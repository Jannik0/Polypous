//! Loading, running and saving graphs: [`Engine::load_graph`], [`Model`] (public-api.md §1.3).
//!
//! A thin layer over `polypous-runtime` (runtime.md): the runtime compiles the plan, checks and
//! loads the weights, binds and runs the graph. The facade picks the device, checks and uploads the
//! call's [`TensorData`] (the runtime never moves values between devices), reads results back and
//! maps [`RuntimeError`]s to [`Error`]s.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, PoisonError, RwLock};

use polypous_backend::{DynTensor, TensorError};
use polypous_formats::safetensors::{
    self, Dtype as StorageDtype, SafetensorsFile, ShardedSafetensors, WriteTensor,
};
use polypous_ir::BindError;
use polypous_runtime::{Plan, PlanOptions, RunOptions, RuntimeError, WeightFile};
use serde::{Deserialize, Serialize};

use crate::cancel::CancellationToken;
use crate::device::{self, DeviceId};
use crate::engine::{Engine, object_from_json};
use crate::error::{Error, ErrorKind, guard};
use crate::graph::{Graph, ValidationReport, invalid_graph_error};
use crate::tensor::{ComputeDtype, Dtype, NamedTensor, TensorData};
use crate::tensor_file::{copy_bytes, format_error, storage_dtype};

/// Options of [`Engine::load_graph`] (public-api.md §1.3).
///
/// JSON: an object, `{}` for the defaults, for example
/// `{"weights":"mlp.safetensors","device":{"backend":"cpu"},"compute_dtype":"f32"}`. Unknown fields
/// are an [`ErrorKind::InvalidInput`] error. The struct is `#[non_exhaustive]`: start from
/// `GraphLoadOptions::default()` and set fields.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct GraphLoadOptions {
    /// A safetensors file, or a sharded checkpoint's `*.safetensors.index.json`. Default `None`:
    /// the graph must declare no params.
    pub weights: Option<PathBuf>,
    /// The device. Default `None`: the engine's primary GPU, or the CPU if there is none.
    pub device: Option<DeviceId>,
    /// The compute dtype (graph-ir.md §2.2). Default `f32`.
    pub compute_dtype: ComputeDtype,
}

impl GraphLoadOptions {
    /// Reads options from their JSON form (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not a JSON object of known fields.
    pub fn from_json(json: &str) -> Result<GraphLoadOptions, Error> {
        guard("graph_load_options_from_json", || {
            object_from_json(json, "The graph load options are not valid.")
        })
    }

    /// The options' JSON form.
    pub fn to_json(&self) -> String {
        // Plain fields; a path that is not UTF-8 can't be written: fall back to the defaults' form.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// One dim of a [`TensorSpec`] shape. JSON: a number for a literal dim, a string for a symbol
/// (`"batch"`) or a dim expression in canonical form (`"seq + 1"`).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Dim {
    /// A literal size.
    Fixed(u64),
    /// A symbol or a dim expression, bound per call.
    Symbolic(String),
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Fixed(size) => write!(f, "{size}"),
            Dim::Symbolic(text) => f.write_str(text),
        }
    }
}

/// Name, dtype and shape of an input, output, state entry or param.
///
/// JSON: `{"name":"image","dtype":"f32","shape":["batch",784]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct TensorSpec {
    /// The name.
    pub name: String,
    /// The value dtype, `float` resolved to the compute dtype.
    pub dtype: Dtype,
    /// The shape as validation inferred it.
    pub shape: Vec<Dim>,
}

/// What a loaded [`Model`] is (public-api.md §1.3, "`ModelInfo` JSON").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct ModelInfo {
    /// The device the model runs on.
    pub device: DeviceId,
    /// The compute dtype.
    pub compute_dtype: ComputeDtype,
    /// Graph inputs, in declaration order.
    pub inputs: Vec<TensorSpec>,
    /// Graph outputs, in declaration order.
    pub outputs: Vec<TensorSpec>,
    /// Root-level state entries (expanded names), in binding order: what
    /// [`Model::run`] takes and returns as `state`.
    pub state: Vec<TensorSpec>,
    /// Declared params, in declaration order, in the compute dtype.
    pub params: Vec<TensorSpec>,
}

/// The result of [`Model::run`].
///
/// JSON: `{"outputs":[NamedTensor…],"state":[NamedTensor…]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RunOutput {
    /// Every graph output, in declaration order.
    pub outputs: Vec<NamedTensor>,
    /// Every root-level state entry, in the order of [`ModelInfo::state`].
    pub state: Vec<NamedTensor>,
}

impl RunOutput {
    /// The output `name`.
    pub fn output(&self, name: &str) -> Option<&TensorData> {
        self.outputs
            .iter()
            .find(|named| named.name == name)
            .map(|named| &named.tensor)
    }
}

fn cancelled() -> Error {
    Error::new(ErrorKind::Cancelled, "The operation was cancelled.")
}

fn check_cancel(cancel: &CancellationToken) -> Result<(), Error> {
    if cancel.is_cancelled() {
        Err(cancelled())
    } else {
        Ok(())
    }
}

/// `text`, ending with a full stop.
pub(crate) fn sentence(text: String) -> String {
    if text.ends_with(['.', '!', '?']) {
        text
    } else {
        format!("{text}.")
    }
}

/// Maps a runtime error to the facade's error (public-api.md §1.3, "Error mapping"): the kind of
/// [`RuntimeError::kind`], `prefix` and the runtime's description as the message, and the node,
/// tensor, file and device in `context`. `weights` is the weight file of a load, if any.
fn runtime_error(prefix: &str, error: RuntimeError, weights: Option<&Path>) -> Error {
    use polypous_runtime::ErrorKind as Rk;
    let kind = match error.kind() {
        Rk::InvalidInput => ErrorKind::InvalidInput,
        Rk::InvalidGraph => ErrorKind::InvalidGraph,
        Rk::UnsupportedModel => ErrorKind::UnsupportedModel,
        Rk::UnsupportedOperation => ErrorKind::UnsupportedOperation,
        Rk::Io => ErrorKind::Io,
        Rk::BackendUnavailable => ErrorKind::BackendUnavailable,
        Rk::Backend => ErrorKind::Backend,
        Rk::Cancelled => ErrorKind::Cancelled,
        _ => ErrorKind::Internal,
    };
    if let RuntimeError::InvalidGraph(report) = &error {
        return invalid_graph_error(&ValidationReport::from_ir(report));
    }
    let text = error.to_string();
    let mut out = if kind == ErrorKind::Cancelled {
        cancelled()
    } else if kind == ErrorKind::Internal {
        Error::new(
            ErrorKind::Internal,
            "Polypous ran into an internal error. Please report this as a bug.",
        )
    } else {
        Error::new(kind, sentence(format!("{prefix}: {text}")))
    }
    .with_details(text);
    if let Some(node) = error.node() {
        out = out.with_node(node);
    }
    match &error {
        RuntimeError::UnsupportedDtype { device, .. }
        | RuntimeError::DeviceUnavailable { device, .. } => out = out.with_device(device.clone()),
        RuntimeError::WrongDevice { name, expected, .. } => {
            out = out.with_tensor(name.clone()).with_device(expected.clone());
        }
        RuntimeError::WeightFile { source } => {
            if let Some(tensor) = source.tensor() {
                out = out.with_tensor(tensor);
            }
            if let Some(path) = source.path() {
                out = out.with_file(path);
            }
        }
        RuntimeError::Weight { tensor, .. } | RuntimeError::WeightUpload { tensor, .. } => {
            out = out.with_tensor(tensor.clone());
        }
        RuntimeError::Bind(bind) => match bind {
            BindError::Missing { name, .. }
            | BindError::Unknown { name, .. }
            | BindError::Duplicate { name, .. }
            | BindError::Dtype { name, .. }
            | BindError::Rank { name, .. }
            | BindError::DimMismatch { name, .. }
            | BindError::DimTooLarge { name, .. } => out = out.with_tensor(name.clone()),
            BindError::InconsistentSymbol { name, .. } => out = out.with_tensor(name.clone()),
            BindError::NonpositiveDim { node, .. }
            | BindError::AttrOutOfRange { node, .. }
            | BindError::TooManyElements { node, .. } => out = out.with_node(node.clone()),
            _ => {}
        },
        _ => {}
    }
    if matches!(
        error,
        RuntimeError::Weight { .. } | RuntimeError::WeightFile { .. }
    ) && let Some(path) = weights
    {
        out = out.with_file(path);
    }
    out
}

/// The kind of a backend error while uploading a call value or reading a result back.
fn tensor_error_kind(error: &TensorError) -> ErrorKind {
    match error {
        TensorError::UnsupportedDType { .. } => ErrorKind::UnsupportedOperation,
        TensorError::DeviceUnavailable { .. } => ErrorKind::BackendUnavailable,
        TensorError::Backend { .. } => ErrorKind::Backend,
        _ => ErrorKind::InvalidInput,
    }
}

fn spec_of(value: &polypous_ir::NamedValue) -> TensorSpec {
    TensorSpec {
        name: value.name.clone(),
        dtype: Dtype::from_ir(value.info.dtype),
        shape: value
            .info
            .shape
            .iter()
            .map(|dim| match dim.as_literal().map(u64::try_from) {
                Some(Ok(size)) => Dim::Fixed(size),
                _ => Dim::Symbolic(dim.to_string()),
            })
            .collect(),
    }
}

fn model_info(graph: &polypous_ir::Graph, plan: &Plan) -> ModelInfo {
    let analysis = plan.analysis();
    let compute_dtype = ComputeDtype::from_ir(plan.compute_dtype());
    ModelInfo {
        device: device::from_backend_id(plan.device()),
        compute_dtype,
        inputs: analysis.inputs.iter().map(spec_of).collect(),
        outputs: analysis.outputs.iter().map(spec_of).collect(),
        state: analysis.state.iter().map(spec_of).collect(),
        params: graph
            .params
            .iter()
            .map(|param| TensorSpec {
                name: param.name.clone(),
                dtype: compute_dtype.dtype(),
                shape: param
                    .shape
                    .iter()
                    .map(|dim| match dim {
                        polypous_ir::model::Dim::Literal(size) => match u64::try_from(*size) {
                            Ok(size) => Dim::Fixed(size),
                            Err(_) => Dim::Symbolic(size.to_string()),
                        },
                        polypous_ir::model::Dim::Expr(text) => Dim::Symbolic(text.clone()),
                    })
                    .collect(),
            })
            .collect(),
    }
}

impl Engine {
    /// The device a load or a trainer uses: `requested`, else the primary GPU, else the CPU. It
    /// must be listed and available; otherwise the id and why it can't be used.
    pub(crate) fn pick_device(
        &self,
        requested: Option<DeviceId>,
    ) -> Result<DeviceId, (DeviceId, String)> {
        let id = requested
            .or_else(|| self.primary_gpu().map(|device| device.id))
            .unwrap_or(DeviceId::Cpu);
        match self.devices().into_iter().find(|device| device.id == id) {
            None => Err((
                id,
                "this engine has no such device (check `polyp devices`; its backend may not be compiled in)"
                    .to_string(),
            )),
            Some(device) if !device.status.is_available() => Err((id, device.status.to_string())),
            Some(_) => Ok(id),
        }
    }

    /// The device a load uses ([`Engine::pick_device`]), as a load error.
    fn load_device(&self, requested: Option<DeviceId>) -> Result<DeviceId, Error> {
        self.pick_device(requested).map_err(|(id, why)| {
            Error::new(
                ErrorKind::BackendUnavailable,
                format!("Device {id} is not available: {why}."),
            )
            .with_device(id.to_string())
        })
    }

    /// Loads `graph` with its weights on one device (public-api.md §1.3, runtime.md §1, §2).
    ///
    /// The device is [`GraphLoadOptions::device`], else the primary GPU, else the CPU; it must be
    /// in [`Engine::devices`] and available. Blocks until the weights are on the device. The token
    /// is checked before each step (device, plan, weights) and after the last one.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::BackendUnavailable`] for a device that is unknown or unavailable,
    /// [`ErrorKind::InvalidGraph`] if the graph doesn't validate with the compute dtype,
    /// [`ErrorKind::UnsupportedOperation`] for a compute dtype or value dtype the device lacks,
    /// [`ErrorKind::InvalidInput`] for missing, extra, wrong-shape or non-float weights and
    /// malformed weight files, [`ErrorKind::UnsupportedModel`] for an unsupported float storage,
    /// [`ErrorKind::Io`], [`ErrorKind::Cancelled`].
    pub fn load_graph(
        &self,
        graph: &Graph,
        options: GraphLoadOptions,
        cancel: &CancellationToken,
    ) -> Result<Model, Error> {
        guard("engine_load_graph", || {
            const PREFIX: &str = "Could not load the model";
            check_cancel(cancel)?;
            let device = self.load_device(options.device)?;
            let plan_options = PlanOptions {
                device: device::to_backend_id(device),
                compute_dtype: options.compute_dtype.to_ir(),
            };
            let plan = Plan::new(graph.ir(), plan_options)
                .map_err(|error| runtime_error(PREFIX, error, None))?;
            check_cancel(cancel)?;
            let weights = options.weights.as_deref();
            let model = match weights {
                Some(path) => {
                    let file = WeightFile::open(path)
                        .map_err(|error| runtime_error(PREFIX, error, Some(path)))?;
                    check_cancel(cancel)?;
                    polypous_runtime::Model::load(plan, &file)
                        .map_err(|error| runtime_error(PREFIX, error, Some(path)))?
                }
                None => polypous_runtime::Model::without_weights(plan)
                    .map_err(|error| runtime_error(PREFIX, error, None))?,
            };
            // Dropping `model` frees its weights.
            check_cancel(cancel)?;
            let info = model_info(graph.ir(), model.plan());
            tracing::info!(
                device = %device,
                compute_dtype = %options.compute_dtype,
                params = info.params.len(),
                "loaded a graph"
            );
            Ok(Model {
                inner: Arc::new(ModelInner {
                    graph: graph.clone(),
                    info,
                    weights: options.weights,
                    model: RwLock::new(Some(Arc::new(model))),
                }),
            })
        })
    }
}

/// A graph loaded with its weights on one device (public-api.md §1.3).
///
/// A cheap-to-clone handle (`Arc` inside), `Send + Sync + 'static`. Clones share the weights;
/// calls take `&self` and may run concurrently. Dropping the last clone frees the weights, or call
/// [`Model::close`].
#[derive(Clone)]
pub struct Model {
    inner: Arc<ModelInner>,
}

struct ModelInner {
    graph: Graph,
    info: ModelInfo,
    /// The weight file it was loaded from (`save` copies unused params from it).
    weights: Option<PathBuf>,
    /// `None` after `close`. Calls hold their own `Arc`, so `close` never waits for them.
    model: RwLock<Option<Arc<polypous_runtime::Model>>>,
}

/// Which list a call value belongs to.
#[derive(Clone, Copy)]
enum ValueKind {
    Input,
    State,
}

impl ValueKind {
    fn label(self) -> &'static str {
        match self {
            ValueKind::Input => "Input",
            ValueKind::State => "State entry",
        }
    }
}

impl Model {
    fn runtime(&self) -> Result<Arc<polypous_runtime::Model>, Error> {
        self.inner
            .model
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    "The model was closed and can't be used any more.",
                )
            })
    }

    /// The device, compute dtype, inputs, outputs, state entries and params.
    pub fn info(&self) -> ModelInfo {
        self.inner.info.clone()
    }

    /// The graph the model was loaded from.
    pub fn graph(&self) -> Graph {
        self.inner.graph.clone()
    }

    /// Frees the weights now (public-api.md §2 rule 2). Later [`Model::run`] and [`Model::save`]
    /// calls on any clone fail with [`ErrorKind::InvalidInput`]; calls that are running finish
    /// first. Idempotent.
    pub fn close(&self) {
        let _ = guard("model_close", || {
            let model = self
                .inner
                .model
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            drop(model);
            Ok(())
        });
    }

    /// Checks one call value (public-api.md §1.3, "Running" step 1): valid `TensorData`, no
    /// dimension of size 0.
    fn check(kind: ValueKind, named: &NamedTensor) -> Result<(), Error> {
        let label = kind.label();
        let name = &named.name;
        named.tensor.validate().map_err(|error| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("{label} `{name}` is not valid: {}", error.message),
            )
            .with_tensor(name.clone())
        })?;
        if named.tensor.shape.contains(&0) {
            return Err(Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "{label} `{name}` has shape {:?} with a dimension of size 0; graphs have no empty dimensions.",
                    named.tensor.shape
                ),
            )
            .with_tensor(name.clone()));
        }
        Ok(())
    }

    /// Uploads one checked call value to `device` ("Running" step 2).
    fn upload(
        kind: ValueKind,
        named: &NamedTensor,
        device: polypous_backend::device::DeviceId,
    ) -> Result<(String, DynTensor), Error> {
        let label = kind.label();
        let name = &named.name;
        let in_value = |error: Error| error.with_tensor(name.clone());
        let data = named.tensor.to_backend().map_err(in_value)?;
        let tensor = DynTensor::from_data(&data, device).map_err(|error| {
            in_value(
                Error::new(
                    tensor_error_kind(&error),
                    sentence(format!("{label} `{name}` could not be used: {error}")),
                )
                .with_details(error.to_string())
                .with_device(device.to_string()),
            )
        })?;
        Ok((name.clone(), tensor))
    }

    /// Runs the graph once (public-api.md §1.3, runtime.md §3): `inputs` are the graph inputs by
    /// name, `state` a value for every root-level state entry ([`ModelInfo::state`]). Blocks until
    /// done; the token is checked before every node and `repeat` iteration.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] for malformed [`TensorData`], a zero-size dimension, missing,
    /// extra, wrong-dtype or wrong-shape values (`context.tensor`), an out-of-range index while
    /// running (`context.node`) or a closed model; [`ErrorKind::UnsupportedOperation`] for a
    /// value dtype the device can't hold; [`ErrorKind::Cancelled`]; [`ErrorKind::Backend`].
    pub fn run(
        &self,
        inputs: Vec<NamedTensor>,
        state: Vec<NamedTensor>,
        cancel: &CancellationToken,
    ) -> Result<RunOutput, Error> {
        guard("model_run", || {
            let model = self.runtime()?;
            let device = model.plan().device();
            for (kind, values) in [(ValueKind::Input, &inputs), (ValueKind::State, &state)] {
                for named in values {
                    Model::check(kind, named)?;
                }
            }
            check_cancel(cancel)?;
            let mut uploaded_inputs = Vec::with_capacity(inputs.len());
            for named in &inputs {
                uploaded_inputs.push(Model::upload(ValueKind::Input, named, device)?);
            }
            let mut uploaded_state = Vec::with_capacity(state.len());
            for named in &state {
                uploaded_state.push(Model::upload(ValueKind::State, named, device)?);
            }
            drop((inputs, state));
            let options = RunOptions {
                cancel: Some(cancel.runtime()),
                trace: false,
            };
            let out = model
                .run(uploaded_inputs, uploaded_state, &options)
                .map_err(|error| runtime_error("The model run failed", error, None))?;
            let read = |(name, tensor): (String, DynTensor)| -> Result<NamedTensor, Error> {
                let data = tensor.into_data().map_err(|error| {
                    Error::new(
                        ErrorKind::Backend,
                        sentence(format!(
                            "Could not read back `{name}` from the device: {error}"
                        )),
                    )
                    .with_details(error.to_string())
                    .with_tensor(name.clone())
                })?;
                Ok(NamedTensor::new(name, TensorData::from_backend(data)?))
            };
            Ok(RunOutput {
                outputs: out
                    .outputs
                    .into_iter()
                    .map(read)
                    .collect::<Result<_, _>>()?,
                state: out.state.into_iter().map(read).collect::<Result<_, _>>()?,
            })
        })
    }

    /// Writes the graph's canonical JSON to `graph_path` and every declared param to
    /// `weights_path` as one safetensors file (public-api.md §1.3, "Saving").
    ///
    /// Params a node uses are read back from the device in the compute dtype; declared params that
    /// no node uses are not on the device and are copied byte for byte from the weight file the
    /// model was loaded from, which is opened again. The weights file is written first, through a
    /// temporary file and a rename.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Io`] if a file can't be read or written (`context.file`);
    /// [`ErrorKind::InvalidInput`] if the original weight file no longer holds an unused param with
    /// its declared shape and a float storage, or the model was closed; [`ErrorKind::Backend`].
    pub fn save(
        &self,
        graph_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
    ) -> Result<(), Error> {
        let graph_path = graph_path.as_ref();
        let weights_path = weights_path.as_ref();
        guard("model_save", || {
            const PREFIX: &str = "Could not save the model";
            let model = self.runtime()?;
            let params = &self.inner.graph.ir().params;
            // (name, storage dtype, shape, bytes) of every param, in declaration order.
            let mut tensors: Vec<(String, StorageDtype, Vec<usize>, Vec<u8>)> =
                Vec::with_capacity(params.len());
            let mut source: Option<Source> = None;
            for param in params {
                let name = &param.name;
                if let Some(tensor) = model.param(name) {
                    let data = tensor.to_data().map_err(|error| {
                        Error::new(
                            ErrorKind::Backend,
                            sentence(format!(
                                "{PREFIX}: could not read param `{name}` from the device: {error}"
                            )),
                        )
                        .with_details(error.to_string())
                        .with_tensor(name.clone())
                    })?;
                    let data = TensorData::from_backend(data)?;
                    let shape = data
                        .shape
                        .iter()
                        .map(|&dim| usize::try_from(dim))
                        .collect::<Result<Vec<usize>, _>>()
                        .map_err(|_| {
                            Error::new(
                                ErrorKind::Internal,
                                format!("{PREFIX}: param `{name}` is too large."),
                            )
                        })?;
                    tensors.push((name.clone(), storage_dtype(data.dtype), shape, data.bytes));
                } else {
                    let source = match &mut source {
                        Some(source) => source,
                        None => source.insert(Source::open(self.inner.weights.as_deref(), PREFIX)?),
                    };
                    tensors.push(source.unused_param(param, PREFIX)?);
                }
            }
            // The copies are owned: close the original file before writing (it may be the same
            // path, and Windows can't rename over a mapped file).
            drop(source);
            let mut entries = Vec::with_capacity(tensors.len());
            for (name, dtype, shape, bytes) in &tensors {
                let tensor = WriteTensor::new(*dtype, shape, bytes).map_err(|error| {
                    format_error(PREFIX, weights_path, &error).with_tensor(name.clone())
                })?;
                entries.push((name.as_str(), tensor));
            }
            safetensors::write_file(weights_path, entries, &BTreeMap::new())
                .map_err(|error| format_error(PREFIX, weights_path, &error))?;
            std::fs::write(graph_path, self.inner.graph.to_json()).map_err(|error| {
                Error::new(
                    ErrorKind::Io,
                    format!(
                        "{PREFIX}: could not write {}: {error}.",
                        graph_path.display()
                    ),
                )
                .with_details(error.to_string())
                .with_file(graph_path)
            })?;
            tracing::info!(
                graph = %graph_path.display(),
                weights = %weights_path.display(),
                params = tensors.len(),
                "saved a model"
            );
            Ok(())
        })
    }
}

/// The original weight file, opened again by `save` for the params that are not on the device.
enum Source {
    Single(PathBuf, SafetensorsFile),
    Sharded(PathBuf, ShardedSafetensors),
}

impl Source {
    fn open(path: Option<&Path>, prefix: &str) -> Result<Source, Error> {
        let Some(path) = path else {
            // A model without a weight file declares no params, so every param is on the device.
            return Err(Error::new(
                ErrorKind::Internal,
                "Polypous ran into an internal error. Please report this as a bug.",
            )
            .with_details("an unused param of a model loaded without weights"));
        };
        let sharded = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.ends_with(".index.json"));
        let message = format!("{prefix}: could not read the original weight file");
        if sharded {
            ShardedSafetensors::open(path)
                .map(|file| Source::Sharded(path.to_path_buf(), file))
                .map_err(|error| format_error(&message, path, &error))
        } else {
            SafetensorsFile::open(path)
                .map(|file| Source::Single(path.to_path_buf(), file))
                .map_err(|error| format_error(&message, path, &error))
        }
    }

    /// A copy of the stored tensor of an unused param, checked against its declared shape.
    fn unused_param(
        &self,
        param: &polypous_ir::model::Param,
        prefix: &str,
    ) -> Result<(String, StorageDtype, Vec<usize>, Vec<u8>), Error> {
        let name = &param.name;
        let (path, view) = match self {
            Source::Single(path, file) => (path, file.tensor(name)),
            Source::Sharded(path, file) => (path, file.tensor(name)),
        };
        let changed = |why: String| {
            Error::new(
                ErrorKind::InvalidInput,
                format!(
                    "{prefix}: param `{name}` is not used by any node, so it is copied from {}, but {why}.",
                    path.display()
                ),
            )
            .with_tensor(name.clone())
            .with_file(path)
        };
        let view = view.map_err(|error| changed(format!("reading it failed: {error}")))?;
        let declared: Option<Vec<usize>> = param
            .shape
            .iter()
            .map(|dim| match dim {
                polypous_ir::model::Dim::Literal(size) => usize::try_from(*size).ok(),
                polypous_ir::model::Dim::Expr(_) => None,
            })
            .collect();
        if declared.as_deref() != Some(view.shape()) {
            return Err(changed(format!(
                "the file now stores it with shape {:?}",
                view.shape()
            )));
        }
        if !matches!(
            view.dtype(),
            StorageDtype::F32 | StorageDtype::F16 | StorageDtype::BF16
        ) {
            return Err(changed(format!(
                "the file now stores it as {}",
                view.dtype().name()
            )));
        }
        let bytes = copy_bytes(view.data())?;
        Ok((name.clone(), view.dtype(), view.shape().to_vec(), bytes))
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let closed = self
            .inner
            .model
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none();
        f.debug_struct("Model")
            .field("device", &self.inner.info.device)
            .field("compute_dtype", &self.inner.info.compute_dtype)
            .field("closed", &closed)
            .finish_non_exhaustive()
    }
}
