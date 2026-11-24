//! Creating a trainer: the checks of `docs/specs/training.md` §2.10, in order, and resuming from
//! a checkpoint (§5.4).
//!
//! Steps 1–8 touch no device: the plan is compiled (which checks the device's dtypes) but nothing
//! is uploaded and nothing runs before step 9. The runtime's training-mode checks (runtime.md
//! §3.1 checks 2 and 3) run at step 6 on the bound shapes, through the graph analysis, with the
//! runtime's own error values, so a last batch of one sample with `nn.batch_norm` fails here
//! rather than in the middle of an epoch.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData};
use polypous_formats::safetensors::{
    Dtype as StorageDtype, SafetensorsFile, ShardedSafetensors, TensorInfo, bf16_to_f32, f16_to_f32,
};
use polypous_ir::{Analysis, BindError, BoundNode, Graph, NodeInfo, TensorDesc, ValidateOptions};
use polypous_runtime::{CancellationToken, Plan, PlanOptions, RuntimeError};

use super::{Core, DeviceResolver, ResumeOptions, TrainerOptions};
use crate::checkpoint::{self, Loaded};
use crate::config::{LossConfig, MetricConfig};
use crate::data::{
    Batcher, DataSources, Dataset, Datasets, FieldSelection, InMemoryDataset, read_datasets,
};
use crate::error::{TrainError, TrainErrorKind};
use crate::events::{Clock, SystemClock};
use crate::init::initial_values;
use crate::json;
use crate::loss::Loss;
use crate::metrics::{Accuracy, LossMetric, MetricError};
use crate::optimizer::{Optimizer, StateTensor, TensorMap};
use crate::schedule::Schedule;
use crate::targets::TargetError;
use crate::training_config::TrainingConfig;

/// The most params named in the message of §2.10 step 7.
const LISTED_PARAMS: usize = 10;

fn check_cancel(cancel: &CancellationToken) -> Result<(), TrainError> {
    if cancel.is_cancelled() {
        Err(TrainError::new(
            TrainErrorKind::Cancelled,
            "cancelled while creating the trainer",
        ))
    } else {
        Ok(())
    }
}

/// A use of a dataset field (§2.10 step 5), with its config pointer.
#[derive(Clone, Debug)]
enum Use {
    /// Feeds this graph input (index into `analysis.inputs`).
    Input(usize),
    /// `loss.target`.
    LossTarget,
    /// The target of metric `i`.
    MetricTarget(usize),
}

#[derive(Clone, Debug)]
struct FieldUse {
    pointer: String,
    field: String,
    usage: Use,
}

/// Every field use, in check order: `inputs` (key order), `loss.target`, the metrics' `target`.
fn field_uses(config: &TrainingConfig, analysis: &Analysis) -> Vec<FieldUse> {
    let mut out = Vec::new();
    for (input, field) in &config.inputs {
        if let Some(i) = analysis.inputs.iter().position(|v| v.name == *input) {
            out.push(FieldUse {
                pointer: json::child("/inputs", input),
                field: field.clone(),
                usage: Use::Input(i),
            });
        }
    }
    out.push(FieldUse {
        pointer: "/loss/target".into(),
        field: config.loss.target().to_owned(),
        usage: Use::LossTarget,
    });
    for (i, m) in config.metrics.iter().enumerate() {
        let MetricConfig::Accuracy(c) = m;
        out.push(FieldUse {
            pointer: format!("/metrics/{i}/target"),
            field: c.target.clone(),
            usage: Use::MetricTarget(i),
        });
    }
    out
}

/// What steps 3–6 produce.
pub(super) struct Prepared {
    pub device: DeviceId,
    pub plan: Plan,
    pub datasets: Datasets,
    pub batcher: Batcher,
    pub val_batcher: Option<Batcher>,
    pub loss: Loss,
    pub metrics: Vec<Accuracy>,
    /// (graph input, field), in graph input order.
    pub input_fields: Vec<(String, String)>,
}

/// Step 2: the graph, validated with compute dtype `f32`, without state.
fn check_graph(graph: &Graph) -> Result<Analysis, TrainError> {
    let analysis = graph
        .analyze(&ValidateOptions::default())
        .map_err(|report| {
            TrainError::invalid("the graph is not valid").with_details(report.to_string())
        })?;
    if let Some(entry) = analysis.state.first() {
        return Err(TrainError::new(
            TrainErrorKind::UnsupportedOperation,
            format!(
                "the graph has state (`{}`); training a graph with state is not supported in \
                 this version",
                entry.name
            ),
        )
        .with_tensor(entry.name.clone()));
    }
    Ok(analysis)
}

/// Step 3: the device the caller resolved, and the plan for it.
fn check_device(
    graph: &Graph,
    device_text: &str,
    resolve: DeviceResolver<'_>,
) -> Result<(DeviceId, Plan), TrainError> {
    let device = resolve(device_text)?;
    let plan = Plan::new(graph, PlanOptions::new(device)).map_err(|e| {
        let err = TrainError::from(e);
        if err.device().is_none() && err.kind() == TrainErrorKind::BackendUnavailable {
            err.with_device(device.to_string())
        } else {
            err
        }
    })?;
    Ok((device, plan))
}

/// Step 4: references between the config and the graph.
fn check_references(config: &TrainingConfig, analysis: &Analysis) -> Result<(), TrainError> {
    let refer = |pointer: &str, name: &str, message: String| {
        TrainError::config(pointer, &message).with_tensor(name)
    };
    for input in config.inputs.keys() {
        if !analysis.inputs.iter().any(|v| v.name == *input) {
            let names: Vec<&str> = analysis.inputs.iter().map(|v| v.name.as_str()).collect();
            return Err(refer(
                &json::child("/inputs", input),
                input,
                format!("`{input}` is not a graph input (the graph's inputs are {names:?})"),
            ));
        }
    }
    for v in &analysis.inputs {
        if !config.inputs.contains_key(&v.name) {
            return Err(refer(
                "/inputs",
                &v.name,
                format!(
                    "graph input `{}` is not mapped to a dataset field; `inputs` must map every \
                     graph input",
                    v.name
                ),
            ));
        }
    }
    let output_names: Vec<&str> = analysis.outputs.iter().map(|v| v.name.as_str()).collect();
    let check_output = |pointer: &str, name: &str| {
        if output_names.contains(&name) {
            Ok(())
        } else {
            Err(refer(
                pointer,
                name,
                format!(
                    "`{name}` is not a graph output (the graph's outputs are {output_names:?})"
                ),
            ))
        }
    };
    check_output("/loss/prediction", config.loss.prediction())?;
    for (i, m) in config.metrics.iter().enumerate() {
        let MetricConfig::Accuracy(c) = m;
        check_output(&format!("/metrics/{i}/prediction"), &c.prediction)?;
    }
    if let Some(name) = config
        .checkpoint
        .as_ref()
        .and_then(|c| c.keep_best.as_deref())
    {
        let reported = config.reported_names();
        if !reported.iter().any(|r| r == name) {
            let why = if name.starts_with("val.") && config.data.validation.is_none() {
                "a `val.*` metric needs a validation set".to_owned()
            } else {
                format!("the reported names are {reported:?}")
            };
            return Err(TrainError::config(
                "/checkpoint/keep_best",
                &format!("`keep_best` {name:?} is not a reported metric: {why}"),
            ));
        }
    }
    Ok(())
}

/// Step 5: read the datasets (only the used fields), then check the fields.
fn read_data(
    config: &TrainingConfig,
    analysis: &Analysis,
    base_dir: &Path,
    in_memory: Vec<InMemoryDataset>,
    cancel: &CancellationToken,
) -> Result<Datasets, TrainError> {
    let uses = field_uses(config, analysis);
    let sources = DataSources {
        base_dir: base_dir.to_path_buf(),
        in_memory,
        fields: FieldSelection::Only(config.used_fields()),
    };
    let datasets = match read_datasets(&config.data, config.seed, sources, cancel) {
        Ok(d) => d,
        Err(e) if e.has_no_used_fields() => {
            // The dataset has none of the used fields: report the first use as §2.10 step 5 says.
            let role = e.dataset().unwrap_or("train");
            // `uses` always holds `loss.target`.
            let Some(first) = uses.first() else {
                return Err(e.into());
            };
            return Err(missing_field(first, role_name(role)).with_details(e.to_string()));
        }
        Err(e) => return Err(e.into()),
    };
    check_cancel(cancel)?;
    for u in &uses {
        let Some((dtype, shape)) = datasets.train.field_info(&u.field) else {
            return Err(missing_field(u, "training"));
        };
        if let Some(val) = &datasets.validation {
            let Some((vd, vs)) = val.field_info(&u.field) else {
                return Err(missing_field(u, "validation"));
            };
            if vd != dtype || vs != shape {
                return Err(TrainError::config(
                    &u.pointer,
                    &format!(
                        "field `{}` is {} {shape:?} per sample in the training set but {vd} \
                         {vs:?} in the validation set",
                        u.field, dtype
                    ),
                )
                .with_tensor(u.field.clone()));
            }
        }
        let allowed: Vec<DType> = match u.usage {
            Use::Input(i) => {
                let want = analysis.inputs[i].info.dtype;
                vec![DType::from_name(want.as_str()).unwrap_or(DType::F32)]
            }
            Use::LossTarget => match config.loss {
                LossConfig::CrossEntropy(_) => vec![DType::I64, DType::I32],
                _ => vec![DType::F32],
            },
            Use::MetricTarget(_) => vec![DType::I64, DType::I32, DType::F32],
        };
        if !allowed.contains(&dtype) {
            let what = match u.usage {
                Use::Input(i) => format!("graph input `{}`", analysis.inputs[i].name),
                Use::LossTarget => format!("the target of `{}`", config.loss.kind()),
                Use::MetricTarget(i) => {
                    format!("the target of metric `{}`", config.metrics[i].name())
                }
            };
            let names: Vec<&str> = allowed.iter().map(|d| d.name()).collect();
            return Err(TrainError::config(
                &u.pointer,
                &format!(
                    "field `{}` has dtype {dtype}, but {what} takes {}",
                    u.field,
                    names.join(" or ")
                ),
            )
            .with_tensor(u.field.clone()));
        }
    }
    Ok(datasets)
}

fn role_name(role: &str) -> &'static str {
    if role == "validation" {
        "validation"
    } else {
        "training"
    }
}

fn missing_field(u: &FieldUse, set: &str) -> TrainError {
    TrainError::config(
        &u.pointer,
        &format!("field `{}` is not in the {set} set", u.field),
    )
    .with_tensor(u.field.clone())
}

/// The batch sizes that occur in a set of `n` samples: the full one and a smaller last one.
fn batch_sizes(n: u64, batch_size: u64) -> Vec<u64> {
    let full = n.min(batch_size);
    let last = n % batch_size;
    if n > batch_size && last != 0 {
        vec![full, last]
    } else {
        vec![full]
    }
}

fn bind_error_name(e: &BindError) -> Option<&str> {
    match e {
        BindError::Missing { name, .. }
        | BindError::Unknown { name, .. }
        | BindError::Duplicate { name, .. }
        | BindError::Dtype { name, .. }
        | BindError::Rank { name, .. }
        | BindError::DimMismatch { name, .. } => Some(name),
        _ => None,
    }
}

/// The runtime's §3.1 check 3 on bound shapes (`exec.rs` `check_values_per_channel`), with its
/// error value.
fn check_batch_norm(infos: &[NodeInfo], bound: &[BoundNode], path: &str) -> Result<(), TrainError> {
    for (info, b) in infos.iter().zip(bound) {
        match (info, b) {
            (NodeInfo::Op(op), BoundNode::Op(b)) if op.op.name == "nn.batch_norm" => {
                let shape = b.outputs.first().map(Vec::as_slice).unwrap_or_default();
                let total = shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d));
                let values = match (total, shape.get(1)) {
                    (Some(t), Some(&c)) if c > 0 => t / c,
                    _ => {
                        return Err(TrainError::internal(format!(
                            "node `{path}{}`: batch_norm has the bound shape {shape:?}",
                            op.id
                        )));
                    }
                };
                if values < 2 {
                    return Err(RuntimeError::BatchNormValues {
                        node: format!("{path}{}", op.id),
                        shape: shape.to_vec(),
                        values,
                    }
                    .into());
                }
            }
            (NodeInfo::Repeat(r), BoundNode::Repeat(b)) if r.count > 0 => {
                check_batch_norm(&r.nodes, &b.nodes, &format!("{path}{}[0]/", r.id))?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// The runtime's §3.1 check 2 (`exec.rs` `check_training`), with its error value.
fn check_running_stats(analysis: &Analysis) -> Result<(), TrainError> {
    for p in &analysis.params {
        if p.running_stat && p.uses != 1 {
            let uses: Vec<String> = p.first_uses.iter().map(ToString::to_string).collect();
            return Err(RuntimeError::SharedRunningStat {
                param: p.name.clone(),
                uses: p.uses,
                first: uses.first().cloned().unwrap_or_default(),
                second: uses.get(1).cloned().unwrap_or_default(),
            }
            .into());
        }
    }
    Ok(())
}

fn target_error(pointer: &str, field: &str, set: &str, e: &TargetError) -> TrainError {
    TrainError::config(pointer, &format!("field `{field}` ({set} set): {e}")).with_tensor(field)
}

/// Step 6: shapes, target values and the training-mode checks.
fn check_shapes(
    config: &TrainingConfig,
    analysis: &Analysis,
    datasets: &Datasets,
    loss: &Loss,
    metrics: &[Accuracy],
) -> Result<(), TrainError> {
    let mut sizes: Vec<(u64, bool)> = batch_sizes(datasets.train.len(), config.batch_size)
        .into_iter()
        .map(|s| (s, true))
        .collect();
    if let Some(v) = &datasets.validation {
        for s in batch_sizes(v.len(), config.batch_size) {
            sizes.push((s, false));
        }
    }
    let info = |name: &str| {
        datasets
            .train
            .field_info(name)
            .map(|(d, s)| (d, s.iter().map(|&x| x as usize).collect::<Vec<_>>()))
    };
    let output = |name: &str| analysis.outputs.iter().find(|o| o.name == name);
    let mut pred_shapes: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for (size, train) in sizes {
        if !seen.insert((size, train)) {
            continue;
        }
        let size = size as usize;
        let mut shapes = Vec::with_capacity(analysis.inputs.len());
        for v in &analysis.inputs {
            let Some((_, sample)) = config.inputs.get(&v.name).and_then(|f| info(f)) else {
                return Err(TrainError::internal(format!(
                    "input `{}` has no field",
                    v.name
                )));
            };
            let mut shape = vec![size];
            shape.extend(sample);
            shapes.push((v.name.as_str(), v.info.dtype, shape));
        }
        let descs: Vec<(&str, TensorDesc<'_>)> = shapes
            .iter()
            .map(|(n, d, s)| {
                (
                    *n,
                    TensorDesc {
                        dtype: *d,
                        shape: s,
                    },
                )
            })
            .collect();
        let bindings = analysis.bind(&descs, &[]).map_err(|e| {
            let name = bind_error_name(&e).unwrap_or_default().to_owned();
            TrainError::config(
                &json::child("/inputs", &name),
                &format!("a batch of {size} samples doesn't fit the graph: {e}"),
            )
            .with_tensor(name)
        })?;
        let shape_of = |name: &str| -> Result<(DType, Vec<usize>), TrainError> {
            let o = output(name)
                .ok_or_else(|| TrainError::internal(format!("output `{name}` vanished")))?;
            let dims = o
                .info
                .shape
                .iter()
                .map(|d| {
                    d.eval(&|s| bindings.symbol(s))
                        .ok()
                        .and_then(|v| usize::try_from(v).ok())
                        .ok_or_else(|| {
                            TrainError::internal(format!("output `{name}` has an unbound shape"))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let dtype = DType::from_name(o.info.dtype.as_str())
                .ok_or_else(|| TrainError::internal("unknown dtype".to_owned()))?;
            Ok((dtype, dims))
        };
        let target_shape = |field: &str| -> Result<(DType, Vec<usize>), TrainError> {
            let (d, sample) = info(field)
                .ok_or_else(|| TrainError::internal(format!("field `{field}` vanished")))?;
            let mut shape = vec![size];
            shape.extend(sample);
            Ok((d, shape))
        };
        let pred = shape_of(config.loss.prediction())?;
        let target = target_shape(config.loss.target())?;
        loss.check_shapes((pred.0, &pred.1), (target.0, &target.1))
            .map_err(|e| {
                TrainError::config(
                    "/loss",
                    &format!(
                        "prediction `{}` and target `{}` don't fit: {e}",
                        config.loss.prediction(),
                        config.loss.target()
                    ),
                )
                .with_tensor(config.loss.prediction())
            })?;
        pred_shapes
            .entry(config.loss.prediction().to_owned())
            .or_insert(pred.1);
        for (i, (m, acc)) in config.metrics.iter().zip(metrics).enumerate() {
            let MetricConfig::Accuracy(c) = m;
            let pred = shape_of(&c.prediction)?;
            let target = target_shape(&c.target)?;
            acc.check_shapes((pred.0, &pred.1), (target.0, &target.1))
                .map_err(|e| match e {
                    MetricError::ThresholdWithMulticlass { .. } => TrainError::config(
                        &format!("/metrics/{i}/threshold"),
                        &format!("{e}; leave `threshold` null for class-index targets"),
                    )
                    .with_tensor(c.target.clone()),
                    other => TrainError::config(
                        &format!("/metrics/{i}"),
                        &format!(
                            "prediction `{}` and target `{}` don't fit: {other}",
                            c.prediction, c.target
                        ),
                    )
                    .with_tensor(c.prediction.clone()),
                })?;
            pred_shapes.entry(c.prediction.clone()).or_insert(pred.1);
        }
        if train {
            check_batch_norm(&analysis.nodes, &bindings.nodes, "")?;
        }
    }
    // Target values on every sample of both sets.
    let sets: Vec<(&Dataset, &str)> = std::iter::once((&datasets.train, "training"))
        .chain(datasets.validation.iter().map(|v| (v, "validation")))
        .collect();
    let pred_shape = |name: &str| pred_shapes.get(name).cloned().unwrap_or_default();
    for (set, set_name) in &sets {
        let field = config.loss.target();
        if let Some(data) = set.field(field) {
            loss.check_target_values(&pred_shape(config.loss.prediction()), data)
                .map_err(|e| target_error("/loss/target", field, set_name, &e))?;
        }
        for (i, (m, acc)) in config.metrics.iter().zip(metrics).enumerate() {
            let MetricConfig::Accuracy(c) = m;
            if let Some(data) = set.field(&c.target) {
                acc.check_target_values(&pred_shape(&c.prediction), data)
                    .map_err(|e| {
                        target_error(&format!("/metrics/{i}/target"), &c.target, set_name, &e)
                    })?;
            }
        }
    }
    check_running_stats(analysis)
}

/// Steps 3–6, shared by a new run and a resume.
fn prepare(
    graph: &Graph,
    analysis: &Analysis,
    config: &TrainingConfig,
    base_dir: &Path,
    in_memory: Vec<InMemoryDataset>,
    resolve: DeviceResolver<'_>,
    cancel: &CancellationToken,
) -> Result<Prepared, TrainError> {
    let (device, plan) = check_device(graph, &config.device, resolve)?;
    check_cancel(cancel)?;
    check_references(config, analysis)?;
    check_cancel(cancel)?;
    let datasets = read_data(config, analysis, base_dir, in_memory, cancel)?;
    check_cancel(cancel)?;
    let loss = Loss::new(&config.loss).map_err(|e| TrainError::internal(e.to_string()))?;
    let metrics = config
        .metrics
        .iter()
        .map(|m| {
            let MetricConfig::Accuracy(c) = m;
            Accuracy::new(c).map_err(|e| TrainError::internal(e.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    check_shapes(config, analysis, &datasets, &loss, &metrics)?;
    check_cancel(cancel)?;
    let batcher = Batcher::new(
        datasets.train.len(),
        config.batch_size,
        config.data.shuffle,
        config.seed,
    )?;
    let val_batcher = datasets
        .validation
        .as_ref()
        .map(|v| Batcher::validation(v.len(), config.batch_size))
        .transpose()?;
    let input_fields = analysis
        .inputs
        .iter()
        .map(|v| {
            // Step 4 proved that `inputs` maps every graph input.
            let field = config.inputs.get(&v.name).cloned().unwrap_or_default();
            (v.name.clone(), field)
        })
        .collect();
    Ok(Prepared {
        device,
        plan,
        datasets,
        batcher,
        val_batcher,
        loss,
        metrics,
        input_fields,
    })
}

/// `base_dir` made absolute (null: the current directory).
fn absolute_base(base_dir: Option<&Path>) -> Result<PathBuf, TrainError> {
    let base = base_dir.unwrap_or_else(|| Path::new("."));
    std::path::absolute(base).map_err(|e| TrainError::io(&e, "resolve", base))
}

/// The weight file of `TrainerOptions::weights`: one safetensors file or a sharded checkpoint.
enum WeightSource {
    Single(SafetensorsFile),
    Sharded(ShardedSafetensors),
}

impl WeightSource {
    fn open(path: &Path) -> Result<Self, TrainError> {
        let sharded = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".index.json"));
        let opened = if sharded {
            ShardedSafetensors::open(path).map(WeightSource::Sharded)
        } else {
            SafetensorsFile::open(path).map(WeightSource::Single)
        };
        opened.map_err(|e| {
            let io = matches!(e.kind(), polypous_formats::safetensors::ErrorKind::Io(_));
            let kind = if io {
                TrainErrorKind::Io
            } else {
                TrainErrorKind::InvalidInput
            };
            TrainError::new(kind, format!("cannot read the weight file: {e}"))
                .with_file(path)
                .with_details(e.to_string())
        })
    }

    fn names(&self) -> Vec<&str> {
        match self {
            Self::Single(f) => f.names().collect(),
            Self::Sharded(f) => f.names().collect(),
        }
    }

    fn info(&self, name: &str) -> Option<&TensorInfo> {
        match self {
            Self::Single(f) => f.info(name),
            Self::Sharded(f) => f.info(name),
        }
    }

    fn data(&self, name: &str) -> Result<&[u8], polypous_formats::safetensors::Error> {
        match self {
            Self::Single(f) => f.tensor(name).map(|v| v.data()),
            Self::Sharded(f) => f.tensor(name).map(|v| v.data()),
        }
    }
}

/// Step 7: the initial weight file (a subset of the params), as `f32` bytes by param name.
fn read_weights(path: &Path, analysis: &Analysis) -> Result<BTreeMap<String, Vec<u8>>, TrainError> {
    let source = WeightSource::open(path)?;
    let mut names = source.names();
    names.sort_unstable();
    let bad = |name: &str, why: String| {
        TrainError::invalid(format!("weight `{name}`: {why}"))
            .with_tensor(name)
            .with_file(path)
    };
    let mut checked = Vec::with_capacity(names.len());
    for name in names {
        let Some(param) = analysis.param(name) else {
            return Err(bad(
                name,
                "the weight file has this tensor, but the graph declares no param of that name"
                    .into(),
            ));
        };
        let Some(info) = source.info(name) else {
            continue;
        };
        let size = match info.dtype() {
            StorageDtype::F32 => 4,
            StorageDtype::F16 | StorageDtype::BF16 => 2,
            other => {
                return Err(bad(
                    name,
                    format!(
                        "stored as {}, but params are float tensors (F32, F16 or BF16)",
                        other.name()
                    ),
                ));
            }
        };
        if info.shape() != param.shape.as_slice() {
            return Err(bad(
                name,
                format!(
                    "stored with shape {:?}, but the graph declares {:?}",
                    info.shape(),
                    param.shape
                ),
            ));
        }
        let bytes = param
            .shape
            .iter()
            .try_fold(size, |acc: usize, &d| acc.checked_mul(d));
        let (start, end) = info.data_offsets();
        if bytes.is_none() || end.checked_sub(start) != bytes || Some(info.byte_len()) != bytes {
            return Err(bad(name, "its byte range doesn't fit its shape".into()));
        }
        checked.push((name.to_owned(), info.dtype()));
    }
    let mut out = BTreeMap::new();
    for (name, dtype) in checked {
        let data = source.data(&name).map_err(|e| {
            TrainError::invalid(format!("cannot read weight `{name}`: {e}"))
                .with_tensor(name.clone())
                .with_file(path)
        })?;
        let widen: Option<fn(u16) -> f32> = match dtype {
            StorageDtype::F16 => Some(f16_to_f32),
            StorageDtype::BF16 => Some(bf16_to_f32),
            _ => None,
        };
        let len = if widen.is_some() {
            data.len() * 2
        } else {
            data.len()
        };
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| {
            TrainError::new(
                TrainErrorKind::OutOfMemory,
                format!("could not reserve {len} bytes for weight `{name}`"),
            )
            .with_tensor(name.clone())
        })?;
        match widen {
            Some(widen) => {
                let (pairs, _) = data.as_chunks::<2>();
                for &pair in pairs {
                    bytes.extend_from_slice(&widen(u16::from_le_bytes(pair)).to_le_bytes());
                }
            }
            None => bytes.extend_from_slice(data),
        }
        out.insert(name, bytes);
    }
    Ok(out)
}

/// The params without a value: neither in the weight file nor with an `init` (step 7).
fn check_values(
    analysis: &Analysis,
    weights: &BTreeMap<String, Vec<u8>>,
) -> Result<(), TrainError> {
    let missing: Vec<&str> = analysis
        .params
        .iter()
        .filter(|p| p.init.is_none() && !weights.contains_key(&p.name))
        .map(|p| p.name.as_str())
        .collect();
    let Some(first) = missing.first() else {
        return Ok(());
    };
    let mut listed = missing
        .iter()
        .take(LISTED_PARAMS)
        .copied()
        .collect::<Vec<_>>()
        .join(", ");
    if missing.len() > LISTED_PARAMS {
        listed.push_str(&format!(" and {} more", missing.len() - LISTED_PARAMS));
    }
    Err(TrainError::invalid(format!(
        "These params have no init and no weights: {listed}"
    ))
    .with_tensor(*first)
    .with_details(missing.join("\n")))
}

/// Step 8: the run directory of a new run.
fn prepare_run_dir(dir: &Path) -> Result<(), TrainError> {
    if dir.exists() {
        if let Some(entry) = checkpoint::list(dir)?.first() {
            return Err(TrainError::invalid(format!(
                "the run directory {} already holds checkpoints ({}); use a new directory, or \
                 resume from it",
                dir.display(),
                entry.path.display()
            ))
            .with_file(&entry.path));
        }
        if !dir.is_dir() {
            return Err(TrainError::invalid(format!(
                "the run directory {} is not a directory",
                dir.display()
            ))
            .with_file(dir));
        }
        return Ok(());
    }
    std::fs::create_dir_all(dir).map_err(|e| TrainError::io(&e, "create", dir))
}

/// Uploads `f32` bytes of `shape` to `device`.
pub(super) fn upload_f32(
    name: &str,
    shape: &[usize],
    bytes: Vec<u8>,
    device: DeviceId,
) -> Result<DynTensor, TrainError> {
    let shape = shape.iter().map(|&d| d as u64).collect();
    let data = TensorData::new(DType::F32, shape, bytes)
        .map_err(|e| TrainError::internal(format!("param `{name}`: {e}")))?;
    DynTensor::from_data(&data, device).map_err(|e| TrainError::from(e).with_tensor(name))
}

fn f32_bytes(values: &[f32]) -> Result<Vec<u8>, TrainError> {
    let mut out = Vec::new();
    out.try_reserve_exact(values.len() * 4).map_err(|_| {
        TrainError::new(
            TrainErrorKind::OutOfMemory,
            "could not reserve host memory for initial values",
        )
    })?;
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(out)
}

/// What [`Core::build`] needs besides [`Prepared`].
pub(super) struct Parts {
    pub config: TrainingConfig,
    pub graph_json: String,
    pub analysis: Analysis,
    pub base_dir: PathBuf,
    pub run_dir: Option<PathBuf>,
    pub params: TensorMap,
    pub clock: Arc<dyn Clock>,
}

/// A new run (§2.10).
pub(super) fn new_core(
    graph: &Graph,
    config: TrainingConfig,
    options: TrainerOptions,
    resolve: DeviceResolver<'_>,
    cancel: &CancellationToken,
) -> Result<Core, TrainError> {
    // Step 1.
    config.validate()?;
    check_cancel(cancel)?;
    // Step 2.
    let analysis = check_graph(graph)?;
    check_cancel(cancel)?;
    let base_dir = absolute_base(options.base_dir.as_deref())?;
    // Steps 3–6.
    let prepared = prepare(
        graph,
        &analysis,
        &config,
        &base_dir,
        options.datasets,
        resolve,
        cancel,
    )?;
    // Step 7.
    let weights = match &options.weights {
        Some(path) => read_weights(path, &analysis)?,
        None => BTreeMap::new(),
    };
    check_values(&analysis, &weights)?;
    check_cancel(cancel)?;
    // Step 8.
    let run_dir = match &config.checkpoint {
        Some(c) => {
            let dir = base_dir.join(&c.dir);
            prepare_run_dir(&dir)?;
            Some(dir)
        }
        None => None,
    };
    check_cancel(cancel)?;
    // Step 9: initial values on the host, then every param uploaded.
    let mut weights = weights;
    let mut params = TensorMap::new();
    for p in &analysis.params {
        let bytes = match weights.remove(&p.name) {
            Some(b) => b,
            None => {
                let init = p.init.as_ref().ok_or_else(|| {
                    TrainError::internal(format!("param `{}` has no value", p.name))
                })?;
                let values = initial_values(config.seed, &p.name, &p.shape, init, p.fans)
                    .map_err(|e| TrainError::invalid(e.to_string()).with_tensor(p.name.clone()))?;
                f32_bytes(&values)?
            }
        };
        params.insert(
            p.name.clone(),
            upload_f32(&p.name, &p.shape, bytes, prepared.device)?,
        );
    }
    let clock = options
        .clock
        .unwrap_or_else(|| Arc::new(SystemClock::default()) as Arc<dyn Clock>);
    Core::build(
        prepared,
        Parts {
            config,
            graph_json: graph.to_json(),
            analysis,
            base_dir,
            run_dir,
            params,
            clock,
        },
    )
}

/// A resume (§5.4).
pub(super) fn resume_core(
    path: &Path,
    options: ResumeOptions,
    resolve: DeviceResolver<'_>,
    cancel: &CancellationToken,
) -> Result<Core, TrainError> {
    let dir = checkpoint::locate(path)?;
    let Loaded {
        dir,
        run_dir,
        state,
        graph,
        analysis,
        weights,
        optimizer,
    } = checkpoint::load(&dir)?;
    check_cancel(cancel)?;
    if let Some(entry) = analysis.state.first() {
        return Err(TrainError::new(
            TrainErrorKind::UnsupportedOperation,
            "the checkpoint's graph has state, which training doesn't support",
        )
        .with_tensor(entry.name.clone())
        .with_file(dir.join(checkpoint::GRAPH_FILE)));
    }
    let mut config = state.config.clone();
    if let Some(device) = &options.device {
        config.device = device.clone();
    }
    let base_dir = match &options.base_dir {
        Some(b) => absolute_base(Some(b))?,
        None => PathBuf::from(&state.base_dir),
    };
    let prepared = prepare(
        &graph,
        &analysis,
        &config,
        &base_dir,
        options.datasets,
        resolve,
        cancel,
    )
    .map_err(|e| {
        // The config comes from the checkpoint's state.json.
        e.prefixed("the checkpoint's config doesn't fit its data or graph: ")
            .or_file(&dir.join(checkpoint::STATE_FILE))
    })?;
    let val_len = prepared
        .datasets
        .validation
        .as_ref()
        .map_or(0, Dataset::len);
    if prepared.datasets.train.len() != state.train_samples
        || val_len != state.validation_samples
        || prepared.datasets.fingerprint_hex() != state.data_fingerprint
    {
        return Err(TrainError::invalid(format!(
            "the training data changed since the checkpoint {} (it was {} training and {} \
             validation samples with fingerprint {}; now {} and {} with {})",
            dir.display(),
            state.train_samples,
            state.validation_samples,
            state.data_fingerprint,
            prepared.datasets.train.len(),
            val_len,
            prepared.datasets.fingerprint_hex()
        ))
        .with_file(&dir));
    }
    if prepared.batcher.steps_per_epoch() != state.steps_per_epoch {
        return Err(TrainError::invalid(format!(
            "{}: `steps_per_epoch` doesn't fit the data",
            dir.display()
        ))
        .with_file(&dir));
    }
    check_cancel(cancel)?;
    // Step 6: newer checkpoints of the run directory.
    let newer: Vec<_> = checkpoint::checkpoints(&run_dir)?
        .into_iter()
        .filter(|e| e.step > state.step)
        .collect();
    if let Some(first) = newer.first()
        && !options.discard_newer
    {
        return Err(TrainError::invalid(format!(
            "the run directory {} holds checkpoints newer than step {} ({}); resume from the \
             newest one, or set `discard_newer` to replace them",
            run_dir.display(),
            state.step,
            first.path.display()
        ))
        .with_file(&first.path));
    }
    // Upload.
    let mut params = TensorMap::new();
    for (name, shape, bytes) in weights {
        let t = upload_f32(&name, &shape, bytes, prepared.device)?;
        params.insert(name, t);
    }
    let mut entries = Vec::with_capacity(optimizer.len());
    for (param, state_name, shape, bytes) in optimizer {
        let tensor = upload_f32(&param, &shape, bytes, prepared.device)?;
        entries.push(StateTensor {
            param,
            state: state_name,
            tensor,
        });
    }
    let clock = options
        .clock
        .unwrap_or_else(|| Arc::new(SystemClock::default()) as Arc<dyn Clock>);
    let mut core = Core::build(
        prepared,
        Parts {
            config,
            graph_json: graph.to_json(),
            analysis,
            base_dir,
            run_dir: Some(run_dir),
            params,
            clock,
        },
    )?;
    let trainable: TensorMap = core
        .params
        .iter()
        .filter(|(n, _)| core.trainable.contains(*n))
        .map(|(n, t)| (n.clone(), t.clone()))
        .collect();
    core.optimizer
        .import_state(&trainable, entries)
        .map_err(|e| TrainError::from(e).or_file(&dir.join(checkpoint::OPTIMIZER_FILE)))?;
    core.restore(&state, &dir, !newer.is_empty())?;
    Ok(core)
}

impl Core {
    /// Assembles a trainer at step 0 from what creation checked and uploaded.
    pub(super) fn build(prepared: Prepared, parts: Parts) -> Result<Self, TrainError> {
        let Prepared {
            device,
            plan,
            datasets,
            batcher,
            val_batcher,
            loss,
            metrics,
            input_fields,
        } = prepared;
        let config = parts.config;
        let steps_per_epoch = batcher.steps_per_epoch();
        let total_steps = config
            .epochs
            .checked_mul(steps_per_epoch)
            .ok_or_else(|| TrainError::internal("too many steps".to_owned()))?;
        let optimizer = Optimizer::new(config.optimizer.clone())
            .map_err(|e| TrainError::internal(e.to_string()))?;
        let schedule = Schedule::new(config.schedule, config.optimizer.lr(), total_steps)
            .map_err(|e| TrainError::internal(e.to_string()))?;
        let trainable: Vec<(String, Vec<usize>)> = parts
            .analysis
            .trainable_params()
            .map(|p| (p.name.clone(), p.shape.clone()))
            .collect();
        let param_order = parts
            .analysis
            .params
            .iter()
            .map(|p| p.name.clone())
            .collect();
        Ok(Core {
            fingerprint: datasets.fingerprint_hex(),
            trainable: {
                // Sorted, for `binary_search` in the loop.
                let mut names: Vec<String> = trainable.iter().map(|(n, _)| n.clone()).collect();
                names.sort_unstable();
                names
            },
            trainable_shapes: trainable,
            param_order,
            config,
            graph_json: parts.graph_json,
            base_dir: parts.base_dir,
            run_dir: parts.run_dir,
            device,
            plan,
            datasets,
            batcher,
            val_batcher,
            loss,
            metrics,
            input_fields,
            params: parts.params,
            optimizer,
            schedule,
            steps_per_epoch,
            total_steps,
            step: 0,
            epoch: 0,
            batch: 0,
            train_loss: LossMetric::new(),
            history: Vec::new(),
            best: None,
            elapsed_before: 0.0,
            last_checkpoint: None,
            last_checkpoint_pos: None,
            discard_newer_than: None,
            values: BTreeMap::new(),
            clock: parts.clock,
        })
    }

    /// Restores the counters, sums, history and best of a checkpoint (§5.4, "What resume
    /// restores").
    fn restore(
        &mut self,
        state: &checkpoint::State,
        dir: &Path,
        newer: bool,
    ) -> Result<(), TrainError> {
        self.step = state.step;
        self.epoch = state.epoch;
        self.batch = state.batch;
        self.train_loss = LossMetric::from_sums(state.sums.loss_sum, state.sums.samples);
        for (acc, (_, correct, counted)) in self.metrics.iter_mut().zip(&state.sums.accuracy) {
            if !acc.restore(*correct, *counted) {
                return Err(TrainError::invalid("inconsistent accuracy counts").with_file(dir));
            }
        }
        self.history = state.history.clone();
        self.best = state.best.clone();
        self.elapsed_before = state.elapsed_seconds;
        self.last_checkpoint = Some(dir.to_path_buf());
        self.last_checkpoint_pos = Some((state.step, state.epoch, state.batch));
        if newer {
            self.discard_newer_than = Some(state.step);
        }
        let value = checkpoint::epoch_value(
            &state.config,
            &state.history,
            state.step,
            state.epoch,
            state.batch,
        );
        self.values.insert(state.step, value);
        if let Some(best) = &state.best
            && let Some(step) = checkpoint::parse_checkpoint_name(&best.checkpoint)
        {
            self.values
                .insert(step, Some((best.value, best.epoch, best.step)));
        }
        Ok(())
    }
}
