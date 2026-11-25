//! Training: [`Engine::trainer`], [`Engine::resume_trainer`], [`Trainer`] and the training config,
//! options, events and outcome (public-api.md §1.4, training.md).
//!
//! A thin layer over `polypous-train`, which owns every behavior: the checks of training.md
//! §2.10, the loop, events, cancellation, checkpoints, resume and export. The facade adds:
//!
//! - its own types for what shows a device or has methods that return [`Error`]
//!   ([`TrainingConfig`], [`TrainerOptions`], [`ResumeOptions`], [`InMemoryDataset`],
//!   [`TrainingEvent`], [`TrainingOutcome`], [`TrainerInfo`], [`Trainer`]), converting to and from
//!   the train crate's; the nested plain-data types (the config's objects, metrics, history) are
//!   `polypous-train`'s, re-exported unchanged here;
//! - the device resolver: `"auto"` and device ids, resolved against the engine's device list as
//!   [`Engine::load_graph`] does it (training.md §2.2, §2.10 step 3);
//! - the error mapping: [`polypous_train::TrainErrorKind`] 1:1 to [`ErrorKind`], with every
//!   context field, the message prefix and `context.operation` of public-api.md §1.4;
//! - panics caught at the boundary (public-api.md §2 rule 8), including a panic in the caller's
//!   `on_event` callback.
//!
//! There is no shared state: every trainer owns its params, optimizer state and datasets, and the
//! facade installs no signal handler or other process-global hook (§2 rule 9).

pub use polypous_train::data::{
    ColumnSource, CsvColumnConfig, CsvDatasetConfig, CsvDtype, ImageFolderConfig,
    SafetensorsDatasetConfig, SplitConfig, TensorsDatasetConfig,
};
pub use polypous_train::{
    AccuracyConfig, AdamConfig, AdamWConfig, Best, BinaryCrossEntropyConfig, CheckpointConfig,
    CrossEntropyConfig, DataConfig, DatasetConfig, GradientClipConfig, HistoryEntry, LossConfig,
    MetricConfig, Metrics, MseConfig, OptimizerConfig, ScheduleConfig, SgdConfig, StopReason,
    TrainableParam,
};

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use polypous_backend::device as backend;
use polypous_train::{Number, TrainError, TrainErrorKind};
use serde::{Deserialize, Serialize, Serializer};

use crate::cancel::CancellationToken;
use crate::device::{self, DeviceId};
use crate::engine::{Engine, object_from_json};
use crate::error::{Error, ErrorKind, guard};
use crate::graph::Graph;
use crate::model::sentence;
use crate::tensor::NamedTensor;

/// The most [`InMemoryDataset`]s of [`TrainerOptions::datasets`] and
/// [`ResumeOptions::datasets`] (public-api.md §1.4).
pub const MAX_DATASETS: usize = 64;

/// The message prefix of [`Engine::trainer`]'s errors.
const CREATE: &str = "Could not create the trainer";
/// The message prefix of [`Engine::resume_trainer`]'s errors.
const RESUME: &str = "Could not resume training";
/// The message prefix of [`Trainer::run`]'s errors.
const RUN: &str = "Training failed";
/// The message prefix of [`Trainer::export`]'s errors.
const EXPORT: &str = "Could not export the model";

// ---- errors ----

/// The facade's kind for a train crate kind: the same name (training.md §6).
fn error_kind(kind: TrainErrorKind) -> ErrorKind {
    match kind {
        TrainErrorKind::InvalidInput => ErrorKind::InvalidInput,
        TrainErrorKind::UnsupportedOperation => ErrorKind::UnsupportedOperation,
        TrainErrorKind::BackendUnavailable => ErrorKind::BackendUnavailable,
        TrainErrorKind::Io => ErrorKind::Io,
        TrainErrorKind::OutOfMemory => ErrorKind::OutOfMemory,
        TrainErrorKind::Backend => ErrorKind::Backend,
        TrainErrorKind::Cancelled => ErrorKind::Cancelled,
        // `Internal`, and any kind added later: a bug until the facade maps it.
        _ => ErrorKind::Internal,
    }
}

/// `error` with `message` and every context field of `source`.
fn with_context(message: String, source: &TrainError) -> Error {
    let mut error = Error::new(error_kind(source.kind()), sentence(message));
    if let Some(pointer) = source.pointer() {
        error = error.with_pointer(pointer);
    }
    if let Some(tensor) = source.tensor() {
        error = error.with_tensor(tensor);
    }
    if let Some(node) = source.node() {
        error = error.with_node(node);
    }
    if let Some(file) = source.file() {
        error = error.with_file(file);
    }
    if let Some(device) = source.device() {
        error = error.with_device(device);
    }
    if let Some(details) = source.details() {
        error = error.with_details(details);
    }
    error
}

/// An error of a trainer operation: `<prefix>: <description>.` (public-api.md §1.4, "Errors").
fn train_error(prefix: &str, source: &TrainError) -> Error {
    with_context(format!("{prefix}: {}", source.message()), source)
}

/// An error of reading a training config (as [`Graph::from_json`] reports a graph): `The
/// training config is not valid: <pointer>: <description>.`, `context.pointer` and, from a file,
/// `context.file`.
fn config_error(source: &TrainError) -> Error {
    let message = match source.kind() {
        TrainErrorKind::InvalidInput => {
            format!("The training config is not valid: {}", source.message())
        }
        TrainErrorKind::UnsupportedOperation => {
            format!("The training config is not supported: {}", source.message())
        }
        _ => capitalized(source.message()),
    };
    with_context(message, source)
}

fn capitalized(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

/// [`guard`] for a trainer operation: errors are `<prefix>: …` (a caught panic too) with
/// `context.operation`.
fn train_guard<T>(
    operation: &str,
    prefix: &str,
    body: impl FnOnce() -> Result<T, Error>,
) -> Result<T, Error> {
    guard(operation, body).map_err(|mut error| {
        if !error.message.starts_with(prefix) {
            error.message = sentence(format!("{prefix}: {}", error.message));
        }
        error
    })
}

// ---- the config ----

/// A training config (training.md §2): plain data, its fields public.
///
/// Read it with [`TrainingConfig::from_json`] or [`TrainingConfig::from_file`]; a value built or
/// changed in Rust is checked exactly as a document when the trainer is created ([`Engine::trainer`],
/// training.md §2.10 step 1). `format` and `format_version` are not fields: they are always
/// `"polypous.training"` and `1`. The nested types are in [`crate::training`].
///
/// JSON: the document of training.md §2; [`TrainingConfig::to_json`] is canonical (every field,
/// defaults filled in, in the order of the spec's tables), and `from_json(&c.to_json()) == c`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    /// Keys every random stream (training.md §1.3). Default `0`.
    pub seed: u64,
    /// `"auto"` (the primary GPU, else the CPU) or a device id (`"cpu"`, `"cuda:0"`,
    /// `"vulkan:discrete_gpu:0"`, [`DeviceId::parse`]). Default `"auto"`.
    pub device: String,
    /// `"f32"` only in M2. Default `"f32"`.
    pub precision: String,
    /// 1–10,000. Required.
    pub epochs: u64,
    /// 1–65,536. Required.
    pub batch_size: u64,
    /// The datasets (training.md §2.3). Required.
    pub data: DataConfig,
    /// Graph input → dataset field (training.md §2.5). Required.
    pub inputs: BTreeMap<String, String>,
    /// The loss (training.md §2.6). Required.
    pub loss: LossConfig,
    /// The metrics (training.md §2.7). Default empty.
    pub metrics: Vec<MetricConfig>,
    /// The optimizer (training.md §2.8). Required.
    pub optimizer: OptimizerConfig,
    /// The learning-rate schedule (training.md §2.8). Default `constant`.
    pub schedule: ScheduleConfig,
    /// Gradient clipping (training.md §3.1.4), or none. Default none.
    pub gradient_clip: Option<GradientClipConfig>,
    /// Checkpoints (training.md §2.9), or none. Default none.
    pub checkpoint: Option<CheckpointConfig>,
}

impl TrainingConfig {
    /// Parses and checks a config document (training.md §2.1–§2.9). The checks against a graph
    /// and data come when the trainer is created (§2.10).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] with `context.pointer` (RFC 6901; `""` for the whole document:
    /// not JSON, not an object, over 1 MiB or nested deeper than 32) for a malformed document or an
    /// unknown, missing, mistyped or out-of-range field; [`ErrorKind::UnsupportedOperation`] at
    /// `/precision` for `f16`/`bf16`.
    pub fn from_json(json: &str) -> Result<TrainingConfig, Error> {
        guard("training_config_from_json", || {
            polypous_train::TrainingConfig::from_json(json)
                .map(TrainingConfig::from_train)
                .map_err(|error| config_error(&error))
        })
    }

    /// Reads a config file (at most 1 MiB is read), then [`TrainingConfig::from_json`].
    ///
    /// # Errors
    ///
    /// [`ErrorKind::Io`] with `context.file` if the file can't be read; [`ErrorKind::InvalidInput`]
    /// for a file over 1 MiB or not UTF-8; otherwise as [`TrainingConfig::from_json`], with
    /// `context.file`.
    pub fn from_file(path: impl AsRef<Path>) -> Result<TrainingConfig, Error> {
        let path = path.as_ref();
        guard("training_config_from_file", || {
            polypous_train::TrainingConfig::from_file(path)
                .map(TrainingConfig::from_train)
                .map_err(|error| config_error(&error))
        })
    }

    /// The canonical JSON form (training.md §2.1): every field, with defaults filled in, in the
    /// order of the spec's tables.
    pub fn to_json(&self) -> String {
        self.to_train().to_json()
    }

    pub(crate) fn from_train(config: polypous_train::TrainingConfig) -> TrainingConfig {
        let polypous_train::TrainingConfig {
            seed,
            device,
            precision,
            epochs,
            batch_size,
            data,
            inputs,
            loss,
            metrics,
            optimizer,
            schedule,
            gradient_clip,
            checkpoint,
        } = config;
        TrainingConfig {
            seed,
            device,
            precision,
            epochs,
            batch_size,
            data,
            inputs,
            loss,
            metrics,
            optimizer,
            schedule,
            gradient_clip,
            checkpoint,
        }
    }

    pub(crate) fn into_train(self) -> polypous_train::TrainingConfig {
        let TrainingConfig {
            seed,
            device,
            precision,
            epochs,
            batch_size,
            data,
            inputs,
            loss,
            metrics,
            optimizer,
            schedule,
            gradient_clip,
            checkpoint,
        } = self;
        polypous_train::TrainingConfig {
            seed,
            device,
            precision,
            epochs,
            batch_size,
            data,
            inputs,
            loss,
            metrics,
            optimizer,
            schedule,
            gradient_clip,
            checkpoint,
        }
    }

    fn to_train(&self) -> polypous_train::TrainingConfig {
        self.clone().into_train()
    }
}

/// The canonical JSON form, fields in table order (the same as [`TrainingConfig::to_json`]).
#[derive(Serialize)]
struct ConfigView<'a> {
    format: &'static str,
    format_version: u64,
    seed: u64,
    device: &'a str,
    precision: &'a str,
    epochs: u64,
    batch_size: u64,
    data: &'a DataConfig,
    inputs: &'a BTreeMap<String, String>,
    loss: &'a LossConfig,
    metrics: &'a [MetricConfig],
    optimizer: &'a OptimizerConfig,
    schedule: &'a ScheduleConfig,
    gradient_clip: &'a Option<GradientClipConfig>,
    checkpoint: &'a Option<CheckpointConfig>,
}

impl Serialize for TrainingConfig {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        ConfigView {
            format: polypous_train::training_config::FORMAT,
            format_version: polypous_train::training_config::FORMAT_VERSION,
            seed: self.seed,
            device: &self.device,
            precision: &self.precision,
            epochs: self.epochs,
            batch_size: self.batch_size,
            data: &self.data,
            inputs: &self.inputs,
            loss: &self.loss,
            metrics: &self.metrics,
            optimizer: &self.optimizer,
            schedule: &self.schedule,
            gradient_clip: &self.gradient_clip,
            checkpoint: &self.checkpoint,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TrainingConfig {
    /// Through [`TrainingConfig::from_json`], so every check applies. A key given twice can't be
    /// detected here (the value is read first); `from_json` detects it.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        polypous_train::TrainingConfig::from_json(&value.to_string())
            .map(TrainingConfig::from_train)
            .map_err(|error| serde::de::Error::custom(error.message()))
    }
}

// ---- options ----

/// A `tensors` dataset (training.md §3.5.1): named fields, dim 0 of each the sample dim.
///
/// JSON: `{"name":"train","fields":[NamedTensor JSON, …]}` (public-api.md §1.3). Wrappers should
/// pass tensors natively, not as JSON. The fields are checked when the trainer reads the dataset
/// (training.md §2.10 step 5): valid [`TensorData`](crate::TensorData) (§2 rule 4), rank 1–7,
/// the same sample count, unique names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InMemoryDataset {
    /// The name a `{"kind": "tensors", "name": …}` dataset selects it by.
    pub name: String,
    /// The fields.
    pub fields: Vec<NamedTensor>,
}

impl InMemoryDataset {
    /// The train crate's host data. The bytes are moved, not copied or checked: the trainer
    /// checks the fields it uses at §2.10 step 5, in its order.
    fn into_train(self) -> polypous_train::InMemoryDataset {
        polypous_train::InMemoryDataset {
            name: self.name,
            fields: self
                .fields
                .into_iter()
                .map(|named| {
                    let data = polypous_backend::TensorData {
                        dtype: named.tensor.dtype.to_backend(),
                        shape: named.tensor.shape,
                        bytes: named.tensor.bytes,
                    };
                    (named.name, data)
                })
                .collect(),
        }
    }
}

/// Checks `list` ([`check_datasets`]) and converts it.
fn datasets(list: Vec<InMemoryDataset>) -> Result<Vec<polypous_train::InMemoryDataset>, String> {
    check_datasets(&list)?;
    Ok(list.into_iter().map(InMemoryDataset::into_train).collect())
}

/// The dataset error of [`datasets`] as `prefix: …` (`InvalidInput`).
fn datasets_error(prefix: &str, why: String) -> Error {
    Error::new(
        ErrorKind::InvalidInput,
        sentence(format!("{prefix}: {why}")),
    )
}

/// What a new training run needs besides the graph and the config (public-api.md §1.4).
///
/// JSON: an object, every field optional, unknown fields rejected:
/// `{"weights":"pretrained.safetensors","base_dir":"/data","datasets":[…]}`. The struct is
/// `#[non_exhaustive]`: start from `TrainerOptions::default()` and set fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct TrainerOptions {
    /// Initial weights: a safetensors file or a sharded checkpoint's index (formats.md §1.5),
    /// holding any subset of the params; the others use their `init` (training.md §2.10 step 7).
    /// Used as given (not joined to `base_dir`). Default `None`: every param uses its `init`.
    pub weights: Option<PathBuf>,
    /// The directory relative paths in the config are resolved against. Default `None`: the
    /// process's current directory.
    pub base_dir: Option<PathBuf>,
    /// The `tensors` datasets (training.md §3.5.1): at most 64, names unique. Default empty.
    pub datasets: Vec<InMemoryDataset>,
}

impl TrainerOptions {
    /// Reads options from their JSON form (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not a JSON object of known fields, or has more
    /// than 64 datasets or two with the same name.
    pub fn from_json(json: &str) -> Result<TrainerOptions, Error> {
        guard("trainer_options_from_json", || {
            const MESSAGE: &str = "The trainer options are not valid";
            let options: TrainerOptions = object_from_json(json, &format!("{MESSAGE}."))?;
            check_datasets(&options.datasets).map_err(|why| datasets_error(MESSAGE, why))?;
            Ok(options)
        })
    }

    /// The options' JSON form.
    pub fn to_json(&self) -> String {
        // Plain data; a path that is not UTF-8 can't be written: fall back to the defaults' form.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

/// At most [`MAX_DATASETS`] datasets, names unique (public-api.md §1.4).
fn check_datasets(list: &[InMemoryDataset]) -> Result<(), String> {
    if list.len() > MAX_DATASETS {
        return Err(format!(
            "{} in-memory datasets were given; at most {MAX_DATASETS} are allowed",
            list.len()
        ));
    }
    let mut names = BTreeSet::new();
    for dataset in list {
        if !names.insert(dataset.name.as_str()) {
            return Err(format!(
                "two in-memory datasets are named `{}`; names must be unique",
                dataset.name
            ));
        }
    }
    Ok(())
}

/// What a resume needs besides the checkpoint (public-api.md §1.4, training.md §5.4).
///
/// JSON: an object, every field optional, unknown fields rejected:
/// `{"device":"cpu","base_dir":null,"datasets":[],"discard_newer":false}`. The struct is
/// `#[non_exhaustive]`: start from `ResumeOptions::default()` and set fields.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct ResumeOptions {
    /// A device id (`"cpu"`, `"cuda:0"`) or `"auto"` that replaces the config's `device`.
    /// Default `None`: the checkpoint's config decides.
    pub device: Option<String>,
    /// Replaces the checkpoint's `base_dir` for the datasets' relative paths. Default `None`.
    pub base_dir: Option<PathBuf>,
    /// The `tensors` datasets again (as [`TrainerOptions::datasets`]). Default empty.
    pub datasets: Vec<InMemoryDataset>,
    /// Delete the run directory's checkpoints newer than the one resumed when the first new
    /// checkpoint is written; otherwise they are an error (training.md §5.4 step 6). Default
    /// `false`.
    pub discard_newer: bool,
}

impl ResumeOptions {
    /// Reads options from their JSON form (public-api.md §2 rule 3).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if `json` is not a JSON object of known fields, or has more
    /// than 64 datasets or two with the same name.
    pub fn from_json(json: &str) -> Result<ResumeOptions, Error> {
        guard("resume_options_from_json", || {
            const MESSAGE: &str = "The resume options are not valid";
            let options: ResumeOptions = object_from_json(json, &format!("{MESSAGE}."))?;
            check_datasets(&options.datasets).map_err(|why| datasets_error(MESSAGE, why))?;
            Ok(options)
        })
    }

    /// The options' JSON form.
    pub fn to_json(&self) -> String {
        // Plain data; a path that is not UTF-8 can't be written: fall back to the defaults' form.
        serde_json::to_string(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---- events, outcome, info ----

fn number<S: Serializer>(value: &f64, serializer: S) -> Result<S::Ok, S::Error> {
    Number(*value).serialize(serializer)
}

fn to_json<T: Serialize>(value: &T) -> String {
    // Numbers are written by `Number` (non-finite values as strings), map keys are strings: the
    // types here always serialize.
    serde_json::to_string(value).unwrap_or_default()
}

/// What [`Trainer::run`] returns and the `finished` event carries (public-api.md §1.4).
///
/// JSON: `{"stop_reason":"completed","step":4220,"epochs_completed":5,"final_metrics":{…},
/// "best":{…} or null,"history":[…],"last_checkpoint":"runs/mnist/step-000000004220" or null,
/// "elapsed_seconds":812.5}`. Numbers follow training.md §4.1 (non-finite values as `"nan"`,
/// `"inf"`, `"-inf"`). Readers must accept new fields.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[non_exhaustive]
pub struct TrainingOutcome {
    /// `completed` or `cancelled`.
    pub stop_reason: StopReason,
    /// Optimizer steps completed.
    pub step: u64,
    /// Epochs finished (their validation pass included).
    pub epochs_completed: u64,
    /// The last finished epoch's metrics, in the reported order; empty if none finished.
    pub final_metrics: Metrics,
    /// The best checkpoint (its path in `checkpoint`); `None` without `checkpoint.keep_best`.
    pub best: Option<Best>,
    /// Every finished epoch.
    pub history: Vec<HistoryEntry>,
    /// The newest checkpoint's path, if any.
    pub last_checkpoint: Option<String>,
    /// Wall-clock seconds of every run of this training (informative).
    #[serde(serialize_with = "number")]
    pub elapsed_seconds: f64,
}

impl TrainingOutcome {
    /// The JSON form (public-api.md §1.4).
    pub fn to_json(&self) -> String {
        to_json(self)
    }

    fn from_train(outcome: polypous_train::TrainingOutcome) -> TrainingOutcome {
        let polypous_train::TrainingOutcome {
            stop_reason,
            step,
            epochs_completed,
            final_metrics,
            best,
            history,
            last_checkpoint,
            elapsed_seconds,
        } = outcome;
        TrainingOutcome {
            stop_reason,
            step,
            epochs_completed,
            final_metrics,
            best,
            history,
            last_checkpoint,
            elapsed_seconds,
        }
    }
}

/// A training event (training.md §4.1), in the order of §4.2; `Finished` is the last one of a
/// run that returns `Ok`.
///
/// JSON: an object with an `"event"` tag, `{"event":"started","device":{"backend":"cpu"},…}`.
/// New kinds may be added: match with a wildcard arm, and readers of the JSON must accept new
/// kinds and fields.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrainingEvent {
    /// The run starts.
    Started {
        /// The training device (DeviceId JSON, public-api.md §1.2).
        device: DeviceId,
        /// `epochs`.
        epochs: u64,
        /// `S`, optimizer steps per epoch.
        steps_per_epoch: u64,
        /// `T = epochs · S`.
        total_steps: u64,
        /// Training samples (after a split).
        train_samples: u64,
        /// Validation samples, `0` without a validation set.
        validation_samples: u64,
        /// The number of values in the trainable params.
        trainable_elements: u64,
        /// The step the run starts at: `0`, or where a resumed or earlier run stopped.
        step: u64,
    },
    /// An epoch starts, or continues.
    EpochStarted {
        /// The epoch, 0-based.
        epoch: u64,
        /// The first batch this run trains in the epoch (`0`, or the resume position).
        batch: u64,
    },
    /// An optimizer step finished (throttled to about 10 per second, training.md §4.2).
    Step {
        /// The epoch.
        epoch: u64,
        /// The batch of the epoch.
        batch: u64,
        /// The step's 0-based index `k`.
        step: u64,
        /// The batch loss.
        #[serde(serialize_with = "number")]
        loss: f64,
        /// The learning rate `lr(k)`.
        #[serde(serialize_with = "number")]
        lr: f64,
    },
    /// The validation pass of an epoch starts.
    ValidationStarted {
        /// The epoch.
        epoch: u64,
    },
    /// An epoch finished.
    EpochFinished {
        /// The epoch.
        epoch: u64,
        /// Steps completed.
        step: u64,
        /// Reported names → values, in the order of training.md §2.7.
        metrics: Metrics,
    },
    /// A checkpoint was written.
    CheckpointSaved {
        /// The checkpoint directory, as resolved.
        path: String,
        /// Its step.
        step: u64,
    },
    /// A checkpoint was deleted (`keep_last`, or a newer checkpoint discarded on resume).
    CheckpointRemoved {
        /// The checkpoint directory.
        path: String,
    },
    /// The run returned `Ok`: the last event.
    Finished {
        /// What `run` returns.
        outcome: Box<TrainingOutcome>,
    },
}

impl TrainingEvent {
    /// The JSON form (training.md §4.1).
    pub fn to_json(&self) -> String {
        to_json(self)
    }

    /// The facade's event for the train crate's; `None` for a kind the facade doesn't know yet.
    fn from_train(event: polypous_train::TrainingEvent) -> Option<TrainingEvent> {
        use polypous_train::TrainingEvent as T;
        Some(match event {
            T::Started {
                device,
                epochs,
                steps_per_epoch,
                total_steps,
                train_samples,
                validation_samples,
                trainable_elements,
                step,
            } => TrainingEvent::Started {
                device: device::from_backend_id(device),
                epochs,
                steps_per_epoch,
                total_steps,
                train_samples,
                validation_samples,
                trainable_elements,
                step,
            },
            T::EpochStarted { epoch, batch } => TrainingEvent::EpochStarted { epoch, batch },
            T::Step {
                epoch,
                batch,
                step,
                loss,
                lr,
            } => TrainingEvent::Step {
                epoch,
                batch,
                step,
                loss,
                lr,
            },
            T::ValidationStarted { epoch } => TrainingEvent::ValidationStarted { epoch },
            T::EpochFinished {
                epoch,
                step,
                metrics,
            } => TrainingEvent::EpochFinished {
                epoch,
                step,
                metrics,
            },
            T::CheckpointSaved { path, step } => TrainingEvent::CheckpointSaved { path, step },
            T::CheckpointRemoved { path } => TrainingEvent::CheckpointRemoved { path },
            T::Finished { outcome } => TrainingEvent::Finished {
                outcome: Box::new(TrainingOutcome::from_train(*outcome)),
            },
            other => {
                tracing::warn!(
                    event = other.name(),
                    "a training event the facade doesn't know was not passed on"
                );
                return None;
            }
        })
    }
}

/// What [`Trainer::info`] returns (public-api.md §1.4): the `started` event's fields with `step`
/// the steps completed so far, `finished`, and the trainable params.
///
/// JSON: `{"device":{"backend":"cpu"},"epochs":5,"steps_per_epoch":844,"total_steps":4220,
/// "train_samples":54000,"validation_samples":6000,"trainable_elements":1199882,"step":0,
/// "finished":false,"trainable":[{"name":"fc1.weight","shape":[128,784]},…]}`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[non_exhaustive]
pub struct TrainerInfo {
    /// The training device.
    pub device: DeviceId,
    /// `epochs`.
    pub epochs: u64,
    /// `S`.
    pub steps_per_epoch: u64,
    /// `T`.
    pub total_steps: u64,
    /// Training samples.
    pub train_samples: u64,
    /// Validation samples (`0` without a validation set).
    pub validation_samples: u64,
    /// The number of values in the trainable params.
    pub trainable_elements: u64,
    /// Steps completed so far.
    pub step: u64,
    /// Whether training has finished (every step and the last epoch's validation).
    pub finished: bool,
    /// The trainable params, in declaration order.
    pub trainable: Vec<TrainableParam>,
}

impl TrainerInfo {
    /// The JSON form.
    pub fn to_json(&self) -> String {
        to_json(self)
    }

    fn from_train(info: polypous_train::TrainerInfo) -> TrainerInfo {
        let polypous_train::TrainerInfo {
            device,
            epochs,
            steps_per_epoch,
            total_steps,
            train_samples,
            validation_samples,
            trainable_elements,
            step,
            finished,
            trainable,
        } = info;
        TrainerInfo {
            device: device::from_backend_id(device),
            epochs,
            steps_per_epoch,
            total_steps,
            train_samples,
            validation_samples,
            trainable_elements,
            step,
            finished,
            trainable,
        }
    }
}

// ---- the engine's side ----

impl Engine {
    /// The trainer's device resolver (training.md §2.2, §2.10 step 3): `"auto"` is the device
    /// [`Engine::load_graph`] picks by default (the primary GPU, else the CPU); any other text is
    /// a device id ([`DeviceId::parse`]). The device must be in [`Engine::devices`] and available;
    /// otherwise `BackendUnavailable` with `context.device`.
    fn train_device(&self, text: &str) -> Result<backend::DeviceId, TrainError> {
        let requested = if text == "auto" {
            None
        } else {
            let id = DeviceId::parse(text).map_err(|error| {
                TrainError::new(
                    TrainErrorKind::BackendUnavailable,
                    format!("`{text}` is not a device id"),
                )
                .with_device(text)
                .with_details(error.message)
            })?;
            Some(id)
        };
        self.pick_device(requested)
            .map(device::to_backend_id)
            .map_err(|(id, why)| {
                TrainError::new(
                    TrainErrorKind::BackendUnavailable,
                    format!("device {id} is not available: {why}"),
                )
                .with_device(id.to_string())
            })
    }

    /// Creates a trainer for a new run (public-api.md §1.4): checks the config against the graph
    /// and the data, reads the datasets, loads the initial weights, computes the initial values
    /// and uploads the params, in the order of training.md §2.10. Blocking; nothing trains yet.
    ///
    /// The device is the config's `device`: `"auto"` is the primary GPU, else the CPU (as
    /// [`Engine::load_graph`]); an id must be in [`Engine::devices`] and available. `cancel` is
    /// checked between the steps and between dataset files.
    ///
    /// # Errors
    ///
    /// The first failing check of training.md §2.10 (kinds of §6), with the message
    /// `Could not create the trainer: …` and `context.operation` `engine_trainer`;
    /// [`ErrorKind::InvalidInput`] for more than 64 datasets or two with the same name;
    /// [`ErrorKind::Cancelled`].
    pub fn trainer(
        &self,
        graph: &Graph,
        config: TrainingConfig,
        options: TrainerOptions,
        cancel: &CancellationToken,
    ) -> Result<Trainer, Error> {
        train_guard("engine_trainer", CREATE, || {
            let TrainerOptions {
                weights,
                base_dir,
                datasets: list,
            } = options;
            let datasets = datasets(list).map_err(|why| datasets_error(CREATE, why))?;
            let options = polypous_train::TrainerOptions {
                weights,
                base_dir,
                datasets,
                clock: None,
            };
            let resolve = |text: &str| self.train_device(text);
            let inner = polypous_train::Trainer::new(
                graph.ir(),
                config.into_train(),
                options,
                &resolve,
                cancel.runtime(),
            )
            .map_err(|error| train_error(CREATE, &error))?;
            let info = inner.info();
            tracing::info!(
                device = %info.device,
                steps = info.total_steps,
                step = info.step,
                "created a trainer"
            );
            Ok(Trainer { inner })
        })
    }

    /// A trainer restored from a checkpoint directory, or from the newest checkpoint of a run
    /// directory (public-api.md §1.4, training.md §5.4). The checkpoint is untrusted: every check
    /// runs before the data it protects is read. Blocking.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] with `context.file` for an invalid or inconsistent checkpoint,
    /// changed data or newer checkpoints (without [`ResumeOptions::discard_newer`]);
    /// [`ErrorKind::Io`]; the errors of training.md §2.10 steps 3–6; [`ErrorKind::Cancelled`].
    /// The message is `Could not resume training: …`, `context.operation`
    /// `engine_resume_trainer`.
    pub fn resume_trainer(
        &self,
        path: impl AsRef<Path>,
        options: ResumeOptions,
        cancel: &CancellationToken,
    ) -> Result<Trainer, Error> {
        let path = path.as_ref();
        train_guard("engine_resume_trainer", RESUME, || {
            let ResumeOptions {
                device,
                base_dir,
                datasets: list,
                discard_newer,
            } = options;
            let datasets = datasets(list).map_err(|why| datasets_error(RESUME, why))?;
            let options = polypous_train::ResumeOptions {
                device,
                base_dir,
                datasets,
                discard_newer,
                clock: None,
            };
            let resolve = |text: &str| self.train_device(text);
            let inner = polypous_train::Trainer::resume(path, options, &resolve, cancel.runtime())
                .map_err(|error| train_error(RESUME, &error))?;
            let info = inner.info();
            tracing::info!(
                checkpoint = %path.display(),
                device = %info.device,
                step = info.step,
                "resumed a trainer"
            );
            Ok(Trainer { inner })
        })
    }
}

/// A training run (public-api.md §1.4): the param store, the optimizer state and the datasets of
/// one graph and config.
///
/// A cheap-to-clone handle (`Arc` inside), `Send + Sync + 'static`. It is idle when created or
/// after a cancelled run, running during [`Trainer::run`], finished after the last epoch, failed
/// after `run` returned an error (or a panic, also one in the callback), and closed after
/// [`Trainer::close`]. Dropping the last clone frees everything.
#[derive(Clone)]
pub struct Trainer {
    inner: polypous_train::Trainer,
}

impl Trainer {
    /// The `started` event's fields with `step` the steps completed so far, `finished`, and the
    /// trainable params. Works in every state, also while running and after `close`.
    pub fn info(&self) -> TrainerInfo {
        TrainerInfo::from_train(self.inner.info())
    }

    /// Trains from the current step to the end, or until `cancel` is set (training.md §1.2, §4).
    /// Blocking. Events go to `on_event` in the order of training.md §4.2, never concurrently;
    /// `Finished` is the last one of a run that returns `Ok`.
    ///
    /// Returns `Ok` also when cancelled (`stop_reason: "cancelled"`): the current step finishes,
    /// a checkpoint is written if configured (§4.3), and calling `run` again continues. The
    /// callback must not call `run`, `export` or `close` expecting them to work: `run` and
    /// `export` are `InvalidInput` while running, and `close` takes effect when the run returns.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if the trainer is running (another clone, or its own
    /// callback), has finished, has failed, or was closed; during the run the kinds of
    /// training.md §6, after which the trainer is failed (resume from the last checkpoint).
    /// A panic, also one in `on_event`, is [`ErrorKind::Internal`] and leaves the trainer failed.
    /// The message is `Training failed: …`, `context.operation` `trainer_run`.
    pub fn run(
        &self,
        cancel: &CancellationToken,
        mut on_event: impl FnMut(TrainingEvent),
    ) -> Result<TrainingOutcome, Error> {
        train_guard("trainer_run", RUN, || {
            self.inner
                .run(cancel.runtime(), |event| {
                    if let Some(event) = TrainingEvent::from_train(event) {
                        on_event(event);
                    }
                })
                .map(TrainingOutcome::from_train)
                .map_err(|error| train_error(RUN, &error))
        })
    }

    /// Writes the current model (training.md §5.5): the graph's canonical JSON to `graph_path`
    /// and every declared param, running statistics included, as `F32` to `weights_path`, each
    /// through a temporary file and a rename. [`Engine::load_graph`] loads the pair; with compute
    /// dtype `f32` on the same device its outputs equal the trainer's inference-mode outputs.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] while running, after a failed run or after `close`;
    /// [`ErrorKind::Io`] with `context.file`; [`ErrorKind::Backend`]. The message is
    /// `Could not export the model: …`, `context.operation` `trainer_export`.
    pub fn export(
        &self,
        graph_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
    ) -> Result<(), Error> {
        let graph_path = graph_path.as_ref();
        let weights_path = weights_path.as_ref();
        train_guard("trainer_export", EXPORT, || {
            self.inner
                .export(graph_path, weights_path)
                .map_err(|error| train_error(EXPORT, &error))?;
            tracing::info!(
                graph = %graph_path.display(),
                weights = %weights_path.display(),
                "exported a trained model"
            );
            Ok(())
        })
    }

    /// Frees the params, the optimizer state and the datasets now (public-api.md §2 rule 2). A
    /// running `run` finishes first. Later `run` and `export` calls on any clone are
    /// [`ErrorKind::InvalidInput`]; `info` still works. Idempotent.
    pub fn close(&self) {
        let _ = guard("trainer_close", || {
            self.inner.close();
            Ok(())
        });
    }
}

impl fmt::Debug for Trainer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let info = self.inner.info();
        f.debug_struct("Trainer")
            .field("device", &device::from_backend_id(info.device))
            .field("step", &info.step)
            .field("finished", &info.finished)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every kind of training.md §6 maps to the facade kind of the same name, with every context
    /// field, the prefix and a full stop.
    #[test]
    fn errors_map_one_to_one_with_every_context_field() {
        let kinds = [
            (TrainErrorKind::InvalidInput, ErrorKind::InvalidInput),
            (
                TrainErrorKind::UnsupportedOperation,
                ErrorKind::UnsupportedOperation,
            ),
            (
                TrainErrorKind::BackendUnavailable,
                ErrorKind::BackendUnavailable,
            ),
            (TrainErrorKind::Io, ErrorKind::Io),
            (TrainErrorKind::OutOfMemory, ErrorKind::OutOfMemory),
            (TrainErrorKind::Backend, ErrorKind::Backend),
            (TrainErrorKind::Cancelled, ErrorKind::Cancelled),
            (TrainErrorKind::Internal, ErrorKind::Internal),
        ];
        for (train, facade) in kinds {
            assert_eq!(error_kind(train), facade);
            assert_eq!(train.as_str(), facade.as_str());
            for prefix in [CREATE, RESUME, RUN, EXPORT] {
                let source = TrainError::new(train, "it broke")
                    .with_pointer("/optimizer/lr")
                    .with_tensor("fc1.weight")
                    .with_node("layers[3]/bn")
                    .with_file("runs/x/state.json")
                    .with_device("cuda:0")
                    .with_details("the details");
                let error = train_error(prefix, &source);
                assert_eq!(error.kind, facade);
                assert_eq!(error.message, format!("{prefix}: it broke."));
                let c = &error.context;
                assert_eq!(c.pointer.as_deref(), Some("/optimizer/lr"));
                assert_eq!(c.tensor.as_deref(), Some("fc1.weight"));
                assert_eq!(c.node.as_deref(), Some("layers[3]/bn"));
                assert_eq!(c.file.as_deref(), Some("runs/x/state.json"));
                assert_eq!(c.device.as_deref(), Some("cuda:0"));
                assert_eq!(c.details.as_deref(), Some("the details"));
                // Absent fields stay absent.
                let bare = train_error(prefix, &TrainError::new(train, "x."));
                assert_eq!(bare.message, format!("{prefix}: x."));
                assert_eq!(bare.context, crate::ErrorContext::default());
            }
        }
    }

    #[test]
    fn a_panic_gets_the_prefix_and_the_operation() {
        let error =
            train_guard::<()>("trainer_run", RUN, || panic!("inside the callback")).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Internal);
        assert_eq!(
            error.message,
            "Training failed: Polypous ran into an internal error. Please report this as a bug."
        );
        assert_eq!(error.context.operation.as_deref(), Some("trainer_run"));
        assert_eq!(
            error.context.details.as_deref(),
            Some("panic: inside the callback")
        );
        // An error of the body keeps its message (already prefixed).
        let source = TrainError::new(TrainErrorKind::Io, "could not write x");
        let error = train_guard::<()>("trainer_export", EXPORT, || {
            Err(train_error(EXPORT, &source))
        })
        .unwrap_err();
        assert_eq!(
            error.message,
            "Could not export the model: could not write x."
        );
        assert_eq!(error.context.operation.as_deref(), Some("trainer_export"));
    }

    #[test]
    fn config_errors_follow_the_graph_convention() {
        let e = TrainError::new(TrainErrorKind::InvalidInput, "/epochs: out of range")
            .with_pointer("/epochs");
        let error = config_error(&e);
        assert_eq!(
            error.message,
            "The training config is not valid: /epochs: out of range."
        );
        assert_eq!(error.context.pointer.as_deref(), Some("/epochs"));
        let e = TrainError::new(TrainErrorKind::UnsupportedOperation, "/precision: f16");
        assert_eq!(
            config_error(&e).message,
            "The training config is not supported: /precision: f16."
        );
        let e = TrainError::new(TrainErrorKind::Io, "could not open the config a.json: gone");
        let error = config_error(&e);
        assert_eq!(error.kind, ErrorKind::Io);
        assert_eq!(error.message, "Could not open the config a.json: gone.");
    }

    fn outcome() -> polypous_train::TrainingOutcome {
        let metrics = Metrics(vec![
            ("train.loss".into(), 0.25),
            ("val.accuracy".into(), f64::NAN),
        ]);
        polypous_train::TrainingOutcome {
            stop_reason: StopReason::Cancelled,
            step: 12,
            epochs_completed: 1,
            final_metrics: metrics.clone(),
            best: Some(Best {
                metric: "train.loss".into(),
                value: f64::INFINITY,
                epoch: 0,
                step: 6,
                checkpoint: "runs/x/step-000000000006".into(),
            }),
            history: vec![HistoryEntry {
                epoch: 0,
                step: 6,
                metrics,
            }],
            last_checkpoint: Some("runs/x/step-000000000012".into()),
            elapsed_seconds: 1e-7,
        }
    }

    /// The facade's events, outcome and info write exactly the train crate's JSON (training.md
    /// §4.1), with the device as the facade's `DeviceId`.
    #[test]
    fn event_json_is_the_train_crates() {
        use polypous_train::TrainingEvent as T;
        let vulkan = backend::DeviceId::Vulkan {
            adapter: backend::AdapterKind::Cpu,
            index: 0,
        };
        let events = vec![
            T::Started {
                device: vulkan,
                epochs: 5,
                steps_per_epoch: 844,
                total_steps: 4220,
                train_samples: 54000,
                validation_samples: 6000,
                trainable_elements: 1_199_882,
                step: 3,
            },
            T::EpochStarted { epoch: 0, batch: 3 },
            T::Step {
                epoch: 0,
                batch: 3,
                step: 3,
                loss: 2.3051,
                lr: f64::NEG_INFINITY,
            },
            T::ValidationStarted { epoch: 0 },
            T::EpochFinished {
                epoch: 0,
                step: 844,
                metrics: Metrics(vec![("train.loss".into(), 0.2412)]),
            },
            T::CheckpointSaved {
                path: "runs/x/step-000000000844".into(),
                step: 844,
            },
            T::CheckpointRemoved {
                path: "runs/x/step-000000000422".into(),
            },
            T::Finished {
                outcome: Box::new(outcome()),
            },
        ];
        for event in events {
            let expected = event.to_json();
            let facade = TrainingEvent::from_train(event).unwrap();
            assert_eq!(facade.to_json(), expected);
            let json: serde_json::Value = serde_json::from_str(&expected).unwrap();
            assert!(json["event"].is_string(), "{json}");
        }
        let started = TrainingEvent::from_train(T::Started {
            device: vulkan,
            epochs: 1,
            steps_per_epoch: 1,
            total_steps: 1,
            train_samples: 1,
            validation_samples: 0,
            trainable_elements: 1,
            step: 0,
        })
        .unwrap();
        let TrainingEvent::Started { device, .. } = started else {
            panic!("{started:?}");
        };
        assert_eq!(device, DeviceId::parse("vulkan:cpu:0").unwrap());
        let json: serde_json::Value = serde_json::from_str(&started.to_json()).unwrap();
        assert_eq!(json["device"], serde_json::to_value(device).unwrap());

        let train = outcome();
        let expected = train.to_json();
        assert!(expected.contains(r#""value":"inf""#), "{expected}");
        assert!(expected.contains(r#""val.accuracy":"nan""#), "{expected}");
        assert_eq!(TrainingOutcome::from_train(train).to_json(), expected);

        let info = polypous_train::TrainerInfo {
            device: backend::DeviceId::Cuda { index: 1 },
            epochs: 2,
            steps_per_epoch: 3,
            total_steps: 6,
            train_samples: 40,
            validation_samples: 0,
            trainable_elements: 7,
            step: 4,
            finished: false,
            trainable: vec![TrainableParam {
                name: "w".into(),
                shape: vec![7],
            }],
        };
        let expected = info.to_json();
        let facade = TrainerInfo::from_train(info);
        assert_eq!(facade.device, DeviceId::Cuda { index: 1 });
        assert_eq!(facade.to_json(), expected);
    }
}
