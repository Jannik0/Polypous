//! Training for Polypous (`docs/specs/training.md`).
//!
//! Responsibility: the training config and its validation, the param store, initial values,
//! datasets, losses and metrics, the optimizers and schedules (Polypous code on `DynTensor`
//! ops, ADR-0014; no Burn dependency), the training loop, checkpoints and events. See
//! `docs/architecture.md` §2 and training.md §1.1.
//!
//! - [`init`]: the initial value of a param from its `init`, on the host (graph-ir.md §2.4.1).
//! - [`config`]: the `loss`, `metrics`, `optimizer`, `schedule` and `gradient_clip` objects of a
//!   training config (training.md §2.6–§2.8), with their defaults and range checks.
//! - [`optimizer`]: `sgd`, `adam` and `adamw` with their state, export and import (§3.1, §5.2).
//! - [`schedule`]: the learning-rate schedules (§3.2).
//! - [`clip`]: gradient clipping by the total L2 norm (§3.1.4).
//! - [`loss`]: `mse`, `cross_entropy` and `binary_cross_entropy`, differentiable (§3.3).
//! - [`metrics`]: the `loss` and `accuracy` epoch accumulators (§3.4).
//! - [`data`]: the datasets (`tensors`, `safetensors`, `csv`, `image_folder`, `split`), their
//!   config types and limits, and the seeded shuffling and batching (§2.3, §2.4, §3.5, §3.6).
//! - [`targets`]: the checks of a loss or metric's prediction and target, and of the target
//!   values, made when the trainer is created (§2.10 step 6).
//! - [`training_config`]: the root of the training config, [`TrainingConfig`] (§2), with its
//!   JSON form and every range check, with full pointers.
//! - [`trainer`]: the [`Trainer`] (§1.2, §2.10, §4, §5): creation with the checks of §2.10, the
//!   loop, events, cancellation, checkpoints, resume and export.
//! - [`events`]: [`TrainingEvent`], [`TrainingOutcome`], [`TrainerInfo`] and the clock (§4).
//! - [`checkpoint`]: the run directory, writing checkpoints atomically, `keep_last`/`keep_best`
//!   and loading an untrusted checkpoint (§5).
//! - [`error`]: [`TrainError`], with the kinds and context of §6.

pub mod checkpoint;
pub mod clip;
pub mod config;
pub mod data;
pub mod error;
pub mod events;
pub mod init;
pub(crate) mod json;
pub mod loss;
pub mod metrics;
pub mod optimizer;
pub mod schedule;
pub mod targets;
pub mod trainer;
pub mod training_config;

pub use clip::clip_gradients;
pub use config::{
    AccuracyConfig, AdamConfig, AdamWConfig, BinaryCrossEntropyConfig, ConfigError,
    CrossEntropyConfig, GradientClipConfig, LossConfig, MetricConfig, MseConfig, OptimizerConfig,
    ScheduleConfig, SgdConfig, validate_metrics,
};
pub use data::{
    Batcher, DataConfig, DataError, DataErrorKind, DataSources, Dataset, DatasetConfig, Datasets,
    FieldSelection, InMemoryDataset, read_datasets,
};
pub use error::{TrainError, TrainErrorKind};
pub use events::{
    Best, Clock, HistoryEntry, ManualClock, Metrics, StopReason, SystemClock, TrainableParam,
    TrainerInfo, TrainingEvent, TrainingOutcome,
};
pub use init::{InitError, initial_values};
pub use json::Number;
pub use loss::{Loss, LossError};
pub use metrics::{Accuracy, LossMetric, MetricError};
pub use optimizer::{Optimizer, OptimizerError, StateTensor, TensorMap};
pub use schedule::Schedule;
pub use targets::{TargetError, TargetValue};
pub use trainer::{DeviceResolver, ResumeOptions, Trainer, TrainerOptions};
pub use training_config::{CheckpointConfig, TrainingConfig};
