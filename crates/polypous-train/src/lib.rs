//! Training for Polypous (`docs/specs/training.md`).
//!
//! Responsibility: the training config and its validation, the param store, initial values,
//! datasets, losses and metrics, the optimizers and schedules (Polypous code on `DynTensor`
//! ops, ADR-0014; no Burn dependency), the training loop, checkpoints and events. See
//! `docs/architecture.md` §2 and training.md §1.1.
//!
//! - [`init`]: the initial value of a param from its `init`, on the host (graph-ir.md §2.4.1).
//! - [`config`]: the `optimizer`, `schedule` and `gradient_clip` objects of a training config
//!   (training.md §2.8), with their defaults and range checks.
//! - [`optimizer`]: `sgd`, `adam` and `adamw` with their state, export and import (§3.1, §5.2).
//! - [`schedule`]: the learning-rate schedules (§3.2).
//! - [`clip`]: gradient clipping by the total L2 norm (§3.1.4).
//!
//! The rest (the trainer, losses, datasets, checkpoint files) comes with later M2 tasks.

pub mod clip;
pub mod config;
pub mod init;
pub mod optimizer;
pub mod schedule;

pub use clip::clip_gradients;
pub use config::{
    AdamConfig, AdamWConfig, ConfigError, GradientClipConfig, OptimizerConfig, ScheduleConfig,
    SgdConfig,
};
pub use init::{InitError, initial_values};
pub use optimizer::{Optimizer, OptimizerError, StateTensor, TensorMap};
pub use schedule::Schedule;
