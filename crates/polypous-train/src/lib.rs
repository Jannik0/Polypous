//! Training for Polypous (`docs/specs/training.md`).
//!
//! Responsibility: the training config and its validation, the param store, initial values,
//! datasets, losses and metrics, the optimizers and schedules (Polypous code on `DynTensor`
//! ops, ADR-0014; no Burn dependency), the training loop, checkpoints and events. See
//! `docs/architecture.md` §2 and training.md §1.1.
//!
//! - [`init`]: the initial value of a param from its `init`, on the host (graph-ir.md §2.4.1).
//!
//! The rest (the trainer, optimizers, losses, datasets) comes with later M2 tasks.

pub mod init;

pub use init::{InitError, initial_values};
