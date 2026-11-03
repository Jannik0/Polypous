//! Compute backend for Polypous.
//!
//! Responsibility: Burn integration: `BackendKind` enum, device discovery/probing, `DynTensor`
//! (dynamic rank + dtype), implementations of every IR op, cross-device transfer, autodiff wrapper.
//!
//! Together with `polypous-quant`, the only crate allowed to use Burn (ADR-0007). See
//! `docs/architecture.md` §2.

pub mod tensor;

pub use tensor::{DType, DynTensor, TensorData, TensorError};
