//! Runtime-typed tensors: [`DynTensor`], [`TensorData`] and [`DType`].
//!
//! Implements the `DynTensor` of ADR-0002 on top of Burn's runtime dispatch (ADR-0011). Dtypes
//! follow `docs/specs/graph-ir.md` §2.2, and [`TensorData`] uses the exchange layout of
//! `docs/specs/public-api.md` §2 rule 4.
//!
//! # How the dispatch works
//!
//! Burn 0.22 picks the **backend** at runtime from the `burn::Device` a tensor lives on, so
//! `DynTensor` needs no backend enum. Burn still fixes two things at compile time: the **rank**
//! (`Tensor<const D: usize, K>`) and the tensor **kind** (`K` = `Float`, `Int` or `Bool`). The
//! precise dtype within a kind (`f32`/`f16`/`bf16`, `i64`/`i32`) is a runtime value in Burn
//! already. `DynTensor` therefore holds two nested enums:
//!
//! - `Inner`: one variant per rank, `R1(Ranked<1>)` … `R6(Ranked<6>)`.
//! - `Ranked<const D>`: one variant per kind, `Float(Tensor<D>)`, `Int(Tensor<D, Int>)`,
//!   `Bool(Tensor<D, Bool>)`.
//!
//! Operations are written **once**, as methods on `Ranked<D>` that are generic over `D` and
//! `match` on the kind. Four small macros in `dyn_tensor.rs` turn the rank into that `D`:
//!
//! - `with_rank!(inner, t => body)`: runs `body` with `t: Ranked<D>` for the rank `inner` has.
//! - `map_rank!(inner, t => body)`: the same, for a body that returns a `Ranked<D>` of equal rank.
//! - `zip_rank!(op, a, b, x, y => body)`: the binary version, for operands of equal rank.
//! - `dispatch_rank!(rank, N => body)`: turns a runtime rank into a constant `N` (a `const` item
//!   in each match arm) for code that creates a tensor of a new rank, such as `from_data` and
//!   `reshape`. Ranks outside 1–6 become [`TensorError::UnsupportedRank`].
//!
//! Adding an op means one method on `Ranked<D>` plus one public method on `DynTensor` that checks
//! the inputs and calls a macro. The IR ops live in [`crate::ops`], which uses these macros and
//! types crate-internally; see its documentation for the file layout. Supporting rank 7 would
//! mean one more variant and one more arm per macro.
//!
//! # No panics on bad input
//!
//! Burn panics on invalid shapes, mixed devices and dtypes a backend can't handle. Every public
//! function checks these first (shape rules, `Device::supports_dtype`) and returns a
//! [`TensorError`].
//!
//! # Devices
//!
//! The public API addresses devices by [`DeviceId`](crate::device::DeviceId), as returned by
//! [`list_devices`](crate::device::list_devices), so no Burn type appears in it (ADR-0007).
//! `from_data`, `to_device` and `device` map ids to Burn devices with the crate-internal mapping of
//! the `device` module; the `burn::Device` variants (`from_data_burn`, `to_device_burn`,
//! `burn_device`) are crate-internal. An id whose backend is not compiled in, or that addresses no
//! device Burn's enumeration lists, is a [`TensorError::DeviceUnavailable`] and never reaches Burn,
//! which would panic on it (`device/resolve.rs`). Each tensor stores the [`DeviceId`] of its device,
//! so `device()` returns exactly the id it was created with.
//!
//! [`DeviceId`]: crate::device::DeviceId

mod data;
mod dtype;
mod dyn_tensor;
mod error;

pub use data::TensorData;
pub use dtype::DType;
pub(crate) use dtype::Kind;
pub use dyn_tensor::DynTensor;
// For the ops module (`crate::ops`).
pub(crate) use data::num_elements;
pub(crate) use dyn_tensor::{
    Inner, Ranked, check_device_supports, dispatch_rank, map_rank, resolve, with_rank, zip_rank,
};
pub use error::{Result, TensorError};
