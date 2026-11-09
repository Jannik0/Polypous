//! The IR ops of `docs/specs/graph-ir.md` §3 as [`DynTensor`] functions.
//!
//! Every op is a method (or associated function) on [`DynTensor`] with plain Rust arguments: no
//! `polypous-ir` type appears here (`docs/architecture.md` §2). `polypous-runtime` maps IR nodes
//! onto these functions, after validation has proven dtypes and shapes (graph-ir.md §2.3.4).
//! The functions still check every argument and return a [`TensorError`] instead of panicking
//! (CLAUDE.md, "No panics on untrusted input").
//!
//! # Layout
//!
//! One file per op family, each with its own `impl DynTensor` block, so that op families can be
//! added in parallel without touching each other's files:
//!
//! | File | Ops (graph-ir.md §3.2) |
//! |---|---|
//! | `mod.rs` | Shared pieces: [`Scalar`], dtype classes, broadcasting, operand checks |
//! | `elementwise.rs` | `core.add sub mul div pow maximum minimum neg abs exp log sqrt rsqrt sin cos tanh clamp where` |
//! | `compare.rs` | `core.eq ne lt le gt ge`, and the logic ops `core.and or not` |
//! | `cast.rs` | `core.cast` |
//!
//! Planned: shape, indexing and creation ops (#41), reductions, `matmul` and softmax (#42) and
//! the `nn` opset (#43) each add their own file here and register it below. `matmul` and
//! `reshape` still live in `tensor/dyn_tensor.rs` until #41/#42 move them.
//!
//! Reference tests live in `tests/reference_*.rs`, one test binary per op family, and share the
//! fixture harness in `tests/reference/common.rs` (testing-strategy.md §6). A new family adds
//! its own `tests/reference_<family>.rs` with `mod common;` pointing at that file.
//!
//! # Method names
//!
//! | IR op | Method |
//! |---|---|
//! | `core.add`, `sub`, `mul`, `div`, `pow`, `maximum`, `minimum` | [`DynTensor::add`], [`sub`](DynTensor::sub), [`mul`](DynTensor::mul), [`div`](DynTensor::div), [`pow`](DynTensor::pow), [`maximum`](DynTensor::maximum), [`minimum`](DynTensor::minimum) |
//! | `core.neg`, `abs`, `exp`, `log`, `sqrt`, `rsqrt`, `sin`, `cos`, `tanh` | The same names |
//! | `core.clamp` | [`DynTensor::clamp`] |
//! | `core.where` | [`DynTensor::where_`] (`where` is a Rust keyword) |
//! | `core.eq`, `ne`, `lt`, `le`, `gt`, `ge` | [`equal`](DynTensor::equal), [`not_equal`](DynTensor::not_equal), [`less`](DynTensor::less), [`less_equal`](DynTensor::less_equal), [`greater`](DynTensor::greater), [`greater_equal`](DynTensor::greater_equal) (not `eq`/`ne`, which read like `PartialEq`) |
//! | `core.and`, `or`, `not` | [`logical_and`](DynTensor::logical_and), [`logical_or`](DynTensor::logical_or), [`logical_not`](DynTensor::logical_not) (PyTorch's names) |
//! | `core.cast` | [`DynTensor::cast`] |
//!
//! # Semantics (graph-ir.md §3.1)
//!
//! - **Strict dtypes** (§3.1.2): operands that must agree have the same dtype, or the op fails
//!   with [`TensorError::DTypeMismatch`]. A dtype outside the op's class is
//!   [`TensorError::UnsupportedOperation`]. There is no promotion and no silent conversion.
//! - **Broadcasting** (§3.1.3): shapes are right-aligned, the shorter one is padded with leading
//!   `1`s, and each pair of dims must be equal or contain a `1`. Ranks may differ; the output has
//!   the largest rank. Every operand is broadcast explicitly (reshape to the output rank, then
//!   `expand`) before Burn sees it, so no backend's own broadcasting rules matter.
//! - **NaN and special values** (§3.1.8) are set explicitly with masks wherever a backend was
//!   found to get them wrong, so results don't depend on the backend:
//!   - `maximum`, `minimum` and `clamp` propagate NaN (`flex`'s `max_pair` returns the non-NaN
//!     operand).
//!   - Comparisons: NaN compares false, except in `ne` (the CubeCL backends compare ordered, so
//!     `ne` gave `false` on lavapipe). Float → `bool` casts map NaN to `true` for the same reason.
//!   - `pow` follows C99 Annex F like `torch.pow` (`pow(1, NaN) = pow(NaN, 0) = 1`, zero and
//!     infinite bases and exponents); CubeCL's `exp2(b · log2(a))` got four of them wrong on
//!     lavapipe.
//! - **Internal precision:** the transcendental ops and `pow` on `f16`/`bf16` compute in `f32`
//!   and round once to the dtype, on every backend (`flex` does this anyway; CubeCL's `f16` `tanh`
//!   missed `rtol 1e-2` near zero on lavapipe). The output dtype is unchanged.
//! - **Integer overflow wraps** (§3.1.8). Burn's CPU backend (`flex`) computes `i64` `add`, `sub`
//!   and `mul` with plain `+ - *`, which panic on overflow when overflow checks are on (debug
//!   builds). On the CPU those three ops on `i64` therefore run on the host with
//!   `wrapping_add`/`wrapping_sub`/`wrapping_mul`. `flex` computes `i32` in `i64` and truncates,
//!   and uses `wrapping_neg`/`wrapping_abs`, so those wrap already. GPU integer arithmetic wraps.
//! - **Devices** (graph-ir.md §2.2): a tensor only exists on a device that supports its dtype
//!   ([`DynTensor::from_data`] and [`DynTensor::to_device`] check), and [`DynTensor::cast`] checks
//!   the target dtype, so no op here can reach a device with an unsupported dtype.
//!   [`DynTensor::device_supports`] answers the question up front.

mod cast;
mod compare;
mod elementwise;

use crate::device::DeviceId;
use burn::tensor::Shape;

use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank};

/// A scalar argument of an op, for example a `clamp` bound (graph-ir.md §3.1.7, type `scalar`).
///
/// Which variants an op accepts is part of its documentation. In general an integer fits every
/// dtype it is in range for, a float fits float dtypes (it is rounded to the dtype), and a bool
/// fits `bool`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum Scalar {
    /// A float value; may be NaN or ±inf.
    Float(f64),
    /// An integer value.
    Int(i64),
    /// A boolean value.
    Bool(bool),
}

/// The dtype classes of graph-ir.md §3.1.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DTypeClass {
    /// `N`: float or integer.
    Numeric,
    /// `F`: `f32`, `f16`, `bf16`.
    Float,
    /// `B`: `bool`.
    Bool,
    /// `A`: any dtype.
    Any,
}

impl DTypeClass {
    fn contains(self, dtype: DType) -> bool {
        match self {
            DTypeClass::Numeric => dtype.is_float() || dtype.is_int(),
            DTypeClass::Float => dtype.is_float(),
            DTypeClass::Bool => dtype == DType::Bool,
            DTypeClass::Any => true,
        }
    }

    /// `UnsupportedOperation` unless `dtype` is in the class.
    pub(crate) fn check(self, op: &'static str, dtype: DType) -> Result<()> {
        if self.contains(dtype) {
            Ok(())
        } else {
            Err(TensorError::UnsupportedOperation { op, dtype })
        }
    }
}

/// The broadcast shape of `shapes` (graph-ir.md §3.1.3): right-aligned, padded with leading `1`s,
/// each dim equal or `1`. `None` if two dims are incompatible.
pub(crate) fn broadcast_shapes(shapes: &[&[usize]]) -> Option<Vec<usize>> {
    let rank = shapes.iter().map(|shape| shape.len()).max()?;
    let mut out = vec![1usize; rank];
    for shape in shapes {
        let offset = rank - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let slot = &mut out[offset + i];
            if *slot == 1 {
                *slot = dim;
            } else if dim != 1 && dim != *slot {
                return None;
            }
        }
    }
    Some(out)
}

/// `DeviceMismatch` unless both devices are the same.
pub(crate) fn check_same_device(op: &'static str, lhs: DeviceId, rhs: DeviceId) -> Result<()> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(TensorError::DeviceMismatch {
            op,
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
        })
    }
}

/// Checks the operands of a broadcasting binary op with dtype class `class` and broadcasts both to
/// the output shape. The checks run in this order: dtype mismatch, dtype class, device, shape.
pub(crate) fn prepare_binary(
    op: &'static str,
    class: DTypeClass,
    lhs: DynTensor,
    rhs: DynTensor,
) -> Result<(DynTensor, DynTensor)> {
    if lhs.dtype() != rhs.dtype() {
        return Err(TensorError::DTypeMismatch {
            op,
            lhs: lhs.dtype(),
            rhs: rhs.dtype(),
        });
    }
    class.check(op, lhs.dtype())?;
    check_same_device(op, lhs.device(), rhs.device())?;
    let (lhs_shape, rhs_shape) = (lhs.shape(), rhs.shape());
    let Some(shape) = broadcast_shapes(&[&lhs_shape, &rhs_shape]) else {
        return Err(TensorError::ShapeMismatch {
            op,
            lhs: lhs_shape,
            rhs: rhs_shape,
        });
    };
    Ok((lhs.broadcast_to(&shape)?, rhs.broadcast_to(&shape)?))
}

impl DynTensor {
    /// Broadcasts the tensor to `shape`, which must be a broadcast of its shape (callers compute
    /// it with [`broadcast_shapes`]): leading `1`s are added with a reshape, then size-1 dims are
    /// expanded. Burn's `expand` makes a view, not a copy.
    pub(crate) fn broadcast_to(self, shape: &[usize]) -> Result<Self> {
        let current = self.shape();
        if current == shape {
            return Ok(self);
        }
        if current.len() > shape.len() {
            return Err(TensorError::ShapeMismatch {
                op: "broadcast",
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        let compatible = current
            .iter()
            .zip(&shape[shape.len() - current.len()..])
            .all(|(&have, &want)| have == want || have == 1);
        if !compatible {
            return Err(TensorError::ShapeMismatch {
                op: "broadcast",
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        let aligned = if current.len() < shape.len() {
            let mut padded = vec![1usize; shape.len() - current.len()];
            padded.extend_from_slice(&current);
            self.reshape(&padded)?
        } else {
            self
        };
        if aligned.shape() == shape {
            return Ok(aligned);
        }
        let (inner, dtype, device) = aligned.into_parts();
        let inner = map_rank!(inner, t => t.expand(shape)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

impl<const D: usize> Ranked<D> {
    /// Burn's `expand` to a shape of the same rank. The caller checked that every dim is equal
    /// or `1` in `self`; a wrong rank is an error, not a panic.
    pub(crate) fn expand(self, shape: &[usize]) -> Result<Self> {
        if shape.len() != D {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        let shape = Shape::from(shape.to_vec());
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.expand(shape)),
            Ranked::Int(t) => Ranked::Int(t.expand(shape)),
            Ranked::Bool(t) => Ranked::Bool(t.expand(shape)),
        })
    }
}

/// Maps a Burn read-back error to [`TensorError::Backend`].
pub(crate) fn backend_error(err: impl std::fmt::Debug) -> TensorError {
    TensorError::Backend {
        message: format!("{err:?}"),
    }
}

#[cfg(test)]
mod tests;
