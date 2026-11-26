//! `core.matmul`: graph-ir.md §3.2, "Linear algebra". General rules: §3.1 (see the module
//! documentation of [`crate::ops`]).
//!
//! The batch dims (all but the last two) broadcast like elementwise ops (§3.1.3), across
//! different ranks. Both operands are broadcast explicitly to `[batch…, n, k]` and
//! `[batch…, k, m]` (reshape, then `expand`: a view, no copy) before Burn's `matmul` sees them,
//! so no backend's own broadcasting rule matters.
//!
//! `f16`/`bf16` operands are cast to `f32` first and the product is rounded once to their dtype
//! (graph-ir.md §3.1.8, "Half-precision accumulation"; issue #67).

use super::{DTypeClass, broadcast_shapes, check_same_device, check_shape};
use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, zip_rank};

impl DynTensor {
    /// `core.matmul`: `self @ other` over the last two dims, `[…, n, k] @ […, k, m] = […, n, m]`.
    ///
    /// Both operands need the same float dtype ([`TensorError::DTypeMismatch`],
    /// [`TensorError::UnsupportedOperation`]), the same device, rank ≥ 2 and the same `k`; the
    /// batch dims broadcast and may have different ranks (the output has the larger one).
    /// Shape errors are [`TensorError::ShapeMismatch`]. Rank-1 operands are not allowed
    /// (PyTorch's vector forms change the output rank); `unsqueeze` them first.
    ///
    /// `f16`/`bf16` compute in `f32` and round once to the dtype (graph-ir.md §3.1.8): both
    /// operands are cast to `f32` after the checks, multiplied on the `f32` path and the result is
    /// cast back. Left to the backend kernel, `f16` missed `rtol 1e-2` with `k = 256` on Mesa ANV
    /// (Iris Xe, #67). `f32` operands (as [`DynTensor::linear`] passes them) run unchanged.
    ///
    /// **Known limitation:** on CUDA GPUs with TF32 support (Ampere or newer), Burn 0.22.1 runs
    /// `f32` matmul on TF32 tensor cores, which misses the spec's f32 accuracy. There is no
    /// workaround here by design; it waits for an upstream precision flag (docs/status.md §4).
    /// `f16` and `bf16` stay correct there: their values are exact in TF32 (10-bit mantissa, the
    /// same as `f16`'s), and the products accumulate in `f32`.
    pub fn matmul(self, other: Self) -> Result<Self> {
        const OP: &str = "matmul";
        if self.dtype() != other.dtype() {
            return Err(TensorError::DTypeMismatch {
                op: OP,
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }
        DTypeClass::Float.check(OP, self.dtype())?;
        check_same_device(OP, self.device(), other.device())?;
        let (lhs, rhs) = (self.shape(), other.shape());
        let (Some(shape), Some((n, k)), Some((k2, m))) =
            (matmul_batch(&lhs, &rhs), last_two(&lhs), last_two(&rhs))
        else {
            return Err(TensorError::ShapeMismatch { op: OP, lhs, rhs });
        };
        if k != k2 {
            return Err(TensorError::ShapeMismatch { op: OP, lhs, rhs });
        }
        let mut out_shape = shape.clone();
        out_shape.extend([n, m]);
        check_shape(OP, &out_shape, self.dtype())?;
        let dtype = self.dtype();
        if dtype != DType::F32 {
            // f16/bf16 (the float class checked above): compute in f32, round once (§3.1.8).
            // The f32 operands and output are at most twice the checked half-precision sizes.
            check_shape(OP, &out_shape, DType::F32)?;
            return self.matmul_in_f32(other, dtype);
        }
        let (mut a_shape, mut b_shape) = (shape.clone(), shape);
        a_shape.extend([n, k]);
        b_shape.extend([k, m]);
        let device = self.device();
        let (a, _, _) = self.broadcast_to(&a_shape)?.into_parts();
        let (b, _, _) = other.broadcast_to(&b_shape)?.into_parts();
        let inner = zip_rank!(OP, a, b, x, y => x.matmul(y)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// The `f16`/`bf16` path of [`DynTensor::matmul`], after its checks: both operands cast to
    /// `f32`, multiplied, the product cast back to `dtype`.
    // #114: own frame. A debug build gives every temporary its own stack slot; inline, the casts'
    // slots made `matmul`'s frame larger on the `f32` path too, which this path calls.
    #[inline(never)]
    fn matmul_in_f32(self, other: Self, dtype: DType) -> Result<Self> {
        let wide = self.cast(DType::F32)?.matmul(other.cast(DType::F32)?)?;
        wide.cast(dtype)
    }
}

/// The last two dims, if the rank is at least 2.
fn last_two(shape: &[usize]) -> Option<(usize, usize)> {
    match shape {
        [.., a, b] => Some((*a, *b)),
        _ => None,
    }
}

/// The broadcast batch dims of two matmul operands (rank ≥ 2 each), or `None`.
fn matmul_batch(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    if lhs.len() < 2 || rhs.len() < 2 {
        return None;
    }
    broadcast_shapes(&[&lhs[..lhs.len() - 2], &rhs[..rhs.len() - 2]])
}

impl<const D: usize> Ranked<D> {
    /// The caller checked dtypes, shapes and devices.
    fn matmul(self, other: Self) -> Result<Self> {
        match (self, other) {
            (Ranked::Float(a), Ranked::Float(b)) => Ok(Ranked::Float(a.matmul(b))),
            (a, _) => Err(a.kind_error("matmul")),
        }
    }
}

#[cfg(test)]
#[path = "linalg_tests.rs"]
mod tests;
