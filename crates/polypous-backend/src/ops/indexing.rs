//! Indexing ops: graph-ir.md §3.2 "Indexing" (`core.gather index_select`) and the out-of-range
//! rule of §3.1.8.

use burn::tensor::{Bool, Int, IntDType, Tensor};

use super::{DTypeClass, backend_error, check_axis, check_same_device, check_shape};
use crate::tensor::{DynTensor, Inner, Ranked, Result, TensorError, map_rank, with_rank};

impl DynTensor {
    /// `core.gather`: `out[i][j]… = self[…][index[i][j]…][…]`, indexing along `dim`; the output
    /// has `index`'s shape and `self`'s dtype (any dtype).
    ///
    /// `index` is `i32` or `i64`, has `self`'s rank and device, and in every dim other than `dim`
    /// is at most as large as `self` (PyTorch's rule; the output reads the leading part of those
    /// dims). Every index must be in `0..self.shape()[dim]`.
    ///
    /// The indices are checked **before** the op runs: their minimum and maximum are read back
    /// to the host, which costs one reduction and one device sync on a GPU. An index out of range
    /// is [`TensorError::IndexOutOfRange`]; nothing has run.
    pub fn gather(self, dim: usize, index: Self) -> Result<Self> {
        const OP: &str = "gather";
        DTypeClass::Index.check(OP, index.dtype())?;
        check_same_device(OP, self.device(), index.device())?;
        let (shape, index_shape) = (self.shape(), index.shape());
        check_axis(OP, "dim", dim, shape.len())?;
        let fits = index_shape.len() == shape.len()
            && (0..shape.len()).all(|d| d == dim || index_shape[d] <= shape[d]);
        if !fits {
            return Err(TensorError::ShapeMismatch {
                op: OP,
                lhs: shape,
                rhs: index_shape,
            });
        }
        check_shape(OP, &index_shape, self.dtype())?;
        check_indices(OP, &index, shape[dim])?;

        // Burn's gather needs equal sizes outside `dim`: keep the leading part of `self` there.
        let mut x = self;
        for d in (0..shape.len()).filter(|&d| d != dim && index_shape[d] < shape[d]) {
            x = x.slice(d, 0, index_shape[d], 1)?;
        }
        let (inner, dtype, device) = x.into_parts();
        let (index, _, _) = index.into_parts();
        let inner = match (inner, index) {
            (Inner::R1(x), Inner::R1(i)) => Inner::R1(x.gather(dim, i)?),
            (Inner::R2(x), Inner::R2(i)) => Inner::R2(x.gather(dim, i)?),
            (Inner::R3(x), Inner::R3(i)) => Inner::R3(x.gather(dim, i)?),
            (Inner::R4(x), Inner::R4(i)) => Inner::R4(x.gather(dim, i)?),
            (Inner::R5(x), Inner::R5(i)) => Inner::R5(x.gather(dim, i)?),
            (Inner::R6(x), Inner::R6(i)) => Inner::R6(x.gather(dim, i)?),
            (x, i) => {
                return Err(TensorError::ShapeMismatch {
                    op: OP,
                    lhs: x.dims(),
                    rhs: i.dims(),
                });
            }
        };
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `core.index_select`: the slices `index[0], index[1], …` of `self` along `dim`; the output
    /// is `self`'s shape with dim `dim` replaced by `index`'s length. Any dtype.
    ///
    /// `index` is a rank-1 `i32` or `i64` tensor on `self`'s device; every index must be in
    /// `0..self.shape()[dim]`. Checked before the op runs, like [`DynTensor::gather`] (one
    /// reduction and one device sync on a GPU); an index out of range is
    /// [`TensorError::IndexOutOfRange`].
    pub fn index_select(self, dim: usize, index: Self) -> Result<Self> {
        const OP: &str = "index_select";
        DTypeClass::Index.check(OP, index.dtype())?;
        check_same_device(OP, self.device(), index.device())?;
        let mut shape = self.shape();
        check_axis(OP, "dim", dim, shape.len())?;
        let index_shape = index.shape();
        if index_shape.len() != 1 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`index` has shape {index_shape:?}, needs rank 1"),
            });
        }
        let size = shape[dim];
        shape[dim] = index_shape[0];
        check_shape(OP, &shape, self.dtype())?;
        check_indices(OP, &index, size)?;

        let (inner, dtype, device) = self.into_parts();
        let (index, _, _) = index.into_parts();
        let Inner::R1(Ranked::Int(index)) = index else {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: "`index` must be a rank-1 integer tensor".into(),
            });
        };
        let inner = map_rank!(inner, t => t.index_select(dim, index.clone()));
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// `IndexOutOfRange` unless every element of the integer tensor `index` is in `0..size`. Reads the
/// minimum and maximum back to the host in one transfer (graph-ir.md §3.1.8: checked before the
/// op runs). Also used by `nn.embedding`.
pub(super) fn check_indices(op: &'static str, index: &DynTensor, size: usize) -> Result<()> {
    let (inner, _, _) = index.clone().into_parts();
    let bounds: Tensor<1, Int> = with_rank!(inner, t => match t {
        Ranked::Int(t) => Tensor::cat(vec![t.clone().min(), t.max()], 0),
        other => return Err(other.kind_error(op)),
    });
    let data = bounds.try_into_data().map_err(backend_error)?;
    let values: Vec<i64> = data.iter::<i64>().collect();
    let [min, max] = values[..] else {
        return Err(TensorError::Backend {
            message: format!("{op}: index bounds read back as {values:?}"),
        });
    };
    let in_range = |v: i64| usize::try_from(v).is_ok_and(|v| v < size);
    if !in_range(min) {
        return Err(TensorError::IndexOutOfRange {
            op,
            index: min,
            size,
        });
    }
    if !in_range(max) {
        return Err(TensorError::IndexOutOfRange {
            op,
            index: max,
            size,
        });
    }
    Ok(())
}

impl<const D: usize> Ranked<D> {
    /// The caller checked the dtypes, the dim, the shapes and the index range.
    fn gather(self, dim: usize, index: Self) -> Result<Self> {
        let Ranked::Int(index) = index else {
            return Err(index.kind_error("gather"));
        };
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.gather(dim, index)),
            Ranked::Int(t) => Ranked::Int(t.gather(dim, index)),
            Ranked::Bool(t) => Ranked::Bool(via_int(t, |t| t.gather(dim, index))),
        })
    }

    /// The caller checked the dim and the index range.
    fn index_select(self, dim: usize, index: Tensor<1, Int>) -> Self {
        match self {
            Ranked::Float(t) => Ranked::Float(t.select(dim, index)),
            Ranked::Int(t) => Ranked::Int(t.select(dim, index)),
            Ranked::Bool(t) => Ranked::Bool(via_int(t, |t| t.select(dim, index))),
        }
    }
}

/// Runs an indexing op on a bool tensor as `i32` and converts the result back to `bool`: Burn
/// 0.22.1's `flex` returns `gather` and `select` of a bool tensor in `u8` storage, which is not a
/// bool dtype (reading it back fails). Done on every backend, so results don't depend on one.
fn via_int<const D: usize>(
    t: Tensor<D, Bool>,
    op: impl FnOnce(Tensor<D, Int>) -> Tensor<D, Int>,
) -> Tensor<D, Bool> {
    op(t.cast(IntDType::I32)).bool()
}

#[cfg(test)]
#[path = "indexing_tests.rs"]
mod tests;
