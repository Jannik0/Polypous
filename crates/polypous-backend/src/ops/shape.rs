//! Shape ops: graph-ir.md §3.2 "Shape" (`core.reshape flatten permute transpose squeeze unsqueeze
//! expand concat split slice pad`). Arguments are concrete and normalized (see the module
//! documentation of [`crate::ops`]); every one is still checked.

use burn::tensor::{Slice, Tensor};

use super::creation::{Fill, fill_value};
use super::{MAX_CONCAT_SPLIT, Scalar, check_axis, check_same_device, check_shape};
use crate::tensor::{
    DType, DynTensor, Inner, Ranked, Result, TensorError, dispatch_rank, map_rank, num_elements,
    with_rank,
};

impl DynTensor {
    /// `core.reshape`: the same elements in row-major order, with a new shape (rank 1–6).
    ///
    /// `shape` is concrete: the IR's `-1` entry is resolved by the caller. Fails with
    /// [`TensorError::UnsupportedRank`] for a rank outside 1–6, [`TensorError::ShapeMismatch`] if
    /// the element counts differ, and [`TensorError::InvalidArgument`] for a zero-size dim.
    pub fn reshape(self, shape: &[usize]) -> Result<Self> {
        const OP: &str = "reshape";
        if !(1..=Self::MAX_RANK).contains(&shape.len()) {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        let current = self.shape();
        if num_elements(shape) != num_elements(&current) {
            return Err(TensorError::ShapeMismatch {
                op: OP,
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        check_shape(OP, shape, self.dtype())?;
        if current == shape {
            return Ok(self);
        }
        let (inner, dtype, device) = self.into_parts();
        let inner = with_rank!(inner, t => {
            dispatch_rank!(shape.len(), N => t.reshape::<N>(shape)?)
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `core.flatten`: dims `start_dim..=end_dim` replaced by their product.
    ///
    /// Needs `start_dim ≤ end_dim < rank` ([`TensorError::InvalidArgument`] otherwise).
    pub fn flatten(self, start_dim: usize, end_dim: usize) -> Result<Self> {
        const OP: &str = "flatten";
        let shape = self.shape();
        check_axis(OP, "start_dim", start_dim, shape.len())?;
        check_axis(OP, "end_dim", end_dim, shape.len())?;
        if start_dim > end_dim {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`start_dim` {start_dim} is after `end_dim` {end_dim}"),
            });
        }
        let merged = num_elements(&shape[start_dim..=end_dim]).ok_or_else(|| too_large(OP))?;
        let mut out = shape[..start_dim].to_vec();
        out.push(merged);
        out.extend_from_slice(&shape[end_dim + 1..]);
        self.reshape(&out).map_err(|err| rename(err, OP))
    }

    /// `core.permute`: dim `i` of the result is dim `dims[i]` of the input. `dims` is a
    /// permutation of `0..rank`. The result is a view.
    pub fn permute(self, dims: &[usize]) -> Result<Self> {
        const OP: &str = "permute";
        let rank = self.rank();
        let mut seen = [false; Self::MAX_RANK];
        let valid = dims.len() == rank
            && dims
                .iter()
                .all(|&d| d < rank && !std::mem::replace(&mut seen[d], true));
        if !valid {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`dims` {dims:?} is not a permutation of 0..{rank}"),
            });
        }
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.permute(dims)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `core.transpose`: swaps dims `dim0` and `dim1` (equal dims leave the tensor unchanged). The
    /// result is a view.
    pub fn transpose(self, dim0: usize, dim1: usize) -> Result<Self> {
        const OP: &str = "transpose";
        let rank = self.rank();
        check_axis(OP, "dim0", dim0, rank)?;
        check_axis(OP, "dim1", dim1, rank)?;
        if dim0 == dim1 {
            return Ok(self);
        }
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.swap_dims(dim0, dim1));
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `core.squeeze`: removes the listed dims, each of which must have size 1. If every dim is
    /// removed, the result is `[1]` (graph-ir.md §3.1.5). An empty list returns the tensor
    /// unchanged (the IR fills in the size-1 dims when `dim` is absent; there may be none).
    pub fn squeeze(self, dims: &[usize]) -> Result<Self> {
        const OP: &str = "squeeze";
        let shape = self.shape();
        let mut removed = [false; Self::MAX_RANK];
        for &d in dims {
            check_axis(OP, "dim", d, shape.len())?;
            if std::mem::replace(&mut removed[d], true) {
                return Err(TensorError::InvalidArgument {
                    op: OP,
                    reason: format!("`dim` {dims:?} lists dim {d} twice"),
                });
            }
            if shape[d] != 1 {
                return Err(TensorError::InvalidArgument {
                    op: OP,
                    reason: format!("dim {d} has size {}, not 1", shape[d]),
                });
            }
        }
        let mut out: Vec<usize> = (0..shape.len())
            .filter(|&d| !removed[d])
            .map(|d| shape[d])
            .collect();
        if out.is_empty() {
            out.push(1);
        }
        self.reshape(&out).map_err(|err| rename(err, OP))
    }

    /// `core.unsqueeze`: inserts a size-1 dim at `dim` (`0..=rank`). Fails with
    /// [`TensorError::UnsupportedRank`] for a rank-6 input.
    pub fn unsqueeze(self, dim: usize) -> Result<Self> {
        const OP: &str = "unsqueeze";
        let mut shape = self.shape();
        check_axis(OP, "dim", dim, shape.len() + 1)?;
        shape.insert(dim, 1);
        if shape.len() > Self::MAX_RANK {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        self.reshape(&shape).map_err(|err| rename(err, OP))
    }

    /// `core.expand`: broadcasts to `shape`, right-aligned against the tensor's shape; new
    /// leading dims are allowed (rank up to 6). Each aligned dim must be equal, or `1` in the
    /// tensor. `shape` is concrete (the IR's `-1` entries are resolved by the caller). The result
    /// is a view with stride 0 along the expanded dims.
    pub fn expand(self, shape: &[usize]) -> Result<Self> {
        const OP: &str = "expand";
        if shape.len() > Self::MAX_RANK {
            return Err(TensorError::UnsupportedRank { rank: shape.len() });
        }
        let current = self.shape();
        let fits = shape.len() >= current.len()
            && current
                .iter()
                .zip(&shape[shape.len() - current.len()..])
                .all(|(&have, &want)| have == want || have == 1);
        if !fits {
            return Err(TensorError::ShapeMismatch {
                op: OP,
                lhs: current,
                rhs: shape.to_vec(),
            });
        }
        check_shape(OP, shape, self.dtype())?;
        self.broadcast_to(shape)
    }

    /// `core.concat`: joins `xs` (1–1,024 tensors) along `dim`. All need the same dtype, rank and
    /// device, and equal sizes in every other dim.
    pub fn concat(xs: &[DynTensor], dim: usize) -> Result<Self> {
        const OP: &str = "concat";
        let Some(first) = xs.first() else {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: "needs at least one input".into(),
            });
        };
        if xs.len() > MAX_CONCAT_SPLIT {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("{} inputs, at most {MAX_CONCAT_SPLIT}", xs.len()),
            });
        }
        let first_shape = first.shape();
        check_axis(OP, "dim", dim, first_shape.len())?;
        let mut size = 0usize;
        for x in xs {
            if x.dtype() != first.dtype() {
                return Err(TensorError::DTypeMismatch {
                    op: OP,
                    lhs: first.dtype(),
                    rhs: x.dtype(),
                });
            }
            check_same_device(OP, first.device(), x.device())?;
            let shape = x.shape();
            let fits = shape.len() == first_shape.len()
                && (0..shape.len()).all(|d| d == dim || shape[d] == first_shape[d]);
            if !fits {
                return Err(TensorError::ShapeMismatch {
                    op: OP,
                    lhs: first_shape,
                    rhs: shape,
                });
            }
            size = size.checked_add(shape[dim]).ok_or_else(|| too_large(OP))?;
        }
        let mut out = first_shape;
        out[dim] = size;
        check_shape(OP, &out, first.dtype())?;
        if xs.len() == 1 {
            return Ok(first.clone());
        }
        let (dtype, device) = (first.dtype(), first.device());
        let inners = xs.iter().map(|x| x.clone().into_parts().0).collect();
        Ok(DynTensor::from_parts(
            cat_inner(inners, dim)?,
            dtype,
            device,
        ))
    }

    /// `core.split`: splits along `dim` into pieces of `sizes` (1–1,024 entries, each ≥ 1, summing
    /// to the dim's size). The pieces are views, in order.
    pub fn split(self, dim: usize, sizes: &[usize]) -> Result<Vec<Self>> {
        const OP: &str = "split";
        let shape = self.shape();
        check_axis(OP, "dim", dim, shape.len())?;
        if sizes.is_empty() || sizes.len() > MAX_CONCAT_SPLIT {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("{} sizes, need 1 to {MAX_CONCAT_SPLIT}", sizes.len()),
            });
        }
        if sizes.contains(&0) {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`sizes` {sizes:?} has a zero entry (not supported in M1)"),
            });
        }
        let total = sizes.iter().try_fold(0usize, |acc, &s| acc.checked_add(s));
        if total != Some(shape[dim]) {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "`sizes` {sizes:?} don't add up to the size {} of dim {dim}",
                    shape[dim]
                ),
            });
        }
        let mut start = 0;
        let mut out = Vec::with_capacity(sizes.len());
        for &size in sizes {
            out.push(self.clone().slice_unchecked(dim, start, start + size, 1)?);
            start += size;
        }
        Ok(out)
    }

    /// `core.slice`: elements `start, start + step, …` below `end` along `dim`.
    ///
    /// The bounds are resolved and clamped by the caller: `0 ≤ start < end ≤ n` for the dim size
    /// `n` (an empty result is not supported in M1), and `step ≥ 1`. The output dim is
    /// `(end - start).div_ceil(step)`. The result is a view.
    pub fn slice(self, dim: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        const OP: &str = "slice";
        let shape = self.shape();
        check_axis(OP, "dim", dim, shape.len())?;
        let n = shape[dim];
        if step == 0 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: "`step` must be at least 1".into(),
            });
        }
        if !(start < end && end <= n) {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "bounds {start}..{end} don't select a non-empty range of dim {dim} (size {n})"
                ),
            });
        }
        self.slice_unchecked(dim, start, end, step)
    }

    /// `core.pad` with `mode: "constant"`: `pad` lists `(left, right)` pairs starting from the
    /// **last** dim (as in `F.pad`), 1 to `rank` pairs; padded elements are `value`.
    ///
    /// `value` follows the fill rules of the module documentation (for the IR default `0`: the
    /// zero of the dtype, [`Scalar::Bool(false)`](Scalar::Bool) for `bool`). Works for every dtype.
    pub fn pad(self, pad: &[usize], value: Scalar) -> Result<Self> {
        const OP: &str = "pad";
        let mut shape = self.shape();
        let rank = shape.len();
        if pad.is_empty() || !pad.len().is_multiple_of(2) || pad.len() > 2 * rank {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "`pad` has {} entries, needs an even number from 2 to {}",
                    pad.len(),
                    2 * rank
                ),
            });
        }
        let fill = fill_value(OP, value, self.dtype())?;
        let widths: Vec<(usize, usize, usize)> = pad
            .as_chunks::<2>()
            .0
            .iter()
            .enumerate()
            .map(|(k, &[left, right])| (rank - 1 - k, left, right))
            .collect();
        for &(dim, left, right) in &widths {
            shape[dim] = shape[dim]
                .checked_add(left)
                .and_then(|d| d.checked_add(right))
                .ok_or_else(|| too_large(OP))?;
        }
        check_shape(OP, &shape, self.dtype())?;
        let (mut inner, dtype, device) = self.into_parts();
        for (dim, left, right) in widths {
            if left == 0 && right == 0 {
                continue;
            }
            inner = map_rank!(inner, t => t.pad_dim(dim, left, right, fill, dtype)?);
        }
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `slice` after the arguments were checked.
    fn slice_unchecked(self, dim: usize, start: usize, end: usize, step: usize) -> Result<Self> {
        let n = self.shape().get(dim).copied().unwrap_or(0);
        // `step ≥ n` selects only `start`, like `step = n`; this keeps the value in `isize`.
        let slice = Slice::new(
            to_isize(start)?,
            Some(to_isize(end)?),
            to_isize(step.min(n.max(1)))?,
        );
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.slice_dim(dim, slice));
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// `InvalidArgument` for a shape whose size overflows.
fn too_large(op: &'static str) -> TensorError {
    TensorError::InvalidArgument {
        op,
        reason: "the result is too large".into(),
    }
}

/// Reports a `reshape` error raised for a derived shape under the op that called it.
fn rename(err: TensorError, op: &'static str) -> TensorError {
    match err {
        TensorError::ShapeMismatch { lhs, rhs, .. } => TensorError::ShapeMismatch { op, lhs, rhs },
        TensorError::InvalidArgument { reason, .. } => TensorError::InvalidArgument { op, reason },
        other => other,
    }
}

fn to_isize(v: usize) -> Result<isize> {
    isize::try_from(v).map_err(|_| too_large("slice"))
}

/// Concatenates tensors of one rank along `dim`. The caller checked dtypes, ranks, devices and
/// shapes; a rank or kind that differs anyway is an error, not a panic.
fn cat_inner(inners: Vec<Inner>, dim: usize) -> Result<Inner> {
    macro_rules! cat {
        ($($variant:ident),*) => {
            match inners.first() {
                $(Some(Inner::$variant(_)) => {
                    let parts = inners
                        .into_iter()
                        .map(|inner| match inner {
                            Inner::$variant(t) => Ok(t),
                            other => Err(TensorError::ShapeMismatch {
                                op: "concat",
                                lhs: Vec::new(),
                                rhs: other.dims(),
                            }),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    Ok(Inner::$variant(Ranked::cat(parts, dim)?))
                })*
                None => Err(TensorError::InvalidArgument {
                    op: "concat",
                    reason: "needs at least one input".into(),
                }),
            }
        };
    }
    cat!(R1, R2, R3, R4, R5, R6)
}

impl<const D: usize> Ranked<D> {
    /// The caller checked that the element counts match.
    pub(crate) fn reshape<const N: usize>(self, shape: &[usize]) -> Result<Ranked<N>> {
        let dims: [usize; N] = shape
            .try_into()
            .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.reshape(dims)),
            Ranked::Int(t) => Ranked::Int(t.reshape(dims)),
            Ranked::Bool(t) => Ranked::Bool(t.reshape(dims)),
        })
    }

    /// The caller checked that `dims` is a permutation of `0..D`.
    fn permute(self, dims: &[usize]) -> Result<Self> {
        let axes: [usize; D] = dims
            .try_into()
            .map_err(|_| TensorError::UnsupportedRank { rank: dims.len() })?;
        Ok(match self {
            Ranked::Float(t) => Ranked::Float(t.permute(axes)),
            Ranked::Int(t) => Ranked::Int(t.permute(axes)),
            Ranked::Bool(t) => Ranked::Bool(t.permute(axes)),
        })
    }

    /// The caller checked both dims.
    fn swap_dims(self, dim0: usize, dim1: usize) -> Self {
        match self {
            Ranked::Float(t) => Ranked::Float(t.swap_dims(dim0, dim1)),
            Ranked::Int(t) => Ranked::Int(t.swap_dims(dim0, dim1)),
            Ranked::Bool(t) => Ranked::Bool(t.swap_dims(dim0, dim1)),
        }
    }

    /// The caller checked the dim and the bounds.
    fn slice_dim(self, dim: usize, slice: Slice) -> Self {
        match self {
            Ranked::Float(t) => Ranked::Float(t.slice_dim(dim, slice)),
            Ranked::Int(t) => Ranked::Int(t.slice_dim(dim, slice)),
            Ranked::Bool(t) => Ranked::Bool(t.slice_dim(dim, slice)),
        }
    }

    /// The caller checked that the parts have one kind and fitting shapes, and `dim < D`.
    pub(crate) fn cat(parts: Vec<Self>, dim: usize) -> Result<Self> {
        macro_rules! collect {
            ($kind:ident) => {
                parts
                    .into_iter()
                    .map(|part| match part {
                        Ranked::$kind(t) => Ok(t),
                        other => Err(other.kind_error("concat")),
                    })
                    .collect::<Result<Vec<_>>>()?
            };
        }
        Ok(match parts.first() {
            Some(Ranked::Float(_)) => Ranked::Float(Tensor::cat(collect!(Float), dim)),
            Some(Ranked::Int(_)) => Ranked::Int(Tensor::cat(collect!(Int), dim)),
            Some(Ranked::Bool(_)) => Ranked::Bool(Tensor::cat(collect!(Bool), dim)),
            None => {
                return Err(TensorError::InvalidArgument {
                    op: "concat",
                    reason: "needs at least one input".into(),
                });
            }
        })
    }

    /// Pads dim `dim` with `left` and `right` elements of `fill`: `concat` of a `full` block, the
    /// tensor and another `full` block. The caller checked the sizes and that `fill` fits `dtype`.
    fn pad_dim(
        self,
        dim: usize,
        left: usize,
        right: usize,
        fill: Fill,
        dtype: DType,
    ) -> Result<Self> {
        let device = self.device();
        let shape = self.dims();
        let block = |width: usize| -> Result<Self> {
            let mut block_shape = shape.clone();
            block_shape[dim] = width;
            let block_shape: [usize; D] = block_shape
                .try_into()
                .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
            Ranked::full(block_shape, fill, dtype, &device)
        };
        let mut parts = Vec::with_capacity(3);
        if left > 0 {
            parts.push(block(left)?);
        }
        parts.push(self);
        if right > 0 {
            parts.push(block(right)?);
        }
        Ranked::cat(parts, dim)
    }
}

#[cfg(test)]
#[path = "shape_tests.rs"]
mod tests;
