//! Reductions: graph-ir.md §3.2, "Reduction" (`core.sum mean max min argmax argmin`), and
//! `core.cumsum` from "Other". General rules: §3.1 (see the module documentation of
//! [`crate::ops`], "Reductions, matmul, softmax and top_k").
//!
//! Every reduction runs along one axis at a time with Burn's `*_dim` ops, which keep the axis
//! with size 1; the result is reshaped to the output shape at the end. NaN, ties and integer
//! overflow are set explicitly, so the results don't depend on a backend's reduction order:
//!
//! - `max`/`min`: NaN is replaced by `-inf`/`+inf` before the reduction, and the result is NaN
//!   where the reduced slice held a NaN.
//! - `argmax`/`argmin`: the candidates are the positions equal to the (NaN-free) extremum, or
//!   the NaN positions in a slice with a NaN; the result is the smallest candidate index (an
//!   integer `min`, exact in any order).
//! - Integer `sum` and `cumsum` wrap. On the CPU, `i64` `sum` and `i32`/`i64` `cumsum` run on the
//!   host with wrapping arithmetic, because Burn's `flex` uses plain `+` there, which panics on
//!   overflow in debug builds.
//! - `sum`, `mean` and `cumsum` of `f16`/`bf16` accumulate in `f32` and round once (graph-ir.md
//!   §3.1.8).

use burn::tensor::{DType as BurnDType, Device, FloatDType, Int, Tensor};

use super::{DTypeClass, backend_error, check_axis, check_shape};
use crate::device::DeviceId;
use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank};

/// The four value reductions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reduce {
    Sum,
    Mean,
    Max,
    Min,
}

impl Reduce {
    fn name(self) -> &'static str {
        match self {
            Reduce::Sum => "sum",
            Reduce::Mean => "mean",
            Reduce::Max => "max",
            Reduce::Min => "min",
        }
    }

    fn class(self) -> DTypeClass {
        match self {
            Reduce::Mean => DTypeClass::Float,
            _ => DTypeClass::Numeric,
        }
    }
}

impl DynTensor {
    /// `core.sum`: the sum over `dims` (all dims for `None`), keeping them as size 1 with
    /// `keepdim`. Float or int; the output dtype is the input dtype (`i32` stays `i32`), and
    /// integers wrap on overflow. A reduction that removes every dim gives shape `[1]`.
    ///
    /// `dims` must be non-empty, in range and without duplicates ([`TensorError::InvalidArgument`]
    /// otherwise). `f16`/`bf16` accumulate in `f32` and round once.
    pub fn sum(self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        self.reduce(Reduce::Sum, dims, keepdim)
    }

    /// `core.mean`: the mean over `dims`, like [`DynTensor::sum`]. Float only. `f16`/`bf16`
    /// accumulate in `f32` (so the intermediate sum doesn't overflow) and round once.
    pub fn mean(self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        self.reduce(Reduce::Mean, dims, keepdim)
    }

    /// `core.max`: the largest value over `dims` (PyTorch's `amax`), like [`DynTensor::sum`].
    /// Float or int. NaN propagates: the result is NaN wherever a reduced value is NaN.
    pub fn max(self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        self.reduce(Reduce::Max, dims, keepdim)
    }

    /// `core.min`: the smallest value over `dims` (PyTorch's `amin`), like [`DynTensor::max`].
    pub fn min(self, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        self.reduce(Reduce::Min, dims, keepdim)
    }

    /// `core.argmax`: the index of the largest value along `dim`, as `dtype` (`i64` or `i32`).
    /// Float or int input. Ties give the lowest index; NaN counts as larger than every number, so
    /// the index of the first NaN is returned (graph-ir.md §3.1.8).
    ///
    /// `dim` must be in range; `dtype` must be an integer dtype the device supports, and `i32`
    /// needs the size of `dim` to fit it.
    pub fn argmax(self, dim: usize, keepdim: bool, dtype: DType) -> Result<Self> {
        self.arg_reduce("argmax", true, dim, keepdim, dtype)
    }

    /// `core.argmin`: the index of the smallest value along `dim`, like [`DynTensor::argmax`].
    /// NaN counts as smaller than every number, so the index of the first NaN is returned.
    pub fn argmin(self, dim: usize, keepdim: bool, dtype: DType) -> Result<Self> {
        self.arg_reduce("argmin", false, dim, keepdim, dtype)
    }

    /// `core.cumsum`: the cumulative sum along `dim`. Float or int; the output dtype is the
    /// input dtype, and integers wrap on overflow. `f16`/`bf16` accumulate in `f32` and round
    /// once.
    pub fn cumsum(self, dim: usize) -> Result<Self> {
        const OP: &str = "cumsum";
        DTypeClass::Numeric.check(OP, self.dtype())?;
        check_axis(OP, "dim", dim, self.rank())?;
        let (inner, dtype, device) = self.into_parts();
        let host = device == DeviceId::Cpu && dtype.is_int();
        let inner = map_rank!(inner, t => t.cumsum(dim, host)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    fn reduce(self, op: Reduce, dims: Option<&[usize]>, keepdim: bool) -> Result<Self> {
        let name = op.name();
        let dtype = self.dtype();
        op.class().check(name, dtype)?;
        let shape = self.shape();
        let axes = reduce_axes(name, dims, shape.len())?;
        let out_shape = reduced_shape(&shape, &axes, keepdim);
        check_shape(name, &out_shape, dtype)?;
        let device = self.device();
        let host = op == Reduce::Sum && dtype == DType::I64 && device == DeviceId::Cpu;
        let (inner, _, _) = self.into_parts();
        let inner = map_rank!(inner, t => t.reduce(op, &axes, host)?);
        DynTensor::from_parts(inner, dtype, device).reshape(&out_shape)
    }

    fn arg_reduce(
        self,
        op: &'static str,
        largest: bool,
        dim: usize,
        keepdim: bool,
        dtype: DType,
    ) -> Result<Self> {
        DTypeClass::Numeric.check(op, self.dtype())?;
        let shape = self.shape();
        check_axis(op, "dim", dim, shape.len())?;
        check_index_dtype(op, dtype, shape[dim], self.device())?;
        let out_shape = reduced_shape(&shape, &[dim], keepdim);
        check_shape(op, &out_shape, dtype)?;
        let device = self.device();
        let (inner, _, _) = self.into_parts();
        let inner = map_rank!(inner, t => t.arg_reduce(op, largest, dim, dtype)?);
        DynTensor::from_parts(inner, dtype, device).reshape(&out_shape)
    }
}

/// The axes of a reduction, sorted: all of `0..rank` for `None`. An empty list, an axis out of
/// range or a duplicate is `InvalidArgument`.
pub(crate) fn reduce_axes(
    op: &'static str,
    dims: Option<&[usize]>,
    rank: usize,
) -> Result<Vec<usize>> {
    let Some(dims) = dims else {
        return Ok((0..rank).collect());
    };
    if dims.is_empty() {
        return Err(TensorError::InvalidArgument {
            op,
            reason: "`dim` must not be empty (leave it out to reduce every dim)".into(),
        });
    }
    let mut axes = Vec::with_capacity(dims.len());
    for &axis in dims {
        check_axis(op, "dim", axis, rank)?;
        if axes.contains(&axis) {
            return Err(TensorError::InvalidArgument {
                op,
                reason: format!("`dim` lists axis {axis} twice"),
            });
        }
        axes.push(axis);
    }
    axes.sort_unstable();
    Ok(axes)
}

/// The output shape of a reduction over `axes` (graph-ir.md §3.2 "Reduction" note): reduced
/// dims become `1` with `keepdim` and are removed otherwise; removing every dim gives `[1]`.
pub(crate) fn reduced_shape(shape: &[usize], axes: &[usize], keepdim: bool) -> Vec<usize> {
    let mut out = Vec::with_capacity(shape.len());
    for (i, &d) in shape.iter().enumerate() {
        if !axes.contains(&i) {
            out.push(d);
        } else if keepdim {
            out.push(1);
        }
    }
    if out.is_empty() {
        out.push(1);
    }
    out
}

/// Checks the `dtype` attribute of `argmax`, `argmin` and `top_k`: an integer dtype the device
/// supports (`UnsupportedDType` otherwise), able to hold every index below `size`.
pub(crate) fn check_index_dtype(
    op: &'static str,
    dtype: DType,
    size: usize,
    device: DeviceId,
) -> Result<()> {
    if !dtype.is_int() {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("the index dtype must be i64 or i32, not {dtype}"),
        });
    }
    if dtype == DType::I32 && i32::try_from(size).is_err() {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("a dim of size {size} has indices that don't fit i32"),
        });
    }
    if !DynTensor::device_supports(device, dtype)? {
        return Err(TensorError::UnsupportedDType {
            dtype,
            device: device.to_string(),
        });
    }
    Ok(())
}

/// The Burn dtype of an integer index dtype.
fn index_burn_dtype(dtype: DType) -> BurnDType {
    if dtype == DType::I32 {
        BurnDType::I32
    } else {
        BurnDType::I64
    }
}

/// Runs `f` on `x` in `f32` and rounds the result back to `x`'s dtype for `f16` and `bf16`;
/// `f32` runs as it is. Used for the accumulating ops (`sum`, `mean`, `cumsum`, `softmax`,
/// `log_softmax`; graph-ir.md §3.1.8). Without it, on lavapipe Burn's CubeCL `cumsum` accumulates
/// `f16` in `f16` (the cumulative sum of 4096 ones ends at 2048), `mean` (a sum, then a division)
/// overflows to `inf` where the sum leaves the `f16` range, and `softmax` in `bf16` misses
/// `rtol 1e-2` on the CPU (each step rounds to `bf16`). The output dtype is unchanged.
pub(crate) fn accumulate_in_f32<const D: usize>(
    x: Tensor<D>,
    f: impl FnOnce(Tensor<D>) -> Tensor<D>,
) -> Tensor<D> {
    let dtype = match x.dtype() {
        BurnDType::F16 => FloatDType::F16,
        BurnDType::BF16 => FloatDType::BF16,
        _ => return f(x),
    };
    f(x.cast(FloatDType::F32)).cast(dtype)
}

/// `max` (or `min`) of a float tensor along `dim`, keeping it as size 1, with NaN propagation:
/// NaN where the slice holds a NaN, independent of how the backend's `max_dim` treats NaN.
pub(crate) fn extremum_dim<const D: usize>(x: Tensor<D>, dim: usize, largest: bool) -> Tensor<D> {
    let nan = x.clone().is_nan();
    let fill = if largest {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };
    let filled = x.mask_fill(nan.clone(), fill);
    let out = if largest {
        filled.max_dim(dim)
    } else {
        filled.min_dim(dim)
    };
    out.mask_fill(any_dim(nan, dim), f64::NAN)
}

/// Whether any element along `dim` is true, keeping `dim` as size 1. Computed as the `max` of
/// the mask as `f32` (exact for 0 and 1), which needs no bool reduction kernel.
fn any_dim<const D: usize>(
    mask: Tensor<D, burn::tensor::Bool>,
    dim: usize,
) -> Tensor<D, burn::tensor::Bool> {
    mask.float()
        .cast(FloatDType::F32)
        .max_dim(dim)
        .greater_elem(0.5)
}

/// `0, 1, …, n-1` along `dim` of a rank-`D` index tensor of `shape`, in `dtype` (`I32`/`I64`).
fn positions<const D: usize>(
    shape: &[usize],
    dim: usize,
    dtype: BurnDType,
    device: &Device,
) -> Result<Tensor<D, Int>> {
    let n = shape[dim];
    let end = i64::try_from(n).map_err(|_| TensorError::InvalidArgument {
        op: "positions",
        reason: format!("dim size {n} is too large"),
    })?;
    let mut dims = [1usize; D];
    dims[dim] = n;
    let full: [usize; D] = shape
        .try_into()
        .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
    Ok(Tensor::<1, Int>::arange(0..end, (device, dtype))
        .reshape(dims)
        .expand(full))
}

impl<const D: usize> Ranked<D> {
    /// The caller checked the dtype class and the axes (sorted, unique, in range).
    fn reduce(self, op: Reduce, axes: &[usize], host: bool) -> Result<Self> {
        match self {
            Ranked::Float(x) => Ok(Ranked::Float(match op {
                Reduce::Sum => accumulate_in_f32(x, |x| sum_axes(x, axes)),
                Reduce::Mean => {
                    let count: usize = axes.iter().map(|&a| x.dims()[a]).product();
                    // Exact up to 2^53 elements; tensors here have far fewer.
                    let count = count as f64;
                    accumulate_in_f32(x, |x| sum_axes(x, axes).div_scalar(count))
                }
                Reduce::Max | Reduce::Min => {
                    let largest = op == Reduce::Max;
                    axes.iter()
                        .fold(x, |x, &axis| extremum_dim(x, axis, largest))
                }
            })),
            Ranked::Int(x) => Ok(Ranked::Int(match op {
                Reduce::Sum if host => host_sum_i64(x, axes)?,
                Reduce::Sum => axes.iter().fold(x, |x, &axis| x.sum_dim(axis)),
                Reduce::Max => axes.iter().fold(x, |x, &axis| x.max_dim(axis)),
                Reduce::Min => axes.iter().fold(x, |x, &axis| x.min_dim(axis)),
                Reduce::Mean => return Err(Ranked::Int(x).kind_error(op.name())),
            })),
            x => Err(x.kind_error(op.name())),
        }
    }

    /// The caller checked the dtype class, `dim` and the index dtype.
    fn arg_reduce(self, op: &'static str, largest: bool, dim: usize, dtype: DType) -> Result<Self> {
        let index_dtype = index_burn_dtype(dtype);
        let (candidates, shape, device) = match self {
            Ranked::Float(x) => {
                let (shape, device) = (x.dims(), x.device());
                let nan = x.clone().is_nan();
                let has_nan = any_dim(nan.clone(), dim).expand(shape_array::<D>(&shape)?);
                let fill = if largest {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                };
                let filled = x.mask_fill(nan.clone(), fill);
                let best = if largest {
                    filled.clone().max_dim(dim)
                } else {
                    filled.clone().min_dim(dim)
                };
                let equal = filled.equal(best.expand(shape_array::<D>(&shape)?));
                (equal.mask_where(has_nan, nan), shape, device)
            }
            Ranked::Int(x) => {
                let (shape, device) = (x.dims(), x.device());
                let best = if largest {
                    x.clone().max_dim(dim)
                } else {
                    x.clone().min_dim(dim)
                };
                (
                    x.equal(best.expand(shape_array::<D>(&shape)?)),
                    shape,
                    device,
                )
            }
            x => return Err(x.kind_error(op)),
        };
        let n = i64::try_from(shape[dim]).map_err(|_| TensorError::InvalidArgument {
            op,
            reason: "dim is too large".into(),
        })?;
        let index = positions::<D>(&shape, dim, index_dtype, &device)?;
        Ok(Ranked::Int(
            index.mask_fill(candidates.bool_not(), n).min_dim(dim),
        ))
    }

    /// The caller checked the dtype class and `dim`. `host`: integers on the CPU, where `flex`
    /// would panic on overflow.
    fn cumsum(self, dim: usize, host: bool) -> Result<Self> {
        match self {
            Ranked::Float(x) => Ok(Ranked::Float(accumulate_in_f32(x, |x| x.cumsum(dim)))),
            Ranked::Int(x) if host => Ok(Ranked::Int(host_cumsum(x, dim)?)),
            Ranked::Int(x) => Ok(Ranked::Int(x.cumsum(dim))),
            x => Err(x.kind_error("cumsum")),
        }
    }
}

/// `shape` as an array of rank `D`, or `UnsupportedRank`.
pub(crate) fn shape_array<const D: usize>(shape: &[usize]) -> Result<[usize; D]> {
    shape
        .try_into()
        .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })
}

fn sum_axes<const D: usize>(x: Tensor<D>, axes: &[usize]) -> Tensor<D> {
    axes.iter().fold(x, |x, &axis| x.sum_dim(axis))
}

/// Row-major strides of `shape`.
fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// `i64` sum over `axes` on the host with `wrapping_add`, keeping the axes as size 1 (see the
/// module documentation). The tensor is read back row-major.
fn host_sum_i64<const D: usize>(x: Tensor<D, Int>, axes: &[usize]) -> Result<Tensor<D, Int>> {
    let device = x.device();
    let shape = x.dims();
    let data = x.try_into_data().map_err(backend_error)?;
    let values = data.as_slice::<i64>().map_err(backend_error)?;
    let mut out_shape = shape;
    for &axis in axes {
        out_shape[axis] = 1;
    }
    let out_strides = strides(&out_shape);
    let mut out = vec![0i64; out_shape.iter().product()];
    let mut index = vec![0usize; shape.len()];
    for &v in values {
        let slot: usize = index
            .iter()
            .zip(&out_strides)
            .enumerate()
            .filter(|(axis, _)| !axes.contains(axis))
            .map(|(_, (&i, &s))| i * s)
            .sum();
        out[slot] = out[slot].wrapping_add(v);
        // Advance the row-major multi-index.
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    let data = burn::tensor::TensorData::new(out, out_shape);
    Ok(Tensor::from_data(data, (&device, BurnDType::I64)))
}

/// Integer cumulative sum along `dim` on the host with wrapping arithmetic (`i32` or `i64`).
fn host_cumsum<const D: usize>(x: Tensor<D, Int>, dim: usize) -> Result<Tensor<D, Int>> {
    let device = x.device();
    let shape = x.dims();
    let data = x.try_into_data().map_err(backend_error)?;
    let n = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let data = match data.dtype() {
        BurnDType::I64 => {
            let values = data.as_slice::<i64>().map_err(backend_error)?;
            burn::tensor::TensorData::new(cumsum_rows(values, n, inner, i64::wrapping_add), shape)
        }
        BurnDType::I32 => {
            let values = data.as_slice::<i32>().map_err(backend_error)?;
            burn::tensor::TensorData::new(cumsum_rows(values, n, inner, i32::wrapping_add), shape)
        }
        other => {
            return Err(TensorError::Backend {
                message: format!("host cumsum: unexpected integer storage {other:?}"),
            });
        }
    };
    let dtype = data.dtype();
    Ok(Tensor::from_data(data, (&device, dtype)))
}

/// Cumulative sums of row-major `values` along an axis of size `n` with `inner` elements after
/// it.
fn cumsum_rows<T: Copy>(values: &[T], n: usize, inner: usize, add: fn(T, T) -> T) -> Vec<T> {
    let mut out = values.to_vec();
    let block = n * inner;
    if block == 0 {
        return out;
    }
    for start in (0..out.len()).step_by(block) {
        for j in 1..n {
            for i in 0..inner {
                let at = start + j * inner + i;
                out[at] = add(out[at - inner], out[at]);
            }
        }
    }
    out
}

#[cfg(test)]
#[path = "reduce_tests.rs"]
pub(crate) mod tests;
