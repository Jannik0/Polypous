//! Creation ops: graph-ir.md §3.2 "Creation" (`core.constant arange zeros ones full
//! full_like`), and the fill values they share with `core.pad` (§3.1.7, `scalar`).

use burn::tensor::{Bool, Device, Int, Tensor, bf16, f16};

use super::{DTypeClass, Scalar, check_shape};
use crate::device::DeviceId;
use crate::tensor::{
    DType, DynTensor, Inner, Kind, Ranked, Result, TensorData, TensorError, check_device_supports,
    dispatch_rank, resolve,
};

/// Most elements of a `core.constant` (graph-ir.md §4.3).
pub(crate) const MAX_CONSTANT_ELEMENTS: usize = 65_536;

/// A fill value converted to a dtype's kind (§3.1.7). A float value is already rounded to the
/// dtype, so converting it to the dtype's element type is exact on every backend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Fill {
    Float(f64),
    Int(i64),
    Bool(bool),
}

/// Converts a scalar argument to a fill value of `dtype` (§3.1.7): `bool` takes a bool, `i32` and
/// `i64` an in-range integer, float dtypes a float or an integer. Floats are rounded from `f64` to
/// `f32`, then to `f16`/`bf16`, as PyTorch rounds Python floats (`torch.full`, `torch.tensor`).
/// `InvalidArgument` if the value doesn't fit.
pub(crate) fn fill_value(op: &'static str, value: Scalar, dtype: DType) -> Result<Fill> {
    let mismatch = || TensorError::InvalidArgument {
        op,
        reason: format!("value {value:?} does not fit dtype {dtype}"),
    };
    match (dtype.kind(), value) {
        (Kind::Bool, Scalar::Bool(b)) => Ok(Fill::Bool(b)),
        (Kind::Int, Scalar::Int(v)) => {
            if dtype == DType::I32 && i32::try_from(v).is_err() {
                return Err(TensorError::InvalidArgument {
                    op,
                    reason: format!("value {v} is outside the i32 range"),
                });
            }
            Ok(Fill::Int(v))
        }
        // Integers above 2^53 round to f64 first, as the IR stores them (polypous-ir `Scalar`).
        (Kind::Float, Scalar::Int(v)) => Ok(Fill::Float(round_to(v as f64, dtype))),
        (Kind::Float, Scalar::Float(v)) => Ok(Fill::Float(round_to(v, dtype))),
        _ => Err(mismatch()),
    }
}

/// `v` rounded to the float dtype `dtype` (to nearest, ties to even), through `f32` like PyTorch.
/// NaN and infinities stay as they are; values beyond the dtype's range become infinities.
fn round_to(v: f64, dtype: DType) -> f64 {
    let single = v as f32;
    match dtype {
        DType::F16 => f16::from_f32(single).to_f64(),
        DType::BF16 => bf16::from_f32(single).to_f64(),
        _ => f64::from(single),
    }
}

impl DynTensor {
    /// `core.constant`: a tensor of `dtype` and `shape` on `device`, with `values` in row-major
    /// order (exactly one per element, at most 65,536). Each value follows the fill rules of the
    /// module documentation of [`crate::ops`].
    ///
    /// Fails with [`TensorError::UnsupportedDType`] if the device doesn't support `dtype`, and
    /// with [`TensorError::InvalidArgument`] for a wrong count or a value that doesn't fit.
    pub fn constant(
        dtype: DType,
        shape: &[usize],
        values: &[Scalar],
        device: DeviceId,
    ) -> Result<Self> {
        const OP: &str = "constant";
        if values.len() > MAX_CONSTANT_ELEMENTS {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("{} values, at most {MAX_CONSTANT_ELEMENTS}", values.len()),
            });
        }
        check_shape(OP, shape, dtype)?;
        let count: usize = shape.iter().product();
        if count != values.len() {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "shape {shape:?} has {count} elements, but {} values are given",
                    values.len()
                ),
            });
        }
        let mut bytes = Vec::with_capacity(count * dtype.size_bytes());
        for &value in values {
            match (fill_value(OP, value, dtype)?, dtype) {
                (Fill::Bool(b), _) => bytes.push(u8::from(b)),
                (Fill::Int(v), DType::I32) => {
                    // In range: checked by `fill_value`.
                    bytes.extend_from_slice(&(v as i32).to_le_bytes());
                }
                (Fill::Int(v), _) => bytes.extend_from_slice(&v.to_le_bytes()),
                // Exact: `fill_value` rounded the value to the dtype.
                (Fill::Float(v), DType::F16) => {
                    bytes.extend_from_slice(&f16::from_f64(v).to_le_bytes());
                }
                (Fill::Float(v), DType::BF16) => {
                    bytes.extend_from_slice(&bf16::from_f64(v).to_le_bytes());
                }
                (Fill::Float(v), _) => bytes.extend_from_slice(&(v as f32).to_le_bytes()),
            }
        }
        let shape = shape.iter().map(|&d| d as u64).collect();
        DynTensor::from_data(&TensorData::new(dtype, shape, bytes)?, device)
    }

    /// `core.arange`: `start, start + step, …` below `end`, as a rank-1 tensor of the numeric
    /// `dtype` on `device`. Needs `step ≥ 1` and `start < end` (at least one element).
    ///
    /// Values are computed as integers and converted once to `dtype`, so they are exact while they
    /// are below 2^24 (`f32`), 2^11 (`f16`) or 2^8 (`bf16`) (graph-ir.md §3.2), and round to
    /// nearest beyond. For `i32` every value must fit, and for the other dtypes every value must
    /// be below `i64::MAX` (`InvalidArgument` otherwise).
    pub fn arange(
        start: usize,
        end: usize,
        step: usize,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self> {
        const OP: &str = "arange";
        DTypeClass::Numeric.check(OP, dtype)?;
        if step == 0 || start >= end {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "start {start}, end {end}, step {step}: needs step ≥ 1 and start < end"
                ),
            });
        }
        let len = (end - start).div_ceil(step);
        check_shape(OP, &[len], dtype)?;
        // The last value is below `end`. It must fit the dtype for an integer result, and stay
        // below i64::MAX so the exclusive range end `last + 1` fits i64.
        let last = start + (len - 1) * step;
        let limit = if dtype == DType::I32 {
            i32::MAX as u64
        } else {
            i64::MAX as u64 - 1
        };
        if last as u64 > limit {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("value {last} is outside the range of {dtype} arange"),
            });
        }
        let burn_device = resolve(device)?;
        check_device_supports(dtype, &burn_device, device)?;
        // Integers are generated in i64, or in i32 when every value fits (so a float result
        // doesn't need i64 support on the device).
        let int_dtype = if dtype == DType::I64 || last as u64 > i32::MAX as u64 {
            DType::I64
        } else {
            DType::I32
        };
        check_device_supports(int_dtype, &burn_device, device)?;
        // `end` may not fit i64, so the range ends at `last + 1`, which gives the same values.
        let range = (start as i64)..(last as i64 + 1);
        let ints = Tensor::<1, Int>::arange_step(range, step, (&burn_device, int_dtype.to_burn()));
        let ranked = match dtype.kind() {
            Kind::Float => Ranked::Float(ints.cast(dtype.to_burn_float())),
            _ => Ranked::Int(ints.cast(dtype.to_burn_int())),
        };
        Ok(DynTensor::from_parts(Inner::R1(ranked), dtype, device))
    }

    /// `core.zeros`: a tensor of `shape` filled with zeros (`false` for `bool`).
    pub fn zeros(shape: &[usize], dtype: DType, device: DeviceId) -> Result<Self> {
        Self::fill("zeros", shape, unit(dtype, false), dtype, device)
    }

    /// `core.ones`: a tensor of `shape` filled with ones (`true` for `bool`).
    pub fn ones(shape: &[usize], dtype: DType, device: DeviceId) -> Result<Self> {
        Self::fill("ones", shape, unit(dtype, true), dtype, device)
    }

    /// `core.full`: a tensor of `shape` filled with `value` (fill rules in the module
    /// documentation of [`crate::ops`]; NaN and infinities are allowed for float dtypes).
    pub fn full(shape: &[usize], value: Scalar, dtype: DType, device: DeviceId) -> Result<Self> {
        Self::fill("full", shape, value, dtype, device)
    }

    /// `core.full_like`: a tensor with the shape and device of `self`, filled with `value`, of
    /// `dtype` (default: `self`'s dtype). Only the shape of `self` is used, not its values.
    pub fn full_like(&self, value: Scalar, dtype: Option<DType>) -> Result<Self> {
        let dtype = dtype.unwrap_or(self.dtype());
        Self::fill("full_like", &self.shape(), value, dtype, self.device())
    }

    fn fill(
        op: &'static str,
        shape: &[usize],
        value: Scalar,
        dtype: DType,
        device: DeviceId,
    ) -> Result<Self> {
        check_shape(op, shape, dtype)?;
        let fill = fill_value(op, value, dtype)?;
        let burn_device = resolve(device)?;
        check_device_supports(dtype, &burn_device, device)?;
        let inner = dispatch_rank!(shape.len(), N => {
            let dims: [usize; N] = shape
                .try_into()
                .map_err(|_| TensorError::UnsupportedRank { rank: shape.len() })?;
            Ranked::<N>::full(dims, fill, dtype, &burn_device)?
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// `0`/`1` (`false`/`true`) as a scalar of `dtype`'s kind.
fn unit(dtype: DType, one: bool) -> Scalar {
    match dtype.kind() {
        Kind::Bool => Scalar::Bool(one),
        Kind::Int => Scalar::Int(i64::from(one)),
        Kind::Float => Scalar::Float(if one { 1.0 } else { 0.0 }),
    }
}

impl<const D: usize> Ranked<D> {
    /// A tensor of `shape` filled with `fill`, of `dtype` on `device`. The caller checked the
    /// shape, the device's dtype support and that `fill` has `dtype`'s kind.
    pub(crate) fn full(
        shape: [usize; D],
        fill: Fill,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Ok(match (fill, dtype.kind()) {
            (Fill::Float(v), Kind::Float) => {
                Ranked::Float(Tensor::full(shape, v, (device, dtype.to_burn())))
            }
            (Fill::Int(v), Kind::Int) => {
                Ranked::Int(Tensor::full(shape, v, (device, dtype.to_burn())))
            }
            // Burn creates bool tensors in the device's own bool storage.
            (Fill::Bool(b), Kind::Bool) => Ranked::Bool(Tensor::<D, Bool>::full(shape, b, device)),
            _ => {
                return Err(TensorError::InvalidArgument {
                    op: "full",
                    reason: format!("{fill:?} does not fit dtype {dtype}"),
                });
            }
        })
    }
}

#[cfg(test)]
#[path = "creation_tests.rs"]
mod tests;
