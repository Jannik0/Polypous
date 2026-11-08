//! `core.cast`: graph-ir.md §3.2 ("Reduction and others", the `cast` note) and §3.1.8.

use burn::tensor::{Bool, Tensor};

use crate::tensor::{DType, DynTensor, Kind, Ranked, Result, check_device_supports, map_rank};

impl DynTensor {
    /// `core.cast`: converts the elements to `dtype`. Casting to the same dtype is a no-op.
    ///
    /// - float → float rounds to nearest, ties to even; out-of-range values become ±inf.
    /// - float → int rounds toward zero. NaN, infinities and values outside the integer range give
    ///   a backend-defined value, never a panic (graph-ir.md §3.1.8).
    /// - int → float rounds to nearest; int → int wraps (two's complement truncation).
    /// - any → `bool` gives `true` for non-zero elements (NaN is non-zero, `-0.0` is zero).
    /// - `bool` → float or int gives `1` and `0`.
    ///
    /// Fails with [`TensorError::UnsupportedDType`](crate::TensorError::UnsupportedDType) if the
    /// device doesn't support `dtype`.
    pub fn cast(self, dtype: DType) -> Result<Self> {
        if dtype == self.dtype() {
            return Ok(self);
        }
        check_device_supports(dtype, &self.burn_device(), self.device())?;
        let (inner, _, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.cast(dtype));
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// Float → `bool`: non-zero is `true`, and so is NaN. Burn's `bool()` is `x != 0`, which the
/// CubeCL backends compute as an ordered comparison (NaN gives `false`; seen on lavapipe), so
/// NaN is or-ed in from an explicit mask.
fn float_to_bool<const D: usize>(t: Tensor<D>) -> Tensor<D, Bool> {
    let nan = t.clone().is_nan();
    t.not_equal_elem(0.0).bool_or(nan)
}

impl<const D: usize> Ranked<D> {
    /// The caller checked that the device supports `dtype`.
    fn cast(self, dtype: DType) -> Self {
        match (self, dtype.kind()) {
            (Ranked::Float(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Float(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Float(t), Kind::Bool) => Ranked::Bool(float_to_bool(t)),
            (Ranked::Int(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Int(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Int(t), Kind::Bool) => Ranked::Bool(t.bool()),
            (Ranked::Bool(t), Kind::Float) => Ranked::Float(t.cast(dtype.to_burn_float())),
            (Ranked::Bool(t), Kind::Int) => Ranked::Int(t.cast(dtype.to_burn_int())),
            (Ranked::Bool(t), Kind::Bool) => Ranked::Bool(t),
        }
    }
}
