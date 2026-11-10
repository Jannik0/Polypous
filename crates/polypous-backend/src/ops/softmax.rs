//! `core.softmax` and `core.log_softmax`: graph-ir.md §3.2, "Other". General rules: §3.1 (see
//! the module documentation of [`crate::ops`]).
//!
//! Both subtract the row maximum before `exp`, so large inputs (`1000`, `1001`) don't overflow:
//! `softmax = exp(x - m) / sum(exp(x - m))` and `log_softmax = (x - m) - log(sum(exp(x - m)))`.
//! `m` is the NaN-propagating maximum of `reduce.rs`. Special rows follow PyTorch: a row with a
//! NaN is NaN everywhere; a row with `+inf` is NaN everywhere (`inf - inf`); a row of only `-inf`
//! is NaN everywhere (`-inf - -inf`); `-inf` next to finite values gives `0` (`-inf` for
//! `log_softmax`). `f16`/`bf16` compute in `f32` and round once (graph-ir.md §3.1.8).

use burn::tensor::Tensor;

use super::reduce::{accumulate_in_f32, extremum_dim, shape_array};
use super::{DTypeClass, check_axis};
use crate::tensor::{DynTensor, Ranked, Result, map_rank};

impl DynTensor {
    /// `core.softmax` along `dim`. Float only. See the module documentation for the special
    /// rows.
    pub fn softmax(self, dim: usize) -> Result<Self> {
        self.softmax_op("softmax", dim, false)
    }

    /// `core.log_softmax` along `dim`. Float only. Computed directly (not as `log(softmax)`), so
    /// it stays finite where `softmax` underflows to 0.
    pub fn log_softmax(self, dim: usize) -> Result<Self> {
        self.softmax_op("log_softmax", dim, true)
    }

    fn softmax_op(self, op: &'static str, dim: usize, log: bool) -> Result<Self> {
        DTypeClass::Float.check(op, self.dtype())?;
        check_axis(op, "dim", dim, self.rank())?;
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.softmax(op, dim, log)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

impl<const D: usize> Ranked<D> {
    /// The caller checked the dtype class and `dim`.
    fn softmax(self, op: &'static str, dim: usize, log: bool) -> Result<Self> {
        let Ranked::Float(x) = self else {
            return Err(self.kind_error(op));
        };
        let shape = shape_array::<D>(&x.dims())?;
        Ok(Ranked::Float(accumulate_in_f32(x, |x| {
            stable_softmax(x, dim, log, shape)
        })))
    }
}

fn stable_softmax<const D: usize>(
    x: Tensor<D>,
    dim: usize,
    log: bool,
    shape: [usize; D],
) -> Tensor<D> {
    let max = extremum_dim(x.clone(), dim, true);
    let shifted = x.sub(max.expand(shape));
    let exp = shifted.clone().exp();
    let total = exp.clone().sum_dim(dim);
    if log {
        shifted.sub(total.log().expand(shape))
    } else {
        exp.div(total.expand(shape))
    }
}

#[cfg(test)]
#[path = "softmax_tests.rs"]
mod tests;
