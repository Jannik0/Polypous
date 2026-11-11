//! Activations: graph-ir.md §3.3, "Activations" (`nn.relu leaky_relu gelu gelu_tanh silu sigmoid
//! softplus mish`). General rules: §3.1 (see the module documentation of [`crate::ops`]).
//!
//! Every activation is written out here from Burn's elementary ops, not taken from
//! `burn::tensor::activation`, so that NaN, infinities, large inputs and the internal precision
//! are the same on every backend:
//!
//! - **Internal precision:** every activation except `relu` computes `f16`/`bf16` in `f32` and
//!   rounds once to the dtype (graph-ir.md §3.1.8, "Half-precision transcendentals"; `relu` only
//!   selects). The output dtype is unchanged.
//! - **NaN propagates** through every activation: selections test `x < 0` or `x > t`, which are
//!   false for NaN, so the NaN input itself (or arithmetic on it) reaches the output. Burn's
//!   `relu` is a `max(x, 0)` on some backends, which may drop NaN.
//! - **No `exp` overflow:** `sigmoid` is `1 / (1 + exp(-x))`, whose `exp` overflows to `inf` only
//!   where the result is `0` anyway; `softplus` uses the overflow-free form
//!   `max(z, 0) + log1p(exp(-|z|))` (with `z = beta·x`), with a `log1p` of its own that stays
//!   accurate for tiny arguments (CubeCL's is `ln(1 + x)`); `mish` takes the `tanh` of that
//!   (`≥ 0`) with a series for tiny arguments, where lavapipe's `tanh` is noise; `gelu_tanh` uses
//!   `0.5·(1 + tanh(u)) = sigmoid(2u)`, so neither needs `tanh` of large arguments.
//! - **Infinities** follow the formulas, as in PyTorch: `silu(-inf)`, `mish(-inf)` and
//!   `gelu_tanh(-inf)` are NaN (`-inf · 0`), `gelu(-inf)` too. `gelu(+inf)` is `+inf`
//!   (PyTorch's `f32` CPU kernel gives NaN there, its `f64` kernel `+inf`; fixtures leave `+inf`
//!   out of `gelu`).

use burn::tensor::Tensor;

use super::DTypeClass;
use super::reduce::accumulate_in_f32;
use crate::tensor::{DynTensor, Ranked, Result, TensorError, map_rank};

/// `sqrt(2 / π)`, the `gelu_tanh` scale.
const SQRT_2_OVER_PI: f64 = 0.797_884_560_802_865_4;

/// The activations, with their attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Activation {
    Relu,
    LeakyRelu { negative_slope: f64 },
    Gelu,
    GeluTanh,
    Silu,
    Sigmoid,
    Softplus { beta: f64, threshold: f64 },
    Mish,
}

impl Activation {
    fn name(self) -> &'static str {
        match self {
            Activation::Relu => "relu",
            Activation::LeakyRelu { .. } => "leaky_relu",
            Activation::Gelu => "gelu",
            Activation::GeluTanh => "gelu_tanh",
            Activation::Silu => "silu",
            Activation::Sigmoid => "sigmoid",
            Activation::Softplus { .. } => "softplus",
            Activation::Mish => "mish",
        }
    }
}

impl DynTensor {
    /// `nn.relu`: `max(x, 0)`, with NaN propagated and `relu(-0.0) = -0.0` (as `F.relu`). Float
    /// only.
    pub fn relu(self) -> Result<Self> {
        self.activation(Activation::Relu)
    }

    /// `nn.leaky_relu`: `x` for `x ≥ 0`, `negative_slope · x` below. Float only.
    /// `negative_slope` must be finite ([`TensorError::InvalidArgument`]); any sign is allowed,
    /// as in PyTorch.
    pub fn leaky_relu(self, negative_slope: f64) -> Result<Self> {
        if !negative_slope.is_finite() {
            return Err(TensorError::InvalidArgument {
                op: "leaky_relu",
                reason: format!("`negative_slope` must be finite, got {negative_slope}"),
            });
        }
        self.activation(Activation::LeakyRelu { negative_slope })
    }

    /// `nn.gelu`: the exact GELU, `x · 0.5 · (1 + erf(x / sqrt(2)))` (`F.gelu`). Float only.
    pub fn gelu(self) -> Result<Self> {
        self.activation(Activation::Gelu)
    }

    /// `nn.gelu_tanh`: the tanh approximation of GELU,
    /// `0.5 · x · (1 + tanh(sqrt(2/π) · (x + 0.044715 · x³)))`
    /// (`F.gelu(x, approximate="tanh")`), computed as `x · sigmoid(2u)`. Float only.
    pub fn gelu_tanh(self) -> Result<Self> {
        self.activation(Activation::GeluTanh)
    }

    /// `nn.silu`: `x · sigmoid(x)`. Float only.
    pub fn silu(self) -> Result<Self> {
        self.activation(Activation::Silu)
    }

    /// `nn.sigmoid`: `1 / (1 + exp(-x))`. Float only.
    pub fn sigmoid(self) -> Result<Self> {
        self.activation(Activation::Sigmoid)
    }

    /// `nn.softplus`: `log(1 + exp(beta · x)) / beta`, and `x` itself where
    /// `beta · x > threshold` (`F.softplus`). Float only. `beta` and `threshold` must be finite
    /// and `> 0` ([`TensorError::InvalidArgument`]).
    pub fn softplus(self, beta: f64, threshold: f64) -> Result<Self> {
        for (name, value) in [("beta", beta), ("threshold", threshold)] {
            if !(value.is_finite() && value > 0.0) {
                return Err(TensorError::InvalidArgument {
                    op: "softplus",
                    reason: format!("`{name}` must be finite and > 0, got {value}"),
                });
            }
        }
        self.activation(Activation::Softplus { beta, threshold })
    }

    /// `nn.mish`: `x · tanh(softplus(x))`. Float only.
    pub fn mish(self) -> Result<Self> {
        self.activation(Activation::Mish)
    }

    fn activation(self, op: Activation) -> Result<Self> {
        DTypeClass::Float.check(op.name(), self.dtype())?;
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.activation(op)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

impl<const D: usize> Ranked<D> {
    /// The caller checked the dtype class and the attributes.
    fn activation(self, op: Activation) -> Result<Self> {
        let Ranked::Float(x) = self else {
            return Err(self.kind_error(op.name()));
        };
        Ok(Ranked::Float(match op {
            Activation::Relu => relu(x),
            op => accumulate_in_f32(x, |x| activate(x, op)),
        }))
    }
}

/// `max(x, 0)` as a selection: `x < 0` is false for NaN and `-0.0`, which therefore pass through.
fn relu<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let negative = x.clone().lower_elem(0.0);
    x.mask_fill(negative, 0.0)
}

/// `1 / (1 + exp(-x))`. `exp(-x)` overflows to `inf` only below about `-88` (in `f32`), where the
/// result is `0`, so the overflow gives the right value and no NaN.
pub(crate) fn sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.neg().exp().add_scalar(1.0).recip()
}

/// `softplus` with `beta` and `threshold`: `max(z, 0) / beta + log1p(exp(-|z|)) / beta` with
/// `z = beta · x`, which is `log(1 + exp(z)) / beta` without overflow, and `x` where
/// `z > threshold`.
fn softplus<const D: usize>(x: Tensor<D>, beta: f64, threshold: f64) -> Tensor<D> {
    let z = x.clone().mul_scalar(beta);
    let linear = z.clone().greater_elem(threshold);
    let smooth = relu(z.clone())
        .add(log1p(z.abs().neg().exp()))
        .div_scalar(beta);
    smooth.mask_where(linear, x)
}

/// `tanh(s)` for `s ≥ 0` (and NaN, `+inf`), accurate for tiny `s` on every backend. Mesa's
/// Vulkan `tanh` (lavapipe) returned noise of about `3e-8` for arguments near `1e-37`, so
/// `mish(-85)` came out near `-2.5e-6` instead of `-1.03e-35`. Below `0.1` this uses the series
/// `s - s³/3 + 2s⁵/15 - 17s⁷/315` (relative error below `3e-10`), above it `(1 - e) / (1 + e)`
/// with `e = exp(-2s)` in `(0, 1)`, which can't overflow (`tanh(+inf) = 1`) and whose relative
/// error there is below `1e-6`.
fn tanh_nonnegative<const D: usize>(s: Tensor<D>) -> Tensor<D> {
    let small = s.clone().lower_elem(0.1);
    let q = s.clone().mul(s.clone());
    // s · (1 - q · (1/3 - q · (2/15 - q · 17/315))), as plain multiply-adds.
    let inner = q.clone().mul_scalar(-17.0 / 315.0).add_scalar(2.0 / 15.0);
    let inner = q.clone().mul(inner).neg().add_scalar(1.0 / 3.0);
    let series = s.clone().mul(q.mul(inner).neg().add_scalar(1.0));
    let e = s.mul_scalar(-2.0).exp();
    let wide = e.clone().neg().add_scalar(1.0).div(e.add_scalar(1.0));
    wide.mask_where(small, series)
}

/// `log(1 + t)` for `t` in `[0, 1]` (and NaN), accurate for small `t` on every backend. Burn
/// 0.22.1's CubeCL `log1p` is `ln(1 + t)`, which rounds `1 + t` to 1 in `f32` below
/// `t ≈ 6e-8`: `softplus(-17)` gave 0 instead of `4.14e-8` on lavapipe and CUDA. Below `0.05`
/// this uses the series `t - t²/2 + t³/3 - t⁴/4 + t⁵/5` (relative error below `3e-8`), written as
/// plain multiply-adds that no compiler rewrites into `ln(1 + t)`; above it `ln(1 + t)`, whose
/// relative error there (one `f32` ulp of 1 against a value ≥ 0.048) is below `2e-6`.
fn log1p<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    let small = t.clone().lower_elem(0.05);
    // t · (1 - t · (1/2 - t · (1/3 - t · (1/4 - t/5)))).
    let inner = t.clone().mul_scalar(-0.2).add_scalar(0.25);
    let inner = t.clone().mul(inner).neg().add_scalar(1.0 / 3.0);
    let inner = t.clone().mul(inner).neg().add_scalar(0.5);
    let series = t.clone().mul(t.clone().mul(inner).neg().add_scalar(1.0));
    t.add_scalar(1.0).log().mask_where(small, series)
}

/// The activations other than `relu`, on `f32` (or an `f16`/`bf16` tensor's `f32` copy).
fn activate<const D: usize>(x: Tensor<D>, op: Activation) -> Tensor<D> {
    match op {
        Activation::Relu => relu(x),
        Activation::LeakyRelu { negative_slope } => {
            let negative = x.clone().lower_elem(0.0);
            x.clone().mask_where(negative, x.mul_scalar(negative_slope))
        }
        Activation::Gelu => {
            let cdf = x
                .clone()
                .mul_scalar(std::f64::consts::FRAC_1_SQRT_2)
                .erf()
                .add_scalar(1.0)
                .mul_scalar(0.5);
            x.mul(cdf)
        }
        Activation::GeluTanh => {
            // 0.5 · (1 + tanh(u)) = sigmoid(2u): no cancellation for negative x, no tanh of large
            // arguments.
            let cube = x.clone().mul(x.clone()).mul(x.clone());
            let u = x.clone().add(cube.mul_scalar(0.044_715));
            x.mul(sigmoid(u.mul_scalar(2.0 * SQRT_2_OVER_PI)))
        }
        Activation::Silu => x.clone().mul(sigmoid(x)),
        Activation::Sigmoid => sigmoid(x),
        Activation::Softplus { beta, threshold } => softplus(x, beta, threshold),
        Activation::Mish => x.clone().mul(tanh_nonnegative(softplus(x, 1.0, 20.0))),
    }
}

#[cfg(test)]
#[path = "activation_tests.rs"]
mod tests;
