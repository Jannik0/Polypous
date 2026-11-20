//! Gradient overrides: ops whose Burn backward differs from graph-ir.md §3.1.11 set their
//! gradient explicitly on their tracked path. See the module documentation of [`crate::ops`],
//! "Gradients", for which ops use this and why.
//!
//! [`with_gradient`] returns a tensor whose **forward value is `out` bit for bit** and whose
//! gradient with respect to each tracked input `x` is `g · d`, where `g` is the upstream gradient
//! and `d` a derivative tensor the op computes from the spec's formula (any value, including
//! `±inf` and NaN). It is built from Burn ops only (no custom backward: Burn 0.22.1's dispatch
//! backend has no way to register one without Burn's backend types), as
//!
//! ```text
//! constant(out) + T(x₁, d₁) + T(x₂, d₂) + …     with every T exactly -0 in the forward pass
//! ```
//!
//! and `x + (-0) = x` for every `x` (also `-0`, `±inf` and NaN), so the forward value is unchanged.
//! Each term is `T = ((s · c) · k) · k`:
//!
//! - `s` is `+0` with derivative `±1`: `xs - constant(xs)` (`+1`) where `d < 0`, else
//!   `constant(xs) - xs` (`-1`), selected with `mask_where` (a selection, so the forward value is one
//!   of the two `+0`s). `xs` is `x` with its non-finite elements replaced by a constant `0`
//!   (`inf - inf` would be NaN); their gradient is unspecified (§3.1.11) and comes out `0`.
//! - `c ≤ 0` (or `-0`) carries the magnitude, so `s · c = -0`: `c = -|d|` for finite `d`, `-1` for
//!   `d = ±inf`, `-0` for NaN. With the sign of `s`, the derivative of `s · c` is `d` for finite
//!   `d`.
//! - `k` is `1` for finite `d` and the dtype's largest finite value otherwise: `0 · k = 0` in the
//!   forward pass, while the backward pass multiplies `g · k · k`, which overflows to `±inf` (for
//!   `|g| > 1 / k`), and then by `c`: `±inf` for `d = ±inf`, and `inf · -0 = NaN` for `d = NaN`.
//!
//! For finite `d` the gradient is `g · d`, rounded once, as PyTorch computes most of these
//! derivatives (`grad * (…)`). For `d = ±inf` it is `±inf` (signed by `g`) unless `|g| ≤ 1 / k`
//! (for example `g = 0`, where PyTorch gives NaN); such `g` together with an infinite derivative
//! don't occur in the fixtures.
//!
//! This costs about a dozen elementwise ops per tracked input, only on the tracked path:
//! untracked calls never reach this module.

use burn::tensor::{DType as BurnDType, Tensor};

/// `t` as an untracked constant (no graph, no gradient). Burn's `detach` alone is not enough: on a
/// leaf it returns a new leaf that still requires a gradient (`float_detach` keeps
/// `require_grad`), so ops on it would still be recorded.
pub(crate) fn constant<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.detach().set_require_grad(false)
}

/// Whether any of the tensors takes part in a recorded graph.
pub(crate) fn any_tracked<const D: usize>(tensors: &[&Tensor<D>]) -> bool {
    tensors.iter().any(|t| t.is_tracked())
}

/// `out` in the forward pass; gradient `g · d` to each tracked `x` of `terms` (see the module
/// documentation). `out`, every `x` and every `d` have the same shape and dtype. Untracked `x`
/// are skipped. `out`'s own graph (if any) is cut: the result's gradient comes from `terms` only.
pub(crate) fn with_gradient<const D: usize, const N: usize>(
    out: Tensor<D>,
    terms: [(Tensor<D>, Tensor<D>); N],
) -> Tensor<D> {
    terms
        .into_iter()
        .filter(|(x, _)| x.is_tracked())
        .fold(constant(out), |acc, (x, d)| acc + zero_with_gradient(x, d))
}

/// `-0` in the forward pass, gradient `g · d` to `x` (see the module documentation).
fn zero_with_gradient<const D: usize>(x: Tensor<D>, d: Tensor<D>) -> Tensor<D> {
    let d = constant(d);
    let xs = x.clone().mask_fill(x.is_finite().bool_not(), 0.0);
    let xd = constant(xs.clone());
    // Both +0; derivative +1 and -1.
    let up = xs.clone() - xd.clone();
    let down = xd - xs;
    let s = down.mask_where(d.clone().lower_elem(0.0), up);
    let c = d
        .clone()
        .abs()
        .neg()
        .mask_fill(d.clone().is_inf(), -1.0)
        .mask_fill(d.clone().is_nan(), -0.0);
    let k = d
        .ones_like()
        .mask_fill(d.is_finite().bool_not(), largest_finite(c.dtype()));
    ((s * c) * k.clone()) * k
}

/// The largest finite value of a float dtype.
fn largest_finite(dtype: BurnDType) -> f64 {
    match dtype {
        BurnDType::F16 => 65504.0,
        // bf16 has f32's exponent range; its largest finite value is 0x7F7F.
        BurnDType::BF16 => 3.389_531_389_251_535_5e38,
        BurnDType::F64 => f64::MAX,
        _ => f64::from(f32::MAX),
    }
}

#[cfg(test)]
#[path = "gradient_tests.rs"]
mod tests;
