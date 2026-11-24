//! Gradient clipping by the L2 norm of all gradients (`docs/specs/training.md` §3.1.4,
//! `torch.nn.utils.clip_grad_norm_` with norm type 2 and `error_if_nonfinite=False`).

use polypous_backend::{DynTensor, Scalar};

use crate::config::GradientClipConfig;
use crate::optimizer::{OptimizerError, TensorMap, check_f32, f32_of, scalar};

/// Clips `grads` (the gradients of every trainable param that has one, after the backward pass
/// and before the optimizer) in place, and returns the total norm (`f32`, shape `[1]`, on the
/// gradients' device), or `None` if there are no gradients (then nothing changes).
///
/// ```text
/// n_i   = sqrt(Σ g_i²)                                    f32
/// total = sqrt(Σ n_i²)                                    f32
/// c     = min(f32(max_norm) / (total + f32(1e-6)), 1)     f32
/// g_i  ← g_i · c                                          every gradient, also when c = 1
/// ```
///
/// A non-finite total gives `c = 0` or NaN and the gradients follow, as in PyTorch; it isn't an
/// error. The summation order of the norms is the backend's (`core.sum`), and the total sums the
/// norms in name order. Either every gradient is clipped, or (on an error) none is.
///
/// # Errors
/// [`OptimizerError::Config`] if `max_norm` is out of range; [`OptimizerError::Mismatch`] if a
/// gradient isn't `f32` or is on another device than the first; [`OptimizerError::Backend`] if a
/// tensor op failed.
pub fn clip_gradients(
    config: &GradientClipConfig,
    grads: &mut TensorMap,
) -> Result<Option<DynTensor>, OptimizerError> {
    config.validate()?;
    let Some((first_name, first)) = grads.iter().next() else {
        return Ok(None);
    };
    let device = first.device();
    let mut norms = Vec::with_capacity(grads.len());
    for (name, g) in grads.iter() {
        check_f32(name, "gradient", g)?;
        if g.device() != device {
            return Err(OptimizerError::Mismatch {
                name: name.clone(),
                reason: format!(
                    "the gradient is on {}, `{first_name}` on {device}",
                    g.device()
                ),
            });
        }
        let g = g.clone().without_autodiff();
        // n_i = sqrt(Σ g_i²), shape [1]
        norms.push(g.clone().mul(g)?.sum(None, false)?.sqrt()?);
    }
    let norms = DynTensor::concat(&norms, 0)?;
    // total = sqrt(Σ n_i²), shape [1]
    let total = norms.clone().mul(norms)?.sum(None, false)?.sqrt()?;
    // c = min(f32(max_norm) / (total + f32(1e-6)), 1); clamp propagates NaN, as torch.clamp.
    let c = scalar(f32_of(config.max_norm), &total)?
        .div(total.clone().add(scalar(f32_of(1e-6), &total)?)?)?
        .clamp(None, Some(Scalar::Float(1.0)))?;
    let mut clipped = Vec::with_capacity(grads.len());
    for (name, g) in grads.iter() {
        let c = c.clone().reshape(&vec![1; g.rank()])?;
        clipped.push((name.clone(), g.clone().without_autodiff().mul(c)?));
    }
    grads.extend(clipped);
    Ok(Some(total))
}
