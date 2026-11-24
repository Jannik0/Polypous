//! The losses `mse`, `cross_entropy{label_smoothing}` and `binary_cross_entropy{with_logits}`
//! (`docs/specs/training.md` §3.3), written in public `DynTensor` ops (ADR-0014).
//!
//! [`Loss::compute`] returns a one-element `f32` tensor of shape `[1]`, the mean over the batch
//! (PyTorch's `reduction="mean"`), that is differentiable with respect to the prediction through
//! the autodiff API: back-propagating it gives PyTorch autograd's gradient of the reference call
//! (`F.mse_loss`, `F.cross_entropy`, `F.binary_cross_entropy(_with_logits)`, 2.15.0), within the
//! gradient tolerance of testing-strategy.md §2.
//!
//! # Formulations
//!
//! - **`mse`:** `d = x − y`, `mean(d · d)`.
//! - **`cross_entropy`:** `lp = log_softmax(x, 1)` (stable: the backend subtracts the row
//!   maximum), then `loss = −((1 − ε) · mean_n lp[n, y_n] + (ε / C) · mean_n Σ_c lp[n, c])`,
//!   which is §3.3's formula with `lse_n − x_{n,c} = −lp[n, c]`. The terms with a zero factor
//!   (`ε = 0`, `ε = 1`) are left out. `y` is gathered as it is, `i64` or `i32`: `core.gather`
//!   takes both, and the values are the same.
//! - **`binary_cross_entropy{with_logits}`:** `mean((1 − y) · x + softplus(−x))`, since
//!   `−log_sigmoid(x) = softplus(−x)`. The backend's `softplus` (β = 1, threshold 20, as
//!   `F.softplus`) returns `z` itself above the threshold and is stable at ±200. Its gradient,
//!   `(1 − y) − σ(−x) = σ(x) − y`, is PyTorch's.
//! - **`binary_cross_entropy`:** the value is §3.3's formula,
//!   `mean(−(y · max(ln x, −100) + (1 − y) · max(ln(1 − x), −100)))` with `clamp(min = −100)`.
//!   Back-propagated as it is, that composite gives NaN at `x = 0` and `x = 1` (the clamp passes
//!   0, and `log`'s backward multiplies it by `1 / 0`), and its gradient isn't PyTorch's where
//!   `x · (1 − x) < 1e-12` either. PyTorch's backward is its own formula,
//!   `(x − y) / max(x · (1 − x), 1e-12) / N`, not the derivative of the clamped loss. So the loss
//!   is assembled as `value + (s − detach(s))` with `s = Σ x · detach(G)` and `G` that formula
//!   computed on the detached prediction: the forward value is `value` exactly (`s − s = 0` for
//!   a finite `s`), and the gradient with respect to `x` is `G` exactly (the backward of `mul`
//!   and `sum`). An untracked prediction (a validation pass) skips `s` and gets the same value.
//!
//! # Checks
//!
//! [`Loss::compute`] checks the dtypes, shapes and devices (no broadcasting: a mismatch is an
//! error, never a panic). It doesn't read the target values back: they are checked when the
//! trainer is created ([`Loss::check_target_values`], §2.10 step 6). An out-of-range class index
//! is still caught by `core.gather`, which checks its indices before it runs.

use polypous_backend::{DType, DynTensor, Scalar, TensorData, TensorError};
use thiserror::Error;

use crate::config::{ConfigError, LossConfig};
use crate::optimizer::{f32_of, scalar};
use crate::targets::{Layout, TargetError, ValueRule, check_layout, check_values};

/// PyTorch's clamp of the log-probabilities in `binary_cross_entropy` (training.md §3.3).
const LOG_CLAMP: f64 = -100.0;
/// PyTorch's lower bound of `x · (1 − x)` in `binary_cross_entropy`'s backward (`EPSILON` in
/// ATen's `Loss.cpp`).
const BCE_GRAD_EPSILON: f64 = 1e-12;
/// `F.softplus`'s default threshold.
const SOFTPLUS_THRESHOLD: f64 = 20.0;

/// Why a loss couldn't be built or computed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LossError {
    /// The config has an out-of-range field.
    #[error("invalid config: {0}")]
    Config(#[from] ConfigError),
    /// The prediction and the target don't fit the loss (dtype, shape, device), or a target value
    /// is out of range.
    #[error(transparent)]
    Target(#[from] TargetError),
    /// A tensor op failed (including an out-of-range class index found by `core.gather`).
    #[error(transparent)]
    Backend(#[from] TensorError),
}

/// The selected loss with its options.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Mse,
    CrossEntropy { label_smoothing: f64 },
    BinaryCrossEntropy { with_logits: bool },
}

/// A loss of training.md §3.3, built from the config's `loss` object.
#[derive(Clone, Debug, PartialEq)]
pub struct Loss {
    kind: Kind,
}

impl Loss {
    /// Validates the config and builds the loss. The names (`prediction`, `target`) are the
    /// trainer's business; the loss only uses the kind and its options.
    ///
    /// # Errors
    /// [`LossError::Config`] for an out-of-range field.
    pub fn new(config: &LossConfig) -> Result<Self, LossError> {
        config.validate()?;
        let kind = match config {
            LossConfig::Mse(_) => Kind::Mse,
            LossConfig::CrossEntropy(c) => Kind::CrossEntropy {
                label_smoothing: c.label_smoothing,
            },
            LossConfig::BinaryCrossEntropy(c) => Kind::BinaryCrossEntropy {
                with_logits: c.with_logits,
            },
        };
        Ok(Self { kind })
    }

    /// The `kind` string of the config.
    pub fn kind(&self) -> &'static str {
        match self.kind {
            Kind::Mse => "mse",
            Kind::CrossEntropy { .. } => "cross_entropy",
            Kind::BinaryCrossEntropy { .. } => "binary_cross_entropy",
        }
    }

    fn layout(&self) -> Layout {
        match self.kind {
            Kind::CrossEntropy { .. } => Layout::ClassIndices,
            Kind::Mse | Kind::BinaryCrossEntropy { .. } => Layout::SameShapeF32,
        }
    }

    /// Checks a prediction's and a target's dtype and shape (with the batch dim, as binding
    /// gives them) against the loss (training.md §2.10 step 6, §3.3): an `f32` prediction; for
    /// `mse` and `binary_cross_entropy` an `f32` target of the same shape; for `cross_entropy`
    /// logits `[B, C]` and `i64` or `i32` class indices `[B]`. Every dim must be at least 1.
    ///
    /// # Errors
    /// The first rule broken, as a [`TargetError`].
    pub fn check_shapes(
        &self,
        prediction: (DType, &[usize]),
        target: (DType, &[usize]),
    ) -> Result<(), TargetError> {
        check_layout(self.kind(), self.layout(), prediction, target)
    }

    /// Checks every value of the target field's data (`[N] + sample shape`, every sample of a
    /// set) for this loss (training.md §2.10 step 6, §3.3): for `cross_entropy` a class index in
    /// `[0, C)` with `C = prediction_shape[1]`; for `binary_cross_entropy` (both forms) a value in
    /// `[0, 1]`, not NaN; `mse` takes any value. The dtype is checked as in
    /// [`Loss::check_shapes`]. `prediction_shape` is the prediction's shape as binding gives it.
    ///
    /// # Errors
    /// [`TargetError::Value`] naming the first bad sample, its element and the value; a wrong
    /// dtype, shape or malformed data.
    pub fn check_target_values(
        &self,
        prediction_shape: &[usize],
        target: &TensorData,
    ) -> Result<(), TargetError> {
        let what = self.kind();
        let (rule, classes) = match self.kind {
            Kind::Mse => (ValueRule::Any, 0),
            Kind::BinaryCrossEntropy { .. } => (ValueRule::UnitInterval, 0),
            Kind::CrossEntropy { .. } => match prediction_shape {
                [_, c] if *c >= 1 => (ValueRule::ClassIndex, *c),
                _ => {
                    return Err(TargetError::Shape {
                        what,
                        prediction: prediction_shape.to_vec(),
                        target: target.shape.iter().map(|&d| d as usize).collect(),
                        expected: "the prediction must be [B, C] with C ≥ 1",
                    });
                }
            },
        };
        if self.layout() == Layout::SameShapeF32 && target.dtype != DType::F32 {
            return Err(TargetError::TargetDType {
                what,
                expected: "f32",
                got: target.dtype,
            });
        }
        check_values(what, rule, classes, target)
    }

    /// The loss of a batch: a one-element `f32` tensor of shape `[1]`, the batch mean, on the
    /// prediction's device. It carries the prediction's autodiff tracking, so back-propagating it
    /// gives PyTorch's gradient with respect to the prediction (module documentation). The target
    /// is never differentiated.
    ///
    /// # Errors
    /// [`LossError::Target`] if the dtypes, shapes or devices don't fit ([`Loss::check_shapes`]);
    /// [`LossError::Backend`] if an op failed, including a class index outside `[0, C)`.
    pub fn compute(
        &self,
        prediction: &DynTensor,
        target: &DynTensor,
    ) -> Result<DynTensor, LossError> {
        self.check_shapes(
            (prediction.dtype(), &prediction.shape()),
            (target.dtype(), &target.shape()),
        )?;
        if prediction.device() != target.device() {
            return Err(TargetError::Device {
                what: self.kind(),
                prediction: prediction.device().to_string(),
                target: target.device().to_string(),
            }
            .into());
        }
        // The target is data: never differentiated, never part of the recorded graph.
        let (x, y) = (prediction.clone(), target.clone().detach());
        Ok(match self.kind {
            Kind::Mse => mse(x, y)?,
            Kind::CrossEntropy { label_smoothing } => cross_entropy(x, y, label_smoothing)?,
            Kind::BinaryCrossEntropy { with_logits: true } => bce_with_logits(x, y)?,
            Kind::BinaryCrossEntropy { with_logits: false } => bce(x, y)?,
        })
    }
}

/// `mean((x − y)²)` over every element.
fn mse(x: DynTensor, y: DynTensor) -> Result<DynTensor, TensorError> {
    let d = x.sub(y)?;
    d.clone().mul(d)?.mean(None, false)
}

/// `−((1 − ε) · mean_n lp[n, y_n] + (ε / C) · mean_n Σ_c lp[n, c])`, `lp = log_softmax(x, 1)`.
fn cross_entropy(x: DynTensor, y: DynTensor, eps: f64) -> Result<DynTensor, TensorError> {
    let shape = x.shape();
    let (b, c) = (shape[0], shape[1]);
    let lp = x.log_softmax(1)?;
    let mut total: Option<DynTensor> = None;
    if eps < 1.0 {
        // mean_n lp[n, y_n], shape [1]
        let picked = lp.clone().gather(1, y.reshape(&[b, 1])?)?;
        let mut nll = picked.mean(None, false)?;
        if eps != 0.0 {
            nll = nll.clone().mul(scalar(f32_of(1.0 - eps), &nll)?)?;
        }
        total = Some(nll);
    }
    if eps > 0.0 {
        // (ε / C) · mean_n Σ_c lp[n, c], shape [1]
        #[allow(
            clippy::cast_precision_loss,
            reason = "C is at most a tensor dim; the factor is rounded to f32 once anyway"
        )]
        let factor = eps / c as f64;
        let smooth = lp.sum(Some(&[1]), false)?.mean(None, false)?;
        let smooth = smooth.clone().mul(scalar(f32_of(factor), &smooth)?)?;
        total = Some(match total {
            Some(t) => t.add(smooth)?,
            None => smooth,
        });
    }
    // 0 ≤ ε ≤ 1 (validated), so at least one of the two branches ran.
    total
        .ok_or_else(|| TensorError::Backend {
            message: "cross_entropy: no term (label_smoothing out of range)".into(),
        })?
        .neg()
}

/// `mean((1 − y) · x + softplus(−x))`.
fn bce_with_logits(x: DynTensor, y: DynTensor) -> Result<DynTensor, TensorError> {
    let one_minus_y = scalar(1.0, &y)?.sub(y)?;
    let linear = one_minus_y.mul(x.clone())?;
    let soft = x.neg()?.softplus(1.0, SOFTPLUS_THRESHOLD)?;
    linear.add(soft)?.mean(None, false)
}

/// §3.3's clamped formula for the value; PyTorch's backward formula for the gradient (module
/// documentation).
fn bce(x: DynTensor, y: DynTensor) -> Result<DynTensor, TensorError> {
    let xd = x.clone().detach();
    let one = |like: &DynTensor| scalar(1.0, like);
    let clamp = Some(Scalar::Float(LOG_CLAMP));
    // −(y · max(ln x, −100) + (1 − y) · max(ln(1 − x), −100)), mean
    let log_x = xd.clone().log()?.clamp(clamp, None)?;
    let one_minus_x = one(&xd)?.sub(xd.clone())?;
    let log_1mx = one_minus_x.clone().log()?.clamp(clamp, None)?;
    let one_minus_y = one(&y)?.sub(y.clone())?;
    let value = y
        .clone()
        .mul(log_x)?
        .add(one_minus_y.mul(log_1mx)?)?
        .neg()?
        .mean(None, false)?;
    if !x.is_tracked() {
        return Ok(value);
    }
    // G = (x − y) / max(x · (1 − x), 1e-12) / N, on the detached prediction
    let n = x.shape().iter().product::<usize>();
    #[allow(
        clippy::cast_precision_loss,
        reason = "PyTorch divides by numel() as a float too"
    )]
    let n = n as f64;
    let denom = xd
        .clone()
        .mul(one_minus_x)?
        .maximum(scalar(f32_of(BCE_GRAD_EPSILON), &xd)?)?;
    let grad = xd
        .clone()
        .sub(y)?
        .div(denom)?
        .div(scalar(f32_of(n), &xd)?)?;
    // value + (s − detach(s)), s = Σ x · G: the value is `value`, the gradient is G.
    let s = x.mul(grad)?.sum(None, false)?;
    let zero = s.clone().sub(s.detach())?;
    value.add(zero)
}

#[cfg(test)]
#[path = "loss_tests.rs"]
mod tests;
