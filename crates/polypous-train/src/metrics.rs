//! The metrics `loss` and `accuracy` (`docs/specs/training.md` §3.4), as epoch accumulators.
//!
//! The trainer (#82) keeps one accumulator per reported metric and per set (`train.*`,
//! `val.*`), feeds it every batch of the epoch, reads the value at the end of the epoch and
//! resets it. Both are exact and deterministic: [`LossMetric`] sums in `f64` in batch order, and
//! [`Accuracy`] counts integers and divides once.
//!
//! Metrics run outside the backward pass: [`Accuracy::update`] reads the prediction back to the
//! host (which records nothing, even for a tracked tensor) and counts in Rust, so the tie and NaN
//! rules of `torch.argmax` hold by construction on every device. Nothing here changes a tensor.
//!
//! **An empty epoch** (no batch fed, or only batches of zero samples) has no value: `value()`
//! returns `None` rather than the NaN of `0 / 0`. So the trainer can't mistake it for a NaN loss,
//! and must decide explicitly. It can't happen in a run that the trainer accepted: every epoch has
//! at least one step, and a validation set has at least one sample.

use polypous_backend::{DType, DynTensor, TensorData, TensorError};
use thiserror::Error;

use crate::config::{AccuracyConfig, ConfigError};
use crate::optimizer::f32_of;
use crate::targets::{Layout, TargetError, ValueRule, check_layout, check_values};

/// The `threshold` of binary accuracy when the config leaves it null (training.md §2.7).
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// Why a metric couldn't be built or updated.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum MetricError {
    /// The config has an out-of-range field.
    #[error("invalid config: {0}")]
    Config(#[from] ConfigError),
    /// `threshold` is given, but the target has class indices (the multiclass form, training.md
    /// §2.7). The trainer reports it at the metric's `/threshold`.
    #[error("`threshold` is for binary accuracy only, but the target has class indices ({target})")]
    ThresholdWithMulticlass {
        /// The target's dtype.
        target: DType,
    },
    /// The prediction and the target don't fit the metric, or a target value is out of range.
    #[error(transparent)]
    Target(#[from] TargetError),
    /// Reading a tensor back failed.
    #[error(transparent)]
    Backend(#[from] TensorError),
}

/// The `loss` metric of one epoch and set (training.md §3.4):
/// `Σ_b loss_b · B_b / Σ_b B_b`, summed in `f64` in batch order from the `f32` batch losses.
///
/// `f64(loss_b) · B_b` is exact (a 24-bit significand times a count below `2^29`), so the only
/// roundings are the sum's, in batch order, and the one division.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LossMetric {
    weighted_sum: f64,
    samples: u64,
}

impl LossMetric {
    /// An empty epoch.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds batch `b`'s loss (the `f32` value of [`crate::Loss::compute`]) and its sample count
    /// `B_b`. A batch of zero samples adds nothing (also if its loss is NaN).
    pub fn add(&mut self, loss: f32, batch_size: usize) {
        if batch_size == 0 {
            return;
        }
        let b = batch_size as u64;
        #[allow(
            clippy::cast_precision_loss,
            reason = "a batch has at most 65,536 samples (training.md §2.2), exact in f64"
        )]
        let weight = b as f64;
        self.weighted_sum += f64::from(loss) * weight;
        self.samples = self.samples.saturating_add(b);
    }

    /// An epoch restored from a checkpoint's `epoch_sums` (training.md §5.2): `Σ loss_b · B_b`
    /// and `Σ B_b`, exactly as [`LossMetric::weighted_sum`] and [`LossMetric::samples`] gave them.
    pub fn from_sums(weighted_sum: f64, samples: u64) -> Self {
        Self {
            weighted_sum,
            samples,
        }
    }

    /// `Σ loss_b · B_b` so far, in `f64` (`loss_sum` of a checkpoint's `epoch_sums`).
    pub fn weighted_sum(&self) -> f64 {
        self.weighted_sum
    }

    /// The samples added so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// The weighted mean loss, or `None` for an empty epoch (module documentation). NaN or ±inf
    /// if a batch loss was.
    pub fn value(&self) -> Option<f64> {
        #[allow(
            clippy::cast_precision_loss,
            reason = "sample counts stay far below 2^53, exact in f64"
        )]
        (self.samples > 0).then(|| self.weighted_sum / self.samples as f64)
    }

    /// Starts a new epoch.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// The `accuracy` metric of one epoch and set (training.md §3.4): exact counts, divided once in
/// `f64`. The form follows the target's dtype:
///
/// - **multiclass** (target `i64`/`i32` `[B]`, prediction `f32` `[B, C]`): sample `n` is correct
///   if `argmax_c x[n, c] = y_n`, with ties to the lowest index and the first NaN winning
///   (`torch.argmax`, 2.15.0);
/// - **binary** (target `f32` of the prediction's shape, every value 0 or 1): each element counts,
///   and is correct if `(x > threshold) = (y = 1)`. The comparison is in `f32`, with the threshold
///   rounded once to `f32`, as PyTorch's `x > threshold` on an `f32` tensor. A NaN is never above
///   the threshold.
#[derive(Clone, Debug, PartialEq)]
pub struct Accuracy {
    threshold: Option<f64>,
    correct: u64,
    total: u64,
}

impl Accuracy {
    /// Validates the config and builds an empty accumulator.
    ///
    /// # Errors
    /// [`MetricError::Config`] for an out-of-range field.
    pub fn new(config: &AccuracyConfig) -> Result<Self, MetricError> {
        config.validate()?;
        Ok(Self {
            threshold: config.threshold,
            correct: 0,
            total: 0,
        })
    }

    fn layout(target: DType) -> Layout {
        if target.is_int() {
            Layout::ClassIndices
        } else {
            Layout::SameShapeF32
        }
    }

    /// Checks a prediction's and a target's dtype and shape (with the batch dim, as binding gives
    /// them; training.md §2.10 step 6, §3.4) and that `threshold` is null for the multiclass form.
    /// An integer target (`i64`, `i32`) selects the multiclass form, any other dtype the binary
    /// form (which needs `f32`).
    ///
    /// # Errors
    /// [`MetricError::Target`] for the first rule broken; [`MetricError::ThresholdWithMulticlass`].
    pub fn check_shapes(
        &self,
        prediction: (DType, &[usize]),
        target: (DType, &[usize]),
    ) -> Result<(), MetricError> {
        let layout = Self::layout(target.0);
        check_layout("accuracy", layout, prediction, target)?;
        if layout == Layout::ClassIndices && self.threshold.is_some() {
            return Err(MetricError::ThresholdWithMulticlass { target: target.0 });
        }
        Ok(())
    }

    /// Checks every value of the target field's data (`[N] + sample shape`): class indices in
    /// `[0, C)` with `C = prediction_shape[1]` (multiclass), or every value exactly 0 or 1
    /// (binary). `prediction_shape` is the prediction's shape as binding gives it.
    ///
    /// # Errors
    /// [`TargetError::Value`] naming the first bad sample, its element and the value; a wrong
    /// dtype, shape or malformed data.
    pub fn check_target_values(
        &self,
        prediction_shape: &[usize],
        target: &TensorData,
    ) -> Result<(), TargetError> {
        if Self::layout(target.dtype) == Layout::ClassIndices {
            let Some(&c) = prediction_shape
                .get(1)
                .filter(|_| prediction_shape.len() == 2)
            else {
                return Err(TargetError::Shape {
                    what: "accuracy",
                    prediction: prediction_shape.to_vec(),
                    target: target.shape.iter().map(|&d| d as usize).collect(),
                    expected: "the prediction must be [B, C] for class-index targets",
                });
            };
            check_values("accuracy", ValueRule::ClassIndex, c, target)
        } else {
            check_values("accuracy", ValueRule::ZeroOrOne, 0, target)
        }
    }

    /// Counts one batch: reads `prediction` back to the host and calls
    /// [`Accuracy::update_host`]. The prediction may be tracked; nothing is recorded.
    ///
    /// # Errors
    /// As [`Accuracy::update_host`]; [`MetricError::Backend`] if the read-back failed.
    pub fn update(
        &mut self,
        prediction: &DynTensor,
        target: &TensorData,
    ) -> Result<(), MetricError> {
        self.update_host(&prediction.to_data()?, target)
    }

    /// Counts one batch from host data: the prediction `[B, C]` or `[B, …]`, `f32`, and the
    /// batch's target. The batch is checked first ([`Accuracy::check_shapes`] and the target
    /// values), so on an error nothing is counted.
    ///
    /// # Errors
    /// [`MetricError::Target`], [`MetricError::ThresholdWithMulticlass`].
    pub fn update_host(
        &mut self,
        prediction: &TensorData,
        target: &TensorData,
    ) -> Result<(), MetricError> {
        let data_error = |source| TargetError::Data {
            what: "accuracy",
            source,
        };
        prediction.validate().map_err(data_error)?;
        target.validate().map_err(data_error)?;
        // `validate` proved every dim's product fits `usize`, so each dim does.
        let p_shape: Vec<usize> = prediction.shape.iter().map(|&d| d as usize).collect();
        let t_shape: Vec<usize> = target.shape.iter().map(|&d| d as usize).collect();
        self.check_shapes((prediction.dtype, &p_shape), (target.dtype, &t_shape))?;
        self.check_target_values(&p_shape, target)?;
        let (x, _) = prediction.bytes.as_chunks::<4>();
        let x = x.iter().map(|&b| f32::from_le_bytes(b));
        let (correct, total) = if target.dtype.is_int() {
            let classes = p_shape[1];
            let rows: Vec<f32> = x.collect();
            let mut correct = 0u64;
            for (n, row) in rows.chunks_exact(classes).enumerate() {
                let y = class_index(target, n);
                if y == Some(argmax(row)) {
                    correct += 1;
                }
            }
            (correct, t_shape[0] as u64)
        } else {
            let threshold = f32_of(self.threshold.unwrap_or(DEFAULT_THRESHOLD));
            let (y, _) = target.bytes.as_chunks::<4>();
            let mut correct = 0u64;
            let mut total = 0u64;
            for (x, &y) in x.zip(y) {
                // NaN > threshold is false: a NaN is never predicted 1.
                if (x > threshold) == (f32::from_le_bytes(y) == 1.0) {
                    correct += 1;
                }
                total += 1;
            }
            (correct, total)
        };
        self.correct += correct;
        self.total += total;
        Ok(())
    }

    /// The correct predictions counted so far.
    pub fn correct(&self) -> u64 {
        self.correct
    }

    /// The predictions counted so far (samples for multiclass, elements for binary).
    pub fn total(&self) -> u64 {
        self.total
    }

    /// `correct / total` in `f64`, or `None` for an empty epoch (module documentation).
    pub fn value(&self) -> Option<f64> {
        #[allow(
            clippy::cast_precision_loss,
            reason = "counts stay far below 2^53, exact in f64"
        )]
        (self.total > 0).then(|| self.correct as f64 / self.total as f64)
    }

    /// Restores the counts of a checkpoint's `epoch_sums` (training.md §5.2). Returns `false`,
    /// and changes nothing, if `correct > total`.
    pub fn restore(&mut self, correct: u64, total: u64) -> bool {
        if correct > total {
            return false;
        }
        self.correct = correct;
        self.total = total;
        true
    }

    /// Starts a new epoch (the threshold stays).
    pub fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }
}

/// `torch.argmax` of one row (2.15.0): the first maximum, and the first NaN if there is one.
/// `row` is not empty (`C ≥ 1`, checked).
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (c, &v) in row.iter().enumerate().skip(1) {
        if row[best].is_nan() {
            break;
        }
        if v.is_nan() || v > row[best] {
            best = c;
        }
    }
    best
}

/// The class index of sample `n` of a checked `i64`/`i32` target, as `usize`.
fn class_index(target: &TensorData, n: usize) -> Option<usize> {
    let v = match target.dtype {
        DType::I64 => {
            let b = target.bytes.get(n * 8..n * 8 + 8)?;
            i64::from_le_bytes(b.try_into().ok()?)
        }
        DType::I32 => {
            let b = target.bytes.get(n * 4..n * 4 + 4)?;
            i64::from(i32::from_le_bytes(b.try_into().ok()?))
        }
        _ => return None,
    };
    usize::try_from(v).ok()
}

#[cfg(test)]
#[path = "metrics_tests.rs"]
mod tests;
