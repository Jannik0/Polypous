//! Target checks of the losses and metrics (`docs/specs/training.md` §2.10 step 6, §3.3, §3.4).
//!
//! The trainer checks, when it is created and before anything runs, that each loss and metric's
//! prediction (dtype and shape, as binding gives them) fits its target field, and then every
//! target value of both sets. This module has the rules; [`crate::Loss`] and [`crate::Accuracy`]
//! apply them (`check_shapes`, `check_target_values`), and the trainer (#82) adds the config
//! pointer and the field name and maps a [`TargetError`] to `InvalidInput`.
//!
//! Targets come from data and are untrusted: the checks read [`TensorData`] after validating it,
//! allocate nothing per element and never panic. A value error names the sample (the index along
//! dim 0 of the field's data), the element inside the sample (flat, row-major) and the value.

use std::fmt;

use polypous_backend::{DType, TensorData, TensorError};
use thiserror::Error;

use ValueRule as Rule;

/// A target value that failed a check, as stored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TargetValue {
    /// An `i64` or `i32` class index.
    Int(i64),
    /// An `f32` value (may be NaN or ±inf).
    Float(f32),
}

impl fmt::Display for TargetValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "{v}"),
            Self::Float(v) => write!(f, "{v}"),
        }
    }
}

/// Why a prediction and its target don't fit a loss or metric, or why a target value is out of
/// range (training.md §3.3, §3.4). Never a panic.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TargetError {
    /// The prediction isn't `f32` (training is `f32` only, and every loss and metric reads an
    /// `f32` prediction).
    #[error("the prediction of `{what}` must be f32, got {got}")]
    PredictionDType {
        /// The loss or metric kind.
        what: &'static str,
        /// The prediction's dtype.
        got: DType,
    },
    /// The target's dtype doesn't fit the loss or metric.
    #[error("the target of `{what}` must be {expected}, got {got}")]
    TargetDType {
        /// The loss or metric kind.
        what: &'static str,
        /// The allowed dtypes, as text.
        expected: &'static str,
        /// The target's dtype.
        got: DType,
    },
    /// The shapes don't fit: not the same shape (no broadcasting), not `[B, C]` and `[B]` for
    /// class indices, or an empty tensor.
    #[error(
        "`{what}`: prediction shape {prediction:?} and target shape {target:?} don't fit: {expected}"
    )]
    Shape {
        /// The loss or metric kind.
        what: &'static str,
        /// The prediction's shape (with the batch dim).
        prediction: Vec<usize>,
        /// The target's shape (with the batch dim, or the field's data shape for value checks).
        target: Vec<usize>,
        /// The rule, as text.
        expected: &'static str,
    },
    /// The prediction and the target are on different devices.
    #[error("`{what}`: the prediction is on {prediction}, the target on {target}")]
    Device {
        /// The loss or metric kind.
        what: &'static str,
        /// The prediction's device.
        prediction: String,
        /// The target's device.
        target: String,
    },
    /// A target value outside the allowed range: a class index outside `[0, C)`, a
    /// `binary_cross_entropy` target outside `[0, 1]` or NaN, a binary-accuracy target that isn't
    /// exactly 0 or 1.
    #[error("`{what}`: sample {sample}, element {element}: target {value} is not {expected}")]
    Value {
        /// The loss or metric kind.
        what: &'static str,
        /// The sample: the index along dim 0 of the target data.
        sample: u64,
        /// The element inside the sample, flat in row-major order (`0` for a scalar sample).
        element: u64,
        /// The value as stored.
        value: TargetValue,
        /// The rule, as text (`"a class index in [0, 10)"`).
        expected: String,
    },
    /// The target data is malformed (byte length, shape).
    #[error("`{what}`: malformed target data: {source}")]
    Data {
        /// The loss or metric kind.
        what: &'static str,
        /// What [`TensorData::validate`] found.
        source: TensorError,
    },
}

/// How a prediction and its target relate (training.md §3.3, §3.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// The same shape `[B, …]`, `f32` target (`mse`, `binary_cross_entropy`, binary accuracy).
    SameShapeF32,
    /// Prediction `[B, C]` with `C ≥ 1`, target `[B]` of `i64` or `i32` (`cross_entropy`,
    /// multiclass accuracy).
    ClassIndices,
}

/// What a target value must be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ValueRule {
    /// Anything (`mse`).
    Any,
    /// A class index in `[0, C)`; `C` comes from the prediction's shape.
    ClassIndex,
    /// In `[0, 1]`, not NaN (`binary_cross_entropy`).
    UnitInterval,
    /// Exactly `0` or `1` (binary accuracy).
    ZeroOrOne,
}

/// Checks the dtypes and shapes of a prediction and its target against `layout`. Shapes include
/// the batch dim, as binding gives them (training.md §2.10 step 6). Every dim must be `≥ 1`.
pub(crate) fn check_layout(
    what: &'static str,
    layout: Layout,
    prediction: (DType, &[usize]),
    target: (DType, &[usize]),
) -> Result<(), TargetError> {
    let ((p_dtype, p_shape), (t_dtype, t_shape)) = (prediction, target);
    if p_dtype != DType::F32 {
        return Err(TargetError::PredictionDType { what, got: p_dtype });
    }
    let shape_error = |expected| TargetError::Shape {
        what,
        prediction: p_shape.to_vec(),
        target: t_shape.to_vec(),
        expected,
    };
    match layout {
        Layout::SameShapeF32 => {
            if t_dtype != DType::F32 {
                return Err(TargetError::TargetDType {
                    what,
                    expected: "f32",
                    got: t_dtype,
                });
            }
            if p_shape != t_shape {
                return Err(shape_error(
                    "the target must have the prediction's shape (no broadcasting)",
                ));
            }
            if p_shape.is_empty() || p_shape.contains(&0) {
                return Err(shape_error(
                    "the prediction must be [B, …] with every dim at least 1",
                ));
            }
        }
        Layout::ClassIndices => {
            if !matches!(t_dtype, DType::I64 | DType::I32) {
                return Err(TargetError::TargetDType {
                    what,
                    expected: "i64 or i32 (class indices)",
                    got: t_dtype,
                });
            }
            let fits = matches!(p_shape, [b, c] if *b >= 1 && *c >= 1)
                && t_shape.len() == 1
                && t_shape[0] == p_shape[0];
            if !fits {
                return Err(shape_error(
                    "the prediction must be [B, C] and the target [B], with B ≥ 1 and C ≥ 1",
                ));
            }
        }
    }
    Ok(())
}

/// Checks every value of a target field's data (`[N] + sample shape`, every sample of a set)
/// against `rule`. `classes` is `C` for [`ValueRule::ClassIndex`]. The dtype must be the one the
/// rule reads (`i64`/`i32` for class indices, `f32` otherwise); [`ValueRule::Any`] checks only
/// that the data is well formed.
pub(crate) fn check_values(
    what: &'static str,
    rule: ValueRule,
    classes: usize,
    target: &TensorData,
) -> Result<(), TargetError> {
    target
        .validate()
        .map_err(|source| TargetError::Data { what, source })?;
    if target.shape.is_empty() {
        return Err(TargetError::Shape {
            what,
            prediction: Vec::new(),
            target: Vec::new(),
            expected: "the target data must have a sample dim (dim 0)",
        });
    }
    // `validate` proved the product of the dims fits `usize`, so these products fit `u64`.
    let per_sample: u64 = target.shape[1..].iter().product::<u64>().max(1);
    let locate = |i: usize| {
        let i = i as u64;
        (i / per_sample, i % per_sample)
    };
    let value_error = |i: usize, value: TargetValue, expected: String| {
        let (sample, element) = locate(i);
        TargetError::Value {
            what,
            sample,
            element,
            value,
            expected,
        }
    };
    let dtype_error = |expected| TargetError::TargetDType {
        what,
        expected,
        got: target.dtype,
    };
    match rule {
        Rule::Any => Ok(()),
        Rule::ClassIndex => {
            let classes = i64::try_from(classes).unwrap_or(i64::MAX);
            let bad = |v: i64| !(0..classes).contains(&v);
            let expected = || format!("a class index in [0, {classes})");
            match target.dtype {
                DType::I64 => {
                    let (chunks, _) = target.bytes.as_chunks::<8>();
                    for (i, &b) in chunks.iter().enumerate() {
                        let v = i64::from_le_bytes(b);
                        if bad(v) {
                            return Err(value_error(i, TargetValue::Int(v), expected()));
                        }
                    }
                    Ok(())
                }
                DType::I32 => {
                    let (chunks, _) = target.bytes.as_chunks::<4>();
                    for (i, &b) in chunks.iter().enumerate() {
                        let v = i64::from(i32::from_le_bytes(b));
                        if bad(v) {
                            return Err(value_error(i, TargetValue::Int(v), expected()));
                        }
                    }
                    Ok(())
                }
                _ => Err(dtype_error("i64 or i32 (class indices)")),
            }
        }
        Rule::UnitInterval | Rule::ZeroOrOne => {
            if target.dtype != DType::F32 {
                return Err(dtype_error("f32"));
            }
            let (ok, expected): (fn(f32) -> bool, &str) = if rule == Rule::UnitInterval {
                (|v| (0.0..=1.0).contains(&v), "in [0, 1]")
            } else {
                (|v| v == 0.0 || v == 1.0, "exactly 0 or 1")
            };
            let (chunks, _) = target.bytes.as_chunks::<4>();
            for (i, &b) in chunks.iter().enumerate() {
                let v = f32::from_le_bytes(b);
                if !ok(v) {
                    return Err(value_error(i, TargetValue::Float(v), expected.to_owned()));
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "targets_tests.rs"]
mod tests;
