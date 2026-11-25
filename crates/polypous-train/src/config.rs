//! The `loss`, `metrics`, `optimizer`, `schedule` and `gradient_clip` objects of a training
//! config (`docs/specs/training.md` §2.1, §2.2, §2.6–§2.8), as Rust types.
//!
//! The types deserialize directly from the JSON of §2.8 (`serde`, internally tagged by `kind`,
//! unknown fields rejected, missing optional fields filled with their defaults) and serialize every
//! field in the order of the spec's tables, so the config parser (#82) can embed them. Serde
//! checks the JSON types (an integer field rejects `5.0`, a negative or too-large integer doesn't
//! fit its Rust type); the ranges are checked by `validate`, which returns a [`ConfigError`] whose
//! pointer is relative to the object (`/betas/1`). The parser prefixes the object's own pointer
//! (`/optimizer`), as §2.1 requires.
//!
//! [`crate::Optimizer::new`], [`crate::Schedule::new`], [`crate::clip_gradients`],
//! [`crate::Loss::new`] and [`crate::Accuracy::new`] validate the config they are given, so an
//! invalid config never reaches the arithmetic.
//!
//! The checks against the graph and the data (training.md §2.10) are not made here: that
//! `prediction` is a graph output and `target` a dataset field, their dtypes and shapes
//! ([`crate::Loss::check_shapes`], [`crate::Accuracy::check_shapes`]) and the target values
//! ([`crate::targets`]).

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// An out-of-range field of a config object (training.md §2.1, "Errors").
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[error("{pointer}: {message}")]
pub struct ConfigError {
    /// The JSON pointer (RFC 6901) of the offending value, relative to the object that was
    /// validated: `/lr`, `/betas/1`, `/min_lr`. The config parser prefixes the object's pointer.
    pub pointer: String,
    /// What is wrong: the field and the allowed values.
    pub message: String,
}

impl ConfigError {
    fn new(pointer: &str, message: impl Into<String>) -> Self {
        Self {
            pointer: pointer.to_owned(),
            message: message.into(),
        }
    }
}

/// The largest `every` and `warmup_steps` (training.md §2.8: `2^32`).
pub const MAX_SCHEDULE_STEPS: u64 = 1 << 32;

/// `optimizer` (training.md §2.8, §3.1). Its JSON form is internally tagged:
/// `{"kind": "adamw", "lr": 1e-3, ...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum OptimizerConfig {
    /// `sgd` (§3.1.1).
    Sgd(SgdConfig),
    /// `adam` (§3.1.2).
    Adam(AdamConfig),
    /// `adamw` (§3.1.3).
    Adamw(AdamWConfig),
}

/// The fields of `sgd` (training.md §2.8). Defaults are `torch.optim.SGD`'s.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct SgdConfig {
    /// Learning rate `lr₀`, `≥ 0`. Default `1e-3`.
    pub lr: f64,
    /// `0 ≤ momentum ≤ 1`. Default `0`.
    pub momentum: f64,
    /// `0 ≤ dampening ≤ 1`. Default `0`.
    pub dampening: f64,
    /// Nesterov momentum; needs `momentum > 0` and `dampening = 0`. Default `false`.
    pub nesterov: bool,
    /// L2 penalty added to the gradient, `≥ 0`. Default `0`.
    pub weight_decay: f64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        }
    }
}

/// The fields of `adam` (training.md §2.8). Defaults are `torch.optim.Adam`'s.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AdamConfig {
    /// Learning rate `lr₀`, `≥ 0`. Default `1e-3`.
    pub lr: f64,
    /// `[β1, β2]`, each `0 ≤ β < 1`. Default `[0.9, 0.999]`.
    pub betas: [f64; 2],
    /// `≥ 0`. Default `1e-8`.
    pub eps: f64,
    /// L2 penalty added to the gradient, `≥ 0`. Default `0`.
    pub weight_decay: f64,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// The fields of `adamw` (training.md §2.8): as [`AdamConfig`], but the weight decay is decoupled
/// (§3.1.3) and defaults to `0.01` (`torch.optim.AdamW`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct AdamWConfig {
    /// Learning rate `lr₀`, `≥ 0`. Default `1e-3`.
    pub lr: f64,
    /// `[β1, β2]`, each `0 ≤ β < 1`. Default `[0.9, 0.999]`.
    pub betas: [f64; 2],
    /// `≥ 0`. Default `1e-8`.
    pub eps: f64,
    /// Decoupled weight decay, `≥ 0`. Default `0.01`.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// `x ≥ 0` and finite in `f32` (training.md §2.1, "Numbers": a float used in `f32` arithmetic
/// must round to a finite `f32`). NaN fails.
fn check_non_negative(pointer: &str, field: &str, x: f64) -> Result<(), ConfigError> {
    if x.is_nan() || x < 0.0 {
        return Err(ConfigError::new(
            pointer,
            format!("`{field}` must be ≥ 0, got {x}"),
        ));
    }
    check_f32_finite(pointer, field, x)
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the check is whether the f32 rounding is finite"
)]
fn check_f32_finite(pointer: &str, field: &str, x: f64) -> Result<(), ConfigError> {
    if (x as f32).is_finite() {
        Ok(())
    } else {
        Err(ConfigError::new(
            pointer,
            format!("`{field}` must round to a finite f32, got {x}"),
        ))
    }
}

/// `lo ≤ x ≤ hi` (NaN fails).
fn check_closed(pointer: &str, field: &str, x: f64, lo: f64, hi: f64) -> Result<(), ConfigError> {
    if x >= lo && x <= hi {
        Ok(())
    } else {
        Err(ConfigError::new(
            pointer,
            format!("`{field}` must be in [{lo}, {hi}], got {x}"),
        ))
    }
}

fn check_betas(betas: [f64; 2]) -> Result<(), ConfigError> {
    for (i, (beta, pointer)) in betas.into_iter().zip(["/betas/0", "/betas/1"]).enumerate() {
        if !(0.0..1.0).contains(&beta) {
            return Err(ConfigError::new(
                pointer,
                format!("`betas[{i}]` must be in [0, 1), got {beta}"),
            ));
        }
    }
    Ok(())
}

impl SgdConfig {
    /// Checks the ranges of training.md §2.8.
    ///
    /// # Errors
    /// The first out-of-range field, in table order, then the `nesterov` rule.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        check_non_negative("/lr", "lr", self.lr)?;
        check_closed("/momentum", "momentum", self.momentum, 0.0, 1.0)?;
        check_closed("/dampening", "dampening", self.dampening, 0.0, 1.0)?;
        if self.nesterov && !(self.momentum > 0.0 && self.dampening == 0.0) {
            return Err(ConfigError::new(
                "/nesterov",
                format!(
                    "`nesterov` needs `momentum` > 0 and `dampening` = 0, got momentum {} and \
                     dampening {}",
                    self.momentum, self.dampening
                ),
            ));
        }
        check_non_negative("/weight_decay", "weight_decay", self.weight_decay)
    }
}

impl AdamConfig {
    /// Checks the ranges of training.md §2.8.
    ///
    /// # Errors
    /// The first out-of-range field, in table order.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        check_non_negative("/lr", "lr", self.lr)?;
        check_betas(self.betas)?;
        check_non_negative("/eps", "eps", self.eps)?;
        check_non_negative("/weight_decay", "weight_decay", self.weight_decay)
    }
}

impl AdamWConfig {
    /// Checks the ranges of training.md §2.8 (those of `adam`).
    ///
    /// # Errors
    /// The first out-of-range field, in table order.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        self.as_adam().validate()
    }

    /// The same numbers as an [`AdamConfig`] (the two differ only in how the decay is applied).
    fn as_adam(&self) -> AdamConfig {
        AdamConfig {
            lr: self.lr,
            betas: self.betas,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

impl OptimizerConfig {
    /// Checks the ranges of training.md §2.8 for the selected kind.
    ///
    /// # Errors
    /// The first out-of-range field.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Sgd(c) => c.validate(),
            Self::Adam(c) => c.validate(),
            Self::Adamw(c) => c.validate(),
        }
    }

    /// The learning rate `lr₀` of the config (the schedules' base, training.md §3.2).
    pub fn lr(&self) -> f64 {
        match self {
            Self::Sgd(c) => c.lr,
            Self::Adam(c) => c.lr,
            Self::Adamw(c) => c.lr,
        }
    }

    /// The `kind` string of the JSON form.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Sgd(_) => "sgd",
            Self::Adam(_) => "adam",
            Self::Adamw(_) => "adamw",
        }
    }
}

/// `schedule` (training.md §2.8, §3.2). Its JSON form is internally tagged:
/// `{"kind": "cosine", "warmup_steps": 100, "min_lr": 1e-5}`. The default is `constant`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
#[non_exhaustive]
pub enum ScheduleConfig {
    /// `constant`: `lr₀` at every step. A struct variant with no fields, so that serde rejects
    /// unknown fields on it (it ignores them on a unit variant of an internally tagged enum).
    Constant {},
    /// `step{every, gamma}`: `lr₀` multiplied by `gamma` at every multiple of `every`.
    Step {
        /// `1 ≤ every ≤ 2^32`, required.
        every: u64,
        /// `0 ≤ gamma ≤ 1`. Default `0.1`.
        #[serde(default = "default_gamma")]
        gamma: f64,
    },
    /// `cosine{warmup_steps, min_lr}`: a linear warmup, then a cosine from `lr₀` to `min_lr`.
    Cosine {
        /// `0 ≤ warmup_steps ≤ 2^32`. Default `0`.
        #[serde(default)]
        warmup_steps: u64,
        /// `0 ≤ min_lr ≤ lr₀`. Default `0`.
        #[serde(default)]
        min_lr: f64,
    },
    /// `linear_warmup{warmup_steps}`: from `lr₀ / W` up to `lr₀` over `W` steps, then `lr₀`.
    LinearWarmup {
        /// `1 ≤ warmup_steps ≤ 2^32`, required.
        warmup_steps: u64,
    },
}

impl Default for ScheduleConfig {
    /// `constant` (training.md §2.2: `schedule` defaults to `{"kind": "constant"}`).
    fn default() -> Self {
        Self::Constant {}
    }
}

fn default_gamma() -> f64 {
    0.1
}

impl ScheduleConfig {
    /// Checks the ranges of training.md §2.8 for the selected kind. `lr` is the optimizer's
    /// `lr₀`, which bounds `min_lr`.
    ///
    /// # Errors
    /// The first out-of-range field; `min_lr > lr` at `/min_lr`.
    pub(crate) fn validate(&self, lr: f64) -> Result<(), ConfigError> {
        let check_steps = |pointer: &str, field: &str, v: u64, min: u64| {
            if (min..=MAX_SCHEDULE_STEPS).contains(&v) {
                Ok(())
            } else {
                Err(ConfigError::new(
                    pointer,
                    format!("`{field}` must be in [{min}, 2^32], got {v}"),
                ))
            }
        };
        match *self {
            Self::Constant {} => Ok(()),
            Self::Step { every, gamma } => {
                check_steps("/every", "every", every, 1)?;
                check_closed("/gamma", "gamma", gamma, 0.0, 1.0)
            }
            Self::Cosine {
                warmup_steps,
                min_lr,
            } => {
                check_steps("/warmup_steps", "warmup_steps", warmup_steps, 0)?;
                if min_lr.is_nan() || min_lr < 0.0 {
                    return Err(ConfigError::new(
                        "/min_lr",
                        format!("`min_lr` must be ≥ 0, got {min_lr}"),
                    ));
                }
                if min_lr > lr {
                    return Err(ConfigError::new(
                        "/min_lr",
                        format!(
                            "`min_lr` ({min_lr}) must not exceed the optimizer's `lr` ({lr}); \
                             the cosine would rise"
                        ),
                    ));
                }
                Ok(())
            }
            Self::LinearWarmup { warmup_steps } => {
                check_steps("/warmup_steps", "warmup_steps", warmup_steps, 1)
            }
        }
    }

    /// The `kind` string of the JSON form.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Constant {} => "constant",
            Self::Step { .. } => "step",
            Self::Cosine { .. } => "cosine",
            Self::LinearWarmup { .. } => "linear_warmup",
        }
    }
}

/// `gradient_clip` (training.md §2.2, §3.1.4): `{"max_norm": float}`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradientClipConfig {
    /// The largest L2 norm of all gradients together: finite and `> 0`, required.
    pub max_norm: f64,
}

impl GradientClipConfig {
    /// Checks the range of training.md §2.2.
    ///
    /// # Errors
    /// `max_norm` not finite (in `f32`) or not `> 0`, at `/max_norm`.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        if self.max_norm.is_nan() || self.max_norm <= 0.0 {
            return Err(ConfigError::new(
                "/max_norm",
                format!("`max_norm` must be > 0, got {}", self.max_norm),
            ));
        }
        check_f32_finite("/max_norm", "max_norm", self.max_norm)
    }
}

/// The longest name of a graph output or dataset field (training.md §2.1, "Strings": 1–256 bytes).
pub const MAX_NAME_BYTES: usize = 256;
/// The longest metric name (training.md §2.7: 1–64 bytes).
pub const MAX_METRIC_NAME_BYTES: usize = 64;
/// The most entries of `metrics` (training.md §2.1, "Limits").
pub const MAX_METRICS: usize = 16;

/// A `prediction` or `target` name: 1–256 bytes (training.md §2.1).
fn check_name(pointer: &str, field: &str, name: &str) -> Result<(), ConfigError> {
    if (1..=MAX_NAME_BYTES).contains(&name.len()) {
        Ok(())
    } else {
        Err(ConfigError::new(
            pointer,
            format!(
                "`{field}` must be 1–{MAX_NAME_BYTES} bytes, got {} bytes",
                name.len()
            ),
        ))
    }
}

/// `loss` (training.md §2.6, §3.3). Its JSON form is internally tagged:
/// `{"kind": "cross_entropy", "prediction": "logits", "target": "label", "label_smoothing": 0.1}`.
/// A field of another kind is an unknown field (`label_smoothing` on `mse`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum LossConfig {
    /// `mse`: `F.mse_loss`.
    Mse(MseConfig),
    /// `cross_entropy{label_smoothing}`: `F.cross_entropy` with class-index targets.
    CrossEntropy(CrossEntropyConfig),
    /// `binary_cross_entropy{with_logits}`: `F.binary_cross_entropy(_with_logits)`.
    BinaryCrossEntropy(BinaryCrossEntropyConfig),
}

/// The fields of `mse` (training.md §2.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MseConfig {
    /// A graph output, required.
    pub prediction: String,
    /// A dataset field, required.
    pub target: String,
}

/// The fields of `cross_entropy` (training.md §2.6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrossEntropyConfig {
    /// A graph output (the logits), required.
    pub prediction: String,
    /// A dataset field (the class indices), required.
    pub target: String,
    /// `0 ≤ label_smoothing ≤ 1`. Default `0`.
    #[serde(default)]
    pub label_smoothing: f64,
}

/// The fields of `binary_cross_entropy` (training.md §2.6).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BinaryCrossEntropyConfig {
    /// A graph output (probabilities, or logits with `with_logits`), required.
    pub prediction: String,
    /// A dataset field (values in `[0, 1]`), required.
    pub target: String,
    /// The prediction is a logit, not a probability. Default `false`.
    #[serde(default)]
    pub with_logits: bool,
}

impl LossConfig {
    /// Checks the names (1–256 bytes) and `0 ≤ label_smoothing ≤ 1` (training.md §2.1, §2.6).
    ///
    /// # Errors
    /// The first bad field, in table order (`/prediction`, `/target`, `/label_smoothing`).
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        check_name("/prediction", "prediction", self.prediction())?;
        check_name("/target", "target", self.target())?;
        if let Self::CrossEntropy(c) = self {
            check_closed(
                "/label_smoothing",
                "label_smoothing",
                c.label_smoothing,
                0.0,
                1.0,
            )?;
        }
        Ok(())
    }

    /// The `kind` string of the JSON form.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Mse(_) => "mse",
            Self::CrossEntropy(_) => "cross_entropy",
            Self::BinaryCrossEntropy(_) => "binary_cross_entropy",
        }
    }

    /// The graph output the loss reads.
    pub fn prediction(&self) -> &str {
        match self {
            Self::Mse(c) => &c.prediction,
            Self::CrossEntropy(c) => &c.prediction,
            Self::BinaryCrossEntropy(c) => &c.prediction,
        }
    }

    /// The dataset field the loss reads.
    pub fn target(&self) -> &str {
        match self {
            Self::Mse(c) => &c.target,
            Self::CrossEntropy(c) => &c.target,
            Self::BinaryCrossEntropy(c) => &c.target,
        }
    }
}

/// One entry of `metrics` (training.md §2.7, §3.4). Its JSON form is internally tagged:
/// `{"kind": "accuracy", "prediction": "logits", "target": "label"}`. `{"kind": "loss"}` is an
/// unknown kind: the loss is always reported and is never listed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum MetricConfig {
    /// `accuracy` (§3.4): multiclass or binary, chosen by the target's dtype.
    Accuracy(AccuracyConfig),
}

/// The fields of `accuracy` (training.md §2.7).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracyConfig {
    /// A graph output, required.
    pub prediction: String,
    /// A dataset field, required.
    pub target: String,
    /// The name in events and checkpoints: `[a-z0-9_]`, 1–64 bytes, not `loss`, unique in the
    /// list. Default `"accuracy"` (the kind).
    #[serde(default = "default_accuracy_name")]
    pub name: String,
    /// Binary accuracy only: an element counts as predicted 1 if it is above the threshold. Null
    /// (`None`) means `0.5`. Must round to a finite `f32`, because it is compared with `f32`
    /// predictions. Whether the metric is binary depends on the target's dtype, so "given with the
    /// multiclass form" is checked against the data ([`crate::Accuracy::check_shapes`]).
    #[serde(default)]
    pub threshold: Option<f64>,
}

fn default_accuracy_name() -> String {
    "accuracy".to_owned()
}

impl AccuracyConfig {
    /// Checks the names and the threshold (training.md §2.1, §2.7).
    ///
    /// # Errors
    /// The first bad field, in table order (`/prediction`, `/target`, `/name`, `/threshold`).
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        check_name("/prediction", "prediction", &self.prediction)?;
        check_name("/target", "target", &self.target)?;
        check_metric_name("/name", &self.name)?;
        if let Some(t) = self.threshold {
            check_f32_finite("/threshold", "threshold", t)?;
        }
        Ok(())
    }
}

/// A metric name: `[a-z0-9_]`, 1–64 bytes, not `loss` (training.md §2.7).
fn check_metric_name(pointer: &str, name: &str) -> Result<(), ConfigError> {
    if !(1..=MAX_METRIC_NAME_BYTES).contains(&name.len()) {
        return Err(ConfigError::new(
            pointer,
            format!(
                "`name` must be 1–{MAX_METRIC_NAME_BYTES} bytes, got {} bytes",
                name.len()
            ),
        ));
    }
    if let Some(c) = name
        .chars()
        .find(|c| !matches!(c, 'a'..='z' | '0'..='9' | '_'))
    {
        return Err(ConfigError::new(
            pointer,
            format!("`name` may only contain a-z, 0-9 and _, got {c:?} in {name:?}"),
        ));
    }
    if name == "loss" {
        return Err(ConfigError::new(
            pointer,
            "`name` must not be `loss`: the loss is always reported under that name",
        ));
    }
    Ok(())
}

impl MetricConfig {
    /// Checks the fields of the selected kind (training.md §2.7).
    ///
    /// # Errors
    /// The first bad field.
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        match self {
            Self::Accuracy(c) => c.validate(),
        }
    }

    /// The `kind` string of the JSON form.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Accuracy(_) => "accuracy",
        }
    }

    /// The metric's name in events and checkpoints (training.md §2.7).
    pub fn name(&self) -> &str {
        match self {
            Self::Accuracy(c) => &c.name,
        }
    }
}

/// Checks the `metrics` array (training.md §2.1, §2.7): at most 16 entries, each valid, the names
/// unique. Pointers are relative to the array (`/1/name`); the config parser prefixes `/metrics`.
///
/// # Errors
/// More than 16 entries at `""`; the first bad entry; a repeated name at the second use's `/name`.
pub(crate) fn validate_metrics(metrics: &[MetricConfig]) -> Result<(), ConfigError> {
    if metrics.len() > MAX_METRICS {
        return Err(ConfigError::new(
            "",
            format!(
                "`metrics` has at most {MAX_METRICS} entries, got {}",
                metrics.len()
            ),
        ));
    }
    for (i, m) in metrics.iter().enumerate() {
        m.validate().map_err(|e| ConfigError {
            pointer: format!("/{i}{}", e.pointer),
            message: e.message,
        })?;
        if let Some(first) = metrics[..i].iter().position(|o| o.name() == m.name()) {
            return Err(ConfigError::new(
                &format!("/{i}/name"),
                format!(
                    "metric name `{}` is already used by entry {first}; names must be unique \
                     (set `name`)",
                    m.name()
                ),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "config_loss_tests.rs"]
mod loss_tests;
