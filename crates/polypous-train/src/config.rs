//! The `optimizer`, `schedule` and `gradient_clip` objects of a training config
//! (`docs/specs/training.md` §2.1, §2.2, §2.8), as Rust types.
//!
//! The types deserialize directly from the JSON of §2.8 (`serde`, internally tagged by `kind`,
//! unknown fields rejected, missing optional fields filled with their defaults) and serialize every
//! field in the order of the spec's tables, so the config parser (#82) can embed them. Serde
//! checks the JSON types (an integer field rejects `5.0`, a negative or too-large integer doesn't
//! fit its Rust type); the ranges are checked by `validate`, which returns a [`ConfigError`] whose
//! pointer is relative to the object (`/betas/1`). The parser prefixes the object's own pointer
//! (`/optimizer`), as §2.1 requires.
//!
//! [`crate::Optimizer::new`], [`crate::Schedule::new`] and [`crate::clip_gradients`] validate the
//! config they are given, so an invalid config never reaches the arithmetic.

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
    pub fn validate(&self) -> Result<(), ConfigError> {
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
    pub fn validate(&self) -> Result<(), ConfigError> {
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
    pub fn validate(&self) -> Result<(), ConfigError> {
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
    pub fn validate(&self) -> Result<(), ConfigError> {
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
    pub fn validate(&self, lr: f64) -> Result<(), ConfigError> {
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
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.max_norm.is_nan() || self.max_norm <= 0.0 {
            return Err(ConfigError::new(
                "/max_norm",
                format!("`max_norm` must be > 0, got {}", self.max_norm),
            ));
        }
        check_f32_finite("/max_norm", "max_norm", self.max_norm)
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
