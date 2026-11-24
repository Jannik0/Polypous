//! The optimizers `sgd`, `adam` and `adamw` (`docs/specs/training.md` §3.1, §3.1.1–§3.1.3) and
//! their state (§5.2), written in `DynTensor` ops on tensors without autodiff context (ADR-0014).
//!
//! Each follows PyTorch 2.15.0's single-tensor algorithm (`foreach=False`) in the op order of the
//! spec's pseudocode. Every scalar is computed in `f64` from the config values as written and
//! rounded once to `f32` where it meets a tensor ([`f32_of`]); `x^y` is [`f64::powf`]. PyTorch's
//! CPU kernels fuse some of the steps (FMA), so the results differ from PyTorch's by a few
//! roundings (testing-strategy.md §2, "Optimizer tolerance").
//!
//! **The step count.** The optimizer keeps no step count of its own: [`Optimizer::step`] takes the
//! trainer's global step counter `k`, and Adam's `t = k + 1`. training.md §1.1 guarantees that the
//! set of params with a gradient is the same at every step, so a param's own count always equals
//! `k`, and a checkpoint stores only `k` (§5.2). A param that has a gradient but no state at a step
//! `k > 0` therefore means the state doesn't fit the graph (§5.4 step 4), and is an error.

use std::collections::BTreeMap;

use polypous_backend::{DType, DynTensor, Scalar, TensorError};
use thiserror::Error;

use crate::config::{AdamConfig, ConfigError, OptimizerConfig, SgdConfig};

/// Tensors by param name: the trainable params, or their gradients.
pub type TensorMap = BTreeMap<String, DynTensor>;

/// The state name of SGD's momentum buffer (training.md §5.2; PyTorch's `state_dict()` name).
pub const MOMENTUM_BUFFER: &str = "momentum_buffer";
/// The state name of Adam's first moment `m` (training.md §5.2).
pub const EXP_AVG: &str = "exp_avg";
/// The state name of Adam's second moment `v` (training.md §5.2).
pub const EXP_AVG_SQ: &str = "exp_avg_sq";

/// Why an optimizer, schedule or clipping call failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OptimizerError {
    /// The config has an out-of-range field.
    #[error("invalid config: {0}")]
    Config(#[from] ConfigError),
    /// A learning rate that is negative, NaN or infinite.
    #[error("the learning rate must be finite and ≥ 0, got {lr}")]
    LearningRate {
        /// The learning rate given.
        lr: f64,
    },
    /// A schedule asked for a step outside `0 … T − 1`, or built with `T = 0`.
    #[error("step {step} is outside the {total_steps} steps of the schedule")]
    StepOutOfRange {
        /// The step asked for.
        step: u64,
        /// `T`.
        total_steps: u64,
    },
    /// A gradient for a name that isn't among the params.
    #[error("gradient for `{name}`, which is not a param")]
    UnknownParam {
        /// The gradient's name.
        name: String,
    },
    /// A param or gradient that isn't `f32`, or a gradient whose shape or device differs from its
    /// param's (or, for clipping, from the first gradient's device).
    #[error("`{name}`: {reason}")]
    Mismatch {
        /// The param name.
        name: String,
        /// What differs.
        reason: String,
    },
    /// A param with a gradient but no optimizer state at a step `k > 0` (training.md §5.4 step 4:
    /// the state doesn't fit the graph).
    #[error("param `{name}` has a gradient but no optimizer state at step {step}")]
    MissingState {
        /// The param name.
        name: String,
        /// The step `k`.
        step: u64,
    },
    /// Optimizer state given to [`Optimizer::import_state`] that doesn't fit the optimizer or the
    /// params: an unknown param or state name, a duplicate, a wrong dtype, shape or device, or a
    /// param with only some of its state names.
    #[error("optimizer state `{name}`: {reason}")]
    InvalidState {
        /// `<param name>:<state name>`, as in `optimizer.safetensors` (§5.2), or the param name.
        name: String,
        /// What is wrong.
        reason: String,
    },
    /// A tensor op failed.
    #[error(transparent)]
    Backend(#[from] TensorError),
}

/// One state tensor of one param, as exported and imported for checkpoints (training.md §5.2).
#[derive(Clone, Debug)]
pub struct StateTensor {
    /// The param name.
    pub param: String,
    /// The state name: [`MOMENTUM_BUFFER`], [`EXP_AVG`] or [`EXP_AVG_SQ`].
    pub state: String,
    /// The state, `f32`, with the param's shape, on the param's device, without autodiff context.
    pub tensor: DynTensor,
}

/// The state of one param.
#[derive(Clone, Debug)]
#[allow(
    clippy::large_enum_variant,
    reason = "one entry per param; the variants differ by one tensor handle, and boxing would add an allocation per param"
)]
enum ParamState {
    /// SGD with `momentum ≠ 0`.
    Momentum { buffer: DynTensor },
    /// Adam and AdamW.
    Adam {
        exp_avg: DynTensor,
        exp_avg_sq: DynTensor,
    },
}

/// An optimizer of training.md §3.1 with its per-param state.
///
/// The state is keyed by param name and created at a param's first step (SGD's
/// `momentum_buffer` as a copy of the gradient, Adam's `exp_avg` and `exp_avg_sq` as zeros). It is
/// `f32` on the params' device.
#[derive(Clone, Debug)]
pub struct Optimizer {
    config: OptimizerConfig,
    state: BTreeMap<String, ParamState>,
}

/// The single rounding of a scalar from `f64` to `f32` (round to nearest, ties to even;
/// training.md §3.1, "Scalars").
#[allow(
    clippy::cast_possible_truncation,
    reason = "the single rounding to f32 of training.md §3.1"
)]
pub(crate) fn f32_of(x: f64) -> f32 {
    x as f32
}

/// A tensor holding the `f32` scalar `v`, broadcastable to `like` (shape `[1, …, 1]` of its rank,
/// so the result keeps `like`'s shape), on `like`'s device.
pub(crate) fn scalar(v: f32, like: &DynTensor) -> Result<DynTensor, TensorError> {
    // `f64::from(v)` is exact, so `full` rounds it back to `v` exactly.
    DynTensor::full(
        &vec![1; like.rank()],
        Scalar::Float(f64::from(v)),
        DType::F32,
        like.device(),
    )
}

/// Checks that a param or gradient is `f32`.
pub(crate) fn check_f32(name: &str, what: &str, t: &DynTensor) -> Result<(), OptimizerError> {
    if t.dtype() == DType::F32 {
        Ok(())
    } else {
        Err(OptimizerError::Mismatch {
            name: name.to_owned(),
            reason: format!("the {what} is {:?}, training is f32 only", t.dtype()),
        })
    }
}

fn check_lr(lr: f64) -> Result<(), OptimizerError> {
    if lr >= 0.0 && lr.is_finite() {
        Ok(())
    } else {
        Err(OptimizerError::LearningRate { lr })
    }
}

impl Optimizer {
    /// An optimizer with no state yet.
    ///
    /// # Errors
    /// [`OptimizerError::Config`] if the config is out of range (training.md §2.8).
    pub fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        config.validate()?;
        Ok(Self {
            config,
            state: BTreeMap::new(),
        })
    }

    /// The config.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// The state names each param with state has (training.md §5.2): `momentum_buffer` for SGD
    /// with `momentum ≠ 0`, none for SGD without momentum, `exp_avg` and `exp_avg_sq` for Adam and
    /// AdamW.
    pub fn state_names(&self) -> &'static [&'static str] {
        match &self.config {
            OptimizerConfig::Sgd(c) if c.momentum != 0.0 => &[MOMENTUM_BUFFER],
            OptimizerConfig::Sgd(_) => &[],
            OptimizerConfig::Adam(_) | OptimizerConfig::Adamw(_) => &[EXP_AVG, EXP_AVG_SQ],
        }
    }

    /// Whether a param has optimizer state.
    pub fn has_state(&self, param: &str) -> bool {
        self.state.contains_key(param)
    }

    /// Optimizer step `k` with learning rate `lr = lr(k)` (training.md §1.2 step 7, §3.1).
    ///
    /// Updates every param in `grads` (the trainable params that have a gradient, after clipping)
    /// in `params`, and its state. Params without a gradient keep their value and state. Either
    /// every param and its state is updated, or (on an error) none is.
    ///
    /// # Errors
    /// - [`OptimizerError::LearningRate`]: `lr` negative or not finite.
    /// - [`OptimizerError::UnknownParam`]: a gradient whose name isn't in `params`.
    /// - [`OptimizerError::Mismatch`]: a param or gradient that isn't `f32`, or a gradient whose
    ///   shape or device differs from its param's.
    /// - [`OptimizerError::MissingState`]: an optimizer with state, and a param with a gradient
    ///   but no state at `k > 0`.
    /// - [`OptimizerError::Backend`]: a tensor op failed.
    pub fn step(
        &mut self,
        k: u64,
        lr: f64,
        params: &mut TensorMap,
        grads: &TensorMap,
    ) -> Result<(), OptimizerError> {
        check_lr(lr)?;
        let stateful = !self.state_names().is_empty();
        for (name, g) in grads {
            let p = params
                .get(name)
                .ok_or_else(|| OptimizerError::UnknownParam { name: name.clone() })?;
            check_f32(name, "param", p)?;
            check_f32(name, "gradient", g)?;
            if g.shape() != p.shape() {
                return Err(OptimizerError::Mismatch {
                    name: name.clone(),
                    reason: format!(
                        "the gradient has shape {:?}, the param {:?}",
                        g.shape(),
                        p.shape()
                    ),
                });
            }
            if g.device() != p.device() {
                return Err(OptimizerError::Mismatch {
                    name: name.clone(),
                    reason: format!(
                        "the gradient is on {}, the param on {}",
                        g.device(),
                        p.device()
                    ),
                });
            }
            if stateful && k > 0 && !self.state.contains_key(name) {
                return Err(OptimizerError::MissingState {
                    name: name.clone(),
                    step: k,
                });
            }
        }
        let mut updates = Vec::with_capacity(grads.len());
        for (name, g) in grads {
            // Checked above: every gradient has a param.
            let Some(p) = params.get(name) else {
                return Err(OptimizerError::UnknownParam { name: name.clone() });
            };
            let p = p.clone().without_autodiff();
            let g = g.clone().without_autodiff();
            let state = self.state.get(name);
            let (p, state) = match &self.config {
                OptimizerConfig::Sgd(c) => sgd(c, lr, p, g, state)?,
                OptimizerConfig::Adam(c) => adam(c, false, k, lr, p, g, state)?,
                OptimizerConfig::Adamw(c) => {
                    let c = AdamConfig {
                        lr: c.lr,
                        betas: c.betas,
                        eps: c.eps,
                        weight_decay: c.weight_decay,
                    };
                    adam(&c, true, k, lr, p, g, state)?
                }
            };
            updates.push((name, p, state));
        }
        for (name, p, state) in updates {
            params.insert(name.clone(), p);
            if let Some(state) = state {
                self.state.insert(name.clone(), state);
            }
        }
        Ok(())
    }

    /// The state of every param that has state, for a checkpoint (training.md §5.2): by param name,
    /// then in the order of [`Optimizer::state_names`].
    pub fn export_state(&self) -> Vec<StateTensor> {
        let mut out = Vec::new();
        for (param, state) in &self.state {
            let mut push = |name: &str, tensor: &DynTensor| {
                out.push(StateTensor {
                    param: param.clone(),
                    state: name.to_owned(),
                    tensor: tensor.clone(),
                });
            };
            match state {
                ParamState::Momentum { buffer } => push(MOMENTUM_BUFFER, buffer),
                ParamState::Adam {
                    exp_avg,
                    exp_avg_sq,
                } => {
                    push(EXP_AVG, exp_avg);
                    push(EXP_AVG_SQ, exp_avg_sq);
                }
            }
        }
        out
    }

    /// Replaces the state with `entries`, for a resume (training.md §5.2, §5.4 step 4). `params`
    /// are the trainable params, which give each state tensor's name, shape and device.
    ///
    /// Every entry must name a param of `params` and a state name of this optimizer
    /// ([`Optimizer::state_names`]), be `f32` with the param's shape on its device, and appear once;
    /// a param has all of its state names or none. On an error the state is unchanged.
    ///
    /// # Errors
    /// [`OptimizerError::InvalidState`] naming the first entry (or param) that breaks a rule.
    pub fn import_state(
        &mut self,
        params: &TensorMap,
        entries: Vec<StateTensor>,
    ) -> Result<(), OptimizerError> {
        let names = self.state_names();
        let mut by_param: BTreeMap<String, BTreeMap<&'static str, DynTensor>> = BTreeMap::new();
        for entry in entries {
            let full = format!("{}:{}", entry.param, entry.state);
            let invalid = |reason: String| OptimizerError::InvalidState {
                name: full.clone(),
                reason,
            };
            let Some(&state) = names.iter().find(|&&n| n == entry.state) else {
                return Err(invalid(format!(
                    "`{}` is not a state of `{}` (expected one of {names:?})",
                    entry.state,
                    self.config.kind()
                )));
            };
            let Some(p) = params.get(&entry.param) else {
                return Err(invalid(format!(
                    "`{}` is not a trainable param",
                    entry.param
                )));
            };
            let t = entry.tensor.without_autodiff();
            if t.dtype() != DType::F32 {
                return Err(invalid(format!("dtype {:?}, expected f32", t.dtype())));
            }
            if t.shape() != p.shape() {
                return Err(invalid(format!(
                    "shape {:?}, the param has {:?}",
                    t.shape(),
                    p.shape()
                )));
            }
            if t.device() != p.device() {
                return Err(invalid(format!(
                    "on {}, the param is on {}",
                    t.device(),
                    p.device()
                )));
            }
            let slot = by_param.entry(entry.param).or_default();
            if slot.insert(state, t).is_some() {
                return Err(invalid("given twice".to_owned()));
            }
        }
        let mut state = BTreeMap::new();
        for (param, mut tensors) in by_param {
            let missing: Vec<&str> = names
                .iter()
                .copied()
                .filter(|n| !tensors.contains_key(n))
                .collect();
            if !missing.is_empty() {
                return Err(OptimizerError::InvalidState {
                    name: param,
                    reason: format!("has only some of its state, missing {missing:?}"),
                });
            }
            let mut take = |n: &str| tensors.remove(n);
            let s = match (take(MOMENTUM_BUFFER), take(EXP_AVG), take(EXP_AVG_SQ)) {
                (Some(buffer), None, None) => ParamState::Momentum { buffer },
                (None, Some(exp_avg), Some(exp_avg_sq)) => ParamState::Adam {
                    exp_avg,
                    exp_avg_sq,
                },
                // `names` is one of the two sets above, and every entry was checked against it.
                _ => {
                    return Err(OptimizerError::InvalidState {
                        name: param,
                        reason: "state names of different optimizers".to_owned(),
                    });
                }
            };
            state.insert(param, s);
        }
        self.state = state;
        Ok(())
    }
}

type Updated = (DynTensor, Option<ParamState>);

/// One SGD step of one param (training.md §3.1.1).
fn sgd(
    c: &SgdConfig,
    lr: f64,
    p: DynTensor,
    g: DynTensor,
    state: Option<&ParamState>,
) -> Result<Updated, OptimizerError> {
    let mut g = g;
    if c.weight_decay != 0.0 {
        // g ← g + weight_decay · p
        g = g.add(p.clone().mul(scalar(f32_of(c.weight_decay), &p)?)?)?;
    }
    let mut new_state = None;
    if c.momentum != 0.0 {
        let momentum = scalar(f32_of(c.momentum), &p)?;
        let buf = match state {
            // The first step: buf ← g (the tensors are immutable, so sharing is a copy).
            None => g.clone(),
            // buf ← buf · momentum + (1 − dampening) · g
            Some(ParamState::Momentum { buffer }) => buffer
                .clone()
                .mul(momentum.clone())?
                .add(scalar(f32_of(1.0 - c.dampening), &p)?.mul(g.clone())?)?,
            Some(ParamState::Adam { .. }) => {
                return Err(OptimizerError::InvalidState {
                    name: String::new(),
                    reason: "Adam state on an SGD optimizer".to_owned(),
                });
            }
        };
        // g ← nesterov ? g + momentum · buf : buf
        g = if c.nesterov {
            g.add(momentum.mul(buf.clone())?)?
        } else {
            buf.clone()
        };
        new_state = Some(ParamState::Momentum { buffer: buf });
    }
    // p ← p + f32(−η) · g
    let p = p.clone().add(scalar(f32_of(-lr), &p)?.mul(g)?)?;
    Ok((p, new_state))
}

/// `t = k + 1` as `f64` (exact for every `k` below `2^53`).
#[allow(
    clippy::cast_precision_loss,
    reason = "step counts stay far below 2^53, where the conversion is exact"
)]
fn step_count(k: u64) -> f64 {
    k.saturating_add(1) as f64
}

/// One Adam (or, with `decoupled`, AdamW) step of one param (training.md §3.1.2, §3.1.3).
fn adam(
    c: &AdamConfig,
    decoupled: bool,
    k: u64,
    lr: f64,
    p: DynTensor,
    g: DynTensor,
    state: Option<&ParamState>,
) -> Result<Updated, OptimizerError> {
    let [beta1, beta2] = c.betas;
    let (mut p, mut g) = (p, g);
    if c.weight_decay != 0.0 {
        if decoupled {
            // p ← p · f32(1 − η · weight_decay), the product and difference in f64
            p = p
                .clone()
                .mul(scalar(f32_of(1.0 - lr * c.weight_decay), &p)?)?;
        } else {
            // g ← g + weight_decay · p
            g = g.add(p.clone().mul(scalar(f32_of(c.weight_decay), &p)?)?)?;
        }
    }
    let (m, v) = match state {
        Some(ParamState::Adam {
            exp_avg,
            exp_avg_sq,
        }) => (exp_avg.clone(), exp_avg_sq.clone()),
        // Zero before the param's first step.
        None => (
            p.full_like(Scalar::Float(0.0), None)?,
            p.full_like(Scalar::Float(0.0), None)?,
        ),
        Some(ParamState::Momentum { .. }) => {
            return Err(OptimizerError::InvalidState {
                name: String::new(),
                reason: "SGD state on an Adam optimizer".to_owned(),
            });
        }
    };
    // m ← lerp(m, g, 1 − β1), ATen's formula. The weight is an f32 scalar in ATen's kernel, and
    // so is its `1 − w`: for `w ≥ 0.5` that difference is exact (Sterbenz), so it is computed in
    // f64 from the rounded `w` and is exact in f32 as well.
    let w = f32_of(1.0 - beta1);
    let diff = g.clone().sub(m.clone())?;
    let m = if w.abs() < 0.5 {
        // m + w · (g − m)
        m.add(scalar(w, &p)?.mul(diff)?)?
    } else {
        // g − (g − m) · (1 − w)
        g.clone()
            .sub(diff.mul(scalar(f32_of(1.0 - f64::from(w)), &p)?)?)?
    };
    // v ← v · β2 + (1 − β2) · (g · g)
    let v = v
        .mul(scalar(f32_of(beta2), &p)?)?
        .add(scalar(f32_of(1.0 - beta2), &p)?.mul(g.clone().mul(g)?)?)?;
    let t = step_count(k);
    let bias_correction1 = 1.0 - beta1.powf(t);
    let bias_correction2 = 1.0 - beta2.powf(t);
    let step_size = lr / bias_correction1;
    // denom = sqrt(v) / f32(bias_correction2^0.5) + f32(eps)
    let denom = v
        .clone()
        .sqrt()?
        .div(scalar(f32_of(bias_correction2.powf(0.5)), &p)?)?
        .add(scalar(f32_of(c.eps), &p)?)?;
    // p ← p + f32(−step_size) · (m / denom)
    let p = p
        .clone()
        .add(scalar(f32_of(-step_size), &p)?.mul(m.clone().div(denom)?)?)?;
    Ok((
        p,
        Some(ParamState::Adam {
            exp_avg: m,
            exp_avg_sq: v,
        }),
    ))
}
