//! Learning-rate schedules (`docs/specs/training.md` §3.2).
//!
//! A schedule is a pure function `lr(k)` of its config, the optimizer's `lr₀` and the total step
//! count `T`, in `f64`, evaluated left to right as the spec writes it. It has no state beyond `k`
//! and `T`, so a checkpoint saves none (§5.2): the same `k` always gives the same value.
//!
//! `step` multiplies `lr₀` by `gamma` once per completed interval, one rounding per
//! multiplication, as PyTorch's chainable `StepLR` does. [`Schedule`] remembers the last product
//! it computed so that consecutive steps don't repeat the multiplications; that memo is a cache of
//! the same function, not state, and the value never depends on it.

use std::f64::consts::PI;
use std::sync::{Mutex, PoisonError};

use crate::config::ScheduleConfig;
use crate::optimizer::OptimizerError;

/// A schedule of training.md §3.2, bound to `lr₀` and `T`.
#[derive(Debug)]
pub struct Schedule {
    config: ScheduleConfig,
    lr0: f64,
    total_steps: u64,
    /// `step` only: `(n, lr₀ multiplied by gamma n times)` of the last call.
    memo: Mutex<(u64, f64)>,
}

impl Clone for Schedule {
    fn clone(&self) -> Self {
        Self {
            config: self.config,
            lr0: self.lr0,
            total_steps: self.total_steps,
            memo: Mutex::new((0, self.lr0)),
        }
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "step counts stay far below 2^53, where the conversion is exact (as Python's int → float)"
)]
fn f(v: u64) -> f64 {
    v as f64
}

impl Schedule {
    /// The schedule of `config` for an optimizer with learning rate `lr0` and `total_steps`
    /// (`T`) optimizer steps.
    ///
    /// # Errors
    /// [`OptimizerError::Config`] if the config is out of range for `lr0` (training.md §2.8);
    /// [`OptimizerError::LearningRate`] if `lr0` is negative or not finite;
    /// [`OptimizerError::StepOutOfRange`] if `total_steps` is 0.
    pub fn new(config: ScheduleConfig, lr0: f64, total_steps: u64) -> Result<Self, OptimizerError> {
        if !(lr0 >= 0.0 && lr0.is_finite()) {
            return Err(OptimizerError::LearningRate { lr: lr0 });
        }
        config.validate(lr0)?;
        if total_steps == 0 {
            return Err(OptimizerError::StepOutOfRange {
                step: 0,
                total_steps,
            });
        }
        Ok(Self {
            config,
            lr0,
            total_steps,
            memo: Mutex::new((0, lr0)),
        })
    }

    /// The config.
    pub fn config(&self) -> &ScheduleConfig {
        &self.config
    }

    /// `T`.
    pub fn total_steps(&self) -> u64 {
        self.total_steps
    }

    /// The learning rate of optimizer step `k` (training.md §3.2).
    ///
    /// # Errors
    /// [`OptimizerError::StepOutOfRange`] if `k ≥ T`.
    pub fn lr(&self, k: u64) -> Result<f64, OptimizerError> {
        let total = self.total_steps;
        if k >= total {
            return Err(OptimizerError::StepOutOfRange {
                step: k,
                total_steps: total,
            });
        }
        let lr0 = self.lr0;
        Ok(match self.config {
            ScheduleConfig::Constant {} => lr0,
            ScheduleConfig::Step { every, gamma } => self.step_lr(k / every, gamma),
            ScheduleConfig::LinearWarmup { warmup_steps } => warmup(lr0, k, warmup_steps),
            ScheduleConfig::Cosine {
                warmup_steps,
                min_lr,
            } => {
                if k < warmup_steps {
                    warmup(lr0, k, warmup_steps)
                } else {
                    // k ≥ W and k < T, so T − W ≥ 1.
                    min_lr
                        + (lr0 - min_lr)
                            * (1.0 + (PI * f(k - warmup_steps) / f(total - warmup_steps)).cos())
                            / 2.0
                }
            }
        })
    }

    /// `lr₀` multiplied by `gamma` `n` times, one rounding per multiplication.
    fn step_lr(&self, n: u64, gamma: f64) -> f64 {
        let mut memo = self.memo.lock().unwrap_or_else(PoisonError::into_inner);
        let (mut i, mut value) = if memo.0 <= n { *memo } else { (0, self.lr0) };
        while i < n {
            let next = value * gamma;
            if next == value {
                // A fixed point (gamma = 1, or the value reached 0): every further product is the
                // same, so the remaining multiplications change nothing.
                break;
            }
            value = next;
            i += 1;
        }
        *memo = (n, value);
        value
    }
}

/// `lr₀ · ((k + 1) / W)` for `k < W`, then `lr₀` (training.md §3.2, `linear_warmup`).
fn warmup(lr0: f64, k: u64, w: u64) -> f64 {
    if k < w { lr0 * (f(k + 1) / f(w)) } else { lr0 }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::float_cmp)] // Test code; exact values are the point.

    use super::*;

    #[test]
    fn same_k_same_value_in_any_order() {
        let s = Schedule::new(
            ScheduleConfig::Step {
                every: 2,
                gamma: 0.3,
            },
            0.1,
            40,
        )
        .unwrap();
        let forward: Vec<f64> = (0..40).map(|k| s.lr(k).unwrap()).collect();
        let backward: Vec<f64> = (0..40).rev().map(|k| s.lr(k).unwrap()).collect();
        let fresh = Schedule::new(*s.config(), 0.1, 40).unwrap();
        for k in [37u64, 3, 20, 0, 39, 39, 1] {
            assert_eq!(fresh.lr(k).unwrap(), forward[k as usize]);
        }
        let reversed: Vec<f64> = backward.into_iter().rev().collect();
        assert_eq!(forward, reversed);
        assert_eq!(s.clone().lr(25).unwrap(), forward[25]);
    }

    #[test]
    fn step_multiplies_once_per_interval() {
        let s = Schedule::new(
            ScheduleConfig::Step {
                every: 3,
                gamma: 0.1,
            },
            0.1,
            10,
        )
        .unwrap();
        let mut want = 0.1f64;
        for k in 0..10 {
            if k > 0 && k % 3 == 0 {
                want *= 0.1;
            }
            assert_eq!(s.lr(k).unwrap(), want, "k = {k}");
        }
        // One pow rounds differently: ((0.01 · 0.1) · 0.1 = 1e-4 exactly, 0.01 · 0.1^2 doesn't.
        let s = Schedule::new(
            ScheduleConfig::Step {
                every: 1,
                gamma: 0.1,
            },
            0.01,
            3,
        )
        .unwrap();
        assert_eq!(s.lr(2).unwrap(), 0.0001);
        assert_ne!(s.lr(2).unwrap(), 0.01 * 0.1f64.powf(2.0));
    }

    #[test]
    fn step_gamma_zero_and_one() {
        let zero = Schedule::new(
            ScheduleConfig::Step {
                every: 4,
                gamma: 0.0,
            },
            0.5,
            12,
        )
        .unwrap();
        assert_eq!(zero.lr(3).unwrap(), 0.5);
        for k in 4..12 {
            assert_eq!(zero.lr(k).unwrap(), 0.0);
        }
        let one = Schedule::new(
            ScheduleConfig::Step {
                every: 1,
                gamma: 1.0,
            },
            0.5,
            u64::MAX,
        )
        .unwrap();
        // The fixed point ends the loop: this is instant.
        assert_eq!(one.lr(u64::MAX - 1).unwrap(), 0.5);
        let tiny = Schedule::new(
            ScheduleConfig::Step {
                every: 1,
                gamma: 0.5,
            },
            0.5,
            u64::MAX,
        )
        .unwrap();
        assert_eq!(tiny.lr(1 << 40).unwrap(), 0.0);
    }

    #[test]
    fn warmup_reaches_lr0_at_w_minus_1() {
        let s = Schedule::new(ScheduleConfig::LinearWarmup { warmup_steps: 4 }, 0.2, 8).unwrap();
        let got: Vec<f64> = (0..8).map(|k| s.lr(k).unwrap()).collect();
        assert_eq!(
            got,
            [0.2 * 0.25, 0.2 * 0.5, 0.2 * 0.75, 0.2, 0.2, 0.2, 0.2, 0.2]
        );
    }

    #[test]
    fn cosine_edges() {
        // W = 0: no warmup, starts at lr₀.
        let s = Schedule::new(
            ScheduleConfig::Cosine {
                warmup_steps: 0,
                min_lr: 0.01,
            },
            0.1,
            5,
        )
        .unwrap();
        assert_eq!(s.lr(0).unwrap(), 0.1);
        assert!(s.lr(4).unwrap() > 0.01);
        // W ≥ T: every step is a warmup step.
        let s = Schedule::new(
            ScheduleConfig::Cosine {
                warmup_steps: 10,
                min_lr: 0.0,
            },
            0.1,
            6,
        )
        .unwrap();
        for k in 0..6 {
            assert_eq!(s.lr(k).unwrap(), 0.1 * (f(k + 1) / 10.0));
        }
        // The first step after the warmup is lr₀.
        let s = Schedule::new(
            ScheduleConfig::Cosine {
                warmup_steps: 3,
                min_lr: 0.0,
            },
            0.1,
            9,
        )
        .unwrap();
        assert_eq!(s.lr(3).unwrap(), 0.1);
    }

    #[test]
    fn errors() {
        assert!(matches!(
            Schedule::new(ScheduleConfig::Constant {}, 0.1, 0),
            Err(OptimizerError::StepOutOfRange { .. })
        ));
        assert!(matches!(
            Schedule::new(ScheduleConfig::Constant {}, f64::NAN, 3),
            Err(OptimizerError::LearningRate { .. })
        ));
        assert!(matches!(
            Schedule::new(
                ScheduleConfig::Cosine {
                    warmup_steps: 0,
                    min_lr: 0.2
                },
                0.1,
                3
            ),
            Err(OptimizerError::Config(_))
        ));
        let s = Schedule::new(ScheduleConfig::Constant {}, 0.1, 3).unwrap();
        assert!(matches!(
            s.lr(3),
            Err(OptimizerError::StepOutOfRange { step: 3, .. })
        ));
    }
}
