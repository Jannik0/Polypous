//! Training events, the outcome of a run and the trainer's info (`docs/specs/training.md` §4,
//! public-api.md §1.4), with their documented JSON, and the clock the trainer times itself with.
//!
//! Numbers follow §4.1: integers are JSON integers; `loss`, `lr`, metric values and
//! `elapsed_seconds` are `f64` in the shortest form that reads back as the same `f64`, and a
//! non-finite value is the string `"nan"`, `"inf"` or `"-inf"` ([`crate::json::Number`]).
//!
//! Every reported metric of a finished epoch has a value: a dataset has at least one sample
//! (§3.5), so every epoch has at least one step and every validation pass at least one batch. The
//! `None` of an accumulator without samples (§3.4) therefore never reaches these types; the
//! trainer reports it as an internal error if it ever did.

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use polypous_backend::device::DeviceId;
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

use crate::json::{self, Number};

fn number<S: Serializer>(x: &f64, s: S) -> Result<S::Ok, S::Error> {
    Number(*x).serialize(s)
}

/// Reported metric names → values (training.md §2.7), in the reported order. JSON: an object in
/// that order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metrics(pub Vec<(String, f64)>);

impl Metrics {
    /// The value of `name`.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.iter().find(|(n, _)| n == name).map(|(_, v)| *v)
    }

    /// Whether there are no metrics.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for Metrics {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (name, value) in &self.0 {
            map.serialize_entry(name, &Number(*value))?;
        }
        map.end()
    }
}

/// One finished epoch (training.md §5.2 `history`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HistoryEntry {
    /// The epoch, 0-based.
    pub epoch: u64,
    /// Optimizer steps completed when it finished.
    pub step: u64,
    /// Its metrics, in the reported order.
    pub metrics: Metrics,
}

/// The best checkpoint by `checkpoint.keep_best` (training.md §5.2, §5.3).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Best {
    /// The metric (`val.accuracy`).
    pub metric: String,
    /// Its best value.
    #[serde(serialize_with = "number")]
    pub value: f64,
    /// The history entry the value comes from: its epoch …
    pub epoch: u64,
    /// … and its step.
    pub step: u64,
    /// In `state.json`: the checkpoint's directory name (`step-000000000844`). In a
    /// [`TrainingOutcome`]: its path (`runs/mnist/step-000000000844`).
    pub checkpoint: String,
}

/// Why a run returned (public-api.md §1.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Every step of every epoch ran.
    Completed,
    /// The cancellation token was set (training.md §4.3).
    Cancelled,
}

/// What `run` returns and the `finished` event carries (public-api.md §1.4).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainingOutcome {
    /// `completed` or `cancelled`.
    pub stop_reason: StopReason,
    /// Optimizer steps completed.
    pub step: u64,
    /// Epochs finished (their validation pass included).
    pub epochs_completed: u64,
    /// The last finished epoch's metrics; empty if none finished.
    pub final_metrics: Metrics,
    /// The best checkpoint, with its path; `None` without `keep_best` or before any value.
    pub best: Option<Best>,
    /// Every finished epoch.
    pub history: Vec<HistoryEntry>,
    /// The path of the newest checkpoint, if any.
    pub last_checkpoint: Option<String>,
    /// Wall-clock seconds of every run of this training (informative).
    #[serde(serialize_with = "number")]
    pub elapsed_seconds: f64,
}

impl TrainingOutcome {
    /// The JSON form (public-api.md §1.4).
    pub fn to_json(&self) -> String {
        json::to_string(self)
    }
}

/// A training event (training.md §4.1). JSON: an object with an `"event"` tag.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TrainingEvent {
    /// The run starts.
    Started {
        /// The training device.
        device: DeviceId,
        /// `epochs`.
        epochs: u64,
        /// `S`.
        steps_per_epoch: u64,
        /// `T = epochs · S`.
        total_steps: u64,
        /// Training samples (after a split).
        train_samples: u64,
        /// Validation samples, `0` without a validation set.
        validation_samples: u64,
        /// Values in the trainable params.
        trainable_elements: u64,
        /// The step the run starts at.
        step: u64,
    },
    /// An epoch starts (or continues: `batch` is the first batch this run trains in it).
    EpochStarted {
        /// The epoch, 0-based.
        epoch: u64,
        /// The first batch this run trains in the epoch (`S` if only its validation is left).
        batch: u64,
    },
    /// An optimizer step finished (throttled, §4.2).
    Step {
        /// The epoch.
        epoch: u64,
        /// The batch of the epoch.
        batch: u64,
        /// The step's 0-based index `k`.
        step: u64,
        /// The batch loss.
        #[serde(serialize_with = "number")]
        loss: f64,
        /// `lr(k)`.
        #[serde(serialize_with = "number")]
        lr: f64,
    },
    /// The validation pass of an epoch starts.
    ValidationStarted {
        /// The epoch.
        epoch: u64,
    },
    /// An epoch finished.
    EpochFinished {
        /// The epoch.
        epoch: u64,
        /// Steps completed.
        step: u64,
        /// Its metrics, in the reported order.
        metrics: Metrics,
    },
    /// A checkpoint was written.
    CheckpointSaved {
        /// The checkpoint directory, as resolved.
        path: String,
        /// Its step.
        step: u64,
    },
    /// A checkpoint was deleted (`keep_last`, or a newer checkpoint discarded on resume).
    CheckpointRemoved {
        /// The checkpoint directory.
        path: String,
    },
    /// The run returned `Ok`: the last event.
    Finished {
        /// What `run` returns.
        outcome: Box<TrainingOutcome>,
    },
}

impl TrainingEvent {
    /// The JSON form (training.md §4.1).
    pub fn to_json(&self) -> String {
        json::to_string(self)
    }

    /// The `"event"` tag.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Started { .. } => "started",
            Self::EpochStarted { .. } => "epoch_started",
            Self::Step { .. } => "step",
            Self::ValidationStarted { .. } => "validation_started",
            Self::EpochFinished { .. } => "epoch_finished",
            Self::CheckpointSaved { .. } => "checkpoint_saved",
            Self::CheckpointRemoved { .. } => "checkpoint_removed",
            Self::Finished { .. } => "finished",
        }
    }
}

/// A trainable param in [`TrainerInfo`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct TrainableParam {
    /// The param name.
    pub name: String,
    /// Its shape.
    pub shape: Vec<usize>,
}

/// `trainer.info()` (public-api.md §1.4): the `started` event's fields with `step` the steps
/// completed so far, `finished`, and the trainable params in declaration order.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TrainerInfo {
    /// The training device.
    pub device: DeviceId,
    /// `epochs`.
    pub epochs: u64,
    /// `S`.
    pub steps_per_epoch: u64,
    /// `T`.
    pub total_steps: u64,
    /// Training samples.
    pub train_samples: u64,
    /// Validation samples (`0` without a validation set).
    pub validation_samples: u64,
    /// Values in the trainable params.
    pub trainable_elements: u64,
    /// Steps completed so far.
    pub step: u64,
    /// Whether training has finished (every step and the last epoch's validation).
    pub finished: bool,
    /// The trainable params, in declaration order.
    pub trainable: Vec<TrainableParam>,
}

impl TrainerInfo {
    /// The JSON form.
    pub fn to_json(&self) -> String {
        json::to_string(self)
    }
}

/// A monotonic clock (training.md §4.2: `step` throttling). Injectable, so tests are
/// deterministic. `elapsed_seconds` is measured with it too.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The time since an arbitrary, fixed origin; never decreases.
    fn now(&self) -> Duration;
}

/// The system's monotonic clock ([`Instant`]).
#[derive(Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now(&self) -> Duration {
        self.origin.elapsed()
    }
}

/// A clock that only moves when told to ([`ManualClock::advance`]), for tests.
#[derive(Debug, Default)]
pub struct ManualClock {
    nanos: AtomicU64,
}

impl ManualClock {
    /// A clock at 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Moves the clock forward by `by`.
    pub fn advance(&self, by: Duration) {
        let nanos = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.nanos.fetch_add(nanos, Ordering::SeqCst);
    }
}

impl Clock for ManualClock {
    fn now(&self) -> Duration {
        Duration::from_nanos(self.nanos.load(Ordering::SeqCst))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome() -> TrainingOutcome {
        TrainingOutcome {
            stop_reason: StopReason::Completed,
            step: 4220,
            epochs_completed: 5,
            final_metrics: Metrics(vec![
                ("train.loss".into(), 0.021),
                ("val.loss".into(), f64::NAN),
            ]),
            best: Some(Best {
                metric: "val.accuracy".into(),
                value: 0.9912,
                epoch: 3,
                step: 3376,
                checkpoint: "runs/mnist/step-000000003376".into(),
            }),
            history: vec![HistoryEntry {
                epoch: 0,
                step: 844,
                metrics: Metrics(vec![("train.loss".into(), f64::INFINITY)]),
            }],
            last_checkpoint: None,
            elapsed_seconds: 812.5,
        }
    }

    #[test]
    fn every_event_kind_has_the_documented_json() {
        let cases = [
            (
                TrainingEvent::Started {
                    device: DeviceId::Cpu,
                    epochs: 5,
                    steps_per_epoch: 844,
                    total_steps: 4220,
                    train_samples: 54000,
                    validation_samples: 6000,
                    trainable_elements: 1_199_882,
                    step: 0,
                },
                r#"{"event":"started","device":{"backend":"cpu"},"epochs":5,"steps_per_epoch":844,"total_steps":4220,"train_samples":54000,"validation_samples":6000,"trainable_elements":1199882,"step":0}"#,
            ),
            (
                TrainingEvent::EpochStarted { epoch: 0, batch: 0 },
                r#"{"event":"epoch_started","epoch":0,"batch":0}"#,
            ),
            (
                TrainingEvent::Step {
                    epoch: 0,
                    batch: 0,
                    step: 0,
                    loss: 2.3051,
                    lr: 1e-5,
                },
                r#"{"event":"step","epoch":0,"batch":0,"step":0,"loss":2.3051,"lr":0.00001}"#,
            ),
            (
                TrainingEvent::Step {
                    epoch: 1,
                    batch: 2,
                    step: 3,
                    loss: f64::NAN,
                    lr: 0.1,
                },
                r#"{"event":"step","epoch":1,"batch":2,"step":3,"loss":"nan","lr":0.1}"#,
            ),
            (
                TrainingEvent::ValidationStarted { epoch: 0 },
                r#"{"event":"validation_started","epoch":0}"#,
            ),
            (
                TrainingEvent::EpochFinished {
                    epoch: 0,
                    step: 844,
                    metrics: Metrics(vec![
                        ("train.loss".into(), 0.2412),
                        ("train.accuracy".into(), 0.9271),
                        ("val.loss".into(), f64::NEG_INFINITY),
                        ("val.accuracy".into(), 0.9779),
                    ]),
                },
                r#"{"event":"epoch_finished","epoch":0,"step":844,"metrics":{"train.loss":0.2412,"train.accuracy":0.9271,"val.loss":"-inf","val.accuracy":0.9779}}"#,
            ),
            (
                TrainingEvent::CheckpointSaved {
                    path: "runs/mnist/step-000000000844".into(),
                    step: 844,
                },
                r#"{"event":"checkpoint_saved","path":"runs/mnist/step-000000000844","step":844}"#,
            ),
            (
                TrainingEvent::CheckpointRemoved {
                    path: "runs/mnist/step-000000000422".into(),
                },
                r#"{"event":"checkpoint_removed","path":"runs/mnist/step-000000000422"}"#,
            ),
        ];
        for (event, expected) in cases {
            assert_eq!(event.to_json(), expected);
            assert!(expected.contains(&format!("\"event\":\"{}\"", event.name())));
        }
        let finished = TrainingEvent::Finished {
            outcome: Box::new(outcome()),
        };
        assert_eq!(
            finished.to_json(),
            r#"{"event":"finished","outcome":{"stop_reason":"completed","step":4220,"epochs_completed":5,"final_metrics":{"train.loss":0.021,"val.loss":"nan"},"best":{"metric":"val.accuracy","value":0.9912,"epoch":3,"step":3376,"checkpoint":"runs/mnist/step-000000003376"},"history":[{"epoch":0,"step":844,"metrics":{"train.loss":"inf"}}],"last_checkpoint":null,"elapsed_seconds":812.5}}"#
        );
    }

    #[test]
    fn manual_clock() {
        let c = ManualClock::new();
        assert_eq!(c.now(), Duration::ZERO);
        c.advance(Duration::from_millis(100));
        assert_eq!(c.now(), Duration::from_millis(100));
        let s = SystemClock::default();
        assert!(s.now() <= s.now());
    }
}
