//! The trainer (`docs/specs/training.md` §1.2, §2.10, §4, §5; public-api.md §1.4).
//!
//! [`Trainer`] is a cheap-to-clone handle (`Arc` inside, `Send + Sync + 'static`, no generics or
//! lifetimes) that the facade (#83) wraps one to one. It is a small state machine:
//!
//! | State | `run` | `export` | `info` |
//! |---|---|---|---|
//! | idle (created, or cancelled) | trains | writes | ✓ |
//! | running | `InvalidInput` | `InvalidInput` | ✓ |
//! | finished (every epoch done) | `InvalidInput` | writes | ✓ |
//! | failed (`run` returned an error) | `InvalidInput` | `InvalidInput` | ✓ |
//! | closed | `InvalidInput` | `InvalidInput` | ✓ |
//!
//! `run` takes the training state out of the handle while it runs, so a callback that calls
//! `run`, `export` or `close` on the same trainer (or another clone) never deadlocks: `run` and
//! `export` find it running, and `close` is recorded and takes effect when the run returns. If a
//! run panics, the handle becomes failed. `close` is idempotent.
//!
//! The device is resolved by the caller: the trainer is given a [`DeviceResolver`] that turns
//! the config's `device` text (`"auto"`, `"cpu"`, `"cuda:0"`, or `ResumeOptions::device`) into a
//! device, at §2.10 step 3, so a device error comes after the config and graph errors. The facade
//! passes the engine's resolution (`"auto"`, the device list and availability, `BackendUnavailable`
//! with `context.device`).

mod run;
mod setup;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use polypous_backend::device::DeviceId;
use polypous_backend::{DynTensor, TensorData};
use polypous_ir::Graph;
use polypous_runtime::{CancellationToken, Plan};

use crate::data::{Batcher, Datasets, InMemoryDataset};
use crate::error::TrainError;
use crate::events::{
    Best, Clock, HistoryEntry, TrainableParam, TrainerInfo, TrainingEvent, TrainingOutcome,
};
use crate::loss::Loss;
use crate::metrics::{Accuracy, LossMetric};
use crate::optimizer::{Optimizer, TensorMap};
use crate::schedule::Schedule;
use crate::training_config::TrainingConfig;

/// Resolves the config's `device` text to a device (§2.2, §2.10 step 3). Errors should be
/// `BackendUnavailable` with `context.device`.
pub type DeviceResolver<'a> = &'a dyn Fn(&str) -> Result<DeviceId, TrainError>;

/// What a new training run needs besides the graph and the config (public-api.md §1.4
/// `TrainerOptions`, with the datasets as host data).
#[derive(Debug, Default)]
pub struct TrainerOptions {
    /// Initial weights: a safetensors file or a sharded checkpoint's index, holding any subset of
    /// the params (§2.10 step 7). Used as given (it is not a config path, so `base_dir` doesn't
    /// apply). `None`: every param uses its `init`.
    pub weights: Option<PathBuf>,
    /// The directory relative config paths are resolved against; `None`: the current directory.
    /// It is made absolute and recorded in every checkpoint.
    pub base_dir: Option<PathBuf>,
    /// The `tensors` datasets (§3.5.1).
    pub datasets: Vec<InMemoryDataset>,
    /// The clock for `step` throttling and `elapsed_seconds`; `None`: the system's monotonic
    /// clock.
    pub clock: Option<Arc<dyn Clock>>,
}

/// What a resume needs besides the checkpoint (public-api.md §1.4 `ResumeOptions`).
#[derive(Debug, Default)]
pub struct ResumeOptions {
    /// A device id text that replaces the config's `device`.
    pub device: Option<String>,
    /// Replaces the checkpoint's `base_dir` for the datasets' relative paths.
    pub base_dir: Option<PathBuf>,
    /// The `tensors` datasets again.
    pub datasets: Vec<InMemoryDataset>,
    /// Delete the run directory's checkpoints newer than the one resumed, when the first new
    /// checkpoint is written (§5.4 step 6); otherwise they are an error.
    pub discard_newer: bool,
    /// As [`TrainerOptions::clock`].
    pub clock: Option<Arc<dyn Clock>>,
}

/// A checkpoint's `keep_best` value with its history entry's epoch and step, if it has one.
type Value = Option<(f64, u64, u64)>;

/// The training state: everything a run reads and changes.
pub(crate) struct Core {
    config: TrainingConfig,
    graph_json: String,
    base_dir: PathBuf,
    run_dir: Option<PathBuf>,
    device: DeviceId,
    plan: Plan,
    datasets: Datasets,
    batcher: Batcher,
    val_batcher: Option<Batcher>,
    loss: Loss,
    /// The training-set accumulators of the current epoch, in list order.
    metrics: Vec<Accuracy>,
    input_fields: Vec<(String, String)>,
    fingerprint: String,
    /// Every declared param, by name.
    params: TensorMap,
    param_order: Vec<String>,
    trainable: Vec<String>,
    trainable_shapes: Vec<(String, Vec<usize>)>,
    optimizer: Optimizer,
    schedule: Schedule,
    steps_per_epoch: u64,
    total_steps: u64,
    step: u64,
    epoch: u64,
    batch: u64,
    train_loss: LossMetric,
    history: Vec<HistoryEntry>,
    /// As in `state.json`: the checkpoint is a directory name.
    best: Option<Best>,
    elapsed_before: f64,
    last_checkpoint: Option<PathBuf>,
    /// `(step, epoch, batch)` of the newest checkpoint: one isn't written twice for a state.
    last_checkpoint_pos: Option<(u64, u64, u64)>,
    /// After a resume with `discard_newer`: checkpoints above this step are deleted with the
    /// first new checkpoint.
    discard_newer_than: Option<u64>,
    /// The `keep_best` value of each checkpoint of the run directory this trainer knows:
    /// `(value, history epoch, history step)`, `None` without one.
    values: BTreeMap<u64, Value>,
    clock: Arc<dyn Clock>,
}

impl Core {
    fn finished(&self) -> bool {
        self.epoch >= self.config.epochs
    }

    fn info(&self) -> TrainerInfo {
        TrainerInfo {
            device: self.device,
            epochs: self.config.epochs,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            train_samples: self.datasets.train.len(),
            validation_samples: self.datasets.validation.as_ref().map_or(0, |v| v.len()),
            trainable_elements: self
                .trainable_shapes
                .iter()
                .map(|(_, s)| s.iter().map(|&d| d as u64).product::<u64>())
                .sum(),
            step: self.step,
            finished: self.finished(),
            trainable: self
                .trainable_shapes
                .iter()
                .map(|(name, shape)| TrainableParam {
                    name: name.clone(),
                    shape: shape.clone(),
                })
                .collect(),
        }
    }

    /// Every declared param as host `f32` data, in declaration order.
    fn host_params(&self) -> Result<Vec<(String, TensorData)>, TrainError> {
        self.param_order
            .iter()
            .map(|name| {
                let t = self
                    .params
                    .get(name)
                    .ok_or_else(|| TrainError::internal(format!("param `{name}` is missing")))?;
                Ok((name.clone(), read_back(name, t)?))
            })
            .collect()
    }

    /// The optimizer state as host data, named `<param>:<state>`, sorted by name.
    fn host_optimizer_state(&self) -> Result<Vec<(String, TensorData)>, TrainError> {
        let mut out: Vec<(String, TensorData)> = self
            .optimizer
            .export_state()
            .into_iter()
            .map(|s| {
                let name = format!("{}:{}", s.param, s.state);
                let data = read_back(&name, &s.tensor)?;
                Ok((name, data))
            })
            .collect::<Result<_, TrainError>>()?;
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }
}

fn read_back(name: &str, t: &DynTensor) -> Result<TensorData, TrainError> {
    t.to_data()
        .map_err(|e| TrainError::from(e).with_tensor(name))
}

enum State {
    Idle(Box<Core>),
    Running,
    Finished(Box<Core>),
    Failed,
    Closed,
}

struct Shared {
    state: Mutex<State>,
    /// `info()` while running: the static part, with `step` and `finished` from the atomics.
    info: Mutex<TrainerInfo>,
    step: AtomicU64,
    finished: AtomicBool,
    close_requested: AtomicBool,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        // A panic while the lock is held leaves a consistent `State` value (every change is one
        // assignment), so a poisoned lock is still usable.
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// A training run (public-api.md §1.4): the param store, the optimizer state and the datasets
/// of one graph and config. See the module documentation for its states.
#[derive(Clone)]
pub struct Trainer {
    shared: Arc<Shared>,
}

impl std::fmt::Debug for Trainer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer").finish_non_exhaustive()
    }
}

fn state_error(state: &State, what: &str) -> TrainError {
    let why = match state {
        State::Running => "the trainer is running (another call, or its own callback)",
        State::Finished(_) => "training has finished",
        State::Failed => {
            "an earlier run failed, so the trainer's state is not reliable; resume from the last checkpoint"
        }
        State::Closed => "the trainer was closed",
        State::Idle(_) => "the trainer is idle",
    };
    TrainError::invalid(format!("cannot {what}: {why}"))
}

/// Sets the state to failed if a run unwinds (a panic) before it stored its result.
struct RunGuard<'a> {
    shared: &'a Shared,
    done: bool,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            *self.shared.lock() = State::Failed;
        }
    }
}

impl Trainer {
    fn from_core(core: Core) -> Self {
        let info = core.info();
        let finished = core.finished();
        let state = if finished {
            State::Finished(Box::new(core))
        } else {
            State::Idle(Box::new(core))
        };
        Self {
            shared: Arc::new(Shared {
                step: AtomicU64::new(info.step),
                finished: AtomicBool::new(finished),
                info: Mutex::new(info),
                state: Mutex::new(state),
                close_requested: AtomicBool::new(false),
            }),
        }
    }

    /// Creates a trainer for a new run: the checks of training.md §2.10 in order (config, graph,
    /// device through `resolve`, references, data, shapes, params, run directory), then the
    /// initial values are computed and every param is uploaded. Nothing trains yet. `cancel` is
    /// checked between the steps and between dataset files.
    ///
    /// # Errors
    /// The first failing check (training.md §6); `Cancelled` if `cancel` was set.
    pub fn new(
        graph: &Graph,
        config: TrainingConfig,
        options: TrainerOptions,
        resolve: DeviceResolver<'_>,
        cancel: &CancellationToken,
    ) -> Result<Self, TrainError> {
        setup::new_core(graph, config, options, resolve, cancel).map(Self::from_core)
    }

    /// A trainer restored from a checkpoint directory, or the newest checkpoint of a run
    /// directory (training.md §5.4). The checkpoint is untrusted: every check runs before the data
    /// it protects is read.
    ///
    /// # Errors
    /// `InvalidInput` with `context.file` for an invalid or inconsistent checkpoint or changed
    /// data; `Io`; the errors of §2.10 steps 3–6; `Cancelled`.
    pub fn resume(
        path: impl AsRef<Path>,
        options: ResumeOptions,
        resolve: DeviceResolver<'_>,
        cancel: &CancellationToken,
    ) -> Result<Self, TrainError> {
        setup::resume_core(path.as_ref(), options, resolve, cancel).map(Self::from_core)
    }

    /// The `started` event's fields with `step` the steps completed so far, `finished`, and the
    /// trainable params. Works in every state.
    pub fn info(&self) -> TrainerInfo {
        let mut info = self
            .shared
            .info
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        info.step = self.shared.step.load(Ordering::SeqCst);
        info.finished = self.shared.finished.load(Ordering::SeqCst);
        info
    }

    /// Trains from the current step to the end, or until `cancel` is set (training.md §1.2,
    /// §4). Returns `Ok` when completed and when cancelled (`stop_reason: "cancelled"`); calling
    /// `run` again continues a cancelled run. Events go to `on_event`, on the calling thread, in
    /// the order of §4.2; `finished` is the last one of an `Ok` run.
    ///
    /// # Errors
    /// `InvalidInput` if the trainer is running, finished, failed or closed. An error during the
    /// run (§6) leaves the trainer failed.
    pub fn run(
        &self,
        cancel: &CancellationToken,
        mut on_event: impl FnMut(TrainingEvent),
    ) -> Result<TrainingOutcome, TrainError> {
        let mut core = {
            let mut state = self.shared.lock();
            match std::mem::replace(&mut *state, State::Running) {
                State::Idle(core) => core,
                other => {
                    let err = state_error(&other, "run");
                    *state = other;
                    return Err(err);
                }
            }
        };
        let mut guard = RunGuard {
            shared: &self.shared,
            done: false,
        };
        let result = core.run(cancel, &mut on_event, &self.shared.step);
        self.shared
            .finished
            .store(core.finished(), Ordering::SeqCst);
        let next = match &result {
            _ if self.shared.close_requested.load(Ordering::SeqCst) => State::Closed,
            Ok(_) if core.finished() => State::Finished(core),
            Ok(_) => State::Idle(core),
            Err(_) => State::Failed,
        };
        *self.shared.lock() = next;
        guard.done = true;
        result
    }

    /// Writes the current model (training.md §5.5): the graph's canonical JSON to `graph_path`
    /// and every declared param, running statistics included, as `F32` to `weights_path`, each
    /// through a temporary file, `fsync` and a rename (the weights first).
    ///
    /// # Errors
    /// `InvalidInput` while running, after a failed run or after `close`; `Io` with
    /// `context.file`; `Backend`.
    pub fn export(
        &self,
        graph_path: impl AsRef<Path>,
        weights_path: impl AsRef<Path>,
    ) -> Result<(), TrainError> {
        let state = self.shared.lock();
        match &*state {
            State::Idle(core) | State::Finished(core) => {
                run::export(core, graph_path.as_ref(), weights_path.as_ref())
            }
            other => Err(state_error(other, "export")),
        }
    }

    /// Frees the params, the optimizer state and the datasets now. A running `run` finishes
    /// first. Later `run` and `export` calls on any clone are `InvalidInput`; `info` still works.
    /// Idempotent.
    pub fn close(&self) {
        let mut state = self.shared.lock();
        match &*state {
            State::Running => self.shared.close_requested.store(true, Ordering::SeqCst),
            _ => *state = State::Closed,
        }
    }

    /// Every declared param as host `f32` data, in declaration order (for tests and tools).
    ///
    /// # Errors
    /// `InvalidInput` unless the trainer is idle or finished; `Backend`.
    pub fn params(&self) -> Result<Vec<(String, TensorData)>, TrainError> {
        self.with_core("read the params", Core::host_params)
    }

    /// The optimizer state as host data, `<param>:<state>`, sorted by name.
    ///
    /// # Errors
    /// As [`Trainer::params`].
    pub fn optimizer_state(&self) -> Result<Vec<(String, TensorData)>, TrainError> {
        self.with_core("read the optimizer state", Core::host_optimizer_state)
    }

    /// The finished epochs and the best checkpoint (as in `state.json`).
    ///
    /// # Errors
    /// As [`Trainer::params`].
    pub fn history(&self) -> Result<(Vec<HistoryEntry>, Option<Best>), TrainError> {
        self.with_core("read the history", |c| {
            Ok((c.history.clone(), c.best.clone()))
        })
    }

    /// The config (with `ResumeOptions::device` applied after a resume).
    ///
    /// # Errors
    /// As [`Trainer::params`].
    pub fn config(&self) -> Result<TrainingConfig, TrainError> {
        self.with_core("read the config", |c| Ok(c.config.clone()))
    }

    fn with_core<T>(
        &self,
        what: &str,
        f: impl FnOnce(&Core) -> Result<T, TrainError>,
    ) -> Result<T, TrainError> {
        let state = self.shared.lock();
        match &*state {
            State::Idle(core) | State::Finished(core) => f(core),
            other => Err(state_error(other, what)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_handle_is_send_sync_static() {
        fn check<T: Clone + Send + Sync + 'static>() {}
        check::<Trainer>();
    }
}
