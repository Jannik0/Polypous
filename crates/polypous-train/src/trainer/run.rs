//! The training loop (`docs/specs/training.md` §1.2), events and cancellation (§4), writing
//! checkpoints with `keep_last`/`keep_best` (§5.1, §5.3) and export (§5.5).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use polypous_backend::{DType, DynTensor, TensorData};
use polypous_runtime::{CancellationToken, Params, RunMode, RunOptions, RunOutput};

use super::Core;
use crate::checkpoint::{self, Contents, EpochSums, HostTensor, State};
use crate::clip::clip_gradients;
use crate::config::MetricConfig;
use crate::error::TrainError;
use crate::events::{Best, HistoryEntry, Metrics, StopReason, TrainingEvent, TrainingOutcome};
use crate::metrics::{Accuracy, LossMetric};
use crate::optimizer::TensorMap;

/// `step` events are at least this far apart, except the first of a run and the last of an
/// epoch (§4.2).
const STEP_EVENT_INTERVAL: Duration = Duration::from_millis(100);

/// One run's context: the callback, the token and the throttle.
struct Ctx<'a> {
    cancel: &'a CancellationToken,
    on_event: &'a mut dyn FnMut(TrainingEvent),
    step_counter: &'a AtomicU64,
    run_start: Duration,
    last_step_event: Option<Duration>,
}

impl Ctx<'_> {
    fn emit(&mut self, event: TrainingEvent) {
        (self.on_event)(event);
    }
}

fn f32_value(t: &DynTensor) -> Result<f32, TrainError> {
    let data = t.to_data()?;
    match data.bytes.as_slice() {
        [a, b, c, d] if data.dtype == DType::F32 => Ok(f32::from_le_bytes([*a, *b, *c, *d])),
        _ => Err(TrainError::internal(format!(
            "the loss is {} {:?}, not one f32",
            data.dtype, data.shape
        ))),
    }
}

fn batch_len(fields: &BTreeMap<String, TensorData>) -> usize {
    fields
        .values()
        .next()
        .and_then(|t| t.shape.first())
        .map_or(0, |&n| usize::try_from(n).unwrap_or(0))
}

fn field<'a>(
    fields: &'a BTreeMap<String, TensorData>,
    name: &str,
) -> Result<&'a TensorData, TrainError> {
    fields
        .get(name)
        .ok_or_else(|| TrainError::internal(format!("field `{name}` is not in the batch")))
}

fn output<'a>(out: &'a RunOutput, name: &str) -> Result<&'a DynTensor, TrainError> {
    out.output(name)
        .ok_or_else(|| TrainError::internal(format!("graph output `{name}` is missing")))
}

impl Core {
    fn elapsed(&self, ctx: &Ctx<'_>) -> f64 {
        self.elapsed_before + (self.clock.now().saturating_sub(ctx.run_start)).as_secs_f64()
    }

    /// The graph inputs of a batch, uploaded (a field feeding several inputs is uploaded once).
    fn upload_inputs(
        &self,
        fields: &BTreeMap<String, TensorData>,
    ) -> Result<Vec<(String, DynTensor)>, TrainError> {
        let mut uploaded: BTreeMap<&str, DynTensor> = BTreeMap::new();
        let mut inputs = Vec::with_capacity(self.input_fields.len());
        for (input, name) in &self.input_fields {
            let t = match uploaded.get(name.as_str()) {
                Some(t) => t.clone(),
                None => {
                    let t = DynTensor::from_data(field(fields, name)?, self.device)
                        .map_err(|e| TrainError::from(e).with_tensor(name.clone()))?;
                    uploaded.insert(name, t.clone());
                    t
                }
            };
            inputs.push((input.clone(), t));
        }
        Ok(inputs)
    }

    fn metric_configs(&self) -> impl Iterator<Item = (&str, &str, &str)> + '_ {
        self.config.metrics.iter().map(|m| {
            let MetricConfig::Accuracy(c) = m;
            (c.name.as_str(), c.prediction.as_str(), c.target.as_str())
        })
    }

    /// Optimizer step `k` (§1.2 steps 1–8, without the event, the token and checkpoints).
    /// Returns the batch loss and `lr(k)`.
    fn train_step(&mut self, order: &[u32]) -> Result<(f32, f64), TrainError> {
        let k = self.step;
        // 1. Batch.
        let fields = self
            .batcher
            .batch(&self.datasets.train, order, self.batch)?;
        let samples = batch_len(&fields);
        let inputs = self.upload_inputs(&fields)?;
        let target_name = self.config.loss.target().to_owned();
        let target = DynTensor::from_data(field(&fields, &target_name)?, self.device)?;
        // 2. Forward, with every trainable param as a leaf.
        let mut given = Params::with_capacity(self.params.len());
        let mut leaves = Vec::with_capacity(self.trainable.len());
        for (name, t) in &self.params {
            if self.trainable.binary_search(name).is_ok() {
                let leaf = t.clone().require_grad()?;
                given.insert(name.clone(), leaf.clone());
                leaves.push((name.clone(), leaf));
            } else {
                given.insert(name.clone(), t.clone());
            }
        }
        let mode = RunMode::Training {
            seed: self.config.seed,
            step: k,
        };
        let out = self
            .plan
            .run(inputs, Vec::new(), &given, mode, &RunOptions::default())?;
        drop(given);
        // 3. Running statistics, right away.
        for (name, t) in &out.running_stats {
            match self.params.get_mut(name) {
                Some(slot) => *slot = t.clone(),
                None => {
                    return Err(TrainError::internal(format!(
                        "running statistic `{name}` is not a param"
                    )));
                }
            }
        }
        // 4. Loss and training metrics.
        let loss = self
            .loss
            .compute(output(&out, self.config.loss.prediction())?, &target)?;
        let value = f32_value(&loss)?;
        self.train_loss.add(value, samples);
        let metric_io: Vec<(String, String)> = self
            .metric_configs()
            .map(|(_, p, t)| (p.to_owned(), t.to_owned()))
            .collect();
        for (acc, (pred, tgt)) in self.metrics.iter_mut().zip(&metric_io) {
            acc.update(output(&out, pred)?, field(&fields, tgt)?)?;
        }
        // 5. Backward, one gradient per trainable param that the pass reaches.
        let mut grads = loss.backward()?;
        let mut gradients = TensorMap::new();
        for (name, leaf) in &leaves {
            if let Some(g) = grads.take(leaf)? {
                gradients.insert(name.clone(), g);
            }
        }
        drop(grads);
        drop(loss);
        drop(out);
        drop(leaves);
        drop(target);
        // 6. Clip.
        if let Some(clip) = &self.config.gradient_clip {
            clip_gradients(clip, &mut gradients)?;
        }
        // 7. Update.
        let lr = self.schedule.lr(k)?;
        self.optimizer.step(k, lr, &mut self.params, &gradients)?;
        // 8. k ← k + 1.
        self.step += 1;
        self.batch += 1;
        Ok((value, lr))
    }

    /// One validation batch in inference mode: adds to `loss` and `accs`.
    fn validation_batch(
        &self,
        order: &[u32],
        batch: u64,
        loss_metric: &mut LossMetric,
        accs: &mut [Accuracy],
    ) -> Result<(), TrainError> {
        let (Some(batcher), Some(set)) = (&self.val_batcher, &self.datasets.validation) else {
            return Err(TrainError::internal("no validation set".to_owned()));
        };
        let fields = batcher.batch(set, order, batch)?;
        let samples = batch_len(&fields);
        let inputs = self.upload_inputs(&fields)?;
        let target = DynTensor::from_data(field(&fields, self.config.loss.target())?, self.device)?;
        let given: Params = self
            .params
            .iter()
            .map(|(n, t)| (n.clone(), t.clone()))
            .collect();
        let out = self.plan.run(
            inputs,
            Vec::new(),
            &given,
            RunMode::Inference,
            &RunOptions::default(),
        )?;
        let loss = self
            .loss
            .compute(output(&out, self.config.loss.prediction())?, &target)?;
        loss_metric.add(f32_value(&loss)?, samples);
        for (acc, (_, pred, tgt)) in accs.iter_mut().zip(self.metric_configs()) {
            acc.update(output(&out, pred)?, field(&fields, tgt)?)?;
        }
        Ok(())
    }

    /// Trains until the end or until `cancel` is set (§1.2, §4.2, §4.3).
    pub(super) fn run(
        &mut self,
        cancel: &CancellationToken,
        on_event: &mut dyn FnMut(TrainingEvent),
        step_counter: &AtomicU64,
    ) -> Result<TrainingOutcome, TrainError> {
        let mut ctx = Ctx {
            cancel,
            on_event,
            step_counter,
            run_start: self.clock.now(),
            last_step_event: None,
        };
        let result = self.run_inner(&mut ctx);
        // The time of this run counts also when it failed or was cancelled.
        self.elapsed_before = self.elapsed(&ctx);
        result
    }

    fn run_inner(&mut self, ctx: &mut Ctx<'_>) -> Result<TrainingOutcome, TrainError> {
        let info = self.info();
        ctx.emit(TrainingEvent::Started {
            device: info.device,
            epochs: info.epochs,
            steps_per_epoch: info.steps_per_epoch,
            total_steps: info.total_steps,
            train_samples: info.train_samples,
            validation_samples: info.validation_samples,
            trainable_elements: info.trainable_elements,
            step: self.step,
        });
        if ctx.cancel.is_cancelled() {
            return self.stop_cancelled(ctx);
        }
        let s = self.steps_per_epoch;
        let mut first_step = true;
        while self.epoch < self.config.epochs {
            ctx.emit(TrainingEvent::EpochStarted {
                epoch: self.epoch,
                batch: self.batch,
            });
            if self.batch < s {
                let order = self.batcher.order(self.epoch)?;
                while self.batch < s {
                    let (epoch, batch, k) = (self.epoch, self.batch, self.step);
                    let (loss, lr) = self.train_step(&order)?;
                    ctx.step_counter.store(self.step, Ordering::SeqCst);
                    let now = self.clock.now();
                    let last_of_epoch = self.batch == s;
                    let due = ctx
                        .last_step_event
                        .is_none_or(|t| now.saturating_sub(t) >= STEP_EVENT_INTERVAL);
                    if first_step || last_of_epoch || due {
                        ctx.last_step_event = Some(now);
                        ctx.emit(TrainingEvent::Step {
                            epoch,
                            batch,
                            step: k,
                            loss: f64::from(loss),
                            lr,
                        });
                    }
                    first_step = false;
                    if ctx.cancel.is_cancelled() {
                        return self.stop_cancelled(ctx);
                    }
                    let by_step = self
                        .config
                        .checkpoint
                        .as_ref()
                        .and_then(|c| c.every_steps)
                        .is_some_and(|n| self.step.is_multiple_of(n));
                    if by_step && !last_of_epoch {
                        // The token was checked right above, before this checkpoint.
                        self.checkpoint(ctx)?;
                    }
                }
            }
            // The validation pass.
            let mut val = None;
            if let Some(batcher) = self.val_batcher {
                ctx.emit(TrainingEvent::ValidationStarted { epoch: self.epoch });
                let order = batcher.order(0)?;
                let mut loss = LossMetric::new();
                let mut accs: Vec<Accuracy> = self
                    .config
                    .metrics
                    .iter()
                    .map(|m| {
                        let MetricConfig::Accuracy(c) = m;
                        Accuracy::new(c).map_err(TrainError::from)
                    })
                    .collect::<Result<_, _>>()?;
                for b in 0..batcher.steps_per_epoch() {
                    if ctx.cancel.is_cancelled() {
                        return self.stop_cancelled(ctx);
                    }
                    self.validation_batch(&order, b, &mut loss, &mut accs)?;
                }
                val = Some((loss, accs));
            }
            // The epoch's metrics.
            let finished_epoch = self.epoch;
            let metrics = self.epoch_metrics(val.as_ref())?;
            self.history.push(HistoryEntry {
                epoch: finished_epoch,
                step: self.step,
                metrics: metrics.clone(),
            });
            self.epoch += 1;
            self.batch = 0;
            self.train_loss.reset();
            for acc in &mut self.metrics {
                acc.reset();
            }
            ctx.emit(TrainingEvent::EpochFinished {
                epoch: finished_epoch,
                step: self.step,
                metrics,
            });
            if let Some(c) = &self.config.checkpoint {
                let by_epoch = c
                    .every_epochs
                    .is_some_and(|n| (finished_epoch + 1).is_multiple_of(n));
                let by_step = c.every_steps.is_some_and(|n| self.step.is_multiple_of(n));
                let last = self.epoch == self.config.epochs;
                if by_epoch || by_step || last {
                    if ctx.cancel.is_cancelled() {
                        return self.stop_cancelled(ctx);
                    }
                    self.checkpoint(ctx)?;
                }
            }
        }
        let outcome = self.outcome(StopReason::Completed, ctx);
        ctx.emit(TrainingEvent::Finished {
            outcome: Box::new(outcome.clone()),
        });
        Ok(outcome)
    }

    /// `train.*` then `val.*` (§2.7).
    fn epoch_metrics(
        &self,
        val: Option<&(LossMetric, Vec<Accuracy>)>,
    ) -> Result<Metrics, TrainError> {
        // Every set has at least one sample, so every value exists (events.rs).
        let value = |v: Option<f64>, name: &str| {
            v.ok_or_else(|| TrainError::internal(format!("metric `{name}` has no samples")))
        };
        let mut out = Vec::new();
        out.push((
            "train.loss".to_owned(),
            value(self.train_loss.value(), "train.loss")?,
        ));
        for (acc, m) in self.metrics.iter().zip(&self.config.metrics) {
            let name = format!("train.{}", m.name());
            out.push((name.clone(), value(acc.value(), &name)?));
        }
        if let Some((loss, accs)) = val {
            out.push(("val.loss".to_owned(), value(loss.value(), "val.loss")?));
            for (acc, m) in accs.iter().zip(&self.config.metrics) {
                let name = format!("val.{}", m.name());
                out.push((name.clone(), value(acc.value(), &name)?));
            }
        }
        Ok(Metrics(out))
    }

    fn stop_cancelled(&mut self, ctx: &mut Ctx<'_>) -> Result<TrainingOutcome, TrainError> {
        let write = self.config.checkpoint.as_ref().is_some_and(|c| c.on_cancel)
            && self.last_checkpoint_pos != Some((self.step, self.epoch, self.batch));
        if write {
            self.checkpoint(ctx)?;
        }
        let outcome = self.outcome(StopReason::Cancelled, ctx);
        ctx.emit(TrainingEvent::Finished {
            outcome: Box::new(outcome.clone()),
        });
        Ok(outcome)
    }

    fn outcome(&self, stop_reason: StopReason, ctx: &Ctx<'_>) -> TrainingOutcome {
        let path = |name: &str| {
            self.run_dir
                .as_ref()
                .map_or_else(|| PathBuf::from(name), |d| d.join(name))
                .to_string_lossy()
                .into_owned()
        };
        TrainingOutcome {
            stop_reason,
            step: self.step,
            epochs_completed: self.history.len() as u64,
            final_metrics: self
                .history
                .last()
                .map(|h| h.metrics.clone())
                .unwrap_or_default(),
            best: self.best.as_ref().map(|b| Best {
                checkpoint: path(&b.checkpoint),
                ..b.clone()
            }),
            history: self.history.clone(),
            last_checkpoint: self
                .last_checkpoint
                .as_ref()
                .map(|p| p.to_string_lossy().into_owned()),
            elapsed_seconds: self.elapsed(ctx),
        }
    }

    fn state(&self, best: Option<Best>, elapsed: f64) -> State {
        State {
            config: self.config.clone(),
            base_dir: self.base_dir.to_string_lossy().into_owned(),
            step: self.step,
            epoch: self.epoch,
            batch: self.batch,
            steps_per_epoch: self.steps_per_epoch,
            train_samples: self.datasets.train.len(),
            validation_samples: self.datasets.validation.as_ref().map_or(0, |v| v.len()),
            data_fingerprint: self.fingerprint.clone(),
            sums: EpochSums {
                loss_sum: self.train_loss.weighted_sum(),
                samples: self.train_loss.samples(),
                accuracy: self
                    .metrics
                    .iter()
                    .zip(&self.config.metrics)
                    .map(|(a, m)| (m.name().to_owned(), a.correct(), a.total()))
                    .collect(),
            },
            history: self.history.clone(),
            best,
            elapsed_seconds: elapsed,
        }
    }

    /// The `keep_best` value of this trainer's current state: its last history entry's.
    fn own_value(&self, metric: &str) -> Option<(f64, u64, u64)> {
        let h = self.history.last()?;
        Some((h.metrics.get(metric)?, h.epoch, h.step))
    }

    /// The best checkpoint among `existing` and the one about to be written (§5.3): the best
    /// value of `metric`, NaN never, the earlier one on a tie.
    fn best_among(&mut self, metric: &str, existing: &[checkpoint::Entry]) -> Option<Best> {
        let mut candidates: Vec<(u64, super::Value)> = Vec::new();
        for e in existing {
            let v = match self.values.get(&e.step) {
                Some(v) => *v,
                None => {
                    let v = checkpoint::last_value(&e.path, metric);
                    self.values.insert(e.step, v);
                    v
                }
            };
            candidates.push((e.step, v));
        }
        candidates.push((self.step, self.own_value(metric)));
        candidates.sort_by_key(|c| c.0);
        let mut best: Option<(u64, (f64, u64, u64))> = None;
        for (step, v) in candidates {
            if let Some(v) = v
                && checkpoint::better(metric, v.0, best.map(|b| b.1.0))
            {
                best = Some((step, v));
            }
        }
        best.map(|(step, (value, epoch, hstep))| Best {
            metric: metric.to_owned(),
            value,
            epoch,
            step: hstep,
            checkpoint: checkpoint::checkpoint_name(step),
        })
    }

    fn host_tensors(&self) -> Result<(Vec<HostTensor>, Vec<HostTensor>), TrainError> {
        let mut weights = Vec::with_capacity(self.params.len());
        for (name, data) in self.host_params()? {
            weights.push(host_tensor(name, data)?);
        }
        let mut optimizer = Vec::new();
        for (name, data) in self.host_optimizer_state()? {
            optimizer.push(host_tensor(name, data)?);
        }
        Ok((weights, optimizer))
    }

    /// Writes a checkpoint of the current state (§5.1) and applies `keep_last`/`keep_best`
    /// (§5.3), with the events.
    fn checkpoint(&mut self, ctx: &mut Ctx<'_>) -> Result<(), TrainError> {
        let (Some(config), Some(run_dir)) = (self.config.checkpoint.clone(), self.run_dir.clone())
        else {
            return Ok(());
        };
        let mut existing = checkpoint::checkpoints(&run_dir)?;
        let discard: Vec<checkpoint::Entry> = match self.discard_newer_than {
            Some(limit) => existing
                .iter()
                .filter(|e| e.step > limit && e.step != self.step)
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        existing.retain(|e| e.step != self.step && !discard.contains(e));
        let best = match &config.keep_best {
            Some(metric) => self.best_among(metric, &existing),
            None => None,
        };
        let state = self.state(best.clone(), self.elapsed(ctx));
        let (weights, optimizer) = self.host_tensors()?;
        let path = checkpoint::write(
            &run_dir,
            self.step,
            &Contents {
                state: state.to_json(),
                graph: &self.graph_json,
                weights: &weights,
                optimizer: &optimizer,
            },
        )?;
        drop((weights, optimizer));
        let own = config.keep_best.as_deref().and_then(|m| self.own_value(m));
        self.values.insert(self.step, own);
        self.best = best;
        self.last_checkpoint = Some(path.clone());
        self.last_checkpoint_pos = Some((self.step, self.epoch, self.batch));
        tracing::info!(path = %path.display(), step = self.step, "wrote a checkpoint");
        ctx.emit(TrainingEvent::CheckpointSaved {
            path: path.to_string_lossy().into_owned(),
            step: self.step,
        });
        // The continuation that this resume replaces (§5.4 step 6).
        for e in discard {
            self.remove(&e, ctx);
        }
        self.discard_newer_than = None;
        // keep_last: all but the newest `n` and the best, oldest first.
        if let Some(n) = config.keep_last {
            let all = match checkpoint::checkpoints(&run_dir) {
                Ok(all) => all,
                Err(e) => {
                    tracing::warn!(error = %e, "can't list the run directory for keep_last");
                    return Ok(());
                }
            };
            let best_step = self
                .best
                .as_ref()
                .and_then(|b| checkpoint::parse_checkpoint_name(&b.checkpoint));
            let keep_from = all
                .len()
                .saturating_sub(usize::try_from(n).unwrap_or(usize::MAX));
            for (i, e) in all.iter().enumerate() {
                if i < keep_from && Some(e.step) != best_step {
                    self.remove(e, ctx);
                }
            }
        }
        Ok(())
    }

    /// Deletes a checkpoint; a failure is logged at `warn` (§5.3).
    fn remove(&mut self, e: &checkpoint::Entry, ctx: &mut Ctx<'_>) {
        match std::fs::remove_dir_all(&e.path) {
            Ok(()) => {
                self.values.remove(&e.step);
                ctx.emit(TrainingEvent::CheckpointRemoved {
                    path: e.path.to_string_lossy().into_owned(),
                });
            }
            Err(err) => {
                tracing::warn!(path = %e.path.display(), error = %err, "can't delete a checkpoint");
            }
        }
    }
}

fn host_tensor(name: String, data: TensorData) -> Result<HostTensor, TrainError> {
    if data.dtype != DType::F32 {
        return Err(TrainError::internal(format!(
            "`{name}` is {}, not f32",
            data.dtype
        )));
    }
    let shape = data
        .shape
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| TrainError::internal(format!("`{name}` is too large")))?;
    Ok((name, shape, data.bytes))
}

/// Writes `bytes` to `path` through a temporary sibling, `fsync` and a rename.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), TrainError> {
    let name = path.file_name().ok_or_else(|| {
        TrainError::invalid(format!("{} is not a file path", path.display())).with_file(path)
    })?;
    let mut tmp_name = std::ffi::OsString::from(".");
    tmp_name.push(name);
    tmp_name.push(format!(".{}.tmp", std::process::id()));
    let tmp = path.with_file_name(tmp_name);
    let _ = std::fs::remove_file(&tmp);
    let result = checkpoint::write_synced(&tmp, bytes)
        .and_then(|()| std::fs::rename(&tmp, path).map_err(|e| TrainError::io(&e, "write", path)));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// §5.5: the weights, then the graph.
pub(super) fn export(
    core: &Core,
    graph_path: &Path,
    weights_path: &Path,
) -> Result<(), TrainError> {
    let mut weights = Vec::with_capacity(core.params.len());
    for (name, data) in core.host_params()? {
        weights.push(host_tensor(name, data)?);
    }
    checkpoint::write_tensors(weights_path, &weights)?;
    write_atomic(graph_path, core.graph_json.as_bytes())?;
    tracing::info!(
        graph = %graph_path.display(),
        weights = %weights_path.display(),
        "exported the trained model"
    );
    Ok(())
}
