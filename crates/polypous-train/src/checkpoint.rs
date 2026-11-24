//! Checkpoints (`docs/specs/training.md` §5.1–§5.4): the run directory and its entry names,
//! `state.json`, writing a checkpoint atomically, and loading an untrusted one.
//!
//! **Names.** A checkpoint is the directory `step-` + its step as decimal zero-padded to 12
//! digits (wider if needed). Only that canonical form is a checkpoint: `step-5` is someone else's
//! entry and is never touched. The temporary directory of a checkpoint being written is
//! `.tmp-step-` + the same digits. These two kinds of entries are the only ones the trainer
//! creates, replaces or deletes.
//!
//! **Writing** (§5.1): delete leftover `.tmp-step-*` directories; write the four files into
//! `.tmp-step-N`, each flushed to disk (`fsync`); flush the directory itself; delete an existing
//! `step-N`; rename; flush the run directory. Flushing a directory is a Unix operation: on Windows
//! the standard library can't open a directory for `sync_all`, so the two directory flushes are
//! compiled only on Unix (the files themselves are flushed everywhere).
//!
//! **Loading** (§5.4) treats every file as untrusted. Each check runs before the data it protects
//! is read or allocated: `state.json` is at most 64 MiB, read with a bounded reader and parsed
//! strictly; `graph.json` is at most the graph limit of graph-ir.md §4.3 and validated; the two
//! safetensors headers are validated by `polypous-formats` (offsets inside the file) and checked
//! against the graph (names, `F32`, shapes, byte ranges) before any tensor is copied, and every
//! copy is a fallible reservation of a size the header proved to be inside the file.

use std::collections::BTreeMap;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use polypous_formats::safetensors::{
    self, Dtype as StorageDtype, Error as FormatError, ErrorKind as FormatErrorKind,
    SafetensorsFile, WriteTensor,
};
use polypous_ir::{Analysis, Graph, ValidateOptions};
use serde::Serialize;
use serde::ser::{SerializeMap, Serializer};

use crate::error::{TrainError, TrainErrorKind};
use crate::events::{Best, HistoryEntry, Metrics};
use crate::json::{self, Json, Number, Obj, Read, as_array, as_f64_or_special, as_str};
use crate::optimizer::Optimizer;
use crate::training_config::TrainingConfig;

/// `format` of `state.json`.
pub const STATE_FORMAT: &str = "polypous.checkpoint";
/// The only supported `format_version` of `state.json`.
pub const STATE_FORMAT_VERSION: u64 = 1;
/// The largest `state.json`: 64 MiB (§5.4 step 1).
pub const MAX_STATE_BYTES: u64 = 64 << 20;

/// The training state file of a checkpoint.
pub const STATE_FILE: &str = "state.json";
/// The graph of a checkpoint (canonical JSON).
pub const GRAPH_FILE: &str = "graph.json";
/// Every declared param, `F32`.
pub const WEIGHTS_FILE: &str = "weights.safetensors";
/// The optimizer state, `F32`, `<param>:<state>`.
pub const OPTIMIZER_FILE: &str = "optimizer.safetensors";

/// The directory name of the checkpoint after `step` optimizer steps: `step-000000000844`.
pub fn checkpoint_name(step: u64) -> String {
    format!("step-{step:012}")
}

/// The step of a checkpoint directory name, if `name` is a canonical checkpoint name
/// ([`checkpoint_name`]).
pub fn parse_checkpoint_name(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("step-")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let step = digits.parse::<u64>().ok()?;
    (checkpoint_name(step) == name).then_some(step)
}

fn is_tmp_name(name: &str) -> bool {
    name.strip_prefix('.')
        .and_then(|rest| rest.strip_prefix("tmp-"))
        .and_then(parse_checkpoint_name)
        .is_some()
}

/// A checkpoint entry of a run directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Entry {
    pub step: u64,
    pub path: PathBuf,
    pub is_dir: bool,
}

/// Every entry of `dir` with a canonical checkpoint name, sorted by step. Entries whose names
/// aren't UTF-8 are not checkpoints.
pub(crate) fn list(dir: &Path) -> Result<Vec<Entry>, TrainError> {
    let read = std::fs::read_dir(dir).map_err(|e| TrainError::io(&e, "list", dir))?;
    let mut out = Vec::new();
    for entry in read {
        let entry = entry.map_err(|e| TrainError::io(&e, "list", dir))?;
        let name = entry.file_name();
        let Some(step) = name.to_str().and_then(parse_checkpoint_name) else {
            continue;
        };
        let path = entry.path();
        // Follows symbolic links, like the dataset readers.
        let is_dir = std::fs::metadata(&path).is_ok_and(|m| m.is_dir());
        out.push(Entry { step, path, is_dir });
    }
    out.sort_by_key(|e| e.step);
    Ok(out)
}

/// The checkpoints (directories) of `dir`, sorted by step.
pub(crate) fn checkpoints(dir: &Path) -> Result<Vec<Entry>, TrainError> {
    let mut all = list(dir)?;
    all.retain(|e| e.is_dir);
    Ok(all)
}

/// The training metric sums of the current epoch (§5.2 `epoch_sums`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EpochSums {
    pub loss_sum: f64,
    pub samples: u64,
    /// Per metric, in list order: (name, correct, counted).
    pub accuracy: Vec<(String, u64, u64)>,
}

/// `state.json` (§5.2).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct State {
    pub config: TrainingConfig,
    pub base_dir: String,
    pub step: u64,
    pub epoch: u64,
    pub batch: u64,
    pub steps_per_epoch: u64,
    pub train_samples: u64,
    pub validation_samples: u64,
    pub data_fingerprint: String,
    pub sums: EpochSums,
    pub history: Vec<HistoryEntry>,
    pub best: Option<Best>,
    pub elapsed_seconds: f64,
}

struct Counts<'a>(&'a [(String, u64, u64)], bool);

impl Serialize for Counts<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut map = s.serialize_map(Some(self.0.len()))?;
        for (name, correct, counted) in self.0 {
            map.serialize_entry(name, if self.1 { correct } else { counted })?;
        }
        map.end()
    }
}

#[derive(Serialize)]
struct SumsView<'a> {
    loss_sum: Number,
    samples: u64,
    correct: Counts<'a>,
    counted: Counts<'a>,
}

#[derive(Serialize)]
struct StateView<'a> {
    format: &'static str,
    format_version: u64,
    polypous_version: &'static str,
    config: &'a TrainingConfig,
    base_dir: &'a str,
    step: u64,
    epoch: u64,
    batch: u64,
    steps_per_epoch: u64,
    train_samples: u64,
    validation_samples: u64,
    data_fingerprint: &'a str,
    epoch_sums: SumsView<'a>,
    history: &'a [HistoryEntry],
    best: &'a Option<Best>,
    elapsed_seconds: Number,
}

impl State {
    /// The JSON of `state.json`.
    pub(crate) fn to_json(&self) -> String {
        json::to_string(&StateView {
            format: STATE_FORMAT,
            format_version: STATE_FORMAT_VERSION,
            polypous_version: env!("CARGO_PKG_VERSION"),
            config: &self.config,
            base_dir: &self.base_dir,
            step: self.step,
            epoch: self.epoch,
            batch: self.batch,
            steps_per_epoch: self.steps_per_epoch,
            train_samples: self.train_samples,
            validation_samples: self.validation_samples,
            data_fingerprint: &self.data_fingerprint,
            epoch_sums: SumsView {
                loss_sum: Number(self.sums.loss_sum),
                samples: self.sums.samples,
                correct: Counts(&self.sums.accuracy, true),
                counted: Counts(&self.sums.accuracy, false),
            },
            history: &self.history,
            best: &self.best,
            elapsed_seconds: Number(self.elapsed_seconds),
        })
    }

    /// Parses and checks `state.json` (§5.4 step 1). Errors are `InvalidInput` (or
    /// `UnsupportedOperation` for a config with `precision` `f16`) with the pointer into the
    /// document.
    pub(crate) fn parse(text: &str) -> Result<State, TrainError> {
        let value = json::parse(text)
            .map_err(|e| TrainError::config("", &format!("not valid JSON: {e}")))?;
        let state = read_state(&value)?;
        state.check()?;
        Ok(state)
    }

    /// The counter rules of §5.4 step 1, and the agreement of `history`, `best` and `epoch_sums`
    /// with the config.
    fn check(&self) -> Result<(), TrainError> {
        let bad = |pointer: &str, message: String| Err(TrainError::config(pointer, &message));
        let c = &self.config;
        let s = self.steps_per_epoch;
        if self.train_samples == 0 || self.train_samples > crate::data::MAX_SAMPLES {
            return bad(
                "/train_samples",
                format!(
                    "`train_samples` must be 1 to 2^32, got {}",
                    self.train_samples
                ),
            );
        }
        if s != self.train_samples.div_ceil(c.batch_size) {
            return bad(
                "/steps_per_epoch",
                format!(
                    "`steps_per_epoch` is {s}, but {} training samples in batches of {} make {}",
                    self.train_samples,
                    c.batch_size,
                    self.train_samples.div_ceil(c.batch_size)
                ),
            );
        }
        if (self.validation_samples > 0) != c.data.validation.is_some()
            || self.validation_samples > crate::data::MAX_SAMPLES
        {
            return bad(
                "/validation_samples",
                format!(
                    "`validation_samples` is {}, which doesn't fit the config's validation set",
                    self.validation_samples
                ),
            );
        }
        if self.batch > s {
            return bad(
                "/batch",
                format!("`batch` is {}, but an epoch has {s} batches", self.batch),
            );
        }
        let finished = self.epoch == c.epochs && self.batch == 0;
        if self.epoch >= c.epochs && !finished {
            return bad(
                "/epoch",
                format!(
                    "`epoch` is {} with `batch` {}, but training has {} epochs",
                    self.epoch, self.batch, c.epochs
                ),
            );
        }
        let expected_step = self
            .epoch
            .checked_mul(s)
            .and_then(|v| v.checked_add(self.batch));
        if expected_step != Some(self.step) {
            return bad(
                "/step",
                format!(
                    "`step` is {}, but `epoch` {} · `steps_per_epoch` {s} + `batch` {} is not",
                    self.step, self.epoch, self.batch
                ),
            );
        }
        if !is_fingerprint(&self.data_fingerprint) {
            return bad(
                "/data_fingerprint",
                "`data_fingerprint` must be 16 lowercase hex digits".to_owned(),
            );
        }
        if !(self.elapsed_seconds.is_finite() && self.elapsed_seconds >= 0.0) {
            return bad(
                "/elapsed_seconds",
                format!(
                    "`elapsed_seconds` must be finite and ≥ 0, got {}",
                    self.elapsed_seconds
                ),
            );
        }
        // epoch_sums: the samples of batches 0 … batch − 1 of the current epoch.
        let samples = self
            .batch
            .saturating_mul(c.batch_size)
            .min(self.train_samples);
        if self.sums.samples != samples {
            return bad(
                "/epoch_sums/samples",
                format!(
                    "`samples` is {}, but {} batches of the epoch hold {samples}",
                    self.sums.samples, self.batch
                ),
            );
        }
        if samples == 0 && self.sums.loss_sum.to_bits() != 0 {
            return bad(
                "/epoch_sums/loss_sum",
                "`loss_sum` must be 0 when no samples are counted".to_owned(),
            );
        }
        let names: Vec<&str> = c.metrics.iter().map(|m| m.name()).collect();
        let sum_names: Vec<&str> = self
            .sums
            .accuracy
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        if names != sum_names {
            return bad(
                "/epoch_sums/correct",
                format!("the counts must name the config's metrics {names:?}, got {sum_names:?}"),
            );
        }
        for (name, correct, counted) in &self.sums.accuracy {
            if correct > counted || (samples == 0 && *counted != 0) {
                return bad(
                    &json::child("/epoch_sums/correct", name),
                    format!("`{name}`: {correct} correct of {counted} counted is not possible"),
                );
            }
        }
        // history: one entry per finished epoch, with the reported names.
        if self.history.len() as u64 != self.epoch {
            return bad(
                "/history",
                format!(
                    "`history` has {} entries, but {} epochs have finished",
                    self.history.len(),
                    self.epoch
                ),
            );
        }
        let reported = c.reported_names();
        for (i, h) in self.history.iter().enumerate() {
            let at = format!("/history/{i}");
            let i = i as u64;
            if h.epoch != i {
                return bad(
                    &format!("{at}/epoch"),
                    format!("entry {i} is for epoch {}", h.epoch),
                );
            }
            if Some(h.step) != i.checked_add(1).and_then(|e| e.checked_mul(s)) {
                return bad(
                    &format!("{at}/step"),
                    format!("epoch {i} can't have finished at step {}", h.step),
                );
            }
            let got: Vec<&str> = h.metrics.0.iter().map(|(n, _)| n.as_str()).collect();
            if got != reported {
                return bad(
                    &format!("{at}/metrics"),
                    format!("the metrics must be {reported:?}, got {got:?}"),
                );
            }
        }
        match (
            &c.checkpoint.as_ref().and_then(|k| k.keep_best.clone()),
            &self.best,
        ) {
            (_, None) => {}
            (None, Some(_)) => {
                return bad(
                    "/best",
                    "`best` must be null without `keep_best`".to_owned(),
                );
            }
            (Some(metric), Some(best)) => {
                let entry = usize::try_from(best.epoch)
                    .ok()
                    .and_then(|i| self.history.get(i));
                let value = entry.and_then(|h| h.metrics.get(metric));
                let ok = best.metric == *metric
                    && entry.is_some_and(|h| h.step == best.step)
                    && value.is_some_and(|v| v.to_bits() == best.value.to_bits())
                    && parse_checkpoint_name(&best.checkpoint)
                        .is_some_and(|n| n == best.step && n <= self.step);
                if !ok {
                    return bad(
                        "/best",
                        "`best` doesn't name a history entry of `keep_best` and the epoch \
                         checkpoint written at its step"
                            .to_owned(),
                    );
                }
            }
        }
        Ok(())
    }
}

fn is_fingerprint(s: &str) -> bool {
    s.len() == 16 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn read_metrics(v: &Json, at: &str) -> Read<Metrics> {
    let o = Obj::new(v, at, "`metrics`")?;
    let mut out = Vec::with_capacity(o.members().len());
    for (name, value) in o.members() {
        out.push((name.clone(), as_f64_or_special(value, &o.ptr(name), name)?));
    }
    Ok(Metrics(out))
}

fn read_counts(v: &Json, at: &str, field: &str) -> Read<Vec<(String, u64)>> {
    let o = Obj::new(v, at, field)?;
    o.members()
        .iter()
        .map(|(name, value)| Ok((name.clone(), json::as_u64(value, &o.ptr(name), name)?)))
        .collect()
}

fn read_state(value: &Json) -> Result<State, TrainError> {
    let o = Obj::new(value, "", "`state.json`")?;
    let format = o.req_str("format")?;
    if format != STATE_FORMAT {
        return Err(TrainError::config(
            "/format",
            &format!("`format` must be {STATE_FORMAT:?}, got {format:?}"),
        ));
    }
    let version = o.req_u64("format_version")?;
    if version != STATE_FORMAT_VERSION {
        return Err(TrainError::config(
            "/format_version",
            &format!(
                "`format_version` {version} is not supported: this version reads \
                 {STATE_FORMAT_VERSION}"
            ),
        ));
    }
    o.allow(
        &[
            "format",
            "format_version",
            "polypous_version",
            "config",
            "base_dir",
            "step",
            "epoch",
            "batch",
            "steps_per_epoch",
            "train_samples",
            "validation_samples",
            "data_fingerprint",
            "epoch_sums",
            "history",
            "best",
            "elapsed_seconds",
        ],
        "`state.json`",
    )?;
    o.req_str("polypous_version")?;
    let config = TrainingConfig::from_value(o.req("config")?, "/config")?;
    let base_dir = o.req_str("base_dir")?.to_owned();
    if base_dir.is_empty() || !Path::new(&base_dir).is_absolute() {
        return Err(TrainError::config(
            "/base_dir",
            &format!("`base_dir` must be an absolute path, got {base_dir:?}"),
        ));
    }
    let step = o.req_u64("step")?;
    let epoch = o.req_u64("epoch")?;
    let batch = o.req_u64("batch")?;
    let steps_per_epoch = o.req_u64("steps_per_epoch")?;
    let train_samples = o.req_u64("train_samples")?;
    let validation_samples = o.req_u64("validation_samples")?;
    let data_fingerprint = o.req_str("data_fingerprint")?.to_owned();
    let sums = {
        let at = o.ptr("epoch_sums");
        let s = Obj::new(o.req("epoch_sums")?, &at, "`epoch_sums`")?;
        s.allow(
            &["loss_sum", "samples", "correct", "counted"],
            "`epoch_sums`",
        )?;
        let loss_sum = as_f64_or_special(s.req("loss_sum")?, &s.ptr("loss_sum"), "loss_sum")?;
        let samples = s.req_u64("samples")?;
        let correct = read_counts(s.req("correct")?, &s.ptr("correct"), "`correct`")?;
        let counted = read_counts(s.req("counted")?, &s.ptr("counted"), "`counted`")?;
        if correct
            .iter()
            .map(|(n, _)| n)
            .ne(counted.iter().map(|(n, _)| n))
        {
            return Err(TrainError::config(
                &s.ptr("counted"),
                "`counted` must name the same metrics as `correct`, in the same order",
            ));
        }
        EpochSums {
            loss_sum,
            samples,
            accuracy: correct
                .into_iter()
                .zip(counted)
                .map(|((n, c), (_, t))| (n, c, t))
                .collect(),
        }
    };
    let history = {
        let at = o.ptr("history");
        let items = as_array(o.req("history")?, &at, "history")?;
        let mut out = Vec::new();
        for (i, item) in items.iter().enumerate() {
            let at = format!("{at}/{i}");
            let h = Obj::new(item, &at, "a history entry")?;
            h.allow(&["epoch", "step", "metrics"], "a history entry")?;
            out.push(HistoryEntry {
                epoch: h.req_u64("epoch")?,
                step: h.req_u64("step")?,
                metrics: read_metrics(h.req("metrics")?, &h.ptr("metrics"))?,
            });
        }
        out
    };
    let best = o.nullable("best", None, |v, at| {
        let b = Obj::new(v, at, "`best`")?;
        b.allow(
            &["metric", "value", "epoch", "step", "checkpoint"],
            "`best`",
        )?;
        Ok(Best {
            metric: b.req_str("metric")?.to_owned(),
            value: as_f64_or_special(b.req("value")?, &b.ptr("value"), "value")?,
            epoch: b.req_u64("epoch")?,
            step: b.req_u64("step")?,
            checkpoint: as_str(b.req("checkpoint")?, &b.ptr("checkpoint"), "checkpoint")?
                .to_owned(),
        })
    })?;
    let elapsed_seconds = o.req_f64("elapsed_seconds")?;
    Ok(State {
        config,
        base_dir,
        step,
        epoch,
        batch,
        steps_per_epoch,
        train_samples,
        validation_samples,
        data_fingerprint,
        sums,
        history,
        best,
        elapsed_seconds,
    })
}

/// A tensor of a checkpoint or export: name, shape and `F32` little-endian bytes.
pub(crate) type HostTensor = (String, Vec<usize>, Vec<u8>);

/// The contents of one checkpoint.
pub(crate) struct Contents<'a> {
    pub state: String,
    pub graph: &'a str,
    pub weights: &'a [HostTensor],
    pub optimizer: &'a [HostTensor],
}

/// Writes `bytes` to a new file at `path` and flushes it to disk.
pub(crate) fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), TrainError> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| TrainError::io(&e, "create", path))?;
    file.write_all(bytes)
        .map_err(|e| TrainError::io(&e, "write", path))?;
    file.sync_all()
        .map_err(|e| TrainError::io(&e, "flush", path))
}

/// Flushes a directory's entries to disk (Unix only; see the module documentation).
fn sync_dir(dir: &Path) -> Result<(), TrainError> {
    #[cfg(unix)]
    {
        std::fs::File::open(dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| TrainError::io(&e, "flush the directory", dir))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

fn format_error(e: &FormatError, path: &Path) -> TrainError {
    let kind = if matches!(e.kind(), FormatErrorKind::Io(_)) {
        TrainErrorKind::Io
    } else {
        TrainErrorKind::InvalidInput
    };
    let mut out = TrainError::new(kind, e.to_string())
        .with_file(path)
        .with_details(e.to_string());
    if let Some(t) = e.tensor() {
        out = out.with_tensor(t);
    }
    out
}

/// Writes `tensors` (all `F32`) as a safetensors file without `__metadata__`, through a
/// temporary file, `fsync` and a rename (`polypous-formats`' writer).
pub(crate) fn write_tensors(path: &Path, tensors: &[HostTensor]) -> Result<(), TrainError> {
    let mut entries = Vec::with_capacity(tensors.len());
    for (name, shape, bytes) in tensors {
        let t = WriteTensor::new(StorageDtype::F32, shape, bytes)
            .map_err(|e| format_error(&e, path).with_tensor(name.clone()))?;
        entries.push((name.as_str(), t));
    }
    safetensors::write_file(path, entries, &BTreeMap::new()).map_err(|e| format_error(&e, path))
}

/// Deletes the leftover `.tmp-step-*` directories of `run_dir` (§5.1 step 1).
fn remove_tmp(run_dir: &Path) -> Result<(), TrainError> {
    let read = std::fs::read_dir(run_dir).map_err(|e| TrainError::io(&e, "list", run_dir))?;
    for entry in read {
        let entry = entry.map_err(|e| TrainError::io(&e, "list", run_dir))?;
        if entry.file_name().to_str().is_some_and(is_tmp_name) {
            let path = entry.path();
            let removed = if std::fs::symlink_metadata(&path).is_ok_and(|m| m.is_dir()) {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            removed.map_err(|e| TrainError::io(&e, "delete", &path))?;
            tracing::info!(path = %path.display(), "deleted an incomplete checkpoint");
        }
    }
    Ok(())
}

/// Writes checkpoint `step` into `run_dir` atomically (§5.1 steps 1–4) and returns its path.
pub(crate) fn write(
    run_dir: &Path,
    step: u64,
    contents: &Contents<'_>,
) -> Result<PathBuf, TrainError> {
    remove_tmp(run_dir)?;
    let name = checkpoint_name(step);
    let tmp = run_dir.join(format!(".tmp-{name}"));
    let target = run_dir.join(&name);
    std::fs::create_dir(&tmp).map_err(|e| TrainError::io(&e, "create", &tmp))?;
    let result = (|| {
        write_synced(&tmp.join(STATE_FILE), contents.state.as_bytes())?;
        write_synced(&tmp.join(GRAPH_FILE), contents.graph.as_bytes())?;
        write_tensors(&tmp.join(WEIGHTS_FILE), contents.weights)?;
        write_tensors(&tmp.join(OPTIMIZER_FILE), contents.optimizer)?;
        sync_dir(&tmp)?;
        if std::fs::symlink_metadata(&target).is_ok() {
            std::fs::remove_dir_all(&target).map_err(|e| TrainError::io(&e, "replace", &target))?;
        }
        std::fs::rename(&tmp, &target).map_err(|e| TrainError::io(&e, "rename", &tmp))?;
        sync_dir(run_dir)
    })();
    if result.is_err() {
        // Best effort: the next checkpoint deletes it otherwise.
        let _ = std::fs::remove_dir_all(&tmp);
    }
    result.map(|()| target)
}

/// Reads at most `limit` bytes of `path` as UTF-8. `InvalidInput` if the file is larger.
fn read_bounded(path: &Path, limit: u64, what: &str) -> Result<String, TrainError> {
    let file = std::fs::File::open(path).map_err(|e| TrainError::io(&e, "open", path))?;
    let len = file
        .metadata()
        .map_err(|e| TrainError::io(&e, "read", path))?
        .len();
    if len > limit {
        return Err(TrainError::invalid(format!(
            "{}: {what} is {len} bytes; at most {limit} are allowed",
            path.display()
        ))
        .with_file(path));
    }
    let mut bytes = Vec::new();
    // `len ≤ limit ≤ 64 MiB` fits usize; the reader stops at the limit if the file grew.
    bytes
        .try_reserve_exact(usize::try_from(len).unwrap_or(0))
        .map_err(|_| {
            TrainError::new(
                TrainErrorKind::OutOfMemory,
                format!("could not reserve {len} bytes to read {}", path.display()),
            )
        })?;
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| TrainError::io(&e, "read", path))?;
    if bytes.len() as u64 > limit {
        return Err(TrainError::invalid(format!(
            "{}: {what} is larger than {limit} bytes",
            path.display()
        ))
        .with_file(path));
    }
    String::from_utf8(bytes).map_err(|_| {
        TrainError::invalid(format!("{}: {what} is not UTF-8", path.display())).with_file(path)
    })
}

/// An untrusted checkpoint, checked and read into host memory (§5.4 steps 1–4). Every file is
/// closed again, so the run directory can be changed.
pub(crate) struct Loaded {
    /// The checkpoint directory.
    pub dir: PathBuf,
    /// Its parent, the run directory (§5.4 step 6).
    pub run_dir: PathBuf,
    pub state: State,
    pub graph: Graph,
    pub analysis: Analysis,
    /// Every declared param, in declaration order.
    pub weights: Vec<HostTensor>,
    /// `(param, state name, shape, bytes)`.
    pub optimizer: Vec<(String, String, Vec<usize>, Vec<u8>)>,
}

/// The checkpoint `path` names: `path` itself if it holds a `state.json`, otherwise the newest
/// checkpoint of the run directory `path` (`.tmp-step-*` ignored).
pub(crate) fn locate(path: &Path) -> Result<PathBuf, TrainError> {
    if path.join(STATE_FILE).is_file() {
        return Ok(path.to_path_buf());
    }
    if !path.is_dir() {
        return Err(TrainError::invalid(format!(
            "{} is neither a checkpoint nor a run directory",
            path.display()
        ))
        .with_file(path));
    }
    checkpoints(path)?.pop().map(|e| e.path).ok_or_else(|| {
        TrainError::invalid(format!(
            "the run directory {} holds no checkpoints",
            path.display()
        ))
        .with_file(path)
    })
}

fn invalid_at(path: &Path, message: impl Into<String>) -> TrainError {
    TrainError::invalid(format!("{}: {}", path.display(), message.into())).with_file(path)
}

/// Copies the bytes of a tensor whose size the header checks proved.
fn copy_bytes(data: &[u8], path: &Path, name: &str) -> Result<Vec<u8>, TrainError> {
    let mut out = Vec::new();
    out.try_reserve_exact(data.len()).map_err(|_| {
        TrainError::new(
            TrainErrorKind::OutOfMemory,
            format!("could not reserve {} bytes for `{name}`", data.len()),
        )
        .with_file(path)
        .with_tensor(name)
    })?;
    out.extend_from_slice(data);
    Ok(out)
}

/// Checks a stored tensor's dtype (`F32`), shape and byte range against the declared shape.
fn check_f32_tensor(
    file: &SafetensorsFile,
    path: &Path,
    name: &str,
    shape: &[usize],
) -> Result<(), TrainError> {
    let Some(info) = file.info(name) else {
        return Err(invalid_at(path, format!("the file has no tensor `{name}`")).with_tensor(name));
    };
    if info.dtype() != StorageDtype::F32 {
        return Err(invalid_at(
            path,
            format!(
                "tensor `{name}` is stored as {}; a checkpoint stores F32",
                info.dtype().name()
            ),
        )
        .with_tensor(name));
    }
    if info.shape() != shape {
        return Err(invalid_at(
            path,
            format!(
                "tensor `{name}` has shape {:?}, but the graph declares {shape:?}",
                info.shape()
            ),
        )
        .with_tensor(name));
    }
    let bytes = shape.iter().try_fold(4usize, |acc, &d| acc.checked_mul(d));
    let (start, end) = info.data_offsets();
    if bytes.is_none() || end.checked_sub(start) != bytes || Some(info.byte_len()) != bytes {
        return Err(invalid_at(
            path,
            format!("tensor `{name}`: its byte range doesn't fit its shape"),
        )
        .with_tensor(name));
    }
    Ok(())
}

/// Loads checkpoint `dir` (§5.4 steps 1–4).
pub(crate) fn load(dir: &Path) -> Result<Loaded, TrainError> {
    let run_dir = match dir.parent() {
        Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
        _ => PathBuf::from("."),
    };
    // Step 1: state.json.
    let state_path = dir.join(STATE_FILE);
    let text = read_bounded(&state_path, MAX_STATE_BYTES, "state.json")?;
    let state = State::parse(&text).map_err(|e| {
        e.prefixed(&format!("{}: ", state_path.display()))
            .with_file(&state_path)
    })?;
    drop(text);
    let expected = checkpoint_name(state.step);
    if dir.file_name().and_then(|n| n.to_str()) != Some(expected.as_str()) {
        return Err(invalid_at(
            dir,
            format!(
                "a checkpoint after {} steps must be a directory named {expected}",
                state.step
            ),
        ));
    }
    // Step 2: graph.json.
    let graph_path = dir.join(GRAPH_FILE);
    let graph_text = read_bounded(
        &graph_path,
        polypous_ir::limits::MAX_DOCUMENT_BYTES as u64,
        "graph.json",
    )?;
    let graph = Graph::from_json(graph_text.as_bytes()).map_err(|report| {
        invalid_at(&graph_path, "the graph is not valid").with_details(report.to_string())
    })?;
    drop(graph_text);
    let analysis = graph
        .analyze(&ValidateOptions::default())
        .map_err(|report| {
            invalid_at(&graph_path, "the graph is not valid").with_details(report.to_string())
        })?;
    // Step 3: weights.safetensors: exactly the declared params, F32, declared shapes.
    let weights_path = dir.join(WEIGHTS_FILE);
    let file = SafetensorsFile::open(&weights_path).map_err(|e| format_error(&e, &weights_path))?;
    for p in &analysis.params {
        check_f32_tensor(&file, &weights_path, &p.name, &p.shape)?;
    }
    let mut extra: Vec<&str> = file
        .names()
        .filter(|n| analysis.param(n).is_none())
        .collect();
    extra.sort_unstable();
    if let Some(name) = extra.first() {
        return Err(invalid_at(
            &weights_path,
            format!("tensor `{name}` is not a param of the graph"),
        )
        .with_tensor(*name));
    }
    let mut weights = Vec::with_capacity(analysis.params.len());
    for p in &analysis.params {
        let view = file
            .tensor(&p.name)
            .map_err(|e| format_error(&e, &weights_path))?;
        weights.push((
            p.name.clone(),
            p.shape.clone(),
            copy_bytes(view.data(), &weights_path, &p.name)?,
        ));
    }
    drop(file);
    // Step 4: optimizer.safetensors: `<trainable param>:<state name>`, all or none per param.
    let opt_path = dir.join(OPTIMIZER_FILE);
    let names = Optimizer::new(state.config.optimizer.clone())
        .map_err(|e| TrainError::internal(e.to_string()))?
        .state_names();
    let file = SafetensorsFile::open(&opt_path).map_err(|e| format_error(&e, &opt_path))?;
    let mut tensor_names: Vec<&str> = file.names().collect();
    tensor_names.sort_unstable();
    let mut per_param: BTreeMap<&str, usize> = BTreeMap::new();
    for &full in &tensor_names {
        let bad = |why: String| {
            invalid_at(&opt_path, format!("tensor `{full}`: {why}")).with_tensor(full)
        };
        let Some((param, state_name)) = full.split_once(':') else {
            return Err(bad("not named `<param>:<state>`".into()));
        };
        let Some(info) = analysis.param(param).filter(|p| p.is_trainable()) else {
            return Err(bad(format!(
                "`{param}` is not a trainable param of the graph"
            )));
        };
        if !names.contains(&state_name) {
            return Err(bad(format!(
                "`{state_name}` is not a state of `{}` (its states are {names:?})",
                state.config.optimizer.kind()
            )));
        }
        check_f32_tensor(&file, &opt_path, full, &info.shape)?;
        *per_param.entry(param).or_default() += 1;
    }
    if let Some((param, n)) = per_param.iter().find(|(_, n)| **n != names.len()) {
        return Err(invalid_at(
            &opt_path,
            format!(
                "param `{param}` has {n} of its {} optimizer states; a param has all or none",
                names.len()
            ),
        )
        .with_tensor(*param));
    }
    let mut optimizer = Vec::with_capacity(tensor_names.len());
    for full in tensor_names {
        let view = file.tensor(full).map_err(|e| format_error(&e, &opt_path))?;
        let (param, state_name) = full.split_once(':').unwrap_or((full, ""));
        optimizer.push((
            param.to_owned(),
            state_name.to_owned(),
            view.shape().to_vec(),
            copy_bytes(view.data(), &opt_path, full)?,
        ));
    }
    drop(file);
    Ok(Loaded {
        dir: dir.to_path_buf(),
        run_dir,
        state,
        graph,
        analysis,
        weights,
        optimizer,
    })
}

/// The `keep_best` value of a checkpoint with these config and counters (§5.3), with the epoch
/// and step of the history entry it comes from.
///
/// Only a checkpoint that §5.1's first rule writes, after an `epoch_finished`, has a value: its
/// state is right after epoch `e` (`epoch = e + 1`, `batch = 0`, `step` the step of `history`'s
/// last entry, which is epoch `e`'s), and that epoch's checkpoint was due: `(e + 1)` is a multiple
/// of `every_epochs`, a multiple of `every_steps` was reached by the epoch's last step, or `e` is
/// the last epoch. A checkpoint written in the middle of an epoch (`every_steps`) or on
/// cancellation has none, even if its last history entry is that of the previous epoch, because
/// its weights are not the ones that entry measured. A cancel checkpoint at an epoch boundary
/// where the epoch checkpoint was due holds exactly that checkpoint's state.
pub(crate) fn epoch_value(
    config: &TrainingConfig,
    history: &[HistoryEntry],
    step: u64,
    epoch: u64,
    batch: u64,
) -> Option<(f64, u64, u64)> {
    let checkpoint = config.checkpoint.as_ref()?;
    let metric = checkpoint.keep_best.as_deref()?;
    let h = history.last()?;
    if batch != 0 || step != h.step || h.epoch.checked_add(1) != Some(epoch) {
        return None;
    }
    let e = h.epoch;
    let due = checkpoint
        .every_epochs
        .is_some_and(|n| (e + 1).is_multiple_of(n))
        || checkpoint
            .every_steps
            .is_some_and(|n| step.is_multiple_of(n))
        || e + 1 == config.epochs;
    if !due {
        return None;
    }
    Some((h.metrics.get(metric)?, h.epoch, h.step))
}

/// The `keep_best` value of checkpoint `dir` ([`epoch_value`]), judged from its own
/// `state.json` (config and counters), or `None` if it has none or its `state.json` can't be
/// read or isn't valid (logged at `warn`). `metric` must be the trainer's `keep_best`.
pub(crate) fn last_value(dir: &Path, metric: &str) -> Option<(f64, u64, u64)> {
    let path = dir.join(STATE_FILE);
    let state = read_bounded(&path, MAX_STATE_BYTES, "state.json").and_then(|t| State::parse(&t));
    let state = match state {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "can't read a checkpoint's state");
            return None;
        }
    };
    let keep_best = state.config.checkpoint.as_ref()?.keep_best.as_deref()?;
    if keep_best != metric {
        return None;
    }
    epoch_value(
        &state.config,
        &state.history,
        state.step,
        state.epoch,
        state.batch,
    )
}

/// Whether `candidate` is better than `current` for `metric` (§5.3): larger for `accuracy`,
/// smaller for `loss`. NaN is never better.
pub(crate) fn better(metric: &str, candidate: f64, current: Option<f64>) -> bool {
    if candidate.is_nan() {
        return false;
    }
    let larger = !metric.ends_with(".loss");
    match current {
        None => true,
        Some(c) if c.is_nan() => true,
        Some(c) => {
            if larger {
                candidate > c
            } else {
                candidate < c
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert_eq!(checkpoint_name(844), "step-000000000844");
        assert_eq!(checkpoint_name(1_234_567_890_123), "step-1234567890123");
        assert_eq!(parse_checkpoint_name("step-000000000844"), Some(844));
        assert_eq!(
            parse_checkpoint_name("step-1234567890123"),
            Some(1_234_567_890_123)
        );
        for bad in [
            "step-5",
            "step-0000000008440",
            "step-",
            "step-00000000084a",
            "x-000000000844",
        ] {
            assert_eq!(parse_checkpoint_name(bad), None, "{bad}");
        }
        assert!(is_tmp_name(".tmp-step-000000000844"));
        assert!(!is_tmp_name(".tmp-step-5"));
        assert!(!is_tmp_name("tmp-step-000000000844"));
    }

    #[test]
    fn only_epoch_checkpoints_have_a_value() {
        use crate::events::Metrics;
        let config = |every_epochs: Option<u64>, every_steps: Option<u64>| {
            let mut c = crate::training_config::TrainingConfig::from_json(
                r#"{"format": "polypous.training", "format_version": 1, "epochs": 5,
                    "batch_size": 8, "data": {"train": {"kind": "tensors", "name": "t"}},
                    "inputs": {"x": "x"},
                    "loss": {"kind": "mse", "prediction": "y", "target": "y"},
                    "optimizer": {"kind": "sgd"},
                    "checkpoint": {"dir": "r", "keep_best": "train.loss"}}"#,
            )
            .unwrap();
            let k = c.checkpoint.as_mut().unwrap();
            k.every_epochs = every_epochs;
            k.every_steps = every_steps;
            c
        };
        // 4 steps per epoch; epochs 0 and 1 finished (steps 4 and 8).
        let history: Vec<HistoryEntry> = (0..2)
            .map(|e| HistoryEntry {
                epoch: e,
                step: 4 * (e + 1),
                metrics: Metrics(vec![("train.loss".into(), 0.5 + e as f64)]),
            })
            .collect();
        let c = config(Some(2), Some(5));
        // After epoch 1 (due: 2 is a multiple of every_epochs): the epoch checkpoint.
        assert_eq!(epoch_value(&c, &history, 8, 2, 0), Some((1.5, 1, 8)));
        // After epoch 0 (not due): a cancel checkpoint there has no value.
        assert_eq!(epoch_value(&c, &history[..1], 4, 1, 0), None);
        // Mid-epoch (every_steps, cancel) and during validation (batch = S).
        assert_eq!(epoch_value(&c, &history, 10, 2, 2), None);
        assert_eq!(epoch_value(&c, &history[..1], 8, 1, 4), None);
        // Due by every_steps reached at the epoch's last step, or by the last epoch.
        assert_eq!(
            epoch_value(&config(None, Some(4)), &history[..1], 4, 1, 0),
            Some((0.5, 0, 4))
        );
        let last: Vec<HistoryEntry> = (0..5)
            .map(|e| HistoryEntry {
                epoch: e,
                step: 4 * (e + 1),
                metrics: Metrics(vec![("train.loss".into(), 1.0)]),
            })
            .collect();
        assert_eq!(
            epoch_value(&config(Some(2), None), &last, 20, 5, 0),
            Some((1.0, 4, 20))
        );
        assert_eq!(
            epoch_value(&config(Some(2), None), &last[..3], 12, 3, 0),
            None
        );
        // Without keep_best: no value.
        let mut none = config(Some(1), None);
        none.checkpoint.as_mut().unwrap().keep_best = None;
        assert_eq!(epoch_value(&none, &history, 8, 2, 0), None);
    }

    #[test]
    fn better_rules() {
        assert!(better("val.accuracy", 0.5, None));
        assert!(better("val.accuracy", 0.6, Some(0.5)));
        assert!(
            !better("val.accuracy", 0.5, Some(0.5)),
            "a tie keeps the earlier"
        );
        assert!(!better("val.accuracy", f64::NAN, None), "NaN is never best");
        assert!(better("train.loss", 0.4, Some(0.5)));
        assert!(!better("train.loss", 0.6, Some(0.5)));
    }
}
