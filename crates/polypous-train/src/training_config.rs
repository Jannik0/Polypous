//! The training config (`docs/specs/training.md` §2.1–§2.9) as a Rust type: [`TrainingConfig`],
//! with [`TrainingConfig::from_json`], [`TrainingConfig::from_file`] and
//! [`TrainingConfig::to_json`].
//!
//! Parsing has two phases, both without a graph:
//!
//! 1. **Structure.** The document (at most 1 MiB, nesting depth at most 32) is read into a
//!    [`Json`] tree and walked object by object. In each object: a key given twice, then (for the
//!    root) `format` and `format_version`, then the first unknown field in document order, then
//!    the fields in the order of the spec's tables: a missing required field or a value of the
//!    wrong JSON type. The nested objects become the existing config types of [`crate::config`]
//!    and [`crate::data`].
//! 2. **Ranges** ([`TrainingConfig::validate`]), in table order, with the nested objects'
//!    `validate` and their pointers prefixed (`/optimizer/betas/1`).
//!
//! Every error is a [`TrainError`] of kind `InvalidInput` with the full pointer, except
//! `precision` `f16`/`bf16`, which is `UnsupportedOperation` (§2.2).
//!
//! The checks against the graph and the data (§2.10) are the trainer's.

use std::collections::BTreeMap;
use std::io::Read as _;
use std::path::Path;

use polypous_backend::device::DeviceId;
use serde::Serialize;

use crate::config::{
    AccuracyConfig, AdamConfig, AdamWConfig, BinaryCrossEntropyConfig, CrossEntropyConfig,
    GradientClipConfig, LossConfig, MetricConfig, MseConfig, OptimizerConfig, ScheduleConfig,
    SgdConfig, validate_metrics,
};
use crate::data::{
    ColumnSource, CsvColumnConfig, CsvDatasetConfig, CsvDtype, DEFAULT_MAX_BYTES, DataConfig,
    DatasetConfig, ImageFolderConfig, SafetensorsDatasetConfig, SplitConfig, TensorsDatasetConfig,
};
use crate::error::{TrainError, TrainErrorKind};
use crate::json::{self, Json, Obj, Read, as_array, as_f64, as_str, as_u64, error};

/// `format` of a training config.
pub const FORMAT: &str = "polypous.training";
/// The only supported `format_version`.
pub const FORMAT_VERSION: u64 = 1;
/// The largest config document: 1 MiB (§2.1, "Limits").
pub const MAX_CONFIG_BYTES: usize = 1 << 20;
/// The deepest JSON nesting of a config (§2.1).
pub const MAX_CONFIG_DEPTH: usize = 32;
/// The most entries of `inputs` (§2.1).
pub const MAX_INPUTS: usize = 4096;
/// The most `epochs` (§2.2).
pub const MAX_EPOCHS: u64 = 10_000;
/// The largest `batch_size` (§2.2).
pub const MAX_BATCH_SIZE: u64 = 65_536;
/// The largest `every_steps` (§2.9: `2^32`).
pub const MAX_EVERY_STEPS: u64 = 1 << 32;
/// The largest `every_epochs` and `keep_last` (§2.9).
pub const MAX_CHECKPOINT_COUNT: u64 = 10_000;

const NAME_BYTES: usize = 256;
const PATH_BYTES: usize = 4096;

/// `checkpoint` (training.md §2.9).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckpointConfig {
    /// The run directory (§5.1), 1–4,096 bytes; relative to the base directory. Required.
    pub dir: String,
    /// A checkpoint after every `every_epochs`-th epoch, 1–10,000; `None` (JSON null): none by
    /// epoch. Default `Some(1)`.
    pub every_epochs: Option<u64>,
    /// A checkpoint after every `every_steps`-th optimizer step, 1 to `2^32`; `None`: none by
    /// step. Default `None`.
    pub every_steps: Option<u64>,
    /// How many of the newest checkpoints to keep, 1–10,000; `None`: all. Default `Some(3)`.
    pub keep_last: Option<u64>,
    /// A reported metric name (`val.accuracy`): also keep the checkpoint with its best value
    /// (§5.3). Default `None`.
    pub keep_best: Option<String>,
    /// Write a checkpoint when the run is cancelled (§4.3). Default `true`.
    pub on_cancel: bool,
}

impl CheckpointConfig {
    /// A config with `dir` and every other field at its default.
    pub fn new(dir: impl Into<String>) -> Self {
        Self {
            dir: dir.into(),
            every_epochs: Some(1),
            every_steps: None,
            keep_last: Some(3),
            keep_best: None,
            on_cancel: true,
        }
    }
}

/// A training config (training.md §2). Plain data: its fields are public, and
/// [`TrainingConfig::validate`] checks a value built in Rust as [`TrainingConfig::from_json`]
/// checks a document. `format` and `format_version` are not fields: they are always
/// `"polypous.training"` and `1`.
#[derive(Clone, Debug, PartialEq)]
pub struct TrainingConfig {
    /// Keys every random stream (§1.3). Default `0`.
    pub seed: u64,
    /// `"auto"` or a device id (`"cpu"`, `"cuda:0"`). Default `"auto"`. The facade resolves it.
    pub device: String,
    /// `"f32"` only in M2. Default `"f32"`.
    pub precision: String,
    /// 1–10,000. Required.
    pub epochs: u64,
    /// 1–65,536. Required.
    pub batch_size: u64,
    /// The datasets (§2.3). Required.
    pub data: DataConfig,
    /// Graph input → dataset field (§2.5). Required.
    pub inputs: BTreeMap<String, String>,
    /// The loss (§2.6). Required.
    pub loss: LossConfig,
    /// The metrics (§2.7). Default empty.
    pub metrics: Vec<MetricConfig>,
    /// The optimizer (§2.8). Required.
    pub optimizer: OptimizerConfig,
    /// The learning-rate schedule (§2.8). Default `constant`.
    pub schedule: ScheduleConfig,
    /// Gradient clipping (§3.1.4), or none. Default none.
    pub gradient_clip: Option<GradientClipConfig>,
    /// Checkpoints (§2.9), or none. Default none.
    pub checkpoint: Option<CheckpointConfig>,
}

/// The canonical JSON form, fields in table order.
#[derive(Serialize)]
struct View<'a> {
    format: &'static str,
    format_version: u64,
    seed: u64,
    device: &'a str,
    precision: &'a str,
    epochs: u64,
    batch_size: u64,
    data: &'a DataConfig,
    inputs: &'a BTreeMap<String, String>,
    loss: &'a LossConfig,
    metrics: &'a [MetricConfig],
    optimizer: &'a OptimizerConfig,
    schedule: &'a ScheduleConfig,
    gradient_clip: &'a Option<GradientClipConfig>,
    checkpoint: &'a Option<CheckpointConfig>,
}

impl TrainingConfig {
    /// Parses and checks a config document (§2.1–§2.9).
    ///
    /// # Errors
    /// `InvalidInput` with the pointer of the offending value (`""` for the whole document: not
    /// JSON, not an object, over 1 MiB or deeper than 32); `UnsupportedOperation` at
    /// `/precision` for `f16`/`bf16`.
    pub fn from_json(text: &str) -> Result<Self, TrainError> {
        if text.len() > MAX_CONFIG_BYTES {
            return Err(TrainError::config(
                "",
                &format!(
                    "the config is {} bytes; at most {MAX_CONFIG_BYTES} (1 MiB) are allowed",
                    text.len()
                ),
            ));
        }
        let value = json::parse(text)
            .map_err(|e| TrainError::config("", &format!("not valid JSON: {e}")))?;
        Self::from_value(&value, "")
    }

    /// Reads and parses a config file: at most 1 MiB is read (`InvalidInput` if the file is
    /// larger), then [`TrainingConfig::from_json`].
    ///
    /// # Errors
    /// `Io` with `context.file` if the file can't be read; as [`TrainingConfig::from_json`].
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TrainError> {
        let path = path.as_ref();
        let file =
            std::fs::File::open(path).map_err(|e| TrainError::io(&e, "open the config", path))?;
        let mut bytes = Vec::new();
        file.take(MAX_CONFIG_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| TrainError::io(&e, "read the config", path))?;
        if bytes.len() > MAX_CONFIG_BYTES {
            return Err(TrainError::config(
                "",
                &format!("the config is larger than {MAX_CONFIG_BYTES} bytes (1 MiB)"),
            )
            .with_file(path));
        }
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| TrainError::config("", &format!("not UTF-8: {e}")).with_file(path))?;
        Self::from_json(text).map_err(|e| e.or_file(path))
    }

    /// Reads a config from a parsed value whose pointer is `at` (`""` for a config document,
    /// `"/config"` inside a checkpoint's `state.json`).
    pub(crate) fn from_value(value: &Json, at: &str) -> Result<Self, TrainError> {
        if value.depth() > MAX_CONFIG_DEPTH {
            return Err(TrainError::config(
                at,
                &format!(
                    "the config is nested {} levels deep; at most {MAX_CONFIG_DEPTH} are allowed",
                    value.depth()
                ),
            ));
        }
        let config = read_root(value, at)?;
        config.validate_at(at)?;
        Ok(config)
    }

    /// The canonical JSON form (§2.1): every field, defaults filled in, in table order.
    /// `from_json(&c.to_json()) == c` for every valid config.
    pub fn to_json(&self) -> String {
        json::to_string(&self.view())
    }

    fn view(&self) -> View<'_> {
        View {
            format: FORMAT,
            format_version: FORMAT_VERSION,
            seed: self.seed,
            device: &self.device,
            precision: &self.precision,
            epochs: self.epochs,
            batch_size: self.batch_size,
            data: &self.data,
            inputs: &self.inputs,
            loss: &self.loss,
            metrics: &self.metrics,
            optimizer: &self.optimizer,
            schedule: &self.schedule,
            gradient_clip: &self.gradient_clip,
            checkpoint: &self.checkpoint,
        }
    }

    /// Checks every range of §2.1–§2.9, in table order (the config's structure is given by the
    /// type). The first failure is returned.
    ///
    /// # Errors
    /// As [`TrainingConfig::from_json`].
    pub fn validate(&self) -> Result<(), TrainError> {
        self.validate_at("")
    }

    fn validate_at(&self, at: &str) -> Result<(), TrainError> {
        let ptr = |field: &str| format!("{at}/{field}");
        let bad = |field: &str, message: String| TrainError::config(&ptr(field), &message);
        if self.device != "auto" && self.device.parse::<DeviceId>().is_err() {
            return Err(bad(
                "device",
                format!(
                    "`device` must be \"auto\" or a device id (cpu, cuda:0, vulkan:discrete_gpu:0), \
                     got {:?}",
                    self.device
                ),
            ));
        }
        match self.precision.as_str() {
            "f32" => {}
            "f16" | "bf16" => {
                return Err(TrainError::new(
                    TrainErrorKind::UnsupportedOperation,
                    format!(
                        "{}: `precision` {:?} is not supported in this version: training runs in \
                         f32 only (mixed precision comes later)",
                        ptr("precision"),
                        self.precision
                    ),
                )
                .with_pointer(ptr("precision")));
            }
            other => {
                return Err(bad(
                    "precision",
                    format!("`precision` must be \"f32\", got {other:?}"),
                ));
            }
        }
        if !(1..=MAX_EPOCHS).contains(&self.epochs) {
            return Err(bad(
                "epochs",
                format!("`epochs` must be 1–{MAX_EPOCHS}, got {}", self.epochs),
            ));
        }
        if !(1..=MAX_BATCH_SIZE).contains(&self.batch_size) {
            return Err(bad(
                "batch_size",
                format!(
                    "`batch_size` must be 1–{MAX_BATCH_SIZE}, got {}",
                    self.batch_size
                ),
            ));
        }
        self.data
            .validate()
            .map_err(|e| TrainError::nested(&ptr("data"), &e))?;
        if self.inputs.len() > MAX_INPUTS {
            return Err(bad(
                "inputs",
                format!(
                    "`inputs` has {} entries; at most {MAX_INPUTS} are allowed",
                    self.inputs.len()
                ),
            ));
        }
        for (input, field) in &self.inputs {
            let at_input = json::child(&ptr("inputs"), input);
            for (what, name) in [("graph input name", input), ("field name", field)] {
                if !(1..=NAME_BYTES).contains(&name.len()) {
                    return Err(TrainError::config(
                        &at_input,
                        &format!(
                            "a {what} in `inputs` must be 1–{NAME_BYTES} bytes, got {} bytes",
                            name.len()
                        ),
                    ));
                }
            }
        }
        self.loss
            .validate()
            .map_err(|e| TrainError::nested(&ptr("loss"), &e))?;
        validate_metrics(&self.metrics).map_err(|e| TrainError::nested(&ptr("metrics"), &e))?;
        self.optimizer
            .validate()
            .map_err(|e| TrainError::nested(&ptr("optimizer"), &e))?;
        self.schedule
            .validate(self.optimizer.lr())
            .map_err(|e| TrainError::nested(&ptr("schedule"), &e))?;
        if let Some(clip) = &self.gradient_clip {
            clip.validate()
                .map_err(|e| TrainError::nested(&ptr("gradient_clip"), &e))?;
        }
        if let Some(c) = &self.checkpoint {
            let cp = |field: &str| format!("{at}/checkpoint/{field}");
            if !(1..=PATH_BYTES).contains(&c.dir.len()) {
                return Err(TrainError::config(
                    &cp("dir"),
                    &format!(
                        "`dir` must be 1–{PATH_BYTES} bytes, got {} bytes",
                        c.dir.len()
                    ),
                ));
            }
            let ranges = [
                (
                    "every_epochs",
                    c.every_epochs,
                    MAX_CHECKPOINT_COUNT,
                    "10,000",
                ),
                ("every_steps", c.every_steps, MAX_EVERY_STEPS, "2^32"),
                ("keep_last", c.keep_last, MAX_CHECKPOINT_COUNT, "10,000"),
            ];
            for (field, value, max, max_text) in ranges {
                if let Some(v) = value
                    && !(1..=max).contains(&v)
                {
                    return Err(TrainError::config(
                        &cp(field),
                        &format!("`{field}` must be 1 to {max_text} or null, got {v}"),
                    ));
                }
            }
            if let Some(name) = &c.keep_best
                && !(1..=NAME_BYTES).contains(&name.len())
            {
                return Err(TrainError::config(
                    &cp("keep_best"),
                    &format!(
                        "`keep_best` must be 1–{NAME_BYTES} bytes, got {} bytes",
                        name.len()
                    ),
                ));
            }
        }
        Ok(())
    }

    /// The fields the training uses (§3.5): `inputs`' values, `loss.target` and the metrics'
    /// `target`.
    pub fn used_fields(&self) -> std::collections::BTreeSet<String> {
        let mut out: std::collections::BTreeSet<String> = self.inputs.values().cloned().collect();
        out.insert(self.loss.target().to_owned());
        for m in &self.metrics {
            match m {
                MetricConfig::Accuracy(c) => {
                    out.insert(c.target.clone());
                }
            }
        }
        out
    }

    /// The reported metric names (§2.7), in order: `train.loss`, `train.<name>`…, and with a
    /// validation set `val.loss`, `val.<name>`….
    pub fn reported_names(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(2 * (self.metrics.len() + 1));
        let sets: &[&str] = if self.data.validation.is_some() {
            &["train", "val"]
        } else {
            &["train"]
        };
        for set in sets {
            out.push(format!("{set}.loss"));
            for m in &self.metrics {
                out.push(format!("{set}.{}", m.name()));
            }
        }
        out
    }
}

impl Serialize for TrainingConfig {
    /// The canonical form of [`TrainingConfig::to_json`].
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.view().serialize(s)
    }
}

fn kind_error<T>(o: &Obj<'_>, kind: &str, allowed: &[&str], what: &str) -> Read<T> {
    error(
        &o.ptr("kind"),
        format!(
            "unknown {what} kind {kind:?}; the kinds are {}",
            allowed
                .iter()
                .map(|k| format!("`{k}`"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    )
}

fn read_root(value: &Json, at: &str) -> Read<TrainingConfig> {
    let o = Obj::new(value, at, "the config")?;
    let format = o.req_str("format")?;
    if format != FORMAT {
        return error(
            &o.ptr("format"),
            format!("`format` must be {FORMAT:?}, got {format:?}"),
        );
    }
    let version = o.req_u64("format_version")?;
    if version != FORMAT_VERSION {
        return error(
            &o.ptr("format_version"),
            format!(
                "`format_version` {version} is not supported: this version reads \
                 {FORMAT_VERSION} (a newer config needs a newer Polypous)"
            ),
        );
    }
    o.allow(
        &[
            "format",
            "format_version",
            "seed",
            "device",
            "precision",
            "epochs",
            "batch_size",
            "data",
            "inputs",
            "loss",
            "metrics",
            "optimizer",
            "schedule",
            "gradient_clip",
            "checkpoint",
        ],
        "the config",
    )?;
    let seed = o.opt_u64("seed", 0)?;
    let device = o.opt_str("device", "auto")?;
    let precision = o.opt_str("precision", "f32")?;
    let epochs = o.req_u64("epochs")?;
    let batch_size = o.req_u64("batch_size")?;
    let data = read_data(o.req("data")?, &o.ptr("data"))?;
    let inputs = read_inputs(o.req("inputs")?, &o.ptr("inputs"))?;
    let loss = read_loss(o.req("loss")?, &o.ptr("loss"))?;
    let metrics = match o.get("metrics") {
        None => Vec::new(),
        Some(v) => {
            let at = o.ptr("metrics");
            as_array(v, &at, "metrics")?
                .iter()
                .enumerate()
                .map(|(i, m)| read_metric(m, &format!("{at}/{i}")))
                .collect::<Read<_>>()?
        }
    };
    let optimizer = read_optimizer(o.req("optimizer")?, &o.ptr("optimizer"))?;
    let schedule = match o.get("schedule") {
        None => ScheduleConfig::default(),
        Some(v) => read_schedule(v, &o.ptr("schedule"))?,
    };
    let gradient_clip = o.nullable("gradient_clip", None, read_clip)?;
    let checkpoint = o.nullable("checkpoint", None, read_checkpoint)?;
    Ok(TrainingConfig {
        seed,
        device,
        precision,
        epochs,
        batch_size,
        data,
        inputs,
        loss,
        metrics,
        optimizer,
        schedule,
        gradient_clip,
        checkpoint,
    })
}

fn read_inputs(v: &Json, at: &str) -> Read<BTreeMap<String, String>> {
    let o = Obj::new(v, at, "`inputs`")?;
    let mut out = BTreeMap::new();
    for (key, value) in o.members() {
        let field = as_str(value, &o.ptr(key), key)?;
        out.insert(key.clone(), field.to_owned());
    }
    Ok(out)
}

fn read_data(v: &Json, at: &str) -> Read<DataConfig> {
    let o = Obj::new(v, at, "`data`")?;
    o.allow(&["train", "validation", "shuffle", "max_bytes"], "`data`")?;
    let train = read_dataset(o.req("train")?, &o.ptr("train"))?;
    let validation = o.nullable("validation", None, read_dataset)?;
    let shuffle = o.opt_bool("shuffle", true)?;
    let max_bytes = o.opt_u64("max_bytes", DEFAULT_MAX_BYTES)?;
    Ok(DataConfig {
        train,
        validation,
        shuffle,
        max_bytes,
    })
}

fn read_dataset(v: &Json, at: &str) -> Read<DatasetConfig> {
    const KINDS: &[&str] = &["tensors", "safetensors", "csv", "image_folder", "split"];
    let o = Obj::new(v, at, "a dataset")?;
    let kind = o.req_str("kind")?;
    let what = format!("a `{kind}` dataset");
    Ok(match kind {
        "tensors" => {
            o.allow(&["kind", "name"], &what)?;
            DatasetConfig::Tensors(TensorsDatasetConfig {
                name: o.req_str("name")?.to_owned(),
            })
        }
        "safetensors" => {
            o.allow(&["kind", "path"], &what)?;
            DatasetConfig::Safetensors(SafetensorsDatasetConfig {
                path: o.req_str("path")?.to_owned(),
            })
        }
        "csv" => {
            o.allow(&["kind", "path", "columns"], &what)?;
            let path = o.req_str("path")?.to_owned();
            let at_columns = o.ptr("columns");
            let columns_obj = Obj::new(o.req("columns")?, &at_columns, "`columns`")?;
            let mut columns = BTreeMap::new();
            for (name, column) in columns_obj.members() {
                columns.insert(
                    name.clone(),
                    read_csv_column(column, &columns_obj.ptr(name))?,
                );
            }
            DatasetConfig::Csv(CsvDatasetConfig { path, columns })
        }
        "image_folder" => {
            o.allow(
                &[
                    "kind",
                    "path",
                    "image_size",
                    "channels",
                    "image_field",
                    "label_field",
                ],
                &what,
            )?;
            let path = o.req_str("path")?.to_owned();
            let at_size = o.ptr("image_size");
            let size = as_array(o.req("image_size")?, &at_size, "image_size")?;
            let [h, w] = size else {
                return error(
                    &at_size,
                    format!("`image_size` must be [H, W], got {} values", size.len()),
                );
            };
            let image_size = [
                as_u64(h, &format!("{at_size}/0"), "image_size[0]")?,
                as_u64(w, &format!("{at_size}/1"), "image_size[1]")?,
            ];
            DatasetConfig::ImageFolder(ImageFolderConfig {
                path,
                image_size,
                channels: o.opt_u64("channels", 1)?,
                image_field: o.opt_str("image_field", "image")?,
                label_field: o.opt_str("label_field", "label")?,
            })
        }
        "split" => {
            o.allow(&["kind", "from", "fraction"], &what)?;
            DatasetConfig::Split(SplitConfig {
                from: o.opt_str("from", "train")?,
                fraction: o.req_f64("fraction")?,
            })
        }
        other => return kind_error(&o, other, KINDS, "dataset"),
    })
}

fn read_csv_column(v: &Json, at: &str) -> Read<CsvColumnConfig> {
    let o = Obj::new(v, at, "a CSV field")?;
    o.allow(&["from", "dtype"], "a CSV field")?;
    let at_from = o.ptr("from");
    let from = match o.req("from")? {
        Json::Str(s) => ColumnSource::One(s.clone()),
        Json::Arr(items) => ColumnSource::Many(
            items
                .iter()
                .enumerate()
                .map(|(i, c)| as_str(c, &format!("{at_from}/{i}"), "from").map(str::to_owned))
                .collect::<Read<_>>()?,
        ),
        other => {
            return error(
                &at_from,
                format!(
                    "`from` must be a column name or an array of them, got {}",
                    other.type_name()
                ),
            );
        }
    };
    let dtype = match o.req_str("dtype")? {
        "f32" => CsvDtype::F32,
        "i64" => CsvDtype::I64,
        "i32" => CsvDtype::I32,
        other => {
            return error(
                &o.ptr("dtype"),
                format!("`dtype` must be \"f32\", \"i64\" or \"i32\", got {other:?}"),
            );
        }
    };
    Ok(CsvColumnConfig { from, dtype })
}

fn read_loss(v: &Json, at: &str) -> Read<LossConfig> {
    const KINDS: &[&str] = &["mse", "cross_entropy", "binary_cross_entropy"];
    let o = Obj::new(v, at, "`loss`")?;
    let kind = o.req_str("kind")?;
    let what = format!("a `{kind}` loss");
    let names = |o: &Obj<'_>| -> Read<(String, String)> {
        Ok((
            o.req_str("prediction")?.to_owned(),
            o.req_str("target")?.to_owned(),
        ))
    };
    Ok(match kind {
        "mse" => {
            o.allow(&["kind", "prediction", "target"], &what)?;
            let (prediction, target) = names(&o)?;
            LossConfig::Mse(MseConfig { prediction, target })
        }
        "cross_entropy" => {
            o.allow(&["kind", "prediction", "target", "label_smoothing"], &what)?;
            let (prediction, target) = names(&o)?;
            LossConfig::CrossEntropy(CrossEntropyConfig {
                prediction,
                target,
                label_smoothing: o.opt_f64("label_smoothing", 0.0)?,
            })
        }
        "binary_cross_entropy" => {
            o.allow(&["kind", "prediction", "target", "with_logits"], &what)?;
            let (prediction, target) = names(&o)?;
            LossConfig::BinaryCrossEntropy(BinaryCrossEntropyConfig {
                prediction,
                target,
                with_logits: o.opt_bool("with_logits", false)?,
            })
        }
        other => return kind_error(&o, other, KINDS, "loss"),
    })
}

fn read_metric(v: &Json, at: &str) -> Read<MetricConfig> {
    let o = Obj::new(v, at, "a metric")?;
    let kind = o.req_str("kind")?;
    match kind {
        "accuracy" => {
            o.allow(
                &["kind", "prediction", "target", "name", "threshold"],
                "an `accuracy` metric",
            )?;
            Ok(MetricConfig::Accuracy(AccuracyConfig {
                prediction: o.req_str("prediction")?.to_owned(),
                target: o.req_str("target")?.to_owned(),
                name: o.opt_str("name", "accuracy")?,
                threshold: o.nullable("threshold", None, |v, at| as_f64(v, at, "threshold"))?,
            }))
        }
        "loss" => error(
            &o.ptr("kind"),
            "the loss is always reported as the metric `loss` and is not listed in `metrics`",
        ),
        other => kind_error(&o, other, &["accuracy"], "metric"),
    }
}

fn read_optimizer(v: &Json, at: &str) -> Read<OptimizerConfig> {
    const KINDS: &[&str] = &["sgd", "adam", "adamw"];
    let o = Obj::new(v, at, "`optimizer`")?;
    let kind = o.req_str("kind")?;
    let what = format!("an `{kind}` optimizer");
    let read_betas = |o: &Obj<'_>, default: [f64; 2]| -> Read<[f64; 2]> {
        let Some(v) = o.get("betas") else {
            return Ok(default);
        };
        let at = o.ptr("betas");
        let items = as_array(v, &at, "betas")?;
        let [b1, b2] = items else {
            return error(
                &at,
                format!("`betas` must be [beta1, beta2], got {} values", items.len()),
            );
        };
        Ok([
            as_f64(b1, &format!("{at}/0"), "betas[0]")?,
            as_f64(b2, &format!("{at}/1"), "betas[1]")?,
        ])
    };
    Ok(match kind {
        "sgd" => {
            o.allow(
                &[
                    "kind",
                    "lr",
                    "momentum",
                    "dampening",
                    "nesterov",
                    "weight_decay",
                ],
                &what,
            )?;
            let d = SgdConfig::default();
            OptimizerConfig::Sgd(SgdConfig {
                lr: o.opt_f64("lr", d.lr)?,
                momentum: o.opt_f64("momentum", d.momentum)?,
                dampening: o.opt_f64("dampening", d.dampening)?,
                nesterov: o.opt_bool("nesterov", d.nesterov)?,
                weight_decay: o.opt_f64("weight_decay", d.weight_decay)?,
            })
        }
        "adam" => {
            o.allow(&["kind", "lr", "betas", "eps", "weight_decay"], &what)?;
            let d = AdamConfig::default();
            OptimizerConfig::Adam(AdamConfig {
                lr: o.opt_f64("lr", d.lr)?,
                betas: read_betas(&o, d.betas)?,
                eps: o.opt_f64("eps", d.eps)?,
                weight_decay: o.opt_f64("weight_decay", d.weight_decay)?,
            })
        }
        "adamw" => {
            o.allow(&["kind", "lr", "betas", "eps", "weight_decay"], &what)?;
            let d = AdamWConfig::default();
            OptimizerConfig::Adamw(AdamWConfig {
                lr: o.opt_f64("lr", d.lr)?,
                betas: read_betas(&o, d.betas)?,
                eps: o.opt_f64("eps", d.eps)?,
                weight_decay: o.opt_f64("weight_decay", d.weight_decay)?,
            })
        }
        other => return kind_error(&o, other, KINDS, "optimizer"),
    })
}

fn read_schedule(v: &Json, at: &str) -> Read<ScheduleConfig> {
    const KINDS: &[&str] = &["constant", "step", "cosine", "linear_warmup"];
    let o = Obj::new(v, at, "`schedule`")?;
    let kind = o.req_str("kind")?;
    let what = format!("a `{kind}` schedule");
    Ok(match kind {
        "constant" => {
            o.allow(&["kind"], &what)?;
            ScheduleConfig::Constant {}
        }
        "step" => {
            o.allow(&["kind", "every", "gamma"], &what)?;
            ScheduleConfig::Step {
                every: o.req_u64("every")?,
                gamma: o.opt_f64("gamma", 0.1)?,
            }
        }
        "cosine" => {
            o.allow(&["kind", "warmup_steps", "min_lr"], &what)?;
            ScheduleConfig::Cosine {
                warmup_steps: o.opt_u64("warmup_steps", 0)?,
                min_lr: o.opt_f64("min_lr", 0.0)?,
            }
        }
        "linear_warmup" => {
            o.allow(&["kind", "warmup_steps"], &what)?;
            ScheduleConfig::LinearWarmup {
                warmup_steps: o.req_u64("warmup_steps")?,
            }
        }
        other => return kind_error(&o, other, KINDS, "schedule"),
    })
}

fn read_clip(v: &Json, at: &str) -> Read<GradientClipConfig> {
    let o = Obj::new(v, at, "`gradient_clip`")?;
    o.allow(&["max_norm"], "`gradient_clip`")?;
    Ok(GradientClipConfig {
        max_norm: o.req_f64("max_norm")?,
    })
}

fn read_checkpoint(v: &Json, at: &str) -> Read<CheckpointConfig> {
    let o = Obj::new(v, at, "`checkpoint`")?;
    o.allow(
        &[
            "dir",
            "every_epochs",
            "every_steps",
            "keep_last",
            "keep_best",
            "on_cancel",
        ],
        "`checkpoint`",
    )?;
    let int = |field: &'static str| move |v: &Json, at: &str| as_u64(v, at, field);
    Ok(CheckpointConfig {
        dir: o.req_str("dir")?.to_owned(),
        every_epochs: o.nullable("every_epochs", Some(1), int("every_epochs"))?,
        every_steps: o.nullable("every_steps", None, int("every_steps"))?,
        keep_last: o.nullable("keep_last", Some(3), int("keep_last"))?,
        keep_best: o.nullable("keep_best", None, |v, at| {
            as_str(v, at, "keep_best").map(str::to_owned)
        })?,
        on_cancel: o.opt_bool("on_cancel", true)?,
    })
}

#[cfg(test)]
#[path = "training_config_tests.rs"]
mod tests;
