//! Datasets, shuffling and batching (`docs/specs/training.md` §2.3, §2.4, §3.5, §3.6).
//!
//! A [`Dataset`] is the §3.5 model: named fields, each a host [`TensorData`] of shape
//! `[n] + sample shape` with dtype `f32`, `i64`, `i32` or `bool`, all with the same sample count
//! `n`, in dataset order. One reader per kind builds it:
//!
//! - `tensors` (§3.5.1): in-memory fields from the app, [`InMemoryDataset`];
//! - `safetensors` (§3.5.2): a file read through `polypous-formats`, only the used tensors;
//! - `csv` (§3.5.3): an RFC 4180 file with Polypous' own parser ([`csv`](self) module);
//! - `image_folder` (§3.5.4): PNG files in class folders, decoded with the `png` crate;
//! - `split` (§3.5.5): a seeded subset of the training set as the validation set.
//!
//! [`read_datasets`] reads both roles of a [`DataConfig`] under one memory budget
//! (`data.max_bytes`) and computes the `data_fingerprint` of §5.2. [`Batcher`] gives the order of
//! any epoch and any batch of it directly from the seed (§3.6), so resuming needs no saved data
//! state.
//!
//! Every file is untrusted: sizes are checked before the memory they need is reserved, and every
//! reservation is fallible ([`Vec::try_reserve`]), so a hostile file is a [`DataError`] of kind
//! [`DataErrorKind::InvalidInput`] (malformed, over a limit) or [`DataErrorKind::OutOfMemory`]
//! (the host can't give memory below the limit), never a panic or an abort.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use polypous_backend::{DType, TensorData};
use polypous_runtime::CancellationToken;

mod batch;
mod config;
mod csv;
mod image_folder;
mod png_decode;
mod safetensors;
mod split;
mod tensors;

pub use batch::{Batcher, SHUFFLE_LABEL, SPLIT_LABEL, gather, permutation};
pub use config::{
    ColumnSource, CsvColumnConfig, CsvDatasetConfig, CsvDtype, DEFAULT_MAX_BYTES, DataConfig,
    DatasetConfig, ImageFolderConfig, MAX_CSV_COLUMNS, MAX_CSV_FIELDS, MAX_DATA_BYTES,
    MAX_IMAGE_SIDE, MAX_NAME_BYTES, MAX_PATH_BYTES, SafetensorsDatasetConfig, SplitConfig,
    TensorsDatasetConfig,
};
pub use split::split;

/// The most samples a dataset may have (§3.5: `2^32`).
pub const MAX_SAMPLES: u64 = 1 << 32;
/// The largest rank of a `tensors` or `safetensors` field, with the sample dim (§3.5.1, §3.5.2).
pub const MAX_FIELD_RANK: usize = 7;

/// What kind of failure a [`DataError`] is. #82 maps it to `polypous::ErrorKind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataErrorKind {
    /// Malformed data, over a limit, or unsupported (§6: `InvalidInput`).
    InvalidInput,
    /// A file or directory can't be read (§6: `Io`).
    Io,
    /// The host couldn't reserve memory below `data.max_bytes` (§6: `OutOfMemory`).
    OutOfMemory,
    /// The cancellation token was set between files (§2.10: `Cancelled`).
    Cancelled,
}

/// Why a dataset couldn't be read (training.md §3.5, §6). Besides the kind and the message it
/// carries the context #82 puts into `polypous::ErrorContext`: the dataset's role, the file, the
/// field or tensor, the CSV record and column.
#[derive(Debug)]
pub struct DataError(Box<Inner>);

#[derive(Debug)]
struct Inner {
    kind: DataErrorKind,
    message: String,
    dataset: Option<String>,
    file: Option<PathBuf>,
    field: Option<String>,
    record: Option<u64>,
    column: Option<String>,
    source: Option<std::io::Error>,
    no_used_fields: bool,
}

impl fmt::Display for Inner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(dataset) = &self.dataset {
            write!(f, "dataset `{dataset}`: ")?;
        }
        if let Some(file) = &self.file {
            write!(f, "{}: ", file.display())?;
        }
        f.write_str(&self.message)
    }
}

impl DataError {
    /// The kind of failure.
    #[must_use]
    pub fn kind(&self) -> DataErrorKind {
        self.0.kind
    }

    /// What is wrong, suitable for users, without the role and file (see `Display` for both).
    /// It names the record number and column (CSV), the tensor (safetensors), the class folder
    /// or the sizes (images) where §3.5 says so.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// The dataset's role (`"train"` or `"validation"`), once known.
    #[must_use]
    pub fn dataset(&self) -> Option<&str> {
        self.0.dataset.as_deref()
    }

    /// The file or directory concerned.
    #[must_use]
    pub fn file(&self) -> Option<&Path> {
        self.0.file.as_deref()
    }

    /// The field or tensor concerned.
    #[must_use]
    pub fn field(&self) -> Option<&str> {
        self.0.field.as_deref()
    }

    /// The CSV record number (the header is record 1).
    #[must_use]
    pub fn record(&self) -> Option<u64> {
        self.0.record
    }

    /// The CSV column name.
    #[must_use]
    pub fn column(&self) -> Option<&str> {
        self.0.column.as_deref()
    }

    /// Whether the dataset has none of the fields the training uses ([`FieldSelection::Only`]).
    /// Such a dataset has no sample count, so the reader fails; the trainer reports it as the
    /// first used field missing from the dataset, with its config pointer (training.md §2.10
    /// step 5).
    #[must_use]
    pub fn has_no_used_fields(&self) -> bool {
        self.0.no_used_fields
    }

    /// The I/O error behind a [`DataErrorKind::Io`] error.
    #[must_use]
    pub fn io_error(&self) -> Option<&std::io::Error> {
        self.0.source.as_ref()
    }

    fn new(kind: DataErrorKind, message: impl Into<String>) -> Self {
        Self(Box::new(Inner {
            kind,
            message: message.into(),
            dataset: None,
            file: None,
            field: None,
            record: None,
            column: None,
            source: None,
            no_used_fields: false,
        }))
    }

    /// `InvalidInput`: the dataset has none of the fields the training uses (see
    /// [`DataError::has_no_used_fields`]).
    pub(crate) fn no_used_fields(message: impl Into<String>) -> Self {
        let mut e = Self::invalid(message);
        e.0.no_used_fields = true;
        e
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(DataErrorKind::InvalidInput, message)
    }

    pub(crate) fn out_of_memory(bytes: u64, what: &str) -> Self {
        Self::new(
            DataErrorKind::OutOfMemory,
            format!("could not reserve {bytes} bytes of host memory for {what}"),
        )
    }

    pub(crate) fn io(err: std::io::Error, path: &Path, what: &str) -> Self {
        let mut e = Self::new(DataErrorKind::Io, format!("could not {what}: {err}"));
        e.0.file = Some(path.to_path_buf());
        e.0.source = Some(err);
        e
    }

    pub(crate) fn cancelled() -> Self {
        Self::new(
            DataErrorKind::Cancelled,
            "cancelled while reading the datasets",
        )
    }

    pub(crate) fn with_file(mut self, path: &Path) -> Self {
        if self.0.file.is_none() {
            self.0.file = Some(path.to_path_buf());
        }
        self
    }

    pub(crate) fn with_field(mut self, name: &str) -> Self {
        if self.0.field.is_none() {
            self.0.field = Some(name.to_owned());
        }
        self
    }

    pub(crate) fn with_dataset(mut self, role: &str) -> Self {
        if self.0.dataset.is_none() {
            self.0.dataset = Some(role.to_owned());
        }
        self
    }

    pub(crate) fn with_record(mut self, record: u64, column: Option<&str>) -> Self {
        self.0.record = Some(record);
        self.0.column = column.map(str::to_owned);
        self
    }
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for DataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0
            .source
            .as_ref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

/// Which fields of a dataset the trainer uses (§3.5: the others are ignored and, where the format
/// allows, not read). The trainer (#82) passes the fields named by `inputs`, `loss.target` and the
/// metrics' `target`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum FieldSelection {
    /// Every field of the dataset.
    #[default]
    All,
    /// Only these fields. Names that the dataset doesn't have are not an error here; the trainer
    /// reports a missing field with its config pointer (§2.10 step 5).
    Only(BTreeSet<String>),
}

impl FieldSelection {
    /// Whether `name` is read.
    #[must_use]
    pub fn wants(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.contains(name),
        }
    }
}

/// The memory budget `data.max_bytes` of both sets together (§3.5, "Memory").
#[derive(Debug)]
pub(crate) struct Budget {
    limit: u64,
    used: u64,
}

impl Budget {
    pub(crate) fn new(limit: u64) -> Self {
        Self { limit, used: 0 }
    }

    /// Counts `bytes` more decoded bytes, or fails with `InvalidInput` if that exceeds the limit.
    pub(crate) fn charge(&mut self, bytes: u64) -> Result<(), DataError> {
        match self.used.checked_add(bytes) {
            Some(total) if total <= self.limit => {
                self.used = total;
                Ok(())
            }
            _ => Err(DataError::invalid(format!(
                "the decoded datasets need more than `data.max_bytes` = {} bytes \
                 ({} bytes already read, {bytes} more needed)",
                self.limit, self.used
            ))),
        }
    }
}

/// A `Vec<u8>` with room for exactly `bytes` bytes, reserved fallibly.
pub(crate) fn try_bytes(bytes: u64, what: &str) -> Result<Vec<u8>, DataError> {
    let len = usize::try_from(bytes).map_err(|_| DataError::out_of_memory(bytes, what))?;
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| DataError::out_of_memory(bytes, what))?;
    Ok(v)
}

/// A dataset (training.md §3.5): named fields with the same sample count, on the host, in dataset
/// order. Each field is a [`TensorData`] of shape `[n] + sample shape`, validated, with dtype
/// `f32`, `i64`, `i32` or `bool`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dataset {
    len: u64,
    fields: BTreeMap<String, TensorData>,
}

impl Dataset {
    /// Builds a dataset from fields that were checked by a reader: every field has rank ≥ 1, dim
    /// 0 equal to `len`, a field dtype and valid bytes.
    pub(crate) fn from_checked(len: u64, fields: BTreeMap<String, TensorData>) -> Self {
        debug_assert!(fields.values().all(|t| t.shape.first() == Some(&len)));
        Self { len, fields }
    }

    /// The sample count `n` (1 to `2^32`).
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Always `false`: a dataset has at least one sample. (For clippy's `len_without_is_empty`.)
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The fields, in name order.
    pub fn fields(&self) -> impl Iterator<Item = (&str, &TensorData)> + '_ {
        self.fields.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// The field's data, shape `[n] + sample shape`.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&TensorData> {
        self.fields.get(name)
    }

    /// The field's dtype and sample shape (its shape without dim 0).
    #[must_use]
    pub fn field_info(&self, name: &str) -> Option<(DType, &[u64])> {
        self.fields
            .get(name)
            .map(|t| (t.dtype, t.shape.get(1..).unwrap_or(&[])))
    }

    /// The bytes the fields take together.
    #[must_use]
    pub fn byte_len(&self) -> u64 {
        self.fields.values().map(|t| t.bytes.len() as u64).sum()
    }

    pub(crate) fn into_fields(self) -> BTreeMap<String, TensorData> {
        self.fields
    }
}

/// A dataset given by the app (`TrainerOptions::datasets`, training.md §3.5.1). The facade (#83)
/// converts its `InMemoryDataset { name, fields: Vec<NamedTensor> }` into this.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InMemoryDataset {
    /// The name a `tensors` dataset selects it by.
    pub name: String,
    /// The fields: a name and its data, dim 0 the sample dim.
    pub fields: Vec<(String, TensorData)>,
}

/// What [`read_datasets`] needs besides the config.
#[derive(Debug, Default)]
pub struct DataSources {
    /// The directory relative paths are resolved against (§2.1, `TrainerOptions::base_dir`).
    pub base_dir: PathBuf,
    /// The `tensors` datasets (§3.5.1). Their fields are moved into the result; a dataset used by
    /// both roles is copied for the second.
    pub in_memory: Vec<InMemoryDataset>,
    /// The fields the trainer uses (the same selection for both sets).
    pub fields: FieldSelection,
}

/// The datasets of a training run, as read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Datasets {
    /// The training set (after a `split`, without the validation samples).
    pub train: Dataset,
    /// The validation set, if the config has one.
    pub validation: Option<Dataset>,
    /// `data_fingerprint` (training.md §5.2): FNV-1a over the datasets as read, before a split.
    pub fingerprint: u64,
}

impl Datasets {
    /// `data_fingerprint` as written in `state.json`: 16 lowercase hex digits (§5.2).
    #[must_use]
    pub fn fingerprint_hex(&self) -> String {
        format!("{:016x}", self.fingerprint)
    }
}

/// Reads the training and validation sets of `config` (training.md §3.5), under one budget of
/// `config.max_bytes` for both, then applies a `split` validation set with `seed` (§3.5.5).
///
/// The config is validated first (`DataConfig`'s range checks); an invalid one is `InvalidInput`
/// whose message starts with the pointer relative to `data`. `cancel` is checked before each
/// dataset and between files.
///
/// # Errors
/// A [`DataError`]; its `dataset` is the role being read.
pub fn read_datasets(
    config: &DataConfig,
    seed: u64,
    mut sources: DataSources,
    cancel: &CancellationToken,
) -> Result<Datasets, DataError> {
    config
        .validate()
        .map_err(|e| DataError::invalid(format!("invalid `data` config: {e}")))?;
    let mut budget = Budget::new(config.max_bytes);
    // A `tensors` dataset used by both roles is copied for the first and moved for the second.
    let shared = match (&config.train, &config.validation) {
        (DatasetConfig::Tensors(a), Some(DatasetConfig::Tensors(b))) => a.name == b.name,
        _ => false,
    };
    let train = read_role(
        &config.train,
        "train",
        &mut sources,
        &mut budget,
        cancel,
        shared,
    )?;
    let validation = match &config.validation {
        None | Some(DatasetConfig::Split(_)) => None,
        Some(dataset) => Some(read_role(
            dataset,
            "validation",
            &mut sources,
            &mut budget,
            cancel,
            false,
        )?),
    };
    let mut hasher = Fnv1a::new();
    hasher.dataset(&train);
    if let Some(validation) = &validation {
        hasher.dataset(validation);
    }
    let fingerprint = hasher.finish();
    let (train, validation) = match &config.validation {
        Some(DatasetConfig::Split(cfg)) => {
            let (train, validation) =
                split(train, cfg.fraction, seed).map_err(|e| e.with_dataset("validation"))?;
            (train, Some(validation))
        }
        _ => (train, validation),
    };
    Ok(Datasets {
        train,
        validation,
        fingerprint,
    })
}

/// Reads one dataset (any kind but `split`).
fn read_role(
    config: &DatasetConfig,
    role: &str,
    sources: &mut DataSources,
    budget: &mut Budget,
    cancel: &CancellationToken,
    keep: bool,
) -> Result<Dataset, DataError> {
    if cancel.is_cancelled() {
        return Err(DataError::cancelled());
    }
    let base = sources.base_dir.as_path();
    let fields = &sources.fields;
    let result = match config {
        DatasetConfig::Tensors(cfg) => {
            tensors::read(&cfg.name, &mut sources.in_memory, fields, budget, keep)
        }
        DatasetConfig::Safetensors(cfg) => safetensors::read(&base.join(&cfg.path), fields, budget),
        DatasetConfig::Csv(cfg) => csv::read(&base.join(&cfg.path), &cfg.columns, fields, budget),
        DatasetConfig::ImageFolder(cfg) => {
            image_folder::read(&base.join(&cfg.path), cfg, fields, budget, cancel)
        }
        DatasetConfig::Split(_) => Err(DataError::invalid(
            "a `split` dataset can only be the validation set",
        )),
    };
    result.map_err(|e| e.with_dataset(role))
}

/// The 64-bit FNV-1a hash over bytes (graph-ir.md §3.1.12's `fnv1a`, which
/// `polypous_ir::rng::fnv1a` computes over a string's bytes).
pub(crate) struct Fnv1a(u64);

impl Fnv1a {
    pub(crate) fn new() -> Self {
        Self(0xCBF2_9CE4_8422_2325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }

    /// §5.2: per field in name order, the name, `0x00`, the dtype name, `0x00`, the shape as
    /// 8-byte little-endian integers, and the data bytes.
    pub(crate) fn dataset(&mut self, dataset: &Dataset) {
        for (name, data) in dataset.fields() {
            self.update(name.as_bytes());
            self.update(&[0]);
            self.update(data.dtype.name().as_bytes());
            self.update(&[0]);
            for &dim in &data.shape {
                self.update(&dim.to_le_bytes());
            }
            self.update(&data.bytes);
        }
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}

/// Checks the sample count of a field (dim 0) against the other fields' and the §3.5 range.
pub(crate) fn check_count(len: &mut Option<u64>, name: &str, n: u64) -> Result<(), DataError> {
    if n == 0 || n > MAX_SAMPLES {
        return Err(DataError::invalid(format!(
            "field `{name}` has {n} samples; a dataset has 1 to 2^32 samples"
        ))
        .with_field(name));
    }
    match *len {
        Some(m) if m != n => Err(DataError::invalid(format!(
            "field `{name}` has {n} samples, but the dataset's other fields have {m}; every field \
             must have the same sample count"
        ))
        .with_field(name)),
        _ => {
            *len = Some(n);
            Ok(())
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
