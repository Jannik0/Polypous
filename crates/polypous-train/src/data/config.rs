//! The `data` object of a training config and its dataset objects (`docs/specs/training.md` §2.1,
//! §2.3, §2.4), as Rust types.
//!
//! As for the other config types ([`crate::config`]), serde checks the JSON shape (unknown fields,
//! wrong JSON types, an unknown `kind`) and fills in defaults; `validate` checks the ranges and
//! returns a [`ConfigError`] whose pointer is relative to the `data` object (`/train/image_size/0`).
//! The config parser (#82) prefixes `/data`.

use std::collections::BTreeMap;
use std::fmt;

use serde::de::{MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};

use crate::config::ConfigError;

/// The default of `data.max_bytes`: 4 GiB (§2.3).
pub const DEFAULT_MAX_BYTES: u64 = 1 << 32;
/// The largest `data.max_bytes`: `2^40` (§2.3).
pub const MAX_DATA_BYTES: u64 = 1 << 40;
/// Names (fields, datasets) are 1–256 bytes (§2.1).
pub const MAX_NAME_BYTES: usize = 256;
/// Paths are 1–4,096 bytes (§2.1).
pub const MAX_PATH_BYTES: usize = 4096;
/// Each of `image_size` is 1–4,096 (§2.4).
pub const MAX_IMAGE_SIDE: u64 = 4096;
/// A CSV `columns` object has at most 1,024 fields (§2.1).
pub const MAX_CSV_FIELDS: usize = 1024;
/// A CSV `columns` object names at most 16,384 columns in all, and one field 1–16,384 (§2.1,
/// §3.5.3). A CSV file has at most as many columns.
pub const MAX_CSV_COLUMNS: usize = 16_384;

/// `data` (training.md §2.3).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// The training set: any kind except `split`. Required.
    pub train: DatasetConfig,
    /// The validation set (any kind), or none. Default null.
    #[serde(default)]
    pub validation: Option<DatasetConfig>,
    /// Shuffle the training samples every epoch (§3.6). Default `true`.
    #[serde(default = "default_shuffle")]
    pub shuffle: bool,
    /// The most host memory the decoded datasets may take together, 1 to `2^40`. Default 4 GiB.
    #[serde(default = "default_max_bytes")]
    pub max_bytes: u64,
}

fn default_shuffle() -> bool {
    true
}

fn default_max_bytes() -> u64 {
    DEFAULT_MAX_BYTES
}

/// A dataset object (training.md §2.4, §3.5). Its JSON form is internally tagged:
/// `{"kind": "image_folder", "path": "data/train", ...}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[non_exhaustive]
pub enum DatasetConfig {
    /// `tensors` (§3.5.1).
    Tensors(TensorsDatasetConfig),
    /// `safetensors` (§3.5.2).
    Safetensors(SafetensorsDatasetConfig),
    /// `csv` (§3.5.3).
    Csv(CsvDatasetConfig),
    /// `image_folder` (§3.5.4).
    ImageFolder(ImageFolderConfig),
    /// `split` (§3.5.5).
    Split(SplitConfig),
}

/// `tensors`: an in-memory dataset of `TrainerOptions::datasets` (§3.5.1).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TensorsDatasetConfig {
    /// The dataset's name, 1–256 bytes. Required.
    pub name: String,
}

/// `safetensors`: a file of named fields (§3.5.2).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SafetensorsDatasetConfig {
    /// The file, 1–4,096 bytes; relative to the base directory. Required.
    pub path: String,
}

/// `csv`: a CSV file and the fields made of its columns (§3.5.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CsvDatasetConfig {
    /// The file, 1–4,096 bytes; relative to the base directory. Required.
    pub path: String,
    /// Field name → its columns and dtype. Required. A JSON key given twice is an error.
    #[serde(deserialize_with = "unique_keys")]
    pub columns: BTreeMap<String, CsvColumnConfig>,
}

/// One field of a `csv` dataset: `{"from": <column> | [<column>, …], "dtype": …}` (§3.5.3).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CsvColumnConfig {
    /// The column (sample shape `[]`) or columns (sample shape `[k]`, in this order). Required.
    pub from: ColumnSource,
    /// The field's dtype. Required.
    pub dtype: CsvDtype,
}

/// The `from` of a CSV field: one column name, or a list of 1–16,384.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ColumnSource {
    /// One column: sample shape `[]`.
    One(String),
    /// Several columns: sample shape `[k]`.
    Many(Vec<String>),
}

impl ColumnSource {
    /// The column names, in order.
    #[must_use]
    pub fn columns(&self) -> &[String] {
        match self {
            Self::One(name) => std::slice::from_ref(name),
            Self::Many(names) => names,
        }
    }
}

/// The dtype of a CSV field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CsvDtype {
    /// `f32`: a finite decimal number.
    F32,
    /// `i64`: a decimal integer.
    I64,
    /// `i32`: a decimal integer.
    I32,
}

/// `image_folder` (§3.5.4).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageFolderConfig {
    /// The folder of class folders, 1–4,096 bytes; relative to the base directory. Required.
    pub path: String,
    /// `[H, W]`, each 1–4,096: every image must have exactly this size. Required.
    pub image_size: [u64; 2],
    /// `1` (gray) or `3` (RGB). Default `1`.
    #[serde(default = "default_channels")]
    pub channels: u64,
    /// The name of the image field, 1–256 bytes. Default `"image"`.
    #[serde(default = "default_image_field")]
    pub image_field: String,
    /// The name of the label field, 1–256 bytes. Default `"label"`.
    #[serde(default = "default_label_field")]
    pub label_field: String,
}

fn default_channels() -> u64 {
    1
}

fn default_image_field() -> String {
    "image".to_owned()
}

fn default_label_field() -> String {
    "label".to_owned()
}

/// `split` (§3.5.5): a seeded subset of the training set as the validation set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitConfig {
    /// The role the samples come from: `"train"` only. Default `"train"`.
    #[serde(default = "default_from")]
    pub from: String,
    /// `0 < fraction < 1`. Required.
    pub fraction: f64,
}

fn default_from() -> String {
    "train".to_owned()
}

/// Deserializes a JSON object into a map, rejecting a key given twice (a strict object, §2.1).
fn unique_keys<'de, D>(deserializer: D) -> Result<BTreeMap<String, CsvColumnConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    struct UniqueKeys;

    impl<'de> Visitor<'de> for UniqueKeys {
        type Value = BTreeMap<String, CsvColumnConfig>;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("an object of field names to column objects")
        }

        fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut out = BTreeMap::new();
            while let Some((key, value)) = map.next_entry::<String, CsvColumnConfig>()? {
                if out.contains_key(&key) {
                    return Err(serde::de::Error::custom(format!(
                        "the field `{key}` is given twice"
                    )));
                }
                out.insert(key, value);
            }
            Ok(out)
        }
    }

    deserializer.deserialize_map(UniqueKeys)
}

/// A JSON pointer token (RFC 6901): `~` → `~0`, `/` → `~1`.
fn token(key: &str) -> String {
    key.replace('~', "~0").replace('/', "~1")
}

fn check_len(
    pointer: &str,
    what: &str,
    value: &str,
    max: usize,
    unit: &str,
) -> Result<(), ConfigError> {
    if value.is_empty() || value.len() > max {
        return Err(ConfigError {
            pointer: pointer.to_owned(),
            message: format!(
                "`{what}` must be 1–{max} bytes long ({unit}), got {} bytes",
                value.len()
            ),
        });
    }
    Ok(())
}

fn err(pointer: String, message: impl Into<String>) -> ConfigError {
    ConfigError {
        pointer,
        message: message.into(),
    }
}

impl DataConfig {
    /// Checks the ranges of training.md §2.1, §2.3 and §2.4. Pointers are relative to `data`.
    ///
    /// # Errors
    /// The first failure: `train` before `validation`, then `max_bytes`.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if matches!(self.train, DatasetConfig::Split(_)) {
            return Err(err(
                "/train/kind".to_owned(),
                "the training set can't be a `split`; only `validation` can",
            ));
        }
        self.train.validate("/train")?;
        if let Some(validation) = &self.validation {
            validation.validate("/validation")?;
        }
        if !(1..=MAX_DATA_BYTES).contains(&self.max_bytes) {
            return Err(err(
                "/max_bytes".to_owned(),
                format!(
                    "`max_bytes` must be 1 to 2^40 ({MAX_DATA_BYTES}), got {}",
                    self.max_bytes
                ),
            ));
        }
        Ok(())
    }
}

impl DatasetConfig {
    /// Checks the fields of one dataset object; `at` is its pointer (`/train`).
    ///
    /// # Errors
    /// The first out-of-range field, in table order.
    pub fn validate(&self, at: &str) -> Result<(), ConfigError> {
        match self {
            Self::Tensors(cfg) => check_len(
                &format!("{at}/name"),
                "name",
                &cfg.name,
                MAX_NAME_BYTES,
                "a name",
            ),
            Self::Safetensors(cfg) => check_len(
                &format!("{at}/path"),
                "path",
                &cfg.path,
                MAX_PATH_BYTES,
                "a path",
            ),
            Self::Csv(cfg) => cfg.validate(at),
            Self::ImageFolder(cfg) => cfg.validate(at),
            Self::Split(cfg) => cfg.validate(at),
        }
    }
}

impl CsvDatasetConfig {
    fn validate(&self, at: &str) -> Result<(), ConfigError> {
        check_len(
            &format!("{at}/path"),
            "path",
            &self.path,
            MAX_PATH_BYTES,
            "a path",
        )?;
        let pointer = format!("{at}/columns");
        if self.columns.len() > MAX_CSV_FIELDS {
            return Err(err(
                pointer,
                format!(
                    "`columns` has {} fields; at most {MAX_CSV_FIELDS} are allowed",
                    self.columns.len()
                ),
            ));
        }
        let mut total = 0usize;
        for (name, column) in &self.columns {
            let at_field = format!("{pointer}/{}", token(name));
            check_len(&at_field, "a field name", name, MAX_NAME_BYTES, "a name")?;
            if let ColumnSource::Many(names) = &column.from
                && !(1..=MAX_CSV_COLUMNS).contains(&names.len())
            {
                return Err(err(
                    format!("{at_field}/from"),
                    format!(
                        "`from` must list 1–{MAX_CSV_COLUMNS} columns, got {}",
                        names.len()
                    ),
                ));
            }
            total += column.from.columns().len();
        }
        if total > MAX_CSV_COLUMNS {
            return Err(err(
                pointer,
                format!(
                    "`columns` names {total} columns in all; at most {MAX_CSV_COLUMNS} are allowed"
                ),
            ));
        }
        Ok(())
    }
}

impl ImageFolderConfig {
    fn validate(&self, at: &str) -> Result<(), ConfigError> {
        check_len(
            &format!("{at}/path"),
            "path",
            &self.path,
            MAX_PATH_BYTES,
            "a path",
        )?;
        for (i, side) in self.image_size.into_iter().enumerate() {
            if !(1..=MAX_IMAGE_SIDE).contains(&side) {
                return Err(err(
                    format!("{at}/image_size/{i}"),
                    format!("`image_size[{i}]` must be 1–{MAX_IMAGE_SIDE}, got {side}"),
                ));
            }
        }
        if self.channels != 1 && self.channels != 3 {
            return Err(err(
                format!("{at}/channels"),
                format!("`channels` must be 1 or 3, got {}", self.channels),
            ));
        }
        check_len(
            &format!("{at}/image_field"),
            "image_field",
            &self.image_field,
            MAX_NAME_BYTES,
            "a name",
        )?;
        check_len(
            &format!("{at}/label_field"),
            "label_field",
            &self.label_field,
            MAX_NAME_BYTES,
            "a name",
        )?;
        if self.image_field == self.label_field {
            return Err(err(
                format!("{at}/label_field"),
                format!(
                    "`label_field` must differ from `image_field`; both are `{}`",
                    self.label_field
                ),
            ));
        }
        Ok(())
    }
}

impl SplitConfig {
    fn validate(&self, at: &str) -> Result<(), ConfigError> {
        if self.from != "train" {
            return Err(err(
                format!("{at}/from"),
                format!(
                    "`from` must be \"train\" (the only role a split can take samples from), got {:?}",
                    self.from
                ),
            ));
        }
        if !(self.fraction > 0.0 && self.fraction < 1.0) {
            return Err(err(
                format!("{at}/fraction"),
                format!("`fraction` must be in (0, 1), got {}", self.fraction),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
