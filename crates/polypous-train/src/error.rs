//! The error of training (`docs/specs/training.md` §6).
//!
//! [`TrainError`] carries one of the §6 kinds ([`TrainErrorKind`]), which the facade (#83) maps
//! 1:1 to `polypous::ErrorKind`, a message for users, and the context fields of
//! `polypous::ErrorContext`: `pointer`, `tensor`, `node`, `file`, `device`, `details`. The errors
//! of the other modules (datasets, losses, metrics, targets, optimizers, the runtime and the
//! backend) are converted here, so that every failure of the trainer is one type and never a
//! panic.

use std::fmt;
use std::path::{Path, PathBuf};

use polypous_backend::TensorError;
use polypous_runtime::{ErrorKind as RuntimeKind, RuntimeError};

use crate::config::ConfigError;
use crate::data::{DataError, DataErrorKind};
use crate::json::ReadError;
use crate::loss::LossError;
use crate::metrics::MetricError;
use crate::optimizer::OptimizerError;

/// The kind of a [`TrainError`]: the `polypous::ErrorKind` the facade maps it to (training.md §6,
/// public-api.md §2 rule 7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TrainErrorKind {
    /// The config, graph references, data, weights or a checkpoint don't fit; `run`, `export`
    /// in the wrong state.
    InvalidInput,
    /// `precision` `f16`/`bf16`; a graph with state; a shared running statistic.
    UnsupportedOperation,
    /// The device can't be used.
    BackendUnavailable,
    /// A file or directory can't be read, created or written.
    Io,
    /// The host (or the device, where the backend reports it) is out of memory.
    OutOfMemory,
    /// The backend failed.
    Backend,
    /// The trainer's creation was cancelled.
    Cancelled,
    /// A broken invariant: a bug in Polypous.
    Internal,
}

impl TrainErrorKind {
    /// The kind's name, as `polypous::ErrorKind`'s JSON writes it (`invalid_input`, …).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::InvalidInput => "invalid_input",
            Self::UnsupportedOperation => "unsupported_operation",
            Self::BackendUnavailable => "backend_unavailable",
            Self::Io => "io",
            Self::OutOfMemory => "out_of_memory",
            Self::Backend => "backend",
            Self::Cancelled => "cancelled",
            Self::Internal => "internal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Inner {
    kind: TrainErrorKind,
    message: String,
    pointer: Option<String>,
    tensor: Option<String>,
    node: Option<String>,
    file: Option<PathBuf>,
    device: Option<String>,
    details: Option<String>,
}

/// An error of the training config, the trainer's creation, a run, a resume or an export
/// (training.md §6). Boxed, so `Result<_, TrainError>` stays one pointer wide.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainError(Box<Inner>);

impl TrainError {
    /// A new error of `kind` with a message for users.
    pub fn new(kind: TrainErrorKind, message: impl Into<String>) -> Self {
        Self(Box::new(Inner {
            kind,
            message: message.into(),
            pointer: None,
            tensor: None,
            node: None,
            file: None,
            device: None,
            details: None,
        }))
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::new(TrainErrorKind::InvalidInput, message)
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        Self::new(
            TrainErrorKind::Internal,
            "Polypous ran into an internal error. Please report this as a bug.",
        )
        .with_details(message)
    }

    pub(crate) fn io(err: &std::io::Error, what: &str, path: &Path) -> Self {
        Self::new(
            TrainErrorKind::Io,
            format!("could not {what} {}: {err}", path.display()),
        )
        .with_file(path)
        .with_details(err.to_string())
    }

    /// The kind.
    pub fn kind(&self) -> TrainErrorKind {
        self.0.kind
    }

    /// The message, for users.
    pub fn message(&self) -> &str {
        &self.0.message
    }

    /// `context.pointer`: the JSON pointer (RFC 6901) into the training config, from its root
    /// (`/optimizer/betas/1`; `""` for the whole document).
    pub fn pointer(&self) -> Option<&str> {
        self.0.pointer.as_deref()
    }

    /// `context.tensor`: a graph input or output, a dataset field, a param or a tensor of a file.
    pub fn tensor(&self) -> Option<&str> {
        self.0.tensor.as_deref()
    }

    /// `context.node`: a node path (`layers[3]/bn`).
    pub fn node(&self) -> Option<&str> {
        self.0.node.as_deref()
    }

    /// `context.file`: the file or directory concerned.
    pub fn file(&self) -> Option<&Path> {
        self.0.file.as_deref()
    }

    /// `context.device`: the device id text (`cuda:0`).
    pub fn device(&self) -> Option<&str> {
        self.0.device.as_deref()
    }

    /// `context.details`: technical details (a validation report, every param without a value,
    /// the source error).
    pub fn details(&self) -> Option<&str> {
        self.0.details.as_deref()
    }

    /// Sets `context.pointer`.
    #[must_use]
    pub fn with_pointer(mut self, pointer: impl Into<String>) -> Self {
        self.0.pointer = Some(pointer.into());
        self
    }

    /// Sets `context.tensor`.
    #[must_use]
    pub fn with_tensor(mut self, tensor: impl Into<String>) -> Self {
        self.0.tensor = Some(tensor.into());
        self
    }

    /// Sets `context.node`.
    #[must_use]
    pub fn with_node(mut self, node: impl Into<String>) -> Self {
        self.0.node = Some(node.into());
        self
    }

    /// Sets `context.file`.
    #[must_use]
    pub fn with_file(mut self, file: impl Into<PathBuf>) -> Self {
        self.0.file = Some(file.into());
        self
    }

    /// Sets `context.file` unless it is set already.
    #[must_use]
    pub(crate) fn or_file(mut self, file: &Path) -> Self {
        if self.0.file.is_none() {
            self.0.file = Some(file.to_path_buf());
        }
        self
    }

    /// Sets `context.device`.
    #[must_use]
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.0.device = Some(device.into());
        self
    }

    /// Sets `context.details`.
    #[must_use]
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.0.details = Some(details.into());
        self
    }

    /// Prefixes the message (`state.json: …`).
    #[must_use]
    pub(crate) fn prefixed(mut self, prefix: &str) -> Self {
        self.0.message = format!("{prefix}{}", self.0.message);
        self
    }

    /// A config error at `pointer` (from the root), `InvalidInput`.
    pub(crate) fn config(pointer: &str, message: &str) -> Self {
        let at = if pointer.is_empty() {
            "the config"
        } else {
            pointer
        };
        Self::invalid(format!("{at}: {message}")).with_pointer(pointer)
    }

    /// A [`ConfigError`] of a nested object whose own pointer is `prefix` (`/optimizer`).
    pub(crate) fn nested(prefix: &str, e: &ConfigError) -> Self {
        Self::config(&format!("{prefix}{}", e.pointer), &e.message)
    }
}

impl fmt::Display for TrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.message)
    }
}

impl std::error::Error for TrainError {}

impl From<ReadError> for TrainError {
    fn from(e: ReadError) -> Self {
        Self::config(&e.pointer, &e.message)
    }
}

/// The kind of a backend error (as the runtime maps it, runtime.md §5).
pub(crate) fn tensor_kind(e: &TensorError) -> TrainErrorKind {
    match e {
        TensorError::IndexOutOfRange { .. } => TrainErrorKind::InvalidInput,
        TensorError::UnsupportedDType { .. } | TensorError::UnsupportedOperation { .. } => {
            TrainErrorKind::UnsupportedOperation
        }
        TensorError::DeviceUnavailable { .. } => TrainErrorKind::BackendUnavailable,
        TensorError::Backend { .. } => TrainErrorKind::Backend,
        _ => TrainErrorKind::Internal,
    }
}

impl From<TensorError> for TrainError {
    fn from(e: TensorError) -> Self {
        let kind = tensor_kind(&e);
        if kind == TrainErrorKind::Internal {
            return Self::internal(e.to_string());
        }
        let mut out = Self::new(kind, e.to_string()).with_details(e.to_string());
        if let TensorError::DeviceUnavailable { device, .. } = &e {
            out = out.with_device(device.to_string());
        }
        out
    }
}

impl From<RuntimeError> for TrainError {
    fn from(e: RuntimeError) -> Self {
        let kind = match e.kind() {
            RuntimeKind::InvalidInput
            | RuntimeKind::InvalidGraph
            | RuntimeKind::UnsupportedModel => TrainErrorKind::InvalidInput,
            RuntimeKind::UnsupportedOperation => TrainErrorKind::UnsupportedOperation,
            RuntimeKind::Io => TrainErrorKind::Io,
            RuntimeKind::BackendUnavailable => TrainErrorKind::BackendUnavailable,
            RuntimeKind::Backend => TrainErrorKind::Backend,
            RuntimeKind::Cancelled => TrainErrorKind::Cancelled,
            _ => TrainErrorKind::Internal,
        };
        if kind == TrainErrorKind::Internal {
            return Self::internal(e.to_string());
        }
        let mut out = Self::new(kind, e.to_string());
        if let Some(node) = e.node() {
            out = out.with_node(node);
        }
        match &e {
            RuntimeError::Weight { tensor, .. } | RuntimeError::WeightUpload { tensor, .. } => {
                out = out.with_tensor(tensor.clone());
            }
            RuntimeError::Param { name, .. } => out = out.with_tensor(name.clone()),
            RuntimeError::SharedRunningStat { param, .. } => out = out.with_tensor(param.clone()),
            RuntimeError::DeviceUnavailable { device, .. }
            | RuntimeError::UnsupportedDtype { device, .. } => {
                out = out.with_device(device.clone());
            }
            RuntimeError::InvalidGraph(report) => out = out.with_details(report.to_string()),
            _ => {}
        }
        out
    }
}

impl From<DataError> for TrainError {
    fn from(e: DataError) -> Self {
        let kind = match e.kind() {
            DataErrorKind::InvalidInput => TrainErrorKind::InvalidInput,
            DataErrorKind::Io => TrainErrorKind::Io,
            DataErrorKind::OutOfMemory => TrainErrorKind::OutOfMemory,
            DataErrorKind::Cancelled => TrainErrorKind::Cancelled,
        };
        let mut out = Self::new(kind, e.to_string());
        if let Some(file) = e.file() {
            out = out.with_file(file);
        }
        if let Some(field) = e.field() {
            out = out.with_tensor(field);
        }
        let mut details = Vec::new();
        if let Some(record) = e.record() {
            details.push(format!("record {record}"));
        }
        if let Some(column) = e.column() {
            details.push(format!("column `{column}`"));
        }
        if let Some(io) = e.io_error() {
            details.push(io.to_string());
        }
        if !details.is_empty() {
            out = out.with_details(details.join(", "));
        }
        out
    }
}

impl From<OptimizerError> for TrainError {
    fn from(e: OptimizerError) -> Self {
        match e {
            OptimizerError::Backend(t) => t.into(),
            OptimizerError::MissingState { ref name, .. } => {
                // training.md §5.4 step 4: the checkpoint doesn't fit its graph.
                Self::invalid(format!(
                    "{e}: the checkpoint's optimizer state doesn't fit its graph"
                ))
                .with_tensor(name.clone())
            }
            OptimizerError::InvalidState { ref name, .. } => {
                Self::invalid(e.to_string()).with_tensor(name.clone())
            }
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<LossError> for TrainError {
    fn from(e: LossError) -> Self {
        match e {
            LossError::Backend(t) => t.into(),
            // Checked when the trainer was created (§2.10 steps 1 and 6).
            other => Self::internal(other.to_string()),
        }
    }
}

impl From<MetricError> for TrainError {
    fn from(e: MetricError) -> Self {
        match e {
            MetricError::Backend(t) => t.into(),
            other => Self::internal(other.to_string()),
        }
    }
}
