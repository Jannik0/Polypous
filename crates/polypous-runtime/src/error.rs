//! Errors of the runtime (`docs/specs/runtime.md` §5).

use polypous_backend::TensorError;
use polypous_formats::safetensors::{Error as FormatError, ErrorKind as FormatErrorKind};
use polypous_ir::{BindError, Dtype, ValidationReport};

/// The category of a [`RuntimeError`]: the `polypous::ErrorKind` it maps to at the facade
/// (`docs/specs/public-api.md` §2 rule 7, runtime.md §5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Inputs, state values or weights don't fit the graph, or an index is out of range.
    InvalidInput,
    /// The graph failed validation.
    InvalidGraph,
    /// The weight file uses a storage format this version doesn't support.
    UnsupportedModel,
    /// The device doesn't support a dtype or an operation of the graph.
    UnsupportedOperation,
    /// Reading the weight file failed.
    Io,
    /// The device can't be used.
    BackendUnavailable,
    /// The backend failed while running.
    Backend,
    /// The call was cancelled through its [`CancellationToken`](crate::CancellationToken).
    Cancelled,
    /// A broken invariant: a bug in Polypous.
    Internal,
}

/// What is wrong with a weight tensor (runtime.md §2). Always reported with the tensor's name in
/// [`RuntimeError::Weight`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum WeightProblem {
    /// The graph declares the param, the file has no tensor of that name.
    #[error("the graph declares this param, but the weight file has no tensor of that name")]
    Missing,
    /// The file has a tensor that the graph doesn't declare as a param.
    #[error("the weight file has this tensor, but the graph declares no param of that name")]
    Extra,
    /// The tensor is not a float tensor.
    #[error("stored as {stored}, but params are float tensors (F32, F16 or BF16)")]
    NotFloat {
        /// The stored dtype, as the safetensors header names it.
        stored: String,
    },
    /// The tensor is a float tensor in a storage format this version can't load.
    #[error("stored as {stored}; this version loads params stored as F32, F16 or BF16")]
    UnsupportedStorage {
        /// The stored dtype, as the safetensors header names it.
        stored: String,
    },
    /// The stored shape differs from the declared one.
    #[error("stored with shape {stored:?}, but the graph declares {declared:?}")]
    Shape {
        /// The declared shape.
        declared: Vec<usize>,
        /// The stored shape.
        stored: Vec<usize>,
    },
    /// The tensor's size doesn't fit the address space, or its byte range doesn't match its
    /// shape and dtype.
    #[error("{reason}")]
    Size {
        /// What is wrong.
        reason: String,
    },
    /// No host memory for the tensor.
    #[error("could not allocate {bytes} bytes of host memory for it")]
    Allocation {
        /// The requested size.
        bytes: usize,
    },
}

/// An error of the runtime (runtime.md §5). [`RuntimeError::kind`] gives its category.
///
/// Errors raised while running a node carry the node path (`layers[3]/attn`, runtime.md §5). The
/// sources of backend, binding and file errors are kept as values ([`std::error::Error::source`]).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RuntimeError {
    /// The graph failed validation.
    #[error("the graph is not valid:\n{0}")]
    InvalidGraph(ValidationReport),

    /// The device doesn't support a dtype of the graph (graph-ir.md §2.2). Nothing has run.
    #[error("{what} has dtype {dtype}, which device {device} doesn't support")]
    UnsupportedDtype {
        /// The value: `compute dtype`, `input \`x\``, `state entry \`k\``, or
        /// `node \`layers/attn\` (nn.linear) output \`out\``.
        what: String,
        /// The node path, if the value is a node output.
        node: Option<String>,
        /// The op of that node.
        op: Option<String>,
        /// The dtype.
        dtype: Dtype,
        /// The device, as its id text.
        device: String,
    },

    /// The device can't be used: its backend is not compiled in, or it doesn't exist.
    #[error("device {device} is not available")]
    DeviceUnavailable {
        /// The device, as its id text.
        device: String,
        /// The backend's error.
        #[source]
        source: TensorError,
    },

    /// The weight file can't be opened or read: I/O, a malformed or truncated file or index.
    #[error("cannot read the weight file: {source}")]
    WeightFile {
        /// The reader's error (it names the file, and the tensor where known).
        #[source]
        source: FormatError,
    },

    /// A weight tensor doesn't fit the graph (runtime.md §2).
    #[error("weight `{tensor}`: {problem}")]
    Weight {
        /// The tensor (param) name.
        tensor: String,
        /// What is wrong.
        problem: WeightProblem,
    },

    /// Uploading or casting a weight tensor failed.
    #[error("weight `{tensor}`: upload to the device failed: {source}")]
    WeightUpload {
        /// The tensor (param) name.
        tensor: String,
        /// The backend's error.
        #[source]
        source: TensorError,
    },

    /// The call's inputs or state values don't fit the graph (graph-ir.md §2.3.4).
    #[error("{0}")]
    Bind(#[from] BindError),

    /// An input or state value lives on another device than the plan's.
    #[error("{kind} `{name}` is on device {actual}, but the model runs on {expected}")]
    WrongDevice {
        /// `input` or `state entry`.
        kind: &'static str,
        /// Its name.
        name: String,
        /// The plan's device.
        expected: String,
        /// The value's device.
        actual: String,
    },

    /// The call was cancelled before `before` ran (runtime.md §4).
    #[error("cancelled before node `{before}`")]
    Cancelled {
        /// The node path (for a repeat iteration: `layers[3]`) that didn't run.
        before: String,
    },

    /// An op failed while running a node.
    #[error("node `{node}` ({op}): {source}")]
    Node {
        /// The node path, `layers[3]/attn`.
        node: String,
        /// The op, `nn.linear`.
        op: String,
        /// The backend's error.
        #[source]
        source: TensorError,
    },

    /// A broken invariant: a bug in Polypous, never caused by a graph, file or input alone.
    #[error("internal error: {message}")]
    Internal {
        /// What went wrong.
        message: String,
    },
}

impl RuntimeError {
    /// The error's category (runtime.md §5).
    pub fn kind(&self) -> ErrorKind {
        match self {
            RuntimeError::InvalidGraph(_) => ErrorKind::InvalidGraph,
            RuntimeError::UnsupportedDtype { .. } => ErrorKind::UnsupportedOperation,
            RuntimeError::DeviceUnavailable { .. } => ErrorKind::BackendUnavailable,
            RuntimeError::WeightFile { source } => match source.kind() {
                FormatErrorKind::Io(_) => ErrorKind::Io,
                _ => ErrorKind::InvalidInput,
            },
            RuntimeError::Weight { problem, .. } => match problem {
                WeightProblem::UnsupportedStorage { .. } => ErrorKind::UnsupportedModel,
                _ => ErrorKind::InvalidInput,
            },
            RuntimeError::WeightUpload { source, .. } => tensor_error_kind(source),
            RuntimeError::Bind(BindError::Unbound { .. }) => ErrorKind::Internal,
            RuntimeError::Bind(_) | RuntimeError::WrongDevice { .. } => ErrorKind::InvalidInput,
            RuntimeError::Cancelled { .. } => ErrorKind::Cancelled,
            RuntimeError::Node { source, .. } => tensor_error_kind(source),
            RuntimeError::Internal { .. } => ErrorKind::Internal,
        }
    }

    /// The node path, for errors raised by or before a node.
    pub fn node(&self) -> Option<&str> {
        match self {
            RuntimeError::UnsupportedDtype { node, .. } => node.as_deref(),
            RuntimeError::Cancelled { before } => Some(before),
            RuntimeError::Node { node, .. } => Some(node),
            _ => None,
        }
    }

    pub(crate) fn internal(message: impl Into<String>) -> Self {
        RuntimeError::Internal {
            message: message.into(),
        }
    }
}

/// The kind of a backend error raised by a node (runtime.md §5). After binding, shape and dtype
/// errors can't happen; if one does, it is a bug.
fn tensor_error_kind(err: &TensorError) -> ErrorKind {
    match err {
        TensorError::IndexOutOfRange { .. } => ErrorKind::InvalidInput,
        TensorError::UnsupportedDType { .. } | TensorError::UnsupportedOperation { .. } => {
            ErrorKind::UnsupportedOperation
        }
        TensorError::DeviceUnavailable { .. } => ErrorKind::BackendUnavailable,
        TensorError::Backend { .. } => ErrorKind::Backend,
        _ => ErrorKind::Internal,
    }
}

/// Result type of the runtime.
pub type Result<T, E = RuntimeError> = std::result::Result<T, E>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds() {
        let node = |source| RuntimeError::Node {
            node: "layers[1]/emb".into(),
            op: "nn.embedding".into(),
            source,
        };
        let oob = node(TensorError::IndexOutOfRange {
            op: "embedding",
            index: 9,
            size: 4,
        });
        assert_eq!(oob.kind(), ErrorKind::InvalidInput);
        assert_eq!(oob.node(), Some("layers[1]/emb"));
        assert!(oob.to_string().contains("layers[1]/emb"));
        assert!(std::error::Error::source(&oob).is_some());
        let shape = node(TensorError::UnsupportedRank { rank: 7 });
        assert_eq!(shape.kind(), ErrorKind::Internal);
        let missing = RuntimeError::Weight {
            tensor: "fc.weight".into(),
            problem: WeightProblem::Missing,
        };
        assert_eq!(missing.kind(), ErrorKind::InvalidInput);
        assert!(missing.to_string().contains("fc.weight"));
        let f64 = RuntimeError::Weight {
            tensor: "w".into(),
            problem: WeightProblem::UnsupportedStorage {
                stored: "F64".into(),
            },
        };
        assert_eq!(f64.kind(), ErrorKind::UnsupportedModel);
        let cancelled = RuntimeError::Cancelled {
            before: "layers[2]".into(),
        };
        assert_eq!(cancelled.kind(), ErrorKind::Cancelled);
    }
}
