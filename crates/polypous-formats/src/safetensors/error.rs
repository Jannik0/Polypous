//! Errors of the safetensors reader and writer (`docs/specs/formats.md` §1.2, §1.5, §1.6).

use std::fmt;
use std::path::{Path, PathBuf};

use super::Dtype;

/// An error from reading or writing safetensors data.
///
/// Carries the reason ([`ErrorKind`]) and, where known, the file and the tensor it concerns, so a
/// caller can report "file X, tensor Y: reason".
#[derive(Debug)]
pub struct Error {
    kind: ErrorKind,
    path: Option<PathBuf>,
    tensor: Option<String>,
}

impl Error {
    /// The reason for the error.
    pub fn kind(&self) -> &ErrorKind {
        &self.kind
    }

    /// The file the error concerns (the shard or index file for sharded checkpoints), if known.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The tensor the error concerns, if any.
    pub fn tensor(&self) -> Option<&str> {
        self.tensor.as_deref()
    }

    /// Consumes the error and returns its reason.
    pub fn into_kind(self) -> ErrorKind {
        self.kind
    }

    pub(crate) fn with_path(mut self, path: &Path) -> Self {
        if self.path.is_none() {
            self.path = Some(path.to_path_buf());
        }
        self
    }

    pub(crate) fn with_tensor(mut self, name: &str) -> Self {
        if self.tensor.is_none() {
            self.tensor = Some(name.to_owned());
        }
        self
    }
}

impl From<ErrorKind> for Error {
    fn from(kind: ErrorKind) -> Self {
        Error {
            kind,
            path: None,
            tensor: None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(path) = &self.path {
            write!(f, "{}: ", path.display())?;
        }
        if let Some(tensor) = &self.tensor {
            write!(f, "tensor {tensor:?}: ")?;
        }
        write!(f, "{}", self.kind)
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        std::error::Error::source(&self.kind)
    }
}

/// The reason for an [`Error`]. Rule numbers refer to `docs/specs/formats.md` §1.2.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ErrorKind {
    /// Reading or writing the file failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The file is shorter than the 8-byte header length (rule 1).
    #[error("file is {len} bytes long, shorter than the 8-byte header length")]
    FileTooSmall {
        /// File length in bytes.
        len: u64,
    },
    /// The file doesn't fit in this platform's address space.
    #[error("file is {len} bytes long, too large for this platform")]
    FileTooLarge {
        /// File length in bytes.
        len: u64,
    },
    /// The header length exceeds [`MAX_HEADER_LEN`](super::MAX_HEADER_LEN) (rule 2).
    #[error("header length {len} exceeds the limit of {max} bytes")]
    HeaderTooLarge {
        /// Declared (or, when writing, produced) header length.
        len: u64,
        /// The limit.
        max: u64,
    },
    /// The header extends past the end of the file (rule 3).
    #[error("header of {header_len} bytes extends past the end of the {file_len}-byte file")]
    HeaderOutOfBounds {
        /// Declared header length.
        header_len: u64,
        /// File length.
        file_len: u64,
    },
    /// The header is not valid UTF-8 (rule 4).
    #[error("header is not valid UTF-8")]
    HeaderNotUtf8,
    /// The header is not valid JSON or doesn't have the expected structure (rules 5 and 7).
    #[error("invalid header: {0}")]
    InvalidHeader(String),
    /// `__metadata__` is not an object of string to string (rule 6).
    #[error("invalid __metadata__: {0}")]
    InvalidMetadata(String),
    /// A key appears twice (rule 8), or the writer got a duplicate or reserved tensor name.
    #[error("duplicate or reserved key {key:?}")]
    DuplicateKey {
        /// The key.
        key: String,
    },
    /// A dtype name the format does not define (rule 9).
    #[error("unknown dtype {name:?}")]
    UnknownDtype {
        /// The name as written in the header (truncated to 64 characters).
        name: String,
    },
    /// A shape has more than [`MAX_RANK`](super::MAX_RANK) dimensions (rule 10).
    #[error("shape has more than {max} dimensions")]
    RankTooLarge {
        /// The limit.
        max: usize,
    },
    /// A number doesn't fit in `usize`, or a size computation overflows (rule 11).
    #[error("size or offset overflows this platform's address space")]
    SizeOverflow,
    /// `numel × bits` is not a multiple of 8 (rule 12).
    #[error("{numel} elements of {dtype} are not a whole number of bytes")]
    PartialByte {
        /// The dtype.
        dtype: Dtype,
        /// Number of elements.
        numel: usize,
    },
    /// `data_offsets` has `begin > end` (rule 13).
    #[error("data_offsets [{begin}, {end}] have begin > end")]
    InvalidOffsets {
        /// Begin offset.
        begin: usize,
        /// End offset.
        end: usize,
    },
    /// The byte length doesn't match the shape and dtype (rule 13).
    #[error("data is {actual} bytes, but dtype and shape need {expected}")]
    LengthMismatch {
        /// Bytes needed for the shape and dtype.
        expected: usize,
        /// Bytes given.
        actual: usize,
    },
    /// Tensors overlap or leave a hole in the data buffer (rule 14).
    #[error("data_offsets begin at {begin}, expected {expected} (tensors must be contiguous)")]
    NonContiguous {
        /// Where the tensor should begin.
        expected: usize,
        /// Where it begins.
        begin: usize,
    },
    /// The tensors don't cover the data buffer exactly (rule 15).
    #[error("tensors cover {covered} bytes, but the data buffer is {data_len} bytes")]
    DataLengthMismatch {
        /// End of the last tensor.
        covered: usize,
        /// Length of the data buffer.
        data_len: usize,
    },
    /// No tensor with this name exists.
    #[error("tensor not found")]
    TensorNotFound,
    /// The tensor's dtype can't be decoded to values (`docs/specs/formats.md` §1.1).
    #[error("dtype {dtype} can't be read as values")]
    UnsupportedDtype {
        /// The dtype.
        dtype: Dtype,
    },
    /// A `BOOL` element is neither 0 nor 1.
    #[error("BOOL element {index} has byte value {value}, expected 0 or 1")]
    InvalidBoolValue {
        /// Element index.
        index: usize,
        /// The byte.
        value: u8,
    },
    /// Memory for decoded values could not be allocated.
    #[error("could not allocate {bytes} bytes for decoded values")]
    AllocationFailed {
        /// Requested size.
        bytes: usize,
    },
    /// The sharded index file exceeds [`MAX_INDEX_LEN`](super::MAX_INDEX_LEN) (§1.5).
    #[error("index file is larger than the limit of {max} bytes")]
    IndexTooLarge {
        /// The limit.
        max: u64,
    },
    /// The sharded index is not valid JSON or doesn't have the expected structure (§1.5).
    #[error("invalid index: {0}")]
    InvalidIndex(String),
    /// A shard path in the index is not a plain file name (§1.5).
    #[error("shard path {path:?} is not a plain file name in the index's directory")]
    InvalidShardPath {
        /// The path as written in the index (truncated to 256 characters).
        path: String,
    },
    /// A tensor listed in the index is missing from its shard (§1.5).
    #[error("tensor listed in the index is missing from shard {shard:?}")]
    MissingInShard {
        /// The shard file name.
        shard: String,
    },
    /// A tensor is present in more than one shard (§1.5).
    #[error("tensor is in shard {found:?}, but the index maps it to {mapped:?}")]
    InMultipleShards {
        /// The shard the index maps the tensor to.
        mapped: String,
        /// The other shard it was found in.
        found: String,
    },
    /// A shard contains a tensor the index doesn't list (§1.5).
    #[error("shard {shard:?} contains a tensor the index doesn't list")]
    UnlistedInShard {
        /// The shard file name.
        shard: String,
    },
}

/// Truncates untrusted text that goes into an error message.
pub(crate) fn truncate_for_message(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((i, _)) => format!("{}…", s.get(..i).unwrap_or_default()),
        None => s.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_path_and_tensor() {
        let e = Error::from(ErrorKind::TensorNotFound)
            .with_tensor("w")
            .with_path(Path::new("a.safetensors"));
        assert_eq!(
            e.to_string(),
            "a.safetensors: tensor \"w\": tensor not found"
        );
        assert_eq!(e.tensor(), Some("w"));
    }

    #[test]
    fn truncate() {
        assert_eq!(truncate_for_message("abc", 2), "ab…");
        assert_eq!(truncate_for_message("äöü", 3), "äöü");
    }
}
