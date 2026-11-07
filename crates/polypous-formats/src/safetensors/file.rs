//! A single safetensors file, memory-mapped or in memory (`docs/specs/formats.md` §1.3, §1.4).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::header::{self, Header};
use super::{Error, ErrorKind, TensorInfo, TensorView};

/// The bytes of a file.
enum Storage {
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Storage {
    fn bytes(&self) -> &[u8] {
        match self {
            Storage::Mapped(map) => map,
            Storage::Owned(vec) => vec,
        }
    }
}

/// Maps `file` read-only.
///
/// The only `unsafe` code in this crate (`docs/specs/formats.md` §1.4).
#[allow(unsafe_code)]
fn map_file(file: &File) -> std::io::Result<memmap2::Mmap> {
    // SAFETY: `Mmap::map` is unsafe because the mapping aliases the file: if the file is modified
    // or truncated while mapped, the `&[u8]` views handed out by `SafetensorsFile` change under
    // the borrow checker or (truncation) fault on access. We assume, and docs/specs/formats.md §1.4
    // documents as the caller's contract, that the file is not modified while mapped. The map is
    // read-only, owned by the `SafetensorsFile`, and outlives every view borrowed from it.
    unsafe { memmap2::Mmap::map(file) }
}

/// A validated safetensors file.
///
/// [`open`](Self::open) memory-maps the file; tensors are then borrowed from the map without
/// copying. The whole header is validated (`docs/specs/formats.md` §1.2) before the value is
/// returned.
pub struct SafetensorsFile {
    storage: Storage,
    path: Option<PathBuf>,
    header: Header,
}

impl std::fmt::Debug for SafetensorsFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafetensorsFile")
            .field("path", &self.path)
            .field("memory_mapped", &self.is_memory_mapped())
            .field("tensors", &self.header.tensors.len())
            .finish_non_exhaustive()
    }
}

impl SafetensorsFile {
    /// Opens and validates the file at `path`, memory-mapped read-only.
    ///
    /// If mapping fails, the file is read into memory instead (§1.4). The file must not be
    /// modified while the returned value exists.
    pub fn open(path: impl AsRef<Path>) -> Result<SafetensorsFile, Error> {
        let path = path.as_ref();
        Self::open_inner(path).map_err(|e| e.with_path(path))
    }

    fn open_inner(path: &Path) -> Result<SafetensorsFile, Error> {
        let mut file = File::open(path).map_err(ErrorKind::Io)?;
        let len = file.metadata().map_err(ErrorKind::Io)?.len();
        if len < header::LEN_PREFIX as u64 {
            // Also avoids mapping an empty file, which fails on some platforms.
            return Err(ErrorKind::FileTooSmall { len }.into());
        }
        let len_usize = usize::try_from(len).map_err(|_| ErrorKind::FileTooLarge { len })?;
        let storage = match map_file(&file) {
            Ok(map) => Storage::Mapped(map),
            Err(err) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %err,
                    "memory-mapping failed; reading the file into memory instead"
                );
                // Copy fallback: the length fits in `usize`; read at most that many bytes, so a
                // file that grows meanwhile can't make us allocate more.
                let mut buf = Vec::new();
                buf.try_reserve_exact(len_usize)
                    .map_err(|_| ErrorKind::FileTooLarge { len })?;
                (&mut file)
                    .take(len)
                    .read_to_end(&mut buf)
                    .map_err(ErrorKind::Io)?;
                Storage::Owned(buf)
            }
        };
        let header = header::parse(storage.bytes())?;
        Ok(SafetensorsFile {
            storage,
            path: Some(path.to_path_buf()),
            header,
        })
    }

    /// Validates a complete safetensors file held in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<SafetensorsFile, Error> {
        let header = header::parse(&bytes)?;
        Ok(SafetensorsFile {
            storage: Storage::Owned(bytes),
            path: None,
            header,
        })
    }

    /// The path the file was opened from; `None` for [`from_bytes`](Self::from_bytes).
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Whether the data is memory-mapped (`false` after the copy fallback or for `from_bytes`).
    pub fn is_memory_mapped(&self) -> bool {
        matches!(self.storage, Storage::Mapped(_))
    }

    /// The `__metadata__` entries; empty if the header has none.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.header.metadata
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.header.tensors.len()
    }

    /// Whether the file has no tensors.
    pub fn is_empty(&self) -> bool {
        self.header.tensors.is_empty()
    }

    /// Tensor names, sorted byte-wise.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.header.tensors.keys().map(String::as_str)
    }

    /// Whether a tensor with this name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.header.tensors.contains_key(name)
    }

    /// Metadata of the named tensor, or `None` if it doesn't exist.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.header.tensors.get(name)
    }

    /// A view of the named tensor, borrowing its bytes from the file.
    ///
    /// Fails with [`ErrorKind::TensorNotFound`] if no tensor has this name.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        let not_found = || self.error(ErrorKind::TensorNotFound, name);
        let (name, info) = self
            .header
            .tensors
            .get_key_value(name)
            .ok_or_else(not_found)?;
        let (begin, end) = info.data_offsets();
        // Validation guarantees `data_start + end == file length`; the checked arithmetic and
        // `get` only keep this path panic-free.
        let range = self
            .header
            .data_start
            .checked_add(begin)
            .zip(self.header.data_start.checked_add(end));
        let data = range
            .and_then(|(b, e)| self.storage.bytes().get(b..e))
            .ok_or_else(|| self.error(ErrorKind::SizeOverflow, name))?;
        Ok(TensorView::new(name, info, data, self.path.as_deref()))
    }

    fn error(&self, kind: ErrorKind, tensor: &str) -> Error {
        let err = Error::from(kind).with_tensor(tensor);
        match &self.path {
            Some(p) => err.with_path(p),
            None => err,
        }
    }
}
