//! safetensors writer (`docs/specs/formats.md` §1.6).

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::header::LEN_PREFIX;
use super::{Dtype, Error, ErrorKind, MAX_HEADER_LEN};

/// A tensor to write: dtype, shape and little-endian, row-major bytes. [`new`](Self::new) checks
/// that the byte length matches the shape and dtype.
#[derive(Debug, Clone, Copy)]
pub struct WriteTensor<'a> {
    dtype: Dtype,
    shape: &'a [usize],
    data: &'a [u8],
}

impl<'a> WriteTensor<'a> {
    /// Checks that `data` is exactly `product(shape) × bits / 8` bytes.
    ///
    /// Fails with [`ErrorKind::SizeOverflow`], [`ErrorKind::PartialByte`] or
    /// [`ErrorKind::LengthMismatch`].
    pub fn new(dtype: Dtype, shape: &'a [usize], data: &'a [u8]) -> Result<Self, Error> {
        let numel = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(ErrorKind::SizeOverflow)?;
        let expected = dtype.byte_len(numel)?;
        if data.len() != expected {
            return Err(ErrorKind::LengthMismatch {
                expected,
                actual: data.len(),
            }
            .into());
        }
        Ok(WriteTensor { dtype, shape, data })
    }

    /// The element type.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The shape.
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }

    /// The bytes.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Header bytes (padded) and the tensors in output order.
struct Prepared<'a> {
    header: Vec<u8>,
    tensors: Vec<WriteTensor<'a>>,
}

fn prepare<'a, N: Into<String>>(
    tensors: impl IntoIterator<Item = (N, WriteTensor<'a>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<Prepared<'a>, Error> {
    let mut sorted = BTreeMap::<String, WriteTensor<'a>>::new();
    for (name, tensor) in tensors {
        let name = name.into();
        if name == "__metadata__" {
            return Err(
                Error::from(ErrorKind::DuplicateKey { key: name.clone() }).with_tensor(&name)
            );
        }
        match sorted.entry(name) {
            Entry::Occupied(o) => {
                let name = o.key();
                return Err(
                    Error::from(ErrorKind::DuplicateKey { key: name.clone() }).with_tensor(name)
                );
            }
            Entry::Vacant(v) => {
                v.insert(tensor);
            }
        }
    }

    let json_err = |e: serde_json::Error| ErrorKind::InvalidHeader(e.to_string());
    let mut header = String::from("{");
    let mut first = true;
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":");
        header.push_str(&serde_json::to_string(metadata).map_err(json_err)?);
        first = false;
    }
    let mut offset = 0usize;
    for (name, tensor) in &sorted {
        let end = offset
            .checked_add(tensor.data.len())
            .ok_or(ErrorKind::SizeOverflow)?;
        if !first {
            header.push(',');
        }
        first = false;
        header.push_str(&serde_json::to_string(name).map_err(json_err)?);
        header.push_str(":{\"dtype\":\"");
        header.push_str(tensor.dtype.name());
        header.push_str("\",\"shape\":[");
        for (i, d) in tensor.shape.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&d.to_string());
        }
        header.push_str(&format!("],\"data_offsets\":[{offset},{end}]}}"));
        offset = end;
    }
    header.push('}');

    // Pad with spaces so the data buffer starts at an 8-byte aligned file offset.
    let mut header = header.into_bytes();
    let padded = header
        .len()
        .checked_next_multiple_of(LEN_PREFIX)
        .ok_or(ErrorKind::SizeOverflow)?;
    if padded as u64 > MAX_HEADER_LEN {
        return Err(ErrorKind::HeaderTooLarge {
            len: padded as u64,
            max: MAX_HEADER_LEN,
        }
        .into());
    }
    header.resize(padded, b' ');
    Ok(Prepared {
        header,
        tensors: sorted.into_values().collect(),
    })
}

/// Serializes tensors and `__metadata__` entries into a complete safetensors file in memory.
///
/// Tensors are ordered by name; see `docs/specs/formats.md` §1.6 for the exact output. Fails on
/// a duplicate tensor name, the name `__metadata__` ([`ErrorKind::DuplicateKey`]) or a header
/// larger than [`MAX_HEADER_LEN`] ([`ErrorKind::HeaderTooLarge`]).
pub fn serialize<'a, N: Into<String>>(
    tensors: impl IntoIterator<Item = (N, WriteTensor<'a>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>, Error> {
    let prepared = prepare(tensors, metadata)?;
    let total = prepared
        .tensors
        .iter()
        .try_fold(LEN_PREFIX + prepared.header.len(), |acc, t| {
            acc.checked_add(t.data.len())
        })
        .ok_or(ErrorKind::SizeOverflow)?;
    let mut out = Vec::new();
    out.try_reserve_exact(total)
        .map_err(|_| ErrorKind::AllocationFailed { bytes: total })?;
    write_prepared(&mut out, &prepared)?;
    Ok(out)
}

/// Writes a safetensors file to `writer`. Same output and errors as [`serialize`], plus
/// [`ErrorKind::Io`].
pub fn write<'a, N: Into<String>, W: Write>(
    mut writer: W,
    tensors: impl IntoIterator<Item = (N, WriteTensor<'a>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<(), Error> {
    let prepared = prepare(tensors, metadata)?;
    write_prepared(&mut writer, &prepared)?;
    writer.flush().map_err(ErrorKind::Io)?;
    Ok(())
}

fn write_prepared<W: Write>(writer: &mut W, prepared: &Prepared<'_>) -> Result<(), Error> {
    let len = prepared.header.len() as u64;
    writer
        .write_all(&len.to_le_bytes())
        .map_err(ErrorKind::Io)?;
    writer.write_all(&prepared.header).map_err(ErrorKind::Io)?;
    for t in &prepared.tensors {
        writer.write_all(t.data).map_err(ErrorKind::Io)?;
    }
    Ok(())
}

/// Writes a safetensors file to `path`. Same output and errors as [`serialize`], plus
/// [`ErrorKind::Io`].
///
/// Writes a new temporary file next to `path` and renames it over `path`, so an existing file is
/// never modified in place (it may still be memory-mapped by a reader, `docs/specs/formats.md`
/// §1.4). On Windows the rename fails while `path` is mapped. The temporary file is removed on
/// error.
pub fn write_file<'a, N: Into<String>>(
    path: impl AsRef<Path>,
    tensors: impl IntoIterator<Item = (N, WriteTensor<'a>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<(), Error> {
    let path = path.as_ref();
    write_file_inner(path, tensors, metadata).map_err(|e| e.with_path(path))
}

fn write_file_inner<'a, N: Into<String>>(
    path: &Path,
    tensors: impl IntoIterator<Item = (N, WriteTensor<'a>)>,
    metadata: &BTreeMap<String, String>,
) -> Result<(), Error> {
    let prepared = prepare(tensors, metadata)?;
    let (tmp_path, file) = create_temp_sibling(path)?;
    let result = (|| -> Result<(), Error> {
        let mut writer = BufWriter::new(file);
        write_prepared(&mut writer, &prepared)?;
        let file = writer
            .into_inner()
            .map_err(|e| ErrorKind::Io(e.into_error()))?;
        file.sync_all().map_err(ErrorKind::Io)?;
        drop(file);
        std::fs::rename(&tmp_path, path).map_err(ErrorKind::Io)?;
        Ok(())
    })();
    if result.is_err() {
        // Best effort: the error being returned is the one that matters.
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

/// Creates a new, uniquely named file next to `path`.
fn create_temp_sibling(path: &Path) -> Result<(PathBuf, File), Error> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_name = path.file_name().ok_or_else(|| {
        ErrorKind::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "path has no file name",
        ))
    })?;
    let dir = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let mut attempts = 0;
    loop {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut name = std::ffi::OsString::from(".");
        name.push(file_name);
        name.push(format!(".{}.{n}.tmp", std::process::id()));
        let tmp = dir.join(name);
        match OpenOptions::new().write(true).create_new(true).open(&tmp) {
            Ok(file) => return Ok((tmp, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists && attempts < 100 => {
                attempts += 1;
            }
            Err(e) => return Err(ErrorKind::Io(e).into()),
        }
    }
}
