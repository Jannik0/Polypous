//! Named tensors from and to safetensors files (public-api.md §1.3, "Tensor files").
//!
//! For apps and the `polyp` CLI, which depend only on the facade. Reading validates the whole
//! header before any tensor is copied (formats.md §1.2) and reserves host memory fallibly; writing
//! validates every tensor first and writes a temporary file that is renamed over `path`
//! (formats.md §1.6).

use std::collections::BTreeMap;
use std::path::Path;

use polypous_formats::safetensors::{self, Dtype as StorageDtype, SafetensorsFile, WriteTensor};

use crate::error::{Error, ErrorKind, guard};
use crate::tensor::{Dtype, NamedTensor, TensorData};

/// The error of a safetensors read or write: `Io` for I/O, `InvalidInput` otherwise (malformed,
/// truncated, duplicate names), with the file and the tensor where known.
pub(crate) fn format_error(
    message: &str,
    path: &Path,
    error: &polypous_formats::safetensors::Error,
) -> Error {
    let kind = match error.kind() {
        safetensors::ErrorKind::Io(_) => ErrorKind::Io,
        safetensors::ErrorKind::AllocationFailed { .. } => ErrorKind::OutOfMemory,
        _ => ErrorKind::InvalidInput,
    };
    let mut out = Error::new(kind, format!("{message}: {error}."))
        .with_details(error.to_string())
        .with_file(error.path().unwrap_or(path));
    if let Some(tensor) = error.tensor() {
        out = out.with_tensor(tensor);
    }
    out
}

pub(crate) fn storage_dtype(dtype: Dtype) -> StorageDtype {
    match dtype {
        Dtype::F32 => StorageDtype::F32,
        Dtype::F16 => StorageDtype::F16,
        Dtype::Bf16 => StorageDtype::BF16,
        Dtype::I64 => StorageDtype::I64,
        Dtype::I32 => StorageDtype::I32,
        Dtype::Bool => StorageDtype::Bool,
    }
}

fn value_dtype(dtype: StorageDtype) -> Option<Dtype> {
    Some(match dtype {
        StorageDtype::F32 => Dtype::F32,
        StorageDtype::F16 => Dtype::F16,
        StorageDtype::BF16 => Dtype::Bf16,
        StorageDtype::I64 => Dtype::I64,
        StorageDtype::I32 => Dtype::I32,
        StorageDtype::Bool => Dtype::Bool,
        _ => return None,
    })
}

/// Copies `data` into a new buffer, reserving the memory fallibly.
pub(crate) fn copy_bytes(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut out = Vec::new();
    out.try_reserve_exact(data.len()).map_err(|_| {
        Error::new(
            ErrorKind::OutOfMemory,
            format!(
                "Not enough memory to read a tensor of {} bytes.",
                data.len()
            ),
        )
    })?;
    out.extend_from_slice(data);
    Ok(out)
}

/// Reads every tensor of the safetensors file at `path`, in name order.
///
/// # Errors
///
/// [`ErrorKind::Io`] if the file can't be read; [`ErrorKind::InvalidInput`] if it is malformed or
/// truncated, or a tensor is stored in a dtype other than `F32`, `F16`, `BF16`, `I64`, `I32` or
/// `BOOL`, or a `BOOL` byte is not `0`/`1`; [`ErrorKind::OutOfMemory`]. `context` names the file
/// and the tensor.
pub fn read(path: impl AsRef<Path>) -> Result<Vec<NamedTensor>, Error> {
    let path = path.as_ref();
    guard("tensor_file_read", || {
        let message = format!("Could not read the tensor file {}", path.display());
        let file =
            SafetensorsFile::open(path).map_err(|error| format_error(&message, path, &error))?;
        let mut out = Vec::with_capacity(file.len());
        for name in file.names() {
            let view = file
                .tensor(name)
                .map_err(|error| format_error(&message, path, &error))?;
            let in_file = |error: Error| error.with_tensor(name).with_file(path);
            let dtype = value_dtype(view.dtype()).ok_or_else(|| {
                in_file(Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "{message}: tensor `{name}` is stored as {}; tensors must be F32, F16, BF16, I64, I32 or BOOL.",
                        view.dtype().name()
                    ),
                ))
            })?;
            let tensor = TensorData {
                dtype,
                shape: view.shape().iter().map(|&dim| dim as u64).collect(),
                bytes: copy_bytes(view.data()).map_err(in_file)?,
            };
            tensor.validate().map_err(|error| {
                in_file(Error::new(
                    ErrorKind::InvalidInput,
                    format!("{message}: tensor `{name}`: {}", error.message),
                ))
            })?;
            out.push(NamedTensor::new(name, tensor));
        }
        Ok(out)
    })
}

/// Writes `tensors` to a safetensors file at `path`, by name, without metadata. An existing file
/// is replaced through a temporary file and a rename.
///
/// # Errors
///
/// [`ErrorKind::InvalidInput`] if a tensor is malformed ([`TensorData::validate`]), a name is
/// given twice or is `__metadata__`; [`ErrorKind::Io`] if the file can't be written.
pub fn write(path: impl AsRef<Path>, tensors: &[NamedTensor]) -> Result<(), Error> {
    let path = path.as_ref();
    guard("tensor_file_write", || {
        let message = format!("Could not write the tensor file {}", path.display());
        let mut shapes = Vec::with_capacity(tensors.len());
        for named in tensors {
            named.tensor.validate().map_err(|error| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!("{message}: tensor `{}`: {}", named.name, error.message),
                )
                .with_tensor(&named.name)
            })?;
            let shape = named
                .tensor
                .shape
                .iter()
                .map(|&dim| usize::try_from(dim))
                .collect::<Result<Vec<usize>, _>>()
                .map_err(|_| {
                    Error::new(
                        ErrorKind::InvalidInput,
                        format!(
                            "{message}: tensor `{}` has a dimension too large for this platform.",
                            named.name
                        ),
                    )
                    .with_tensor(&named.name)
                })?;
            shapes.push(shape);
        }
        let mut entries = Vec::with_capacity(tensors.len());
        for (named, shape) in tensors.iter().zip(&shapes) {
            let tensor = WriteTensor::new(
                storage_dtype(named.tensor.dtype),
                shape,
                &named.tensor.bytes,
            )
            .map_err(|error| format_error(&message, path, &error).with_tensor(&named.name))?;
            entries.push((named.name.as_str(), tensor));
        }
        safetensors::write_file(path, entries, &BTreeMap::new())
            .map_err(|error| format_error(&message, path, &error))
    })
}
