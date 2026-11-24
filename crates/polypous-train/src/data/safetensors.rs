//! The `safetensors` dataset (`docs/specs/training.md` §3.5.2), read through `polypous-formats`
//! (formats.md §1).
//!
//! `SafetensorsFile::open` validates the whole header and checks every tensor's offsets against
//! the file before any data is touched. The file is memory-mapped, so only the used tensors' bytes
//! are read (if mapping fails, formats.md §1.4 reads the whole file instead). Every used tensor's
//! dtype, rank and sample count, and the memory for all of them, are checked from the header
//! before any data is decoded.

use std::collections::BTreeMap;
use std::path::Path;

use polypous_backend::{DType, TensorData};
use polypous_formats::safetensors::{
    self as st, Dtype, ErrorKind, SafetensorsFile, bf16_to_f32, f16_to_f32,
};

use super::{Budget, DataError, Dataset, FieldSelection, MAX_FIELD_RANK, check_count, try_bytes};

pub(crate) fn read(
    path: &Path,
    fields: &FieldSelection,
    budget: &mut Budget,
) -> Result<Dataset, DataError> {
    let file = SafetensorsFile::open(path).map_err(|e| map_error(e, path))?;
    let used: Vec<&str> = file.names().filter(|n| fields.wants(n)).collect();
    // Metadata first: dtype, rank, sample count and memory of every used tensor.
    let mut len = None;
    let mut plan = Vec::with_capacity(used.len());
    for &name in &used {
        let at = |e: DataError| e.with_field(name).with_file(path);
        let Some(info) = file.info(name) else {
            continue;
        };
        let dtype = field_dtype(info.dtype()).ok_or_else(|| {
            at(DataError::invalid(format!(
                "tensor `{name}` has storage dtype {:?}; a dataset tensor is F32, F16, BF16, I64, \
                 I32 or BOOL",
                info.dtype()
            )))
        })?;
        let shape = info.shape();
        if shape.is_empty() || shape.len() > MAX_FIELD_RANK {
            return Err(at(DataError::invalid(format!(
                "tensor `{name}` has rank {}; a dataset tensor has rank 1–7 (dim 0 is the sample \
                 dim)",
                shape.len()
            ))));
        }
        check_count(&mut len, name, shape[0] as u64).map_err(at)?;
        let bytes = (info.numel() as u64)
            .checked_mul(dtype.size_bytes() as u64)
            .ok_or_else(|| at(DataError::invalid("tensor too large")))?;
        budget.charge(bytes).map_err(at)?;
        plan.push((name, dtype, bytes));
    }
    let Some(len) = len else {
        let names: Vec<&str> = file.names().collect();
        return Err(DataError::invalid(format!(
            "the file has none of the tensors the training uses (it has {names:?})"
        ))
        .with_file(path));
    };
    let mut out = BTreeMap::new();
    for (name, dtype, bytes) in plan {
        let at = |e: DataError| e.with_field(name).with_file(path);
        let view = file.tensor(name).map_err(|e| map_error(e, path))?;
        let data = decode(&view, bytes).map_err(at)?;
        let shape = view.shape().iter().map(|&d| d as u64).collect();
        out.insert(
            name.to_owned(),
            TensorData {
                dtype,
                shape,
                bytes: data,
            },
        );
    }
    Ok(Dataset::from_checked(len, out))
}

/// The field dtype of a storage dtype (§3.5.2), `None` for the others.
fn field_dtype(dtype: Dtype) -> Option<DType> {
    Some(match dtype {
        Dtype::F32 | Dtype::F16 | Dtype::BF16 => DType::F32,
        Dtype::I64 => DType::I64,
        Dtype::I32 => DType::I32,
        Dtype::Bool => DType::Bool,
        _ => return None,
    })
}

/// The tensor's bytes in the field layout (`F16`/`BF16` widened exactly), `bytes` long.
fn decode(view: &st::TensorView<'_>, bytes: u64) -> Result<Vec<u8>, DataError> {
    let mut out = try_bytes(bytes, "a dataset field")?;
    let raw = view.data();
    match view.dtype() {
        Dtype::F16 | Dtype::BF16 => {
            let convert = if view.dtype() == Dtype::F16 {
                f16_to_f32
            } else {
                bf16_to_f32
            };
            let (chunks, _) = raw.as_chunks::<2>();
            for &c in chunks {
                out.extend_from_slice(&convert(u16::from_le_bytes(c)).to_le_bytes());
            }
        }
        Dtype::Bool => {
            if let Some(index) = raw.iter().position(|&b| b > 1) {
                return Err(DataError::invalid(format!(
                    "tensor `{}`: BOOL element {index} is {}, but must be 0 or 1",
                    view.name(),
                    raw[index]
                )));
            }
            out.extend_from_slice(raw);
        }
        // F32, I64 and I32 are stored little-endian, row-major, as the field layout.
        _ => out.extend_from_slice(raw),
    }
    if out.len() as u64 != bytes {
        return Err(DataError::invalid(format!(
            "tensor `{}` has {} data bytes, expected {bytes}",
            view.name(),
            raw.len()
        )));
    }
    Ok(out)
}

/// A reader error as a [`DataError`]: `Io` stays `Io`, a failed allocation is `OutOfMemory`,
/// everything else is a malformed file (`InvalidInput`).
fn map_error(err: st::Error, path: &Path) -> DataError {
    let tensor = err.tensor().map(str::to_owned);
    let message = err.kind().to_string();
    let out = match err.into_kind() {
        ErrorKind::Io(io) => DataError::io(io, path, "read the safetensors file"),
        ErrorKind::AllocationFailed { bytes } => {
            DataError::out_of_memory(bytes as u64, "a safetensors file")
        }
        _ => DataError::invalid(format!("not a valid safetensors file: {message}")),
    };
    let out = out.with_file(path);
    match tensor {
        Some(tensor) => out.with_field(&tensor),
        None => out,
    }
}
