//! The `tensors` dataset: in-memory fields from the app (`docs/specs/training.md` §3.5.1).

use std::collections::{BTreeMap, BTreeSet};

use polypous_backend::{DType, TensorData};
use polypous_formats::safetensors::{bf16_to_f32, f16_to_f32};

use super::{
    Budget, DataError, Dataset, FieldSelection, InMemoryDataset, MAX_FIELD_RANK, check_count,
    try_bytes,
};

/// Reads the dataset `name` of `in_memory`. Its used fields are moved out, unless `keep` (the
/// other role uses the same dataset), in which case they are copied.
pub(crate) fn read(
    name: &str,
    in_memory: &mut [InMemoryDataset],
    fields: &FieldSelection,
    budget: &mut Budget,
    keep: bool,
) -> Result<Dataset, DataError> {
    let Some(dataset) = in_memory.iter_mut().find(|d| d.name == name) else {
        return Err(DataError::invalid(format!(
            "`TrainerOptions::datasets` has no dataset named `{name}`"
        )));
    };
    let mut seen = BTreeSet::new();
    for (field, _) in &dataset.fields {
        if !seen.insert(field.as_str()) {
            return Err(DataError::invalid(format!(
                "the in-memory dataset `{name}` has two fields named `{field}`"
            ))
            .with_field(field));
        }
    }
    // Check every used field and the memory before moving or copying anything.
    let mut len = None;
    for (field, data) in dataset.fields.iter().filter(|(f, _)| fields.wants(f)) {
        check_field(field, data)?;
        check_count(&mut len, field, data.shape[0])?;
    }
    let Some(len) = len else {
        return Err(no_used_fields(name, dataset));
    };
    for (field, data) in dataset.fields.iter().filter(|(f, _)| fields.wants(f)) {
        budget
            .charge(decoded_bytes(data))
            .map_err(|e| e.with_field(field))?;
    }
    let mut out = BTreeMap::new();
    for (field, data) in dataset.fields.iter_mut().filter(|(f, _)| fields.wants(f)) {
        let decoded = match data.dtype {
            DType::F16 | DType::BF16 => widen(data)?,
            _ if keep => {
                let mut bytes = try_bytes(data.bytes.len() as u64, "a dataset field")?;
                bytes.extend_from_slice(&data.bytes);
                TensorData {
                    dtype: data.dtype,
                    shape: data.shape.clone(),
                    bytes,
                }
            }
            _ => std::mem::replace(
                data,
                TensorData {
                    dtype: data.dtype,
                    shape: Vec::new(),
                    bytes: Vec::new(),
                },
            ),
        };
        out.insert(field.clone(), decoded);
    }
    if !keep {
        // The moved-out fields are left as empty placeholders; drop them so nothing reads them.
        dataset
            .fields
            .retain(|(f, d)| !(fields.wants(f) && d.shape.is_empty()));
    }
    Ok(Dataset::from_checked(len, out))
}

fn no_used_fields(name: &str, dataset: &InMemoryDataset) -> DataError {
    let names: Vec<&str> = dataset.fields.iter().map(|(f, _)| f.as_str()).collect();
    DataError::no_used_fields(format!(
        "the in-memory dataset `{name}` has none of the fields the training uses (it has {names:?})"
    ))
}

/// public-api.md §2 rule 4, then the §3.5.1 rank rule.
fn check_field(name: &str, data: &TensorData) -> Result<(), DataError> {
    data.validate().map_err(|e| {
        DataError::invalid(format!("field `{name}` is malformed: {e}")).with_field(name)
    })?;
    if data.shape.is_empty() || data.shape.len() > MAX_FIELD_RANK {
        return Err(DataError::invalid(format!(
            "field `{name}` has rank {}; a field has rank 1–7 (dim 0 is the sample dim)",
            data.shape.len()
        ))
        .with_field(name));
    }
    Ok(())
}

/// The bytes a field takes once read: `f16`/`bf16` are widened to 4-byte `f32`.
fn decoded_bytes(data: &TensorData) -> u64 {
    match data.dtype {
        DType::F16 | DType::BF16 => data.bytes.len() as u64 * 2,
        _ => data.bytes.len() as u64,
    }
}

/// `f16`/`bf16` widened to `f32` exactly (§3.5).
fn widen(data: &TensorData) -> Result<TensorData, DataError> {
    let convert = if data.dtype == DType::F16 {
        f16_to_f32
    } else {
        bf16_to_f32
    };
    let mut bytes = try_bytes(decoded_bytes(data), "a widened dataset field")?;
    let (chunks, _) = data.bytes.as_chunks::<2>();
    for &c in chunks {
        bytes.extend_from_slice(&convert(u16::from_le_bytes(c)).to_le_bytes());
    }
    Ok(TensorData {
        dtype: DType::F32,
        shape: data.shape.clone(),
        bytes,
    })
}
