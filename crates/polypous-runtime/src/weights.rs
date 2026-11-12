//! Weight loading (`docs/specs/runtime.md` §2): strict name match, storage dtype and shape checks
//! on the header before anything is read, one rounding to the compute dtype.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use polypous_backend::{DType, DynTensor, TensorData};
use polypous_formats::safetensors::{
    Dtype as StorageDtype, SafetensorsFile, ShardedSafetensors, TensorInfo, bf16_to_f32, f16_to_f32,
};
use polypous_ir::names::expand_prefix;

use crate::error::{Result, RuntimeError, WeightProblem};
use crate::ops::to_backend_dtype;
use crate::plan::{Block, Plan, StepKind};

/// A weight file: one safetensors file or a sharded checkpoint (formats.md §1, §1.5). Its header
/// (and index) is validated when it is opened.
#[derive(Debug)]
pub enum WeightFile {
    /// One `*.safetensors` file.
    Single(SafetensorsFile),
    /// A sharded checkpoint, opened through its `*.safetensors.index.json`.
    Sharded(ShardedSafetensors),
}

impl WeightFile {
    /// Opens `path`: a sharded checkpoint if the file name ends in `.index.json`, otherwise one
    /// safetensors file. Errors are [`RuntimeError::WeightFile`] (I/O, malformed or truncated
    /// file).
    pub fn open(path: impl AsRef<Path>) -> Result<WeightFile> {
        let path = path.as_ref();
        let sharded = path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".index.json"));
        let opened = if sharded {
            ShardedSafetensors::open(path).map(WeightFile::Sharded)
        } else {
            SafetensorsFile::open(path).map(WeightFile::Single)
        };
        opened.map_err(|source| RuntimeError::WeightFile { source })
    }

    fn names(&self) -> Vec<&str> {
        match self {
            WeightFile::Single(f) => f.names().collect(),
            WeightFile::Sharded(f) => f.names().collect(),
        }
    }

    fn info(&self, name: &str) -> Option<&TensorInfo> {
        match self {
            WeightFile::Single(f) => f.info(name),
            WeightFile::Sharded(f) => f.info(name),
        }
    }

    fn data(&self, name: &str) -> Result<&[u8]> {
        let view = match self {
            WeightFile::Single(f) => f.tensor(name),
            WeightFile::Sharded(f) => f.tensor(name),
        };
        view.map(|v| v.data())
            .map_err(|source| RuntimeError::WeightFile { source })
    }
}

impl From<SafetensorsFile> for WeightFile {
    fn from(file: SafetensorsFile) -> Self {
        WeightFile::Single(file)
    }
}

impl From<ShardedSafetensors> for WeightFile {
    fn from(file: ShardedSafetensors) -> Self {
        WeightFile::Sharded(file)
    }
}

/// A param that passed the header checks.
struct Checked {
    name: String,
    shape: Vec<usize>,
    storage: DType,
    /// Bytes of the stored tensor.
    bytes: usize,
}

fn weight_error(tensor: &str, problem: WeightProblem) -> RuntimeError {
    RuntimeError::Weight {
        tensor: tensor.to_owned(),
        problem,
    }
}

/// Loads the params of `plan` from `file` onto the plan's device (runtime.md §2). Returns the
/// uploaded params by graph param name; declared but unused params are checked, not uploaded.
pub(crate) fn load(plan: &Plan, file: Option<&WeightFile>) -> Result<HashMap<String, DynTensor>> {
    // Step 1–4: every check on header metadata, before any data is read.
    let mut checked = Vec::with_capacity(plan.params.len());
    for (name, declared) in &plan.params {
        let info = file
            .and_then(|f| f.info(name))
            .ok_or_else(|| weight_error(name, WeightProblem::Missing))?;
        checked.push(check(name, declared, info)?);
    }
    if let Some(file) = file {
        let declared: HashSet<&str> = plan.params.iter().map(|(n, _)| n.as_str()).collect();
        let mut extra: Vec<&str> = file
            .names()
            .into_iter()
            .filter(|n| !declared.contains(n))
            .collect();
        extra.sort_unstable();
        if let Some(name) = extra.first() {
            return Err(weight_error(name, WeightProblem::Extra));
        }
    }

    // Step 5–6: upload every param a node uses, once.
    let used = used_params(plan)?;
    let compute = to_backend_dtype(plan.compute_dtype().dtype());
    let mut params = HashMap::with_capacity(used.len());
    for c in checked {
        if !used.contains(c.name.as_str()) {
            tracing::debug!(param = %c.name, "param is declared but unused; not uploaded");
            continue;
        }
        // `file` is `Some`: every declared param was found in it above.
        let Some(file) = file else {
            return Err(weight_error(&c.name, WeightProblem::Missing));
        };
        let data = file.data(&c.name)?;
        let tensor = upload(plan, &c, data, compute)?;
        params.insert(c.name, tensor);
    }
    tracing::debug!(
        params = params.len(),
        device = %plan.device(),
        "loaded weights"
    );
    Ok(params)
}

/// Storage dtype, shape and size checks of one param (runtime.md §2 steps 2–4).
fn check(name: &str, declared: &[usize], info: &TensorInfo) -> Result<Checked> {
    let stored = info.dtype();
    let storage = match stored {
        StorageDtype::F32 => DType::F32,
        StorageDtype::F16 => DType::F16,
        StorageDtype::BF16 => DType::BF16,
        StorageDtype::F64
        | StorageDtype::F8E5M2
        | StorageDtype::F8E4M3
        | StorageDtype::F8E4M3Fnuz
        | StorageDtype::F8E5M2Fnuz
        | StorageDtype::F8E8M0
        | StorageDtype::F6E2M3
        | StorageDtype::F6E3M2
        | StorageDtype::F4 => {
            return Err(weight_error(
                name,
                WeightProblem::UnsupportedStorage {
                    stored: stored.name().to_owned(),
                },
            ));
        }
        _ => {
            return Err(weight_error(
                name,
                WeightProblem::NotFloat {
                    stored: stored.name().to_owned(),
                },
            ));
        }
    };
    if info.shape() != declared {
        return Err(weight_error(
            name,
            WeightProblem::Shape {
                declared: declared.to_vec(),
                stored: info.shape().to_vec(),
            },
        ));
    }
    let size_error = |reason: String| weight_error(name, WeightProblem::Size { reason });
    let numel = declared
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| size_error(format!("shape {declared:?} has too many elements")))?;
    let bytes = numel
        .checked_mul(storage.size_bytes())
        .ok_or_else(|| size_error(format!("shape {declared:?} is too large")))?;
    // The f32 buffer of a widened 16-bit tensor (step 5).
    numel
        .checked_mul(4)
        .filter(|&b| isize::try_from(b).is_ok())
        .ok_or_else(|| size_error(format!("shape {declared:?} is too large")))?;
    let (start, end) = info.data_offsets();
    if end.checked_sub(start) != Some(bytes) || info.byte_len() != bytes {
        return Err(size_error(format!(
            "its byte range {start}..{end} doesn't hold {numel} {} elements",
            stored.name()
        )));
    }
    Ok(Checked {
        name: name.to_owned(),
        shape: declared.to_vec(),
        storage,
        bytes,
    })
}

/// Uploads one param and casts it to the compute dtype, rounding once (runtime.md §2 step 5).
fn upload(plan: &Plan, c: &Checked, data: &[u8], compute: DType) -> Result<DynTensor> {
    if data.len() != c.bytes {
        return Err(weight_error(
            &c.name,
            WeightProblem::Size {
                reason: format!("holds {} bytes, expected {}", data.len(), c.bytes),
            },
        ));
    }
    let (dtype, bytes) = if c.storage == compute || c.storage == DType::F32 {
        (c.storage, copy(&c.name, data)?)
    } else {
        // F16/BF16 stored, another compute dtype: widen to f32 on the host (exact).
        let widen: fn(u16) -> f32 = if c.storage == DType::F16 {
            f16_to_f32
        } else {
            bf16_to_f32
        };
        let len = data.len() / 2 * 4;
        let mut out = Vec::new();
        out.try_reserve_exact(len)
            .map_err(|_| weight_error(&c.name, WeightProblem::Allocation { bytes: len }))?;
        let (chunks, _) = data.as_chunks::<2>();
        for &pair in chunks {
            out.extend_from_slice(&widen(u16::from_le_bytes(pair)).to_le_bytes());
        }
        (DType::F32, out)
    };
    let shape = c.shape.iter().map(|&d| d as u64).collect();
    let upload_error = |source| RuntimeError::WeightUpload {
        tensor: c.name.clone(),
        source,
    };
    let host = TensorData::new(dtype, shape, bytes).map_err(upload_error)?;
    let tensor = DynTensor::from_data(&host, plan.device()).map_err(upload_error)?;
    if dtype == compute {
        Ok(tensor)
    } else {
        tensor.cast(compute).map_err(upload_error)
    }
}

fn copy(name: &str, data: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(data.len())
        .map_err(|_| weight_error(name, WeightProblem::Allocation { bytes: data.len() }))?;
    out.extend_from_slice(data);
    Ok(out)
}

/// Every graph param a node uses, with the body prefixes expanded (graph-ir.md §2.6). A prefix
/// without `{i}` is the same in every iteration and is expanded once.
fn used_params(plan: &Plan) -> Result<HashSet<String>> {
    fn walk(block: &Block, prefix: &str, out: &mut HashSet<String>) {
        for step in &block.steps {
            match &step.kind {
                StepKind::Op(op) => {
                    for name in op.params.iter().flatten() {
                        out.insert(format!("{prefix}{name}"));
                    }
                }
                StepKind::Repeat(r) => {
                    let iterations = if r.params_prefix.contains("{i}") {
                        r.count
                    } else {
                        1
                    };
                    for i in 0..iterations {
                        let inner = format!("{prefix}{}", expand_prefix(&r.params_prefix, i));
                        walk(&r.body, &inner, out);
                    }
                }
            }
        }
    }
    let mut out = HashSet::new();
    walk(&plan.program.root, "", &mut out);
    let declared: HashSet<&str> = plan.params.iter().map(|(n, _)| n.as_str()).collect();
    if let Some(missing) = out.iter().find(|n| !declared.contains(n.as_str())) {
        return Err(RuntimeError::internal(format!(
            "a node uses param `{missing}`, which the graph doesn't declare, after validation"
        )));
    }
    Ok(out)
}
