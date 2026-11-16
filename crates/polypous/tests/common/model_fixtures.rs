//! End-to-end model fixtures (docs/testing-strategy.md §3.1, §3.5): loading the cases of
//! `fixtures/small/models/<model>/` and comparing outputs with a case's tolerance (§3.2).
//!
//! Shared by `crates/polypous/tests/models.rs` (the facade on every device) and
//! `crates/polypous-cli/tests/models.rs` (`polyp run`), which include this file with `#[path]`.
//! Only the public `polypous` API is used. A missing or malformed fixture fails the test, never
//! skips it (§3.3).

#![allow(
    dead_code,
    reason = "each test binary uses a different part of this module"
)]

use std::path::{Path, PathBuf};

use polypous::{ComputeDtype, Dtype, NamedTensor, TensorData, tensor_file};
use serde_json::Value;

/// The case sidecar tolerance (testing-strategy.md §3.2).
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub exact: bool,
    pub atol: f64,
    pub rtol: f64,
}

/// One case of a model: its inputs and expected outputs, by name, in the graph's order.
#[derive(Debug)]
pub struct ModelCase {
    pub model: String,
    pub name: String,
    pub compute_dtype: ComputeDtype,
    pub inputs: Vec<NamedTensor>,
    pub outputs: Vec<NamedTensor>,
    pub tolerance: Tolerance,
}

/// `fixtures/small/models/<model>/`, from the repository root. `manifest_dir` is the test crate's
/// `CARGO_MANIFEST_DIR` (both crates sit in `crates/`).
pub fn model_dir(manifest_dir: &str, model: &str) -> PathBuf {
    Path::new(manifest_dir)
        .join("../../fixtures/small/models")
        .join(model)
}

/// The graph document of a model directory.
pub fn graph_path(dir: &Path) -> PathBuf {
    dir.join("model.graph.json")
}

/// The weights of a model directory.
pub fn weights_path(dir: &Path) -> PathBuf {
    dir.join("model.safetensors")
}

/// Every case of a model directory, sorted by name. Panics if there is none or one is malformed.
pub fn load_cases(dir: &Path) -> Vec<ModelCase> {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("missing model fixture directory {}: {err}", dir.display()));
    let mut sidecars: Vec<PathBuf> = entries
        .map(|entry| entry.expect("readable fixture directory").path())
        .filter(|path| {
            path.extension().is_some_and(|ext| ext == "json")
                && path.file_name().is_some_and(|n| n != "model.graph.json")
        })
        .collect();
    sidecars.sort();
    assert!(!sidecars.is_empty(), "no cases in {}", dir.display());
    sidecars.iter().map(|path| load_case(path)).collect()
}

fn load_case(sidecar: &Path) -> ModelCase {
    let show = sidecar.display();
    let text = std::fs::read_to_string(sidecar).unwrap_or_else(|err| panic!("{show}: {err}"));
    let json: Value = serde_json::from_str(&text).unwrap_or_else(|err| panic!("{show}: {err}"));
    let field = |key: &str| {
        json.get(key)
            .unwrap_or_else(|| panic!("{show}: missing key {key}"))
    };
    assert_eq!(field("schema_version"), 1, "{show}");
    let stem = sidecar
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    assert_eq!(field("case"), stem, "{show}");
    let compute_dtype: ComputeDtype = serde_json::from_value(field("compute_dtype").clone())
        .unwrap_or_else(|err| panic!("{show}: compute_dtype: {err}"));
    let tensors_path = sidecar.with_extension("safetensors");
    let mut tensors = tensor_file::read(&tensors_path).unwrap_or_else(|err| {
        panic!(
            "missing or bad fixture {}: {}",
            tensors_path.display(),
            err.message
        )
    });
    let mut take = |key: &str| -> Vec<NamedTensor> {
        let specs = field(key)
            .as_array()
            .unwrap_or_else(|| panic!("{show}: {key} is not an array"));
        specs
            .iter()
            .map(|spec| {
                let name = spec["name"].as_str().expect("tensor name");
                let dtype: Dtype = serde_json::from_value(spec["dtype"].clone())
                    .unwrap_or_else(|err| panic!("{show}: dtype of {name}: {err}"));
                let shape: Vec<u64> = serde_json::from_value(spec["shape"].clone())
                    .unwrap_or_else(|err| panic!("{show}: shape of {name}: {err}"));
                let at = tensors
                    .iter()
                    .position(|t| t.name == name)
                    .unwrap_or_else(|| panic!("{}: no tensor {name}", tensors_path.display()));
                let tensor = tensors.swap_remove(at);
                assert_eq!(tensor.tensor.dtype, dtype, "{show}: dtype of {name}");
                assert_eq!(tensor.tensor.shape, shape, "{show}: shape of {name}");
                tensor
            })
            .collect()
    };
    let inputs = take("inputs");
    let outputs = take("outputs");
    assert!(
        tensors.is_empty(),
        "{}: undeclared tensors",
        tensors_path.display()
    );
    let tolerance = field("tolerance");
    let number = |key: &str| tolerance[key].as_f64().expect("tolerance number");
    ModelCase {
        model: field("model").as_str().expect("model name").to_string(),
        name: stem.to_string(),
        compute_dtype,
        inputs,
        outputs,
        tolerance: Tolerance {
            exact: tolerance["exact"].as_bool().expect("tolerance.exact"),
            atol: number("atol"),
            rtol: number("rtol"),
        },
    }
}

/// Compares the outputs of a run with a case's expected outputs, by name. `Err` lists every
/// mismatch, one per line.
pub fn compare_outputs(case: &ModelCase, actual: &[NamedTensor]) -> Result<(), String> {
    let mut problems = Vec::new();
    for expected in &case.outputs {
        match actual.iter().find(|t| t.name == expected.name) {
            None => problems.push(format!("{}: missing from the run's outputs", expected.name)),
            Some(got) => {
                if let Err(message) = compare(&expected.tensor, &got.tensor, case.tolerance) {
                    problems.push(format!("{}: {message}", expected.name));
                }
            }
        }
    }
    if actual.len() != case.outputs.len() {
        problems.push(format!(
            "{} outputs, expected {}",
            actual.len(),
            case.outputs.len()
        ));
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems.join("\n"))
    }
}

/// Compares an output with its expected tensor as testing-strategy.md §3.2 defines it (the same
/// rule as the op harness, `crates/polypous-backend/tests/reference/common.rs`). `Err` gives the
/// number of bad elements, the max abs and rel error and the worst element.
pub fn compare(expected: &TensorData, actual: &TensorData, tol: Tolerance) -> Result<(), String> {
    if actual.dtype != expected.dtype {
        return Err(format!(
            "dtype {}, expected {}",
            actual.dtype.as_str(),
            expected.dtype.as_str()
        ));
    }
    if actual.shape != expected.shape {
        return Err(format!(
            "shape {:?}, expected {:?}",
            actual.shape, expected.shape
        ));
    }
    let (e, a) = (floats(expected), floats(actual));
    let (eb, ab) = (float_bits(expected), float_bits(actual));
    let mut bad = 0usize;
    let mut max_abs = 0.0f64;
    let mut max_rel = 0.0f64;
    // Worst element: a NaN/inf mismatch beats any finite error; otherwise the largest error
    // relative to the allowed one.
    let mut worst: Option<(usize, f64)> = None;
    for i in 0..e.len() {
        let (x, y) = (e[i], a[i]);
        let ok;
        let mut score = f64::INFINITY;
        if tol.exact {
            ok = eb[i] == ab[i] || (x.is_nan() && y.is_nan());
            if !ok && x.is_finite() && y.is_finite() {
                score = (y - x).abs();
            }
        } else if x.is_nan() || y.is_nan() {
            ok = x.is_nan() && y.is_nan();
        } else if x.is_infinite() || y.is_infinite() {
            ok = x == y;
        } else {
            let err = (y - x).abs();
            let allowed = tol.atol + tol.rtol * x.abs();
            ok = err <= allowed;
            score = if allowed > 0.0 { err / allowed } else { err };
        }
        if x.is_finite() && y.is_finite() {
            max_abs = max_abs.max((y - x).abs());
            if x != 0.0 {
                max_rel = max_rel.max((y - x).abs() / x.abs());
            }
        }
        if !ok {
            bad += 1;
            if worst.is_none_or(|(_, s)| score > s) {
                worst = Some((i, score));
            }
        }
    }
    match worst {
        None => Ok(()),
        Some((i, _)) => Err(format!(
            "{bad} of {} elements outside the tolerance {tol:?}; max abs error {max_abs:e}, max rel \
             error {max_rel:e} (finite elements); worst at {:?}: expected {:e}, got {:e}",
            e.len(),
            unravel(i, &expected.shape),
            e[i],
            a[i]
        )),
    }
}

/// Float elements as `f64` (exact for f32, f16 and bf16). Panics for other dtypes: model
/// outputs here are floats.
pub fn floats(data: &TensorData) -> Vec<f64> {
    float_bits(data)
        .into_iter()
        .map(|bits| match data.dtype {
            Dtype::F32 => f64::from(f32::from_bits(bits)),
            Dtype::F16 => f16_to_f64(bits as u16),
            Dtype::Bf16 => f64::from(f32::from_bits(bits << 16)),
            other => panic!("not a float dtype: {}", other.as_str()),
        })
        .collect()
}

fn float_bits(data: &TensorData) -> Vec<u32> {
    match data.dtype {
        Dtype::F32 => {
            let (chunks, _) = data.bytes.as_chunks::<4>();
            chunks.iter().map(|&b| u32::from_le_bytes(b)).collect()
        }
        Dtype::F16 | Dtype::Bf16 => {
            let (chunks, _) = data.bytes.as_chunks::<2>();
            chunks
                .iter()
                .map(|&b| u32::from(u16::from_le_bytes(b)))
                .collect()
        }
        other => panic!("not a float dtype: {}", other.as_str()),
    }
}

/// IEEE 754 binary16 to f64 (exact).
fn f16_to_f64(bits: u16) -> f64 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let mantissa = f64::from(bits & 0x3ff);
    sign * match exponent {
        0 => mantissa * 2f64.powi(-24),
        31 if mantissa == 0.0 => f64::INFINITY,
        31 => f64::NAN,
        _ => (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// The multi-dimensional index of flat (row-major) index `i`.
fn unravel(mut i: usize, shape: &[u64]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for (slot, &dim) in index.iter_mut().zip(shape).rev() {
        let dim = dim as usize;
        *slot = i % dim;
        i /= dim;
    }
    index
}

#[test]
fn f16_decoding() {
    assert_eq!(f16_to_f64(0x3c00), 1.0);
    assert_eq!(f16_to_f64(0xc000), -2.0);
    assert_eq!(f16_to_f64(0x7bff), 65504.0);
    assert_eq!(f16_to_f64(0x0400), 2f64.powi(-14));
    assert_eq!(f16_to_f64(0x0001), 2f64.powi(-24));
    assert_eq!(f16_to_f64(0x3555), 0.333_251_953_125);
    assert_eq!(f16_to_f64(0xfc00), f64::NEG_INFINITY);
    assert!(f16_to_f64(0x7e00).is_nan());
}
