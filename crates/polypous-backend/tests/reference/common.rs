//! Shared op reference-test harness (docs/testing-strategy.md §3.2, §6).
//!
//! Each op family has its own test binary, `tests/reference_<family>.rs`, which includes this
//! file with `#[path = "reference/common.rs"] mod common;` and gives [`run_cpu`] / [`run_gpu`] a
//! function that runs one case. The harness:
//!
//! - loads every `.json` sidecar of `fixtures/small/ops/<op>/` and the safetensors file next to it
//!   through `polypous-formats`. A missing directory, an empty one or a missing tensor file is a
//!   test failure, never a skip (testing-strategy.md §3.3).
//! - runs every case on every device: the CPU in a normal test, every GPU device of every
//!   compiled-in GPU backend in a `#[ignore = "gpu"]` test. Software rasterizers (lavapipe) only
//!   with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, as in the other GPU tests.
//! - skips a case on a device that doesn't support one of its dtypes, and prints each skip with
//!   its reason (`SKIP …`), plus an op × device × dtype summary (`SUMMARY …`). Run with
//!   `--nocapture` to see them.
//! - compares each output with the sidecar tolerance exactly as testing-strategy.md §3.2 defines
//!   it, and on failure prints the max abs/rel error and the worst element's index.

#![allow(
    dead_code,
    reason = "each test binary uses a different part of the harness"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use polypous_backend::device::{BackendKind, DeviceId, EnumerateOptions, list_devices};
use polypous_backend::{DType, DynTensor, Scalar, TensorData, TensorError};
use polypous_formats::safetensors::{SafetensorsFile, bf16_to_f32, f16_to_f32};
use serde_json::{Map, Value};

/// The sidecar tolerance (testing-strategy.md §3.2).
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub exact: bool,
    pub atol: f64,
    pub rtol: f64,
}

/// One fixture case: sidecar fields and the tensors from its safetensors file.
#[derive(Debug)]
pub struct Case {
    pub op: String,
    pub name: String,
    pub attributes: Map<String, Value>,
    /// Inputs in the op's order.
    pub inputs: Vec<TensorData>,
    /// Expected outputs in order, with their names.
    pub outputs: Vec<(String, TensorData)>,
    pub tolerance: Tolerance,
}

impl Case {
    /// The dtype under test, for the summary: the last input's dtype (for `where`, the dtype of
    /// `a`/`b`, not of `cond`), with the target for ops with a `dtype` attribute (`f32->i64`).
    pub fn label(&self) -> String {
        let input = self
            .inputs
            .last()
            .map_or("none", |data| data.dtype.name())
            .to_string();
        match self.attributes.get("dtype").and_then(Value::as_str) {
            Some(target) => format!("{input}->{target}"),
            None => input,
        }
    }

    /// A `scalar` attribute (graph-ir.md §3.1.7): a JSON integer is [`Scalar::Int`], any other
    /// number or `"nan"`/`"inf"`/`"-inf"` is [`Scalar::Float`], a bool is [`Scalar::Bool`].
    pub fn scalar(&self, name: &str) -> Option<Scalar> {
        let value = self.attributes.get(name)?;
        Some(match value {
            Value::Bool(b) => Scalar::Bool(*b),
            Value::Number(n) => match n.as_i64() {
                Some(i) => Scalar::Int(i),
                None => Scalar::Float(n.as_f64().expect("a JSON number is an f64")),
            },
            Value::String(s) => Scalar::Float(match s.as_str() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                "-inf" => f64::NEG_INFINITY,
                other => panic!(
                    "{}/{}: bad scalar attribute {name}: {other:?}",
                    self.op, self.name
                ),
            }),
            other => panic!(
                "{}/{}: bad scalar attribute {name}: {other}",
                self.op, self.name
            ),
        })
    }

    /// A `dtype` attribute.
    pub fn dtype(&self, name: &str) -> DType {
        let value = self.attributes.get(name).and_then(Value::as_str);
        value.and_then(DType::from_name).unwrap_or_else(|| {
            panic!(
                "{}/{}: bad dtype attribute {name}: {value:?}",
                self.op, self.name
            )
        })
    }
}

/// `fixtures/small/ops/<op>/` from the repository root.
pub fn fixture_dir(op: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/small/ops")
        .join(op)
}

/// Every case of `op`, sorted by name. Panics (fails the test) if a fixture is missing or
/// malformed.
pub fn load_cases(op: &str) -> Vec<Case> {
    let dir = fixture_dir(op);
    let entries = std::fs::read_dir(&dir)
        .unwrap_or_else(|err| panic!("missing fixture directory {}: {err}", dir.display()));
    let mut sidecars: Vec<PathBuf> = entries
        .map(|entry| entry.expect("readable fixture directory").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    sidecars.sort();
    assert!(!sidecars.is_empty(), "no fixtures in {}", dir.display());
    sidecars.iter().map(|path| load_case(op, path)).collect()
}

fn load_case(op: &str, sidecar: &Path) -> Case {
    let text = std::fs::read_to_string(sidecar)
        .unwrap_or_else(|err| panic!("{}: {err}", sidecar.display()));
    let json: Value =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", sidecar.display()));
    let field = |key: &str| {
        json.get(key)
            .unwrap_or_else(|| panic!("{}: missing key {key}", sidecar.display()))
    };
    assert_eq!(field("schema_version"), 1, "{}", sidecar.display());
    assert_eq!(field("op"), op, "{}", sidecar.display());
    let stem = sidecar
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default();
    assert_eq!(field("case"), stem, "{}", sidecar.display());

    let tensors = sidecar.with_extension("safetensors");
    let file = SafetensorsFile::open(&tensors)
        .unwrap_or_else(|err| panic!("missing or bad fixture {}: {err}", tensors.display()));
    let load = |spec: &Value| -> (String, TensorData) {
        let name = spec["name"].as_str().expect("tensor name").to_string();
        let dtype_name = spec["dtype"].as_str().expect("tensor dtype");
        let dtype = DType::from_name(dtype_name)
            .unwrap_or_else(|| panic!("{}: unknown dtype {dtype_name}", sidecar.display()));
        let shape: Vec<u64> = spec["shape"]
            .as_array()
            .expect("tensor shape")
            .iter()
            .map(|d| d.as_u64().expect("dimension"))
            .collect();
        let view = file
            .tensor(&name)
            .unwrap_or_else(|err| panic!("{}: tensor {name}: {err}", tensors.display()));
        let stored: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
        assert_eq!(stored, shape, "{}: shape of {name}", tensors.display());
        assert_eq!(
            format!("{:?}", view.dtype()).to_lowercase(),
            dtype_name,
            "{}: dtype of {name}",
            tensors.display()
        );
        // safetensors and `TensorData` are both row-major and little-endian.
        let data = TensorData::new(dtype, shape, view.data().to_vec())
            .unwrap_or_else(|err| panic!("{}: tensor {name}: {err}", tensors.display()));
        (name, data)
    };
    let list = |key: &str| -> Vec<(String, TensorData)> {
        field(key)
            .as_array()
            .unwrap_or_else(|| panic!("{}: {key} is not an array", sidecar.display()))
            .iter()
            .map(load)
            .collect()
    };
    let tolerance = field("tolerance");
    let number = |key: &str| tolerance[key].as_f64().expect("tolerance number");
    Case {
        op: op.to_string(),
        name: stem.to_string(),
        attributes: field("attributes")
            .as_object()
            .expect("attributes object")
            .clone(),
        inputs: list("inputs").into_iter().map(|(_, data)| data).collect(),
        outputs: list("outputs"),
        tolerance: Tolerance {
            exact: tolerance["exact"].as_bool().expect("tolerance.exact"),
            atol: number("atol"),
            rtol: number("rtol"),
        },
    }
}

/// Runs one case: inputs (on the device under test) in, outputs out.
pub type RunCase = fn(&Case, Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError>;

/// Runs every case of `op` on the CPU.
pub fn run_cpu(op: &str, run: RunCase) {
    run_on(op, &[DeviceId::Cpu], run);
}

/// Runs every case of `op` on every GPU device of every compiled-in GPU backend.
pub fn run_gpu(op: &str, run: RunCase) {
    let devices = gpu_devices();
    if devices.is_empty() {
        eprintln!("SKIP {op}: no GPU device to test (see the messages above)");
        return;
    }
    run_on(op, &devices, run);
}

/// Every listed GPU device, without software rasterizers unless `POLYPOUS_TEST_SOFTWARE_VULKAN=1`.
/// Panics if a GPU backend is compiled in but lists no device at all.
pub fn gpu_devices() -> Vec<DeviceId> {
    let listed: Vec<DeviceId> = list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu)
        .collect();
    assert!(
        !listed.is_empty(),
        "no GPU device found for the compiled-in GPU backends"
    );
    let options = EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    };
    let (skipped, devices): (Vec<DeviceId>, Vec<DeviceId>) =
        listed.into_iter().partition(|&id| options.skips(id));
    for id in skipped {
        eprintln!(
            "SKIP device {id}: software rasterizer (set POLYPOUS_TEST_SOFTWARE_VULKAN=1 to test it)"
        );
    }
    devices
}

#[derive(Default)]
struct Tally {
    passed: usize,
    skipped: usize,
    failed: usize,
}

/// Runs every case of `op` on `devices` and fails the test if any case fails.
pub fn run_on(op: &str, devices: &[DeviceId], run: RunCase) {
    let cases = load_cases(op);
    let mut failures = Vec::new();
    let mut tally: BTreeMap<(String, String), Tally> = BTreeMap::new();
    for &device in devices {
        for case in &cases {
            let entry = tally.entry((device.to_string(), case.label())).or_default();
            match run_case(case, device, run) {
                Outcome::Passed => entry.passed += 1,
                Outcome::Skipped(reason) => {
                    eprintln!("SKIP {op}/{} on {device}: {reason}", case.name);
                    entry.skipped += 1;
                }
                Outcome::Failed(message) => {
                    failures.push(format!("{op}/{} on {device}: {message}", case.name));
                    entry.failed += 1;
                }
            }
        }
    }
    // One write, so that the lines of parallel tests don't interleave.
    let mut summary = String::new();
    for ((device, label), t) in &tally {
        let _ = writeln!(
            summary,
            "SUMMARY {op} {device} {label}: {} passed, {} skipped, {} failed",
            t.passed, t.skipped, t.failed
        );
    }
    eprint!("{summary}");
    assert!(
        failures.is_empty(),
        "{} case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

enum Outcome {
    Passed,
    Skipped(String),
    Failed(String),
}

fn run_case(case: &Case, device: DeviceId, run: RunCase) -> Outcome {
    let mut dtypes: Vec<DType> = case.inputs.iter().map(|d| d.dtype).collect();
    dtypes.extend(case.outputs.iter().map(|(_, d)| d.dtype));
    for dtype in dtypes {
        match DynTensor::device_supports(device, dtype) {
            Ok(true) => {}
            Ok(false) => return Outcome::Skipped(format!("dtype {dtype} is not supported")),
            Err(err) => return Outcome::Failed(format!("device_supports: {err}")),
        }
    }
    let mut inputs = Vec::with_capacity(case.inputs.len());
    for data in &case.inputs {
        match DynTensor::from_data(data, device) {
            Ok(t) => inputs.push(t),
            Err(err) => return Outcome::Failed(format!("upload: {err}")),
        }
    }
    let outputs = match run(case, inputs) {
        Ok(outputs) => outputs,
        // An op that lacks an implementation for a dtype on this device (graph-ir.md §2.2).
        Err(err @ TensorError::UnsupportedDType { .. }) => {
            return Outcome::Skipped(format!("the op reports: {err}"));
        }
        Err(err) => return Outcome::Failed(format!("op failed: {err}")),
    };
    if outputs.len() != case.outputs.len() {
        return Outcome::Failed(format!(
            "{} outputs, expected {}",
            outputs.len(),
            case.outputs.len()
        ));
    }
    let mut problems = String::new();
    for (actual, (name, expected)) in outputs.into_iter().zip(&case.outputs) {
        if actual.device() != device {
            let _ = write!(problems, "\n  {name}: on {}, not {device}", actual.device());
            continue;
        }
        match actual.into_data() {
            Ok(actual) => {
                if let Err(message) = compare(expected, &actual, case.tolerance) {
                    let _ = write!(problems, "\n  {name}: {message}");
                }
            }
            Err(err) => {
                let _ = write!(problems, "\n  {name}: read back: {err}");
            }
        }
    }
    if problems.is_empty() {
        Outcome::Passed
    } else {
        Outcome::Failed(problems)
    }
}

/// Compares an output with its expected tensor (testing-strategy.md §3.2). `Err` describes the
/// mismatch: for tolerance comparisons the number of bad elements, max abs and rel error and the
/// worst element.
pub fn compare(expected: &TensorData, actual: &TensorData, tol: Tolerance) -> Result<(), String> {
    if actual.dtype != expected.dtype {
        return Err(format!(
            "dtype {}, expected {}",
            actual.dtype, expected.dtype
        ));
    }
    if actual.shape != expected.shape {
        return Err(format!(
            "shape {:?}, expected {:?}",
            actual.shape, expected.shape
        ));
    }
    let shape = &expected.shape;
    if !expected.dtype.is_float() {
        // Integers and bools: exact, element by element.
        let width = expected.dtype.size_bytes();
        let bad: Vec<usize> = expected
            .bytes
            .chunks_exact(width)
            .zip(actual.bytes.chunks_exact(width))
            .enumerate()
            .filter(|(_, (e, a))| e != a)
            .map(|(i, _)| i)
            .collect();
        return match bad.first() {
            None => Ok(()),
            Some(&i) => Err(format!(
                "{} of {} elements differ; first at {:?}: expected {}, got {}",
                bad.len(),
                expected.bytes.len() / width,
                unravel(i, shape),
                show_exact(expected, i),
                show_exact(actual, i)
            )),
        };
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
            unravel(i, shape),
            e[i],
            a[i]
        )),
    }
}

/// Float elements as `f64` (exact for f32, f16 and bf16).
fn floats(data: &TensorData) -> Vec<f64> {
    float_bits(data)
        .into_iter()
        .map(|bits| match data.dtype {
            DType::F32 => f64::from(f32::from_bits(bits)),
            DType::F16 => f64::from(f16_to_f32(bits as u16)),
            DType::BF16 => f64::from(bf16_to_f32(bits as u16)),
            other => panic!("not a float dtype: {other}"),
        })
        .collect()
}

/// Float elements as their bit patterns.
fn float_bits(data: &TensorData) -> Vec<u32> {
    match data.dtype {
        DType::F32 => {
            let (chunks, _) = data.bytes.as_chunks::<4>();
            chunks.iter().map(|&b| u32::from_le_bytes(b)).collect()
        }
        _ => {
            let (chunks, _) = data.bytes.as_chunks::<2>();
            chunks
                .iter()
                .map(|&b| u32::from(u16::from_le_bytes(b)))
                .collect()
        }
    }
}

/// Element `i` of an integer or bool tensor, as text.
fn show_exact(data: &TensorData, i: usize) -> String {
    let width = data.dtype.size_bytes();
    let b = &data.bytes[i * width..(i + 1) * width];
    match data.dtype {
        DType::I64 => i64::from_le_bytes(b.try_into().expect("8 bytes")).to_string(),
        DType::I32 => i32::from_le_bytes(b.try_into().expect("4 bytes")).to_string(),
        DType::Bool => (b[0] != 0).to_string(),
        _ => format!("{b:?}"),
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
fn compare_follows_the_tolerance_rules() {
    let f32s = |values: &[f32]| {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData::new(DType::F32, vec![values.len() as u64], bytes).unwrap()
    };
    let tol = Tolerance {
        exact: false,
        atol: 1e-5,
        rtol: 1e-4,
    };
    let exact = Tolerance {
        exact: true,
        atol: 0.0,
        rtol: 0.0,
    };
    let e = f32s(&[1.0, f32::NAN, f32::INFINITY, -0.0]);
    assert!(compare(&e, &f32s(&[1.00005, -f32::NAN, f32::INFINITY, 0.0]), tol).is_ok());
    assert!(compare(&e, &f32s(&[1.001, f32::NAN, f32::INFINITY, 0.0]), tol).is_err());
    assert!(compare(&e, &f32s(&[1.0, 0.0, f32::INFINITY, 0.0]), tol).is_err());
    assert!(compare(&e, &f32s(&[1.0, f32::NAN, f32::NEG_INFINITY, 0.0]), tol).is_err());
    assert!(compare(&e, &f32s(&[1.0, f32::NAN, f32::MAX, 0.0]), tol).is_err());
    // Exact: any NaN matches any NaN, but -0.0 is not 0.0.
    assert!(compare(&e, &f32s(&[1.0, -f32::NAN, f32::INFINITY, -0.0]), exact).is_ok());
    assert!(compare(&e, &f32s(&[1.0, f32::NAN, f32::INFINITY, 0.0]), exact).is_err());
    let message = compare(&e, &f32s(&[1.5, f32::NAN, f32::INFINITY, 0.0]), tol).unwrap_err();
    assert!(message.contains("worst at [0]"), "{message}");
    assert_eq!(unravel(7, &[2, 2, 3]), vec![1, 0, 1]);
}
