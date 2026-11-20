//! Shared op reference-test harness (docs/testing-strategy.md §3.2, §6).
//!
//! Each op family has its own test binary, `tests/reference_<family>.rs`, which includes this
//! file with `#[path = "reference/common.rs"] mod common;` and gives [`run_cpu`] / [`run_gpu`] a
//! function that runs one case ([`run_cpu_on`] / [`run_gpu_on`] for ops that also need the device,
//! such as the creation ops). The harness:
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
//!
//! **Params** (graph-ir.md §2.4): a sidecar of an op with params has a `params` list after
//! `inputs` (testing-strategy.md §3.2): the params that are present, in catalog order. The harness
//! uploads them like the inputs and passes them to the run function **after the inputs**, in that
//! order; [`Case::params`] has their names, so a run function can tell which optional ones are
//! there. Sidecars without `params` work as before.
//!
//! A run function may call [`feed_layout`] to get the first input of a case as a non-contiguous
//! view (permuted, transposed, expanded or a stepped slice) when the case name asks for one
//! ([`Layout`]); the sidecar and fixture stay plain.

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
#[derive(Clone, Debug)]
pub struct Case {
    pub op: String,
    pub name: String,
    pub attributes: Map<String, Value>,
    /// Inputs in the op's order.
    pub inputs: Vec<TensorData>,
    /// Params that are present, in catalog order, with their names (empty for ops without
    /// params). The run function gets them after the inputs.
    pub params: Vec<(String, TensorData)>,
    /// Expected outputs in order, with their names.
    pub outputs: Vec<(String, TensorData)>,
    pub tolerance: Tolerance,
    /// The names of `inputs`, in order.
    pub input_names: Vec<String>,
    /// `"inference"`, or `"training"` for the training-mode gradient cases (schema 2,
    /// testing-strategy.md §3.6). Forward cases are `"inference"`.
    pub mode: String,
    /// The `mask` of a schema-2 case (`nn.dropout` in training mode). The run function gets it
    /// after the params.
    pub mask: Option<TensorData>,
    /// Whether [`feed_layout`] still has to build the case's input view. The gradient harness
    /// builds it before it marks the inputs as leaves and clears this.
    pub layout_pending: bool,
}

impl Case {
    /// The dtype under test, for the summary: the last input's dtype (for `where`, the dtype of
    /// `a`/`b`, not of `cond`), with the target for ops with a `dtype` attribute (`f32->i64`),
    /// and the params' dtype where it differs from the input's (`nn.embedding`: `i64+f16`).
    pub fn label(&self) -> String {
        let mut input = self
            .inputs
            .last()
            .map_or("none", |data| data.dtype.name())
            .to_string();
        if let Some((_, param)) = self.params.first()
            && param.dtype.name() != input
        {
            input = format!("{input}+{}", param.dtype.name());
        }
        match self.attributes.get("dtype").and_then(Value::as_str) {
            Some(target) => format!("{input}->{target}"),
            None => input,
        }
    }

    /// A `scalar` attribute (graph-ir.md §3.1.7): a JSON integer is [`Scalar::Int`], any other
    /// number or `"nan"`/`"inf"`/`"-inf"` is [`Scalar::Float`], a bool is [`Scalar::Bool`].
    pub fn scalar(&self, name: &str) -> Option<Scalar> {
        let value = self.attributes.get(name)?;
        Some(self.parse_scalar(name, value))
    }

    /// A `scalar[]` attribute (`core.constant` `values`), each entry as in [`Case::scalar`].
    pub fn scalars(&self, name: &str) -> Option<Vec<Scalar>> {
        let values = self.attributes.get(name)?.as_array().unwrap_or_else(|| {
            panic!(
                "{}/{}: attribute {name} is not an array",
                self.op, self.name
            )
        });
        Some(values.iter().map(|v| self.parse_scalar(name, v)).collect())
    }

    fn parse_scalar(&self, name: &str, value: &Value) -> Scalar {
        match value {
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
        }
    }

    /// An `int` (or literal `dim`) attribute.
    pub fn int(&self, name: &str) -> Option<i64> {
        let value = self.attributes.get(name)?;
        Some(value.as_i64().unwrap_or_else(|| {
            panic!(
                "{}/{}: attribute {name} is not an integer: {value}",
                self.op, self.name
            )
        }))
    }

    /// An `int[]` (or literal `dim[]`) attribute.
    pub fn ints(&self, name: &str) -> Option<Vec<i64>> {
        let value = self.attributes.get(name)?;
        let list = value
            .as_array()
            .and_then(|a| a.iter().map(Value::as_i64).collect());
        Some(list.unwrap_or_else(|| {
            panic!(
                "{}/{}: attribute {name} is not an integer list: {value}",
                self.op, self.name
            )
        }))
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
    list_sidecars(&fixture_dir(op))
        .iter()
        .map(|path| load_case(op, path))
        .collect()
}

/// The `.json` sidecars of a fixture directory, sorted. Panics (fails the test) if the directory
/// is missing or holds none.
pub fn list_sidecars(dir: &Path) -> Vec<PathBuf> {
    let entries = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("missing fixture directory {}: {err}", dir.display()));
    let mut sidecars: Vec<PathBuf> = entries
        .map(|entry| entry.expect("readable fixture directory").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    sidecars.sort();
    assert!(!sidecars.is_empty(), "no fixtures in {}", dir.display());
    sidecars
}

fn load_case(op: &str, sidecar: &Path) -> Case {
    let (json, file) = read_sidecar(op, sidecar, 1);
    case_from(op, sidecar, &json, &file)
}

/// Reads a sidecar and its safetensors file and checks `schema_version`, `op` and `case`.
pub fn read_sidecar(op: &str, sidecar: &Path, schema: u64) -> (Value, SafetensorsFile) {
    let text = std::fs::read_to_string(sidecar)
        .unwrap_or_else(|err| panic!("{}: {err}", sidecar.display()));
    let json: Value =
        serde_json::from_str(&text).unwrap_or_else(|err| panic!("{}: {err}", sidecar.display()));
    assert_eq!(
        sidecar_field(&json, sidecar, "schema_version"),
        schema,
        "{}",
        sidecar.display()
    );
    assert_eq!(
        sidecar_field(&json, sidecar, "op"),
        op,
        "{}",
        sidecar.display()
    );
    assert_eq!(
        sidecar_field(&json, sidecar, "case"),
        case_stem(sidecar),
        "{}",
        sidecar.display()
    );
    let tensors = sidecar.with_extension("safetensors");
    let file = SafetensorsFile::open(&tensors)
        .unwrap_or_else(|err| panic!("missing or bad fixture {}: {err}", tensors.display()));
    (json, file)
}

/// A required sidecar key.
pub fn sidecar_field<'a>(json: &'a Value, sidecar: &Path, key: &str) -> &'a Value {
    json.get(key)
        .unwrap_or_else(|| panic!("{}: missing key {key}", sidecar.display()))
}

fn case_stem(sidecar: &Path) -> &str {
    sidecar
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_default()
}

/// Loads the tensor a sidecar entry `{"name", "dtype", "shape"}` names, checking its dtype and
/// shape against the safetensors header.
pub fn load_tensor(file: &SafetensorsFile, sidecar: &Path, spec: &Value) -> (String, TensorData) {
    let tensors = sidecar.with_extension("safetensors");
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
}

/// Every tensor of a sidecar list (`inputs`, `outputs`, …).
pub fn load_list(
    json: &Value,
    file: &SafetensorsFile,
    sidecar: &Path,
    key: &str,
) -> Vec<(String, TensorData)> {
    sidecar_field(json, sidecar, key)
        .as_array()
        .unwrap_or_else(|| panic!("{}: {key} is not an array", sidecar.display()))
        .iter()
        .map(|spec| load_tensor(file, sidecar, spec))
        .collect()
}

/// A `{"exact", "atol", "rtol"}` object.
pub fn parse_tolerance(value: &Value) -> Tolerance {
    let number = |key: &str| value[key].as_f64().expect("tolerance number");
    Tolerance {
        exact: value["exact"].as_bool().expect("tolerance.exact"),
        atol: number("atol"),
        rtol: number("rtol"),
    }
}

/// The [`Case`] of a sidecar (the keys schema 1 and 2 share).
pub fn case_from(op: &str, sidecar: &Path, json: &Value, file: &SafetensorsFile) -> Case {
    // Only ops with params have the key (testing-strategy.md §3.2).
    let params = if json.get("params").is_some() {
        load_list(json, file, sidecar, "params")
    } else {
        Vec::new()
    };
    let (input_names, inputs) = load_list(json, file, sidecar, "inputs").into_iter().unzip();
    Case {
        op: op.to_string(),
        name: case_stem(sidecar).to_string(),
        attributes: sidecar_field(json, sidecar, "attributes")
            .as_object()
            .expect("attributes object")
            .clone(),
        inputs,
        params,
        outputs: load_list(json, file, sidecar, "outputs"),
        tolerance: parse_tolerance(sidecar_field(json, sidecar, "tolerance")),
        input_names,
        mode: json
            .get("mode")
            .map_or("inference", |m| m.as_str().expect("mode is a string"))
            .to_string(),
        mask: json
            .get("mask")
            .map(|spec| load_tensor(file, sidecar, spec).1),
        layout_pending: true,
    }
}

/// Runs one case: inputs, then the params ([`Case::params`]), on the device under test in;
/// outputs out.
pub type RunCase = fn(&Case, Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError>;

/// Like [`RunCase`], with the device under test, for ops that create tensors (`core.zeros`, …).
pub type RunCaseOn = fn(&Case, Vec<DynTensor>, DeviceId) -> Result<Vec<DynTensor>, TensorError>;

/// Runs every case of `op` on the CPU.
pub fn run_cpu(op: &str, run: RunCase) {
    run_on(op, &[DeviceId::Cpu], run);
}

/// Runs every case of `op` on every GPU device of every compiled-in GPU backend.
pub fn run_gpu(op: &str, run: RunCase) {
    run_gpu_with(op, &move |case, inputs, _| run(case, inputs));
}

/// [`run_cpu`] for a [`RunCaseOn`].
pub fn run_cpu_on(op: &str, run: RunCaseOn) {
    run_with(op, &[DeviceId::Cpu], &run);
}

/// [`run_gpu`] for a [`RunCaseOn`].
pub fn run_gpu_on(op: &str, run: RunCaseOn) {
    run_gpu_with(op, &run);
}

fn run_gpu_with(op: &str, run: &Runner) {
    let devices = gpu_devices();
    if devices.is_empty() {
        eprintln!("SKIP {op}: no GPU device to test (see the messages above)");
        return;
    }
    run_with(op, &devices, run);
}

type Runner = dyn Fn(&Case, Vec<DynTensor>, DeviceId) -> Result<Vec<DynTensor>, TensorError>;

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
pub struct Tally {
    pub passed: usize,
    pub skipped: usize,
    pub failed: usize,
    /// Cases whose rerun with tracked inputs (float inputs and params marked with
    /// `require_grad`) gave the same outputs as the untracked run (by [`compare`]'s exact rule:
    /// bit for bit, except that any NaN matches any NaN). Every case with a float input or param
    /// whose untracked outputs could be read back has one, also when those outputs miss the
    /// fixture.
    pub tracked: usize,
    /// Cases whose tracked rerun gave different outputs (a failure).
    pub tracked_differ: usize,
}

/// Runs every case of `op` on `devices` and fails the test if any case fails.
pub fn run_on(op: &str, devices: &[DeviceId], run: RunCase) {
    run_with(op, devices, &move |case, inputs, _| run(case, inputs));
}

fn run_with(op: &str, devices: &[DeviceId], run: &Runner) {
    let cases = load_cases(op);
    let mut failures = Vec::new();
    let mut tally: BTreeMap<(String, String), Tally> = BTreeMap::new();
    for &device in devices {
        for case in &cases {
            let entry = tally.entry((device.to_string(), case.label())).or_default();
            let (outcome, tracked) = run_case(case, device, run);
            match tracked {
                Tracked::NotRun => {}
                Tracked::Identical => entry.tracked += 1,
                Tracked::Differ => entry.tracked_differ += 1,
            }
            match outcome {
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
        let _ = writeln!(
            summary,
            "TRACKED {op} {device} {label}: {} identical with tracked inputs, {} differ",
            t.tracked, t.tracked_differ
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

/// The result of a case's rerun with tracked inputs.
enum Tracked {
    /// No rerun: no float input or param, or the untracked run gave no outputs to compare.
    NotRun,
    Identical,
    Differ,
}

/// Runs a case and, where it has a float input or param, reruns it with those inputs tracked.
/// The rerun runs whether or not the untracked outputs meet the fixture (the CUDA f32 `matmul`
/// cases of #64 don't): tracking must not change a value even where the backend misses the
/// reference.
fn run_case(case: &Case, device: DeviceId, run: &Runner) -> (Outcome, Tracked) {
    let mut dtypes: Vec<DType> = case.inputs.iter().map(|d| d.dtype).collect();
    dtypes.extend(case.params.iter().map(|(_, d)| d.dtype));
    dtypes.extend(case.outputs.iter().map(|(_, d)| d.dtype));
    for dtype in dtypes {
        match DynTensor::device_supports(device, dtype) {
            Ok(true) => {}
            Ok(false) => {
                let reason = format!("dtype {dtype} is not supported");
                return (Outcome::Skipped(reason), Tracked::NotRun);
            }
            Err(err) => {
                let reason = format!("device_supports: {err}");
                return (Outcome::Failed(reason), Tracked::NotRun);
            }
        }
    }
    let outputs = match upload_and_run(case, device, run, false) {
        Ok(outputs) => outputs,
        Err(outcome) => return (outcome, Tracked::NotRun),
    };
    if outputs.len() != case.outputs.len() {
        let reason = format!("{} outputs, expected {}", outputs.len(), case.outputs.len());
        return (Outcome::Failed(reason), Tracked::NotRun);
    }
    let mut problems = String::new();
    let mut read = Vec::with_capacity(outputs.len());
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
                read.push(actual);
            }
            Err(err) => {
                let _ = write!(problems, "\n  {name}: read back: {err}");
            }
        }
    }
    let has_float = case
        .inputs
        .iter()
        .chain(case.params.iter().map(|(_, data)| data))
        .any(|data| data.dtype.is_float());
    let tracked = if has_float && read.len() == case.outputs.len() {
        match tracked_rerun(case, device, run, &read) {
            Ok(()) => Tracked::Identical,
            Err(message) => {
                let _ = write!(
                    problems,
                    "\n  tracked rerun differs from the untracked run:{message}"
                );
                Tracked::Differ
            }
        }
    } else {
        Tracked::NotRun
    };
    if problems.is_empty() {
        (Outcome::Passed, tracked)
    } else {
        (Outcome::Failed(problems), tracked)
    }
}

/// Uploads the case's inputs and params to `device`, marks the float ones as leaves if `track`,
/// and runs the op. An `UnsupportedDType` from the op is a skip.
fn upload_and_run(
    case: &Case,
    device: DeviceId,
    run: &Runner,
    track: bool,
) -> Result<Vec<DynTensor>, Outcome> {
    let mut inputs = Vec::with_capacity(case.inputs.len() + case.params.len());
    let params = case.params.iter().map(|(_, data)| data);
    for data in case.inputs.iter().chain(params) {
        let tensor = DynTensor::from_data(data, device).and_then(|t| {
            if track && t.dtype().is_float() {
                t.require_grad()
            } else {
                Ok(t)
            }
        });
        match tensor {
            Ok(t) => inputs.push(t),
            Err(err) => return Err(Outcome::Failed(format!("upload: {err}"))),
        }
    }
    match run(case, inputs, device) {
        Ok(outputs) => Ok(outputs),
        // An op that lacks an implementation for a dtype on this device (graph-ir.md §2.2).
        Err(err @ TensorError::UnsupportedDType { .. }) => {
            Err(Outcome::Skipped(format!("the op reports: {err}")))
        }
        Err(err) => Err(Outcome::Failed(format!("op failed: {err}"))),
    }
}

/// Runs the case again with every float input and param marked as a leaf
/// ([`DynTensor::require_grad`]) and requires the outputs to equal `untracked` by
/// [`same_values`]: tracking must not change a value (#75).
fn tracked_rerun(
    case: &Case,
    device: DeviceId,
    run: &Runner,
    untracked: &[TensorData],
) -> Result<(), String> {
    let outputs = match upload_and_run(case, device, run, true) {
        Ok(outputs) => outputs,
        Err(Outcome::Skipped(reason) | Outcome::Failed(reason)) => {
            return Err(format!(" {reason}"));
        }
        Err(Outcome::Passed) => return Err(" no outputs".into()),
    };
    if outputs.len() != untracked.len() {
        return Err(format!(
            "{} outputs, expected {}",
            outputs.len(),
            untracked.len()
        ));
    }
    let mut problems = String::new();
    for ((actual, expected), (name, _)) in outputs.into_iter().zip(untracked).zip(&case.outputs) {
        match actual.into_data() {
            Ok(actual) => {
                if let Err(message) = same_values(expected, &actual) {
                    let _ = write!(problems, "\n  {name}: {message}");
                }
            }
            Err(err) => {
                let _ = write!(problems, "\n  {name}: read back: {err}");
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}

/// Whether two tensors hold the same values, by the rule of testing-strategy.md §3.2 for
/// `exact: true`: same dtype and shape, every element bit for bit (`-0` and `+0` differ), except
/// that any NaN matches any NaN (NaN payloads and signs are not part of op semantics,
/// graph-ir.md §3.1.8). Used wherever a tracked run is compared with an untracked one. `Err`
/// describes the first mismatch like [`compare`].
pub fn same_values(expected: &TensorData, actual: &TensorData) -> Result<(), String> {
    let exact = Tolerance {
        exact: true,
        atol: 0.0,
        rtol: 0.0,
    };
    compare(expected, actual, exact)
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

/// How [`feed_layout`] feeds a case's first input, chosen by the case name's prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// No prefix: the input as uploaded (contiguous).
    Contiguous,
    /// `permuted_`: uploaded with its dims reversed, then permuted back on the device.
    Permuted,
    /// `transposed_`: uploaded with its last two dims swapped, then transposed back.
    Transposed,
    /// `expanded_`: its first row uploaded alone, then expanded along dim 0 (stride 0). The
    /// fixture's input repeats one row along dim 0.
    Expanded,
    /// `strided_`: every row along dim 0 uploaded twice, then every second row sliced out from row
    /// 1 (a view with an offset and a step).
    Strided,
}

impl Layout {
    /// The layout a case name asks for.
    pub fn of(case: &str) -> Layout {
        [
            ("permuted_", Layout::Permuted),
            ("transposed_", Layout::Transposed),
            ("expanded_", Layout::Expanded),
            ("strided_", Layout::Strided),
        ]
        .into_iter()
        .find(|(prefix, _)| case.starts_with(prefix))
        .map_or(Layout::Contiguous, |(_, layout)| layout)
    }
}

/// Feeds the first input of a case as a non-contiguous view if the case name asks for one
/// ([`Layout`]), so that layout bugs of an op show up. Sidecars and fixtures stay plain: the input
/// tensor holds the logical values, and the view is built on the device with the shape ops, then
/// checked to read back exactly those values. Ops call this from their [`RunCase`]; cases without
/// a prefix are unchanged.
pub fn feed_layout(case: &Case, mut inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let layout = Layout::of(&case.name);
    if layout == Layout::Contiguous || inputs.is_empty() || !case.layout_pending {
        return Ok(inputs);
    }
    let x = inputs.remove(0);
    // A tracked input (the tracked rerun) is rebuilt as a view of an untracked upload and marked
    // again: the forward value is what that rerun checks.
    let tracked = x.is_tracked();
    let (device, shape, rank) = (x.device(), x.shape(), x.rank());
    let upload = |t: DynTensor| -> Result<DynTensor, TensorError> {
        DynTensor::from_data(&t.into_data()?, device)
    };
    let view = match layout {
        Layout::Contiguous => x.clone(),
        Layout::Permuted => {
            assert!(
                rank >= 2,
                "{}/{}: permuted_ needs rank ≥ 2",
                case.op,
                case.name
            );
            let reversed: Vec<usize> = (0..rank).rev().collect();
            upload(x.clone().permute(&reversed)?)?.permute(&reversed)?
        }
        Layout::Transposed => {
            assert!(
                rank >= 2,
                "{}/{}: transposed_ needs rank ≥ 2",
                case.op,
                case.name
            );
            upload(x.clone().transpose(rank - 2, rank - 1)?)?.transpose(rank - 2, rank - 1)?
        }
        Layout::Expanded => upload(x.clone().slice(0, 0, 1, 1)?)?.expand(&shape)?,
        Layout::Strided => {
            let rows = x.clone().unsqueeze(1)?;
            let doubled = DynTensor::concat(&[rows.clone(), rows], 1)?;
            let mut tall = shape.clone();
            tall[0] *= 2;
            upload(doubled.reshape(&tall)?)?.slice(0, 1, tall[0], 2)?
        }
    };
    assert_eq!(
        view.to_data()?,
        x.to_data()?,
        "{}/{}: the {layout:?} view doesn't hold the fixture's input (an expanded_ input must \
         repeat one row along dim 0)",
        case.op,
        case.name
    );
    let view = if tracked { view.require_grad()? } else { view };
    inputs.insert(0, view);
    Ok(inputs)
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
    // `same_values`: NaN payloads don't matter (the GPUs' canonical NaN 0x7FFFFFFF against the
    // CPU's 0x7FC00000), signed zeros and every other bit do.
    let quiet = f32::from_bits(0x7FC0_0000);
    let all_ones = f32::from_bits(0x7FFF_FFFF);
    let negative_nan = f32::from_bits(0xFFC0_0001);
    let v = f32s(&[quiet, 0.0, -0.0, 1.5]);
    assert!(same_values(&v, &f32s(&[all_ones, 0.0, -0.0, 1.5])).is_ok());
    assert!(same_values(&v, &f32s(&[negative_nan, 0.0, -0.0, 1.5])).is_ok());
    assert!(same_values(&v, &f32s(&[quiet, -0.0, -0.0, 1.5])).is_err());
    assert!(same_values(&v, &f32s(&[quiet, 0.0, 0.0, 1.5])).is_err());
    // A real difference fails: one ulp, a NaN for a number, a number for a NaN.
    let next = f32::from_bits(1.5f32.to_bits() + 1);
    assert!(same_values(&v, &f32s(&[quiet, 0.0, -0.0, next])).is_err());
    assert!(same_values(&v, &f32s(&[quiet, 0.0, -0.0, f32::NAN])).is_err());
    assert!(same_values(&v, &f32s(&[1.0, 0.0, -0.0, 1.5])).is_err());
    assert!(compare(&e, &f32s(&[1.0, f32::NAN, f32::INFINITY, 0.0]), exact).is_err());
    let message = compare(&e, &f32s(&[1.5, f32::NAN, f32::INFINITY, 0.0]), tol).unwrap_err();
    assert!(message.contains("worst at [0]"), "{message}");
    assert_eq!(unravel(7, &[2, 2, 3]), vec![1, 0, 1]);
}
