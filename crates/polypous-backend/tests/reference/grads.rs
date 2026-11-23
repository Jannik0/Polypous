//! Gradient reference-test harness (docs/testing-strategy.md §3.6, graph-ir.md §3.1.11).
//!
//! The gradient counterpart of `common.rs`, on the same pieces (sidecar and tensor loading,
//! [`compare`], [`feed_layout`], the device list). Each op family has a test binary
//! `tests/grad_reference_<family>.rs` that includes both files:
//!
//! ```text
//! #[path = "reference/common.rs"] mod common;
//! #[path = "reference/grads.rs"] mod grads;
//! ```
//!
//! and gives [`run_grad_cpu`] / [`run_grad_gpu`] the same kind of run function as the forward
//! harness ([`RunCase`]): inputs, then params, then the `mask` if the sidecar has one, in; the
//! outputs in port order, then the updated running statistics (`nn.batch_norm` in training mode),
//! out. The run function reads `case.mode` for the training-mode cases. Families share their
//! run function with the forward test (`tests/reference/run_<family>.rs`). The harness:
//!
//! - loads every `.json` sidecar (schema 2) of `fixtures/small/grads/<op>/` and its safetensors
//!   file. A missing directory, an empty one or a missing tensor file fails the test.
//! - runs every case on every device: the CPU in a normal test, every GPU device in a
//!   `#[ignore = "gpu"]` test (lavapipe only with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`).
//! - uploads the inputs and params, builds the first input's view if the case name asks for one
//!   ([`feed_layout`], before marking, so the view itself is the leaf), and marks every float
//!   input and param as a leaf ([`DynTensor::require_grad`]): those the sidecar lists a gradient
//!   for, and the others (`nn.batch_norm`'s running statistics), which must get none.
//! - compares the outputs and running statistics with `tolerance`, back-propagates the
//!   one-element loss `Σ_p sum(out_p · grad_p)` over the float outputs `p` (whose gradient with
//!   respect to `out_p` is exactly the upstream gradient `grad_p`, §3.6), and compares each leaf's
//!   gradient with `gradient_tolerance`: NaN must match NaN, `±inf` the same inf (§3.2). A leaf
//!   without a gradient is a failure, and so is a gradient of a float input or param that the
//!   sidecar lists none for. The returned running statistics must carry no autodiff context
//!   (runtime.md §3.3). On a mismatch it prints the max abs/rel error and the worst
//!   element's index.
//! - `same_value_*` cases (two float inputs holding the same values) run a second time with one
//!   leaf fed to both inputs; its gradient must match the sum of the two expected gradients
//!   (gradients add up over every use, §3.1.11).
//! - prints `SUMMARY grad …` lines per op × device. Run with `--nocapture` to see them.

#![allow(
    dead_code,
    reason = "each test binary uses a different part of the harness"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::common::{
    Case, RunCase, Tolerance, case_from, compare, feed_layout, gpu_devices, list_sidecars,
    load_list, parse_tolerance, read_sidecar, sidecar_field,
};
use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData, TensorError};

/// One gradient case: the forward fields as a [`Case`] (with `mode` and `mask`), plus the
/// upstream and expected gradients.
#[derive(Debug)]
pub struct GradCase {
    pub case: Case,
    /// The upstream gradient of each float output port, in port order.
    pub upstream: Vec<(String, TensorData)>,
    /// Expected updated running statistics (`nn.batch_norm` in training mode).
    pub running_stats: Vec<(String, TensorData)>,
    /// Expected gradients: name, index of the input or param (inputs first, then params, as the
    /// run function gets them) and value.
    pub gradients: Vec<(String, usize, TensorData)>,
    pub gradient_tolerance: Tolerance,
}

/// `fixtures/small/grads/<op>/` from the repository root.
pub fn grad_fixture_dir(op: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/small/grads")
        .join(op)
}

/// Every gradient case of `op`, sorted by name. Panics (fails the test) if a fixture is missing
/// or malformed.
pub fn load_grad_cases(op: &str) -> Vec<GradCase> {
    list_sidecars(&grad_fixture_dir(op))
        .iter()
        .map(|path| load_grad_case(op, path))
        .collect()
}

fn load_grad_case(op: &str, sidecar: &Path) -> GradCase {
    let (json, file) = read_sidecar(op, sidecar, 2);
    let case = case_from(op, sidecar, &json, &file);
    let optional = |key: &str| {
        if json.get(key).is_some() {
            load_list(&json, &file, sidecar, key)
        } else {
            Vec::new()
        }
    };
    let names: Vec<&str> = case
        .input_names
        .iter()
        .map(String::as_str)
        .chain(case.params.iter().map(|(name, _)| name.as_str()))
        .collect();
    let gradients = sidecar_field(&json, sidecar, "gradients")
        .as_array()
        .expect("gradients array")
        .iter()
        .zip(load_list(&json, &file, sidecar, "gradients"))
        .map(|(spec, (name, data))| {
            let of = spec["of"].as_str().expect("gradient `of`");
            let index = names.iter().position(|&n| n == of).unwrap_or_else(|| {
                panic!(
                    "{}: gradient {name} of unknown input {of}",
                    sidecar.display()
                )
            });
            (name, index, data)
        })
        .collect();
    let upstream = load_list(&json, &file, sidecar, "upstream");
    GradCase {
        upstream,
        running_stats: optional("running_stats"),
        gradients,
        gradient_tolerance: parse_tolerance(sidecar_field(&json, sidecar, "gradient_tolerance")),
        case,
    }
}

impl GradCase {
    /// The summary label: the dtype of the gradients (`f32` in M2) and the mode.
    fn label(&self) -> String {
        let dtype = self
            .gradients
            .first()
            .map_or("none", |(_, _, data)| data.dtype.name());
        format!("{dtype} {}", self.case.mode)
    }
}

/// Runs every gradient case of `op` on the CPU.
pub fn run_grad_cpu(op: &str, run: RunCase) {
    run_grad_on(op, &[DeviceId::Cpu], run);
}

/// Runs every gradient case of `op` on every GPU device of every compiled-in GPU backend.
pub fn run_grad_gpu(op: &str, run: RunCase) {
    let devices = gpu_devices();
    if devices.is_empty() {
        eprintln!("SKIP grad {op}: no GPU device to test (see the messages above)");
        return;
    }
    run_grad_on(op, &devices, run);
}

#[derive(Default)]
struct Tally {
    passed: usize,
    skipped: usize,
    failed: usize,
}

/// Runs every gradient case of `op` on `devices` and fails the test if any case fails.
pub fn run_grad_on(op: &str, devices: &[DeviceId], run: RunCase) {
    let cases = load_grad_cases(op);
    let mut failures = Vec::new();
    let mut tally: BTreeMap<(String, String), Tally> = BTreeMap::new();
    for &device in devices {
        for case in &cases {
            let entry = tally.entry((device.to_string(), case.label())).or_default();
            match run_grad_case(case, device, run) {
                Ok(true) => entry.passed += 1,
                Ok(false) => entry.skipped += 1,
                Err(message) => {
                    failures.push(format!("{op}/{} on {device}: {message}", case.case.name));
                    entry.failed += 1;
                }
            }
        }
    }
    let mut summary = String::new();
    for ((device, label), t) in &tally {
        let _ = writeln!(
            summary,
            "SUMMARY grad {op} {device} {label}: {} passed, {} skipped, {} failed",
            t.passed, t.skipped, t.failed
        );
    }
    eprint!("{summary}");
    assert!(
        failures.is_empty(),
        "{} gradient case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// `Ok(true)` passed, `Ok(false)` skipped (a dtype the device lacks), `Err` failed.
fn run_grad_case(case: &GradCase, device: DeviceId, run: RunCase) -> Result<bool, String> {
    let c = &case.case;
    let tensors = c
        .inputs
        .iter()
        .chain(c.params.iter().map(|(_, d)| d))
        .chain(c.mask.iter())
        .chain(case.upstream.iter().map(|(_, d)| d))
        .chain(c.outputs.iter().map(|(_, d)| d));
    for data in tensors {
        match DynTensor::device_supports(device, data.dtype) {
            Ok(true) => {}
            Ok(false) => {
                eprintln!(
                    "SKIP grad {}/{} on {device}: dtype {} is not supported",
                    c.op, c.name, data.dtype
                );
                return Ok(false);
            }
            Err(err) => return Err(format!("device_supports: {err}")),
        }
    }
    let leaves: Vec<usize> = case.gradients.iter().map(|(_, i, _)| *i).collect();
    // Float inputs and params without an expected gradient (`batch_norm`'s running statistics)
    // are marked too, and must get none (graph-ir.md §3.1.11, §3.3).
    let no_gradient: Vec<usize> = c
        .inputs
        .iter()
        .chain(c.params.iter().map(|(_, d)| d))
        .enumerate()
        .filter(|(i, data)| data.dtype.is_float() && !leaves.contains(i))
        .map(|(i, _)| i)
        .collect();
    let leaves: Vec<usize> = leaves.into_iter().chain(no_gradient).collect();
    run_and_check(case, device, run, &leaves, None)?;
    if c.name.starts_with("same_value") {
        let pair = same_value_pair(case)?;
        run_and_check(case, device, run, &leaves, Some(pair))
            .map_err(|message| format!("one value fed to both inputs: {message}"))?;
    }
    Ok(true)
}

/// The two inputs of a `same_value_*` case: the first two inputs with gradients, which must hold
/// the same values.
fn same_value_pair(case: &GradCase) -> Result<(usize, usize), String> {
    let [(_, i, _), (_, j, _), ..] = case.gradients.as_slice() else {
        return Err("a same_value case needs two inputs with gradients".into());
    };
    let (a, b) = (&case.case.inputs[*i], &case.case.inputs[*j]);
    if a != b {
        return Err(format!(
            "inputs {} and {} don't hold the same values",
            case.case.input_names[*i], case.case.input_names[*j]
        ));
    }
    Ok((*i, *j))
}

/// Uploads, marks, runs, back-propagates and compares one case. With `shared = Some((i, j))`,
/// input `j` is the same leaf as input `i`, and the expected gradient of that leaf is the sum of
/// both expected gradients.
fn run_and_check(
    case: &GradCase,
    device: DeviceId,
    run: RunCase,
    leaves: &[usize],
    shared: Option<(usize, usize)>,
) -> Result<(), String> {
    let c = &case.case;
    let upload = |data: &TensorData| {
        DynTensor::from_data(data, device).map_err(|err| format!("upload: {err}"))
    };
    let mut tensors = Vec::new();
    for data in c.inputs.iter().chain(c.params.iter().map(|(_, d)| d)) {
        tensors.push(upload(data)?);
    }
    // The view first, so that the view itself is the leaf.
    let mut tensors = feed_layout(c, tensors).map_err(|err| format!("layout: {err}"))?;
    for &i in leaves {
        tensors[i] = tensors[i]
            .clone()
            .require_grad()
            .map_err(|err| format!("require_grad: {err}"))?;
    }
    if let Some((i, j)) = shared {
        tensors[j] = tensors[i].clone();
    }
    let marked: Vec<DynTensor> = tensors.clone();
    if let Some(mask) = &c.mask {
        tensors.push(upload(mask)?);
    }

    // The run function must not build the view again.
    let fed = Case {
        layout_pending: false,
        ..c.clone()
    };
    let outputs = run(&fed, tensors).map_err(|err| format!("op failed: {err}"))?;
    let expected: Vec<&(String, TensorData)> =
        c.outputs.iter().chain(&case.running_stats).collect();
    if outputs.len() != expected.len() {
        return Err(format!(
            "{} outputs, expected {} (outputs, then running statistics)",
            outputs.len(),
            expected.len()
        ));
    }

    let mut problems = String::new();
    for (actual, (name, _)) in outputs[c.outputs.len()..].iter().zip(&case.running_stats) {
        if actual.has_autodiff() {
            let _ = write!(
                problems,
                "\n  {name}: carries autodiff context (runtime.md §3.3)"
            );
        }
    }
    for (actual, (name, want)) in outputs.iter().zip(&expected) {
        if actual.device() != device {
            let _ = write!(problems, "\n  {name}: on {}, not {device}", actual.device());
            continue;
        }
        match actual.to_data() {
            Ok(got) => {
                if let Err(message) = compare(want, &got, c.tolerance) {
                    let _ = write!(problems, "\n  {name}: {message}");
                }
            }
            Err(err) => {
                let _ = write!(problems, "\n  {name}: read back: {err}");
            }
        }
    }

    let loss = loss(&outputs[..c.outputs.len()], &case.upstream, device)
        .map_err(|err| format!("loss: {err}"))?;
    let grads = loss.backward().map_err(|err| format!("backward: {err}"))?;
    for (name, index, want) in &case.gradients {
        let (index, want) = match shared {
            Some((_, j)) if *index == j => continue,
            Some((i, j)) if *index == i => {
                let other = case
                    .gradients
                    .iter()
                    .find(|(_, idx, _)| *idx == j)
                    .map(|(_, _, d)| d)
                    .ok_or("no gradient for the second input")?;
                (i, sum_f32(want, other))
            }
            _ => (*index, want.clone()),
        };
        match grads.get(&marked[index]) {
            Ok(Some(grad)) => match grad.to_data() {
                Ok(got) => {
                    if grad.device() != device {
                        let _ = write!(problems, "\n  {name}: on {}, not {device}", grad.device());
                    } else if let Err(message) = compare(&want, &got, case.gradient_tolerance) {
                        let _ = write!(problems, "\n  {name}: {message}");
                    }
                }
                Err(err) => {
                    let _ = write!(problems, "\n  {name}: read back: {err}");
                }
            },
            Ok(None) => {
                let _ = write!(problems, "\n  {name}: no gradient (the leaf took no part)");
            }
            Err(err) => {
                let _ = write!(problems, "\n  {name}: {err}");
            }
        }
    }
    let names: Vec<&String> = c
        .input_names
        .iter()
        .chain(c.params.iter().map(|(name, _)| name))
        .collect();
    for &index in leaves {
        if case.gradients.iter().any(|(_, i, _)| *i == index) {
            continue;
        }
        match grads.get(&marked[index]) {
            Ok(None) => {}
            Ok(Some(_)) => {
                let _ = write!(
                    problems,
                    "\n  {}: got a gradient, expected none",
                    names[index]
                );
            }
            Err(err) => {
                let _ = write!(problems, "\n  {}: {err}", names[index]);
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}

/// `Σ_p sum(out_p · grad_p)` over the float outputs, as a `[1]` tensor. Its gradient with
/// respect to `out_p` is `grad_p`.
fn loss(
    outputs: &[DynTensor],
    upstream: &[(String, TensorData)],
    device: DeviceId,
) -> Result<DynTensor, TensorError> {
    let floats: Vec<&DynTensor> = outputs.iter().filter(|t| t.dtype().is_float()).collect();
    assert_eq!(
        floats.len(),
        upstream.len(),
        "one upstream gradient per float output port"
    );
    let mut total: Option<DynTensor> = None;
    for (out, (_, grad)) in floats.into_iter().zip(upstream) {
        let grad = DynTensor::from_data(grad, device)?;
        let term = out.clone().mul(grad)?.sum(None, true)?.reshape(&[1])?;
        total = Some(match total {
            Some(acc) => acc.add(term)?,
            None => term,
        });
    }
    total.ok_or_else(|| TensorError::InvalidArgument {
        op: "loss",
        reason: "no float output".into(),
    })
}

/// `a + b` for f32 tensors of the same shape, rounded once.
fn sum_f32(a: &TensorData, b: &TensorData) -> TensorData {
    assert_eq!(a.dtype, DType::F32, "gradients are f32 in M2");
    assert_eq!(a.shape, b.shape, "same_value inputs have the same shape");
    let (x, _) = a.bytes.as_chunks::<4>();
    let (y, _) = b.bytes.as_chunks::<4>();
    let bytes = x
        .iter()
        .zip(y)
        .flat_map(|(&x, &y)| (f32::from_le_bytes(x) + f32::from_le_bytes(y)).to_le_bytes())
        .collect();
    TensorData::new(DType::F32, a.shape.clone(), bytes).expect("same shape and dtype")
}
