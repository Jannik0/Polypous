//! Training-mode graphs against PyTorch (`fixtures/small/training/<model>/`, testing-strategy.md
//! §3.10; runtime.md §3.1–§3.3): `linear → relu → linear`, and a graph with `conv2d`,
//! `batch_norm`, `dropout` and a `repeat` that drops in each iteration.
//!
//! Per case: the weights of `model.safetensors` become the given params (runtime.md §3.2), the
//! trainable params (training.md §1.1) marked as autodiff leaves; the plan runs in
//! `training { seed, step }` from the sidecar; the outputs and the returned running statistics
//! (names and order too) are compared with `tolerance`; the loss `Σ out · grad_out` is
//! back-propagated and every trainable param's gradient is compared with `gradient_tolerance`,
//! params without an expected gradient must get none. The dropout masks the runtime draws are
//! checked twice: through the outputs and gradients (the reference uses the masks of the Python
//! port of graph-ir.md §3.1.12), and directly, `polypous_ir::rng::dropout_mask` against the stored
//! masks.
//!
//! ```text
//! cargo test -p polypous-runtime --test training_reference -- --nocapture
//! cargo test -p polypous-runtime --features vulkan --test training_reference -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "../../polypous-backend/tests/reference/common.rs"]
mod common;

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use polypous_backend::device::DeviceId;
use polypous_backend::{DynTensor, TensorData};
use polypous_formats::safetensors::SafetensorsFile;
use polypous_ir::{Graph, rng};
use polypous_runtime::{Params, Plan, PlanOptions, RunMode, RunOptions};
use serde_json::Value;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/small/training")
}

fn models() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root())
        .expect("fixtures/small/training")
        .map(|e| e.unwrap().path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert!(dirs.len() >= 2, "expected the mlp and bn_dropout models");
    dirs
}

fn cases(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "json")
                && p.file_name().is_some_and(|n| n != "model.graph.json")
        })
        .collect();
    out.sort();
    assert!(!out.is_empty(), "{}: no cases", dir.display());
    out
}

fn upload(data: &TensorData, device: DeviceId) -> Result<DynTensor, String> {
    DynTensor::from_data(data, device).map_err(|e| format!("upload on {device}: {e}"))
}

fn bools(data: &TensorData) -> Vec<bool> {
    data.bytes.iter().map(|&b| b != 0).collect()
}

/// Runs every case of every model on each of `devices`, prints a `SUMMARY training_reference`
/// line per device, and fails with every failing case (case, device, message) once all have run.
/// Returns the number of cases per device.
fn run_on(devices: &[DeviceId]) -> usize {
    let mut failures = Vec::new();
    let mut summary = String::new();
    let mut total = 0;
    for &device in devices {
        let (mut passed, mut failed) = (0, 0);
        total = 0;
        for dir in models() {
            let model = dir.file_name().unwrap().to_string_lossy().into_owned();
            let graph = Graph::from_json(&std::fs::read(dir.join("model.graph.json")).unwrap())
                .unwrap_or_else(|r| panic!("{model}: {r}"));
            let weights = SafetensorsFile::open(dir.join("model.safetensors")).unwrap();
            let sidecars = cases(&dir);
            total += sidecars.len();
            let plan = match Plan::new(&graph, PlanOptions::new(device)) {
                Ok(plan) => plan,
                Err(e) => {
                    // Every case of the model fails with the plan's error.
                    for sidecar in &sidecars {
                        failures.push(format!("{model}/{} on {device}: plan: {e}", stem(sidecar)));
                    }
                    failed += sidecars.len();
                    continue;
                }
            };
            for sidecar in sidecars {
                match run_case(&model, &plan, &weights, &sidecar, device) {
                    Ok(()) => passed += 1,
                    Err(message) => {
                        eprintln!("FAIL {message}");
                        failures.push(message);
                        failed += 1;
                    }
                }
            }
        }
        let _ = writeln!(
            summary,
            "SUMMARY training_reference {device}: {passed} passed, {failed} failed"
        );
    }
    eprint!("{summary}");
    assert!(
        failures.is_empty(),
        "{} training case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
    total
}

/// The file name of a sidecar without its extension.
fn stem(sidecar: &Path) -> String {
    sidecar.file_stem().unwrap().to_string_lossy().into_owned()
}

/// `Err(message)` if `got != want`, with the text of `assert_eq!`.
fn ensure_eq<T: PartialEq + std::fmt::Debug>(
    got: T,
    want: T,
    message: String,
) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!(
            "{message}: assertion `left == right` failed\n  left: {got:?}\n right: {want:?}"
        ))
    }
}

/// Runs one case on `device`. A mismatch or a failing call on the device is an `Err` with the
/// case, the device and what differs; a malformed fixture panics (it fails on every device).
fn run_case(
    model: &str,
    plan: &Plan,
    weights: &SafetensorsFile,
    sidecar: &Path,
    device: DeviceId,
) -> Result<(), String> {
    let json: Value = serde_json::from_slice(&std::fs::read(sidecar).unwrap()).unwrap();
    let file = SafetensorsFile::open(sidecar.with_extension("safetensors")).unwrap();
    let case = format!("{model}/{}", json["case"].as_str().unwrap());
    assert_eq!(json["schema_version"], 2, "{case}");
    assert_eq!(json["mode"], "training", "{case}");
    let seed = json["training"]["seed"].as_u64().unwrap();
    let step = json["training"]["step"].as_u64().unwrap();

    // Params: every declared param, the trainable ones as leaves.
    let analysis = plan.analysis();
    let mut params: Params = HashMap::new();
    for info in &analysis.params {
        let spec = serde_json::json!({ "name": info.name, "dtype": "f32", "shape": info.shape });
        let (_, data) = common::load_tensor(weights, &sidecar.with_file_name("model.json"), &spec);
        let t = upload(&data, device).map_err(|e| format!("{case}: {e}"))?;
        let t = if info.is_trainable() {
            t.require_grad()
                .map_err(|e| format!("{case} on {device}: require_grad of {}: {e}", info.name))?
        } else {
            t
        };
        params.insert(info.name.clone(), t);
    }

    // The masks the runtime draws (graph-ir.md §3.1.12) equal Python's.
    for m in entries(&json, "masks") {
        let path = m["node"].as_str().unwrap();
        let (_, stored) = common::load_tensor(&file, sidecar, &strip(&m));
        let shape: Vec<usize> = stored.shape.iter().map(|&d| d as usize).collect();
        let p = mask_p(plan, path);
        let mask = rng::dropout_mask(seed, step, path, &shape, p).unwrap();
        ensure_eq(mask, bools(&stored), format!("{case}: mask of {path}"))?;
    }

    let mut inputs: Vec<(String, DynTensor)> = Vec::new();
    for (n, d) in common::load_list(&json, &file, sidecar, "inputs") {
        inputs.push((n, upload(&d, device).map_err(|e| format!("{case}: {e}"))?));
    }
    let out = plan
        .run(
            inputs,
            vec![],
            &params,
            RunMode::Training { seed, step },
            &RunOptions::default(),
        )
        .map_err(|e| format!("{case} on {device}: {e}"))?;
    let tol = common::parse_tolerance(&json["tolerance"]);
    let gtol = common::parse_tolerance(&json["gradient_tolerance"]);

    // Outputs.
    let expected = common::load_list(&json, &file, sidecar, "outputs");
    for (name, want) in &expected {
        let got = out
            .output(name)
            .ok_or_else(|| format!("{case} on {device}: no output {name}"))?
            .to_data()
            .map_err(|e| format!("{case} on {device}: output {name}: {e}"))?;
        common::compare(want, &got, tol)
            .map_err(|e| format!("{case} on {device}: output {name}: {e}"))?;
    }
    // Running statistics: names and order (runtime.md §3.3), values, no autodiff context.
    let want_stats: Vec<String> = entries(&json, "running_stats")
        .iter()
        .map(|r| r["param"].as_str().unwrap().to_owned())
        .collect();
    let got_stats: Vec<String> = out.running_stats.iter().map(|(n, _)| n.clone()).collect();
    ensure_eq(
        &got_stats,
        &want_stats,
        format!("{case}: running statistics"),
    )?;
    for r in entries(&json, "running_stats") {
        let param = r["param"].as_str().unwrap();
        let (_, want) = common::load_tensor(&file, sidecar, &strip(&r));
        let t = out
            .running_stat(param)
            .ok_or_else(|| format!("{case} on {device}: no running statistic {param}"))?;
        if t.has_autodiff() {
            return Err(format!("{case}: {param} carries autodiff context"));
        }
        let got = t
            .to_data()
            .map_err(|e| format!("{case} on {device}: running statistic {param}: {e}"))?;
        common::compare(&want, &got, tol)
            .map_err(|e| format!("{case} on {device}: running statistic {param}: {e}"))?;
    }

    // Gradients of Σ out · grad_out.
    let mut loss: Option<DynTensor> = None;
    for u in entries(&json, "upstream") {
        let output = u["output"].as_str().unwrap();
        let (_, g) = common::load_tensor(&file, sidecar, &strip(&u));
        let loss_error = |e| format!("{case} on {device}: loss term of {output}: {e}");
        let term = out
            .output(output)
            .ok_or_else(|| format!("{case} on {device}: no output {output}"))?
            .clone()
            .mul(upload(&g, device).map_err(|e| format!("{case}: {e}"))?)
            .map_err(loss_error)?
            .sum(None, false)
            .map_err(loss_error)?;
        loss = Some(match loss {
            None => term,
            Some(l) => l.add(term).map_err(loss_error)?,
        });
    }
    let grads = loss
        .expect("an upstream gradient")
        .backward()
        .map_err(|e| format!("{case} on {device}: backward: {e}"))?;
    let mut with_grad = Vec::new();
    for g in entries(&json, "gradients") {
        let param = g["param"].as_str().unwrap();
        let (_, want) = common::load_tensor(&file, sidecar, &strip(&g));
        let got = grads
            .get(&params[param])
            .map_err(|e| format!("{case} on {device}: gradient of {param}: {e}"))?
            .ok_or_else(|| format!("{case} on {device}: {param} got no gradient"))?;
        let got = got
            .to_data()
            .map_err(|e| format!("{case} on {device}: gradient of {param}: {e}"))?;
        common::compare(&want, &got, gtol)
            .map_err(|e| format!("{case} on {device}: gradient of {param}: {e}"))?;
        with_grad.push(param.to_owned());
    }
    for info in &analysis.params {
        if with_grad.contains(&info.name) {
            continue;
        }
        let t = &params[&info.name];
        if info.is_trainable() {
            let got = grads
                .get(t)
                .map_err(|e| format!("{case} on {device}: gradient of {}: {e}", info.name))?;
            if got.is_some() {
                return Err(format!("{case}: {} got a gradient", info.name));
            }
        } else if t.is_tracked() {
            // Given untracked: never a leaf, so no gradient (and the call didn't track it).
            return Err(format!("{case}: {} is tracked", info.name));
        }
    }
    eprintln!(
        "PASS {case} on {device}: {} outputs, {} running statistics, {} gradients",
        expected.len(),
        got_stats.len(),
        with_grad.len()
    );
    Ok(())
}

/// The entries of the sidecar list `key`.
fn entries(json: &Value, key: &str) -> Vec<Value> {
    json[key]
        .as_array()
        .unwrap_or_else(|| panic!("sidecar key {key} is not a list"))
        .clone()
}

/// A sidecar entry as `{"name", "dtype", "shape"}` (dropping `param`, `node`, `output`).
fn strip(entry: &Value) -> Value {
    serde_json::json!({ "name": entry["name"], "dtype": entry["dtype"], "shape": entry["shape"] })
}

/// `p` of the dropout node at `path` (`blocks[1]/drop`): the attribute of the node with that id in
/// its scope, as the analysis normalized it.
fn mask_p(plan: &Plan, path: &str) -> f64 {
    let mut nodes = &plan.analysis().nodes;
    let parts: Vec<&str> = path.split('/').collect();
    for (k, part) in parts.iter().enumerate() {
        let id = part.split('[').next().unwrap();
        let node = nodes
            .iter()
            .find(|n| n.id() == id)
            .unwrap_or_else(|| panic!("{path}"));
        match node {
            polypous_ir::NodeInfo::Repeat(r) if k + 1 < parts.len() => nodes = &r.nodes,
            polypous_ir::NodeInfo::Op(op) => return op.attrs.float("p").unwrap(),
            _ => panic!("{path}"),
        }
    }
    panic!("{path}")
}

#[test]
fn training_graphs_match_pytorch_cpu() {
    assert_eq!(run_on(&[DeviceId::Cpu]), 3);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn training_graphs_match_pytorch_gpu() {
    run_on(&common::gpu_devices());
}
