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

fn upload(data: &TensorData, device: DeviceId) -> DynTensor {
    DynTensor::from_data(data, device).unwrap()
}

fn bools(data: &TensorData) -> Vec<bool> {
    data.bytes.iter().map(|&b| b != 0).collect()
}

/// Runs every case of every model on `device`; returns the number of cases.
fn run_all(device: DeviceId) -> usize {
    let mut n = 0;
    for dir in models() {
        let model = dir.file_name().unwrap().to_string_lossy().into_owned();
        let graph = Graph::from_json(&std::fs::read(dir.join("model.graph.json")).unwrap())
            .unwrap_or_else(|r| panic!("{model}: {r}"));
        let plan = Plan::new(&graph, PlanOptions::new(device)).unwrap();
        let weights = SafetensorsFile::open(dir.join("model.safetensors")).unwrap();
        for sidecar in cases(&dir) {
            run_case(&model, &plan, &weights, &sidecar, device);
            n += 1;
        }
    }
    n
}

fn run_case(model: &str, plan: &Plan, weights: &SafetensorsFile, sidecar: &Path, device: DeviceId) {
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
        let t = upload(&data, device);
        let t = if info.is_trainable() {
            t.require_grad().unwrap()
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
        assert_eq!(mask, bools(&stored), "{case}: mask of {path}");
    }

    let inputs: Vec<(String, DynTensor)> = common::load_list(&json, &file, sidecar, "inputs")
        .into_iter()
        .map(|(n, d)| (n, upload(&d, device)))
        .collect();
    let out = plan
        .run(
            inputs,
            vec![],
            &params,
            RunMode::Training { seed, step },
            &RunOptions::default(),
        )
        .unwrap_or_else(|e| panic!("{case} on {device}: {e}"));
    let tol = common::parse_tolerance(&json["tolerance"]);
    let gtol = common::parse_tolerance(&json["gradient_tolerance"]);

    // Outputs.
    let expected = common::load_list(&json, &file, sidecar, "outputs");
    for (name, want) in &expected {
        let got = out.output(name).unwrap().to_data().unwrap();
        common::compare(want, &got, tol)
            .unwrap_or_else(|e| panic!("{case} on {device}: output {name}: {e}"));
    }
    // Running statistics: names and order (runtime.md §3.3), values, no autodiff context.
    let want_stats: Vec<String> = entries(&json, "running_stats")
        .iter()
        .map(|r| r["param"].as_str().unwrap().to_owned())
        .collect();
    let got_stats: Vec<String> = out.running_stats.iter().map(|(n, _)| n.clone()).collect();
    assert_eq!(got_stats, want_stats, "{case}: running statistics");
    for r in entries(&json, "running_stats") {
        let param = r["param"].as_str().unwrap();
        let (_, want) = common::load_tensor(&file, sidecar, &strip(&r));
        let t = out.running_stat(param).unwrap();
        assert!(
            !t.has_autodiff(),
            "{case}: {param} carries autodiff context"
        );
        common::compare(&want, &t.to_data().unwrap(), tol)
            .unwrap_or_else(|e| panic!("{case} on {device}: running statistic {param}: {e}"));
    }

    // Gradients of Σ out · grad_out.
    let mut loss: Option<DynTensor> = None;
    for u in entries(&json, "upstream") {
        let output = u["output"].as_str().unwrap();
        let (_, g) = common::load_tensor(&file, sidecar, &strip(&u));
        let term = out
            .output(output)
            .unwrap()
            .clone()
            .mul(upload(&g, device))
            .unwrap()
            .sum(None, false)
            .unwrap();
        loss = Some(match loss {
            None => term,
            Some(l) => l.add(term).unwrap(),
        });
    }
    let grads = loss.expect("an upstream gradient").backward().unwrap();
    let mut with_grad = Vec::new();
    for g in entries(&json, "gradients") {
        let param = g["param"].as_str().unwrap();
        let (_, want) = common::load_tensor(&file, sidecar, &strip(&g));
        let got = grads
            .get(&params[param])
            .unwrap()
            .unwrap_or_else(|| panic!("{case} on {device}: {param} got no gradient"));
        common::compare(&want, &got.to_data().unwrap(), gtol)
            .unwrap_or_else(|e| panic!("{case} on {device}: gradient of {param}: {e}"));
        with_grad.push(param.to_owned());
    }
    for info in &analysis.params {
        if with_grad.contains(&info.name) {
            continue;
        }
        let t = &params[&info.name];
        if info.is_trainable() {
            assert!(
                grads.get(t).unwrap().is_none(),
                "{case}: {} got a gradient",
                info.name
            );
        } else {
            // Given untracked: never a leaf, so no gradient (and the call didn't track it).
            assert!(!t.is_tracked(), "{case}: {} is tracked", info.name);
        }
    }
    eprintln!(
        "PASS {case} on {device}: {} outputs, {} running statistics, {} gradients",
        expected.len(),
        got_stats.len(),
        with_grad.len()
    );
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
    let n = run_all(DeviceId::Cpu);
    eprintln!("SUMMARY training_reference cpu: {n} passed, 0 failed");
    assert_eq!(n, 3);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn training_graphs_match_pytorch_gpu() {
    for device in common::gpu_devices() {
        let n = run_all(device);
        eprintln!("SUMMARY training_reference {device}: {n} passed, 0 failed");
    }
}
