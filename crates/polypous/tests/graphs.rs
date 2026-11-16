//! Graphs and models through the public facade (public-api.md §1.3, issue #45), on the CPU.
//!
//! The engine gets its device list from `EngineConfig::probe_report` (an available CPU), so no
//! probe runs. Weights and tensors are written by the tests with `polypous::tensor_file`. Expected
//! values are computed in plain Rust (f32, the same formula as PyTorch's `F.linear`, `F.conv2d`,
//! `F.relu`, `F.max_pool2d`); results are compared with an absolute tolerance of 1e-5, which covers
//! differences in f32 summation order for values of magnitude ≤ 10.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

use std::path::PathBuf;

use polypous::{
    CancellationToken, ComputeDtype, DeviceId, Dim, Dtype, Engine, EngineConfig, ErrorKind, Graph,
    GraphLoadOptions, GraphValidateOptions, Model, NamedTensor, TensorData, tensor_file,
};
use serde_json::{Value, json};

// ---- Helpers ----

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("facade-graphs-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn device_json(id: Value, backend: &str, kind: &str, status: Value) -> Value {
    json!({
        "id": id, "backend": backend, "name": format!("test {backend}"), "vendor": null,
        "driver": null, "kind": kind, "dedicated_memory": null, "free_memory": null,
        "unified_memory": true,
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": status, "physical_gpu": null, "primary": false
    })
}

/// An engine whose only devices are an available CPU and an unavailable Vulkan software
/// rasterizer (as the probe lists a skipped lavapipe). No probe runs.
fn engine() -> Engine {
    let cpu = device_json(
        json!({"backend": "cpu"}),
        "cpu",
        "cpu",
        json!({"state": "available", "isolated": true}),
    );
    let lavapipe = device_json(
        json!({"backend": "vulkan", "adapter": "cpu", "index": 0}),
        "vulkan",
        "cpu",
        json!({"state": "unavailable", "reason": {"code": "software_rasterizer_skipped"}, "isolated": true}),
    );
    let config = json!({ "probe_report": { "devices": [lavapipe, cpu], "isolated": true } });
    Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap()
}

fn f32_tensor(shape: &[u64], values: &[f32]) -> TensorData {
    TensorData::new(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn f32_values(tensor: &TensorData) -> Vec<f32> {
    assert_eq!(tensor.dtype, Dtype::F32);
    tensor
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}

/// Deterministic values in [-1, 1).
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

#[track_caller]
fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= 1e-5, "element {i}: {a} vs {e}");
    }
}

fn options(weights: Option<&PathBuf>) -> GraphLoadOptions {
    let mut options = GraphLoadOptions::default();
    options.weights = weights.cloned();
    options
}

fn load(graph: &Graph, weights: Option<&PathBuf>) -> Result<Model, polypous::Error> {
    engine().load_graph(graph, options(weights), &CancellationToken::new())
}

fn run(model: &Model, inputs: Vec<NamedTensor>) -> Result<polypous::RunOutput, polypous::Error> {
    model.run(inputs, vec![], &CancellationToken::new())
}

// ---- MLP ----

/// `x [batch, 4]` → linear 4→3 → relu → linear 3→2.
fn mlp_doc() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "name": "mlp",
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "fc2" } ],
        "params": [
            { "name": "fc1.weight", "dtype": "float", "shape": [3, 4] },
            { "name": "fc1.bias", "dtype": "float", "shape": [3] },
            { "name": "fc2.weight", "dtype": "float", "shape": [2, 3] },
            { "name": "fc2.bias", "dtype": "float", "shape": [2] }
        ],
        "nodes": [
            { "id": "fc1", "op": "nn.linear", "inputs": { "x": "x" },
              "params": { "weight": "fc1.weight", "bias": "fc1.bias" } },
            { "id": "act", "op": "nn.relu", "inputs": { "x": "fc1" } },
            { "id": "fc2", "op": "nn.linear", "inputs": { "x": "act" },
              "params": { "weight": "fc2.weight", "bias": "fc2.bias" } }
        ]
    })
}

fn mlp_graph() -> Graph {
    Graph::from_json(&mlp_doc().to_string()).unwrap()
}

struct MlpWeights {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn mlp_weights() -> MlpWeights {
    MlpWeights {
        w1: values(12, 1),
        b1: values(3, 2),
        w2: values(6, 3),
        b2: values(2, 4),
    }
}

fn mlp_tensors(w: &MlpWeights) -> Vec<NamedTensor> {
    vec![
        NamedTensor::new("fc1.weight", f32_tensor(&[3, 4], &w.w1)),
        NamedTensor::new("fc1.bias", f32_tensor(&[3], &w.b1)),
        NamedTensor::new("fc2.weight", f32_tensor(&[2, 3], &w.w2)),
        NamedTensor::new("fc2.bias", f32_tensor(&[2], &w.b2)),
    ]
}

fn write_mlp_weights(dir: &std::path::Path) -> (PathBuf, MlpWeights) {
    let w = mlp_weights();
    let path = dir.join("mlp.safetensors");
    tensor_file::write(&path, &mlp_tensors(&w)).unwrap();
    (path, w)
}

/// `F.linear`: `y[r, o] = sum_i x[r, i] * w[o, i] + b[o]`.
fn linear(x: &[f32], rows: usize, w: &[f32], b: &[f32], inputs: usize) -> Vec<f32> {
    let outputs = b.len();
    let mut y = Vec::with_capacity(rows * outputs);
    for r in 0..rows {
        for o in 0..outputs {
            let dot: f32 = (0..inputs)
                .map(|i| x[r * inputs + i] * w[o * inputs + i])
                .sum();
            y.push(dot + b[o]);
        }
    }
    y
}

fn relu(x: Vec<f32>) -> Vec<f32> {
    x.into_iter().map(|v| v.max(0.0)).collect()
}

fn mlp_reference(x: &[f32], batch: usize, w: &MlpWeights) -> Vec<f32> {
    let h = relu(linear(x, batch, &w.w1, &w.b1, 4));
    linear(&h, batch, &w.w2, &w.b2, 3)
}

// ---- Graph JSON and validation ----

#[test]
fn graph_json_round_trips() {
    let graph = mlp_graph();
    let json = graph.to_json();
    let again = Graph::from_json(&json).unwrap();
    assert_eq!(again, graph);
    assert_eq!(again.to_json(), json);
    let report = graph.validate(&GraphValidateOptions::default());
    assert!(report.valid);
    assert_eq!(report.to_json(), r#"{"valid":true,"diagnostics":[]}"#);
    let f16 = GraphValidateOptions::from_json(r#"{"compute_dtype":"f16"}"#).unwrap();
    assert!(graph.validate(&f16).valid);
    assert!(format!("{graph:?}").contains("nodes: 3"));
}

/// Two errors: an unknown value reference in `fc1`, an unknown op in `act`.
fn invalid_doc() -> Value {
    let mut doc = mlp_doc();
    doc["nodes"][0]["inputs"]["x"] = json!("imgae");
    doc["nodes"][1]["op"] = json!("nn.relux");
    doc
}

#[test]
fn validation_reports_every_diagnostic_with_node_and_pointer() {
    let doc = invalid_doc().to_string();
    let report = Graph::validate_json(&doc, &GraphValidateOptions::default());
    assert!(!report.valid);
    let found: Vec<(&str, Option<&str>, &str)> = report
        .diagnostics
        .iter()
        .map(|d| (d.code.as_str(), d.node.as_deref(), d.pointer.as_str()))
        .collect();
    assert!(
        found.contains(&("unknown_reference", Some("fc1"), "/nodes/0/inputs/x")),
        "{found:?}"
    );
    assert!(
        found.contains(&("unknown_op", Some("act"), "/nodes/1/op")),
        "{found:?}"
    );
    // The JSON form.
    let value: Value = serde_json::from_str(&report.to_json()).unwrap();
    assert_eq!(value["valid"], false);
    let first = &value["diagnostics"][0];
    let keys: Vec<&str> = first
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(keys, ["code", "message", "node", "pointer"]);

    // `from_json`: InvalidGraph with the first diagnostic in the context, every one in details.
    let error = Graph::from_json(&doc).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidGraph);
    let first = &report.diagnostics[0];
    assert_eq!(error.context.node, first.node);
    assert_eq!(
        error.context.pointer.as_deref(),
        Some(first.pointer.as_str())
    );
    assert_eq!(error.context.operation.as_deref(), Some("graph_from_json"));
    assert!(error.message.starts_with("The graph is not valid: "));
    assert!(error.message.contains(&first.message));
    let more = report.diagnostics.len() - 1;
    assert!(error.message.contains(&format!("and {more} more error")));
    assert_eq!(
        error.context.details.as_deref().unwrap().lines().count(),
        report.diagnostics.len()
    );
}

#[test]
fn malformed_graph_json_is_an_invalid_graph() {
    for doc in ["{", "", "[]", r#"{"format":"polypous.graph"}"#] {
        let error = Graph::from_json(doc).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidGraph, "{doc}");
        assert!(error.context.pointer.is_some());
    }
    let report = Graph::validate_json("{", &GraphValidateOptions::default());
    assert_eq!(report.diagnostics.len(), 1);
    assert_eq!(report.diagnostics[0].code, "invalid_json");
    assert_eq!(report.diagnostics[0].node, None);
    assert_eq!(report.diagnostics[0].pointer, "");
    // Bad options JSON.
    let error = GraphValidateOptions::from_json(r#"{"compute":"f16"}"#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(GraphValidateOptions::from_json(r#"{"compute_dtype":"f64"}"#).is_err());
}

#[test]
fn graph_files() {
    let dir = dir("files");
    let path = dir.join("mlp.graph.json");
    std::fs::write(&path, mlp_doc().to_string()).unwrap();
    assert_eq!(Graph::from_file(&path).unwrap(), mlp_graph());
    // A UTF-8 string path works too.
    assert!(Graph::from_file(path.to_str().unwrap()).is_ok());
    let report = Graph::validate_file(&path, &GraphValidateOptions::default()).unwrap();
    assert!(report.valid);

    let missing = dir.join("missing.graph.json");
    for error in [
        Graph::from_file(&missing).unwrap_err(),
        Graph::validate_file(&missing, &GraphValidateOptions::default()).unwrap_err(),
    ] {
        assert_eq!(error.kind, ErrorKind::Io);
        assert_eq!(error.context.file, Some(missing.display().to_string()));
    }

    let bad = dir.join("bad.graph.json");
    std::fs::write(&bad, invalid_doc().to_string()).unwrap();
    let error = Graph::from_file(&bad).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidGraph);
    assert_eq!(error.context.file, Some(bad.display().to_string()));

    let binary = dir.join("binary.graph.json");
    std::fs::write(&binary, [0xff, 0xfe, b'{']).unwrap();
    let report = Graph::validate_file(&binary, &GraphValidateOptions::default()).unwrap();
    assert_eq!(report.diagnostics[0].code, "invalid_json");
}

// ---- Loading and running ----

#[test]
fn mlp_runs_with_symbolic_batch_sizes() {
    let dir = dir("mlp");
    let (weights, w) = write_mlp_weights(&dir);
    let model = load(&mlp_graph(), Some(&weights)).unwrap();

    let info = model.info();
    assert_eq!(info.device, DeviceId::Cpu);
    assert_eq!(info.compute_dtype, ComputeDtype::F32);
    assert_eq!(
        serde_json::to_value(&info).unwrap(),
        json!({
            "device": {"backend": "cpu"},
            "compute_dtype": "f32",
            "inputs": [ {"name": "x", "dtype": "f32", "shape": ["batch", 4]} ],
            "outputs": [ {"name": "y", "dtype": "f32", "shape": ["batch", 2]} ],
            "state": [],
            "params": [
                {"name": "fc1.weight", "dtype": "f32", "shape": [3, 4]},
                {"name": "fc1.bias", "dtype": "f32", "shape": [3]},
                {"name": "fc2.weight", "dtype": "f32", "shape": [2, 3]},
                {"name": "fc2.bias", "dtype": "f32", "shape": [2]}
            ]
        })
    );
    assert_eq!(info.inputs[0].shape[0], Dim::Symbolic("batch".into()));
    assert_eq!(model.graph(), mlp_graph());

    for batch in [1usize, 3, 5] {
        let x = values(batch * 4, 10 + batch as u32);
        let out = run(
            &model,
            vec![NamedTensor::new("x", f32_tensor(&[batch as u64, 4], &x))],
        )
        .unwrap();
        assert_eq!(out.outputs.len(), 1);
        assert_eq!(out.outputs[0].name, "y");
        let y = out.output("y").unwrap();
        assert_eq!(y.shape, vec![batch as u64, 2]);
        assert_close(&f32_values(y), &mlp_reference(&x, batch, &w));
        assert!(out.state.is_empty());
    }

    // Clones share the model, and calls may come from several threads.
    let threads: Vec<_> = (0..3)
        .map(|i| {
            let model = model.clone();
            std::thread::spawn(move || {
                let x = values(8, 100 + i);
                run(&model, vec![NamedTensor::new("x", f32_tensor(&[2, 4], &x))])
                    .unwrap()
                    .outputs
            })
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap()[0].tensor.shape, vec![2, 2]);
    }
}

/// `image [batch, 1, 4, 4]` → conv2d 1→2 (3×3) → relu → max_pool2d 2×2 → flatten → linear 2→3.
fn cnn_doc() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "image", "dtype": "float", "shape": ["batch", 1, 4, 4] } ],
        "outputs": [ { "name": "logits", "value": "fc" } ],
        "params": [
            { "name": "conv.weight", "dtype": "float", "shape": [2, 1, 3, 3] },
            { "name": "conv.bias", "dtype": "float", "shape": [2] },
            { "name": "fc.weight", "dtype": "float", "shape": [3, 2] },
            { "name": "fc.bias", "dtype": "float", "shape": [3] }
        ],
        "nodes": [
            { "id": "conv", "op": "nn.conv2d", "inputs": { "x": "image" },
              "params": { "weight": "conv.weight", "bias": "conv.bias" } },
            { "id": "act", "op": "nn.relu", "inputs": { "x": "conv" } },
            { "id": "pool", "op": "nn.max_pool2d", "inputs": { "x": "act" },
              "attrs": { "kernel_size": [2, 2] } },
            { "id": "flat", "op": "core.flatten", "inputs": { "x": "pool" }, "attrs": { "start_dim": 1 } },
            { "id": "fc", "op": "nn.linear", "inputs": { "x": "flat" },
              "params": { "weight": "fc.weight", "bias": "fc.bias" } }
        ]
    })
}

/// The CNN in plain Rust: valid 3×3 conv (4×4 → 2×2), relu, 2×2 max pool (→ 1×1), linear.
fn cnn_reference(
    x: &[f32],
    batch: usize,
    cw: &[f32],
    cb: &[f32],
    fw: &[f32],
    fb: &[f32],
) -> Vec<f32> {
    let mut pooled = Vec::with_capacity(batch * 2);
    for n in 0..batch {
        for c in 0..2 {
            let mut max = f32::NEG_INFINITY;
            for oy in 0..2 {
                for ox in 0..2 {
                    let mut sum = cb[c];
                    for ky in 0..3 {
                        for kx in 0..3 {
                            sum += x[n * 16 + (oy + ky) * 4 + ox + kx] * cw[c * 9 + ky * 3 + kx];
                        }
                    }
                    max = max.max(sum.max(0.0));
                }
            }
            pooled.push(max);
        }
    }
    linear(&pooled, batch, fw, fb, 2)
}

#[test]
fn cnn_runs() {
    let dir = dir("cnn");
    let (cw, cb, fw, fb) = (values(18, 5), values(2, 6), values(6, 7), values(3, 8));
    let weights = dir.join("cnn.safetensors");
    tensor_file::write(
        &weights,
        &[
            NamedTensor::new("conv.weight", f32_tensor(&[2, 1, 3, 3], &cw)),
            NamedTensor::new("conv.bias", f32_tensor(&[2], &cb)),
            NamedTensor::new("fc.weight", f32_tensor(&[3, 2], &fw)),
            NamedTensor::new("fc.bias", f32_tensor(&[3], &fb)),
        ],
    )
    .unwrap();
    let graph = Graph::from_json(&cnn_doc().to_string()).unwrap();
    let model = load(&graph, Some(&weights)).unwrap();
    for batch in [1usize, 4] {
        let x = values(batch * 16, 20 + batch as u32);
        let out = run(
            &model,
            vec![NamedTensor::new(
                "image",
                f32_tensor(&[batch as u64, 1, 4, 4], &x),
            )],
        )
        .unwrap();
        let logits = out.output("logits").unwrap();
        assert_eq!(logits.shape, vec![batch as u64, 3]);
        assert_close(
            &f32_values(logits),
            &cnn_reference(&x, batch, &cw, &cb, &fw, &fb),
        );
    }
}

#[test]
fn state_is_passed_in_and_returned() {
    // y = acc + x; acc ← y.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2] } ],
        "state": [ { "name": "acc", "dtype": "float", "shape": [2], "init": "zeros" } ],
        "outputs": [ { "name": "y", "value": "sum" } ],
        "nodes": [ { "id": "sum", "op": "core.add", "inputs": { "a": "acc", "b": "x" } } ],
        "state_updates": { "acc": "sum" }
    });
    let model = load(&Graph::from_json(&doc.to_string()).unwrap(), None).unwrap();
    assert_eq!(model.info().state[0].name, "acc");
    let mut acc = f32_tensor(&[2], &[0.0, 0.0]);
    for step in 1..=3 {
        let out = model
            .run(
                vec![NamedTensor::new("x", f32_tensor(&[2], &[1.0, 0.5]))],
                vec![NamedTensor::new("acc", acc)],
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(out.state[0].name, "acc");
        acc = out.state[0].tensor.clone();
        assert_eq!(f32_values(&acc), [step as f32, 0.5 * step as f32]);
    }
    // A missing state value.
    let error = run(
        &model,
        vec![NamedTensor::new("x", f32_tensor(&[2], &[1.0, 0.5]))],
    )
    .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("acc"));
}

#[test]
fn cancellation() {
    let dir = dir("cancel");
    let (weights, _) = write_mlp_weights(&dir);
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = engine()
        .load_graph(&mlp_graph(), options(Some(&weights)), &cancel)
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::Cancelled);
    assert_eq!(
        error.context.operation.as_deref(),
        Some("engine_load_graph")
    );

    let model = load(&mlp_graph(), Some(&weights)).unwrap();
    let x = NamedTensor::new("x", f32_tensor(&[1, 4], &[0.0; 4]));
    let error = model.run(vec![x.clone()], vec![], &cancel).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Cancelled);
    assert_eq!(error.message, "The operation was cancelled.");
    // A fresh token runs.
    assert!(run(&model, vec![x]).is_ok());
}

// ---- Saving ----

#[test]
fn save_and_reload_give_identical_outputs() {
    let dir = dir("save");
    let (weights, _) = write_mlp_weights(&dir);
    let x = NamedTensor::new("x", f32_tensor(&[3, 4], &values(12, 30)));
    for compute in [ComputeDtype::F32, ComputeDtype::F16, ComputeDtype::Bf16] {
        let mut opts = options(Some(&weights));
        opts.compute_dtype = compute;
        let model = engine()
            .load_graph(&mlp_graph(), opts.clone(), &CancellationToken::new())
            .unwrap();
        let mut x = x.clone();
        if compute != ComputeDtype::F32 {
            // `float` inputs have the compute dtype; zeros are exact in every dtype (the outputs
            // still depend on every param through the biases and relu).
            x.tensor = TensorData::new(compute.dtype(), vec![3, 4], vec![0; 24]).unwrap();
        }
        let before = run(&model, vec![x.clone()]).unwrap();

        let graph_path = dir.join(format!("model-{compute}.graph.json"));
        let weights_path = dir.join(format!("model-{compute}.safetensors"));
        model.save(&graph_path, &weights_path).unwrap();
        // The params are saved in the compute dtype.
        let saved = tensor_file::read(&weights_path).unwrap();
        assert_eq!(saved.len(), 4);
        assert!(saved.iter().all(|t| t.tensor.dtype == compute.dtype()));

        opts.weights = Some(weights_path);
        let reloaded = engine()
            .load_graph(
                &Graph::from_file(&graph_path).unwrap(),
                opts,
                &CancellationToken::new(),
            )
            .unwrap();
        assert_eq!(reloaded.graph(), mlp_graph());
        assert_eq!(run(&reloaded, vec![x]).unwrap(), before);
    }
}

#[test]
fn save_copies_unused_params_from_the_weight_file() {
    let dir = dir("save-unused");
    let mut doc = mlp_doc();
    doc["params"]
        .as_array_mut()
        .unwrap()
        .push(json!({ "name": "spare", "dtype": "float", "shape": [2, 2] }));
    let graph = Graph::from_json(&doc.to_string()).unwrap();
    let mut tensors = mlp_tensors(&mlp_weights());
    // Stored as f16, so the copy keeps the stored dtype (1.0, 2.0, -1.0, 0.5).
    let spare = TensorData::new(
        Dtype::F16,
        vec![2, 2],
        [0x3c00u16, 0x4000, 0xbc00, 0x3800]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .unwrap();
    tensors.push(NamedTensor::new("spare", spare.clone()));
    let weights = dir.join("weights.safetensors");
    tensor_file::write(&weights, &tensors).unwrap();
    let model = load(&graph, Some(&weights)).unwrap();

    let (graph_path, weights_path) = (dir.join("m.graph.json"), dir.join("m.safetensors"));
    model.save(&graph_path, &weights_path).unwrap();
    let saved = tensor_file::read(&weights_path).unwrap();
    let copied = saved.iter().find(|t| t.name == "spare").unwrap();
    assert_eq!(copied.tensor, spare);
    assert!(load(&graph, Some(&weights_path)).is_ok());

    // Saving over the original weight file works.
    model.save(&graph_path, &weights).unwrap();
    assert!(load(&graph, Some(&weights)).is_ok());

    // The original file changed: the unused param no longer has its declared shape.
    let mut changed = mlp_tensors(&mlp_weights());
    changed.push(NamedTensor::new("spare", f32_tensor(&[4], &[0.0; 4])));
    tensor_file::write(&weights, &changed).unwrap();
    let error = model.save(&graph_path, &weights_path).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("spare"));
    assert_eq!(error.context.file, Some(weights.display().to_string()));

    // The original file is gone.
    std::fs::remove_file(&weights).unwrap();
    let error = model.save(&graph_path, &weights_path).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert_eq!(error.context.operation.as_deref(), Some("model_save"));
}

#[test]
fn close_frees_the_model() {
    let dir = dir("close");
    let (weights, _) = write_mlp_weights(&dir);
    let model = load(&mlp_graph(), Some(&weights)).unwrap();
    let clone = model.clone();
    model.close();
    model.close();
    let x = NamedTensor::new("x", f32_tensor(&[1, 4], &[0.0; 4]));
    let error = run(&clone, vec![x]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert!(error.message.contains("closed"));
    let error = clone
        .save(dir.join("g.json"), dir.join("w.safetensors"))
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(clone.info().device, DeviceId::Cpu);
    assert!(format!("{clone:?}").contains("closed: true"));
}

// ---- Load errors ----

#[test]
fn weight_errors_name_the_tensor_and_the_file() {
    let dir = dir("weights");
    let w = mlp_weights();
    let all = mlp_tensors(&w);
    let file = |name: &str, tensors: &[NamedTensor]| {
        let path = dir.join(name);
        tensor_file::write(&path, tensors).unwrap();
        path
    };
    let check = |path: &PathBuf, kind: ErrorKind, tensor: &str, text: &str| {
        let error = load(&mlp_graph(), Some(path)).unwrap_err();
        assert_eq!(error.kind, kind, "{error:?}");
        assert_eq!(error.context.tensor.as_deref(), Some(tensor), "{error:?}");
        assert_eq!(error.context.file, Some(path.display().to_string()));
        assert!(
            error.message.starts_with("Could not load the model: "),
            "{}",
            error.message
        );
        assert!(error.message.contains(text), "{}", error.message);
    };

    let missing = file("missing.safetensors", &all[..3]);
    check(
        &missing,
        ErrorKind::InvalidInput,
        "fc2.bias",
        "no tensor of that name",
    );

    let mut extra = all.clone();
    extra.push(NamedTensor::new("fc3.weight", f32_tensor(&[1], &[0.0])));
    let extra = file("extra.safetensors", &extra);
    check(
        &extra,
        ErrorKind::InvalidInput,
        "fc3.weight",
        "declares no param",
    );

    let mut shape = all.clone();
    shape[1] = NamedTensor::new("fc1.bias", f32_tensor(&[4], &[0.0; 4]));
    let shape = file("shape.safetensors", &shape);
    check(&shape, ErrorKind::InvalidInput, "fc1.bias", "shape");

    let mut dtype = all.clone();
    dtype[1] = NamedTensor::new(
        "fc1.bias",
        TensorData::new(Dtype::I64, vec![3], vec![0; 24]).unwrap(),
    );
    let dtype = file("dtype.safetensors", &dtype);
    check(&dtype, ErrorKind::InvalidInput, "fc1.bias", "I64");

    // F64 storage (written with the formats crate; the facade has no f64 dtype).
    let f64_path = dir.join("f64.safetensors");
    let f64_bytes = vec![0u8; 24];
    let mut entries = Vec::new();
    for t in &all {
        if t.name != "fc1.bias" {
            let shape: Vec<usize> = t.tensor.shape.iter().map(|&d| d as usize).collect();
            entries.push((t.name.clone(), shape, t.tensor.bytes.clone()));
        }
    }
    {
        use polypous_formats::safetensors::{Dtype as St, WriteTensor, write_file};
        let mut list: Vec<(String, WriteTensor<'_>)> = entries
            .iter()
            .map(|(n, s, b)| (n.clone(), WriteTensor::new(St::F32, s, b).unwrap()))
            .collect();
        let bias_shape = [3usize];
        list.push((
            "fc1.bias".into(),
            WriteTensor::new(St::F64, &bias_shape, &f64_bytes).unwrap(),
        ));
        write_file(&f64_path, list, &Default::default()).unwrap();
    }
    check(&f64_path, ErrorKind::UnsupportedModel, "fc1.bias", "F64");

    // Truncated: the header promises more data than the file holds.
    let good = file("good.safetensors", &all);
    let bytes = std::fs::read(&good).unwrap();
    let truncated = dir.join("truncated.safetensors");
    std::fs::write(&truncated, &bytes[..bytes.len() - 5]).unwrap();
    let error = load(&mlp_graph(), Some(&truncated)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.file, Some(truncated.display().to_string()));

    // No file at all.
    let absent = dir.join("absent.safetensors");
    let error = load(&mlp_graph(), Some(&absent)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert_eq!(error.context.file, Some(absent.display().to_string()));

    // No weights for a graph with params.
    let error = load(&mlp_graph(), None).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("fc1.weight"));
}

#[test]
fn unknown_and_unavailable_devices() {
    let dir = dir("devices");
    let (weights, _) = write_mlp_weights(&dir);
    for (id, text) in [
        ("cuda:7", "no such device"),
        ("vulkan:cpu:0", "software rasterizer"),
    ] {
        let mut opts = options(Some(&weights));
        opts.device = Some(DeviceId::parse(id).unwrap());
        let error = engine()
            .load_graph(&mlp_graph(), opts, &CancellationToken::new())
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::BackendUnavailable, "{id}");
        assert_eq!(error.context.device.as_deref(), Some(id));
        assert!(error.message.contains(text), "{}", error.message);
    }
    // The default device without a GPU is the CPU.
    assert_eq!(
        load(&mlp_graph(), Some(&weights)).unwrap().info().device,
        DeviceId::Cpu
    );
}

#[test]
fn load_options_json() {
    let options = GraphLoadOptions::from_json(
        r#"{"weights":"w.safetensors","device":{"backend":"cuda","index":0},"compute_dtype":"f16"}"#,
    )
    .unwrap();
    assert_eq!(options.weights, Some(PathBuf::from("w.safetensors")));
    assert_eq!(options.device, Some(DeviceId::Cuda { index: 0 }));
    assert_eq!(options.compute_dtype, ComputeDtype::F16);
    assert_eq!(
        GraphLoadOptions::from_json(&options.to_json()).unwrap(),
        options
    );
    assert_eq!(
        GraphLoadOptions::default().to_json(),
        r#"{"weights":null,"device":null,"compute_dtype":"f32"}"#
    );
    assert_eq!(
        GraphLoadOptions::from_json("{}").unwrap(),
        GraphLoadOptions::default()
    );
    for bad in [r#"{"weight":"w"}"#, r#"{"compute_dtype":"f64"}"#, "[]", "{"] {
        let error = GraphLoadOptions::from_json(bad).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{bad}");
    }
}

#[test]
fn a_graph_invalid_with_the_compute_dtype_is_an_invalid_graph() {
    // `x` is declared f32 and added to the `float` input `z`: valid with f32, a dtype mismatch
    // with f16.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [2] }, { "name": "z", "dtype": "float", "shape": [2] } ],
        "outputs": [ { "name": "y", "value": "sum" } ],
        "nodes": [ { "id": "sum", "op": "core.add", "inputs": { "a": "x", "b": "z" } } ]
    });
    let graph = Graph::from_json(&doc.to_string()).unwrap();
    let mut opts = GraphLoadOptions::default();
    opts.compute_dtype = ComputeDtype::F16;
    let error = engine()
        .load_graph(&graph, opts, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidGraph);
    assert_eq!(error.context.node.as_deref(), Some("sum"));
    assert!(error.context.pointer.is_some());
}

// ---- Run errors ----

#[test]
fn bad_inputs_are_invalid_input() {
    let dir = dir("inputs");
    let (weights, _) = write_mlp_weights(&dir);
    let model = load(&mlp_graph(), Some(&weights)).unwrap();
    let x = |shape: &[u64]| {
        let n: u64 = shape.iter().product();
        f32_tensor(shape, &vec![0.0; n as usize])
    };
    let cases: Vec<(Vec<NamedTensor>, &str, &str)> = vec![
        (vec![], "x", "missing"),
        (
            vec![
                NamedTensor::new("x", x(&[1, 4])),
                NamedTensor::new("z", x(&[1])),
            ],
            "z",
            "not declared",
        ),
        (
            vec![NamedTensor::new("image", x(&[1, 4]))],
            "image",
            "not declared",
        ),
        (
            vec![
                NamedTensor::new("x", x(&[1, 4])),
                NamedTensor::new("x", x(&[1, 4])),
            ],
            "x",
            "twice",
        ),
        (vec![NamedTensor::new("x", x(&[1, 5]))], "x", "dim 1"),
        (vec![NamedTensor::new("x", x(&[4]))], "x", "rank"),
        (
            vec![NamedTensor::new(
                "x",
                TensorData::new(Dtype::I64, vec![1, 4], vec![0; 32]).unwrap(),
            )],
            "x",
            "dtype i64",
        ),
        // Malformed TensorData: checked before anything else.
        (
            vec![NamedTensor::new(
                "x",
                TensorData {
                    dtype: Dtype::F32,
                    shape: vec![1, 4],
                    bytes: vec![0; 15],
                },
            )],
            "x",
            "15 bytes given",
        ),
        (
            vec![NamedTensor::new(
                "x",
                TensorData {
                    dtype: Dtype::Bool,
                    shape: vec![1, 4],
                    bytes: vec![0, 1, 7, 0],
                },
            )],
            "x",
            "must be 0 or 1",
        ),
        (
            vec![NamedTensor::new(
                "x",
                TensorData {
                    dtype: Dtype::F32,
                    shape: vec![u64::MAX, 4],
                    bytes: vec![],
                },
            )],
            "x",
            "overflows",
        ),
        (vec![NamedTensor::new("x", x(&[0, 4]))], "x", "size 0"),
        (
            vec![NamedTensor::new("x", x(&[1, 1, 1, 1, 1, 1, 4]))],
            "x",
            "rank",
        ),
    ];
    for (inputs, tensor, text) in cases {
        let error = run(&model, inputs).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{error:?}");
        assert_eq!(error.context.tensor.as_deref(), Some(tensor), "{error:?}");
        assert_eq!(error.context.operation.as_deref(), Some("model_run"));
        assert!(error.message.contains(text), "{text}: {}", error.message);
    }
}

#[test]
fn errors_while_running_name_the_node() {
    // An out-of-range embedding index.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "ids", "dtype": "i64", "shape": [2] } ],
        "outputs": [ { "name": "e", "value": "emb" } ],
        "params": [ { "name": "table", "dtype": "float", "shape": [3, 2] } ],
        "nodes": [ { "id": "emb", "op": "nn.embedding", "inputs": { "ids": "ids" },
                     "params": { "weight": "table" } } ]
    });
    let dir = dir("node-error");
    let weights = dir.join("table.safetensors");
    tensor_file::write(
        &weights,
        &[NamedTensor::new("table", f32_tensor(&[3, 2], &[0.0; 6]))],
    )
    .unwrap();
    let model = load(&Graph::from_json(&doc.to_string()).unwrap(), Some(&weights)).unwrap();
    let ids = |values: [i64; 2]| {
        NamedTensor::new(
            "ids",
            TensorData::new(
                Dtype::I64,
                vec![2],
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            )
            .unwrap(),
        )
    };
    assert!(run(&model, vec![ids([0, 2])]).is_ok());
    let error = run(&model, vec![ids([0, 3])]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.node.as_deref(), Some("emb"));
    assert!(error.message.starts_with("The model run failed: "));
}

// ---- Tensor files ----

#[test]
fn tensor_files_round_trip_and_reject_bad_data() {
    let dir = dir("tensor-files");
    let tensors = vec![
        NamedTensor::new("b", f32_tensor(&[2], &[1.0, -2.0])),
        NamedTensor::new(
            "a",
            TensorData::new(Dtype::Bool, vec![3], vec![1, 0, 1]).unwrap(),
        ),
        NamedTensor::new(
            "c",
            TensorData::new(Dtype::Bf16, vec![1], vec![0x80, 0x3f]).unwrap(),
        ),
        NamedTensor::new(
            "d",
            TensorData::new(Dtype::I32, vec![1, 1], vec![7, 0, 0, 0]).unwrap(),
        ),
    ];
    let path = dir.join("t.safetensors");
    tensor_file::write(&path, &tensors).unwrap();
    let read = tensor_file::read(&path).unwrap();
    let names: Vec<&str> = read.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, ["a", "b", "c", "d"]);
    for t in &tensors {
        assert_eq!(read.iter().find(|r| r.name == t.name), Some(t));
    }

    // Writing: malformed data and duplicate names.
    let bad = vec![NamedTensor::new(
        "x",
        TensorData {
            dtype: Dtype::F32,
            shape: vec![2],
            bytes: vec![0; 3],
        },
    )];
    let error = tensor_file::write(dir.join("bad.safetensors"), &bad).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("x"));
    let twice = vec![tensors[0].clone(), tensors[0].clone()];
    let error = tensor_file::write(dir.join("twice.safetensors"), &twice).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);

    // Reading: a dtype the facade doesn't have, a bool byte that isn't 0/1, garbage, no file.
    use polypous_formats::safetensors::{Dtype as St, WriteTensor, write_file};
    let shape = [2usize];
    let u8_path = dir.join("u8.safetensors");
    write_file(
        &u8_path,
        [("q", WriteTensor::new(St::U8, &shape, &[1, 2]).unwrap())],
        &Default::default(),
    )
    .unwrap();
    let error = tensor_file::read(&u8_path).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("q"));
    assert!(error.message.contains("U8"), "{}", error.message);

    let bool_path = dir.join("bool.safetensors");
    write_file(
        &bool_path,
        [(
            "flags",
            WriteTensor::new(St::Bool, &shape, &[1, 2]).unwrap(),
        )],
        &Default::default(),
    )
    .unwrap();
    let error = tensor_file::read(&bool_path).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.tensor.as_deref(), Some("flags"));

    let garbage = dir.join("garbage.safetensors");
    std::fs::write(&garbage, b"\xff\xff\xff\xff\xff\xff\xff\xffxx").unwrap();
    let error = tensor_file::read(&garbage).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.file, Some(garbage.display().to_string()));

    let error = tensor_file::read(dir.join("none.safetensors")).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
}

// ---- GPU (Vulkan; lavapipe in CI) ----

/// Load and run on every available Vulkan device, and check that `bf16` (which wgpu lacks) is
/// rejected as a compute dtype and as an input dtype. Software rasterizers are probed only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (the CI lavapipe job sets it).
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "gpu"]
fn mlp_runs_on_vulkan_and_bf16_is_rejected() {
    let dir = dir("vulkan");
    let mut config = EngineConfig::default();
    config.software_rasterizers =
        std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    config.probe_cache_dir = Some(dir.join("probe-cache"));
    let engine = Engine::new(config).unwrap();
    let devices: Vec<DeviceId> = engine
        .devices()
        .into_iter()
        .filter(|d| d.backend == polypous::BackendKind::Vulkan && d.status.is_available())
        .map(|d| d.id)
        .collect();
    assert!(!devices.is_empty(), "no available Vulkan device");
    let (weights, w) = write_mlp_weights(&dir);
    for device in devices {
        eprintln!("testing {device}");
        let mut opts = options(Some(&weights));
        opts.device = Some(device);
        let model = engine
            .load_graph(&mlp_graph(), opts.clone(), &CancellationToken::new())
            .unwrap();
        assert_eq!(model.info().device, device);
        let x = values(12, 40);
        let out = run(&model, vec![NamedTensor::new("x", f32_tensor(&[3, 4], &x))]).unwrap();
        assert_close(
            &f32_values(out.output("y").unwrap()),
            &mlp_reference(&x, 3, &w),
        );

        // A bf16 input can't be uploaded.
        let bf16 = TensorData::new(Dtype::Bf16, vec![3, 4], vec![0; 24]).unwrap();
        let error = run(&model, vec![NamedTensor::new("x", bf16)]).unwrap_err();
        assert_eq!(error.kind, ErrorKind::UnsupportedOperation);
        assert_eq!(error.context.tensor.as_deref(), Some("x"));

        // bf16 compute: rejected by the plan's device check.
        opts.compute_dtype = ComputeDtype::Bf16;
        let error = engine
            .load_graph(&mlp_graph(), opts, &CancellationToken::new())
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::UnsupportedOperation);
        assert_eq!(error.context.device, Some(device.to_string()));
        assert!(error.message.contains("bf16"), "{}", error.message);
    }
}
