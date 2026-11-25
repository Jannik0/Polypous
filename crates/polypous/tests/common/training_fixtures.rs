//! Small training problems for the facade and CLI tests (public-api.md §1.4): an MLP graph, seeded
//! synthetic datasets (in memory, or written as a safetensors file) and configs.
//!
//! Shared by `crates/polypous/tests/training.rs` and `crates/polypous-cli/tests/train.rs`, which
//! include this file with `#[path]`. Only the public `polypous` API is used.

#![allow(
    dead_code,
    reason = "each test binary uses a different part of this module"
)]

use std::path::{Path, PathBuf};

use polypous::{Dtype, Graph, InMemoryDataset, NamedTensor, TensorData, tensor_file};
use serde_json::{Value, json};

fn param(name: &str, shape: &[usize], init: Value) -> Value {
    json!({ "name": name, "dtype": "float", "shape": shape, "init": init })
}

/// `x [batch, n_in]` → linear → relu → linear → `y [batch, n_out]`, PyTorch's default inits.
pub fn mlp_json(n_in: usize, hidden: usize, n_out: usize) -> Value {
    let kaiming = json!({ "kind": "kaiming_uniform", "a": 2.23606797749979 });
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", n_in] } ],
        "outputs": [ { "name": "y", "value": "fc2" } ],
        "params": [
            param("fc1.weight", &[hidden, n_in], kaiming.clone()),
            param("fc1.bias", &[hidden], json!({ "kind": "uniform", "low": -0.3, "high": 0.3 })),
            param("fc2.weight", &[n_out, hidden], kaiming),
            param("fc2.bias", &[n_out], json!({ "kind": "zeros" })),
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

pub fn mlp(n_in: usize, hidden: usize, n_out: usize) -> Graph {
    Graph::from_json(&mlp_json(n_in, hidden, n_out).to_string()).unwrap()
}

pub fn f32_tensor(shape: &[u64], values: &[f32]) -> TensorData {
    TensorData::new(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

pub fn i64_tensor(values: &[i64]) -> TensorData {
    TensorData::new(
        Dtype::I64,
        vec![values.len() as u64],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

pub fn read_f32(data: &TensorData) -> Vec<f32> {
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// Deterministic values in `[-1, 1)` (a seeded LCG).
pub fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// `n` samples of `y = x · w + 0.5` in 4 dims: fields `x` `[n, 4]` and `y` `[n, 1]`.
pub fn regression_fields(n: usize, seed: u64) -> Vec<NamedTensor> {
    let x = values(n * 4, seed);
    let w = [0.7f32, -1.2, 0.4, 0.9];
    let y: Vec<f32> = (0..n)
        .map(|i| (0..4).map(|d| x[i * 4 + d] * w[d]).sum::<f32>() + 0.5)
        .collect();
    vec![
        NamedTensor::new("x", f32_tensor(&[n as u64, 4], &x)),
        NamedTensor::new("y", f32_tensor(&[n as u64, 1], &y)),
    ]
}

/// `n` samples of 3 blobs in 4 dims: fields `x` `[n, 4]` and the class `label` (i64).
pub fn blobs(name: &str, n: usize, seed: u64) -> InMemoryDataset {
    let noise = values(n * 4, seed);
    let centers = [
        [1.5f32, 0.0, -1.0, 0.5],
        [-1.0, 1.5, 0.5, -1.0],
        [0.0, -1.5, 1.0, 1.5],
    ];
    let mut x = Vec::with_capacity(n * 4);
    let mut label = Vec::with_capacity(n);
    for i in 0..n {
        let c = i % 3;
        for d in 0..4 {
            x.push(centers[c][d] + 0.6 * noise[i * 4 + d]);
        }
        label.push(c as i64);
    }
    InMemoryDataset {
        name: name.into(),
        fields: vec![
            NamedTensor::new("x", f32_tensor(&[n as u64, 4], &x)),
            NamedTensor::new("label", i64_tensor(&label)),
        ],
    }
}

/// Writes the regression set to `dir/train.safetensors` (creating `dir`) and returns its path.
pub fn write_regression(dir: &Path, n: usize, seed: u64) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join("train.safetensors");
    tensor_file::write(&path, &regression_fields(n, seed)).unwrap();
    path
}

/// Merges `extra` into the root of `doc`; a `null` value removes the field.
pub fn merge(doc: &mut Value, extra: Value) {
    if let (Value::Object(a), Value::Object(b)) = (doc, extra) {
        for (k, v) in b {
            if v.is_null() {
                a.remove(&k);
            } else {
                a.insert(k, v);
            }
        }
    }
}

/// The MSE regression on the safetensors file `train.safetensors` (relative to the base
/// directory), for the MLP `mlp(4, 16, 1)`, with `extra` merged into the root.
pub fn regression_config(extra: Value) -> Value {
    let mut doc = json!({
        "format": "polypous.training", "format_version": 1, "seed": 3, "device": "cpu",
        "epochs": 30, "batch_size": 16,
        "data": { "train": { "kind": "safetensors", "path": "train.safetensors" } },
        "inputs": { "x": "x" },
        "loss": { "kind": "mse", "prediction": "y", "target": "y" },
        "optimizer": { "kind": "adam", "lr": 0.01 }
    });
    merge(&mut doc, extra);
    doc
}

/// The classification on the in-memory datasets `train` and `val` (cross-entropy, accuracy), for
/// `mlp(4, 16, 3)`.
pub fn classification_config(extra: Value) -> Value {
    let mut doc = json!({
        "format": "polypous.training", "format_version": 1, "seed": 7, "device": "cpu",
        "epochs": 8, "batch_size": 16,
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "tensors", "name": "val" } },
        "inputs": { "x": "x" },
        "loss": { "kind": "cross_entropy", "prediction": "y", "target": "label" },
        "metrics": [ { "kind": "accuracy", "prediction": "y", "target": "label" } ],
        "optimizer": { "kind": "adamw", "lr": 0.01 }
    });
    merge(&mut doc, extra);
    doc
}
