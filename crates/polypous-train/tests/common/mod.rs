//! Helpers shared by the trainer tests: temporary directories, graphs, datasets, configs and an
//! event recorder.

#![allow(dead_code, reason = "each test binary uses a different part")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, TensorData};
use polypous_ir::Graph;
use polypous_runtime::CancellationToken;
use polypous_train::{
    InMemoryDataset, ManualClock, TrainError, Trainer, TrainerOptions, TrainingConfig,
    TrainingEvent, TrainingOutcome,
};
use serde_json::{Value, json};

/// A directory under the system's temporary directory, deleted on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(name: &str) -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("polypous-train-{name}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, p: &str) -> PathBuf {
        self.0.join(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Resolves every device text to the CPU.
pub fn cpu(_: &str) -> Result<DeviceId, TrainError> {
    Ok(DeviceId::Cpu)
}

pub fn graph(doc: &Value) -> Graph {
    Graph::from_json(doc.to_string().as_bytes()).unwrap_or_else(|r| panic!("{r}"))
}

fn param(name: &str, shape: &[usize], init: Value) -> Value {
    json!({ "name": name, "dtype": "float", "shape": shape, "init": init })
}

fn kaiming() -> Value {
    json!({ "kind": "kaiming_uniform", "a": 2.23606797749979 })
}

/// `x [batch, n_in]` → linear → relu → linear → `y [batch, n_out]`, PyTorch's default inits.
pub fn mlp(n_in: usize, hidden: usize, n_out: usize) -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", n_in] } ],
        "outputs": [ { "name": "y", "value": "fc2" } ],
        "params": [
            param("fc1.weight", &[hidden, n_in], kaiming()),
            param("fc1.bias", &[hidden], json!({ "kind": "uniform", "low": -0.3, "high": 0.3 })),
            param("fc2.weight", &[n_out, hidden], kaiming()),
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

/// `x [batch, n_in]` → linear → batch_norm → relu → dropout 0.25 → linear → `y`, plus an unused
/// trainable param `unused` (no gradient, no optimizer state).
pub fn bn_dropout(n_in: usize, hidden: usize, n_out: usize) -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", n_in] } ],
        "outputs": [ { "name": "y", "value": "fc2" } ],
        "params": [
            param("fc1.weight", &[hidden, n_in], kaiming()),
            param("fc1.bias", &[hidden], json!({ "kind": "zeros" })),
            param("bn.weight", &[hidden], json!({ "kind": "ones" })),
            param("bn.bias", &[hidden], json!({ "kind": "zeros" })),
            param("bn.running_mean", &[hidden], json!({ "kind": "zeros" })),
            param("bn.running_var", &[hidden], json!({ "kind": "ones" })),
            param("fc2.weight", &[n_out, hidden], kaiming()),
            param("fc2.bias", &[n_out], json!({ "kind": "zeros" })),
            param("unused", &[3], json!({ "kind": "constant", "value": 0.5 })),
        ],
        "nodes": [
            { "id": "fc1", "op": "nn.linear", "inputs": { "x": "x" },
              "params": { "weight": "fc1.weight", "bias": "fc1.bias" } },
            { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "fc1" }, "attrs": { "momentum": 0.1 },
              "params": { "weight": "bn.weight", "bias": "bn.bias",
                          "running_mean": "bn.running_mean", "running_var": "bn.running_var" } },
            { "id": "act", "op": "nn.relu", "inputs": { "x": "bn" } },
            { "id": "drop", "op": "nn.dropout", "inputs": { "x": "act" }, "attrs": { "p": 0.25 } },
            { "id": "fc2", "op": "nn.linear", "inputs": { "x": "drop" },
              "params": { "weight": "fc2.weight", "bias": "fc2.bias" } }
        ]
    })
}

pub fn f32_data(shape: &[u64], values: &[f32]) -> TensorData {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorData::new(DType::F32, shape.to_vec(), bytes).unwrap()
}

pub fn i64_data(values: &[i64]) -> TensorData {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    TensorData::new(DType::I64, vec![values.len() as u64], bytes).unwrap()
}

pub fn read_f32(data: &TensorData) -> Vec<f32> {
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// Deterministic values in `[-1, 1)`.
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

/// `n` samples of 3 Gaussian-ish blobs in 4 dims, `x` and the class `label` (i64).
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
            ("x".into(), f32_data(&[n as u64, 4], &x)),
            ("label".into(), i64_data(&label)),
        ],
    }
}

/// `n` samples of `y = x · w + 0.5` in 4 dims (a learnable regression), `x` and `y` [1].
pub fn regression(name: &str, n: usize, seed: u64) -> InMemoryDataset {
    let x = values(n * 4, seed);
    let w = [0.7f32, -1.2, 0.4, 0.9];
    let y: Vec<f32> = (0..n)
        .map(|i| (0..4).map(|d| x[i * 4 + d] * w[d]).sum::<f32>() + 0.5)
        .collect();
    InMemoryDataset {
        name: name.into(),
        fields: vec![
            ("x".into(), f32_data(&[n as u64, 4], &x)),
            ("y".into(), f32_data(&[n as u64, 1], &y)),
        ],
    }
}

/// A classification config on the `tensors` dataset `train` (cross-entropy, accuracy), with
/// `extra` merged into the root.
pub fn classification_config(extra: Value) -> TrainingConfig {
    let mut doc = json!({
        "format": "polypous.training", "format_version": 1, "seed": 7, "device": "cpu",
        "epochs": 3, "batch_size": 8,
        "data": { "train": { "kind": "tensors", "name": "train" } },
        "inputs": { "x": "x" },
        "loss": { "kind": "cross_entropy", "prediction": "y", "target": "label" },
        "metrics": [ { "kind": "accuracy", "prediction": "y", "target": "label" } ],
        "optimizer": { "kind": "adamw", "lr": 0.01 }
    });
    merge(&mut doc, extra);
    TrainingConfig::from_json(&doc.to_string()).unwrap_or_else(|e| panic!("{e}"))
}

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

/// A clock that moves forward by one second every time it is read, so every `step` event is
/// emitted (none is throttled).
#[derive(Debug, Default)]
pub struct TickingClock(AtomicU64);

impl polypous_train::Clock for TickingClock {
    fn now(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.0.fetch_add(1, Ordering::SeqCst))
    }
}

pub fn ticking() -> Arc<dyn polypous_train::Clock> {
    Arc::new(TickingClock::default())
}

/// Options for `tensors` datasets with `base` as the base directory and a [`TickingClock`].
pub fn options(
    base: &Path,
    datasets: Vec<InMemoryDataset>,
    clock: Option<Arc<ManualClock>>,
) -> TrainerOptions {
    TrainerOptions {
        weights: None,
        base_dir: Some(base.to_path_buf()),
        datasets,
        clock: Some(clock.map_or_else(ticking, |c| c as Arc<dyn polypous_train::Clock>)),
    }
}

pub fn trainer(graph_doc: &Value, config: TrainingConfig, opts: TrainerOptions) -> Trainer {
    Trainer::new(
        &graph(graph_doc),
        config,
        opts,
        &cpu,
        &CancellationToken::new(),
    )
    .unwrap_or_else(|e| panic!("{e} ({:?})", e.pointer()))
}

/// Runs to the end (or until `cancel`), recording every event.
pub fn run(t: &Trainer, cancel: &CancellationToken) -> (TrainingOutcome, Vec<TrainingEvent>) {
    let mut events = Vec::new();
    let outcome = t
        .run(cancel, |e| events.push(e))
        .unwrap_or_else(|e| panic!("{e}"));
    (outcome, events)
}

/// The event names, for order checks.
pub fn names(events: &[TrainingEvent]) -> Vec<&'static str> {
    events.iter().map(TrainingEvent::name).collect()
}
