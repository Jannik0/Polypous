//! Helpers shared by the interpreter tests: tensors, graphs and in-memory weight files.

#![allow(dead_code, reason = "each test binary uses a different part")]

use std::collections::BTreeMap;

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData};
use polypous_formats::safetensors::{
    Dtype as StorageDtype, SafetensorsFile, WriteTensor, serialize,
};
use polypous_ir::Graph;
use polypous_runtime::{Model, Plan, PlanOptions, RunOptions, RunOutput, WeightFile};
use serde_json::Value;

/// Deterministic values in `[-1, 1)` (a small LCG), different per `seed`.
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

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn tensor(dtype: DType, shape: &[usize], bytes: Vec<u8>, device: DeviceId) -> DynTensor {
    let shape = shape.iter().map(|&d| d as u64).collect();
    DynTensor::from_data(&TensorData::new(dtype, shape, bytes).unwrap(), device).unwrap()
}

pub fn f32_tensor(shape: &[usize], values: &[f32], device: DeviceId) -> DynTensor {
    tensor(DType::F32, shape, f32_bytes(values), device)
}

pub fn i64_tensor(shape: &[usize], values: &[i64], device: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    tensor(DType::I64, shape, bytes, device)
}

/// The f32 values of a tensor.
pub fn read_f32(t: &DynTensor) -> Vec<f32> {
    let data = t.to_data().unwrap();
    assert_eq!(data.dtype, DType::F32);
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

pub fn graph(doc: &Value) -> Graph {
    Graph::from_json(&serde_json::to_vec(doc).unwrap())
        .unwrap_or_else(|report| panic!("invalid test graph:\n{report}\n{doc:#}"))
}

/// A safetensors file in memory: name, storage dtype, shape and bytes per tensor.
pub fn weight_bytes(tensors: &[(&str, StorageDtype, Vec<usize>, Vec<u8>)]) -> Vec<u8> {
    let writes: Vec<(String, WriteTensor<'_>)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                (*name).to_owned(),
                WriteTensor::new(*dtype, shape, bytes).unwrap(),
            )
        })
        .collect();
    serialize(writes, &BTreeMap::new()).unwrap()
}

pub fn weight_file(tensors: &[(&str, StorageDtype, Vec<usize>, Vec<u8>)]) -> WeightFile {
    WeightFile::from(SafetensorsFile::from_bytes(weight_bytes(tensors)).unwrap())
}

/// F32 weights with deterministic values: name and shape per tensor. Returns the file and the
/// values by name.
pub fn f32_weights(params: &[(&str, &[usize])]) -> (WeightFile, BTreeMap<String, Vec<f32>>) {
    let mut values_by_name = BTreeMap::new();
    let mut tensors = Vec::new();
    for (k, (name, shape)) in params.iter().enumerate() {
        let n = shape.iter().product();
        let v = values(n, 1000 + k as u64);
        tensors.push((*name, StorageDtype::F32, shape.to_vec(), f32_bytes(&v)));
        values_by_name.insert((*name).to_owned(), v);
    }
    (weight_file(&tensors), values_by_name)
}

pub fn model(graph: &Graph, device: DeviceId, weights: &WeightFile) -> Model {
    let plan = Plan::new(graph, PlanOptions::new(device)).unwrap();
    Model::load(plan, weights).unwrap()
}

/// Runs with inputs and state by name.
pub fn run(
    model: &Model,
    inputs: Vec<(&str, DynTensor)>,
    state: Vec<(&str, DynTensor)>,
) -> RunOutput {
    let own = |v: Vec<(&str, DynTensor)>| v.into_iter().map(|(n, t)| (n.to_owned(), t)).collect();
    model
        .run(own(inputs), own(state), &RunOptions::default())
        .unwrap()
}

/// Asserts two tensors are bit-identical (dtype, shape, bytes).
pub fn assert_same(actual: &DynTensor, expected: &DynTensor, what: &str) {
    let (a, e) = (actual.to_data().unwrap(), expected.to_data().unwrap());
    assert_eq!(a.dtype, e.dtype, "{what}: dtype");
    assert_eq!(a.shape, e.shape, "{what}: shape");
    assert!(
        a.bytes == e.bytes,
        "{what}: values differ:\n{:?}\n{:?}",
        read_f32(actual),
        read_f32(expected)
    );
}

/// Every listed GPU device of the compiled-in GPU backends, without software rasterizers unless
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (as in the backend's GPU tests). Panics if a GPU backend is
/// compiled in but lists no device.
pub fn gpu_devices() -> Vec<DeviceId> {
    use polypous_backend::device::{BackendKind, EnumerateOptions, list_devices};
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
