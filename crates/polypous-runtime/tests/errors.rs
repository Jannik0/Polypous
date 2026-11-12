//! Error cases of the interpreter (runtime.md §2–§5): weights, call inputs, out-of-range indices,
//! cancellation, invalid graphs and device support. Each test asserts the error kind and that the
//! message names the tensor, value or node.
//!
//! ```text
//! cargo test -p polypous-runtime --test errors
//! VK_DRIVER_FILES=… POLYPOUS_TEST_SOFTWARE_VULKAN=1 cargo test -p polypous-runtime --features vulkan --test errors -- --include-ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, TensorError};
use polypous_formats::safetensors::Dtype as St;
use polypous_ir::BindError;
use polypous_runtime::{
    CancellationToken, ErrorKind, Model, Plan, PlanOptions, RunOptions, RuntimeError, WeightFile,
    WeightProblem,
};
use serde_json::{Value, json};
use support::*;

const CPU: DeviceId = DeviceId::Cpu;

/// `y = linear(x, w, b)` with `x: [batch, 2]`, `w: [3, 2]`, `b: [3]`.
fn linear_graph() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "dims": { "batch": { "max": 8 } },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 2] } ],
        "outputs": [ { "name": "y", "value": "fc" } ],
        "params": [
            { "name": "fc.weight", "dtype": "float", "shape": [3, 2] },
            { "name": "fc.bias", "dtype": "float", "shape": [3] }
        ],
        "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "x" },
                     "params": { "weight": "fc.weight", "bias": "fc.bias" } } ]
    })
}

fn linear_plan() -> Plan {
    Plan::new(&graph(&linear_graph()), PlanOptions::new(CPU)).unwrap()
}

fn f32w<'a>(name: &'a str, shape: &[usize]) -> (&'a str, St, Vec<usize>, Vec<u8>) {
    let n = shape.iter().product();
    (name, St::F32, shape.to_vec(), f32_bytes(&values(n, 1)))
}

fn load_err(tensors: &[(&str, St, Vec<usize>, Vec<u8>)]) -> RuntimeError {
    Model::load(linear_plan(), &weight_file(tensors)).unwrap_err()
}

fn assert_weight(
    err: &RuntimeError,
    tensor: &str,
    kind: ErrorKind,
    check: impl Fn(&WeightProblem) -> bool,
) {
    match err {
        RuntimeError::Weight { tensor: t, problem } => {
            assert_eq!(t, tensor, "{err}");
            assert!(check(problem), "{err}");
        }
        other => panic!("expected a weight error for `{tensor}`, got {other:?}"),
    }
    assert_eq!(err.kind(), kind, "{err}");
    assert!(err.to_string().contains(tensor), "{err}");
}

// ---- Weights ----

#[test]
fn weights_load_and_run() {
    let model = Model::load(
        linear_plan(),
        &weight_file(&[f32w("fc.weight", &[3, 2]), f32w("fc.bias", &[3])]),
    )
    .unwrap();
    let out = run(
        &model,
        vec![("x", f32_tensor(&[1, 2], &[1.0, 2.0], CPU))],
        vec![],
    );
    assert_eq!(out.output("y").unwrap().shape(), [1, 3]);
}

#[test]
fn missing_weight() {
    let err = load_err(&[f32w("fc.weight", &[3, 2])]);
    assert_weight(&err, "fc.bias", ErrorKind::InvalidInput, |p| {
        *p == WeightProblem::Missing
    });
    let err = Model::without_weights(linear_plan()).unwrap_err();
    assert_weight(&err, "fc.weight", ErrorKind::InvalidInput, |p| {
        *p == WeightProblem::Missing
    });
}

#[test]
fn extra_weight() {
    let err = load_err(&[
        f32w("fc.weight", &[3, 2]),
        f32w("fc.bias", &[3]),
        f32w("fc.extra", &[1]),
    ]);
    assert_weight(&err, "fc.extra", ErrorKind::InvalidInput, |p| {
        *p == WeightProblem::Extra
    });
}

#[test]
fn wrong_dtype_weights() {
    let ints = ("fc.bias", St::I64, vec![3], vec![0u8; 24]);
    let err = load_err(&[f32w("fc.weight", &[3, 2]), ints]);
    assert_weight(
        &err,
        "fc.bias",
        ErrorKind::InvalidInput,
        |p| matches!(p, WeightProblem::NotFloat { stored } if stored == "I64"),
    );
    let doubles = ("fc.bias", St::F64, vec![3], vec![0u8; 24]);
    let err = load_err(&[f32w("fc.weight", &[3, 2]), doubles]);
    assert_weight(
        &err,
        "fc.bias",
        ErrorKind::UnsupportedModel,
        |p| matches!(p, WeightProblem::UnsupportedStorage { stored } if stored == "F64"),
    );
}

#[test]
fn wrong_shape_weights() {
    for shape in [vec![2, 3], vec![3, 2, 1], vec![6], vec![30, 2]] {
        let err = load_err(&[f32w("fc.weight", &shape), f32w("fc.bias", &[3])]);
        assert_weight(
            &err,
            "fc.weight",
            ErrorKind::InvalidInput,
            |p| matches!(p, WeightProblem::Shape { declared, stored } if declared == &[3, 2] && stored == &shape),
        );
    }
}

fn scratch_dir(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("polypous-runtime-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn truncated_and_oversized_weight_files() {
    let dir = scratch_dir("truncated");
    let bytes = weight_bytes(&[f32w("fc.weight", &[3, 2]), f32w("fc.bias", &[3])]);
    // Truncated: the last tensor's data is cut short.
    let path = dir.join("truncated.safetensors");
    std::fs::write(&path, &bytes[..bytes.len() - 5]).unwrap();
    let err = WeightFile::open(&path).unwrap_err();
    assert!(matches!(err, RuntimeError::WeightFile { .. }), "{err:?}");
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    assert!(err.to_string().contains("truncated.safetensors"), "{err}");
    // Oversized: a header that claims a tensor far larger than the file holds.
    let header = br#"{"fc.weight":{"dtype":"F32","shape":[1099511627776,2],"data_offsets":[0,24]},"fc.bias":{"dtype":"F32","shape":[3],"data_offsets":[24,36]}}"#;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header);
    file.extend_from_slice(&[0u8; 36]);
    let path = dir.join("oversized.safetensors");
    std::fs::write(&path, &file).unwrap();
    let err = WeightFile::open(&path).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
    assert!(err.to_string().contains("fc.weight"), "{err}");
    // A file that doesn't exist: I/O.
    let err = WeightFile::open(dir.join("missing.safetensors")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Io, "{err}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sharded_checkpoint() {
    let dir = scratch_dir("sharded");
    std::fs::write(
        dir.join("a.safetensors"),
        weight_bytes(&[f32w("fc.weight", &[3, 2])]),
    )
    .unwrap();
    std::fs::write(
        dir.join("b.safetensors"),
        weight_bytes(&[f32w("fc.bias", &[3])]),
    )
    .unwrap();
    let index =
        json!({ "weight_map": { "fc.weight": "a.safetensors", "fc.bias": "b.safetensors" } });
    let index_path = dir.join("model.safetensors.index.json");
    std::fs::write(&index_path, serde_json::to_vec(&index).unwrap()).unwrap();
    let weights = WeightFile::open(&index_path).unwrap();
    assert!(matches!(weights, WeightFile::Sharded(_)));
    let model = Model::load(linear_plan(), &weights).unwrap();
    let single = Model::load(
        linear_plan(),
        &weight_file(&[f32w("fc.weight", &[3, 2]), f32w("fc.bias", &[3])]),
    )
    .unwrap();
    let x = || vec![("x", f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0], CPU))];
    assert_same(
        run(&model, x(), vec![]).output("y").unwrap(),
        run(&single, x(), vec![]).output("y").unwrap(),
        "sharded vs single",
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

/// `y = linear(ones[1, 1], w)` returns `w` (as a column) in the compute dtype.
fn identity_model(compute: polypous_ir::ComputeDtype, weight: (St, Vec<u8>)) -> Model {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [1, 1] } ],
        "outputs": [ { "name": "y", "value": "fc" } ],
        "params": [ { "name": "w", "dtype": "float", "shape": [3, 1] } ],
        "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "x" }, "params": { "weight": "w" } } ]
    });
    let plan = Plan::new(
        &graph(&doc),
        PlanOptions {
            device: CPU,
            compute_dtype: compute,
        },
    )
    .unwrap();
    Model::load(plan, &weight_file(&[("w", weight.0, vec![3, 1], weight.1)])).unwrap()
}

fn half_bits(model: &Model, dtype: DType) -> Vec<u16> {
    let one = match dtype {
        DType::F16 => 0x3C00u16,
        _ => 0x3F80,
    };
    let x = tensor(dtype, &[1, 1], one.to_le_bytes().to_vec(), CPU);
    let out = run(model, vec![("x", x)], vec![]);
    let data = out.output("y").unwrap().to_data().unwrap();
    assert_eq!(data.dtype, dtype);
    let (chunks, _) = data.bytes.as_chunks::<2>();
    chunks.iter().map(|&b| u16::from_le_bytes(b)).collect()
}

#[test]
fn params_are_cast_to_the_compute_dtype_once() {
    use polypous_ir::ComputeDtype;
    // f32 storage, f16 compute: 0.1 → 0x2E66, 1/3 → 0x3555, 65520 → inf (round to nearest even).
    let m = identity_model(
        ComputeDtype::F16,
        (St::F32, f32_bytes(&[0.1, 1.0 / 3.0, 65520.0])),
    );
    assert_eq!(half_bits(&m, DType::F16), [0x2E66, 0x3555, 0x7C00]);
    // bf16 storage, f16 compute: widened exactly, then rounded once (bf16 0x3DCD = 0.10009765625
    // is exact in f16: 0x2E68).
    let bf16: Vec<u8> = [0x3DCDu16, 0x3F80, 0xC040]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let m = identity_model(ComputeDtype::F16, (St::BF16, bf16));
    assert_eq!(half_bits(&m, DType::F16), [0x2E68, 0x3C00, 0xC200]);
    // f16 storage, f32 compute: exact.
    let f16: Vec<u8> = [0x2E66u16, 0x3555, 0xC200]
        .iter()
        .flat_map(|b| b.to_le_bytes())
        .collect();
    let m = identity_model(ComputeDtype::F32, (St::F16, f16));
    let x = f32_tensor(&[1, 1], &[1.0], CPU);
    let out = run(&m, vec![("x", x)], vec![]);
    let widened: Vec<f64> = read_f32(out.output("y").unwrap())
        .into_iter()
        .map(f64::from)
        .collect();
    assert_eq!(widened, [0.0999755859375, 0.333251953125, -3.0]);
    // f32 storage, bf16 compute: 0.1 → 0x3DCD.
    let m = identity_model(ComputeDtype::Bf16, (St::F32, f32_bytes(&[0.1, 1.0, -3.0])));
    assert_eq!(half_bits(&m, DType::BF16), [0x3DCD, 0x3F80, 0xC040]);
}

#[test]
fn tied_and_unused_params() {
    // `w` is used by two nodes, `unused` by none: both must be in the file.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [1, 2] } ],
        "outputs": [ { "name": "y", "value": "b" } ],
        "params": [
            { "name": "w", "dtype": "float", "shape": [2, 2] },
            { "name": "unused", "dtype": "float", "shape": [4] }
        ],
        "nodes": [
            { "id": "a", "op": "nn.linear", "inputs": { "x": "x" }, "params": { "weight": "w" } },
            { "id": "b", "op": "nn.linear", "inputs": { "x": "a" }, "params": { "weight": "w" } }
        ]
    });
    let plan = || Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap();
    let w = [1.0f32, 2.0, 3.0, 4.0];
    let file = weight_file(&[
        ("w", St::F32, vec![2, 2], f32_bytes(&w)),
        f32w("unused", &[4]),
    ]);
    let model = Model::load(plan(), &file).unwrap();
    let out = run(
        &model,
        vec![("x", f32_tensor(&[1, 2], &[1.0, 1.0], CPU))],
        vec![],
    );
    // a = [3, 7]; b = [3 + 14, 9 + 28].
    assert_eq!(read_f32(out.output("y").unwrap()), [17.0, 37.0]);
    let err = Model::load(
        plan(),
        &weight_file(&[("w", St::F32, vec![2, 2], f32_bytes(&w))]),
    )
    .unwrap_err();
    assert_weight(&err, "unused", ErrorKind::InvalidInput, |p| {
        *p == WeightProblem::Missing
    });
    let err = Model::load(
        plan(),
        &weight_file(&[
            ("w", St::F32, vec![2, 2], f32_bytes(&w)),
            f32w("unused", &[5]),
        ]),
    )
    .unwrap_err();
    assert_weight(&err, "unused", ErrorKind::InvalidInput, |p| {
        matches!(p, WeightProblem::Shape { .. })
    });
}

// ---- Call inputs ----

fn linear_model() -> Model {
    Model::load(
        linear_plan(),
        &weight_file(&[f32w("fc.weight", &[3, 2]), f32w("fc.bias", &[3])]),
    )
    .unwrap()
}

fn call_err(model: &Model, inputs: Vec<(&str, polypous_backend::DynTensor)>) -> RuntimeError {
    let inputs = inputs.into_iter().map(|(n, t)| (n.to_owned(), t)).collect();
    model
        .run(inputs, vec![], &RunOptions::default())
        .unwrap_err()
}

fn assert_input(err: &RuntimeError, name: &str, check: impl Fn(&BindError) -> bool) {
    match err {
        RuntimeError::Bind(b) => assert!(check(b), "{err}"),
        other => panic!("expected a binding error, got {other:?}"),
    }
    assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
    assert!(err.to_string().contains(name), "{err}");
}

#[test]
fn wrong_inputs() {
    let m = linear_model();
    let x = || f32_tensor(&[2, 2], &[1.0; 4], CPU);
    let err = call_err(&m, vec![]);
    assert_input(&err, "`x`", |b| matches!(b, BindError::Missing { .. }));
    let err = call_err(&m, vec![("x", x()), ("z", x())]);
    assert_input(&err, "`z`", |b| matches!(b, BindError::Unknown { .. }));
    let err = call_err(&m, vec![("x", i64_tensor(&[2, 2], &[1; 4], CPU))]);
    assert_input(&err, "`x`", |b| matches!(b, BindError::Dtype { .. }));
    let err = call_err(&m, vec![("x", f32_tensor(&[4], &[1.0; 4], CPU))]);
    assert_input(&err, "`x`", |b| matches!(b, BindError::Rank { .. }));
    let err = call_err(&m, vec![("x", f32_tensor(&[2, 3], &[1.0; 6], CPU))]);
    assert_input(&err, "`x`", |b| matches!(b, BindError::DimMismatch { .. }));
    let err = call_err(&m, vec![("x", f32_tensor(&[9, 2], &[1.0; 18], CPU))]);
    assert_input(&err, "batch", |b| {
        matches!(b, BindError::SymbolOutOfRange { .. })
    });
}

#[test]
fn inconsistent_symbols() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "dims": { "batch": {} },
        "inputs": [
            { "name": "a", "dtype": "f32", "shape": ["batch", 2] },
            { "name": "b", "dtype": "f32", "shape": ["batch", 2] }
        ],
        "outputs": [ { "name": "y", "value": "sum" } ],
        "nodes": [ { "id": "sum", "op": "core.add", "inputs": { "a": "a", "b": "b" } } ]
    });
    let m =
        Model::without_weights(Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap()).unwrap();
    let err = call_err(
        &m,
        vec![
            ("a", f32_tensor(&[2, 2], &[0.0; 4], CPU)),
            ("b", f32_tensor(&[3, 2], &[0.0; 6], CPU)),
        ],
    );
    assert_input(&err, "batch", |b| {
        matches!(b, BindError::InconsistentSymbol { .. })
    });
}

#[test]
fn missing_and_unknown_state() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [2] } ],
        "state": [ { "name": "s", "dtype": "f32", "shape": [2] } ],
        "outputs": [ { "name": "y", "value": "y" } ],
        "nodes": [ { "id": "y", "op": "core.add", "inputs": { "a": "s", "b": "x" } } ],
        "state_updates": { "s": "y" }
    });
    let m =
        Model::without_weights(Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap()).unwrap();
    let x = || ("x".to_owned(), f32_tensor(&[2], &[1.0, 2.0], CPU));
    let err = m
        .run(vec![x()], vec![], &RunOptions::default())
        .unwrap_err();
    assert_input(&err, "`s`", |b| matches!(b, BindError::Missing { .. }));
    let s = || ("s".to_owned(), f32_tensor(&[2], &[1.0, 2.0], CPU));
    let other = ("t".to_owned(), f32_tensor(&[2], &[1.0, 2.0], CPU));
    let err = m
        .run(vec![x()], vec![s(), other], &RunOptions::default())
        .unwrap_err();
    assert_input(&err, "`t`", |b| matches!(b, BindError::Unknown { .. }));
    // A state value passed as an input is an unknown input.
    let err = m
        .run(vec![x(), s()], vec![], &RunOptions::default())
        .unwrap_err();
    assert_input(&err, "`s`", |b| matches!(b, BindError::Unknown { .. }));
}

// ---- Out-of-range indices ----

#[test]
fn out_of_range_gather_names_the_node() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [
            { "name": "x", "dtype": "f32", "shape": [2, 3] },
            { "name": "i", "dtype": "i64", "shape": [2, 1] }
        ],
        "outputs": [ { "name": "y", "value": "pick" } ],
        "nodes": [ { "id": "pick", "op": "core.gather", "inputs": { "x": "x", "index": "i" }, "attrs": { "dim": 1 } } ]
    });
    let m =
        Model::without_weights(Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap()).unwrap();
    for bad in [3, -1] {
        let err = call_err(
            &m,
            vec![
                ("x", f32_tensor(&[2, 3], &[0.0; 6], CPU)),
                ("i", i64_tensor(&[2, 1], &[0, bad], CPU)),
            ],
        );
        assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
        assert_eq!(err.node(), Some("pick"));
        assert!(
            matches!(
                &err,
                RuntimeError::Node { source: TensorError::IndexOutOfRange { index, .. }, .. } if *index == bad
            ),
            "{err:?}"
        );
        assert!(err.to_string().contains("`pick`"), "{err}");
    }
}

#[test]
fn out_of_range_embedding_in_a_repeat_names_the_iteration() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "inputs": [
            { "name": "ids", "dtype": "i64", "shape": [2] },
            { "name": "h0", "dtype": "f32", "shape": [2, 3] }
        ],
        "outputs": [ { "name": "y", "value": "layers:h" } ],
        "params": [
            { "name": "layers.0.emb", "dtype": "float", "shape": [4, 3] },
            { "name": "layers.1.emb", "dtype": "float", "shape": [2, 3] }
        ],
        "nodes": [
            { "id": "layers", "op": "repeat", "count": 2, "carry": { "h": "h0" },
              "body": {
                "params_prefix": "layers.{i}.",
                "nodes": [
                    { "id": "emb", "op": "nn.embedding", "inputs": { "ids": "ids" }, "params": { "weight": "emb" } },
                    { "id": "add", "op": "core.add", "inputs": { "a": "h", "b": "emb" } }
                ],
                "carry_out": { "h": "add" }
              } }
        ]
    });
    let unvalidated = polypous_ir::load(&serde_json::to_vec(&doc).unwrap(), &Default::default());
    let plan = Plan::new(&unvalidated.graph.unwrap(), PlanOptions::new(CPU));
    // Body params must have the same shape in every iteration (graph-ir.md §2.6): invalid graph.
    let err = plan.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidGraph, "{err}");
    // With equal shapes, id 3 is out of range in both iterations: the error names iteration 0.
    let mut doc = doc;
    doc["params"][1]["shape"] = json!([4, 3]);
    let plan = Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap();
    let file = weight_file(&[f32w("layers.0.emb", &[4, 3]), f32w("layers.1.emb", &[4, 3])]);
    let m = Model::load(plan, &file).unwrap();
    let ok = run(
        &m,
        vec![
            ("ids", i64_tensor(&[2], &[3, 0], CPU)),
            ("h0", f32_tensor(&[2, 3], &[0.0; 6], CPU)),
        ],
        vec![],
    );
    assert_eq!(ok.output("y").unwrap().shape(), [2, 3]);
    let err = call_err(
        &m,
        vec![
            ("ids", i64_tensor(&[2], &[4, 0], CPU)),
            ("h0", f32_tensor(&[2, 3], &[0.0; 6], CPU)),
        ],
    );
    assert_eq!(err.kind(), ErrorKind::InvalidInput, "{err}");
    assert_eq!(err.node(), Some("layers[0]/emb"), "{err}");
    assert!(err.to_string().contains("layers[0]/emb"), "{err}");
}

// ---- Cancellation ----

#[test]
fn cancelled_before_the_first_node() {
    let m = linear_model();
    let token = CancellationToken::new();
    token.cancel();
    let options = RunOptions {
        cancel: Some(&token),
        trace: false,
    };
    let err = m
        .run(
            vec![("x".into(), f32_tensor(&[1, 2], &[1.0, 2.0], CPU))],
            vec![],
            &options,
        )
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Cancelled);
    assert_eq!(err.node(), Some("fc"));
    assert!(err.to_string().contains("`fc`"), "{err}");
}

/// A long chain: `repeat` 65,536 times around a body of 8 nodes. Cancelled from another thread
/// while it runs, the call stops between nodes (not before the first one) with no outputs.
#[test]
fn cancelled_between_nodes() {
    let body: Vec<Value> = (0..8)
        .map(|k| {
            let input = if k == 0 {
                "h".to_owned()
            } else {
                format!("n{}", k - 1)
            };
            json!({ "id": format!("n{k}"), "op": "core.add", "inputs": { "a": input, "b": "one" } })
        })
        .collect();
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [4] }, { "name": "one", "dtype": "f32", "shape": [1] } ],
        "outputs": [ { "name": "y", "value": "loop:h" } ],
        "nodes": [
            { "id": "first", "op": "core.neg", "inputs": { "x": "x" } },
            { "id": "loop", "op": "repeat", "count": 65536, "carry": { "h": "first" },
              "body": { "nodes": body, "carry_out": { "h": "n7" } } }
        ]
    });
    let m =
        Model::without_weights(Plan::new(&graph(&doc), PlanOptions::new(CPU)).unwrap()).unwrap();
    let token = CancellationToken::new();
    let canceller = token.clone();
    let started = Instant::now();
    let err = std::thread::scope(|s| {
        let handle = s.spawn(|| {
            let options = RunOptions {
                cancel: Some(&token),
                trace: false,
            };
            m.run(
                vec![
                    ("x".into(), f32_tensor(&[4], &[0.0; 4], CPU)),
                    ("one".into(), f32_tensor(&[1], &[1.0], CPU)),
                ],
                vec![],
                &options,
            )
        });
        std::thread::sleep(Duration::from_millis(50));
        canceller.cancel();
        handle.join().unwrap()
    })
    .expect_err("the call should be cancelled before 524,288 nodes have run");
    assert_eq!(err.kind(), ErrorKind::Cancelled, "{err}");
    let before = err.node().unwrap().to_owned();
    assert!(before.starts_with("loop["), "stopped before {before}");
    assert!(before != "first", "{before}");
    assert!(started.elapsed() < Duration::from_secs(60));
}

// ---- Graph and device ----

#[test]
fn invalid_graph() {
    let mut doc = linear_graph();
    doc["nodes"][0]["inputs"]["x"] = json!("nope");
    let g = polypous_ir::load(&serde_json::to_vec(&doc).unwrap(), &Default::default())
        .graph
        .unwrap();
    let err = Plan::new(&g, PlanOptions::new(CPU)).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidGraph);
    assert!(err.to_string().contains("nope"), "{err}");
    // A graph that's valid for f32 but not for f16 (an `f32` input feeding `nn.linear`).
    let mut doc = linear_graph();
    doc["inputs"][0]["dtype"] = json!("f32");
    let g = graph(&doc);
    let options = PlanOptions {
        device: CPU,
        compute_dtype: polypous_ir::ComputeDtype::F16,
    };
    assert_eq!(
        Plan::new(&g, options).unwrap_err().kind(),
        ErrorKind::InvalidGraph
    );
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn device_checks_gpu() {
    use polypous_backend::DynTensor;
    for device in gpu_devices() {
        // A value on another device: an error, never a silent transfer.
        let plan = Plan::new(&graph(&linear_graph()), PlanOptions::new(device)).unwrap();
        let file = weight_file(&[f32w("fc.weight", &[3, 2]), f32w("fc.bias", &[3])]);
        let m = Model::load(plan, &file).unwrap();
        let err = call_err(&m, vec![("x", f32_tensor(&[1, 2], &[1.0, 2.0], CPU))]);
        assert!(matches!(err, RuntimeError::WrongDevice { .. }), "{err:?}");
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        assert!(err.to_string().contains("`x`"), "{err}");

        if DynTensor::device_supports(device, DType::BF16).unwrap() {
            eprintln!("SKIP device_checks_gpu bf16 on {device}: the device supports bf16");
            continue;
        }
        // bf16 compute dtype on a device without bf16: before anything runs.
        let options = PlanOptions {
            device,
            compute_dtype: polypous_ir::ComputeDtype::Bf16,
        };
        let err = Plan::new(&graph(&linear_graph()), options).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnsupportedOperation, "{err}");
        assert!(
            err.to_string().contains("compute dtype") && err.to_string().contains("bf16"),
            "{err}"
        );
        assert!(err.to_string().contains(&device.to_string()), "{err}");
        // An f32 graph that casts to bf16 inside a repeat: names the node, op, dtype and device.
        let doc = json!({
            "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
            "inputs": [ { "name": "x", "dtype": "f32", "shape": [2] } ],
            "outputs": [ { "name": "y", "value": "r:h" } ],
            "nodes": [ { "id": "r", "op": "repeat", "count": 2, "carry": { "h": "x" },
                "body": { "nodes": [
                    { "id": "down", "op": "core.cast", "inputs": { "x": "h" }, "attrs": { "dtype": "bf16" } },
                    { "id": "up", "op": "core.cast", "inputs": { "x": "down" }, "attrs": { "dtype": "f32" } }
                ], "carry_out": { "h": "up" } } } ]
        });
        let err = Plan::new(&graph(&doc), PlanOptions::new(device)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::UnsupportedOperation, "{err}");
        assert_eq!(err.node(), Some("r/down"), "{err}");
        let text = err.to_string();
        for part in ["r/down", "core.cast", "bf16", &device.to_string()] {
            assert!(text.contains(part), "`{part}` missing in: {text}");
        }
        eprintln!("SUMMARY device_checks_gpu {device}: passed");
    }
}
