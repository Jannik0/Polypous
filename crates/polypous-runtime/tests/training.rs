//! Training mode and params given by the caller (`docs/specs/runtime.md` §3.1–§3.3): the checks
//! and their order, dropout masks keyed by seed, step and node path (graph-ir.md §3.1.12), the
//! returned running statistics, and gradients through given params. The PyTorch comparisons of
//! whole graphs are in `training_reference.rs`.
//!
//! Each test runs on the CPU; `training_gpu` runs the device-independent ones on every GPU device
//! (`#[ignore = "gpu"]`, lavapipe with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use std::collections::HashMap;

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor};
use polypous_ir::{ComputeDtype, TensorDesc, rng};
use polypous_runtime::{
    ErrorKind, Model, ParamProblem, Params, Plan, PlanOptions, RunMode, RunOptions, RunOutput,
    RuntimeError,
};
use serde_json::{Value, json};
use support::*;

const CPU: DeviceId = DeviceId::Cpu;
const TRAIN: RunMode = RunMode::Training { seed: 42, step: 7 };

fn plan(doc: &Value, device: DeviceId) -> Plan {
    Plan::new(&graph(doc), PlanOptions::new(device)).unwrap()
}

fn own(v: Vec<(&str, DynTensor)>) -> Vec<(String, DynTensor)> {
    v.into_iter().map(|(n, t)| (n.to_owned(), t)).collect()
}

/// Params with deterministic values for every declared param of `doc`.
fn params_for(doc: &Value, device: DeviceId) -> Params {
    let mut out = HashMap::new();
    for (k, p) in doc["params"].as_array().unwrap().iter().enumerate() {
        let shape: Vec<usize> = p["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        let n: usize = shape.iter().product();
        let mut v = values(n, 77 + k as u64);
        if p["name"].as_str().unwrap().ends_with("running_var") {
            v.iter_mut().for_each(|x| *x = x.abs() + 0.5);
        }
        out.insert(
            p["name"].as_str().unwrap().to_owned(),
            f32_tensor(&shape, &v, device),
        );
    }
    out
}

fn run_plan(
    plan: &Plan,
    inputs: Vec<(&str, DynTensor)>,
    params: &Params,
    mode: RunMode,
) -> Result<RunOutput, RuntimeError> {
    plan.run(own(inputs), vec![], params, mode, &RunOptions::default())
}

// ---- Graphs ----

/// `x [batch, 4]` → `fc1` (linear 4→3) → `act` (relu) → `fc2` (linear 3→2), plus an unused param.
fn mlp_graph() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "fc2" } ],
        "params": [
            { "name": "fc1.weight", "dtype": "float", "shape": [3, 4] },
            { "name": "fc1.bias", "dtype": "float", "shape": [3], "trainable": false },
            { "name": "fc2.weight", "dtype": "float", "shape": [2, 3] },
            { "name": "fc2.bias", "dtype": "float", "shape": [2] },
            { "name": "spare", "dtype": "float", "shape": [5] }
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

/// Two `batch_norm` nodes on `x [batch, 3, len]`, with their own running statistics.
fn bn_graph() -> Value {
    let bn = |id: &str, x: &str| {
        json!({ "id": id, "op": "nn.batch_norm", "inputs": { "x": x }, "attrs": { "momentum": 0.25 },
                "params": { "running_mean": format!("{id}.running_mean"),
                            "running_var": format!("{id}.running_var") } })
    };
    let stats = |id: &str| {
        vec![
            json!({ "name": format!("{id}.running_mean"), "dtype": "float", "shape": [3] }),
            json!({ "name": format!("{id}.running_var"), "dtype": "float", "shape": [3] }),
        ]
    };
    let params = [stats("bn_b"), stats("bn_a")].concat();
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "dims": { "batch": {}, "len": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 3, "len"] } ],
        "outputs": [ { "name": "y", "value": "bn_b" } ],
        "params": params,
        // Document order bn_b, bn_a; plan order bn_a, bn_b (bn_b reads bn_a).
        "nodes": [ bn("bn_b", "bn_a"), bn("bn_a", "x") ]
    })
}

/// A repeat `r` (`count`) whose body drops its carry: `h ← dropout(h)`.
fn dropout_repeat(count: u64, p: f64) -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [4, 8] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "nodes": [ { "id": "r", "op": "repeat", "count": count, "carry": { "h": "x" }, "body": {
            "nodes": [ { "id": "drop", "op": "nn.dropout", "inputs": { "x": "h" }, "attrs": { "p": p } } ],
            "carry_out": { "h": "drop" } } } ]
    })
}

/// `outer` (count 2) around `inner` (count 2) around `drop`.
fn nested_dropout() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [4, 8] } ],
        "outputs": [ { "name": "y", "value": "outer:h" } ],
        "nodes": [ { "id": "outer", "op": "repeat", "count": 2, "carry": { "h": "x" }, "body": {
            "nodes": [ { "id": "inner", "op": "repeat", "count": 2, "carry": { "g": "h" }, "body": {
                "nodes": [ { "id": "drop", "op": "nn.dropout", "inputs": { "x": "g" }, "attrs": { "p": 0.3 } } ],
                "carry_out": { "g": "drop" } } } ],
            "carry_out": { "h": "inner:g" } } } ]
    })
}

/// The expected result of dropping `x` once per path, in order (graph-ir.md §3.3): `x · m` per
/// drop, with `m = s` kept and `0` dropped, in f32.
fn expected_drops(
    x: &[f32],
    shape: &[usize],
    p: f64,
    seed: u64,
    step: u64,
    paths: &[&str],
) -> Vec<f32> {
    #[allow(clippy::cast_possible_truncation)]
    let s = 1.0f32 / ((1.0 - p) as f32);
    let mut out = x.to_vec();
    for path in paths {
        let mask = rng::dropout_mask(seed, step, path, shape, p).unwrap();
        for (v, keep) in out.iter_mut().zip(mask) {
            *v *= if keep { s } else { 0.0 };
        }
    }
    out
}

fn ones(n: usize) -> Vec<f32> {
    vec![1.0; n]
}

// ---- Inference mode with given params ----

fn inference_is_unchanged_with_given_params(device: DeviceId) {
    let doc = mlp_graph();
    let params = params_for(&doc, device);
    let names: Vec<String> = params.keys().cloned().collect();
    let host: Vec<(String, Vec<usize>, Vec<f32>)> = names
        .iter()
        .map(|n| (n.clone(), params[n].shape(), read_f32(&params[n])))
        .collect();
    let tensors: Vec<(
        &str,
        polypous_formats::safetensors::Dtype,
        Vec<usize>,
        Vec<u8>,
    )> = host
        .iter()
        .map(|(n, s, v)| {
            (
                n.as_str(),
                polypous_formats::safetensors::Dtype::F32,
                s.clone(),
                f32_bytes(v),
            )
        })
        .collect();
    let model = Model::load(plan(&doc, device), &weight_file(&tensors)).unwrap();
    let x = || f32_tensor(&[2, 4], &values(8, 5), device);
    let loaded = run(&model, vec![("x", x())], vec![]);
    let given = model
        .run_with(
            own(vec![("x", x())]),
            vec![],
            Some(&params),
            RunMode::Inference,
            &RunOptions::default(),
        )
        .unwrap();
    let via_plan = run_plan(
        &plan(&doc, device),
        vec![("x", x())],
        &params,
        RunMode::Inference,
    )
    .unwrap();
    let loaded_with = model
        .run_with(
            own(vec![("x", x())]),
            vec![],
            None,
            RunMode::Inference,
            &RunOptions::default(),
        )
        .unwrap();
    for (what, out) in [
        ("given", &given),
        ("plan", &via_plan),
        ("run_with(None)", &loaded_with),
    ] {
        assert_same(out.output("y").unwrap(), loaded.output("y").unwrap(), what);
        assert!(out.running_stats.is_empty(), "{what}");
    }
    assert!(loaded.running_stats.is_empty());
    // The given tensors are unchanged and untracked.
    for (n, _, v) in &host {
        assert_eq!(read_f32(&params[n]), *v, "{n}");
        assert!(!params[n].has_autodiff(), "{n}");
    }
}

#[test]
fn inference_is_unchanged_with_given_params_cpu() {
    inference_is_unchanged_with_given_params(CPU);
}

#[test]
fn inference_mode_ignores_dropout_and_returns_no_running_stats() {
    let out = run_plan(
        &plan(&dropout_repeat(3, 0.5), CPU),
        vec![("x", f32_tensor(&[4, 8], &values(32, 1), CPU))],
        &Params::new(),
        RunMode::Inference,
    )
    .unwrap();
    assert_eq!(read_f32(out.output("y").unwrap()), values(32, 1));
    let doc = bn_graph();
    let out = run_plan(
        &plan(&doc, CPU),
        vec![("x", f32_tensor(&[1, 3, 2], &values(6, 2), CPU))],
        &params_for(&doc, CPU),
        RunMode::Inference,
    )
    .unwrap();
    assert!(out.running_stats.is_empty());
}

// ---- Given params: errors (§3.2) ----

fn param_error(plan: &Plan, params: &Params) -> (String, ParamProblem) {
    let x = f32_tensor(&[2, 4], &values(8, 5), plan.device());
    match run_plan(plan, vec![("x", x)], params, TRAIN) {
        Err(e @ RuntimeError::Param { .. }) => {
            assert_eq!(e.kind(), ErrorKind::InvalidInput);
            let RuntimeError::Param { name, problem } = e else {
                unreachable!()
            };
            (name, problem)
        }
        other => panic!("expected a param error, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn given_params_are_checked_in_order() {
    let doc = mlp_graph();
    let p = plan(&doc, CPU);
    let good = params_for(&doc, CPU);
    // Missing, unused ones included; the first in declaration order.
    let mut m = good.clone();
    m.remove("spare");
    m.remove("fc2.bias");
    m.insert("zzz".into(), f32_tensor(&[1], &[0.0], CPU));
    assert_eq!(
        param_error(&p, &m),
        ("fc2.bias".into(), ParamProblem::Missing)
    );
    // Extra: the first name in sorted order, before any dtype or shape problem.
    let mut m = good.clone();
    m.insert("zzz".into(), f32_tensor(&[1], &[0.0], CPU));
    m.insert("aaa".into(), f32_tensor(&[1], &[0.0], CPU));
    m.insert(
        "fc1.weight".into(),
        f32_tensor(&[4, 3], &values(12, 1), CPU),
    );
    assert_eq!(param_error(&p, &m), ("aaa".into(), ParamProblem::Extra));
    // Dtype.
    let mut m = good.clone();
    m.insert(
        "fc2.weight".into(),
        tensor(DType::F16, &[2, 3], vec![0; 12], CPU),
    );
    let (name, problem) = param_error(&p, &m);
    assert_eq!(name, "fc2.weight");
    assert_eq!(
        problem,
        ParamProblem::Dtype {
            expected: "f32".into(),
            actual: "f16".into()
        }
    );
    // Shape, also of an unused param; declaration order decides.
    let mut m = good.clone();
    m.insert("spare".into(), f32_tensor(&[6], &values(6, 1), CPU));
    m.insert("fc1.bias".into(), f32_tensor(&[1, 3], &values(3, 1), CPU));
    let (name, problem) = param_error(&p, &m);
    assert_eq!(name, "fc1.bias");
    assert_eq!(
        problem,
        ParamProblem::Shape {
            expected: vec![3],
            actual: vec![1, 3]
        }
    );
    let msg = RuntimeError::Param { name, problem }.to_string();
    assert!(
        msg.contains("fc1.bias") && msg.contains("[1, 3]") && msg.contains("[3]"),
        "{msg}"
    );
    // Binding errors come first (§3 step 1 before step 3).
    let err = p
        .run(vec![], vec![], &m, TRAIN, &RunOptions::default())
        .unwrap_err();
    assert!(matches!(err, RuntimeError::Bind(_)), "{err}");
}

#[test]
fn a_plan_without_params_runs_with_an_empty_map() {
    let doc = dropout_repeat(1, 0.0);
    let x = f32_tensor(&[4, 8], &values(32, 1), CPU);
    let out = run_plan(&plan(&doc, CPU), vec![("x", x)], &Params::new(), TRAIN).unwrap();
    // p = 0 is the identity, also in training mode.
    assert_eq!(read_f32(out.output("y").unwrap()), values(32, 1));
    // A graph with params needs them.
    let doc = mlp_graph();
    let (name, problem) = param_error(&plan(&doc, CPU), &Params::new());
    assert_eq!(
        (name.as_str(), problem),
        ("fc1.weight", ParamProblem::Missing)
    );
}

// ---- Training mode: errors (§3.1) ----

fn train_error(doc: &Value, compute: ComputeDtype, x_shape: &[usize]) -> RuntimeError {
    let mut options = PlanOptions::new(CPU);
    options.compute_dtype = compute;
    let p = Plan::new(&graph(doc), options).unwrap();
    let n: usize = x_shape.iter().product();
    let to = |t: DynTensor| {
        if compute == ComputeDtype::F32 {
            t
        } else {
            t.cast(match compute {
                ComputeDtype::F16 => DType::F16,
                _ => DType::BF16,
            })
            .unwrap()
        }
    };
    let mut params = params_for(doc, CPU);
    for t in params.values_mut() {
        *t = to(t.clone());
    }
    let x = to(f32_tensor(x_shape, &values(n, 3), CPU));
    let err = run_plan(&p, vec![("x", x.clone())], &params, TRAIN).unwrap_err();
    // An inference-mode call makes none of these checks.
    run_plan(&p, vec![("x", x)], &params, RunMode::Inference).unwrap();
    err
}

/// `bn_graph` with `bn_a` also using `bn_b`'s running statistics in `role`.
fn shared_bn(role: &str) -> Value {
    let mut doc = bn_graph();
    doc["nodes"][1]["params"][role] = json!(format!("bn_b.{role}"));
    doc["params"]
        .as_array_mut()
        .unwrap()
        .retain(|p| p["name"] != format!("bn_a.{role}"));
    doc
}

#[test]
fn training_needs_the_compute_dtype_f32() {
    let err = train_error(&bn_graph(), ComputeDtype::F16, &[2, 3, 2]);
    assert!(matches!(err, RuntimeError::TrainingDtype { .. }), "{err}");
    assert_eq!(err.kind(), ErrorKind::UnsupportedOperation);
    assert!(err.to_string().contains("f16"), "{err}");
    // Check 1 comes before checks 2 and 3.
    let err = train_error(&shared_bn("running_mean"), ComputeDtype::Bf16, &[1, 3, 1]);
    assert!(matches!(err, RuntimeError::TrainingDtype { .. }), "{err}");
}

#[test]
fn a_running_statistic_needs_exactly_one_use() {
    // Shared between two batch_norm nodes.
    let err = train_error(&shared_bn("running_var"), ComputeDtype::F32, &[2, 3, 2]);
    assert_eq!(err.kind(), ErrorKind::UnsupportedOperation);
    let msg = err.to_string();
    assert!(
        msg.contains("`bn_b.running_var`")
            && msg.contains("node `bn_b` (nn.batch_norm) as `running_var`")
            && msg.contains("node `bn_a` (nn.batch_norm) as `running_var`"),
        "{msg}"
    );
    // Check 2 comes before check 3 (one value per channel).
    let err = train_error(&shared_bn("running_mean"), ComputeDtype::F32, &[1, 3, 1]);
    assert!(
        matches!(err, RuntimeError::SharedRunningStat { .. }),
        "{err}"
    );
    // Used in both roles of one node.
    let mut doc = bn_graph();
    doc["nodes"][1]["params"]["running_var"] = json!("bn_a.running_mean");
    doc["params"]
        .as_array_mut()
        .unwrap()
        .retain(|p| p["name"] != "bn_a.running_var");
    let msg = train_error(&doc, ComputeDtype::F32, &[2, 3, 2]).to_string();
    assert!(
        msg.contains("as `running_mean`")
            && msg.contains("node `bn_a` (nn.batch_norm) as `running_var`"),
        "{msg}"
    );
    // Every iteration of a repeat without `{i}`.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 3] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "params": [ { "name": "bn.running_mean", "dtype": "float", "shape": [3] },
                    { "name": "bn.running_var", "dtype": "float", "shape": [3] } ],
        "nodes": [ { "id": "r", "op": "repeat", "count": 3, "carry": { "h": "x" }, "body": {
            "params_prefix": "bn.",
            "nodes": [ { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "h" },
                         "params": { "running_mean": "running_mean", "running_var": "running_var" } } ],
            "carry_out": { "h": "bn" } } } ]
    });
    let msg = train_error(&doc, ComputeDtype::F32, &[2, 3]).to_string();
    assert!(
        msg.contains("`bn.running_mean` has 3 uses")
            && msg.contains("node `r[0]/bn`")
            && msg.contains("node `r[1]/bn`"),
        "{msg}"
    );
    // With `{i}` every iteration has its own: valid.
    let mut doc = doc;
    doc["nodes"][0]["body"]["params_prefix"] = json!("bn{i}.");
    let mut params = Vec::new();
    for i in 0..3 {
        for s in ["running_mean", "running_var"] {
            params.push(json!({ "name": format!("bn{i}.{s}"), "dtype": "float", "shape": [3] }));
        }
    }
    doc["params"] = json!(params);
    let p = plan(&doc, CPU);
    let out = run_plan(
        &p,
        vec![("x", f32_tensor(&[2, 3], &values(6, 1), CPU))],
        &params_for(&doc, CPU),
        TRAIN,
    )
    .unwrap();
    let names: Vec<&str> = out.running_stats.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "bn0.running_mean",
            "bn0.running_var",
            "bn1.running_mean",
            "bn1.running_var",
            "bn2.running_mean",
            "bn2.running_var"
        ]
    );
}

#[test]
fn a_running_statistic_used_by_another_op_is_shared() {
    // `bn.running_mean` [3] is also the bias of a linear 4→3.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 4] } ],
        "outputs": [ { "name": "y", "value": "bn" } ],
        "params": [ { "name": "w", "dtype": "float", "shape": [3, 4] },
                    { "name": "bn.running_mean", "dtype": "float", "shape": [3] },
                    { "name": "bn.running_var", "dtype": "float", "shape": [3] } ],
        "nodes": [
            { "id": "fc", "op": "nn.linear", "inputs": { "x": "x" }, "params": { "weight": "w", "bias": "bn.running_mean" } },
            { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "fc" },
              "params": { "running_mean": "bn.running_mean", "running_var": "bn.running_var" } }
        ]
    });
    let msg = train_error(&doc, ComputeDtype::F32, &[2, 4]).to_string();
    assert!(
        msg.contains("node `fc` (nn.linear) as `bias`")
            && msg.contains("node `bn` (nn.batch_norm) as `running_mean`"),
        "{msg}"
    );
}

#[test]
fn batch_norm_needs_two_values_per_channel() {
    let err = train_error(&bn_graph(), ComputeDtype::F32, &[1, 3, 1]);
    assert_eq!(err.kind(), ErrorKind::InvalidInput);
    // Plan order: bn_a runs first.
    assert_eq!(err.node(), Some("bn_a"));
    assert!(err.to_string().contains("[1, 3, 1] has 1"), "{err}");
    // [1, 3, 2] has 2 values per channel: valid.
    let doc = bn_graph();
    run_plan(
        &plan(&doc, CPU),
        vec![("x", f32_tensor(&[1, 3, 2], &values(6, 1), CPU))],
        &params_for(&doc, CPU),
        TRAIN,
    )
    .unwrap();
    // Inside a repeat the path names iteration 0.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [1, 3] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "params": [ { "name": "l0.m", "dtype": "float", "shape": [3] }, { "name": "l0.v", "dtype": "float", "shape": [3] },
                    { "name": "l1.m", "dtype": "float", "shape": [3] }, { "name": "l1.v", "dtype": "float", "shape": [3] } ],
        "nodes": [ { "id": "r", "op": "repeat", "count": 2, "carry": { "h": "x" }, "body": {
            "params_prefix": "l{i}.",
            "nodes": [ { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "h" },
                         "params": { "running_mean": "m", "running_var": "v" } } ],
            "carry_out": { "h": "bn" } } } ]
    });
    let err = train_error(&doc, ComputeDtype::F32, &[1, 3]);
    assert!(matches!(err, RuntimeError::BatchNormValues { .. }), "{err}");
    assert_eq!(err.node(), Some("r[0]/bn"));
}

#[test]
fn given_param_errors_come_before_training_errors() {
    let doc = shared_bn("running_mean");
    let p = plan(&doc, CPU);
    let mut params = params_for(&doc, CPU);
    params.remove("bn_a.running_var");
    let err = run_plan(
        &p,
        vec![("x", f32_tensor(&[1, 3, 1], &values(3, 1), CPU))],
        &params,
        TRAIN,
    )
    .unwrap_err();
    assert!(matches!(err, RuntimeError::Param { .. }), "{err}");
}

// ---- Dropout masks ----

fn dropout_masks_follow_seed_step_and_node_path(device: DeviceId) {
    let x = values(32, 9);
    let shape = [4, 8];
    let run_drop = |doc: &Value, mode| {
        let out = run_plan(
            &plan(doc, device),
            vec![("x", f32_tensor(&shape, &x, device))],
            &Params::new(),
            mode,
        )
        .unwrap();
        read_f32(out.output("y").unwrap())
    };
    // A repeat draws a different mask in each iteration: `r[0]/drop`, `r[1]/drop`, `r[2]/drop`.
    let got = run_drop(&dropout_repeat(3, 0.5), TRAIN);
    let expected = expected_drops(
        &x,
        &shape,
        0.5,
        42,
        7,
        &["r[0]/drop", "r[1]/drop", "r[2]/drop"],
    );
    assert_eq!(got, expected, "repeat on {device}");
    let m0 = rng::dropout_mask(42, 7, "r[0]/drop", &shape, 0.5).unwrap();
    let m1 = rng::dropout_mask(42, 7, "r[1]/drop", &shape, 0.5).unwrap();
    assert_ne!(m0, m1);
    // Nested repeats: outer[o]/inner[i]/drop, iterations in order.
    let got = run_drop(&nested_dropout(), TRAIN);
    let paths = [
        "outer[0]/inner[0]/drop",
        "outer[0]/inner[1]/drop",
        "outer[1]/inner[0]/drop",
        "outer[1]/inner[1]/drop",
    ];
    assert_eq!(
        got,
        expected_drops(&x, &shape, 0.3, 42, 7, &paths),
        "nested on {device}"
    );
    // Same seed and step: identical. Another step or seed: different.
    let doc = dropout_repeat(1, 0.5);
    let a = run_drop(&doc, TRAIN);
    assert_eq!(a, run_drop(&doc, TRAIN));
    assert_eq!(a, expected_drops(&x, &shape, 0.5, 42, 7, &["r[0]/drop"]));
    let other_step = run_drop(&doc, RunMode::Training { seed: 42, step: 8 });
    assert_ne!(a, other_step);
    assert_eq!(
        other_step,
        expected_drops(&x, &shape, 0.5, 42, 8, &["r[0]/drop"])
    );
    assert_ne!(a, run_drop(&doc, RunMode::Training { seed: 43, step: 7 }));
    // Kept elements are x · s exactly, dropped ones 0.
    let s = 2.0f32;
    for (v, xv) in a.iter().zip(&x) {
        assert!(*v == 0.0 || *v == xv * s);
    }
    // The ones vector shows the mask itself: m · s.
    let n = ones(32);
    let out = run_plan(
        &plan(&doc, device),
        vec![("x", f32_tensor(&shape, &n, device))],
        &Params::new(),
        TRAIN,
    )
    .unwrap();
    let mask = rng::dropout_mask(42, 7, "r[0]/drop", &shape, 0.5).unwrap();
    let want: Vec<f32> = mask.iter().map(|&k| if k { s } else { 0.0 }).collect();
    assert_eq!(read_f32(out.output("y").unwrap()), want);
}

#[test]
fn dropout_masks_follow_seed_step_and_node_path_cpu() {
    dropout_masks_follow_seed_step_and_node_path(CPU);
}

// ---- Running statistics (§3.3) ----

fn running_stats_are_returned_in_plan_order(device: DeviceId) {
    let doc = bn_graph();
    let params = params_for(&doc, device);
    let x = f32_tensor(&[4, 3, 2], &values(24, 4), device)
        .require_grad()
        .unwrap();
    let out = run_plan(&plan(&doc, device), vec![("x", x.clone())], &params, TRAIN).unwrap();
    let names: Vec<&str> = out.running_stats.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "bn_a.running_mean",
            "bn_a.running_var",
            "bn_b.running_mean",
            "bn_b.running_var"
        ]
    );
    for (name, t) in &out.running_stats {
        assert_eq!(t.shape(), [3], "{name}");
        assert_eq!(t.dtype(), DType::F32, "{name}");
        assert_eq!(t.device(), device, "{name}");
        assert!(!t.has_autodiff(), "{name} carries autodiff context");
    }
    // The output is tracked through x; the params are unchanged.
    let y = out.output("y").unwrap();
    assert!(y.is_tracked());
    for (n, t) in &params {
        assert!(!t.has_autodiff(), "{n}");
    }
    // bn_a's new running mean: (1 - 0.25) · old + 0.25 · batch mean, per channel.
    let xv = read_f32(&x);
    let old = read_f32(&params["bn_a.running_mean"]);
    let new = read_f32(out.running_stat("bn_a.running_mean").unwrap());
    for c in 0..3 {
        let mut sum = 0.0f64;
        for n in 0..4 {
            for l in 0..2 {
                sum += f64::from(xv[n * 6 + c * 2 + l]);
            }
        }
        let want = 0.75 * f64::from(old[c]) + 0.25 * sum / 8.0;
        assert!(
            (f64::from(new[c]) - want).abs() < 1e-6,
            "channel {c}: {} vs {want}",
            new[c]
        );
    }
}

#[test]
fn running_stats_are_returned_in_plan_order_cpu() {
    running_stats_are_returned_in_plan_order(CPU);
}

// ---- Gradients through given params ----

fn gradients_flow_through_given_params(device: DeviceId) {
    let doc = mlp_graph();
    let mut params = params_for(&doc, device);
    // Mark the trainable params as leaves (as the trainer does); `fc1.bias` is
    // `trainable: false` and stays untracked.
    let p = plan(&doc, device);
    let trainable: Vec<String> = p
        .analysis()
        .trainable_params()
        .map(|i| i.name.clone())
        .collect();
    assert_eq!(trainable, ["fc1.weight", "fc2.weight", "fc2.bias", "spare"]);
    for name in &trainable {
        let t = params.remove(name).unwrap();
        params.insert(name.clone(), t.require_grad().unwrap());
    }
    let x = f32_tensor(&[2, 4], &values(8, 5), device);
    let out = run_plan(&p, vec![("x", x)], &params, TRAIN).unwrap();
    let y = out.output("y").unwrap();
    assert!(y.is_tracked());
    let g = f32_tensor(&[2, 2], &values(4, 6), device);
    let loss = y.clone().mul(g).unwrap().sum(None, false).unwrap();
    let grads = loss.backward().unwrap();
    for name in ["fc1.weight", "fc2.weight", "fc2.bias"] {
        let grad = grads.get(&params[name]).unwrap();
        assert_eq!(
            grad.map(|t| t.shape()),
            Some(params[name].shape()),
            "{name}"
        );
    }
    // Unused: no gradient. `trainable: false`, given untracked: none either.
    assert!(grads.get(&params["spare"]).unwrap().is_none());
    assert!(!params["fc1.bias"].is_tracked());
    assert!(
        grads.get(&params["fc1.bias"]).is_err(),
        "not a leaf, so it has no gradient"
    );
}

#[test]
fn gradients_flow_through_given_params_cpu() {
    gradients_flow_through_given_params(CPU);
}

// ---- GPU ----

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn training_gpu() {
    for device in gpu_devices() {
        inference_is_unchanged_with_given_params(device);
        dropout_masks_follow_seed_step_and_node_path(device);
        running_stats_are_returned_in_plan_order(device);
        gradients_flow_through_given_params(device);
        // A given param on another device (§3.2).
        let doc = mlp_graph();
        let mut params = params_for(&doc, device);
        params.insert("fc2.bias".into(), f32_tensor(&[2], &[0.0, 1.0], CPU));
        let (name, problem) = param_error(&plan(&doc, device), &params);
        assert_eq!(name, "fc2.bias");
        assert!(matches!(problem, ParamProblem::Device { .. }), "{problem}");
        eprintln!(
            "SUMMARY training {device}: given params, dropout masks, running stats, gradients passed"
        );
    }
}

/// The public check (`Plan::check_training_mode`, runtime.md §3.1) on the bound shapes alone,
/// next to a training-mode `Plan::run` with tensors of those shapes: the same result.
fn check_both(
    doc: &Value,
    compute: ComputeDtype,
    x_shape: &[usize],
) -> (Result<(), String>, Result<(), String>) {
    let mut options = PlanOptions::new(CPU);
    options.compute_dtype = compute;
    let p = Plan::new(&graph(doc), options).unwrap();
    let dtype = p.analysis().inputs[0].info.dtype;
    let bindings = p
        .analysis()
        .bind(
            &[(
                "x",
                TensorDesc {
                    dtype,
                    shape: x_shape,
                },
            )],
            &[],
        )
        .unwrap();
    let checked = p
        .check_training_mode(&bindings)
        .map_err(|e| format!("{e:?}"));
    let ran = if compute == ComputeDtype::F32 {
        let n: usize = x_shape.iter().product();
        run_plan(
            &p,
            vec![("x", f32_tensor(x_shape, &values(n, 3), CPU))],
            &params_for(doc, CPU),
            TRAIN,
        )
        .map(|_| ())
        .map_err(|e| format!("{e:?}"))
    } else {
        Err(format!("{:?}", train_error(doc, compute, x_shape)))
    };
    (checked, ran)
}

#[test]
fn the_public_check_and_a_training_call_agree() {
    // A repeat whose body's batch_norm has its own statistics per iteration (`{i}`).
    let mut params = Vec::new();
    for i in 0..3 {
        for s in ["running_mean", "running_var"] {
            params.push(json!({ "name": format!("bn{i}.{s}"), "dtype": "float", "shape": [3] }));
        }
    }
    let repeat = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 3] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "params": params,
        "nodes": [ { "id": "r", "op": "repeat", "count": 3, "carry": { "h": "x" }, "body": {
            "params_prefix": "bn{i}.",
            "nodes": [ { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "h" },
                         "params": { "running_mean": "running_mean", "running_var": "running_var" } } ],
            "carry_out": { "h": "bn" } } } ]
    });
    let failures = [
        (
            bn_graph(),
            ComputeDtype::F16,
            vec![2, 3, 2],
            "TrainingDtype",
        ),
        (
            bn_graph(),
            ComputeDtype::Bf16,
            vec![1, 3, 1],
            "TrainingDtype",
        ),
        (
            shared_bn("running_var"),
            ComputeDtype::F32,
            vec![2, 3, 2],
            "SharedRunningStat",
        ),
        (
            shared_bn("running_mean"),
            ComputeDtype::F32,
            vec![1, 3, 1],
            "SharedRunningStat",
        ),
        (
            bn_graph(),
            ComputeDtype::F32,
            vec![1, 3, 1],
            "BatchNormValues",
        ),
        (
            repeat.clone(),
            ComputeDtype::F32,
            vec![1, 3],
            "BatchNormValues",
        ),
    ];
    for (doc, compute, shape, kind) in failures {
        let (checked, ran) = check_both(&doc, compute, &shape);
        let checked = checked.expect_err("the check fails");
        assert!(checked.starts_with(kind), "{checked}");
        assert_eq!(Err(checked), ran, "{kind} {shape:?}");
    }
    // The repeat's node path is the runtime's `r[0]/bn`.
    let (checked, _) = check_both(&repeat, ComputeDtype::F32, &[1, 3]);
    assert!(checked.unwrap_err().contains("node: \"r[0]/bn\""));
    // Valid graphs pass both.
    for (doc, shape) in [
        (bn_graph(), vec![2, 3, 2]),
        (repeat, vec![2, 3]),
        (mlp_graph(), vec![2, 4]),
    ] {
        assert_eq!(
            check_both(&doc, ComputeDtype::F32, &shape),
            (Ok(()), Ok(())),
            "{shape:?}"
        );
    }
    // Bindings of another plan don't fit.
    let p = Plan::new(&graph(&bn_graph()), PlanOptions::new(CPU)).unwrap();
    let other = Plan::new(&graph(&mlp_graph()), PlanOptions::new(CPU)).unwrap();
    let dtype = other.analysis().inputs[0].info.dtype;
    let bindings = other
        .analysis()
        .bind(
            &[(
                "x",
                TensorDesc {
                    dtype,
                    shape: &[2, 4],
                },
            )],
            &[],
        )
        .unwrap();
    let err = p.check_training_mode(&bindings).unwrap_err();
    assert!(matches!(err, RuntimeError::Internal { .. }), "{err}");
}
