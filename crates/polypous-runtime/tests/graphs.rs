//! Multi-node graphs through the interpreter (runtime.md §1, §3): an MLP and a CNN at several
//! batch sizes from one plan, `repeat` against the same graph unrolled by hand (carry,
//! `params_prefix` with and without `{i}`, body state, nesting, `count = 1`), explicit state, and
//! value lifetimes.
//!
//! References are the same ops composed directly on `DynTensor` in the same order, so results
//! must match bit for bit. Each test runs on the CPU; the `_gpu` variants run on every GPU device
//! (`#[ignore = "gpu"]`, lavapipe with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`).
//!
//! ```text
//! cargo test -p polypous-runtime --test graphs
//! cargo test -p polypous-runtime --features vulkan --test graphs -- --include-ignored
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use polypous_backend::device::DeviceId;
use polypous_runtime::RunOptions;
use serde_json::{Value, json};
use support::*;

// ---- MLP ----

fn mlp_graph() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "probs", "value": "probs" } ],
        "params": [
            { "name": "fc1.weight", "dtype": "float", "shape": [8, 4] },
            { "name": "fc1.bias", "dtype": "float", "shape": [8] },
            { "name": "fc2.weight", "dtype": "float", "shape": [3, 8] },
            { "name": "fc2.bias", "dtype": "float", "shape": [3] }
        ],
        "nodes": [
            { "id": "fc1", "op": "nn.linear", "inputs": { "x": "x" },
              "params": { "weight": "fc1.weight", "bias": "fc1.bias" } },
            { "id": "act1", "op": "nn.relu", "inputs": { "x": "fc1" } },
            { "id": "fc2", "op": "nn.linear", "inputs": { "x": "act1" },
              "params": { "weight": "fc2.weight", "bias": "fc2.bias" } },
            { "id": "probs", "op": "core.softmax", "inputs": { "x": "fc2" }, "attrs": { "dim": -1 } }
        ]
    })
}

fn mlp(device: DeviceId) {
    let (weights, w) = f32_weights(&[
        ("fc1.weight", &[8, 4]),
        ("fc1.bias", &[8]),
        ("fc2.weight", &[3, 8]),
        ("fc2.bias", &[3]),
    ]);
    let model = model(&graph(&mlp_graph()), device, &weights);
    let p = |name: &str, shape: &[usize]| f32_tensor(shape, &w[name], device);
    for batch in [1, 3, 7] {
        let xv = values(batch * 4, batch as u64);
        let out = run(
            &model,
            vec![("x", f32_tensor(&[batch, 4], &xv, device))],
            vec![],
        );
        let x = f32_tensor(&[batch, 4], &xv, device);
        let expected = x
            .linear(p("fc1.weight", &[8, 4]), Some(p("fc1.bias", &[8])))
            .unwrap()
            .relu()
            .unwrap()
            .linear(p("fc2.weight", &[3, 8]), Some(p("fc2.bias", &[3])))
            .unwrap()
            .softmax(1)
            .unwrap();
        assert_same(
            out.output("probs").unwrap(),
            &expected,
            &format!("mlp batch {batch}"),
        );
    }
}

// ---- CNN ----

fn cnn_graph() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "image", "dtype": "float", "shape": ["batch", 1, 6, 6] } ],
        "outputs": [ { "name": "logits", "value": "fc" } ],
        "params": [
            { "name": "conv.weight", "dtype": "float", "shape": [2, 1, 3, 3] },
            { "name": "conv.bias", "dtype": "float", "shape": [2] },
            { "name": "bn.running_mean", "dtype": "float", "shape": [2] },
            { "name": "bn.running_var", "dtype": "float", "shape": [2] },
            { "name": "bn.weight", "dtype": "float", "shape": [2] },
            { "name": "bn.bias", "dtype": "float", "shape": [2] },
            { "name": "fc.weight", "dtype": "float", "shape": [3, 18] },
            { "name": "fc.bias", "dtype": "float", "shape": [3] }
        ],
        "nodes": [
            { "id": "conv", "op": "nn.conv2d", "inputs": { "x": "image" },
              "params": { "weight": "conv.weight", "bias": "conv.bias" },
              "attrs": { "padding": [1, 1] } },
            { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "conv" },
              "params": { "running_mean": "bn.running_mean", "running_var": "bn.running_var",
                          "weight": "bn.weight", "bias": "bn.bias" } },
            { "id": "act", "op": "nn.relu", "inputs": { "x": "bn" } },
            { "id": "pool", "op": "nn.max_pool2d", "inputs": { "x": "act" },
              "attrs": { "kernel_size": [2, 2] } },
            { "id": "flat", "op": "core.flatten", "inputs": { "x": "pool" }, "attrs": { "start_dim": 1 } },
            { "id": "fc", "op": "nn.linear", "inputs": { "x": "flat" },
              "params": { "weight": "fc.weight", "bias": "fc.bias" } }
        ]
    })
}

fn cnn(device: DeviceId) {
    let (weights, mut w) = f32_weights(&[
        ("conv.weight", &[2, 1, 3, 3]),
        ("conv.bias", &[2]),
        ("bn.running_mean", &[2]),
        ("bn.running_var", &[2]),
        ("bn.weight", &[2]),
        ("bn.bias", &[2]),
        ("fc.weight", &[3, 18]),
        ("fc.bias", &[3]),
    ]);
    // A variance must be positive: rewrite the file with |v| + 0.5.
    let var: Vec<f32> = w["bn.running_var"].iter().map(|v| v.abs() + 0.5).collect();
    w.insert("bn.running_var".into(), var);
    let shapes: [(&str, Vec<usize>); 8] = [
        ("conv.weight", vec![2, 1, 3, 3]),
        ("conv.bias", vec![2]),
        ("bn.running_mean", vec![2]),
        ("bn.running_var", vec![2]),
        ("bn.weight", vec![2]),
        ("bn.bias", vec![2]),
        ("fc.weight", vec![3, 18]),
        ("fc.bias", vec![3]),
    ];
    drop(weights);
    let tensors: Vec<_> = shapes
        .iter()
        .map(|(n, s)| {
            (
                *n,
                polypous_formats::safetensors::Dtype::F32,
                s.clone(),
                f32_bytes(&w[*n]),
            )
        })
        .collect();
    let weights = weight_file(&tensors);
    let model = model(&graph(&cnn_graph()), device, &weights);
    let p = |name: &str| {
        let shape = &shapes.iter().find(|(n, _)| *n == name).unwrap().1;
        f32_tensor(shape, &w[name], device)
    };
    for batch in [1, 3, 7] {
        let xv = values(batch * 36, 50 + batch as u64);
        let out = run(
            &model,
            vec![("image", f32_tensor(&[batch, 1, 6, 6], &xv, device))],
            vec![],
        );
        let expected = f32_tensor(&[batch, 1, 6, 6], &xv, device)
            .conv2d(
                p("conv.weight"),
                Some(p("conv.bias")),
                [1, 1],
                [1, 1],
                [1, 1],
                1,
            )
            .unwrap()
            .batch_norm(
                p("bn.running_mean"),
                p("bn.running_var"),
                Some(p("bn.weight")),
                Some(p("bn.bias")),
                1e-5,
            )
            .unwrap()
            .relu()
            .unwrap()
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1])
            .unwrap()
            .flatten(1, 3)
            .unwrap()
            .linear(p("fc.weight"), Some(p("fc.bias")))
            .unwrap();
        assert_eq!(expected.shape(), vec![batch, 3]);
        assert_same(
            out.output("logits").unwrap(),
            &expected,
            &format!("cnn batch {batch}"),
        );
    }
}

// ---- repeat vs unrolled ----

/// How the repeat graphs name their params.
#[derive(Clone, Copy)]
enum Params {
    /// `params_prefix: "layers.{i}."`: one set per iteration.
    PerLayer,
    /// `params_prefix: "shared."`: every iteration uses the same params.
    Shared,
}

fn param_name(params: Params, i: u64, local: &str) -> String {
    match params {
        Params::PerLayer => format!("layers.{i}.{local}"),
        Params::Shared => format!("shared.{local}"),
    }
}

fn param_decls(count: u64, params: Params) -> (Vec<Value>, Vec<(String, Vec<usize>)>) {
    let mut names = Vec::new();
    let n = match params {
        Params::PerLayer => count,
        Params::Shared => 1,
    };
    for i in 0..n {
        names.push((param_name(params, i, "w"), vec![4, 4]));
        names.push((param_name(params, i, "b"), vec![4]));
    }
    let decls = names
        .iter()
        .map(|(n, s)| json!({ "name": n, "dtype": "float", "shape": s }))
        .collect();
    (decls, names)
}

/// `pre → repeat(count) { lin, act, res = h + act, mix = res + x (enclosing value),
/// upd = acc + mix (body state) } → head`.
fn repeat_graph(count: u64, params: Params) -> Value {
    let prefix = match params {
        Params::PerLayer => "layers.{i}.",
        Params::Shared => "shared.",
    };
    let (decls, _) = param_decls(count, params);
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "head" }, { "name": "h", "value": "layers:h" } ],
        "params": decls,
        "nodes": [
            { "id": "pre", "op": "nn.relu", "inputs": { "x": "x" } },
            { "id": "layers", "op": "repeat", "count": count, "carry": { "h": "pre" },
              "body": {
                "params_prefix": prefix, "state_prefix": "layers.{i}.",
                "state": [ { "name": "acc", "dtype": "float", "shape": ["batch", 4] } ],
                "nodes": [
                    { "id": "lin", "op": "nn.linear", "inputs": { "x": "h" },
                      "params": { "weight": "w", "bias": "b" } },
                    { "id": "act", "op": "core.tanh", "inputs": { "x": "lin" } },
                    { "id": "res", "op": "core.add", "inputs": { "a": "h", "b": "act" } },
                    { "id": "mix", "op": "core.add", "inputs": { "a": "res", "b": "x" } },
                    { "id": "upd", "op": "core.add", "inputs": { "a": "acc", "b": "mix" } }
                ],
                "state_updates": { "acc": "upd" },
                "carry_out": { "h": "mix" }
              } },
            { "id": "head", "op": "core.sum", "inputs": { "x": "layers:h" }, "attrs": { "dim": [1] } }
        ]
    })
}

/// The same graph with the body written out `count` times.
fn unrolled_graph(count: u64, params: Params) -> Value {
    let (decls, _) = param_decls(count, params);
    let mut nodes = vec![json!({ "id": "pre", "op": "nn.relu", "inputs": { "x": "x" } })];
    let mut state = Vec::new();
    let mut updates = serde_json::Map::new();
    let mut h = "pre".to_owned();
    for i in 0..count {
        let id = |n: &str| format!("l{i}_{n}");
        let acc = format!("layers.{i}.acc");
        state.push(json!({ "name": acc, "dtype": "float", "shape": ["batch", 4] }));
        nodes.push(json!({ "id": id("lin"), "op": "nn.linear", "inputs": { "x": h },
            "params": { "weight": param_name(params, i, "w"), "bias": param_name(params, i, "b") } }));
        nodes.push(json!({ "id": id("act"), "op": "core.tanh", "inputs": { "x": id("lin") } }));
        nodes.push(
            json!({ "id": id("res"), "op": "core.add", "inputs": { "a": h, "b": id("act") } }),
        );
        nodes.push(
            json!({ "id": id("mix"), "op": "core.add", "inputs": { "a": id("res"), "b": "x" } }),
        );
        nodes.push(
            json!({ "id": id("upd"), "op": "core.add", "inputs": { "a": acc, "b": id("mix") } }),
        );
        updates.insert(acc, json!(id("upd")));
        h = id("mix");
    }
    nodes.push(
        json!({ "id": "head", "op": "core.sum", "inputs": { "x": h }, "attrs": { "dim": [1] } }),
    );
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "head" }, { "name": "h", "value": h } ],
        "params": decls, "state": state, "nodes": nodes, "state_updates": updates
    })
}

fn repeat_matches_unrolled(device: DeviceId, count: u64, params: Params) {
    let (_, names) = param_decls(count, params);
    let shapes: Vec<(&str, &[usize])> = names
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let (weights, _) = f32_weights(&shapes);
    let repeated = model(&graph(&repeat_graph(count, params)), device, &weights);
    let unrolled = model(&graph(&unrolled_graph(count, params)), device, &weights);
    for batch in [1, 3] {
        let xv = values(batch * 4, 7 + batch as u64);
        let state: Vec<(String, Vec<f32>)> = (0..count)
            .map(|i| (format!("layers.{i}.acc"), values(batch * 4, 100 + i)))
            .collect();
        let call = |m: &polypous_runtime::Model| {
            let st = state
                .iter()
                .map(|(n, v)| (n.as_str(), f32_tensor(&[batch, 4], v, device)))
                .collect();
            run(m, vec![("x", f32_tensor(&[batch, 4], &xv, device))], st)
        };
        let (a, b) = (call(&repeated), call(&unrolled));
        for name in ["y", "h"] {
            assert_same(
                a.output(name).unwrap(),
                b.output(name).unwrap(),
                &format!("output {name}, count {count}, batch {batch}"),
            );
        }
        assert_eq!(a.state.len(), count as usize);
        for ((na, ta), (nb, tb)) in a.state.iter().zip(&b.state) {
            assert_eq!(na, nb);
            assert_same(ta, tb, &format!("state {na}"));
        }
    }
}

/// Two levels: `blocks` (count 2, `blocks.{i}.`) around `layers` (count 2, `layers.{i}.`, with
/// body state), against the hand-unrolled graph.
fn nested(device: DeviceId) {
    let mut decls = Vec::new();
    let mut shapes = Vec::new();
    for b in 0..2 {
        for l in 0..2 {
            for (p, s) in [("w", vec![4usize, 4]), ("b", vec![4])] {
                let name = format!("blocks.{b}.layers.{l}.{p}");
                decls.push(json!({ "name": name, "dtype": "float", "shape": s }));
                shapes.push((name, s));
            }
        }
    }
    let nested = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "blocks:h" } ],
        "params": decls,
        "nodes": [
            { "id": "blocks", "op": "repeat", "count": 2, "carry": { "h": "x" },
              "body": {
                "params_prefix": "blocks.{i}.", "state_prefix": "blocks.{i}.",
                "nodes": [
                    { "id": "layers", "op": "repeat", "count": 2, "carry": { "g": "h" },
                      "body": {
                        "params_prefix": "layers.{i}.", "state_prefix": "layers.{i}.",
                        "state": [ { "name": "s", "dtype": "float", "shape": ["batch", 4] } ],
                        "nodes": [
                            { "id": "lin", "op": "nn.linear", "inputs": { "x": "g" },
                              "params": { "weight": "w", "bias": "b" } },
                            { "id": "res", "op": "core.add", "inputs": { "a": "g", "b": "lin" } },
                            { "id": "upd", "op": "core.add", "inputs": { "a": "s", "b": "lin" } }
                        ],
                        "state_updates": { "s": "upd" },
                        "carry_out": { "g": "res" }
                      } },
                    { "id": "norm", "op": "core.tanh", "inputs": { "x": "layers:g" } }
                ],
                "carry_out": { "h": "norm" }
              } }
        ]
    });
    let mut nodes = Vec::new();
    let mut state = Vec::new();
    let mut updates = serde_json::Map::new();
    let mut h = "x".to_owned();
    for b in 0..2 {
        for l in 0..2 {
            let id = |n: &str| format!("b{b}l{l}_{n}");
            let s = format!("blocks.{b}.layers.{l}.s");
            state.push(json!({ "name": s, "dtype": "float", "shape": ["batch", 4] }));
            let p = |n: &str| format!("blocks.{b}.layers.{l}.{n}");
            nodes.push(
                json!({ "id": id("lin"), "op": "nn.linear", "inputs": { "x": h },
                "params": { "weight": p("w"), "bias": p("b") } }),
            );
            nodes.push(
                json!({ "id": id("res"), "op": "core.add", "inputs": { "a": h, "b": id("lin") } }),
            );
            nodes.push(
                json!({ "id": id("upd"), "op": "core.add", "inputs": { "a": s, "b": id("lin") } }),
            );
            updates.insert(s, json!(id("upd")));
            h = id("res");
        }
        let norm = format!("b{b}_norm");
        nodes.push(json!({ "id": norm, "op": "core.tanh", "inputs": { "x": h } }));
        h = norm;
    }
    let unrolled = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": h } ],
        "params": decls, "state": state, "nodes": nodes, "state_updates": updates
    });
    let refs: Vec<(&str, &[usize])> = shapes
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let (weights, _) = f32_weights(&refs);
    let a_model = model(&graph(&nested), device, &weights);
    let b_model = model(&graph(&unrolled), device, &weights);
    let names: Vec<String> = a_model
        .plan()
        .analysis()
        .state
        .iter()
        .map(|s| s.name.clone())
        .collect();
    assert_eq!(
        names,
        [
            "blocks.0.layers.0.s",
            "blocks.0.layers.1.s",
            "blocks.1.layers.0.s",
            "blocks.1.layers.1.s"
        ]
    );
    let xv = values(12, 3);
    let call = |m: &polypous_runtime::Model| {
        let st = names
            .iter()
            .enumerate()
            .map(|(k, n)| {
                (
                    n.as_str(),
                    f32_tensor(&[3, 4], &values(12, 200 + k as u64), device),
                )
            })
            .collect();
        run(m, vec![("x", f32_tensor(&[3, 4], &xv, device))], st)
    };
    let (a, b) = (call(&a_model), call(&b_model));
    assert_same(
        a.output("y").unwrap(),
        b.output("y").unwrap(),
        "nested output",
    );
    for ((na, ta), (nb, tb)) in a.state.iter().zip(&b.state) {
        assert_eq!(na, nb);
        assert_same(ta, tb, &format!("nested state {na}"));
    }
}

// ---- state ----

fn state(device: DeviceId) {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [2] } ],
        "state": [
            { "name": "s", "dtype": "f32", "shape": [2] },
            { "name": "t", "dtype": "float", "shape": [2], "init": "zeros" }
        ],
        "outputs": [ { "name": "later", "value": "later" }, { "name": "t_out", "value": "t" } ],
        "nodes": [
            { "id": "upd", "op": "core.add", "inputs": { "a": "s", "b": "x" } },
            { "id": "later", "op": "core.add", "inputs": { "a": "s", "b": "upd" } }
        ],
        "state_updates": { "s": "upd" }
    });
    let model = polypous_runtime::Model::without_weights(
        polypous_runtime::Plan::new(&graph(&doc), polypous_runtime::PlanOptions::new(device))
            .unwrap(),
    )
    .unwrap();
    let mut s = vec![1.0f32, 2.0];
    for _call in 0..3 {
        let out = run(
            &model,
            vec![("x", f32_tensor(&[2], &[10.0, 20.0], device))],
            vec![
                ("s", f32_tensor(&[2], &s, device)),
                ("t", f32_tensor(&[2], &[5.0, 6.0], device)),
            ],
        );
        let next: Vec<f32> = s.iter().zip([10.0, 20.0]).map(|(a, b)| a + b).collect();
        // `later` reads `s` after the node that computes its update: it sees the start value.
        let later: Vec<f32> = s.iter().zip(&next).map(|(a, b)| a + b).collect();
        assert_eq!(read_f32(out.output("later").unwrap()), later);
        assert_eq!(
            out.state
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            ["s", "t"]
        );
        assert_eq!(read_f32(out.state_entry("s").unwrap()), next);
        // No update: returned unchanged (and an output may name a state entry).
        assert_eq!(read_f32(out.state_entry("t").unwrap()), [5.0, 6.0]);
        assert_eq!(read_f32(out.output("t_out").unwrap()), [5.0, 6.0]);
        s = next;
    }
}

// ---- lifetimes ----

#[test]
fn intermediates_are_dropped_after_their_last_use() {
    let device = DeviceId::Cpu;
    let (weights, _) = f32_weights(&[
        ("fc1.weight", &[8, 4]),
        ("fc1.bias", &[8]),
        ("fc2.weight", &[3, 8]),
        ("fc2.bias", &[3]),
    ]);
    let model = model(&graph(&mlp_graph()), device, &weights);
    let schedule: Vec<(Option<String>, Vec<String>)> = model
        .plan()
        .schedule()
        .into_iter()
        .map(|s| (s.node, s.drops))
        .collect();
    let step = |n: &str, d: &[&str]| {
        (
            Some(n.to_owned()),
            d.iter().map(|s| (*s).to_owned()).collect(),
        )
    };
    assert_eq!(
        schedule,
        [
            step("fc1", &["x"]),
            step("act1", &["fc1"]),
            step("fc2", &["act1"]),
            step("probs", &["fc2"])
        ]
    );
    let out = model
        .run(
            vec![("x".into(), f32_tensor(&[2, 4], &values(8, 1), device))],
            vec![],
            &RunOptions {
                cancel: None,
                trace: true,
            },
        )
        .unwrap();
    let live: Vec<(String, Vec<String>)> =
        out.trace.into_iter().map(|t| (t.node, t.live)).collect();
    let after = |n: &str, l: &[&str]| (n.to_owned(), l.iter().map(|s| (*s).to_owned()).collect());
    assert_eq!(
        live,
        [
            after("fc1", &["fc1"]),
            after("act1", &["act1"]),
            after("fc2", &["fc2"]),
            after("probs", &["probs"])
        ]
    );
}

#[test]
fn body_values_are_dropped_per_iteration() {
    let device = DeviceId::Cpu;
    let (_, names) = param_decls(3, Params::PerLayer);
    let shapes: Vec<(&str, &[usize])> = names
        .iter()
        .map(|(n, s)| (n.as_str(), s.as_slice()))
        .collect();
    let (weights, _) = f32_weights(&shapes);
    let model = model(&graph(&repeat_graph(3, Params::PerLayer)), device, &weights);
    let state = (0..3)
        .map(|i| {
            (
                format!("layers.{i}.acc"),
                f32_tensor(&[2, 4], &values(8, i), device),
            )
        })
        .collect();
    let out = model
        .run(
            vec![("x".into(), f32_tensor(&[2, 4], &values(8, 9), device))],
            state,
            &RunOptions {
                cancel: None,
                trace: true,
            },
        )
        .unwrap();
    let trace: Vec<(String, Vec<String>)> =
        out.trace.into_iter().map(|t| (t.node, t.live)).collect();
    let find = |node: &str| {
        trace
            .iter()
            .find(|(n, _)| n == node)
            .unwrap_or_else(|| panic!("no trace step {node}: {trace:?}"))
            .1
            .clone()
    };
    // `pre` is the carry init and nothing in the body reads it: moved into the carry.
    assert_eq!(find("pre"), ["pre", "x"]);
    for i in 0..3 {
        // `h` is dropped after `res`, its last use; `lin` after `act`, `act` after `res`.
        assert_eq!(
            find(&format!("layers[{i}]/act")),
            ["layers/acc", "layers/act", "layers/h", "x"]
        );
        assert_eq!(
            find(&format!("layers[{i}]/res")),
            ["layers/acc", "layers/res", "x"]
        );
        // At the end of the body only the carry-out and the state update (and `x`, which the body
        // reads in every iteration) are alive.
        assert_eq!(
            find(&format!("layers[{i}]/upd")),
            ["layers/mix", "layers/upd", "x"]
        );
    }
    // After the repeat: its port; `x` is dropped after its last use, the repeat.
    assert_eq!(find("layers"), ["layers:h"]);
    assert_eq!(find("head"), ["head", "layers:h"]);
    assert_eq!(trace.len(), 1 + 3 * 5 + 2);
}

// ---- CPU and GPU entry points ----

#[test]
fn mlp_cpu() {
    mlp(DeviceId::Cpu);
}

#[test]
fn cnn_cpu() {
    cnn(DeviceId::Cpu);
}

#[test]
fn repeat_matches_unrolled_cpu() {
    repeat_matches_unrolled(DeviceId::Cpu, 3, Params::PerLayer);
}

#[test]
fn repeat_with_shared_params_matches_unrolled_cpu() {
    repeat_matches_unrolled(DeviceId::Cpu, 3, Params::Shared);
}

#[test]
fn repeat_count_one_matches_unrolled_cpu() {
    repeat_matches_unrolled(DeviceId::Cpu, 1, Params::PerLayer);
}

#[test]
fn nested_repeat_matches_unrolled_cpu() {
    nested(DeviceId::Cpu);
}

#[test]
fn state_cpu() {
    state(DeviceId::Cpu);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn graphs_gpu() {
    for device in gpu_devices() {
        mlp(device);
        cnn(device);
        repeat_matches_unrolled(device, 3, Params::PerLayer);
        repeat_matches_unrolled(device, 3, Params::Shared);
        repeat_matches_unrolled(device, 1, Params::PerLayer);
        nested(device);
        state(device);
        eprintln!("SUMMARY graphs {device}: mlp, cnn, repeat (3 variants), nested, state passed");
    }
}
