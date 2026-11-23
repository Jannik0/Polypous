//! Param uses, fans and trainable params (`docs/specs/graph-ir.md` §2.4, §2.4.1; training.md
//! §1.1), the `invalid_init` checks of §2.4.1, and the `invalid_value` checks of `init` values
//! (§2.4).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_ir::{Analysis, ErrorCode, Fans, Graph, ValidateOptions, load};
use serde_json::{Value, json};

fn analyze(doc: &Value) -> Analysis {
    let graph = Graph::from_json(doc.to_string().as_bytes()).unwrap_or_else(|r| panic!("{r}"));
    graph.analyze(&ValidateOptions::default()).unwrap()
}

/// (code, pointer, message) of every diagnostic.
fn errors(doc: &Value) -> Vec<(String, String, String)> {
    load(doc.to_string().as_bytes(), &ValidateOptions::default())
        .report
        .diagnostics
        .iter()
        .map(|d| {
            (
                d.code.as_str().to_owned(),
                d.pointer.clone(),
                d.message.clone(),
            )
        })
        .collect()
}

/// The message of the only diagnostic, which must have `code` at `pointer`.
#[track_caller]
fn only(doc: &Value, code: ErrorCode, pointer: &str) -> String {
    let errs = errors(doc);
    assert_eq!(errs.len(), 1, "{errs:?}");
    assert_eq!(
        (errs[0].0.as_str(), errs[0].1.as_str()),
        (code.as_str(), pointer),
        "{errs:?}"
    );
    errs[0].2.clone()
}

fn param(name: &str, shape: Value, init: Value) -> Value {
    json!({ "name": name, "dtype": "float", "shape": shape, "init": init })
}

/// `x [batch, 4]` → `fc` (`nn.linear` with `fc.weight [3, 4]`, `fc.bias [3]`).
fn linear(weight_init: Value, bias_init: Value) -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "fc" } ],
        "params": [ param("fc.weight", json!([3, 4]), weight_init), param("fc.bias", json!([3]), bias_init) ],
        "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "x" },
                     "params": { "weight": "fc.weight", "bias": "fc.bias" } } ]
    })
}

fn kaiming() -> Value {
    json!({ "kind": "kaiming_uniform", "a": 2.2360679 })
}

/// A repeat `r` (count 2, `params_prefix` as given) around one `nn.linear` `fc` on `[batch, 4]`.
fn repeated(prefix: &str, params: Value) -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "params": params,
        "nodes": [ { "id": "r", "op": "repeat", "count": 2, "carry": { "h": "x" }, "body": {
            "params_prefix": prefix,
            "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "h" }, "params": { "weight": "w" } } ],
            "carry_out": { "h": "fc" } } } ]
    })
}

#[test]
fn each_fan_role_gives_its_fans() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "inputs": [ { "name": "ids", "dtype": "i64", "shape": [2, 3] },
                    { "name": "x1", "dtype": "float", "shape": [1, 4, 9] },
                    { "name": "x2", "dtype": "float", "shape": [1, 4, 9, 9] } ],
        "outputs": [ { "name": "e", "value": "emb" }, { "name": "c1", "value": "conv1" },
                     { "name": "c2", "value": "conv2" }, { "name": "l", "value": "fc" } ],
        "params": [
            param("emb.weight", json!([10, 4]), json!({ "kind": "normal", "mean": 0, "std": 1 })),
            param("conv1.weight", json!([6, 2, 3]), json!({ "kind": "xavier_uniform" })),
            param("conv2.weight", json!([8, 2, 3, 5]), json!({ "kind": "kaiming_normal" })),
            param("conv2.bias", json!([8]), json!({ "kind": "uniform", "low": -1, "high": 1 })),
            param("fc.weight", json!([5, 4]), json!({ "kind": "xavier_normal" }))
        ],
        "nodes": [
            { "id": "emb", "op": "nn.embedding", "inputs": { "ids": "ids" }, "params": { "weight": "emb.weight" } },
            { "id": "conv1", "op": "nn.conv1d", "inputs": { "x": "x1" }, "params": { "weight": "conv1.weight" },
              "attrs": { "groups": 2 } },
            { "id": "conv2", "op": "nn.conv2d", "inputs": { "x": "x2" },
              "params": { "weight": "conv2.weight", "bias": "conv2.bias" }, "attrs": { "groups": 2 } },
            { "id": "fc", "op": "nn.linear", "inputs": { "x": "emb" }, "params": { "weight": "fc.weight" } }
        ]
    });
    let a = analyze(&doc);
    let f = |fan_in, fan_out| Some(Fans { fan_in, fan_out });
    // nn.embedding [num, dim]: fan_in = dim, fan_out = num.
    assert_eq!(a.param_fans("emb.weight"), f(4, 10));
    // conv1d [C_out, C_in/groups, k]: C_in/groups · k, C_out · k (not divided by groups).
    assert_eq!(a.param_fans("conv1.weight"), f(6, 18));
    // conv2d [C_out, C_in/groups, kh, kw].
    assert_eq!(a.param_fans("conv2.weight"), f(30, 120));
    assert_eq!(a.param_fans("fc.weight"), f(4, 5));
    // Roles without fans.
    assert_eq!(a.param_fans("conv2.bias"), None);
    assert_eq!(a.param_fans("missing"), None);
    let names: Vec<&str> = a.trainable_params().map(|p| p.name.as_str()).collect();
    assert_eq!(names.len(), 5);
    let conv2 = a.param("conv2.weight").unwrap();
    assert_eq!(conv2.shape, [8, 2, 3, 5]);
    assert_eq!(conv2.uses, 1);
    assert_eq!(
        conv2.first_uses[0].to_string(),
        "node `conv2` (nn.conv2d) as `weight`"
    );
}

#[test]
fn an_unused_param_with_a_fan_based_init_is_invalid() {
    for kind in [
        "kaiming_uniform",
        "kaiming_normal",
        "xavier_uniform",
        "xavier_normal",
    ] {
        let mut doc = linear(json!({ "kind": "zeros" }), json!({ "kind": "zeros" }));
        doc["params"].as_array_mut().unwrap().push(param(
            "spare",
            json!([3, 4]),
            json!({ "kind": kind }),
        ));
        let msg = only(&doc, ErrorCode::InvalidInit, "/params/2/init");
        assert!(
            msg.contains("`spare`") && msg.contains(kind) && msg.contains("no node uses it"),
            "{msg}"
        );
    }
    // The kinds without fans work on unused params.
    for init in [
        json!({ "kind": "zeros" }),
        json!({ "kind": "ones" }),
        json!({ "kind": "constant", "value": 2 }),
        json!({ "kind": "uniform", "low": 0, "high": 1 }),
        json!({ "kind": "normal", "mean": 0, "std": 1 }),
    ] {
        let mut doc = linear(json!({ "kind": "zeros" }), json!({ "kind": "zeros" }));
        doc["params"]
            .as_array_mut()
            .unwrap()
            .push(param("spare", json!([3, 4]), init));
        let a = analyze(&doc);
        let spare = a.param("spare").unwrap();
        assert_eq!((spare.uses, spare.fans), (0, None));
        // Unused params are still trainable (they just get no gradient).
        assert!(spare.is_trainable());
    }
}

#[test]
fn a_fan_based_init_in_a_role_without_fans_is_invalid() {
    let msg = only(
        &linear(kaiming(), kaiming()),
        ErrorCode::InvalidInit,
        "/params/1/init",
    );
    assert!(
        msg.contains("`fc.bias`")
            && msg.contains("node `fc` (nn.linear) as `bias`")
            && msg.contains("role without fans"),
        "{msg}"
    );
    // A multi-dimensional layer_norm weight has no fans either.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 3, 4] } ],
        "outputs": [ { "name": "y", "value": "ln" } ],
        "params": [ param("ln.weight", json!([3, 4]), json!({ "kind": "xavier_normal" })) ],
        "nodes": [ { "id": "ln", "op": "nn.layer_norm", "inputs": { "x": "x" }, "params": { "weight": "ln.weight" } } ]
    });
    let msg = only(&doc, ErrorCode::InvalidInit, "/params/0/init");
    assert!(
        msg.contains("node `ln` (nn.layer_norm) as `weight`"),
        "{msg}"
    );
    // Inside a repeat the node path names the iteration.
    let doc = repeated(
        "",
        json!([
            param("w", json!([4, 4]), json!({ "kind": "zeros" })),
            param("b", json!([4]), kaiming())
        ]),
    );
    let mut doc = doc;
    doc["nodes"][0]["body"]["nodes"][0]["params"]["bias"] = json!("b");
    let msg = only(&doc, ErrorCode::InvalidInit, "/params/1/init");
    assert!(
        msg.contains("node `r[0]/fc` (nn.linear) as `bias`"),
        "{msg}"
    );
}

#[test]
fn weight_tying_counts_every_use() {
    let mut doc = linear(kaiming(), json!({ "kind": "zeros" }));
    doc["inputs"][0]["shape"] = json!(["batch", 3]);
    doc["params"][0]["shape"] = json!([3, 3]);
    doc["nodes"].as_array_mut().unwrap().push(json!(
        { "id": "fc2", "op": "nn.linear", "inputs": { "x": "fc" }, "params": { "weight": "fc.weight" } }
    ));
    doc["outputs"][0]["value"] = json!("fc2");
    let a = analyze(&doc);
    let w = a.param("fc.weight").unwrap();
    assert_eq!(w.uses, 2);
    let nodes: Vec<&str> = w.first_uses.iter().map(|u| u.node.as_str()).collect();
    assert_eq!(nodes, ["fc", "fc2"]);
    assert_eq!(
        w.fans,
        Some(Fans {
            fan_in: 3,
            fan_out: 3
        })
    );
}

#[test]
fn repeats_expand_params_with_and_without_i() {
    // With `{i}`: one param per iteration, one use each.
    let doc = repeated(
        "layers.{i}.",
        json!([
            param("layers.0.w", json!([4, 4]), kaiming()),
            param(
                "layers.1.w",
                json!([4, 4]),
                json!({ "kind": "xavier_uniform" })
            )
        ]),
    );
    let a = analyze(&doc);
    for (i, name) in ["layers.0.w", "layers.1.w"].iter().enumerate() {
        let p = a.param(name).unwrap();
        assert_eq!(p.uses, 1, "{name}");
        assert_eq!(p.first_uses[0].node, format!("r[{i}]/fc"));
        assert_eq!(
            p.fans,
            Some(Fans {
                fan_in: 4,
                fan_out: 4
            })
        );
    }
    // An iteration's param that no iteration exists for is unused.
    let doc = repeated(
        "layers.{i}.",
        json!([
            param("layers.0.w", json!([4, 4]), kaiming()),
            param("layers.1.w", json!([4, 4]), kaiming()),
            param("layers.2.w", json!([4, 4]), kaiming())
        ]),
    );
    let msg = only(&doc, ErrorCode::InvalidInit, "/params/2/init");
    assert!(
        msg.contains("`layers.2.w`") && msg.contains("no node uses it"),
        "{msg}"
    );
    // Without `{i}`: every iteration uses the same param.
    let doc = repeated(
        "shared.",
        json!([param("shared.w", json!([4, 4]), kaiming())]),
    );
    let a = analyze(&doc);
    let p = a.param("shared.w").unwrap();
    assert_eq!(p.uses, 2);
    let nodes: Vec<&str> = p.first_uses.iter().map(|u| u.node.as_str()).collect();
    assert_eq!(nodes, ["r[0]/fc", "r[1]/fc"]);
    assert_eq!(
        p.fans,
        Some(Fans {
            fan_in: 4,
            fan_out: 4
        })
    );
}

#[test]
fn nested_repeats_give_runtime_node_paths() {
    // outer (count 2, `o{i}.`) around inner (count 3, no `{i}`) around fc.
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 4] } ],
        "outputs": [ { "name": "y", "value": "outer:h" } ],
        "params": [ param("o0.w", json!([4, 4]), kaiming()), param("o1.w", json!([4, 4]), kaiming()) ],
        "nodes": [ { "id": "outer", "op": "repeat", "count": 2, "carry": { "h": "x" }, "body": {
            "params_prefix": "o{i}.",
            "nodes": [ { "id": "inner", "op": "repeat", "count": 3, "carry": { "g": "h" }, "body": {
                "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "g" }, "params": { "weight": "w" } } ],
                "carry_out": { "g": "fc" } } } ],
            "carry_out": { "h": "inner:g" } } } ]
    });
    let a = analyze(&doc);
    let p = a.param("o1.w").unwrap();
    assert_eq!(p.uses, 3);
    let nodes: Vec<&str> = p.first_uses.iter().map(|u| u.node.as_str()).collect();
    assert_eq!(nodes, ["outer[1]/inner[0]/fc", "outer[1]/inner[1]/fc"]);
}

#[test]
fn trainable_params_exclude_running_statistics() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 3] } ],
        "outputs": [ { "name": "y", "value": "bn" } ],
        "params": [
            { "name": "bn.running_mean", "dtype": "float", "shape": [3], "trainable": true },
            { "name": "bn.running_var", "dtype": "float", "shape": [3] },
            { "name": "bn.weight", "dtype": "float", "shape": [3], "trainable": false },
            { "name": "bn.bias", "dtype": "float", "shape": [3] }
        ],
        "nodes": [ { "id": "bn", "op": "nn.batch_norm", "inputs": { "x": "x" },
                     "params": { "running_mean": "bn.running_mean", "running_var": "bn.running_var",
                                 "weight": "bn.weight", "bias": "bn.bias" } } ]
    });
    let a = analyze(&doc);
    let names: Vec<&str> = a.trainable_params().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["bn.bias"]);
    let mean = a.param("bn.running_mean").unwrap();
    assert!(mean.running_stat && mean.trainable_written && !mean.is_trainable());
    assert!(!a.param("bn.weight").unwrap().is_trainable());
}

#[test]
fn a_cycle_doesnt_report_its_params_as_unused() {
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "inputs": [ { "name": "x", "dtype": "float", "shape": [2, 4] } ],
        "outputs": [ { "name": "y", "value": "b" } ],
        "params": [ param("w", json!([4, 4]), kaiming()) ],
        "nodes": [
            { "id": "a", "op": "nn.linear", "inputs": { "x": "b" }, "params": { "weight": "w" } },
            { "id": "b", "op": "core.add", "inputs": { "a": "a", "b": "x" } }
        ]
    });
    let errs = errors(&doc);
    assert!(errs.iter().any(|e| e.0 == "cycle"), "{errs:?}");
    assert!(!errs.iter().any(|e| e.0 == "invalid_init"), "{errs:?}");
}

#[test]
fn init_values_must_be_usable() {
    let bias = |init: Value| linear(json!({ "kind": "zeros" }), init);
    let cases = [
        (
            json!({ "kind": "uniform", "low": 1, "high": 0.5 }),
            "/params/1/init/high",
            "less than `low`",
        ),
        (
            json!({ "kind": "normal", "mean": 0, "std": -0.1 }),
            "/params/1/init/std",
            "negative",
        ),
        (
            json!({ "kind": "constant", "value": 1e39 }),
            "/params/1/init/value",
            "finite f32",
        ),
        (
            json!({ "kind": "uniform", "low": -1e39, "high": 0 }),
            "/params/1/init/low",
            "finite f32",
        ),
        (
            json!({ "kind": "uniform", "low": 0, "high": 3.5e38 }),
            "/params/1/init/high",
            "finite f32",
        ),
        (
            json!({ "kind": "normal", "mean": -4e38, "std": 1 }),
            "/params/1/init/mean",
            "finite f32",
        ),
        (
            json!({ "kind": "normal", "mean": 0, "std": 1e300 }),
            "/params/1/init/std",
            "finite f32",
        ),
    ];
    for (init, pointer, text) in cases {
        let msg = only(&bias(init.clone()), ErrorCode::InvalidValue, pointer);
        assert!(msg.contains(text), "{init}: {msg}");
    }
    // Allowed: low = high, std = 0, the largest finite f32, and any finite `a`.
    for init in [
        json!({ "kind": "uniform", "low": 0.5, "high": 0.5 }),
        json!({ "kind": "normal", "mean": 1, "std": 0 }),
        json!({ "kind": "constant", "value": 3.4028234663852886e38 }),
        json!({ "kind": "constant", "value": -0.0 }),
    ] {
        analyze(&bias(init));
    }
    analyze(&linear(
        json!({ "kind": "kaiming_uniform", "a": 1e300 }),
        json!({ "kind": "zeros" }),
    ));
}
