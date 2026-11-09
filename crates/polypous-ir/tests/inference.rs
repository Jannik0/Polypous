//! Whole-graph inference (`docs/specs/graph-ir.md` §2.5, §2.6, §4.1 checks 6–8): repeat bodies,
//! carries, state updates, the [`Analysis`] the runtime gets, and symbol binding (§2.3.4).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_ir::{
    Analysis, BindError, BoundDim, BoundNode, ComputeDtype, Dtype, Graph, NodeInfo, TensorDesc,
    ValidateOptions, ValueKind,
};
use serde_json::{Value, json};

const MLP: &str = include_str!("data/mlp.graph.json");
const REPEAT: &str = include_str!("data/repeat.graph.json");

fn load(doc: &Value) -> Result<Graph, String> {
    Graph::from_json(doc.to_string().as_bytes()).map_err(|r| r.to_string())
}

fn analyze(doc: &Value) -> Analysis {
    let g = load(doc).unwrap_or_else(|e| panic!("{e}"));
    g.analyze(&ValidateOptions::default()).unwrap()
}

/// (code, pointer, node id) of every diagnostic.
fn errors(doc: &Value) -> Vec<(String, String, Option<String>)> {
    polypous_ir::load(doc.to_string().as_bytes(), &ValidateOptions::default())
        .report
        .diagnostics
        .into_iter()
        .map(|d| (d.code.as_str().to_owned(), d.pointer, d.node_id))
        .collect()
}

#[track_caller]
fn assert_only(doc: &Value, code: &str, pointer: &str, node: Option<&str>) {
    assert_eq!(
        errors(doc),
        vec![(code.to_owned(), pointer.to_owned(), node.map(str::to_owned))]
    );
}

fn desc(dtype: Dtype, shape: &[usize]) -> TensorDesc<'_> {
    TensorDesc { dtype, shape }
}

fn port_shape(info: &NodeInfo, port: &str) -> String {
    info.outputs()
        .iter()
        .find(|(p, _)| p == port)
        .map(|(_, v)| format!("{} {}", v.dtype, v.shape_string()))
        .unwrap()
}

// ---- The spec examples ----

#[test]
fn mlp_analysis() {
    let a = analyze(&serde_json::from_str(MLP).unwrap());
    assert_eq!(a.inputs[0].name, "image");
    assert_eq!(a.inputs[0].info.shape_string(), "[batch, 784]");
    assert_eq!(port_shape(&a.nodes[0], "out"), "f32 [batch, 128]");
    assert_eq!(port_shape(&a.nodes[2], "out"), "f32 [batch, 10]");

    let b = a
        .bind(&[("image", desc(Dtype::F32, &[3, 784]))], &[])
        .unwrap();
    assert_eq!(b.symbol("batch"), Some(3));
    let BoundNode::Op(fc2) = &b.nodes[2] else {
        panic!()
    };
    assert_eq!(fc2.outputs, vec![vec![3, 10]]);

    // Compute dtype f16: `float` inputs and params resolve to it.
    let g = load(&serde_json::from_str(MLP).unwrap()).unwrap();
    let a16 = g
        .analyze(&ValidateOptions {
            compute_dtype: ComputeDtype::F16,
        })
        .unwrap();
    assert_eq!(a16.inputs[0].info.dtype, Dtype::F16);
    assert_eq!(port_shape(&a16.nodes[2], "out"), "f16 [batch, 10]");
    assert_eq!(
        a16.bind(&[("image", desc(Dtype::F32, &[3, 784]))], &[]),
        Err(BindError::Dtype {
            kind: ValueKind::Input,
            name: "image".into(),
            expected: Dtype::F16,
            actual: Dtype::F32
        })
    );
}

#[test]
fn repeat_example_analysis() {
    let a = analyze(&serde_json::from_str(REPEAT).unwrap());
    let NodeInfo::Repeat(r) = &a.nodes[0] else {
        panic!()
    };
    assert_eq!(r.count, 2);
    assert_eq!(r.carry[0].0, "h");
    assert_eq!(r.carry[0].1.shape_string(), "[batch, seq, 16]");
    assert_eq!(r.state[0].shape_string(), "[batch, seq, 16]");
    assert_eq!(port_shape(&r.nodes[1], "out"), "f32 [batch, seq, 16]");
    assert_eq!(port_shape(&a.nodes[1], "1"), "f32 [batch, seq, 8]");
    assert_eq!(port_shape(&a.nodes[2], "out"), "f32 [batch, seq, 4]");
    let names: Vec<&str> = a.state.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["step", "layers.0.cache", "layers.1.cache"]);

    let state_shape = [2usize, 5, 16];
    let state = [
        ("step", desc(Dtype::I64, &[1])),
        ("layers.0.cache", desc(Dtype::F32, &state_shape)),
        ("layers.1.cache", desc(Dtype::F32, &state_shape)),
    ];
    let b = a
        .bind(&[("x", desc(Dtype::F32, &[2, 5, 16]))], &state)
        .unwrap();
    assert_eq!(b.symbol("seq"), Some(5));
    let BoundNode::Repeat(rb) = &b.nodes[0] else {
        panic!()
    };
    assert_eq!(rb.carry, vec![vec![2, 5, 16]]);
    // `batch` has max 64.
    let big = [65usize, 5, 16];
    let big_state = [
        ("step", desc(Dtype::I64, &[1])),
        ("layers.0.cache", desc(Dtype::F32, &big)),
        ("layers.1.cache", desc(Dtype::F32, &big)),
    ];
    assert!(matches!(
        a.bind(&[("x", desc(Dtype::F32, &big))], &big_state),
        Err(BindError::SymbolOutOfRange { ref symbol, value: 65, max: Some(64), .. }) if symbol == "batch"
    ));
    // A missing state value.
    assert_eq!(
        a.bind(&[("x", desc(Dtype::F32, &[2, 5, 16]))], &state[..2]),
        Err(BindError::Missing {
            kind: ValueKind::State,
            name: "layers.1.cache".into()
        })
    );
}

// ---- Repeat carries and state updates ----

fn base() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "r:h" } ],
        "nodes": []
    })
}

fn with_body(body_nodes: Value, carry_out: &str) -> Value {
    let mut g = base();
    g["nodes"] = json!([ { "id": "r", "op": "repeat", "count": 3, "carry": { "h": "x" },
        "body": { "nodes": body_nodes, "carry_out": { "h": carry_out } } } ]);
    g
}

#[test]
fn repeat_carry_keeps_dtype_and_shape() {
    let ok = with_body(
        json!([ { "id": "a", "op": "nn.relu", "inputs": { "x": "h" } } ]),
        "a",
    );
    let a = analyze(&ok);
    assert_eq!(port_shape(&a.nodes[0], "h"), "f32 [batch, 4]");

    let grows = with_body(
        json!([ { "id": "c", "op": "core.concat", "inputs": { "xs": ["h", "h"] }, "attrs": { "dim": 1 } } ]),
        "c",
    );
    assert_only(
        &grows,
        "repeat_carry_mismatch",
        "/nodes/0/body/carry_out/h",
        Some("r"),
    );
    let casts = with_body(
        json!([ { "id": "c", "op": "core.cast", "inputs": { "x": "h" }, "attrs": { "dtype": "f16" } } ]),
        "c",
    );
    assert_only(
        &casts,
        "repeat_carry_mismatch",
        "/nodes/0/body/carry_out/h",
        Some("r"),
    );
    // Equal only up to provability: (seq // 2) * 2 is not seq.
    let mut g = base();
    g["inputs"] = json!([ { "name": "x", "dtype": "f32", "shape": ["batch", 4] } ]);
    g["nodes"] = json!([ { "id": "r", "op": "repeat", "count": 1, "carry": { "h": "x" },
        "body": { "nodes": [
            { "id": "s", "op": "core.reshape", "inputs": { "x": "h" }, "attrs": { "shape": ["batch // 2", 2, 4] } },
            { "id": "t", "op": "core.reshape", "inputs": { "x": "s" }, "attrs": { "shape": ["(batch // 2) * 2", 4] } }
        ], "carry_out": { "h": "t" } } } ]);
    let errs = errors(&g);
    assert!(
        errs.iter()
            .any(|(c, _, n)| c == "shape_mismatch" && n.as_deref() == Some("s")),
        "{errs:?}"
    );
}

#[test]
fn body_nodes_errors_name_the_body_node() {
    let g = with_body(
        json!([ { "id": "bad", "op": "core.exp", "inputs": { "x": "h" } },
                { "id": "c", "op": "core.cast", "inputs": { "x": "bad" }, "attrs": { "dtype": "i64" } },
                { "id": "e", "op": "core.exp", "inputs": { "x": "c" } } ]),
        "h",
    );
    assert_only(
        &g,
        "unsupported_dtype",
        "/nodes/0/body/nodes/2/inputs/x",
        Some("e"),
    );
}

#[test]
fn body_params_must_agree_across_iterations() {
    let mut g = with_body(
        json!([ { "id": "fc", "op": "nn.linear", "inputs": { "x": "h" }, "params": { "weight": "w" } } ]),
        "fc",
    );
    g["nodes"][0]["body"]["params_prefix"] = json!("l.{i}.");
    g["params"] = json!([
        { "name": "l.0.w", "dtype": "float", "shape": [4, 4] },
        { "name": "l.1.w", "dtype": "float", "shape": [4, 4] },
        { "name": "l.2.w", "dtype": "float", "shape": [4, 4] }
    ]);
    analyze(&g);
    g["params"][2]["shape"] = json!([8, 4]);
    assert_only(
        &g,
        "shape_mismatch",
        "/nodes/0/body/nodes/0/params/weight",
        Some("fc"),
    );
}

#[test]
fn state_updates_match_their_entry() {
    let mut g = base();
    g["outputs"] = json!([ { "name": "y", "value": "a" } ]);
    g["state"] = json!([ { "name": "s", "dtype": "f32", "shape": ["batch", 4] } ]);
    g["nodes"] = json!([ { "id": "a", "op": "core.add", "inputs": { "a": "x", "b": "s" } } ]);
    g["state_updates"] = json!({ "s": "a" });
    analyze(&g);

    let mut wide = g.clone();
    wide["nodes"] = json!([
        { "id": "a", "op": "core.add", "inputs": { "a": "x", "b": "s" } },
        { "id": "w", "op": "core.concat", "inputs": { "xs": ["a", "a"] }, "attrs": { "dim": 1 } }
    ]);
    wide["state_updates"] = json!({ "s": "w" });
    assert_only(&wide, "state_update_mismatch", "/state_updates/s", None);

    let mut cast = g.clone();
    cast["nodes"] = json!([
        { "id": "a", "op": "core.add", "inputs": { "a": "x", "b": "s" } },
        { "id": "c", "op": "core.cast", "inputs": { "x": "a" }, "attrs": { "dtype": "bf16" } }
    ]);
    cast["state_updates"] = json!({ "s": "c" });
    assert_only(&cast, "state_update_mismatch", "/state_updates/s", None);

    // A body's state update.
    let mut body = with_body(
        json!([ { "id": "a", "op": "core.sum", "inputs": { "x": "h" }, "attrs": { "dim": [1], "keepdim": true } } ]),
        "h",
    );
    body["nodes"][0]["body"]["state_prefix"] = json!("l.{i}.");
    body["nodes"][0]["body"]["state"] =
        json!([ { "name": "acc", "dtype": "float", "shape": ["batch", 1] } ]);
    body["nodes"][0]["body"]["state_updates"] = json!({ "acc": "a" });
    let a = analyze(&body);
    let names: Vec<&str> = a.state.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["l.0.acc", "l.1.acc", "l.2.acc"]);
    body["nodes"][0]["body"]["state_updates"] = json!({ "acc": "h" });
    assert_only(
        &body,
        "state_update_mismatch",
        "/nodes/0/body/state_updates/acc",
        Some("r"),
    );
}

#[test]
fn nodes_are_inferred_in_dependency_order() {
    // `b` uses `a`, which comes later in the document.
    let mut g = base();
    g["outputs"] = json!([ { "name": "y", "value": "b" } ]);
    g["nodes"] = json!([
        { "id": "b", "op": "core.exp", "inputs": { "x": "a" } },
        { "id": "a", "op": "nn.relu", "inputs": { "x": "x" } }
    ]);
    let a = analyze(&g);
    assert_eq!(port_shape(&a.nodes[0], "out"), "f32 [batch, 4]");
}

#[test]
fn errors_dont_cascade() {
    // One bad node; its users are skipped without follow-up errors.
    let mut g = base();
    g["outputs"] = json!([ { "name": "y", "value": "c" } ]);
    g["nodes"] = json!([
        { "id": "a", "op": "core.add", "inputs": { "a": "x", "b": "x" }, "attrs": { "bad": 1 } },
        { "id": "b", "op": "core.exp", "inputs": { "x": "a" } },
        { "id": "c", "op": "core.sum", "inputs": { "x": "b" }, "attrs": { "dim": [7] } }
    ]);
    let errs = errors(&g);
    assert_eq!(
        errs,
        vec![(
            "unknown_attr".to_owned(),
            "/nodes/0/attrs/bad".to_owned(),
            Some("a".to_owned())
        )]
    );
}

#[test]
fn float_attributes_resolve_to_the_compute_dtype() {
    let mut g = base();
    g["outputs"] = json!([ { "name": "y", "value": "c" } ]);
    g["nodes"] = json!([
        { "id": "z", "op": "core.zeros", "attrs": { "shape": ["batch", 4], "dtype": "float" } },
        { "id": "c", "op": "core.add", "inputs": { "a": "x", "b": "z" } }
    ]);
    let graph = load(&g).unwrap();
    for c in [ComputeDtype::F32, ComputeDtype::F16, ComputeDtype::Bf16] {
        let a = graph
            .analyze(&ValidateOptions { compute_dtype: c })
            .unwrap();
        assert_eq!(a.nodes[1].outputs()[0].1.dtype, c.dtype());
    }
    // A concrete f32 zeros only validates with compute dtype f32.
    g["nodes"][0]["attrs"]["dtype"] = json!("f32");
    let graph = load(&g).unwrap();
    let report = graph
        .analyze(&ValidateOptions {
            compute_dtype: ComputeDtype::F16,
        })
        .unwrap_err();
    assert!(
        report.has(polypous_ir::ErrorCode::DtypeMismatch),
        "{report}"
    );
}

// ---- Binding (§2.3.4) ----

fn one_input(shape: Value, dims: Value, nodes: Value, out: &str) -> Analysis {
    analyze(&json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": dims,
        "inputs": [ { "name": "x", "dtype": "f32", "shape": shape } ],
        "outputs": [ { "name": "y", "value": out } ],
        "nodes": nodes
    }))
}

#[test]
fn bind_inputs() {
    let a = analyze(&json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "dims": { "batch": { "min": 2, "max": 8 }, "seq": {} },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": ["batch", "seq", 4] },
                    { "name": "m", "dtype": "bool", "shape": ["batch", "seq", 1] } ],
        "outputs": [ { "name": "y", "value": "w" } ],
        "nodes": [ { "id": "w", "op": "core.where", "inputs": { "cond": "m", "a": "x", "b": "x" } } ]
    }));
    let ok = a
        .bind(
            &[
                ("x", desc(Dtype::F32, &[2, 3, 4])),
                ("m", desc(Dtype::Bool, &[2, 3, 1])),
            ],
            &[],
        )
        .unwrap();
    assert_eq!(ok.symbols, vec![("batch".into(), 2), ("seq".into(), 3)]);

    let bind = |x: &[usize], m: &[usize]| {
        a.bind(
            &[("x", desc(Dtype::F32, x)), ("m", desc(Dtype::Bool, m))],
            &[],
        )
    };
    assert!(matches!(
        bind(&[2, 3, 4], &[2, 5, 1]),
        Err(BindError::InconsistentSymbol { ref symbol, first: 3, value: 5, ref name, axis: 1, .. })
            if symbol == "seq" && name == "m"
    ));
    assert!(matches!(
        bind(&[2, 3, 5], &[2, 3, 1]),
        Err(BindError::DimMismatch {
            expected: 4,
            actual: 5,
            axis: 2,
            ..
        })
    ));
    assert!(matches!(
        bind(&[1, 3, 4], &[1, 3, 1]),
        Err(BindError::SymbolOutOfRange {
            value: 1,
            min: 2,
            ..
        })
    ));
    assert!(matches!(
        bind(&[9, 3, 4], &[9, 3, 1]),
        Err(BindError::SymbolOutOfRange {
            value: 9,
            max: Some(8),
            ..
        })
    ));
    assert!(matches!(
        bind(&[2, 3], &[2, 3, 1]),
        Err(BindError::Rank {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert_eq!(
        a.bind(&[("x", desc(Dtype::F32, &[2, 3, 4]))], &[]),
        Err(BindError::Missing {
            kind: ValueKind::Input,
            name: "m".into()
        })
    );
    let x = desc(Dtype::F32, &[2, 3, 4]);
    let m = desc(Dtype::Bool, &[2, 3, 1]);
    assert!(matches!(
        a.bind(&[("x", x), ("m", m), ("z", m)], &[]),
        Err(BindError::Unknown { .. })
    ));
    assert!(matches!(
        a.bind(&[("x", x), ("m", m), ("x", x)], &[]),
        Err(BindError::Duplicate { .. })
    ));
    assert!(matches!(
        a.bind(&[("x", x), ("m", m)], &[("s", m)]),
        Err(BindError::Unknown {
            kind: ValueKind::State,
            ..
        })
    ));
    assert!(matches!(
        a.bind(
            &[("x", desc(Dtype::F32, &[usize::MAX, 3, 4])), ("m", m)],
            &[]
        ),
        Err(BindError::DimTooLarge { .. })
    ));
}

#[test]
fn bind_state_expressions() {
    let a = analyze(&json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "dims": { "past": {}, "seq": {} },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": ["seq", 4] } ],
        "state": [ { "name": "k", "dtype": "f32", "shape": ["past", 4] },
                   { "name": "kk", "dtype": "f32", "shape": ["past + seq", 4] } ],
        "outputs": [ { "name": "y", "value": "c" } ],
        "nodes": [ { "id": "c", "op": "core.concat", "inputs": { "xs": ["k", "x"] }, "attrs": { "dim": 0 } } ],
        "state_updates": { "kk": "c" }
    }));
    assert_eq!(port_shape(&a.nodes[0], "out"), "f32 [past + seq, 4]");
    let x = desc(Dtype::F32, &[3, 4]);
    let k = desc(Dtype::F32, &[5, 4]);
    let b = a
        .bind(&[("x", x)], &[("k", k), ("kk", desc(Dtype::F32, &[8, 4]))])
        .unwrap();
    let BoundNode::Op(c) = &b.nodes[0] else {
        panic!()
    };
    assert_eq!(c.outputs, vec![vec![8, 4]]);
    assert!(matches!(
        a.bind(&[("x", x)], &[("k", k), ("kk", desc(Dtype::F32, &[7, 4]))]),
        Err(BindError::DimMismatch {
            kind: ValueKind::State,
            expected: 8,
            actual: 7,
            ..
        })
    ));
}

#[test]
fn bind_inferred_dims_and_attributes() {
    // conv output H - 2 is < 1 for H = 2.
    let a = analyze(&json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 },
        "dims": { "H": {} },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [1, 1, "H", 8] } ],
        "params": [ { "name": "w", "dtype": "float", "shape": [1, 1, 3, 3] } ],
        "outputs": [ { "name": "y", "value": "c" } ],
        "nodes": [ { "id": "c", "op": "nn.conv2d", "inputs": { "x": "x" }, "params": { "weight": "w" } } ]
    }));
    assert!(
        a.bind(&[("x", desc(Dtype::F32, &[1, 1, 3, 8]))], &[])
            .is_ok()
    );
    assert_eq!(
        a.bind(&[("x", desc(Dtype::F32, &[1, 1, 2, 8]))], &[]),
        Err(BindError::NonpositiveDim {
            node: "c".into(),
            port: "out".into(),
            axis: 2,
            value: 0
        })
    );

    // slice bounds: evaluated, and `end >= 0` checked.
    let s = one_input(
        json!(["seq", 4]),
        json!({ "seq": {} }),
        json!([ { "id": "s", "op": "core.slice", "inputs": { "x": "x" },
                  "attrs": { "dim": 0, "start": 1, "end": -3 } } ]),
        "s",
    );
    let b = s.bind(&[("x", desc(Dtype::F32, &[10, 4]))], &[]).unwrap();
    let BoundNode::Op(op) = &b.nodes[0] else {
        panic!()
    };
    assert_eq!(op.dim("start"), Some(&BoundDim::One(1)));
    assert_eq!(op.dim("end"), Some(&BoundDim::One(7)));
    assert_eq!(op.outputs, vec![vec![6, 4]]);
    assert!(matches!(
        s.bind(&[("x", desc(Dtype::F32, &[2, 4]))], &[]),
        Err(BindError::AttrOutOfRange { ref node, ref attr, value: -1, .. }) if node == "s" && attr == "end"
    ));

    // top_k: k <= the dim size.
    let t = one_input(
        json!(["vocab"]),
        json!({ "vocab": {} }),
        json!([ { "id": "t", "op": "core.top_k", "inputs": { "x": "x" }, "attrs": { "k": 5 } } ]),
        "t:values",
    );
    assert!(t.bind(&[("x", desc(Dtype::F32, &[5]))], &[]).is_ok());
    assert!(matches!(
        t.bind(&[("x", desc(Dtype::F32, &[4]))], &[]),
        Err(BindError::AttrOutOfRange { ref attr, value: -1, .. }) if attr == "k"
    ));
}

#[test]
fn bind_overflow_and_element_counts() {
    let f = one_input(
        json!(["a", "b"]),
        json!({ "a": {}, "b": {} }),
        json!([ { "id": "f", "op": "core.flatten", "inputs": { "x": "x" } } ]),
        "f",
    );
    let big = 1usize << 40;
    assert!(matches!(
        f.bind(&[("x", desc(Dtype::F32, &[big, big]))], &[]),
        Err(BindError::Overflow { ref context }) if context.contains("node `f`")
    ));
    let e = one_input(
        json!(["a", "b", "c"]),
        json!({ "a": {}, "b": {}, "c": {} }),
        json!([ { "id": "e", "op": "core.exp", "inputs": { "x": "x" } } ]),
        "e",
    );
    let d = 1usize << 22;
    assert!(matches!(
        e.bind(&[("x", desc(Dtype::F32, &[d, d, d]))], &[]),
        Err(BindError::TooManyElements { ref node, .. }) if node == "e"
    ));
}

#[test]
fn bind_names_body_nodes_with_their_repeat() {
    let mut g = with_body(
        json!([ { "id": "c", "op": "core.slice", "inputs": { "x": "h" },
                  "attrs": { "dim": 0, "end": -2 } },
                { "id": "p", "op": "core.pad", "inputs": { "x": "c" }, "attrs": { "pad": [0, 0, 0, 2] } } ]),
        "p",
    );
    g["nodes"][0]["count"] = json!(2);
    let a = analyze(&g);
    assert!(matches!(
        a.bind(&[("x", desc(Dtype::F32, &[1, 4]))], &[]),
        Err(BindError::AttrOutOfRange { ref node, .. }) if node == "r/c"
    ));
    let b = a.bind(&[("x", desc(Dtype::F32, &[3, 4]))], &[]).unwrap();
    let BoundNode::Repeat(r) = &b.nodes[0] else {
        panic!()
    };
    let BoundNode::Op(c) = &r.nodes[0] else {
        panic!()
    };
    assert_eq!(c.outputs, vec![vec![1, 4]]);
    assert_eq!(r.carry, vec![vec![3, 4]]);
}

#[test]
fn bind_error_messages_name_the_culprit() {
    let e = BindError::InconsistentSymbol {
        symbol: "seq".into(),
        first: 3,
        first_kind: ValueKind::Input,
        first_name: "x".into(),
        first_axis: 1,
        value: 5,
        kind: ValueKind::State,
        name: "k".into(),
        axis: 0,
    };
    assert_eq!(
        e.to_string(),
        "dim `seq` is 3 from input `x` dim 1, but 5 from state entry `k` dim 0"
    );
    let e = BindError::SymbolOutOfRange {
        symbol: "batch".into(),
        value: 70,
        min: 1,
        max: Some(64),
    };
    assert_eq!(
        e.to_string(),
        "dim `batch` is 70, outside its declared range 1..=64"
    );
}

#[test]
fn many_ops_example() {
    let a = analyze(&serde_json::from_str(include_str!("data/ops.graph.json")).unwrap());
    let shape = |id: &str| {
        let n = a.nodes.iter().find(|n| n.id() == id).unwrap();
        port_shape(n, &n.outputs()[0].0)
    };
    assert_eq!(shape("conv"), "f32 [batch, 8, (H + 1) // 2, (W + 1) // 2]");
    assert_eq!(shape("head"), "f32 [batch, 10]");
    assert_eq!(shape("ctx"), "f32 [batch, past + seq, 16]");
    assert_eq!(shape("last"), "f32 [batch, 1, 16]");
    assert_eq!(shape("pos2"), "f32 [1, 1, seq]");
    assert_eq!(shape("padded"), "f32 [batch, 1, seq + 2]");
    let top = a.nodes.iter().find(|n| n.id() == "top").unwrap();
    assert_eq!(port_shape(top, "indices"), "i32 [batch, 3]");

    let b = a
        .bind(
            &[
                ("image", desc(Dtype::F32, &[2, 3, 9, 16])),
                ("tokens", desc(Dtype::I64, &[2, 5])),
                ("mask", desc(Dtype::Bool, &[2, 1, 5])),
            ],
            &[("cache", desc(Dtype::F32, &[2, 7, 16]))],
        )
        .unwrap();
    let shapes: Vec<Vec<usize>> = b
        .nodes
        .iter()
        .map(|n| match n {
            BoundNode::Op(op) => op.outputs[0].clone(),
            BoundNode::Repeat(r) => r.carry[0].clone(),
        })
        .collect();
    assert_eq!(shapes[0], vec![2, 8, 5, 8]);
    assert_eq!(shapes[8], vec![2, 12, 16]);
    assert_eq!(shapes[20], vec![2, 1, 5]);
    // `H` has min 8.
    assert!(matches!(
        a.bind(
            &[
                ("image", desc(Dtype::F32, &[2, 3, 7, 16])),
                ("tokens", desc(Dtype::I64, &[2, 5])),
                ("mask", desc(Dtype::Bool, &[2, 1, 5])),
            ],
            &[("cache", desc(Dtype::F32, &[2, 7, 16]))],
        ),
        Err(BindError::SymbolOutOfRange { ref symbol, .. }) if symbol == "H"
    ));
}
