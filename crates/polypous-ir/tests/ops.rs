//! Per-op signature, dtype and shape inference (`docs/specs/graph-ir.md` §3, §4.1 checks 4, 6, 7):
//! for every op a valid case and the main invalid cases, with the expected code and node id, and the
//! symbolic cases of §2.3.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_ir::catalog;
use polypous_ir::{AttrValue, ComputeDtype, Diagnostic, Graph, NodeInfo, Scalar, ValidateOptions};
use serde_json::{Value, json};

/// A one-node graph: graph inputs named like the op inputs, node `n`, output `y` = `n:<port>`.
struct Case {
    op: String,
    inputs: Vec<(String, String, Value)>,
    /// Graph inputs that only bind symbols; not wired to the node.
    extra: Vec<(String, String, Value)>,
    node_inputs: Option<Value>,
    attrs: Value,
    params: Vec<(String, String, Value)>,
    port: String,
    compute: ComputeDtype,
    dims: Vec<String>,
}

fn case(op: &str) -> Case {
    Case {
        op: op.to_owned(),
        inputs: Vec::new(),
        extra: Vec::new(),
        node_inputs: None,
        attrs: json!({}),
        params: Vec::new(),
        port: "out".to_owned(),
        compute: ComputeDtype::F32,
        dims: Vec::new(),
    }
}

impl Case {
    fn input(mut self, name: &str, dtype: &str, shape: Value) -> Self {
        self.inputs.push((name.into(), dtype.into(), shape));
        self
    }
    fn extra(mut self, name: &str, dtype: &str, shape: Value) -> Self {
        self.extra.push((name.into(), dtype.into(), shape));
        self
    }
    fn x(self, dtype: &str, shape: Value) -> Self {
        self.input("x", dtype, shape)
    }
    fn ab(self, dtype: &str, a: Value, b: Value) -> Self {
        self.input("a", dtype, a).input("b", dtype, b)
    }
    fn node_inputs(mut self, v: Value) -> Self {
        self.node_inputs = Some(v);
        self
    }
    fn attrs(mut self, v: Value) -> Self {
        self.attrs = v;
        self
    }
    /// Op param `name` bound to a graph param of the same name with `shape`.
    fn param(mut self, name: &str, shape: Value) -> Self {
        self.params.push((name.into(), name.into(), shape));
        self
    }
    fn port(mut self, p: &str) -> Self {
        self.port = p.into();
        self
    }
    fn compute(mut self, c: ComputeDtype) -> Self {
        self.compute = c;
        self
    }
    fn dim(mut self, d: &str) -> Self {
        self.dims.push(d.into());
        self
    }

    fn doc(&self) -> Value {
        let mut dims = serde_json::Map::new();
        for (_, _, shape) in self.inputs.iter().chain(&self.extra) {
            for d in shape.as_array().unwrap() {
                if let Some(s) = d.as_str() {
                    dims.insert(s.into(), json!({}));
                }
            }
        }
        for d in &self.dims {
            dims.insert(d.clone(), json!({}));
        }
        let inputs: Vec<Value> = self
            .inputs
            .iter()
            .chain(&self.extra)
            .map(|(n, d, s)| json!({ "name": n, "dtype": d, "shape": s }))
            .collect();
        let node_inputs = self.node_inputs.clone().unwrap_or_else(|| {
            Value::Object(
                self.inputs
                    .iter()
                    .map(|(n, _, _)| (n.clone(), json!(n)))
                    .collect(),
            )
        });
        let params: Vec<Value> = self
            .params
            .iter()
            .map(|(_, g, s)| json!({ "name": g, "dtype": "float", "shape": s }))
            .collect();
        let node_params: serde_json::Map<String, Value> = self
            .params
            .iter()
            .map(|(p, g, _)| (p.clone(), json!(g)))
            .collect();
        json!({
            "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
            "dims": dims, "inputs": inputs, "params": params,
            "outputs": [ { "name": "y", "value": format!("n:{}", self.port) } ],
            "nodes": [ { "id": "n", "op": self.op, "inputs": node_inputs, "params": node_params,
                         "attrs": self.attrs } ]
        })
    }

    fn options(&self) -> ValidateOptions {
        ValidateOptions {
            compute_dtype: self.compute,
        }
    }

    fn diagnostics(&self) -> Vec<Diagnostic> {
        let bytes = self.doc().to_string();
        polypous_ir::load(bytes.as_bytes(), &self.options())
            .report
            .diagnostics
    }

    fn graph(&self) -> Graph {
        let bytes = self.doc().to_string();
        let outcome = polypous_ir::load(bytes.as_bytes(), &self.options());
        assert!(
            outcome.report.is_ok(),
            "{}: {}\n{}",
            self.op,
            outcome.report,
            self.doc()
        );
        outcome.graph.unwrap()
    }

    fn info(&self) -> polypous_ir::OpInfo {
        let analysis = self.graph().analyze(&self.options()).unwrap();
        match analysis.nodes.into_iter().next().unwrap() {
            NodeInfo::Op(op) => op,
            NodeInfo::Repeat(_) => panic!("repeat"),
        }
    }

    /// Asserts the graph is valid and returns `"<dtype> <shape>"` of the output port.
    #[track_caller]
    fn ok(&self) -> String {
        let info = self.info();
        let (_, v) = info
            .outputs
            .iter()
            .find(|(p, _)| *p == self.port)
            .unwrap_or_else(|| panic!("no port {}", self.port));
        format!("{} {}", v.dtype, v.shape_string())
    }

    /// Asserts an error with `code` on node `n` and returns its pointer.
    #[track_caller]
    fn err(&self, code: &str) -> String {
        let diags = self.diagnostics();
        let found = diags
            .iter()
            .find(|d| d.code.as_str() == code && d.node_id.as_deref() == Some("n"));
        match found {
            Some(d) => d.pointer.clone(),
            None => panic!(
                "{}: expected {code} on node `n`, got:\n{}",
                self.op,
                diags
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
        }
    }
}

// ---- Elementwise ----

#[test]
fn binary_elementwise() {
    for op in ["add", "sub", "mul", "maximum", "minimum"] {
        let op = format!("core.{op}");
        assert_eq!(
            case(&op).ab("f32", json!([2, 3]), json!([3])).ok(),
            "f32 [2, 3]"
        );
        assert_eq!(
            case(&op).ab("i32", json!([4, 1]), json!([1, 5])).ok(),
            "i32 [4, 5]"
        );
        assert_eq!(
            case(&op)
                .ab("bool", json!([2]), json!([2]))
                .err("unsupported_dtype"),
            "/nodes/0/inputs/a"
        );
        let mixed = case(&op)
            .input("a", "f16", json!([2]))
            .input("b", "f32", json!([2]));
        assert_eq!(mixed.err("dtype_mismatch"), "/nodes/0/inputs/b");
        case(&op)
            .ab("f32", json!([2, 3]), json!([4]))
            .err("broadcast_mismatch");
        case(&op).x("f32", json!([2])).err("missing_input");
        case(&op)
            .ab("f32", json!([2]), json!([2]))
            .attrs(json!({ "alpha": 1 }))
            .err("unknown_attr");
    }
    for op in ["div", "pow"] {
        let op = format!("core.{op}");
        assert_eq!(
            case(&op).ab("bf16", json!([2, 1]), json!([3])).ok(),
            "bf16 [2, 3]"
        );
        case(&op)
            .ab("i64", json!([2]), json!([2]))
            .err("unsupported_dtype");
    }
}

#[test]
fn unary_elementwise() {
    for op in ["neg", "abs"] {
        let op = format!("core.{op}");
        assert_eq!(case(&op).x("i64", json!([3, 4])).ok(), "i64 [3, 4]");
        case(&op).x("bool", json!([3])).err("unsupported_dtype");
    }
    for op in ["exp", "log", "sqrt", "rsqrt", "sin", "cos", "tanh"] {
        let op = format!("core.{op}");
        assert_eq!(
            case(&op).x("f16", json!(["batch", 4])).ok(),
            "f16 [batch, 4]"
        );
        case(&op).x("i32", json!([3])).err("unsupported_dtype");
        case(&op)
            .input("y", "f32", json!([3]))
            .node_inputs(json!({ "x": "y", "y": "y" }))
            .err("unknown_input");
    }
}

#[test]
fn clamp() {
    let c = |dtype: &str, attrs: Value| case("core.clamp").x(dtype, json!([3])).attrs(attrs);
    assert_eq!(
        c("f32", json!({ "min": -1.5, "max": 0.75 })).ok(),
        "f32 [3]"
    );
    assert_eq!(
        c("f32", json!({ "min": "nan", "max": 1.0 })).ok(),
        "f32 [3]"
    );
    assert_eq!(c("i32", json!({ "max": 7 })).ok(), "i32 [3]");
    let info = c("f16", json!({ "min": 0 })).info();
    assert_eq!(info.attrs.scalar("min"), Some(Scalar::Float(0.0)));
    assert_eq!(info.attrs.scalar("max"), None);
    c("f32", json!({})).err("missing_attr");
    assert_eq!(
        c("f32", json!({ "min": 2, "max": 1 })).err("invalid_attr"),
        "/nodes/0/attrs/min"
    );
    c("i32", json!({ "min": 0.5 })).err("invalid_attr");
    c("i32", json!({ "min": 3_000_000_000_i64 })).err("invalid_attr");
    c("f32", json!({ "min": true })).err("invalid_attr");
    c("f32", json!({ "min": "big" })).err("invalid_attr");
    c("bool", json!({ "min": 0 })).err("unsupported_dtype");
}

#[test]
fn where_() {
    let w = case("core.where")
        .input("cond", "bool", json!([4, 1]))
        .input("a", "f32", json!([1, 5]))
        .input("b", "f32", json!([5]));
    assert_eq!(w.ok(), "f32 [4, 5]");
    case("core.where")
        .input("cond", "i64", json!([4]))
        .ab("f32", json!([4]), json!([4]))
        .err("unsupported_dtype");
    case("core.where")
        .input("cond", "bool", json!([4]))
        .input("a", "f32", json!([4]))
        .input("b", "i64", json!([4]))
        .err("dtype_mismatch");
    case("core.where")
        .input("cond", "bool", json!([3]))
        .ab("i64", json!([4]), json!([4]))
        .err("broadcast_mismatch");
}

#[test]
fn comparison_and_logic() {
    for op in ["eq", "ne"] {
        let op = format!("core.{op}");
        assert_eq!(
            case(&op).ab("bool", json!([2, 3]), json!([1])).ok(),
            "bool [2, 3]"
        );
        case(&op)
            .input("a", "i32", json!([2]))
            .input("b", "i64", json!([2]))
            .err("dtype_mismatch");
    }
    for op in ["lt", "le", "gt", "ge"] {
        let op = format!("core.{op}");
        assert_eq!(case(&op).ab("i64", json!([2]), json!([2])).ok(), "bool [2]");
        case(&op)
            .ab("bool", json!([2]), json!([2]))
            .err("unsupported_dtype");
    }
    for op in ["and", "or"] {
        let op = format!("core.{op}");
        assert_eq!(
            case(&op).ab("bool", json!([2, 1]), json!([3])).ok(),
            "bool [2, 3]"
        );
        assert_eq!(
            case(&op)
                .input("a", "bool", json!([2]))
                .input("b", "f32", json!([2]))
                .err("unsupported_dtype"),
            "/nodes/0/inputs/b"
        );
    }
    assert_eq!(case("core.not").x("bool", json!([2])).ok(), "bool [2]");
    case("core.not")
        .x("i32", json!([2]))
        .err("unsupported_dtype");
}

// ---- Shape ----

#[test]
fn reshape() {
    let r = |shape: Value, to: Value| {
        case("core.reshape")
            .x("f32", shape)
            .attrs(json!({ "shape": to }))
    };
    assert_eq!(r(json!([12]), json!([3, 4])).ok(), "f32 [3, 4]");
    assert_eq!(r(json!([2, 6]), json!([-1, 3])).ok(), "f32 [4, 3]");
    // Symbolic: `-1` divides on normal forms; `batch * seq` as an attribute.
    assert_eq!(
        r(json!(["batch", "seq", 64]), json!(["batch * seq", 64])).ok(),
        "f32 [batch*seq, 64]"
    );
    assert_eq!(
        r(json!(["batch", "seq", 64]), json!([-1, 64])).ok(),
        "f32 [batch*seq, 64]"
    );
    assert_eq!(
        r(json!(["batch", "seq", 64]), json!(["batch", -1, 8])).ok(),
        "f32 [batch, 8*seq, 8]"
    );
    let info = r(json!(["batch", 6]), json!(["batch", -1, 2])).info();
    assert_eq!(info.attrs.dims("shape").unwrap()[1].to_string(), "3");
    // Not exact: 6*batch / 4.
    assert_eq!(
        r(json!(["batch", 6]), json!([4, -1])).err("shape_mismatch"),
        "/nodes/0/attrs/shape"
    );
    // A product of two terms doesn't divide.
    r(json!(["batch", "seq"]), json!(["batch + seq", -1])).err("shape_mismatch");
    r(json!([12]), json!([5, 2])).err("shape_mismatch");
    r(json!(["batch", 4]), json!(["seq", 4]))
        .dim("seq")
        .err("unbound_dim");
    assert_eq!(
        r(json!([12]), json!([-1, -1])).err("invalid_attr"),
        "/nodes/0/attrs/shape/1"
    );
    r(json!([12]), json!([0, 12])).err("invalid_attr");
    r(json!([12]), json!([1, 1, 1, 1, 1, 1, 12])).err("invalid_attr");
    r(json!([12]), json!("12")).err("invalid_attr");
    r(json!([12]), json!(["(12"])).err("invalid_dim_expr");
    case("core.reshape")
        .x("f32", json!([12]))
        .err("missing_attr");
}

#[test]
fn flatten_permute_transpose() {
    let f = |attrs: Value| {
        case("core.flatten")
            .x("f32", json!(["batch", 3, 4, 5]))
            .attrs(attrs)
    };
    assert_eq!(f(json!({})).ok(), "f32 [60*batch]");
    assert_eq!(f(json!({ "start_dim": 1 })).ok(), "f32 [batch, 60]");
    assert_eq!(
        f(json!({ "start_dim": 1, "end_dim": -2 })).ok(),
        "f32 [batch, 12, 5]"
    );
    f(json!({ "start_dim": 2, "end_dim": 1 })).err("invalid_attr");
    assert_eq!(
        f(json!({ "start_dim": 4 })).err("invalid_attr"),
        "/nodes/0/attrs/start_dim"
    );
    f(json!({ "start_dim": 1.0 })).err("invalid_attr");

    let p = |dims: Value| {
        case("core.permute")
            .x("i64", json!(["batch", 3, 4]))
            .attrs(json!({ "dims": dims }))
    };
    assert_eq!(p(json!([2, 0, 1])).ok(), "i64 [4, batch, 3]");
    assert_eq!(p(json!([-1, 0, 1])).ok(), "i64 [4, batch, 3]");
    assert_eq!(
        p(json!([2, 0, 1])).info().attrs.axes("dims"),
        Some(&[2usize, 0, 1][..])
    );
    p(json!([0, 1])).err("invalid_attr");
    assert_eq!(
        p(json!([0, 1, -3])).err("invalid_attr"),
        "/nodes/0/attrs/dims/2"
    );
    p(json!([0, 1, 3])).err("invalid_attr");

    let t = |a: i64, b: i64| {
        case("core.transpose")
            .x("bool", json!(["batch", 3, 4]))
            .attrs(json!({ "dim0": a, "dim1": b }))
    };
    assert_eq!(t(0, -1).ok(), "bool [4, 3, batch]");
    t(0, 3).err("invalid_attr");
    case("core.transpose")
        .x("f32", json!([3]))
        .attrs(json!({ "dim0": 0 }))
        .err("missing_attr");
}

#[test]
fn squeeze_unsqueeze() {
    let s = |shape: Value, attrs: Value| case("core.squeeze").x("f32", shape).attrs(attrs);
    assert_eq!(
        s(json!([1, "batch", 1, 4]), json!({})).ok(),
        "f32 [batch, 4]"
    );
    assert_eq!(
        s(json!([1, "batch", 1, 4]), json!({}))
            .info()
            .attrs
            .axes("dim"),
        Some(&[0usize, 2][..])
    );
    assert_eq!(s(json!([1, 1]), json!({})).ok(), "f32 [1]");
    assert_eq!(
        s(json!([1, 3, 1]), json!({ "dim": [-1] })).ok(),
        "f32 [1, 3]"
    );
    // A symbolic dim is never squeezed.
    assert_eq!(s(json!(["batch", 3]), json!({})).ok(), "f32 [batch, 3]");
    assert_eq!(
        s(json!(["batch", 3]), json!({ "dim": [0] })).err("invalid_attr"),
        "/nodes/0/attrs/dim/0"
    );
    s(json!([1, 3]), json!({ "dim": [0, -2] })).err("invalid_attr");
    s(json!([1, 3]), json!({ "dim": [] })).err("invalid_attr");

    let u = |shape: Value, dim: i64| {
        case("core.unsqueeze")
            .x("f32", shape)
            .attrs(json!({ "dim": dim }))
    };
    assert_eq!(u(json!([3, 4]), 0).ok(), "f32 [1, 3, 4]");
    assert_eq!(u(json!([3, 4]), 2).ok(), "f32 [3, 4, 1]");
    assert_eq!(u(json!([3, 4]), -1).ok(), "f32 [3, 4, 1]");
    assert_eq!(u(json!([3, 4]), -3).ok(), "f32 [1, 3, 4]");
    u(json!([3, 4]), 3).err("invalid_attr");
    u(json!([3, 4]), -4).err("invalid_attr");
    u(json!([1, 1, 1, 1, 1, 1]), 0).err("rank_too_large");
}

#[test]
fn expand() {
    let e = |shape: Value, to: Value| {
        case("core.expand")
            .x("f32", shape)
            .attrs(json!({ "shape": to }))
    };
    assert_eq!(e(json!([1, 4]), json!([3, 4])).ok(), "f32 [3, 4]");
    assert_eq!(e(json!([1, 4]), json!([2, 3, -1])).ok(), "f32 [2, 3, 4]");
    assert_eq!(
        e(json!(["batch", 1]), json!([-1, "batch"])).ok(),
        "f32 [batch, batch]"
    );
    assert_eq!(
        e(json!([1, 4]), json!([2, 3, -1]))
            .info()
            .attrs
            .dims("shape")
            .unwrap()[2]
            .to_string(),
        "4"
    );
    e(json!([2, 4]), json!([3, 4])).err("shape_mismatch");
    e(json!(["batch", 4]), json!([2, 4])).err("shape_mismatch");
    e(json!([1, 4]), json!([4])).err("invalid_attr");
    assert_eq!(
        e(json!([1, 4]), json!([-1, 1, 4])).err("invalid_attr"),
        "/nodes/0/attrs/shape/0"
    );
    e(json!([1, 4]), json!([0, 4])).err("invalid_attr");
}

#[test]
fn concat() {
    let c = case("core.concat")
        .input("past", "f16", json!(["batch", "past", 64]))
        .input("cur", "f16", json!(["batch", "seq", 64]))
        .node_inputs(json!({ "xs": ["past", "cur"] }))
        .attrs(json!({ "dim": 1 }));
    assert_eq!(c.ok(), "f16 [batch, past + seq, 64]");
    let one = case("core.concat")
        .x("i32", json!([3]))
        .node_inputs(json!({ "xs": ["x"] }))
        .attrs(json!({ "dim": 0 }));
    assert_eq!(one.ok(), "i32 [3]");
    let two = |a: Value, b: Value, da: &str, db: &str| {
        case("core.concat")
            .input("p", da, a)
            .input("q", db, b)
            .node_inputs(json!({ "xs": ["p", "q"] }))
            .attrs(json!({ "dim": -1 }))
    };
    assert_eq!(
        two(json!([2, 3]), json!([2, 5]), "f32", "f32").ok(),
        "f32 [2, 8]"
    );
    assert_eq!(
        two(json!([2, 3]), json!([4, 5]), "f32", "f32").err("shape_mismatch"),
        "/nodes/0/inputs/xs/1"
    );
    assert_eq!(
        two(json!([2, 3]), json!([2, 5]), "f32", "i64").err("dtype_mismatch"),
        "/nodes/0/inputs/xs/1"
    );
    two(json!([2, 3]), json!([5]), "f32", "f32").err("rank_mismatch");
    case("core.concat")
        .x("f32", json!([3]))
        .node_inputs(json!({ "xs": "x" }))
        .attrs(json!({ "dim": 0 }))
        .err("invalid_json");
    case("core.concat")
        .x("f32", json!([3]))
        .node_inputs(json!({ "xs": [] }))
        .attrs(json!({ "dim": 0 }))
        .err("invalid_value");
    let many: Vec<&str> = vec!["x"; 1025];
    case("core.concat")
        .x("f32", json!([3]))
        .node_inputs(json!({ "xs": many }))
        .attrs(json!({ "dim": 0 }))
        .err("invalid_value");
    let max: Vec<&str> = vec!["x"; 1024];
    assert_eq!(
        case("core.concat")
            .x("f32", json!([3]))
            .node_inputs(json!({ "xs": max }))
            .attrs(json!({ "dim": 0 }))
            .ok(),
        "f32 [3072]"
    );
}

#[test]
fn split() {
    let s = |shape: Value, attrs: Value| case("core.split").x("f32", shape).attrs(attrs).port("1");
    assert_eq!(
        s(json!([2, 10]), json!({ "dim": 1, "sizes": [3, 7] })).ok(),
        "f32 [2, 7]"
    );
    assert_eq!(
        s(
            json!(["batch", "seq"]),
            json!({ "dim": -1, "sizes": ["seq - 2", 2] })
        )
        .port("0")
        .ok(),
        "f32 [batch, seq - 2]"
    );
    s(
        json!(["batch", "seq"]),
        json!({ "dim": -1, "sizes": ["seq // 2", "seq // 2"] }),
    )
    .err("shape_mismatch");
    s(json!([2, 10]), json!({ "dim": 1, "sizes": [3, 6] })).err("shape_mismatch");
    s(json!([2, 10]), json!({ "dim": 1, "sizes": [10, 0] })).err("invalid_attr");
    s(json!([2, 10]), json!({ "dim": 1, "sizes": [] })).err("invalid_attr");
    let ones: Vec<i64> = vec![1; 1025];
    s(json!([1025]), json!({ "dim": 0, "sizes": ones })).err("invalid_attr");
    let ones: Vec<i64> = vec![1; 1024];
    assert_eq!(
        s(json!([1024]), json!({ "dim": 0, "sizes": ones }))
            .port("1023")
            .ok(),
        "f32 [1]"
    );
    s(json!([2, 10]), json!({ "dim": 1 })).err("missing_attr");
}

#[test]
fn slice() {
    let s = |shape: Value, attrs: Value| case("core.slice").x("f32", shape).attrs(attrs);
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "start": 2, "end": 7 })).ok(),
        "f32 [5]"
    );
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "start": 1, "step": 3 })).ok(),
        "f32 [3]"
    );
    // Bounds clamp to n, as in PyTorch.
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "start": 4, "end": 100 })).ok(),
        "f32 [6]"
    );
    // Negative literal bounds count from the end.
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "start": -3 })).ok(),
        "f32 [3]"
    );
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "end": -1 })).ok(),
        "f32 [9]"
    );
    assert_eq!(
        s(json!(["seq", 4]), json!({ "dim": 0, "start": -3 })).ok(),
        "f32 [3, 4]"
    );
    assert_eq!(
        s(json!(["seq", 4]), json!({ "dim": 0, "end": -1 })).ok(),
        "f32 [seq - 1, 4]"
    );
    assert_eq!(
        s(json!(["seq", 4]), json!({ "dim": 0, "start": 2 })).ok(),
        "f32 [0 - min(2, seq) + seq, 4]"
    );
    let info = s(
        json!(["seq", 4]),
        json!({ "dim": 0, "start": -2, "end": -1 }),
    )
    .info();
    assert_eq!(info.attrs.dim("start").unwrap().to_string(), "seq - 2");
    assert_eq!(info.attrs.dim("end").unwrap().to_string(), "seq - 1");
    assert_eq!(info.checks.len(), 2, "{:?}", info.checks);
    assert_eq!(info.outputs[0].1.shape_string(), "[1, 4]");
    // Empty results.
    assert_eq!(
        s(json!([10]), json!({ "dim": 0, "start": 7, "end": 7 })).err("nonpositive_dim"),
        "/nodes/0"
    );
    s(json!([10]), json!({ "dim": 0, "start": 12 })).err("nonpositive_dim");
    s(json!([10]), json!({ "dim": 0, "start": -11 })).err("shape_mismatch");
    s(json!([10]), json!({ "dim": 0, "step": 0 })).err("invalid_attr");
    s(json!([10]), json!({ "dim": 1 })).err("invalid_attr");
    s(json!([10]), json!({})).err("missing_attr");
}

#[test]
fn pad() {
    let p = |dtype: &str, shape: Value, attrs: Value| case("core.pad").x(dtype, shape).attrs(attrs);
    assert_eq!(
        p("f32", json!([2, 3, 4]), json!({ "pad": [1, 2] })).ok(),
        "f32 [2, 3, 7]"
    );
    assert_eq!(
        p(
            "f32",
            json!([2, 3, 4]),
            json!({ "pad": [1, 2, 0, 3], "value": "-inf" })
        )
        .ok(),
        "f32 [2, 6, 7]"
    );
    assert_eq!(
        p("bool", json!([2]), json!({ "pad": [1, 1], "value": true })).ok(),
        "bool [4]"
    );
    let info = p("i64", json!([2]), json!({ "pad": [1, 1] })).info();
    assert_eq!(info.attrs.scalar("value"), Some(Scalar::Int(0)));
    assert_eq!(info.attrs.enum_value("mode"), Some("constant"));
    p("f32", json!([2]), json!({ "pad": [1] })).err("invalid_attr");
    p("f32", json!([2]), json!({ "pad": [1, 1, 1, 1] })).err("invalid_attr");
    assert_eq!(
        p("f32", json!([2]), json!({ "pad": [1, -1] })).err("invalid_attr"),
        "/nodes/0/attrs/pad/1"
    );
    p(
        "f32",
        json!([2]),
        json!({ "pad": [1, 1], "mode": "reflect" }),
    )
    .err("invalid_attr");
    p("i32", json!([2]), json!({ "pad": [1, 1], "value": 0.5 })).err("invalid_attr");
    // The default value 0 is the zero of the dtype; a given 0 is not a bool.
    assert_eq!(
        p("bool", json!([2]), json!({ "pad": [1, 1] }))
            .info()
            .attrs
            .scalar("value"),
        Some(Scalar::Bool(false))
    );
    p("bool", json!([2]), json!({ "pad": [1, 1], "value": 0 })).err("invalid_attr");
}

// ---- Indexing ----

#[test]
fn gather_index_select() {
    let g = |x: Value, index: Value, idx_dtype: &str, dim: i64| {
        case("core.gather")
            .x("f32", x)
            .input("index", idx_dtype, index)
            .attrs(json!({ "dim": dim }))
    };
    assert_eq!(g(json!([4, 5]), json!([4, 2]), "i64", 1).ok(), "f32 [4, 2]");
    assert_eq!(g(json!([4, 5]), json!([3, 2]), "i32", 1).ok(), "f32 [3, 2]");
    assert_eq!(
        g(json!(["batch", 5]), json!(["batch", 9]), "i32", 1).ok(),
        "f32 [batch, 9]"
    );
    g(json!([4, 5]), json!([5, 2]), "i64", 1).err("shape_mismatch");
    g(json!(["batch", 5]), json!([2, 2]), "i64", 1).err("shape_mismatch");
    g(json!([4, 5]), json!([4]), "i64", 0).err("rank_mismatch");
    g(json!([4, 5]), json!([4, 2]), "f32", 1).err("unsupported_dtype");
    g(json!([4, 5]), json!([4, 2]), "i64", 2).err("invalid_attr");

    let s = |x: Value, index: Value| {
        case("core.index_select")
            .x("bf16", x)
            .input("index", "i32", index)
            .attrs(json!({ "dim": 0 }))
    };
    assert_eq!(s(json!([10, 3]), json!(["seq"])).ok(), "bf16 [seq, 3]");
    s(json!([10, 3]), json!([2, 2])).err("rank_mismatch");
}

// ---- Creation ----

#[test]
fn constant() {
    let c = |attrs: Value| case("core.constant").attrs(attrs);
    assert_eq!(
        c(json!({ "dtype": "f16", "shape": [2, 2], "values": [1, 2.5, "-inf", "nan"] })).ok(),
        "f16 [2, 2]"
    );
    assert_eq!(
        c(json!({ "dtype": "float", "shape": [1], "values": [0.5] }))
            .compute(ComputeDtype::Bf16)
            .ok(),
        "bf16 [1]"
    );
    let info = c(json!({ "dtype": "i32", "shape": [2], "values": [1, -2] })).info();
    assert_eq!(
        info.attrs.scalars("values"),
        Some(&[Scalar::Int(1), Scalar::Int(-2)][..])
    );
    c(json!({ "dtype": "f32", "shape": [3], "values": [1, 2] })).err("invalid_attr");
    assert_eq!(
        c(json!({ "dtype": "i32", "shape": [2], "values": [1, 2.5] })).err("invalid_attr"),
        "/nodes/0/attrs/values/1"
    );
    c(json!({ "dtype": "bool", "shape": [1], "values": [1] })).err("invalid_attr");
    c(json!({ "dtype": "f32", "shape": ["seq"], "values": [1] })).err("invalid_attr");
    c(json!({ "dtype": "f32", "shape": [0], "values": [] })).err("invalid_attr");
    c(json!({ "dtype": "f64", "shape": [1], "values": [1] })).err("invalid_attr");
    let max: Vec<i64> = vec![0; 65_536];
    assert_eq!(
        c(json!({ "dtype": "i64", "shape": [256, 256], "values": max })).ok(),
        "i64 [256, 256]"
    );
    let over: Vec<i64> = vec![0; 65_537];
    assert_eq!(
        c(json!({ "dtype": "i64", "shape": [65_537], "values": over })).err("too_many_elements"),
        "/nodes/0/attrs/shape"
    );
    c(json!({ "dtype": "i64", "shape": [4_294_967_296_i64, 4_294_967_296_i64], "values": [] }))
        .err("too_many_elements");
    c(json!({ "dtype": "i64", "shape": [4_294_967_297_i64], "values": [] })).err("invalid_attr");
}

#[test]
fn arange() {
    let a = |attrs: Value| {
        case("core.arange")
            .attrs(attrs)
            .extra("t", "f32", json!(["seq"]))
    };
    assert_eq!(a(json!({ "end": 10 })).ok(), "i64 [10]");
    assert_eq!(
        a(json!({ "start": 2, "end": 10, "step": 3 })).ok(),
        "i64 [3]"
    );
    assert_eq!(
        a(json!({ "end": "seq", "dtype": "float" })).ok(),
        "f32 [seq]"
    );
    assert_eq!(
        a(json!({ "start": "seq", "end": "2 * seq", "dtype": "i32" })).ok(),
        "i32 [seq]"
    );
    let info = a(json!({ "start": "seq - 1", "end": "seq + 3" })).info();
    assert_eq!(info.checks.len(), 1);
    assert_eq!(info.outputs[0].1.shape_string(), "[4]");
    a(json!({ "end": 10, "dtype": "bool" })).err("invalid_attr");
    a(json!({ "start": -1, "end": 10 })).err("invalid_attr");
    a(json!({ "start": 5, "end": 5 })).err("nonpositive_dim");
    a(json!({})).err("missing_attr");
}

#[test]
fn zeros_ones_full() {
    for op in ["core.zeros", "core.ones"] {
        assert_eq!(
            case(op).attrs(json!({ "shape": [2, 3] })).ok(),
            "f32 [2, 3]"
        );
        assert_eq!(
            case(op)
                .attrs(json!({ "shape": ["batch * 2"], "dtype": "float" }))
                .extra("t", "f32", json!(["batch"]))
                .compute(ComputeDtype::F16)
                .ok(),
            "f16 [2*batch]"
        );
        case(op).attrs(json!({ "shape": [0] })).err("invalid_attr");
        case(op).attrs(json!({ "shape": [] })).err("invalid_attr");
        case(op)
            .attrs(json!({ "shape": [1], "dtype": "q4" }))
            .err("invalid_attr");
    }
    let f = |attrs: Value| case("core.full").attrs(attrs);
    assert_eq!(f(json!({ "shape": [2], "value": "-inf" })).ok(), "f32 [2]");
    assert_eq!(
        f(json!({ "shape": [2], "value": 7, "dtype": "i32" }))
            .info()
            .attrs
            .scalar("value"),
        Some(Scalar::Int(7))
    );
    f(json!({ "shape": [2], "value": 7.5, "dtype": "i32" })).err("invalid_attr");
    f(json!({ "shape": [2] })).err("missing_attr");

    let fl = |dtype: &str, attrs: Value| {
        case("core.full_like")
            .x(dtype, json!(["batch", 3]))
            .attrs(attrs)
    };
    assert_eq!(fl("bf16", json!({ "value": 0 })).ok(), "bf16 [batch, 3]");
    assert_eq!(
        fl("bf16", json!({ "value": 1, "dtype": "i64" })).ok(),
        "i64 [batch, 3]"
    );
    assert_eq!(
        fl("f16", json!({ "value": 0 })).info().attrs.dtype("dtype"),
        Some(polypous_ir::Dtype::F16)
    );
    fl("i64", json!({ "value": 0.5 })).err("invalid_attr");
    fl("f32", json!({ "value": 1, "dtype": "bool" })).err("invalid_attr");
}

// ---- Reduction ----

#[test]
fn reductions() {
    for (op, dtype) in [
        ("core.sum", "i32"),
        ("core.mean", "f32"),
        ("core.max", "i64"),
        ("core.min", "f16"),
    ] {
        let r = |attrs: Value| case(op).x(dtype, json!(["batch", 3, 4])).attrs(attrs);
        assert_eq!(r(json!({})).ok(), format!("{dtype} [1]"));
        assert_eq!(
            r(json!({ "keepdim": true })).ok(),
            format!("{dtype} [1, 1, 1]")
        );
        assert_eq!(r(json!({ "dim": [-1, 0] })).ok(), format!("{dtype} [3]"));
        assert_eq!(
            r(json!({ "dim": [1], "keepdim": true })).ok(),
            format!("{dtype} [batch, 1, 4]")
        );
        assert_eq!(
            r(json!({})).info().attrs.axes("dim"),
            Some(&[0usize, 1, 2][..])
        );
        r(json!({ "dim": [] })).err("invalid_attr");
        r(json!({ "dim": [1, -2] })).err("invalid_attr");
        r(json!({ "dim": [3] })).err("invalid_attr");
        r(json!({ "dim": 1 })).err("invalid_attr");
        r(json!({ "keepdim": 1 })).err("invalid_attr");
    }
    case("core.mean")
        .x("i32", json!([3]))
        .err("unsupported_dtype");
    case("core.sum")
        .x("bool", json!([3]))
        .err("unsupported_dtype");
    for op in ["core.argmax", "core.argmin"] {
        let r = |attrs: Value| case(op).x("f32", json!(["batch", 10])).attrs(attrs);
        assert_eq!(r(json!({ "dim": -1 })).ok(), "i64 [batch]");
        assert_eq!(
            r(json!({ "dim": 1, "keepdim": true, "dtype": "i32" })).ok(),
            "i32 [batch, 1]"
        );
        assert_eq!(
            case(op)
                .x("i32", json!([5]))
                .attrs(json!({ "dim": 0 }))
                .ok(),
            "i64 [1]"
        );
        r(json!({})).err("missing_attr");
        assert_eq!(
            r(json!({ "dim": 0, "dtype": "float" })).err("invalid_attr"),
            "/nodes/0/attrs/dtype"
        );
        r(json!({ "dim": 0, "dtype": "f32" })).err("invalid_attr");
    }
}

// ---- Linear algebra, other ----

#[test]
fn matmul() {
    let m = |a: Value, b: Value| case("core.matmul").ab("f32", a, b);
    assert_eq!(
        m(json!(["batch", 784]), json!([784, 128])).ok(),
        "f32 [batch, 128]"
    );
    assert_eq!(
        m(json!([2, 1, 3, 4]), json!([5, 4, 6])).ok(),
        "f32 [2, 5, 3, 6]"
    );
    assert_eq!(
        m(json!(["b", "n", "k"]), json!(["k", "m"])).ok(),
        "f32 [b, n, m]"
    );
    assert_eq!(
        m(json!([3, 4]), json!([5, 6])).err("shape_mismatch"),
        "/nodes/0/inputs/b"
    );
    m(json!(["b", 3, 4]), json!(["c", 4, 6])).err("broadcast_mismatch");
    m(json!([4]), json!([4, 2])).err("rank_mismatch");
    case("core.matmul")
        .ab("i64", json!([2, 2]), json!([2, 2]))
        .err("unsupported_dtype");
    case("core.matmul")
        .input("a", "f32", json!([2, 2]))
        .input("b", "f16", json!([2, 2]))
        .err("dtype_mismatch");
}

#[test]
fn cast_softmax_cumsum() {
    assert_eq!(
        case("core.cast")
            .x("i64", json!([3]))
            .attrs(json!({ "dtype": "bool" }))
            .ok(),
        "bool [3]"
    );
    assert_eq!(
        case("core.cast")
            .x("i64", json!([3]))
            .attrs(json!({ "dtype": "float" }))
            .compute(ComputeDtype::F16)
            .ok(),
        "f16 [3]"
    );
    case("core.cast").x("i64", json!([3])).err("missing_attr");
    case("core.cast")
        .x("i64", json!([3]))
        .attrs(json!({ "dtype": 3 }))
        .err("invalid_attr");
    for op in ["core.softmax", "core.log_softmax"] {
        assert_eq!(
            case(op)
                .x("f32", json!(["batch", 10]))
                .attrs(json!({ "dim": -1 }))
                .ok(),
            "f32 [batch, 10]"
        );
        assert_eq!(
            case(op)
                .x("f32", json!(["batch", 10]))
                .attrs(json!({ "dim": -1 }))
                .info()
                .attrs
                .axis("dim"),
            Some(1)
        );
        case(op)
            .x("i32", json!([3]))
            .attrs(json!({ "dim": 0 }))
            .err("unsupported_dtype");
        case(op)
            .x("f32", json!([3]))
            .attrs(json!({ "dim": 1 }))
            .err("invalid_attr");
    }
    assert_eq!(
        case("core.cumsum")
            .x("i32", json!([4, 5]))
            .attrs(json!({ "dim": 1 }))
            .ok(),
        "i32 [4, 5]"
    );
    case("core.cumsum")
        .x("bool", json!([4]))
        .attrs(json!({ "dim": 0 }))
        .err("unsupported_dtype");
}

#[test]
fn top_k() {
    let t = |shape: Value, attrs: Value| case("core.top_k").x("f32", shape).attrs(attrs);
    assert_eq!(
        t(json!([4, 10]), json!({ "k": 3 })).port("values").ok(),
        "f32 [4, 3]"
    );
    assert_eq!(
        t(json!([4, 10]), json!({ "k": 3 })).port("indices").ok(),
        "i64 [4, 3]"
    );
    assert_eq!(
        t(json!([4, 10]), json!({ "k": 2, "dim": 0, "dtype": "i32" }))
            .port("indices")
            .ok(),
        "i32 [2, 10]"
    );
    let info = t(json!(["batch", "vocab"]), json!({ "k": 5 }))
        .port("values")
        .info();
    assert_eq!(info.outputs[0].1.shape_string(), "[batch, 5]");
    assert_eq!(info.checks.len(), 1, "k <= vocab is a runtime check");
    assert_eq!(
        t(json!([4, 10]), json!({ "k": 11 }))
            .port("values")
            .err("shape_mismatch"),
        "/nodes/0/attrs/k"
    );
    t(json!([4, 10]), json!({ "k": 0 }))
        .port("values")
        .err("invalid_attr");
    t(json!([4, 10]), json!({ "k": 2, "dtype": "float" }))
        .port("values")
        .err("invalid_attr");
    case("core.top_k")
        .x("i64", json!([4]))
        .attrs(json!({ "k": 1 }))
        .port("values")
        .err("unsupported_dtype");
    // A top_k has no `out` port.
    let diags = t(json!([4]), json!({ "k": 1 })).diagnostics();
    assert!(diags.iter().any(|d| d.code.as_str() == "unknown_reference"));
}

// ---- nn ----

#[test]
fn linear() {
    let l = |x: Value| {
        case("nn.linear")
            .x("float", x)
            .param("weight", json!([128, 784]))
            .param("bias", json!([128]))
    };
    assert_eq!(l(json!(["batch", 784])).ok(), "f32 [batch, 128]");
    assert_eq!(l(json!([784])).ok(), "f32 [128]");
    assert_eq!(
        l(json!(["batch", 784])).compute(ComputeDtype::Bf16).ok(),
        "bf16 [batch, 128]"
    );
    assert_eq!(
        case("nn.linear")
            .x("f32", json!([2, 3]))
            .param("weight", json!([4, 3]))
            .ok(),
        "f32 [2, 4]"
    );
    assert_eq!(
        l(json!(["batch", 783])).err("shape_mismatch"),
        "/nodes/0/inputs/x"
    );
    l(json!(["batch", "k"])).err("shape_mismatch");
    // `x` must have the compute dtype.
    assert_eq!(
        case("nn.linear")
            .x("f16", json!([2, 3]))
            .param("weight", json!([4, 3]))
            .err("dtype_mismatch"),
        "/nodes/0/inputs/x"
    );
    case("nn.linear")
        .x("i32", json!([2, 3]))
        .param("weight", json!([4, 3]))
        .err("unsupported_dtype");
    assert_eq!(
        case("nn.linear")
            .x("f32", json!([2, 3]))
            .param("weight", json!([4, 3, 1]))
            .err("rank_mismatch"),
        "/nodes/0/params/weight"
    );
    assert_eq!(
        case("nn.linear")
            .x("f32", json!([2, 3]))
            .param("weight", json!([4, 3]))
            .param("bias", json!([3]))
            .err("shape_mismatch"),
        "/nodes/0/params/bias"
    );
    case("nn.linear")
        .x("f32", json!([2, 3]))
        .err("missing_param");
    case("nn.linear")
        .x("f32", json!([2, 3]))
        .param("weight", json!([4, 3]))
        .param("scale", json!([4]))
        .err("unknown_param");
}

#[test]
fn embedding() {
    let e = |dtype: &str, ids: Value| {
        case("nn.embedding")
            .input("ids", dtype, ids)
            .param("weight", json!([32000, 64]))
    };
    assert_eq!(
        e("i64", json!(["batch", "seq"])).ok(),
        "f32 [batch, seq, 64]"
    );
    assert_eq!(e("i32", json!([7])).ok(), "f32 [7, 64]");
    e("f32", json!([7])).err("unsupported_dtype");
    e("i64", json!([1, 1, 1, 1, 1, 1])).err("rank_too_large");
    assert_eq!(
        e("i64", json!([1, 1, 1, 1, 1])).ok(),
        "f32 [1, 1, 1, 1, 1, 64]"
    );
}

#[test]
fn conv() {
    let c2 = |x: Value, w: Value, attrs: Value| {
        case("nn.conv2d")
            .x("f32", x)
            .param("weight", w)
            .param("bias", json!([8]))
            .attrs(attrs)
    };
    assert_eq!(
        c2(json!([1, 3, 32, 32]), json!([8, 3, 3, 3]), json!({})).ok(),
        "f32 [1, 8, 30, 30]"
    );
    assert_eq!(
        c2(
            json!([1, 3, 32, 32]),
            json!([8, 3, 3, 3]),
            json!({ "stride": [2, 2], "padding": [1, 1] })
        )
        .ok(),
        "f32 [1, 8, 16, 16]"
    );
    // Symbolic H and W: (H + 2*1 - 1*(3 - 1) - 1) // 2 + 1 = (H - 1) // 2 + 1, which normalizes
    // to (H + 1) // 2; W: (W + 0 - 2*(3 - 1) - 1) // 1 + 1 = W - 4.
    assert_eq!(
        c2(
            json!(["batch", 3, "H", "W"]),
            json!([8, 3, 3, 3]),
            json!({ "stride": [2, 1], "padding": [1, 0], "dilation": [1, 2] })
        )
        .ok(),
        "f32 [batch, 8, (H + 1) // 2, W - 4]"
    );
    assert_eq!(
        c2(
            json!([1, 4, 8, 8]),
            json!([8, 2, 1, 1]),
            json!({ "groups": 2 })
        )
        .ok(),
        "f32 [1, 8, 8, 8]"
    );
    c2(
        json!([1, 3, 8, 8]),
        json!([8, 3, 3, 3]),
        json!({ "groups": 3 }),
    )
    .err("invalid_attr");
    c2(json!([1, 4, 8, 8]), json!([8, 3, 3, 3]), json!({})).err("shape_mismatch");
    c2(json!([1, 3, 2, 2]), json!([8, 3, 3, 3]), json!({})).err("nonpositive_dim");
    c2(
        json!([1, 3, 8, 8]),
        json!([8, 3, 3, 3]),
        json!({ "stride": [0, 1] }),
    )
    .err("invalid_attr");
    c2(
        json!([1, 3, 8, 8]),
        json!([8, 3, 3, 3]),
        json!({ "stride": 2 }),
    )
    .err("invalid_attr");
    c2(
        json!([1, 3, 8, 8]),
        json!([8, 3, 3, 3]),
        json!({ "padding": [-1, 0] }),
    )
    .err("invalid_attr");
    c2(json!([3, 8, 8]), json!([8, 3, 3, 3]), json!({})).err("rank_mismatch");
    c2(json!([1, 3, 8, 8]), json!([8, 3, 3]), json!({})).err("rank_mismatch");

    let c1 = |x: Value, attrs: Value| {
        case("nn.conv1d")
            .x("f32", x)
            .param("weight", json!([16, 4, 5]))
            .attrs(attrs)
    };
    assert_eq!(c1(json!([2, 4, 100]), json!({})).ok(), "f32 [2, 16, 96]");
    assert_eq!(
        c1(json!(["n", 4, "L"]), json!({ "stride": 2, "padding": 2 })).ok(),
        "f32 [n, 16, (L + 1) // 2]"
    );
    c1(json!([2, 4, 100]), json!({ "stride": [2] })).err("invalid_attr");
    c1(json!([2, 4, 100, 1]), json!({})).err("rank_mismatch");
}

#[test]
fn pools() {
    let mp = |x: Value, attrs: Value| case("nn.max_pool2d").x("f32", x).attrs(attrs);
    assert_eq!(
        mp(json!([1, 8, 32, 32]), json!({ "kernel_size": [2, 2] })).ok(),
        "f32 [1, 8, 16, 16]"
    );
    assert_eq!(
        mp(json!([1, 8, 32, 32]), json!({ "kernel_size": [2, 2] }))
            .info()
            .attrs
            .ints("stride"),
        Some(&[2i64, 2][..])
    );
    assert_eq!(
        mp(
            json!(["n", "c", "H", 9]),
            json!({ "kernel_size": [3, 3], "stride": [1, 2], "padding": [1, 1], "dilation": [2, 1] })
        )
        .ok(),
        "f32 [n, c, H - 2, 5]"
    );
    mp(
        json!([1, 8, 32, 32]),
        json!({ "kernel_size": [2, 2], "padding": [2, 0] }),
    )
    .err("invalid_attr");
    mp(json!([1, 8, 32, 32]), json!({})).err("missing_attr");
    mp(json!([1, 8, 32, 32]), json!({ "kernel_size": [2] })).err("invalid_attr");
    mp(json!([1, 8, 1, 1]), json!({ "kernel_size": [2, 2] })).err("nonpositive_dim");

    let ap = |attrs: Value| {
        case("nn.avg_pool2d")
            .x("f32", json!([1, 8, 7, 7]))
            .attrs(attrs)
    };
    assert_eq!(
        ap(json!({ "kernel_size": [3, 3], "stride": [2, 2], "padding": [1, 1] })).ok(),
        "f32 [1, 8, 4, 4]"
    );
    ap(json!({ "kernel_size": [3, 3], "dilation": [1, 1] })).err("unknown_attr");
    ap(json!({ "kernel_size": [3, 3], "count_include_pad": "no" })).err("invalid_attr");

    let aap = |attrs: Value| {
        case("nn.adaptive_avg_pool2d")
            .x("f32", json!(["batch", 8, "H", "W"]))
            .attrs(attrs)
    };
    assert_eq!(
        aap(json!({ "output_size": [1, 1] })).ok(),
        "f32 [batch, 8, 1, 1]"
    );
    aap(json!({ "output_size": [0, 1] })).err("invalid_attr");
    aap(json!({})).err("missing_attr");
}

#[test]
fn dropout_and_activations() {
    let d = |attrs: Value| case("nn.dropout").x("f32", json!([3])).attrs(attrs);
    assert_eq!(d(json!({})).ok(), "f32 [3]");
    assert_eq!(d(json!({ "p": 0 })).ok(), "f32 [3]");
    assert_eq!(d(json!({})).info().attrs.float("p"), Some(0.5));
    d(json!({ "p": 1.0 })).err("invalid_attr");
    d(json!({ "p": -0.1 })).err("invalid_attr");
    d(json!({ "p": "nan" })).err("invalid_attr");

    for op in [
        "relu",
        "gelu",
        "gelu_tanh",
        "silu",
        "sigmoid",
        "mish",
        "leaky_relu",
        "softplus",
    ] {
        let op = format!("nn.{op}");
        assert_eq!(
            case(&op).x("float", json!(["batch", 4])).ok(),
            "f32 [batch, 4]"
        );
        case(&op).x("i64", json!([3])).err("unsupported_dtype");
        case(&op).x("f16", json!([3])).err("dtype_mismatch");
        assert_eq!(
            case(&op)
                .x("f16", json!([3]))
                .compute(ComputeDtype::F16)
                .ok(),
            "f16 [3]"
        );
    }
    assert_eq!(
        case("nn.leaky_relu")
            .x("f32", json!([3]))
            .attrs(json!({ "negative_slope": 0.2 }))
            .info()
            .attrs
            .float("negative_slope"),
        Some(0.2)
    );
    let sp = |attrs: Value| case("nn.softplus").x("f32", json!([3])).attrs(attrs);
    assert_eq!(sp(json!({})).info().attrs.float("threshold"), Some(20.0));
    sp(json!({ "beta": 0 })).err("invalid_attr");
    sp(json!({ "threshold": "-inf" })).err("invalid_attr");
    case("nn.relu")
        .x("f32", json!([3]))
        .attrs(json!({ "inplace": true }))
        .err("unknown_attr");
}

#[test]
fn normalization() {
    let bn = |x: Value| {
        case("nn.batch_norm")
            .x("f32", x)
            .param("running_mean", json!([8]))
            .param("running_var", json!([8]))
            .param("weight", json!([8]))
    };
    assert_eq!(
        bn(json!(["batch", 8, "H", "W"])).ok(),
        "f32 [batch, 8, H, W]"
    );
    assert_eq!(bn(json!([2, 8])).ok(), "f32 [2, 8]");
    bn(json!([2, 4])).err("shape_mismatch");
    bn(json!([2, "c"])).err("shape_mismatch");
    bn(json!([8])).err("rank_mismatch");
    case("nn.batch_norm")
        .x("f32", json!([2, 8]))
        .param("running_mean", json!([8]))
        .err("missing_param");
    case("nn.batch_norm")
        .x("f32", json!([2, 8]))
        .param("running_mean", json!([8]))
        .param("running_var", json!([4]))
        .err("shape_mismatch");
    case("nn.batch_norm")
        .x("f32", json!([2, 8]))
        .param("running_mean", json!([8]))
        .param("running_var", json!([8]))
        .attrs(json!({ "eps": 0 }))
        .err("invalid_attr");

    let ln = |x: Value, attrs: Value| case("nn.layer_norm").x("f32", x).attrs(attrs);
    assert_eq!(
        ln(
            json!(["batch", "seq", 64]),
            json!({ "normalized_shape": [64] })
        )
        .ok(),
        "f32 [batch, seq, 64]"
    );
    let with_weight = ln(json!(["batch", 4, 64]), json!({}))
        .param("weight", json!([4, 64]))
        .param("bias", json!([4, 64]));
    assert_eq!(with_weight.ok(), "f32 [batch, 4, 64]");
    assert_eq!(
        with_weight.info().attrs.ints("normalized_shape"),
        Some(&[4i64, 64][..])
    );
    assert_eq!(
        ln(json!(["batch", 64]), json!({})).err("missing_attr"),
        "/nodes/0"
    );
    ln(json!(["batch", 64]), json!({}))
        .param("bias", json!([64]))
        .err("missing_attr");
    ln(json!(["batch", 64]), json!({ "normalized_shape": [32] })).err("shape_mismatch");
    ln(
        json!(["batch", 64]),
        json!({ "normalized_shape": [1, 1, 64] }),
    )
    .err("invalid_attr");
    ln(json!(["batch", 64]), json!({ "normalized_shape": [64] }))
        .param("weight", json!([32]))
        .err("shape_mismatch");
    ln(json!(["batch", "d"]), json!({ "normalized_shape": [64] })).err("shape_mismatch");

    let gn = |x: Value, attrs: Value| case("nn.group_norm").x("f32", x).attrs(attrs);
    assert_eq!(
        gn(json!(["batch", 32, "H", "W"]), json!({ "num_groups": 8 }))
            .param("weight", json!([32]))
            .ok(),
        "f32 [batch, 32, H, W]"
    );
    gn(json!([2, 32]), json!({ "num_groups": 5 })).err("invalid_attr");
    gn(json!([2, "c"]), json!({ "num_groups": 1 })).err("shape_mismatch");
    gn(json!([2, 32]), json!({})).err("missing_attr");
    gn(json!([2, 32]), json!({ "num_groups": 0 })).err("invalid_attr");

    let rn = |attrs: Value| {
        case("nn.rms_norm")
            .x("f32", json!(["batch", 16]))
            .attrs(attrs)
    };
    assert_eq!(
        rn(json!({ "eps": 1e-6 })).param("weight", json!([16])).ok(),
        "f32 [batch, 16]"
    );
    rn(json!({ "normalized_shape": [16] })).err("missing_attr");
    rn(json!({ "eps": 1e-6 })).err("missing_attr");
    rn(json!({ "eps": 1e-6, "normalized_shape": [16] }))
        .param("bias", json!([16]))
        .err("unknown_param");
}

// ---- General rules ----

#[test]
fn broadcasting_symbols() {
    let add = |a: Value, b: Value| case("core.add").ab("f32", a, b);
    assert_eq!(
        add(json!(["batch", 4]), json!([1, 4])).ok(),
        "f32 [batch, 4]"
    );
    assert_eq!(
        add(json!(["batch", 1]), json!(["seq"])).ok(),
        "f32 [batch, seq]"
    );
    assert_eq!(
        add(json!(["batch", 4]), json!(["batch", 4])).ok(),
        "f32 [batch, 4]"
    );
    // A symbol never broadcasts against a different symbol or a literal other than 1.
    add(json!(["batch", 4]), json!(["seq", 4])).err("broadcast_mismatch");
    add(json!(["batch"]), json!([4])).err("broadcast_mismatch");
}

#[test]
fn every_op_is_exercised() {
    // Keep this file in step with the catalog: every op name appears above.
    let source = include_str!("ops.rs");
    for op in catalog::OPS {
        let short = op.name.split_once('.').unwrap().1;
        assert!(
            source.contains(&format!("\"{}\"", op.name))
                || source.contains(&format!("\"{short}\"")),
            "{} has no test case",
            op.name
        );
    }
}

#[test]
fn attribute_values_are_normalized() {
    let info = case("core.slice")
        .x("f32", json!([10, 4]))
        .attrs(json!({ "dim": -2 }))
        .info();
    let attrs: Vec<(&str, &AttrValue)> = info.attrs.iter().collect();
    assert_eq!(
        attrs.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        vec!["dim", "start", "end", "step"]
    );
    assert_eq!(info.attrs.axis("dim"), Some(0));
    assert_eq!(info.attrs.dim("end").unwrap().to_string(), "10");
    assert_eq!(info.attrs.int("step"), Some(1));
}
