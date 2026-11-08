//! Saving graph documents in the canonical form, `docs/specs/graph-ir.md` §6.
//!
//! Key order: fields defined by the schema are written in schema order (the order of the field
//! tables in this module, which follows §2); maps whose keys are names (opsets, dims, node inputs,
//! params and attrs, carries, state updates) keep their stored (document) order; `meta` is written
//! exactly as stored. Optional fields are written only if present; optional lists and maps only if
//! non-empty. Layout: see [`crate::json::write_value`].

use crate::json::{Number, Value, write_value};
use crate::model::{
    Body, Dim, FORMAT, Graph, GraphInput, GraphOutput, Init, InputRef, Node, NodeKind, Param,
    StateEntry, StateInit,
};

/// Writes `graph` as canonical JSON (UTF-8, two-space indent, trailing newline).
pub fn to_json(graph: &Graph) -> String {
    let mut out = String::new();
    write_value(&mut out, &graph_value(graph), 0);
    out.push('\n');
    out
}

fn s(v: &str) -> Value {
    Value::String(v.to_owned())
}

fn uint(v: u64) -> Value {
    Value::Number(Number::PosInt(v))
}

fn int(v: i64) -> Value {
    Value::Number(if v < 0 {
        Number::NegInt(v)
    } else {
        Number::PosInt(v.unsigned_abs())
    })
}

fn float(v: f64) -> Value {
    Value::Number(Number::Float(v))
}

fn string_map(entries: &[(String, String)]) -> Value {
    Value::Object(entries.iter().map(|(k, v)| (k.clone(), s(v))).collect())
}

struct Obj(Vec<(String, Value)>);

impl Obj {
    fn new() -> Obj {
        Obj(Vec::new())
    }

    fn put(&mut self, key: &str, v: Value) {
        self.0.push((key.to_owned(), v));
    }

    fn opt(&mut self, key: &str, v: Option<Value>) {
        if let Some(v) = v {
            self.put(key, v);
        }
    }

    fn list<T>(&mut self, key: &str, items: &[T], f: impl Fn(&T) -> Value) {
        if !items.is_empty() {
            self.put(key, Value::Array(items.iter().map(f).collect()));
        }
    }

    fn map(&mut self, key: &str, entries: &[(String, String)]) {
        if !entries.is_empty() {
            self.put(key, string_map(entries));
        }
    }

    fn done(self) -> Value {
        Value::Object(self.0)
    }
}

fn graph_value(g: &Graph) -> Value {
    let mut o = Obj::new();
    o.put("format", s(FORMAT));
    o.put("format_version", uint(g.format_version));
    o.put(
        "opsets",
        Value::Object(
            g.opsets
                .iter()
                .map(|(k, v)| (k.clone(), uint(*v)))
                .collect(),
        ),
    );
    o.opt("name", g.name.as_deref().map(s));
    o.opt("meta", g.meta.clone());
    if !g.dims.is_empty() {
        let dims = g
            .dims
            .iter()
            .map(|d| {
                let mut e = Obj::new();
                e.opt("min", d.min.map(int));
                e.opt("max", d.max.map(int));
                (d.name.clone(), e.done())
            })
            .collect();
        o.put("dims", Value::Object(dims));
    }
    o.list("inputs", &g.inputs, input_value);
    o.put(
        "outputs",
        Value::Array(g.outputs.iter().map(output_value).collect()),
    );
    o.list("params", &g.params, param_value);
    o.list("state", &g.state, state_value);
    o.put(
        "nodes",
        Value::Array(g.nodes.iter().map(node_value).collect()),
    );
    o.map("state_updates", &g.state_updates);
    o.done()
}

fn shape_value(shape: &[Dim]) -> Value {
    Value::Array(
        shape
            .iter()
            .map(|d| match d {
                Dim::Literal(v) => int(*v),
                Dim::Expr(e) => s(e),
            })
            .collect(),
    )
}

fn input_value(i: &GraphInput) -> Value {
    let mut o = Obj::new();
    o.put("name", s(&i.name));
    o.put("dtype", s(i.dtype.as_str()));
    o.put("shape", shape_value(&i.shape));
    o.opt("meta", i.meta.clone());
    o.done()
}

fn output_value(out: &GraphOutput) -> Value {
    let mut o = Obj::new();
    o.put("name", s(&out.name));
    o.put("value", s(&out.value));
    o.opt("meta", out.meta.clone());
    o.done()
}

fn param_value(p: &Param) -> Value {
    let mut o = Obj::new();
    o.put("name", s(&p.name));
    o.put("dtype", s(p.dtype.as_str()));
    o.put("shape", shape_value(&p.shape));
    o.opt("init", p.init.as_ref().map(init_value));
    o.opt("trainable", p.trainable.map(Value::Bool));
    o.opt("meta", p.meta.clone());
    o.done()
}

fn init_value(init: &Init) -> Value {
    let mut o = Obj::new();
    o.put("kind", s(init.kind()));
    match *init {
        Init::Constant { value } => o.put("value", float(value)),
        Init::Uniform { low, high } => {
            o.put("low", float(low));
            o.put("high", float(high));
        }
        Init::Normal { mean, std } => {
            o.put("mean", float(mean));
            o.put("std", float(std));
        }
        Init::KaimingUniform { a } => o.opt("a", a.map(float)),
        Init::Zeros
        | Init::Ones
        | Init::XavierUniform
        | Init::XavierNormal
        | Init::KaimingNormal => {}
    }
    o.done()
}

fn state_value(st: &StateEntry) -> Value {
    let mut o = Obj::new();
    o.put("name", s(&st.name));
    o.put("dtype", s(st.dtype.as_str()));
    o.put("shape", shape_value(&st.shape));
    o.opt(
        "init",
        st.init.map(|i| match i {
            StateInit::Zeros => s("zeros"),
        }),
    );
    o.done()
}

fn node_value(n: &Node) -> Value {
    let mut o = Obj::new();
    o.put("id", s(&n.id));
    match &n.kind {
        NodeKind::Op(op) => {
            o.put("op", s(&op.op));
            if !op.inputs.is_empty() {
                let inputs = op
                    .inputs
                    .iter()
                    .map(|(k, r)| {
                        let v = match r {
                            InputRef::One(x) => s(x),
                            InputRef::Many(xs) => Value::Array(xs.iter().map(|x| s(x)).collect()),
                        };
                        (k.clone(), v)
                    })
                    .collect();
                o.put("inputs", Value::Object(inputs));
            }
            o.map("params", &op.params);
            if !op.attrs.is_empty() {
                o.put("attrs", Value::Object(op.attrs.clone()));
            }
        }
        NodeKind::Repeat(r) => {
            o.put("op", s("repeat"));
            o.put("count", int(r.count));
            o.put("carry", string_map(&r.carry));
            o.put("body", body_value(&r.body));
        }
    }
    o.opt("meta", n.meta.clone());
    o.done()
}

fn body_value(b: &Body) -> Value {
    let mut o = Obj::new();
    o.opt("params_prefix", b.params_prefix.as_deref().map(s));
    o.opt("state_prefix", b.state_prefix.as_deref().map(s));
    o.list("state", &b.state, state_value);
    o.put(
        "nodes",
        Value::Array(b.nodes.iter().map(node_value).collect()),
    );
    o.map("state_updates", &b.state_updates);
    o.put("carry_out", string_map(&b.carry_out));
    o.done()
}
