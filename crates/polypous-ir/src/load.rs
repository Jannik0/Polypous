//! Loading graph documents: JSON limits, schema conversion into [`Graph`], then structural
//! validation (`docs/specs/graph-ir.md` §4.1 checks 1–5, 9 and the structural parts of 8).
//!
//! Conversion collects every schema error (wrong JSON type, unknown or missing field, bad enum
//! value) with its JSON pointer instead of stopping at the first. Errors that make later checks
//! meaningless stop early: a JSON syntax error, a resource limit, an unsupported format.

use std::collections::HashSet;

use crate::diag::{Diagnostic, ErrorCode, ValidationReport};
use crate::dtype::DtypeSpec;
use crate::json::{self, Number, ParseError, Value, push_pointer};
use crate::limits;
use crate::model::{
    Body, Dim, DimDecl, FORMAT, FORMAT_VERSION, Graph, GraphInput, GraphOutput, Init, InputRef,
    Node, NodeKind, OpNode, Param, Repeat, StateEntry, StateInit,
};
use crate::validate::{self, ValidateOptions};

/// The result of [`load`].
#[derive(Debug, Clone, PartialEq)]
pub struct LoadOutcome {
    /// The graph, if the document converted into the typed model without loss (no JSON, schema or
    /// limit errors). It may still have validation errors: check [`LoadOutcome::report`].
    pub graph: Option<Graph>,
    /// Every error found.
    pub report: ValidationReport,
}

/// Loads and validates a graph document.
///
/// Never panics, on any input. The document size is checked before parsing, the JSON nesting depth
/// before the parser recurses, and the repeat limits before anything is expanded.
///
/// Parsing recurses once per JSON nesting level (at most 128). A document at the depth limit needs
/// less than 256 KiB of stack in debug builds (measured), so call this from a thread with at least
/// that much (Rust's default for spawned threads is 2 MiB).
///
/// Memory is bounded: the pre-scan rejects documents with more than 4,000,000 JSON values before
/// the parser allocates, which bounds the parsed tree to about 128 MiB.
pub fn load(bytes: &[u8], options: &ValidateOptions) -> LoadOutcome {
    let mut report = ValidationReport::default();
    let value = match json::parse(bytes) {
        Ok(v) => v,
        Err(e) => {
            let code = match e {
                ParseError::TooLarge { .. } | ParseError::TooManyValues { .. } => {
                    ErrorCode::DocumentTooLarge
                }
                ParseError::TooDeep { .. } => ErrorCode::JsonTooDeep,
                ParseError::Syntax(_) => ErrorCode::InvalidJson,
            };
            report.push(Diagnostic::new(code, "", e.to_string()));
            return LoadOutcome {
                graph: None,
                report,
            };
        }
    };
    let mut cx = Cx::default();
    check_strings_and_keys(&value, &mut Vec::new(), &mut cx);
    if cx.stop {
        return LoadOutcome {
            graph: None,
            report: cx.report,
        };
    }
    let graph = convert_graph(&value, &mut cx);
    let Some(graph) = graph else {
        return LoadOutcome {
            graph: None,
            report: cx.report,
        };
    };
    if cx.stop {
        return LoadOutcome {
            graph: None,
            report: cx.report,
        };
    }
    let structural = validate::validate_structure(&graph, options, !cx.lossy);
    // A location the conversion already reported (a missing or malformed field) would only repeat
    // the same problem from the placeholder the conversion left there.
    let reported: HashSet<String> = cx
        .report
        .diagnostics
        .iter()
        .map(|d| d.pointer.clone())
        .collect();
    let mut report = cx.report;
    for d in structural.diagnostics {
        if !reported.contains(&d.pointer) {
            report.push(d);
        }
    }
    LoadOutcome {
        graph: (!cx.lossy).then_some(graph),
        report,
    }
}

#[derive(Default)]
struct Cx {
    report: ValidationReport,
    /// Some part of the document could not be represented in the typed graph.
    lossy: bool,
    /// A limit or format error: skip the remaining checks.
    stop: bool,
}

impl Cx {
    fn err(&mut self, code: ErrorCode, pointer: impl Into<String>, message: impl Into<String>) {
        self.lossy = true;
        self.report.push(Diagnostic::new(code, pointer, message));
    }

    fn type_err(&mut self, pointer: &str, expected: &str, got: &Value) {
        self.err(
            ErrorCode::InvalidJson,
            pointer,
            format!("expected {expected}, got {}", got.type_name()),
        );
    }
}

/// One step of a JSON pointer, materialized only when an error is reported.
#[derive(Clone, Copy)]
enum Seg<'a> {
    Key(&'a str),
    Index(usize),
}

fn pointer_of(path: &[Seg<'_>]) -> String {
    let mut p = String::new();
    for seg in path {
        p = match seg {
            Seg::Key(k) => push_pointer(&p, k),
            Seg::Index(i) => push_pointer(&p, &i.to_string()),
        };
    }
    p
}

/// Checks the length of every string and key, and rejects duplicate keys (everywhere, `meta`
/// included). Recursion depth is bounded by the JSON depth limit.
fn check_strings_and_keys<'a>(value: &'a Value, path: &mut Vec<Seg<'a>>, cx: &mut Cx) {
    match value {
        Value::String(s) if s.len() > limits::MAX_STRING_BYTES => {
            cx.err(
                ErrorCode::StringTooLong,
                pointer_of(path),
                format!(
                    "string is {} bytes, more than the limit of {} bytes",
                    s.len(),
                    limits::MAX_STRING_BYTES
                ),
            );
            cx.stop = true;
        }
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                path.push(Seg::Index(i));
                check_strings_and_keys(item, path, cx);
                path.pop();
            }
        }
        Value::Object(members) => {
            let mut seen: HashSet<&str> = HashSet::with_capacity(members.len());
            for (key, item) in members {
                if key.len() > limits::MAX_STRING_BYTES {
                    cx.err(
                        ErrorCode::StringTooLong,
                        pointer_of(path),
                        format!(
                            "object key is {} bytes, more than the limit of {} bytes",
                            key.len(),
                            limits::MAX_STRING_BYTES
                        ),
                    );
                    cx.stop = true;
                    continue;
                }
                path.push(Seg::Key(key));
                if !seen.insert(key) {
                    cx.err(
                        ErrorCode::InvalidJson,
                        pointer_of(path),
                        format!("duplicate key `{key}`"),
                    );
                }
                check_strings_and_keys(item, path, cx);
                path.pop();
            }
        }
        _ => {}
    }
}

/// The members of a JSON object, with the unknown-field check done.
struct Fields<'a> {
    members: &'a [(String, Value)],
    pointer: String,
}

impl<'a> Fields<'a> {
    /// Checks that `value` is an object whose keys are all in `allowed`.
    fn new(
        value: &'a Value,
        pointer: &str,
        allowed: &[&str],
        what: &str,
        cx: &mut Cx,
    ) -> Option<Fields<'a>> {
        let Value::Object(members) = value else {
            cx.type_err(pointer, &format!("{what} (an object)"), value);
            return None;
        };
        for (key, _) in members {
            if !allowed.contains(&key.as_str()) {
                cx.err(
                    ErrorCode::UnknownField,
                    push_pointer(pointer, key),
                    format!("unknown field `{key}` in {what}"),
                );
            }
        }
        Some(Fields {
            members,
            pointer: pointer.to_owned(),
        })
    }

    /// The first member named `key` (duplicates are reported by [`check_strings_and_keys`]).
    fn get(&self, key: &str) -> Option<&'a Value> {
        self.members.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn ptr(&self, key: &str) -> String {
        push_pointer(&self.pointer, key)
    }

    fn required(&self, key: &str, what: &str, cx: &mut Cx) -> Option<&'a Value> {
        let v = self.get(key);
        if v.is_none() {
            cx.err(
                ErrorCode::MissingField,
                self.ptr(key),
                format!("{what} has no `{key}` field"),
            );
        }
        v
    }

    fn opt_str(&self, key: &str, cx: &mut Cx) -> Option<String> {
        let v = self.get(key)?;
        as_str(v, &self.ptr(key), cx).map(str::to_owned)
    }

    fn req_str(&self, key: &str, what: &str, cx: &mut Cx) -> Option<String> {
        let v = self.required(key, what, cx)?;
        as_str(v, &self.ptr(key), cx).map(str::to_owned)
    }

    fn opt_array(&self, key: &str, cx: &mut Cx) -> Option<&'a [Value]> {
        let v = self.get(key)?;
        as_array(v, &self.ptr(key), cx)
    }

    fn opt_meta(&self) -> Option<Value> {
        self.get("meta").cloned()
    }
}

fn as_str<'a>(v: &'a Value, pointer: &str, cx: &mut Cx) -> Option<&'a str> {
    match v {
        Value::String(s) => Some(s),
        _ => {
            cx.type_err(pointer, "a string", v);
            None
        }
    }
}

fn as_array<'a>(v: &'a Value, pointer: &str, cx: &mut Cx) -> Option<&'a [Value]> {
    match v {
        Value::Array(a) => Some(a),
        _ => {
            cx.type_err(pointer, "an array", v);
            None
        }
    }
}

fn as_object<'a>(v: &'a Value, pointer: &str, cx: &mut Cx) -> Option<&'a [(String, Value)]> {
    match v {
        Value::Object(m) => Some(m),
        _ => {
            cx.type_err(pointer, "an object", v);
            None
        }
    }
}

fn as_number(v: &Value, pointer: &str, cx: &mut Cx) -> Option<f64> {
    match v {
        Value::Number(n) => Some(n.as_f64()),
        _ => {
            cx.type_err(pointer, "a number", v);
            None
        }
    }
}

fn as_bool(v: &Value, pointer: &str, cx: &mut Cx) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        _ => {
            cx.type_err(pointer, "a boolean", v);
            None
        }
    }
}

/// An integer that fits in `i64`. Integers outside the range are `invalid_value`.
fn as_i64(v: &Value, pointer: &str, cx: &mut Cx) -> Option<i64> {
    match v {
        Value::Number(n) if n.is_integer() => match n.as_i64() {
            Some(i) => Some(i),
            None => {
                cx.err(ErrorCode::InvalidValue, pointer, "integer is out of range");
                None
            }
        },
        _ => {
            cx.type_err(pointer, "an integer", v);
            None
        }
    }
}

/// Checks the length of a list against a resource limit before its entries are converted.
fn check_len(items: &[Value], max: usize, pointer: &str, what: &str, cx: &mut Cx) -> bool {
    if items.len() > max {
        cx.err(
            ErrorCode::InvalidValue,
            pointer,
            format!("{} {what}, more than the limit of {max}", items.len()),
        );
        cx.stop = true;
        return false;
    }
    true
}

/// Converts a JSON object of string values (`state_updates`, `carry`, node `params`).
fn string_map(v: &Value, pointer: &str, cx: &mut Cx) -> Vec<(String, String)> {
    let Some(members) = as_object(v, pointer, cx) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(members.len());
    for (k, item) in members {
        if let Some(s) = as_str(item, &push_pointer(pointer, k), cx) {
            out.push((k.clone(), s.to_owned()));
        }
    }
    out
}

const ROOT_FIELDS: &[&str] = &[
    "format",
    "format_version",
    "opsets",
    "name",
    "meta",
    "dims",
    "inputs",
    "outputs",
    "params",
    "state",
    "nodes",
    "state_updates",
    "functions",
];

fn convert_graph(value: &Value, cx: &mut Cx) -> Option<Graph> {
    let Some(f) = Fields::new(value, "", ROOT_FIELDS, "the graph document", cx) else {
        cx.stop = true;
        return None;
    };

    // Format and version first: if they are wrong, the rest of the schema is unknown.
    match f.get("format") {
        None => {
            cx.err(
                ErrorCode::MissingField,
                "/format",
                "the graph document has no `format` field",
            );
            cx.stop = true;
        }
        Some(Value::String(s)) if s == FORMAT => {}
        Some(other) => {
            cx.err(
                ErrorCode::UnsupportedFormat,
                "/format",
                format!(
                    "`format` must be \"{FORMAT}\", got {}",
                    other.to_canonical_string()
                ),
            );
            cx.stop = true;
        }
    }
    let format_version = match f.get("format_version") {
        None => {
            cx.err(
                ErrorCode::MissingField,
                "/format_version",
                "the graph document has no `format_version` field",
            );
            cx.stop = true;
            0
        }
        Some(Value::Number(Number::PosInt(v))) if *v == FORMAT_VERSION => *v,
        Some(other) => {
            cx.err(
                ErrorCode::UnsupportedFormat,
                "/format_version",
                format!(
                    "`format_version` {} is not supported (supported: {FORMAT_VERSION})",
                    other.to_canonical_string()
                ),
            );
            cx.stop = true;
            0
        }
    };
    if cx.stop {
        return None;
    }

    if f.get("functions").is_some() {
        cx.err(
            ErrorCode::UnsupportedInVersion,
            "/functions",
            "`functions` are not supported in this version (planned for M2)",
        );
    }

    let mut opsets = Vec::new();
    if let Some(v) = f.required("opsets", "the graph document", cx)
        && let Some(members) = as_object(v, "/opsets", cx)
    {
        for (name, ver) in members {
            let p = push_pointer("/opsets", name);
            match ver {
                Value::Number(Number::PosInt(n)) => opsets.push((name.clone(), *n)),
                Value::Number(Number::NegInt(_)) => cx.err(
                    ErrorCode::UnsupportedOpset,
                    p,
                    format!("opset `{name}` has a negative version"),
                ),
                other => cx.type_err(&p, "an integer opset version", other),
            }
        }
    }

    let name = f.opt_str("name", cx);
    let meta = f.opt_meta();

    let mut dims = Vec::new();
    if let Some(v) = f.get("dims")
        && let Some(members) = as_object(v, "/dims", cx)
    {
        if members.len() > limits::MAX_DIMS {
            cx.err(
                ErrorCode::InvalidValue,
                "/dims",
                format!(
                    "{} dims declared, more than the limit of {}",
                    members.len(),
                    limits::MAX_DIMS
                ),
            );
            cx.stop = true;
            return None;
        }
        for (sym, decl) in members {
            let p = push_pointer("/dims", sym);
            if let Some(df) = Fields::new(decl, &p, &["min", "max"], "a dim declaration", cx) {
                let min = df.get("min").and_then(|v| as_i64(v, &df.ptr("min"), cx));
                let max = df.get("max").and_then(|v| as_i64(v, &df.ptr("max"), cx));
                dims.push(DimDecl {
                    name: sym.clone(),
                    min,
                    max,
                });
            }
        }
    }

    let mut inputs = Vec::new();
    if let Some(items) = f.opt_array("inputs", cx) {
        if !check_len(items, limits::MAX_GRAPH_IO, "/inputs", "graph inputs", cx) {
            return None;
        }
        for (i, item) in items.iter().enumerate() {
            if let Some(input) = convert_input(item, &format!("/inputs/{i}"), cx) {
                inputs.push(input);
            }
        }
    }

    let mut outputs = Vec::new();
    if let Some(v) = f.required("outputs", "the graph document", cx)
        && let Some(items) = as_array(v, "/outputs", cx)
    {
        if !check_len(items, limits::MAX_GRAPH_IO, "/outputs", "graph outputs", cx) {
            return None;
        }
        for (i, item) in items.iter().enumerate() {
            let p = format!("/outputs/{i}");
            if let Some(of) =
                Fields::new(item, &p, &["name", "value", "meta"], "a graph output", cx)
            {
                let name = of.req_str("name", "a graph output", cx).unwrap_or_default();
                let value = of
                    .req_str("value", "a graph output", cx)
                    .unwrap_or_default();
                outputs.push(GraphOutput {
                    name,
                    value,
                    meta: of.opt_meta(),
                });
            }
        }
    }

    let mut params = Vec::new();
    if let Some(items) = f.opt_array("params", cx) {
        for (i, item) in items.iter().enumerate() {
            if let Some(param) = convert_param(item, &format!("/params/{i}"), cx) {
                params.push(param);
            }
        }
    }

    let mut state = Vec::new();
    if let Some(items) = f.opt_array("state", cx) {
        if !check_len(items, limits::MAX_GRAPH_IO, "/state", "state entries", cx) {
            return None;
        }
        state = convert_state_list(items, "/state", cx);
    }

    let mut nodes = Vec::new();
    if let Some(v) = f.required("nodes", "the graph document", cx)
        && let Some(items) = as_array(v, "/nodes", cx)
    {
        nodes = convert_nodes(items, "/nodes", 0, cx);
    }

    let state_updates = f
        .get("state_updates")
        .map(|v| string_map(v, "/state_updates", cx))
        .unwrap_or_default();

    Some(Graph {
        format_version,
        opsets,
        name,
        meta,
        dims,
        inputs,
        outputs,
        params,
        state,
        nodes,
        state_updates,
    })
}

fn convert_dtype(v: Option<&Value>, pointer: &str, what: &str, cx: &mut Cx) -> DtypeSpec {
    let Some(v) = v else {
        cx.err(
            ErrorCode::MissingField,
            pointer,
            format!("{what} has no `dtype` field"),
        );
        return DtypeSpec::Float;
    };
    let Some(s) = as_str(v, pointer, cx) else {
        return DtypeSpec::Float;
    };
    match DtypeSpec::parse(s) {
        Some(d) => d,
        None => {
            cx.err(
                ErrorCode::InvalidValue,
                pointer,
                format!("unknown dtype `{s}`"),
            );
            DtypeSpec::Float
        }
    }
}

fn convert_shape(v: Option<&Value>, pointer: &str, what: &str, cx: &mut Cx) -> Vec<Dim> {
    let Some(v) = v else {
        cx.err(
            ErrorCode::MissingField,
            pointer,
            format!("{what} has no `shape` field"),
        );
        return Vec::new();
    };
    let Some(items) = as_array(v, pointer, cx) else {
        return Vec::new();
    };
    let mut dims = Vec::with_capacity(items.len().min(limits::MAX_RANK + 1));
    for (i, item) in items.iter().enumerate() {
        let p = push_pointer(pointer, &i.to_string());
        match item {
            Value::String(s) => dims.push(Dim::Expr(s.clone())),
            Value::Number(n) if n.is_integer() => match n.as_i64() {
                Some(v) => dims.push(Dim::Literal(v)),
                None => cx.err(ErrorCode::InvalidValue, p, "literal dim is out of range"),
            },
            other => cx.type_err(&p, "an integer or a dim expression string", other),
        }
    }
    dims
}

fn convert_input(item: &Value, pointer: &str, cx: &mut Cx) -> Option<GraphInput> {
    let f = Fields::new(
        item,
        pointer,
        &["name", "dtype", "shape", "meta"],
        "a graph input",
        cx,
    )?;
    Some(GraphInput {
        name: f.req_str("name", "a graph input", cx).unwrap_or_default(),
        dtype: convert_dtype(f.get("dtype"), &f.ptr("dtype"), "a graph input", cx),
        shape: convert_shape(f.get("shape"), &f.ptr("shape"), "a graph input", cx),
        meta: f.opt_meta(),
    })
}

fn convert_param(item: &Value, pointer: &str, cx: &mut Cx) -> Option<Param> {
    let f = Fields::new(
        item,
        pointer,
        &["name", "dtype", "shape", "init", "trainable", "meta"],
        "a param",
        cx,
    )?;
    let init = f
        .get("init")
        .and_then(|v| convert_init(v, &f.ptr("init"), cx));
    let trainable = f
        .get("trainable")
        .and_then(|v| as_bool(v, &f.ptr("trainable"), cx));
    Some(Param {
        name: f.req_str("name", "a param", cx).unwrap_or_default(),
        dtype: convert_dtype(f.get("dtype"), &f.ptr("dtype"), "a param", cx),
        shape: convert_shape(f.get("shape"), &f.ptr("shape"), "a param", cx),
        init,
        trainable,
        meta: f.opt_meta(),
    })
}

fn convert_init(v: &Value, pointer: &str, cx: &mut Cx) -> Option<Init> {
    let Value::Object(members) = v else {
        cx.type_err(pointer, "an init object", v);
        return None;
    };
    let kind_ptr = push_pointer(pointer, "kind");
    let Some(kind_v) = members.iter().find(|(k, _)| k == "kind").map(|(_, v)| v) else {
        cx.err(
            ErrorCode::MissingField,
            kind_ptr,
            "init has no `kind` field",
        );
        return None;
    };
    let kind = as_str(kind_v, &kind_ptr, cx)?;
    let fields: &[&str] = match kind {
        "zeros" | "ones" | "xavier_uniform" | "xavier_normal" | "kaiming_normal" => &["kind"],
        "constant" => &["kind", "value"],
        "uniform" => &["kind", "low", "high"],
        "normal" => &["kind", "mean", "std"],
        "kaiming_uniform" => &["kind", "a"],
        other => {
            cx.err(
                ErrorCode::InvalidValue,
                kind_ptr,
                format!("unknown init kind `{other}`"),
            );
            return None;
        }
    };
    let f = Fields::new(v, pointer, fields, &format!("init `{kind}`"), cx)?;
    let num = |key: &str, required: bool, cx: &mut Cx| -> Option<f64> {
        let v = if required {
            f.required(key, &format!("init `{kind}`"), cx)?
        } else {
            f.get(key)?
        };
        as_number(v, &f.ptr(key), cx)
    };
    Some(match kind {
        "zeros" => Init::Zeros,
        "ones" => Init::Ones,
        "xavier_uniform" => Init::XavierUniform,
        "xavier_normal" => Init::XavierNormal,
        "kaiming_normal" => Init::KaimingNormal,
        "constant" => Init::Constant {
            value: num("value", true, cx)?,
        },
        "uniform" => {
            let low = num("low", true, cx);
            let high = num("high", true, cx);
            Init::Uniform {
                low: low?,
                high: high?,
            }
        }
        "normal" => {
            let mean = num("mean", true, cx);
            let std = num("std", true, cx);
            Init::Normal {
                mean: mean?,
                std: std?,
            }
        }
        _ => Init::KaimingUniform {
            a: num("a", false, cx),
        },
    })
}

fn convert_state_list(items: &[Value], pointer: &str, cx: &mut Cx) -> Vec<StateEntry> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        let p = push_pointer(pointer, &i.to_string());
        let Some(f) = Fields::new(
            item,
            &p,
            &["name", "dtype", "shape", "init"],
            "a state entry",
            cx,
        ) else {
            continue;
        };
        let init = match f.get("init") {
            None => None,
            Some(Value::String(s)) if s == "zeros" => Some(StateInit::Zeros),
            Some(Value::String(s)) => {
                cx.err(
                    ErrorCode::InvalidValue,
                    f.ptr("init"),
                    format!("unknown state init `{s}` (M1 supports only \"zeros\")"),
                );
                None
            }
            Some(other) => {
                cx.type_err(&f.ptr("init"), "a string", other);
                None
            }
        };
        out.push(StateEntry {
            name: f.req_str("name", "a state entry", cx).unwrap_or_default(),
            dtype: convert_dtype(f.get("dtype"), &f.ptr("dtype"), "a state entry", cx),
            shape: convert_shape(f.get("shape"), &f.ptr("shape"), "a state entry", cx),
            init,
        });
    }
    out
}

fn convert_nodes(items: &[Value], pointer: &str, depth: usize, cx: &mut Cx) -> Vec<Node> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        if let Some(node) = convert_node(item, &push_pointer(pointer, &i.to_string()), depth, cx) {
            out.push(node);
        }
        if cx.stop {
            break;
        }
    }
    out
}

const OP_NODE_FIELDS: &[&str] = &["id", "op", "inputs", "params", "attrs", "meta"];
const REPEAT_FIELDS: &[&str] = &["id", "op", "count", "carry", "body", "meta"];
const BODY_FIELDS: &[&str] = &[
    "params_prefix",
    "state_prefix",
    "state",
    "nodes",
    "state_updates",
    "carry_out",
];

fn convert_node(item: &Value, pointer: &str, depth: usize, cx: &mut Cx) -> Option<Node> {
    let Value::Object(members) = item else {
        cx.type_err(pointer, "a node (an object)", item);
        return None;
    };
    let op = members.iter().find(|(k, _)| k == "op").map(|(_, v)| v);
    let op = match op {
        Some(Value::String(s)) => Some(s.as_str()),
        _ => None,
    };
    let allowed = match op {
        Some("repeat") => REPEAT_FIELDS,
        // `call` is unsupported in M1; its fields aren't checked or kept.
        Some("call") => {
            cx.lossy = true;
            &[]
        }
        _ => OP_NODE_FIELDS,
    };
    let f = if op == Some("call") {
        Fields {
            members,
            pointer: pointer.to_owned(),
        }
    } else {
        Fields::new(item, pointer, allowed, "a node", cx)?
    };
    let id = f.req_str("id", "a node", cx).unwrap_or_default();
    let op_name = f.req_str("op", "a node", cx).unwrap_or_default();
    let meta = f.opt_meta();
    let kind = if op_name == "repeat" {
        NodeKind::Repeat(convert_repeat(&f, depth, cx))
    } else if op_name == "call" {
        NodeKind::Op(OpNode {
            op: op_name,
            inputs: Vec::new(),
            params: Vec::new(),
            attrs: Vec::new(),
        })
    } else {
        let mut inputs = Vec::new();
        if let Some(v) = f.get("inputs") {
            let p = f.ptr("inputs");
            if let Some(members) = as_object(v, &p, cx) {
                for (name, r) in members {
                    let rp = push_pointer(&p, name);
                    match r {
                        Value::String(s) => inputs.push((name.clone(), InputRef::One(s.clone()))),
                        Value::Array(list) => {
                            let mut refs = Vec::with_capacity(list.len());
                            for (k, e) in list.iter().enumerate() {
                                if let Some(s) = as_str(e, &push_pointer(&rp, &k.to_string()), cx) {
                                    refs.push(s.to_owned());
                                }
                            }
                            inputs.push((name.clone(), InputRef::Many(refs)));
                        }
                        other => {
                            cx.type_err(&rp, "a value reference or an array of references", other)
                        }
                    }
                }
            }
        }
        let params = f
            .get("params")
            .map(|v| string_map(v, &f.ptr("params"), cx))
            .unwrap_or_default();
        let attrs = match f.get("attrs") {
            Some(v) => as_object(v, &f.ptr("attrs"), cx)
                .map(<[_]>::to_vec)
                .unwrap_or_default(),
            None => Vec::new(),
        };
        NodeKind::Op(OpNode {
            op: op_name,
            inputs,
            params,
            attrs,
        })
    };
    Some(Node { id, kind, meta })
}

fn convert_repeat(f: &Fields<'_>, depth: usize, cx: &mut Cx) -> Repeat {
    let count = match f.required("count", "a repeat node", cx) {
        Some(Value::Number(n)) if n.is_integer() => n.as_i64().unwrap_or_else(|| {
            // Out of the i64 range: kept as i64::MAX, which validation reports as
            // `invalid_repeat`; the graph no longer matches the document.
            cx.lossy = true;
            i64::MAX
        }),
        Some(other) => {
            cx.type_err(&f.ptr("count"), "an integer", other);
            1
        }
        None => 1,
    };
    let carry = f
        .required("carry", "a repeat node", cx)
        .map(|v| string_map(v, &f.ptr("carry"), cx))
        .unwrap_or_default();
    let mut body = Body {
        params_prefix: None,
        state_prefix: None,
        state: Vec::new(),
        nodes: Vec::new(),
        state_updates: Vec::new(),
        carry_out: Vec::new(),
    };
    let Some(bv) = f.required("body", "a repeat node", cx) else {
        return Repeat { count, carry, body };
    };
    let bp = f.ptr("body");
    if depth + 1 > limits::MAX_REPEAT_DEPTH {
        cx.err(
            ErrorCode::RepeatTooDeep,
            &bp,
            format!(
                "repeat blocks nest more than {} levels deep",
                limits::MAX_REPEAT_DEPTH
            ),
        );
        cx.stop = true;
        return Repeat { count, carry, body };
    }
    let Some(bf) = Fields::new(bv, &bp, BODY_FIELDS, "a repeat body", cx) else {
        return Repeat { count, carry, body };
    };
    body.params_prefix = bf.opt_str("params_prefix", cx);
    body.state_prefix = bf.opt_str("state_prefix", cx);
    if let Some(items) = bf.opt_array("state", cx) {
        if !check_len(
            items,
            limits::MAX_GRAPH_IO,
            &bf.ptr("state"),
            "state entries",
            cx,
        ) {
            return Repeat { count, carry, body };
        }
        body.state = convert_state_list(items, &bf.ptr("state"), cx);
    }
    if let Some(v) = bf.required("nodes", "a repeat body", cx)
        && let Some(items) = as_array(v, &bf.ptr("nodes"), cx)
    {
        body.nodes = convert_nodes(items, &bf.ptr("nodes"), depth + 1, cx);
    }
    body.state_updates = bf
        .get("state_updates")
        .map(|v| string_map(v, &bf.ptr("state_updates"), cx))
        .unwrap_or_default();
    body.carry_out = bf
        .required("carry_out", "a repeat body", cx)
        .map(|v| string_map(v, &bf.ptr("carry_out"), cx))
        .unwrap_or_default();
    Repeat { count, carry, body }
}
