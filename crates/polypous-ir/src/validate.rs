//! Structural validation of graph documents, `docs/specs/graph-ir.md` §4.1 checks 2–5, 9 and the
//! structural parts of 8 (state update targets, `repeat` carry keys).
//!
//! Dtype and shape inference (checks 6–7 and the dtype/shape parts of 8) and the per-op signature
//! checks of check 4 (inputs, params, attrs) need the op catalog and are #39. This pass only knows
//! op names ([`crate::ops`]).
//!
//! Resource limits (§4.3) are computed from the unexpanded document with checked arithmetic before
//! any other check; a repeat is never expanded to count it. Expanded names (param and state names
//! with `{i}` replaced) are generated only after the limits bound their number.

use std::collections::{HashMap, HashSet};

use crate::diag::{Diagnostic, ErrorCode, Item, ValidationReport};
use crate::dim::{DimError, DimExpr};
use crate::dtype::{ComputeDtype, DtypeSpec};
use crate::json::push_pointer;
use crate::limits;
use crate::model::{
    Body, Dim, FORMAT_VERSION, Graph, Init, InputRef, Node, NodeKind, OpNode, Repeat, StateEntry,
};
use crate::names;
use crate::ops::{self, Ports};

/// Options for validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ValidateOptions {
    /// The compute dtype that `float` inputs, state entries and `dtype` attributes resolve to
    /// (§2.2). Default `f32`. Structural validation only checks where `float` is allowed; dtype
    /// inference uses the resolved dtype.
    pub compute_dtype: ComputeDtype,
}

/// Validates a typed graph structurally. Assumes the graph came from conversion (repeat nesting is
/// bounded); [`Graph::validate`] goes through the canonical JSON to guarantee that.
pub(crate) fn validate_structure(graph: &Graph, _options: &ValidateOptions) -> ValidationReport {
    let mut v = Validator::new(graph);
    if !v.check_limits() {
        return v.report;
    }
    v.check_versions();
    v.check_dims();
    v.check_params();
    v.check_root();
    v.check_unbound();
    v.report
}

#[derive(Debug, Clone)]
enum NodePorts {
    Op(Ports),
    Carry(Vec<String>),
}

impl NodePorts {
    fn has(&self, port: &str) -> bool {
        match self {
            NodePorts::Op(p) => p.has(port),
            NodePorts::Carry(keys) => keys.iter().any(|k| k == port),
        }
    }

    fn has_out(&self) -> bool {
        match self {
            NodePorts::Op(p) => p.has_out(),
            NodePorts::Carry(_) => false,
        }
    }
}

#[derive(Debug, Clone)]
enum ValueKind {
    Input,
    State,
    Carry,
    Node { index: usize, ports: NodePorts },
}

/// One scope of the value namespace: the root, or a `repeat` body.
struct Frame {
    values: HashMap<String, ValueKind>,
    /// Dependency edges between the scope's nodes (by index): `adj[u]` lists nodes `u` uses.
    adj: Vec<Vec<usize>>,
    /// The node of this scope whose references are being resolved, if any.
    current: Option<usize>,
}

/// One enclosing `repeat` level, for expanding prefixes.
struct Level<'g> {
    /// `false` if the prefix is invalid (already reported): expanded params aren't checked.
    params_ok: bool,
    /// `false` if the state prefix is invalid or lacks `{i}` (already reported).
    state_ok: bool,
    params_prefix: &'g str,
    params_placeholders: usize,
    state_prefix: &'g str,
    count: u64,
}

struct Validator<'g> {
    graph: &'g Graph,
    report: ValidationReport,
    declared_dims: HashSet<&'g str>,
    params: HashSet<&'g str>,
    /// Symbols that appear as a plain dim of an input or state shape.
    bound: HashSet<String>,
    /// Every symbol use: (symbol, pointer, node id).
    used: Vec<(String, String)>,
    /// Root state names and expanded body state names.
    state_names: HashSet<String>,
    frames: Vec<Frame>,
    levels: Vec<Level<'g>>,
}

/// Saturating counts of the expanded graph.
#[derive(Default)]
struct Counts {
    nodes: u64,
    bindings: u64,
    state: u64,
}

fn valid_count(count: i64) -> Option<u64> {
    u64::try_from(count)
        .ok()
        .filter(|c| (limits::MIN_REPEAT_COUNT..=limits::MAX_REPEAT_COUNT).contains(c))
}

fn count_nodes(nodes: &[Node]) -> Counts {
    let mut c = Counts::default();
    for node in nodes {
        match &node.kind {
            NodeKind::Op(op) => {
                c.nodes = c.nodes.saturating_add(1);
                c.bindings = c.bindings.saturating_add(op.params.len() as u64);
            }
            NodeKind::Repeat(r) => {
                // An invalid count is reported as `invalid_repeat`; 1 avoids a second error here.
                let n = valid_count(r.count).unwrap_or(1);
                let body = count_nodes(&r.body.nodes);
                let state = body.state.saturating_add(r.body.state.len() as u64);
                c.nodes = c.nodes.saturating_add(body.nodes.saturating_mul(n));
                c.bindings = c.bindings.saturating_add(body.bindings.saturating_mul(n));
                c.state = c.state.saturating_add(state.saturating_mul(n));
            }
        }
    }
    c
}

fn node_ptr(base: &str, i: usize) -> String {
    format!("{base}/nodes/{i}")
}

impl<'g> Validator<'g> {
    fn new(graph: &'g Graph) -> Self {
        Validator {
            graph,
            report: ValidationReport::default(),
            declared_dims: graph.dims.iter().map(|d| d.name.as_str()).collect(),
            params: graph.params.iter().map(|p| p.name.as_str()).collect(),
            bound: HashSet::new(),
            used: Vec::new(),
            state_names: HashSet::new(),
            frames: Vec::new(),
            levels: Vec::new(),
        }
    }

    fn err(&mut self, code: ErrorCode, pointer: impl Into<String>, message: impl Into<String>) {
        self.report.push(Diagnostic::new(code, pointer, message));
    }

    fn node_err(
        &mut self,
        code: ErrorCode,
        pointer: impl Into<String>,
        node: &str,
        item: Option<Item>,
        message: impl Into<String>,
    ) {
        let mut d = Diagnostic::new(code, pointer, message).with_node(node);
        d.item = item;
        self.report.push(d);
    }

    /// §4.3 limits that bound the work of every later check. Returns `false` (stop) on a failure.
    fn check_limits(&mut self) -> bool {
        let g = self.graph;
        let mut ok = true;
        for (len, max, pointer, what) in [
            (
                g.inputs.len(),
                limits::MAX_GRAPH_IO,
                "/inputs",
                "graph inputs",
            ),
            (
                g.outputs.len(),
                limits::MAX_GRAPH_IO,
                "/outputs",
                "graph outputs",
            ),
            (g.dims.len(), limits::MAX_DIMS, "/dims", "dims"),
        ] {
            if len > max {
                self.err(
                    ErrorCode::InvalidValue,
                    pointer,
                    format!("{len} {what}, more than the limit of {max}"),
                );
                ok = false;
            }
        }
        let c = count_nodes(&g.nodes);
        if c.nodes > limits::MAX_EXPANDED_NODES {
            self.err(
                ErrorCode::TooManyNodes,
                "/nodes",
                format!(
                    "{} nodes after expanding repeats, more than the limit of {}",
                    fmt_count(c.nodes),
                    limits::MAX_EXPANDED_NODES
                ),
            );
            ok = false;
        }
        if c.bindings > limits::MAX_EXPANDED_PARAM_BINDINGS {
            self.err(
                ErrorCode::TooManyNodes,
                "/nodes",
                format!(
                    "{} param bindings after expanding repeats, more than the limit of {}",
                    fmt_count(c.bindings),
                    limits::MAX_EXPANDED_PARAM_BINDINGS
                ),
            );
            ok = false;
        }
        let state = c.state.saturating_add(g.state.len() as u64);
        if state > limits::MAX_GRAPH_IO as u64 {
            self.err(
                ErrorCode::InvalidValue,
                "/state",
                format!(
                    "{} state entries after expanding repeats, more than the limit of {}",
                    fmt_count(state),
                    limits::MAX_GRAPH_IO
                ),
            );
            ok = false;
        }
        ok
    }

    /// §4.1 check 2.
    fn check_versions(&mut self) {
        let g = self.graph;
        if g.format_version != FORMAT_VERSION {
            self.err(
                ErrorCode::UnsupportedFormat,
                "/format_version",
                format!(
                    "`format_version` {} is not supported (supported: {FORMAT_VERSION})",
                    g.format_version
                ),
            );
        }
        let mut seen = HashSet::new();
        for (name, version) in &g.opsets {
            let p = push_pointer("/opsets", name);
            if !seen.insert(name.as_str()) {
                self.err(
                    ErrorCode::DuplicateName,
                    p,
                    format!("opset `{name}` is declared twice"),
                );
                continue;
            }
            match ops::SUPPORTED_OPSETS.iter().find(|(s, _)| s == name) {
                None => self.err(
                    ErrorCode::UnsupportedOpset,
                    p,
                    format!("unknown opset `{name}`"),
                ),
                Some((_, v)) if v != version => self.err(
                    ErrorCode::UnsupportedOpset,
                    p,
                    format!("opset `{name}` version {version} is not supported (supported: {v})"),
                ),
                Some(_) => {}
            }
        }
    }

    /// §4.1 check 5, declarations.
    fn check_dims(&mut self) {
        let g = self.graph;
        let mut seen = HashSet::new();
        for d in &g.dims {
            let p = push_pointer("/dims", &d.name);
            if !seen.insert(d.name.as_str()) {
                self.err(
                    ErrorCode::DuplicateName,
                    &p,
                    format!("dim `{}` is declared twice", d.name),
                );
            }
            if !names::is_symbol(&d.name) {
                self.err(
                    ErrorCode::InvalidName,
                    &p,
                    format!(
                        "`{}` is not a valid dim symbol (letter or `_`, then letters, digits or `_`; 1–{} bytes; not `min`/`max`)",
                        truncate(&d.name),
                        limits::MAX_SYMBOL_BYTES
                    ),
                );
            }
            let min = d.min.unwrap_or(1);
            if min < 1 {
                self.err(
                    ErrorCode::InvalidValue,
                    push_pointer(&p, "min"),
                    format!("`min` must be >= 1, got {min}"),
                );
            }
            if let Some(max) = d.max
                && max < min.max(1)
            {
                self.err(
                    ErrorCode::InvalidValue,
                    push_pointer(&p, "max"),
                    format!("`max` must be >= `min` ({}), got {max}", min.max(1)),
                );
            }
        }
    }

    /// Params: names, dtype, literal shapes, element counts, init.
    fn check_params(&mut self) {
        let g = self.graph;
        let mut seen = HashSet::new();
        let mut total: Option<u64> = Some(0);
        for (i, param) in g.params.iter().enumerate() {
            let p = format!("/params/{i}");
            if !names::is_param_name(&param.name) {
                self.err(
                    ErrorCode::InvalidName,
                    format!("{p}/name"),
                    format!(
                        "`{}` is not a valid param name (1–{} bytes of letters, digits, `_`, `-`, `.`)",
                        truncate(&param.name),
                        limits::MAX_PARAM_NAME_BYTES
                    ),
                );
            } else if !seen.insert(param.name.as_str()) {
                self.err(
                    ErrorCode::DuplicateName,
                    format!("{p}/name"),
                    format!("param `{}` is declared twice", param.name),
                );
            }
            if let DtypeSpec::Concrete(d) = param.dtype {
                self.err(
                    ErrorCode::UnsupportedInVersion,
                    format!("{p}/dtype"),
                    format!("param dtype `{d}` is reserved; M1 params must have dtype `float`"),
                );
            }
            let shape_ptr = format!("{p}/shape");
            if let Some(elements) = self.check_shape(&param.shape, &shape_ptr, Place::Param) {
                match elements {
                    Some(n) => total = total.and_then(|t| t.checked_add(n)),
                    None => self.err(
                        ErrorCode::TooManyElements,
                        &shape_ptr,
                        format!(
                            "the element count of param `{}` doesn't fit in u64",
                            truncate(&param.name)
                        ),
                    ),
                }
            }
            if let Some(init) = &param.init {
                self.check_init(init, &format!("{p}/init"));
            }
        }
        if total.is_none() {
            self.err(
                ErrorCode::TooManyElements,
                "/params",
                "the total param element count doesn't fit in u64",
            );
        }
    }

    fn check_init(&mut self, init: &Init, pointer: &str) {
        let fields: &[(&str, f64)] = match init {
            Init::Constant { value } => &[("value", *value)],
            Init::Uniform { low, high } => &[("low", *low), ("high", *high)],
            Init::Normal { mean, std } => &[("mean", *mean), ("std", *std)],
            Init::KaimingUniform { a: Some(a) } => &[("a", *a)],
            _ => &[],
        };
        let bad: Vec<&str> = fields
            .iter()
            .filter(|(_, v)| !v.is_finite())
            .map(|(k, _)| *k)
            .collect();
        for k in bad {
            self.err(
                ErrorCode::InvalidValue,
                push_pointer(pointer, k),
                format!("init `{k}` must be a finite number"),
            );
        }
    }

    /// Checks a shape for its place (§2.3 table). For params, returns `Some(element count)` (`None`
    /// inside on u64 overflow) if every dim is a valid literal.
    fn check_shape(&mut self, shape: &[Dim], pointer: &str, place: Place) -> Option<Option<u64>> {
        if shape.is_empty() {
            self.err(
                ErrorCode::InvalidValue,
                pointer,
                "a shape must have rank 1–6, got 0",
            );
            return None;
        }
        if shape.len() > limits::MAX_RANK {
            self.err(
                ErrorCode::RankTooLarge,
                pointer,
                format!(
                    "rank {} is more than the maximum of {}",
                    shape.len(),
                    limits::MAX_RANK
                ),
            );
            return None;
        }
        let mut product: Option<u64> = Some(1);
        let mut all_literal = true;
        for (k, dim) in shape.iter().enumerate() {
            let p = push_pointer(pointer, &k.to_string());
            match dim {
                Dim::Literal(v) => {
                    if self.check_literal(*v, &p) {
                        // check_literal proved 1 ≤ v ≤ 2^32.
                        product = product.and_then(|x| x.checked_mul(v.unsigned_abs()));
                    } else {
                        all_literal = false;
                    }
                }
                Dim::Expr(s) => {
                    all_literal = false;
                    self.check_dim_string(s, &p, place);
                }
            }
        }
        (place == Place::Param && all_literal).then_some(product)
    }

    fn check_literal(&mut self, v: i64, pointer: &str) -> bool {
        if v < 1 {
            self.err(
                ErrorCode::NonpositiveDim,
                pointer,
                format!("dim {v} is not >= 1"),
            );
            false
        } else if v > limits::MAX_LITERAL_DIM {
            self.err(
                ErrorCode::InvalidValue,
                pointer,
                format!("literal dim {v} is more than the limit of 2^32"),
            );
            false
        } else {
            true
        }
    }

    fn check_dim_string(&mut self, s: &str, pointer: &str, place: Place) {
        match place {
            Place::Param => {
                self.err(
                    ErrorCode::InvalidValue,
                    pointer,
                    format!(
                        "param shapes allow only integer literals, got \"{}\"",
                        truncate(s)
                    ),
                );
            }
            Place::Input => {
                if !names::is_symbol(s) {
                    self.err(
                        ErrorCode::InvalidValue,
                        pointer,
                        format!("graph input shapes allow only integer literals and plain symbols, got \"{}\"", truncate(s)),
                    );
                } else if !self.declared_dims.contains(s) {
                    self.err(
                        ErrorCode::UnknownReference,
                        pointer,
                        format!("dim symbol `{s}` is not declared in `dims`"),
                    );
                } else {
                    self.bound.insert(s.to_owned());
                }
            }
            Place::State => {
                if names::is_symbol(s) {
                    if self.declared_dims.contains(s) {
                        self.bound.insert(s.to_owned());
                    } else {
                        self.err(
                            ErrorCode::UnknownReference,
                            pointer,
                            format!("dim symbol `{s}` is not declared in `dims`"),
                        );
                    }
                    return;
                }
                let declared = &self.declared_dims;
                match DimExpr::parse(s, &|name| declared.contains(name)) {
                    Ok(e) => {
                        if let Some(v) = e.as_literal() {
                            self.check_literal(v, pointer);
                        }
                        for sym in e.symbols() {
                            self.used.push((sym, pointer.to_owned()));
                        }
                    }
                    Err(e) => self.dim_err(&e, pointer),
                }
            }
        }
    }

    fn dim_err(&mut self, e: &DimError, pointer: &str) {
        self.err(e.code(), pointer, e.to_string());
    }

    /// The root scope: inputs, state, nodes, outputs, state updates; then cycles.
    fn check_root(&mut self) {
        let g = self.graph;
        self.frames.push(Frame {
            values: HashMap::new(),
            adj: Vec::new(),
            current: None,
        });
        for (i, input) in g.inputs.iter().enumerate() {
            let p = format!("/inputs/{i}");
            self.declare_value(
                &input.name,
                &format!("{p}/name"),
                ValueKind::Input,
                "graph input",
            );
            self.check_shape(&input.shape, &format!("{p}/shape"), Place::Input);
        }
        for (i, entry) in g.state.iter().enumerate() {
            let p = format!("/state/{i}");
            if self.declare_value(
                &entry.name,
                &format!("{p}/name"),
                ValueKind::State,
                "state entry",
            ) {
                self.state_names.insert(entry.name.clone());
            }
            self.check_shape(&entry.shape, &format!("{p}/shape"), Place::State);
        }
        self.check_scope(&g.nodes, "");

        let mut output_names = HashSet::new();
        for (i, out) in g.outputs.iter().enumerate() {
            let p = format!("/outputs/{i}");
            if !names::is_value_name(&out.name) {
                self.err(
                    ErrorCode::InvalidName,
                    format!("{p}/name"),
                    invalid_value_name(&out.name, "output name"),
                );
            } else if !output_names.insert(out.name.as_str()) {
                self.err(
                    ErrorCode::DuplicateName,
                    format!("{p}/name"),
                    format!("output `{}` is declared twice", out.name),
                );
            }
            self.resolve(&out.value, &format!("{p}/value"), None, None);
        }
        if g.outputs.is_empty() {
            self.err(
                ErrorCode::InvalidValue,
                "/outputs",
                "a graph needs at least one output",
            );
        }
        let root_state: HashSet<&str> = g.state.iter().map(|s| s.name.as_str()).collect();
        self.check_state_updates(&g.state_updates, &root_state, "/state_updates", None);
        self.finish_frame("", &g.nodes);
    }

    /// Declares a value in the innermost frame. Returns `false` if the name is invalid or taken.
    fn declare_value(&mut self, name: &str, pointer: &str, kind: ValueKind, what: &str) -> bool {
        if !names::is_value_name(name) {
            self.err(
                ErrorCode::InvalidName,
                pointer,
                invalid_value_name(name, what),
            );
            return false;
        }
        if self.frames.iter().any(|f| f.values.contains_key(name)) {
            let msg = if self
                .frames
                .last()
                .is_some_and(|f| f.values.contains_key(name))
            {
                format!("`{name}` is already used in this scope")
            } else {
                format!(
                    "`{name}` is already used in an enclosing scope; body names must not reuse it"
                )
            };
            self.err(ErrorCode::DuplicateName, pointer, msg);
            return false;
        }
        if let Some(frame) = self.frames.last_mut() {
            frame.values.insert(name.to_owned(), kind);
        }
        true
    }

    /// Declares and checks the nodes of the innermost frame.
    fn check_scope(&mut self, nodes: &'g [Node], base: &str) {
        if let Some(frame) = self.frames.last_mut() {
            frame.adj = vec![Vec::new(); nodes.len()];
        }
        for (i, node) in nodes.iter().enumerate() {
            let ports = match &node.kind {
                NodeKind::Op(op) => NodePorts::Op(self.op_ports(op)),
                NodeKind::Repeat(r) => {
                    NodePorts::Carry(r.carry.iter().map(|(k, _)| k.clone()).collect())
                }
            };
            self.declare_value(
                &node.id,
                &format!("{}/id", node_ptr(base, i)),
                ValueKind::Node { index: i, ports },
                "node id",
            );
        }
        for (i, node) in nodes.iter().enumerate() {
            if let Some(frame) = self.frames.last_mut() {
                frame.current = Some(i);
            }
            let p = node_ptr(base, i);
            match &node.kind {
                NodeKind::Op(op) => self.check_op_node(node, op, &p),
                NodeKind::Repeat(r) => self.check_repeat(node, r, &p),
            }
        }
        if let Some(frame) = self.frames.last_mut() {
            frame.current = None;
        }
    }

    fn op_ports(&self, op: &OpNode) -> Ports {
        let opset_ok = ops::opset_of(&op.op)
            .is_some_and(|set| self.graph.opsets.iter().any(|(s, _)| s == set));
        if !opset_ok {
            return Ports::Unknown;
        }
        let split_sizes = op
            .attrs
            .iter()
            .find(|(k, _)| k == "sizes")
            .and_then(|(_, v)| match v {
                crate::json::Value::Array(a) => Some(a.len()),
                _ => None,
            });
        ops::ports_of(&op.op, split_sizes)
    }

    fn check_op_node(&mut self, node: &'g Node, op: &'g OpNode, p: &str) {
        let id = node.id.as_str();
        let op_ptr = format!("{p}/op");
        if op.op == "call" {
            self.node_err(
                ErrorCode::UnsupportedInVersion,
                op_ptr,
                id,
                None,
                "`call` nodes are not supported in this version (functions are planned for M2)",
            );
            return;
        }
        match ops::opset_of(&op.op) {
            None => self.node_err(
                ErrorCode::UnknownOp,
                op_ptr,
                id,
                None,
                format!(
                    "`{}` is not an op name of the form `opset.op_name`",
                    truncate(&op.op)
                ),
            ),
            Some(set) => match self.graph.opsets.iter().find(|(s, _)| s == set) {
                None => self.node_err(
                    ErrorCode::UnknownOp,
                    op_ptr,
                    id,
                    None,
                    format!(
                        "op `{}`: opset `{set}` is not declared in `opsets`",
                        truncate(&op.op)
                    ),
                ),
                Some((_, version)) => {
                    let supported = ops::SUPPORTED_OPSETS
                        .iter()
                        .any(|(s, v)| *s == set && v == version);
                    // An unsupported opset version is already `unsupported_opset`.
                    if supported && !ops::is_known_op(&op.op) {
                        self.node_err(
                            ErrorCode::UnknownOp,
                            op_ptr,
                            id,
                            None,
                            format!(
                                "op `{}` is not in opset `{set}` version {version}",
                                truncate(&op.op)
                            ),
                        );
                    }
                }
            },
        }
        for (name, r) in &op.inputs {
            let ip = push_pointer(&format!("{p}/inputs"), name);
            match r {
                InputRef::One(s) => self.resolve(s, &ip, Some(id), Some(Item::Input(name.clone()))),
                InputRef::Many(list) => {
                    for (k, s) in list.iter().enumerate() {
                        self.resolve(
                            s,
                            &format!("{ip}/{k}"),
                            Some(id),
                            Some(Item::Input(name.clone())),
                        );
                    }
                }
            }
        }
        for (name, param) in &op.params {
            let pp = push_pointer(&format!("{p}/params"), name);
            self.resolve_param(param, &pp, id, name);
        }
    }

    fn resolve_param(&mut self, param: &str, pointer: &str, node: &str, op_param: &str) {
        let item = || Some(Item::Param(op_param.to_owned()));
        if !names::is_param_name(param) {
            self.node_err(
                ErrorCode::InvalidName,
                pointer,
                node,
                item(),
                format!(
                    "`{}` is not a valid param name (1–{} bytes of letters, digits, `_`, `-`, `.`)",
                    truncate(param),
                    limits::MAX_PARAM_NAME_BYTES
                ),
            );
            return;
        }
        if self.levels.iter().any(|l| !l.params_ok) {
            return;
        }
        if self.levels.is_empty() {
            if !self.params.contains(param) {
                self.node_err(
                    ErrorCode::UnknownReference,
                    pointer,
                    node,
                    item(),
                    format!("param `{param}` is not declared in `params`"),
                );
            }
            return;
        }
        // Longest expanded name: every `{i}` replaced by the largest index of its level.
        let mut max_len = param.len();
        for level in &self.levels {
            let digits = names::decimal_digits(level.count - 1);
            max_len = max_len
                .saturating_add(level.params_prefix.len())
                .saturating_add(level.params_placeholders.saturating_mul(digits))
                .saturating_sub(level.params_placeholders * 3);
        }
        if max_len > limits::MAX_PARAM_NAME_BYTES {
            self.node_err(
                ErrorCode::InvalidName,
                pointer,
                node,
                item(),
                format!(
                    "param `{param}` with the repeat prefixes is up to {max_len} bytes, more than {}",
                    limits::MAX_PARAM_NAME_BYTES
                ),
            );
            return;
        }
        // Iterate the index tuples of the levels whose prefix varies. The number of lookups is
        // bounded by MAX_EXPANDED_PARAM_BINDINGS (checked in `check_limits`).
        let varying: Vec<usize> = (0..self.levels.len())
            .filter(|&l| self.levels[l].params_placeholders > 0)
            .collect();
        let mut index = vec![0u64; self.levels.len()];
        loop {
            let mut name = String::with_capacity(max_len);
            for (l, level) in self.levels.iter().enumerate() {
                if level.params_placeholders > 0 {
                    name.push_str(&names::expand_prefix(level.params_prefix, index[l]));
                } else {
                    name.push_str(level.params_prefix);
                }
            }
            name.push_str(param);
            if !self.params.contains(name.as_str()) {
                let at = if varying.is_empty() {
                    String::new()
                } else {
                    let idx: Vec<String> = varying.iter().map(|&l| index[l].to_string()).collect();
                    format!(" (iteration {})", idx.join(", "))
                };
                self.node_err(
                    ErrorCode::UnknownReference,
                    pointer,
                    node,
                    item(),
                    format!("param `{name}`{at} is not declared in `params`"),
                );
                return;
            }
            // Odometer over the varying levels, innermost fastest.
            let mut advanced = false;
            for &l in varying.iter().rev() {
                index[l] += 1;
                if index[l] < self.levels[l].count {
                    advanced = true;
                    break;
                }
                index[l] = 0;
            }
            if !advanced {
                return;
            }
        }
    }

    fn check_repeat(&mut self, node: &'g Node, r: &'g Repeat, p: &str) {
        let id = node.id.as_str();
        let count = match valid_count(r.count) {
            Some(c) => c,
            None => {
                self.node_err(
                    ErrorCode::InvalidRepeat,
                    format!("{p}/count"),
                    id,
                    None,
                    format!(
                        "`count` must be {}–{}, got {}",
                        limits::MIN_REPEAT_COUNT,
                        limits::MAX_REPEAT_COUNT,
                        r.count
                    ),
                );
                1
            }
        };
        if r.carry.is_empty() {
            self.node_err(
                ErrorCode::InvalidRepeat,
                format!("{p}/carry"),
                id,
                None,
                "`carry` must have at least one entry",
            );
        }
        for (key, value) in &r.carry {
            self.resolve(
                value,
                &push_pointer(&format!("{p}/carry"), key),
                Some(id),
                None,
            );
        }
        let body = &r.body;
        let bp = format!("{p}/body");
        let params_prefix = self.check_prefix(
            body.params_prefix.as_deref(),
            &format!("{bp}/params_prefix"),
            id,
        );
        let state_prefix = self.check_prefix(
            body.state_prefix.as_deref(),
            &format!("{bp}/state_prefix"),
            id,
        );
        let state_prefix_missing_i =
            !body.state.is_empty() && state_prefix.is_some_and(|(_, n)| n == 0);
        if state_prefix_missing_i {
            let ptr = if body.state_prefix.is_some() {
                format!("{bp}/state_prefix")
            } else {
                format!("{bp}/state")
            };
            self.node_err(
                ErrorCode::InvalidRepeat,
                ptr,
                id,
                None,
                "a body that declares state needs a `state_prefix` that contains `{i}`",
            );
        }

        self.frames.push(Frame {
            values: HashMap::new(),
            adj: Vec::new(),
            current: None,
        });
        self.levels.push(Level {
            params_ok: params_prefix.is_some(),
            state_ok: state_prefix.is_some() && !state_prefix_missing_i,
            params_prefix: params_prefix.map_or("", |(s, _)| s),
            params_placeholders: params_prefix.map_or(0, |(_, n)| n),
            state_prefix: state_prefix.map_or("", |(s, _)| s),
            count,
        });
        for (key, _) in &r.carry {
            self.declare_value(
                key,
                &push_pointer(&format!("{p}/carry"), key),
                ValueKind::Carry,
                "carry name",
            );
        }
        self.check_body_state(body, &bp, id);
        self.check_scope(&body.nodes, &bp);

        let local_state: HashSet<&str> = body.state.iter().map(|s| s.name.as_str()).collect();
        self.check_state_updates(
            &body.state_updates,
            &local_state,
            &format!("{bp}/state_updates"),
            Some(id),
        );
        let carry_keys: HashSet<&str> = r.carry.iter().map(|(k, _)| k.as_str()).collect();
        let out_keys: HashSet<&str> = body.carry_out.iter().map(|(k, _)| k.as_str()).collect();
        for (key, value) in &body.carry_out {
            let cp = push_pointer(&format!("{bp}/carry_out"), key);
            if !carry_keys.contains(key.as_str()) {
                self.node_err(
                    ErrorCode::RepeatCarryMismatch,
                    &cp,
                    id,
                    None,
                    format!("`carry_out` has key `{key}`, which is not in `carry`"),
                );
            }
            self.resolve(value, &cp, Some(id), None);
        }
        for (key, _) in &r.carry {
            if !out_keys.contains(key.as_str()) {
                self.node_err(
                    ErrorCode::RepeatCarryMismatch,
                    format!("{bp}/carry_out"),
                    id,
                    None,
                    format!("`carry_out` has no entry for carry `{key}`"),
                );
            }
        }
        self.finish_frame(&bp, &body.nodes);
        self.levels.pop();
    }

    /// Checks a prefix; returns it with its number of `{i}` placeholders if valid (absent = "").
    fn check_prefix(
        &mut self,
        prefix: Option<&'g str>,
        pointer: &str,
        node: &str,
    ) -> Option<(&'g str, usize)> {
        let s = prefix.unwrap_or("");
        match names::prefix_placeholders(s) {
            Some(n) => Some((s, n)),
            None => {
                self.node_err(
                    ErrorCode::InvalidName,
                    pointer,
                    node,
                    None,
                    format!(
                        "`{}` is not a valid prefix (up to {} bytes of letters, digits, `_`, `-`, `.` and `{{i}}`)",
                        truncate(s),
                        limits::MAX_PARAM_NAME_BYTES
                    ),
                );
                None
            }
        }
    }

    /// Body state: local names, shapes, and the expanded root-level names.
    fn check_body_state(&mut self, body: &'g Body, bp: &str, node: &str) {
        for (k, entry) in body.state.iter().enumerate() {
            let sp = format!("{bp}/state/{k}");
            let declared = self.declare_value(
                &entry.name,
                &format!("{sp}/name"),
                ValueKind::State,
                "state entry",
            );
            self.check_shape(&entry.shape, &format!("{sp}/shape"), Place::State);
            if declared && self.levels.iter().all(|l| l.state_ok) {
                self.check_expanded_state(entry, &format!("{sp}/name"), node);
            }
        }
    }

    /// Generates the root-level names of a body state entry (bounded by the state limit checked in
    /// `check_limits`) and checks that they are valid and unique.
    fn check_expanded_state(&mut self, entry: &StateEntry, pointer: &str, node: &str) {
        let mut index = vec![0u64; self.levels.len()];
        loop {
            let mut name = String::new();
            for (l, level) in self.levels.iter().enumerate() {
                name.push_str(&names::expand_prefix(level.state_prefix, index[l]));
            }
            name.push_str(&entry.name);
            if !names::is_value_name(&name) {
                self.node_err(
                    ErrorCode::InvalidName,
                    pointer,
                    node,
                    None,
                    format!(
                        "state name `{}` with the repeat prefixes is longer than {} bytes",
                        truncate(&name),
                        limits::MAX_VALUE_NAME_BYTES
                    ),
                );
                return;
            }
            if !self.state_names.insert(name.clone()) {
                self.node_err(
                    ErrorCode::DuplicateName,
                    pointer,
                    node,
                    None,
                    format!(
                        "state name `{name}` (after expanding the repeat prefixes) is used twice"
                    ),
                );
                return;
            }
            let mut advanced = false;
            for l in (0..self.levels.len()).rev() {
                index[l] += 1;
                if index[l] < self.levels[l].count {
                    advanced = true;
                    break;
                }
                index[l] = 0;
            }
            if !advanced {
                return;
            }
        }
    }

    fn check_state_updates(
        &mut self,
        updates: &[(String, String)],
        state: &HashSet<&str>,
        pointer: &str,
        node: Option<&str>,
    ) {
        for (key, value) in updates {
            let up = push_pointer(pointer, key);
            if !state.contains(key.as_str()) {
                let msg = if node.is_some() {
                    format!("`state_updates` target `{key}` is not a state entry of this body")
                } else {
                    format!("`state_updates` target `{key}` is not a root state entry")
                };
                let mut d = Diagnostic::new(ErrorCode::UnknownReference, &up, msg);
                d.node_id = node.map(str::to_owned);
                self.report.push(d);
            }
            self.resolve(value, &up, node, None);
        }
    }

    /// Resolves a value reference in the current scopes and records the dependency edge.
    fn resolve(&mut self, reference: &str, pointer: &str, node: Option<&str>, item: Option<Item>) {
        let (id, port) = match reference.split_once(':') {
            Some((id, port)) => (id, Some(port)),
            None => (reference, None),
        };
        let found = self
            .frames
            .iter()
            .enumerate()
            .rev()
            .find_map(|(level, f)| f.values.get(id).map(|k| (level, k.clone())));
        let problem = match &found {
            None => Some((
                ErrorCode::UnknownReference,
                format!("`{}` doesn't name a value in scope", truncate(id)),
            )),
            Some((_, ValueKind::Node { ports, .. })) => match port {
                Some(port) if !ports.has(port) => Some((
                    ErrorCode::UnknownReference,
                    format!("node `{id}` has no output port `{}`", truncate(port)),
                )),
                None if !ports.has_out() => Some((
                    ErrorCode::AmbiguousOutput,
                    format!("node `{id}` has no `out` port; name a port (`{id}:<port>`)"),
                )),
                _ => None,
            },
            Some((_, kind)) => port.map(|_| {
                let what = match kind {
                    ValueKind::Input => "graph input",
                    ValueKind::State => "state entry",
                    _ => "carry",
                };
                (
                    ErrorCode::UnknownReference,
                    format!(
                        "`{id}` is a {what}, which is referred to by its name only, without a port"
                    ),
                )
            }),
        };
        if let Some((code, msg)) = problem {
            let mut d = Diagnostic::new(code, pointer, msg);
            d.node_id = node.map(str::to_owned);
            d.item = item;
            self.report.push(d);
            return;
        }
        if let Some((level, ValueKind::Node { index, .. })) = found
            && let Some(frame) = self.frames.get_mut(level)
            && let Some(user) = frame.current
            && let Some(edges) = frame.adj.get_mut(user)
        {
            edges.push(index);
        }
    }

    /// Reports cycles among the innermost frame's nodes and pops the frame.
    fn finish_frame(&mut self, base: &str, nodes: &[Node]) {
        let Some(frame) = self.frames.pop() else {
            return;
        };
        for scc in cyclic_sccs(&frame.adj) {
            let first = scc[0];
            let mut ids: Vec<&str> = scc.iter().take(8).map(|&i| nodes[i].id.as_str()).collect();
            if scc.len() > 8 {
                ids.push("…");
            }
            self.node_err(
                ErrorCode::Cycle,
                node_ptr(base, first),
                &nodes[first].id,
                None,
                format!("nodes form a cycle: {}", ids.join(", ")),
            );
        }
    }

    /// §2.3: every used symbol appears as a plain dim of an input or state shape.
    fn check_unbound(&mut self) {
        let used = std::mem::take(&mut self.used);
        let mut reported = HashSet::new();
        for (sym, pointer) in used {
            if !self.bound.contains(&sym) && reported.insert((sym.clone(), pointer.clone())) {
                self.err(
                    ErrorCode::UnboundDim,
                    pointer,
                    format!("dim symbol `{sym}` doesn't appear as a plain dim in any graph input or state shape, so it can't be bound"),
                );
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Place {
    Input,
    State,
    Param,
}

fn invalid_value_name(name: &str, what: &str) -> String {
    format!(
        "`{}` is not a valid {what} (1–{} bytes of letters, digits, `_`, `-`, `.`)",
        truncate(name),
        limits::MAX_VALUE_NAME_BYTES
    )
}

fn truncate(s: &str) -> String {
    const MAX: usize = 80;
    if s.chars().count() <= MAX {
        s.to_owned()
    } else {
        let mut t: String = s.chars().take(MAX).collect();
        t.push('…');
        t
    }
}

fn fmt_count(n: u64) -> String {
    if n == u64::MAX {
        "more than 2^64".to_owned()
    } else {
        n.to_string()
    }
}

/// Strongly connected components that contain a cycle (size > 1, or a self-loop), each sorted by
/// node index, in order of their first node. Iterative Tarjan, so deep graphs can't overflow the
/// stack.
fn cyclic_sccs(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    const UNVISITED: usize = usize::MAX;
    let n = adj.len();
    let mut index = vec![UNVISITED; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut stack = Vec::new();
    let mut next = 0usize;
    let mut out = Vec::new();
    // (node, next edge position)
    let mut call: Vec<(usize, usize)> = Vec::new();
    for root in 0..n {
        if index[root] != UNVISITED {
            continue;
        }
        call.push((root, 0));
        index[root] = next;
        low[root] = next;
        next += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut pos)) = call.last_mut() {
            if let Some(&w) = adj[v].get(*pos) {
                *pos += 1;
                if w >= n {
                    continue;
                }
                if index[w] == UNVISITED {
                    index[w] = next;
                    low[w] = next;
                    next += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            call.pop();
            if let Some(&(parent, _)) = call.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                let mut scc = Vec::new();
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                let cyclic = scc.len() > 1 || adj[v].contains(&v);
                if cyclic {
                    scc.sort_unstable();
                    out.push(scc);
                }
            }
        }
    }
    out.sort_by_key(|scc| scc[0]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tarjan() {
        assert!(cyclic_sccs(&[vec![1], vec![2], vec![]]).is_empty());
        assert_eq!(
            cyclic_sccs(&[vec![1], vec![0], vec![2]]),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(
            cyclic_sccs(&[vec![], vec![2], vec![3], vec![1]]),
            vec![vec![1, 2, 3]]
        );
        // A long chain doesn't overflow the stack.
        let n = 200_000;
        let adj: Vec<Vec<usize>> = (0..n)
            .map(|i| if i + 1 < n { vec![i + 1] } else { vec![0] })
            .collect();
        assert_eq!(cyclic_sccs(&adj).len(), 1);
    }
}
