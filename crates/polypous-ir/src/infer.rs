//! Op signatures, dtype inference and symbolic shape inference, `docs/specs/graph-ir.md` §4.1
//! checks 4 (per-op inputs, params and attributes), 6, 7 and 8 (dtypes and shapes of state updates
//! and `repeat` carries).
//!
//! Runs after structural validation ([`crate::validate`]) has checked the limits, so the work here
//! is bounded. References are resolved leniently: a value whose producer failed is unknown, and
//! nodes that use it are skipped without follow-up errors. Each scope is inferred in dependency
//! order; a `repeat` body is inferred once.
//!
//! Every rule follows the op's row and notes in §3.2/§3.3 and the general rules of §3.1; the rule
//! for each op is the function named after its [`ShapeRule`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::analysis::{
    Analysis, AttrValue, Attrs, DimCheck, NamedValue, NodeInfo, OpInfo, RepeatInfo, Scalar,
    SymbolInfo, ValueInfo, shape_string,
};
use crate::catalog::{
    self, AttrDef, AttrType, InDtype, OpDef, OutDtype, Outputs, Presence, ShapeRule,
};
use crate::diag::{Diagnostic, ErrorCode, Item, ValidationReport};
use crate::dim::{DimError, DimExpr};
use crate::dtype::{ComputeDtype, Dtype};
use crate::json::{self, Value, push_pointer};
use crate::limits;
use crate::model::{Body, Dim, Graph, InputRef, Node, NodeKind, OpNode, Param, Repeat, StateEntry};
use crate::names;
use crate::validate::ValidateOptions;

use limits::{MAX_CONCAT_SPLIT, MAX_CONSTANT_ELEMENTS};

/// Runs checks 4 (signatures), 6, 7 and 8 on a graph whose limits hold. Returns the diagnostics and,
/// if every node was inferred, the analysis.
pub(crate) fn run(
    graph: &Graph,
    options: &ValidateOptions,
) -> (ValidationReport, Option<Analysis>) {
    let mut inf = Inferrer::new(graph, options);
    let mut scope = Scope::default();
    let mut inputs = Vec::new();
    for input in &graph.inputs {
        let info = inf.input_info(input.dtype.resolve(options.compute_dtype), &input.shape);
        if info.is_none() {
            inf.complete = false;
        }
        scope
            .names
            .entry(input.name.clone())
            .or_insert(Slot::Value(info.clone()));
        inputs.push(info.map(|info| NamedValue {
            name: input.name.clone(),
            info,
        }));
    }
    let mut root_state = Vec::new();
    for entry in &graph.state {
        let info = inf.state_info(entry);
        scope
            .names
            .entry(entry.name.clone())
            .or_insert(Slot::Value(info.clone()));
        root_state.push(info);
    }
    inf.scopes.push(scope);
    let nodes = inf.process_scope(&graph.nodes, "");
    inf.check_state_updates(
        &graph.state_updates,
        &graph.state,
        &root_state,
        "/state_updates",
        None,
    );
    inf.scopes.pop();

    let mut state = Vec::new();
    for (entry, info) in graph.state.iter().zip(&root_state) {
        if let Some(info) = info {
            state.push(NamedValue {
                name: entry.name.clone(),
                info: info.clone(),
            });
        }
    }
    let nodes: Option<Vec<NodeInfo>> = nodes.into_iter().collect();
    let inputs: Option<Vec<NamedValue>> = inputs.into_iter().collect();
    let analysis = match (inf.complete, nodes, inputs) {
        (true, Some(nodes), Some(inputs)) if root_state.iter().all(Option::is_some) => {
            expand_body_state(&graph.nodes, &nodes, &mut Vec::new(), &mut state);
            Some(Analysis {
                compute_dtype: options.compute_dtype,
                symbols: graph
                    .dims
                    .iter()
                    .map(|d| SymbolInfo {
                        name: d.name.clone(),
                        min: d.min.unwrap_or(1),
                        max: d.max,
                    })
                    .collect(),
                inputs,
                state,
                nodes,
            })
        }
        _ => None,
    };
    (inf.report, analysis)
}

/// Appends the expanded body state entries (§2.6) in binding order.
fn expand_body_state(
    nodes: &[Node],
    infos: &[NodeInfo],
    prefix: &mut Vec<(String, u64)>,
    out: &mut Vec<NamedValue>,
) {
    for (node, info) in nodes.iter().zip(infos) {
        let (NodeKind::Repeat(r), NodeInfo::Repeat(ri)) = (&node.kind, info) else {
            continue;
        };
        let state_prefix = r.body.state_prefix.as_deref().unwrap_or("");
        for i in 0..ri.count {
            prefix.push((state_prefix.to_owned(), i));
            let mut name_prefix = String::new();
            for (p, idx) in prefix.iter() {
                name_prefix.push_str(&names::expand_prefix(p, *idx));
            }
            for (entry, sinfo) in r.body.state.iter().zip(&ri.state) {
                out.push(NamedValue {
                    name: format!("{name_prefix}{}", entry.name),
                    info: sinfo.clone(),
                });
            }
            expand_body_state(&r.body.nodes, &ri.nodes, prefix, out);
            prefix.pop();
        }
    }
}

enum Slot {
    Value(Option<ValueInfo>),
    Node(usize),
}

#[derive(Default)]
struct Scope {
    names: HashMap<String, Slot>,
    /// Output ports of the scope's nodes, by node index; `None` if unknown.
    ports: Vec<Option<Vec<(String, ValueInfo)>>>,
}

struct Level<'g> {
    /// `false` if the prefix or count is invalid (already reported): params aren't resolved.
    ok: bool,
    params_prefix: &'g str,
    varies: bool,
    count: u64,
}

struct Inferrer<'g> {
    graph: &'g Graph,
    compute_dtype: ComputeDtype,
    compute: Dtype,
    report: ValidationReport,
    declared: HashSet<&'g str>,
    /// Symbols that appear as a plain dim of an input or state shape (§2.3).
    bound: HashSet<String>,
    params: HashMap<&'g str, &'g Param>,
    levels: Vec<Level<'g>>,
    scopes: Vec<Scope>,
    /// Every node and value got a result.
    complete: bool,
}

fn node_ptr(base: &str, i: usize) -> String {
    format!("{base}/nodes/{i}")
}

fn collect_bound(nodes: &[Node], bound: &mut HashSet<String>) {
    for node in nodes {
        if let NodeKind::Repeat(r) = &node.kind {
            for entry in &r.body.state {
                plain_symbols(&entry.shape, bound);
            }
            collect_bound(&r.body.nodes, bound);
        }
    }
}

fn plain_symbols(shape: &[Dim], bound: &mut HashSet<String>) {
    for d in shape {
        if let Dim::Expr(s) = d
            && names::is_symbol(s)
        {
            bound.insert(s.clone());
        }
    }
}

/// Every value reference a node makes, including inside a `repeat` body.
fn collect_refs<'g>(node: &'g Node, out: &mut Vec<&'g str>) {
    match &node.kind {
        NodeKind::Op(op) => {
            for (_, r) in &op.inputs {
                match r {
                    InputRef::One(s) => out.push(s),
                    InputRef::Many(list) => out.extend(list.iter().map(String::as_str)),
                }
            }
        }
        NodeKind::Repeat(r) => {
            out.extend(r.carry.iter().map(|(_, v)| v.as_str()));
            for n in &r.body.nodes {
                collect_refs(n, out);
            }
            out.extend(r.body.state_updates.iter().map(|(_, v)| v.as_str()));
            out.extend(r.body.carry_out.iter().map(|(_, v)| v.as_str()));
        }
    }
}

fn ref_id(reference: &str) -> &str {
    reference.split_once(':').map_or(reference, |(id, _)| id)
}

impl<'g> Inferrer<'g> {
    fn new(graph: &'g Graph, options: &ValidateOptions) -> Self {
        let mut bound = HashSet::new();
        for input in &graph.inputs {
            plain_symbols(&input.shape, &mut bound);
        }
        for entry in &graph.state {
            plain_symbols(&entry.shape, &mut bound);
        }
        collect_bound(&graph.nodes, &mut bound);
        let mut params = HashMap::new();
        for p in &graph.params {
            params.entry(p.name.as_str()).or_insert(p);
        }
        Inferrer {
            graph,
            compute_dtype: options.compute_dtype,
            compute: options.compute_dtype.dtype(),
            report: ValidationReport::default(),
            declared: graph.dims.iter().map(|d| d.name.as_str()).collect(),
            bound,
            params,
            levels: Vec::new(),
            scopes: Vec::new(),
            complete: true,
        }
    }

    fn push(&mut self, d: Diagnostic) {
        self.report.push(d);
    }

    /// A graph input's shape: literals and declared plain symbols (checked structurally).
    fn input_info(&self, dtype: Dtype, shape: &[Dim]) -> Option<ValueInfo> {
        if shape.is_empty() || shape.len() > limits::MAX_RANK {
            return None;
        }
        let mut dims = Vec::with_capacity(shape.len());
        for d in shape {
            dims.push(match d {
                Dim::Literal(v) if (1..=limits::MAX_LITERAL_DIM).contains(v) => {
                    DimExpr::literal(*v).ok()?
                }
                Dim::Expr(s) if names::is_symbol(s) && self.declared.contains(s.as_str()) => {
                    DimExpr::symbol(s)
                }
                _ => return None,
            });
        }
        Some(ValueInfo { dtype, shape: dims })
    }

    /// A state entry's dtype and shape (literals, symbols and expressions; checked structurally).
    fn state_info(&mut self, entry: &StateEntry) -> Option<ValueInfo> {
        let dtype = entry.dtype.resolve(self.compute_dtype);
        let info = self
            .parse_shape(&entry.shape)
            .map(|shape| ValueInfo { dtype, shape });
        if info.is_none() {
            self.complete = false;
        }
        info
    }

    fn parse_shape(&self, shape: &[Dim]) -> Option<Vec<DimExpr>> {
        if shape.is_empty() || shape.len() > limits::MAX_RANK {
            return None;
        }
        let declared = &self.declared;
        let mut dims = Vec::with_capacity(shape.len());
        for d in shape {
            let e = match d {
                Dim::Literal(v) => DimExpr::literal(*v).ok()?,
                Dim::Expr(s) => DimExpr::parse(s, &|n| declared.contains(n)).ok()?,
            };
            if e.as_literal()
                .is_some_and(|v| !(1..=limits::MAX_LITERAL_DIM).contains(&v))
            {
                return None;
            }
            dims.push(e);
        }
        Some(dims)
    }

    /// Infers the nodes of the innermost scope in dependency order.
    fn process_scope(&mut self, nodes: &'g [Node], base: &str) -> Vec<Option<NodeInfo>> {
        let n = nodes.len();
        let mut index: HashMap<&str, usize> = HashMap::with_capacity(n);
        for (i, node) in nodes.iter().enumerate() {
            index.entry(node.id.as_str()).or_insert(i);
        }
        if let Some(scope) = self.scopes.last_mut() {
            for (i, node) in nodes.iter().enumerate() {
                scope.names.entry(node.id.clone()).or_insert(Slot::Node(i));
            }
            scope.ports = vec![None; n];
        }
        let mut users: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut indegree = vec![0usize; n];
        for (i, node) in nodes.iter().enumerate() {
            let mut refs = Vec::new();
            collect_refs(node, &mut refs);
            for r in refs {
                if let Some(&j) = index.get(ref_id(r)) {
                    users[j].push(i);
                    indegree[i] += 1;
                }
            }
        }
        let mut ready: BinaryHeap<Reverse<usize>> =
            (0..n).filter(|&i| indegree[i] == 0).map(Reverse).collect();
        let mut infos: Vec<Option<NodeInfo>> = vec![None; n];
        let scope_index = self.scopes.len().saturating_sub(1);
        while let Some(Reverse(i)) = ready.pop() {
            let node = &nodes[i];
            let p = node_ptr(base, i);
            let (info, ports) = match &node.kind {
                NodeKind::Op(op) => self.infer_op(node, op, &p),
                NodeKind::Repeat(r) => self.infer_repeat(node, r, &p),
            };
            if info.is_none() {
                self.complete = false;
            }
            infos[i] = info;
            if let Some(scope) = self.scopes.get_mut(scope_index) {
                scope.ports[i] = ports;
            }
            for &u in &users[i] {
                indegree[u] -= 1;
                if indegree[u] == 0 {
                    ready.push(Reverse(u));
                }
            }
        }
        if infos.iter().any(Option::is_none) {
            // Nodes on a cycle (reported structurally) are never ready.
            self.complete = false;
        }
        infos
    }

    /// Resolves a value reference to its dtype and shape, if known.
    fn resolve(&self, reference: &str) -> Option<ValueInfo> {
        let (id, port) = match reference.split_once(':') {
            Some((id, port)) => (id, Some(port)),
            None => (reference, None),
        };
        for scope in self.scopes.iter().rev() {
            if let Some(slot) = scope.names.get(id) {
                return match slot {
                    Slot::Value(v) if port.is_none() => v.clone(),
                    Slot::Value(_) => None,
                    Slot::Node(i) => {
                        let port = port.unwrap_or("out");
                        scope
                            .ports
                            .get(*i)?
                            .as_ref()?
                            .iter()
                            .find(|(p, _)| p == port)
                            .map(|(_, v)| v.clone())
                    }
                };
            }
        }
        None
    }

    /// The op definition, if the op is in a declared opset of a supported version (otherwise the
    /// structural pass reported it).
    fn available(&self, op: &str) -> Option<&'static OpDef> {
        let def = catalog::lookup(op)?;
        let set = def.opset();
        let declared = self.graph.opsets.iter().find(|(s, _)| s == set)?;
        catalog::SUPPORTED_OPSETS
            .iter()
            .any(|(s, v)| *s == set && *v == declared.1)
            .then_some(def)
    }

    fn infer_repeat(
        &mut self,
        node: &'g Node,
        r: &'g Repeat,
        p: &str,
    ) -> (Option<NodeInfo>, Option<Vec<(String, ValueInfo)>>) {
        let id = node.id.as_str();
        let count = u64::try_from(r.count)
            .ok()
            .filter(|c| (limits::MIN_REPEAT_COUNT..=limits::MAX_REPEAT_COUNT).contains(c));
        let carry: Vec<(String, Option<ValueInfo>)> = r
            .carry
            .iter()
            .map(|(k, v)| (k.clone(), self.resolve(v)))
            .collect();
        let body: &'g Body = &r.body;
        let bp = format!("{p}/body");
        let prefix = body.params_prefix.as_deref().unwrap_or("");
        let placeholders = names::prefix_placeholders(prefix);
        self.levels.push(Level {
            ok: count.is_some() && placeholders.is_some(),
            params_prefix: prefix,
            varies: placeholders.is_some_and(|n| n > 0),
            count: count.unwrap_or(1),
        });
        let mut scope = Scope::default();
        for (k, info) in &carry {
            scope
                .names
                .entry(k.clone())
                .or_insert(Slot::Value(info.clone()));
        }
        let mut state = Vec::with_capacity(body.state.len());
        for entry in &body.state {
            let info = self.state_info(entry);
            scope
                .names
                .entry(entry.name.clone())
                .or_insert(Slot::Value(info.clone()));
            state.push(info);
        }
        self.scopes.push(scope);
        let nodes = self.process_scope(&body.nodes, &bp);
        self.check_state_updates(
            &body.state_updates,
            &body.state,
            &state,
            &format!("{bp}/state_updates"),
            Some(id),
        );
        for (key, value) in &body.carry_out {
            let Some((_, Some(cin))) = carry.iter().find(|(k, _)| k == key) else {
                continue;
            };
            let Some(cout) = self.resolve(value) else {
                continue;
            };
            let cp = push_pointer(&format!("{bp}/carry_out"), key);
            if cin.dtype != cout.dtype {
                self.push(
                    Diagnostic::new(
                        ErrorCode::RepeatCarryMismatch,
                        cp,
                        format!(
                            "carry `{key}` enters the body as {} but `carry_out` gives {}; the dtype must stay the same across iterations",
                            cin.dtype, cout.dtype
                        ),
                    )
                    .with_node(id),
                );
            } else if cin.shape != cout.shape {
                self.push(
                    Diagnostic::new(
                        ErrorCode::RepeatCarryMismatch,
                        cp,
                        format!(
                            "carry `{key}` enters the body with shape {} but `carry_out` gives {}, which is not provably equal",
                            shape_string(&cin.shape),
                            shape_string(&cout.shape)
                        ),
                    )
                    .with_node(id),
                );
            }
        }
        self.scopes.pop();
        self.levels.pop();

        let ports: Option<Vec<(String, ValueInfo)>> = carry
            .iter()
            .map(|(k, v)| v.clone().map(|v| (k.clone(), v)))
            .collect();
        let info = match (count, &ports, nodes.into_iter().collect::<Option<Vec<_>>>()) {
            (Some(count), Some(carry), Some(nodes)) if state.iter().all(Option::is_some) => {
                Some(NodeInfo::Repeat(RepeatInfo {
                    id: id.to_owned(),
                    count,
                    carry: carry.clone(),
                    state: state.into_iter().flatten().collect(),
                    nodes,
                }))
            }
            _ => None,
        };
        (info, ports)
    }

    fn check_state_updates(
        &mut self,
        updates: &[(String, String)],
        entries: &[StateEntry],
        infos: &[Option<ValueInfo>],
        pointer: &str,
        node: Option<&str>,
    ) {
        for (key, value) in updates {
            let Some(k) = entries.iter().position(|e| &e.name == key) else {
                continue;
            };
            let (Some(Some(target)), Some(v)) = (infos.get(k), self.resolve(value)) else {
                continue;
            };
            let msg = if target.dtype != v.dtype {
                Some(format!(
                    "state `{key}` has dtype {} but the update has {}",
                    target.dtype, v.dtype
                ))
            } else if target.shape != v.shape {
                Some(format!(
                    "state `{key}` has shape {} but the update has {}, which is not provably equal",
                    shape_string(&target.shape),
                    shape_string(&v.shape)
                ))
            } else {
                None
            };
            if let Some(msg) = msg {
                let mut d = Diagnostic::new(
                    ErrorCode::StateUpdateMismatch,
                    push_pointer(pointer, key),
                    msg,
                );
                d.node_id = node.map(str::to_owned);
                self.push(d);
            }
        }
    }

    /// The literal shape of the graph param a node param refers to. Inside repeat bodies every
    /// iteration's param must have the same shape, because the body is inferred once.
    fn param_shape(
        &mut self,
        name: &str,
        pointer: &str,
        node: &str,
        item: &str,
    ) -> Option<Vec<i64>> {
        let literal = |p: &Param| -> Option<Vec<i64>> {
            p.shape
                .iter()
                .map(|d| match d {
                    Dim::Literal(v) if (1..=limits::MAX_LITERAL_DIM).contains(v) => Some(*v),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .filter(|s| (1..=limits::MAX_RANK).contains(&s.len()))
        };
        if self.levels.iter().any(|l| !l.ok) {
            return None;
        }
        let varying: Vec<usize> = (0..self.levels.len())
            .filter(|&l| self.levels[l].varies)
            .collect();
        let mut index = vec![0u64; self.levels.len()];
        let mut first: Option<Vec<i64>> = None;
        loop {
            let mut full = String::new();
            for (l, level) in self.levels.iter().enumerate() {
                if level.varies {
                    full.push_str(&names::expand_prefix(level.params_prefix, index[l]));
                } else {
                    full.push_str(level.params_prefix);
                }
            }
            full.push_str(name);
            let shape = literal(self.params.get(full.as_str())?)?;
            match &first {
                None => first = Some(shape),
                Some(f) if *f != shape => {
                    let idx: Vec<String> = varying.iter().map(|&l| index[l].to_string()).collect();
                    let msg = format!(
                        "param `{full}` (iteration {}) has shape {:?} but iteration 0 has {:?}; a repeat body is inferred once, so every iteration's params must have the same shape",
                        idx.join(", "),
                        shape,
                        f
                    );
                    self.push(
                        Diagnostic::new(ErrorCode::ShapeMismatch, pointer, msg)
                            .with_node(node)
                            .with_item(Item::Param(item.to_owned())),
                    );
                    return None;
                }
                Some(_) => {}
            }
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
                return first;
            }
        }
    }

    fn infer_op(
        &mut self,
        node: &'g Node,
        op: &'g OpNode,
        p: &str,
    ) -> (Option<NodeInfo>, Option<Vec<(String, ValueInfo)>>) {
        let Some(def) = self.available(&op.op) else {
            return (None, None);
        };
        let id = node.id.as_str();
        let mut cx = Cx {
            id,
            p,
            def,
            compute: self.compute,
            attrs: Attrs::default(),
            defaulted: Vec::new(),
            checks: Vec::new(),
            diags: Vec::new(),
        };
        let mut ok = cx.check_signature(op);
        if !cx.parse_attrs(op, &self.declared, &self.bound) {
            ok = false;
        }

        // Input values, parallel to `def.inputs`.
        let mut known = true;
        let mut inputs: Vec<Vec<ValueInfo>> = Vec::with_capacity(def.inputs.len());
        for d in def.inputs {
            let mut vals = Vec::new();
            match op.inputs.iter().find(|(n, _)| n == d.name).map(|(_, r)| r) {
                Some(InputRef::One(s)) if !d.variadic => match self.resolve(s) {
                    Some(v) => vals.push(v),
                    None => known = false,
                },
                Some(InputRef::Many(list)) if d.variadic => {
                    for s in list {
                        match self.resolve(s) {
                            Some(v) => vals.push(v),
                            None => known = false,
                        }
                    }
                }
                _ => known = false,
            }
            inputs.push(vals);
        }
        if known && !cx.check_inputs(&inputs) {
            ok = false;
        }

        // Param shapes, parallel to `def.params`.
        let mut params: Vec<Option<Vec<i64>>> = Vec::with_capacity(def.params.len());
        for d in def.params {
            match op.params.iter().find(|(n, _)| n == d.name) {
                Some((_, gname)) => {
                    let pp = push_pointer(&format!("{p}/params"), d.name);
                    let shape = self.param_shape(gname, &pp, id, d.name);
                    if shape.is_none() {
                        known = false;
                    }
                    params.push(shape);
                }
                None => params.push(None),
            }
        }

        let result = if ok && known {
            cx.infer_shapes(&inputs, &params).ok()
        } else {
            None
        };
        let Cx {
            attrs,
            checks,
            diags,
            ..
        } = cx;
        for d in diags {
            self.push(d);
        }
        let Some(shapes) = result else {
            return (None, None);
        };

        // Output dtypes.
        let mut outputs = Vec::with_capacity(shapes.len());
        match def.outputs {
            Outputs::Fixed(defs) => {
                for (o, shape) in defs.iter().zip(shapes) {
                    let input_dtype = |name: &str| {
                        def.inputs
                            .iter()
                            .position(|i| i.name == name)
                            .and_then(|k| inputs.get(k))
                            .and_then(|v| v.first())
                            .map(|v| v.dtype)
                    };
                    let dtype = match o.dtype {
                        OutDtype::Same(n) => input_dtype(n),
                        OutDtype::Bool => Some(Dtype::Bool),
                        OutDtype::Attr => attrs.dtype("dtype"),
                        OutDtype::AttrOr(n) => attrs.dtype("dtype").or_else(|| input_dtype(n)),
                        OutDtype::Compute => Some(self.compute),
                    };
                    let Some(dtype) = dtype else {
                        return (None, None);
                    };
                    outputs.push((o.name.to_owned(), ValueInfo { dtype, shape }));
                }
            }
            Outputs::Split => {
                let Some(dtype) = inputs.first().and_then(|v| v.first()).map(|v| v.dtype) else {
                    return (None, None);
                };
                for (j, shape) in shapes.into_iter().enumerate() {
                    outputs.push((j.to_string(), ValueInfo { dtype, shape }));
                }
            }
        }
        let ports = outputs.clone();
        (
            Some(NodeInfo::Op(OpInfo {
                id: id.to_owned(),
                op: def,
                attrs,
                outputs,
                checks,
            })),
            Some(ports),
        )
    }
}

/// A failure that has been reported.
struct Fail;

type R<T> = Result<T, Fail>;

/// The context of one op node: reports diagnostics on it and holds its attributes.
struct Cx<'a> {
    id: &'a str,
    p: &'a str,
    def: &'static OpDef,
    compute: Dtype,
    attrs: Attrs,
    /// Attributes that took their default.
    defaulted: Vec<&'static str>,
    checks: Vec<DimCheck>,
    diags: Vec<Diagnostic>,
}

fn class_list(c: InDtype) -> &'static str {
    match c {
        InDtype::Class(catalog::Class::F) | InDtype::Compute => "f32, f16, bf16",
        InDtype::Class(catalog::Class::I) => "i64, i32",
        InDtype::Class(catalog::Class::B) => "bool",
        InDtype::Class(catalog::Class::N) => "f32, f16, bf16, i64, i32",
        InDtype::Class(catalog::Class::A) | InDtype::Same(_) => "any dtype",
    }
}

fn lit(v: i64) -> R<DimExpr> {
    DimExpr::literal(v).map_err(|_| Fail)
}

impl Cx<'_> {
    fn diag(&mut self, code: ErrorCode, pointer: String, item: Option<Item>, msg: String) -> Fail {
        let mut d = Diagnostic::new(code, pointer, msg).with_node(self.id);
        d.item = item;
        self.diags.push(d);
        Fail
    }

    fn node_err(&mut self, code: ErrorCode, msg: impl Into<String>) -> Fail {
        self.diag(code, self.p.to_owned(), None, msg.into())
    }

    fn attr_err(&mut self, code: ErrorCode, name: &str, msg: impl Into<String>) -> Fail {
        let ptr = push_pointer(&format!("{}/attrs", self.p), name);
        self.diag(code, ptr, Some(Item::Attr(name.to_owned())), msg.into())
    }

    fn attr_entry_err(&mut self, name: &str, k: usize, msg: impl Into<String>) -> Fail {
        let ptr = format!("{}/{k}", push_pointer(&format!("{}/attrs", self.p), name));
        self.diag(
            ErrorCode::InvalidAttr,
            ptr,
            Some(Item::Attr(name.to_owned())),
            msg.into(),
        )
    }

    fn input_err(
        &mut self,
        code: ErrorCode,
        name: &str,
        k: Option<usize>,
        msg: impl Into<String>,
    ) -> Fail {
        let mut ptr = push_pointer(&format!("{}/inputs", self.p), name);
        if let Some(k) = k {
            ptr = format!("{ptr}/{k}");
        }
        self.diag(code, ptr, Some(Item::Input(name.to_owned())), msg.into())
    }

    fn param_err(&mut self, code: ErrorCode, name: &str, msg: impl Into<String>) -> Fail {
        let ptr = push_pointer(&format!("{}/params", self.p), name);
        self.diag(code, ptr, Some(Item::Param(name.to_owned())), msg.into())
    }

    /// Reports a dim arithmetic error on the node.
    fn dim(&mut self, r: Result<DimExpr, DimError>) -> R<DimExpr> {
        r.map_err(|e| self.node_err(e.code(), e.to_string()))
    }

    fn product(&mut self, dims: &[DimExpr]) -> R<DimExpr> {
        let mut acc = lit(1)?;
        for d in dims {
            acc = self.dim(acc.mul(d))?;
        }
        Ok(acc)
    }

    // ---- Check 4: signature ----

    /// Unknown and missing inputs, params and attributes; variadic forms.
    fn check_signature(&mut self, op: &OpNode) -> bool {
        let def = self.def;
        let mut ok = true;
        for (name, r) in &op.inputs {
            match def.input(name) {
                None => {
                    let names: Vec<&str> = def.inputs.iter().map(|i| i.name).collect();
                    self.input_err(
                        ErrorCode::UnknownInput,
                        name,
                        None,
                        format!(
                            "`{}` has no input `{name}` (inputs: {})",
                            def.name,
                            list_or_none(&names)
                        ),
                    );
                    ok = false;
                }
                Some(d) => match (d.variadic, r) {
                    (true, InputRef::One(_)) => {
                        self.input_err(
                            ErrorCode::InvalidJson,
                            name,
                            None,
                            format!("input `{name}` is variadic and takes an array of references"),
                        );
                        ok = false;
                    }
                    (false, InputRef::Many(_)) => {
                        self.input_err(
                            ErrorCode::InvalidJson,
                            name,
                            None,
                            format!("input `{name}` takes one reference, not an array"),
                        );
                        ok = false;
                    }
                    (true, InputRef::Many(list))
                        if list.is_empty() || list.len() > MAX_CONCAT_SPLIT =>
                    {
                        self.input_err(
                            ErrorCode::InvalidValue,
                            name,
                            None,
                            format!(
                                "input `{name}` has {} references; it takes 1–{MAX_CONCAT_SPLIT}",
                                list.len()
                            ),
                        );
                        ok = false;
                    }
                    _ => {}
                },
            }
        }
        for d in def.inputs {
            if !op.inputs.iter().any(|(n, _)| n == d.name) {
                self.diag(
                    ErrorCode::MissingInput,
                    self.p.to_owned(),
                    Some(Item::Input(d.name.to_owned())),
                    format!("`{}` needs input `{}`", def.name, d.name),
                );
                ok = false;
            }
        }
        for (name, _) in &op.params {
            if def.param(name).is_none() {
                let names: Vec<&str> = def.params.iter().map(|i| i.name).collect();
                self.param_err(
                    ErrorCode::UnknownParam,
                    name,
                    format!(
                        "`{}` has no param `{name}` (params: {})",
                        def.name,
                        list_or_none(&names)
                    ),
                );
                ok = false;
            }
        }
        for d in def.params {
            if !d.optional && !op.params.iter().any(|(n, _)| n == d.name) {
                self.diag(
                    ErrorCode::MissingParam,
                    self.p.to_owned(),
                    Some(Item::Param(d.name.to_owned())),
                    format!("`{}` needs param `{}`", def.name, d.name),
                );
                ok = false;
            }
        }
        ok
    }

    /// Parses every attribute of the catalog entry (given or default) by type (§3.1.7).
    fn parse_attrs(
        &mut self,
        op: &OpNode,
        declared: &HashSet<&str>,
        bound: &HashSet<String>,
    ) -> bool {
        let def = self.def;
        let mut ok = true;
        for (name, _) in &op.attrs {
            if def.attr(name).is_none() {
                let names: Vec<&str> = def.attrs.iter().map(|a| a.name).collect();
                self.attr_err(
                    ErrorCode::UnknownAttr,
                    name,
                    format!(
                        "`{}` has no attribute `{name}` (attributes: {})",
                        def.name,
                        list_or_none(&names)
                    ),
                );
                ok = false;
            }
        }
        for a in def.attrs {
            let given = op.attrs.iter().find(|(n, _)| n == a.name).map(|(_, v)| v);
            let default;
            let value = match (given, a.presence) {
                (Some(v), _) => v,
                (None, Presence::Default(text)) => match json::parse(text.as_bytes()) {
                    Ok(v) => {
                        self.defaulted.push(a.name);
                        default = v;
                        &default
                    }
                    // The catalog's defaults are valid JSON (unit test `defaults_parse`).
                    Err(_) => {
                        ok = false;
                        continue;
                    }
                },
                (None, Presence::Optional) => continue,
                (None, Presence::Required) => {
                    self.diag(
                        ErrorCode::MissingAttr,
                        self.p.to_owned(),
                        Some(Item::Attr(a.name.to_owned())),
                        format!("`{}` needs attribute `{}`", def.name, a.name),
                    );
                    ok = false;
                    continue;
                }
            };
            match parse_attr(a, value, self.compute, declared, bound) {
                Ok(v) => self.attrs.set(a.name, v),
                Err(e) => {
                    let ptr = match e.index {
                        Some(k) => {
                            format!("{}/{k}", push_pointer(&format!("{}/attrs", self.p), a.name))
                        }
                        None => push_pointer(&format!("{}/attrs", self.p), a.name),
                    };
                    self.diag(e.code, ptr, Some(Item::Attr(a.name.to_owned())), e.msg);
                    ok = false;
                }
            }
        }
        ok
    }

    // ---- Check 6: dtypes, and input ranks ----

    fn check_inputs(&mut self, inputs: &[Vec<ValueInfo>]) -> bool {
        let def = self.def;
        let mut ok = true;
        let mut class_ok = vec![true; def.inputs.len()];
        for (k, d) in def.inputs.iter().enumerate() {
            let Some(vals) = inputs.get(k) else { continue };
            for (j, v) in vals.iter().enumerate() {
                let at = d.variadic.then_some(j);
                let dt = v.dtype;
                match d.dtype {
                    InDtype::Class(c) => {
                        if !c.contains(dt) {
                            self.input_err(
                                ErrorCode::UnsupportedDtype,
                                d.name,
                                at,
                                format!(
                                    "input `{}` has dtype {dt}; `{}` takes {} here",
                                    d.name,
                                    def.name,
                                    class_list(d.dtype)
                                ),
                            );
                            class_ok[k] = false;
                            ok = false;
                        } else if j > 0 && vals[0].dtype != dt {
                            self.input_err(
                                ErrorCode::DtypeMismatch,
                                d.name,
                                at,
                                format!(
                                    "input `{}` entry {j} has dtype {dt} but entry 0 has {}; all entries must have the same dtype",
                                    d.name, vals[0].dtype
                                ),
                            );
                            ok = false;
                        }
                    }
                    InDtype::Same(other) => {
                        let o = def.inputs.iter().position(|i| i.name == other);
                        if let Some(o) = o
                            && class_ok[o]
                            && let Some(ov) = inputs.get(o).and_then(|v| v.first())
                            && ov.dtype != dt
                        {
                            self.input_err(
                                ErrorCode::DtypeMismatch,
                                d.name,
                                at,
                                format!(
                                    "input `{}` has dtype {dt} but `{other}` has {}; they must have the same dtype (there is no implicit promotion, use `core.cast`)",
                                    d.name, ov.dtype
                                ),
                            );
                            ok = false;
                        }
                    }
                    InDtype::Compute => {
                        if !dt.is_float() {
                            self.input_err(
                                ErrorCode::UnsupportedDtype,
                                d.name,
                                at,
                                format!(
                                    "input `{}` has dtype {dt}; `{}` takes a float input",
                                    d.name, def.name
                                ),
                            );
                            ok = false;
                        } else if dt != self.compute {
                            self.input_err(
                                ErrorCode::DtypeMismatch,
                                d.name,
                                at,
                                format!(
                                    "input `{}` has dtype {dt} but `nn` ops take the compute dtype {} (their params are seen in it)",
                                    d.name, self.compute
                                ),
                            );
                            ok = false;
                        }
                    }
                }
                let r = v.shape.len();
                if r < d.min_rank || r > d.max_rank {
                    let want = if d.min_rank == d.max_rank {
                        format!("rank {}", d.min_rank)
                    } else {
                        format!("rank {}–{}", d.min_rank, d.max_rank)
                    };
                    self.input_err(
                        ErrorCode::RankMismatch,
                        d.name,
                        at,
                        format!(
                            "input `{}` has rank {r} ({}); `{}` takes {want}",
                            d.name,
                            v.shape_string(),
                            def.name
                        ),
                    );
                    ok = false;
                }
            }
        }
        ok
    }

    // ---- Attribute helpers used by the shape rules ----

    /// Normalizes an axis attribute against rank `r` (§3.1.4) and stores it.
    fn axis(&mut self, name: &'static str, r: usize) -> R<usize> {
        let Some(v) = self.attrs.int(name) else {
            return match self.attrs.axis(name) {
                Some(a) => Ok(a),
                None => Err(Fail),
            };
        };
        let a = normalize_axis(v, r).ok_or_else(|| {
            self.attr_err(
                ErrorCode::InvalidAttr,
                name,
                format!(
                    "axis {v} is out of range for rank {r} (valid: {}…{})",
                    -(r as i64),
                    r as i64 - 1
                ),
            )
        })?;
        self.attrs.set(name, AttrValue::Axis(a));
        Ok(a)
    }

    /// Normalizes an axis list (non-empty, distinct, at most `r` entries) and stores it.
    fn axes(&mut self, name: &'static str, r: usize) -> R<Option<Vec<usize>>> {
        let Some(list) = self.attrs.ints(name).map(<[i64]>::to_vec) else {
            return Ok(None);
        };
        let mut out = Vec::with_capacity(list.len());
        for (k, &v) in list.iter().enumerate() {
            let Some(a) = normalize_axis(v, r) else {
                return Err(self.attr_entry_err(
                    name,
                    k,
                    format!("axis {v} is out of range for rank {r}"),
                ));
            };
            if out.contains(&a) {
                return Err(self.attr_entry_err(
                    name,
                    k,
                    format!("axis {v} is listed twice (normalized: {a})"),
                ));
            }
            out.push(a);
        }
        self.attrs.set(name, AttrValue::Axes(out.clone()));
        Ok(Some(out))
    }

    /// Converts a scalar attribute to the dtype the op fills (§3.1.7) and stores it.
    fn scalar_for(&mut self, name: &'static str, dtype: Dtype) -> R<Option<Scalar>> {
        let Some(s) = self.attrs.scalar(name) else {
            return Ok(None);
        };
        match coerce_scalar(s, dtype) {
            Ok(v) => {
                self.attrs.set(name, AttrValue::Scalar(v));
                Ok(Some(v))
            }
            Err(msg) => Err(self.attr_err(ErrorCode::InvalidAttr, name, msg)),
        }
    }

    /// A runtime condition `expr >= min` on a dim attribute; decided now if `expr` is a literal.
    fn check_dim(
        &mut self,
        attr: &'static str,
        expr: DimExpr,
        min: i64,
        code: ErrorCode,
        condition: String,
    ) -> R<()> {
        match expr.as_literal() {
            Some(v) if v < min => Err(self.attr_err(code, attr, format!("{condition}, got {v}"))),
            Some(_) => Ok(()),
            None => {
                self.checks.push(DimCheck {
                    attr,
                    expr,
                    min,
                    condition,
                });
                Ok(())
            }
        }
    }

    // ---- Checks 7: shapes ----

    fn infer_shapes(
        &mut self,
        inputs: &[Vec<ValueInfo>],
        params: &[Option<Vec<i64>>],
    ) -> R<Vec<Vec<DimExpr>>> {
        let first = |k: usize| inputs.get(k).and_then(|v| v.first());
        let x = first(0);
        let shapes = match self.def.shape {
            ShapeRule::Unary => vec![x.ok_or(Fail)?.shape.clone()],
            ShapeRule::Broadcast => {
                let (a, b) = (first(0).ok_or(Fail)?, first(1).ok_or(Fail)?);
                vec![self.broadcast(&[&a.shape, &b.shape])?]
            }
            ShapeRule::Where => {
                let (c, a, b) = (
                    first(0).ok_or(Fail)?,
                    first(1).ok_or(Fail)?,
                    first(2).ok_or(Fail)?,
                );
                vec![self.broadcast(&[&c.shape, &a.shape, &b.shape])?]
            }
            ShapeRule::Clamp => vec![self.clamp(x.ok_or(Fail)?)?],
            ShapeRule::Reshape => vec![self.reshape(x.ok_or(Fail)?)?],
            ShapeRule::Flatten => vec![self.flatten(x.ok_or(Fail)?)?],
            ShapeRule::Permute => vec![self.permute(x.ok_or(Fail)?)?],
            ShapeRule::Transpose => vec![self.transpose(x.ok_or(Fail)?)?],
            ShapeRule::Squeeze => vec![self.squeeze(x.ok_or(Fail)?)?],
            ShapeRule::Unsqueeze => vec![self.unsqueeze(x.ok_or(Fail)?)?],
            ShapeRule::Expand => vec![self.expand(x.ok_or(Fail)?)?],
            ShapeRule::Concat => vec![self.concat(inputs.first().ok_or(Fail)?)?],
            ShapeRule::Split => self.split(x.ok_or(Fail)?)?,
            ShapeRule::Slice => vec![self.slice(x.ok_or(Fail)?)?],
            ShapeRule::Pad => vec![self.pad(x.ok_or(Fail)?)?],
            ShapeRule::Gather => vec![self.gather(x.ok_or(Fail)?, first(1).ok_or(Fail)?)?],
            ShapeRule::IndexSelect => {
                vec![self.index_select(x.ok_or(Fail)?, first(1).ok_or(Fail)?)?]
            }
            ShapeRule::Constant => vec![self.constant()?],
            ShapeRule::Arange => vec![self.arange()?],
            ShapeRule::Fill => vec![self.fill()?],
            ShapeRule::FullLike => {
                let x = x.ok_or(Fail)?;
                let dtype = self.attrs.dtype("dtype").unwrap_or(x.dtype);
                self.attrs.set("dtype", AttrValue::Dtype(dtype));
                self.scalar_for("value", dtype)?;
                vec![x.shape.clone()]
            }
            ShapeRule::Reduce => vec![self.reduce(x.ok_or(Fail)?, true)?],
            ShapeRule::ArgReduce => vec![self.reduce(x.ok_or(Fail)?, false)?],
            ShapeRule::Matmul => vec![self.matmul(x.ok_or(Fail)?, first(1).ok_or(Fail)?)?],
            ShapeRule::TopK => {
                let s = self.top_k(x.ok_or(Fail)?)?;
                vec![s.clone(), s]
            }
            ShapeRule::UnaryAxis => {
                let x = x.ok_or(Fail)?;
                self.axis("dim", x.shape.len())?;
                vec![x.shape.clone()]
            }
            ShapeRule::Linear => vec![self.linear(x.ok_or(Fail)?, params)?],
            ShapeRule::Embedding => vec![self.embedding(x.ok_or(Fail)?, params)?],
            ShapeRule::Conv1d | ShapeRule::Conv2d => vec![self.conv(x.ok_or(Fail)?, params)?],
            ShapeRule::MaxPool2d | ShapeRule::AvgPool2d => vec![self.pool(x.ok_or(Fail)?)?],
            ShapeRule::AdaptiveAvgPool2d => {
                let x = x.ok_or(Fail)?;
                let os = self.attrs.ints("output_size").ok_or(Fail)?.to_vec();
                let mut s = x.shape[..2].to_vec();
                for v in os {
                    s.push(lit(v)?);
                }
                vec![s]
            }
            ShapeRule::BatchNorm => vec![self.batch_norm(x.ok_or(Fail)?, params)?],
            ShapeRule::LayerNorm | ShapeRule::RmsNorm => {
                vec![self.layer_norm(x.ok_or(Fail)?, params)?]
            }
            ShapeRule::GroupNorm => vec![self.group_norm(x.ok_or(Fail)?, params)?],
        };
        self.attrs.sort_by_catalog(self.def);
        let names = self.output_names(shapes.len());
        for (port, shape) in names.iter().zip(&shapes) {
            self.check_output(port, shape)?;
        }
        Ok(shapes)
    }

    fn output_names(&self, n: usize) -> Vec<String> {
        match self.def.outputs {
            Outputs::Fixed(defs) => defs.iter().map(|o| o.name.to_owned()).collect(),
            Outputs::Split => (0..n).map(|j| j.to_string()).collect(),
        }
    }

    /// Rank 1–6 and no literal dim ≤ 0 (§2.3, §3.1.5).
    fn check_output(&mut self, port: &str, shape: &[DimExpr]) -> R<()> {
        if shape.len() > limits::MAX_RANK {
            return Err(self.node_err(
                ErrorCode::RankTooLarge,
                format!(
                    "output `{port}` would have rank {}, more than the maximum of {}",
                    shape.len(),
                    limits::MAX_RANK
                ),
            ));
        }
        if shape.is_empty() {
            return Err(self.node_err(
                ErrorCode::ShapeMismatch,
                format!("output `{port}` would have rank 0"),
            ));
        }
        for (axis, d) in shape.iter().enumerate() {
            if let Some(v) = d.as_literal()
                && v < 1
            {
                return Err(self.node_err(
                    ErrorCode::NonpositiveDim,
                    format!(
                        "output `{port}` has size {v} in dim {axis} ({}); every dim must be >= 1",
                        shape_string(shape)
                    ),
                ));
            }
        }
        Ok(())
    }

    /// §3.1.3.
    fn broadcast(&mut self, shapes: &[&Vec<DimExpr>]) -> R<Vec<DimExpr>> {
        let rank = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
        let one = lit(1)?;
        let mut out = Vec::with_capacity(rank);
        for i in 0..rank {
            // Right-aligned: position i of the result is dim `len - rank + i` of each shape.
            let mut acc: Option<DimExpr> = None;
            for s in shapes {
                let Some(k) = (s.len() + i).checked_sub(rank) else {
                    continue;
                };
                let d = &s[k];
                acc = Some(match acc {
                    None => d.clone(),
                    Some(p) if p == *d => p,
                    Some(p) if p == one => d.clone(),
                    Some(p) if *d == one => p,
                    Some(p) => {
                        let all: Vec<String> = shapes.iter().map(|s| shape_string(s)).collect();
                        return Err(self.node_err(
                            ErrorCode::BroadcastMismatch,
                            format!(
                                "dims `{p}` and `{d}` can't be broadcast (shapes {}); dims broadcast only if provably equal or one is the literal 1",
                                all.join(", ")
                            ),
                        ));
                    }
                });
            }
            out.push(acc.ok_or(Fail)?);
        }
        Ok(out)
    }

    fn clamp(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let min = self.scalar_for("min", x.dtype)?;
        let max = self.scalar_for("max", x.dtype)?;
        if min.is_none() && max.is_none() {
            return Err(self.diag(
                ErrorCode::MissingAttr,
                self.p.to_owned(),
                Some(Item::Attr("min".to_owned())),
                "`core.clamp` needs at least one of `min` and `max`".to_owned(),
            ));
        }
        let greater = match (min, max) {
            (Some(Scalar::Int(a)), Some(Scalar::Int(b))) => a > b,
            // NaN compares false: a NaN bound is allowed (§3.2 clamp note: invalid only if
            // `min > max`).
            (Some(Scalar::Float(a)), Some(Scalar::Float(b))) => a > b,
            _ => false,
        };
        if greater {
            return Err(self.attr_err(ErrorCode::InvalidAttr, "min", "`min` must be <= `max`"));
        }
        Ok(x.shape.clone())
    }

    fn reshape(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let entries = self.attrs.dims("shape").ok_or(Fail)?.to_vec();
        let mut minus_one = None;
        for (k, e) in entries.iter().enumerate() {
            match e.as_literal() {
                Some(-1) if minus_one.is_none() => minus_one = Some(k),
                Some(-1) => {
                    return Err(self.attr_entry_err("shape", k, "at most one entry may be -1"));
                }
                Some(v) if v < 1 => {
                    return Err(self.attr_entry_err(
                        "shape",
                        k,
                        format!("entry {v} is not allowed (entries are >= 1, or one -1)"),
                    ));
                }
                _ => {}
            }
        }
        let count = self.product(&x.shape)?;
        let mut out = entries.clone();
        match minus_one {
            Some(k) => {
                let others: Vec<DimExpr> = entries
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != k)
                    .map(|(_, e)| e.clone())
                    .collect();
                let p = self.product(&others)?;
                match count.exact_div(&p) {
                    Some(q) => out[k] = self.dim(q)?,
                    None => {
                        return Err(self.attr_err(
                            ErrorCode::ShapeMismatch,
                            "shape",
                            format!(
                                "the input's element count `{count}` ({}) is not provably divisible by `{p}`, the product of the other entries; the product must be a single term that divides every term",
                                x.shape_string()
                            ),
                        ));
                    }
                }
            }
            None => {
                let p = self.product(&entries)?;
                if p != count {
                    return Err(self.attr_err(
                        ErrorCode::ShapeMismatch,
                        "shape",
                        format!(
                            "the new shape has `{p}` elements but the input ({}) has `{count}`, which is not provably equal",
                            x.shape_string()
                        ),
                    ));
                }
            }
        }
        self.attrs.set("shape", AttrValue::Dims(out.clone()));
        Ok(out)
    }

    fn flatten(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let start = self.axis("start_dim", r)?;
        let end = self.axis("end_dim", r)?;
        if start > end {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "start_dim",
                format!("`start_dim` ({start}) must not come after `end_dim` ({end})"),
            ));
        }
        let mut out = x.shape[..start].to_vec();
        out.push(self.product(&x.shape[start..=end])?);
        out.extend_from_slice(&x.shape[end + 1..]);
        Ok(out)
    }

    fn permute(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let dims = self.attrs.ints("dims").ok_or(Fail)?.to_vec();
        if dims.len() != r {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "dims",
                format!(
                    "`dims` has {} entries but the input has rank {r}",
                    dims.len()
                ),
            ));
        }
        let axes = self.axes("dims", r)?.ok_or(Fail)?;
        Ok(axes.iter().map(|&a| x.shape[a].clone()).collect())
    }

    fn transpose(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let a = self.axis("dim0", r)?;
        let b = self.axis("dim1", r)?;
        let mut out = x.shape.clone();
        out.swap(a, b);
        Ok(out)
    }

    fn squeeze(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let one = lit(1)?;
        let axes = match self.axes("dim", r)? {
            Some(axes) => {
                for (k, &a) in axes.iter().enumerate() {
                    if x.shape[a] != one {
                        return Err(self.attr_entry_err(
                            "dim",
                            k,
                            format!(
                                "dim {a} has size `{}`, not the literal 1 ({})",
                                x.shape[a],
                                x.shape_string()
                            ),
                        ));
                    }
                }
                axes
            }
            None => {
                let axes: Vec<usize> = (0..r).filter(|&a| x.shape[a] == one).collect();
                self.attrs.set("dim", AttrValue::Axes(axes.clone()));
                axes
            }
        };
        let mut out: Vec<DimExpr> = (0..r)
            .filter(|a| !axes.contains(a))
            .map(|a| x.shape[a].clone())
            .collect();
        if out.is_empty() {
            out.push(one);
        }
        Ok(out)
    }

    fn unsqueeze(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let v = self.attrs.int("dim").ok_or(Fail)?;
        let a = normalize_axis(v, r + 1).ok_or_else(|| {
            self.attr_err(
                ErrorCode::InvalidAttr,
                "dim",
                format!(
                    "axis {v} is out of range for unsqueeze of rank {r} (valid: {}…{r})",
                    -(r as i64) - 1
                ),
            )
        })?;
        self.attrs.set("dim", AttrValue::Axis(a));
        let mut out = x.shape.clone();
        out.insert(a, lit(1)?);
        Ok(out)
    }

    fn expand(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let entries = self.attrs.dims("shape").ok_or(Fail)?.to_vec();
        if entries.len() < r {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "shape",
                format!(
                    "`shape` has {} entries but the input has rank {r}; it needs {r}–6",
                    entries.len()
                ),
            ));
        }
        let lead = entries.len() - r;
        let one = lit(1)?;
        let mut out = Vec::with_capacity(entries.len());
        for (k, e) in entries.iter().enumerate() {
            let lit_v = e.as_literal();
            if k < lead {
                if let Some(v) = lit_v
                    && v < 1
                {
                    return Err(self.attr_entry_err(
                        "shape",
                        k,
                        format!("entry {v} is not allowed for a new leading dim (needs >= 1)"),
                    ));
                }
                out.push(e.clone());
                continue;
            }
            let d = &x.shape[k - lead];
            match lit_v {
                Some(-1) => out.push(d.clone()),
                Some(v) if v < 1 => {
                    return Err(self.attr_entry_err(
                        "shape",
                        k,
                        format!("entry {v} is not allowed (entries are >= 1 or -1)"),
                    ));
                }
                _ if e == d => out.push(d.clone()),
                _ if *d == one => out.push(e.clone()),
                _ => {
                    return Err(self.attr_err(
                        ErrorCode::ShapeMismatch,
                        "shape",
                        format!(
                            "entry {k} (`{e}`) is not provably equal to the input dim `{d}`, which is not the literal 1 ({})",
                            x.shape_string()
                        ),
                    ));
                }
            }
        }
        self.attrs.set("shape", AttrValue::Dims(out.clone()));
        Ok(out)
    }

    fn concat(&mut self, xs: &[ValueInfo]) -> R<Vec<DimExpr>> {
        let first = xs.first().ok_or(Fail)?;
        let r = first.shape.len();
        for (j, v) in xs.iter().enumerate().skip(1) {
            if v.shape.len() != r {
                return Err(self.input_err(
                    ErrorCode::RankMismatch,
                    "xs",
                    Some(j),
                    format!(
                        "entry {j} has rank {} ({}) but entry 0 has rank {r}",
                        v.shape.len(),
                        v.shape_string()
                    ),
                ));
            }
        }
        let dim = self.axis("dim", r)?;
        let mut out = first.shape.clone();
        for (j, v) in xs.iter().enumerate().skip(1) {
            for a in (0..r).filter(|&a| a != dim) {
                if v.shape[a] != first.shape[a] {
                    return Err(self.input_err(
                        ErrorCode::ShapeMismatch,
                        "xs",
                        Some(j),
                        format!(
                            "entry {j} ({}) differs from entry 0 ({}) in dim {a}, which is not the concat dim",
                            v.shape_string(),
                            first.shape_string()
                        ),
                    ));
                }
            }
            out[dim] = self.dim(out[dim].add(&v.shape[dim]))?;
        }
        Ok(out)
    }

    fn split(&mut self, x: &ValueInfo) -> R<Vec<Vec<DimExpr>>> {
        let dim = self.axis("dim", x.shape.len())?;
        let sizes = self.attrs.dims("sizes").ok_or(Fail)?.to_vec();
        for (k, s) in sizes.iter().enumerate() {
            if let Some(v) = s.as_literal()
                && v < 1
            {
                return Err(self.attr_entry_err("sizes", k, format!("size {v} is not >= 1")));
            }
        }
        let mut sum = lit(0)?;
        for s in &sizes {
            sum = self.dim(sum.add(s))?;
        }
        if sum != x.shape[dim] {
            return Err(self.attr_err(
                ErrorCode::ShapeMismatch,
                "sizes",
                format!(
                    "the sizes add up to `{sum}`, which is not provably equal to dim {dim} of the input (`{}`)",
                    x.shape[dim]
                ),
            ));
        }
        Ok(sizes
            .into_iter()
            .map(|s| {
                let mut out = x.shape.clone();
                out[dim] = s;
                out
            })
            .collect())
    }

    /// A slice bound (§3.2 slice note): a negative literal `v` counts from the end (`n + v`), then
    /// the bound is clamped to `n`. A literal bound ≤ 1 needs no clamp, since every dim is ≥ 1.
    fn slice_bound(&mut self, name: &'static str, e: DimExpr, n: &DimExpr) -> R<DimExpr> {
        let resolved = match e.as_literal() {
            Some(v) if v < 0 => {
                let r = self.dim(n.add(&lit(v)?))?;
                if let Some(rv) = r.as_literal()
                    && rv < 0
                {
                    return Err(self.attr_err(
                        ErrorCode::ShapeMismatch,
                        name,
                        format!("`{name}` {v} is out of range for a dim of size {n}"),
                    ));
                }
                r
            }
            _ => e,
        };
        self.check_dim(
            name,
            resolved.clone(),
            0,
            ErrorCode::InvalidAttr,
            format!("`{name}` must be >= 0"),
        )?;
        if resolved.as_literal().is_some_and(|v| v <= 1) {
            Ok(resolved)
        } else {
            self.dim(resolved.min(n))
        }
    }

    fn slice(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let dim = self.axis("dim", x.shape.len())?;
        let n = x.shape[dim].clone();
        let start = self.attrs.dim("start").ok_or(Fail)?.clone();
        let end = self.attrs.dim("end").cloned().unwrap_or_else(|| n.clone());
        let step = self.attrs.int("step").ok_or(Fail)?;
        let start = self.slice_bound("start", start, &n)?;
        let end = self.slice_bound("end", end, &n)?;
        let len = self.dim(end.sub(&start))?;
        let len = self.dim(len.add(&lit(step - 1)?))?;
        let len = self.dim(len.floor_div(step))?;
        self.attrs.set("start", AttrValue::Dim(start));
        self.attrs.set("end", AttrValue::Dim(end));
        let mut out = x.shape.clone();
        out[dim] = len;
        Ok(out)
    }

    fn pad(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let pad = self.attrs.dims("pad").ok_or(Fail)?.to_vec();
        if pad.len() % 2 != 0 || pad.len() > 2 * r {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "pad",
                format!(
                    "`pad` has {} entries; it needs an even number from 2 to {} (2 × rank)",
                    pad.len(),
                    2 * r
                ),
            ));
        }
        for (k, e) in pad.iter().enumerate() {
            if let Some(v) = e.as_literal() {
                if v < 0 {
                    return Err(self.attr_entry_err("pad", k, format!("entry {v} is not >= 0")));
                }
            } else {
                self.checks.push(DimCheck {
                    attr: "pad",
                    expr: e.clone(),
                    min: 0,
                    condition: format!("`pad` entry {k} must be >= 0"),
                });
            }
        }
        if x.dtype == Dtype::Bool && self.defaulted.contains(&"value") {
            // The default `0` is the zero of the dtype: `false` for bool.
            self.attrs
                .set("value", AttrValue::Scalar(Scalar::Bool(false)));
        } else {
            self.scalar_for("value", x.dtype)?;
        }
        let mut out = x.shape.clone();
        for (j, pair) in pad.chunks(2).enumerate() {
            let a = r - 1 - j;
            let grown = self.dim(out[a].add(&pair[0]))?;
            out[a] = self.dim(grown.add(&pair[1]))?;
        }
        Ok(out)
    }

    fn gather(&mut self, x: &ValueInfo, index: &ValueInfo) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        if index.shape.len() != r {
            return Err(self.input_err(
                ErrorCode::RankMismatch,
                "index",
                None,
                format!(
                    "`index` has rank {} but `x` has rank {r}; they must be equal",
                    index.shape.len()
                ),
            ));
        }
        let dim = self.axis("dim", r)?;
        for a in (0..r).filter(|&a| a != dim) {
            let (i, d) = (&index.shape[a], &x.shape[a]);
            let fits = i == d
                || matches!((i.as_literal(), d.as_literal()), (Some(iv), Some(dv)) if iv <= dv);
            if !fits {
                return Err(self.input_err(
                    ErrorCode::ShapeMismatch,
                    "index",
                    None,
                    format!(
                        "`index` dim {a} is `{i}` and `x` dim {a} is `{d}`; outside `dim` they must be provably equal, or literals with index <= x"
                    ),
                ));
            }
        }
        Ok(index.shape.clone())
    }

    fn index_select(&mut self, x: &ValueInfo, index: &ValueInfo) -> R<Vec<DimExpr>> {
        let dim = self.axis("dim", x.shape.len())?;
        let mut out = x.shape.clone();
        out[dim] = index.shape[0].clone();
        Ok(out)
    }

    fn constant(&mut self) -> R<Vec<DimExpr>> {
        let dtype = self.attrs.dtype("dtype").ok_or(Fail)?;
        let shape = self.attrs.ints("shape").ok_or(Fail)?.to_vec();
        let mut count: u64 = 1;
        for (k, &v) in shape.iter().enumerate() {
            if v > limits::MAX_LITERAL_DIM {
                return Err(self.attr_entry_err(
                    "shape",
                    k,
                    format!("literal dim {v} is more than the limit of 2^32"),
                ));
            }
            // Entries are >= 1 (attribute type) and <= 2^32.
            count = count.saturating_mul(v.unsigned_abs());
        }
        if count > MAX_CONSTANT_ELEMENTS {
            return Err(self.attr_err(
                ErrorCode::TooManyElements,
                "shape",
                format!(
                    "a constant has at most {MAX_CONSTANT_ELEMENTS} elements, this one {}; larger constants are params",
                    if count == u64::MAX { "more than 2^64".to_owned() } else { count.to_string() }
                ),
            ));
        }
        let values = self.attrs.scalars("values").ok_or(Fail)?.to_vec();
        if values.len() as u64 != count {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "values",
                format!(
                    "`values` has {} entries but `shape` has {count} elements",
                    values.len()
                ),
            ));
        }
        let mut out = Vec::with_capacity(values.len());
        for (k, v) in values.into_iter().enumerate() {
            match coerce_scalar(v, dtype) {
                Ok(s) => out.push(s),
                Err(msg) => return Err(self.attr_entry_err("values", k, msg)),
            }
        }
        self.attrs.set("values", AttrValue::Scalars(out));
        shape.into_iter().map(lit).collect()
    }

    fn arange(&mut self) -> R<Vec<DimExpr>> {
        let start = self.attrs.dim("start").ok_or(Fail)?.clone();
        let end = self.attrs.dim("end").ok_or(Fail)?.clone();
        let step = self.attrs.int("step").ok_or(Fail)?;
        self.check_dim(
            "start",
            start.clone(),
            0,
            ErrorCode::InvalidAttr,
            "`start` must be >= 0".to_owned(),
        )?;
        let len = self.dim(end.sub(&start))?;
        let len = self.dim(len.add(&lit(step - 1)?))?;
        Ok(vec![self.dim(len.floor_div(step))?])
    }

    fn fill(&mut self) -> R<Vec<DimExpr>> {
        let shape = self.attrs.dims("shape").ok_or(Fail)?.to_vec();
        for (k, e) in shape.iter().enumerate() {
            if let Some(v) = e.as_literal()
                && v < 1
            {
                return Err(self.attr_entry_err("shape", k, format!("entry {v} is not >= 1")));
            }
        }
        if self.def.attr("value").is_some() {
            let dtype = self.attrs.dtype("dtype").ok_or(Fail)?;
            self.scalar_for("value", dtype)?;
        }
        Ok(shape)
    }

    /// `keep_list`: reductions with an optional axis list (`sum`, …); otherwise one axis (`argmax`).
    fn reduce(&mut self, x: &ValueInfo, keep_list: bool) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let axes = if keep_list {
            match self.axes("dim", r)? {
                Some(a) => a,
                None => {
                    let all: Vec<usize> = (0..r).collect();
                    self.attrs.set("dim", AttrValue::Axes(all.clone()));
                    all
                }
            }
        } else {
            vec![self.axis("dim", r)?]
        };
        let keepdim = self.attrs.bool("keepdim").ok_or(Fail)?;
        let one = lit(1)?;
        let mut out = Vec::with_capacity(r);
        for (a, d) in x.shape.iter().enumerate() {
            if !axes.contains(&a) {
                out.push(d.clone());
            } else if keepdim {
                out.push(one.clone());
            }
        }
        if out.is_empty() {
            out.push(one);
        }
        Ok(out)
    }

    fn matmul(&mut self, a: &ValueInfo, b: &ValueInfo) -> R<Vec<DimExpr>> {
        let (ra, rb) = (a.shape.len(), b.shape.len());
        let (n, k) = (&a.shape[ra - 2], &a.shape[ra - 1]);
        let (k2, m) = (&b.shape[rb - 2], &b.shape[rb - 1]);
        if k != k2 {
            return Err(self.input_err(
                ErrorCode::ShapeMismatch,
                "b",
                None,
                format!(
                    "`a` {} has inner dim `{k}` but `b` {} has `{k2}`; they must be provably equal",
                    a.shape_string(),
                    b.shape_string()
                ),
            ));
        }
        let (ba, bb) = (a.shape[..ra - 2].to_vec(), b.shape[..rb - 2].to_vec());
        let mut out = self.broadcast(&[&ba, &bb])?;
        out.push(n.clone());
        out.push(m.clone());
        Ok(out)
    }

    fn top_k(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let dim = self.axis("dim", x.shape.len())?;
        let k = self.attrs.dim("k").ok_or(Fail)?.clone();
        self.check_dim(
            "k",
            k.clone(),
            1,
            ErrorCode::InvalidAttr,
            "`k` must be >= 1".to_owned(),
        )?;
        let d = &x.shape[dim];
        let room = self.dim(d.sub(&k))?;
        self.check_dim(
            "k",
            room,
            0,
            ErrorCode::ShapeMismatch,
            format!("`k` must be <= the size of dim {dim} (`{d}`); size minus `k`"),
        )?;
        let mut out = x.shape.clone();
        out[dim] = k;
        Ok(out)
    }

    // ---- nn ----

    /// A required or optional param's shape with an exact rank, else `rank_mismatch`.
    fn param_rank(
        &mut self,
        params: &[Option<Vec<i64>>],
        name: &'static str,
        rank: usize,
    ) -> R<Option<Vec<i64>>> {
        let k = self
            .def
            .params
            .iter()
            .position(|p| p.name == name)
            .ok_or(Fail)?;
        let Some(shape) = params.get(k).cloned().flatten() else {
            return Ok(None);
        };
        if shape.len() != rank {
            return Err(self.param_err(
                ErrorCode::RankMismatch,
                name,
                format!(
                    "param `{name}` has shape {shape:?}; `{}` needs rank {rank} ({})",
                    self.def.name, self.def.params[k].shape
                ),
            ));
        }
        Ok(Some(shape))
    }

    /// An optional param that must have exactly `want`.
    fn param_exact(
        &mut self,
        params: &[Option<Vec<i64>>],
        name: &'static str,
        want: &[i64],
    ) -> R<()> {
        if let Some(shape) = self.param_rank(params, name, want.len())?
            && shape != want
        {
            return Err(self.param_err(
                ErrorCode::ShapeMismatch,
                name,
                format!("param `{name}` has shape {shape:?} but needs {want:?}"),
            ));
        }
        Ok(())
    }

    /// `x`'s dim `a` must be provably equal to the literal `v`.
    fn x_dim_is(&mut self, x: &ValueInfo, a: usize, v: i64, what: &str) -> R<()> {
        if x.shape[a] != lit(v)? {
            return Err(self.input_err(
                ErrorCode::ShapeMismatch,
                self.def.inputs[0].name,
                None,
                format!(
                    "dim {a} of `{}` is `{}` ({}) but {what} is {v}; it must be provably equal",
                    self.def.inputs[0].name,
                    x.shape[a],
                    x.shape_string()
                ),
            ));
        }
        Ok(())
    }

    fn linear(&mut self, x: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let w = self.param_rank(params, "weight", 2)?.ok_or(Fail)?;
        let r = x.shape.len();
        self.x_dim_is(x, r - 1, w[1], "`weight` in (dim 1)")?;
        self.param_exact(params, "bias", &[w[0]])?;
        let mut out = x.shape[..r - 1].to_vec();
        out.push(lit(w[0])?);
        Ok(out)
    }

    fn embedding(&mut self, ids: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let w = self.param_rank(params, "weight", 2)?.ok_or(Fail)?;
        if ids.shape.len() >= limits::MAX_RANK {
            return Err(self.input_err(
                ErrorCode::RankTooLarge,
                "ids",
                None,
                format!(
                    "`ids` has rank {}; the output would have rank {}, more than {}",
                    ids.shape.len(),
                    ids.shape.len() + 1,
                    limits::MAX_RANK
                ),
            ));
        }
        let mut out = ids.shape.clone();
        out.push(lit(w[1])?);
        Ok(out)
    }

    /// Per-spatial-dim values of an `int` (conv1d) or `int[2]` attribute.
    fn spatial(&mut self, name: &str, n: usize) -> R<Vec<i64>> {
        match self.attrs.get(name) {
            Some(AttrValue::Int(v)) => Ok(vec![*v; n]),
            Some(AttrValue::Ints(v)) if v.len() == n => Ok(v.clone()),
            _ => Err(Fail),
        }
    }

    /// `(H + 2*p - dl*(k - 1) - 1) // s + 1` (§3.3).
    fn conv_out(&mut self, h: &DimExpr, k: i64, s: i64, p: i64, dl: i64) -> R<DimExpr> {
        let c = p.checked_mul(2).and_then(|pp| {
            dl.checked_mul(k - 1)
                .and_then(|e| pp.checked_sub(e))
                .and_then(|v| v.checked_sub(1))
        });
        let Some(c) = c else {
            return Err(self.node_err(
                ErrorCode::DimOverflow,
                "integer overflow in the output size formula",
            ));
        };
        let v = self.dim(h.add(&lit(c)?))?;
        let v = self.dim(v.floor_div(s))?;
        self.dim(v.add(&lit(1)?))
    }

    fn conv(&mut self, x: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let n = x.shape.len() - 2;
        let w = self.param_rank(params, "weight", n + 2)?.ok_or(Fail)?;
        let groups = self.attrs.int("groups").ok_or(Fail)?;
        let (c_out, c_in_g) = (w[0], w[1]);
        if c_out % groups != 0 {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "groups",
                format!("`groups` ({groups}) must divide C_out ({c_out}, `weight` dim 0)"),
            ));
        }
        let Some(c_in) = c_in_g.checked_mul(groups) else {
            return Err(self.node_err(ErrorCode::DimOverflow, "C_in/groups × groups overflows"));
        };
        self.x_dim_is(x, 1, c_in, "`weight` dim 1 × groups")?;
        self.param_exact(params, "bias", &[c_out])?;
        let stride = self.spatial("stride", n)?;
        let padding = self.spatial("padding", n)?;
        let dilation = self.spatial("dilation", n)?;
        let mut out = vec![x.shape[0].clone(), lit(c_out)?];
        for i in 0..n {
            out.push(self.conv_out(
                &x.shape[2 + i],
                w[2 + i],
                stride[i],
                padding[i],
                dilation[i],
            )?);
        }
        Ok(out)
    }

    fn pool(&mut self, x: &ValueInfo) -> R<Vec<DimExpr>> {
        let kernel = self.attrs.ints("kernel_size").ok_or(Fail)?.to_vec();
        let stride = match self.attrs.ints("stride") {
            Some(s) => s.to_vec(),
            None => {
                self.attrs.set("stride", AttrValue::Ints(kernel.clone()));
                kernel.clone()
            }
        };
        let padding = self.attrs.ints("padding").ok_or(Fail)?.to_vec();
        let dilation = match self.attrs.ints("dilation") {
            Some(d) => d.to_vec(),
            None => vec![1, 1],
        };
        for i in 0..2 {
            if padding[i] > kernel[i] / 2 {
                return Err(self.attr_entry_err(
                    "padding",
                    i,
                    format!(
                        "padding {} is more than half the kernel size {}",
                        padding[i], kernel[i]
                    ),
                ));
            }
        }
        let mut out = x.shape[..2].to_vec();
        for i in 0..2 {
            out.push(self.conv_out(
                &x.shape[2 + i],
                kernel[i],
                stride[i],
                padding[i],
                dilation[i],
            )?);
        }
        Ok(out)
    }

    /// `C` (`x` dim 1) as a literal, for `batch_norm` and `group_norm` (§3.3 note).
    fn channels(&mut self, x: &ValueInfo) -> R<i64> {
        match x.shape[1].as_literal() {
            Some(c) => Ok(c),
            None => Err(self.input_err(
                ErrorCode::ShapeMismatch,
                "x",
                None,
                format!(
                    "the channel dim of `x` ({}) is `{}`; `{}` needs a literal channel count",
                    x.shape_string(),
                    x.shape[1],
                    self.def.name
                ),
            )),
        }
    }

    fn batch_norm(&mut self, x: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let mean = self.param_rank(params, "running_mean", 1)?.ok_or(Fail)?;
        self.x_dim_is(x, 1, mean[0], "`running_mean`'s size")?;
        for name in ["running_var", "weight", "bias"] {
            self.param_exact(params, name, &mean)?;
        }
        Ok(x.shape.clone())
    }

    fn group_norm(&mut self, x: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let c = self.channels(x)?;
        let groups = self.attrs.int("num_groups").ok_or(Fail)?;
        if c % groups != 0 {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "num_groups",
                format!("`num_groups` ({groups}) must divide the channel count ({c})"),
            ));
        }
        self.param_exact(params, "weight", &[c])?;
        self.param_exact(params, "bias", &[c])?;
        Ok(x.shape.clone())
    }

    fn layer_norm(&mut self, x: &ValueInfo, params: &[Option<Vec<i64>>]) -> R<Vec<DimExpr>> {
        let r = x.shape.len();
        let weight = self
            .def
            .params
            .iter()
            .position(|p| p.name == "weight")
            .and_then(|k| params.get(k).cloned().flatten());
        let ns = match (self.attrs.ints("normalized_shape"), weight) {
            (Some(ns), _) => ns.to_vec(),
            (None, Some(w)) => {
                self.attrs
                    .set("normalized_shape", AttrValue::Ints(w.clone()));
                w
            }
            (None, None) => {
                return Err(self.diag(
                    ErrorCode::MissingAttr,
                    self.p.to_owned(),
                    Some(Item::Attr("normalized_shape".to_owned())),
                    format!(
                        "`{}` needs `normalized_shape` or a `weight` param to take it from",
                        self.def.name
                    ),
                ));
            }
        };
        if ns.len() > r {
            return Err(self.attr_err(
                ErrorCode::InvalidAttr,
                "normalized_shape",
                format!(
                    "`normalized_shape` has {} entries but the input has rank {r}",
                    ns.len()
                ),
            ));
        }
        for (j, &v) in ns.iter().enumerate() {
            let a = r - ns.len() + j;
            if x.shape[a] != lit(v)? {
                return Err(self.attr_err(
                    ErrorCode::ShapeMismatch,
                    "normalized_shape",
                    format!(
                        "`normalized_shape` {ns:?} doesn't match the trailing dims of `x` ({}): dim {a} is `{}`",
                        x.shape_string(),
                        x.shape[a]
                    ),
                ));
            }
        }
        self.param_exact(params, "weight", &ns)?;
        if self.def.param("bias").is_some() {
            self.param_exact(params, "bias", &ns)?;
        }
        Ok(x.shape.clone())
    }
}

fn list_or_none(names: &[&str]) -> String {
    if names.is_empty() {
        "none".to_owned()
    } else {
        names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `v` as an axis of rank `r`: `[-r, r-1]` → `0…r-1`.
fn normalize_axis(v: i64, r: usize) -> Option<usize> {
    let r = i64::try_from(r).ok()?;
    let a = if v < 0 { v.checked_add(r)? } else { v };
    if (0..r).contains(&a) {
        usize::try_from(a).ok()
    } else {
        None
    }
}

/// A scalar for the dtype the op fills (§3.1.7): an integer in range for `I`, a number for `F`, a
/// bool for `B`.
fn coerce_scalar(s: Scalar, dtype: Dtype) -> Result<Scalar, String> {
    match (s, dtype) {
        (Scalar::Bool(b), Dtype::Bool) => Ok(Scalar::Bool(b)),
        (_, Dtype::Bool) => Err("needs true or false for dtype bool".to_owned()),
        (Scalar::Int(v), Dtype::I64) => Ok(Scalar::Int(v)),
        (Scalar::Int(v), Dtype::I32) if i32::try_from(v).is_ok() => Ok(Scalar::Int(v)),
        (Scalar::Int(v), Dtype::I32) => Err(format!("{v} is out of range for dtype i32")),
        (_, Dtype::I64 | Dtype::I32) => Err(format!(
            "needs an integer within the range of {dtype} for dtype {dtype}"
        )),
        // Rounding to the float dtype happens at runtime; the value is kept as written.
        (Scalar::Int(v), _) => Ok(Scalar::Float(v as f64)),
        (Scalar::Float(v), _) => Ok(Scalar::Float(v)),
        (Scalar::Bool(_), _) => Err(format!("needs a number for dtype {dtype}")),
    }
}

struct AttrError {
    code: ErrorCode,
    index: Option<usize>,
    msg: String,
}

fn invalid(index: Option<usize>, msg: impl Into<String>) -> AttrError {
    AttrError {
        code: ErrorCode::InvalidAttr,
        index,
        msg: msg.into(),
    }
}

fn json_int(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        _ => None,
    }
}

fn json_float(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => Some(n.as_f64()),
        Value::String(s) => match s.as_str() {
            "inf" => Some(f64::INFINITY),
            "-inf" => Some(f64::NEG_INFINITY),
            "nan" => Some(f64::NAN),
            _ => None,
        },
        _ => None,
    }
}

fn json_scalar(v: &Value) -> Option<Scalar> {
    match v {
        Value::Bool(b) => Some(Scalar::Bool(*b)),
        Value::Number(n) => Some(match n.as_i64() {
            Some(i) if n.is_integer() => Scalar::Int(i),
            _ => Scalar::Float(n.as_f64()),
        }),
        _ => json_float(v).map(Scalar::Float),
    }
}

fn parse_dim(
    v: &Value,
    index: Option<usize>,
    declared: &HashSet<&str>,
    bound: &HashSet<String>,
) -> Result<DimExpr, AttrError> {
    let e = match v {
        Value::Number(n) => {
            let Some(i) = n.as_i64().filter(|_| n.is_integer()) else {
                return Err(invalid(
                    index,
                    "a dim must be an integer (i64) or a dim expression string",
                ));
            };
            DimExpr::literal(i).map_err(|e| AttrError {
                code: e.code(),
                index,
                msg: e.to_string(),
            })?
        }
        Value::String(s) => {
            DimExpr::parse(s, &|n| declared.contains(n)).map_err(|e| AttrError {
                code: e.code(),
                index,
                msg: e.to_string(),
            })?
        }
        other => {
            return Err(invalid(
                index,
                format!(
                    "a dim must be an integer or a dim expression string, got {}",
                    other.type_name()
                ),
            ));
        }
    };
    if let Some(sym) = e.symbols().into_iter().find(|s| !bound.contains(s)) {
        return Err(AttrError {
            code: ErrorCode::UnboundDim,
            index,
            msg: format!(
                "dim symbol `{sym}` doesn't appear as a plain dim in any graph input or state shape, so it can't be bound"
            ),
        });
    }
    Ok(e)
}

/// Parses an attribute value by its type (§3.1.7).
fn parse_attr(
    def: &AttrDef,
    v: &Value,
    compute: Dtype,
    declared: &HashSet<&str>,
    bound: &HashSet<String>,
) -> Result<AttrValue, AttrError> {
    let want = || {
        invalid(
            None,
            format!(
                "`{}` must be {}, got {}",
                def.name,
                def.ty.describe(),
                v.type_name()
            ),
        )
    };
    let ints = |v: &Value| -> Result<Vec<i64>, AttrError> {
        let Value::Array(items) = v else {
            return Err(want());
        };
        items
            .iter()
            .enumerate()
            .map(|(k, item)| {
                json_int(item).ok_or_else(|| invalid(Some(k), "entries must be integers"))
            })
            .collect()
    };
    match def.ty {
        AttrType::Int { min } => {
            let i = json_int(v).ok_or_else(want)?;
            match min {
                Some(m) if i < m => Err(invalid(
                    None,
                    format!("`{}` must be >= {m}, got {i}", def.name),
                )),
                _ => Ok(AttrValue::Int(i)),
            }
        }
        AttrType::Axis | AttrType::AxisInsert => json_int(v).map(AttrValue::Int).ok_or_else(want),
        AttrType::Axes => {
            let list = ints(v)?;
            if list.is_empty() {
                return Err(invalid(None, format!("`{}` must not be empty", def.name)));
            }
            Ok(AttrValue::Ints(list))
        }
        AttrType::Permutation => Ok(AttrValue::Ints(ints(v)?)),
        AttrType::Ints { len, min } => {
            let list = ints(v)?;
            if !len.allows(list.len()) {
                return Err(invalid(
                    None,
                    format!(
                        "`{}` has {} entries; it must be {}",
                        def.name,
                        list.len(),
                        def.ty.describe()
                    ),
                ));
            }
            if let Some(k) = list.iter().position(|&i| i < min) {
                return Err(invalid(
                    Some(k),
                    format!("entries must be >= {min}, got {}", list[k]),
                ));
            }
            Ok(AttrValue::Ints(list))
        }
        AttrType::Float(range) => {
            let f = json_float(v).ok_or_else(want)?;
            if !range.allows(f) {
                return Err(invalid(
                    None,
                    format!("`{}` must be {}, got {f}", def.name, def.ty.describe()),
                ));
            }
            Ok(AttrValue::Float(f))
        }
        AttrType::Bool => match v {
            Value::Bool(b) => Ok(AttrValue::Bool(*b)),
            _ => Err(want()),
        },
        AttrType::Enum(values) => match v {
            Value::String(s) if values.contains(&s.as_str()) => Ok(AttrValue::Enum(s.clone())),
            Value::String(s) => Err(invalid(
                None,
                format!(
                    "`{}` is \"{}\"; it must be one of {}",
                    def.name,
                    truncate(s),
                    values.join(", ")
                ),
            )),
            _ => Err(want()),
        },
        AttrType::Dim => parse_dim(v, None, declared, bound).map(AttrValue::Dim),
        AttrType::Dims(len) => {
            let Value::Array(items) = v else {
                return Err(want());
            };
            if !len.allows(items.len()) {
                return Err(invalid(
                    None,
                    format!(
                        "`{}` has {} entries; it must be {}",
                        def.name,
                        items.len(),
                        def.ty.describe()
                    ),
                ));
            }
            items
                .iter()
                .enumerate()
                .map(|(k, item)| parse_dim(item, Some(k), declared, bound))
                .collect::<Result<Vec<_>, _>>()
                .map(AttrValue::Dims)
        }
        AttrType::Dtype(set) => {
            let Value::String(s) = v else {
                return Err(want());
            };
            if s == "float" {
                return if set.allows_float() {
                    Ok(AttrValue::Dtype(compute))
                } else {
                    Err(invalid(
                        None,
                        format!(
                            "`{}` doesn't allow `float`; it must be {}",
                            def.name,
                            def.ty.describe()
                        ),
                    ))
                };
            }
            match Dtype::parse(s) {
                Some(d) if set.allows(d) => Ok(AttrValue::Dtype(d)),
                _ => Err(invalid(
                    None,
                    format!(
                        "`{}` is \"{}\"; it must be {}",
                        def.name,
                        truncate(s),
                        def.ty.describe()
                    ),
                )),
            }
        }
        AttrType::Scalar => json_scalar(v).map(AttrValue::Scalar).ok_or_else(want),
        AttrType::Scalars => {
            let Value::Array(items) = v else {
                return Err(want());
            };
            items
                .iter()
                .enumerate()
                .map(|(k, item)| {
                    json_scalar(item).ok_or_else(|| {
                        invalid(
                            Some(k),
                            "entries must be numbers, \"inf\", \"-inf\", \"nan\" or booleans",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map(AttrValue::Scalars)
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_parse() {
        // Every default in the catalog is valid JSON of its attribute's type.
        let declared = HashSet::new();
        let bound = HashSet::new();
        for op in catalog::OPS {
            for a in op.attrs {
                if let Presence::Default(text) = a.presence {
                    let v = json::parse(text.as_bytes())
                        .unwrap_or_else(|e| panic!("{} {}: {e}", op.name, a.name));
                    assert!(
                        parse_attr(a, &v, Dtype::F32, &declared, &bound).is_ok(),
                        "{} {}",
                        op.name,
                        a.name
                    );
                }
            }
        }
    }

    #[test]
    fn axes() {
        assert_eq!(normalize_axis(-1, 3), Some(2));
        assert_eq!(normalize_axis(-3, 3), Some(0));
        assert_eq!(normalize_axis(-4, 3), None);
        assert_eq!(normalize_axis(3, 3), None);
        assert_eq!(normalize_axis(i64::MIN, 3), None);
    }

    #[test]
    fn scalars() {
        assert_eq!(
            coerce_scalar(Scalar::Int(5), Dtype::F16),
            Ok(Scalar::Float(5.0))
        );
        assert!(coerce_scalar(Scalar::Float(1.0), Dtype::I64).is_err());
        assert!(coerce_scalar(Scalar::Int(1 << 40), Dtype::I32).is_err());
        assert_eq!(
            coerce_scalar(Scalar::Int(i64::MIN), Dtype::I64),
            Ok(Scalar::Int(i64::MIN))
        );
        assert!(coerce_scalar(Scalar::Int(0), Dtype::Bool).is_err());
        assert!(coerce_scalar(Scalar::Bool(true), Dtype::F32).is_err());
    }
}
