//! What validation proves about a graph, for the runtime: resolved dtypes, inferred shapes and
//! normalized attributes ([`Analysis`]), and the symbol binding of `docs/specs/graph-ir.md` §2.3.4
//! ([`Analysis::bind`]).
//!
//! An [`Analysis`] exists only for a graph that validated without errors ([`Graph::analyze`]). Its
//! node lists mirror the graph's: `analysis.nodes[i]` belongs to `graph.nodes[i]`, and a `repeat`'s
//! [`RepeatInfo::nodes`] to its body's nodes. A body is inferred once; its shapes hold for every
//! iteration.
//!
//! [`Analysis::bind`] runs steps 1–5 of §2.3.4 on the concrete input and state values of one call:
//! it checks them, binds the symbols, checks `min`/`max`, and evaluates every inferred dim and every
//! dim attribute with checked arithmetic. After it succeeds, no op can fail on shapes.

use std::collections::HashMap;

use crate::catalog::OpDef;
use crate::diag::{Diagnostic, ErrorCode, ValidationReport};
use crate::dim::{DimExpr, EvalError};
use crate::dtype::{ComputeDtype, Dtype};
use crate::model::Graph;
use crate::validate::ValidateOptions;

/// The resolved dtype and inferred shape of a value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueInfo {
    /// Concrete dtype (`float` resolved to the compute dtype).
    pub dtype: Dtype,
    /// Shape as normal forms, rank 1–6.
    pub shape: Vec<DimExpr>,
}

impl ValueInfo {
    /// The shape as canonical strings, `[batch, 128]`.
    pub fn shape_string(&self) -> String {
        shape_string(&self.shape)
    }
}

/// Writes a shape as `[d0, d1, …]` with canonical dim strings.
pub fn shape_string(shape: &[DimExpr]) -> String {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    format!("[{}]", dims.join(", "))
}

/// A value of a `scalar` attribute, converted to the dtype the op fills (§3.1.7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// For an `I` dtype: in range for it.
    Int(i64),
    /// For an `F` dtype: as written; the runtime rounds it to the dtype.
    Float(f64),
    /// For `bool`.
    Bool(bool),
}

/// A normalized attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue {
    /// `int`.
    Int(i64),
    /// `float`.
    Float(f64),
    /// `bool`.
    Bool(bool),
    /// `enum`.
    Enum(String),
    /// `int[]` that isn't an axis list.
    Ints(Vec<i64>),
    /// An axis, normalized to `0…r-1` (§3.1.4; `unsqueeze`: `0…r`).
    Axis(usize),
    /// An axis list, normalized (`squeeze` and reductions: absent means every reduced or removed
    /// axis, filled in).
    Axes(Vec<usize>),
    /// `dim`, as a normal form.
    Dim(DimExpr),
    /// `dim[]`, as normal forms.
    Dims(Vec<DimExpr>),
    /// `dtype`, `float` resolved to the compute dtype.
    Dtype(Dtype),
    /// `scalar`, converted to the filled dtype.
    Scalar(Scalar),
    /// `scalar[]`.
    Scalars(Vec<Scalar>),
}

/// The normalized attributes of an op node: every attribute of the catalog entry, defaults filled
/// in, in catalog order. Optional attributes without a default are absent when not given, except
/// where the spec defines the default in words; then the derived value is filled in:
///
/// - `core.squeeze` `dim`, `core.sum`/`mean`/`max`/`min` `dim`: the removed or reduced axes;
/// - `core.slice` `end`: the dim size. `start` and `end` are stored resolved: a negative literal
///   `v` becomes `n + v` and the bound is clamped to `n` (`min(v, n)`), so the interpreter slices
///   `start..end` with `step`;
/// - `core.reshape` and `core.expand` `shape`: the output shape (`-1` replaced);
/// - `core.full_like` `dtype`: `x`'s dtype; `nn.max_pool2d`/`nn.avg_pool2d` `stride`:
///   `kernel_size`; `nn.layer_norm`/`nn.rms_norm` `normalized_shape`: the weight's shape.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Attrs {
    entries: Vec<(&'static str, AttrValue)>,
}

impl Attrs {
    /// Every attribute, in catalog order.
    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &AttrValue)> {
        self.entries.iter().map(|(k, v)| (*k, v))
    }

    /// The attribute `name`, if present.
    pub fn get(&self, name: &str) -> Option<&AttrValue> {
        self.entries
            .iter()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v)
    }

    /// An `int` attribute.
    pub fn int(&self, name: &str) -> Option<i64> {
        match self.get(name) {
            Some(AttrValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// A `float` attribute.
    pub fn float(&self, name: &str) -> Option<f64> {
        match self.get(name) {
            Some(AttrValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// A `bool` attribute.
    pub fn bool(&self, name: &str) -> Option<bool> {
        match self.get(name) {
            Some(AttrValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    /// An `int[]` attribute that isn't an axis list.
    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.get(name) {
            Some(AttrValue::Ints(v)) => Some(v),
            _ => None,
        }
    }

    /// A normalized axis.
    pub fn axis(&self, name: &str) -> Option<usize> {
        match self.get(name) {
            Some(AttrValue::Axis(v)) => Some(*v),
            _ => None,
        }
    }

    /// A normalized axis list.
    pub fn axes(&self, name: &str) -> Option<&[usize]> {
        match self.get(name) {
            Some(AttrValue::Axes(v)) => Some(v),
            _ => None,
        }
    }

    /// A `dim` attribute.
    pub fn dim(&self, name: &str) -> Option<&DimExpr> {
        match self.get(name) {
            Some(AttrValue::Dim(v)) => Some(v),
            _ => None,
        }
    }

    /// A `dim[]` attribute.
    pub fn dims(&self, name: &str) -> Option<&[DimExpr]> {
        match self.get(name) {
            Some(AttrValue::Dims(v)) => Some(v),
            _ => None,
        }
    }

    /// A `dtype` attribute (resolved).
    pub fn dtype(&self, name: &str) -> Option<Dtype> {
        match self.get(name) {
            Some(AttrValue::Dtype(v)) => Some(*v),
            _ => None,
        }
    }

    /// A `scalar` attribute.
    pub fn scalar(&self, name: &str) -> Option<Scalar> {
        match self.get(name) {
            Some(AttrValue::Scalar(v)) => Some(*v),
            _ => None,
        }
    }

    /// A `scalar[]` attribute.
    pub fn scalars(&self, name: &str) -> Option<&[Scalar]> {
        match self.get(name) {
            Some(AttrValue::Scalars(v)) => Some(v),
            _ => None,
        }
    }

    /// An `enum` attribute.
    pub fn enum_value(&self, name: &str) -> Option<&str> {
        match self.get(name) {
            Some(AttrValue::Enum(v)) => Some(v),
            _ => None,
        }
    }

    pub(crate) fn set(&mut self, name: &'static str, value: AttrValue) {
        match self.entries.iter_mut().find(|(k, _)| *k == name) {
            Some(entry) => entry.1 = value,
            None => self.entries.push((name, value)),
        }
    }

    /// Puts the entries in catalog order.
    pub(crate) fn sort_by_catalog(&mut self, def: &OpDef) {
        self.entries.sort_by_key(|(k, _)| {
            def.attrs
                .iter()
                .position(|a| a.name == *k)
                .unwrap_or(usize::MAX)
        });
    }
}

/// A condition on dim attributes that validation couldn't decide because it involves symbols, and
/// that [`Analysis::bind`] checks: `expr >= min` (for example `slice` `start >= 0`, `top_k`
/// `d(dim) - k >= 0`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimCheck {
    /// The attribute the condition is about.
    pub attr: &'static str,
    /// The checked expression.
    pub expr: DimExpr,
    /// Its smallest allowed value.
    pub min: i64,
    /// The condition in words, for error messages (`` `start` must be >= 0 ``).
    pub condition: String,
}

/// What validation inferred for an op node.
#[derive(Debug, Clone, PartialEq)]
pub struct OpInfo {
    /// The node id.
    pub id: String,
    /// The catalog entry.
    pub op: &'static OpDef,
    /// Normalized attributes.
    pub attrs: Attrs,
    /// Output ports in catalog order (`core.split`: `0` … `n-1`), with dtype and shape.
    pub outputs: Vec<(String, ValueInfo)>,
    /// Runtime conditions on dim attributes.
    pub checks: Vec<DimCheck>,
}

/// What validation inferred for a `repeat` node.
#[derive(Debug, Clone, PartialEq)]
pub struct RepeatInfo {
    /// The node id.
    pub id: String,
    /// `count`.
    pub count: u64,
    /// Each carry (in `carry` order) with the dtype and shape it keeps across iterations; also the
    /// repeat's output ports.
    pub carry: Vec<(String, ValueInfo)>,
    /// The body's state entries (body-local, in declaration order). Every iteration's expanded
    /// entry has this dtype and shape.
    pub state: Vec<ValueInfo>,
    /// The body's nodes, parallel to `body.nodes`.
    pub nodes: Vec<NodeInfo>,
}

/// What validation inferred for a node.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeInfo {
    /// An op node.
    Op(OpInfo),
    /// A `repeat` node.
    Repeat(RepeatInfo),
}

impl NodeInfo {
    /// The node id.
    pub fn id(&self) -> &str {
        match self {
            NodeInfo::Op(op) => &op.id,
            NodeInfo::Repeat(r) => &r.id,
        }
    }

    /// The output ports with dtype and shape (a repeat's ports are its carries).
    pub fn outputs(&self) -> &[(String, ValueInfo)] {
        match self {
            NodeInfo::Op(op) => &op.outputs,
            NodeInfo::Repeat(r) => &r.carry,
        }
    }
}

/// A named value (graph input or root-level state entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NamedValue {
    /// Name (state: expanded, with the repeat prefixes).
    pub name: String,
    /// Dtype and declared shape.
    pub info: ValueInfo,
}

/// A declared dim symbol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymbolInfo {
    /// The symbol.
    pub name: String,
    /// `min` (default 1).
    pub min: i64,
    /// `max`, if declared.
    pub max: Option<i64>,
}

/// The result of validating a graph (§4.1 checks 6–8), for the runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct Analysis {
    /// The compute dtype the graph was validated with.
    pub compute_dtype: ComputeDtype,
    /// Declared dim symbols, in document order.
    pub symbols: Vec<SymbolInfo>,
    /// Graph inputs, in declaration order.
    pub inputs: Vec<NamedValue>,
    /// Every root-level state entry after expanding repeats (§2.5, §2.6): root state in
    /// declaration order, then body state, depth-first in node order, iterations in order
    /// (`layers.0.k`, `layers.0.v`, `layers.1.k`, …). This is the order in which state values bind
    /// symbols (§2.3.4 step 2).
    pub state: Vec<NamedValue>,
    /// Root nodes, parallel to `graph.nodes`.
    pub nodes: Vec<NodeInfo>,
}

impl Graph {
    /// Validates the graph and returns what validation proved: dtypes and shapes of every value and
    /// the normalized attributes of every node. Err with every error if the graph doesn't validate.
    pub fn analyze(&self, options: &ValidateOptions) -> Result<Analysis, ValidationReport> {
        let report = self.validate(options);
        if !report.is_ok() {
            return Err(report);
        }
        let (report, analysis) = crate::infer::run(self, options);
        match analysis {
            Some(a) if report.is_ok() => Ok(a),
            _ if !report.is_ok() => Err(report),
            _ => {
                // Inference produces a result for every node of a graph without errors; this is
                // only reachable if that invariant is broken.
                let mut r = ValidationReport::default();
                r.push(Diagnostic::new(
                    ErrorCode::InvalidValue,
                    "",
                    "internal error: shape inference produced no result for a valid graph",
                ));
                Err(r)
            }
        }
    }
}

// ---- Binding (§2.3.4) ----

/// A concrete value given to a call: its dtype and shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDesc<'a> {
    /// Dtype.
    pub dtype: Dtype,
    /// Shape.
    pub shape: &'a [usize],
}

/// Whether a bound value is a graph input or a state entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueKind {
    /// A graph input.
    Input,
    /// A state entry.
    State,
}

impl std::fmt::Display for ValueKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ValueKind::Input => "input",
            ValueKind::State => "state entry",
        })
    }
}

/// Why a call's values don't fit the graph (§2.3.4). Every variant names the input, state entry,
/// symbol or node. The facade maps it to `InvalidInput`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum BindError {
    /// A declared input or state value is missing.
    #[error("{kind} `{name}` is missing")]
    Missing {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
    },
    /// A value is given that the graph doesn't declare.
    #[error("{kind} `{name}` is not declared by the graph")]
    Unknown {
        /// Input or state.
        kind: ValueKind,
        /// The given name.
        name: String,
    },
    /// A value is given twice.
    #[error("{kind} `{name}` is given twice")]
    Duplicate {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
    },
    /// A value has a different dtype than declared.
    #[error("{kind} `{name}` has dtype {actual}, but the graph declares {expected}")]
    Dtype {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
        /// Declared dtype (`float` resolved).
        expected: Dtype,
        /// Given dtype.
        actual: Dtype,
    },
    /// A value has a different rank than declared.
    #[error("{kind} `{name}` has rank {actual}, but the graph declares rank {expected}")]
    Rank {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
        /// Declared rank.
        expected: usize,
        /// Given rank.
        actual: usize,
    },
    /// A dim differs from the declared literal or evaluated expression.
    #[error("{kind} `{name}` has size {actual} in dim {axis}, but the graph declares {expected}")]
    DimMismatch {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
        /// The axis.
        axis: usize,
        /// The declared (or evaluated) size.
        expected: i64,
        /// The given size.
        actual: usize,
    },
    /// A symbol occurs with two different values.
    #[error(
        "dim `{symbol}` is {first} from {first_kind} `{first_name}` dim {first_axis}, but {value} from {kind} `{name}` dim {axis}"
    )]
    InconsistentSymbol {
        /// The symbol.
        symbol: String,
        /// The value it was bound to.
        first: i64,
        /// Where it was bound.
        first_kind: ValueKind,
        /// Where it was bound.
        first_name: String,
        /// Where it was bound.
        first_axis: usize,
        /// The conflicting value.
        value: i64,
        /// Where the conflict is.
        kind: ValueKind,
        /// Where the conflict is.
        name: String,
        /// Where the conflict is.
        axis: usize,
    },
    /// A symbol's value is outside its declared `min`/`max`.
    #[error("dim `{symbol}` is {value}, outside its declared range {min}..={}", max.map_or("".to_owned(), |m| m.to_string()))]
    SymbolOutOfRange {
        /// The symbol.
        symbol: String,
        /// Its value.
        value: i64,
        /// Declared `min`.
        min: i64,
        /// Declared `max`.
        max: Option<i64>,
    },
    /// A given dim doesn't fit in `i64`.
    #[error("{kind} `{name}` has size {actual} in dim {axis}, more than i64 can hold")]
    DimTooLarge {
        /// Input or state.
        kind: ValueKind,
        /// Its name.
        name: String,
        /// The axis.
        axis: usize,
        /// The given size.
        actual: usize,
    },
    /// An inferred dim evaluates below 1.
    #[error("node `{node}` output `{port}` has size {value} in dim {axis}; every dim must be >= 1")]
    NonpositiveDim {
        /// Node path (ids of enclosing repeats and the node, joined by `/`).
        node: String,
        /// Output port.
        port: String,
        /// The axis.
        axis: usize,
        /// The evaluated size.
        value: i64,
    },
    /// A dim attribute is outside its range.
    #[error("node `{node}` attribute `{attr}`: {condition}, but it evaluates to {value}")]
    AttrOutOfRange {
        /// Node path.
        node: String,
        /// The attribute.
        attr: String,
        /// The violated condition.
        condition: String,
        /// The evaluated expression of the condition.
        value: i64,
    },
    /// An output's element count doesn't fit in `u64` or `usize`.
    #[error("node `{node}` output `{port}` has more elements than u64/usize can count")]
    TooManyElements {
        /// Node path.
        node: String,
        /// Output port.
        port: String,
    },
    /// `i64` overflow while evaluating a dim.
    #[error("integer overflow while evaluating {context}")]
    Overflow {
        /// What was evaluated: an input or state dim, or a node output or attribute.
        context: String,
    },
    /// A symbol has no value. Validation proves that every used symbol can be bound, so this
    /// indicates an analysis that doesn't belong to the graph.
    #[error("dim `{symbol}` has no value while evaluating {context}")]
    Unbound {
        /// The symbol.
        symbol: String,
        /// What was evaluated.
        context: String,
    },
}

/// The evaluated value of a dim attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundDim {
    /// `dim`.
    One(i64),
    /// `dim[]`.
    Many(Vec<i64>),
}

/// Concrete shapes and dim attributes of an op node for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundOp {
    /// Output shapes, parallel to [`OpInfo::outputs`].
    pub outputs: Vec<Vec<usize>>,
    /// The evaluated `dim`/`dim[]` attributes, in catalog order. Other attributes are in
    /// [`OpInfo::attrs`].
    pub dims: Vec<(&'static str, BoundDim)>,
}

impl BoundOp {
    /// The evaluated dim attribute `name`.
    pub fn dim(&self, name: &str) -> Option<&BoundDim> {
        self.dims.iter().find(|(k, _)| *k == name).map(|(_, v)| v)
    }
}

/// Concrete shapes of a `repeat` node for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRepeat {
    /// Carry shapes, parallel to [`RepeatInfo::carry`].
    pub carry: Vec<Vec<usize>>,
    /// Body nodes, parallel to [`RepeatInfo::nodes`] (the same in every iteration).
    pub nodes: Vec<BoundNode>,
}

/// Concrete shapes of a node for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundNode {
    /// An op node.
    Op(BoundOp),
    /// A `repeat` node.
    Repeat(BoundRepeat),
}

/// The result of [`Analysis::bind`]: symbol values and concrete shapes of every node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bindings {
    /// Bound symbols, in the order they were bound.
    pub symbols: Vec<(String, i64)>,
    /// Root nodes, parallel to [`Analysis::nodes`].
    pub nodes: Vec<BoundNode>,
}

impl Bindings {
    /// The value of `symbol`.
    pub fn symbol(&self, symbol: &str) -> Option<i64> {
        self.symbols
            .iter()
            .find(|(s, _)| s == symbol)
            .map(|(_, v)| *v)
    }
}

struct Binding {
    value: i64,
    kind: ValueKind,
    name: String,
    axis: usize,
}

impl Analysis {
    /// Checks the input and state values of one call and binds the symbols (§2.3.4 steps 1–5).
    ///
    /// `inputs` and `state` are keyed by name (state: the expanded root-level names of
    /// [`Analysis::state`]). Every declared value must be given exactly once and nothing else.
    /// Returns the symbol values and the concrete output shapes and dim attributes of every node,
    /// or the first problem found. Never panics; all arithmetic is checked.
    pub fn bind(
        &self,
        inputs: &[(&str, TensorDesc<'_>)],
        state: &[(&str, TensorDesc<'_>)],
    ) -> Result<Bindings, BindError> {
        // Step 1: presence, dtype, rank.
        let given_inputs = given_map(inputs, &self.inputs, ValueKind::Input)?;
        let given_state = given_map(state, &self.state, ValueKind::State)?;

        // Step 2: bind symbols from plain-symbol dims; literal dims must match.
        let mut bound: HashMap<&str, Binding> = HashMap::new();
        let mut order: Vec<String> = Vec::new();
        for (values, given, kind) in [
            (&self.inputs, &given_inputs, ValueKind::Input),
            (&self.state, &given_state, ValueKind::State),
        ] {
            for (v, desc) in values.iter().zip(given.iter()) {
                for (axis, (dim, &actual)) in v.info.shape.iter().zip(desc.shape).enumerate() {
                    let value = i64::try_from(actual).map_err(|_| BindError::DimTooLarge {
                        kind,
                        name: v.name.clone(),
                        axis,
                        actual,
                    })?;
                    if let Some(expected) = dim.as_literal() {
                        if expected != value {
                            return Err(BindError::DimMismatch {
                                kind,
                                name: v.name.clone(),
                                axis,
                                expected,
                                actual,
                            });
                        }
                    } else if let Some(sym) = dim.as_symbol() {
                        match bound.get(sym) {
                            Some(b) if b.value != value => {
                                return Err(BindError::InconsistentSymbol {
                                    symbol: sym.to_owned(),
                                    first: b.value,
                                    first_kind: b.kind,
                                    first_name: b.name.clone(),
                                    first_axis: b.axis,
                                    value,
                                    kind,
                                    name: v.name.clone(),
                                    axis,
                                });
                            }
                            Some(_) => {}
                            None => {
                                bound.insert(
                                    sym,
                                    Binding {
                                        value,
                                        kind,
                                        name: v.name.clone(),
                                        axis,
                                    },
                                );
                                order.push(sym.to_owned());
                            }
                        }
                    }
                }
            }
        }

        // Step 3: min/max.
        for s in &self.symbols {
            if let Some(b) = bound.get(s.name.as_str())
                && (b.value < s.min || s.max.is_some_and(|m| b.value > m))
            {
                return Err(BindError::SymbolOutOfRange {
                    symbol: s.name.clone(),
                    value: b.value,
                    min: s.min,
                    max: s.max,
                });
            }
        }
        let value_of = |name: &str| bound.get(name).map(|b| b.value);
        let eval = |e: &DimExpr, context: &dyn Fn() -> String| -> Result<i64, BindError> {
            e.eval(&value_of).map_err(|err| match err {
                EvalError::Overflow => BindError::Overflow { context: context() },
                EvalError::Unbound(symbol) => BindError::Unbound {
                    symbol,
                    context: context(),
                },
            })
        };

        // Step 4: expression dims of state shapes.
        for (v, desc) in self.state.iter().zip(given_state.iter()) {
            for (axis, (dim, &actual)) in v.info.shape.iter().zip(desc.shape).enumerate() {
                if dim.as_literal().is_some() || dim.as_symbol().is_some() {
                    continue;
                }
                let expected = eval(dim, &|| format!("state entry `{}` dim {axis}", v.name))?;
                if i64::try_from(actual).ok() != Some(expected) {
                    return Err(BindError::DimMismatch {
                        kind: ValueKind::State,
                        name: v.name.clone(),
                        axis,
                        expected,
                        actual,
                    });
                }
            }
        }

        // Step 5: every inferred dim and dim attribute.
        let mut path = Vec::new();
        let nodes = bind_nodes(&self.nodes, &mut path, &eval)?;
        let symbols = order
            .into_iter()
            .filter_map(|s| bound.get(s.as_str()).map(|b| (s.clone(), b.value)))
            .collect();
        Ok(Bindings { symbols, nodes })
    }
}

type Eval<'a> = dyn Fn(&DimExpr, &dyn Fn() -> String) -> Result<i64, BindError> + 'a;

fn given_map<'a>(
    given: &'a [(&str, TensorDesc<'a>)],
    declared: &[NamedValue],
    kind: ValueKind,
) -> Result<Vec<TensorDesc<'a>>, BindError> {
    let mut by_name: HashMap<&str, TensorDesc<'a>> = HashMap::with_capacity(given.len());
    for (name, desc) in given {
        if !declared.iter().any(|d| d.name == *name) {
            return Err(BindError::Unknown {
                kind,
                name: (*name).to_owned(),
            });
        }
        if by_name.insert(name, *desc).is_some() {
            return Err(BindError::Duplicate {
                kind,
                name: (*name).to_owned(),
            });
        }
    }
    let mut out = Vec::with_capacity(declared.len());
    for d in declared {
        let Some(desc) = by_name.get(d.name.as_str()) else {
            return Err(BindError::Missing {
                kind,
                name: d.name.clone(),
            });
        };
        if desc.dtype != d.info.dtype {
            return Err(BindError::Dtype {
                kind,
                name: d.name.clone(),
                expected: d.info.dtype,
                actual: desc.dtype,
            });
        }
        if desc.shape.len() != d.info.shape.len() {
            return Err(BindError::Rank {
                kind,
                name: d.name.clone(),
                expected: d.info.shape.len(),
                actual: desc.shape.len(),
            });
        }
        out.push(*desc);
    }
    Ok(out)
}

fn bind_nodes(
    nodes: &[NodeInfo],
    path: &mut Vec<String>,
    eval: &Eval<'_>,
) -> Result<Vec<BoundNode>, BindError> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes {
        path.push(node.id().to_owned());
        let bound = match node {
            NodeInfo::Op(op) => bind_op(op, path, eval).map(BoundNode::Op),
            NodeInfo::Repeat(r) => r
                .carry
                .iter()
                .map(|(port, info)| bind_shape(&info.shape, path, port, eval))
                .collect::<Result<Vec<_>, _>>()
                .and_then(|carry| {
                    Ok(BoundNode::Repeat(BoundRepeat {
                        carry,
                        nodes: bind_nodes(&r.nodes, path, eval)?,
                    }))
                }),
        };
        path.pop();
        out.push(bound?);
    }
    Ok(out)
}

fn node_path(path: &[String]) -> String {
    path.join("/")
}

fn bind_shape(
    shape: &[DimExpr],
    path: &[String],
    port: &str,
    eval: &Eval<'_>,
) -> Result<Vec<usize>, BindError> {
    let node = node_path(path);
    let mut dims = Vec::with_capacity(shape.len());
    let mut count: u64 = 1;
    for (axis, d) in shape.iter().enumerate() {
        let v = eval(d, &|| format!("node `{node}` output `{port}` dim {axis}"))?;
        if v < 1 {
            return Err(BindError::NonpositiveDim {
                node: node.clone(),
                port: port.to_owned(),
                axis,
                value: v,
            });
        }
        let too_many = || BindError::TooManyElements {
            node: node.clone(),
            port: port.to_owned(),
        };
        let u = u64::try_from(v).map_err(|_| too_many())?;
        count = count.checked_mul(u).ok_or_else(too_many)?;
        dims.push(usize::try_from(u).map_err(|_| too_many())?);
    }
    usize::try_from(count).map_err(|_| BindError::TooManyElements {
        node,
        port: port.to_owned(),
    })?;
    Ok(dims)
}

fn bind_op(op: &OpInfo, path: &[String], eval: &Eval<'_>) -> Result<BoundOp, BindError> {
    let node = node_path(path);
    // Attribute conditions first: they explain an output dim < 1 better than the dim itself.
    for check in &op.checks {
        let value = eval(&check.expr, &|| {
            format!("node `{node}` attribute `{}`", check.attr)
        })?;
        if value < check.min {
            return Err(BindError::AttrOutOfRange {
                node: node.clone(),
                attr: check.attr.to_owned(),
                condition: check.condition.clone(),
                value,
            });
        }
    }
    let mut dims = Vec::new();
    for (name, value) in op.attrs.iter() {
        let ctx = || format!("node `{node}` attribute `{name}`");
        match value {
            AttrValue::Dim(e) => dims.push((name, BoundDim::One(eval(e, &ctx)?))),
            AttrValue::Dims(es) => {
                let vals = es
                    .iter()
                    .map(|e| eval(e, &ctx))
                    .collect::<Result<Vec<_>, _>>()?;
                dims.push((name, BoundDim::Many(vals)));
            }
            _ => {}
        }
    }
    let mut outputs = Vec::with_capacity(op.outputs.len());
    for (port, info) in &op.outputs {
        outputs.push(bind_shape(&info.shape, path, port, eval)?);
    }
    Ok(BoundOp { outputs, dims })
}
