//! Typed graph documents, `docs/specs/graph-ir.md` §2.
//!
//! The types hold the document as written: dims are kept as written (integer or string), `float`
//! dtypes stay `float`, attribute values and `meta` are raw [`Value`]s. Validation
//! ([`crate::validate`]) checks them; saving ([`crate::save`]) writes them back unchanged.
//!
//! Maps of the document (opsets, dims, node inputs, params and attrs, carries, state updates) are
//! vectors of entries in document order, which is also the order they are saved in.

use crate::dtype::DtypeSpec;
use crate::json::Value;

/// The only `format` value.
pub const FORMAT: &str = "polypous.graph";
/// The only supported `format_version`.
pub const FORMAT_VERSION: u64 = 1;

/// A graph document (§2).
#[derive(Debug, Clone, PartialEq)]
pub struct Graph {
    /// `format_version` (only [`FORMAT_VERSION`] is supported).
    pub format_version: u64,
    /// `opsets`: opset name → version, in document order.
    pub opsets: Vec<(String, u64)>,
    /// Optional graph name.
    pub name: Option<String>,
    /// Opaque graph `meta` (§2.8).
    pub meta: Option<Value>,
    /// Declared dim symbols (§2.3), in document order.
    pub dims: Vec<DimDecl>,
    /// Graph inputs.
    pub inputs: Vec<GraphInput>,
    /// Graph outputs (at least one).
    pub outputs: Vec<GraphOutput>,
    /// Parameters (§2.4).
    pub params: Vec<Param>,
    /// Root state entries (§2.5).
    pub state: Vec<StateEntry>,
    /// Root nodes.
    pub nodes: Vec<Node>,
    /// Root `state_updates`: state name → value reference, in document order.
    pub state_updates: Vec<(String, String)>,
}

impl Graph {
    /// An empty graph with the current format version and the `core` and `nn` opsets at version 1.
    /// It doesn't validate until it has an output.
    pub fn new() -> Graph {
        Graph {
            format_version: FORMAT_VERSION,
            opsets: vec![("core".into(), 1), ("nn".into(), 1)],
            name: None,
            meta: None,
            dims: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            params: Vec::new(),
            state: Vec::new(),
            nodes: Vec::new(),
            state_updates: Vec::new(),
        }
    }
}

impl Default for Graph {
    fn default() -> Self {
        Graph::new()
    }
}

/// A declared dim symbol: `"name": { "min": …, "max": … }`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimDecl {
    /// The symbol.
    pub name: String,
    /// `min` as written (default 1).
    pub min: Option<i64>,
    /// `max` as written (no default).
    pub max: Option<i64>,
}

/// One dimension of a shape, as written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dim {
    /// An integer literal.
    Literal(i64),
    /// A string: a plain symbol or a dim expression (§2.3.1).
    Expr(String),
}

/// A graph input.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphInput {
    /// Value name.
    pub name: String,
    /// Dtype (`float` is the compute dtype).
    pub dtype: DtypeSpec,
    /// Shape: literals and plain symbols.
    pub shape: Vec<Dim>,
    /// Opaque `meta`.
    pub meta: Option<Value>,
}

/// A graph output: `{ "name": …, "value": <reference> }`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphOutput {
    /// Output name (its own namespace).
    pub name: String,
    /// Value reference.
    pub value: String,
    /// Opaque `meta`.
    pub meta: Option<Value>,
}

/// A parameter (§2.4).
#[derive(Debug, Clone, PartialEq)]
pub struct Param {
    /// Parameter name (may contain dots).
    pub name: String,
    /// Dtype; must be `float` in M1.
    pub dtype: DtypeSpec,
    /// Shape: literals only.
    pub shape: Vec<Dim>,
    /// Initializer (parsed and validated in M1, applied in M2).
    pub init: Option<Init>,
    /// `trainable` as written (default `true`).
    pub trainable: Option<bool>,
    /// Opaque `meta`.
    pub meta: Option<Value>,
}

/// A parameter initializer (§2.4).
#[derive(Debug, Clone, PartialEq)]
pub enum Init {
    /// `zeros`
    Zeros,
    /// `ones`
    Ones,
    /// `constant{value}`
    Constant {
        /// The value.
        value: f64,
    },
    /// `uniform{low, high}`
    Uniform {
        /// Lower bound.
        low: f64,
        /// Upper bound.
        high: f64,
    },
    /// `normal{mean, std}`
    Normal {
        /// Mean.
        mean: f64,
        /// Standard deviation.
        std: f64,
    },
    /// `xavier_uniform`
    XavierUniform,
    /// `xavier_normal`
    XavierNormal,
    /// `kaiming_uniform{a}`, `a` optional.
    KaimingUniform {
        /// Negative slope `a`, if written.
        a: Option<f64>,
    },
    /// `kaiming_normal`
    KaimingNormal,
}

impl Init {
    /// The `kind` string.
    pub fn kind(&self) -> &'static str {
        match self {
            Init::Zeros => "zeros",
            Init::Ones => "ones",
            Init::Constant { .. } => "constant",
            Init::Uniform { .. } => "uniform",
            Init::Normal { .. } => "normal",
            Init::XavierUniform => "xavier_uniform",
            Init::XavierNormal => "xavier_normal",
            Init::KaimingUniform { .. } => "kaiming_uniform",
            Init::KaimingNormal => "kaiming_normal",
        }
    }
}

/// A state entry (§2.5).
#[derive(Debug, Clone, PartialEq)]
pub struct StateEntry {
    /// State name (body-local in a `repeat` body).
    pub name: String,
    /// Dtype (`float` is the compute dtype).
    pub dtype: DtypeSpec,
    /// Shape: literals, symbols and expressions.
    pub shape: Vec<Dim>,
    /// `init` as written; `"zeros"` is the only kind (and the default).
    pub init: Option<StateInit>,
}

/// The initial value of a state entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateInit {
    /// `"zeros"`
    Zeros,
}

/// A node (§2.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// Node id (a value name).
    pub id: String,
    /// What the node does.
    pub kind: NodeKind,
    /// Opaque `meta`.
    pub meta: Option<Value>,
}

/// An op node or a `repeat` block.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeKind {
    /// An op (`opset.op_name`).
    Op(OpNode),
    /// A `repeat` block (§2.6).
    Repeat(Repeat),
}

/// An op node.
#[derive(Debug, Clone, PartialEq)]
pub struct OpNode {
    /// Full op name, `opset.op_name`.
    pub op: String,
    /// Op input name → reference(s), in document order.
    pub inputs: Vec<(String, InputRef)>,
    /// Op param name → graph param name (body-local in a `repeat` body), in document order.
    pub params: Vec<(String, String)>,
    /// Attribute name → raw value, in document order. Typed by the op catalog.
    pub attrs: Vec<(String, Value)>,
}

/// The reference(s) bound to an op input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputRef {
    /// One reference.
    One(String),
    /// A variadic input: a list of references.
    Many(Vec<String>),
}

/// A `repeat` block (§2.6).
#[derive(Debug, Clone, PartialEq)]
pub struct Repeat {
    /// `count` as written (valid: 1–65,536).
    pub count: i64,
    /// Carry name → initial value reference (enclosing scope), in document order.
    pub carry: Vec<(String, String)>,
    /// The body.
    pub body: Body,
}

/// The body of a `repeat` block.
#[derive(Debug, Clone, PartialEq)]
pub struct Body {
    /// `params_prefix` as written (may contain `{i}`).
    pub params_prefix: Option<String>,
    /// `state_prefix` as written (may contain `{i}`).
    pub state_prefix: Option<String>,
    /// Body state entries, named without the prefix.
    pub state: Vec<StateEntry>,
    /// Body nodes.
    pub nodes: Vec<Node>,
    /// Body-local state name → body value reference, in document order.
    pub state_updates: Vec<(String, String)>,
    /// Carry name → body value reference, in document order.
    pub carry_out: Vec<(String, String)>,
}
