//! The M1 op catalog, `docs/specs/graph-ir.md` §3.2 and §3.3.
//!
//! One [`OpDef`] per table row, in table order. Each entry is written so that it can be checked
//! against its row at a glance: inputs with their dtype class (§3.1.1) and rank range, params,
//! attributes with their type, default (spelled exactly as in the spec, as JSON) and range, output
//! ports with their dtype rule, the shape rule and the differentiable flag. A test parses the spec
//! tables and compares them with this catalog (`tests/catalog.rs`), so the two can't drift apart.
//!
//! The rules that need the op's other inputs (axis ranges, rank-dependent list lengths, scalar values
//! for the filled dtype, shape relations) are applied by shape inference (`infer.rs`), keyed by
//! [`ShapeRule`].

use crate::dtype::Dtype;

/// The opsets this version supports, with their supported version.
pub const SUPPORTED_OPSETS: &[(&str, u64)] = &[("core", 1), ("nn", 1)];

/// A dtype class (§3.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// `F`: `f32 f16 bf16`.
    F,
    /// `I`: `i64 i32`.
    I,
    /// `B`: `bool`.
    B,
    /// `N`: `F` or `I`.
    N,
    /// `A`: any dtype.
    A,
}

impl Class {
    /// Whether `dtype` is in the class.
    pub fn contains(self, dtype: Dtype) -> bool {
        match self {
            Class::F => dtype.is_float(),
            Class::I => matches!(dtype, Dtype::I64 | Dtype::I32),
            Class::B => dtype == Dtype::Bool,
            Class::N => dtype != Dtype::Bool,
            Class::A => true,
        }
    }

    /// The letter used in the spec tables.
    pub fn letter(self) -> &'static str {
        match self {
            Class::F => "F",
            Class::I => "I",
            Class::B => "B",
            Class::N => "N",
            Class::A => "A",
        }
    }
}

/// The dtype rule of an op input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InDtype {
    /// Any dtype of the class. Index inputs (`index`, `ids`) are `Class(I)` and independent of the
    /// other inputs (§3.1.2).
    Class(Class),
    /// `T`: exactly the dtype of the named input.
    Same(&'static str),
    /// `C`: a float input that must have the compute dtype (`nn` ops, §3.3).
    Compute,
}

impl InDtype {
    /// The letter used in the spec tables.
    pub fn letter(self) -> &'static str {
        match self {
            InDtype::Class(c) => c.letter(),
            InDtype::Same(_) => "T",
            InDtype::Compute => "C",
        }
    }
}

/// An op input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputDef {
    /// Input name.
    pub name: &'static str,
    /// Dtype rule.
    pub dtype: InDtype,
    /// A variadic input takes a list of references (only `core.concat`, 1–1,024 entries).
    pub variadic: bool,
    /// Smallest allowed rank (`rank_mismatch` below).
    pub min_rank: usize,
    /// Largest allowed rank (`rank_mismatch` above).
    pub max_rank: usize,
}

/// An op param (`nn` ops only). Params are graph params of dtype `float`, seen in the compute dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParamDef {
    /// Param name.
    pub name: &'static str,
    /// Whether the param may be left out (`?` in the spec).
    pub optional: bool,
    /// The shape as the spec writes it, for documentation (`[out, in]`).
    pub shape: &'static str,
}

/// The length rule of a list attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListLen {
    /// Exactly `n` entries (`int[2]`).
    Exactly(usize),
    /// `min..=max` entries. Rules that depend on the input rank are applied by shape inference.
    Range(usize, usize),
}

impl ListLen {
    /// Whether `len` entries are allowed.
    pub fn allows(self, len: usize) -> bool {
        match self {
            ListLen::Exactly(n) => len == n,
            ListLen::Range(lo, hi) => (lo..=hi).contains(&len),
        }
    }

    fn describe(self) -> String {
        match self {
            ListLen::Exactly(n) => format!("exactly {n} entries"),
            ListLen::Range(lo, hi) => format!("{lo}–{hi} entries"),
        }
    }
}

/// The range of a `float` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatRange {
    /// Any value, `inf` and `nan` included.
    Any,
    /// `> 0`.
    Positive,
    /// `0 ≤ p < 1`.
    Probability,
}

impl FloatRange {
    /// Whether `v` is in the range (NaN is in no range but `Any`).
    pub fn allows(self, v: f64) -> bool {
        match self {
            FloatRange::Any => true,
            FloatRange::Positive => v > 0.0,
            FloatRange::Probability => (0.0..1.0).contains(&v),
        }
    }

    fn describe(self) -> &'static str {
        match self {
            FloatRange::Any => "any number",
            FloatRange::Positive => "> 0",
            FloatRange::Probability => "0 <= p < 1",
        }
    }
}

/// The dtypes a `dtype` attribute allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeSet {
    /// Any concrete dtype, and `float` (the compute dtype).
    AnyOrFloat,
    /// `N` or `float` (`core.arange`).
    NumericOrFloat,
    /// `I` only; `float` is not allowed (§2.2).
    Index,
}

impl DtypeSet {
    /// Whether a concrete `dtype` is allowed.
    pub fn allows(self, dtype: Dtype) -> bool {
        match self {
            DtypeSet::AnyOrFloat => true,
            DtypeSet::NumericOrFloat => dtype != Dtype::Bool,
            DtypeSet::Index => Class::I.contains(dtype),
        }
    }

    /// Whether `"float"` is allowed.
    pub fn allows_float(self) -> bool {
        !matches!(self, DtypeSet::Index)
    }

    fn describe(self) -> &'static str {
        match self {
            DtypeSet::AnyOrFloat => "f32, f16, bf16, i64, i32, bool or float",
            DtypeSet::NumericOrFloat => "f32, f16, bf16, i64, i32 or float",
            DtypeSet::Index => "i64 or i32",
        }
    }
}

/// The type of an attribute (§3.1.7), with the range that doesn't depend on the inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrType {
    /// `int`, at least `min` if given.
    Int {
        /// Smallest allowed value.
        min: Option<i64>,
    },
    /// `int`: an axis of the first input, `[-r, r-1]` (§3.1.4).
    Axis,
    /// `int`: an axis of the output of `core.unsqueeze`, `[-(r+1), r]`.
    AxisInsert,
    /// `int[]`: non-empty list of distinct axes of the first input (at most `r` entries).
    Axes,
    /// `int[]`: a permutation of the first input's axes (length `r`).
    Permutation,
    /// `int[]` or `int[n]`: integers, each at least `min`.
    Ints {
        /// Length rule.
        len: ListLen,
        /// Smallest allowed entry.
        min: i64,
    },
    /// `float` (number, `"inf"`, `"-inf"` or `"nan"`).
    Float(FloatRange),
    /// `bool`.
    Bool,
    /// `enum`: one of the listed strings.
    Enum(&'static [&'static str]),
    /// `dim`: an integer or a dim expression; ranges are per op (shape inference).
    Dim,
    /// `dim[]`.
    Dims(ListLen),
    /// `dtype`.
    Dtype(DtypeSet),
    /// `scalar`: a value of the dtype the op fills (checked by shape inference).
    Scalar,
    /// `scalar[]` (`core.constant` `values`).
    Scalars,
}

impl AttrType {
    /// The type as the spec tables write it (`int`, `int[2]`, `dim[]`, …).
    pub fn spec_type(self) -> String {
        match self {
            AttrType::Int { .. } | AttrType::Axis | AttrType::AxisInsert => "int".into(),
            AttrType::Axes | AttrType::Permutation => "int[]".into(),
            AttrType::Ints {
                len: ListLen::Exactly(n),
                ..
            } => format!("int[{n}]"),
            AttrType::Ints { .. } => "int[]".into(),
            AttrType::Float(_) => "float".into(),
            AttrType::Bool => "bool".into(),
            AttrType::Enum(_) => "enum".into(),
            AttrType::Dim => "dim".into(),
            AttrType::Dims(_) => "dim[]".into(),
            AttrType::Dtype(_) => "dtype".into(),
            AttrType::Scalar => "scalar".into(),
            AttrType::Scalars => "scalar[]".into(),
        }
    }

    /// A short description of the allowed values, for error messages.
    pub fn describe(self) -> String {
        match self {
            AttrType::Int { min: Some(m) } => format!("an integer >= {m}"),
            AttrType::Int { min: None } | AttrType::Axis | AttrType::AxisInsert => {
                "an integer".into()
            }
            AttrType::Axes => "a non-empty array of distinct axes".into(),
            AttrType::Permutation => "a permutation of the input's axes".into(),
            AttrType::Ints { len, min } => {
                format!("an array of integers >= {min} with {}", len.describe())
            }
            AttrType::Float(r) => {
                format!("a number or \"inf\", \"-inf\", \"nan\" ({})", r.describe())
            }
            AttrType::Bool => "true or false".into(),
            AttrType::Enum(values) => format!("one of {}", values.join(", ")),
            AttrType::Dim => "an integer or a dim expression string".into(),
            AttrType::Dims(len) => format!(
                "an array of integers or dim expression strings with {}",
                len.describe()
            ),
            AttrType::Dtype(set) => format!("a dtype name ({})", set.describe()),
            AttrType::Scalar => "a number, \"inf\", \"-inf\", \"nan\", true or false".into(),
            AttrType::Scalars => "an array of scalars".into(),
        }
    }
}

/// Whether an attribute is required, optional, or has a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    /// Required (no default in the spec).
    Required,
    /// Optional with no default (`?` in the spec). Shape inference fills in derived defaults where
    /// the spec gives one in words (for example `max_pool2d` `stride` = `kernel_size`).
    Optional,
    /// Optional with this default, as JSON exactly as the spec writes it (`1e-5`, `[1, 1]`,
    /// `"i64"`).
    Default(&'static str),
}

/// An op attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttrDef {
    /// Attribute name.
    pub name: &'static str,
    /// Type and static range.
    pub ty: AttrType,
    /// Required, optional or default.
    pub presence: Presence,
}

/// The dtype rule of an output port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutDtype {
    /// `T`: the dtype of the named input.
    Same(&'static str),
    /// `B`: `bool`.
    Bool,
    /// `dtype`: the `dtype` attribute, `float` resolved to the compute dtype.
    Attr,
    /// `dtype`: the `dtype` attribute if given, else the dtype of the named input (`full_like`).
    AttrOr(&'static str),
    /// `C`: the compute dtype.
    Compute,
}

impl OutDtype {
    /// The notation of the spec tables.
    pub fn spec(self) -> &'static str {
        match self {
            OutDtype::Same(_) => "T",
            OutDtype::Bool => "B",
            OutDtype::Attr | OutDtype::AttrOr(_) => "dtype",
            OutDtype::Compute => "C",
        }
    }
}

/// An output port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputDef {
    /// Port name.
    pub name: &'static str,
    /// Dtype rule.
    pub dtype: OutDtype,
}

/// The output ports of an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outputs {
    /// Fixed ports.
    Fixed(&'static [OutputDef]),
    /// `core.split`: ports `0` … `n-1`, one per `sizes` entry, all with the input's dtype.
    Split,
}

/// Which gradients an op has (M2; the column fixes them now, §3.1.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Differentiable {
    /// With respect to every float input and every param.
    Yes,
    /// None.
    No,
    /// Only with respect to (or, for `top_k`, only for) the listed inputs, params or ports.
    Only(&'static [&'static str]),
    /// `core.cast`: only from a float dtype to a float dtype.
    FloatToFloat,
}

impl Differentiable {
    /// The notation of the spec tables.
    pub fn spec(self) -> String {
        match self {
            Differentiable::Yes => "yes".into(),
            Differentiable::No => "no".into(),
            Differentiable::Only(names) => {
                let list: Vec<String> = names.iter().map(|n| format!("`{n}`")).collect();
                format!("yes ({})", list.join(", "))
            }
            Differentiable::FloatToFloat => "float → float only".into(),
        }
    }
}

/// How the output shape is inferred; each rule is one function of shape inference (`infer.rs`) and follows the
/// op's row and notes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShapeRule {
    /// `x.shape`.
    Unary,
    /// Broadcast of `a` and `b` (§3.1.3).
    Broadcast,
    /// `x.shape`; `min`/`max` scalars of `x`'s dtype, at least one, `min <= max`.
    Clamp,
    /// Broadcast of `cond`, `a` and `b`.
    Where,
    /// `core.reshape`.
    Reshape,
    /// `core.flatten`.
    Flatten,
    /// `core.permute`.
    Permute,
    /// `core.transpose`.
    Transpose,
    /// `core.squeeze`.
    Squeeze,
    /// `core.unsqueeze`.
    Unsqueeze,
    /// `core.expand`.
    Expand,
    /// `core.concat`.
    Concat,
    /// `core.split`.
    Split,
    /// `core.slice`.
    Slice,
    /// `core.pad`.
    Pad,
    /// `core.gather`.
    Gather,
    /// `core.index_select`.
    IndexSelect,
    /// `core.constant`.
    Constant,
    /// `core.arange`.
    Arange,
    /// `core.zeros`, `core.ones`, `core.full`: the `shape` attribute.
    Fill,
    /// `core.full_like`: `x.shape`.
    FullLike,
    /// `core.sum`, `mean`, `max`, `min`.
    Reduce,
    /// `core.argmax`, `argmin`.
    ArgReduce,
    /// `core.matmul`.
    Matmul,
    /// `core.top_k`.
    TopK,
    /// `x.shape` with a `dim` axis attribute (`softmax`, `log_softmax`, `cumsum`).
    UnaryAxis,
    /// `nn.linear`.
    Linear,
    /// `nn.embedding`.
    Embedding,
    /// `nn.conv1d`.
    Conv1d,
    /// `nn.conv2d`.
    Conv2d,
    /// `nn.max_pool2d`.
    MaxPool2d,
    /// `nn.avg_pool2d`.
    AvgPool2d,
    /// `nn.adaptive_avg_pool2d`.
    AdaptiveAvgPool2d,
    /// `nn.batch_norm`.
    BatchNorm,
    /// `nn.layer_norm`.
    LayerNorm,
    /// `nn.group_norm`.
    GroupNorm,
    /// `nn.rms_norm`.
    RmsNorm,
}

/// One op of the catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpDef {
    /// Full op name, `opset.op_name`.
    pub name: &'static str,
    /// Inputs, in table order.
    pub inputs: &'static [InputDef],
    /// Params, in table order (`nn` ops only).
    pub params: &'static [ParamDef],
    /// Attributes, in table order.
    pub attrs: &'static [AttrDef],
    /// Output ports.
    pub outputs: Outputs,
    /// Shape rule.
    pub shape: ShapeRule,
    /// Differentiable column.
    pub differentiable: Differentiable,
}

impl OpDef {
    /// The opset of the op (`nn` for `nn.linear`).
    pub fn opset(&self) -> &'static str {
        self.name.split_once('.').map_or(self.name, |(s, _)| s)
    }

    /// The input definition named `name`.
    pub fn input(&self, name: &str) -> Option<&'static InputDef> {
        self.inputs.iter().find(|i| i.name == name)
    }

    /// The param definition named `name`.
    pub fn param(&self, name: &str) -> Option<&'static ParamDef> {
        self.params.iter().find(|p| p.name == name)
    }

    /// The attribute definition named `name`.
    pub fn attr(&self, name: &str) -> Option<&'static AttrDef> {
        self.attrs.iter().find(|a| a.name == name)
    }
}

/// Looks an op up by its full name.
pub fn lookup(name: &str) -> Option<&'static OpDef> {
    OPS.iter().find(|op| op.name == name)
}

// ---- Builders, so that each table row below fits on a few lines. ----

const fn input(name: &'static str, dtype: InDtype) -> InputDef {
    InputDef {
        name,
        dtype,
        variadic: false,
        min_rank: 1,
        max_rank: 6,
    }
}

const fn ranked(name: &'static str, dtype: InDtype, min_rank: usize, max_rank: usize) -> InputDef {
    InputDef {
        name,
        dtype,
        variadic: false,
        min_rank,
        max_rank,
    }
}

const fn param(name: &'static str, shape: &'static str) -> ParamDef {
    ParamDef {
        name,
        optional: false,
        shape,
    }
}

const fn opt_param(name: &'static str, shape: &'static str) -> ParamDef {
    ParamDef {
        name,
        optional: true,
        shape,
    }
}

const fn req(name: &'static str, ty: AttrType) -> AttrDef {
    AttrDef {
        name,
        ty,
        presence: Presence::Required,
    }
}

const fn opt(name: &'static str, ty: AttrType) -> AttrDef {
    AttrDef {
        name,
        ty,
        presence: Presence::Optional,
    }
}

const fn def(name: &'static str, ty: AttrType, default: &'static str) -> AttrDef {
    AttrDef {
        name,
        ty,
        presence: Presence::Default(default),
    }
}

const fn port(dtype: OutDtype) -> OutputDef {
    OutputDef { name: "out", dtype }
}

const OUT_T_A: Outputs = Outputs::Fixed(&[port(OutDtype::Same("a"))]);
const OUT_T_X: Outputs = Outputs::Fixed(&[port(OutDtype::Same("x"))]);
const OUT_B: Outputs = Outputs::Fixed(&[port(OutDtype::Bool)]);
const OUT_DTYPE: Outputs = Outputs::Fixed(&[port(OutDtype::Attr)]);
const OUT_C: Outputs = Outputs::Fixed(&[port(OutDtype::Compute)]);

use AttrType as T;
use Class::{A, B, F, I, N};
use InDtype::{Class as Cl, Compute, Same};

const AB_N: &[InputDef] = &[input("a", Cl(N)), input("b", Same("a"))];
const AB_F: &[InputDef] = &[input("a", Cl(F)), input("b", Same("a"))];
const AB_A: &[InputDef] = &[input("a", Cl(A)), input("b", Same("a"))];
const AB_B: &[InputDef] = &[input("a", Cl(B)), input("b", Cl(B))];
const X_N: &[InputDef] = &[input("x", Cl(N))];
const X_F: &[InputDef] = &[input("x", Cl(F))];
const X_B: &[InputDef] = &[input("x", Cl(B))];
const X_A: &[InputDef] = &[input("x", Cl(A))];
const X_C: &[InputDef] = &[input("x", Compute)];
const X_C_NCHW: &[InputDef] = &[ranked("x", Compute, 4, 4)];
const X_C_NC: &[InputDef] = &[ranked("x", Compute, 2, 6)];

const AXIS: AttrDef = req("dim", T::Axis);
const SHAPE_DIMS: AttrDef = req("shape", T::Dims(ListLen::Range(1, 6)));
const DTYPE_F32: AttrDef = def("dtype", T::Dtype(DtypeSet::AnyOrFloat), "\"f32\"");
const DTYPE_I64: AttrDef = def("dtype", T::Dtype(DtypeSet::Index), "\"i64\"");
const KEEPDIM: AttrDef = def("keepdim", T::Bool, "false");
const REDUCE_ATTRS: &[AttrDef] = &[opt("dim", T::Axes), KEEPDIM];
const ARG_ATTRS: &[AttrDef] = &[AXIS, KEEPDIM, DTYPE_I64];
const EPS: AttrDef = def("eps", T::Float(FloatRange::Positive), "1e-5");
const INT2_1: AttrType = T::Ints {
    len: ListLen::Exactly(2),
    min: 1,
};
const INT2_0: AttrType = T::Ints {
    len: ListLen::Exactly(2),
    min: 0,
};
const NORMALIZED_SHAPE: AttrDef = opt(
    "normalized_shape",
    T::Ints {
        len: ListLen::Range(1, 6),
        min: 1,
    },
);
const WEIGHT_BIAS: &[ParamDef] = &[param("weight", "[out, in]"), opt_param("bias", "[out]")];

const fn binary(name: &'static str, inputs: &'static [InputDef]) -> OpDef {
    OpDef {
        name,
        inputs,
        params: &[],
        attrs: &[],
        outputs: OUT_T_A,
        shape: ShapeRule::Broadcast,
        differentiable: Differentiable::Yes,
    }
}

const fn compare(name: &'static str, inputs: &'static [InputDef]) -> OpDef {
    OpDef {
        name,
        inputs,
        params: &[],
        attrs: &[],
        outputs: OUT_B,
        shape: ShapeRule::Broadcast,
        differentiable: Differentiable::No,
    }
}

const fn unary(name: &'static str, inputs: &'static [InputDef]) -> OpDef {
    OpDef {
        name,
        inputs,
        params: &[],
        attrs: &[],
        outputs: OUT_T_X,
        shape: ShapeRule::Unary,
        differentiable: Differentiable::Yes,
    }
}

const fn shape_op(name: &'static str, attrs: &'static [AttrDef], shape: ShapeRule) -> OpDef {
    OpDef {
        name,
        inputs: X_A,
        params: &[],
        attrs,
        outputs: OUT_T_X,
        shape,
        differentiable: Differentiable::Yes,
    }
}

const fn creation(name: &'static str, attrs: &'static [AttrDef], shape: ShapeRule) -> OpDef {
    OpDef {
        name,
        inputs: &[],
        params: &[],
        attrs,
        outputs: OUT_DTYPE,
        shape,
        differentiable: Differentiable::No,
    }
}

const fn reduce(name: &'static str, inputs: &'static [InputDef]) -> OpDef {
    OpDef {
        name,
        inputs,
        params: &[],
        attrs: REDUCE_ATTRS,
        outputs: OUT_T_X,
        shape: ShapeRule::Reduce,
        differentiable: Differentiable::Yes,
    }
}

const fn arg_reduce(name: &'static str) -> OpDef {
    OpDef {
        name,
        inputs: X_N,
        params: &[],
        attrs: ARG_ATTRS,
        outputs: OUT_DTYPE,
        shape: ShapeRule::ArgReduce,
        differentiable: Differentiable::No,
    }
}

const fn axis_op(name: &'static str, inputs: &'static [InputDef]) -> OpDef {
    OpDef {
        name,
        inputs,
        params: &[],
        attrs: &[AXIS],
        outputs: OUT_T_X,
        shape: ShapeRule::UnaryAxis,
        differentiable: Differentiable::Yes,
    }
}

const fn nn(
    name: &'static str,
    inputs: &'static [InputDef],
    params: &'static [ParamDef],
    attrs: &'static [AttrDef],
    shape: ShapeRule,
) -> OpDef {
    OpDef {
        name,
        inputs,
        params,
        attrs,
        outputs: OUT_C,
        shape,
        differentiable: Differentiable::Yes,
    }
}

const fn activation(name: &'static str, attrs: &'static [AttrDef]) -> OpDef {
    nn(name, X_C, &[], attrs, ShapeRule::Unary)
}

/// Every M1 op, in the order of the spec tables (§3.2, then §3.3).
pub const OPS: &[OpDef] = &[
    // §3.2 Elementwise
    binary("core.add", AB_N),
    binary("core.sub", AB_N),
    binary("core.mul", AB_N),
    binary("core.div", AB_F),
    binary("core.pow", AB_F),
    binary("core.maximum", AB_N),
    binary("core.minimum", AB_N),
    unary("core.neg", X_N),
    unary("core.abs", X_N),
    unary("core.exp", X_F),
    unary("core.log", X_F),
    unary("core.sqrt", X_F),
    unary("core.rsqrt", X_F),
    unary("core.sin", X_F),
    unary("core.cos", X_F),
    unary("core.tanh", X_F),
    OpDef {
        name: "core.clamp",
        inputs: X_N,
        params: &[],
        attrs: &[opt("min", T::Scalar), opt("max", T::Scalar)],
        outputs: OUT_T_X,
        shape: ShapeRule::Clamp,
        differentiable: Differentiable::Yes,
    },
    OpDef {
        name: "core.where",
        inputs: &[
            input("cond", Cl(B)),
            input("a", Cl(A)),
            input("b", Same("a")),
        ],
        params: &[],
        attrs: &[],
        outputs: OUT_T_A,
        shape: ShapeRule::Where,
        differentiable: Differentiable::Only(&["a", "b"]),
    },
    // §3.2 Comparison and logic
    compare("core.eq", AB_A),
    compare("core.ne", AB_A),
    compare("core.lt", AB_N),
    compare("core.le", AB_N),
    compare("core.gt", AB_N),
    compare("core.ge", AB_N),
    compare("core.and", AB_B),
    compare("core.or", AB_B),
    OpDef {
        name: "core.not",
        inputs: X_B,
        params: &[],
        attrs: &[],
        outputs: OUT_B,
        shape: ShapeRule::Unary,
        differentiable: Differentiable::No,
    },
    // §3.2 Shape
    shape_op("core.reshape", &[SHAPE_DIMS], ShapeRule::Reshape),
    shape_op(
        "core.flatten",
        &[
            def("start_dim", T::Axis, "0"),
            def("end_dim", T::Axis, "-1"),
        ],
        ShapeRule::Flatten,
    ),
    shape_op(
        "core.permute",
        &[req("dims", T::Permutation)],
        ShapeRule::Permute,
    ),
    shape_op(
        "core.transpose",
        &[req("dim0", T::Axis), req("dim1", T::Axis)],
        ShapeRule::Transpose,
    ),
    shape_op("core.squeeze", &[opt("dim", T::Axes)], ShapeRule::Squeeze),
    shape_op(
        "core.unsqueeze",
        &[req("dim", T::AxisInsert)],
        ShapeRule::Unsqueeze,
    ),
    shape_op("core.expand", &[SHAPE_DIMS], ShapeRule::Expand),
    OpDef {
        name: "core.concat",
        inputs: &[InputDef {
            name: "xs",
            dtype: Cl(A),
            variadic: true,
            min_rank: 1,
            max_rank: 6,
        }],
        params: &[],
        attrs: &[AXIS],
        outputs: Outputs::Fixed(&[port(OutDtype::Same("xs"))]),
        shape: ShapeRule::Concat,
        differentiable: Differentiable::Yes,
    },
    OpDef {
        name: "core.split",
        inputs: X_A,
        params: &[],
        attrs: &[
            AXIS,
            req(
                "sizes",
                T::Dims(ListLen::Range(1, crate::limits::MAX_CONCAT_SPLIT)),
            ),
        ],
        outputs: Outputs::Split,
        shape: ShapeRule::Split,
        differentiable: Differentiable::Yes,
    },
    shape_op(
        "core.slice",
        &[
            AXIS,
            def("start", T::Dim, "0"),
            opt("end", T::Dim),
            def("step", T::Int { min: Some(1) }, "1"),
        ],
        ShapeRule::Slice,
    ),
    shape_op(
        "core.pad",
        &[
            req("pad", T::Dims(ListLen::Range(2, 12))),
            def("mode", T::Enum(&["constant"]), "\"constant\""),
            def("value", T::Scalar, "0"),
        ],
        ShapeRule::Pad,
    ),
    // §3.2 Indexing
    OpDef {
        name: "core.gather",
        inputs: &[input("x", Cl(A)), input("index", Cl(I))],
        params: &[],
        attrs: &[AXIS],
        outputs: OUT_T_X,
        shape: ShapeRule::Gather,
        differentiable: Differentiable::Only(&["x"]),
    },
    OpDef {
        name: "core.index_select",
        inputs: &[input("x", Cl(A)), ranked("index", Cl(I), 1, 1)],
        params: &[],
        attrs: &[AXIS],
        outputs: OUT_T_X,
        shape: ShapeRule::IndexSelect,
        differentiable: Differentiable::Only(&["x"]),
    },
    // §3.2 Creation
    creation(
        "core.constant",
        &[
            req("dtype", T::Dtype(DtypeSet::AnyOrFloat)),
            req(
                "shape",
                T::Ints {
                    len: ListLen::Range(1, 6),
                    min: 1,
                },
            ),
            req("values", T::Scalars),
        ],
        ShapeRule::Constant,
    ),
    creation(
        "core.arange",
        &[
            def("start", T::Dim, "0"),
            req("end", T::Dim),
            def("step", T::Int { min: Some(1) }, "1"),
            def("dtype", T::Dtype(DtypeSet::NumericOrFloat), "\"i64\""),
        ],
        ShapeRule::Arange,
    ),
    creation("core.zeros", &[SHAPE_DIMS, DTYPE_F32], ShapeRule::Fill),
    creation("core.ones", &[SHAPE_DIMS, DTYPE_F32], ShapeRule::Fill),
    creation(
        "core.full",
        &[SHAPE_DIMS, req("value", T::Scalar), DTYPE_F32],
        ShapeRule::Fill,
    ),
    OpDef {
        name: "core.full_like",
        inputs: X_A,
        params: &[],
        attrs: &[
            req("value", T::Scalar),
            opt("dtype", T::Dtype(DtypeSet::AnyOrFloat)),
        ],
        outputs: Outputs::Fixed(&[port(OutDtype::AttrOr("x"))]),
        shape: ShapeRule::FullLike,
        differentiable: Differentiable::No,
    },
    // §3.2 Reduction
    reduce("core.sum", X_N),
    reduce("core.mean", X_F),
    reduce("core.max", X_N),
    reduce("core.min", X_N),
    arg_reduce("core.argmax"),
    arg_reduce("core.argmin"),
    // §3.2 Linear algebra
    OpDef {
        name: "core.matmul",
        inputs: &[ranked("a", Cl(F), 2, 6), ranked("b", Same("a"), 2, 6)],
        params: &[],
        attrs: &[],
        outputs: OUT_T_A,
        shape: ShapeRule::Matmul,
        differentiable: Differentiable::Yes,
    },
    // §3.2 Other
    OpDef {
        name: "core.cast",
        inputs: X_A,
        params: &[],
        attrs: &[req("dtype", T::Dtype(DtypeSet::AnyOrFloat))],
        outputs: OUT_DTYPE,
        shape: ShapeRule::Unary,
        differentiable: Differentiable::FloatToFloat,
    },
    axis_op("core.softmax", X_F),
    axis_op("core.log_softmax", X_F),
    axis_op("core.cumsum", X_N),
    OpDef {
        name: "core.top_k",
        inputs: X_F,
        params: &[],
        attrs: &[
            req("k", T::Dim),
            def("dim", T::Axis, "-1"),
            def("largest", T::Bool, "true"),
            DTYPE_I64,
        ],
        outputs: Outputs::Fixed(&[
            OutputDef {
                name: "values",
                dtype: OutDtype::Same("x"),
            },
            OutputDef {
                name: "indices",
                dtype: OutDtype::Attr,
            },
        ]),
        shape: ShapeRule::TopK,
        differentiable: Differentiable::Only(&["values"]),
    },
    // §3.3 Layers
    nn("nn.linear", X_C, WEIGHT_BIAS, &[], ShapeRule::Linear),
    OpDef {
        name: "nn.embedding",
        // Rank 6 passes this range and is `rank_too_large` (§3.1.5): the output would have rank 7.
        inputs: &[input("ids", Cl(I))],
        params: &[param("weight", "[num, dim]")],
        attrs: &[],
        outputs: OUT_C,
        shape: ShapeRule::Embedding,
        differentiable: Differentiable::Only(&["weight"]),
    },
    nn(
        "nn.conv1d",
        &[ranked("x", Compute, 3, 3)],
        &[
            param("weight", "[C_out, C_in/groups, k]"),
            opt_param("bias", "[C_out]"),
        ],
        &[
            def("stride", T::Int { min: Some(1) }, "1"),
            def("padding", T::Int { min: Some(0) }, "0"),
            def("dilation", T::Int { min: Some(1) }, "1"),
            def("groups", T::Int { min: Some(1) }, "1"),
        ],
        ShapeRule::Conv1d,
    ),
    nn(
        "nn.conv2d",
        X_C_NCHW,
        &[
            param("weight", "[C_out, C_in/groups, kh, kw]"),
            opt_param("bias", "[C_out]"),
        ],
        &[
            def("stride", INT2_1, "[1, 1]"),
            def("padding", INT2_0, "[0, 0]"),
            def("dilation", INT2_1, "[1, 1]"),
            def("groups", T::Int { min: Some(1) }, "1"),
        ],
        ShapeRule::Conv2d,
    ),
    nn(
        "nn.max_pool2d",
        X_C_NCHW,
        &[],
        &[
            req("kernel_size", INT2_1),
            opt("stride", INT2_1),
            def("padding", INT2_0, "[0, 0]"),
            def("dilation", INT2_1, "[1, 1]"),
        ],
        ShapeRule::MaxPool2d,
    ),
    nn(
        "nn.avg_pool2d",
        X_C_NCHW,
        &[],
        &[
            req("kernel_size", INT2_1),
            opt("stride", INT2_1),
            def("padding", INT2_0, "[0, 0]"),
            def("count_include_pad", T::Bool, "true"),
        ],
        ShapeRule::AvgPool2d,
    ),
    nn(
        "nn.adaptive_avg_pool2d",
        X_C_NCHW,
        &[],
        &[req("output_size", INT2_1)],
        ShapeRule::AdaptiveAvgPool2d,
    ),
    nn(
        "nn.dropout",
        X_C,
        &[],
        &[def("p", T::Float(FloatRange::Probability), "0.5")],
        ShapeRule::Unary,
    ),
    // §3.3 Normalization
    OpDef {
        name: "nn.batch_norm",
        inputs: X_C_NC,
        params: &[
            param("running_mean", "[C]"),
            param("running_var", "[C]"),
            opt_param("weight", "[C]"),
            opt_param("bias", "[C]"),
        ],
        attrs: &[EPS],
        outputs: OUT_C,
        shape: ShapeRule::BatchNorm,
        differentiable: Differentiable::Only(&["x", "weight", "bias"]),
    },
    nn(
        "nn.layer_norm",
        X_C,
        &[
            opt_param("weight", "normalized_shape"),
            opt_param("bias", "normalized_shape"),
        ],
        &[NORMALIZED_SHAPE, EPS],
        ShapeRule::LayerNorm,
    ),
    nn(
        "nn.group_norm",
        X_C_NC,
        &[opt_param("weight", "[C]"), opt_param("bias", "[C]")],
        &[req("num_groups", T::Int { min: Some(1) }), EPS],
        ShapeRule::GroupNorm,
    ),
    nn(
        "nn.rms_norm",
        X_C,
        &[opt_param("weight", "normalized_shape")],
        &[NORMALIZED_SHAPE, req("eps", T::Float(FloatRange::Positive))],
        ShapeRule::RmsNorm,
    ),
    // §3.3 Activations
    activation("nn.relu", &[]),
    activation(
        "nn.leaky_relu",
        &[def("negative_slope", T::Float(FloatRange::Any), "0.01")],
    ),
    activation("nn.gelu", &[]),
    activation("nn.gelu_tanh", &[]),
    activation("nn.silu", &[]),
    activation("nn.sigmoid", &[]),
    activation(
        "nn.softplus",
        &[
            def("beta", T::Float(FloatRange::Positive), "1.0"),
            def("threshold", T::Float(FloatRange::Positive), "20.0"),
        ],
    ),
    activation("nn.mish", &[]),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_in_supported_opsets() {
        assert_eq!(OPS.len(), 78);
        let unique: std::collections::HashSet<_> = OPS.iter().map(|o| o.name).collect();
        assert_eq!(unique.len(), OPS.len());
        for op in OPS {
            assert!(
                SUPPORTED_OPSETS.iter().any(|(s, _)| *s == op.opset()),
                "{}",
                op.name
            );
        }
    }

    #[test]
    fn same_dtype_rules_name_an_earlier_input() {
        for op in OPS {
            for (k, i) in op.inputs.iter().enumerate() {
                if let InDtype::Same(other) = i.dtype {
                    assert!(
                        op.inputs[..k].iter().any(|j| j.name == other),
                        "{}",
                        op.name
                    );
                }
            }
            if let Outputs::Fixed(outs) = op.outputs {
                for o in outs {
                    if let OutDtype::Same(n) | OutDtype::AttrOr(n) = o.dtype {
                        assert!(op.input(n).is_some(), "{} {}", op.name, o.name);
                    }
                    if matches!(o.dtype, OutDtype::Attr | OutDtype::AttrOr(_)) {
                        assert!(
                            matches!(op.attr("dtype").map(|a| a.ty), Some(AttrType::Dtype(_))),
                            "{}",
                            op.name
                        );
                    }
                }
            }
        }
    }
}
