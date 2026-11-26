//! IR op → `DynTensor` mapping (`docs/specs/runtime.md` §3; the method table is in
//! `polypous_backend::ops`).
//!
//! [`run_op`] calls the backend function of one op node with its inputs (catalog order), its
//! params (catalog order) and its attributes as the analysis normalized them (defaults filled in,
//! axes `0…r-1`), with the `dim`/`dim[]` attributes evaluated by the binding
//! (`polypous_ir::BoundOp`). Every catalog op has an arm ([`MAPPED_OPS`] lists them; a test checks
//! the list against the catalog). An attribute the analysis should have filled in but didn't is an
//! internal error, never a panic.

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, Scalar, TensorData, TensorError};
use polypous_ir::{AttrValue, Attrs, BoundDim, BoundOp, Dtype, OpInfo};

/// The value dtype of the backend for an IR dtype.
pub(crate) fn to_backend_dtype(dtype: Dtype) -> DType {
    match dtype {
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::Bf16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::I32 => DType::I32,
        Dtype::Bool => DType::Bool,
    }
}

/// The IR dtype of a backend dtype.
pub(crate) fn from_backend_dtype(dtype: DType) -> Option<Dtype> {
    Some(match dtype {
        DType::F32 => Dtype::F32,
        DType::F16 => Dtype::F16,
        DType::BF16 => Dtype::Bf16,
        DType::I64 => Dtype::I64,
        DType::I32 => Dtype::I32,
        DType::Bool => Dtype::Bool,
        _ => return None,
    })
}

/// Every op [`run_op`] maps onto the backend: the M1 catalog (graph-ir.md §3.2, §3.3).
pub const MAPPED_OPS: &[&str] = &[
    "core.add",
    "core.sub",
    "core.mul",
    "core.div",
    "core.pow",
    "core.maximum",
    "core.minimum",
    "core.neg",
    "core.abs",
    "core.exp",
    "core.log",
    "core.sqrt",
    "core.rsqrt",
    "core.sin",
    "core.cos",
    "core.tanh",
    "core.clamp",
    "core.where",
    "core.eq",
    "core.ne",
    "core.lt",
    "core.le",
    "core.gt",
    "core.ge",
    "core.and",
    "core.or",
    "core.not",
    "core.reshape",
    "core.flatten",
    "core.permute",
    "core.transpose",
    "core.squeeze",
    "core.unsqueeze",
    "core.expand",
    "core.concat",
    "core.split",
    "core.slice",
    "core.pad",
    "core.gather",
    "core.index_select",
    "core.constant",
    "core.arange",
    "core.zeros",
    "core.ones",
    "core.full",
    "core.full_like",
    "core.sum",
    "core.mean",
    "core.max",
    "core.min",
    "core.argmax",
    "core.argmin",
    "core.matmul",
    "core.cast",
    "core.softmax",
    "core.log_softmax",
    "core.cumsum",
    "core.top_k",
    "nn.linear",
    "nn.embedding",
    "nn.conv1d",
    "nn.conv2d",
    "nn.max_pool2d",
    "nn.avg_pool2d",
    "nn.adaptive_avg_pool2d",
    "nn.dropout",
    "nn.batch_norm",
    "nn.layer_norm",
    "nn.group_norm",
    "nn.rms_norm",
    "nn.relu",
    "nn.leaky_relu",
    "nn.gelu",
    "nn.gelu_tanh",
    "nn.silu",
    "nn.sigmoid",
    "nn.softplus",
    "nn.mish",
];

/// Whether [`run_op`] maps `op` (a full op name).
pub fn is_mapped(op: &str) -> bool {
    MAPPED_OPS.contains(&op)
}

/// The tensors bound to one op input.
pub(crate) enum Arg {
    /// Boxed: a `DynTensor` is a few hundred bytes (one variant per rank and kind).
    One(Box<DynTensor>),
    Many(Vec<DynTensor>),
}

/// The mode an op node runs in (runtime.md §3.1). Only `nn.batch_norm` and `nn.dropout` look
/// at it (graph-ir.md §3.1.10).
#[derive(Debug, Clone, Copy)]
pub(crate) enum OpMode<'a> {
    Inference,
    Training {
        seed: u64,
        step: u64,
        /// The node path of runtime.md §5, which keys the dropout stream (graph-ir.md §3.1.12).
        node_path: &'a str,
    },
}

/// Why [`run_op`] failed.
pub(crate) enum OpFailure {
    /// The backend returned an error.
    Tensor(TensorError),
    /// A broken invariant (an attribute or input the analysis guarantees is missing).
    Internal(String),
}

impl From<TensorError> for OpFailure {
    fn from(err: TensorError) -> Self {
        OpFailure::Tensor(err)
    }
}

type R<T> = std::result::Result<T, OpFailure>;

/// One output.
fn one(t: DynTensor) -> R<Vec<DynTensor>> {
    Ok(vec![t])
}

fn internal<T>(message: String) -> R<T> {
    Err(OpFailure::Internal(message))
}

/// Reads a node's inputs, params and attributes.
struct Node<'a> {
    info: &'a OpInfo,
    bound: &'a BoundOp,
    inputs: Vec<Option<Arg>>,
    params: Vec<Option<DynTensor>>,
}

impl Node<'_> {
    fn attrs(&self) -> &Attrs {
        &self.info.attrs
    }

    fn missing<T>(&self, what: &str, name: &str) -> R<T> {
        internal(format!(
            "`{}`: {what} `{name}` is missing or has an unexpected form after validation",
            self.info.op.name
        ))
    }

    /// Input `k` (catalog order), a single tensor.
    fn input(&mut self, k: usize) -> R<DynTensor> {
        match self.inputs.get_mut(k).and_then(Option::take) {
            Some(Arg::One(t)) => Ok(*t),
            _ => self.missing("input", self.input_name(k)),
        }
    }

    /// Input `k`, a variadic list.
    fn inputs_list(&mut self, k: usize) -> R<Vec<DynTensor>> {
        match self.inputs.get_mut(k).and_then(Option::take) {
            Some(Arg::Many(ts)) => Ok(ts),
            _ => self.missing("input", self.input_name(k)),
        }
    }

    fn input_name(&self, k: usize) -> &'static str {
        self.info.op.inputs.get(k).map_or("?", |i| i.name)
    }

    /// Param `k` (catalog order), if bound.
    fn param(&mut self, k: usize) -> Option<DynTensor> {
        self.params.get_mut(k).and_then(Option::take)
    }

    /// A required param.
    fn required_param(&mut self, k: usize) -> R<DynTensor> {
        match self.param(k) {
            Some(t) => Ok(t),
            None => {
                let name = self.info.op.params.get(k).map_or("?", |p| p.name);
                self.missing("param", name)
            }
        }
    }

    fn axis(&self, name: &str) -> R<usize> {
        match self.attrs().get(name) {
            Some(AttrValue::Axis(a)) => Ok(*a),
            _ => self.missing("attribute", name),
        }
    }

    fn axes(&self, name: &str) -> R<Vec<usize>> {
        match self.attrs().get(name) {
            Some(AttrValue::Axes(a)) => Ok(a.clone()),
            _ => self.missing("attribute", name),
        }
    }

    /// A non-negative `int` attribute.
    fn uint(&self, name: &str) -> R<usize> {
        match self.attrs().get(name) {
            Some(AttrValue::Int(v)) => match usize::try_from(*v) {
                Ok(v) => Ok(v),
                Err(_) => self.missing("attribute", name),
            },
            _ => self.missing("attribute", name),
        }
    }

    /// A non-negative `int[]` attribute.
    fn uints(&self, name: &str) -> R<Vec<usize>> {
        match self.attrs().get(name) {
            Some(AttrValue::Ints(v)) => match v.iter().map(|&x| usize::try_from(x)).collect() {
                Ok(v) => Ok(v),
                Err(_) => self.missing("attribute", name),
            },
            _ => self.missing("attribute", name),
        }
    }

    /// An `int[2]` attribute.
    fn pair(&self, name: &str) -> R<[usize; 2]> {
        match <[usize; 2]>::try_from(self.uints(name)?) {
            Ok(p) => Ok(p),
            Err(_) => self.missing("attribute", name),
        }
    }

    fn float(&self, name: &str) -> R<f64> {
        match self.attrs().get(name) {
            Some(AttrValue::Float(v)) => Ok(*v),
            _ => self.missing("attribute", name),
        }
    }

    fn flag(&self, name: &str) -> R<bool> {
        match self.attrs().get(name) {
            Some(AttrValue::Bool(v)) => Ok(*v),
            _ => self.missing("attribute", name),
        }
    }

    fn dtype(&self, name: &str) -> R<DType> {
        match self.attrs().get(name) {
            Some(AttrValue::Dtype(d)) => Ok(to_backend_dtype(*d)),
            _ => self.missing("attribute", name),
        }
    }

    /// An optional `scalar` attribute.
    fn scalar_opt(&self, name: &str) -> R<Option<Scalar>> {
        match self.attrs().get(name) {
            None => Ok(None),
            Some(AttrValue::Scalar(s)) => Ok(Some(scalar(*s))),
            Some(_) => self.missing("attribute", name),
        }
    }

    fn scalar(&self, name: &str) -> R<Scalar> {
        match self.scalar_opt(name)? {
            Some(s) => Ok(s),
            None => self.missing("attribute", name),
        }
    }

    fn scalars(&self, name: &str) -> R<Vec<Scalar>> {
        match self.attrs().get(name) {
            Some(AttrValue::Scalars(v)) => Ok(v.iter().map(|s| scalar(*s)).collect()),
            _ => self.missing("attribute", name),
        }
    }

    /// An evaluated, non-negative `dim` attribute.
    fn dim(&self, name: &str) -> R<usize> {
        match self.bound.dim(name) {
            Some(BoundDim::One(v)) => match usize::try_from(*v) {
                Ok(v) => Ok(v),
                Err(_) => self.missing("dim attribute", name),
            },
            _ => self.missing("dim attribute", name),
        }
    }

    /// An evaluated `dim[]` attribute whose entries are all non-negative.
    fn dims(&self, name: &str) -> R<Vec<usize>> {
        match self.bound.dim(name) {
            Some(BoundDim::Many(v)) => match v.iter().map(|&x| usize::try_from(x)).collect() {
                Ok(v) => Ok(v),
                Err(_) => self.missing("dim attribute", name),
            },
            _ => self.missing("dim attribute", name),
        }
    }
}

fn scalar(s: polypous_ir::Scalar) -> Scalar {
    match s {
        polypous_ir::Scalar::Int(v) => Scalar::Int(v),
        polypous_ir::Scalar::Float(v) => Scalar::Float(v),
        polypous_ir::Scalar::Bool(v) => Scalar::Bool(v),
    }
}

/// Runs one op node. `inputs` and `params` are in catalog order (`None`: not bound). Returns the
/// outputs in port order.
///
/// In training mode `nn.batch_norm` puts its new `(running_mean, running_var)` into `running`
/// (runtime.md §3.3), and `nn.dropout` builds its keep mask on the host
/// ([`polypous_ir::rng::dropout_mask`]) and uploads it as a constant `bool` tensor.
///
/// Each arm calls its own `#[inline(never)]` function. A debug build gives every arm's
/// temporaries their own stack slots, and a `DynTensor` result is several hundred bytes, so with
/// the arms inline this one frame was about 600 KB with `vulkan` and overflowed the 1 MB main
/// thread on Windows (#114). Split, only the arm that runs is on the stack.
pub(crate) fn run_op(
    info: &OpInfo,
    bound: &BoundOp,
    inputs: Vec<Option<Arg>>,
    params: Vec<Option<DynTensor>>,
    device: DeviceId,
    mode: OpMode<'_>,
    running: &mut Option<(DynTensor, DynTensor)>,
) -> R<Vec<DynTensor>> {
    let mut n = Node {
        info,
        bound,
        inputs,
        params,
    };
    let name = info.op.name;
    match name {
        // Elementwise (binary: `a`, `b`).
        "core.add" | "core.sub" | "core.mul" | "core.div" | "core.pow" | "core.maximum"
        | "core.minimum" | "core.eq" | "core.ne" | "core.lt" | "core.le" | "core.gt"
        | "core.ge" | "core.and" | "core.or" => elementwise_binary(&mut n, name),
        // Elementwise (unary: `x`).
        "core.neg" | "core.abs" | "core.exp" | "core.log" | "core.sqrt" | "core.rsqrt"
        | "core.sin" | "core.cos" | "core.tanh" | "core.not" | "nn.relu" | "nn.gelu"
        | "nn.gelu_tanh" | "nn.silu" | "nn.sigmoid" | "nn.mish" => elementwise_unary(&mut n, name),
        "core.clamp" => core_clamp(&mut n),
        "core.where" => core_where(&mut n),
        "core.cast" => core_cast(&mut n),

        // Shape.
        "core.reshape" => core_reshape(&mut n),
        "core.flatten" => core_flatten(&mut n),
        "core.permute" => core_permute(&mut n),
        "core.transpose" => core_transpose(&mut n),
        "core.squeeze" => core_squeeze(&mut n),
        "core.unsqueeze" => core_unsqueeze(&mut n),
        "core.expand" => core_expand(&mut n),
        "core.concat" => core_concat(&mut n),
        "core.split" => core_split(&mut n),
        "core.slice" => core_slice(&mut n),
        "core.pad" => core_pad(&mut n),

        // Indexing.
        "core.gather" | "core.index_select" => gather_or_index_select(&mut n, name),

        // Creation.
        "core.constant" => core_constant(&mut n, device),
        "core.arange" => core_arange(&mut n, device),
        "core.zeros" | "core.ones" => zeros_or_ones(&mut n, name, device),
        "core.full" => core_full(&mut n, device),
        "core.full_like" => core_full_like(&mut n),

        // Reduction.
        "core.sum" | "core.mean" | "core.max" | "core.min" => reduce(&mut n, name),
        "core.argmax" | "core.argmin" => arg_reduce(&mut n, name),

        // Linear algebra and other.
        "core.matmul" => core_matmul(&mut n),
        "core.softmax" | "core.log_softmax" | "core.cumsum" => softmax_or_cumsum(&mut n, name),
        "core.top_k" => core_top_k(&mut n),

        // nn layers. Params in catalog order (graph-ir.md §3.3).
        "nn.linear" => nn_linear(&mut n),
        "nn.embedding" => nn_embedding(&mut n),
        "nn.conv1d" => nn_conv1d(&mut n),
        "nn.conv2d" => nn_conv2d(&mut n),
        "nn.max_pool2d" => nn_max_pool2d(&mut n),
        "nn.avg_pool2d" => nn_avg_pool2d(&mut n),
        "nn.adaptive_avg_pool2d" => nn_adaptive_avg_pool2d(&mut n),
        "nn.dropout" => nn_dropout(&mut n, device, mode),

        // Normalization.
        "nn.batch_norm" => nn_batch_norm(&mut n, mode, running),
        "nn.layer_norm" => nn_layer_norm(&mut n),
        "nn.group_norm" => nn_group_norm(&mut n),
        "nn.rms_norm" => nn_rms_norm(&mut n),

        // Activations with attributes.
        "nn.leaky_relu" => nn_leaky_relu(&mut n),
        "nn.softplus" => nn_softplus(&mut n),

        other => internal(format!("op `{other}` has no backend mapping")),
    }
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn elementwise_binary(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let (a, b) = (n.input(0)?, n.input(1)?);
    // One call site for all of them: a debug build gives every call its own stack slots (#114).
    let op: fn(DynTensor, DynTensor) -> Result<DynTensor, TensorError> = match name {
        "core.add" => DynTensor::add,
        "core.sub" => DynTensor::sub,
        "core.mul" => DynTensor::mul,
        "core.div" => DynTensor::div,
        "core.pow" => DynTensor::pow,
        "core.maximum" => DynTensor::maximum,
        "core.minimum" => DynTensor::minimum,
        "core.eq" => DynTensor::equal,
        "core.ne" => DynTensor::not_equal,
        "core.lt" => DynTensor::less,
        "core.le" => DynTensor::less_equal,
        "core.gt" => DynTensor::greater,
        "core.ge" => DynTensor::greater_equal,
        "core.and" => DynTensor::logical_and,
        _ => DynTensor::logical_or,
    };
    one(op(a, b)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn elementwise_unary(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let x = n.input(0)?;
    // One call site for all of them: a debug build gives every call its own stack slots (#114).
    let op: fn(DynTensor) -> Result<DynTensor, TensorError> = match name {
        "core.neg" => DynTensor::neg,
        "core.abs" => DynTensor::abs,
        "core.exp" => DynTensor::exp,
        "core.log" => DynTensor::log,
        "core.sqrt" => DynTensor::sqrt,
        "core.rsqrt" => DynTensor::rsqrt,
        "core.sin" => DynTensor::sin,
        "core.cos" => DynTensor::cos,
        "core.tanh" => DynTensor::tanh,
        "core.not" => DynTensor::logical_not,
        "nn.relu" => DynTensor::relu,
        "nn.gelu" => DynTensor::gelu,
        "nn.gelu_tanh" => DynTensor::gelu_tanh,
        "nn.silu" => DynTensor::silu,
        "nn.sigmoid" => DynTensor::sigmoid,
        _ => DynTensor::mish,
    };
    one(op(x)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_clamp(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (min, max) = (n.scalar_opt("min")?, n.scalar_opt("max")?);
    one(n.input(0)?.clamp(min, max)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_where(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (cond, a, b) = (n.input(0)?, n.input(1)?, n.input(2)?);
    one(DynTensor::where_(cond, a, b)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_cast(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let dtype = n.dtype("dtype")?;
    one(n.input(0)?.cast(dtype)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_reshape(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let shape = n.dims("shape")?;
    one(n.input(0)?.reshape(&shape)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_flatten(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (start, end) = (n.axis("start_dim")?, n.axis("end_dim")?);
    one(n.input(0)?.flatten(start, end)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_permute(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let dims = n.axes("dims")?;
    one(n.input(0)?.permute(&dims)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_transpose(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (d0, d1) = (n.axis("dim0")?, n.axis("dim1")?);
    one(n.input(0)?.transpose(d0, d1)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_squeeze(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let dims = n.axes("dim")?;
    one(n.input(0)?.squeeze(&dims)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_unsqueeze(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let dim = n.axis("dim")?;
    one(n.input(0)?.unsqueeze(dim)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_expand(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let shape = n.dims("shape")?;
    one(n.input(0)?.expand(&shape)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_concat(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let dim = n.axis("dim")?;
    one(DynTensor::concat(&n.inputs_list(0)?, dim)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_split(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (dim, sizes) = (n.axis("dim")?, n.dims("sizes")?);
    Ok(n.input(0)?.split(dim, &sizes)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_slice(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (dim, start, end, step) = (
        n.axis("dim")?,
        n.dim("start")?,
        n.dim("end")?,
        n.uint("step")?,
    );
    one(n.input(0)?.slice(dim, start, end, step)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_pad(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (pad, value) = (n.dims("pad")?, n.scalar("value")?);
    one(n.input(0)?.pad(&pad, value)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn gather_or_index_select(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let dim = n.axis("dim")?;
    let (x, index) = (n.input(0)?, n.input(1)?);
    one(if name == "core.gather" {
        x.gather(dim, index)?
    } else {
        x.index_select(dim, index)?
    })
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_constant(n: &mut Node<'_>, device: DeviceId) -> R<Vec<DynTensor>> {
    let (dtype, shape, values) = (n.dtype("dtype")?, n.uints("shape")?, n.scalars("values")?);
    one(DynTensor::constant(dtype, &shape, &values, device)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_arange(n: &mut Node<'_>, device: DeviceId) -> R<Vec<DynTensor>> {
    let (start, end, step, dtype) = (
        n.dim("start")?,
        n.dim("end")?,
        n.uint("step")?,
        n.dtype("dtype")?,
    );
    one(DynTensor::arange(start, end, step, dtype, device)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn zeros_or_ones(n: &mut Node<'_>, name: &str, device: DeviceId) -> R<Vec<DynTensor>> {
    let (shape, dtype) = (n.dims("shape")?, n.dtype("dtype")?);
    one(if name == "core.zeros" {
        DynTensor::zeros(&shape, dtype, device)?
    } else {
        DynTensor::ones(&shape, dtype, device)?
    })
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_full(n: &mut Node<'_>, device: DeviceId) -> R<Vec<DynTensor>> {
    let (shape, value, dtype) = (n.dims("shape")?, n.scalar("value")?, n.dtype("dtype")?);
    one(DynTensor::full(&shape, value, dtype, device)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_full_like(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (value, dtype) = (n.scalar("value")?, n.dtype("dtype")?);
    one(n.input(0)?.full_like(value, Some(dtype))?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn reduce(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let (dims, keepdim) = (n.axes("dim")?, n.flag("keepdim")?);
    let x = n.input(0)?;
    one(match name {
        "core.sum" => x.sum(Some(&dims), keepdim)?,
        "core.mean" => x.mean(Some(&dims), keepdim)?,
        "core.max" => x.max(Some(&dims), keepdim)?,
        _ => x.min(Some(&dims), keepdim)?,
    })
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn arg_reduce(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let (dim, keepdim, dtype) = (n.axis("dim")?, n.flag("keepdim")?, n.dtype("dtype")?);
    let x = n.input(0)?;
    one(if name == "core.argmax" {
        x.argmax(dim, keepdim, dtype)?
    } else {
        x.argmin(dim, keepdim, dtype)?
    })
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_matmul(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (a, b) = (n.input(0)?, n.input(1)?);
    one(a.matmul(b)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn softmax_or_cumsum(n: &mut Node<'_>, name: &str) -> R<Vec<DynTensor>> {
    let dim = n.axis("dim")?;
    let x = n.input(0)?;
    one(match name {
        "core.softmax" => x.softmax(dim)?,
        "core.log_softmax" => x.log_softmax(dim)?,
        _ => x.cumsum(dim)?,
    })
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn core_top_k(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (k, dim, largest, dtype) = (
        n.dim("k")?,
        n.axis("dim")?,
        n.flag("largest")?,
        n.dtype("dtype")?,
    );
    let (values, indices) = n.input(0)?.top_k(k, dim, largest, dtype)?;
    Ok(vec![values, indices])
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_linear(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (weight, bias) = (n.required_param(0)?, n.param(1));
    one(n.input(0)?.linear(weight, bias)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_embedding(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let weight = n.required_param(0)?;
    one(n.input(0)?.embedding(weight)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_conv1d(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (weight, bias) = (n.required_param(0)?, n.param(1));
    let (stride, padding, dilation, groups) = (
        n.uint("stride")?,
        n.uint("padding")?,
        n.uint("dilation")?,
        n.uint("groups")?,
    );
    one(n
        .input(0)?
        .conv1d(weight, bias, stride, padding, dilation, groups)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_conv2d(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (weight, bias) = (n.required_param(0)?, n.param(1));
    let (stride, padding, dilation, groups) = (
        n.pair("stride")?,
        n.pair("padding")?,
        n.pair("dilation")?,
        n.uint("groups")?,
    );
    one(n
        .input(0)?
        .conv2d(weight, bias, stride, padding, dilation, groups)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_max_pool2d(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (kernel, stride, padding, dilation) = (
        n.pair("kernel_size")?,
        n.pair("stride")?,
        n.pair("padding")?,
        n.pair("dilation")?,
    );
    one(n.input(0)?.max_pool2d(kernel, stride, padding, dilation)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_avg_pool2d(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (kernel, stride, padding, include) = (
        n.pair("kernel_size")?,
        n.pair("stride")?,
        n.pair("padding")?,
        n.flag("count_include_pad")?,
    );
    one(n.input(0)?.avg_pool2d(kernel, stride, padding, include)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_adaptive_avg_pool2d(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let size = n.pair("output_size")?;
    one(n.input(0)?.adaptive_avg_pool2d(size)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_dropout(n: &mut Node<'_>, device: DeviceId, mode: OpMode<'_>) -> R<Vec<DynTensor>> {
    let p = n.float("p")?;
    let x = n.input(0)?;
    match mode {
        // `p = 0` is the identity in both modes and draws nothing (graph-ir.md §3.3).
        OpMode::Training {
            seed,
            step,
            node_path,
        } if p > 0.0 => {
            let keep = dropout_keep(seed, step, node_path, &x.shape(), p, device)?;
            one(x.dropout_training(p, keep)?)
        }
        _ => one(x.dropout(p)?),
    }
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_batch_norm(
    n: &mut Node<'_>,
    mode: OpMode<'_>,
    running: &mut Option<(DynTensor, DynTensor)>,
) -> R<Vec<DynTensor>> {
    let (mean, var) = (n.required_param(0)?, n.required_param(1)?);
    let (weight, bias, eps) = (n.param(2), n.param(3), n.float("eps")?);
    match mode {
        OpMode::Training { .. } => {
            let momentum = n.float("momentum")?;
            let (out, new_mean, new_var) = n
                .input(0)?
                .batch_norm_training(mean, var, weight, bias, eps, momentum)?;
            *running = Some((new_mean, new_var));
            one(out)
        }
        OpMode::Inference => one(n.input(0)?.batch_norm(mean, var, weight, bias, eps)?),
    }
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_layer_norm(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (weight, bias) = (n.param(0), n.param(1));
    let (shape, eps) = (n.uints("normalized_shape")?, n.float("eps")?);
    one(n.input(0)?.layer_norm(&shape, weight, bias, eps)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_group_norm(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (weight, bias) = (n.param(0), n.param(1));
    let (groups, eps) = (n.uint("num_groups")?, n.float("eps")?);
    one(n.input(0)?.group_norm(groups, weight, bias, eps)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_rms_norm(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let weight = n.param(0);
    let (shape, eps) = (n.uints("normalized_shape")?, n.float("eps")?);
    one(n.input(0)?.rms_norm(&shape, weight, eps)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_leaky_relu(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let slope = n.float("negative_slope")?;
    one(n.input(0)?.leaky_relu(slope)?)
}

// #114: own frame, see `run_op`.
#[inline(never)]
fn nn_softplus(n: &mut Node<'_>) -> R<Vec<DynTensor>> {
    let (beta, threshold) = (n.float("beta")?, n.float("threshold")?);
    one(n.input(0)?.softplus(beta, threshold)?)
}

/// The keep mask of a training-mode `nn.dropout` (graph-ir.md §3.3, §3.1.12), built on the host
/// and uploaded to `device` as a `bool` tensor of `shape`. It is a constant: uploaded data carries
/// no autodiff tracking. `shape` is the shape of a tensor that exists, so its element count fits;
/// a mask that can't be allocated is reported as an internal error, never a panic.
fn dropout_keep(
    seed: u64,
    step: u64,
    node_path: &str,
    shape: &[usize],
    p: f64,
    device: DeviceId,
) -> R<DynTensor> {
    let mask = polypous_ir::rng::dropout_mask(seed, step, node_path, shape, p)
        .map_err(|e| OpFailure::Internal(format!("dropout mask: {e}")))?;
    // `bool` and `u8` have the same layout, so this collects in place, without a new allocation.
    let bytes: Vec<u8> = mask.into_iter().map(u8::from).collect();
    let dims = shape.iter().map(|&d| d as u64).collect();
    let data = TensorData::new(DType::Bool, dims, bytes)?;
    Ok(DynTensor::from_data(&data, device)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalog_op_is_mapped_and_nothing_else() {
        let catalog: Vec<&str> = polypous_ir::catalog::OPS.iter().map(|o| o.name).collect();
        for op in &catalog {
            assert!(is_mapped(op), "catalog op `{op}` has no backend mapping");
        }
        for op in MAPPED_OPS {
            assert!(
                catalog.contains(op),
                "`{op}` is mapped but not in the catalog"
            );
        }
        assert_eq!(MAPPED_OPS.len(), catalog.len());
    }

    #[test]
    fn dtypes_round_trip() {
        for d in [
            Dtype::F32,
            Dtype::F16,
            Dtype::Bf16,
            Dtype::I64,
            Dtype::I32,
            Dtype::Bool,
        ] {
            assert_eq!(from_backend_dtype(to_backend_dtype(d)), Some(d));
            assert_eq!(to_backend_dtype(d).name(), d.as_str());
        }
    }
}
