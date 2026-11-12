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
use polypous_backend::{DType, DynTensor, Scalar, TensorError};
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
pub(crate) fn run_op(
    info: &OpInfo,
    bound: &BoundOp,
    inputs: Vec<Option<Arg>>,
    params: Vec<Option<DynTensor>>,
    device: DeviceId,
) -> R<Vec<DynTensor>> {
    let mut n = Node {
        info,
        bound,
        inputs,
        params,
    };
    let one = |t: DynTensor| Ok(vec![t]);
    let name = info.op.name;
    match name {
        // Elementwise (binary: `a`, `b`).
        "core.add" | "core.sub" | "core.mul" | "core.div" | "core.pow" | "core.maximum"
        | "core.minimum" | "core.eq" | "core.ne" | "core.lt" | "core.le" | "core.gt"
        | "core.ge" | "core.and" | "core.or" => {
            let (a, b) = (n.input(0)?, n.input(1)?);
            one(match name {
                "core.add" => a.add(b)?,
                "core.sub" => a.sub(b)?,
                "core.mul" => a.mul(b)?,
                "core.div" => a.div(b)?,
                "core.pow" => a.pow(b)?,
                "core.maximum" => a.maximum(b)?,
                "core.minimum" => a.minimum(b)?,
                "core.eq" => a.equal(b)?,
                "core.ne" => a.not_equal(b)?,
                "core.lt" => a.less(b)?,
                "core.le" => a.less_equal(b)?,
                "core.gt" => a.greater(b)?,
                "core.ge" => a.greater_equal(b)?,
                "core.and" => a.logical_and(b)?,
                _ => a.logical_or(b)?,
            })
        }
        // Elementwise (unary: `x`).
        "core.neg" | "core.abs" | "core.exp" | "core.log" | "core.sqrt" | "core.rsqrt"
        | "core.sin" | "core.cos" | "core.tanh" | "core.not" | "nn.relu" | "nn.gelu"
        | "nn.gelu_tanh" | "nn.silu" | "nn.sigmoid" | "nn.mish" => {
            let x = n.input(0)?;
            one(match name {
                "core.neg" => x.neg()?,
                "core.abs" => x.abs()?,
                "core.exp" => x.exp()?,
                "core.log" => x.log()?,
                "core.sqrt" => x.sqrt()?,
                "core.rsqrt" => x.rsqrt()?,
                "core.sin" => x.sin()?,
                "core.cos" => x.cos()?,
                "core.tanh" => x.tanh()?,
                "core.not" => x.logical_not()?,
                "nn.relu" => x.relu()?,
                "nn.gelu" => x.gelu()?,
                "nn.gelu_tanh" => x.gelu_tanh()?,
                "nn.silu" => x.silu()?,
                "nn.sigmoid" => x.sigmoid()?,
                _ => x.mish()?,
            })
        }
        "core.clamp" => {
            let (min, max) = (n.scalar_opt("min")?, n.scalar_opt("max")?);
            one(n.input(0)?.clamp(min, max)?)
        }
        "core.where" => {
            let (cond, a, b) = (n.input(0)?, n.input(1)?, n.input(2)?);
            one(DynTensor::where_(cond, a, b)?)
        }
        "core.cast" => {
            let dtype = n.dtype("dtype")?;
            one(n.input(0)?.cast(dtype)?)
        }

        // Shape.
        "core.reshape" => {
            let shape = n.dims("shape")?;
            one(n.input(0)?.reshape(&shape)?)
        }
        "core.flatten" => {
            let (start, end) = (n.axis("start_dim")?, n.axis("end_dim")?);
            one(n.input(0)?.flatten(start, end)?)
        }
        "core.permute" => {
            let dims = n.axes("dims")?;
            one(n.input(0)?.permute(&dims)?)
        }
        "core.transpose" => {
            let (d0, d1) = (n.axis("dim0")?, n.axis("dim1")?);
            one(n.input(0)?.transpose(d0, d1)?)
        }
        "core.squeeze" => {
            let dims = n.axes("dim")?;
            one(n.input(0)?.squeeze(&dims)?)
        }
        "core.unsqueeze" => {
            let dim = n.axis("dim")?;
            one(n.input(0)?.unsqueeze(dim)?)
        }
        "core.expand" => {
            let shape = n.dims("shape")?;
            one(n.input(0)?.expand(&shape)?)
        }
        "core.concat" => {
            let dim = n.axis("dim")?;
            one(DynTensor::concat(&n.inputs_list(0)?, dim)?)
        }
        "core.split" => {
            let (dim, sizes) = (n.axis("dim")?, n.dims("sizes")?);
            Ok(n.input(0)?.split(dim, &sizes)?)
        }
        "core.slice" => {
            let (dim, start, end, step) = (
                n.axis("dim")?,
                n.dim("start")?,
                n.dim("end")?,
                n.uint("step")?,
            );
            one(n.input(0)?.slice(dim, start, end, step)?)
        }
        "core.pad" => {
            let (pad, value) = (n.dims("pad")?, n.scalar("value")?);
            one(n.input(0)?.pad(&pad, value)?)
        }

        // Indexing.
        "core.gather" | "core.index_select" => {
            let dim = n.axis("dim")?;
            let (x, index) = (n.input(0)?, n.input(1)?);
            one(if name == "core.gather" {
                x.gather(dim, index)?
            } else {
                x.index_select(dim, index)?
            })
        }

        // Creation.
        "core.constant" => {
            let (dtype, shape, values) =
                (n.dtype("dtype")?, n.uints("shape")?, n.scalars("values")?);
            one(DynTensor::constant(dtype, &shape, &values, device)?)
        }
        "core.arange" => {
            let (start, end, step, dtype) = (
                n.dim("start")?,
                n.dim("end")?,
                n.uint("step")?,
                n.dtype("dtype")?,
            );
            one(DynTensor::arange(start, end, step, dtype, device)?)
        }
        "core.zeros" | "core.ones" => {
            let (shape, dtype) = (n.dims("shape")?, n.dtype("dtype")?);
            one(if name == "core.zeros" {
                DynTensor::zeros(&shape, dtype, device)?
            } else {
                DynTensor::ones(&shape, dtype, device)?
            })
        }
        "core.full" => {
            let (shape, value, dtype) = (n.dims("shape")?, n.scalar("value")?, n.dtype("dtype")?);
            one(DynTensor::full(&shape, value, dtype, device)?)
        }
        "core.full_like" => {
            let (value, dtype) = (n.scalar("value")?, n.dtype("dtype")?);
            one(n.input(0)?.full_like(value, Some(dtype))?)
        }

        // Reduction.
        "core.sum" | "core.mean" | "core.max" | "core.min" => {
            let (dims, keepdim) = (n.axes("dim")?, n.flag("keepdim")?);
            let x = n.input(0)?;
            one(match name {
                "core.sum" => x.sum(Some(&dims), keepdim)?,
                "core.mean" => x.mean(Some(&dims), keepdim)?,
                "core.max" => x.max(Some(&dims), keepdim)?,
                _ => x.min(Some(&dims), keepdim)?,
            })
        }
        "core.argmax" | "core.argmin" => {
            let (dim, keepdim, dtype) = (n.axis("dim")?, n.flag("keepdim")?, n.dtype("dtype")?);
            let x = n.input(0)?;
            one(if name == "core.argmax" {
                x.argmax(dim, keepdim, dtype)?
            } else {
                x.argmin(dim, keepdim, dtype)?
            })
        }

        // Linear algebra and other.
        "core.matmul" => {
            let (a, b) = (n.input(0)?, n.input(1)?);
            one(a.matmul(b)?)
        }
        "core.softmax" | "core.log_softmax" | "core.cumsum" => {
            let dim = n.axis("dim")?;
            let x = n.input(0)?;
            one(match name {
                "core.softmax" => x.softmax(dim)?,
                "core.log_softmax" => x.log_softmax(dim)?,
                _ => x.cumsum(dim)?,
            })
        }
        "core.top_k" => {
            let (k, dim, largest, dtype) = (
                n.dim("k")?,
                n.axis("dim")?,
                n.flag("largest")?,
                n.dtype("dtype")?,
            );
            let (values, indices) = n.input(0)?.top_k(k, dim, largest, dtype)?;
            Ok(vec![values, indices])
        }

        // nn layers. Params in catalog order (graph-ir.md §3.3).
        "nn.linear" => {
            let (weight, bias) = (n.required_param(0)?, n.param(1));
            one(n.input(0)?.linear(weight, bias)?)
        }
        "nn.embedding" => {
            let weight = n.required_param(0)?;
            one(n.input(0)?.embedding(weight)?)
        }
        "nn.conv1d" => {
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
        "nn.conv2d" => {
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
        "nn.max_pool2d" => {
            let (kernel, stride, padding, dilation) = (
                n.pair("kernel_size")?,
                n.pair("stride")?,
                n.pair("padding")?,
                n.pair("dilation")?,
            );
            one(n.input(0)?.max_pool2d(kernel, stride, padding, dilation)?)
        }
        "nn.avg_pool2d" => {
            let (kernel, stride, padding, include) = (
                n.pair("kernel_size")?,
                n.pair("stride")?,
                n.pair("padding")?,
                n.flag("count_include_pad")?,
            );
            one(n.input(0)?.avg_pool2d(kernel, stride, padding, include)?)
        }
        "nn.adaptive_avg_pool2d" => {
            let size = n.pair("output_size")?;
            one(n.input(0)?.adaptive_avg_pool2d(size)?)
        }
        "nn.dropout" => {
            let p = n.float("p")?;
            one(n.input(0)?.dropout(p)?)
        }

        // Normalization.
        "nn.batch_norm" => {
            let (mean, var) = (n.required_param(0)?, n.required_param(1)?);
            let (weight, bias, eps) = (n.param(2), n.param(3), n.float("eps")?);
            one(n.input(0)?.batch_norm(mean, var, weight, bias, eps)?)
        }
        "nn.layer_norm" => {
            let (weight, bias) = (n.param(0), n.param(1));
            let (shape, eps) = (n.uints("normalized_shape")?, n.float("eps")?);
            one(n.input(0)?.layer_norm(&shape, weight, bias, eps)?)
        }
        "nn.group_norm" => {
            let (weight, bias) = (n.param(0), n.param(1));
            let (groups, eps) = (n.uint("num_groups")?, n.float("eps")?);
            one(n.input(0)?.group_norm(groups, weight, bias, eps)?)
        }
        "nn.rms_norm" => {
            let weight = n.param(0);
            let (shape, eps) = (n.uints("normalized_shape")?, n.float("eps")?);
            one(n.input(0)?.rms_norm(&shape, weight, eps)?)
        }

        // Activations with attributes.
        "nn.leaky_relu" => {
            let slope = n.float("negative_slope")?;
            one(n.input(0)?.leaky_relu(slope)?)
        }
        "nn.softplus" => {
            let (beta, threshold) = (n.float("beta")?, n.float("threshold")?);
            one(n.input(0)?.softplus(beta, threshold)?)
        }

        other => internal(format!("op `{other}` has no backend mapping")),
    }
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
