//! Elementwise ops: graph-ir.md §3.2, "Elementwise". General rules: §3.1 (see the module
//! documentation of [`crate::ops`]).

use burn::tensor::{DType as BurnDType, FloatDType, Int, Tensor};

use super::{
    DTypeClass, Scalar, backend_error, broadcast_shapes, check_same_device, prepare_binary,
};
use crate::device::DeviceId;
use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank, zip_rank};

/// Binary elementwise ops, which broadcast.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            BinaryOp::Add => "add",
            BinaryOp::Sub => "sub",
            BinaryOp::Mul => "mul",
            BinaryOp::Div => "div",
            BinaryOp::Pow => "pow",
            BinaryOp::Maximum => "maximum",
            BinaryOp::Minimum => "minimum",
        }
    }

    fn class(self) -> DTypeClass {
        match self {
            BinaryOp::Div | BinaryOp::Pow => DTypeClass::Float,
            _ => DTypeClass::Numeric,
        }
    }

    /// The wrapping host implementation for `i64` on the CPU (see the module documentation of
    /// [`crate::ops`], "Integer overflow wraps"), for the ops that need one.
    fn host_i64(self) -> Option<fn(i64, i64) -> i64> {
        match self {
            BinaryOp::Add => Some(i64::wrapping_add),
            BinaryOp::Sub => Some(i64::wrapping_sub),
            BinaryOp::Mul => Some(i64::wrapping_mul),
            _ => None,
        }
    }
}

/// Unary elementwise ops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnaryOp {
    Neg,
    Abs,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tanh,
}

impl UnaryOp {
    fn name(self) -> &'static str {
        match self {
            UnaryOp::Neg => "neg",
            UnaryOp::Abs => "abs",
            UnaryOp::Exp => "exp",
            UnaryOp::Log => "log",
            UnaryOp::Sqrt => "sqrt",
            UnaryOp::Rsqrt => "rsqrt",
            UnaryOp::Sin => "sin",
            UnaryOp::Cos => "cos",
            UnaryOp::Tanh => "tanh",
        }
    }

    fn class(self) -> DTypeClass {
        match self {
            UnaryOp::Neg | UnaryOp::Abs => DTypeClass::Numeric,
            _ => DTypeClass::Float,
        }
    }
}

/// Three-operand version of `zip_rank!` for `where`. The caller broadcast all three to the same
/// shape; differing ranks are a `ShapeMismatch` error, not a panic.
macro_rules! zip3_rank {
    ($op:expr, $c:expr, $a:expr, $b:expr, $x:ident, $y:ident, $z:ident => $body:expr) => {
        match ($c, $a, $b) {
            (
                $crate::tensor::Inner::R1($x),
                $crate::tensor::Inner::R1($y),
                $crate::tensor::Inner::R1($z),
            ) => $crate::tensor::Inner::R1($body),
            (
                $crate::tensor::Inner::R2($x),
                $crate::tensor::Inner::R2($y),
                $crate::tensor::Inner::R2($z),
            ) => $crate::tensor::Inner::R2($body),
            (
                $crate::tensor::Inner::R3($x),
                $crate::tensor::Inner::R3($y),
                $crate::tensor::Inner::R3($z),
            ) => $crate::tensor::Inner::R3($body),
            (
                $crate::tensor::Inner::R4($x),
                $crate::tensor::Inner::R4($y),
                $crate::tensor::Inner::R4($z),
            ) => $crate::tensor::Inner::R4($body),
            (
                $crate::tensor::Inner::R5($x),
                $crate::tensor::Inner::R5($y),
                $crate::tensor::Inner::R5($z),
            ) => $crate::tensor::Inner::R5($body),
            (
                $crate::tensor::Inner::R6($x),
                $crate::tensor::Inner::R6($y),
                $crate::tensor::Inner::R6($z),
            ) => $crate::tensor::Inner::R6($body),
            (_, a, b) => {
                return Err($crate::tensor::TensorError::ShapeMismatch {
                    op: $op,
                    lhs: a.dims(),
                    rhs: b.dims(),
                });
            }
        }
    };
}

impl DynTensor {
    /// `core.add`: `self + other`, broadcasting. Float or int; integers wrap on overflow.
    ///
    /// Not `std::ops::Add`, because it can fail.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Add` does not fit"
    )]
    pub fn add(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Add)
    }

    /// `core.sub`: `self - other`, broadcasting. Float or int; integers wrap on overflow.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Sub` does not fit"
    )]
    pub fn sub(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Sub)
    }

    /// `core.mul`: `self * other`, broadcasting. Float or int; integers wrap on overflow.
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Mul` does not fit"
    )]
    pub fn mul(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Mul)
    }

    /// `core.div`: `self / other`, broadcasting. Float only (graph-ir.md §3.2: integer division
    /// has no single meaning; cast first).
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Div` does not fit"
    )]
    pub fn div(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Div)
    }

    /// `core.pow`: `self` raised to `other`, elementwise, broadcasting. Float only. A negative
    /// base with an integer exponent gives the real result, as `torch.pow` does.
    pub fn pow(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Pow)
    }

    /// `core.maximum`: the larger of each pair, broadcasting. Float or int. NaN propagates: if
    /// either value is NaN, the result is NaN.
    pub fn maximum(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Maximum)
    }

    /// `core.minimum`: the smaller of each pair, broadcasting. Float or int. NaN propagates.
    pub fn minimum(self, other: Self) -> Result<Self> {
        self.binary(other, BinaryOp::Minimum)
    }

    /// `core.neg`: `-self`. Float or int; the integer minimum stays unchanged (wraps).
    #[expect(
        clippy::should_implement_trait,
        reason = "fallible, so `std::ops::Neg` does not fit"
    )]
    pub fn neg(self) -> Result<Self> {
        self.unary(UnaryOp::Neg)
    }

    /// `core.abs`: `|self|`. Float or int; the integer minimum stays unchanged (wraps).
    pub fn abs(self) -> Result<Self> {
        self.unary(UnaryOp::Abs)
    }

    /// `core.exp`: `e^self`. Float only.
    pub fn exp(self) -> Result<Self> {
        self.unary(UnaryOp::Exp)
    }

    /// `core.log`: natural logarithm. Float only; `log(0) = -inf`, `log(x < 0) = NaN`.
    pub fn log(self) -> Result<Self> {
        self.unary(UnaryOp::Log)
    }

    /// `core.sqrt`: square root. Float only; `sqrt(x < 0) = NaN`.
    pub fn sqrt(self) -> Result<Self> {
        self.unary(UnaryOp::Sqrt)
    }

    /// `core.rsqrt`: `1 / sqrt(self)`. Float only; `rsqrt(0) = inf`. Computed as a reciprocal of
    /// the square root (Burn has no fused `rsqrt`), so it is rounded twice.
    pub fn rsqrt(self) -> Result<Self> {
        self.unary(UnaryOp::Rsqrt)
    }

    /// `core.sin`: sine (radians). Float only.
    pub fn sin(self) -> Result<Self> {
        self.unary(UnaryOp::Sin)
    }

    /// `core.cos`: cosine (radians). Float only.
    pub fn cos(self) -> Result<Self> {
        self.unary(UnaryOp::Cos)
    }

    /// `core.tanh`: hyperbolic tangent. Float only.
    pub fn tanh(self) -> Result<Self> {
        self.unary(UnaryOp::Tanh)
    }

    /// `core.clamp`: limits every element to `[min, max]`. Float or int.
    ///
    /// At least one bound is required, and `min ≤ max` if both are given
    /// ([`TensorError::InvalidArgument`] otherwise). Bounds: [`Scalar::Int`] for every numeric
    /// dtype (it must be in range for an integer dtype), [`Scalar::Float`] for float dtypes only.
    /// Float bounds are rounded to the tensor's dtype. NaN propagates: a NaN element stays NaN,
    /// and a NaN bound makes every element NaN (as in PyTorch).
    pub fn clamp(self, min: Option<Scalar>, max: Option<Scalar>) -> Result<Self> {
        const OP: &str = "clamp";
        DTypeClass::Numeric.check(OP, self.dtype())?;
        if min.is_none() && max.is_none() {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: "at least one of `min` and `max` is required".into(),
            });
        }
        let dtype = self.dtype();
        let bounds = if dtype.is_float() {
            let min = min.map(|s| float_bound(OP, s)).transpose()?;
            let max = max.map(|s| float_bound(OP, s)).transpose()?;
            if let (Some(lo), Some(hi)) = (min, max)
                && lo > hi
            {
                return Err(TensorError::InvalidArgument {
                    op: OP,
                    reason: format!("`min` ({lo}) is greater than `max` ({hi})"),
                });
            }
            ClampBounds::Float(min, max)
        } else {
            let min = min.map(|s| int_bound(OP, s, dtype)).transpose()?;
            let max = max.map(|s| int_bound(OP, s, dtype)).transpose()?;
            if let (Some(lo), Some(hi)) = (min, max)
                && lo > hi
            {
                return Err(TensorError::InvalidArgument {
                    op: OP,
                    reason: format!("`min` ({lo}) is greater than `max` ({hi})"),
                });
            }
            ClampBounds::Int(min, max)
        };
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.clamp(bounds)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `core.where`: `a` where `cond` is true, `b` elsewhere, with all three broadcast together
    /// (graph-ir.md §3.1.3). `cond` is `bool`; `a` and `b` have the same dtype, any dtype.
    ///
    /// Named `where_` because `where` is a Rust keyword.
    pub fn where_(cond: Self, a: Self, b: Self) -> Result<Self> {
        const OP: &str = "where";
        DTypeClass::Bool.check(OP, cond.dtype())?;
        if a.dtype() != b.dtype() {
            return Err(TensorError::DTypeMismatch {
                op: OP,
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }
        check_same_device(OP, cond.device(), a.device())?;
        check_same_device(OP, a.device(), b.device())?;
        let shapes = [cond.shape(), a.shape(), b.shape()];
        let Some(shape) = broadcast_shapes(&[&shapes[0], &shapes[1], &shapes[2]]) else {
            let [_, lhs, rhs] = shapes;
            // Name the pair that fails; with only `cond` at fault, report it against `a`.
            let (lhs, rhs) = if broadcast_shapes(&[&lhs, &rhs]).is_some() {
                (cond.shape(), lhs)
            } else {
                (lhs, rhs)
            };
            return Err(TensorError::ShapeMismatch { op: OP, lhs, rhs });
        };
        let (dtype, device) = (a.dtype(), a.device());
        let (c, _, _) = cond.broadcast_to(&shape)?.into_parts();
        let (a, _, _) = a.broadcast_to(&shape)?.into_parts();
        let (b, _, _) = b.broadcast_to(&shape)?.into_parts();
        let inner = zip3_rank!(OP, c, a, b, x, y, z => Ranked::select(x, y, z)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    fn binary(self, other: Self, op: BinaryOp) -> Result<Self> {
        let name = op.name();
        let (lhs, rhs) = prepare_binary(name, op.class(), self, other)?;
        let (lhs, dtype, device) = lhs.into_parts();
        let (rhs, _, _) = rhs.into_parts();
        let host = match (dtype, device) {
            (DType::I64, DeviceId::Cpu) => op.host_i64(),
            _ => None,
        };
        let inner = zip_rank!(name, lhs, rhs, a, b => a.binary(b, op, host)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    fn unary(self, op: UnaryOp) -> Result<Self> {
        op.class().check(op.name(), self.dtype())?;
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.unary(op)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// Validated `clamp` bounds in the tensor's kind.
#[derive(Clone, Copy, Debug)]
enum ClampBounds {
    Float(Option<f64>, Option<f64>),
    Int(Option<i64>, Option<i64>),
}

fn float_bound(op: &'static str, bound: Scalar) -> Result<f64> {
    match bound {
        Scalar::Float(v) => Ok(v),
        // Integers are exact in f64 up to 2^53; beyond that they round like any float bound.
        Scalar::Int(v) => Ok(v as f64),
        Scalar::Bool(_) => Err(TensorError::InvalidArgument {
            op,
            reason: "a bool bound does not fit a float tensor".into(),
        }),
    }
}

fn int_bound(op: &'static str, bound: Scalar, dtype: DType) -> Result<i64> {
    let Scalar::Int(v) = bound else {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("{bound:?} is not an integer bound for a {dtype} tensor"),
        });
    };
    if dtype == DType::I32 && i32::try_from(v).is_err() {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("bound {v} is outside the i32 range"),
        });
    }
    Ok(v)
}

/// Runs `f` on `x` in `f32` and rounds the result back to `x`'s dtype, for `f16` and `bf16`;
/// `f32` runs as it is. Used for the transcendental ops and `pow`: the CubeCL backends evaluate
/// them in the storage dtype, and on lavapipe `tanh` in `f16` misses the spec's `rtol 1e-2` near
/// zero (cancellation). Burn's CPU backend computes them in `f32` already, so this makes every
/// backend round once from `f32`. The output dtype is unchanged; this is internal precision, not
/// a dtype conversion of the graph (graph-ir.md §2.2).
fn in_f32<const D: usize>(x: Tensor<D>, f: impl FnOnce(Tensor<D>) -> Tensor<D>) -> Tensor<D> {
    let dtype = match x.dtype() {
        BurnDType::F16 => FloatDType::F16,
        BurnDType::BF16 => FloatDType::BF16,
        _ => return f(x),
    };
    f(x.cast(FloatDType::F32)).cast(dtype)
}

/// `pow` with the special cases of C99 Annex F / IEEE 754 `pow`, which `torch.pow` follows, set
/// explicitly: the CubeCL backends compute `pow` as `exp2(b · log2(a))` (with a sign fix for
/// negative bases), which gets `pow(1, NaN)`, `pow(NaN, 0)`, `pow(0, 0)` and `pow(0, -1)` wrong
/// (seen on lavapipe). Finite non-zero bases with finite exponents keep the backend's result.
/// Later rules win over earlier ones.
fn pow_ieee<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> Tensor<D> {
    let inf = f64::INFINITY;
    let p = a.clone().powf(b.clone());
    // The sign of the base, -0.0 included (1 / -0.0 = -inf).
    let negative = a.clone().lower_elem(0.0).bool_or(
        a.clone()
            .equal_elem(0.0)
            .bool_and(a.clone().recip().lower_elem(0.0)),
    );
    let odd = b.clone().remainder_scalar(2.0).abs().equal_elem(1.0);
    let neg_odd = negative.bool_and(odd);
    let b_pos = b.clone().greater_elem(0.0);
    let b_neg = b.clone().lower_elem(0.0);
    // Base ±0: pow(±0, b > 0) = ±0, pow(±0, b < 0) = ±inf; the sign only for odd integers b.
    let zero = a.clone().equal_elem(0.0);
    let p = p
        .mask_fill(zero.clone().bool_and(b_pos.clone()), 0.0)
        .mask_fill(zero.clone().bool_and(b_neg.clone()), inf)
        .mask_fill(
            zero.clone()
                .bool_and(b_pos.clone())
                .bool_and(neg_odd.clone()),
            -0.0,
        )
        .mask_fill(zero.bool_and(b_neg.clone()).bool_and(neg_odd.clone()), -inf);
    // Base ±inf: like ±0 with the exponent's sign flipped.
    let infinite = a.clone().is_inf();
    let p = p
        .mask_fill(infinite.clone().bool_and(b_pos.clone()), inf)
        .mask_fill(infinite.clone().bool_and(b_neg.clone()), 0.0)
        .mask_fill(
            infinite
                .clone()
                .bool_and(b_pos.clone())
                .bool_and(neg_odd.clone()),
            -inf,
        )
        .mask_fill(infinite.bool_and(b_neg).bool_and(neg_odd), -0.0);
    // Exponent ±inf: 0 or inf by |a| < 1 or > 1, and 1 for |a| = 1.
    let magnitude = a.clone().abs();
    let small = magnitude.clone().lower_elem(1.0);
    let large = magnitude.clone().greater_elem(1.0);
    let exp_inf = b.clone().is_inf();
    let pos_inf = exp_inf.clone().bool_and(b_pos);
    let neg_inf = exp_inf.clone().bool_and(b.clone().lower_elem(0.0));
    let p = p
        .mask_fill(pos_inf.clone().bool_and(small.clone()), 0.0)
        .mask_fill(pos_inf.bool_and(large.clone()), inf)
        .mask_fill(neg_inf.clone().bool_and(small), inf)
        .mask_fill(neg_inf.bool_and(large), 0.0)
        .mask_fill(exp_inf.bool_and(magnitude.equal_elem(1.0)), 1.0);
    // NaN propagates, except pow(1, b) = 1 and pow(a, ±0) = 1 for every a and b.
    let nan = a.clone().is_nan().bool_or(b.clone().is_nan());
    p.mask_fill(nan, f64::NAN)
        .mask_fill(a.equal_elem(1.0), 1.0)
        .mask_fill(b.equal_elem(0.0), 1.0)
}

/// Replaces the result with `x` wherever `x` is NaN (NaN propagation for `maximum`, `minimum`
/// and `clamp`, independent of how the backend's `max`/`min` treat NaN).
fn keep_nan<const D: usize>(result: Tensor<D>, x: Tensor<D>) -> Tensor<D> {
    let nan = x.clone().is_nan();
    result.mask_where(nan, x)
}

impl<const D: usize> Ranked<D> {
    /// The caller checked dtypes, class, devices and broadcast both operands to the same shape.
    fn binary(self, other: Self, op: BinaryOp, host: Option<fn(i64, i64) -> i64>) -> Result<Self> {
        match (self, other) {
            (Ranked::Float(a), Ranked::Float(b)) => Ok(Ranked::Float(match op {
                BinaryOp::Add => a.add(b),
                BinaryOp::Sub => a.sub(b),
                BinaryOp::Mul => a.mul(b),
                BinaryOp::Div => a.div(b),
                BinaryOp::Pow => in_f32(a, |a| {
                    let b = b.cast(FloatDType::F32);
                    pow_ieee(a, b)
                }),
                BinaryOp::Maximum => {
                    let out = a.clone().max_pair(b.clone());
                    keep_nan(keep_nan(out, a), b)
                }
                BinaryOp::Minimum => {
                    let out = a.clone().min_pair(b.clone());
                    keep_nan(keep_nan(out, a), b)
                }
            })),
            (Ranked::Int(a), Ranked::Int(b)) => {
                if let Some(f) = host {
                    return Ok(Ranked::Int(host_i64(a, b, f)?));
                }
                Ok(Ranked::Int(match op {
                    BinaryOp::Add => a.add(b),
                    BinaryOp::Sub => a.sub(b),
                    BinaryOp::Mul => a.mul(b),
                    BinaryOp::Maximum => a.max_pair(b),
                    BinaryOp::Minimum => a.min_pair(b),
                    BinaryOp::Div | BinaryOp::Pow => {
                        return Err(Ranked::Int(a).kind_error(op.name()));
                    }
                }))
            }
            (a, _) => Err(a.kind_error(op.name())),
        }
    }

    /// The caller checked the dtype class.
    fn unary(self, op: UnaryOp) -> Result<Self> {
        match self {
            Ranked::Float(x) => Ok(Ranked::Float(match op {
                UnaryOp::Neg => x.neg(),
                UnaryOp::Abs => x.abs(),
                UnaryOp::Exp => in_f32(x, Tensor::exp),
                UnaryOp::Log => in_f32(x, Tensor::log),
                UnaryOp::Sqrt => in_f32(x, Tensor::sqrt),
                UnaryOp::Rsqrt => in_f32(x, |x| x.sqrt().recip()),
                UnaryOp::Sin => in_f32(x, Tensor::sin),
                UnaryOp::Cos => in_f32(x, Tensor::cos),
                UnaryOp::Tanh => in_f32(x, Tensor::tanh),
            })),
            Ranked::Int(x) => match op {
                UnaryOp::Neg => Ok(Ranked::Int(x.neg())),
                UnaryOp::Abs => Ok(Ranked::Int(x.abs())),
                _ => Err(Ranked::Int(x).kind_error(op.name())),
            },
            x => Err(x.kind_error(op.name())),
        }
    }

    /// The caller validated the bounds (in range, `min ≤ max`, at least one given).
    fn clamp(self, bounds: ClampBounds) -> Result<Self> {
        match (self, bounds) {
            (Ranked::Float(x), ClampBounds::Float(min, max)) => {
                if min.is_some_and(f64::is_nan) || max.is_some_and(f64::is_nan) {
                    return Ok(Ranked::Float(x.full_like(f64::NAN)));
                }
                let out = match (min, max) {
                    (Some(lo), Some(hi)) => x.clone().clamp(lo, hi),
                    (Some(lo), None) => x.clone().clamp_min(lo),
                    (None, Some(hi)) => x.clone().clamp_max(hi),
                    (None, None) => x.clone(),
                };
                Ok(Ranked::Float(keep_nan(out, x)))
            }
            (Ranked::Int(x), ClampBounds::Int(min, max)) => Ok(Ranked::Int(match (min, max) {
                (Some(lo), Some(hi)) => x.clamp(lo, hi),
                (Some(lo), None) => x.clamp_min(lo),
                (None, Some(hi)) => x.clamp_max(hi),
                (None, None) => x,
            })),
            (x, _) => Err(x.kind_error("clamp")),
        }
    }

    /// `where`: `a` where `cond`, else `b`. The caller checked dtypes and broadcast all three to
    /// the same shape.
    fn select(cond: Self, a: Self, b: Self) -> Result<Self> {
        let Ranked::Bool(cond) = cond else {
            return Err(cond.kind_error("where"));
        };
        match (a, b) {
            (Ranked::Float(a), Ranked::Float(b)) => Ok(Ranked::Float(b.mask_where(cond, a))),
            (Ranked::Int(a), Ranked::Int(b)) => Ok(Ranked::Int(b.mask_where(cond, a))),
            (Ranked::Bool(a), Ranked::Bool(b)) => Ok(Ranked::Bool(b.mask_where(cond, a))),
            (a, _) => Err(a.kind_error("where")),
        }
    }
}

/// `i64` binary op on the host with a wrapping function, for the CPU backend (see the module
/// documentation of [`crate::ops`]). Both operands have the same shape (broadcast by the caller;
/// reading back an expanded view materializes it).
fn host_i64<const D: usize>(
    a: Tensor<D, Int>,
    b: Tensor<D, Int>,
    f: fn(i64, i64) -> i64,
) -> Result<Tensor<D, Int>> {
    let device = a.device();
    let shape = a.shape();
    let lhs = a.try_into_data().map_err(backend_error)?;
    let rhs = b.try_into_data().map_err(backend_error)?;
    let lhs = lhs.as_slice::<i64>().map_err(backend_error)?;
    let rhs = rhs.as_slice::<i64>().map_err(backend_error)?;
    if lhs.len() != rhs.len() {
        return Err(TensorError::Backend {
            message: format!(
                "host i64 op: operands have {} and {} elements",
                lhs.len(),
                rhs.len()
            ),
        });
    }
    let out: Vec<i64> = lhs.iter().zip(rhs).map(|(&x, &y)| f(x, y)).collect();
    let data = burn::tensor::TensorData::new(out, shape);
    Ok(Tensor::from_data(data, (&device, BurnDType::I64)))
}
