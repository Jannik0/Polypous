//! Comparison and logic ops: graph-ir.md §3.2, "Comparison and logic". They broadcast, and their
//! outputs are `bool`. NaN compares false, except in `not_equal`, where it compares true
//! (§3.1.8).

use super::{DTypeClass, prepare_binary};
use crate::tensor::{DType, DynTensor, Ranked, Result, map_rank, zip_rank};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl CompareOp {
    /// The IR op's name without the opset.
    fn name(self) -> &'static str {
        match self {
            CompareOp::Eq => "eq",
            CompareOp::Ne => "ne",
            CompareOp::Lt => "lt",
            CompareOp::Le => "le",
            CompareOp::Gt => "gt",
            CompareOp::Ge => "ge",
        }
    }

    fn class(self) -> DTypeClass {
        match self {
            CompareOp::Eq | CompareOp::Ne => DTypeClass::Any,
            _ => DTypeClass::Numeric,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogicOp {
    And,
    Or,
}

impl LogicOp {
    fn name(self) -> &'static str {
        match self {
            LogicOp::And => "and",
            LogicOp::Or => "or",
        }
    }
}

impl DynTensor {
    /// `core.eq`: `self == other` elementwise, broadcasting. Any dtype; the output is `bool`.
    pub fn equal(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Eq)
    }

    /// `core.ne`: `self != other` elementwise, broadcasting. Any dtype; NaN is unequal to
    /// everything, itself included.
    pub fn not_equal(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Ne)
    }

    /// `core.lt`: `self < other`, broadcasting. Float or int.
    pub fn less(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Lt)
    }

    /// `core.le`: `self <= other`, broadcasting. Float or int.
    pub fn less_equal(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Le)
    }

    /// `core.gt`: `self > other`, broadcasting. Float or int.
    pub fn greater(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Gt)
    }

    /// `core.ge`: `self >= other`, broadcasting. Float or int.
    pub fn greater_equal(self, other: Self) -> Result<Self> {
        self.compare(other, CompareOp::Ge)
    }

    /// `core.and`: logical and of two `bool` tensors, broadcasting.
    pub fn logical_and(self, other: Self) -> Result<Self> {
        self.logic(other, LogicOp::And)
    }

    /// `core.or`: logical or of two `bool` tensors, broadcasting.
    pub fn logical_or(self, other: Self) -> Result<Self> {
        self.logic(other, LogicOp::Or)
    }

    /// `core.not`: logical not of a `bool` tensor.
    pub fn logical_not(self) -> Result<Self> {
        const OP: &str = "not";
        DTypeClass::Bool.check(OP, self.dtype())?;
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Bool(t) => Ranked::Bool(t.bool_not()),
            t => return Err(t.kind_error(OP)),
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    fn compare(self, other: Self, op: CompareOp) -> Result<Self> {
        let name = op.name();
        let (lhs, rhs) = prepare_binary(name, op.class(), self, other)?;
        let (lhs, _, device) = lhs.into_parts();
        let (rhs, _, _) = rhs.into_parts();
        let inner = zip_rank!(name, lhs, rhs, a, b => a.compare(b, op)?);
        Ok(DynTensor::from_parts(inner, DType::Bool, device))
    }

    fn logic(self, other: Self, op: LogicOp) -> Result<Self> {
        let name = op.name();
        let (lhs, rhs) = prepare_binary(name, DTypeClass::Bool, self, other)?;
        let (lhs, dtype, device) = lhs.into_parts();
        let (rhs, _, _) = rhs.into_parts();
        let inner = zip_rank!(name, lhs, rhs, a, b => match (a, b) {
            (Ranked::Bool(a), Ranked::Bool(b)) => Ranked::Bool(match op {
                LogicOp::And => a.bool_and(b),
                LogicOp::Or => a.bool_or(b),
            }),
            (a, _) => return Err(a.kind_error(name)),
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

impl<const D: usize> Ranked<D> {
    /// The caller checked dtypes, class and devices and broadcast both operands.
    fn compare(self, other: Self, op: CompareOp) -> Result<Self> {
        let out = match (self, other) {
            (Ranked::Float(a), Ranked::Float(b)) => {
                // NaN is set explicitly: the CubeCL backends compare ordered, so `ne` with a NaN
                // gave `false` on lavapipe. Masking the other comparisons too keeps them
                // independent of how a backend (or a fast-math compiler) treats NaN.
                let nan = a.clone().is_nan().bool_or(b.clone().is_nan());
                let out = match op {
                    CompareOp::Eq => a.equal(b),
                    CompareOp::Ne => a.not_equal(b),
                    CompareOp::Lt => a.lower(b),
                    CompareOp::Le => a.lower_equal(b),
                    CompareOp::Gt => a.greater(b),
                    CompareOp::Ge => a.greater_equal(b),
                };
                if op == CompareOp::Ne {
                    out.bool_or(nan)
                } else {
                    out.bool_and(nan.bool_not())
                }
            }
            (Ranked::Int(a), Ranked::Int(b)) => match op {
                CompareOp::Eq => a.equal(b),
                CompareOp::Ne => a.not_equal(b),
                CompareOp::Lt => a.lower(b),
                CompareOp::Le => a.lower_equal(b),
                CompareOp::Gt => a.greater(b),
                CompareOp::Ge => a.greater_equal(b),
            },
            (Ranked::Bool(a), Ranked::Bool(b)) => match op {
                CompareOp::Eq => a.equal(b),
                CompareOp::Ne => a.not_equal(b),
                _ => return Err(Ranked::Bool(a).kind_error(op.name())),
            },
            (a, _) => return Err(a.kind_error(op.name())),
        };
        Ok(Ranked::Bool(out))
    }
}
