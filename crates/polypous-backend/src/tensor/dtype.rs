//! Value dtypes of [`DynTensor`](super::DynTensor) (`docs/specs/graph-ir.md` §2.2).

use std::fmt;

use burn::tensor::{BoolStore, DType as BurnDType, FloatDType, IntDType};

/// Element type of a tensor value.
///
/// These are the value dtypes of `docs/specs/graph-ir.md` §2.2. The parameter-only dtype `float`
/// and quantized storage formats are not value dtypes and have no variant here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float (half precision).
    F16,
    /// 16-bit brain float (8 exponent bits, 7 mantissa bits).
    BF16,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// Boolean. One byte per element in [`TensorData`](super::TensorData): `0` is false, `1` is
    /// true.
    Bool,
}

/// Which family of Burn tensor holds a dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Float,
    Int,
    Bool,
}

impl DType {
    /// Every value dtype, in declaration order.
    pub const ALL: [DType; 6] = [
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::Bool,
    ];

    /// Size of one element in [`TensorData`](super::TensorData), in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }

    /// The dtype's name as written in graphs (`"f32"`, `"bool"`, …).
    pub const fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I64 => "i64",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }

    /// Parses a dtype name as written in graphs. Returns `None` for unknown names.
    pub fn from_name(name: &str) -> Option<DType> {
        DType::ALL.into_iter().find(|dtype| dtype.name() == name)
    }

    /// Whether this is a floating-point dtype.
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F32 | DType::F16 | DType::BF16)
    }

    /// Whether this is an integer dtype.
    pub const fn is_int(self) -> bool {
        matches!(self, DType::I64 | DType::I32)
    }

    pub(crate) const fn kind(self) -> Kind {
        match self {
            DType::F32 | DType::F16 | DType::BF16 => Kind::Float,
            DType::I64 | DType::I32 => Kind::Int,
            DType::Bool => Kind::Bool,
        }
    }

    /// The Burn dtype used to store this dtype. For `Bool` this is the host layout (`Native`, one
    /// byte); Burn picks the device's own bool storage when a tensor is created.
    pub(crate) const fn to_burn(self) -> BurnDType {
        match self {
            DType::F32 => BurnDType::F32,
            DType::F16 => BurnDType::F16,
            DType::BF16 => BurnDType::BF16,
            DType::I64 => BurnDType::I64,
            DType::I32 => BurnDType::I32,
            DType::Bool => BurnDType::Bool(BoolStore::Native),
        }
    }

    /// The Burn float dtype. Only called for float dtypes; other dtypes map to `F32`, and callers
    /// check the kind first.
    pub(crate) const fn to_burn_float(self) -> FloatDType {
        match self {
            DType::F16 => FloatDType::F16,
            DType::BF16 => FloatDType::BF16,
            _ => FloatDType::F32,
        }
    }

    /// The Burn int dtype. Only called for int dtypes; other dtypes map to `I64`, and callers
    /// check the kind first.
    pub(crate) const fn to_burn_int(self) -> IntDType {
        match self {
            DType::I32 => IntDType::I32,
            _ => IntDType::I64,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip() {
        for dtype in DType::ALL {
            assert_eq!(DType::from_name(dtype.name()), Some(dtype));
            assert_eq!(dtype.size_bytes(), dtype.to_burn().size());
        }
        assert_eq!(DType::from_name("float"), None);
        assert_eq!(DType::from_name("q4_k"), None);
    }
}
