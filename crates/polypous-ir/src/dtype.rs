//! Dtypes of graph documents, `docs/specs/graph-ir.md` §2.2.

use std::fmt;

/// A concrete value dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
    /// bfloat16.
    Bf16,
    /// 64-bit signed integer.
    I64,
    /// 32-bit signed integer.
    I32,
    /// Boolean.
    Bool,
}

impl Dtype {
    /// The dtype name as written in documents.
    pub fn as_str(self) -> &'static str {
        match self {
            Dtype::F32 => "f32",
            Dtype::F16 => "f16",
            Dtype::Bf16 => "bf16",
            Dtype::I64 => "i64",
            Dtype::I32 => "i32",
            Dtype::Bool => "bool",
        }
    }

    /// Parses a concrete dtype name.
    pub fn parse(s: &str) -> Option<Dtype> {
        Some(match s {
            "f32" => Dtype::F32,
            "f16" => Dtype::F16,
            "bf16" => Dtype::Bf16,
            "i64" => Dtype::I64,
            "i32" => Dtype::I32,
            "bool" => Dtype::Bool,
            _ => return None,
        })
    }

    /// Whether this is a float dtype (class `F`).
    pub fn is_float(self) -> bool {
        matches!(self, Dtype::F32 | Dtype::F16 | Dtype::Bf16)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A dtype as written in a document: a concrete dtype or `"float"`.
///
/// On graph inputs and state entries `float` means the compute dtype; on params it means "any float
/// or quantized storage, seen in the compute dtype" (§2.2). Saving keeps `float` as written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DtypeSpec {
    /// A concrete dtype.
    Concrete(Dtype),
    /// `"float"`.
    Float,
}

impl DtypeSpec {
    /// The name as written in documents.
    pub fn as_str(self) -> &'static str {
        match self {
            DtypeSpec::Concrete(d) => d.as_str(),
            DtypeSpec::Float => "float",
        }
    }

    /// Parses a dtype name, `float` included.
    pub fn parse(s: &str) -> Option<DtypeSpec> {
        if s == "float" {
            Some(DtypeSpec::Float)
        } else {
            Dtype::parse(s).map(DtypeSpec::Concrete)
        }
    }

    /// Resolves `float` to the compute dtype.
    pub fn resolve(self, compute: ComputeDtype) -> Dtype {
        match self {
            DtypeSpec::Concrete(d) => d,
            DtypeSpec::Float => compute.dtype(),
        }
    }
}

impl fmt::Display for DtypeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The compute dtype of a loaded model (§2.2): `f32` by default, `f16` or `bf16` where supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ComputeDtype {
    /// `f32` (default).
    #[default]
    F32,
    /// `f16`.
    F16,
    /// `bf16`.
    Bf16,
}

impl ComputeDtype {
    /// The concrete dtype.
    pub fn dtype(self) -> Dtype {
        match self {
            ComputeDtype::F32 => Dtype::F32,
            ComputeDtype::F16 => Dtype::F16,
            ComputeDtype::Bf16 => Dtype::Bf16,
        }
    }
}
