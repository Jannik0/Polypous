//! Tensors across the API boundary (public-api.md §2 rule 4, §1.3): [`TensorData`],
//! [`NamedTensor`], [`Dtype`] and [`ComputeDtype`].
//!
//! These are the facade's own serde types. They convert to and from `polypous_backend`'s
//! `TensorData` and `DType` and `polypous_ir`'s dtypes, so those crates can change without changing
//! the public JSON.

use std::fmt;

use polypous_backend as backend;
use serde::{Deserialize, Serialize};

use crate::error::{Error, ErrorKind};

/// A value dtype (graph-ir.md §2.2). JSON: `"f32"`, `"f16"`, `"bf16"`, `"i64"`, `"i32"`, `"bool"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Dtype {
    /// 32-bit IEEE 754 float.
    F32,
    /// 16-bit IEEE 754 float.
    F16,
    /// bfloat16.
    Bf16,
    /// 64-bit two's complement integer.
    I64,
    /// 32-bit two's complement integer.
    I32,
    /// Boolean, one byte per element: `0` false, `1` true.
    Bool,
}

impl Dtype {
    /// The stable name, the same as in JSON and in graph documents.
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

    /// Bytes per element in [`TensorData::bytes`].
    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::I64 => 8,
            Dtype::Bool => 1,
        }
    }

    pub(crate) fn to_backend(self) -> backend::DType {
        match self {
            Dtype::F32 => backend::DType::F32,
            Dtype::F16 => backend::DType::F16,
            Dtype::Bf16 => backend::DType::BF16,
            Dtype::I64 => backend::DType::I64,
            Dtype::I32 => backend::DType::I32,
            Dtype::Bool => backend::DType::Bool,
        }
    }

    pub(crate) fn from_backend(dtype: backend::DType) -> Option<Dtype> {
        Some(match dtype {
            backend::DType::F32 => Dtype::F32,
            backend::DType::F16 => Dtype::F16,
            backend::DType::BF16 => Dtype::Bf16,
            backend::DType::I64 => Dtype::I64,
            backend::DType::I32 => Dtype::I32,
            backend::DType::Bool => Dtype::Bool,
            _ => return None,
        })
    }

    pub(crate) fn from_ir(dtype: polypous_ir::Dtype) -> Dtype {
        match dtype {
            polypous_ir::Dtype::F32 => Dtype::F32,
            polypous_ir::Dtype::F16 => Dtype::F16,
            polypous_ir::Dtype::Bf16 => Dtype::Bf16,
            polypous_ir::Dtype::I64 => Dtype::I64,
            polypous_ir::Dtype::I32 => Dtype::I32,
            polypous_ir::Dtype::Bool => Dtype::Bool,
        }
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The compute dtype of a loaded model (graph-ir.md §2.2): what `float` inputs, state entries and
/// params resolve to. JSON: `"f32"` (default), `"f16"`, `"bf16"`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
    /// The stable name, the same as in JSON.
    pub fn as_str(self) -> &'static str {
        self.dtype().as_str()
    }

    /// The value dtype.
    pub fn dtype(self) -> Dtype {
        match self {
            ComputeDtype::F32 => Dtype::F32,
            ComputeDtype::F16 => Dtype::F16,
            ComputeDtype::Bf16 => Dtype::Bf16,
        }
    }

    pub(crate) fn to_ir(self) -> polypous_ir::ComputeDtype {
        match self {
            ComputeDtype::F32 => polypous_ir::ComputeDtype::F32,
            ComputeDtype::F16 => polypous_ir::ComputeDtype::F16,
            ComputeDtype::Bf16 => polypous_ir::ComputeDtype::Bf16,
        }
    }

    pub(crate) fn from_ir(dtype: polypous_ir::ComputeDtype) -> ComputeDtype {
        match dtype {
            polypous_ir::ComputeDtype::F32 => ComputeDtype::F32,
            polypous_ir::ComputeDtype::F16 => ComputeDtype::F16,
            polypous_ir::ComputeDtype::Bf16 => ComputeDtype::Bf16,
        }
    }
}

impl fmt::Display for ComputeDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A tensor in host memory (public-api.md §2 rule 4): row-major, every element little-endian;
/// IEEE 754 for `f32`/`f16`, bfloat16 for `bf16`, two's complement for `i64`/`i32`, one byte `0`
/// or `1` per `bool`.
///
/// The fields are public, so a value can be malformed; [`TensorData::validate`] checks it, and
/// every function that takes one validates it first.
///
/// JSON: `{"dtype":"f32","shape":[2,3],"bytes":[0,0,128,63,…]}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorData {
    /// The element type.
    pub dtype: Dtype,
    /// Dimensions, outermost first. A dimension may be `0` here; [`Model::run`](crate::Model::run)
    /// rejects it.
    pub shape: Vec<u64>,
    /// The elements, row-major, little-endian.
    pub bytes: Vec<u8>,
}

impl TensorData {
    /// A tensor, validated ([`TensorData::validate`]).
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] if the value is malformed.
    pub fn new(dtype: Dtype, shape: Vec<u64>, bytes: Vec<u8>) -> Result<TensorData, Error> {
        let data = TensorData {
            dtype,
            shape,
            bytes,
        };
        data.validate()?;
        Ok(data)
    }

    /// Checks the value: `bytes.len()` must equal the product of `shape` times the element size
    /// (computed with checked arithmetic), and every `bool` byte must be `0` or `1`.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] naming the problem.
    pub fn validate(&self) -> Result<(), Error> {
        let invalid = |details: String| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("The tensor data is malformed: {details}."),
            )
            .with_details(details)
        };
        let elements = self
            .shape
            .iter()
            .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
            .ok_or_else(|| {
                invalid(format!(
                    "the element count of shape {:?} overflows",
                    self.shape
                ))
            })?;
        let expected = elements
            .checked_mul(self.dtype.size_bytes() as u64)
            .ok_or_else(|| invalid(format!("the byte size of shape {:?} overflows", self.shape)))?;
        if self.bytes.len() as u64 != expected {
            return Err(invalid(format!(
                "{} bytes given, but shape {:?} with dtype {} needs {expected}",
                self.bytes.len(),
                self.shape,
                self.dtype
            )));
        }
        if self.dtype == Dtype::Bool
            && let Some((index, byte)) = self.bytes.iter().enumerate().find(|(_, b)| **b > 1)
        {
            return Err(invalid(format!(
                "bool element {index} is {byte}, but must be 0 or 1"
            )));
        }
        Ok(())
    }

    /// The backend's tensor data. The value must be valid.
    pub(crate) fn to_backend(&self) -> Result<backend::TensorData, Error> {
        backend::TensorData::new(
            self.dtype.to_backend(),
            self.shape.clone(),
            self.bytes.clone(),
        )
        .map_err(|error| {
            Error::new(
                ErrorKind::InvalidInput,
                format!("The tensor data is malformed: {error}."),
            )
            .with_details(error.to_string())
        })
    }

    pub(crate) fn from_backend(data: backend::TensorData) -> Result<TensorData, Error> {
        let dtype = Dtype::from_backend(data.dtype).ok_or_else(|| {
            Error::new(
                ErrorKind::Internal,
                "Polypous ran into an internal error. Please report this as a bug.",
            )
            .with_details(format!("unknown backend dtype {}", data.dtype))
        })?;
        Ok(TensorData {
            dtype,
            shape: data.shape,
            bytes: data.bytes,
        })
    }
}

/// A tensor with a name: a graph input, output or state entry, or a tensor of a tensor file.
///
/// JSON: `{"name":"x","tensor":{"dtype":"f32","shape":[1,2],"bytes":[…]}}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NamedTensor {
    /// The name.
    pub name: String,
    /// The tensor.
    pub tensor: TensorData,
}

impl NamedTensor {
    /// A named tensor.
    pub fn new(name: impl Into<String>, tensor: TensorData) -> NamedTensor {
        NamedTensor {
            name: name.into(),
            tensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtypes_have_stable_names() {
        for (dtype, name, size) in [
            (Dtype::F32, "f32", 4),
            (Dtype::F16, "f16", 2),
            (Dtype::Bf16, "bf16", 2),
            (Dtype::I64, "i64", 8),
            (Dtype::I32, "i32", 4),
            (Dtype::Bool, "bool", 1),
        ] {
            assert_eq!(dtype.as_str(), name);
            assert_eq!(dtype.size_bytes(), size);
            assert_eq!(dtype.to_backend().size_bytes(), size);
            assert_eq!(Dtype::from_backend(dtype.to_backend()), Some(dtype));
            assert_eq!(
                serde_json::to_string(&dtype).unwrap(),
                format!("\"{name}\"")
            );
        }
        assert_eq!(ComputeDtype::default(), ComputeDtype::F32);
        for compute in [ComputeDtype::F32, ComputeDtype::F16, ComputeDtype::Bf16] {
            assert_eq!(ComputeDtype::from_ir(compute.to_ir()), compute);
            assert_eq!(
                serde_json::to_string(&compute).unwrap(),
                format!("\"{compute}\"")
            );
        }
    }

    #[test]
    fn tensor_data_is_validated() {
        let ok = TensorData::new(Dtype::F32, vec![2], 1.5f32.to_le_bytes().repeat(2)).unwrap();
        assert_eq!(
            serde_json::to_string(&ok).unwrap(),
            r#"{"dtype":"f32","shape":[2],"bytes":[0,0,192,63,0,0,192,63]}"#
        );
        assert!(TensorData::new(Dtype::Bool, vec![3], vec![0, 1, 1]).is_ok());
        // A zero-size dim is valid data (rule 4); `Model::run` rejects it.
        assert!(TensorData::new(Dtype::I64, vec![0, 4], vec![]).is_ok());

        let cases = [
            (Dtype::F32, vec![2], vec![0; 7], "7 bytes given"),
            (Dtype::Bool, vec![2], vec![1, 2], "bool element 1 is 2"),
            (Dtype::I64, vec![u64::MAX, 2], vec![], "element count"),
            (Dtype::I64, vec![1 << 62], vec![], "byte size"),
        ];
        for (dtype, shape, bytes, text) in cases {
            let error = TensorData::new(dtype, shape, bytes).unwrap_err();
            assert_eq!(error.kind, ErrorKind::InvalidInput);
            assert!(error.message.contains(text), "{}", error.message);
        }
    }
}
