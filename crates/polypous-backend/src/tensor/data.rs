//! Host-side tensor data (`docs/specs/public-api.md` §2 rule 4).

use super::{DType, Result, TensorError};

/// A tensor in host memory, in Polypous' plain exchange layout.
///
/// - `bytes` holds the elements in **row-major** order, each one **little-endian**, with
///   [`DType::size_bytes`] bytes per element. `bool` uses one byte per element, `0` or `1`.
/// - `shape` lists the dimensions, outermost first. A dimension may be `0` (an empty tensor).
///
/// This is the layout of `docs/specs/public-api.md` §2 rule 4. The fields are public, so a value
/// can be malformed. [`TensorData::validate`] checks it, and every consumer validates before use.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorData {
    /// Element type.
    pub dtype: DType,
    /// Dimensions, outermost first.
    pub shape: Vec<u64>,
    /// Elements, row-major, little-endian.
    pub bytes: Vec<u8>,
}

impl TensorData {
    /// Creates tensor data and validates it (see [`TensorData::validate`]).
    pub fn new(dtype: DType, shape: Vec<u64>, bytes: Vec<u8>) -> Result<Self> {
        let data = TensorData {
            dtype,
            shape,
            bytes,
        };
        data.validate()?;
        Ok(data)
    }

    /// Number of elements: the product of the dimensions.
    ///
    /// Fails if the product does not fit in `usize`.
    pub fn num_elements(&self) -> Result<usize> {
        let dims = self.dims()?;
        num_elements(&dims).ok_or_else(|| TensorError::InvalidData {
            reason: format!("element count of shape {:?} overflows", self.shape),
        })
    }

    /// Checks that the byte length matches shape and dtype, and that every `bool` byte is `0` or
    /// `1`.
    pub fn validate(&self) -> Result<()> {
        let elements = self.num_elements()?;
        let expected = elements
            .checked_mul(self.dtype.size_bytes())
            .ok_or_else(|| TensorError::InvalidData {
                reason: format!("byte size of shape {:?} overflows", self.shape),
            })?;
        if self.bytes.len() != expected {
            return Err(TensorError::InvalidData {
                reason: format!(
                    "{} bytes given, but shape {:?} with dtype {} needs {expected}",
                    self.bytes.len(),
                    self.shape,
                    self.dtype
                ),
            });
        }
        if self.dtype == DType::Bool
            && let Some(index) = self.bytes.iter().position(|&byte| byte > 1)
        {
            return Err(TensorError::InvalidData {
                reason: format!(
                    "bool element {index} is {}, but must be 0 or 1",
                    self.bytes[index]
                ),
            });
        }
        Ok(())
    }

    /// The shape as `usize` dimensions. Fails if a dimension doesn't fit in `usize`.
    pub(crate) fn dims(&self) -> Result<Vec<usize>> {
        self.shape
            .iter()
            .map(|&dim| {
                usize::try_from(dim).map_err(|_| TensorError::InvalidData {
                    reason: format!("dimension {dim} does not fit in this platform's usize"),
                })
            })
            .collect()
    }
}

/// Product of the dimensions, or `None` on overflow.
pub(crate) fn num_elements(dims: &[usize]) -> Option<usize> {
    dims.iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_well_formed_data() {
        assert!(TensorData::new(DType::F32, vec![2, 3], vec![0; 24]).is_ok());
        assert!(TensorData::new(DType::Bool, vec![3], vec![0, 1, 1]).is_ok());
        assert!(TensorData::new(DType::I64, vec![0, 5], vec![]).is_ok());
    }

    #[test]
    fn validate_rejects_wrong_byte_length() {
        let err = TensorData::new(DType::F16, vec![2, 2], vec![0; 7]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidData { .. }), "{err}");
    }

    #[test]
    fn validate_rejects_bad_bool_bytes() {
        let err = TensorData::new(DType::Bool, vec![2], vec![1, 2]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidData { .. }), "{err}");
    }

    #[test]
    fn validate_rejects_overflowing_shape() {
        let err = TensorData::new(DType::F32, vec![u64::MAX, u64::MAX], vec![]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidData { .. }), "{err}");
        let err = TensorData::new(DType::I64, vec![u64::MAX / 4], vec![]).unwrap_err();
        assert!(matches!(err, TensorError::InvalidData { .. }), "{err}");
    }
}
