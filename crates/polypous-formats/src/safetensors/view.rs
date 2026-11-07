//! Tensor views and value decoding (`docs/specs/formats.md` §1.1, §1.3).

use std::path::Path;

use super::{Dtype, Error, ErrorKind, TensorInfo};

/// A tensor of a [`SafetensorsFile`](super::SafetensorsFile): dtype, shape and bytes, borrowed
/// from the file without copying.
///
/// The bytes are little-endian, row-major, and not necessarily aligned to the element size.
#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    name: &'a str,
    info: &'a TensorInfo,
    data: &'a [u8],
    path: Option<&'a Path>,
}

/// Decoded tensor values (`docs/specs/formats.md` §1.1).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum TensorValues {
    /// `F32` values.
    F32(Vec<f32>),
    /// `F16` values as IEEE binary16 bit patterns; see [`f16_to_f32`].
    F16(Vec<u16>),
    /// `BF16` values as bfloat16 bit patterns; see [`bf16_to_f32`].
    BF16(Vec<u16>),
    /// `I64` values.
    I64(Vec<i64>),
    /// `I32` values.
    I32(Vec<i32>),
    /// `BOOL` values.
    Bool(Vec<bool>),
}

impl<'a> TensorView<'a> {
    pub(crate) fn new(
        name: &'a str,
        info: &'a TensorInfo,
        data: &'a [u8],
        path: Option<&'a Path>,
    ) -> Self {
        TensorView {
            name,
            info,
            data,
            path,
        }
    }

    /// The tensor's name.
    pub fn name(&self) -> &'a str {
        self.name
    }

    /// The tensor's metadata.
    pub fn info(&self) -> &'a TensorInfo {
        self.info
    }

    /// The element type.
    pub fn dtype(&self) -> Dtype {
        self.info.dtype()
    }

    /// The shape, outermost dimension first. Empty for a scalar.
    pub fn shape(&self) -> &'a [usize] {
        self.info.shape()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.info.numel()
    }

    /// The raw little-endian bytes, borrowed from the file.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Decodes the values. Supported for F32, F16, BF16, I64, I32 and BOOL; other dtypes fail with
    /// [`ErrorKind::UnsupportedDtype`]. `BOOL` bytes other than 0 and 1 fail with
    /// [`ErrorKind::InvalidBoolValue`]. Copies the data.
    pub fn values(&self) -> Result<TensorValues, Error> {
        let d = self.data;
        let values = match self.dtype() {
            Dtype::F32 => TensorValues::F32(decode(d, f32::from_le_bytes)?),
            Dtype::F16 => TensorValues::F16(decode(d, u16::from_le_bytes)?),
            Dtype::BF16 => TensorValues::BF16(decode(d, u16::from_le_bytes)?),
            Dtype::I64 => TensorValues::I64(decode(d, i64::from_le_bytes)?),
            Dtype::I32 => TensorValues::I32(decode(d, i32::from_le_bytes)?),
            Dtype::Bool => {
                let mut out = alloc::<bool>(d.len())?;
                for (index, &value) in d.iter().enumerate() {
                    match value {
                        0 => out.push(false),
                        1 => out.push(true),
                        _ => return Err(self.error(ErrorKind::InvalidBoolValue { index, value })),
                    }
                }
                TensorValues::Bool(out)
            }
            dtype => return Err(self.error(ErrorKind::UnsupportedDtype { dtype })),
        };
        Ok(values)
    }

    /// Decodes F32, F16 or BF16 values to `f32`. F16 and BF16 widen exactly. Other dtypes fail
    /// with [`ErrorKind::UnsupportedDtype`]. Copies the data.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>, Error> {
        let d = self.data;
        match self.dtype() {
            Dtype::F32 => decode(d, f32::from_le_bytes),
            Dtype::F16 => decode(d, |b| f16_to_f32(u16::from_le_bytes(b))),
            Dtype::BF16 => decode(d, |b| bf16_to_f32(u16::from_le_bytes(b))),
            dtype => Err(self.error(ErrorKind::UnsupportedDtype { dtype })),
        }
        .map_err(|e| self.context(e))
    }

    fn error(&self, kind: ErrorKind) -> Error {
        self.context(kind.into())
    }

    fn context(&self, err: Error) -> Error {
        let err = err.with_tensor(self.name);
        match self.path {
            Some(p) => err.with_path(p),
            None => err,
        }
    }
}

/// Allocates room for `n` values, reporting failure instead of aborting.
fn alloc<T>(n: usize) -> Result<Vec<T>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| ErrorKind::AllocationFailed {
            bytes: n.saturating_mul(size_of::<T>()),
        })?;
    Ok(v)
}

/// Decodes `N`-byte little-endian elements. A trailing partial element can't occur for validated
/// tensors (the length is `numel × N`); it would be ignored.
fn decode<T, const N: usize>(data: &[u8], f: impl Fn([u8; N]) -> T) -> Result<Vec<T>, Error> {
    let (chunks, _rest) = data.as_chunks::<N>();
    let mut out = alloc::<T>(chunks.len())?;
    out.extend(chunks.iter().map(|c| f(*c)));
    Ok(out)
}

/// Converts an IEEE 754 binary16 bit pattern to `f32`, exactly.
///
/// Subnormals, infinities and signed zeros are preserved; NaN stays NaN with its payload shifted
/// into the `f32` mantissa.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let man = u32::from(bits & 0x03ff);
    let out = match (exp, man) {
        (0, 0) => sign,
        (0, _) => {
            // Subnormal: man × 2^-24. Normalize: the highest set bit `p` (0..=9) becomes the
            // implicit 1, so the value is 1.f × 2^(p - 24) and the f32 exponent field p + 103.
            let p = 31 - man.leading_zeros();
            let frac = (man ^ (1 << p)) << (23 - p);
            sign | ((p + 103) << 23) | frac
        }
        (0x1f, _) => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(out)
}

/// Converts a bfloat16 bit pattern to `f32`, exactly (bfloat16 is the upper half of an `f32`).
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent reference: the binary16 definition evaluated in `f64`.
    fn f16_reference(bits: u16) -> f64 {
        let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = i32::from((bits >> 10) & 0x1f);
        let man = f64::from(bits & 0x03ff);
        match exp {
            0 => sign * man * 2f64.powi(-24),
            0x1f if man == 0.0 => sign * f64::INFINITY,
            0x1f => f64::NAN,
            _ => sign * (1.0 + man / 1024.0) * 2f64.powi(exp - 15),
        }
    }

    #[test]
    fn f16_to_f32_is_exact_for_every_bit_pattern() {
        for bits in 0..=u16::MAX {
            let got = f16_to_f32(bits);
            let want = f16_reference(bits);
            if want.is_nan() {
                assert!(got.is_nan(), "{bits:#06x}");
            } else {
                assert_eq!(f64::from(got), want, "{bits:#06x}");
                // Signed zero and sign in general.
                assert_eq!(got.is_sign_negative(), bits & 0x8000 != 0, "{bits:#06x}");
            }
        }
    }

    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert_eq!(f16_to_f32(0x8000).to_bits(), 0x8000_0000);
    }

    #[test]
    fn bf16_to_f32_is_the_upper_half() {
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0xc040), -3.0);
        assert_eq!(bf16_to_f32(0x0001), f32::from_bits(0x0001_0000));
        assert_eq!(bf16_to_f32(0xff80), f32::NEG_INFINITY);
        assert!(bf16_to_f32(0x7fc0).is_nan());
    }
}
