//! Element types of the safetensors format (`docs/specs/formats.md` §1.1).

use std::fmt;

/// A safetensors element type.
///
/// Every dtype the format defines is recognized; see `docs/specs/formats.md` §1.1 for which ones
/// can be decoded to values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Dtype {
    /// `BOOL`: one byte per element, 0 or 1.
    Bool,
    /// `U8`: unsigned 8-bit integer.
    U8,
    /// `I8`: signed 8-bit integer.
    I8,
    /// `I16`: signed 16-bit integer.
    I16,
    /// `U16`: unsigned 16-bit integer.
    U16,
    /// `F16`: IEEE 754 binary16.
    F16,
    /// `BF16`: bfloat16.
    BF16,
    /// `I32`: signed 32-bit integer.
    I32,
    /// `U32`: unsigned 32-bit integer.
    U32,
    /// `F32`: IEEE 754 binary32.
    F32,
    /// `F64`: IEEE 754 binary64.
    F64,
    /// `I64`: signed 64-bit integer.
    I64,
    /// `U64`: unsigned 64-bit integer.
    U64,
    /// `C64`: complex number of two `f32` parts.
    C64,
    /// `F8_E5M2`: 8-bit float, 5 exponent and 2 mantissa bits.
    F8E5M2,
    /// `F8_E4M3`: 8-bit float, 4 exponent and 3 mantissa bits.
    F8E4M3,
    /// `F8_E4M3FNUZ`: 8-bit float E4M3, finite, no negative zero.
    F8E4M3Fnuz,
    /// `F8_E5M2FNUZ`: 8-bit float E5M2, finite, no negative zero.
    F8E5M2Fnuz,
    /// `F8_E8M0`: 8-bit exponent-only scale (OCP microscaling).
    F8E8M0,
    /// `F6_E2M3`: 6-bit float (OCP microscaling), packed.
    F6E2M3,
    /// `F6_E3M2`: 6-bit float (OCP microscaling), packed.
    F6E3M2,
    /// `F4`: 4-bit float E2M1 (OCP microscaling), packed.
    F4,
}

impl Dtype {
    /// Every dtype, in the order of `docs/specs/formats.md` §1.1.
    pub const ALL: [Dtype; 22] = [
        Dtype::Bool,
        Dtype::U8,
        Dtype::I8,
        Dtype::I16,
        Dtype::U16,
        Dtype::F16,
        Dtype::BF16,
        Dtype::I32,
        Dtype::U32,
        Dtype::F32,
        Dtype::F64,
        Dtype::I64,
        Dtype::U64,
        Dtype::C64,
        Dtype::F8E5M2,
        Dtype::F8E4M3,
        Dtype::F8E4M3Fnuz,
        Dtype::F8E5M2Fnuz,
        Dtype::F8E8M0,
        Dtype::F6E2M3,
        Dtype::F6E3M2,
        Dtype::F4,
    ];

    /// The dtype's name in a safetensors header, for example `"BF16"` or `"F8_E4M3"`.
    pub const fn name(self) -> &'static str {
        match self {
            Dtype::Bool => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
            Dtype::C64 => "C64",
            Dtype::F8E5M2 => "F8_E5M2",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E4M3Fnuz => "F8_E4M3FNUZ",
            Dtype::F8E5M2Fnuz => "F8_E5M2FNUZ",
            Dtype::F8E8M0 => "F8_E8M0",
            Dtype::F6E2M3 => "F6_E2M3",
            Dtype::F6E3M2 => "F6_E3M2",
            Dtype::F4 => "F4",
        }
    }

    /// Parses a header dtype name. Names are case-sensitive, as in the format.
    pub fn from_name(name: &str) -> Option<Dtype> {
        Dtype::ALL.into_iter().find(|d| d.name() == name)
    }

    /// Size of one element in bits (4, 6, 8, 16, 32 or 64).
    pub const fn bits(self) -> usize {
        match self {
            Dtype::F4 => 4,
            Dtype::F6E2M3 | Dtype::F6E3M2 => 6,
            Dtype::Bool
            | Dtype::U8
            | Dtype::I8
            | Dtype::F8E5M2
            | Dtype::F8E4M3
            | Dtype::F8E4M3Fnuz
            | Dtype::F8E5M2Fnuz
            | Dtype::F8E8M0 => 8,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 16,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 32,
            Dtype::F64 | Dtype::I64 | Dtype::U64 | Dtype::C64 => 64,
        }
    }

    /// Alignment in bytes the writer gives tensors of this dtype: the element size, at least 1
    /// (sub-byte dtypes are byte-aligned). `docs/specs/formats.md` §1.6.
    pub const fn alignment(self) -> usize {
        let bytes = self.bits() / 8;
        if bytes == 0 { 1 } else { bytes }
    }

    /// Whether [`TensorView::values`](super::TensorView::values) can decode this dtype
    /// (F32, F16, BF16, I64, I32, BOOL).
    pub const fn is_readable_as_values(self) -> bool {
        matches!(
            self,
            Dtype::F32 | Dtype::F16 | Dtype::BF16 | Dtype::I64 | Dtype::I32 | Dtype::Bool
        )
    }

    /// Byte size of `numel` elements; an error if it overflows `usize` or isn't a whole number of
    /// bytes (sub-byte dtypes, `docs/specs/formats.md` §1.2 rules 11 and 12).
    pub(crate) fn byte_len(self, numel: usize) -> Result<usize, super::ErrorKind> {
        let bits = numel
            .checked_mul(self.bits())
            .ok_or(super::ErrorKind::SizeOverflow)?;
        if bits % 8 != 0 {
            return Err(super::ErrorKind::PartialByte { dtype: self, numel });
        }
        Ok(bits / 8)
    }
}

impl fmt::Display for Dtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_are_unique() {
        for d in Dtype::ALL {
            assert_eq!(Dtype::from_name(d.name()), Some(d));
        }
        let mut names: Vec<_> = Dtype::ALL.iter().map(|d| d.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), Dtype::ALL.len());
        assert_eq!(Dtype::from_name("f32"), None);
        assert_eq!(Dtype::from_name("Q4_K"), None);
    }

    #[test]
    fn byte_len_checks_sub_byte_and_overflow() {
        assert!(matches!(Dtype::F4.byte_len(4), Ok(2)));
        assert!(matches!(
            Dtype::F4.byte_len(3),
            Err(super::super::ErrorKind::PartialByte { .. })
        ));
        assert!(matches!(Dtype::F6E2M3.byte_len(4), Ok(3)));
        assert!(matches!(Dtype::F64.byte_len(0), Ok(0)));
        assert!(matches!(
            Dtype::F32.byte_len(usize::MAX),
            Err(super::super::ErrorKind::SizeOverflow)
        ));
    }
}
