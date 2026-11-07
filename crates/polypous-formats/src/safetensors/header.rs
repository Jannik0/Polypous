//! Header validation (`docs/specs/formats.md` §1.2).

use std::collections::BTreeMap;

use super::json::{self, dtype_name_for_message};
use super::{Dtype, Error, ErrorKind, MAX_HEADER_LEN};

/// Size of the header length prefix.
pub(crate) const LEN_PREFIX: usize = 8;

/// Metadata of one tensor: dtype, shape and position in the data buffer. Validated: the byte
/// range matches the shape and dtype, and lies inside the data buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    dtype: Dtype,
    shape: Vec<usize>,
    numel: usize,
    data_offsets: (usize, usize),
}

impl TensorInfo {
    /// The element type.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    /// The shape, outermost dimension first. Empty for a scalar.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of elements: the product of the shape (1 for a scalar).
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// `[begin, end)` byte offsets relative to the start of the data buffer.
    pub fn data_offsets(&self) -> (usize, usize) {
        self.data_offsets
    }

    /// Size of the tensor's data in bytes.
    pub fn byte_len(&self) -> usize {
        // `end >= begin` is checked in `validate_tensor`; saturating keeps this panic-free anyway.
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }
}

/// A validated header.
#[derive(Debug)]
pub(crate) struct Header {
    pub(crate) metadata: BTreeMap<String, String>,
    /// Tensors sorted by name.
    pub(crate) tensors: BTreeMap<String, TensorInfo>,
    /// File offset of the data buffer (`8 + N`).
    pub(crate) data_start: usize,
}

/// Validates a complete file's bytes and returns its header (rules 1–15).
pub(crate) fn parse(bytes: &[u8]) -> Result<Header, Error> {
    let file_len = bytes.len();
    let Some(prefix) = bytes.first_chunk::<LEN_PREFIX>() else {
        return Err(ErrorKind::FileTooSmall {
            len: file_len as u64,
        }
        .into());
    };
    let header_len = u64::from_le_bytes(*prefix);
    if header_len > MAX_HEADER_LEN {
        return Err(ErrorKind::HeaderTooLarge {
            len: header_len,
            max: MAX_HEADER_LEN,
        }
        .into());
    }
    let out_of_bounds = || ErrorKind::HeaderOutOfBounds {
        header_len,
        file_len: file_len as u64,
    };
    let data_start = usize::try_from(header_len)
        .ok()
        .and_then(|n| n.checked_add(LEN_PREFIX))
        .ok_or_else(out_of_bounds)?;
    let header_bytes = bytes
        .get(LEN_PREFIX..data_start)
        .ok_or_else(out_of_bounds)?;
    let text = std::str::from_utf8(header_bytes).map_err(|_| ErrorKind::HeaderNotUtf8)?;
    let raw = json::parse_header(text)?;

    let mut tensors = BTreeMap::new();
    for (name, raw_tensor) in raw.tensors {
        let info = validate_tensor(raw_tensor).map_err(|e| Error::from(e).with_tensor(&name))?;
        tensors.insert(name, info);
    }

    // Rules 14 and 15: sorted by offsets, the tensors tile the data buffer exactly.
    let mut by_offset: Vec<(&str, (usize, usize))> = tensors
        .iter()
        .map(|(name, info)| (name.as_str(), info.data_offsets))
        .collect();
    by_offset.sort_unstable_by_key(|&(_, offsets)| offsets);
    let mut expected = 0usize;
    for (name, (begin, end)) in by_offset {
        if begin != expected {
            return Err(Error::from(ErrorKind::NonContiguous { expected, begin }).with_tensor(name));
        }
        expected = end;
    }
    // `data_start <= file_len` because `bytes.get(..data_start)` succeeded above.
    let data_len = file_len.saturating_sub(data_start);
    if expected != data_len {
        return Err(ErrorKind::DataLengthMismatch {
            covered: expected,
            data_len,
        }
        .into());
    }

    Ok(Header {
        metadata: raw.metadata,
        tensors,
        data_start,
    })
}

/// Rules 9 and 11–13 for one tensor.
fn validate_tensor(raw: json::RawTensor) -> Result<TensorInfo, ErrorKind> {
    let dtype = Dtype::from_name(&raw.dtype).ok_or_else(|| ErrorKind::UnknownDtype {
        name: dtype_name_for_message(&raw.dtype),
    })?;
    let shape = raw
        .shape
        .iter()
        .map(|&d| usize::try_from(d).map_err(|_| ErrorKind::SizeOverflow))
        .collect::<Result<Vec<usize>, _>>()?;
    let numel = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or(ErrorKind::SizeOverflow)?;
    let expected = dtype.byte_len(numel)?;
    let to_usize = |v: u64| usize::try_from(v).map_err(|_| ErrorKind::SizeOverflow);
    let (begin, end) = (to_usize(raw.data_offsets.0)?, to_usize(raw.data_offsets.1)?);
    let actual = end
        .checked_sub(begin)
        .ok_or(ErrorKind::InvalidOffsets { begin, end })?;
    if actual != expected {
        return Err(ErrorKind::LengthMismatch { expected, actual });
    }
    Ok(TensorInfo {
        dtype,
        shape,
        numel,
        data_offsets: (begin, end),
    })
}
