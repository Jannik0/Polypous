//! `core.top_k`: graph-ir.md §3.2, "Other". General rules: §3.1 (see the module documentation
//! of [`crate::ops`]).
//!
//! **Runs on the host on every backend.** The tensor is read back once, each slice along `dim`
//! is sorted with a stable sort (NaN largest, ties in index order, graph-ir.md §3.1.8), and the
//! `k` values and indices are uploaded to the tensor's device. Burn 0.22.1 offers nothing better:
//! - it has no GPU sort; its CubeCL `sort`/`topk` read the tensor back to the host as well
//!   (`burn-backend` `float_sort`, `try_read_sync`),
//! - its sorts are unstable (`sort_unstable_by`), so ties come back in any order (seen on `flex`
//!   and lavapipe: a descending sort of a row with 41 equal values started at index 1943, not
//!   100),
//! - CubeCL's `topk_with_indices` returns `-3.4e38` with index `-1` for the slots after a NaN on
//!   lavapipe.
//!
//! The cost is **one device sync per call** and a transfer of the whole input; `top_k` is used on
//! logits for sampling, where that is small.

use super::reduce::check_index_dtype;
use super::{DTypeClass, check_axis, check_shape};
use crate::tensor::{DType, DynTensor, Result, TensorData, TensorError};

impl DynTensor {
    /// `core.top_k`: the `k` largest (or, with `largest: false`, smallest) values along `dim` and
    /// their indices, as `(values, indices)`. Float only.
    ///
    /// The output is sorted (largest first, or smallest first), equal values keep index order,
    /// and NaN counts as larger than every number (graph-ir.md §3.1.8). `values` has the input
    /// dtype; `indices` has `dtype` (`i64` or `i32`). Both have the input shape with `dim`'s size
    /// replaced by `k`.
    ///
    /// `dim` must be in range, `1 ≤ k ≤` the size of `dim`, and `dtype` an integer dtype the
    /// device supports. Runs on the host (one device sync, see the module documentation).
    pub fn top_k(self, k: usize, dim: usize, largest: bool, dtype: DType) -> Result<(Self, Self)> {
        const OP: &str = "top_k";
        DTypeClass::Float.check(OP, self.dtype())?;
        let shape = self.shape();
        check_axis(OP, "dim", dim, shape.len())?;
        let n = shape[dim];
        if k == 0 || k > n {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`k` {k} is outside 1..={n} (the size of dim {dim})"),
            });
        }
        let device = self.device();
        check_index_dtype(OP, dtype, n, device)?;
        let mut out_shape = shape.clone();
        out_shape[dim] = k;
        check_shape(OP, &out_shape, self.dtype())?;
        check_shape(OP, &out_shape, dtype)?;

        let data = self.into_data()?;
        let width = data.dtype.size_bytes();
        let keys: Vec<u32> = data
            .bytes
            .chunks_exact(width)
            .map(|b| order_key(data.dtype, b))
            .collect();
        let outer: usize = shape[..dim].iter().product();
        let inner: usize = shape[dim + 1..].iter().product();
        let count = outer * k * inner;
        let mut values = Vec::with_capacity(count * width);
        let mut indices: Vec<usize> = Vec::with_capacity(count);
        let mut order: Vec<usize> = Vec::with_capacity(n);
        // Output element (o, j, i) comes from input element (o, order[j], i).
        let mut picked = vec![0usize; k * inner];
        for o in 0..outer {
            for i in 0..inner {
                let at = |j: usize| (o * n + j) * inner + i;
                order.clear();
                order.extend(0..n);
                // `sort_by` is stable: equal keys keep index order.
                if largest {
                    order.sort_by(|&a, &b| keys[at(b)].cmp(&keys[at(a)]));
                } else {
                    order.sort_by(|&a, &b| keys[at(a)].cmp(&keys[at(b)]));
                }
                for (j, &src) in order[..k].iter().enumerate() {
                    picked[j * inner + i] = src;
                }
            }
            for j in 0..k {
                for i in 0..inner {
                    let src = picked[j * inner + i];
                    let at = ((o * n + src) * inner + i) * width;
                    values.extend_from_slice(&data.bytes[at..at + width]);
                    indices.push(src);
                }
            }
        }
        let dims: Vec<u64> = out_shape.iter().map(|&d| d as u64).collect();
        let index_bytes: Vec<u8> = if dtype == DType::I32 {
            // check_index_dtype proved that every index below n fits i32.
            indices
                .iter()
                .flat_map(|&i| (i as i32).to_le_bytes())
                .collect()
        } else {
            indices
                .iter()
                .flat_map(|&i| (i as i64).to_le_bytes())
                .collect()
        };
        let values = TensorData::new(data.dtype, dims.clone(), values)?;
        let indices = TensorData::new(dtype, dims, index_bytes)?;
        Ok((
            DynTensor::from_data(&values, device)?,
            DynTensor::from_data(&indices, device)?,
        ))
    }
}

/// A key whose unsigned order is the numeric order of a little-endian float element of `dtype`,
/// with `-0.0` equal to `0.0` and every NaN above `+inf` (and equal to each other).
fn order_key(dtype: DType, bytes: &[u8]) -> u32 {
    let (bits, sign, exponent_mask, mantissa_mask) = match dtype {
        DType::F32 => (
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            0x8000_0000,
            0x7f80_0000,
            0x007f_ffff,
        ),
        // f16: 5 exponent bits; bf16: 8 exponent bits. Both are 16 bits wide.
        DType::F16 => (
            u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16,
            0x8000_0000,
            0x7c00_0000,
            0x03ff_0000,
        ),
        _ => (
            u32::from(u16::from_le_bytes([bytes[0], bytes[1]])) << 16,
            0x8000_0000,
            0x7f80_0000,
            0x007f_0000,
        ),
    };
    let is_nan = bits & exponent_mask == exponent_mask && bits & mantissa_mask != 0;
    if is_nan {
        return u32::MAX;
    }
    if bits & !sign == 0 {
        // ±0.0: both map to the key of +0.0.
        return sign;
    }
    if bits & sign != 0 {
        // Negative: larger magnitudes are smaller numbers.
        !bits
    } else {
        bits | sign
    }
}

#[cfg(test)]
#[path = "topk_tests.rs"]
mod tests;
