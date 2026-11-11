//! `nn.linear`, `nn.embedding` and `nn.dropout`: graph-ir.md §3.3, "Layers" (conv: `conv.rs`,
//! pools: `pool.rs`). Also the param checks shared by every `nn` op. General rules: §3.1 (see
//! the module documentation of [`crate::ops`]).
//!
//! Params are ordinary [`DynTensor`] arguments (graph-ir.md §2.4), already in the compute dtype
//! (§2.2), with PyTorch's layouts, so PyTorch weights load unchanged. Every `nn` op checks each
//! param against `x`: the same float dtype ([`TensorError::DTypeMismatch`]), the same device
//! ([`TensorError::DeviceMismatch`]) and the shape the spec table gives
//! ([`TensorError::InvalidArgument`], naming the param).

use super::indexing::check_indices;
use super::{DTypeClass, check_same_device, check_shape};
use crate::tensor::{DType, DynTensor, Inner, Ranked, Result, TensorError};

/// Checks a param of an `nn` op against `x`: same dtype, same device, shape `expected`.
/// `layout` is the spec's shape expression (`[C_out]`), for the message.
pub(crate) fn check_param(
    op: &'static str,
    name: &str,
    param: &DynTensor,
    dtype: DType,
    x: &DynTensor,
    expected: &[usize],
    layout: &str,
) -> Result<()> {
    if param.dtype() != dtype {
        return Err(TensorError::DTypeMismatch {
            op,
            lhs: dtype,
            rhs: param.dtype(),
        });
    }
    check_same_device(op, x.device(), param.device())?;
    let shape = param.shape();
    if shape != expected {
        return Err(TensorError::InvalidArgument {
            op,
            reason: format!("`{name}` has shape {shape:?}, needs {expected:?} ({layout})"),
        });
    }
    Ok(())
}

/// [`check_param`] for an optional param.
pub(crate) fn check_optional_param(
    op: &'static str,
    name: &str,
    param: Option<&DynTensor>,
    dtype: DType,
    x: &DynTensor,
    expected: &[usize],
    layout: &str,
) -> Result<()> {
    match param {
        Some(param) => check_param(op, name, param, dtype, x, expected, layout),
        None => Ok(()),
    }
}

/// `InvalidArgument` unless `x` has a rank in `min..=max`. `layout` is the spec's shape (`[N, C,
/// H, W]`).
pub(crate) fn check_rank(
    op: &'static str,
    x: &DynTensor,
    min: usize,
    max: usize,
    layout: &str,
) -> Result<()> {
    let rank = x.rank();
    if (min..=max).contains(&rank) {
        Ok(())
    } else {
        Err(TensorError::InvalidArgument {
            op,
            reason: format!(
                "`x` has shape {:?}, needs rank {} ({layout})",
                x.shape(),
                if min == max {
                    min.to_string()
                } else {
                    format!("{min} to {max}")
                }
            ),
        })
    }
}

/// `InvalidArgument` unless `value` is finite and `> 0` (`eps`, `beta`, `threshold`).
pub(crate) fn check_positive(op: &'static str, name: &str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(TensorError::InvalidArgument {
            op,
            reason: format!("`{name}` must be finite and > 0, got {value}"),
        })
    }
}

impl DynTensor {
    /// `nn.linear`: `x @ weight^T + bias` (`F.linear`). `x` is `[…, in]` with rank 1–6, `weight`
    /// is `[out, in]` and the optional `bias` is `[out]` (PyTorch's layout); the output is
    /// `[…, out]`. Float only; `x` and the params share the dtype and the device.
    ///
    /// Built on [`DynTensor::matmul`] (the transposed weight is a view), then the bias is added.
    /// `f16`/`bf16` compute in `f32` and round once (graph-ir.md §3.1.8).
    ///
    /// **Known limitation:** it shares matmul's: on CUDA GPUs with TF32 (Ampere or newer), Burn
    /// 0.22.1 runs `f32` matmul on TF32 tensor cores, which misses the spec's f32 accuracy (7 of 8
    /// `f32` reference cases fail on the reference machine's RTX 3060). No workaround here by
    /// design; it waits for an upstream precision flag (docs/status.md §4).
    pub fn linear(self, weight: Self, bias: Option<Self>) -> Result<Self> {
        const OP: &str = "linear";
        DTypeClass::Float.check(OP, self.dtype())?;
        let dtype = self.dtype();
        let shape = self.shape();
        let weight_shape = weight.shape();
        let [out, inp] = weight_shape[..] else {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`weight` has shape {weight_shape:?}, needs rank 2 ([out, in])"),
            });
        };
        check_param(
            OP,
            "weight",
            &weight,
            dtype,
            &self,
            &[out, inp],
            "[out, in]",
        )?;
        check_optional_param(OP, "bias", bias.as_ref(), dtype, &self, &[out], "[out]")?;
        let Some(&last) = shape.last() else {
            return Err(TensorError::UnsupportedRank { rank: 0 });
        };
        if last != inp {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "the last dim of `x` {shape:?} is not `in` = {inp} of `weight` {weight_shape:?}"
                ),
            });
        }
        let mut out_shape = shape.clone();
        if let Some(slot) = out_shape.last_mut() {
            *slot = out;
        }
        check_shape(OP, &out_shape, dtype)?;

        // f16/bf16 compute in f32 and round once (graph-ir.md §3.1.8, "Half-precision
        // accumulation"): in f16, Burn's matmul missed rtol 1e-2 on `flex` and lavapipe where the
        // output cancels to near zero (see the module documentation of `crate::ops`).
        let wide = |t: Self| {
            if dtype == DType::F32 {
                Ok(t)
            } else {
                t.cast(DType::F32)
            }
        };
        let rank1 = shape.len() == 1;
        let x = wide(if rank1 { self.unsqueeze(0)? } else { self })?;
        let y = x.matmul(wide(weight)?.transpose(0, 1)?)?;
        let y = match bias {
            Some(bias) => y.add(wide(bias)?)?,
            None => y,
        };
        let y = y.cast(dtype)?;
        if rank1 { y.squeeze(&[0]) } else { Ok(y) }
    }

    /// `nn.embedding`: row `ids[…]` of `weight` for every id (`F.embedding`). `self` is `ids`
    /// (`i32` or `i64`, rank 1–5), `weight` is `[num, dim]` (float); the output is
    /// `ids.shape + [dim]` in `weight`'s dtype. Every id must be in `0..num`.
    ///
    /// The ids are checked **before** the lookup, as for `core.index_select` (graph-ir.md §3.1.8):
    /// their minimum and maximum are read back to the host (one reduction and one device sync on
    /// a GPU). An id out of range, negative ids included, is [`TensorError::IndexOutOfRange`];
    /// nothing has run. Burn 0.22.1 does not check (`flex` panics, lavapipe reads out of bounds).
    pub fn embedding(self, weight: Self) -> Result<Self> {
        const OP: &str = "embedding";
        DTypeClass::Index.check(OP, self.dtype())?;
        DTypeClass::Float.check(OP, weight.dtype())?;
        check_same_device(OP, self.device(), weight.device())?;
        let ids_shape = self.shape();
        if ids_shape.len() > DynTensor::MAX_RANK - 1 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "`ids` has shape {ids_shape:?}; rank 6 gives an output of rank 7 (rank_too_large)"
                ),
            });
        }
        let weight_shape = weight.shape();
        let [num, dim] = weight_shape[..] else {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`weight` has shape {weight_shape:?}, needs rank 2 ([num, dim])"),
            });
        };
        let mut out_shape = ids_shape.clone();
        out_shape.push(dim);
        check_shape(OP, &out_shape, weight.dtype())?;
        check_indices(OP, &self, num)?;

        let count: usize = ids_shape.iter().product();
        let (ids, _, _) = self.reshape(&[count])?.into_parts();
        let (rows, dtype, device) = weight.into_parts();
        let (Inner::R1(Ranked::Int(ids)), Inner::R2(Ranked::Float(rows))) = (ids, rows) else {
            return Err(TensorError::UnsupportedOperation { op: OP, dtype });
        };
        let picked = Inner::R2(Ranked::Float(rows.select(0, ids)));
        DynTensor::from_parts(picked, dtype, device).reshape(&out_shape)
    }

    /// `nn.dropout` in inference mode: the identity (graph-ir.md §3.3; there is no training mode
    /// before M2). Float only. `p` is still checked, `0 ≤ p < 1`
    /// ([`TensorError::InvalidArgument`]), so a graph stays valid for training.
    pub fn dropout(self, p: f64) -> Result<Self> {
        const OP: &str = "dropout";
        DTypeClass::Float.check(OP, self.dtype())?;
        if !(0.0..1.0).contains(&p) {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`p` must be in [0, 1), got {p}"),
            });
        }
        Ok(self)
    }
}

#[cfg(test)]
#[path = "nn_layers_tests.rs"]
mod tests;
