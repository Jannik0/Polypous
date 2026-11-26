//! `nn.linear`, `nn.embedding` and `nn.dropout` (inference and training mode): graph-ir.md §3.3,
//! "Layers" (conv: `conv.rs`, pools: `pool.rs`). Also the param checks shared by every `nn` op.
//! General rules: §3.1 (see the module documentation of [`crate::ops`]). Gradients are Burn's:
//! `linear` through `matmul`, `embedding` through `select` (repeated ids add up; the ids get
//! none), training-mode `dropout` through a product with a constant.
//!
//! Params are ordinary [`DynTensor`] arguments (graph-ir.md §2.4), already in the compute dtype
//! (§2.2), with PyTorch's layouts, so PyTorch weights load unchanged. Every `nn` op checks each
//! param against `x`: the same float dtype ([`TensorError::DTypeMismatch`]), the same device
//! ([`TensorError::DeviceMismatch`]) and the shape the spec table gives
//! ([`TensorError::InvalidArgument`], naming the param).

use burn::tensor::{Bool, Tensor};

use super::gradient::constant;
use super::indexing::check_indices;
use super::reduce::accumulate_in_f32;
use super::{DTypeClass, check_same_device, check_shape};
use crate::tensor::{DType, DynTensor, Inner, Ranked, Result, TensorError, zip_rank};

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

/// `t` in `f32` for the computation of `nn.linear` (a no-op for `f32`).
fn widen(t: DynTensor, dtype: DType) -> Result<DynTensor> {
    if dtype == DType::F32 {
        Ok(t)
    } else {
        t.cast(DType::F32)
    }
}

/// `x @ weightᵀ` of `nn.linear`, in `f32`. The caller checked the operands.
// #114: `linear_product` and `linear_bias` have their own frames. A debug build gives every
// temporary its own stack slot; inline, they made `linear`'s frame about 40 KB with `vulkan`, on
// the stack under both its `matmul` and its bias `add`.
#[inline(never)]
fn linear_product(x: DynTensor, weight: DynTensor, dtype: DType) -> Result<DynTensor> {
    let x = widen(x, dtype)?;
    x.matmul(widen(weight, dtype)?.transpose(0, 1)?)
}

/// `y + bias` of `nn.linear`, in `f32`. See [`linear_product`].
// #114: own frame, see `linear_product`.
#[inline(never)]
fn linear_bias(y: DynTensor, bias: DynTensor, dtype: DType) -> Result<DynTensor> {
    y.add(widen(bias, dtype)?)
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
        let rank1 = shape.len() == 1;
        let x = if rank1 { self.unsqueeze(0)? } else { self };
        let y = linear_product(x, weight, dtype)?;
        let y = match bias {
            Some(bias) => linear_bias(y, bias, dtype)?,
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

    /// `nn.dropout` in inference mode: the identity (graph-ir.md §3.3), for every `p`. Float
    /// only. `p` is still checked, `0 ≤ p < 1` ([`TensorError::InvalidArgument`]), so a graph stays
    /// valid for training. Training mode is [`DynTensor::dropout_training`].
    pub fn dropout(self, p: f64) -> Result<Self> {
        const OP: &str = "dropout";
        DTypeClass::Float.check(OP, self.dtype())?;
        check_dropout_p(OP, p)?;
        Ok(self)
    }

    /// `nn.dropout` in training mode (graph-ir.md §3.3, §3.1.10) with the caller's keep mask:
    /// `out = x · m`, where `m = s` where `keep` is true and `0` where it is false, and
    /// `s = 1 / (1 - p)` computed as PyTorch's CPU kernel computes it (`1 - p` in `f64`, rounded to
    /// `f32`, then the division in `f32`). The gradient is `g · m` (Burn's `mul` backward with the
    /// constant `m`), and both are exact for a given mask: one `f32` product per element.
    ///
    /// - `keep` is a `bool` tensor of `x`'s shape on `x`'s device. The backend never draws random
    ///   numbers (ADR-0014): the runtime builds the mask on the host with the Polypous generator
    ///   (`polypous_ir::rng::dropout_mask`, graph-ir.md §3.1.12) and uploads it. `keep` is a
    ///   constant: it has no gradient.
    /// - `p = 0` is the identity: `x` is returned as it is and `keep` isn't read (the rule keeps
    ///   every element at `p = 0`). It is still checked.
    /// - NaN or `±inf` in `x` at a dropped element gives NaN (`x · 0`), as in PyTorch.
    /// - `f16`/`bf16` `x` (outside M2's training mode, which requires `f32`) multiply in `f32`
    ///   and round once.
    ///
    /// Errors: a non-float `x` ([`TensorError::UnsupportedOperation`]), `p` outside `[0, 1)`
    /// ([`TensorError::InvalidArgument`]), a non-`bool` `keep` ([`TensorError::DTypeMismatch`]),
    /// another device ([`TensorError::DeviceMismatch`]) or another shape
    /// ([`TensorError::ShapeMismatch`]).
    pub fn dropout_training(self, p: f64, keep: Self) -> Result<Self> {
        const OP: &str = "dropout";
        DTypeClass::Float.check(OP, self.dtype())?;
        check_dropout_p(OP, p)?;
        if keep.dtype() != DType::Bool {
            return Err(TensorError::DTypeMismatch {
                op: OP,
                lhs: DType::Bool,
                rhs: keep.dtype(),
            });
        }
        check_same_device(OP, self.device(), keep.device())?;
        if keep.shape() != self.shape() {
            return Err(TensorError::ShapeMismatch {
                op: OP,
                lhs: self.shape(),
                rhs: keep.shape(),
            });
        }
        if p == 0.0 {
            return Ok(self);
        }
        let scale = dropout_scale(p);
        let (x, dtype, device) = self.into_parts();
        let (keep, _, _) = keep.into_parts();
        let inner = zip_rank!(OP, x, keep, x, keep => match (x, keep) {
            (Ranked::Float(x), Ranked::Bool(keep)) => {
                Ranked::Float(accumulate_in_f32(x, |x| apply_keep(x, keep, scale)))
            }
            (x, _) => return Err(x.kind_error(OP)),
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }
}

/// `InvalidArgument` unless `0 ≤ p < 1` (NaN fails).
fn check_dropout_p(op: &'static str, p: f64) -> Result<()> {
    if (0.0..1.0).contains(&p) {
        Ok(())
    } else {
        Err(TensorError::InvalidArgument {
            op,
            reason: format!("`p` must be in [0, 1), got {p}"),
        })
    }
}

/// The dropout scale `s = 1 / (1 - p)` as PyTorch's CPU kernel computes it (graph-ir.md §3.3):
/// `1 - p` in `f64`, rounded to `f32`, then the division in `f32`.
pub(crate) fn dropout_scale(p: f64) -> f32 {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "the rounding of `1 - p` to f32 is the spec's"
    )]
    let keep = (1.0 - p) as f32;
    1.0 / keep
}

/// `x · m` with `m = scale` where `keep`, else `0`. `m` is a constant: the gradient is `g · m`.
fn apply_keep<const D: usize>(x: Tensor<D>, keep: Tensor<D, Bool>, scale: f32) -> Tensor<D> {
    let m = constant(x.zeros_like()).mask_fill(keep, f64::from(scale));
    x * m
}

#[cfg(test)]
#[path = "nn_layers_tests.rs"]
mod tests;
