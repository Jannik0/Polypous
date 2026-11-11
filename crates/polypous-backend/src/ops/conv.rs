//! `nn.conv1d` and `nn.conv2d`: graph-ir.md §3.3, "Layers". General rules: §3.1 (see the module
//! documentation of [`crate::ops`]).
//!
//! Zero padding, stride, dilation and groups as in `F.conv1d`/`F.conv2d`, with PyTorch's weight
//! layout `[C_out, C_in/groups, k…]`, which is Burn's too. Every argument is checked before
//! Burn's `conv1d`/`conv2d` sees it (Burn asserts on bad options and shapes). The output size per
//! spatial dim is `(H + 2·p - dl·(k - 1) - 1) / s + 1` ([`window_output`], checked arithmetic),
//! which must be ≥ 1.
//!
//! `f16`/`bf16` compute in `f32` and round once (graph-ir.md §3.1.8): Burn's f16 convolution
//! missed `rtol 1e-2` on `flex` and lavapipe where outputs cancel to near zero. `f32` runs Burn's
//! kernel as it is. Unlike `matmul`, Burn's `f32` convolutions met the spec's accuracy on CUDA
//! (RTX 3060, every dtype), so the TF32 limitation of docs/status.md §4 doesn't apply here.

use burn::tensor::module;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{DType as BurnDType, FloatDType, Tensor};

use super::DTypeClass;
use super::check_shape;
use super::nn_layers::{check_optional_param, check_param, check_rank};
use crate::tensor::{DynTensor, Inner, Ranked, Result, TensorError};

/// Output size of a sliding window along one dim: `(size + 2·padding - dilation·(kernel - 1) -
/// 1) / stride + 1` (graph-ir.md §3.3), or `InvalidArgument` if it would be below 1 or overflow.
/// The caller checked `kernel`, `stride` and `dilation` ≥ 1.
pub(crate) fn window_output(
    op: &'static str,
    dim: &str,
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    let span = kernel
        .checked_sub(1)
        .and_then(|k| k.checked_mul(dilation))
        .and_then(|k| k.checked_add(1));
    let padded = padding.checked_mul(2).and_then(|p| p.checked_add(size));
    match (span, padded) {
        (Some(span), Some(padded)) if padded >= span && stride > 0 => {
            Ok((padded - span) / stride + 1)
        }
        _ => Err(TensorError::InvalidArgument {
            op,
            reason: format!(
                "the window (kernel {kernel}, dilation {dilation}) doesn't fit {dim} = {size} \
                 with padding {padding}: the output size would be below 1"
            ),
        }),
    }
}

/// `InvalidArgument` unless every entry of `values` is ≥ 1.
pub(crate) fn check_at_least_one(op: &'static str, name: &str, values: &[usize]) -> Result<()> {
    if values.contains(&0) {
        Err(TensorError::InvalidArgument {
            op,
            reason: format!("`{name}` entries must be ≥ 1, got {values:?}"),
        })
    } else {
        Ok(())
    }
}

/// The checked arguments of a convolution with `N` spatial dims.
struct Conv<const N: usize> {
    stride: [usize; N],
    padding: [usize; N],
    dilation: [usize; N],
    groups: usize,
}

impl<const N: usize> Conv<N> {
    /// Checks `x`, `weight`, `bias` and the attributes, and returns the output shape.
    fn check(
        &self,
        op: &'static str,
        x: &DynTensor,
        weight: &DynTensor,
        bias: Option<&DynTensor>,
    ) -> Result<Vec<usize>> {
        let (x_layout, w_layout) = if N == 1 {
            ("[N, C_in, L]", "[C_out, C_in/groups, k]")
        } else {
            ("[N, C_in, H, W]", "[C_out, C_in/groups, kh, kw]")
        };
        DTypeClass::Float.check(op, x.dtype())?;
        check_rank(op, x, N + 2, N + 2, x_layout)?;
        check_at_least_one(op, "stride", &self.stride)?;
        check_at_least_one(op, "dilation", &self.dilation)?;
        check_at_least_one(op, "groups", &[self.groups])?;
        let shape = x.shape();
        let weight_shape = weight.shape();
        if weight_shape.len() != N + 2 || weight_shape.contains(&0) {
            return Err(TensorError::InvalidArgument {
                op,
                reason: format!("`weight` has shape {weight_shape:?}, needs {w_layout}"),
            });
        }
        let (c_in, c_out, groups) = (shape[1], weight_shape[0], self.groups);
        if c_in % groups != 0 || c_out % groups != 0 {
            return Err(TensorError::InvalidArgument {
                op,
                reason: format!("`groups` {groups} must divide C_in = {c_in} and C_out = {c_out}"),
            });
        }
        let mut expected = weight_shape.clone();
        expected[1] = c_in / groups;
        check_param(op, "weight", weight, x.dtype(), x, &expected, w_layout)?;
        check_optional_param(op, "bias", bias, x.dtype(), x, &[c_out], "[C_out]")?;
        let mut out = vec![shape[0], c_out];
        for i in 0..N {
            out.push(window_output(
                op,
                if N == 1 { "L" } else { ["H", "W"][i] },
                shape[2 + i],
                weight_shape[2 + i],
                self.stride[i],
                self.padding[i],
                self.dilation[i],
            )?);
        }
        check_shape(op, &out, x.dtype())?;
        Ok(out)
    }

    fn options(&self) -> ConvOptions<N> {
        ConvOptions::new(self.stride, self.padding, self.dilation, self.groups)
    }
}

impl DynTensor {
    /// `nn.conv1d` (`F.conv1d(x, weight, bias, stride, padding, dilation, groups)`): `x` is
    /// `[N, C_in, L]`, `weight` `[C_out, C_in/groups, k]`, the optional `bias` `[C_out]`; the
    /// output is `[N, C_out, L_out]`. Float only; `x` and the params share dtype and device.
    ///
    /// `stride`, `dilation` and `groups` must be ≥ 1, `groups` must divide `C_in` and `C_out`,
    /// and `L_out` must be ≥ 1; otherwise [`TensorError::InvalidArgument`]. Zero padding only.
    pub fn conv1d(
        self,
        weight: Self,
        bias: Option<Self>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        const OP: &str = "conv1d";
        let conv = Conv {
            stride: [stride],
            padding: [padding],
            dilation: [dilation],
            groups,
        };
        conv.check(OP, &self, &weight, bias.as_ref())?;
        let (x, dtype, device) = self.into_parts();
        let (Inner::R3(Ranked::Float(x)), Inner::R3(Ranked::Float(w))) = (x, weight.into_parts().0)
        else {
            return Err(TensorError::UnsupportedOperation { op: OP, dtype });
        };
        let bias = float_vector(OP, bias)?;
        let out = in_f32(x, w, bias, |x, w, b| {
            module::conv1d(x, w, b, conv.options())
        });
        Ok(DynTensor::from_parts(
            Inner::R3(Ranked::Float(out)),
            dtype,
            device,
        ))
    }

    /// `nn.conv2d` (`F.conv2d(x, weight, bias, stride, padding, dilation, groups)`): `x` is
    /// `[N, C_in, H, W]`, `weight` `[C_out, C_in/groups, kh, kw]`, the optional `bias`
    /// `[C_out]`; the output is `[N, C_out, H_out, W_out]`. `stride`, `padding` and `dilation`
    /// are `[h, w]`. Float only; `x` and the params share dtype and device.
    ///
    /// `stride`, `dilation` and `groups` must be ≥ 1, `groups` must divide `C_in` and `C_out`,
    /// and both output sizes must be ≥ 1; otherwise [`TensorError::InvalidArgument`]. Zero
    /// padding only.
    pub fn conv2d(
        self,
        weight: Self,
        bias: Option<Self>,
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        groups: usize,
    ) -> Result<Self> {
        const OP: &str = "conv2d";
        let conv = Conv {
            stride,
            padding,
            dilation,
            groups,
        };
        conv.check(OP, &self, &weight, bias.as_ref())?;
        let (x, dtype, device) = self.into_parts();
        let (Inner::R4(Ranked::Float(x)), Inner::R4(Ranked::Float(w))) = (x, weight.into_parts().0)
        else {
            return Err(TensorError::UnsupportedOperation { op: OP, dtype });
        };
        let bias = float_vector(OP, bias)?;
        let out = in_f32(x, w, bias, |x, w, b| {
            module::conv2d(x, w, b, conv.options())
        });
        Ok(DynTensor::from_parts(
            Inner::R4(Ranked::Float(out)),
            dtype,
            device,
        ))
    }
}

/// Runs a convolution in `f32` for `f16`/`bf16` operands (all three the same dtype, checked by
/// the caller) and rounds the result once to their dtype; `f32` runs as it is.
fn in_f32<const D: usize>(
    x: Tensor<D>,
    w: Tensor<D>,
    bias: Option<Tensor<1>>,
    conv: impl FnOnce(Tensor<D>, Tensor<D>, Option<Tensor<1>>) -> Tensor<D>,
) -> Tensor<D> {
    let dtype = match x.dtype() {
        BurnDType::F16 => FloatDType::F16,
        BurnDType::BF16 => FloatDType::BF16,
        _ => return conv(x, w, bias),
    };
    let wide = FloatDType::F32;
    conv(x.cast(wide), w.cast(wide), bias.map(|b| b.cast(wide))).cast(dtype)
}

/// The Burn tensor of a checked rank-1 float param.
fn float_vector(op: &'static str, param: Option<DynTensor>) -> Result<Option<Tensor<1>>> {
    match param.map(DynTensor::into_parts) {
        None => Ok(None),
        Some((Inner::R1(Ranked::Float(t)), _, _)) => Ok(Some(t)),
        Some((_, dtype, _)) => Err(TensorError::UnsupportedOperation { op, dtype }),
    }
}

#[cfg(test)]
#[path = "conv_tests.rs"]
mod tests;
