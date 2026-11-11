//! `nn.max_pool2d`, `nn.avg_pool2d` and `nn.adaptive_avg_pool2d`: graph-ir.md §3.3, "Layers".
//! General rules: §3.1 (see the module documentation of [`crate::ops`]).
//!
//! - Output sizes as for conv ([`window_output`]; the pools use `dl = 1` where they have no
//!   dilation), which must be ≥ 1. `padding` entries are at most `kernel_size / 2` (integer
//!   division, as PyTorch checks it); `ceil_mode`, `return_indices` and `divisor_override` don't
//!   exist (PyTorch's defaults).
//! - `max_pool2d` pads with `-inf`. With dilation, a window can still lie entirely in the padding
//!   (`H = 2`, `k = 2`, `p = 1`, `dl = 3`: the elements at `-1` and `2`); its result is `-inf`, as
//!   in PyTorch. A NaN in a window makes the result NaN, as in PyTorch. Both are set explicitly
//!   with one extra max-pool of a code tensor, because Burn 0.22.1's CubeCL `max_pool2d` pads with
//!   `-f32::MAX` and dropped NaN on lavapipe (`flex` got both right).
//! - The average pools compute `f16`/`bf16` in `f32` and round once (graph-ir.md §3.1.8,
//!   "Half-precision accumulation"), like `mean`.

use burn::tensor::module;
use burn::tensor::ops::{AvgPoolOptions, MaxPoolOptions};
use burn::tensor::{FloatDType, Tensor};

use super::DTypeClass;
use super::check_shape;
use super::conv::{check_at_least_one, window_output};
use super::nn_layers::check_rank;
use super::reduce::accumulate_in_f32;
use crate::tensor::{DynTensor, Inner, Ranked, Result, TensorError};

const NCHW: &str = "[N, C, H, W]";

/// Checks the common pool arguments and returns the output shape `[N, C, H_out, W_out]`.
fn check_pool(
    op: &'static str,
    x: &DynTensor,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<Vec<usize>> {
    DTypeClass::Float.check(op, x.dtype())?;
    check_rank(op, x, 4, 4, NCHW)?;
    check_at_least_one(op, "kernel_size", &kernel_size)?;
    check_at_least_one(op, "stride", &stride)?;
    check_at_least_one(op, "dilation", &dilation)?;
    for i in 0..2 {
        if padding[i] > kernel_size[i] / 2 {
            return Err(TensorError::InvalidArgument {
                op,
                reason: format!(
                    "`padding` {padding:?} must be at most half of `kernel_size` {kernel_size:?}"
                ),
            });
        }
    }
    let shape = x.shape();
    let mut out = vec![shape[0], shape[1]];
    for i in 0..2 {
        out.push(window_output(
            op,
            ["H", "W"][i],
            shape[2 + i],
            kernel_size[i],
            stride[i],
            padding[i],
            dilation[i],
        )?);
    }
    check_shape(op, &out, x.dtype())?;
    Ok(out)
}

/// The rank-4 Burn float tensor of a checked `x`.
fn nchw(
    op: &'static str,
    x: DynTensor,
) -> Result<(Tensor<4>, crate::tensor::DType, crate::device::DeviceId)> {
    match x.into_parts() {
        (Inner::R4(Ranked::Float(t)), dtype, device) => Ok((t, dtype, device)),
        (_, dtype, _) => Err(TensorError::UnsupportedOperation { op, dtype }),
    }
}

fn wrap(t: Tensor<4>, dtype: crate::tensor::DType, device: crate::device::DeviceId) -> DynTensor {
    DynTensor::from_parts(Inner::R4(Ranked::Float(t)), dtype, device)
}

impl DynTensor {
    /// `nn.max_pool2d` (`F.max_pool2d(x, kernel_size, stride, padding, dilation)`): the maximum
    /// of each window of `x` (`[N, C, H, W]`), padded with `-inf`. Float only. The runtime fills
    /// in `stride`'s default (`kernel_size`).
    ///
    /// `kernel_size`, `stride` and `dilation` entries must be ≥ 1, `padding` entries at most
    /// `kernel_size / 2`, and the output sizes ≥ 1; otherwise [`TensorError::InvalidArgument`].
    /// NaN propagates: a window with a NaN gives NaN.
    pub fn max_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        const OP: &str = "max_pool2d";
        check_pool(OP, &self, kernel_size, stride, padding, dilation)?;
        let (x, dtype, device) = nchw(OP, self)?;
        let options = MaxPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_dilation(dilation);
        let out = module::max_pool2d(x.clone(), options.clone());
        // Per element a code: 2 for NaN, 0 for -inf, 1 otherwise. Its max-pool is 2 where a
        // window holds a NaN, and below 0.5 where it holds nothing above -inf: only -inf inputs,
        // or only padding (the backend's padding value: -inf on `flex`, -f32::MAX on CubeCL).
        let code = x
            .clone()
            .is_nan()
            .float()
            .cast(FloatDType::F32)
            .add_scalar(1.0)
            .mask_fill(x.equal_elem(f64::NEG_INFINITY), 0.0);
        let pooled = module::max_pool2d(code, options);
        let out = out
            .mask_fill(pooled.clone().lower_elem(0.5), f64::NEG_INFINITY)
            .mask_fill(pooled.greater_elem(1.5), f64::NAN);
        Ok(wrap(out, dtype, device))
    }

    /// `nn.avg_pool2d` (`F.avg_pool2d(x, kernel_size, stride, padding,
    /// count_include_pad=count_include_pad)`): the mean of each window of `x` (`[N, C, H, W]`),
    /// zero-padded. With `count_include_pad` the divisor is the full window (`kh · kw`), without
    /// it only the elements inside `x`. Float only; the runtime fills in `stride`'s default.
    ///
    /// `kernel_size` and `stride` entries must be ≥ 1, `padding` entries at most
    /// `kernel_size / 2`, and the output sizes ≥ 1; otherwise [`TensorError::InvalidArgument`].
    pub fn avg_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Result<Self> {
        const OP: &str = "avg_pool2d";
        check_pool(OP, &self, kernel_size, stride, padding, [1, 1])?;
        let (x, dtype, device) = nchw(OP, self)?;
        let options = AvgPoolOptions::new(kernel_size)
            .with_stride(stride)
            .with_padding(padding)
            .with_count_include_pad(count_include_pad);
        let out = accumulate_in_f32(x, |x| module::avg_pool2d(x, options));
        Ok(wrap(out, dtype, device))
    }

    /// `nn.adaptive_avg_pool2d` (`F.adaptive_avg_pool2d(x, output_size)`): output cell `(i, j)`
    /// of `x` (`[N, C, H, W]`) averages rows `floor(i·H/oh)` to `ceil((i+1)·H/oh) - 1` and the
    /// columns likewise, so cells overlap where `oh` doesn't divide `H`. Float only.
    /// `output_size` entries must be ≥ 1 ([`TensorError::InvalidArgument`]); they may exceed the
    /// input size (cells then repeat input elements).
    pub fn adaptive_avg_pool2d(self, output_size: [usize; 2]) -> Result<Self> {
        const OP: &str = "adaptive_avg_pool2d";
        DTypeClass::Float.check(OP, self.dtype())?;
        check_rank(OP, &self, 4, 4, NCHW)?;
        check_at_least_one(OP, "output_size", &output_size)?;
        let shape = self.shape();
        check_shape(
            OP,
            &[shape[0], shape[1], output_size[0], output_size[1]],
            self.dtype(),
        )?;
        let (x, dtype, device) = nchw(OP, self)?;
        let out = accumulate_in_f32(x, |x| module::adaptive_avg_pool2d(x, output_size));
        Ok(wrap(out, dtype, device))
    }
}

#[cfg(test)]
#[path = "pool_tests.rs"]
mod tests;
