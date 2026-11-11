//! Normalization: graph-ir.md §3.3, "Normalization" (`nn.batch_norm layer_norm group_norm
//! rms_norm`). General rules: §3.1 (see the module documentation of [`crate::ops`]).
//!
//! All four follow PyTorch's formulas, with `eps` inside the square root:
//!
//! - `batch_norm` (inference mode): `(x - running_mean) / sqrt(running_var + eps) · weight +
//!   bias`, per channel (dim 1).
//! - `layer_norm`: `(x - mean) / sqrt(var + eps) · weight + bias` over the trailing
//!   `normalized_shape` dims, with the **biased** variance (divided by the count).
//! - `group_norm`: the same over each group of `C / num_groups` channels together with all
//!   spatial dims, then `weight` and `bias` per channel.
//! - `rms_norm`: `x / sqrt(mean(x²) + eps) · weight` over the trailing dims.
//!
//! The variance is computed in **two passes** (`mean((x - mean)²)`), not as
//! `mean(x²) - mean(x)²`, which cancels catastrophically for inputs with a large mean.
//! `f16`/`bf16` compute in `f32` (params included) and round once to the dtype (graph-ir.md
//! §3.1.8, "Half-precision accumulation"). Absent optional params are left out of the formula
//! (PyTorch's `weight = 1`, `bias = 0`).

use burn::tensor::Tensor;

use super::nn_layers::{check_optional_param, check_param, check_positive, check_rank};
use super::{DTypeClass, check_axis};
use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank};

/// What a normalization subtracts and divides by, over the trailing dims from `start`.
#[derive(Clone, Copy, Debug)]
enum Stats {
    /// Mean and biased variance (`layer_norm`, `group_norm`).
    MeanVar,
    /// Root mean square, no centering (`rms_norm`).
    Rms,
}

impl DynTensor {
    /// `nn.batch_norm` in inference mode (`F.batch_norm(x, running_mean, running_var, weight,
    /// bias, training=False, eps=eps)`): `x` is `[N, C, …]` (rank 2–6); `running_mean`,
    /// `running_var` and the optional `weight` and `bias` are `[C]`. The running statistics are
    /// read, never updated. Float only; `x` and the params share dtype and device. `eps` must be
    /// finite and `> 0` ([`TensorError::InvalidArgument`]).
    pub fn batch_norm(
        self,
        running_mean: Self,
        running_var: Self,
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f64,
    ) -> Result<Self> {
        const OP: &str = "batch_norm";
        DTypeClass::Float.check(OP, self.dtype())?;
        check_rank(OP, &self, 2, 6, "[N, C, …]")?;
        check_positive(OP, "eps", eps)?;
        let (dtype, c) = (self.dtype(), self.shape()[1]);
        let channel = [c];
        check_param(
            OP,
            "running_mean",
            &running_mean,
            dtype,
            &self,
            &channel,
            "[C]",
        )?;
        check_param(
            OP,
            "running_var",
            &running_var,
            dtype,
            &self,
            &channel,
            "[C]",
        )?;
        check_optional_param(OP, "weight", weight.as_ref(), dtype, &self, &channel, "[C]")?;
        check_optional_param(OP, "bias", bias.as_ref(), dtype, &self, &channel, "[C]")?;
        let rank = self.rank();
        let per_channel = |t: Self| -> Result<Self> { channel_view(t, c, rank)?.cast(DType::F32) };
        let x = self.cast(DType::F32)?;
        let mean = per_channel(running_mean)?;
        let inv_std = per_channel(running_var)?.add_eps_rsqrt(eps)?;
        let mut out = x.sub(mean)?.mul(inv_std)?;
        if let Some(weight) = weight {
            out = out.mul(per_channel(weight)?)?;
        }
        if let Some(bias) = bias {
            out = out.add(per_channel(bias)?)?;
        }
        out.cast(dtype)
    }

    /// `nn.layer_norm` (`F.layer_norm(x, normalized_shape, weight, bias, eps)`): normalizes over
    /// the trailing dims `normalized_shape` (1 to `rank` entries, equal to `x`'s trailing dims),
    /// with the biased variance; the optional `weight` and `bias` have shape
    /// `normalized_shape`. Float only. The runtime fills `normalized_shape` in from `weight`'s
    /// shape where the graph leaves it out. `eps` must be finite and `> 0`.
    pub fn layer_norm(
        self,
        normalized_shape: &[usize],
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f64,
    ) -> Result<Self> {
        const OP: &str = "layer_norm";
        let start = self.check_trailing(OP, normalized_shape, eps)?;
        let dtype = self.dtype();
        check_optional_param(
            OP,
            "weight",
            weight.as_ref(),
            dtype,
            &self,
            normalized_shape,
            "normalized_shape",
        )?;
        check_optional_param(
            OP,
            "bias",
            bias.as_ref(),
            dtype,
            &self,
            normalized_shape,
            "normalized_shape",
        )?;
        let out = self
            .cast(DType::F32)?
            .normalize(start, eps, Stats::MeanVar)?;
        out.affine(weight, bias)?.cast(dtype)
    }

    /// `nn.rms_norm` (`F.rms_norm(x, normalized_shape, weight, eps)`): `x / sqrt(mean(x²) + eps)`
    /// over the trailing dims `normalized_shape`, times the optional `weight` (shape
    /// `normalized_shape`). Float only. `eps` is required (no dtype-dependent default) and must be
    /// finite and `> 0`.
    pub fn rms_norm(
        self,
        normalized_shape: &[usize],
        weight: Option<Self>,
        eps: f64,
    ) -> Result<Self> {
        const OP: &str = "rms_norm";
        let start = self.check_trailing(OP, normalized_shape, eps)?;
        let dtype = self.dtype();
        check_optional_param(
            OP,
            "weight",
            weight.as_ref(),
            dtype,
            &self,
            normalized_shape,
            "normalized_shape",
        )?;
        let out = self.cast(DType::F32)?.normalize(start, eps, Stats::Rms)?;
        out.affine(weight, None)?.cast(dtype)
    }

    /// `nn.group_norm` (`F.group_norm(x, num_groups, weight, bias, eps)`): `x` is `[N, C, …]`
    /// (rank 2–6); each of the `num_groups` groups of `C / num_groups` consecutive channels is
    /// normalized together with every spatial dim (biased variance), then the optional `weight`
    /// and `bias` (`[C]`) apply per channel. Float only. `num_groups` must be ≥ 1 and divide `C`;
    /// `eps` must be finite and `> 0`.
    pub fn group_norm(
        self,
        num_groups: usize,
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f64,
    ) -> Result<Self> {
        const OP: &str = "group_norm";
        DTypeClass::Float.check(OP, self.dtype())?;
        check_rank(OP, &self, 2, 6, "[N, C, …]")?;
        check_positive(OP, "eps", eps)?;
        let shape = self.shape();
        let (dtype, rank, c) = (self.dtype(), shape.len(), shape[1]);
        if num_groups == 0 || c % num_groups != 0 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`num_groups` {num_groups} must be ≥ 1 and divide C = {c}"),
            });
        }
        check_optional_param(OP, "weight", weight.as_ref(), dtype, &self, &[c], "[C]")?;
        check_optional_param(OP, "bias", bias.as_ref(), dtype, &self, &[c], "[C]")?;
        // [N, C, …] -> [N, G, C/G · spatial]: each group's elements are contiguous in row-major
        // order, so a reshape groups them.
        let rest: usize = shape[1..].iter().product::<usize>() / num_groups;
        let grouped = self
            .cast(DType::F32)?
            .reshape(&[shape[0], num_groups, rest])?;
        let mut out = grouped.normalize(2, eps, Stats::MeanVar)?.reshape(&shape)?;
        if let Some(weight) = weight {
            out = out.mul(channel_view(weight, c, rank)?.cast(DType::F32)?)?;
        }
        if let Some(bias) = bias {
            out = out.add(channel_view(bias, c, rank)?.cast(DType::F32)?)?;
        }
        out.cast(dtype)
    }

    /// Checks `x`, `normalized_shape` and `eps` of `layer_norm`/`rms_norm`; returns the first
    /// normalized dim.
    fn check_trailing(
        &self,
        op: &'static str,
        normalized_shape: &[usize],
        eps: f64,
    ) -> Result<usize> {
        DTypeClass::Float.check(op, self.dtype())?;
        check_positive(op, "eps", eps)?;
        let shape = self.shape();
        let k = normalized_shape.len();
        if k == 0 || k > shape.len() || shape[shape.len() - k..] != *normalized_shape {
            return Err(TensorError::InvalidArgument {
                op,
                reason: format!(
                    "`normalized_shape` {normalized_shape:?} must be 1 to {} trailing dims of `x` {shape:?}",
                    shape.len()
                ),
            });
        }
        Ok(shape.len() - k)
    }

    /// `f32` only: normalizes over the dims `start..rank` with the statistics `stats`.
    fn normalize(self, start: usize, eps: f64, stats: Stats) -> Result<Self> {
        check_axis("normalize", "start", start, self.rank())?;
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => t.normalize(start, eps, stats)?);
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `1 / sqrt(self + eps)`, `f32` only.
    fn add_eps_rsqrt(self, eps: f64) -> Result<Self> {
        let (inner, dtype, device) = self.into_parts();
        let inner = map_rank!(inner, t => match t {
            Ranked::Float(t) => Ranked::Float(t.add_scalar(eps).sqrt().recip()),
            other => return Err(other.kind_error("batch_norm")),
        });
        Ok(DynTensor::from_parts(inner, dtype, device))
    }

    /// `self · weight + bias` (each optional) in `f32`, with the params broadcast over the
    /// leading dims.
    fn affine(self, weight: Option<Self>, bias: Option<Self>) -> Result<Self> {
        let mut out = self;
        if let Some(weight) = weight {
            out = out.mul(weight.cast(DType::F32)?)?;
        }
        if let Some(bias) = bias {
            out = out.add(bias.cast(DType::F32)?)?;
        }
        Ok(out)
    }
}

/// A `[C]` param as `[1, C, 1, …]` of rank `rank`, which broadcasts over `[N, C, …]`.
fn channel_view(t: DynTensor, c: usize, rank: usize) -> Result<DynTensor> {
    let mut shape = vec![1; rank];
    shape[1] = c;
    t.reshape(&shape)
}

impl<const D: usize> Ranked<D> {
    /// Normalizes an `f32` tensor over the dims `start..D` (the caller checked `start < D`).
    fn normalize(self, start: usize, eps: f64, stats: Stats) -> Result<Self> {
        let Ranked::Float(x) = self else {
            return Err(self.kind_error("normalize"));
        };
        let shape = x.dims();
        let count: usize = shape[start..].iter().product();
        let mean_over = |t: Tensor<D>| -> Tensor<D> {
            (start..D)
                .fold(t, |t, d| t.sum_dim(d))
                .div_scalar(count as f64)
        };
        Ok(Ranked::Float(match stats {
            Stats::MeanVar => {
                let centered = x.clone().sub(mean_over(x).expand(shape));
                let var = mean_over(centered.clone().mul(centered.clone()));
                centered.mul(var.add_scalar(eps).sqrt().recip().expand(shape))
            }
            Stats::Rms => {
                let mean_square = mean_over(x.clone().mul(x.clone()));
                x.mul(mean_square.add_scalar(eps).sqrt().recip().expand(shape))
            }
        }))
    }
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
