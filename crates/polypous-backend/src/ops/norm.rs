//! Normalization: graph-ir.md §3.3, "Normalization" (`nn.batch_norm layer_norm group_norm
//! rms_norm`). General rules: §3.1 (see the module documentation of [`crate::ops`]).
//!
//! All four follow PyTorch's formulas, with `eps` inside the square root:
//!
//! - `batch_norm` (inference mode): `(x - running_mean) / sqrt(running_var + eps) · weight +
//!   bias`, per channel (dim 1). In training mode ([`DynTensor::batch_norm_training`]) with the
//!   batch mean and biased variance over every dim except `C` instead, returning the new running
//!   statistics. The running statistics never get a gradient (graph-ir.md §3.3).
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
//! (PyTorch's `weight = 1`, `bias = 0`). Gradients (graph-ir.md §3.1.11) are Burn's autodiff of
//! these ops, through the statistics and `eps`.

use burn::tensor::Tensor;

use super::nn_layers::{check_optional_param, check_param, check_positive, check_rank};
use super::{DTypeClass, check_axis};
use crate::tensor::{DType, DynTensor, Ranked, Result, TensorError, map_rank, zip_rank};

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
    /// read, never updated, and never get a gradient (graph-ir.md §3.3): tracked ones are read as
    /// constants. Float only; `x` and the params share dtype and device. `eps` must be finite and
    /// `> 0` ([`TensorError::InvalidArgument`]). Training mode is
    /// [`DynTensor::batch_norm_training`].
    pub fn batch_norm(
        self,
        running_mean: Self,
        running_var: Self,
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f64,
    ) -> Result<Self> {
        const OP: &str = "batch_norm";
        let c = self.check_batch_norm(
            OP,
            &running_mean,
            &running_var,
            weight.as_ref(),
            bias.as_ref(),
            eps,
        )?;
        let (dtype, rank) = (self.dtype(), self.rank());
        let per_channel = |t: Self| -> Result<Self> { channel_view(t, c, rank)?.cast(DType::F32) };
        let x = self.cast(DType::F32)?;
        let mean = per_channel(running_mean.detach())?;
        let inv_std = per_channel(running_var.detach())?.add_eps_rsqrt(eps)?;
        x.sub(mean)?
            .mul(inv_std)?
            .channel_affine(weight, bias, c, rank)?
            .cast(dtype)
    }

    /// `nn.batch_norm` in training mode (`F.batch_norm(x, running_mean, running_var, weight,
    /// bias, training=True, momentum=momentum, eps=eps)`; graph-ir.md §3.3, §3.1.10): normalizes
    /// `x` (`[N, C, …]`, rank 2–6) with the batch statistics of each channel and returns
    /// `(out, new_running_mean, new_running_var)`.
    ///
    /// - The batch statistics run over every dim except `C`, `n` values per channel: the mean
    ///   `mean_c` and the **biased** variance `var_c = mean((x - mean_c)²)` (two passes).
    ///   `out = (x - mean_c) / sqrt(var_c + eps) · weight + bias`, and the gradient flows through
    ///   `mean_c` and `var_c` to `x` (Burn's autodiff of these ops), as in PyTorch.
    /// - The running statistics aren't read for `out`. The new ones are
    ///   `(1 - momentum) · running_mean + momentum · mean_c` and
    ///   `(1 - momentum) · running_var + momentum · var_c · n / (n - 1)` (the unbiased variance),
    ///   computed in `f32` from values without autodiff context, with `momentum`, `1 - momentum`
    ///   and `n / (n - 1)` computed in `f64` and rounded once to `f32` (runtime.md §3.3). They are
    ///   `[C]` in `x`'s dtype on `x`'s device, carry no autodiff context, and the running
    ///   statistics never get a gradient.
    /// - `f16`/`bf16` (outside M2's training mode, which requires `f32`) compute in `f32` and
    ///   round once, as in inference mode.
    ///
    /// Errors ([`TensorError::InvalidArgument`] unless noted): those of
    /// [`DynTensor::batch_norm`]; `momentum` outside `[0, 1]` (or NaN); fewer than 2 values per
    /// channel (`n < 2`, where PyTorch raises too).
    pub fn batch_norm_training(
        self,
        running_mean: Self,
        running_var: Self,
        weight: Option<Self>,
        bias: Option<Self>,
        eps: f64,
        momentum: f64,
    ) -> Result<(Self, Self, Self)> {
        const OP: &str = "batch_norm";
        let c = self.check_batch_norm(
            OP,
            &running_mean,
            &running_var,
            weight.as_ref(),
            bias.as_ref(),
            eps,
        )?;
        if !(0.0..=1.0).contains(&momentum) {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!("`momentum` must be in [0, 1], got {momentum}"),
            });
        }
        let shape = self.shape();
        // `check_shape` held for `x` when it was built, so the product doesn't overflow; c ≥ 1.
        let n = shape.iter().product::<usize>() / c;
        if n < 2 {
            return Err(TensorError::InvalidArgument {
                op: OP,
                reason: format!(
                    "training mode needs at least 2 values per channel, `x` {shape:?} has {n}"
                ),
            });
        }
        let (dtype, rank) = (self.dtype(), shape.len());
        let dims: Vec<usize> = (0..rank).filter(|&d| d != 1).collect();
        let x = self.cast(DType::F32)?;
        let mean = x.clone().mean(Some(&dims), true)?;
        let centered = x.sub(mean.clone())?;
        let var = centered
            .clone()
            .mul(centered.clone())?
            .mean(Some(&dims), true)?;
        let out = centered
            .mul(var.clone().add_eps_rsqrt(eps)?)?
            .channel_affine(weight, bias, c, rank)?
            .cast(dtype)?;

        // The running statistics, from values without autodiff context.
        let round = |v: f64| {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "the rounding to f32 is the spec's (runtime.md §3.3)"
            )]
            let v = v as f32;
            f64::from(v)
        };
        let keep = round(1.0 - momentum);
        let momentum = round(momentum);
        #[allow(
            clippy::cast_precision_loss,
            reason = "n is a tensor's element count; n / (n - 1) is rounded to f32 anyway"
        )]
        let bessel = round(n as f64 / (n - 1) as f64);
        let stat = |t: Self| -> Result<Self> { t.without_autodiff().reshape(&[c]) };
        let update = |running: Self, batch: Self, factor: Option<f64>| -> Result<Self> {
            let running = running.without_autodiff().cast(DType::F32)?;
            let (running, _, device) = running.into_parts();
            let (batch, _, _) = stat(batch)?.into_parts();
            let inner = zip_rank!(OP, running, batch, r, b => match (r, b) {
                (Ranked::Float(r), Ranked::Float(b)) => {
                    let b = b.mul_scalar(momentum);
                    let b = match factor {
                        Some(f) => b.mul_scalar(f),
                        None => b,
                    };
                    Ranked::Float(r.mul_scalar(keep).add(b))
                }
                (r, _) => return Err(r.kind_error(OP)),
            });
            DynTensor::from_parts(inner, DType::F32, device).cast(dtype)
        };
        let new_mean = update(running_mean, mean, None)?;
        let new_var = update(running_var, var, Some(bessel))?;
        Ok((out, new_mean, new_var))
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
        grouped
            .normalize(2, eps, Stats::MeanVar)?
            .reshape(&shape)?
            .channel_affine(weight, bias, c, rank)?
            .cast(dtype)
    }

    /// Checks `x`, the params and `eps` of `batch_norm` (both modes); returns `C`.
    fn check_batch_norm(
        &self,
        op: &'static str,
        running_mean: &Self,
        running_var: &Self,
        weight: Option<&Self>,
        bias: Option<&Self>,
        eps: f64,
    ) -> Result<usize> {
        DTypeClass::Float.check(op, self.dtype())?;
        check_rank(op, self, 2, 6, "[N, C, …]")?;
        check_positive(op, "eps", eps)?;
        let (dtype, c) = (self.dtype(), self.shape()[1]);
        let channel = [c];
        check_param(
            op,
            "running_mean",
            running_mean,
            dtype,
            self,
            &channel,
            "[C]",
        )?;
        check_param(op, "running_var", running_var, dtype, self, &channel, "[C]")?;
        check_optional_param(op, "weight", weight, dtype, self, &channel, "[C]")?;
        check_optional_param(op, "bias", bias, dtype, self, &channel, "[C]")?;
        Ok(c)
    }

    /// `self · weight + bias` (each optional, `[C]`) per channel of an `f32` `[N, C, …]` tensor.
    fn channel_affine(
        self,
        weight: Option<Self>,
        bias: Option<Self>,
        c: usize,
        rank: usize,
    ) -> Result<Self> {
        let mut out = self;
        if let Some(weight) = weight {
            out = out.mul(channel_view(weight, c, rank)?.cast(DType::F32)?)?;
        }
        if let Some(bias) = bias {
            out = out.add(channel_view(bias, c, rank)?.cast(DType::F32)?)?;
        }
        Ok(out)
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
