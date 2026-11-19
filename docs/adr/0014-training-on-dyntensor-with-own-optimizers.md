# ADR-0014: Training on `DynTensor` with a Polypous param store and own optimizers

- **Status:** Proposed
- **Date:** 2026-11-19
- **Deciders:** project owner (+ authoring agent: the autodiff spike, issue #71)
- **Supersedes / Superseded by:** supersedes, in part, [ADR-0002](0002-runtime-graph-ir-interpreter.md):
  the clause "`DynamicModel`, which implements Burn's `Module` trait by hand so Burn optimizers and
  autodiff work unchanged". The rest of ADR-0002 (Graph IR, interpreter, `DynTensor`) stands.

## Context

ADR-0002 planned a hand-written `DynamicModel: burn::module::Module` so that Burn's optimizers
train runtime-defined graphs unchanged ([training.md §1](../specs/training.md#1-how-it-works), draft).
ADR-0007 keeps Burn inside `polypous-backend` and `polypous-quant`, and `polypous-train` owns the
optimizers ([architecture.md §2](../architecture.md)). Owner decision 3 for M2 (#71) prefers own
optimizers on `DynTensor` in `polypous-train`, if the autodiff spike confirms that this works.

The spike (#71, Burn 0.22.1, CPU `flex` and Vulkan on lavapipe; no CUDA) found:

1. **Burn's autodiff works on `DynTensor` as it is.** With Burn's `autodiff` feature, the autodiff
   context lives on each *tensor* (`Tensor::autodiff()`, `require_grad()`); its device reads as
   `DispatchDevice::Autodiff(inner)`, and Burn compares it equal to the plain device. Every rank
   1–6 and `f32`/`f16`/`bf16` record and back-propagate through the `Ranked`/`Inner` enums; ops mix
   tracked tensors with plain constants, bool masks, int indices and host-built tensors;
   `to_device` (also CPU ↔ lavapipe) keeps the graph. Nothing in the M1 op code needed to change
   for this, except one line: `device_id_of` sees through the autodiff wrapper.
2. **Gradients flow through almost every M1 op**, including the NaN masks, the `max_pool2d` code
   tensor, the own `log1p`/`tanh` series and the f32 compute paths for f16/bf16. `top_k` cuts the
   graph (host values); the M1 special-value masks of `pow`, `maximum`/`minimum`, `clamp` and
   `max`/`min` give gradients that differ from PyTorch at ties, kinks and special values. These are
   op-level fixes for #75–#77, not architecture questions.
3. **Burn's optimizers don't match PyTorch's exactly.** Fed PyTorch's own gradients, Burn's
   `Adam`/`AdamW` drift to 28–29 ulp (relative 2.4e-6) from `torch.optim.Adam`/`AdamW` after 3
   steps, because Burn keeps `beta_1`, `beta_2`, `eps` and the bias corrections in `f32`
   (`burn-optim-0.22.1/src/optim/adam.rs:195-209`, `adamw.rs:206-223`), while PyTorch computes the
   scalars in `f64`. Burn's defaults also differ (`eps` 1e-5, momentum `dampening` 0.1, AdamW
   `weight_decay` 1e-4; PyTorch: 1e-8, 0, 1e-2). Optimizers written on `DynTensor` ops in PyTorch's
   single-tensor op order, with scalars in `f64` rounded once, stay within 1 ulp (Adam, AdamW) and
   4 ulp (SGD with momentum and Nesterov; PyTorch fuses `add(alpha=…)` with FMA) over the same 3
   steps, on both devices.
4. **A Burn `Module` adds nothing we need.** A runtime param list fits `Module` only as
   `Vec<Param<Tensor<D>>>` per rank plus a name map, and Burn's optimizers then need Burn's
   `GradientsParams`, `ParamId`s and `ModuleOptimizer` records, all Burn types that must stay
   inside `polypous-backend` and change between Burn releases. Burn's per-tensor
   `Optimizer::step` would avoid the `Module` but keeps finding 3.
5. **Burn's RNG is process-global** (`burn-flex-0.22.1/src/backend.rs:20,131-135`,
   `cubek-random-0.3.1/src/base.rs:14,21-25`: `Device::seed` ignores the device), which conflicts
   with public-api.md §2 rule 9; owner decision 2 already moves init and dropout masks to the host.

## Decision

- **Autodiff stays Burn's.** `polypous-backend` enables Burn's `autodiff` feature and gives
  `DynTensor` a small autodiff API with no Burn types: mark a float tensor as a param leaf that keeps
  its gradient, back-propagate from a one-element loss, read a leaf's gradient as a plain
  `DynTensor` from an opaque Polypous gradients type, detach, and drop the autodiff context.
  Tracking is per tensor: a tensor keeps its `DeviceId`, inference tensors carry no autodiff
  context and pay nothing, and there is no separate "autodiff device" in the public API. Op
  gradients that differ from the conventions of [graph-ir.md §3](../specs/graph-ir.md) are fixed
  in `polypous-backend` (#75–#77).
- **Params live in a Polypous param store**, as `DynTensor`s keyed by their IR param path, owned
  by the runtime/trainer. There is no `DynamicModel` implementing Burn's `Module`, no Burn
  `Param`, `ParamId` or `GradientsParams`. Each step marks the trainable params as leaves, runs
  the interpreter, back-propagates the loss and reads one gradient per param.
- **Optimizers are Polypous code in `polypous-train`**, written in public `DynTensor` ops on
  tensors without autodiff context: `sgd{momentum, dampening, nesterov, weight_decay}`, `adam`,
  `adamw`, gradient clipping and the schedules. Each follows PyTorch's single-tensor algorithm
  (`torch.optim`, `foreach=False`) step by step; scalars (bias corrections, step sizes, decay
  factors) are computed in `f64` and rounded once to the param dtype. Optimizer state is
  `DynTensor`s plus step counts, saved by Polypous (training.md §5). Burn's `burn-optim` is not used.
- **No Burn RNG.** Init values and dropout masks come from a Polypous-owned, counter-based host
  generator keyed by seed, step and node path (owner decision 2), uploaded as data.

## Alternatives considered

- **(a) Hand-written Burn `Module` with Burn's optimizers, wrapped inside `polypous-backend`**
  (ADR-0002 as written): works (spike, `design_a` in `src/tensor/autodiff_spike_tests.rs`), but
  doesn't match PyTorch's Adam/AdamW exactly (finding 3), needs Burn's `optim` feature and module
  types behind a wrapper, keeps optimizer state in Burn's record format, and puts optimizer
  behavior under Burn's release cycle. Matching PyTorch would mean patching or re-implementing
  Burn's optimizers anyway.
- **(a') Burn's per-tensor `Optimizer::step` on our own param store:** no `Module`, but the same
  numerics as (a), and Burn types in the optimizer state.
- **Own autodiff (tape) on `DynTensor`:** full control over every gradient convention, but a large
  amount of new numerical code for something Burn already does correctly for almost every op.
  Not needed: the conventions that differ are fixed per op.

## Consequences

- `polypous-train` stays free of Burn (ADR-0007 holds); a Burn upgrade touches only the autodiff
  wrapper in `polypous-backend` (`tensor/autodiff.rs`, about 150 lines) and the op gradients.
- The optimizers are about 100 lines of `DynTensor` code (spike: `step_b`, SGD + Adam + AdamW) plus
  tests, and are checked against PyTorch with 3-step fixtures (#79). Expected differences are a few
  ulp (FMA in PyTorch's `add(alpha=…)`, `lerp`, `addcmul`, `addcdiv`), so the optimizer fixtures
  need a stated tolerance of a few ulp, not bit-exactness (#73).
- Optimizer steps run as several small elementwise ops per param. For the MNIST CNN of the spike
  (1.2 M params, batch 64, 4 vCPU), the AdamW update takes 22–29 ms of a 185–200 ms step on the CPU
  (`flex`) and 48 of 411 ms on lavapipe (spike, question 5). A fused update is a later
  optimization if profiles ask for it.
- Burn does not retain the graph (`retain_graph`): one backward per forward. The trainer combines
  losses before backward.
- Specs and docs to update (#73): training.md §1 (replace the `DynamicModel` paragraph),
  architecture.md §2 (`polypous-train`: "optimizers/schedulers via Burn") and the data-flow sketch,
  roadmap.md M2 ("`DynamicModel` implementing Burn `Module`"), glossary.md (`DynamicModel`),
  status.md.
