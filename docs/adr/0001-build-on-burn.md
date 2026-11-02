# ADR-0001: Build on Burn

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Polypous needs tensors, autodiff, optimizers and GPU backends for CUDA, Metal, Vulkan and CPU.
Writing these is a multi-year effort and not where Polypous adds value.

## Decision

Polypous is a layer on top of **Burn** (and its kernel language **CubeCL**). Burn provides tensor
operations, autodiff, optimizers and backends. Polypous adds the runtime graph, placement,
quantized-format kernels, model loading, pipelines and a wrapper-friendly API.

## Alternatives considered

- **candle:** strong inference story and GGUF support, but weaker training and fewer
  portable GPU backends (no Vulkan).
- **tch / libtorch:** large native dependency, hard to ship to casual users, CUDA-centric.
- **Own engine on raw wgpu/CUDA:** far too much scope.

## Consequences

- Burn's backends and their quirks become ours. We depend on Burn's pre-1.0 API, which is handled by [ADR-0007](0007-burn-version-pinning.md).
- Features Burn lacks (GGUF k-quants, disk streaming, runtime graphs) are built in Polypous.
- Upstream contributions to Burn are welcome when a fix belongs there.
