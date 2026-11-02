# ADR-0004: Tiered GPU → CPU → disk placement

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Casual users have limited VRAM (often none, or an integrated GPU) and 8–16 GB RAM. The project's
priority is that models *run*, even slowly, rather than fail.

## Decision

Models are split into placement blocks (usually one transformer layer each). A deterministic planner
fills tiers in order: **GPU** (one device, CUDA/Metal before Vulkan), then **CPU RAM**, then
**Disk** (memory-mapped, prefetched, streamed). The tiers are combined: one model may use all three.
Plans and estimates are available before loading. Details: [placement.md](../specs/placement.md).

## Alternatives considered

- **All-or-nothing device choice:** simple, but fails for any model larger than VRAM.
- **Manual layer counts (llama.cpp `-ngl` style):** too technical for the target users. It remains available as an override.
- **Dynamic, cost-based optimization:** better performance, but non-deterministic and hard to test. May come later.

## Consequences

- Memory estimation per block must be accurate (target ±25%). This needs symbolic shape inference and liveness analysis.
- Disk-streamed inference is slow (limited by disk bandwidth). Apps get speed estimates to communicate this.
- Splitting a model across several GPUs is out of scope for v1.
