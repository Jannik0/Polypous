# ADR-0011: Use Burn's runtime `Dispatch` backend for backend selection

- **Status:** Accepted (2026-11-02)
- **Date:** 2026-11-02
- **Deciders:** project owner (+ authoring agent)
- **Supersedes / Superseded by:** supersedes part of [ADR-0003](0003-runtime-backend-selection.md) (how the `BackendKind` enum dispatch is built), not its goals

## Context

[ADR-0003](0003-runtime-backend-selection.md) was written when Burn selected backends only
through a type parameter (`Tensor<B, D>`, `B: Backend`). It decides that `polypous-backend` writes
backend-generic code and instantiates it for each compiled-in backend through a `BackendKind` enum
and `match`. It names "Burn's own router/multi-backend facilities" as an alternative to evaluate in
M0. This ADR records that evaluation (issue #2), done against **Burn 0.22.1**, the version pinned
in M0 ([ADR-0007](0007-burn-version-pinning.md)).

What Burn 0.22.1 provides (sources: the `burn`, `burn-tensor` and `burn-dispatch` 0.22.1 crates):

- **The public tensor type has no backend type parameter anymore.** It is
  `burn::Tensor<const D: usize, K = Float>` (`burn-tensor/src/tensor/api/base.rs:76`). The backend
  is chosen by the `burn::Device` that a tensor was created on (`burn-tensor/src/device.rs`).
- `burn::Device` wraps `burn_dispatch::DispatchDevice`, an **enum with one variant per compiled-in
  backend family** (`burn-dispatch/src/device.rs`): `Cube` (every CubeCL runtime: CUDA, ROCm,
  Metal, Vulkan, WebGPU, CubeCL-CPU; the inner device names which), `Flex` (pure-Rust CPU),
  `NdArray`/`LibTorch` (deprecated), `Remote`, `Capture` and `Autodiff` (a wrapper).
- `burn_dispatch::Dispatch` implements Burn's `Backend` trait by **generated enum dispatch**
  (`#[backend_dispatch]` in `burn-backend-extension`): exactly the mechanism ADR-0003 plans to
  write by hand.
- `Tensor::to_device` across different backends goes **through host memory** (`TensorData`), and
  keeps the autodiff graph connected (`burn-tensor/src/tensor/api/base.rs`, doc of `to_device`).
  This matches ADR-0003's transfer rule.
- Autodiff is a property of the device (`device.autodiff()`), not a separate backend type.
- Device enumeration exists (`Device::enumerate`, `Device::enumerate_physical`), and device
  identity (`DeviceIdentity`, PCI address, adapter LUID) is reported for CubeCL devices. This can
  help with "a physical GPU reachable via both CUDA and Vulkan counts once"
  ([placement.md §1](../specs/placement.md#1-device-discovery)).
- Custom backend operations (our quantized kernels, [ADR-0005](0005-native-quantized-kernels.md))
  are added through `#[backend_extension]` traits that the same generator routes per backend.
- `Device::default()` picks CUDA > Metal > ROCm > Vulkan > WebGPU > wgpu > CPU > … The order and
  the probing rules in placement.md differ, so Polypous would still choose devices itself and never
  call `Device::default()` (it also panics when no backend is compiled in).

What Burn does **not** give us:

- **Dynamic rank.** `Tensor<D>` still has a compile-time rank. `DynTensor` (dynamic rank + dtype)
  is still needed.
- **Failure handling that matches ADR-0003.** CUDA enumeration returns no devices when `libcuda`
  is missing (`cubecl-cuda-0.11.3/src/runtime.rs:463-477`), which is what we want. But device
  initialization `unwrap()`s driver calls (`cubecl-cuda-0.11.3/src/runtime.rs:70-72`), so a driver
  that is present but broken panics. Probing has to isolate such failures (for example
  `catch_unwind` around the probe) whichever dispatch mechanism we use.
- An API we control. Burn's API changes between releases, so the facade must still not expose it.

## Decision

Proposed:

1. `polypous-backend` builds on `burn::Tensor` + `burn::Device` (Burn's `Dispatch` backend)
   instead of writing its own `BackendKind` enum over generic `fn f<B: Backend>` code.
2. `BackendKind` stays as a **Polypous** value type (`Cuda` / `Metal` / `Vulkan` / `Cpu`) used for
   discovery results, priorities, overrides and error messages. `polypous-backend` maps it to and
   from `burn::Device`. No Burn type leaks out of `polypous-backend` ([ADR-0007](0007-burn-version-pinning.md)).
3. `DynTensor` wraps `burn::Tensor` for each supported rank (or Burn's untyped primitive), so other
   crates still see only runtime values (`DeviceId`, `DynTensor`).
4. Everything else in ADR-0003 is kept: backends are compiled in through cargo features (`cpu`
   always on), cross-device transfers go through host memory, a missing GPU library makes that
   backend unavailable instead of failing, and priority/probing follow placement.md §1.

The public API of the `polypous` crate does not change, which is the condition ADR-0003 sets for
switching.

## Alternatives considered

- **Enum dispatch as planned in ADR-0003 (status quo):** still possible in Burn 0.22 through the
  per-backend types (`burn_dispatch::backends::{Cube, Flex}` and the `Backend` trait). It would
  duplicate the generated dispatch that Burn now ships, fight the direction of Burn's API (the
  high-level `Tensor`, `nn` modules and optimizers no longer take a backend parameter), and make
  every Burn upgrade larger. Its only advantage, full control over the dispatch, is not needed:
  device choice and probing stay ours either way.
- **Expose `burn::Tensor` beyond `polypous-backend`:** rejected, violates ADR-0007.

## Consequences

- Less code in `polypous-backend`: no hand-written match per op, and autodiff comes from
  `device.autodiff()` instead of an `Autodiff<B>` wrapper type per backend.
- Upgrades follow Burn's main API instead of a lower-level layer that Burn may change more freely.
- Our probing must defend against Burn/CubeCL panics during device initialization (needed either way).
- Specs to update if accepted: ADR-0003 marked "Superseded in part by 0011", `docs/architecture.md`
  §2 (`polypous-backend` row), `docs/specs/placement.md` §1 (mention `Device::enumerate_physical`
  and device identity), and the DynTensor issue (#5).
- If this ADR is **not** accepted, ADR-0003 stands unchanged and is implementable with Burn 0.22.1
  through the per-backend types.
