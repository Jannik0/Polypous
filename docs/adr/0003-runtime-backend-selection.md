# ADR-0003: Runtime backend selection via enum dispatch

- **Status:** Accepted, superseded in part by [ADR-0011](0011-use-burn-dispatch-for-backend-selection.md) (Burn's runtime dispatch replaces the hand-written enum dispatch; the goals and rules below still apply)
- **Date:** 2026-11-02

## Context

Burn selects backends through a type parameter (`B: Backend`). Polypous must choose CUDA, Metal,
Vulkan or CPU **at runtime** depending on the machine, and fall back when one fails.

## Decision

- Backends are compiled in through cargo features (`cuda`, `metal`, `vulkan`, `cpu`; `cpu` always on).
- Code that needs a concrete backend is written as generic code (`fn f<B: Backend>`) *inside*
  `polypous-backend` only. A `BackendKind` enum and `match` instantiate it for each compiled-in backend.
- Every other crate sees only runtime values (`DeviceId`, `DynTensor`), with no backend type parameters.
- Moving tensors between backends goes through host memory (`to_data` / `from_data`).
- GPU runtime libraries are loaded at runtime, and a missing library makes that backend unavailable instead of failing.
- Priority and probing follow [placement.md §1](../specs/placement.md#1-device-discovery).

## Alternatives considered

- **Trait objects over Burn backends:** Burn's `Backend` trait is not object-safe.
- **Burn's own router/multi-backend facilities:** to be evaluated during M0. If they cover our needs,
  a new ADR may switch to them, as long as our public API does not change.

## Consequences

- Compile time and binary size grow with each enabled backend.
- Cross-device transfers cost a host round trip. That is acceptable because transfers happen only at tier boundaries.
