# ADR-0002: Runtime graph IR executed by an interpreter

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

In Burn, a model is a Rust struct (`#[derive(Module)]`) whose structure is fixed at compile time,
and tensors carry their rank and kind as type parameters. Visual builders and `.polyp`
definitions need models defined **at runtime**, from data.

## Decision

- Polypous defines a serializable **Graph IR** ([spec](../specs/graph-ir.md)), independent of Burn.
- An **interpreter** in `polypous-runtime` executes the graph by calling Burn tensor operations
  through `polypous-backend`.
- `polypous-backend` provides `DynTensor` (rank 1–6 and dtype as runtime values, wrapping
  Burn's typed tensors) and `DynamicModel`, which implements Burn's `Module` trait by hand so Burn
  optimizers and autodiff work unchanged.
- One IR serves the visual builder, training, and pretrained architectures.

## Alternatives considered

- **Generate Rust code from graphs and compile it:** fast execution, but needs a Rust toolchain at
  runtime. That is impossible for end-user apps.
- **ONNX as the IR:** a large spec with poor fit for KV-cache state, quantized params and
  builder metadata. ONNX *import* may still be added later.
- **Separate formats for builder graphs and pretrained models:** two systems to maintain.

## Consequences

- Each op is dispatched at runtime. That overhead is negligible because ops are large tensor kernels,
  and Burn's runtime fusion still applies.
- The rank/dtype dispatch code in `DynTensor` is verbose but simple, and a good candidate for macros.
- The IR's op catalog becomes a central, versioned contract.
