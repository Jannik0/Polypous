# ADR-0005: Native kernels for GGUF quantization formats

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Most models casual users download are GGUF files with block quantization (Q4_K_M etc.). Unpacking
them to f16 when loading roughly quadruples memory use and defeats the purpose on laptops.

## Decision

Polypous implements native kernels for GGUF block formats, starting with `Q8_0`, `Q4_K`, `Q5_K`,
`Q6_K`: dequantization, matrix-vector multiply, matrix-matrix multiply and row gather. GPU kernels
are written once in **CubeCL**. CPU kernels are Rust with runtime SIMD dispatch. Other formats
fall back to dequantizing when loading. Details: [quantization.md](../specs/quantization.md).

## Alternatives considered

- **Dequantize when loading only:** simple but memory-prohibitive.
- **Linking ggml/llama.cpp for quantized ops:** a C/C++ build dependency, a second tensor runtime, and
  awkward interaction with Burn's memory management.
- **Requiring Burn's own quantization format:** would mean re-quantizing every GGUF file with
  quality loss. It doesn't cover k-quants.

## Consequences

- A significant kernel-engineering effort, mitigated by bit-exact reference tests against `gguf-py`.
- How quantized blocks are stored on the device (Burn tensor vs CubeCL handle) will be decided by an M3 spike and recorded in a follow-up ADR.
