//! Quantization support for Polypous.
//!
//! Responsibility: quantized tensor representation, GGUF block formats, dequantization, quantized
//! matmul/matvec kernels (CubeCL for GPUs, SIMD Rust for CPU).
//!
//! See `docs/architecture.md` §2 and `docs/specs/quantization.md`.
