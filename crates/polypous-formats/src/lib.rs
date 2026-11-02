//! Model file format readers for Polypous.
//!
//! Responsibility: memory-mapped safetensors (single + sharded) and GGUF readers; tensor metadata;
//! lazy tensor views; GGUF metadata KV access.
//!
//! Pure Rust, with no GPU or Burn dependency. See `docs/architecture.md` §2.
