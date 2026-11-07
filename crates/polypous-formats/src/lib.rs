//! Model file format readers for Polypous.
//!
//! Responsibility: memory-mapped safetensors (single + sharded) and GGUF readers; tensor metadata;
//! lazy tensor views; GGUF metadata KV access.
//!
//! Pure Rust, with no GPU or Burn dependency and no dependency on other Polypous crates. See
//! `docs/architecture.md` §2 and the format spec, `docs/specs/formats.md`.
//!
//! - [`safetensors`]: reader (mmap, sharded) and writer, `docs/specs/formats.md` §1.
//!
//! Every model file is untrusted input: the readers validate before they expose data, use checked
//! arithmetic for every offset and size, and return errors instead of panicking.

pub mod safetensors;
