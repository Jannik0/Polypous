//! Graph IR for Polypous.
//!
//! Responsibility: graph IR data model, op catalog (opset), validation, symbolic shape inference,
//! JSON (de)serialization, versioning/migration.
//!
//! Pure Rust, with no GPU or Burn dependency. See `docs/architecture.md` §2 and
//! `docs/specs/graph-ir.md`.
