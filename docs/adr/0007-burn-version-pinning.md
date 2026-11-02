# ADR-0007: Pin Burn and isolate it in one crate

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Burn is pre-1.0, and its API changes between releases. Letting Burn types spread through the codebase would make every upgrade touch everything.

## Decision

- Burn and CubeCL are pinned to an **exact version** (`=x.y.z`) in the workspace `Cargo.toml`. The
  version is chosen at M0 (latest stable release at that time) and recorded in this ADR's follow-up note.
- Only `polypous-backend` and `polypous-quant` may depend on `burn*` / `cubecl*` crates. No Burn type
  appears in the public API of any other crate. This is enforced by a CI check on `Cargo.toml` dependencies.
- An upgrade is a **dedicated task**: bump, fix, run the full reference test suite on CPU, CUDA and
  Vulkan on the reference machine, and record notable changes in `CHANGELOG.md`.
- Upgrades happen deliberately, for example when a Burn release brings something we need, not automatically.

## Consequences

- Upgrades are contained in two crates and checked against the full test suite.
- We may lag behind Burn releases, which is acceptable.

## Follow-up notes

- *(M0: record the chosen Burn/CubeCL versions and Rust toolchain here.)*
