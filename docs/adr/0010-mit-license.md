# ADR-0010: MIT license

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Polypous should be usable by anyone building apps, including commercial ones. Burn is MIT/Apache-2.0.

## Decision

Polypous is licensed under **MIT** for now. Dependencies must have licenses compatible with MIT
distribution (MIT, Apache-2.0, BSD, Zlib, ISC, Unicode-3.0, …). Copyleft dependencies (GPL, LGPL,
AGPL) are not allowed. This is checked by `cargo deny` in CI.

## Consequences

- Dual-licensing as MIT OR Apache-2.0 (the Rust ecosystem convention, which adds a patent grant) may be
  considered later in a superseding ADR.
- Model weights and `.polyp` files contributed by third parties carry their own licenses. Official
  `.polyp` files in `polyps/` are MIT.
