# ADR-0012: MPL-2.0 allowed for unmodified dependencies, per crate

- **Status:** Accepted
- **Date:** 2026-11-02
- **Deciders:** project owner
- **Supersedes / Superseded by:** extends [ADR-0010](0010-mit-license.md); does not change it

## Context

[ADR-0010](0010-mit-license.md) allows only MIT-compatible licenses and rules out copyleft (GPL,
LGPL, AGPL). It does not mention MPL-2.0. Burn 0.22.1 ([ADR-0007](0007-burn-version-pinning.md))
pulls in three MPL-2.0 crates that cannot be avoided:

- `colored`: always, through `burn-tensor`'s `std` feature
- `buildid`: with `cuda`, `vulkan` or `metal`, through `cubecl-runtime`
- `option-ext`: with `cuda`, through `dirs-sys`

MPL-2.0 is weak copyleft **per file**: it only requires that changes to MPL-licensed files are
published under MPL-2.0. Code that merely uses an MPL crate is not affected and can stay MIT.

## Decision

- MPL-2.0 dependencies are allowed **if Polypous uses them unmodified** (no forks, no patched copies).
- They are allowed **per crate**, through `exceptions` in `deny.toml`, not by adding MPL-2.0 to the
  general allow list. Every new MPL crate fails `cargo deny` and needs the owner's approval and an
  entry with the reason it is needed.
- GPL, LGPL and AGPL stay disallowed for anything that ships with Polypous (ADR-0010). Tools that
  only run in CI and never ship (for example a GitHub Action) are not covered by these rules.

## Alternatives considered

- **Allow MPL-2.0 globally:** less work, but new MPL crates would arrive unnoticed.
- **Reject MPL-2.0:** would rule out Burn (ADR-0001).

## Consequences

- Before the first release, Polypous needs a third-party license notice (for example generated with
  `cargo-about`). Apps that ship Polypous must include it. For MPL crates this notice also says where
  their source code is available (crates.io).
- Patching an MPL crate (for example with `[patch.crates-io]`) would make that file's changes
  subject to MPL-2.0. Avoid it, or publish the patched source.
