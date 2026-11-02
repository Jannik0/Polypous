# Architecture Decision Records

An ADR records one significant decision: its context, the decision, and its consequences. ADRs
stop settled questions from being reopened, which matters especially when many AI agents work
on the project over time.

## Rules

- One decision per file: `NNNN-short-title.md`, numbered sequentially. Copy [`0000-template.md`](0000-template.md).
- Status: `Proposed` → `Accepted` (by the project owner) → optionally `Superseded by NNNN` or `Deprecated`.
- Accepted ADRs are **not edited** except to change their status or fix typos. To change a
  decision, write a new ADR that supersedes the old one.
- Agents may write `Proposed` ADRs. Only the project owner accepts them.
- If code or a spec contradicts an accepted ADR, the ADR wins until superseded.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-build-on-burn.md) | Build on Burn | Accepted |
| [0002](0002-runtime-graph-ir-interpreter.md) | Runtime graph IR executed by an interpreter | Accepted |
| [0003](0003-runtime-backend-selection.md) | Runtime backend selection via enum dispatch | Accepted |
| [0004](0004-tiered-placement.md) | Tiered GPU → CPU → disk placement | Accepted |
| [0005](0005-native-quantized-kernels.md) | Native kernels for GGUF quantization formats | Accepted |
| [0006](0006-polyp-files-are-data-only.md) | `.polyp` files are data only | Accepted |
| [0007](0007-burn-version-pinning.md) | Pin Burn and isolate it in one crate | Accepted |
| [0008](0008-wrapper-friendly-api.md) | Wrapper-friendly Rust API, C ABI later | Accepted |
| [0009](0009-reference-based-testing.md) | Testing against reference implementations | Accepted |
| [0010](0010-mit-license.md) | MIT license | Accepted |
| [0011](0011-use-burn-dispatch-for-backend-selection.md) | Use Burn's runtime `Dispatch` backend for backend selection | Proposed |
