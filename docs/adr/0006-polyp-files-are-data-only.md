# ADR-0006: `.polyp` files are data only

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

New architecture families should be addable without changing Polypous, through external `.polyp`
files. Those files may be downloaded by non-technical users from untrusted sources.

## Decision

`.polyp` files contain **only data**: graph templates, hyperparameter bindings, a small pure
expression language, weight-name maps with a fixed catalog of transforms, and configuration of built-in
pipeline components. They never contain or reference executable code (no WASM, no scripts, no
native plugins). Anything that needs code is a built-in component implemented in Polypous.
Details: [polyp-format.md](../specs/polyp-format.md).

## Alternatives considered

- **WASM plugins (sandboxed):** flexible, but adds a runtime, increases attack surface and
  makes definitions harder to review and validate statically.
- **Native dynamic libraries:** no stable Rust ABI, and arbitrary code execution. Rejected.

## Consequences

- Loading a malicious `.polyp` can at worst produce a load error, provided the parser limits are enforced.
- Architectures that need genuinely new computation require a Polypous release with a new op or
  component. This is intentional: the op catalog and component library grow centrally and are tested.
- The Graph IR and expression language must be expressive enough for real architectures, and are
  designed against Llama, Whisper and Stable Diffusion from the start.
