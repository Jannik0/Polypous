# ADR-0008: Wrapper-friendly Rust API, C ABI later

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

The main users are app developers, often building Electron apps that will wrap Polypous in a
Node.js native addon. Language wrappers are not part of this repository, but writing them must be easy.

## Decision

- The public Rust API follows the rules in [public-api.md §2](../specs/public-api.md#2-wrapper-friendliness-rules-binding-for-all-public-items):
  no generics/lifetimes in public types, `Send + Sync` handles, JSON-compatible plain data,
  blocking calls with callbacks and cancellation, stable error kinds, no panics, no global state.
- A stable **C ABI** (`polypous-capi`) mirroring the facade is delivered in milestone M7.
- No language wrappers live in this repository. A guide for wrapper authors is provided.

## Alternatives considered

- **Async-first Rust API:** forces a runtime choice on embedders and complicates FFI.
- **Ship an official Node addon:** out of scope by the owner's decision. Others can build it on the C ABI or with napi-rs.

## Consequences

- Some idiomatic Rust patterns (borrowed views, generic builders) are avoided in the public API.
- JSON shapes of configs, requests and events become a contract, published as JSON Schema.
