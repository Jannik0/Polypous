# ADR-0009: Testing against reference implementations

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Polypous is mostly written by AI agents. Numerical code can look correct and be subtly wrong.
Agents need an objective, automatic way to check their work.

## Decision

Every numerical component is tested against **reference fixtures** produced by established
implementations: PyTorch (ops, gradients, optimizers), Hugging Face transformers/diffusers (models,
tokenizers, schedulers), `gguf-py` (quantization) and llama.cpp (end-to-end GGUF inference).
Fixture generation scripts live in `tools/reference/`. Details and tolerances:
[testing-strategy.md](../testing-strategy.md).

## Consequences

- Python (with PyTorch etc.) is a **development** dependency for generating fixtures, never a runtime or test-time dependency.
- Committed fixtures are small. Large ones are downloaded and verified by hash.
- A change that requires loosening a tolerance must explain why in the PR.
