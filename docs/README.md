# Polypous Documentation

## Start here

| Document | Read it to… |
|---|---|
| [project-brief.md](project-brief.md) | understand the vision, users, priorities, goals, non-goals and the v1 target |
| [architecture.md](architecture.md) | see how the crates fit together and how data flows |
| [roadmap.md](roadmap.md) | know what gets built in which milestone, and when a milestone is done |
| [status.md](status.md) | see what exists now, what's verified, known limitations, and where the next milestone starts |
| [glossary.md](glossary.md) | look up terms (block, tier, `.polyp`, session, …) |

## Specifications ([`specs/`](specs/))

| Spec | Covers | Milestone |
|---|---|---|
| [graph-ir.md](specs/graph-ir.md) | The serializable model graph, opsets, validation, versioning | M1, M3 |
| [formats.md](specs/formats.md) | Model file formats: safetensors reader and writer (§1), GGUF (§2, later) | M1, M3 |
| [training.md](specs/training.md) | Declarative training config, catalog, events, checkpoints | M2 |
| [polyp-format.md](specs/polyp-format.md) | `.polyp` architecture definitions | M3 |
| [quantization.md](specs/quantization.md) | GGUF quantized formats and kernels | M3 |
| [pipelines.md](specs/pipelines.md) | Text generation, ASR, image generation, components | M3, M5, M6 |
| [placement.md](specs/placement.md) | Device discovery, GPU → CPU → disk placement, fallback | M0, M4 |
| [public-api.md](specs/public-api.md) | Rust facade, wrapper-friendliness rules, C ABI | M1 → M7 |

Specs are **drafts** until their milestone starts, then refined and kept in sync with the code.

## Decisions ([`adr/`](adr/))

See the [ADR index](adr/README.md).

## Working on Polypous

| Document | For |
|---|---|
| [../CLAUDE.md](../CLAUDE.md) | AI agents (and humans): rules, commands, conventions |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Humans: quick start and PR checklist |
| [development-workflow.md](development-workflow.md) | Tasks, lifecycle, definition of done, parallel agents, lead and worker agents |
| [testing-strategy.md](testing-strategy.md) | Test levels, tolerances, fixtures, backends in CI |
| [dev-environment.md](dev-environment.md) | Setting up Windows/Debian, CUDA, Vulkan, Python |
| [hardware-targets.md](hardware-targets.md) | Target machines, memory budgets, expected placements |
