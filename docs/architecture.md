# Architecture

This document describes the planned structure of Polypous. Detailed behavior lives in
[`specs/`](specs/); the reasons behind major choices live in [`adr/`](adr/).

## 1. Layered overview

```
┌───────────────────────────────────────────────────────────────────────────┐
│ Apps (separate projects): visual builder, chat app, language wrappers     │
└───────────────────────────────────────────────────────────────────────────┘
                 │ Rust API                         │ C ABI (M7)
┌────────────────▼─────────────────────────────────▼────────────────────────┐
│ polypous            Facade: public API, sessions, events, errors          │
├───────────────────────────────────────────────────────────────────────────┤
│ polypous-pipelines  Tasks: text-gen, ASR, image-gen; tokenizers, samplers,│
│                     schedulers, audio/image pre/post-processing           │
│ polypous-train      Declarative training loop, optimizers, metrics        │
├───────────────────────────────────────────────────────────────────────────┤
│ polypous-runtime    Graph interpreter, placement planner, layer streaming │
├──────────────────────────────────┬────────────────────────────────────────┤
│ polypous-backend                 │ polypous-quant                         │
│ DynTensor, backend enum, device  │ Quantized tensor types, CubeCL GPU     │
│ discovery, op implementations    │ kernels, SIMD CPU kernels              │
│ ── the ONLY crate using Burn ──  │                                        │
├──────────────────────────────────┴────────────────────────────────────────┤
│ polypous-polyp      .polyp parsing, validation, expressions, weight maps  │
│ polypous-formats    safetensors + GGUF readers (mmap), metadata           │
│ polypous-ir         Graph IR types, validation, shape inference, serde    │
└───────────────────────────────────────────────────────────────────────────┘
                 │
           Burn  /  CubeCL  (CUDA · Metal · Vulkan(wgpu) · CPU)
```

Lower layers never depend on upper layers. `polypous-ir`, `polypous-formats` and `polypous-polyp` are
**pure Rust with no Burn dependency.** They can be tested quickly, without a GPU, and are reusable on
their own (for example, a visual builder can validate graphs without linking a backend).

## 2. Crates

Planned Cargo workspace under `crates/`:

| Crate | Responsibility | Depends on |
|---|---|---|
| `polypous-ir` | Graph IR data model, op catalog (opset), validation, symbolic shape inference, JSON (de)serialization, versioning/migration | — |
| `polypous-formats` | Memory-mapped safetensors (single + sharded) and GGUF readers; tensor metadata; lazy tensor views; GGUF metadata KV access | — |
| `polypous-polyp` | `.polyp` schema, parsing, validation, hyperparameter expression evaluation, match rules, weight-name mapping | `ir`, `formats` |
| `polypous-backend` | Burn integration on `burn::Tensor` + `burn::Device` (Burn's runtime dispatch, ADR-0011): `BackendKind` value type, device discovery/probing, `DynTensor` (dynamic rank + dtype), implementations of every IR op as `DynTensor` functions with plain Rust arguments and no `polypous-ir` types (`polypous-runtime` maps IR nodes onto them), cross-device transfer, autodiff | `burn` |
| `polypous-quant` | Quantized tensor representation, GGUF block formats, dequantization, quantized matmul/matvec kernels (CubeCL for GPUs, SIMD Rust for CPU) | `backend`, `cubecl` |
| `polypous-runtime` | Graph interpreter ([spec](specs/runtime.md)), executable plan compilation, placement planner, weight residency (resident / streamed), prefetching, KV/state management | `ir`, `backend`, `quant`, `formats` |
| `polypous-train` | Training config, training loop, optimizers/schedulers via Burn, metrics, checkpoints, events | `runtime`, `backend` |
| `polypous-pipelines` | Task pipelines and their built-in components (tokenizers, chat templates, samplers, mel spectrogram, diffusion schedulers, image codecs) | `runtime`, `polyp` |
| `polypous` | Public facade: `Engine`, `Model`, `Session`, `Pipeline`, `Trainer`, events, errors | all of the above |
| `polypous-cli` | `polyp` developer CLI: inspect files, validate `.polyp`, probe devices, plan placement, run pipelines, benchmark | `polypous` |
| `polypous-capi` (M7) | Stable C ABI over the facade (handles, JSON configs, callbacks) | `polypous` |

Other top-level directories:

| Path | Content |
|---|---|
| `docs/` | Brief, architecture, specs, ADRs, guides |
| `polyps/` | Official `.polyp` architecture definitions (Llama, Qwen, Whisper, SD 1.5, …), embedded into the binary and also usable as standalone files |
| `tools/reference/` | Python scripts that generate reference fixtures from PyTorch / transformers / gguf-py / llama.cpp |
| `fixtures/` | Small committed test fixtures (`fixtures/small/`) + manifest of large fixtures downloaded into `fixtures/cache/` |

## 3. Core concepts

- **Graph** ([spec](specs/graph-ir.md)): a directed acyclic graph of typed ops with named
  parameters, symbolic shapes and explicit state inputs/outputs (for example the KV cache). It is data
  only and serialized as JSON.
- **`.polyp` definition** ([spec](specs/polyp-format.md)): a data-only package that describes an
  architecture *family*. It contains match rules, hyperparameter bindings, one or more graph templates, a
  weight-name map and a pipeline description.
- **Weights source** ([spec](specs/formats.md)): safetensors or GGUF files. They are memory-mapped and never fully copied into
  RAM unless placement says so.
- **Model** = graph(s) + weights + placement plan, loaded on an `Engine`.
- **Placement plan** ([spec](specs/placement.md)): assigns every *placement block* (usually one
  transformer layer) to `Gpu`, `Cpu` or `Disk`. Disk blocks are streamed.
- **Pipeline** ([spec](specs/pipelines.md)): task-level orchestration around one or more graphs:
  tokenize → loop(graph forward → sample) → detokenize, or text-encode → loop(UNet, scheduler) → VAE.
- **Session:** a stateful use of a model (for example a chat conversation with its KV cache). Many
  sessions can share one loaded model.

## 4. Data flow: loading and running a pretrained model

```
model file(s) ─► polypous-formats: mmap + read metadata
                      │
                      ▼
polypous-polyp: find matching .polyp (match rules) ─► bind hyperparameters from metadata
                      │                                 ─► instantiate graph template(s)
                      ▼                                 ─► resolve weight-name map
polypous-runtime: device discovery (polypous-backend)
                      │
                      ▼
                  placement planner ─► PlacementPlan + Estimate (shown to app before loading)
                      │
                      ▼
                  load: GPU blocks uploaded · CPU blocks copied/mapped · Disk blocks left mapped
                      │
                      ▼
polypous-pipelines: run the task loop; runtime runs blocks in order, transfers activations at tier
                    boundaries and prefetches disk blocks on a background thread
                      │
                      ▼
                  events (progress, tokens, partial transcripts, preview images) ─► app
```

## 5. Data flow: building and training a graph

```
app/builder ─► Graph JSON ─► polypous-ir: validate + infer shapes (errors carry node ids)
                                   │
training config JSON ─► polypous-train: build DynamicModel (parameters initialized per IR init specs)
                                   │       on autodiff backend chosen by device discovery
                                   ▼
                       loop: data batch → forward (interpreter) → loss → backward → optimizer step
                                   │
                                   ▼
                       events (epoch, step, loss, metrics) · checkpoints (graph JSON + safetensors)
```

## 6. Key design decisions

| Decision | ADR |
|---|---|
| Build on Burn, keep Burn out of the public API | [0001](adr/0001-build-on-burn.md), [0007](adr/0007-burn-version-pinning.md) |
| Runtime graph IR, executed by an interpreter | [0002](adr/0002-runtime-graph-ir-interpreter.md) |
| Choose the backend at runtime, using Burn's runtime dispatch | [0003](adr/0003-runtime-backend-selection.md), [0011](adr/0011-use-burn-dispatch-for-backend-selection.md) |
| Tiered GPU → CPU → disk placement | [0004](adr/0004-tiered-placement.md) |
| Native kernels for GGUF quantization formats | [0005](adr/0005-native-quantized-kernels.md) |
| `.polyp` files contain data only | [0006](adr/0006-polyp-files-are-data-only.md) |
| Rust API designed for wrapping, C ABI later | [0008](adr/0008-wrapper-friendly-api.md) |
| Testing against reference implementations | [0009](adr/0009-reference-based-testing.md) |
| MIT license; MPL-2.0 only for unmodified dependencies, per crate | [0010](adr/0010-mit-license.md), [0012](adr/0012-mpl-for-unmodified-dependencies.md) |

## 7. Cross-cutting concerns

- **Errors:** one error type per crate (`thiserror`), mapped at the facade to a stable `ErrorKind`
  enum + human-readable message + optional context (node id, file, tensor name). Errors are
  returned, never panics. A panic in Polypous is a bug.
- **Events and cancellation:** long operations (load, generate, train) emit events through a
  callback/channel and take a `CancellationToken`. See [Public API](specs/public-api.md).
- **Logging:** `tracing` throughout. The facade lets apps attach a subscriber. No `println!`.
- **Threading:** the public API is `Send + Sync`. Heavy work runs on Polypous-managed threads.
  Apps never need an async runtime (async APIs may be added later, on top).
- **Determinism:** with a fixed seed, CPU results are reproducible on the same machine. GPU results may differ within tolerance.
- **Security:** model and `.polyp` files are untrusted input. Parsers check bounds and sizes, and
  limit graph size and expression depth. Untrusted input must never cause unbounded allocation.
