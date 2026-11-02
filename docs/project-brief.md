# Polypous — Project Brief

## 1. Vision

Polypous makes neural networks **easy to build, train and run, reliably, on the hardware people
already own.** It is a layer on top of [Burn](https://github.com/tracel-ai/burn), not a new tensor
library. Burn provides tensors, autodiff and backends. Polypous adds:

1. A **runtime-defined, serializable model graph** that tools such as visual builders can create and
   edit, and that Polypous can train and run.
2. An **inference engine** that runs a model on whatever hardware is available. It splits the model
   across GPU, CPU RAM and disk on its own.
3. A **data-only extension system** (`.polyp` files) that teaches Polypous new pretrained
   architecture families (chat LLMs, speech recognition, image generation) without code changes.

## 2. Users

**Primary:** application developers who build ML tools for **non-technical end users**. Typical
products:

- a visual neural network builder (create → train → export)
- a local chat assistant with speech input and image generation
- a transcription tool

These developers usually ship desktop apps (often Electron) to machines they don't control. They
need Polypous to **pick the hardware, fall back gracefully, report progress, and fail with errors
they can show to a user.**

**Secondary:** Rust developers who want a simple, high-level ML API, and authors of `.polyp`
architecture definitions.

**Follow-up projects** (separate repositories, built on Polypous): a visual NN builder and a
chat/ASR/image-generation app. Polypous must stay generic. Nothing app-specific belongs here.

## 3. Priorities

Order matters. When these conflict, the higher one wins:

1. **Correctness:** outputs match reference implementations within documented tolerances.
2. **Stability and graceful degradation:** a model that can run at all should run, even if slowly.
   No crashes because the hardware differs from the developer's.
3. **Ease of use:** good defaults, few required decisions, clear errors, progress reporting.
4. **Portability:** Windows, Linux, macOS; NVIDIA, AMD, Intel, Apple GPUs; CPU-only machines.
5. **Performance:** good enough for casual use. We do *not* compete with llama.cpp or PyTorch on
   speed, but we avoid needless slowness (for example, quantized weights run natively, not unpacked).

## 4. Goals

### G1 — Graph-based model building and training
- Describe networks as a versioned, serializable graph ([Graph IR spec](specs/graph-ir.md)).
- Validate graphs and infer shapes when the graph is built, with errors that point at the faulty
  node (for UI display).
- Train such graphs with a declarative training configuration ([Training spec](specs/training.md)):
  optimizers, schedules, losses, metrics, checkpoints, progress events, cancellation.
- Export trained models as graph + safetensors weights.

### G2 — Run on any hardware
- Backends: **CUDA, Metal, Vulkan, CPU**. GPU preference: CUDA or Metal first, then Vulkan.
- **Tiered placement** ([Placement spec](specs/placement.md)): fill the GPU first, then CPU RAM,
  then stream the rest from disk. The tiers are combined: a large model can end up partly on each.
- Estimate before loading: memory per tier and expected speed, so apps can warn users.
- Automatic fallback when a backend fails to initialize or runs out of memory.

### G3 — Pretrained transformer models
- Load weights from **safetensors** (including sharded Hugging Face checkpoints) and **GGUF**.
- **Native quantized kernels** for the common GGUF formats ([Quantization spec](specs/quantization.md)),
  so quantized models stay small in memory and fast.
- Architecture families defined in **data-only `.polyp` files** ([`.polyp` spec](specs/polyp-format.md)).
- Ready-to-use **pipelines** for text generation (chat), speech recognition and image generation
  ([Pipelines spec](specs/pipelines.md)).

### G4 — Easy to embed and wrap
- A Rust API designed so bindings to other languages are straightforward
  ([Public API spec](specs/public-api.md)), in particular Node.js/Electron.
- A stable C ABI (milestone M7). Language wrappers themselves are out of scope for this repository.

## 5. Non-goals

- Writing a tensor library or autodiff engine (Burn does this).
- Top-of-the-leaderboard inference speed or training throughput.
- Distributed or multi-node training. Training or fine-tuning large LLMs (LoRA may come later, not v1).
- Executable code in `.polyp` files. They are data only, permanently ([ADR-0006](adr/0006-polyp-files-are-data-only.md)).
- Language wrappers (JS, Python, …) in this repository.
- A GUI. Visual builders and chat apps are separate projects.
- WebAssembly/browser and mobile targets in v1. We don't rule them out, but we don't design for them yet.

## 6. v1 target

> **A 7B-parameter chat model in GGUF Q4_K_M format runs interactively on a laptop with 16 GB RAM
> and only integrated graphics, and also on the reference machine (below), using whichever GPU
> is available plus CPU, with no manual configuration.**

Further acceptance scenarios for v1:

- **Disk streaming:** a model that does *not* fit in VRAM + RAM (for example a 30B+ Q4 model on the
  reference machine) loads and produces correct output, with streamed layers and a pre-load speed
  estimate.
- **CPU-only:** the same 7B model runs with no usable GPU (GPU drivers missing or disabled).
- **Builder workflow:** an MLP and a small CNN can be defined as a graph, trained on a small
  dataset with progress events, saved, reloaded and run for inference.
- **Whisper** (small/base) transcribes audio, and **Stable Diffusion 1.5** generates an image,
  on the reference machine.

See [hardware-targets.md](hardware-targets.md) for concrete numbers.

## 7. Reference hardware

The primary development and test machine:

| Component | Spec |
|---|---|
| Machine | Razer Blade 15 (2022) |
| GPU | NVIDIA RTX 3060 Laptop, **6 GB dedicated VRAM** (Windows also reports 7.8 GB "shared" GPU memory, which is system RAM) |
| CPU | Intel (12th gen class) with integrated graphics (NVIDIA Optimus) |
| RAM | 16 GB |
| OS | Dual boot: Windows 11 and Debian |

What this means for us: CUDA and Vulkan get tested regularly. **Metal cannot be tested yet** (no Mac),
so Metal support is best-effort until a Mac is available. The machine has two GPUs (Intel iGPU +
NVIDIA dGPU), so device selection gets exercised for real.

## 8. How the project is developed

Polypous is developed **mainly by AI agents**, with a human owner who makes the decisions. That shapes
the repository:

- **Specs before code.** Behavior is defined in `docs/specs/` before it is implemented. Code that
  disagrees with a spec is a bug in one of them, and the spec gets updated in the same change.
- **Decisions are recorded** as ADRs in `docs/adr/` so they are not reopened by accident.
- **Reference-based testing.** Every numerical component is checked against outputs from
  established implementations (PyTorch, Hugging Face, llama.cpp/ggml). Agents check their own
  work against these fixtures ([Testing strategy](testing-strategy.md)).
- **Small crates, clear boundaries,** so several agents can work in parallel without conflicts.

## 9. Licensing

Polypous is MIT-licensed. Burn is MIT/Apache-2.0. Model weights carry their own licenses. Polypous
reports license metadata from model files where available but does not enforce it.

## 10. Open questions

Tracked here until resolved by an ADR:

- **Q1** Whether to ship a built-in OpenAI-compatible HTTP server (`polypous-server`) or leave it to apps. Leaning: optional crate after M7.
- **Q2** Whether disk-streamed layers run on the CPU (simple) or are uploaded to the GPU for
  compute (faster once disk reads are the bottleneck). v1 plan: CPU, revisit after M4.
- **Q3** Whether `.polyp` files may reference other `.polyp` files (for example, SDXL reusing a CLIP
  text encoder definition). v1 plan: no, each file is self-contained.
- **Q4** Quantizing fp16 safetensors models when loading them so they fit on small machines. Leaning: post-v1.
- **Q5** LoRA / fine-tuning support for pretrained models. Post-v1.
