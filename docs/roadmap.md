# Roadmap

Milestones are ordered by dependency. Each has **exit criteria**. A milestone is done when every
criterion is demonstrated by an automated test or a documented manual check on the reference
machine. Work items for each milestone are tracked as issues or task files; this document only
defines scope.

Legend: **[T1]** reference machine (CUDA + Vulkan), **[CI]** runs in hosted CI (CPU, plus software
Vulkan where possible).

---

## M0 — Foundations

Scope: Cargo workspace, CI, Burn pin, backend selection, device discovery.

- Workspace with the crates from [architecture.md](architecture.md) (empty where not yet needed).
- Burn version pinned ([ADR-0007](adr/0007-burn-version-pinning.md)); `rust-toolchain.toml` pinned.
- `polypous-backend`: `BackendKind { Cuda, Metal, Vulkan, Cpu }`, compiled in via cargo features;
  device discovery that reports name, kind, dedicated memory, unified-memory flag; init failures are
  caught and reported, not fatal.
- `DynTensor` with dynamic rank (1–6) and dtype (f32, f16, bf16, i64, i32, bool).
- `polyp devices` CLI command.
- CI: fmt, clippy, tests on Linux + Windows (CPU), and Vulkan via a software driver (lavapipe) if feasible.

**Exit:** `polyp devices` on T1 lists the RTX 3060 (CUDA + Vulkan), the Intel iGPU (Vulkan) and the
CPU, with correct dedicated VRAM. A simple matmul gives matching results on all of them. Building
without CUDA installed still succeeds and falls back cleanly.

## M1 — Graph IR and interpreter

Scope: [Graph IR spec](specs/graph-ir.md) core opset; interpreter; safetensors.

- `polypous-ir`: types, JSON serde, validation, symbolic shape inference, `repeat` blocks.
- `polypous-backend`: implementations of the core opset.
- `polypous-runtime`: interpreter running on a single device.
- `polypous-formats`: safetensors reader (mmap, sharded).
- Graph + safetensors save/load.

**Exit:** reference tests for every core op pass on CPU **[CI]** and CUDA/Vulkan **[T1]**. An MLP and a
small CNN defined as JSON give the same outputs as identical PyTorch models with the same weights.

## M2 — Training toolkit

Scope: [Training spec](specs/training.md).

- `DynamicModel` implementing Burn `Module`; parameter init from IR specs.
- Declarative training config; SGD, Adam, AdamW; constant / step / cosine / warmup schedules;
  MSE, cross-entropy, BCE losses; accuracy/loss metrics.
- Built-in datasets: in-memory tensors, CSV, image folder.
- Events, cancellation, checkpoints, resume.

**Exit:** MNIST-sized CNN trains to the expected accuracy from a JSON graph + JSON config on CPU and
CUDA. Gradients of every differentiable op match PyTorch. Training resumes correctly from a checkpoint.

## M3 — Pretrained LLMs: GGUF, quantization, text generation

Scope: [Quantization spec](specs/quantization.md), [`.polyp` spec](specs/polyp-format.md),
[Pipelines spec](specs/pipelines.md) (text generation).

- GGUF reader (metadata, tensor infos, tokenizer data, chat template).
- `.polyp` parser, expression evaluator, weight-map resolver; `polyp validate` CLI.
- Quantized kernels: Q8_0, Q4_K, Q5_K, Q6_K (GPU via CubeCL, CPU via SIMD); dequantize-on-load
  fallback for every other GGUF type.
- Transformer opset additions: RoPE variants, RMSNorm, grouped-query attention with KV cache,
  SwiGLU, sliding window.
- `polyps/llama.polyp` (covers Llama 2/3, Mistral), then Qwen2/2.5, Gemma 2, Phi-3.
- Text-generation pipeline: tokenizer from GGUF/`tokenizer.json`, chat template (minijinja),
  samplers, streaming, stop conditions.
- Model runs on a **single device** in this milestone.

**Exit:** a Llama-3-8B Q4_K_M GGUF produces the same greedy tokens as llama.cpp for a fixed prompt
(first N tokens, within documented logit tolerance), on CPU **[CI-nightly]** and CUDA/Vulkan **[T1]**.
The same model loads from HF safetensors (F16).

## M4 — Tiered placement and streaming → **v1 target**

Scope: [Placement spec](specs/placement.md).

- Memory estimation per block; placement planner; pre-load `Estimate`.
- Mixed GPU + CPU execution with activation transfer at boundaries.
- Disk streaming: mmap, background prefetch, bounded staging buffers.
- Fallback on backend init failure and on out-of-memory (re-plan with smaller budget).

**Exit:** all v1 LLM scenarios from the [project brief](project-brief.md#6-v1-target) pass:
7B Q4_K_M interactive on T1 and on a CPU-only run; a 30B+ Q4 model runs on T1 using all three tiers
with correct output; the estimate is within ±25% of the measured peak memory.

## M5 — Speech recognition

- Audio decoding (WAV, plus MP3/FLAC/OGG via pure-Rust decoders), resampling, log-mel spectrogram.
- Whisper `.polyp` (encoder-decoder, cross-attention), timestamps, long-audio chunking, language detection.

**Exit:** Whisper base/small transcripts match the reference implementation (exact text under greedy decoding) on the fixture clips **[T1]**.

## M6 — Image generation

- Diffusion pipeline: CLIP text encoder, UNet, VAE decoder, schedulers (DDIM, Euler, Euler a, DPM++ 2M).
- SD 1.5 `.polyp`, then SDXL, then a DiT/flow model (for example Flux, if feasible on the reference hardware via quantization + streaming).

**Exit:** SD 1.5 at a fixed seed and scheduler gives an image whose latents match the diffusers reference within tolerance **[T1]**.

## M7 — Embedding interface

- `polypous-capi`: stable C ABI with handles, JSON configs, event callbacks, cancellation.
- Generated C header, ABI version, example C program.
- Guide: "Writing a language wrapper" (including a Node.js N-API sketch).

**Exit:** the C example loads a model, streams tokens and cancels generation. ABI checks run in CI.

## Later (unscheduled)

OpenAI-compatible HTTP server crate · quantizing models when loading · LoRA fine-tuning · more
architectures (MoE such as Mixtral/Qwen-MoE, multimodal) · splitting a model across multiple GPUs ·
ONNX import for the builder · speculative decoding.
