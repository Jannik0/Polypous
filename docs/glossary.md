# Glossary

| Term | Meaning |
|---|---|
| **Backend** | A Burn compute backend (CUDA, Metal, Vulkan via wgpu, CPU). In Polypous: a `BackendKind` value chosen at runtime. |
| **Block** (placement block) | The unit the placement planner assigns to a tier. Usually one iteration of a `repeat` (one transformer layer). Also embeddings, the output head, etc. |
| **Compute dtype** | The float dtype (`f32`, `f16` or `bf16`) a loaded model computes in. `float` params are cast to it when loaded, and graph inputs, state and some op attributes may declare `float` to mean it. See [graph-ir.md §2.2](specs/graph-ir.md#22-dtypes). |
| **DynTensor** | Polypous' runtime-typed tensor wrapper over Burn's compile-time typed `Tensor<B, D, K>`: dynamic rank and dtype. |
| **DynamicModel** | A `Module` implementation whose parameters come from a graph at runtime, so Burn optimizers can train it. |
| **Engine** | Top-level public object: owns device discovery, the backend instances and loaded models. |
| **Estimate** | The planner's pre-load prediction of memory per tier and expected speed. |
| **Graph / Graph IR** | Polypous' serializable, runtime-defined network description. See [spec](specs/graph-ir.md). |
| **Hyperparameter binding** | In a `.polyp`, a rule mapping a model-file metadata key (for example GGUF `llama.block_count`) to a named hyperparameter. |
| **KV cache** | Keys/values of previous tokens kept between transformer forward passes. Modeled as explicit graph *state*. |
| **Opset** | The versioned catalog of operations the IR may use (`core`, `nn`, `transformer`, …). |
| **Pipeline** | Task-level orchestration around graphs (text generation, ASR, image generation). See [spec](specs/pipelines.md). |
| **Pipeline component** | A built-in, named, code-implemented building block a pipeline references (tokenizer, sampler, scheduler, mel spectrogram). `.polyp` files configure components but never contain code. |
| **`.polyp`** | A data-only architecture-family definition. See [spec](specs/polyp-format.md). |
| **Placement plan** | Assignment of each block to `Gpu`, `Cpu` or `Disk`. See [spec](specs/placement.md). |
| **Quant format** | A block-wise quantized weight encoding (GGUF `Q4_K`, `Q8_0`, …). See [spec](specs/quantization.md). |
| **Reference fixture** | Inputs + expected outputs produced by an established implementation, used as ground truth in tests. |
| **Session** | A stateful use of a loaded model (for example a conversation with its KV cache). |
| **Streaming (disk)** | Running a block whose weights stay on disk and are read into memory just before use. Not to be confused with **token streaming** (sending output tokens to the app as they are generated). |
| **Symbolic dim** | A tensor dimension given by a name declared in a graph's `dims` (`batch`, `seq`) or an expression over such names, bound to a value on each call. See [graph-ir.md §2.3](specs/graph-ir.md#23-shapes-and-symbolic-dims). |
| **Tier** | One of `Gpu`, `Cpu`, `Disk`. |
| **Unified memory** | GPU and CPU share physical RAM (integrated GPUs, Apple Silicon). The GPU and CPU tiers then share one budget. |
