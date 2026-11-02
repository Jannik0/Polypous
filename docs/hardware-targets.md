# Hardware Targets

Concrete numbers the [placement planner](specs/placement.md) and the acceptance tests are designed
against. All budgets are *usable* memory after OS and other apps, not nominal capacity.

## Target classes

| Class | Example | GPU | Usable VRAM | Usable RAM | Role |
|---|---|---|---|---|---|
| **T1 — Reference** | Razer Blade 15 (2022) | RTX 3060 Laptop (CUDA, Vulkan) | ~5.0 GB of 6 GB | ~10–11 GB of 16 GB | Main dev/test machine |
| **T2 — iGPU laptop** | 16 GB Intel/AMD laptop | Integrated (Vulkan) | shares RAM | ~10–11 GB total, shared | v1 headline target |
| **T3 — CPU only** | Any 16 GB machine, no usable GPU | — | — | ~10–11 GB | Fallback must work |
| **T4 — Apple Silicon** | M1/M2 16 GB | Metal, unified memory | shares RAM | ~10–11 GB total, shared | Best-effort until a Mac is available |

## Memory notes

- **Windows "shared GPU memory" is not VRAM.** On the reference machine Windows reports 6 GB
  dedicated + 7.8 GB shared. The shared part is system RAM reachable over PCIe. The planner counts
  **only dedicated VRAM** for discrete GPUs, and uses RAM through the CPU tier. Counting both would
  use the same RAM twice.
- **Unified-memory devices** (integrated GPUs, Apple Silicon) have *one* memory pool. GPU and CPU
  tiers share a budget.
- **Optimus / dual GPU:** the reference machine exposes both the Intel iGPU and the NVIDIA dGPU over
  Vulkan. The planner must prefer the discrete GPU and must not split a model across both GPUs in v1.

## Model size reference (approximate)

Weights only; KV cache and activations come on top.

| Model | Format | Weight size |
|---|---|---|
| 7B / 8B LLM | Q4_K_M | ~4.1–4.9 GB |
| 7B / 8B LLM | Q8_0 | ~7.2–8.5 GB |
| 7B LLM | F16 | ~13.5 GB |
| 13B LLM | Q4_K_M | ~7.9 GB |
| 30–34B LLM | Q4_K_M | ~19–20 GB |
| Whisper small / base | F16 | ~0.5 GB / ~0.15 GB |
| Stable Diffusion 1.5 | F16 | ~2.0 GB |

## Expected placement on the reference machine (T1)

| Model | Expected placement | Expected use |
|---|---|---|
| 7B Q4_K_M, 4k context | All on GPU, or nearly | Interactive chat |
| 13B Q4_K_M | ~60% GPU, rest CPU | Usable chat, slower |
| 30B+ Q4_K_M | GPU + CPU + disk streaming | Works, slow (seconds per token). Shows that disk streaming works |
| Whisper small | GPU | Faster than real time |
| SD 1.5, 512×512 | GPU | Tens of seconds per image is acceptable |

## Speed expectations (v1, not guarantees)

- 7B Q4_K_M on T1, fully on the GPU: **≥ 10 tokens/s** generation.
- 7B Q4_K_M on T2/T3: **≥ 2 tokens/s** generation.
- Disk-streamed layers: speed is limited by disk bandwidth. The planner reports an estimate before loading.

These numbers are sanity targets, used to catch pathological slowness. They are not optimization goals.
