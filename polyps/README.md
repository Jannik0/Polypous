# Official `.polyp` architecture definitions

This directory holds the architecture-family definitions shipped with Polypous. They are embedded into
the binary and are also valid standalone files. Format: [docs/specs/polyp-format.md](../docs/specs/polyp-format.md).

Planned (see [roadmap](../docs/roadmap.md)):

| File | Families | Milestone |
|---|---|---|
| `llama.polyp` | Llama 2/3, Mistral | M3 |
| `qwen2.polyp` | Qwen2 / Qwen2.5 | M3 |
| `gemma2.polyp` | Gemma 2 | M3 |
| `phi3.polyp` | Phi-3 | M3 |
| `whisper.polyp` | Whisper | M5 |
| `sd15.polyp` | Stable Diffusion 1.5 | M6 |
| `sdxl.polyp` | Stable Diffusion XL | M6 |

Every definition here must have a model reference test ([testing strategy](../docs/testing-strategy.md)).
Ids use the reserved `polypous.` prefix. License: MIT.
