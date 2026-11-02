# Spec: Pipelines and Pipeline Components

**Status:** Draft. Text generation in M3, ASR in M5, image generation in M6. **Crate:** `polypous-pipelines`

A **pipeline** turns a loaded model into a task: messages in → streamed text out; audio in →
transcript out; prompt in → image out. Pipelines run the loops (token generation, denoising,
audio chunking) around graph executions and use **built-in components** for everything that needs
code. `.polyp` files select and configure components by name ([polyp-format.md](polyp-format.md#37-pipeline)).

## 1. Common behavior

- Every pipeline call takes a request (serde struct, JSON-compatible), an event callback and a
  `CancellationToken`, and returns a final result.
- Events are delivered in order, on a Polypous thread ([public-api.md](public-api.md#4-events-and-threading)).
- Cancellation is checked at least once per generated token, diffusion step or audio chunk.
  A cancelled call returns `Cancelled` with any partial result.
- Randomness takes an explicit `seed` (optional; random if absent and reported in the result).

## 2. Task: `text-generation`

**Request:**

| Field | Notes |
|---|---|
| `messages` *or* `prompt` | Chat messages (`role`, `content`) rendered with the chat template, or a raw prompt |
| `max_tokens` | Default from `.polyp` `defaults` |
| `temperature`, `top_k`, `top_p`, `min_p`, `repeat_penalty`, `repeat_last_n`, `seed` | Sampling |
| `stop` | Stop strings. EOS/EOT tokens from model metadata always stop |
| `grammar` / `json_schema` | Later (constrained decoding) |

**Session:** a `TextSession` owns the KV cache (allocated at `max_context` on creation). Across turns
the session reuses the longest common token prefix, so it does not recompute earlier conversation turns.
When the context is full, the policy is `error` (default) or `truncate_oldest` (keeps system message).

**Events:** `PromptProcessed { tokens, elapsed }`, `Token { id, text_delta }`, `Finished { reason:
stop|length|cancelled, usage { prompt_tokens, completion_tokens }, timings }`.

`text_delta` is always valid UTF-8. Tokens that end in the middle of a multi-byte character are held back until the character is complete.

**Components:**

| Component | Kinds |
|---|---|
| `tokenizer` | `auto` (pick from available sources), `hf_json` (`tokenizer.json`), `gguf` (build from GGUF `tokenizer.ggml.*` metadata: `llama` SentencePiece-style and `gpt2` byte-level BPE first). Built on the `tokenizers` crate |
| `chat_template` | `from_model` (GGUF `tokenizer.chat_template` / `tokenizer_config.json`), named built-ins (`chatml`, `llama3`, `mistral`, `gemma`, …) or an inline template string. Rendered with `minijinja` in a sandbox |
| `sampler` | `standard` (the parameters above), `greedy` |

## 3. Task: `speech-recognition` (M5)

**Input:** PCM samples (`f32`, any sample rate, mono or stereo) or encoded audio bytes (WAV, MP3,
FLAC, OGG/Vorbis via pure-Rust decoders).

**Request:** `language` (or auto-detect), `task` (`transcribe` | `translate`), `timestamps`
(`none` | `segment` | `word` later), `initial_prompt`, `temperature` with fallback.

**Events:** `LanguageDetected`, `Segment { start, end, text }`, `Progress { fraction }`, `Finished`.

**Components:** `audio_decode`, `resample` (to 16 kHz), `log_mel` (`n_mels` 80 or 128, Whisper
parameters), `chunker` (30 s windows with timestamp-driven seeking), `tokenizer`, `sampler`.
Graph roles: `encoder`, `decoder` (with cross-attention state).

## 4. Task: `image-generation` (M6)

**Request:** `prompt`, `negative_prompt`, `width`, `height`, `steps`, `guidance_scale`, `seed`,
`scheduler`, optional `init_image` + `strength` (img2img, later).

**Events:** `Step { index, total, preview: Option<Image> }` (low-res preview decoded every N steps if
requested), `Finished`.

**Output:** RGB8 image (`width × height × 3` bytes) + helper to encode PNG.

**Components:** `tokenizer` (CLIP BPE; T5 for later models), `scheduler` (`ddim`, `euler`,
`euler_ancestral`, `dpmpp_2m`, `flow_match_euler` later), `image_codec`.
Graph roles: `text_encoder` (one or more), `denoiser` (UNet or DiT), `vae_decoder`, `vae_encoder` (img2img).

## 5. Task: `graph` (generic)

Runs any graph: named input tensors (`TensorData { dtype, shape, bytes }`) in, named outputs out.
This is what the visual builder uses for inference on custom models. There are no components and no loops.

## 6. Adding components

New component kinds are added **in Rust inside Polypous** (never through `.polyp`). Each component:
has a stable `kind` string, a serde config struct (published in the JSON Schema), unit tests against
a reference (Hugging Face tokenizers, librosa/whisper mel, diffusers schedulers), and an entry in this spec.
