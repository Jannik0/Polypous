# Spec: `.polyp` Architecture Definitions

**Status:** Draft. Refined during M3. **Crate:** `polypous-polyp` · **Decision:** [ADR-0006](../adr/0006-polyp-files-are-data-only.md)

A `.polyp` file teaches Polypous an **architecture family** (for example "Llama-style decoder-only
transformers"). Together with a weight file (GGUF or safetensors) it becomes a runnable model.
`.polyp` files are **data only**: they cannot contain or reference executable code. Anything that
needs code (tokenizers, samplers, schedulers, audio processing) is a **built-in pipeline component**
that the file can only select and configure.

## 1. What a `.polyp` contains

| Section | Purpose |
|---|---|
| Header | Identity, version, license, required Polypous and opset versions |
| `match` | Which weight files this definition applies to |
| `hyperparameters` | Named values read from weight-file metadata, with defaults and derived expressions |
| `graphs` | One or more graph templates (Graph IR + `${…}` expressions) |
| `weights` | How tensor names in the weight file map to graph parameters, per source format, with transforms |
| `pipeline` | The task and the built-in components it uses |

A `.polyp` does **not** contain weights, vocabularies or other large assets. Those come from the model files.

## 2. Example (abbreviated)

```json
{
  "format": "polypous.polyp",
  "format_version": 1,
  "id": "polypous.llama",
  "version": "1.0.0",
  "name": "Llama-style decoder-only transformer",
  "description": "Llama 2/3, Mistral, and compatible models.",
  "license": "MIT",
  "requires": { "polypous": ">=0.1.0", "opsets": { "core": 1, "nn": 1, "transformer": 1 } },

  "match": [
    { "source": "gguf", "metadata": { "general.architecture": ["llama", "mistral"] } },
    { "source": "hf",   "config":   { "model_type": ["llama", "mistral"] } }
  ],

  "hyperparameters": {
    "arch":        { "type": "string", "from": { "gguf": "general.architecture" }, "default": "llama" },
    "n_layers":    { "type": "int",   "from": { "gguf": "${arch}.block_count",        "hf": "num_hidden_layers" } },
    "hidden":      { "type": "int",   "from": { "gguf": "${arch}.embedding_length",   "hf": "hidden_size" } },
    "ffn":         { "type": "int",   "from": { "gguf": "${arch}.feed_forward_length","hf": "intermediate_size" } },
    "n_heads":     { "type": "int",   "from": { "gguf": "${arch}.attention.head_count", "hf": "num_attention_heads" } },
    "n_kv_heads":  { "type": "int",   "from": { "gguf": "${arch}.attention.head_count_kv", "hf": "num_key_value_heads" }, "default": "${n_heads}" },
    "head_dim":    { "type": "int",   "expr": "hidden / n_heads" },
    "rms_eps":     { "type": "float", "from": { "gguf": "${arch}.attention.layer_norm_rms_epsilon", "hf": "rms_norm_eps" }, "default": 1e-5 },
    "rope_theta":  { "type": "float", "from": { "gguf": "${arch}.rope.freq_base", "hf": "rope_theta" }, "default": 10000.0 },
    "vocab":       { "type": "int",   "from": { "gguf": "@tokenizer.vocab_size", "hf": "vocab_size" } },
    "max_ctx":     { "type": "int",   "from": { "gguf": "${arch}.context_length", "hf": "max_position_embeddings" } },
    "tied_embeddings": { "type": "bool", "expr": "!has_tensor('output.weight') && !has_tensor('lm_head.weight')" }
  },

  "graphs": {
    "decoder": {
      "format": "polypous.graph", "format_version": 1,
      "opsets": { "core": 1, "nn": 1, "transformer": 1 },
      "dims": { "batch": {}, "seq": {}, "past": {} },
      "inputs": [ { "name": "tokens", "dtype": "i64", "shape": ["batch", "seq"] } ],
      "outputs": [ { "name": "logits", "value": "head" } ],
      "nodes": [
        { "id": "embed", "op": "nn.embedding", "inputs": { "ids": "tokens" }, "params": { "weight": "embed.weight" } },
        { "id": "layers", "op": "repeat", "count": "${n_layers}", "carry": { "h": "embed" },
          "body": { "params_prefix": "layers.{i}.", "...": "attention + SwiGLU block" } },
        { "id": "norm", "op": "nn.rms_norm", "inputs": { "x": "layers:h" }, "params": { "weight": "norm.weight" }, "attrs": { "eps": "${rms_eps}" } },
        { "id": "head", "op": "nn.linear", "inputs": { "x": "norm" },
          "params": { "weight": "${tied_embeddings ? 'embed.weight' : 'head.weight'}" } }
      ]
    }
  },

  "weights": {
    "decoder": [
      { "param": "embed.weight", "gguf": "token_embd.weight", "hf": "model.embed_tokens.weight" },
      { "param": "layers.{i}.attn.q.weight", "gguf": "blk.{i}.attn_q.weight",
        "hf": "model.layers.{i}.self_attn.q_proj.weight",
        "hf_transform": [ { "kind": "rope_permute", "n_heads": "${n_heads}" } ] },
      { "param": "head.weight", "gguf": "output.weight", "hf": "lm_head.weight", "when": "!tied_embeddings" }
    ]
  },

  "pipeline": {
    "task": "text-generation",
    "graph": "decoder",
    "components": {
      "tokenizer":     { "kind": "auto" },
      "chat_template": { "kind": "from_model", "fallback": "chatml" },
      "sampler":       { "kind": "standard" }
    },
    "defaults": { "temperature": 0.7, "top_p": 0.9, "max_context": "${min(max_ctx, 8192)}" }
  }
}
```

## 3. Sections in detail

### 3.1 Header

`format`, `format_version`, `id` (reverse-DNS-like, unique), `version` (semver of the definition),
`name`, `description`, `license`, `authors`, `homepage`, `requires`. Definitions shipped with
Polypous use the `polypous.` id prefix, which third-party files may not use.

### 3.2 `match`

A list of alternatives, any of which may match. Each alternative names a `source` (`gguf`, `hf`, or
`safetensors` with embedded metadata) and conditions on metadata (`gguf`), `config.json` keys
(`hf`) and/or tensor presence (`tensors: ["blk.0.attn_q.weight"]`).

**Resolution:** among all registered definitions that match, pick the one with the most specific
match (most conditions satisfied). If several are equally specific, the error lists them all.
Apps can always choose a definition explicitly by `id`.

### 3.3 `hyperparameters`

Each entry has a `type` (`int`, `float`, `bool`, `string`, `int[]`, `float[]`) and one of:
- `from`: per-source key. Keys may use `${…}` (for example `${arch}.block_count`). `@tokenizer.vocab_size`
  and similar `@`-prefixed keys are derived values that Polypous computes.
- `expr`: an expression over other hyperparameters.
- `default`: used if `from` finds nothing. Without a default, a missing key is an error.

Hyperparameters are evaluated in dependency order, and cycles are an error.

### 3.4 Expressions

A small, pure, total expression language, used in `${…}` inside any string field and in `expr`:

- Literals: integers, floats, booleans, single-quoted strings.
- Operators: `+ - * / // % ** == != < <= > >= && || !`, ternary `c ? a : b`.
- Functions: `min max abs floor ceil round sqrt int float` and `has_tensor(name)`, `has_meta(key)`.
- Names refer to hyperparameters only.

No loops, no user-defined functions, no I/O. Limits are expression length ≤ 4096 chars and nesting
depth ≤ 64. Integer division by zero and overflow are errors, not panics.

A string field that consists of exactly one `${…}` takes the expression's type, so
`"count": "${n_layers}"` becomes an integer.

### 3.5 `graphs`

Named Graph IR templates ([graph-ir.md](graph-ir.md)) with `${…}` allowed in counts, shapes,
attributes and parameter names. Nodes may have `"when": "<expr>"`. When false, the node is
removed and its output becomes an alias of the value named in `"else"` (for example optional QK-norm).

**Instantiation:** expressions are substituted, `when` is resolved, then the resulting plain graph is
validated like any other graph. After instantiation, nothing of the template remains.

### 3.6 `weights`

For each graph, a list of mapping rules:

| Field | Meaning |
|---|---|
| `param` | Graph parameter name (with `{i}` for repeat indices) |
| `gguf` / `hf` / `safetensors` | Tensor name in that source (with `{i}`) |
| `<source>_transform` | Ordered list of built-in transforms applied when loading |
| `when` | Optional condition |
| `optional` | If true, a missing tensor is allowed and the param initializes per its `init` |

**Built-in transforms** (fixed catalog, data-parameterized): `transpose`, `permute{dims}`,
`reshape{shape}`, `split{dim, parts, index}` (for example fused QKV), `concat{sources, dim}`,
`rope_permute{n_heads}`, `scale{factor}`, `add{value}`, `cast{dtype}`.

On **quantized** tensors, only transforms that move whole rows along dim 0 (for example `split` on dim 0,
`rope_permute`) run without dequantizing. Any other transform dequantizes that tensor and logs a
warning with the memory cost.

Load-time checks: every non-optional param resolves to exactly one tensor, shapes match the
instantiated graph, and every tensor in the file is either mapped or listed in `"ignore": [patterns]`.
Unmapped tensors produce a warning, not an error.

### 3.7 `pipeline`

Selects the task and its built-in components ([pipelines.md](pipelines.md)). Component
configuration is data (names, numbers, strings, lists). For multi-graph tasks (ASR:
encoder + decoder; diffusion: text encoder + UNet + VAE) the pipeline maps its roles to the graph names.

## 4. File format rules

- UTF-8 JSON, extension `.polyp`. Fields named `_comment` (any type) are allowed anywhere and ignored.
- Maximum file size 16 MiB. Limits from the Graph IR spec apply after instantiation.
- Unknown fields are errors (except `_comment` and `meta`).
- A JSON Schema for `.polyp` is generated from the Rust types and published in `docs/schemas/`
  (M3), so editors can validate files.

## 5. Distribution and registry

- Official definitions live in `polyps/` and are compiled into the binary.
- Apps can register more files or directories at runtime (`engine.register_polyp(path)`). A
  registered definition with the same `id` and a higher `version` overrides the built-in one, and the
  override is logged.
- `polyp validate <file>` checks a definition without weights. `polyp validate <file> --weights <model>` checks it against a real model file.

## 6. Security model

`.polyp` files and model files may come from untrusted sources. Because they are data only, the worst
case must be a **load error**, never code execution or unbounded resource use. Every limit in this
spec and in the Graph IR spec is enforced before allocating memory proportional to untrusted values.
Chat templates (Jinja) from model files are rendered in minijinja's sandbox with no filesystem
access and with fuel/recursion limits.

## 7. Open questions

- Allow JSONC (`//` comments) for easier hand-authoring?
- Cross-file reuse (`imports`) for shared components such as CLIP or T5 encoders. v1: copy instead.
