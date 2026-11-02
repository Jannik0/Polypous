# Spec: Graph IR

**Status:** Draft. Refined during M1 (core) and M3 (transformer ops). **Crate:** `polypous-ir`

The Graph IR is Polypous' serializable description of a neural network. Visual builders write it,
`.polyp` files generate it, the trainer trains it, and the runtime executes it. It is **pure data**
with no dependency on Burn.

## 1. Design goals

1. **Editable by tools.** Stable node ids, opaque `meta` that is preserved for UI data (positions,
   colors, comments), and validation errors that point at a node.
2. **Big models stay compact.** `repeat` blocks describe N identical layers once.
3. **State is explicit.** Values that persist between calls (KV cache, recurrent state) are
   declared, not hidden in ops.
4. **Versioned.** Every graph declares its format version and the opset versions it uses. Older
   graphs are migrated forward when loaded.
5. **Checked before running.** Dtypes and (symbolic) shapes are inferred and checked when the graph
   is built, before any computation.

## 2. Document structure (JSON)

```json
{
  "format": "polypous.graph",
  "format_version": 1,
  "opsets": { "core": 1, "nn": 1 },
  "name": "mnist-mlp",
  "meta": { "builder": { "canvas": { "zoom": 1.0 } } },

  "dims": { "batch": { "min": 1 } },

  "inputs":  [ { "name": "image", "dtype": "f32", "shape": ["batch", 784] } ],
  "outputs": [ { "name": "logits", "value": "fc2" } ],

  "params": [
    { "name": "fc1.weight", "dtype": "float", "shape": [128, 784], "init": { "kind": "kaiming_uniform" } },
    { "name": "fc1.bias",   "dtype": "float", "shape": [128],      "init": { "kind": "zeros" } },
    { "name": "fc2.weight", "dtype": "float", "shape": [10, 128],  "init": { "kind": "kaiming_uniform" } },
    { "name": "fc2.bias",   "dtype": "float", "shape": [10],       "init": { "kind": "zeros" } }
  ],

  "nodes": [
    { "id": "fc1",  "op": "nn.linear", "inputs": { "x": "image" },
      "params": { "weight": "fc1.weight", "bias": "fc1.bias" } },
    { "id": "act1", "op": "nn.relu",   "inputs": { "x": "fc1" } },
    { "id": "fc2",  "op": "nn.linear", "inputs": { "x": "act1" },
      "params": { "weight": "fc2.weight", "bias": "fc2.bias" },
      "meta": { "builder": { "x": 420, "y": 80 } } }
  ]
}
```

### 2.1 Values and references

- Graph inputs, state entries and node outputs share **one namespace of values**.
- A node's default output is referred to by its id (`"fc1"`). Ops with several outputs name them,
  and they are referred to as `"node_id:port"` (for example `"split_qkv:q"`).
- Parameter names are separate from value names and may contain dots (`layers.3.attn.q.weight`).

### 2.2 Dtypes

| Dtype | Meaning |
|---|---|
| `f32`, `f16`, `bf16` | Float tensors |
| `i64`, `i32` | Integer tensors (token ids, indices) |
| `bool` | Masks |
| `float` (params only) | "Any float or quantized storage": the actual storage dtype comes from the weight file (`f16`, `q4_k`, …). Compute dtype is chosen by the runtime |

Quantized formats are **storage dtypes of parameters, not value dtypes.** Ops that accept quantized
parameters (`nn.linear`, `nn.embedding`, …) say so in the op catalog. See [quantization.md](quantization.md).

### 2.3 Shapes

A shape is a list of dimensions. Each dimension is:

- an integer literal (`784`)
- a **symbol** declared in `dims` (`"batch"`, `"seq"`, `"past"`), which can bind to different values on each call. A symbol may declare `min` / `max`.
- after shape inference, a symbolic **expression** over symbols (`"past + seq"`). Expressions may
  appear in inferred shapes and in state shapes. Inputs should use plain symbols.

Shape inference propagates symbols through every op, and every node gets an inferred output shape.
The runtime binds symbols to concrete values per call (for example `batch=1, seq=17, past=230`).

### 2.4 Parameters

```json
{ "name": "conv1.weight", "dtype": "float", "shape": [32, 1, 3, 3],
  "init": { "kind": "kaiming_uniform", "a": 2.2360679 }, "trainable": true }
```

`init` kinds (v1): `zeros`, `ones`, `constant{value}`, `uniform{low,high}`, `normal{mean,std}`,
`xavier_uniform`, `xavier_normal`, `kaiming_uniform{a}`, `kaiming_normal`. Fan-in and fan-out are
derived from the op that uses the parameter. Pretrained models ignore `init`: weights come from files.

### 2.5 State

```json
"state": [
  { "name": "k_cache", "dtype": "f16", "shape": ["batch", "n_kv_heads", "max_ctx", "head_dim"], "init": "zeros" }
]
```

A node reads state by referencing its name like any value. A node writes state through
`"state_updates": { "k_cache": "attn:k_cache_out" }` on the graph root (or inside a `repeat` body).
The runtime keeps state per **session** and places it on the device of the block that owns it.

### 2.6 Repeat blocks

```json
{ "id": "layers", "op": "repeat", "count": 32,
  "carry": { "h": "embed" },
  "body": {
    "params_prefix": "layers.{i}.",
    "state_prefix":  "layers.{i}.",
    "nodes": [ "... nodes using value 'h' and params like 'attn_norm.weight' ..." ],
    "carry_out": { "h": "ffn_residual" }
  }
}
```

- The body runs `count` times. Value `h` is passed from one iteration to the next.
- `{i}` is replaced by the iteration index in parameter and state names.
- Each iteration is a **placement block** by default ([placement.md](placement.md)).
- `"repeat:h"` refers to the final carried value.

### 2.7 Subgraphs (functions)

Reusable bodies can be declared once under `"functions": { "name": { inputs, outputs, params, nodes } }`
and used with `{ "op": "call", "function": "name", ... }`. Builders use this for user-defined
"composite layers". There is no recursion.

### 2.8 `meta`

`meta` may appear on the graph, on nodes, params, inputs and outputs. Its content is opaque. Polypous preserves it
through load/save and never interprets it. The `meta.builder` namespace is reserved for visual tools.

## 3. Opsets

Ops are grouped into versioned opsets. Each op definition in `polypous-ir` specifies: named inputs
(with allowed dtypes), named params, attributes (typed, with defaults), outputs, a shape-inference
rule, whether it is differentiable, and whether it accepts quantized params. The **catalog in code is
authoritative**. This list is the planned scope.

### `core` (v1, M1)

Elementwise: `add sub mul div pow neg abs exp log sqrt rsqrt sin cos tanh clamp maximum minimum where`
Comparison: `eq ne lt le gt ge` · Logic: `and or not`
Shape: `reshape flatten permute transpose squeeze unsqueeze expand concat split slice gather index_select pad`
Reduction: `sum mean max min argmax argmin` (with `dims`, `keepdim`)
Linear algebra: `matmul` (batched, broadcasting)
Creation: `constant arange zeros ones full like`
Other: `cast softmax log_softmax cumsum top_k`

### `nn` (v1, M1–M2)

`linear embedding conv1d conv2d conv_transpose2d max_pool2d avg_pool2d adaptive_avg_pool2d
batch_norm layer_norm group_norm rms_norm dropout
relu leaky_relu gelu gelu_tanh silu sigmoid softplus mish
upsample_nearest upsample_bilinear`

### `transformer` (M3)

- `rope`: attrs `style` (`neox` | `interleaved`), `theta`, `scaling` (`none` | `linear` | `ntk` |
  `yarn` | `llama3`, with parameters), `rotary_dim`.
- `attention`: scaled dot-product attention with `n_heads`, `n_kv_heads` (GQA/MQA), `mask`
  (`none` | `causal` | `sliding{window}` | explicit input), `softcap`, `scale`. Optional KV cache
  inputs/outputs plus a `past` position. Cross-attention when K/V come from another input.
- `swiglu`, `geglu`: fused gated activations.
- `moe` (later): router + experts.

### `audio` / `diffusion` (M5–M6)

Added as needed (for example `group_norm` with SiLU fusions, `timestep_embedding`). Preprocessing
such as mel spectrograms is **not** a graph op. It is a pipeline component.

## 4. Validation

Validation runs on load and on every edit through the builder API. Each error has a stable
`code`, a `message`, and a location (`node_id`, optional `input`/`param`/`attr` name, JSON pointer).

Checks:
1. Format and opset versions are supported (otherwise migrate or reject).
2. Ids are unique; every reference resolves; the graph is acyclic.
3. Op exists in the declared opset version; required inputs, params and attrs are present and typed.
4. Dtype inference succeeds.
5. Symbolic shape inference succeeds, and incompatible dims produce an error on the offending node.
6. Every state entry that is updated has a matching shape and dtype.
7. Resource limits (untrusted input): ≤ 1,000,000 nodes after expanding repeats, ≤ 64 nesting
   levels, rank ≤ 8, total declared parameter elements fit in u64.

Errors are collected (not fail-fast), so a builder can show all problems at once.

## 5. Versioning and migration

- `format_version` covers document structure; `opsets` cover op semantics.
- `polypous-ir` can read any older version and migrates it in memory. Saving writes the current version.
- Removing or changing an op's semantics requires a new opset version. The old version stays
  supported, through migration or a legacy implementation, for at least one major Polypous release.
- Unknown top-level or node fields are **errors** (catches typos). The exception is `meta`, which is free-form.

## 6. Serialization

- Canonical form: UTF-8 JSON, keys in a stable order, so diffs are readable.
- A trained or exported model is `model.graph.json` + `model.safetensors` (param names = graph
  param names). A bundled single-file container may be added later.

## 7. Open questions

- Explicit control flow (`if`, `while`) inside graphs. v1: no. Loops live in pipelines.
- A binary encoding for huge graphs. Probably unnecessary because `repeat` keeps graphs small.
