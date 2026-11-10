# Spec: Graph IR

**Status:** Binding for M1: §2 (all), §3.1–§3.3 (general rules and the M1 op catalog), §4, §5 and §6.
Draft: §3.4 (planned ops) and §7. The `transformer` opset is refined in M3. **Crate:** `polypous-ir`

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

  "inputs":  [ { "name": "image", "dtype": "float", "shape": ["batch", 784] } ],
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

### 2.1 Nodes, values and references

A node is a JSON object with these fields:

| Field | Required | Content |
|---|---|---|
| `id` | yes | The node's name in the value namespace |
| `op` | yes | Full op name, `opset.op_name` (`nn.linear`), or `repeat` (§2.6) |
| `inputs` | if the op has inputs | Op input name → value reference. A variadic input (only `core.concat` in M1) takes a JSON array of references |
| `params` | if the op has params | Op param name → graph param name |
| `attrs` | if the op has attributes | Attribute name → value (§3.1.7) |
| `meta` | no | Opaque (§2.8) |

Optional inputs, params and attrs may be left out. A name the op doesn't define is an error.

- Graph inputs, state entries and node outputs share **one namespace of values**.
- Every op output is a named **port**. An op with one output has the single port `out`. A reference
  is either `"node_id:port"` or a bare `"node_id"`, which means `"node_id:out"`. A bare id of a
  node that has no `out` port (a multi-output op) is an error. Graph inputs and state entries are
  referred to by their name only. Port names of multi-output ops are listed in the op tables (§3).
- Parameter names are separate from value names and may contain dots (`layers.3.attn.q.weight`).
- **Names.** Node ids, graph input, output and state names: 1–128 bytes of ASCII letters, digits,
  `_`, `-` and `.`, never `:`. Parameter names: 1–256 bytes of the same characters. `{i}` may
  appear only in `params_prefix` and `state_prefix` (§2.6), which are up to 256 bytes of these
  characters and `{i}`; body param references themselves are plain parameter names. Dim symbols:
  see §2.3.
- `outputs` entries are `{ "name": …, "value": <reference> }`. Output names are their own
  namespace and must be unique.

### 2.2 Dtypes

| Dtype | Meaning |
|---|---|
| `f32`, `f16`, `bf16` | Float tensors |
| `i64`, `i32` | Integer tensors (token ids, indices) |
| `bool` | Masks |
| `float` | On params: "any float or quantized storage". The actual storage dtype comes from the weight file (`f16`, `q4_k`, …), and ops see the param in the **compute dtype**. On graph inputs, state entries and some `dtype` attributes: the compute dtype itself (see below) |

Value tensors have **rank 1 to 6**. There is no rank 0: a scalar is a tensor of shape `[1]` (§3.1.5).

**Compute dtype.** Each loaded model has a compute dtype: `f32` by default, `f16` or `bf16` where
the device supports them. The runtime casts every `float` param to it when it loads the param.
Validation takes the compute dtype as an input (default `f32`) and types every `float` param with
it.

**`float` as a value dtype.** Graph inputs and state entries declare their dtype. Besides the
concrete dtypes they may declare `"float"`, which means the compute dtype. So may the `dtype`
attribute of `core.cast`, `core.constant`, `core.zeros`, `core.ones`, `core.full`,
`core.full_like` and `core.arange`. With it, one graph is valid for every compute dtype:
- Validation resolves every `"float"` to the compute dtype it is given, before dtype inference.
  The strict rule of §3.1.2 is unchanged: a `float` input has exactly the compute dtype. A graph
  whose inputs declare a concrete `f32` still validates only with compute dtype `f32`, unless it
  casts.
- `"float"` is not allowed anywhere else. In particular, index dtypes (`dtype` of `core.argmax`,
  `core.argmin`, `core.top_k`) must be `I`. Elsewhere it is `invalid_attr` (attributes) or
  `invalid_value` (other fields).
- Scalar values of `core.constant`, `core.full` and `core.full_like` with dtype `float` are rounded
  to the compute dtype.
- The runtime checks given input and state values against the resolved dtype (`InvalidInput` on a
  mismatch, §2.3.4).
- Saving keeps `"float"` as written; it is never replaced by the resolved dtype.

**Device support.** Not every device supports every dtype. On Vulkan, for example, `bf16` depends on
device extensions (Mesa lavapipe has none), and `i64` on `shaderInt64`. The backend reports support
per device, and per op where an op lacks an implementation for a dtype on a backend. The rule:

- The graph is valid independently of any device. Device support is checked by the runtime.
- When a graph is prepared for a device, and at the latest before the first node of a call runs,
  the runtime checks every value dtype and every (op, dtype) pair of the graph against the device,
  and the compute dtype against the device.
- A failure is a `polypous::Error` with kind `UnsupportedOperation`
  ([public-api.md §2](public-api.md#2-wrapper-friendliness-rules-binding-for-all-public-items), rule 7).
  Its context names the node, op, dtype and device. Nothing has run on the device yet.
- The runtime never converts a dtype silently. It casts only where the graph has `core.cast` or
  where the compute dtype rule above applies.

Quantized formats are **storage dtypes of parameters, not value dtypes.** No M1 op accepts quantized
params. `nn.linear` and `nn.embedding` will in M3 ([quantization.md](quantization.md)).

### 2.3 Shapes and symbolic dims

A shape is a list of 1–6 dimensions. Each dimension is:

- an integer literal ≥ 1 (`784`)
- a **symbol** declared in `dims` (`"batch"`, `"seq"`), which binds to a value on each call. A
  symbol may declare `min` (default 1, must be ≥ 1) and `max` (≥ `min`, no default).
- a **dim expression** over symbols (`"past + seq"`), §2.3.1.

Where dims may appear:

| Place | Allowed |
|---|---|
| Graph input shapes | Literals and plain symbols. This is where symbols get bound |
| State shapes | Literals, symbols and expressions |
| Param shapes | Literals only |
| Dim attributes of ops (`reshape`, `slice`, `pad`, `full`, …) | Literals and expressions (§3.1.7) |
| Inferred shapes | Anything shape inference produces |

Every symbol used anywhere must appear as a plain symbol in some graph input or state shape, so
that the runtime can bind it. Symbols declared in `dims` but never used are allowed.

**Zero-size dims are not allowed in M1.** Every literal is ≥ 1, every symbol binds to ≥ 1, and every
inferred dim must evaluate to ≥ 1. PyTorch allows empty tensors, but Burn 0.22.1's support for them
across backends is unverified and M1 models don't need them. Relaxing this later is backward
compatible.

#### 2.3.1 Grammar

Dim expressions are JSON strings. Whitespace between tokens is ignored.

```
expr   = term   { ("+" | "-") term }
term   = factor { ("*" | "//" | "%") factor }
factor = INT | SYMBOL | "(" expr ")" | ("min" | "max") "(" expr "," expr ")"
INT    = digit { digit }                       (decimal, fits in i64)
SYMBOL = (letter | "_") { letter | digit | "_" }   (1–64 bytes, declared in `dims`, not `min`/`max`)
```

- `+ - *` are integer arithmetic. `//` is **floor** division and `%` the matching remainder
  (`a % k = a - k * (a // k)`, always in `[0, k)`), as in Python.
- The right operand of `//` and `%` must normalize to an integer literal ≥ 1. Division by a symbol
  is not allowed.
- There is no unary minus.

Why these operators: `+` and `-` for concatenated or sliced dims (`past + seq`), `*` for
`flatten`/`reshape` (`batch * seq`), `//` for conv/pool output sizes
(`(H + 2*p - d*(k - 1) - 1) // s + 1`) and `split` sizes, `%` for completeness of `//`, and
`min`/`max` for `slice`, where PyTorch clamps the bounds to the dim size. Divisors are literals
because conv/pool strides are literal attributes, and a literal divisor keeps the normal form simple.

#### 2.3.2 Normal form

Every dim normalizes to a **sum of terms** `c * m`: an integer coefficient `c ≠ 0` times a
monomial `m`, which is a product of zero or more **atoms** (a monomial with no atoms is the
constant term). An atom is a symbol or an opaque term `(e) // k`, `min(e1, e2)` or `max(e1, e2)`
whose arguments are themselves in normal form. Normalization works bottom-up:

1. `+`, `-`, `*`: expand products, collect equal monomials, drop zero coefficients. Literal-only
   subexpressions fold to a literal.
2. `e // k`: if `e` is a literal, fold it. If `k = 1`, the result is `e`. Otherwise split every
   coefficient (and the constant) of `e` as `c = k*q + r` with `0 ≤ r < k`. Then
   `e // k = Q + (R // k)`, where `Q` collects the `q` parts and `R` the `r` parts. `R // k` folds
   if `R` is a literal and is otherwise an opaque atom. So `(2*seq + 1) // 2 = seq`.
3. `e % k` normalizes as `e - k * (e // k)`.
4. `min(a, b)`, `max(a, b)`: if `a - b` normalizes to a literal, pick the smaller or larger
   argument. Otherwise it is an opaque atom whose two arguments are sorted by their canonical string.

**Canonical order and string.** Atoms in a monomial are sorted by their canonical string (symbols
by name, byte order). Terms are sorted by degree (descending), then by monomial string (the atoms'
strings joined with `*`, without the coefficient), with the constant last.
The canonical string uses the grammar's syntax: `past + 2*seq - 1`, `(H + 1) // 2`,
`min(10, seq)`. It parses back to the same normal form. Error messages and inferred shapes print it.
- Coefficients `±1` are not written; other coefficients are written as `c*m`.
- An opaque `//` atom is written in parentheses when it is multiplied by a coefficient or another
  atom (`2*((H + 1) // 2)`), because `*` and `//` bind equally and associate to the left. Its
  dividend is written in parentheses unless it is a single symbol or `min`/`max` (`H // 2`).
- There is no unary minus, so a normal form whose first term is negative starts with `0 - `
  (`0 - seq + 1` for `1 - seq`). The zero dim is `0`. A coefficient of `i64::MIN` is `dim_overflow`,
  so every coefficient's absolute value is an `i64` and the string parses back.

#### 2.3.3 Equality

Two dims are **equal** when their normal forms are identical. This is sound but not complete:
`(seq // 2) * 2` and `seq` are not provably equal even when `seq` is always even. Wherever an op
requires equal dims (broadcasting, `concat`, `matmul`, param shapes, `reshape` element counts,
`repeat` carries, state updates), "not provably equal" is a validation error on that node. Graphs
fix this with an explicit `reshape` or by using the same expression.

#### 2.3.4 What validation proves and what the runtime checks

Validation proves:
- every node's output dtype and shape (as normal forms), and the equalities each op requires
- every literal or literal-folded dim is ≥ 1, and every literal bound (`top_k` `k`, `slice`
  bounds) is in range when both sides are literals
- every used symbol can be bound (§2.3)

On each call, before any node runs, the runtime:
1. Checks every graph input and state value: present, declared dtype (`float` resolved to the
   compute dtype, §2.2), declared rank.
2. Binds symbols: a plain-symbol dim of an input or state value (inputs in declaration order, then
   state, dims left to right) binds the symbol on its first occurrence. Every later occurrence must
   have the same value. Literal dims must match exactly. State means every root-level entry after
   expanding repeats: root state in declaration order, then body state depth-first in node order,
   iteration by iteration (`layers.0.k`, `layers.0.v`, `layers.1.k`, …).
3. Checks every bound value against the symbol's declared `min`/`max`.
4. Evaluates the expression dims of state shapes and compares them with the given state values.
5. Evaluates every inferred dim and every dim attribute of every node with checked `i64`
   arithmetic. Each shape dim must be ≥ 1, each attribute must be in its range (for example
   `slice` `start ≥ 0`, `top_k` `k ≤` the dim size), and each value's element count must fit
   in `u64` and `usize`.

Any failure is an `InvalidInput` error naming the input, symbol or node, and nothing runs. After
these checks no op can fail on shapes, so backend ops only fail on values (out-of-range indices,
§3.1.8) or device errors.

#### 2.3.5 Limits

Dim expressions come from untrusted documents:
- an expression string is at most 256 bytes, and its parentheses and `min`/`max` calls nest at
  most 16 deep
- a normal form has at most 64 terms, and each monomial has degree at most 8
- literals, coefficients and all intermediate values are `i64`, with checked arithmetic. Overflow is
  the validation error `dim_overflow`, or `InvalidInput` when it happens while the runtime
  evaluates bound dims.

### 2.4 Parameters

```json
{ "name": "conv1.weight", "dtype": "float", "shape": [32, 1, 3, 3],
  "init": { "kind": "kaiming_uniform", "a": 2.2360679 }, "trainable": true }
```

- In M1, `dtype` must be `float` and `shape` must be literals (rank 1–6). Other dtypes are reserved
  (`unsupported_in_version`).
- A param may be used by several nodes (weight tying). An unused param is not an error.
- `init` kinds (v1): `zeros`, `ones`, `constant{value}`, `uniform{low,high}`, `normal{mean,std}`,
  `xavier_uniform`, `xavier_normal`, `kaiming_uniform{a}`, `kaiming_normal`. Fan-in and fan-out are
  derived from the op that uses the parameter. Pretrained models ignore `init`: weights come from
  files. M1 parses and validates `init` but doesn't apply it; that is M2 (training).
- `init` fields: `constant.value`, `uniform.low`/`high` and `normal.mean`/`std` are required,
  `kaiming_uniform.a` is optional (default as in `torch.nn.init.kaiming_uniform_`, `0`). All are
  JSON numbers and must be finite. Other fields are `unknown_field`. `init` itself is optional.
- `trainable` defaults to `true`. It has no effect before M2.

### 2.5 State

```json
"state": [
  { "name": "k_cache", "dtype": "f16", "shape": ["batch", "n_kv_heads", "max_ctx", "head_dim"], "init": "zeros" }
]
```

A node reads state by referencing its name like any value. A node writes state through
`"state_updates": { "k_cache": "attn:k_cache_out" }` on the graph root (or inside a `repeat` body).
The runtime keeps state per **session** and places it on the device of the block that owns it.

**What M1 implements:**
- `state` entries on the graph root and in `repeat` bodies (§2.6), with a concrete value dtype
  or `float` (the compute dtype, §2.2), and `init: "zeros"` (the only init kind in M1).
- `state_updates`: each state entry is updated at most once per call (JSON object keys are unique,
  §4.2 `invalid_json`). Root `state_updates` target root state entries; a body's `state_updates`
  target the body's own state entries, by their local names. Any other target is
  `unknown_reference`. The update's dtype must equal
  the entry's dtype and its shape must be provably equal to the entry's shape. Nodes that read a
  state entry during a call see the value from the start of the call; the update takes effect for
  the next call.
- **Until sessions exist (M3)**, the interpreter has no state of its own. A call takes the state
  values as extra inputs, keyed by state name next to the graph inputs, and returns every state
  entry as an extra output in a separate map keyed by state name: the updated value, or the input
  value unchanged if the entry has no update. A missing state value is `InvalidInput`. The caller
  creates the initial values (zeros of the declared dtype and shape).

### 2.6 Repeat blocks

```json
{ "id": "layers", "op": "repeat", "count": 32,
  "carry": { "h": "embed" },
  "body": {
    "params_prefix": "layers.{i}.",
    "state_prefix":  "layers.{i}.",
    "state": [ "... optional state entries, named without the prefix ..." ],
    "nodes": [ "... nodes using value 'h' and params like 'attn_norm.weight' ..." ],
    "state_updates": { "... body-local state name": "body value reference" },
    "carry_out": { "h": "ffn_residual" }
  }
}
```

- The body runs `count` times (an integer literal, 1–65,536). Value `h` is passed from one
  iteration to the next. `carry` must have at least one entry, and `carry_out` must have exactly the
  same keys. For each key, the `carry_out` value must have the same dtype as the `carry` value and a
  provably equal shape.
- `{i}` in `params_prefix` and `state_prefix` is replaced by the iteration index (`0`…`count-1`).
  A body param name `n` refers to the graph param `params_prefix(i) + n`. Without `{i}`, all
  iterations share the params. Body `state` entries become root-level state entries named
  `state_prefix(i) + n`, so `state_prefix` must contain `{i}` if the body declares state.
- Scoping: body nodes may reference the carry names, other body nodes, and every value of the
  enclosing scope (graph inputs, state, nodes of the enclosing scope that don't depend on this
  repeat). Body ids, carry names and body state names must not reuse a name of any enclosing scope
  (`duplicate_name`), so a name never shadows another. Outside the body, only
  `"<repeat id>:<carry name>"` refers to the final carried value; the repeat node has no `out` port.
- Repeats may nest (at most 8 levels). Prefixes of nested repeats concatenate, outer first.
- Validation infers a body **once**, for all iterations. So a body param must have the same shape
  in every iteration: if `params_prefix(i) + n` differs in shape between iterations, the node is
  `shape_mismatch` (naming the param and the iteration).
- Expanded names follow the naming rules: `params_prefix(i) + n` is a parameter name (≤ 256 bytes)
  and must be declared in `params` for every iteration, and `state_prefix(i) + n` is a state name
  (≤ 128 bytes) that must be unique among all root-level state names after expansion.
- Each iteration is a **placement block** by default ([placement.md](placement.md)). In M1 the
  interpreter runs on a single device, so this has no effect yet.

### 2.7 Subgraphs (functions)

Reusable bodies can be declared once under `"functions": { "name": { inputs, outputs, params, nodes } }`
and used with `{ "op": "call", "function": "name", ... }`. Builders use this for user-defined
"composite layers". There is no recursion.

**Not in M1.** Functions arrive in M2 with the builder's composite layers. In M1 a `functions` field
on the graph or a node with `op: "call"` is the validation error `unsupported_in_version`.

### 2.8 `meta`

`meta` may appear on the graph, on nodes, params, inputs and outputs. Its content is opaque (any JSON
value). Polypous preserves it through load/save and never interprets it. The `meta.builder` namespace
is reserved for visual tools. `meta` is subject to the JSON limits of §4.3.

### 2.9 Fields

Every object of the document has exactly these fields; any other field is `unknown_field`
(outside `meta`), a missing required field is `missing_field`.

| Object | Required | Optional |
|---|---|---|
| Graph | `format`, `format_version`, `opsets`, `outputs` (at least one entry), `nodes` | `name`, `meta`, `dims`, `inputs`, `params`, `state`, `state_updates` |
| `dims` entry | — | `min`, `max` |
| Input | `name`, `dtype`, `shape` | `meta` |
| Output | `name`, `value` | `meta` |
| Param | `name`, `dtype`, `shape` | `init`, `trainable`, `meta` |
| State entry | `name`, `dtype`, `shape` | `init` (default `"zeros"`) |
| Op node | `id`, `op` | `inputs`, `params`, `attrs`, `meta` |
| `repeat` node | `id`, `op`, `count`, `carry`, `body` | `meta` |
| `repeat` body | `nodes`, `carry_out` | `params_prefix`, `state_prefix` (both default `""`), `state`, `state_updates` |

`format` and `format_version` are checked first; if either is missing or unsupported, nothing else
is checked. `functions` on the graph is `unsupported_in_version` (§2.7), not `unknown_field`.

## 3. Opsets

Ops are grouped into versioned opsets. **The tables in §3.2 and §3.3 are the M1 op catalog.** The
catalog in `polypous-ir` implements exactly these rows, and a test checks that its op names match the
first column of these tables (#39). Each M1 op is one table row whose first cell is the full op name
in backticks. Ops in §3.4 are planned and not part of M1.

### 3.1 General rules

These rules apply to every op unless its row or note says otherwise.

#### 3.1.1 Notation in the tables

- **Dtype classes:** `F` = `f32 f16 bf16`, `I` = `i64 i32`, `B` = `bool`, `N` = `F` or `I`,
  `A` = any dtype. `T` means "the same dtype as `x`" (or `a`). `C` is the compute dtype (§2.2).
- `x.shape = [d0, …, d(r-1)]`, `r` = rank of `x`.
- Attributes are written `name: type = default`. No default means required. `?` means optional
  with no default.
- `F.` is `torch.nn.functional.`.
- **Differentiable** "yes" means: with respect to every float input and every param. Integer and
  bool inputs are never differentiable. Gradients are M2; this column fixes which ops get them.

#### 3.1.2 Dtypes (strict)

All inputs (and params) of an op that must agree have **the same dtype**. There is no implicit
promotion: `add(f16, f32)` and `linear(f32 x, f16 weight)` are `dtype_mismatch` errors. Graphs
convert with `core.cast`. Index inputs (`index`, `ids`) are an exception: they accept `i32` or
`i64` regardless of the other inputs. An input whose dtype is outside its class is
`unsupported_dtype`.

The output dtype is the input dtype unless the table says otherwise. In particular, `core.sum` and
`core.cumsum` of `i32` give `i32` (PyTorch gives `i64`; fixtures cast the reference).

#### 3.1.3 Broadcasting

Ops marked "broadcast" use PyTorch/NumPy rules: shapes are right-aligned, the shorter one is padded
on the left with `1`, and each pair of dims `(p, q)` combines as:

| `p` | `q` | Result |
|---|---|---|
| provably equal to `q` | | `p` |
| literal `1` | anything | `q` |
| anything | literal `1` | `p` |
| otherwise | | `broadcast_mismatch` error on the node |

A symbol broadcasts only against a literal `1` or a provably equal dim. A symbol is never treated as
`1`, even if it may bind to `1`, and a symbol never broadcasts against a different literal, even if
it may bind to it. Because validation proves every pair, broadcasting needs no runtime shape checks.
Three-input `core.where` broadcasts all three shapes the same way. The result rank is the largest
input rank.

#### 3.1.4 Axes

- Axis attributes (`dim`, `dims`, `start_dim`, …) use PyTorch's names and meaning.
- Negative axes are allowed and count from the end: `-1` is the last dim. For an input of rank `r`
  the valid range is `[-r, r-1]` (`core.unsqueeze`: `[-(r+1), r]`, because it refers to the output).
- Validation normalizes axes to `0…r-1` internally. Saving writes the value as written.
- An axis outside the range is `invalid_attr`. An axis list with a duplicate (after normalization)
  is `invalid_attr`.

#### 3.1.5 Rank and scalars

- Every input and output has rank 1–6. An op whose output rank would exceed 6 (`unsqueeze` of a
  rank-6 value, `embedding` of rank-6 `ids`) is `rank_too_large`.
- There is no rank 0. A **scalar** is shape `[1]`. Where PyTorch returns a 0-dim tensor (a full
  reduction with `keepdim=False`, `squeeze` of `[1]`), the result here is shape `[1]` with the same
  value, and fixtures compare it that way.
- To combine a scalar with a tensor, use broadcasting: `[1]` broadcasts against anything.

#### 3.1.6 Output ports

- Single-output ops have the port `out`.
- `core.split` with `n` sizes has ports `0`, `1`, …, `n-1` (decimal, no leading zeros):
  `"qkv:0"`, `"qkv:1"`, `"qkv:2"`.
- `core.top_k` has ports `values` and `indices`.
- `repeat` has one port per carry name (§2.6).

#### 3.1.7 Attribute types

| Type | JSON | Notes |
|---|---|---|
| `int` | integer | `i64` |
| `float` | number, or `"inf"`, `"-inf"`, `"nan"` | `f64`. Strings are needed because JSON has no infinities (masks use `-inf`) |
| `bool` | `true`/`false` | |
| `enum` | string | One of the values the op lists |
| `int[]` | array of integers | Length rules per op |
| `dim` | integer, or string | A dim expression (§2.3.1). Its symbols must be declared in `dims` |
| `dim[]` | array of `dim` | |
| `dtype` | string | `f32 f16 bf16 i64 i32 bool`, and `float` (the compute dtype, §2.2) where the op says so |
| `scalar` | number, `"inf"`/`"-inf"`/`"nan"`, or `true`/`false` | A value of the dtype the op is filling: an integer for `I` (in range for the dtype), a `float` for `F` (rounded to the dtype), a bool for `B` |

**Rounding fill values.** A `float` that fills an `F` tensor (`core.constant`, `core.full`,
`core.full_like`, `core.pad`) rounds to nearest, ties to even, first to `f32` and then to `f16` or
`bf16`, as PyTorch rounds Python floats (`torch.full`, `torch.tensor`). A finite value beyond the
dtype's range becomes `±inf` (IEEE 754).

**Shape-like arguments are attributes** (reshape target, `expand`, `slice` bounds, `pad`, `arange`,
`zeros`/`ones`/`full` shape, `top_k` `k`, `split` sizes). They are `dim` or `dim[]` attributes,
never tensor inputs, so no output shape depends on tensor values. An attribute refers to a `dims`
symbol by using its name in a dim expression: `{ "shape": ["batch * seq", 64] }`.

The rule "every dim ≥ 1" of §2.3 is for shapes. A `dim` attribute has the range its op states
(`pad` entries may be `0`, `reshape` and `expand` accept the literal `-1`, `slice` bounds may be
negative literals).

A wrong JSON type, a value outside the listed range, an unknown enum value or a wrong list length is
`invalid_attr`. More precisely, for the codes of §4.2:
- An attribute outside its own range (`pad` entry `< 0`, `top_k` `k < 1`, a `reshape` entry `0`) is
  `invalid_attr`. A literal bound that is out of range for an input dim (`top_k` `k` above the dim
  size, a negative `slice` bound below `-n`) is `shape_mismatch`. An inferred dim that folds to a
  literal `≤ 0` (an empty `slice`, a conv output) is `nonpositive_dim`.
- A `dim` attribute that uses an undeclared symbol is `invalid_dim_expr`; one that uses a declared
  symbol that no input or state shape binds is `unbound_dim`.
- A `dim` attribute with symbols whose range validation can't decide (for example `slice`
  `start ≥ 0` for `"seq - 4"`) is checked by the runtime after binding (§2.3.4 step 5).
- A missing required attribute is `missing_attr`, also where a rule asks for one of several
  (`clamp` without `min` and `max` reports `min`).

#### 3.1.8 Numerics

- Float ops follow IEEE 754 (with the backend's rounding of transcendental functions):
  `log(-1)` and `sqrt(-1)` are NaN, `rsqrt(0)` is `inf`, NaN propagates through arithmetic.
- **Integer overflow wraps** (two's complement), like PyTorch: `add`, `sub`, `mul`, `neg`, `abs`,
  `sum` and `cumsum` of integers never fail and never panic. `abs` and `neg` of the minimum value
  return it unchanged.
- **NaN in comparisons and selections**, matching PyTorch:
  - `maximum`, `minimum`, `clamp`, `max`, `min`: if any compared value is NaN, the result is NaN.
  - `argmax`, `argmin`: NaN counts as larger (and smaller) than every number, so the index of the
    first NaN is returned.
  - `top_k`: NaN counts as larger than every number.
  - `eq`, `lt`, …: NaN compares false, except `ne`, which is true.
- **Ties:** `argmax`, `argmin` and `top_k` return the lowest index among equal values. `top_k`
  output is sorted (largest first, or smallest first with `largest: false`), and equal values keep
  index order.
- **Out-of-range indices** in `core.gather`, `core.index_select` and `nn.embedding` are an error,
  never undefined behavior and never a panic. Valid indices are `0 ≤ i < size`; negative indices are
  out of range (as in PyTorch for these ops). The backend checks the index tensor before it runs
  the op and returns an error; the facade maps it to `InvalidInput`, naming the node.
- Float-to-integer `cast` truncates toward zero. NaN, infinities and values outside the integer
  range give a backend-defined value, but never undefined behavior or a panic. Fixtures avoid them.
- **Subnormals are backend-defined.** A device may treat subnormal float inputs and results as
  zero (with their sign), as GPU drivers commonly do. Results that involve subnormal values are
  therefore backend-defined. Reference results are compared on normal values only: fixtures keep
  subnormals out of their cases.
- **Half-precision transcendentals:** for `f16` and `bf16`, the transcendental ops (`exp`, `log`,
  `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`) and `pow` compute in `f32` and round once to the dtype,
  so results don't depend on a backend's half-precision kernels. The output dtype is unchanged.
- **Half-precision accumulation:** for `f16` and `bf16`, `sum`, `mean`, `cumsum`, `softmax` and
  `log_softmax` compute in `f32` and round once to the dtype. The output dtype is unchanged.

#### 3.1.9 The PyTorch column

The PyTorch column is the reference the op must match (ADR-0009): the fixture for the op runs exactly
this call with the node's attributes as arguments, and the result must agree within the tolerances
of [testing-strategy.md §2](../testing-strategy.md#2-tolerances-defaults). Where PyTorch accepts
more forms than Polypous (for example an `int` where we require `int[2]`), the call uses the form
written here.

### 3.2 `core` (v1, M1)

No `core` op has params.

#### Elementwise

Binary ops broadcast `a` and `b` (§3.1.3). Their output shape is the broadcast shape.

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.add` | `a`: `N`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.add(a, b)` |
| `core.sub` | `a`: `N`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.sub(a, b)` |
| `core.mul` | `a`: `N`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.mul(a, b)` |
| `core.div` | `a`: `F`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.div(a, b)` |
| `core.pow` | `a`: `F`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.pow(a, b)` |
| `core.maximum` | `a`: `N`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.maximum(a, b)` |
| `core.minimum` | `a`: `N`, `b`: `T` | — | `out`: `T` | broadcast | yes | `torch.minimum(a, b)` |
| `core.neg` | `x`: `N` | — | `out`: `T` | `x.shape` | yes | `torch.neg(x)` |
| `core.abs` | `x`: `N` | — | `out`: `T` | `x.shape` | yes | `torch.abs(x)` |
| `core.exp` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.exp(x)` |
| `core.log` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.log(x)` |
| `core.sqrt` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.sqrt(x)` |
| `core.rsqrt` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.rsqrt(x)` |
| `core.sin` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.sin(x)` |
| `core.cos` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.cos(x)` |
| `core.tanh` | `x`: `F` | — | `out`: `T` | `x.shape` | yes | `torch.tanh(x)` |
| `core.clamp` | `x`: `N` | `min: scalar?`, `max: scalar?` | `out`: `T` | `x.shape` | yes | `torch.clamp(x, min, max)` |
| `core.where` | `cond`: `B`, `a`: `A`, `b`: `T` | — | `out`: `T` | broadcast of all three | yes (`a`, `b`) | `torch.where(cond, a, b)` |

- `div` and `pow` take floats only. Integer division has two common meanings (C truncation, Python
  floor) and PyTorch's `torch.div` of integers returns a float, which the strict dtype rule
  forbids. Graphs that need it cast to a float dtype first.
- `pow`: the exponent is a tensor. For `x²`, use a `[1]` constant and broadcasting, or `mul(x, x)`.
- `clamp`: at least one of `min` and `max` is required. If both are given, `min > max` is
  `invalid_attr` (PyTorch's result for it is surprising). A NaN bound is allowed (it compares
  false, so it is not `min > max`); the result is NaN (§3.1.8).

#### Comparison and logic

Binary ops broadcast. Outputs are `bool`. None is differentiable.

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.eq` | `a`: `A`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.eq(a, b)` |
| `core.ne` | `a`: `A`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.ne(a, b)` |
| `core.lt` | `a`: `N`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.lt(a, b)` |
| `core.le` | `a`: `N`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.le(a, b)` |
| `core.gt` | `a`: `N`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.gt(a, b)` |
| `core.ge` | `a`: `N`, `b`: `T` | — | `out`: `B` | broadcast | no | `torch.ge(a, b)` |
| `core.and` | `a`: `B`, `b`: `B` | — | `out`: `B` | broadcast | no | `torch.logical_and(a, b)` |
| `core.or` | `a`: `B`, `b`: `B` | — | `out`: `B` | broadcast | no | `torch.logical_or(a, b)` |
| `core.not` | `x`: `B` | — | `out`: `B` | `x.shape` | no | `torch.logical_not(x)` |

#### Shape

All inputs `x`: `A`, outputs `T`. Differentiable for float `x`.

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.reshape` | `x`: `A` | `shape: dim[]` (1–6 entries) | `out`: `T` | `shape`, see note | yes | `torch.reshape(x, shape)` |
| `core.flatten` | `x`: `A` | `start_dim: int = 0`, `end_dim: int = -1` | `out`: `T` | dims `start_dim…end_dim` replaced by their product | yes | `torch.flatten(x, start_dim, end_dim)` |
| `core.permute` | `x`: `A` | `dims: int[]` (a permutation, length `r`) | `out`: `T` | `[d(dims[0]), …]` | yes | `torch.permute(x, dims)` |
| `core.transpose` | `x`: `A` | `dim0: int`, `dim1: int` | `out`: `T` | `d(dim0)` and `d(dim1)` swapped | yes | `torch.transpose(x, dim0, dim1)` |
| `core.squeeze` | `x`: `A` | `dim: int[]?` | `out`: `T` | listed dims removed; see note | yes | `torch.squeeze(x, dim)` |
| `core.unsqueeze` | `x`: `A` | `dim: int` | `out`: `T` | a `1` inserted at `dim` | yes | `torch.unsqueeze(x, dim)` |
| `core.expand` | `x`: `A` | `shape: dim[]` (`r`–6 entries) | `out`: `T` | `shape`, see note | yes | `x.expand(shape)` |
| `core.concat` | `xs`: `A`, variadic (1–1024), all `T` | `dim: int` | `out`: `T` | `dim` is the sum; other dims provably equal | yes | `torch.cat(xs, dim)` |
| `core.split` | `x`: `A` | `dim: int`, `sizes: dim[]` (1–1024 entries) | `0`…`n-1`: `T` | port `j` has `sizes[j]` at `dim` | yes | `torch.split(x, sizes, dim)` |
| `core.slice` | `x`: `A` | `dim: int`, `start: dim = 0`, `end: dim?`, `step: int = 1` (≥ 1) | `out`: `T` | see note | yes | `x[(slice(None),) * dim + (slice(start, end, step),)]` |
| `core.pad` | `x`: `A` | `pad: dim[]` (even length 2…`2r`), `mode: enum = "constant"` (`constant`), `value: scalar = 0` | `out`: `T` | see note | yes | `F.pad(x, pad, mode, value)` |

- `reshape`: no `0` entries. At most one entry may be the literal `-1`; its size is the input element
  count divided by the product of the other entries. That division must be exact on normal forms:
  the product of the other entries must be a single term `c * m` that divides every term of the
  input's element count. Otherwise (and if the element counts are not provably equal)
  `shape_mismatch`.
- `squeeze`: `dim`, if given, is non-empty (as for the reductions). With `dim`, every listed dim must
  be the literal `1` (`invalid_attr` otherwise; PyTorch silently keeps non-1 dims, but a symbolic
  dim can't be decided at validation). Without `dim`, every
  literal-`1` dim is removed; symbolic dims are never removed, even if they bind to `1`. If every dim
  is removed, the result is `[1]`.
- `expand`: right-aligned against `x.shape`, new leading dims allowed. An entry `-1` keeps the input
  dim (not allowed for new leading dims). Every other entry must be provably equal to the input dim,
  or the input dim is the literal `1`.
- `slice`: `start` and `end` may be negative **integer literals**, which count from the end (`n + v`);
  expressions are never negative (`InvalidInput` at runtime if one evaluates below 0). Missing `end`
  means `n`. As in PyTorch, `start` and `end` are clamped to `n`. The output dim is
  `(min(end, n) - min(start, n) + step - 1) // step`, which must be ≥ 1.
  A negative literal below `-n` is `shape_mismatch` when `n` is a literal. A literal bound `≤ 1`
  needs no clamp (every dim is `≥ 1`), so inference writes `min(v, n)` only for other bounds:
  slicing `[seq]` with `end: -1` gives `seq - 1`, not an opaque `min`.
- `pad`: as in `F.pad`, `pad` lists `(left, right)` pairs starting from the **last** dim. Entries are
  ≥ 0 (no cropping). Each padded dim becomes `d + left + right`. M1 supports only `mode: "constant"`.
  The default `value` `0` means the zero of `x`'s dtype (`false` for `bool`); a given `value`
  follows the `scalar` rule of §3.1.7.

#### Indexing

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.gather` | `x`: `A`, `index`: `I` (rank `r`) | `dim: int` | `out`: `T` | `index.shape` | yes (`x`) | `torch.gather(x, dim, index)` |
| `core.index_select` | `x`: `A`, `index`: `I` (rank 1) | `dim: int` | `out`: `T` | `x.shape` with `d(dim)` replaced by `index.shape[0]` | yes (`x`) | `torch.index_select(x, dim, index)` |

- `gather`: for every axis other than `dim`, the `index` dim must be provably equal to the `x` dim, or
  both are literals with `index ≤ x` (PyTorch allows `≤`; a symbolic `≤` can't be proven).
- Out-of-range indices: §3.1.8.

#### Creation

These ops have no tensor inputs (except `full_like`) and are not differentiable.

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.constant` | — | `dtype: dtype` (`float` allowed), `shape: int[]` (literals, 1–6 entries), `values: scalar[]` | `out`: `dtype` | `shape` | no | `torch.tensor(values, dtype=dtype).reshape(shape)` |
| `core.arange` | — | `start: dim = 0`, `end: dim`, `step: int = 1` (≥ 1), `dtype: dtype = "i64"` (`N` or `float`) | `out`: `dtype` | `[(end - start + step - 1) // step]` | no | `torch.arange(start, end, step, dtype=dtype)` |
| `core.zeros` | — | `shape: dim[]` (1–6 entries), `dtype: dtype = "f32"` (`float` allowed) | `out`: `dtype` | `shape` | no | `torch.zeros(shape, dtype=dtype)` |
| `core.ones` | — | `shape: dim[]` (1–6 entries), `dtype: dtype = "f32"` (`float` allowed) | `out`: `dtype` | `shape` | no | `torch.ones(shape, dtype=dtype)` |
| `core.full` | — | `shape: dim[]` (1–6 entries), `value: scalar`, `dtype: dtype = "f32"` (`float` allowed) | `out`: `dtype` | `shape` | no | `torch.full(shape, value, dtype=dtype)` |
| `core.full_like` | `x`: `A` | `value: scalar`, `dtype: dtype?` (default: `x`'s dtype; `float` allowed) | `out`: `dtype` | `x.shape` | no | `torch.full_like(x, value, dtype=dtype)` |

- `constant`: `values` is row-major and has exactly as many entries as `shape` has elements, at
  most 65,536. Larger constants are params.
- `arange`: `start` and `end` are evaluated at runtime (they may use symbols); `start ≥ 0` and the
  length must be ≥ 1. For float dtypes, values are exact while they are below 2^24 (`f32`), 2^11
  (`f16`) or 2^8 (`bf16`).
- `dtype: "float"` means the compute dtype (§2.2); scalar values are rounded to it. In fixtures,
  the PyTorch call gets the resolved dtype.
- `full_like` replaces the earlier `like`: shape (and by default dtype) come from `x`, values from
  `value`. `zeros_like`/`ones_like` are `full_like` with `0`/`1`.

#### Reduction

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.sum` | `x`: `N` | `dim: int[]?` (non-empty), `keepdim: bool = false` | `out`: `T` | reduced, see note | yes | `torch.sum(x, dim, keepdim)` |
| `core.mean` | `x`: `F` | `dim: int[]?` (non-empty), `keepdim: bool = false` | `out`: `T` | reduced | yes | `torch.mean(x, dim, keepdim)` |
| `core.max` | `x`: `N` | `dim: int[]?` (non-empty), `keepdim: bool = false` | `out`: `T` | reduced | yes | `torch.amax(x, dim, keepdim)` |
| `core.min` | `x`: `N` | `dim: int[]?` (non-empty), `keepdim: bool = false` | `out`: `T` | reduced | yes | `torch.amin(x, dim, keepdim)` |
| `core.argmax` | `x`: `N` | `dim: int`, `keepdim: bool = false`, `dtype: dtype = "i64"` (`I`) | `out`: `dtype` | reduced | no | `torch.argmax(x, dim, keepdim)` |
| `core.argmin` | `x`: `N` | `dim: int`, `keepdim: bool = false`, `dtype: dtype = "i64"` (`I`) | `out`: `dtype` | reduced | no | `torch.argmin(x, dim, keepdim)` |

- Reduced shape: without `dim`, all dims are reduced. Reduced dims become `1` with `keepdim: true` and
  are removed otherwise. If every dim is removed, the result is `[1]` (§3.1.5).
- `max`/`min` reduce values only (PyTorch's `amax`/`amin`). There is no `(values, indices)` form;
  use `argmax`/`argmin` for indices.
- `argmax`/`argmin`: `dim` is required (PyTorch's flattening form is `reshape` + `argmax`).
  `dtype: "i32"` exists for devices without `i64`; the reference result is cast.

#### Linear algebra

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.matmul` | `a`: `F` (rank ≥ 2), `b`: `T` (rank ≥ 2) | — | `out`: `T` | `[batch…, n, m]`, see note | yes | `torch.matmul(a, b)` |

- `a` is `[…, n, k]`, `b` is `[…, k, m]`. The two `k` must be provably equal. The batch dims (all but
  the last two) broadcast (§3.1.3). Rank-1 operands are not allowed (PyTorch's vector forms change
  the output rank); use `unsqueeze`.

#### Other

| Op | Inputs | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|
| `core.cast` | `x`: `A` | `dtype: dtype` (`float` allowed) | `out`: `dtype` | `x.shape` | float → float only | `x.to(dtype)` |
| `core.softmax` | `x`: `F` | `dim: int` | `out`: `T` | `x.shape` | yes | `torch.softmax(x, dim)` |
| `core.log_softmax` | `x`: `F` | `dim: int` | `out`: `T` | `x.shape` | yes | `torch.log_softmax(x, dim)` |
| `core.cumsum` | `x`: `N` | `dim: int` | `out`: `T` | `x.shape` | yes | `torch.cumsum(x, dim)` |
| `core.top_k` | `x`: `F` | `k: dim` (≥ 1, ≤ `d(dim)`), `dim: int = -1`, `largest: bool = true`, `dtype: dtype = "i64"` (`I`) | `values`: `T`, `indices`: `dtype` | `x.shape` with `d(dim)` replaced by `k` | yes (`values`) | `torch.topk(x, k, dim, largest, sorted=True)` |

- `cast`: any dtype to any dtype. Casting to the same dtype is a no-op. To `bool`: non-zero is
  `true`. From `bool`: `true` is `1`. Float to integer: §3.1.8.
- `top_k`: floats only in M1 (sampling works on logits). Output is always sorted; ties and NaN:
  §3.1.8. `torch.topk` leaves the order of equal values open, so fixtures use the stable sort it
  otherwise equals, `torch.sort(x, dim, descending=largest, stable=True)` cut to the first `k`.

### 3.3 `nn` (v1, M1)

All `nn` ops take float inputs. Their params are graph params with dtype `float`, seen in the
compute dtype `C` (§2.2), so a float input `x` must have dtype `C`. **No `nn` op accepts quantized
params in M1.** Param shapes are literals, so channel dims of `x` that must match a param must be
provably equal to that literal. Optional params are marked `?`.

#### Layers

| Op | Inputs | Params | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|---|
| `nn.linear` | `x`: `C`, `[…, in]` | `weight` `[out, in]`, `bias?` `[out]` | — | `out`: `C` | `[…, out]` | yes | `F.linear(x, weight, bias)` |
| `nn.embedding` | `ids`: `I`, rank 1–5 | `weight` `[num, dim]` | — | `out`: `C` | `ids.shape + [dim]` | yes (`weight`) | `F.embedding(ids, weight)` |
| `nn.conv1d` | `x`: `C`, `[N, C_in, L]` | `weight` `[C_out, C_in/groups, k]`, `bias?` `[C_out]` | `stride: int = 1`, `padding: int = 0`, `dilation: int = 1`, `groups: int = 1` | `out`: `C` | `[N, C_out, L_out]` | yes | `F.conv1d(x, weight, bias, stride, padding, dilation, groups)` |
| `nn.conv2d` | `x`: `C`, `[N, C_in, H, W]` | `weight` `[C_out, C_in/groups, kh, kw]`, `bias?` `[C_out]` | `stride: int[2] = [1, 1]`, `padding: int[2] = [0, 0]`, `dilation: int[2] = [1, 1]`, `groups: int = 1` | `out`: `C` | `[N, C_out, H_out, W_out]` | yes | `F.conv2d(x, weight, bias, stride, padding, dilation, groups)` |
| `nn.max_pool2d` | `x`: `C`, `[N, C, H, W]` | — | `kernel_size: int[2]`, `stride: int[2]?` (default `kernel_size`), `padding: int[2] = [0, 0]`, `dilation: int[2] = [1, 1]` | `out`: `C` | `[N, C, H_out, W_out]` | yes | `F.max_pool2d(x, kernel_size, stride, padding, dilation)` |
| `nn.avg_pool2d` | `x`: `C`, `[N, C, H, W]` | — | `kernel_size: int[2]`, `stride: int[2]?` (default `kernel_size`), `padding: int[2] = [0, 0]`, `count_include_pad: bool = true` | `out`: `C` | `[N, C, H_out, W_out]` | yes | `F.avg_pool2d(x, kernel_size, stride, padding, count_include_pad=count_include_pad)` |
| `nn.adaptive_avg_pool2d` | `x`: `C`, `[N, C, H, W]` | — | `output_size: int[2]` (≥ 1) | `out`: `C` | `[N, C, oh, ow]` | yes | `F.adaptive_avg_pool2d(x, output_size)` |
| `nn.dropout` | `x`: `C` | — | `p: float = 0.5` (`0 ≤ p < 1`) | `out`: `C` | `x.shape` | yes | `F.dropout(x, p, training=False)` |

- The input `x` of `nn` ops has dtype `C`; the tables write it as `C`, and the allowed class is `F`.
- `linear`: `x` has rank 1–6; its last dim must be provably equal to `in`.
- `embedding`: the output rank is the rank of `ids` + 1, so `ids` has rank ≤ 5. Out-of-range
  indices: §3.1.8. `padding_idx`, `max_norm` and sparse gradients are not supported.
- Conv and pool output size per spatial dim (`H` with `kh`, `s`, `p`, `dl`; pools use the kernel
  size as `k` and `dl = 1` where there is no dilation):
  `H_out = (H + 2*p - dl*(k - 1) - 1) // s + 1`, which must be ≥ 1.
- Conv: `stride` and `dilation` entries ≥ 1, `padding` ≥ 0, `groups` ≥ 1 and divides both `C_in` and
  `C_out`. `x`'s `C_in` must be provably equal to `weight.shape[1] * groups`. Only zero padding
  (PyTorch's `padding_mode="zeros"`); string paddings (`"same"`, `"valid"`) are not supported.
- Pools: `padding` entries ≤ `kernel_size / 2` (as in PyTorch). `ceil_mode`, `return_indices` and
  `divisor_override` are not supported (always PyTorch's defaults). `max_pool2d` pads with `-inf`.
- `adaptive_avg_pool2d`: output cell `i` averages input rows `floor(i*H/oh)` to
  `ceil((i+1)*H/oh) - 1`, as in PyTorch.
- `dropout`: identity in M1 (there is no training mode before M2). `p` is validated so graphs stay
  valid for training.
- Unbatched inputs (PyTorch's conv/pool on rank 2 or 3 without `N`) are not supported; `unsqueeze`.

#### Normalization

| Op | Inputs | Params | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|---|
| `nn.batch_norm` | `x`: `C`, `[N, C, …]`, rank 2–6 | `running_mean` `[C]`, `running_var` `[C]`, `weight?` `[C]`, `bias?` `[C]` | `eps: float = 1e-5` (> 0) | `out`: `C` | `x.shape` | yes (`x`, `weight`, `bias`) | `F.batch_norm(x, running_mean, running_var, weight, bias, training=False, eps=eps)` |
| `nn.layer_norm` | `x`: `C` | `weight?` `normalized_shape`, `bias?` `normalized_shape` | `normalized_shape: int[]?` (1…`r` entries), `eps: float = 1e-5` (> 0) | `out`: `C` | `x.shape` | yes | `F.layer_norm(x, normalized_shape, weight, bias, eps)` |
| `nn.group_norm` | `x`: `C`, `[N, C, …]`, rank 2–6 | `weight?` `[C]`, `bias?` `[C]` | `num_groups: int` (≥ 1, divides `C`), `eps: float = 1e-5` (> 0) | `out`: `C` | `x.shape` | yes | `F.group_norm(x, num_groups, weight, bias, eps)` |
| `nn.rms_norm` | `x`: `C` | `weight?` `normalized_shape` | `normalized_shape: int[]?` (1…`r` entries), `eps: float` (> 0) | `out`: `C` | `x.shape` | yes | `F.rms_norm(x, normalized_shape, weight, eps)` |

- `batch_norm` is inference mode only in M1: it normalizes with the running statistics, which are
  never updated. Training mode (batch statistics, `momentum`, updating the running statistics) is
  specified with M2. `running_mean` and `running_var` are params, so they are not differentiable.
- `layer_norm`, `rms_norm`: `normalized_shape` lists literal dims that must be provably equal to the
  trailing dims of `x`. The params have exactly that shape. Without the attribute, it is the shape
  of `weight`; without both, `missing_attr`.
- `group_norm`, `batch_norm`: `C` must be a literal (it is checked against param shapes and
  `num_groups`).
- `rms_norm`: `eps` is required. PyTorch's default (`None`, meaning the dtype's machine epsilon)
  differs between dtypes, and models always state their epsilon.

#### Activations

All take `x`: `C` and return `out`: `C` with `x.shape`. All are differentiable.

| Op | Inputs | Params | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|---|
| `nn.relu` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `F.relu(x)` |
| `nn.leaky_relu` | `x`: `C` | — | `negative_slope: float = 0.01` | `out`: `C` | `x.shape` | yes | `F.leaky_relu(x, negative_slope)` |
| `nn.gelu` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `F.gelu(x)` (exact, erf) |
| `nn.gelu_tanh` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `F.gelu(x, approximate="tanh")` |
| `nn.silu` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `F.silu(x)` |
| `nn.sigmoid` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `torch.sigmoid(x)` |
| `nn.softplus` | `x`: `C` | — | `beta: float = 1.0` (> 0), `threshold: float = 20.0` (> 0) | `out`: `C` | `x.shape` | yes | `F.softplus(x, beta, threshold)` |
| `nn.mish` | `x`: `C` | — | — | `out`: `C` | `x.shape` | yes | `F.mish(x)` |

### 3.4 Planned ops (not M1)

Not in the M1 catalog. A graph that uses them fails validation with `unknown_op`.

- `core`: `pad` modes `reflect`, `replicate` and `circular`; integer division and remainder;
  reading a param as a value (for example a learned position embedding added with `core.add`);
  `any`/`all`.
- `nn` (M2 or later, as needed): `conv_transpose2d`, `upsample_nearest`, `upsample_bilinear`;
  training mode of `batch_norm` and `dropout`.

#### `transformer` (M3)

- `rope`: attrs `style` (`neox` | `interleaved`), `theta`, `scaling` (`none` | `linear` | `ntk` |
  `yarn` | `llama3`, with parameters), `rotary_dim`.
- `attention`: scaled dot-product attention with `n_heads`, `n_kv_heads` (GQA/MQA), `mask`
  (`none` | `causal` | `sliding{window}` | explicit input), `softcap`, `scale`. Optional KV cache
  inputs/outputs plus a `past` position. Cross-attention when K/V come from another input.
- `swiglu`, `geglu`: fused gated activations.
- `moe` (later): router + experts.

#### `audio` / `diffusion` (M5–M6)

Added as needed (for example `group_norm` with SiLU fusions, `timestep_embedding`). Preprocessing
such as mel spectrograms is **not** a graph op. It is a pipeline component.

## 4. Validation

Validation runs on load and on every edit through the builder API. Each error has a stable
`code`, a `message`, and a location (`node_id`, optional `input`/`param`/`attr` name, JSON pointer).
Errors are collected (not fail-fast), so a builder can show all problems at once. Errors that make
later checks meaningless (a JSON syntax error, a resource limit) stop validation early.

### 4.1 Checks

1. The document is within the JSON limits (§4.3) and parses.
2. Format and opset versions are supported (otherwise migrate or reject).
3. Names are valid and unique; every reference resolves; the graph is acyclic.
4. Each op exists in the declared opset version; required inputs, params and attrs are present and
   typed; no unknown fields.
5. Dims are declared and well-formed (§2.3).
6. Dtype inference succeeds (§3.1.2).
7. Symbolic shape inference succeeds, and incompatible dims produce an error on the offending node.
8. Every state update and every `repeat` carry has a matching dtype and shape.
9. The resource limits of §4.3 hold.

### 4.2 Error codes

Codes are stable `snake_case` strings. New codes may be added; existing ones keep their meaning.

| Code | Raised when |
|---|---|
| `invalid_json` | The document isn't valid UTF-8 JSON, a field has the wrong JSON type (including a single reference for a variadic input, or an array for a non-variadic one), or an object has a duplicate key (anywhere, `meta` included; JSON parsers disagree on which value wins) |
| `document_too_large` | The document exceeds the size limit or the JSON value limit (§4.3) |
| `json_too_deep` | JSON nesting exceeds the depth limit |
| `string_too_long` | A JSON string exceeds its length limit |
| `unsupported_format` | `format` isn't `polypous.graph`, or `format_version` is unknown |
| `unsupported_opset` | An opset in `opsets` is unknown or its version isn't supported |
| `unknown_field` | A field that the schema doesn't define (except inside `meta`) |
| `missing_field` | A required field is missing (for example a node's `op`, a param's `shape`) |
| `invalid_value` | A field has a value outside its allowed set (a bad dtype name, `float` where it isn't allowed, an unknown `init` kind, `min`/`max` of a dim, an empty shape, a literal dim above 2^32, an expression where §2.3 allows only literals or plain symbols, no `outputs`), or a list exceeds its §4.3 limit (`dims`, inputs, outputs, state entries, the references of a variadic input: 1–1,024) |
| `invalid_name` | A name or id breaks the naming rules (§2.1, §2.3.1) |
| `duplicate_name` | Two values, params, dims, state entries or outputs share a name |
| `unknown_reference` | A reference to an undeclared value, port, param, state entry or dim symbol (a shape entry that is a plain symbol; an undeclared symbol inside a longer expression is `invalid_dim_expr`) |
| `ambiguous_output` | A bare node id refers to a node without an `out` port |
| `cycle` | The nodes form a cycle |
| `unknown_op` | The op isn't in the catalog of the declared opset version, or its opset isn't declared |
| `unsupported_in_version` | A feature that this version specifies but doesn't implement: `functions`, `call`, non-`float` params |
| `missing_input` / `missing_param` / `missing_attr` | A required op input, param or attribute is absent (reported at the node, with the input, param or attribute as the item) |
| `unknown_input` / `unknown_param` / `unknown_attr` | The op defines no input, param or attribute of that name |
| `invalid_attr` | An attribute has the wrong type, is out of range, has a wrong list length, or is an invalid axis (§3.1.4, §3.1.7) |
| `unsupported_dtype` | An input's or param's dtype is outside the op's allowed class |
| `dtype_mismatch` | Inputs or params that must have the same dtype don't (§3.1.2) |
| `rank_mismatch` | An input or param has a rank the op doesn't allow |
| `rank_too_large` | A declared or inferred rank exceeds 6 |
| `shape_mismatch` | Dims that must be equal are not provably equal (§2.3.3), or a literal bound is out of range |
| `broadcast_mismatch` | Two dims can't be broadcast (§3.1.3) |
| `invalid_dim_expr` | A dim expression doesn't parse, uses an undeclared symbol, divides by a non-literal or non-positive value, or exceeds the expression limits (§2.3.5) |
| `dim_overflow` | `i64` overflow while normalizing a dim expression |
| `nonpositive_dim` | A literal or literal-folded dim is ≤ 0 |
| `unbound_dim` | A used symbol doesn't appear as a plain dim in any input or state shape |
| `state_update_mismatch` | A state update has a different dtype or not provably the same shape, or updates an entry twice |
| `repeat_carry_mismatch` | `carry` and `carry_out` have different keys, or a carry's dtype or shape differs between them |
| `invalid_repeat` | `count` is out of range, `carry` is empty, or a body with state has a `state_prefix` without `{i}` |
| `too_many_nodes` | More than 1,000,000 nodes, or more than 4,000,000 param bindings, after expanding repeats |
| `repeat_too_deep` | `repeat` nesting exceeds 8 levels |
| `too_many_elements` | A param's or constant's element count, or the total param element count, doesn't fit in `u64` or exceeds its limit |

### 4.3 Resource limits

Graph documents are untrusted. Each limit is checked before the memory it protects is allocated.

| Limit | Value |
|---|---|
| Document size | ≤ 64 MiB |
| JSON nesting depth (arrays and objects, including `meta`) | ≤ 128 |
| JSON values (scalars, arrays and objects, including `meta`; object keys don't count) | ≤ 4,000,000 (`document_too_large`), counted before parsing |
| Names, ids, dim symbols | §2.1, §2.3.1 (≤ 256 bytes) |
| Dim expressions | §2.3.5 (≤ 256 bytes, depth 16, 64 terms, degree 8) |
| Any other JSON string (enum values, `meta` content) | ≤ 64 KiB |
| Rank of any value, param or state entry | 1–6 |
| Nodes after expanding repeats | ≤ 1,000,000, computed from counts before expanding |
| Param bindings after expanding repeats (node `params` entries × iterations) | ≤ 4,000,000, computed from counts before expanding |
| `repeat` nesting | ≤ 8 levels |
| `repeat` `count` | 1–65,536 |
| Declared `dims` symbols | ≤ 256 |
| Graph inputs, outputs, state entries | ≤ 4,096 each; state entries counted after expanding repeats |
| `core.concat` inputs, `core.split` outputs | ≤ 1,024 |
| `core.constant` elements | ≤ 65,536 |
| Literal dims | ≤ 2^32 |
| Total declared param elements | fits in `u64` (checked arithmetic) |

The value limit bounds the memory of the parsed document: without it, a 64 MiB document of tiny
values (`[0,0,…]`) parses into a tree of about 1 GiB. Builder and `.polyp`-generated graphs are far
below it (repeats keep big models small; a maximal `core.constant` is 65,536 values).

The param binding limit bounds the work of checking that every iteration's params exist (§2.6):
no M1 op has more than 4 params, so a graph within the node limit is within it too.

The repeat nesting limit is 8 (the earlier draft said 64) because each nesting level costs about four
JSON levels, and real models nest at most twice.

### 4.4 Runtime errors

These are not validation codes; the runtime raises them as `polypous::Error` kinds:

| Kind | When |
|---|---|
| `InvalidInput` | A binding check of §2.3.4 fails (missing or extra input, wrong dtype or rank, inconsistent symbol, out of `min`/`max`, an inferred dim < 1, overflow), or an index is out of range (§3.1.8) |
| `UnsupportedOperation` | The device doesn't support a dtype or (op, dtype) pair of the graph, or the compute dtype (§2.2) |

## 5. Versioning and migration

- `format_version` covers document structure; `opsets` cover op semantics.
- `polypous-ir` can read any older version and migrates it in memory. Saving writes the current version.
- Removing or changing an op's semantics requires a new opset version. The old version stays
  supported, through migration or a legacy implementation, for at least one major Polypous release.
- Unknown top-level or node fields are **errors** (catches typos). The exception is `meta`, which is free-form.

## 6. Serialization

- Canonical form: UTF-8 JSON, keys in a stable order, so diffs are readable:
  - Fields defined by the schema are written in schema order, as in the §2 example: graph
    `format`, `format_version`, `opsets`, `name`, `meta`, `dims`, `inputs`, `outputs`, `params`,
    `state`, `nodes`, `state_updates`; nodes `id`, `op`, `count`, `carry`, `body`, `inputs`,
    `params`, `attrs`, `meta`; bodies `params_prefix`, `state_prefix`, `state`, `nodes`,
    `state_updates`, `carry_out`; inputs, params and state entries `name`, `dtype`, `shape`, `init`,
    `trainable`, `meta`; outputs `name`, `value`, `meta`; `init` objects `kind` first; `dims`
    entries `min`, `max`.
  - Maps keyed by names (`opsets`, `dims`, node `inputs`, `params` and `attrs`, `carry`,
    `carry_out`, `state_updates`) keep the order they were written or built in. `meta` and
    attribute values are written exactly as stored, key order included.
  - Optional fields are written only if present; optional lists and maps only if non-empty.
    Values are written as written (`"float"` stays `"float"`, dims stay literal or string).
  - Layout: two-space indentation, one object member per line; arrays that contain only scalars on
    one line (`[128, 784]`); a trailing newline. Integers are written exactly; other numbers in
    the shortest form that parses back to the same `f64`.
  - load → save → load gives the same graph, and save → load → save the same bytes.
- A trained or exported model is `model.graph.json` + `model.safetensors` (param names = graph
  param names). A bundled single-file container may be added later.

## 7. Open questions

- Explicit control flow (`if`, `while`) inside graphs. v1: no. Loops live in pipelines.
- A binary encoding for huge graphs. Probably unnecessary because `repeat` keeps graphs small.
