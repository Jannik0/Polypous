# Spec: Graph IR

**Status:** Binding for M1: §2 (all), §3.1–§3.3 (general rules and the M1 op catalog), §4, §5 and §6.
Binding for M2 (training): initial values and `trainable` (§2.4, §2.4.1), training mode (§3.1.10),
gradients (§3.1.11), random values (§3.1.12), the training mode of `nn.batch_norm` and `nn.dropout`
(§3.3) and `invalid_init` (§4.2). Draft: §3.4 (planned ops) and §7. The `transformer` opset is
refined in M3. **Crate:** `polypous-ir`

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
- A param may be used by several nodes (weight tying). An unused param is not an error, unless its
  `init` is fan-based (§2.4.1).
- `init` kinds (v1): `zeros`, `ones`, `constant{value}`, `uniform{low,high}`, `normal{mean,std}`,
  `xavier_uniform`, `xavier_normal`, `kaiming_uniform{a}`, `kaiming_normal`. The values each kind
  gives, and the fan-in and fan-out that the fan-based kinds take from the ops using the param, are
  in §2.4.1. Pretrained models ignore `init`: weights come from files. Validation checks `init`
  whether or not weights are loaded later; the trainer applies it to the params that get no loaded
  weights ([training.md](training.md)).
- `init` fields: `constant.value`, `uniform.low`/`high` and `normal.mean`/`std` are required,
  `kaiming_uniform.a` is optional (default as in `torch.nn.init.kaiming_uniform_`, `0`). All are
  JSON numbers and must be finite. Other fields are `unknown_field`. `init` itself is optional.
  As in `torch.nn.init`, which raises for each of these, the values must also be usable:
  `uniform.low ≤ uniform.high`, `normal.std ≥ 0`, and `constant.value`, `uniform.low`/`high` and
  `normal.mean`/`std` must round to a finite `f32`. A violation is `invalid_value` on the field.
  (`low = high` and `std = 0` are allowed and give a constant.)
- `trainable` defaults to `true`. A param with `trainable: false` gets no gradient, is never
  updated by the optimizer and has no optimizer state; it is still initialized (from `init` or a
  weight file) like every other param. A param used as `running_mean` or `running_var` of
  `nn.batch_norm` is never trained, whatever `trainable` says (§3.3). `trainable` doesn't change
  what a call computes.
- A graph without `init` is valid. A trainable param with neither `init` nor a loaded weight is an
  error when the trainer is created, listing those params (owner decision 8 for M2; the error is
  specified in [training.md](training.md)).

#### 2.4.1 Initial values

The trainer computes initial values on the host, never on the device, from the training seed (a
`u64`, [training.md](training.md)), the param's name and shape, and its `init`. Each param has its
own random stream (§3.1.12), so its values don't depend on the device, on other params or on the
order params are initialized in. Every value is computed in `f64` exactly as written below and
rounded once to `f32` (round to nearest, ties to even). Training is `f32` only in M2 (§3.1.11), so
initial values are `f32`.

Element `i` is the element at row-major index `i` of the param. With
`key = stream_key(seed, 0, "init:" + name)` (§3.1.12), `U(i)` and `N(i)` are the uniform and normal
values of §3.1.12 for that key. Integers (fans) convert to `f64` with round to nearest.

| Kind | Value of element `i` |
|---|---|
| `zeros` | `0` |
| `ones` | `1` |
| `constant{value}` | `value` |
| `uniform{low, high}` | `low + (high - low) · U(i)` |
| `normal{mean, std}` | `mean + std · N(i)` |
| `kaiming_uniform{a}` | `uniform` with `low = -bound`, `high = bound`, where `gain = sqrt(2 / (1 + a · a))`, `std = gain / sqrt(fan_in)`, `bound = sqrt(3) · std` |
| `kaiming_normal` | `normal` with `mean = 0`, `std = gain / sqrt(fan_in)`, `gain = sqrt(2 / (1 + 0))` |
| `xavier_uniform` | `uniform` with `low = -bound`, `high = bound`, where `std = 1 · sqrt(2 / (fan_in + fan_out))`, `bound = sqrt(3) · std` |
| `xavier_normal` | `normal` with `mean = 0`, `std = 1 · sqrt(2 / (fan_in + fan_out))` |

- These are the formulas of `torch.nn.init` (`kaiming_uniform_`, `kaiming_normal_`,
  `xavier_uniform_`, `xavier_normal_`) in the same order of operations. The values themselves
  differ from PyTorch's, because the random numbers come from the Polypous generator.
- `kaiming_*` use `fan_in` (PyTorch's `mode="fan_in"`) and the leaky-relu gain
  (`nonlinearity="leaky_relu"`). Neither is configurable. `kaiming_normal` has no `a`: it is
  always `0` (PyTorch's default), and a field `a` on it is `unknown_field`. Adding `a` later is
  backward compatible. `xavier_*` use gain `1`.
- `uniform` values lie in `[low, high]` after rounding. `normal` values are unbounded; with an
  extreme `mean` or `std` they may round to `±inf` (IEEE 754), as in PyTorch.

**Fans.** The fan-based kinds (`kaiming_*`, `xavier_*`) take `fan_in` and `fan_out` from the
nodes that use the param, by the param's role in the op:

| Op | Param | `fan_in` | `fan_out` |
|---|---|---|---|
| `nn.linear` | `weight` `[out, in]` | `in` | `out` |
| `nn.embedding` | `weight` `[num, dim]` | `dim` | `num` |
| `nn.conv1d` | `weight` `[C_out, C_in/groups, k]` | `C_in/groups · k` | `C_out · k` |
| `nn.conv2d` | `weight` `[C_out, C_in/groups, kh, kw]` | `C_in/groups · kh · kw` | `C_out · kh · kw` |

- This is PyTorch's `torch.nn.init._calculate_fan_in_and_fan_out`: `fan_in = shape[1] · rf`,
  `fan_out = shape[0] · rf`, where `rf` is the product of the dims after the second. `groups` is
  already in the weight shape, and `fan_out` is not divided by it, as in PyTorch.
- `nn.embedding`'s fans are PyTorch's computation for its 2-D weight. (PyTorch's `nn.Embedding`
  initializes with `normal(0, 1)`, which is the kind `normal` here.)
- Every other role has no fans: biases, the params of the normalization ops (also a
  multi-dimensional `layer_norm` weight) and `batch_norm`'s running statistics. PyTorch raises for
  tensors with fewer than 2 dims; Polypous goes further and accepts only the roles of the table.
- A fan-based kind is the validation error `invalid_init` (§4.2) on the param when:
  - no node uses the param (after expanding repeats);
  - a node uses it in a role without fans;
  - two uses give different `(fan_in, fan_out)`. With the roles of the table the fans depend only
    on the param's shape, so this can't happen in `nn` v1; the rule is there for later ops whose
    fans follow another rule.
- The kinds without fans (`zeros`, `ones`, `constant`, `uniform`, `normal`) work on every param,
  used or not.

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
  creates the initial values (zeros of the declared dtype and shape). The call contract of the
  interpreter is in [runtime.md §3](runtime.md#3-calls).

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

**Not in M1 or M2** (owner decision 4 for M2). Functions arrive in a later milestone with the
builder's composite layers. Until then a `functions` field on the graph or a node with `op: "call"`
is the validation error `unsupported_in_version`.

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
  bool inputs are never differentiable. What the gradients are is §3.1.11.

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
  `sqrt`, `rsqrt`, `sin`, `cos`, `tanh`), `pow` and the `nn` activations other than `relu`
  (`leaky_relu`, `gelu`, `gelu_tanh`, `silu`, `sigmoid`, `softplus`, `mish`) compute in `f32` and
  round once to the dtype, so results don't depend on a backend's half-precision kernels. The
  output dtype is unchanged.
- **Half-precision accumulation:** for `f16` and `bf16`, `sum`, `mean`, `cumsum`, `softmax`,
  `log_softmax`, `core.matmul`, `nn.linear`, `nn.conv1d`, `nn.conv2d`, `nn.avg_pool2d`, `nn.adaptive_avg_pool2d`
  and the normalization ops (`nn.batch_norm`, `nn.layer_norm`, `nn.group_norm`, `nn.rms_norm`,
  params included) compute in `f32` and round once to the dtype. The output dtype is unchanged.

#### 3.1.9 The PyTorch column

The PyTorch column is the reference the op must match (ADR-0009): the fixture for the op runs exactly
this call with the node's attributes as arguments, and the result must agree within the tolerances
of [testing-strategy.md §2](../testing-strategy.md#2-tolerances-defaults). Where PyTorch accepts
more forms than Polypous (for example an `int` where we require `int[2]`), the call uses the form
written here.

#### 3.1.10 Training mode

A call runs in **inference mode** (the default, and the behavior of M1) or in **training mode**.
The mode belongs to the call, not to the graph: validation doesn't depend on it, and a valid graph
can be called in both modes. [runtime.md §3.1](runtime.md#31-mode) says how a call carries the mode
and which checks a training-mode call adds (compute dtype `f32`, one use per running statistic,
at least 2 values per channel for `batch_norm`).

- Only two ops behave differently in training mode: `nn.batch_norm` normalizes with the batch
  statistics and returns updated running statistics, and `nn.dropout` with `p > 0` zeroes random
  elements (§3.3). Every other op computes the same in both modes.
- A training-mode call carries a `seed` and a `step` (both `u64`), which key the dropout masks
  (§3.1.12).
- The mode and gradients are independent. Gradients come from tracked params and inputs
  ([ADR-0014](../adr/0014-training-on-dyntensor-with-own-optimizers.md)), and an inference-mode
  call records them the same way.
- In M2, training mode requires the compute dtype `f32` (owner decision 6 for M2, §3.1.11).
- Training mode adds the attribute `momentum` to `nn.batch_norm` (with a default) and changes no
  inference result, so it is not a new opset version (§5): `nn` stays at version 1.

#### 3.1.11 Gradients

The "Diff." column says which inputs and params of an op get gradients: "yes" means every float
input and every param, except `batch_norm`'s running statistics, which never get one (§3.3).
Integer and bool values never have gradients. An op marked "no" passes no gradient to its inputs.

**Reference.** An op's gradient is the one PyTorch's autograd computes for the call in the PyTorch
column (with the argument forms written there, §3.1.9), at the same inputs and upstream gradient,
within the gradient tolerance of [testing-strategy.md §2](../testing-strategy.md#2-tolerances-defaults).
The training-mode ops use their training-mode call (§3.3). Gradients add up over every use of a
value: a value read by several nodes, a param used by several nodes or by every iteration of a
`repeat` (a `params_prefix` without `{i}`), and repeated indices of `core.gather`,
`core.index_select` and `nn.embedding` all receive the sum.

**Finite inputs.** Where every input of the op and its upstream gradient `g` are finite, the
gradient is PyTorch's, also where that is `±inf` or NaN: for example `sqrt'(0) = +inf`,
`rsqrt'(0) = -inf`, `log'(0) = +inf`, and the gradients of `div` at `b = 0`. At `x = -0` the sign
follows the formulas `g / (2 · sqrt(x))`, `-g · rsqrt(x)³ / 2` and `g / x` with `sqrt(-0) = -0`
(IEEE 754), as in PyTorch: `sqrt'(-0) = -inf`, `rsqrt'(-0) = +inf` and `log'(-0) = -inf` for
`g > 0`. Where the derivative is not unique (ties and kinks), PyTorch's choice is binding, as
follows (checked against PyTorch 2.15.0, #72):

| Op | Where | Gradient |
|---|---|---|
| `core.max`, `core.min` | `k` elements of a reduced slice equal the result | each gets `g / k` (`torch.amax`/`amin` split it evenly) |
| `core.maximum`, `core.minimum` | `a = b` | `g / 2` to `a` and `g / 2` to `b` |
| `nn.max_pool2d` | several elements of a window equal its maximum | all of `g` to the first of them, in the window's row-major order (`kh`, then `kw`); overlapping windows add up |
| `core.top_k` | equal values | to the elements at the returned `indices` (§3.1.8 ties) |
| `core.clamp` | `x = min` or `x = max` (also `min = max`) | `0`. `g` passes only where `min < x < max` (with one bound: `x > min`, or `x < max`) |
| `core.where` | — | `g` to `a` where `cond` is true, to `b` elsewhere, `0` to the other |
| `core.abs` | `x = 0` (also `-0`) | `0` |
| `nn.relu` | `x = 0` (also `-0`) | `0` |
| `nn.leaky_relu` | `x = 0` (also `-0`) | `negative_slope · g` |
| `nn.softplus` | `β·x = 0` | `g / 2` (`= σ(0)`). Above the threshold (`β·x > threshold`), `g` |
| `core.sqrt` | `x = 0` | `g / (2 · sqrt(x))`, so `+inf` for `g > 0` |
| `core.pow` | `b = 0` | `∂/∂a = 0` (also at `a = 0`) |
| `core.pow` | `a = 0` and `b ≥ 0` | `∂/∂b = 0` |
| `core.pow` | elsewhere | `∂/∂a = g · b · a^(b-1)`, `∂/∂b = g · a^b · ln(a)`: NaN for `a < 0`, `±inf` at `a = 0` with `b < 0` or `0 < b < 1` |

- `clamp`: this is PyTorch's form with scalar bounds, which the PyTorch column uses
  (`torch.clamp(x, min, max)` with the attributes as Python numbers). PyTorch's form with tensor
  bounds gives `g / 2` at a bound and is not the reference.
- `max_pool2d`: a window that lies entirely in the padding (possible with `dilation > 1`, §3.3)
  reads no element and passes no gradient. PyTorch 2.15.0's CPU backward corrupts its heap in that
  case (`malloc(): invalid size`), so it has no fixture.

**Non-finite inputs.** Where an input of the op or its upstream gradient is NaN or `±inf`, the
gradient is unspecified: any value, but never a panic, and the backward pass completes. PyTorch's
own results there are irregular and are not the reference. For example (informative): `amax` of a
slice holding NaN gives NaN to the whole slice, `maximum(NaN, 1)` passes `g` to both inputs,
`clamp(NaN)` gives `0` with scalar bounds and `g` with tensor bounds, `relu'(NaN) = 1`, and
`max_pool2d` sends the gradient to a NaN in the window.

**Dtypes.** Gradients are `f32` only in M2 (owner decision 6 for M2): the trainer and training mode
require the compute dtype `f32`, and other compute dtypes are `UnsupportedOperation`
([training.md](training.md), [runtime.md §3.1](runtime.md#31-mode)). A graph that casts values to
`f16` or `bf16` inside gets gradients through them by the same rules, but M2's reference tests cover
`f32` only. The `f16`/`bf16` compute paths of §3.1.8 are not part of M2's gradients.

#### 3.1.12 Random values

Dropout masks (§3.3) and initial values (§2.4.1) come from one Polypous generator on the host,
never from a device RNG (owner decision 2 for M2,
[ADR-0014](../adr/0014-training-on-dyntensor-with-own-optimizers.md)). They are therefore the same
on every device and backend, and independent of threads and of the order in which nodes run. The
generator is SplitMix64 (Steele, Lea and Flood, 2014) used as a counter-based hash: value `i` of a
stream is a hash of the stream's key and `i`, so no generator state is kept or saved.

All arithmetic is on `u64` and wraps (modulo 2^64); `^` is exclusive or and `>>` a logical shift.

```text
GAMMA = 0x9E3779B97F4A7C15

mix(z):         z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
                z = (z ^ (z >> 27)) * 0x94D049BB133111EB
                return z ^ (z >> 31)

fnv1a(label):   h = 0xCBF29CE484222325
                for each byte b of label:  h = (h ^ b) * 0x00000100000001B3
                return h

stream_key(seed, step, label) = mix(mix(seed ^ GAMMA) ^ mix(step) ^ fnv1a(label))
draw(key, i)                  = mix(key + (i + 1) * GAMMA)
```

**Streams.** Labels are ASCII strings:

| Use | `seed` | `step` | `label` |
|---|---|---|---|
| `nn.dropout` | the call's `seed` | the call's `step` | `"dropout:"` + the node path of [runtime.md §5](runtime.md#5-errors) (`drop1`, `layers[3]/drop`, `outer[1]/inner[0]/drop`) |
| `init` (§2.4.1) | the training seed | `0` | `"init:"` + the param name |

- A node inside a `repeat` has a different path, and so a different stream, in each iteration.
- `:` appears in neither node paths nor param names, so a dropout label never equals an init label.
  Two different labels hash to the same `fnv1a` value with a probability of about 2^-64 per pair;
  that would only correlate two streams.

**Values from draws.** In `f64`, from `z = draw(key, i)`:

- `u24(z) = (z >> 40) · 2^-24`, uniform in `[0, 1)` on a grid of 2^-24. Exact in `f32` and `f64`.
  Dropout uses it.
- `U(i) = (z >> 11) · 2^-53`, uniform in `[0, 1)` on a grid of 2^-53. Exact in `f64`.
- `N(i)`, standard normal by the Box–Muller transform, one value from each pair of draws:
  `v1 = ((draw(key, 2i) >> 11) + 1) · 2^-53` (in `(0, 1]`, so the logarithm is finite),
  `v2 = (draw(key, 2i + 1) >> 11) · 2^-53`, and `N(i) = sqrt(-2 · ln(v1)) · cos(TAU · v2)`, with
  `TAU` the `f64` nearest to 2π (Rust's `std::f64::consts::TAU`).

Every index `i` (and `2i + 1`) of a tensor that can be allocated is far below 2^63, so the counters
don't wrap.

**Exactness.** Everything above except `ln` and `cos` is exact or correctly rounded in IEEE 754
(integer arithmetic, multiplication by powers of two, `+`, `-`, `·`, `/`, `sqrt`). So the streams,
`u24`, `U`, dropout masks and the initial values of `zeros`, `ones`, `constant`, `uniform`,
`kaiming_uniform` and `xavier_uniform` are bit-identical on every platform and in every correct
implementation. `ln` and `cos` come from the platform's math library, which IEEE 754 doesn't require
to round correctly; the values of `normal`, `kaiming_normal` and `xavier_normal` may therefore
differ between platforms in the last bit of the `f64` result, which after rounding to `f32` rarely
shows and is then 1 ulp. Their reference test allows 1 ulp (`f32`); the others are bit-exact.

**Known values** (for tests of an implementation):

| Quantity | Value |
|---|---|
| `draw(0, 0)` (SplitMix64's first output for seed 0) | `0xE220A8397B1DCDAF` |
| `fnv1a("a")` | `0xAF63DC4C8601EC8C` |
| `stream_key(42, 7, "dropout:layers[3]/drop")` | `0x559FE2396A2BE18C` |
| `draw` 0–3 of that key | `0xDEAFBD2049BBA567`, `0x97B9C0C7F5A35D08`, `0xC5DA3B226895E44F`, `0x09407FF20490E439` |
| their `u24` | `0.8698690533638`, `0.5926780700683594`, `0.7728611826896667`, `0.036140382289886475` |
| `stream_key(42, 0, "init:fc1.weight")` | `0x4A64D327A84495D0` |
| `U(0)`, `U(1)`, `U(2)` of that key | `0.9418494217379093`, `0.952777646315502`, `0.21886262424309422` |
| `N(0)`, `N(1)`, `N(2)` of that key (`f64`) | `0.33102468465391816`, `1.1951877566467608`, `0.2241557644014358` |
| `kaiming_uniform` (`a = 0`) of `fc1.weight` `[128, 784]`, seed 42, elements 0–2 (`f32`) | `0.077307545`, `0.07921959`, `-0.049188793` |

The generator's statistical test (uniformity of `u24`, dropout rates, moments of `N`) belongs to
[testing-strategy.md](../testing-strategy.md).

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
  the output rank); use `unsqueeze`. `f16`/`bf16` compute in `f32` and round once (§3.1.8).

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
| `nn.dropout` | `x`: `C` | — | `p: float = 0.5` (`0 ≤ p < 1`) | `out`: `C` | `x.shape` | yes | `F.dropout(x, p, training=t)`, `t` = training mode; see note |

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
- Pools: `padding` entries ≤ `kernel_size / 2` (integer division, as in PyTorch). `ceil_mode`,
  `return_indices` and `divisor_override` are not supported (always PyTorch's defaults).
  `max_pool2d` pads with `-inf`, and a NaN in a window gives NaN. With `dilation` > 1 a window can
  lie entirely in the padding (`H = 2`, `kh = 2`, `p = 1`, `dl = 3` reads rows `-1` and `2`); its
  result is `-inf`, as in PyTorch.
- `adaptive_avg_pool2d`: output cell `i` averages input rows `floor(i*H/oh)` to
  `ceil((i+1)*H/oh) - 1`, as in PyTorch.
- `dropout` in inference mode is the identity (`out = x`), for every `p`. In training mode
  (§3.1.10), as PyTorch's `F.dropout(x, p, training=True)` on the CPU, but with Polypous masks:
  - `p = 0` is the identity and draws nothing.
  - Otherwise element `i` of `x` (row-major index at the bound shape) is kept if
    `u24(draw(key, i)) ≥ p`, compared in `f64`, with `key = stream_key(seed, step, "dropout:" +
    node path)` from the call (§3.1.12), and dropped otherwise. Each element is kept with
    probability `1 - p` (to within 2^-24).
  - `out = x · m`, in `f32`, where `m = s` for a kept element and `0` for a dropped one, and
    `s = 1 / (1 - p)` is computed as PyTorch's CPU kernel computes it: `1 - p` in `f64`, rounded to
    `f32`, then the division in `f32`. Because `out` is a product, NaN or `±inf` in `x` gives NaN
    at a dropped element, as in PyTorch.
  - The gradient is `g · m`, with the same mask.
  - Fixtures can't reproduce PyTorch's masks, which come from PyTorch's generator. The op's
    reference test therefore takes the mask as an input and checks output and gradient against
    PyTorch's `x * (mask / (1 - p))` (`mask` a float tensor of 0 and 1), which is what
    `F.dropout` computes for that mask. The masks themselves are tested statistically and against
    the known values of §3.1.12 (testing-strategy.md).
- Unbatched inputs (PyTorch's conv/pool on rank 2 or 3 without `N`) are not supported; `unsqueeze`.

#### Normalization

| Op | Inputs | Params | Attributes | Outputs | Shape | Diff. | PyTorch |
|---|---|---|---|---|---|---|---|
| `nn.batch_norm` | `x`: `C`, `[N, C, …]`, rank 2–6 | `running_mean` `[C]`, `running_var` `[C]`, `weight?` `[C]`, `bias?` `[C]` | `eps: float = 1e-5` (> 0), `momentum: float = 0.1` (`0 ≤ momentum ≤ 1`) | `out`: `C` | `x.shape` | yes (`x`, `weight`, `bias`) | `F.batch_norm(x, running_mean, running_var, weight, bias, training=t, momentum=momentum, eps=eps)`, `t` = training mode |
| `nn.layer_norm` | `x`: `C` | `weight?` `normalized_shape`, `bias?` `normalized_shape` | `normalized_shape: int[]?` (1…`r` entries), `eps: float = 1e-5` (> 0) | `out`: `C` | `x.shape` | yes | `F.layer_norm(x, normalized_shape, weight, bias, eps)` |
| `nn.group_norm` | `x`: `C`, `[N, C, …]`, rank 2–6 | `weight?` `[C]`, `bias?` `[C]` | `num_groups: int` (≥ 1, divides `C`), `eps: float = 1e-5` (> 0) | `out`: `C` | `x.shape` | yes | `F.group_norm(x, num_groups, weight, bias, eps)` |
| `nn.rms_norm` | `x`: `C` | `weight?` `normalized_shape` | `normalized_shape: int[]?` (1…`r` entries), `eps: float` (> 0) | `out`: `C` | `x.shape` | yes | `F.rms_norm(x, normalized_shape, weight, eps)` |

- `batch_norm` in inference mode normalizes with the running statistics, per channel:
  `out = (x - running_mean) / sqrt(running_var + eps) · weight + bias`. Nothing is updated, and
  `momentum` has no effect.
- `batch_norm` in training mode (§3.1.10) is PyTorch's `F.batch_norm(…, training=True,
  momentum=momentum)`:
  - `n` is the number of values per channel: `N` times the product of the dims after `C`. `n ≥ 2`
    is required. With `n = 1` the call fails with `InvalidInput` naming the node, checked with the
    bound shapes before any node runs ([runtime.md §3](runtime.md#3-calls)). PyTorch raises too
    ("Expected more than 1 value per channel when training"). Validation can't check it, because
    it doesn't know the mode.
  - It normalizes with the batch statistics over every dim except `C`: the mean `mean_c` and the
    biased variance `var_c = mean((x - mean_c)²)`.
    `out = (x - mean_c) / sqrt(var_c + eps) · weight + bias`. The gradient flows through `mean_c`
    and `var_c` to `x`, as in PyTorch.
  - It doesn't read the running statistics. The call returns their new values
    ([runtime.md §3.3](runtime.md#33-running-statistics)):
    `running_mean' = (1 - momentum) · running_mean + momentum · mean_c` and
    `running_var' = (1 - momentum) · running_var + momentum · var_c · n / (n - 1)` (the unbiased
    variance). They are computed in `f32` from values without autodiff tracking, with `momentum`,
    `1 - momentum` and `n / (n - 1)` computed in `f64` and rounded once to `f32`. `momentum = 0`
    keeps the running statistics, `1` replaces them with the batch statistics.
  - PyTorch's `momentum=None` (a cumulative average, with the `num_batches_tracked` counter) is not
    supported. PyTorch also accepts a `momentum` outside `[0, 1]`, which extrapolates; here that is
    `invalid_attr`.
- `running_mean` and `running_var` are params in both modes (owner decision 7 for M2), so PyTorch's
  `state_dict` names and the M1 weights keep working. They are never differentiable and never
  trained, whatever `trainable` says (§2.4). In training mode the caller writes the returned values
  back ([training.md](training.md)). A training-mode call requires each of them to have exactly one
  use ([runtime.md §3.1](runtime.md#31-mode)).
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

- NaN propagates through every activation. Infinities follow the formulas: `silu`, `mish`, `gelu`
  and `gelu_tanh` of `-inf` are NaN (`-inf · 0`), as in PyTorch. `gelu(+inf)` is `+inf` (PyTorch's
  `f64` kernel agrees; its `f32` CPU kernel gives NaN).

### 3.4 Planned ops (not M1)

Not in the M1 catalog. A graph that uses them fails validation with `unknown_op`.

- `core`: `pad` modes `reflect`, `replicate` and `circular`; integer division and remainder;
  reading a param as a value (for example a learned position embedding added with `core.add`);
  `any`/`all`.
- `nn` (M2 or later, as needed): `conv_transpose2d`, `upsample_nearest`, `upsample_bilinear`.

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
| `invalid_value` | A field has a value outside its allowed set (a bad dtype name, `float` where it isn't allowed, an unknown `init` kind, an `init` value that isn't finite or usable (§2.4), `min`/`max` of a dim, an empty shape, a literal dim above 2^32, an expression where §2.3 allows only literals or plain symbols, no `outputs`), or a list exceeds its §4.3 limit (`dims`, inputs, outputs, state entries, the references of a variadic input: 1–1,024) |
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
| `invalid_init` | A fan-based `init` kind (`kaiming_*`, `xavier_*`) on a param that no node uses, that a node uses in a role without fans, or whose uses give different fans (§2.4.1). The message says which, and names the node and role |
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

These are not validation codes; the runtime raises them as `polypous::Error` kinds (the full list
of runtime errors, with weight loading and cancellation, is [runtime.md §5](runtime.md#5-errors)):

| Kind | When |
|---|---|
| `InvalidInput` | A binding check of §2.3.4 fails (missing or extra input, wrong dtype or rank, inconsistent symbol, out of `min`/`max`, an inferred dim < 1, overflow), an index is out of range (§3.1.8), or a training-mode `nn.batch_norm` has one value per channel (§3.3) |
| `UnsupportedOperation` | The device doesn't support a dtype or (op, dtype) pair of the graph, or the compute dtype (§2.2); or a training-mode call has a compute dtype other than `f32`, or a running statistic with more than one use ([runtime.md §3.1](runtime.md#31-mode)) |

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
