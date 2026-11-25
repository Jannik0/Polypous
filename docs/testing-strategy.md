# Testing Strategy

**Decision:** [ADR-0009](adr/0009-reference-based-testing.md). Correctness is priority #1. Every
numerical result is checked against an established reference implementation.

## 1. Test levels

| Level | What | Where it runs | Gate |
|---|---|---|---|
| **Unit** | Parsers, validation, shape inference, planner logic, expression evaluator | CI, every platform | Every PR |
| **Op reference** | Each IR op forward (and gradient) vs PyTorch fixtures | CPU in CI; CUDA/Vulkan on T1 | Every PR (CPU); before merge for backend changes (GPU) |
| **Kernel reference** | Quant dequant (bit-exact), qmatvec/qmatmul vs f64 | CPU in CI; GPU on T1 | Same |
| **Component reference** | Tokenizers, chat templates, mel spectrogram, schedulers vs HF/librosa/diffusers | CI | Every PR |
| **Training reference** | Optimizers, LR schedules, gradient clipping and losses vs PyTorch (§3.7); the random generator, initial values, dropout masks and shuffles: known values and statistics; bit-identical resume (§3.8) | CI | Every PR |
| **Model reference** | Small real models end-to-end vs transformers / llama.cpp; the MNIST training run (§3.8) | T1, nightly or on demand; MNIST also on the CPU | Milestone exits |
| **Scenario** | v1 acceptance scenarios (placement, streaming, CPU-only, builder flow) | T1 | Milestone exits, releases |
| **Fuzz / robustness** | GGUF/safetensors/`.polyp`/graph parsers with malformed input | CI (short), local (long) | Weekly / before release |
| **Placement estimate** | Predicted vs measured peak memory (±25%) | T1 | M4 exit, releases |

## 2. Tolerances (defaults)

| Comparison | Tolerance |
|---|---|
| Integer, shape, token ids, tokenizer output | exact |
| Dequantization | bit-exact (f32) |
| f32 op vs PyTorch f32 | `atol = 1e-5`, `rtol = 1e-4` (reductions/matmul: `rtol = 1e-3`) |
| f16/bf16 compute | `rtol = 1e-2` |
| Gradients, f32 (op fixtures, loss gradients; §3.6, §3.7) | `atol = 1e-5`, `rtol = 1e-3` |
| Optimizer steps, f32: params and state after each of 3 steps (§3.7) | `atol = 1e-6`, `rtol = 1e-6`, against PyTorch's own f32 result |
| Loss values (§3.7) | as f32 reductions: `atol = 1e-5`, `rtol = 1e-3` |
| Gradient clipping: total norm and clipped gradients (§3.7) | as f32 reductions: `atol = 1e-5`, `rtol = 1e-3` |
| Learning-rate schedules (§3.7) | exact (`constant`, `step`, `linear_warmup`); `cosine`: `\|actual − expected\| ≤ 1e-14 · lr` |
| Metrics (`accuracy` counts, the `loss` metric's sums) | exact |
| Random generator, dropout masks, shuffles, uniform-based initial values (§3.8) | bit-exact |
| Normal-based initial values (`normal`, `kaiming_normal`, `xavier_normal`; §3.8) | 1 ulp (f32) |
| MNIST training: the first 10 training losses and one inference loss vs PyTorch (§3.8) | `rtol = 1e-3` |
| Training resumed from a checkpoint vs uninterrupted, CPU (§3.8) | bit-exact |
| Model end-to-end, f32 compute (MLP, small CNN; §3.5) | `atol = 1e-5`, `rtol = 1e-3` |
| Model end-to-end, f16 compute | `atol = 2e-3`, `rtol = 1e-2` |
| Model end-to-end, bf16 compute | `atol = 2e-2`, `rtol = 1e-2` |
| LLM end-to-end | Greedy decoding: first 32 tokens identical to the reference for fixture prompts; logits of the first step: cosine similarity ≥ 0.999 and top-5 overlap = 5 |
| Whisper | Greedy transcript text identical on fixture clips |
| Diffusion | Latents after N steps at fixed seed: cosine similarity ≥ 0.99. Final image PSNR ≥ 30 dB vs diffusers |

The reductions/matmul class covers every op that sums over many elements: `core.sum mean cumsum matmul
softmax log_softmax`, `nn.linear`, `nn.conv1d`/`conv2d` and the normalization ops. Activations, pools and
`nn.dropout` use the elementwise `rtol`.

Op-specific tolerances may differ and are documented next to the test. Loosening a tolerance requires
an explanation in the PR.

**Model end-to-end tolerances.** A model output is compared with the module in f64 on the same weights
and inputs, rounded once to the compute dtype (§3.5). The models are chains of the reductions/matmul
class (`nn.linear`, `nn.conv2d`, `nn.batch_norm`, `nn.adaptive_avg_pool2d`, `log_softmax`) and exact or
elementwise ops (`relu`, `max_pool2d`, `flatten`), with outputs of scale ≤ 1 (logits up to 0.6,
log-probabilities up to 2.6 in magnitude):

- **f32** uses the reductions/matmul class unchanged. Each op meets it on its own, and the chain stays
  far inside it: PyTorch's own f32 result is within 1e-7 of the reference (worst element at 0.001 of
  the allowed error). The margin is for other summation orders and kernels (GPU tiling, Winograd
  convolution), not for lower precision: f32 matmul on TF32 tensor cores misses it by 1.5–5×
  (generator `--report`, info only), which is the known CUDA limitation of [status.md §4](status.md#4-known-limitations-and-open-items).
- **f16/bf16** keep the class's `rtol = 1e-2`, which covers the final rounding (at most one ulp, `2u`,
  between a device's rounding and the reference's; `u` is the unit roundoff, `2^-11` for f16 and
  `2^-8` for bf16). Unlike a single op, a model rounds every intermediate value to the compute dtype
  (graph-ir.md §3.1.8), so an output that sums many rounded terms has an absolute error of order `u`
  times the size of the terms, however close to zero the sum is. `atol = 4u` rounded up (`2e-3` for f16,
  `2e-2` for bf16) covers that for outputs of scale ≤ 1. Measured: PyTorch's own f16/bf16 result and an
  emulation of the runtime's numerics (every layer in f64, rounded to the dtype, subnormals flushed to
  zero) have the same worst errors, at most about `u` absolute (f16: 2.4e-4 on the CNN logits; bf16:
  3.9e-3), so the worst element uses at most 0.08 (f16) and 0.37 (bf16, a log-probability one ulp off)
  of the allowed error. A model with larger outputs needs its own `atol`, scaled to its output.

The generator checks both computations against every case's tolerance before writing it
(`tools/reference/gen_models.py`, `--report` prints the numbers).

**Gradient tolerance.** Gradients use the f32 reductions/matmul class for every op, `atol = 1e-5` and
`rtol = 1e-3`: a backward pass sums over every broadcast dim, every use of a value and every output a
value contributes to (the gradient of an elementwise op on a broadcast input is a reduction), and the
summation order of backward kernels is not specified. The `atol` is the f32 op one: gradients that
should be zero, or that cancel, have an absolute error of order the f32 unit roundoff times the size of
the summed terms (≤ 1 in the fixtures). Expected gradients are computed in f64 and rounded once to f32,
and the generator checks that PyTorch's own f32 gradient meets the tolerance (§3.3, §3.6). Gradients are
f32 only in M2 (graph-ir.md §3.1.11).

**Optimizer tolerance.** Polypous's optimizers run PyTorch's single-tensor algorithm as separate `f32`
`DynTensor` ops (training.md §3.1). PyTorch's CPU kernels fuse `add(alpha=…)`, `addcmul`, `addcdiv` and
`lerp` with FMA, so the two round differently; near zero, where an update cancels the param, the
difference is large in ulps but tiny in absolute terms. The spike (#71) measured at most 1 ulp (Adam,
AdamW) and 4 ulp (SGD with momentum and Nesterov) on one small case. For the tolerance, an emulation of
training.md §3.1's op sequence without FMA (NumPy, f32) was compared with `torch.optim` 2.15.0
(`foreach=False`) on 300 random cases of 256 elements per optimizer, 3 steps, params and gradients from
a standard normal, `lr` 0.1 (SGD) and 0.01 (Adam, AdamW): with `rtol = 1e-6`, the largest `atol` any
element of the params and state needed was `1.7e-7` (SGD with momentum and dampening; Adam and AdamW
`2.6e-8`). `atol = 1e-6`, `rtol = 1e-6` leaves a margin of about 6×, and is far tighter than the error
of a wrong formula (Burn's own Adam missed PyTorch by 2.4e-6 relative after 3 steps, ADR-0014). The
expected values are PyTorch's own f32 results, not an f64 computation (an exception to §3.3): both
implementations are defined as the same f32 op sequence, and an f64 value would be neither. Fixtures
keep params and gradients of scale ≤ 5 and `lr ≤ 0.1`; a case outside that needs its own, explained
tolerance.

**Schedule tolerance.** Schedules are `f64` formulas evaluated in a fixed order (training.md §3.2), so
`constant`, `step` and `linear_warmup` equal PyTorch's `StepLR`/`LambdaLR` values bit for bit (checked
over 1,000 steps). `cosine` uses the platform's `cos`, which IEEE 754 doesn't require to round
correctly; it equals `CosineAnnealingLR`'s closed form exactly on Linux, and PyTorch's chainable
schedulers agree with it within `1.1e-15 · lr`. `1e-14 · lr` allows for another platform's `cos`.

## 3. Reference fixtures

- Generated by scripts in [`tools/reference/`](../tools/reference/) using PyTorch, transformers,
  diffusers, `gguf-py`, librosa and llama.cpp. Each script is deterministic (fixed seeds) and records
  the library versions it used in the fixture's metadata.
- **Format:** safetensors (tensors) + a JSON sidecar (inputs, attributes, tolerances, versions).
- **Small fixtures** (< 1 MiB each, < 50 MiB total) are committed in `fixtures/small/`.
- **Large fixtures and test models** are listed in `fixtures/manifest.json` (URL, SHA-256, size,
  license). `tools/reference/fetch_fixtures.py` downloads them into `fixtures/cache/` (git-ignored).
  Tests that need them skip with a clear message when they are missing. In CI they are a failure, not a skip.
- Test models are the smallest representative models available (for example tiny Llama-architecture
  test models, Whisper tiny, SD 1.5 at low resolution and few steps).

### 3.1 Layout

```text
fixtures/
  manifest.json                    large fixtures and test models (§3.4)
  cache/                           downloads from the manifest, git-ignored
  small/                           committed, size-checked in CI
    ops/<op>/<case>.safetensors    one op case: inputs and expected outputs
    ops/<op>/<case>.json           its sidecar (§3.2)
    models/<model>/model.graph.json      an end-to-end model's graph (§3.5)
    models/<model>/model.safetensors     its weights
    models/<model>/<case>.safetensors    one model case: inputs and expected outputs
    models/<model>/<case>.json           its sidecar (§3.5)
    grads/<op>/<case>.safetensors  one gradient case of an op (§3.6)
    grads/<op>/<case>.json         its sidecar (op sidecar schema version 2)
    train/<component>/<case>.safetensors   one optimizer, clipping or loss case (§3.7)
    train/<component>/<case>.json          its sidecar (§3.7); schedule cases have only the sidecar
    init/<kind>/<case>.safetensors   the initial values of one param (§3.9)
    init/<kind>/<case>.json          its sidecar (§3.9)
    training/<model>/model.graph.json    a training-mode graph (§3.10)
    training/<model>/model.safetensors   its weights, given to the call as params
    training/<model>/<case>.safetensors  one training-mode call: inputs, expected outputs, running statistics, masks, gradients
    training/<model>/<case>.json         its sidecar (model case schema version 2, §3.10)
```

- `<op>` is the IR op name from [graph-ir.md §3](specs/graph-ir.md#3-opsets), for example
  `fixtures/small/ops/core.add/broadcast_row_f32.json`. `<case>` is `[a-z0-9_]`, unique per op,
  and usually ends in the dtype under test.
- A **fixture** is one case: its `.safetensors` file and its `.json` sidecar. The size budget counts
  both together. A model's `model.graph.json` and `model.safetensors` are one fixture too: files
  belong to the same fixture when their paths agree up to the first `.` of the file name.
- `<model>` is `[a-z0-9_]`, for example `fixtures/small/models/mlp/batch4_f32.json`. Its cases are
  usually named `batch<N>_<dtype>`.
- Other areas (`quant/`, `tokenizers/`, …) get their own directories under `small/` when they get
  their first fixtures, with the same pairing of tensors and sidecar.

### 3.2 Op sidecar (schema version 1)

The safetensors file holds every input, every param and every expected output under the names the
sidecar gives, **in the dtype under test** (`F32`, `F16`, `BF16`, `I64`, `I32`, `BOOL`;
little-endian, row-major) and with no `__metadata__`. safetensors stores raw bytes, so `-0.0`, NaN,
infinities and f16/bf16 values round-trip exactly. The sidecar is UTF-8 JSON with exactly these
keys, in this order (`params` only for ops that have params; 2-space indent, LF line endings,
written by the generator only):

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `op` | string | IR op name, equal to the directory name |
| `case` | string | Case name, equal to the file name without extension |
| `description` | string | What the case covers |
| `attributes` | object | The op's attributes (for example `{"dtype": "f16"}` for `core.cast`, `{"shape": [4, 6]}` for `core.reshape`). Names follow the op catalog; the generator is the only place that writes them |
| `inputs` | array | The op's inputs **in order**: `{"name", "dtype", "shape"}` with dtype `f32`/`f16`/`bf16`/`i64`/`i32`/`bool` |
| `params` | array | Only for ops with params ([graph-ir.md §2.4](specs/graph-ir.md#24-parameters), the op catalog), and then always (possibly empty): the params that are present, **in catalog order**, same form as `inputs`. Each holds the param in the compute dtype the case tests. An absent optional param is left out, and the case name says so (`no_bias_f32`). Ops without params have no `params` key |
| `outputs` | array | Expected outputs in order, same form. Tensor names are unique across inputs, params and outputs |
| `tolerance` | object | `{"exact", "atol", "rtol"}`, see below |
| `reference` | string | The reference expression, for example `torch.add(a, b)` |
| `notes` | string | Semantics the case relies on and inputs left out on purpose (may be empty) |
| `seed` | int | Seed of the case's input data |
| `generator` | string | Script that wrote it, relative to the repository root |
| `versions` | object | Versions it was generated with: `python` (major.minor), `torch`, `safetensors`, `numpy` |

**Comparing with the tolerance.** With `exact: true` (then `atol` and `rtol` are `0`), the output
must equal the expected tensor bit for bit, except that any NaN matches any NaN (NaN payloads and
signs are not part of op semantics). Otherwise every element must satisfy
`|actual − expected| ≤ atol + rtol · |expected|`, a NaN is expected exactly where the expected
tensor has one, and an expected ±inf must be the same inf. Tolerances default to §2 (f32:
`atol 1e-5`, `rtol 1e-4`, matmul `rtol 1e-3`; f16/bf16: `rtol 1e-2`; ints, bool, reshape and cast:
exact) and are written into every sidecar, so a test reads them from there and never hard-codes them.

### 3.3 Generating and checking

- Each generator declares all its cases in one place, each with a fixed seed. Input data comes from
  NumPy's PCG64 generator, so it is the same on every CPU and OS. Where a library's result depends
  on the CPU's kernels (matmul summation order), the generator computes in f64 and rounds once to the
  dtype under test, and checks that the library's own result in that dtype meets the tolerance.
- Generators need the exact versions pinned in `tools/reference/requirements.txt` and refuse to run
  with others. `--check` regenerates into a temporary directory and fails on a fixture that differs
  from the committed one, a generated fixture that is not committed (missing), and a committed file
  that no case produces (extra). See [`tools/reference/README.md`](../tools/reference/README.md).
- CI runs `tools/ci/check_fixtures.py` (standard library only): the size budget, and every sidecar
  against the schema and its safetensors header. CI does not run PyTorch.

**How tests find fixtures.** Paths are relative to the repository root: Rust tests build them from
`CARGO_MANIFEST_DIR` (`../../fixtures/small/ops/<op>/`), read every `.json` in the op's directory,
load the safetensors file next to it, and compare with the sidecar's tolerance. A missing committed
fixture is a test failure, never a skip. The Rust loader comes with the first backend op task, through
`polypous-formats`.

### 3.4 Manifest

`fixtures/manifest.json` is `{"schema_version": 1, "entries": [...]}`. Every entry has exactly
`name` (unique, `[A-Za-z0-9._-]`), `url` (`https`, or `file` for local mirrors), `sha256` (lowercase
hex), `size` (bytes), `license` and `path` (relative, `/`-separated, inside `fixtures/cache/`).
`fetch_fixtures.py` validates the whole manifest first, never reads more than `size` bytes of a
download (or `--max-bytes`, default 8 GiB, for any entry), writes to a temporary file, checks the
SHA-256 and only then moves the file into place. A file already in the cache with the right hash is
kept.

### 3.5 Model fixtures (schema version 1)

End-to-end reference tests for whole models (roadmap M1 exit: "an MLP and a small CNN defined as JSON
give the same outputs as identical PyTorch models with the same weights"). Generated by
[`tools/reference/gen_models.py`](../tools/reference/gen_models.py) with the rules of §3.3.

- **`model.graph.json`**: the graph ([graph-ir.md](specs/graph-ir.md)), written by the generator from
  the PyTorch module's structure (one node per layer, node ids equal to the layer names), with the
  batch dim as the symbol `batch`, so one loaded model runs every batch size. Every param is named
  exactly like the PyTorch `state_dict` key (`fc1.weight`, `bn.running_mean`), in PyTorch's layout
  (graph-ir.md §3.3), so PyTorch's weights load unchanged.
- **`model.safetensors`**: exactly the graph's params (the runtime's strict match, runtime.md §2), stored
  as `F32`, no `__metadata__`. State that inference doesn't read is left out (BatchNorm's
  `num_batches_tracked`). Weights have no values below the smallest normal f16 (`2^-14`) other than zero,
  since the f16 cases round them to f16 (runtime.md §2 step 5) and subnormals are backend-defined.
- **Cases**: `<case>.safetensors` holds the graph inputs and the expected outputs under their graph
  names, in the case's compute dtype (names unique across both). The sidecar is UTF-8 JSON with exactly
  these keys, in this order (2-space indent, LF line endings, written by the generator only):

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `model` | string | Model name, equal to the directory name |
| `case` | string | Case name, equal to the file name without extension |
| `description` | string | The model and what the case covers (batch size, dtype) |
| `compute_dtype` | string | `f32`, `f16` or `bf16`: the `GraphLoadOptions::compute_dtype` to load the model with. `float` inputs are stored in it |
| `inputs` | array | The graph inputs **in graph order**, `{"name", "dtype", "shape"}` as in §3.2 |
| `outputs` | array | The graph outputs **in graph order**, same form |
| `tolerance` | object | `{"exact", "atol", "rtol"}` (§3.2), from the model rows of §2 |
| `reference` | string | How the expected outputs were computed |
| `notes` | string | Semantics and choices the case relies on |
| `seed` | int | Seed of the case's input data |
| `generator` | string | Script that wrote it, relative to the repository root |
| `versions` | object | As in §3.2 |

**References.** The module runs in f64 on the weights and inputs **as the runtime holds them**: the f32
weights (for an f16/bf16 case, rounded once to that dtype first) and the inputs as stored in the case
file. The outputs are rounded once to the compute dtype. Weights and inputs come from NumPy's PCG64 with
fixed seeds; non-trivial state (BatchNorm running statistics and affine params) is set by the generator,
never left at PyTorch's defaults. Inputs and outputs have no subnormal values.

**Comparing** follows §3.2. Tests load each graph through the facade (`Graph::from_file`,
`Engine::load_graph` with `model.safetensors` and the case's compute dtype), run every case on every
available device, skip a compute dtype a device doesn't support with a `SKIP` line, and print a
`SUMMARY <model> <device> <dtype>` line ([`crates/polypous/tests/models.rs`](../crates/polypous/tests/models.rs)).
`crates/polypous-cli/tests/models.rs` runs one case through `polyp run`.

**Checks.** `tools/ci/check_fixtures.py` checks every model directory: the graph lists `inputs`,
`outputs` and `params`; `model.safetensors` holds exactly the params with their declared shapes as
`F32`; every case sidecar follows this schema, names the graph's inputs and outputs in order, stores
`float` inputs in the compute dtype and matches its safetensors header; no float tensor holds a
subnormal; no other files. Full graph validation is `polypous-ir`'s: the Rust tests load every graph.

### 3.6 Gradient fixtures (op sidecar schema version 2)

Gradient cases of the ops marked differentiable (graph-ir.md §3, "Diff."; conventions in §3.1.11), and
the training-mode forward passes of `nn.batch_norm` and `nn.dropout` (graph-ir.md §3.3). They live in
`fixtures/small/grads/<op>/<case>.{json,safetensors}`, apart from the forward cases, so the forward
harness and its schema stay as they are. `<case>` follows §3.1 and ends in `_f32` (gradients are f32
only in M2).

The safetensors file holds every input, param, mask and upstream gradient, every expected forward
output and running statistic, and every expected gradient, under the names the sidecar gives, as in
§3.2. The sidecar is schema version 2 of the op sidecar: §3.2's keys plus five, with exactly these
keys in this order (keys marked "only" are left out otherwise):

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `op`, `case`, `description` | string | As in §3.2 |
| `mode` | string | `"inference"` or `"training"` (only `nn.batch_norm` and `nn.dropout` have training cases) |
| `attributes`, `inputs` | | As in §3.2 (integer and bool inputs included; they get no gradient) |
| `params` | array | As in §3.2, only for ops with params |
| `mask` | object | Only for `nn.dropout` in training mode: `{"name": "mask", "dtype": "f32", "shape"}`, the keep mask (`1` kept, `0` dropped) with `x`'s shape. The op's test feeds it instead of a generated mask (graph-ir.md §3.3) |
| `upstream` | array | The upstream gradient of every float output port, in port order: `{"name": "grad_<port>", "port", "dtype", "shape"}` (`grad_out` for the usual single port) |
| `outputs` | array | The expected forward outputs, as in §3.2 |
| `running_stats` | array | Only for `nn.batch_norm` in training mode: `{"name": "new_running_mean", "param": "running_mean", …}`, then `new_running_var`: the updated running statistics the call returns (runtime.md §3.3) |
| `gradients` | array | The expected gradient of every differentiable float input, in input order, then of every differentiable param, in `params` order: `{"name": "grad_<input or param>", "of", "dtype", "shape"}`. `batch_norm`'s running statistics have none |
| `tolerance` | object | For `outputs` and `running_stats`: §3.2, from §2's op rows |
| `gradient_tolerance` | object | For `gradients`: `{"exact", "atol", "rtol"}` from §2's gradient row |
| `reference` | string | The PyTorch expression, for example `torch.autograd.grad(F.linear(x, weight, bias), (x, weight, bias), grad_out)` |
| `notes`, `seed`, `generator`, `versions` | | As in §3.2 |

- **Names.** No op input or param name in the catalog starts with `grad_` or `new_`, and `mask` is no
  input or param name, so names stay unique across the file.
- **Expected values.** The generator runs the reference in f64 on the f32 inputs, params and upstream
  gradients and rounds once to f32 (§3.3), and checks that PyTorch's own f32 result meets `tolerance`
  and `gradient_tolerance`. Tie and kink cases (graph-ir.md §3.1.11) use exactly representable values,
  so f64 and f32 agree on which convention applies. Cases cover broadcasting, every use of a value,
  the conventions of graph-ir.md §3.1.11, conv and pool `groups`/`stride`/`padding`/`dilation`, and
  `batch_norm` in training mode.
- **Dropout** cases compute `x * (mask / (1 - p))` (the mask as an f32 tensor, graph-ir.md §3.3).
- **Comparing** follows §3.2: outputs and running statistics with `tolerance`, gradients with
  `gradient_tolerance`. A test may back-propagate a one-element loss `Σ out · grad_out` (summed over
  every float port), whose gradient with respect to each output is exactly its upstream gradient.
- **Checks.** `tools/ci/check_fixtures.py` validates this schema like §3.2's (keys and their order, names,
  dtypes and shapes against the safetensors header, `mask` and `running_stats` only where allowed).

### 3.7 Training component fixtures (schema version 1)

Fixtures for the optimizers, gradient clipping, losses and schedules of training.md §3, in
`fixtures/small/train/<component>/<case>.{json,safetensors}`. `<component>` is `optimizer.sgd`,
`optimizer.adam`, `optimizer.adamw`, `gradient_clip`, `loss.mse`, `loss.cross_entropy`,
`loss.binary_cross_entropy` or `schedule.<kind>`. The tensors follow §3.2's rules (all `F32`, class
indices `I64`). The sidecar has exactly these keys, in this order:

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `component`, `case`, `description` | string | As `op`, `case`, `description` in §3.2 |
| `config` | object | The component's config object exactly as in training.md §2: the `optimizer` object, `{"max_norm": …}`, the `loss` object without `prediction`/`target`, or `{"lr": …, "total_steps": T, "schedule": {…}}` |
| `inputs` | array | `{"name", "dtype", "shape"}`, below. Empty for schedules |
| `outputs` | array | The expected tensors, same form. Empty for schedules |
| `values` | array | Schedules only: the expected `lr(k)` for `k = 0 … T−1`, JSON numbers in shortest round-trip form |
| `tolerance` | object | `{"exact", "atol", "rtol"}` from §2 (for `cosine`: `{"exact": false, "atol": 1e-14 · lr, "rtol": 0}`) |
| `reference` | string | The PyTorch call, for example `torch.optim.AdamW([p], lr=0.01, weight_decay=0.1, foreach=False)` |
| `notes`, `seed`, `generator`, `versions` | | As in §3.2 |

| Component | `inputs` | `outputs` |
|---|---|---|
| `optimizer.*` | `param` (the initial value), `grad_1`, `grad_2`, `grad_3` (the gradient given at each step) | after each step `s = 1, 2, 3`: `param_<s>` and every state tensor `<state>_<s>` (`momentum_buffer_2`, `exp_avg_sq_3`; training.md §5.2's names) |
| `gradient_clip` | `grad_0`, `grad_1`, … (the gradients of several params) | `total_norm` (`[1]`), `clipped_0`, `clipped_1`, … |
| `loss.*` | `prediction`, `target` | `loss` (`[1]`), `grad_prediction` (the gradient of the loss, upstream gradient 1) |

- **Optimizer cases** run 3 steps from identical state with a constant learning rate, and compare the
  params and every state tensor after each step. Their expected values are PyTorch's own f32 results
  (§2, "Optimizer tolerance"). Cases cover weight decay, `momentum` with and without `dampening`,
  Nesterov, and the default and non-default `betas` and `eps`.
- **Loss cases** use f64 references rounded once (§3.3), and cover large logits (the stable
  formulations), `label_smoothing` 0, 0.1 and 1, `i32` and `i64` class indices, BCE probabilities of
  exactly 0 and 1 (the `−100` clamp) and logits of ±200.
- **Schedule cases** have no tensors and no safetensors file; `values` comes from PyTorch's scheduler
  stepped once per optimizer step (training.md §3.2's reference column), covering the warmup, a
  `step` boundary and the last step.
- **Metrics** (`accuracy`, the `loss` metric's weighting) are exact integer and `f64` computations; they
  are tested with Rust unit tests on hand-written cases (ties, NaN, a smaller last batch), not fixtures.
- `tools/ci/check_fixtures.py` validates this schema; each task that adds a component (#79, #80) extends it.

### 3.8 Training tests

**Generator, known values.** Implementations of graph-ir.md §3.1.12 are tested against its table of
known values (bit-exact), and the shuffle and split against training.md §3.6's table.

**Generator, statistics.** With fixed seeds these tests are deterministic; their thresholds catch
implementation bugs (a wrong shift, a reused counter), not bad luck. Each statistic must lie within 5
standard errors of its expectation; a chi-square statistic with `d` degrees of freedom must be at most
`d + 5·sqrt(2d)`. Sample sizes and the values the Python implementation of the spec gives (#73):

| Test | Sample | Pass if | Python value |
|---|---|---|---|
| `u24` uniformity: 256 equal buckets | 2^20 draws of `stream_key(42, 7, "dropout:layers[3]/drop")` | chi-square ≤ 367.9 (255 dof) | 241.8 |
| `u24` mean | the same | `|mean − 0.5| ≤ 5 · sqrt(1/12/n)` | 0.73 SE |
| Dropout keep rate, `p` ∈ {0.1, 0.25, 0.5, 0.9} | the same draws, rule of graph-ir.md §3.3 | kept fraction within 5 SE of `1 − p` (`SE = sqrt(p(1−p)/n)`) | ≤ 0.58 SE |
| Dropout independence across steps | `p = 0.5`, steps 7 and 8 of the same node | fraction kept in both within 5 SE of 0.25 | 0.63 SE |
| `N`: mean, variance, skewness, excess kurtosis | 2^19 values of `stream_key(42, 0, "init:fc1.weight")` | each within 5 SE (`sqrt(1/n)`, `sqrt(2/n)`, `sqrt(6/n)`, `sqrt(24/n)`) | ≤ 1.33 SE |
| Shuffle uniformity | `n = 4`, 24,000 epochs of seed 7 | chi-square over the 24 permutations ≤ 56.9 (23 dof) | 20.2 |

The dropout tests also check that kept elements equal `x · s` exactly and that the masks of two node
paths and of two steps differ.

**Initial values.** Every `init` kind (graph-ir.md §2.4.1) on params of each fan role, against values
the reference generator computes from the spec's formulas (a Python implementation in `tools/reference/`,
f64 rounded once; fixtures in §3.9): `zeros`, `ones`, `constant`, `uniform`, `kaiming_uniform` and `xavier_uniform`
bit-exact; `normal`, `kaiming_normal` and `xavier_normal` within 1 ulp, because their `ln` and `cos`
come from the platform's math library (graph-ir.md §3.1.12). The table of graph-ir.md §3.1.12 is one of
the cases.

**Resume** (training.md §5.4, in CI on the CPU): a small model with `nn.batch_norm` and `nn.dropout`,
trained with AdamW, a cosine schedule with warmup, gradient clipping, shuffling, a `split` validation
set and `every_steps` checkpoints, for 3 epochs: (a) uninterrupted; (b) cancelled in the middle of an
epoch and continued with `Engine::resume_trainer` from the checkpoint written on cancel; (c) cancelled
at an epoch boundary and continued by calling `run` again. The final params, optimizer state, `history`
and `best`, and the bytes of the final checkpoint's `weights.safetensors` and `optimizer.safetensors`,
must be identical in all three.

**MNIST** (roadmap M2 exit; #84). An MNIST-sized CNN (`conv2d` → `relu` → `conv2d` → `batch_norm` →
`relu` → `max_pool2d` → `dropout` → `flatten` → `linear` → `relu` → `dropout` → `linear`), a JSON graph
and a JSON config (AdamW, cross-entropy, accuracy), both committed as fixtures; the data is MNIST
converted to safetensors and fetched through the manifest (§3.4; owner decision 9 for M2).

- **Accuracy:** trained on the CPU through the facade and through `polyp train`, with the MNIST test
  set as the validation set, `val.accuracy` reaches at least 0.98 within 5 epochs.
- **Reference:** the same graph with every `dropout` at `p = 0` (the identity, graph-ir.md §3.3), the
  same config with `shuffle: false` and a constant learning rate, and initial weights from a
  safetensors file that the generator writes (given as `TrainerOptions::weights`, and loaded into the
  PyTorch module). The first **10** training losses, and the inference-mode loss of a fixed batch of
  test images after step 10 (which reads the running statistics written back in each step), must match
  the reference within `rtol = 1e-3`. The reference is the PyTorch module run in f64 on the same f32
  weights and data; the generator checks that PyTorch's own f32 run meets the tolerance and prints both
  errors (as `gen_models.py --report`, §2).
- **Why 10 steps and `rtol = 1e-3`.** Training amplifies rounding differences step by step, so the
  error of a loss grows with the number of steps before it. Measured with this CNN (batch 64, AdamW
  `lr = 1e-3`, PyTorch 2.15.0, CPU) on synthetic 28×28 images in `[0, 1]` (10 class templates plus
  noise, not MNIST itself; #73): over the first 10 losses, PyTorch
  f32 differs from f64 by at most `5.3e-7` relative, and two f32 implementations (oneDNN and PyTorch's
  native convolutions) differ by at most `1.6e-5`; after 20 steps the latter reaches `3.6e-4`, after 50
  steps `1e-3`. SGD with momentum 0.9 diverges faster (`1e-4` after 10 steps, `1e-2` after 20). The
  first 10 steps already exercise the forward, backward, optimizer and running-statistics paths, and
  `rtol = 1e-3` leaves a margin of about 60× over the measured difference between f32 implementations
  (10× for the faster-diverging SGD), while a wrong gradient or optimizer formula changes the losses by
  far more after a few steps.
- **CI.** The run takes about 15 minutes on 4 vCPUs (spike, #71) and needs the downloaded data, so the
  tests are `#[ignore = "mnist"]`: CI doesn't run them; they run in the M2 exit check (#85) and with
  `cargo test -- --ignored`. The reference comparison takes 10 steps and may run in CI once the data is
  available there. GPU runs (CUDA, Vulkan) are part of the exit check: on a GPU the accuracy criterion
  applies, and the reference losses are reported but, for CUDA's f32 `matmul` on TF32 (#64), not
  required to meet the tolerance (owner decision 1 for M2).

### 3.9 Initial-value fixtures (schema version 1)

The reference values of the "Initial values" tests of §3.8, in `fixtures/small/init/<kind>/<case>.{json,safetensors}`.
`<kind>` is the `init` kind (graph-ir.md §2.4.1). Generated by
[`tools/reference/gen_training.py`](../tools/reference/gen_training.py) from
[`tools/reference/polyrng.py`](../tools/reference/polyrng.py), a Python implementation of graph-ir.md
§2.4.1 and §3.1.12 written from the spec (`f64`, rounded once to `f32`); it is itself tested against the
known values of §3.1.12 and the Python column of §3.8. PyTorch isn't involved: its `torch.nn.init` uses
PyTorch's generator, and Polypous's values differ by design (§2.4.1).

The safetensors file holds one tensor, `values`: the param's initial values, `F32`, with the param's
shape (row-major). The sidecar has exactly these keys, in this order:

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `1` |
| `kind` | string | The `init` kind, equal to the directory name |
| `case`, `description` | string | As in §3.2 |
| `seed` | int | The training seed (a `u64`) the values are drawn with |
| `param` | object | `{"name", "shape", "init", "role", "fans"}`: the param name (it keys the stream, `"init:" + name`), its literal shape, its `init` object exactly as in a graph, the use it has (`{"op", "param"}`, for example `{"op": "nn.conv2d", "param": "weight"}`, or `null` for an unused param), and its fans from that role (`{"fan_in", "fan_out"}` by graph-ir.md §2.4.1's table, or `null` for a role without fans) |
| `outputs` | array | `[{"name": "values", "dtype": "f32", "shape"}]` |
| `tolerance` | object | `{"ulp": n}`: the largest allowed distance in `f32` ulps. `0` (bit-exact) for `zeros`, `ones`, `constant`, `uniform`, `kaiming_uniform`, `xavier_uniform`; `1` for `normal`, `kaiming_normal`, `xavier_normal` (§2, graph-ir.md §3.1.12, "Exactness"), `0` allowed for them too |
| `reference`, `notes`, `generator`, `versions` | | As in §3.2 |

- **Cases** cover every kind on params of each fan role: the four fan-based kinds on the weights of
  `nn.linear`, `nn.embedding`, `nn.conv1d` and `nn.conv2d` (`kaiming_uniform` with `a = sqrt(5)`, and the
  table case of graph-ir.md §3.1.12, `kaiming_uniform` of `fc1.weight` `[128, 784]` with seed 42), and the
  other kinds on a `linear` weight, roles without fans (a bias, a `batch_norm` running statistic, a
  multi-dimensional `layer_norm` weight) and an unused param, plus `low = high`, `std = 0` and a `normal`
  whose values round to `±inf`.
- **Comparing.** The test computes the values with `polypous-train` from `seed`, `param.name`,
  `param.shape` and `param.init`, takes the fans from the IR's table for `param.role` (they must equal
  `param.fans`), and compares each element with `tolerance` (`±inf` must be the same infinity).
- **Checks.** `tools/ci/check_fixtures.py`: keys and their order, `kind` against the directory, the fans
  against the shape and role (a fan-based kind needs a role with fans), the tolerance of the kind, and one
  `F32` `values` tensor with the param's shape.

### 3.10 Training-mode graph fixtures (model case schema version 2)

Whole graphs in training mode against PyTorch (#78): the outputs, the new running statistics and the
gradient of every trainable param of a training-mode call (runtime.md §3.1–§3.3) on given params and
inputs. They follow the model fixtures of §3.5, extended: a directory
`fixtures/small/training/<model>/` with `model.graph.json`, `model.safetensors` (exactly the graph's
params, `F32`, checked as in §3.5) and cases whose sidecars are **model case schema version 2**. They live
apart from `models/`, so §3.5's inference harness and `gen_models.py --check` stay as they are. Generated by
[`tools/reference/gen_training.py`](../tools/reference/gen_training.py) with the rules of §3.3.

The case file holds the graph inputs, the expected outputs, the running statistics, the dropout masks,
the upstream gradients and the expected gradients, under the names the sidecar gives (`/` appears in no
graph name, so the prefixed names are unique). The sidecar has exactly these keys, in this order (keys
marked "training" only in training mode):

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | int | `2` |
| `model`, `case`, `description` | string | As in §3.5 |
| `compute_dtype` | string | `f32` (training mode and gradients are `f32` only in M2) |
| `mode` | string | `"training"` or `"inference"` (the runtime's call mode) |
| `training` | object | Training: `{"seed", "step"}`, the call's `seed` and `step` (`u64`) |
| `inputs`, `outputs` | array | As in §3.5: the graph inputs and outputs in graph order |
| `running_stats` | array | Training: `{"name": "new/<param>", "param", "dtype", "shape"}` for every running statistic of every `nn.batch_norm` node, in the order the call returns them (runtime.md §3.3) |
| `masks` | array | Training: `{"name": "mask/<node path>", "node", "dtype": "bool", "shape"}` for every `nn.dropout` with `p > 0`, in the order the nodes run: the keep mask of graph-ir.md §3.3 for `training.seed`, `training.step` and the node path of runtime.md §5 (`drop1`, `blocks[1]/drop`) |
| `upstream` | array | `{"name": "grad/<output>", "output", "dtype", "shape"}` for every float graph output, in graph order |
| `gradients` | array | `{"name": "grad/<param>", "param", "dtype", "shape"}`: the expected gradient of each trainable param (training.md §1.1) that the loss reaches, in declaration order. Other params get none |
| `tolerance` | object | For `outputs` and `running_stats`: §3.2, from §2's f32 model row |
| `gradient_tolerance` | object | For `gradients`: §2's gradient row |
| `reference`, `notes`, `seed`, `generator`, `versions` | | As in §3.5 (`seed`: of the input data and upstream gradients) |

- **Expected values.** The graph as PyTorch functional ops in `f64` on the `f32` weights and inputs,
  rounded once to `f32` (§3.3): the outputs, the running statistics as `F.batch_norm(…, training=True)`
  updates them, and `torch.autograd.grad` of the loss `Σ out · grad_out` (summed over every float
  output, §3.6 "Comparing") with respect to the trainable params. Dropout multiplies by
  `x * (mask / (1 - p))` with the case's stored mask (graph-ir.md §3.3), which comes from
  `tools/reference/polyrng.py`, not from PyTorch's generator. The generator checks that PyTorch's own
  `f32` run meets both tolerances (`--report` prints the errors).
- **Comparing.** The test runs the graph through the runtime in the case's mode, with
  `model.safetensors` as the given params (runtime.md §3.2) and the trainable params marked as autodiff
  leaves, compares outputs and running statistics (names and order too) with `tolerance`, back-propagates
  the loss and compares the gradients with `gradient_tolerance`; a param not in `gradients` must get none.
  It also compares the runtime's masks (`polypous_ir::rng::dropout_mask` for each node path) with `masks`
  exactly, so a wrong node path or stream fails even where it would not change a value.
- **Cases** cover `linear → relu → linear` with a `trainable: false` bias, and a graph with `conv2d`,
  `batch_norm`, `dropout` and a `repeat` whose body drops in each iteration, at two `step`s.
- **Checks.** `tools/ci/check_fixtures.py`: graph and weights as in §3.5; keys and their order for the
  mode; `training` as two `u64`s; `running_stats` naming every running statistic of the graph; `masks`
  naming every dropout node path with `p > 0` (repeats expanded) as `bool`; one upstream gradient per float
  output; gradients only of trainable params, in declaration order, with their shapes; everything against
  the safetensors header; no subnormal values.

## 4. Backends in tests

- Op and kernel tests are written once and run on every compiled-in backend through a test helper
  that iterates over available devices.
- **CPU** runs everywhere.
- **Vulkan** runs in CI through a software driver (Mesa lavapipe on Linux) if practical. This is slow but
  catches correctness issues. Real-GPU Vulkan runs on T1 (both NVIDIA and the Intel iGPU).
- **CUDA** runs on T1 only.
- **Metal:** compile-checked in CI on a macOS runner. No real tests until Mac hardware is available.
  Report this in PRs that touch Metal paths.

## 5. CI

GitHub Actions, set up in M0 (issue #3). Workflows in [`.github/workflows/`](../.github/workflows/),
helper scripts in [`tools/ci/`](../tools/ci/). The toolchain is always the one in `rust-toolchain.toml`.

| Job (stable name, used for required checks) | Workflow | Runs on | What |
|---|---|---|---|
| `Lint & test (Linux)` | `ci.yml` | every PR, push to `main` | `cargo fmt --check`, `clippy -D warnings`, `cargo test --workspace` (CPU features) |
| `Lint & test (Windows)` | `ci.yml` | every PR, push to `main` | same on Windows |
| `cargo-deny (licenses, advisories)` | `ci.yml` | every PR, push to `main` | `cargo deny check` with [`deny.toml`](../deny.toml): MIT-compatible licenses only ([ADR-0010](adr/0010-mit-license.md)), RustSec advisories, crates.io only |
| `Dependency boundary` | `ci.yml` | every PR, push to `main` | `tools/ci/check_dependency_boundary.py` (and its tests): Burn/CubeCL only in `polypous-backend`/`polypous-quant` ([ADR-0007](adr/0007-burn-version-pinning.md)); no GPU dependency, direct or transitive, in `polypous-ir`/`-formats`/`-polyp`. Also the fixture checks (§3.3): `tools/ci/check_fixtures.py` and the standard-library tests of `tools/reference/` |
| `Vulkan tests (Linux, lavapipe)` | `ci.yml` | every PR, push to `main` | `cargo test --workspace --features polypous-cli/vulkan -- --include-ignored --skip mnist --nocapture` on Mesa lavapipe (software Vulkan): the GPU tests of `polypous-backend`, `polypous-runtime`, `polypous` and `polypous-cli` |
| `Build (macOS, Metal)` | `macos.yml` | push to `main`, manual dispatch, PRs touching `crates/polypous-backend/**`, `crates/polypous-quant/**`, `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml` | `cargo build --workspace --all-targets --features polypous-cli/metal` (build only, no Mac hardware; compiles the Metal-gated test code of the backend, runtime, facade and CLI) |

- **Minutes budget.** The repository is private on the free plan, where macOS minutes count 10×.
  The macOS job therefore runs only on PRs that touch the backend crates or the dependency set (so it
  cannot be a required check); trigger it by hand for other branches if needed. Superseded runs on PR
  branches are cancelled. Cargo caching (`Swatinem/rust-cache`) keeps a no-change run at roughly
  10 minutes or less.
- **Backend features.** `tools/ci/backend_feature.py` selects the `cpu` feature of
  `polypous-backend` and the `vulkan` and `metal` features of `polypous-cli` (which enable the
  facade's, the runtime's and the backend's, [dev-environment.md](dev-environment.md#gpu-tests))
  when they exist, and falls back to the default features with a warning annotation otherwise.
- **Actions** are pinned to full commit SHAs with a version comment. Update them deliberately.
- Nightly on T1 (self-hosted runner, optional, later): GPU and model reference tests.

## 6. Writing a reference test (checklist)

1. Add or extend a generator in `tools/reference/` and run it. Commit the small fixture or add a manifest entry.
2. Load the fixture in the test, run the Polypous implementation on every available device, and compare with the documented tolerance.
3. Include edge cases: empty/size-1 dims, non-contiguous inputs (after permute), large values, broadcasting.
4. On failure, print max abs/rel error and the index of the worst element.
