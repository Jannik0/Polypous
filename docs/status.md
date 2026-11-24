# Project Status

What exists, what is known to be missing or fragile, and where the next milestone starts. Update
this file when a milestone closes or when a known limitation is fixed or found. Last updated:
2026-11-18, M1 closed; M2 not started.

## 1. Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 — Foundations | **Done** (closed 2026-11-06) | Exit check results: [#7](https://github.com/Jannik0/Polypous/issues/7#issuecomment-6684237754), on Windows 11 and Debian 13 |
| M1 — Graph IR and interpreter | **Done** (closed 2026-11-18) with one known limitation: f32 `matmul` and `nn.linear` on CUDA (TF32 in Burn, #64, §4) | Exit check results: [#47](https://github.com/Jannik0/Polypous/issues/47#issuecomment-6990026543), on Windows 11 and Debian 13 |
| M2–M7 | Not started | [roadmap.md](roadmap.md) |

## 2. What exists after M1

| Crate | State |
|---|---|
| `polypous-backend` | Real code. Device discovery, probing, primary GPU selection and physical-GPU grouping (`device`). `DynTensor` with runtime rank 1–6 and dtypes f32/f16/bf16/i64/i32/bool over `burn::Tensor` (`tensor`). Crash-safe probing: prober protocol, supervisor, probe cache, driver fingerprints (Linux and Windows), crash journal, reasons for backends without devices (`probe`). Core ops on `DynTensor` (`ops`, M1): elementwise, comparison, logic and `cast` (#40); shape, indexing (out-of-range indices checked before the op runs) and creation ops (#41); reductions, `matmul`, softmax, `cumsum` and `top_k` (#42); the `nn` ops: layers, convolution, pools, norms and activations (#43), with a fixture-based reference harness that runs every case on every device and dtype. Autodiff (M2, #75, ADR-0014): Burn's autodiff always compiled in; `DynTensor::require_grad`, `backward` from a one-element loss into an opaque `Gradients`, `Gradients::get`/`take`, `detach`, `without_autodiff`, `is_tracked`, every misuse a `TensorError`. Gradients of the elementwise ops, `cast`, the shape and indexing ops (#75), of the reductions, `matmul`, `softmax`, `log_softmax`, `cumsum` and `top_k` (#76), and of the `nn` ops (#77) match all 253 gradient fixtures on `cpu` and lavapipe (gradient harness, `tests/grad_reference_*.rs`), with the spec's tie and kink rules for `maximum`/`minimum`/`clamp`/`pow`/`sqrt`/`rsqrt`, for `max`/`min` (ties split evenly) and for `relu`/`leaky_relu`/`softplus` (plus PyTorch's `sigmoid`/`silu` derivatives in their tails) set in the backend; `top_k` takes its values with a `gather` on the device, so they are differentiable; `batch_norm`'s running statistics never get a gradient. Training modes (#77): `DynTensor::batch_norm_training` (batch statistics, gradient through them, the new running statistics returned untracked) and `DynTensor::dropout_training` (the caller's `bool` keep mask; output and gradient exact for a given mask); the forward harness reruns every case with tracked inputs and requires identical outputs (bit for bit, except NaN payloads, which aren't part of op semantics) |
| `polypous` (facade) | Real code. `Engine`, `EngineConfig`, facade device types (`DeviceInfo` with `physical_gpu` and `primary`, `DeviceId::parse`), probe API (`polypous::probe`), `Error`/`ErrorKind` with `node`/`pointer`/`tensor`/`file`/`device` context, panics caught at the boundary. Graphs and models (M1, #45, [public-api.md §1.3](specs/public-api.md)): `Graph` (JSON, files, `ValidationReport`), `Engine::load_graph`, `Model` (`info`, `run` with named `TensorData`, `save`, `close`), `tensor_file`, one `CancellationToken`. Tested on the CPU and lavapipe |
| `polypous-cli` (`polyp`) | `polyp devices [--json] [--report] [probe options]`, `polyp probe --json`, `polyp graph validate <file> [--json]`, `polyp run <graph> --weights … --input NAME=FILE … --output FILE [--device ID]` (no state yet) |
| `polypous-formats` | safetensors reader (memory-mapped, validated header, sharded checkpoints) and writer (M1, #37). GGUF follows in M3 |
| `polypous-ir` | Real code (M1, #38, #39). Graph document model, loading under the limits of graph-ir.md §4.3, canonical JSON, structural validation with collected errors, dim expressions, the catalog of the 78 M1 ops, dtype and symbolic shape inference, and the runtime API (`Graph::analyze`, `Analysis::bind`). The Polypous random generator of graph-ir.md §3.1.12 (`rng`, M2, #77): `mix`, `fnv1a`, `stream_key`, `draw`, `u24`, `u53`, `normal`, and the dropout keep mask for a shape, tested against the spec's known values (bit-exact; `normal` within 1 ulp) and testing-strategy.md §3.8's statistics. Param analysis (M2, #78): every param's uses after expanding repeats, its fans by graph-ir.md §2.4.1's table, `invalid_init` for a fan-based `init` on an unused param, in a role without fans or with disagreeing fans, the `invalid_value` checks of usable `init` values (§2.4), and `Analysis::params`, `param_fans`, `trainable_params` (training.md §1.1) |
| `polypous-runtime` | Real code (M1, #44). The single-device graph interpreter of [runtime.md](specs/runtime.md): plans (validation with a compute dtype, device dtype check, topological order, value lifetimes), strict weight loading from safetensors files and sharded checkpoints (header checks before any allocation, one rounding to the compute dtype), calls with inputs and state by name, `repeat`, cancellation, errors with node paths. Every M1 op runs as a one-node graph against its fixture (verified on every device of the reference machine in #47). Training mode (M2, #78, runtime.md §3.1–§3.3): `RunMode::Training { seed, step }`, params given by the caller (`Model::run_with`, `Plan::run`) with the metadata-only checks of §3.2, the three training-mode checks of §3.1, `batch_norm` with batch statistics returning the new running statistics, and dropout with host masks keyed by seed, step and node path; checked against PyTorch on two training-mode graphs (outputs, running statistics, gradients) on `cpu` and lavapipe. Inference mode is unchanged |
| `polypous-train` | Initial values (M2, #78, graph-ir.md §2.4.1): `initial_values` computes every `init` kind on the host with the Polypous generator, in f64 rounded once to f32; bit-exact against the Python reference fixtures (`fixtures/small/init/`, all 45 cases, normal-based kinds included on Linux) and checked for bounds, means and variances. Optimizers, schedules and gradient clipping (M2, #79, training.md §3.1, §3.2, §5.2): `sgd`, `adam`, `adamw` on `DynTensor` ops with state export and import, `constant`/`step`/`cosine`/`linear_warmup` schedules, `clip_gradients`, and the `optimizer`/`schedule`/`gradient_clip` config types with their ranges; within the optimizer tolerance of PyTorch's own f32 results on the 13 optimizer fixtures (worst 0.14 of it) on the CPU and lavapipe, bit-exact on the CPU with training.md §3.1's op sequence, schedules exact (cosine within its tolerance) on 11 fixtures, clipping on 5 (`fixtures/small/train/`). Losses and metrics (M2, #80, training.md §2.6, §2.7, §3.3, §3.4): `Loss` (`mse`, `cross_entropy{label_smoothing}`, `binary_cross_entropy{with_logits}`) in public `DynTensor` ops, differentiable with respect to the prediction, within testing-strategy.md §2's loss tolerance of PyTorch's f64 value and gradient on the 17 loss fixtures (worst 2.7e-4 of it, `cross_entropy` with `i64` and `i32` targets) on the CPU and lavapipe; the `LossMetric` and `Accuracy` epoch accumulators (exact, unit-tested); the host-side target checks and the `loss`/`metrics` config types. Datasets, shuffling and batching (M2, #81, training.md §2.3, §2.4, §3.5, §3.6): `read_datasets` with the `tensors`, `safetensors`, `csv` (own RFC 4180 parser) and `image_folder` (PNG via the `png` crate, every color type and bit depth) readers and the seeded `split`, every limit of §3.5 checked before allocating (fallible reservations, `data.max_bytes` over both sets), errors naming the file and record/column/tensor; `Batcher` with the epoch order and any batch computed directly, bit-exact against §3.6's known values, the shuffle uniformity test of testing-strategy.md §3.8, generated test PNGs (Adam7 included) and fuzz-style loops over truncated and bit-flipped PNG, CSV and safetensors files; the `data` config types; the `data_fingerprint` of §5.2. The trainer (M2, #82, training.md §1.2, §2, §4, §5): `TrainingConfig` with full-pointer errors and a canonical `to_json`; `Trainer::new` with the checks of §2.10 in order; the loop, events (throttled on an injectable clock), cancellation, atomic checkpoints with `keep_last`/`keep_best`, `Trainer::resume` of untrusted checkpoints (5,400 fuzzed checkpoints, no panic) and `Trainer::export`. Verified on the CPU: an MLP learns (cross-entropy and MSE), and resume is bit-exact against uninterrupted training in testing-strategy.md §3.8's scenarios (mid-epoch, epoch boundary, during validation, from an `every_steps` checkpoint); lavapipe: the MSE run reaches its loss threshold. The facade and `polyp train` follow (#83) |
| `-polyp`, `-quant`, `-pipelines` | Skeletons (a doc comment each) |
| `tools/reference/`, `fixtures/` | PyTorch fixture generators with `--check`: 885 op cases (`gen_ops.py`, #36, #40–#43) and the end-to-end MLP and CNN (`gen_models.py`, #46); 253 gradient cases of every differentiable op (`gen_grads.py`, schema 2, `fixtures/small/grads/`, #74); `polyrng.py`, a Python port of the Polypous generator and the initial values; 45 init cases and 3 training-mode graph cases (`gen_training.py`, `fixtures/small/init/`, `fixtures/small/training/`, #78); `fetch_fixtures.py` and the fixture checks in CI |

Specs that describe implemented behavior: [placement.md §1](specs/placement.md#1-device-discovery)
(discovery, probing, cache, fingerprints, enumeration rules), [public-api.md §1.1–§1.3, §2](specs/public-api.md)
and [formats.md §1](specs/formats.md) (safetensors). [graph-ir.md](specs/graph-ir.md) and [runtime.md](specs/runtime.md) are implemented:
validation and inference (#38, #39), the backend ops (#40–#43, #67), the interpreter (#44) and the facade (#45). All other specs are drafts until their milestone starts.

CI (`.github/workflows/`): seven checks. Six are required by the `main` ruleset:
- Lint & test (Linux)
- Lint & test (Windows)
- cargo-deny (licenses, advisories)
- Dependency boundary
- Vulkan tests (Linux, lavapipe)
- PR title and description

The seventh, Build (macOS, Metal), runs only on pushes to `main`, by hand, and on PRs that touch
the backend crates or the dependency set, so it can't be required ([testing-strategy.md §5](testing-strategy.md#5-ci)).
When it runs on a PR, it must be green too. CodeQL code scanning (Rust, Actions, Python, GitHub's
default setup) also runs on PRs; it isn't required, but its findings are reviewed. The lavapipe job runs the GPU tests
of every crate (`polypous-cli/vulkan`, #65); the macOS job builds with `polypous-cli/metal`.

## 3. Verified on the reference machine

The reference machine is the Razer Blade 15: RTX 3060 Laptop 6 GB, Intel Iris Xe, i7-12800H. See
[hardware-targets.md](hardware-targets.md).

| | Windows 11 | Debian 13 |
|---|---|---|
| CUDA (RTX 3060), f32/f16/bf16 | ✓ | ✓ |
| Vulkan RTX 3060, Vulkan Iris Xe, f32/f16 (no bf16) | ✓ | ✓ |
| Isolated probing, probe cache hits, re-probe after a driver update | ✓ (real NVIDIA update 617.42 → 621.14) | ✓ |
| Cache survives a reboot | ✓ (#30) | ✓ (PCI ids are stable) |
| Real native driver crash contained in the prober | – (no crashing driver available) | ✓ (lavapipe, opt-in) |
| Backend GPU tests in parallel | ✓ | ✓ (after #27) |

Driver versions at the M0 exit check:
- Windows: NVIDIA 621.14 and Intel 32.0.101.7092.
- Debian: NVIDIA 620.70.06 and Mesa 25.0.7.

**M1 exit check (#47, both systems on `ddb6bf0`, drivers unchanged from M0):**

| | Windows 11 | Debian 13 |
|---|---|---|
| All 885 op fixture cases (78 ops), backend and as one-node graphs through the interpreter: CPU, Vulkan RTX 3060, Vulkan Iris Xe (bf16 skipped on Vulkan) | ✓ | ✓ |
| The same on CUDA, f16/bf16 and every f32 op except `matmul`/`nn.linear` | ✓ | ✓ |
| f32 `core.matmul` (9 of 11) and `nn.linear` (7 of 8) on CUDA | ✗ (#64, TF32) | ✗ (#64, TF32) |
| MLP and CNN from JSON vs PyTorch through the facade and `polyp run`, every device and dtype except f32 on CUDA | ✓ | ✓ |
| Out-of-range indices and `embedding` ids are errors on every device | ✓ | ✓ |
| No crash, no GPU error in the system/kernel log, failures identical in three runs | ✓ | ✓ |

GPU builds of the whole workspace need `-j 2` on the 16 GB reference machine (dev-environment.md, "GPU tests").

## 4. Known limitations and open items

Fix an item when its area is touched, or file an issue first.

Every finding in a dependency (Burn, CubeCL, cubek, GPU drivers) has an ID, its workaround and its
upstream state in the [upstream findings register](upstream-findings.md); the IDs are given below.

- **f32 `matmul` and `nn.linear` on CUDA are not f32-accurate ([#64](https://github.com/Jannik0/Polypous/issues/64), M1-9; M1 closed with it, owner decision).**
  - Burn 0.22.1 / cubek-matmul 0.3.1 run f32 `matmul` on TF32 tensor cores when the GPU supports TF32 (Ampere or newer): `adjust_dtypes` in cubek-matmul's `multi_level/definition/blueprint.rs`, mirrored in burn-cubecl's `kernel/matmul/tune/base.rs`. TF32 rounds the inputs to a 10-bit mantissa. There is no setting to turn it off.
  - On the RTX 3060 (Windows, NVIDIA 621.14), 9 of 11 `core.matmul` f32 cases fail on `cuda:0`: relative errors of 6e-4 to 6.5e-2 against `rtol 1e-3`. The output equals an f32 matmul with TF32-rounded inputs (max rel difference 6.7e-6). Vulkan on the same GPU, the Iris Xe, the CPU and lavapipe pass; f16/bf16 pass on CUDA.
  - **No workaround in Polypous (owner decision):** the spec (graph-ir.md §3.2, testing-strategy.md §2) stays as it is, and the tests keep failing on `cuda:0`. A trial computing in f64 on CUDA was exact but 61× slower than TF32 at 2048³, and was discarded.
  - Upstream: [tracel-ai/burn#6140](https://github.com/tracel-ai/burn/issues/6140) asked for control over TF32 in f32 matmul; the PRs answering it are merged into Burn's `main` (not in a release yet). They don't add a Rust-level API to switch it off easily, which [tracel-ai/burn#6480](https://github.com/tracel-ai/burn/issues/6480) asks for. Fix when Burn is updated (ADR-0007): select full f32 precision, then the CUDA tests must pass.
  - `nn.linear` runs on `matmul` and fails the same way: 7 of 8 f32 cases on `cuda:0` (#43, PR 60). `nn.conv1d`/`conv2d` pass on CUDA in every dtype, f32 included.
  - f16/bf16 `matmul` computes in f32 since #67 and is not affected: f16 and bf16 values are exact in TF32 (10-bit mantissa, the same as f16's).
  - The M1 exit check confirmed it on Windows and Debian: 16 op cases (in the backend and the runtime tests) and the f32 cases of both end-to-end models (1.5–5.2× the allowed error) fail on `cuda:0`; nothing else does.

- **Burn 0.22.1 bool bug** in `bool_to_device` across Dispatch backends (M0-1):
  - Burn hands native bool storage to CubeCL, which panics on storage that isn't its own.
  - `DynTensor` works around it by going through host memory (`tensor/dyn_tensor.rs`).
  - The upstream report and a standalone reproduction are handled outside this repository, so there's no issue here. Re-check when Burn is updated (ADR-0007).
- **Burn 0.22.1 / CubeCL 0.11.3 numerics found in M1 (#40, PR 54; M1-1 … M1-8, M1-10 … M1-20 in the register)**, each worked around in `polypous-backend` (`ops/mod.rs` documents them). Re-check when Burn is updated (ADR-0007):
  - `flex` computes `i64` `add`/`sub`/`mul` with plain arithmetic, which panics on overflow in debug builds. The CPU runs these on the host with wrapping arithmetic.
  - `max_pair`/`min_pair` (generic code in `burn-tensor`, so every backend) drop a NaN in the second operand. `maximum`, `minimum` and `clamp` use NaN masks.
  - CubeCL's SPIR-V lowering maps float `!=` to `OpFOrdNotEqual`: `ne` with NaN and float → `bool` casts of NaN gave `false` on Vulkan. The LLVM path (CUDA) uses the unordered form. Explicit masks.
  - CubeCL's `powf` polyfill falls back to the native pow (`GLSL.std.450 Pow` on SPIR-V, undefined for some inputs), which is wrong for `1^NaN`, `NaN^0`, `0^0` and `0^-1` on lavapipe. Explicit C99 special cases.
  - CubeCL's f16 `tanh` missed `rtol 1e-2` near zero on lavapipe. f16/bf16 transcendentals and `pow` compute in f32 on every backend.
  - On the reference machine (Windows, NVIDIA 621.14, Intel 32.0.101.7092): Intel's Vulkan driver returns `pow(|a|, b)` for a negative base with a non-integer exponent (GLSL `pow` is undefined there). `pow` sets NaN explicitly for it. Both Vulkan drivers flush f32 subnormals to zero in comparisons. Subnormals are backend-defined (graph-ir.md §3.1.8, owner decision), and fixtures contain none.
  - Found with #41 (PR 57): out-of-range indices in `gather`/`select` make `flex` panic, and on lavapipe they read out of bounds (wrong values; `select` with `-1` crashed the process with SIGSEGV). Polypous checks indices before every indexing op (one min/max read-back, one device sync on GPUs). `flex` also returns `gather`/`select` of a bool tensor in `u8` storage, which can't be read back as bool; bool indexing runs as `i32` on every backend.
  - Found with #42 (PR 59): `flex` panics on overflow in `i64` `sum_dim` and in `i32`/`i64` `cumsum` (integer `sum`/`cumsum` wrap on the host on the CPU). CubeCL's `f16` `cumsum` accumulates in `f16` on lavapipe (the cumulative sum of 4096 ones gives 2048); f16/bf16 `sum`, `mean`, `cumsum`, `softmax` and `log_softmax` compute in f32 on every backend (graph-ir.md §3.1.8). Burn's sorts are unstable and its CubeCL `sort`/`topk` read back to the host anyway; CubeCL's `topk_with_indices` returned garbage after a NaN on lavapipe. `top_k` sorts stably on the host, one device sync per call.
  - Found with #43 (PR 60): CubeCL's `max_pool2d` pads with `-f32::MAX` instead of `-inf` (an all-padding window, or a padded window of only `-inf` inputs, gives `-3.4028235e38`) and dropped NaN on lavapipe (the window `1, NaN, 5, 6` gave 6); `max_pool2d` sets both explicitly with one extra pool of a code tensor. CubeCL's `erf` is the Abramowitz–Stegun 7.1.26 polynomial (absolute error 1.5e-7, inside the `gelu` tolerances), and its `log1p` is `ln(1 + x)`, which is 0 in f32 below `x ≈ 6e-8`: `softplus(x)` gave 0 below `x ≈ -16.6` on every CubeCL device (CUDA and Vulkan; `softplus(-17)`: 0 instead of 4.14e-8), and so did `mish`. Mesa's Vulkan `tanh` (lavapipe) returns noise of about 3e-8 for tiny arguments. `softplus` uses its own `log1p` and `mish` its own `tanh` for non-negative arguments, with series for small arguments. In f16, Burn's `matmul` and conv (`flex` and lavapipe) miss `rtol 1e-2` on outputs that cancel to near zero, and f16 norms and activations fail without f32 compute; f16/bf16 `nn.linear`, conv, average pools, norms and activations compute in f32 (graph-ir.md §3.1.8).
  - Found with the M1 exit check (#47, #67): `core.matmul/k256_f16` (`[2, 4, 256] @ [256, 3]`, `rtol 1e-2`) missed on the Iris Xe with Mesa 25.0.7 ANV (Debian 13.7), deterministically: 1 of 24 elements, max abs error 3.125e-2, max rel error 2.17e-2 (expected 9.009e-2, got 8.813e-2, about 32 f16 ulps), which looks like half-precision partial sums. The CPU, CUDA, the RTX 3060 through Vulkan and every Windows driver passed. Cause not confirmed: cubek-matmul 0.3.1 accumulates f16 in f32 in registers, but `MatmulElems::adjust_stage_dtypes` stages the accumulator in the output dtype, which the autotune picks on ANV may use. f16/bf16 `matmul` computes in f32 and rounds once since #67 (graph-ir.md §3.1.8). With the f32 path off, the unit test `half_precision_computes_in_f32_and_rounds_once` (k = 4096, products not exact in f16, exact f32 sums) misses in f16 on `flex` (−0.6201 instead of −0.6099), passes in f16 on lavapipe, and passes in bf16 on `flex`; the `k256_f16` fixture passes on both without it.
  - The owner follows these up in Burn and CubeCL upstream, outside this repository.
- **Burn 0.22.1 autodiff findings of M2 (M2-1 … M2-8; the lavapipe crash is M2-9).** Six were handed off from the spike (#71, PR 86): no
  `retain_graph`, `backward` panics on an untracked tensor, `sqrt'(0)` on CubeCL, tie rules, Adam
  scalars, the global seed. New with #75, each worked around in `polypous-backend` (`ops/mod.rs`,
  "Gradients"; `tensor/autodiff.rs`). Re-check when Burn is updated (ADR-0007):
  - `Tensor::detach` keeps `require_grad` on a leaf: it returns a new leaf that is still tracked,
    unlike PyTorch's `detach` (documented in `burn-tensor`, `float.rs` `detach`;
    `burn-autodiff` `float_detach`). `DynTensor::detach` and the gradient fixes use
    `detach().set_require_grad(false)`.
  - The `sqrt` backward (`g · x^-0.5 / 2` through the backend's `powf`) is wrong at `-0` on `flex`
    too, not only at `0` on CubeCL: `(-0)^-0.5 = +inf`, so `sqrt'(-0)` is `+inf · g` instead of
    `-inf · g` (graph-ir.md §3.1.11). Our `rsqrt` (`1 / sqrt(x)`) inherits both: the wrong sign at
    `-0` on `flex`, NaN at `±0` on lavapipe.
  - Burn's "graph tape has already been consumed" panic (a second backward) leaves the graph usable:
    `DynTensor::backward` catches it and returns an error; fresh forwards from the same leaves work.
  - **Lavapipe crashes (SIGSEGV) in `select`'s backward: a Mesa (lavapipe) bug, not a Burn or
    CubeCL one; hardware GPUs aren't affected** (#75 found it, #92 found the cause). The API test
    `autodiff_gpu` (`tests/autodiff.rs`) crashes in about 1 of 4 runs on lavapipe (11 of 40 in
    #92), so it skips lavapipe unless `POLYPOUS_TEST_LAVAPIPE_AUTODIFF=1` (owner decision,
    2026-11-20). The skip stays: lavapipe is a documented limitation here, not a training target.
    - **Faulting kernel:** Burn's `select_assign_kernel` (`burn-cubecl` `kernel/index/select_assign.rs`),
      which `select`'s backward runs (a scatter-add into zeros). Each invocation loops over the
      indices, reads `indices[i]`, then `tensor[indices[i]]`.
    - **Cause:** Mesa's gallivm doesn't branch on an empty exec mask, so it runs the rest of the
      loop body once more after the last invocation has left the loop, with `i` equal to the
      index count. Its load from a uniform global address (`visit_load_global` in
      `src/gallium/auxiliary/gallivm/lp_bld_nir_soa.c`, Mesa 26.0.8 lines 4666–4673) ignores the
      exec mask. CubeCL 0.11.3's SPIR-V reaches buffers through buffer device addresses
      (`PhysicalStorageBuffer64`), which Vulkan's robustness doesn't cover. So the extra pass reads
      `indices[len]`, whatever lies after the index buffer in CubeCL's memory pool, and then
      `tensor[that value]`, which faults when it isn't mapped. The store stays masked, so the values
      are right whenever the address is mapped. The masks, the earlier backward passes and
      `gather` instead of `select` (whose backward has no such loop) only changed what lay after the
      index buffer.
    - **Evidence:** gdb at the fault (both reproductions and `autodiff_gpu`): the only active lane has
      just left the loop (loop counter 3 = index count 3) when the load of `tensor[0xFFC00000]` faults.
      The kernel's SPIR-V reads `indices[i]` only inside the loop; Vulkan validation reports nothing
      at dispatch.
    - **Reproductions** in `polypous-backend/src/tensor/autodiff_tests.rs`, opt-in with
      `POLYPOUS_REPRO_LVP_CRASH=1`: `lavapipe_select_assign_stale_index_crash` (one
      `select_assign` after leaving `0xFFC00000` in freed pool slots, no autodiff) and
      `lavapipe_where_select_backward_crash` (#75's). Each crashes 3 of 3 runs with Ubuntu's Mesa 25.2.8,
      26.0.3 and 26.0.8 (LLVM 20 and 21). With a small value after the indices it passes with the
      right values.
    - **Hardware GPUs:** they don't execute loads for invocations that have left a loop, and no
      active invocation reads out of bounds. On lavapipe, random `select` and `select_assign` cases
      matched `flex` bit for bit, with canary buffers around them unchanged (2000 cases). On the
      reference machine #75's reproduction passed 5 of 5 runs with correct values on `cuda:0` and
      both Vulkan GPUs (PR 91).
    - **Upstream, unrelated to the crash:** `select_assign_kernel` claims checked mode clamps
      out-of-range indices, but its read `tensor[index]` is not clamped in the SPIR-V (only the write
      is checked). Polypous checks indices before every indexing op, so it isn't reachable from here.
- **Found with #76 (M2-10, M2-11; worked around in `polypous-backend`, `ops/mod.rs`).** Re-check when Burn is updated (ADR-0007):
  - On lavapipe, `max_dim`/`min_dim` return either zero of a tie between `0` and `-0`:
    `max([0, -0, -1, -2, -0.5])` gave `-0` and `min([-0, 0, 1, 2, 0.5])` gave `0`; `flex` and
    PyTorch (which returns the first zero of the slice) give `0` and `-0` (`core.max/zero_ties_f32`,
    `core.min/zero_ties_f32`, exact).
    SPIR-V's `FMax`/`FMin` allow either zero. Float `max`/`min` now set the sign of a zero result
    by IEEE 754-2019's order (`-0 < +0`) on every backend (owner decision, graph-ir.md §3.1.8).
  - `flex` checks the dtypes of binary float ops with a `debug_assert_eq!` only, so mixing `f16`
    and `f32` panics in debug builds (`binary_op: dtype mismatch`) instead of returning an error.
    Burn's `Bool::float` gives the default float dtype (`f32`), whatever dtype the mask came from,
    which is how a tracked `f16` `max` hit it during #76 (fixed in Polypous: the tie mask is built
    in the input's dtype). Not reachable from Polypous's ops now.
- **Found with #77 (M2-12; worked around in `polypous-backend`, `ops/pool.rs`).** Re-check when Burn is updated (ADR-0007):
  - On lavapipe, Burn's `max_pool2d` and the `max_pool2d_with_indices` its autodiff runs return
    different zeros at a `0`/`-0` tie in a window: the window `[0, -0]` gave `-0` untracked and
    `+0` tracked (PyTorch's first-maximum rule gives `+0`). Both kernels take SPIR-V's `FMax`,
    which may return either zero. The tracked `max_pool2d` keeps the untracked pool's value and
    takes only the gradient from the indexed pool, so tracked and untracked forwards agree bit for
    bit. Which zero `max_pool2d` returns at such a tie is backend-defined (graph-ir.md §3.3, owner
    decision; the forward fixtures have no such tie).
- **The first open of a Vulkan device still enumerates once inside CubeCL** (M0-7; M0-6 is the race it is left over from), and that can race with Vulkan compute on other threads ([placement.md §1, "Remaining window"](specs/placement.md); #26).
  - `Engine::new` opens every device it probes, so only a first use of a device discovery didn't open is exposed.
- **Lavapipe on Debian 13** (Mesa 25.0.7, LLVM 19) segfaults on its first compute work (M0-9).
  - Software rasterizers are skipped unless opted in (#21).
  - Ubuntu's newer Mesa in CI is fine.
- **Worker-thread panics in probe reasons** are collected only in the prober process (#32, M0-8). In-process probing shows the calling thread's view, for example `RecvError`.
- **Not verified on hardware:**
  - Metal: compile-checked in CI only.
  - A `cuda` build on Windows with no NVIDIA driver installed: approximated with `CUDA_VISIBLE_DEVICES=-1`.
  - The "driver library is not installed" reason on Windows: `nvcuda.dll` can't be hidden without admin rights. It is unit-tested, and verified on Debian.
- **macOS driver fingerprint:** unknown, so macOS takes the enumerate-only probe path.
- **Dependabot alerts 1 and 2** (M0-14):
  - Both concern `zip` 0.6.6 (one critical, one moderate). It enters `Cargo.lock` only through Burn's optional LibTorch backend (`burn-tch` → `tch`/`torch-sys`), which Polypous never compiles: `cargo tree --all-features --target all` doesn't list it.
  - There's no fix through an update while Burn is pinned at 0.22.1.
  - If cargo-deny starts failing once RustSec publishes the advisories, add a `deny.toml` ignore entry with this reason. The owner decides whether to dismiss the GitHub alerts.
- **`target/` size:** debug test binaries with `cuda` or `vulkan` are about 1.2 GB each. Plan for 20 GB.

## 5. Where M2 starts

[roadmap.md, M2](roadmap.md#m2--training-toolkit): training on `DynTensor` with Burn's autodiff, a
Polypous param store and own optimizers ([ADR-0014](adr/0014-training-on-dyntensor-with-own-optimizers.md)),
parameter init from the IR (`init`, graph-ir.md §2.4), the declarative training config,
optimizers, schedules, losses, metrics, datasets, events, checkpoints and resume. Exit: an
MNIST-sized CNN trains to the expected accuracy from a JSON graph and a JSON config on CPU and CUDA,
the gradients of every differentiable op match PyTorch, and training resumes from a checkpoint.

Specs to refine first: [training.md](specs/training.md), and graph-ir.md for the training modes of
`nn.batch_norm` and `nn.dropout` and the `init` kinds. M2's issues are not filed yet.

Open items to keep in mind: #64 (CUDA f32 `matmul`, a Burn upgrade task once Burn ships the
precision switch; M2's exit check includes CUDA) and the Burn/CubeCL findings of §4.
