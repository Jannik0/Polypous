# Project Status

What exists, what is known to be missing or fragile, and where the next milestone starts. Update
this file when a milestone closes or when a known limitation is fixed or found. Last updated:
2026-11-10, M1 in progress (#35–#42 done).

## 1. Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 — Foundations | **Done** (closed 2026-11-06) | Exit check results: [#7](https://github.com/Jannik0/Polypous/issues/7#issuecomment-6684237754), on Windows 11 and Debian 13 |
| M1 — Graph IR and interpreter | In progress. Issues #35–#47 on the [M1 milestone](https://github.com/Jannik0/Polypous/milestone/2); done: #35 (spec pass), #36 (fixture tooling), #37 (safetensors), #38 and #39 (`polypous-ir`), #40 (elementwise ops), #41 (shape, indexing and creation ops), #42 (reductions, `matmul`, softmax, `cumsum`, `top_k`; f32 `matmul` on CUDA misses the spec, see §4). Next: #43 (nn ops), then #44 (interpreter). M1 can't close before the Burn TF32 fix (§4) | [roadmap.md](roadmap.md#m1--graph-ir-and-interpreter) |
| M2–M7 | Not started | [roadmap.md](roadmap.md) |

## 2. What exists after M0

| Crate | State |
|---|---|
| `polypous-backend` | Real code. Device discovery, probing, primary GPU selection and physical-GPU grouping (`device`). `DynTensor` with runtime rank 1–6 and dtypes f32/f16/bf16/i64/i32/bool over `burn::Tensor` (`tensor`). Crash-safe probing: prober protocol, supervisor, probe cache, driver fingerprints (Linux and Windows), crash journal, reasons for backends without devices (`probe`). Core ops on `DynTensor` (`ops`, M1): elementwise, comparison, logic and `cast` (#40); shape, indexing (out-of-range indices checked before the op runs) and creation ops (#41); reductions, `matmul`, softmax, `cumsum` and `top_k` (#42), with a fixture-based reference harness that runs every case on every device and dtype |
| `polypous` (facade) | Real code. `Engine`, `EngineConfig`, facade device types (`DeviceInfo` with `physical_gpu` and `primary`), probe API (`polypous::probe`), `Error`/`ErrorKind`, panics caught at the boundary |
| `polypous-cli` (`polyp`) | `polyp devices [--json] [--report] [probe options]`, `polyp probe --json` |
| `polypous-formats` | safetensors reader (memory-mapped, validated header, sharded checkpoints) and writer (M1, #37). GGUF follows in M3 |
| `polypous-ir` | Real code (M1, #38, #39). Graph document model, loading under the limits of graph-ir.md §4.3, canonical JSON, structural validation with collected errors, dim expressions, the catalog of the 78 M1 ops, dtype and symbolic shape inference, and the runtime API (`Graph::analyze`, `Analysis::bind`) |
| `polypous-runtime`, `-polyp`, `-quant`, `-train`, `-pipelines` | Skeletons (a doc comment each). M1 fills `-runtime` (#44) |
| `tools/reference/`, `fixtures/` | PyTorch fixture generator with `--check`, first op fixtures, `fetch_fixtures.py` and the fixture checks in CI (M1, #36) |

Specs that describe implemented behavior: [placement.md §1](specs/placement.md#1-device-discovery)
(discovery, probing, cache, fingerprints, enumeration rules), [public-api.md §1.1, §1.2, §2](specs/public-api.md)
and [formats.md §1](specs/formats.md) (safetensors). [graph-ir.md](specs/graph-ir.md) is binding for M1 as its
status line says: validation and inference are implemented (#38, #39), the backend ops in part (#40, #41), the interpreter not yet. All other specs are drafts until their milestone starts.

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
default setup) also runs on PRs; it isn't required, but its findings are reviewed.

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

## 4. Known limitations and open items

Fix an item when its area is touched, or file an issue first. Only the first item blocks M1.

- **f32 `matmul` and `nn.linear` on CUDA are not f32-accurate (blocks the M1 exit check, #47).**
  - Burn 0.22.1 / cubek-matmul 0.3.1 run f32 `matmul` on TF32 tensor cores when the GPU supports TF32 (Ampere or newer): `adjust_dtypes` in cubek-matmul's `multi_level/definition/blueprint.rs`, mirrored in burn-cubecl's `kernel/matmul/tune/base.rs`. TF32 rounds the inputs to a 10-bit mantissa. There is no setting to turn it off.
  - On the RTX 3060 (Windows, NVIDIA 621.14), 9 of 11 `core.matmul` f32 cases fail on `cuda:0`: relative errors of 6e-4 to 6.5e-2 against `rtol 1e-3`. The output equals an f32 matmul with TF32-rounded inputs (max rel difference 6.7e-6). Vulkan on the same GPU, the Iris Xe, the CPU and lavapipe pass; f16/bf16 pass on CUDA.
  - **No workaround in Polypous (owner decision):** the spec (graph-ir.md §3.2, testing-strategy.md §2) stays as it is, and the tests keep failing on `cuda:0`. A trial computing in f64 on CUDA was exact but 61× slower than TF32 at 2048³, and was discarded.
  - Upstream: [tracel-ai/burn#6140](https://github.com/tracel-ai/burn/issues/6140) asked for control over TF32 in f32 matmul; the PRs answering it are merged into Burn's `main` (not in a release yet). They don't add a Rust-level API to switch it off easily, which [tracel-ai/burn#6480](https://github.com/tracel-ai/burn/issues/6480) asks for. Fix when Burn is updated (ADR-0007): select full f32 precision, then the CUDA tests must pass.
  - `nn.linear` runs on `matmul` and fails the same way: 7 of 8 f32 cases on `cuda:0` (#43, PR 60). `nn.conv1d`/`conv2d` pass on CUDA in every dtype, f32 included.

- **Burn 0.22.1 bool bug** in `bool_to_device` across Dispatch backends:
  - Burn hands native bool storage to CubeCL, which panics on storage that isn't its own.
  - `DynTensor` works around it by going through host memory (`tensor/dyn_tensor.rs`).
  - The upstream report and a standalone reproduction are handled outside this repository, so there's no issue here. Re-check when Burn is updated (ADR-0007).
- **Burn 0.22.1 / CubeCL 0.11.3 numerics found in M1 (#40, PR 54)**, each worked around in `polypous-backend` (`ops/mod.rs` documents them). Re-check when Burn is updated (ADR-0007):
  - `flex` computes `i64` `add`/`sub`/`mul` with plain arithmetic, which panics on overflow in debug builds. The CPU runs these on the host with wrapping arithmetic.
  - `max_pair`/`min_pair` (generic code in `burn-tensor`, so every backend) drop a NaN in the second operand. `maximum`, `minimum` and `clamp` use NaN masks.
  - CubeCL's SPIR-V lowering maps float `!=` to `OpFOrdNotEqual`: `ne` with NaN and float → `bool` casts of NaN gave `false` on Vulkan. The LLVM path (CUDA) uses the unordered form. Explicit masks.
  - CubeCL's `powf` polyfill falls back to the native pow (`GLSL.std.450 Pow` on SPIR-V, undefined for some inputs), which is wrong for `1^NaN`, `NaN^0`, `0^0` and `0^-1` on lavapipe. Explicit C99 special cases.
  - CubeCL's f16 `tanh` missed `rtol 1e-2` near zero on lavapipe. f16/bf16 transcendentals and `pow` compute in f32 on every backend.
  - On the reference machine (Windows, NVIDIA 621.14, Intel 32.0.101.7092): Intel's Vulkan driver returns `pow(|a|, b)` for a negative base with a non-integer exponent (GLSL `pow` is undefined there). `pow` sets NaN explicitly for it. Both Vulkan drivers flush f32 subnormals to zero in comparisons. Subnormals are backend-defined (graph-ir.md §3.1.8, owner decision), and fixtures contain none.
  - Found with #41 (PR 57): out-of-range indices in `gather`/`select` make `flex` panic, and on lavapipe they read out of bounds (wrong values; `select` with `-1` crashed the process with SIGSEGV). Polypous checks indices before every indexing op (one min/max read-back, one device sync on GPUs). `flex` also returns `gather`/`select` of a bool tensor in `u8` storage, which can't be read back as bool; bool indexing runs as `i32` on every backend.
  - Found with #42 (PR 59): `flex` panics on overflow in `i64` `sum_dim` and in `i32`/`i64` `cumsum` (integer `sum`/`cumsum` wrap on the host on the CPU). CubeCL's `f16` `cumsum` accumulates in `f16` on lavapipe (the cumulative sum of 4096 ones gives 2048); f16/bf16 `sum`, `mean`, `cumsum`, `softmax` and `log_softmax` compute in f32 on every backend (graph-ir.md §3.1.8). Burn's sorts are unstable and its CubeCL `sort`/`topk` read back to the host anyway; CubeCL's `topk_with_indices` returned garbage after a NaN on lavapipe. `top_k` sorts stably on the host, one device sync per call.
  - Found with #43 (PR 60): CubeCL's `max_pool2d` pads with `-f32::MAX` instead of `-inf` (an all-padding window, or a padded window of only `-inf` inputs, gives `-3.4028235e38`) and dropped NaN on lavapipe (the window `1, NaN, 5, 6` gave 6); `max_pool2d` sets both explicitly with one extra pool of a code tensor. CubeCL's `erf` is the Abramowitz–Stegun 7.1.26 polynomial (absolute error 1.5e-7, inside the `gelu` tolerances), and its `log1p` is `ln(1 + x)`, which is 0 in f32 below `x ≈ 6e-8`: `softplus(x)` gave 0 below `x ≈ -16.6` on every CubeCL device (CUDA and Vulkan; `softplus(-17)`: 0 instead of 4.14e-8), and so did `mish`. Mesa's Vulkan `tanh` (lavapipe) returns noise of about 3e-8 for tiny arguments. `softplus` uses its own `log1p` and `mish` its own `tanh` for non-negative arguments, with series for small arguments. In f16, Burn's `matmul` and conv (`flex` and lavapipe) miss `rtol 1e-2` on outputs that cancel to near zero, and f16 norms and activations fail without f32 compute; f16/bf16 `nn.linear`, conv, average pools, norms and activations compute in f32 (graph-ir.md §3.1.8).
  - The owner follows these up in Burn and CubeCL upstream, outside this repository.
- **The first open of a Vulkan device still enumerates once inside CubeCL**, and that can race with Vulkan compute on other threads ([placement.md §1, "Remaining window"](specs/placement.md); #26).
  - `Engine::new` opens every device it probes, so only a first use of a device discovery didn't open is exposed.
- **Lavapipe on Debian 13** (Mesa 25.0.7, LLVM 19) segfaults on its first compute work.
  - Software rasterizers are skipped unless opted in (#21).
  - Ubuntu's newer Mesa in CI is fine.
- **Worker-thread panics in probe reasons** are collected only in the prober process (#32). In-process probing shows the calling thread's view, for example `RecvError`.
- **Not verified on hardware:**
  - Metal: compile-checked in CI only.
  - A `cuda` build on Windows with no NVIDIA driver installed: approximated with `CUDA_VISIBLE_DEVICES=-1`.
  - The "driver library is not installed" reason on Windows: `nvcuda.dll` can't be hidden without admin rights. It is unit-tested, and verified on Debian.
- **macOS driver fingerprint:** unknown, so macOS takes the enumerate-only probe path.
- **Dependabot alerts 1 and 2:**
  - Both concern `zip` 0.6.6 (one critical, one moderate). It enters `Cargo.lock` only through Burn's optional LibTorch backend (`burn-tch` → `tch`/`torch-sys`), which Polypous never compiles: `cargo tree --all-features --target all` doesn't list it.
  - There's no fix through an update while Burn is pinned at 0.22.1.
  - If cargo-deny starts failing once RustSec publishes the advisories, add a `deny.toml` ignore entry with this reason. The owner decides whether to dismiss the GitHub alerts.
- **`target/` size:** debug test binaries with `cuda` or `vulkan` are about 1.2 GB each. Plan for 20 GB.

## 5. Where M1 starts

[roadmap.md, M1](roadmap.md#m1--graph-ir-and-interpreter). Specs to refine first:
- [graph-ir.md](specs/graph-ir.md): core opset, validation, symbolic shapes, `repeat` blocks.
- [testing-strategy.md](testing-strategy.md): reference fixtures.
- [public-api.md](specs/public-api.md): the facade surface for graphs.

The order below is filed as issues #35–#47 on the [M1 milestone](https://github.com/Jannik0/Polypous/milestone/2).
Each issue names its dependencies and whether it runs in the cloud or on the reference machine.

1. **Spec pass on `graph-ir.md` for the core opset.** Decide the M1 op list and its exact semantics, with a Proposed ADR for anything that changes a decision.
2. **Reference fixture tooling.** Write `tools/reference/requirements.txt`, the first PyTorch generators and the fixture layout from testing-strategy.md. The cloud environment already allows `download.pytorch.org`.
3. **`polypous-ir`:** types, JSON serde, validation, shape inference, `repeat`. Pure Rust, so cloud-friendly.
4. **`polypous-formats`:** safetensors reader (mmap, sharded, size checks on untrusted input). Cloud-friendly.
5. **`polypous-backend` core-op implementations on `DynTensor`**, with reference tests on CPU (cloud) and CUDA/Vulkan (reference machine).
6. **`polypous-runtime`:** an interpreter on a single device.
7. **Facade and CLI surface** for loading and running a graph.
8. **M1 exit check on the reference machine:** an MLP and a small CNN from JSON match PyTorch on CPU, CUDA and Vulkan.

Items 3 and 4 can run in parallel with 2. Item 5 needs 1. Item 6 needs 3 and 5.
