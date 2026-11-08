# Project Status

What exists, what is known to be missing or fragile, and where the next milestone starts. Update
this file when a milestone closes or when a known limitation is fixed or found. Last updated:
2026-11-08, M1 in progress (#35–#37 done).

## 1. Milestones

| Milestone | State | Evidence |
|---|---|---|
| M0 — Foundations | **Done** (closed 2026-11-06) | Exit check results: [#7](https://github.com/Jannik0/Polypous/issues/7#issuecomment-6684237754), on Windows 11 and Debian 13 |
| M1 — Graph IR and interpreter | In progress. Issues #35–#47 on the [M1 milestone](https://github.com/Jannik0/Polypous/milestone/2); done: #35 (spec pass), #36 (fixture tooling), #37 (safetensors) | [roadmap.md](roadmap.md#m1--graph-ir-and-interpreter) |
| M2–M7 | Not started | [roadmap.md](roadmap.md) |

## 2. What exists after M0

| Crate | State |
|---|---|
| `polypous-backend` | Real code. Device discovery, probing, primary GPU selection and physical-GPU grouping (`device`). `DynTensor` with runtime rank 1–6 and dtypes f32/f16/bf16/i64/i32/bool over `burn::Tensor` (`tensor`). Crash-safe probing: prober protocol, supervisor, probe cache, driver fingerprints (Linux and Windows), crash journal, reasons for backends without devices (`probe`) |
| `polypous` (facade) | Real code. `Engine`, `EngineConfig`, facade device types (`DeviceInfo` with `physical_gpu` and `primary`), probe API (`polypous::probe`), `Error`/`ErrorKind`, panics caught at the boundary |
| `polypous-cli` (`polyp`) | `polyp devices [--json] [--report] [probe options]`, `polyp probe --json` |
| `polypous-formats` | safetensors reader (memory-mapped, validated header, sharded checkpoints) and writer (M1, #37). GGUF follows in M3 |
| `polypous-ir`, `-runtime`, `-polyp`, `-quant`, `-train`, `-pipelines` | Skeletons (a doc comment each). M1 fills `-ir` and `-runtime` |
| `tools/reference/`, `fixtures/` | PyTorch fixture generator with `--check`, first op fixtures, `fetch_fixtures.py` and the fixture checks in CI (M1, #36) |

Specs that describe implemented behavior: [placement.md §1](specs/placement.md#1-device-discovery)
(discovery, probing, cache, fingerprints, enumeration rules), [public-api.md §1.1, §1.2, §2](specs/public-api.md)
and [formats.md §1](specs/formats.md) (safetensors). [graph-ir.md](specs/graph-ir.md) is binding for M1 as its
status line says, but not implemented yet. All other specs are drafts until their milestone starts.

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

Nothing here blocks M1. Fix an item when its area is touched, or file an issue first.

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
