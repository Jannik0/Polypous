# ADR-0007: Pin Burn and isolate it in one crate

- **Status:** Accepted
- **Date:** 2026-11-02

## Context

Burn is pre-1.0, and its API changes between releases. Letting Burn types spread through the codebase would make every upgrade touch everything.

## Decision

- Burn and CubeCL are pinned to an **exact version** (`=x.y.z`) in the workspace `Cargo.toml`. The
  version is chosen at M0 (latest stable release at that time) and recorded in this ADR's follow-up note.
- Only `polypous-backend` and `polypous-quant` may depend on `burn*` / `cubecl*` crates. No Burn type
  appears in the public API of any other crate. This is enforced by a CI check on `Cargo.toml` dependencies.
- An upgrade is a **dedicated task**: bump, fix, run the full reference test suite on CPU, CUDA and
  Vulkan on the reference machine, and record notable changes in `CHANGELOG.md`.
- Upgrades happen deliberately, for example when a Burn release brings something we need, not automatically.

## Consequences

- Upgrades are contained in two crates and checked against the full test suite.
- We may lag behind Burn releases, which is acceptable.

## Follow-up notes

### M0 pin (2026-11-02, issue #2)

- **Burn `=0.22.1`** (released 2026-10-20) and **CubeCL `=0.11.3`** (2026-10-22), the latest stable
  releases. Burn 0.22.1 requires `cubecl ^0.11.1`, so 0.11.3 is the version Burn itself resolves to.
  `0.23.0-pre.1` / `0.12.0-pre.1` exist but are pre-releases and were not considered.
- **Rust toolchain 1.99.1**, pinned in `rust-toolchain.toml` (MSRV `rust-version = "1.99"`). Burn
  0.22.1 builds with it.
- The `=` pin covers the top-level `burn` and `cubecl` crates. Their sub-crates (`burn-core`,
  `burn-cubecl`, `cubecl-cuda`, `cudarc`, `wgpu`, …) use caret requirements and are held by the
  committed `Cargo.lock`. Running `cargo update` on them is part of an upgrade task, not routine work.
- `cubecl` is an optional dependency of `polypous-backend` (enabled by the GPU features). It is not
  used in code yet. It makes the exact CubeCL pin part of dependency resolution, so `polypous-quant`
  later gets the same CubeCL as Burn.
- Burn is used with `default-features = false`. Only `std` plus one backend feature per Polypous
  feature are enabled. Burn's default features would pull in `optim`/`autodiff`, which are enabled
  later when training needs them.

**Feature mapping (`polypous-backend`)**

| Polypous feature | Burn feature | Burn backend / runtime |
|---|---|---|
| `cpu` (default) | `flex` | `burn-flex`: pure-Rust CPU backend. Burn's own `cpu` feature is the CubeCL CPU runtime (MLIR/LLVM), much heavier, not used. `ndarray` is deprecated in Burn 0.22 |
| `cuda` | `cuda` | CubeCL CUDA runtime (`cubecl-cuda` 0.11.3 → `cudarc` 0.19.10) |
| `vulkan` | `vulkan` | CubeCL wgpu runtime (`wgpu` 30) pinned to Vulkan, SPIR-V code generation |
| `metal` | `metal` | CubeCL wgpu runtime pinned to Metal, MSL code generation |

**CUDA requirements (from source, not yet verified on hardware)**

- *Build:* **no CUDA toolkit needed, but network access (or a pre-filled cache) is.**
  - `cubecl-cuda` 0.11 always depends on `cubecl-llvm` (`cubecl-cuda-0.11.3/Cargo.toml`,
    `[dependencies.cubecl-llvm]`), whose build dependency `tracel-llvm-bundler` downloads a prebuilt
    LLVM 23.1 bundle (~45 MB `.tar.xz`) from `github.com/tracel-ai/tracel-llvm/releases` and checks
    its SHA-256 (`tracel-llvm-bundler-23.1.0-4/build.rs:76-170`, `src/config.rs:11-16`). The
    archive is cached in `~/.cache/tracel/`, unpacked into the platform's local data directory
    (`~/.local/share/tracel/` on Linux, `%LOCALAPPDATA%\tracel\` on Windows), and linked **statically**
    (`src/llvm_sys.rs:49-66`, `llvm-config --link-static`). The download uses `ureq` with its own
    root certificates, so it fails behind TLS-intercepting proxies (it did in the cloud session,
    `InvalidCertificate(UnknownIssuer)`). Workaround: download both files with `curl` into
    `~/.cache/tracel/` under the names `tracel-llvm-23.1.0-4-<platform>.tar.xz` and
    `tracel-llvm-23.1.0-4-<platform>.checksums.json` and build with
    `TRACEL_LLVM_BUNDLER_SKIP_CHECKSUM_DOWNLOAD=1`. The archive hash is still checked.
  - cudarc's build script runs `nvcc --version` and, if that fails, uses the bindings for the
    newest CUDA version it knows, 13.4 (`cudarc-0.19.10/build.rs:11-31, 158-171`). With `nvcc` 13.2 on
    `PATH` it uses the 13.2 bindings. `CUDARC_CUDA_VERSION=13020` forces them (`build.rs:57-70`).
    cudarc supports CUDA 11.4–13.4.
- *Kernel compilation:* by default CubeCL 0.11 compiles CUDA kernels **with its bundled LLVM to PTX**,
  not with NVRTC (`cubecl-cuda-0.11.3/src/compiler.rs:11-31`: `CudaBackend::Llvm` unless the `cpp`
  feature, exposed as `cubecl/cuda-cpp`, is on; Polypous does not enable it). The PTX version is
  chosen from the **driver** version, for example PTX 9.2 for a CUDA 13.2 driver
  (`cubecl-llvm-0.11.3/src/nvptx/ptx_version.rs:18-43`). NVRTC is only called on the C++ path
  (`cubecl-cuda-0.11.3/src/compute/context.rs:319-460`).
- *Runtime:* only the **NVIDIA driver** (`libcuda.so` / `nvcuda.dll`), loaded with `libloading` on
  first use rather than linked (cudarc `fallback-dynamic-loading`, `cubecl-cuda-0.11.3/Cargo.toml`;
  `cudarc-0.19.10/src/driver/sys/mod.rs`, `culib()`). The CUDA Toolkit is not needed to run
  (confirmed on the reference machine on Windows and Debian, see below).
- *Missing driver:* device enumeration checks for `libcuda` first and returns no devices if it is
  missing (`cubecl-cuda-0.11.3/src/runtime.rs:463-477`). Checked in the cloud session (Linux, no
  CUDA): the `cuda` build of `burn_smoke` starts and reports no CUDA device. But device
  initialization `unwrap()`s driver calls (`runtime.rs:70-72`), so a driver that is present but
  broken panics. Device probing (#4) must isolate such panics.
- **Verified on the reference machine (Windows), see "Reference machine results" below:** the CUDA
  backend runs on the RTX 3060 with driver 591.74 / CUDA 13.2, also with the CUDA toolkit removed
  from `PATH` (no NVRTC reachable), and on Debian with driver 615.71.10 (`LD_DEBUG=libs` shows only
  `libcuda.so` from the driver, no NVRTC or `cudart`). Use
  `crates/polypous-backend/examples/burn_smoke.rs`.

- *Binary size:* because LLVM is linked statically, a `cuda` build is large. The `burn_smoke`
  example with `cuda` is about 1 GB in debug on Linux (debug info inside the binary; test binaries
  about 1.2 GB), and on Windows 137 MB in debug (debug info in a separate PDB) and **90 MB in release**.

**Vulkan requirements (from source)**

- A Vulkan loader and an ICD for the GPU (the normal driver install). wgpu 30 needs Vulkan ≥ 1.1 or
  `VK_KHR_get_physical_device_properties2`.
- For CubeCL's SPIR-V path the device needs Vulkan 1.1 (SPIR-V ≥ 1.3), `vulkanMemoryModel` with
  device scope and `bufferDeviceAddress` (`cubecl-wgpu-0.11.3/src/backend/vulkan/features.rs:315-327`).
  Devices without them fall back to WGSL through wgpu/naga (`cubecl-wgpu-0.11.3/src/backend/vulkan.rs:380-386`).
  Vulkan 1.3 drivers give SPIR-V 1.6. Current NVIDIA and Intel Iris Xe drivers report Vulkan 1.3 or
  newer (to be confirmed on the reference machine with `vulkaninfo --summary`). Mesa lavapipe in the
  cloud session reports 1.4.318.
- Checked in the cloud session on lavapipe: `burn_smoke` runs a 2×2 matmul correctly, and with no
  Vulkan ICD available (`VK_DRIVER_FILES=/nonexistent.json`) it starts and reports no Vulkan device.
- No Vulkan SDK needed to build or run. The SDK only adds validation layers.

**Reference machine results (Windows 11, 2026-11-02)**

Razer Blade 15 (2022): RTX 3060 Laptop (driver 591.74, CUDA toolkit 13.2, Vulkan 1.4.325) and
Intel Iris Xe (driver 30.0.101.1298, Vulkan 1.2.201). Rust 1.99.1.

| Check | Result |
|---|---|
| fmt, clippy `-D warnings` (default, `vulkan`, `cpu,vulkan,cuda`), tests | ✅ |
| `burn_smoke cpu` (`flex`) | ✅ `[7.0, 10.0, 15.0, 22.0]` |
| `burn_smoke vulkan 0`: RTX 3060 (Burn lists it first, as `DiscreteGpu(0)`) | ✅ correct result |
| `burn_smoke vulkan 1`: Intel Iris Xe (`IntegratedGpu(0)`) | ❌ **process crash**: access violation `0xC0000005` inside Intel's Vulkan driver `igvk64.dll` 30.0.101.1298 (Windows event log). Not a Rust panic, so `catch_unwind` cannot contain it |
| First `cuda` build on a normal network | ✅ the LLVM bundle downloaded without the workaround (2 min 12 s build) |
| `burn_smoke cuda`: RTX 3060 | ✅ correct result |
| `burn_smoke cuda` with the CUDA toolkit removed from `PATH` and `CUDA_PATH` unset (NVRTC not reachable) | ✅ correct result: only the driver is needed |
| `cuda` release build size | 90 MB |

**Rerun after driver updates (2026-11-03):** Intel Iris Xe driver 32.0.101.7092 (Vulkan 1.4.323)
and NVIDIA driver 617.42 (Vulkan 1.4.351).

| Check | Result |
|---|---|
| `burn_smoke vulkan 0`: RTX 3060 | ✅ correct result |
| `burn_smoke vulkan 1`: Intel Iris Xe, 3 runs | ✅ correct result every time, no crash events |
| `burn_smoke cuda`, also with the CUDA toolkit removed from `PATH` | ✅ correct result |

So the crash was a bug in the old Intel driver, not in Burn or Polypous.

**Reference machine results (Debian 13, 2026-11-04)**

Debian 13.7, kernel 6.12.115, NVIDIA driver 615.71.10 (open kernel module), Mesa 25.0.7 (ANV for
the Iris Xe, llvmpipe with LLVM 19.1.7), Vulkan loader 1.4.309, CUDA toolkit 13.4 installed but not
on `PATH`. Rust 1.99.1. Tested at commit `6f0bd20`.

| Check | Result |
|---|---|
| fmt, clippy `-D warnings`, `cargo test --workspace` | ✅ |
| `burn_smoke cuda`: RTX 3060, also with the toolkit hidden | ✅ correct result. `LD_DEBUG=libs`: only `libcuda.so` from the driver, no `libnvrtc`/`libcudart`, although the toolkit's lib dirs are in `ld.so.conf.d` |
| `burn_smoke vulkan 0`: RTX 3060 / `vulkan 1`: Iris Xe | ✅ correct result |
| `burn_smoke vulkan 2`: lavapipe (llvmpipe), 3 runs | ❌ **process crash** (SIGSEGV), also with lavapipe as the only ICD. Not a Rust panic |
| Device discovery and DynTensor GPU tests (`cuda,vulkan`, `--include-ignored`) | ❌ crash on lavapipe; ✅ all pass with lavapipe hidden by `VK_DRIVER_FILES` |

Device discovery with lavapipe hidden: RTX 3060 through CUDA (5.72 GiB; Windows 6.00 GiB) and
Vulkan (6.00 GiB, the device-local heap in `vulkaninfo`; Windows 5.85 GiB) with the same PCI id
`0000:01:00.0`, no LUID on Linux. Iris Xe with its own PCI id `0000:00:02.0`, 7.66 GiB, unified. The
primary GPU is `cuda:0`. The dtype support matches Windows: CUDA has bf16, Vulkan doesn't.

Lavapipe on Ubuntu's Mesa 25.2.8 (CI, cloud sessions) works, so this looks like a Mesa 25.0 /
LLVM 19 bug. No backtrace yet (needs root for `dmesg`, and `gdb`). Like the old Intel driver on
Windows, it shows that a stock install can crash the process during probing: a second real-hardware
case for ADR-0013.

Not checked yet: a `cuda` build on a Windows machine without any NVIDIA driver (confirmed on Linux
only), Metal (CI build only).

**Consequence for device probing (#4):** a GPU driver can crash the whole process during device
initialization. That can't be caught inside the process, and casual users often run old drivers.
[ADR-0013](0013-crash-safe-device-probing.md) (accepted) moves probing into a separate prober process.

**Licenses of the new dependencies**

Everything Burn pulls in is under permissive licenses, with these exceptions (checked with
`cargo metadata` and `cargo tree -i`):

- `colored` 3.1.1 (**MPL-2.0**), always: `burn-tensor`'s `std` feature enables it, and Polypous needs `std`.
- `buildid` 1.1.0 (**MPL-2.0**, via `cubecl-runtime`) with `vulkan`/`cuda`/`metal`, and
  `option-ext` 0.2.0 (**MPL-2.0**, via `dirs-sys`) with `cuda`.
- `webpki-roots` (CDLA-Permissive-2.0) and LLVM (Apache-2.0 WITH LLVM-exception), both permissive.

Decision: the three MPL-2.0 crates are allowed as per-crate `exceptions` in `deny.toml`, because
Polypous uses them unmodified ([ADR-0012](0012-mpl-for-unmodified-dependencies.md)).

**Router evaluation (ADR-0003)**

Burn 0.22 has a runtime-dispatched backend: `burn::Tensor<D, K>` has no backend type parameter, and
the backend is chosen by `burn::Device` (an enum over the compiled-in backends, generated dispatch,
cross-backend `to_device` through host memory). This covers what ADR-0003 plans to build by hand, so
the result is [ADR-0011](0011-use-burn-dispatch-for-backend-selection.md), **accepted** on 2026-11-02.
`polypous-backend` builds on `burn::Tensor` + `burn::Device`; ADR-0003's goals and rules still apply.
