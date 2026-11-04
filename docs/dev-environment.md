# Development Environment

Setup for the reference machine (Windows 11 + Debian dual boot, RTX 3060 Laptop). Exact versions
are pinned in M0: Rust 1.99.1 in `rust-toolchain.toml`, Burn 0.22.1 and CubeCL 0.11.3 in the workspace
`Cargo.toml` ([ADR-0007](adr/0007-burn-version-pinning.md), which also lists the sources for the GPU
requirements below).

## Required (all platforms)

| Tool | Purpose |
|---|---|
| **Rust** via [rustup](https://rustup.rs) | The toolchain pinned in `rust-toolchain.toml` is installed automatically |
| **Git** | — |
| **Python 3.12–3.14** | Generating reference fixtures only (`tools/reference/`), never needed at runtime. Use a version PyTorch publishes wheels for. Newly released Python versions often lack them for a few months |

## GPU backends (optional, for GPU tests)

| Backend | Windows | Debian |
|---|---|---|
| **Vulkan** | Comes with NVIDIA/Intel drivers. [Vulkan SDK](https://vulkan.lunarg.com/) for validation layers is recommended | `mesa-vulkan-drivers`, `vulkan-tools` (`vulkaninfo`); NVIDIA proprietary driver for the dGPU. Validation layers: `vulkan-validationlayers` |
| **CUDA** | NVIDIA driver. CUDA Toolkit 11.4–13.4 optional (only `nvcc` is used, at build time; reference machine: 13.2) | NVIDIA driver; CUDA Toolkit 11.4–13.4 from NVIDIA's Debian repository optional, as on Windows |
| **Metal** | — | — (macOS only, not available on the reference machine) |

**Vulkan requirements:** a Vulkan loader and the GPU driver's ICD, Vulkan ≥ 1.1. For the fast
SPIR-V path the device also needs `vulkanMemoryModel` (device scope) and `bufferDeviceAddress`
(standard in Vulkan 1.2+ drivers). Other devices fall back to WGSL. Nothing Vulkan-specific is
needed to *build* with `--features vulkan`.

**CUDA requirements:**

- *Build:* no CUDA toolkit needed. But the first `--features cuda` build **downloads a prebuilt LLVM
  bundle (~45 MB) from GitHub** (`tracel-llvm-bundler`, used by CubeCL to compile CUDA kernels) and
  links it statically. Behind a TLS-intercepting proxy that download fails. Pre-fill the cache as
  described in [ADR-0007's follow-up notes](adr/0007-burn-version-pinning.md#follow-up-notes).
  If `nvcc` is on `PATH`, cudarc uses bindings for its version, otherwise the newest supported one
  (13.4). `CUDARC_CUDA_VERSION=13020` selects the 13.2 bindings without `nvcc`. NVIDIA's Debian
  packages install the toolkit under `/usr/local/cuda-<version>` without putting `nvcc` on `PATH`,
  so set `CUDARC_CUDA_VERSION` when the installed toolkit is not 13.4.
- *Run:* only the NVIDIA driver (`libcuda.so` / `nvcuda.dll`), loaded at runtime. Kernels are
  compiled to PTX with the bundled LLVM, so NVRTC and the rest of the CUDA Toolkit are not needed
  to run (confirmed on the reference machine on Windows and Debian, see ADR-0007's follow-up
  notes). Toolkit versions 11.4–13.4 are supported.
- Without a driver, a `cuda` build starts and finds no CUDA devices (checked on Linux without
  CUDA; Windows not checked yet).

Smoke test for a backend (runs a 2×2 matmul on the first device found, or reports that there is none):

```bash
cargo run -p polypous-backend --example burn_smoke --features cuda -- cuda
```

```bash
cargo run -p polypous-backend --example burn_smoke --features vulkan -- vulkan
```

Check with:

```bash
nvidia-smi
```

```bash
vulkaninfo --summary
```

Once M0 is done, `cargo run -p polypous-cli -- devices` shows what Polypous detects.

## Windows notes

- Use the MSVC toolchain (`x86_64-pc-windows-msvc`). Install "Desktop development with C++" from the Visual Studio Build Tools.
- Optimus: for CUDA nothing is needed. For Vulkan, both GPUs are listed, and Polypous picks the discrete one.
- Long paths: `git config --global core.longpaths true`.

## Debian notes

- `build-essential pkg-config`, and `vulkan-tools` for `vulkaninfo`. (`libssl-dev` is not needed:
  nothing in `Cargo.lock` links OpenSSL.)
- `mesa-vulkan-drivers` provides the Intel driver (ANV) **and** lavapipe (llvmpipe), and Vulkan always
  lists lavapipe next to the real GPUs (as `vulkan:cpu:0`). **On Debian 13 (Mesa 25.0.7, LLVM 19)
  lavapipe crashes the process (SIGSEGV) on its first compute work**. Polypous therefore lists
  software rasterizers but does not open or probe them unless the caller opts in
  (`software_rasterizers`, placement.md §1; issue #21), and the GPU tests skip them the same way, so
  **GPU tests and `polyp devices` work on a stock install without any `VK_DRIVER_FILES`
  workaround**: `vulkan:cpu:0` is shown as skipped. (Mesa 25.2.8 lavapipe on Ubuntu, as in CI and
  the cloud sessions, works.)
- For Vulkan software-driver testing (as in CI), select lavapipe alone with `VK_DRIVER_FILES`
  pointing at `lvp_icd.*.json` (for example `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json`)
  **and** set the test-only variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, so the GPU tests open and
  test it instead of skipping it.
- A first `--features vulkan` or `--features cuda` build compiles wgpu/CubeCL and takes several
  minutes (about 3.5 min for `vulkan` on a 4-vCPU cloud VM). The default `cpu` build is much lighter.
- **Disk space:** each debug test binary with `cuda` or `vulkan` is about 1.2 GB on Linux (debug info
  inside the binary), and every feature set builds its own. Running the CPU, GPU and smoke checks
  filled `target/` with about 17 GB. Plan for 20 GB, or run `cargo clean` between feature sets.

## Python reference environment

```bash
uv venv --python 3.14 tools/reference/.venv
```

(or `py -3.14 -m venv tools/reference/.venv` without uv). Activate it (`tools/reference/.venv/Scripts/activate` on Windows, `tools/reference/.venv/bin/activate` on Linux), then:

```bash
pip install -r tools/reference/requirements.txt
```

`requirements.txt` is added together with the first fixture generator (M1).

## Cloud environments for agents

Agents can run as Claude Code cloud sessions (claude.ai/code, the Claude apps, or `claude --cloud`).
A cloud session is an Ubuntu 24.04 VM with **4 vCPUs, 16 GB RAM, 30 GB disk and no GPU**. Rust and
Python are pre-installed.

**One-time setup**

1. Install the Claude GitHub App on the repository (github.com/apps/claude). The repository is
   private, so sessions need it to clone and push. Auto-fix on PRs also depends on it.
2. Create a cloud environment:
   - **Network access:** *Custom*, keeping the default (Trusted) list, which covers crates.io,
     rustup, PyPI, GitHub and Ubuntu apt. Add `huggingface.co`, `*.hf.co` (test models) and
     `download.pytorch.org` (CPU-only PyTorch for fixture generation).
   - **Setup script:** the contents of [`tools/cloud/setup.sh`](../tools/cloud/setup.sh).
   - **Environment variables:** none needed. Never put secrets there: everyone using the environment can read them.

**What cloud agents can and cannot verify**

| Works in the cloud | Needs the reference machine |
|---|---|
| Building, fmt, clippy, all CPU tests | CUDA tests |
| Vulkan tests on lavapipe (software, slow) | Real-GPU Vulkan tests (NVIDIA + Intel iGPU) |
| Parsers, IR, planner logic, CPU kernels, tokenizers | Model-level and placement scenarios (7B / 30B models) |

Cloud agents state in their PR which backends they ran (normally CPU + lavapipe). The owner, or a
local session on the reference machine, runs the GPU tests before merging changes to backend or kernel code.
