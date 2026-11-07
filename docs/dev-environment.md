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
| **Python 3.14** | Generating and checking reference fixtures only (`tools/reference/`, see [below](#python-reference-environment)), never needed at runtime. The CI helper scripts and the fixture tooling tests need only Python 3.12+ and its standard library |

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
- Without a driver, a `cuda` build starts, finds no CUDA devices and says why, for example
  `cuda: compiled in, but no device found: the driver library libcuda.so.1 is not installed`
  (checked on Linux by hiding `libcuda`; on Windows approximated with `CUDA_VISIBLE_DEVICES=-1`,
  since `nvcuda.dll` can't be hidden without admin rights).

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

What Polypous detects, with isolated probing and the probe cache (the `GPU` column groups rows that
are one physical card):

```bash
cargo run -p polypous-cli --features cuda,vulkan -- devices
```

Add `RUST_LOG=polypous=info` to see prober starts and cache hits, `--json --report` for the full
probe report, and `--probe-cache-dir <dir>` for a cold run without touching the default cache.
After `cargo test`, run `cargo build -p polypous-cli --features cuda,vulkan` again before using
`target/debug/polyp`: the test build contains the prober's fault injection.

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
  `polyp devices` works on a stock install without any `VK_DRIVER_FILES` workaround: `vulkan:cpu:0`
  is shown as skipped. (Mesa 25.2.8 lavapipe on Ubuntu, as in CI and the cloud sessions, works.)
- The backend GPU tests run in parallel on a stock install (no `--test-threads=1`, no
  `VK_DRIVER_FILES`). Enumerating Vulkan devices while another thread computes on a Vulkan GPU used
  to crash the process here, because the Vulkan loader loads and unloads Mesa's device-less ICDs on
  every enumeration; Polypous now enumerates each backend once per process (#26, placement.md §1
  "When enumeration happens"). The first open of a Vulkan device still enumerates once inside
  CubeCL, so tests open their Vulkan GPUs before computing on several threads.
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

Only needed to generate or check reference fixtures ([testing-strategy.md §3](testing-strategy.md#3-reference-fixtures)).
The committed fixtures are generated with **Python 3.14** and the versions pinned in
[`tools/reference/requirements.txt`](../tools/reference/requirements.txt) (CPU-only PyTorch,
`safetensors`, `numpy`). The generators refuse to run with other versions.

With [uv](https://docs.astral.sh/uv/) (fetches Python 3.14 if needed):

```bash
uv venv --python 3.14 tools/reference/.venv
# Linux:   source tools/reference/.venv/bin/activate
# Windows: tools\reference\.venv\Scripts\activate
uv pip install -r tools/reference/requirements.txt
```

Without uv: `py -3.14 -m venv tools/reference/.venv` (Windows) or `python3.14 -m venv
tools/reference/.venv` (Linux), activate it, then `pip install -r tools/reference/requirements.txt`.

`requirements.txt` adds PyTorch's CPU wheel index (`https://download.pytorch.org/whl/cpu`) with
`--extra-index-url` and pins `torch==X.Y.Z+cpu`. The `+cpu` version exists only on that index, so
pip and uv can't pick a CUDA build from PyPI by accident, and the same file works on Linux (x86-64)
and Windows (x64). PyTorch publishes no `+cpu` wheels for macOS. That index's links point to
`download-r2.pytorch.org`, so a firewall or proxy must allow that host as well as
`download.pytorch.org`.

Then, from the repository root:

```bash
python tools/reference/gen_ops.py            # regenerate fixtures/small/ops/
python tools/reference/gen_ops.py --check    # compare with the committed fixtures
python -m unittest discover -s tools/reference -v
```

## Cloud environments for agents

Agents can run as Claude Code cloud sessions (claude.ai/code, the Claude apps, or `claude --cloud`).
A cloud session is an Ubuntu 24.04 VM with **4 vCPUs, 16 GB RAM, 30 GB disk and no GPU**. Rust and
Python are pre-installed.

**One-time setup**

1. Install the Claude GitHub App on the repository (github.com/apps/claude). The repository is
   private, so sessions need it to clone and push. Auto-fix on PRs also depends on it.
2. Create a cloud environment:
   - **Network access:** *Custom*, keeping the default (Trusted) list, which covers crates.io,
     rustup, PyPI, GitHub and Ubuntu apt. Add `huggingface.co`, `*.hf.co` (test models),
     `download.pytorch.org` and `download-r2.pytorch.org` (CPU-only PyTorch for fixture generation).
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
