# Development Environment

Setup for the reference machine (Windows 11 + Debian dual boot, RTX 3060 Laptop). Exact versions
are pinned in M0 (`rust-toolchain.toml`, [ADR-0007](adr/0007-burn-version-pinning.md)).

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
| **CUDA** | NVIDIA driver + CUDA Toolkit (version per Burn's CUDA backend requirements, recorded in M0) | NVIDIA driver + CUDA Toolkit from NVIDIA's Debian repository |
| **Metal** | — | — (macOS only, not available on the reference machine) |

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

- `build-essential pkg-config libssl-dev` (if dependencies need them).
- For Vulkan software-driver testing (as in CI): `mesa-vulkan-drivers` provides lavapipe. Select it with
  `VK_ICD_FILENAMES` / `VK_DRIVER_FILES` pointing at `lvp_icd.*.json`.

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
