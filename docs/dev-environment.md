# Development Environment

Setup for the reference machine (Windows 11 + Debian dual boot, RTX 3060 Laptop). Exact versions
are pinned in M0 (`rust-toolchain.toml`, [ADR-0007](adr/0007-burn-version-pinning.md)).

## Required (all platforms)

| Tool | Purpose |
|---|---|
| **Rust** via [rustup](https://rustup.rs) | The toolchain pinned in `rust-toolchain.toml` is installed automatically |
| **Git** | — |
| **Python 3.11+** | Generating reference fixtures only (`tools/reference/`), never needed at runtime |

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
python -m venv tools/reference/.venv
```

Activate it (`tools/reference/.venv/Scripts/activate` on Windows, `tools/reference/.venv/bin/activate` on Linux), then:

```bash
pip install -r tools/reference/requirements.txt
```

`requirements.txt` is added together with the first fixture generator (M1).
