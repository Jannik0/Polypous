# Changelog

All notable changes to Polypous are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

### Added
- Project brief, architecture, roadmap, specifications, ADRs and contributor/agent guides.
- Cloud environment setup script (`tools/cloud/setup.sh`) and documentation for cloud agents.
- Cargo workspace with empty skeleton crates (`polypous-ir`, `-formats`, `-polyp`, `-backend`, `-quant`,
  `-runtime`, `-train`, `-pipelines`, `polypous`, and `polypous-cli` with the `polyp` binary), workspace
  lints (`unsafe_code` denied, `unwrap_used`/`expect_used` warned outside tests) and the Rust toolchain
  pinned to 1.99.1 in `rust-toolchain.toml`.
- CI on GitHub Actions: fmt, clippy and CPU tests on Linux and Windows, `cargo deny` (licenses and
  advisories, `deny.toml`), a dependency-boundary check (`tools/ci/check_dependency_boundary.py`),
  Vulkan tests on Mesa lavapipe, and a macOS build-only job for Metal (on `main`, manual dispatch, and
  PRs that touch the backend crates or the dependency set).
- Burn pinned to `=0.22.1` and CubeCL to `=0.11.3` as workspace dependencies, used only by
  `polypous-backend`. Backend features `cpu` (default, Burn's pure-Rust `flex` backend), `cuda`,
  `vulkan` and `metal`. `burn_smoke` example to check a backend on a given machine.
- ADR-0011 (accepted): use Burn's runtime `Dispatch` backend for backend selection; supersedes part of ADR-0003.
- ADR-0012: MPL-2.0 allowed for unmodified dependencies, per crate (`colored`, `buildid`, `option-ext`
  as `deny.toml` exceptions).
- ADR-0013: device probing runs in a separate prober process with a probe cache, crash and hang
  recovery, and an in-process fallback. Apps can re-run the probe through the API (`probe_devices`,
  `Engine::refresh_devices`) or pass in their own report.
- Device discovery and probing in `polypous-backend` (`device` module): `BackendKind`, `DeviceId`,
  `DeviceInfo` (memory excluding Windows' shared GPU memory, PCI/LUID identity to match one GPU
  across CUDA and Vulkan, capabilities, probe status; serde JSON with stable `snake_case` names),
  `enumerate_devices`, `describe_device`, `probe_device` (panics caught) and `select_primary_gpu`.
