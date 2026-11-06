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
- CI job "PR title and description" (`tools/ci/check_pr_text.py`): fails a PR whose title or
  description contains a Claude Code session link, a `Claude-Session` trailer or a model name, or an
  agent-authored PR whose description does not end with the fixed Claude Code footer.
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
- `DynTensor` in `polypous-backend` (`tensor` module): tensors with runtime rank (1–6) and dtype
  (`f32`, `f16`, `bf16`, `i64`, `i32`, `bool`) over `burn::Tensor`, with `TensorData` conversion
  (little-endian, row-major), `shape`, `dtype`, `reshape`, `cast`, `add`, `matmul` and `to_device`.
  Bad input gives a `TensorError` instead of a panic.
- `DynTensor::from_data`, `to_device` and `device` are public and take or return a `DeviceId` from
  `list_devices()`. A device whose backend is not compiled in, or that doesn't exist, gives the new
  `TensorError::DeviceUnavailable` instead of a panic. Device names in `TensorError` messages are
  `DeviceId` texts such as `vulkan:discrete_gpu:0`.
- `polypous` facade: `Engine::new(EngineConfig::default())` discovers and probes devices (in process
  until the prober process lands), `engine.devices()` / `engine.primary_gpu()` return the facade's own
  serde device types (placement.md §1 fields plus `primary`), and `polypous::Error` with the stable
  `ErrorKind`; panics at the API boundary become `Internal` errors. Backend features `cpu`, `cuda`,
  `vulkan`, `metal` are forwarded through `polypous` and `polypous-cli`.
- `polyp devices` prints a table of devices (backend, name, kind, dedicated memory, unified memory,
  status) with the primary GPU marked; `polyp devices --json` prints the JSON documented in
  public-api.md §1.2. Logs go to stderr (`RUST_LOG=polypous=debug` shows probe details).
- `EnumerateOptions { software_rasterizers }`, `enumerate_devices_with`, `skipped_device` and
  `DeviceId::is_software_rasterizer` in `polypous-backend`, and the `EngineConfig` field
  `software_rasterizers` (default `false`) in `polypous`.
- Crash-safe device probing (ADR-0013, #14): `Engine::new` takes devices from a probe cache (no
  prober process when every device's driver fingerprint is unchanged), else probes them in a prober
  process (`polyp` or the app itself re-launched via `polypous::probe::run_prober_if_requested()`,
  or `EngineConfig::prober_path`), restarting it after a crash or hang so one bad driver only makes
  its own device unavailable (`crashed during probe` / `probe timed out`), else in process with a
  crash journal so a crashing driver takes the app down at most once. Linux driver fingerprints from
  sysfs, `/proc/driver/nvidia/version` and the Vulkan ICD manifests; Windows and macOS compare the
  enumeration instead (#28).
- Probe API: `polypous::probe::{probe_devices, run_prober_if_requested, run_prober}`,
  `Engine::refresh_devices`, `Engine::probe_report`, `ProbeOptions`, `ProbeEvent`, `ProbeReport`,
  `Isolation`, `CancellationToken`, and `EngineConfig` fields `probe_report`, `prober_path`,
  `probe_cache_dir`, `probe_isolation`, `probe_timeout_per_device_ms`.
- `polyp probe --json` runs the prober and prints its JSON-lines protocol. `polyp devices` gets
  `--software-rasterizers`, `--isolation`, `--probe-cache-dir`, `--prober-path`,
  `--timeout-per-device-ms` and `--json --report`, and prints how devices were probed under the table.

### Changed
- `polyp devices` and `Engine::new` in apps that call `run_prober_if_requested()` report probe
  results as `isolated: true`. Apps without a prober probe in process as before, now with a crash
  journal and a warning in the report.
- Device discovery lists software rasterizers (Vulkan/Metal adapters of type `cpu`, such as
  lavapipe) but no longer opens or probes them unless `software_rasterizers` is set. They are shown
  as unavailable with the new reason code `software_rasterizer_skipped`. This keeps lavapipe on
  Debian 13 (Mesa 25.0.7) from crashing `polyp devices`, `Engine::new` and the GPU tests; the
  `VK_DRIVER_FILES` workaround is no longer needed for that. The GPU tests opt in with the test-only
  variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, which the CI lavapipe job sets.
- Each backend's devices are enumerated at most once per process; `list_devices`,
  `list_backend_devices`, `enumerate_devices*`, `DynTensor` device checks and `Engine` discovery reuse
  the cached list, and unlisted ids fail against it. `describe_device` and `probe_device` reject ids
  that aren't listed without opening them. Devices added after the first enumeration are not seen
  until restart (placement.md §1, "When enumeration happens").
- On Windows the CPU device shows its marketing name from the registry (`ProcessorNameString`)
  instead of `PROCESSOR_IDENTIFIER` (new Windows-only dependency `windows-registry`, MIT OR Apache-2.0).

### Fixed
- Listing devices, or using a `DynTensor` with an id that doesn't exist, while another thread
  computed on a Vulkan GPU could crash the process with SIGSEGV in the Vulkan loader (Debian 13 with
  stock Mesa, #26). The backend GPU tests no longer need `--test-threads=1` there. The first open of
  each Vulkan device still enumerates once inside CubeCL; open devices before computing on other
  threads (`Engine::new` does).
