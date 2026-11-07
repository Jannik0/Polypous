# Changelog

All notable changes to Polypous are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

### Added
- `polypous-formats`: safetensors reader and writer (`docs/specs/formats.md` §1, #37). Memory-mapped
  reader with a copy fallback that validates the whole header before exposing tensors (100 MB header
  limit, contiguous offsets covering the data buffer, checked sizes, no duplicate keys, rank ≤ 64),
  zero-copy tensor views, all safetensors dtypes recognized and F32/F16/BF16/I64/I32/BOOL decodable;
  sharded checkpoints (`*.safetensors.index.json`, 64 MiB limit, plain shard file names, every
  tensor in exactly one shard); a writer whose output is ordered by element size, then name, so every
  tensor is aligned to its element size,
  checked against Python `safetensors`.
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
  sysfs, `/proc/driver/nvidia/version` and the Vulkan ICD manifests; Windows driver fingerprints and
  stable device keys from the registry (display adapters' PCI location, hardware id and driver
  version/date/provider, Windows build, Vulkan loader), so the cache also survives reboots there and
  an Intel iGPU, which Vulkan reports without a PCI address, is matched by its PCI ids (#28); macOS
  compares the enumeration instead.
- Probe API: `polypous::probe::{probe_devices, run_prober_if_requested, run_prober}`,
  `Engine::refresh_devices`, `Engine::probe_report`, `ProbeOptions`, `ProbeEvent`, `ProbeReport`,
  `Isolation`, `CancellationToken`, and `EngineConfig` fields `probe_report`, `prober_path`,
  `probe_cache_dir`, `probe_isolation`, `probe_timeout_per_device_ms`.
- `polyp probe --json` runs the prober and prints its JSON-lines protocol. `polyp devices` gets
  `--software-rasterizers`, `--isolation`, `--probe-cache-dir`, `--prober-path`,
  `--timeout-per-device-ms` and `--json --report`, and prints how devices were probed under the table.
- `DeviceInfo::physical_gpu` (JSON `physical_gpu`): devices with the same number are one physical
  GPU reached through different backends (the RTX 3060 through CUDA and Vulkan); `null` for the CPU
  and software rasterizers. `polyp devices` shows it in a new `GPU` column.
- The probe report says why a compiled-in GPU backend has no device, for example
  `cuda: compiled in, but no device found: CUDA_VISIBLE_DEVICES is set to "-1"` or
  `… the driver library nvcuda.dll is not installed`; `polyp devices` prints it as a warning.
- Reference fixture tooling (#36): `tools/reference/gen_ops.py` generates op fixtures from
  PyTorch (pinned CPU-only versions in `tools/reference/requirements.txt`, Python 3.14) into
  `fixtures/small/ops/<op>/<case>.{safetensors,json}`, with `--check` to compare against the
  committed files. First fixtures for `core.add`, `core.matmul`, `core.cast` (all 36 pairs of
  `f32`/`f16`/`bf16`/`i64`/`i32`/`bool`) and `core.reshape`. Sidecar schema and layout in
  testing-strategy.md §3. `tools/reference/fetch_fixtures.py` downloads `fixtures/manifest.json`
  entries into `fixtures/cache/` with a size limit and SHA-256 check. CI checks the fixture size
  budget and sidecars (`tools/ci/check_fixtures.py`) in the "Dependency boundary" job.

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
- When a device could not be opened because a worker thread panicked (CubeCL's device runner, for
  example `Unable to acquire wgpu device: RequestDevice { … "Initialization of an object has
  failed" }`), the `probe failed` reason only said `… RecvError`. The prober process now records
  panics on every thread and appends the ones from other threads to the reason (`… (worker thread
  panicked: …)`, bounded). In-process probing is unchanged: it installs no panic hook.
- With a warm probe cache, removing (or installing, or updating) the CUDA driver library was
  ignored: `cuda:0` was still reported available and chosen as the primary GPU. The library
  (`libcuda.so.1` on Linux, `nvcuda.dll` on Windows: path, size and modification time, or
  `missing`) is now part of every driver fingerprint, so a change re-probes. Existing probe caches
  are re-probed once after the update.
- With a warm probe cache, setting `CUDA_VISIBLE_DEVICES` (or `CUDA_DEVICE_ORDER`) was ignored: a
  hidden CUDA device was still reported available and chosen as the primary GPU. Both variables are
  now part of the driver fingerprint, so changing them re-probes.
- Listing devices, or using a `DynTensor` with an id that doesn't exist, while another thread
  computed on a Vulkan GPU could crash the process with SIGSEGV in the Vulkan loader (Debian 13 with
  stock Mesa, #26). The backend GPU tests no longer need `--test-threads=1` there. The first open of
  each Vulkan device still enumerates once inside CubeCL; open devices before computing on other
  threads (`Engine::new` does).
