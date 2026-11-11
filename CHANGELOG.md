# Changelog

All notable changes to Polypous are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break the API).

## [Unreleased]

### Added
- `polypous-ir`: the M1 op catalog, dtype inference and symbolic shape inference
  (`docs/specs/graph-ir.md` §3, §4.1 checks 4, 6–8, #39). `catalog::OPS` has one entry per row of
  the §3.2/§3.3 tables (inputs with dtype class and rank, params, attributes with type, default and
  range, output ports with their dtype rule, shape rule, differentiable flag); a test parses the
  spec tables and another checks every op fixture's output dtypes and shapes. Validation now checks
  op inputs, params and attributes (types, ranges, axes, `float` only where allowed, dim attributes
  and their symbols, the `concat`/`split` and `constant` limits), infers every value's dtype
  (`float` resolved to the compute dtype, strict dtypes) and shape as normal forms (broadcasting,
  `reshape` with `-1`, conv/pool sizes, clamped `slice` bounds), infers each `repeat` body once and
  checks carries and state updates (`repeat_carry_mismatch`, `state_update_mismatch`).
  `Graph::analyze` returns what validation proved for the runtime (resolved dtypes, inferred shapes,
  normalized attributes, expanded state entries), and `Analysis::bind` binds the symbols for one
  call (§2.3.4 steps 1–5: input and state checks, `min`/`max`, every inferred dim and dim attribute
  evaluated with checked arithmetic), with a `BindError` that names the input, symbol or node.
- graph-ir.md: clarifications from the inference work: body params must have the same shape in
  every iteration; which code a range error gets (`invalid_attr`, `shape_mismatch`,
  `nonpositive_dim`, `unbound_dim`); a NaN `clamp` bound is allowed; `squeeze` `dim` is non-empty;
  `slice` bounds `≤ 1` aren't clamped; `pad`'s default value is the dtype's zero; variadic input
  errors; the binding order of expanded state.
- `polypous-backend`: the first IR ops on `DynTensor` (`ops` module, graph-ir.md §3.2, #40):
  elementwise `add sub mul div pow maximum minimum neg abs exp log sqrt rsqrt sin cos tanh clamp`,
  `where_` (`core.where`), comparisons `equal not_equal less less_equal greater greater_equal`, logic
  `logical_and logical_or logical_not`, and `cast`. Broadcasting across different ranks, strict
  dtypes (typed errors, no promotion), NaN propagation in `maximum`/`minimum`/`clamp`, NaN in
  comparisons and float → bool casts, IEEE special cases of `pow`, wrapping integer arithmetic, and
  f16/bf16 transcendentals computed in f32 on every backend. New `Scalar` argument type,
  `TensorError::InvalidArgument` and `DynTensor::device_supports`. `add` now broadcasts operands
  of different ranks.
- Op reference tests for these ops against PyTorch fixtures on the CPU and on every GPU device
  (`tests/reference_*.rs`, shared harness in `tests/reference/common.rs`), with fixture cases for
  every op in `fixtures/small/ops/` (`tools/reference/gen_ops.py`).
- `polypous-backend`: shape, indexing and creation ops on `DynTensor` (graph-ir.md §3.2, #41):
  `reshape` (moved into `ops`) `flatten permute transpose squeeze unsqueeze expand concat split
  slice pad`, `gather index_select`, and `constant arange zeros ones full full_like` (associated
  functions with a `DeviceId`). Plain, normalized arguments; every bad argument is a typed error
  (no panics, sizes checked before allocating, zero-size results refused, §4.3 list limits).
  Out-of-range `gather`/`index_select` indices are checked before the op runs and give the new
  `TensorError::IndexOutOfRange` (Burn 0.22.1 panics on the CPU and reads stray memory, or crashes,
  on Vulkan). `pad` and every op work for every dtype, including `bool`. Fill values follow
  §3.1.7 and round through f32 to f16/bf16 like PyTorch.
- Reference fixtures for these ops (190 cases: every op × every allowed dtype, rank 1 and 6,
  size-1 dims, negative axes and `slice` bounds, steps > 1, `i32` and `i64` indices, and inputs fed
  as permuted, transposed, expanded and stepped-slice views), `tests/reference_shape.rs`,
  `reference_indexing.rs` (with out-of-range checks on every device) and `reference_creation.rs`.
  The harness gains integer and list attribute getters, a runner that passes the device to
  creation ops, and `feed_layout` for non-contiguous inputs; existing tests are unchanged.
- `polypous-backend`: reductions, `matmul`, softmax, `cumsum` and `top_k` on `DynTensor`
  (graph-ir.md §3.2, #42): `sum mean max min` over any set of dims (`None`: all, `[1]` for a full
  reduction), `argmax argmin` with an `i64` or `i32` result, `cumsum`, `softmax log_softmax`,
  `top_k` (returns `(values, indices)`) and `matmul` (moved into `ops`; batch dims now broadcast
  across different ranks). NaN and ties follow §3.1.8 on every backend (explicit masks; `top_k`
  sorts stably on the host, one device sync per call), integer `sum`/`cumsum` wrap instead of
  panicking on the CPU, softmax subtracts the row maximum, and f16/bf16 `sum mean cumsum softmax
  log_softmax` accumulate in f32.
  Known limitation: f32 `matmul` on CUDA GPUs with TF32 (Ampere or newer) misses the spec's
  f32 accuracy, because Burn 0.22.1 uses TF32 tensor cores; it waits for an upstream fix
  (docs/status.md §4).
- Reference fixtures for these ops (184 new cases, 192 with the 8 earlier `matmul` cases): every
  op × every allowed dtype, one/several/all dims with and without `keepdim`, `matmul` from 2-D to
  rank 6 with broadcast batch dims of different ranks, `k = 1` and `k = 256`, ties and NaN for
  `argmax`/`argmin`/`top_k`, wrap-around for `sum`/`cumsum`, large values for softmax and f16 sums,
  and non-contiguous inputs; `tests/reference_reduce.rs`.
- `polypous-backend`: the M1 `nn` ops on `DynTensor` (graph-ir.md §3.3, #43): `linear` (on
  `matmul`), `embedding`, `conv1d` and `conv2d` (stride, padding, dilation, groups), `max_pool2d`,
  `avg_pool2d` (`count_include_pad`), `adaptive_avg_pool2d`, `dropout` (the identity in inference),
  `batch_norm` (inference), `layer_norm`, `group_norm`, `rms_norm`, and the activations `relu
  leaky_relu gelu gelu_tanh silu sigmoid softplus mish`. Params are plain `DynTensor` arguments in
  PyTorch's layouts (optional ones `Option`), so PyTorch weights load unchanged; every argument is
  checked (typed errors, output sizes with checked arithmetic). Out-of-range `embedding` ids are a
  `TensorError::IndexOutOfRange`, checked before the lookup. NaN propagates through activations and
  `max_pool2d`, `max_pool2d` pads with `-inf` on every backend (CubeCL pads with `-f32::MAX` and
  dropped NaN on lavapipe), norms use a two-pass variance, and f16/bf16 compute in f32. Known
  limitation: like `matmul`, f32 `linear` (and possibly conv) on CUDA GPUs with TF32 may miss the
  spec's accuracy (docs/status.md §4).
- Reference fixtures for the `nn` ops (265 cases, the first with params): every op × f32/f16/bf16
  (`embedding` ids i64 and i32), stride/padding/dilation/groups incl. depthwise and non-square
  kernels, `count_include_pad`, overlapping adaptive cells, all-padding pool windows, NaN/±inf,
  large and tail inputs, large-mean norms, absent optional params and non-contiguous inputs;
  `tests/reference_nn.rs` (with out-of-range ids on every device). Sidecars of ops with params
  gain a `params` key after `inputs` (testing-strategy.md §3.2); existing sidecars are unchanged.
- graph-ir.md §3.1.8: f16/bf16 `sum`, `mean`, `cumsum`, `softmax` and `log_softmax` compute in f32;
  §3.2 `top_k` note: fixtures use a stable sort, since `torch.topk` leaves the order of ties open.
- graph-ir.md §3.1.8: subnormal float values are backend-defined (a device may flush them to
  zero). Fixtures contain none; `tools/ci/check_fixtures.py` rejects them.
- `polypous-ir`: graph document model, canonical JSON and structural validation
  (`docs/specs/graph-ir.md` §2, §4, §6, #38). Typed `Graph` for every M1 document field (dims,
  inputs, outputs, params with `init`, state, nodes, `repeat` bodies, `meta`); loading untrusted JSON
  under the §4.3 limits (64 MiB, nesting depth 128, 4,000,000 JSON values, 64 KiB strings, duplicate
  keys rejected) with
  every error collected as a stable code, message, node id and JSON pointer; canonical save (schema
  key order, `meta` and `"float"` kept as written, load → save → load identity); structural checks
  (versions, names and namespaces, references including `node:port` and `repeat:carry`, cycles, op
  names of the M1 opsets, dim declarations and binding, repeat limits computed without expanding,
  carry keys and state update targets); dim expressions with their normal form, canonical string
  and equality (§2.3).
- `polypous-formats`: safetensors reader and writer (`docs/specs/formats.md` §1, #37). Memory-mapped
  reader with a copy fallback that validates the whole header before exposing tensors (100 MB header
  limit, contiguous offsets covering the data buffer, checked sizes, no duplicate keys, rank ≤ 64),
  zero-copy tensor views, all safetensors dtypes recognized and F32/F16/BF16/I64/I32/BOOL decodable;
  sharded checkpoints (`*.safetensors.index.json`, 64 MiB limit, plain shard file names, every
  tensor in exactly one shard); a writer whose output is ordered by element size, then name, so every
  tensor is aligned to its element size, checked against Python `safetensors`.
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
- Op fixtures name their tensors after the op's inputs (`a`, `b`, `x`, `cond`; outputs `out`),
  and `core.add/empty_dim_f32` is gone (zero-size dims are not allowed in M1, graph-ir.md §2.3).
- `polypous-backend` no longer depends on `polypous-ir`.
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
