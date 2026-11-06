# Spec: Device Discovery and Tiered Placement

**Status:** Draft. Device discovery in M0, placement in M4. **Crates:** `polypous-backend`, `polypous-runtime`
**Decisions:** [ADR-0003](../adr/0003-runtime-backend-selection.md), [ADR-0004](../adr/0004-tiered-placement.md), [ADR-0011](../adr/0011-use-burn-dispatch-for-backend-selection.md)

Goal: **if a model can run on this machine at all, it runs, with no configuration.** As many blocks
as fit go on the GPU, the next ones in CPU RAM, and the rest are streamed from disk.

## 1. Device discovery

At `Engine` creation Polypous enumerates devices for every compiled-in backend and **probes** each:
initialize, allocate a small buffer, run a tiny kernel, compare the result. A failing probe marks the
device unavailable with a reason, and never aborts.

**Crash safety** ([ADR-0013](../adr/0013-crash-safe-device-probing.md)): GPU drivers can crash or
hang the process natively, which no in-process mechanism can catch. Therefore:

- Enumeration and probing run in a separate **prober process** that reports one JSON line per device.
  If it crashes or hangs on a device (timeout per device, default 60 s), that device is marked
  `crashed during probe` / `probe timed out`, and the prober is restarted for the remaining devices.
- Results are kept in a **probe cache** keyed by device identity, backend, driver fingerprint (from
  OS sources that don't load GPU drivers) and Polypous version. A normal start starts no prober.
- The prober is the tiny `polypous-probe` executable on top of the shared library (M7), the app
  re-launched through `polypous::probe::run_prober_if_requested()`, or `polyp probe --json` (M0–M6).
- **Fallback** when no prober can be started: in-process probing in GPU priority order, with a
  crash journal so the app crashes at most once per device and driver version. Rust panics are
  caught with `catch_unwind` on both paths ([ADR-0011](../adr/0011-use-burn-dispatch-for-backend-selection.md)).
- Apps can run the probe themselves at any time through the API ([public-api.md §1](public-api.md#1-shape-of-the-api-illustrative)).
- The app process initializes only devices that passed the probe, and only when needed.

Enumeration builds on Burn's `Device::enumerate_physical`, which also reports device identity (PCI
address, adapter LUID) for CubeCL devices. Polypous never uses Burn's `Device::default()`: its
priority order and probing rules are the ones in this section.

Each device reports:

| Field | Source |
|---|---|
| `backend` | `Cuda` / `Metal` / `Vulkan` / `Cpu` |
| `name`, `vendor`, `driver` | Backend APIs |
| `kind` | `Discrete` / `Integrated` / `Cpu` |
| `dedicated_memory` | CUDA `cuMemGetInfo` total; Vulkan device-local heaps (excluding host-visible "shared" heaps on discrete GPUs); Metal `recommendedMaxWorkingSetSize` |
| `free_memory` | Queried at plan time (CUDA `cuMemGetInfo`, Vulkan `VK_EXT_memory_budget` if available, else an estimate) |
| `unified_memory` | true for integrated GPUs and Apple Silicon |
| `capabilities` | f16 support, subgroup size, max buffer size, etc. |
| `status` | `available`, or `unavailable` with a reason (`crashed during probe`, `probe timed out`, `probe failed: …`, `library missing`, `software rasterizer skipped`) and whether the probe ran isolated |

**Software rasterizers** (issue #21). Vulkan and Metal adapters of type `cpu` (lavapipe/llvmpipe,
SwiftShader, …) are **listed but not opened or probed by default**: they are never the primary GPU
(the CPU backend is faster), and opening one only risks a crash (lavapipe with Mesa 25.0.7 / LLVM 19
on Debian 13 segfaults the process on its first compute work). Their status is `unavailable` with
the reason `software rasterizer skipped (set software_rasterizers to probe it)` (code
`software_rasterizer_skipped`, `isolated: false`); the other fields are what the id alone tells:
name = the id (`vulkan:cpu:0`), kind `cpu`, `unified_memory` true, everything else null/empty. The
caller opts in with the option `software_rasterizers: true` (backend `EnumerateOptions`, facade
`EngineConfig` and `ProbeOptions`, and the prober's `ProbeRequest`); then they are described and probed like any
other device. Explicit `describe_device(id)` / `probe_device(id)` calls always open the device they
are given. Selection is unchanged either way. Polypous never reads an environment variable for this
(public-api.md §2 rule 9); the GPU tests use the test-only variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1`
to opt in.

**Implementation (M0, issue #4; `polypous_backend::device`).** The building blocks the prober runs:

- `list_devices()` / `list_backend_devices(backend)`: device ids from Burn's `Device::enumerate`
  per compiled-in backend, without opening devices. A missing runtime library means no devices.
  (`Device::enumerate_physical` skips devices without a card identity, such as the CPU and
  lavapipe, so it is not used directly; the same identity data is used for matching, see below.)
- `describe_device(id)`: opens the device (creates its CubeCL client, which loads the driver) and
  reads the fields above. An id that `list_backend_devices` doesn't list is not opened: the result
  is `unavailable: probe failed: could not open the device: no such <backend> device (devices
  found: …)`. `probe_device` rejects such ids without opening them too (`unavailable: probe failed:
  no such <backend> device (devices found: …)`).
  `enumerate_devices_with(&EnumerateOptions { software_rasterizers })` = list + describe, where the devices the options skip (`EnumerateOptions::skips(id)`) get
  `skipped_device(id)` instead and are not opened; `enumerate_devices()` uses the default options.
- `probe_device(id)`: `[[1,2],[3,4]] @ [[5,6],[7,8]] + 1` on the device, read back and compared
  exactly with `[20, 23, 44, 51]` (small integers, exact in f32/tf32/f16).
- None of them fails or panics. Errors and Rust panics (`catch_unwind`) become
  `unavailable: probe failed: …`; the in-process functions report `isolated: false`, the prober
  marks its results isolated. Calls are serialized by a process-wide lock.

**When enumeration happens (issue #26).** Each compiled-in backend is enumerated **at most once per
process**: on the first successful `list_backend_devices` call for it (directly, through
`list_devices`/`enumerate_devices*`, through `describe_device`/`probe_device`/`DynTensor` checking an
id, or through `Engine` discovery). Vulkan and Metal enumeration also reads wgpu's adapter info
(driver strings for `describe_device`) at that moment. Every later call reuses the cached list, and
an id that isn't in it fails against the list without enumerating again. Only successful
enumerations are cached; a failed one (a caught Burn panic) is retried on the next call. Devices
that appear later in the process lifetime (a newly plugged-in eGPU) are not seen until the process
restarts. The cache is process-wide hardware state, identical for every `Engine`, and holds no
configuration (public-api.md §2 rule 9).

Why: enumerating Vulkan devices creates a temporary wgpu instance, and `vkEnumeratePhysicalDevices`
on a fresh instance makes the Vulkan loader load all installed ICDs and unload the device-less ones
(`dlclose`). With stock Mesa on Debian 13 (Mesa 25.0.7, loader 1.4.309) that crashes the process
(SIGSEGV in `libvulkan.so.1`) when another thread computes on a Vulkan GPU at the same time.

**Remaining window.** The first time a Vulkan device is opened in a process (`describe_device`,
`probe_device`, or the first `DynTensor` on it), CubeCL (cubecl-wgpu 0.11.3, `try_create_setup`)
enumerates adapters once on a temporary wgpu instance before creating the device; Polypous can't
change that. Opening a device that was already opened touches no instance. So: **open every Vulkan
device you will use before Vulkan compute starts on other threads** (the first enumeration itself
counts as well). `Engine::new` does this for every device it probes, so after discovery only an
explicit first use of a device discovery didn't open (a skipped software rasterizer, a device that
failed to open) can still race with compute. CUDA and the CPU backend are not affected.

Tensors address devices by the same ids (issue #19): `DynTensor::from_data(&data, id)`,
`DynTensor::to_device(id)` and `DynTensor::device()`. An id is usable if its backend is compiled in
and `list_backend_devices` lists it (checked against the cached enumeration, see above); anything else
is `TensorError::DeviceUnavailable`, never a panic. `from_data(&data, id)?.device() == id` for every
id that `list_devices()` returns. Whether the device passed its probe is not checked there: that is
the `Engine`'s decision (#14).

`DeviceInfo` fields, serialized with serde exactly as named (JSON lines of the prober protocol):

| Field | Type / values | Notes |
|---|---|---|
| `id` | `{"backend":"cuda","index":0}`, `{"backend":"vulkan"\|"metal","adapter":"discrete_gpu"\|"integrated_gpu"\|"virtual_gpu"\|"cpu"\|"other","index":0}`, `{"backend":"cpu"}` | Addresses the device for `describe_device`/`probe_device`, stable within one boot. Index = CUDA ordinal, or index among adapters of that type (Burn's addressing) |
| `backend` | `cuda` / `metal` / `vulkan` / `cpu` | `BackendKind` |
| `name` | string | GPUs: the backend's adapter name. CPU: `model name` from `/proc/cpuinfo` on Linux; on Windows the registry value `HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0\ProcessorNameString` (the marketing name, for example `12th Gen Intel(R) Core(TM) i7-12800H`), else `PROCESSOR_IDENTIFIER`; elsewhere `CPU (<arch>)`. Surrounding whitespace is trimmed |
| `vendor`, `driver` | string or null | Vendor from the PCI vendor id; driver from wgpu's adapter info (`driver` + `driver_info`) for Vulkan/Metal. CUDA reports no driver string yet |
| `kind` | `discrete` / `integrated` / `cpu` / `other` | `cpu` includes Vulkan software rasterizers (lavapipe); `other` is a virtual or unclassified adapter |
| `dedicated_memory` | bytes or null | From CubeCL's device properties: CUDA `cuDeviceTotalMem`; Vulkan the **largest `DEVICE_LOCAL` heap** (Windows' shared GPU memory is a host heap, never counted); Metal `recommendedMaxWorkingSetSize`. For integrated GPUs, the RAM share the driver exposes. `null` for the CPU (RAM comes from OS queries, §3) |
| `free_memory` | bytes or null | CUDA `cuMemGetInfo` at discovery; null elsewhere until M4 |
| `unified_memory` | bool | true unless `kind` is `discrete` |
| `identity` | `{"pci_bus_id","luid","registry_id"}`, each nullable | PCI address `0000:01:00.0`; Windows LUID as 16 hex digits (changes on reboot, never a cache key); macOS registry id. Two devices are the same card if their PCI ids match, else their LUIDs, else their registry ids (CubeCL's rule, as in `enumerate_physical`). Empty for the CPU and software rasterizers |
| `capabilities` | `f16`, `bf16`, `subgroup_size_min`, `subgroup_size_max`, `max_buffer_size`, `max_shared_memory`, `compute_units` | Missing values are null/false |
| `status` | `{"state":"not_probed"}`, `{"state":"available","isolated":bool}`, `{"state":"unavailable","reason":{"code":"crashed_during_probe"\|"probe_timed_out"\|"probe_failed","message"\|"library_missing","library"\|"software_rasterizer_skipped"},"isolated":bool}` | `not_probed` after `describe_device`; the probe sets the rest. `software_rasterizer_skipped` (always `isolated: false`): listed, not opened (see "Software rasterizers" above) |

**`Engine::new` (M0, issues #6 and #14; `polypous::Engine`).** The engine gets its devices from
`EngineConfig::probe_report` if the app passed one; otherwise it runs the probe below (cache →
prober → fallback). Devices are reported in **probe order** (CUDA, Metal, Vulkan discrete, Vulkan
integrated, other Vulkan adapters, Vulkan/Metal software rasterizers, CPU; the backend's order
within each group; `polypous_backend::device::probe_order`), and the facade marks the result of
`select_primary_gpu` as `primary` (public-api.md §1.2). Software rasterizers are skipped unless
`EngineConfig::software_rasterizers` is set (same rule as the backend's `EnumerateOptions`).

**The probe (issue #14; `polypous_backend::probe`, facade `polypous::probe`).** One probe runs these
steps; `Engine::new`, `Engine::refresh_devices` and `probe::probe_devices` share them:

1. **Crash journal recovery.** Leftover journal entries in the cache directory (see "Fallback")
   become cache entries `crashed during probe`, `isolated: false`.
2. **Complete cache hit** (not with `force`): the OS fingerprint is known, the cache was written by a
   probe of all devices that ran to the end, its GPU key set equals the current one, every entry's
   fingerprint still matches, and no entry is a skipped software rasterizer while the caller now opts
   in. Then every device comes from the cache and **no prober is started** (log:
   `probe cache hit: every device from the cache, no prober started`; report: `prober_starts: 0`).
3. Otherwise **skip rules** are built from the valid cache entries (by stable key, else by id; with
   an unknown fingerprint, by id + enumeration key + enumeration fingerprint), and a **prober
   process** enumerates every device and probes those that match no rule (a match is reported
   `skipped` and not opened). With an unknown fingerprint this is the enumerate-only comparison of
   ADR-0013 §2: if every device matches, the prober opens nothing.
4. **Fallback:** without a prober (none configured, none could start, or `isolation = in_process`),
   the same session runs in process with the crash journal.

The report (`ProbeReport`) lists the devices with status, `isolated` (every probe result in the
report, cached or new, came from a prober process), `cache_hits`, `prober_starts`,
`fingerprint_known` and `warnings`. `polyp devices` prints a summary line under the table
(`Probing: isolated … / NOT isolated …`, prober processes started, cache hits) and the warnings;
`polyp devices --json --report` prints the report.

**A compiled-in GPU backend without devices** is not an error, but the report says why, so a
`cuda` build on a machine without usable CUDA still explains itself. When a report covers all
devices (no `devices` subset), every compiled-in GPU backend (CUDA, Vulkan, Metal) with no device in
it, and whose enumeration did not fail (that already has its own warning), adds the warning
`<backend>: compiled in, but no device found: <reason>`. The reason is the first that applies:

- a variable that selects devices or drivers is set: CUDA `CUDA_VISIBLE_DEVICES`; Vulkan
  `VK_DRIVER_FILES`, `VK_ICD_FILENAMES` (for example `CUDA_VISIBLE_DEVICES is set to "-1"`);
- the driver library is not installed, checked by looking for the file without loading it: CUDA
  `nvcuda.dll` in `%SystemRoot%\System32` (Windows), `libcuda.so.1` (Linux); Vulkan `vulkan-1.dll`
  (Windows), `libvulkan.so.1` (Linux). On Linux the directories are those of `LD_LIBRARY_PATH`,
  `/usr/lib/<arch>-linux-gnu`, `/lib/<arch>-linux-gnu`, `/usr/lib64`, `/usr/lib`, `/usr/local/lib`
  and `/usr/lib/wsl/lib`;
- otherwise: no device was reported (`no CUDA device was reported`, `no Vulkan device was reported`,
  `no Metal device was reported`).

**Prober protocol** (`polypous_backend::probe::protocol`, protocol version 1). The app starts
`<prober> --polypous-prober --request <json>` with stdin, stdout and stderr piped. The request
(`ProbeRequest`) carries `protocol`, `software_rasterizers`, `devices` (subset), `skip` (skip rules:
`{"by":"key","key":…}`, `{"by":"id","id":…}`, `{"by":"enumeration","id":…,"key":…,"enum_fingerprint":…}`),
`skip_backends` and `exclude` (devices an earlier prober of the same probe already reported). The
prober writes one JSON object per line to stdout, `type` naming it:

| Line | Meaning |
|---|---|
| `{"type":"header","protocol":1,"polypous_version":"…"}` | First line |
| `{"type":"enumerating","backend":"vulkan"}` | Enumeration of a backend starts |
| `{"type":"enumerated","device":{"id":…,"key":…,"enum_fingerprint":…}}` | A device found, not opened. `key`: `pci:<address>` where the graphics API reports it at enumeration (Vulkan, Metal); else the key of the one GPU in the OS fingerprint with the same PCI vendor and device id (Intel iGPUs through Vulkan on Windows; see "Driver fingerprint"); else null. `enum_fingerprint`: adapter name, PCI vendor/device ids and driver strings from wgpu's adapter info, else null |
| `{"type":"enumeration_failed","backend":…,"message":…}` | Enumeration panicked (caught) |
| `{"type":"skipped","id":…,"rule":N}` | Matched skip rule N; not opened |
| `{"type":"probing","id":…}` | Written and flushed **before** the device is opened |
| `{"type":"probed","device":{DeviceInfo}}` | Description and probe result |
| `{"type":"done"}` | Last line; the prober exits |

After all backends are enumerated, devices are probed in probe order. Logs go to stderr. The app
treats the output as **untrusted**: at most 64 KiB per line, 1024 lines and 128 devices per process;
a line that is not valid JSON of the protocol, or comes out of order (for example `probed` without
`probing`, a device that was not enumerated, a second header, anything after `done`), is a protocol
error. The app decides `isolated` itself (`true` for prober results, `false` for skipped software
rasterizers) and takes `backend` from the id.

**Supervision.** stdout is read on a thread (bounded lines), stderr is drained on another thread
into the log at `debug` (target `polypous::prober`; the last lines are kept for warnings), and stdin
stays open: the prober's watchdog thread exits the prober (code 3) when stdin reaches end of file,
so a prober never outlives its app. If no line arrives within the **timeout per device** (default
60 s, `timeout_per_device_ms`; it also covers start-up and enumeration), or on a protocol error or
cancellation, the prober is killed; every path waits for the child, so no zombie is left. Then:

- the device in progress (`probing` without `probed`) is marked `crashed during probe` (the
  prober exited, for example SIGSEGV), `probe timed out` (killed after the timeout), or
  `probe failed: the device prober sent invalid output: …` (protocol error); all `isolated: true`.
  A prober that dies while enumerating a backend makes that backend's devices unknown (a warning)
  and it is skipped (`skip_backends`) from then on;
- the prober is restarted with `exclude` = the devices already reported, so results survive, until
  every wanted device has a result. A restart happens only after something was blamed or reported,
  so each device or backend causes at most one restart. Devices no prober got to are
  `probe failed: the device prober ended before probing it`.

A prober that cannot be started, or ends, hangs or misbehaves **before its header**, counts as
"no prober": `isolation = auto` falls back to in-process probing (with a warning in the report),
`out_of_process` is a `BackendUnavailable` error.

**Locating the prober** (ADR-0013 §3): `EngineConfig::prober_path` / `ProbeOptions::prober_path` if
set; otherwise the current executable if it called `polypous::probe::run_prober_if_requested()`
(the first line of `main`; `polyp` does); otherwise none (in-process fallback). Every prober
understands `--polypous-prober --request <json>`: `run_prober_if_requested` runs the prober and exits
when it sees that, so `polyp`, an app that embeds the library, and the tiny `polypous-probe` of M7
are started the same way. `polyp probe --json [--software-rasterizers] [--request JSON]` runs the
same prober by hand (unsupervised: no stdin watchdog). The hook is a process-wide flag that
describes the executable, the same for every `Engine` (public-api.md §2 rule 9).

**Probe cache.** One JSON file per Polypous version and set of compiled-in backends,
`probe-cache-v1-<backends>.json` (for example `probe-cache-v1-cuda-vulkan-cpu.json`), in
`EngineConfig::probe_cache_dir` / `ProbeOptions::probe_cache_dir`, else the platform cache
directory (`$XDG_CACHE_HOME/polypous` or `~/.cache/polypous`; `%LOCALAPPDATA%\polypous\cache`;
`~/Library/Caches/polypous`). Each entry is a `DeviceInfo` with its stable key, enumeration
fingerprint and the OS fingerprint it was probed under. The file is read with a 4 MiB limit, any
parse error or other version means "no cache", it is written atomically (temporary file + rename),
and read/write failures are only logged. A `software_rasterizer_skipped` result never replaces a
probed result of the same device. A probe of all devices that ran to the end marks the cache
complete and drops devices that are gone; in-process probing saves after every device but marks
the cache incomplete until it finishes.

**Driver fingerprint** (`polypous_backend::probe::fingerprint`, one function per OS: a stable key and
fingerprint per GPU plus a "shared" fingerprint, or unknown). **Linux:** GPUs are the PCI display
controllers (class `0x03xxxx`) in `/sys/bus/pci/devices`, keyed `pci:<address>`. A GPU's fingerprint
is its PCI vendor/device ids, kernel driver, `/sys/module/<driver>/version` (else the kernel release
`/proc/sys/kernel/osrelease`), the NVIDIA version line of `/proc/driver/nvidia/version` for the
`nvidia` driver, and the Vulkan driver set: `VK_DRIVER_FILES`, `VK_ICD_FILENAMES`,
`VK_ADD_DRIVER_FILES`, and every ICD manifest in `/etc/vulkan/icd.d`, `/usr/share/vulkan/icd.d`,
`/usr/local/share/vulkan/icd.d`, `~/.local/share/vulkan/icd.d` with size and modification time, and
the same for the library it names. The shared fingerprint (CPU, software rasterizers, devices without
a key) has the kernel release, NVIDIA version, Vulkan driver set and the GPU key set. A device's
fingerprint is its GPU's by enumeration key; else by the PCI address its description reported, plus
the GPU key set (it is matched by id, which depends on the device set); else the shared one. Nothing
here opens a device node or loads a library. **Windows:** GPUs are the present display
adapters among the PCI device instances under `HKLM\SYSTEM\CurrentControlSet\Enum\PCI` (class GUID
`{4d36e968-e325-11ce-bfc1-08002be10318}`; present means the instance has its volatile `Control`
subkey, which Windows creates while the device is started). The key is `pci:0000:<bus>:<device>.<function>`
from the `(bus,device,function)` triple in `LocationInformation` (Windows doesn't store the PCI
segment there; 0 is assumed). A GPU's fingerprint is its instance id (PCI vendor, device, subsystem,
revision) and, from its driver key (`HKLM\SYSTEM\CurrentControlSet\Control\Class\<Driver>`),
`DriverVersion`, `DriverDate`, `ProviderName` and `MatchingDeviceId`. The shared fingerprint has the
Windows build (`CurrentBuild` and `UBR` under `HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion`),
size and modification time of the Vulkan loader `%SystemRoot%\System32\vulkan-1.dll`, the loader
variables `VK_DRIVER_FILES`, `VK_ICD_FILENAMES`, `VK_ADD_DRIVER_FILES`, every GPU's fingerprint and
the GPU key set. If the registry can't be read, the fingerprint is unknown. **Adapters without a PCI
address at enumeration** (Intel iGPUs through Vulkan on Windows report only a LUID): the enumeration
key is that of the one present GPU with the same PCI vendor and device id; if none or several match,
there is no key. **macOS** reports unknown for now, which takes the enumerate-only path. LUIDs are
never a key. **On every OS** the variables that change which CUDA devices exist or how they are
numbered, `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER`, are part of every GPU fingerprint and of
the shared one: setting or changing them invalidates the cached devices, so a hidden CUDA device is
never served from the cache. **On Linux and Windows** the CUDA driver library is part of every GPU
fingerprint and of the shared one too, as `cuda_library=<path>:<size>:<mtime>`. On Linux that is the
first `libcuda.so.1` found in the directories listed under "A compiled-in GPU backend without
devices" (first match wins); on Windows, `%SystemRoot%\System32\nvcuda.dll`. Symbolic links are
followed. The value is `missing` if the library isn't found, and `unknown` if the directories can't
be determined. The file is only stat'ed, never loaded. So installing, removing or updating the CUDA
driver library invalidates the cached devices, even when the NVIDIA version line and the Vulkan
ICDs stay the same. It counts for every GPU, not only CUDA devices, because NVIDIA's Vulkan driver
needs it too: with NVIDIA 620.70.06 on Linux, opening the RTX through Vulkan fails while
`libcuda.so.1` is missing.

A device recorded as `crashed during probe` or `probe timed out` stays unavailable, without being
opened, while its fingerprint is unchanged. `force` re-probes it **out of process only**: in-process
probing never re-opens such a device under the same driver, even with `force` (it would crash the
app again).

**Fallback: in-process probing with crash journal.** Before each in-process probe, the device's
cache entry with the status `crashed during probe`, `isolated: false` is written and `fsync`ed to
`probe-journal-<pid>-<nanos>-<n>.json` in the cache directory; after the probe the file is emptied,
and it is removed when the probe ends. The file is held with an exclusive file lock (`File::lock`)
while in use, so recovery (step 1) only reads journals no running process holds. The fallback logs a
warning, results say `isolated: false`, and the report and `polyp devices` say probing was not
isolated. In-process probing has no timeout (a hung driver hangs the app; that is what the prober is
for). Without a cache directory there is no journal (a warning says so).

**Testing the prober** (CI, Linux and Windows). The cargo feature `test-fault-injection` of
`polypous-backend` (forwarded as `polypous/test-fault-injection`) compiles a fault-injection module.
Only `polypous-cli`'s dev-dependency enables it, so `cargo build`, `cargo run` and `cargo install` of
`polyp`, and every app, don't contain it; without the feature the variables below are never read.
`POLYPOUS_TEST_PROBER_FAULTS` (prober and in-process path) replaces the real devices by fakes
(`fake_devices=cuda:0,…`, `fake_driver=…`) and injects `crash=<id>` (abort), `hang=<id>`,
`fail=<id>`, `crash_enumeration=<backend>`, `garbage=<id>` / `oversized=<id>` / `spam=<id>`
(malformed, too long, or out-of-order output after `probing`), `exit_before_header` and
`pid_file=<path>`. `POLYPOUS_TEST_FINGERPRINT` (app process) replaces the OS fingerprint (`unknown`
or any text). `crates/polypous-cli/tests/prober.rs` uses them; `crates/polypous/tests/embedded_prober.rs`
is an app that probes through itself re-launched.

**Primary GPU selection** (only one GPU is used in v1):
1. CUDA device (discrete NVIDIA)
2. Metal device
3. Vulkan discrete GPU
4. Vulkan integrated GPU
5. none, so CPU only

Only devices whose probe passed count. Vulkan software rasterizers and virtual/unclassified
adapters are never the primary GPU (the CPU backend is faster than lavapipe). If several devices
share a rank, choose the one with the most dedicated memory (unknown counts as 0), then the
earlier one in the list. A physical GPU
reachable via both CUDA and Vulkan counts once (matched by device identity), and CUDA wins; if
only its CUDA probe failed, its Vulkan device is used. On the reference machine this picks the
RTX 3060 via CUDA and ignores the Intel iGPU.

Apps can override: force a backend or device, disable GPU, set memory budgets.

**Dynamic libraries:** GPU runtime libraries (CUDA driver/runtime, Vulkan loader) are loaded at
runtime. If they are missing, that backend is unavailable and Polypous moves to the next option. It must never crash.

## 2. Placement blocks

The runtime splits an instantiated graph into **blocks**, the unit of placement:

- each `repeat` iteration is one block
- nodes before the first repeat form the **prologue** block (for example embeddings); nodes after the
  last repeat form the **epilogue** block (final norm, output head)
- a graph without repeats is a single block (small models, builder graphs)

For each block the planner computes:

| Quantity | How |
|---|---|
| `weight_bytes` | Sum of parameter storage sizes **in their stored format** (quantized size for GGUF) |
| `state_bytes` | Size of state (KV cache) owned by the block at the session's `max_context` and `batch` |
| `workspace_bytes` | Peak live activation size inside the block, from shape inference with concrete dims plus liveness analysis, + kernel scratch |
| `compute_class` | `lookup` (embeddings), `dense` (matmul-heavy) or `light` |

## 3. Budgets

```
gpu_budget  = free_vram  − reserve_gpu − max(workspace over GPU blocks) − transfer buffers
cpu_budget  = available_ram − reserve_cpu − max(workspace over CPU blocks) − streaming staging
reserve_gpu = max(512 MiB, 10% of dedicated)
reserve_cpu = max(2 GiB, 15% of total RAM)
```

- **Discrete GPU:** VRAM and RAM are separate budgets. "Shared GPU memory" on Windows is ignored.
- **Unified memory:** one budget, `available_ram − reserve_cpu`. GPU blocks are filled first from it,
  then CPU blocks from what remains.
- **Disk:** unlimited, but needs a **streaming staging area** in RAM of 2× the largest disk block
  (double buffering).
- If even the staging area plus the minimal working set does not fit in RAM, the model is
  **unrunnable**, and the error reports the shortfall.

## 4. Assignment algorithm (v1)

1. Fill the GPU in **priority order** while blocks fit the remaining budget:
   1. epilogue (output head is dense and used every step)
   2. repeat blocks in execution order (0, 1, 2, …)
   3. prologue, only if it is not purely `lookup` (embedding lookups are cheap on CPU and often large)
2. Assign the remaining blocks to the CPU in the same order while they fit.
3. Assign everything left to **Disk**.
4. State follows its block: a GPU block's KV cache lives in VRAM, a CPU/Disk block's in RAM. State
   is counted in step 1 and 2, so a long context reduces how many layers fit on the GPU.

This keeps segments contiguous, so a forward pass crosses a tier boundary only a few times
(for example CPU → GPU → CPU → Disk → GPU). Activations crossing a boundary are small (one hidden
state), so transfer cost is negligible compared to weights.

The algorithm is deterministic. The same inputs always produce the same plan, which keeps tests stable.

## 5. Execution by tier

| Tier | Weights | Compute |
|---|---|---|
| **Gpu** | Uploaded to VRAM at load | GPU backend |
| **Cpu** | Memory-mapped and pre-touched at load (counted against the RAM budget) | CPU backend, quantized CPU kernels operate directly on mapped bytes where possible |
| **Disk** | Memory-mapped, **not** pre-touched; may be evicted by the OS | CPU backend (v1). A background thread prefetches the next K disk blocks (`madvise(WILLNEED)` / `PrefetchVirtualMemory`, or explicit reads into the staging area) while the current block runs |

Weights stay in their stored format in every tier. Whether disk blocks should instead be uploaded to
the GPU for compute is open question Q2 in the [project brief](../project-brief.md#10-open-questions).

## 6. Estimate (before loading)

`engine.plan(source, options)` returns a `PlacementPlan` without loading weights:

- per tier: block list, weight bytes, state bytes, workspace bytes
- `peak_vram`, `peak_ram`, `disk_streamed_bytes_per_forward`
- **speed estimate:** decode is memory-bound, so
  `t_token ≈ Σ_tier bytes_read_per_token / bandwidth_tier` with bandwidth from a built-in table
  per device class, refined by a quick measurement cached per machine (GPU memcpy, RAM memcpy, disk
  sequential read)
- warnings, for example "17 of 48 layers will stream from disk; expect about 0.4 tokens/s"

Acceptance: estimated peak memory within ±25% of measured, on the reference machine.

## 7. Failure handling and fallback

| Situation | Behavior |
|---|---|
| Backend fails to initialize | Mark unavailable, use next GPU option or CPU only; warning event |
| GPU allocation fails during load | Free that model's GPU allocations, reduce `gpu_budget` by 20%, re-plan, retry (max 3 times), then CPU+Disk |
| Kernel launch or device-lost error during inference | Return a `BackendError`. The model is marked degraded and the app may reload (later: automatic CPU fallback for that block) |
| RAM too small for staging + working set | `OutOfMemory` error with the required/available numbers |
| Context exceeds the session's `max_context` | Pipeline policy (error, or truncate oldest) |

KV cache and workspace are **allocated up front** when a session is created, so out-of-memory
cannot happen in the middle of a generation.

Every fallback emits an event and a `tracing` warning with the reason, so apps can tell users why
something is slower than expected.

## 8. Configuration (all optional)

```jsonc
{
  "device": "auto",              // or "cpu", "cuda:0", "vulkan:1", "metal:0"
  "gpu_memory_limit": null,      // bytes; overrides detected budget
  "ram_memory_limit": null,
  "allow_disk_streaming": true,
  "max_context": 4096,
  "prefetch_blocks": 2
}
```

## 9. Out of scope for v1

Splitting a model across several GPUs · pipeline/tensor parallelism · moving blocks between tiers
while running · paging the KV cache to disk.
