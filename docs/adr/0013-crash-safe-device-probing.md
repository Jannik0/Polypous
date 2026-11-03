# ADR-0013: Crash-safe device probing in a separate prober process

- **Status:** Accepted (2026-11-03)
- **Date:** 2026-11-03
- **Deciders:** project owner (+ authoring agent)
- **Supersedes / Superseded by:** extends [placement.md §1](../specs/placement.md#1-device-discovery) and [ADR-0011](0011-use-burn-dispatch-for-backend-selection.md); does not replace them

## Context

Polypous probes devices (initialize, allocate, run a tiny kernel, compare the result) and promises
that a broken device makes only that device unavailable, never the app
([placement.md §1](../specs/placement.md#1-device-discovery)). Rust panics from Burn/CubeCL can be
caught with `catch_unwind`. Native crashes cannot.

The reference machine showed that native crashes happen
([ADR-0007 follow-up notes](0007-burn-version-pinning.md#follow-up-notes)): with the Intel Iris Xe
driver 30.0.101.1298 (January 2022), a 2×2 matmul through Vulkan on the iGPU **crashed the whole
process** with an access violation (`0xC0000005`) inside Intel's driver (`igvk64.dll`). With driver
32.0.101.7092 the same test passed three times in a row, so the crash depends on the driver version,
not on Polypous or Burn. Old drivers are common on the casual users' laptops Polypous targets, and
the crash hits the worst moment: the app's first start.

Requirements from the owner:
- No crash of the app because of a device probe, as far as technically possible. Refusing GPU
  acceleration is not an acceptable answer.
- **Every** device is probed, so the planner knows all available tiers (GPU, CPU, disk) and their
  properties.
- Polypous starts probing automatically. Apps can also run the probe themselves through the API,
  at any time.

## Decision

### 1. Probing runs in a separate prober process

All device enumeration and probing that loads GPU drivers runs in a **prober process**, never in the
app's process. A crashing or hanging driver only takes down the prober.

```
App process (Polypous)                          Prober process
──────────────────────                          ──────────────
Engine::new()
  ├─ probe cache valid for all devices? → use it, no prober
  └─ otherwise start the prober ───────────────► enumerate devices → 1 JSON line
       read its output line by line ◄────────── probe device 1      → 1 JSON line
       (timeout per device)                      probe device 2      → 1 JSON line
                                                 …                   → "done" line
  ├─ prober exited normally → use the results
  ├─ prober died while probing device N → N = "crashed during probe";
  │     start the prober again for the devices not yet reported
  └─ no line within the timeout → kill the prober, device = "probe timed out";
        continue with the remaining devices as above
  save results in the probe cache
```

- **Protocol:** the prober writes one JSON object per line to stdout, starting with a header that
  carries the protocol version and the Polypous version. One line per enumerated device and one per
  probed device. A final `done` line. Logs go to stderr. Output is untrusted input to the app
  process: size limits apply and malformed lines are errors.
- **Recovery:** results already reported survive a crash. The prober is restarted with the list of
  devices still to probe (at most once per device), so one bad driver never hides the other devices.
- **Timeouts:** per device, default 60 s for the first probe of a device (it includes kernel
  compilation), configurable. A hung prober is killed.
- **Scope:** the prober probes every device of every compiled-in backend, including the CPU backend,
  so the report is complete. RAM size and disk properties come from operating-system queries in the
  app process; they don't load GPU drivers.

### 2. Probe cache

Results are cached per user (platform cache directory, overridable in `EngineConfig`), keyed by
device identity, backend, **driver fingerprint** and Polypous version. The driver fingerprint comes
from operating-system sources that don't load GPU drivers (Windows: adapter driver versions from the
registry/WMI; Linux: `/sys/class/drm` and `/proc/driver/nvidia/version`; macOS: OS build). If the
fingerprint is unchanged, `Engine::new` uses the cache and starts no prober. If it can't be
determined, the prober runs in a fast enumerate-only mode to compare device lists, and probes only
new or changed devices.

A device recorded as "crashed during probe" or "probe timed out" stays unavailable until its driver
fingerprint changes or the app forces a re-probe.

### 3. Which program the prober is

In order of preference:

1. **`polypous-probe`, a tiny executable** shipped next to the Polypous shared library (M7). It loads
   that same library and calls its probe entry point, so it adds a few hundred KB, not a second copy
   of Burn and LLVM (about 90 MB with `cuda`), and it tests exactly the code the app uses.
   Polypous finds it next to its own library, or at `EngineConfig::prober_path`.
2. **The app itself, re-launched** (for Rust apps that link Polypous statically): the app calls
   `polypous::probe::run_prober_if_requested()` as the first line of `main`. When Polypous starts
   the current executable with its prober arguments, that call runs the prober and exits.
3. **`polyp probe --json`** (the developer CLI), which is the prober during M0–M6, before the shared
   library exists.

### 4. Probing through the API

Apps can run the probe themselves at any time, for example for a "Re-scan hardware" button or after
a driver update:

- `polypous::probe::probe_devices(options, cancel, on_event) -> ProbeReport`: runs the probe with the
  same machinery as `Engine::new` (prober process, cache, recovery) and returns the report. Options:
  `force` (ignore the cache), `devices` (a subset), `isolation` (`auto` | `out_of_process` |
  `in_process`), `timeout_per_device`. Events: `ProbeStarted`, `DeviceEnumerated`, `DeviceProbed`,
  `ProbeFinished`.
- `engine.refresh_devices(options, cancel, on_event) -> ProbeReport`: the same for a running
  `Engine`. It updates the engine's device list. Models that are already loaded keep their placement;
  new loads use the new list.
- `EngineConfig::probe_report`: an app may pass a `ProbeReport` it produced itself (for example in
  its own child process). Polypous then starts no prober.
- `ProbeReport` is plain JSON-compatible data ([public-api.md §2](../specs/public-api.md#2-wrapper-friendliness-rules-binding-for-all-public-items))
  and is mirrored in the C ABI (`pp_probe_devices`, `pp_engine_refresh_devices`).

### 5. Fallback: in-process probing

If no prober can be started (no prober executable found and no re-launch hook, sandboxed app
stores, locked-down machines) or the app sets `isolation = in_process`, Polypous probes inside the
app process, with two safeguards:

- **Crash journal:** before each device probe, an entry ("probing device X, driver fingerprint Y") is
  written and flushed to the cache directory, and removed afterwards. An entry left over at the next
  start means that probe crashed the process: the device is marked "crashed during probe" until its
  driver changes. The app crashes at most once per device and driver version.
- **Priority order:** devices are probed in the GPU priority order of placement.md §1, so the most
  likely primary GPU is known before a risky one is touched.

The fallback logs a warning, and `ProbeReport` and `polyp devices` show that probing was not isolated.

### 6. Using devices after probing

The app process initializes only devices that passed the probe, and only when they are needed
(model load). A device marked "crashed during probe" or "timed out" is never initialized in the app
process. Initializing a backend can still load every installed driver of that API (for example all
Vulkan ICDs); that residual risk is accepted and documented.

## Alternatives considered

- **`catch_unwind` only:** misses native crashes.
- **Probe only the primary GPU candidate:** avoids most crashes, but the planner needs all devices.
  Kept only as the fallback's ordering.
- **Crash journal as the main mechanism:** still one crash per device and driver version, at the
  app's first start. Kept only as the fallback.
- **Refuse GPU acceleration without isolation:** rejected by the owner.
- **A full-size helper executable:** about 90 MB extra with `cuda`. The tiny `polypous-probe` on top
  of the shared library avoids that.
- **Driver blocklist:** can be added later as data, but only covers crashes already reported.

## Consequences

- [placement.md §1](../specs/placement.md#1-device-discovery) describes the prober, protocol, cache
  and fallback; [public-api.md](../specs/public-api.md) describes the probe API and `EngineConfig`
  fields.
- Issue #4 implements the prober protocol, recovery, cache and fallback, with `polyp probe --json`
  as the prober. The tiny `polypous-probe` executable arrives with the shared library and C ABI (M7).
- Shipped apps must ship and **sign** `polypous-probe` like their own binaries (Windows SmartScreen,
  macOS notarization), and Electron apps must keep it outside the `asar` archive. The wrapper guide
  (M7) covers this.
- The first start on a machine takes longer (kernel compilation per device). Later starts use the cache.
- Crashes **after** probing (a driver that crashes only on certain kernels, during real use) are not
  covered. Extending the journal to the first model load on a device is possible later.
