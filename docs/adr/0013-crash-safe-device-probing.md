# ADR-0013: Crash-safe device probing

- **Status:** Proposed
- **Date:** 2026-11-03
- **Deciders:** project owner (+ authoring agent)
- **Supersedes / Superseded by:** extends [placement.md §1](../specs/placement.md#1-device-discovery) and [ADR-0011](0011-use-burn-dispatch-for-backend-selection.md); does not replace them

## Context

Polypous probes every device at `Engine` creation (initialize, allocate, run a tiny kernel, compare
the result) and promises that a broken device makes only that device unavailable, never the app
([placement.md §1](../specs/placement.md#1-device-discovery)). The current plan contains failures
with `catch_unwind`, because Burn/CubeCL `unwrap()` some driver calls.

The reference machine showed that this is not enough
([ADR-0007 follow-up notes](0007-burn-version-pinning.md#follow-up-notes)): with the Intel Iris Xe
driver 30.0.101.1298 (January 2022), running a matmul through Vulkan on the iGPU **crashed the whole
process** with an access violation (`0xC0000005`) inside Intel's driver (`igvk64.dll`). A native
crash in a driver is not a Rust panic: no in-process mechanism can catch it reliably. On Linux the
equivalent is a `SIGSEGV`. After updating to driver 32.0.101.7092 the same test passed three times
in a row, so the crash depends on the driver version, not on Polypous or Burn.

Old and buggy GPU drivers are common on exactly the machines Polypous targets: casual users' laptops
that never update graphics drivers. An app that crashes on start, every time, because of the iGPU it
doesn't even need, breaks the project's stability priority.

Constraints:
- Polypous is a library. It runs inside apps it does not control (Rust programs, Node/Electron
  addons via the C ABI). It cannot assume it may re-launch the host executable with special
  arguments: in an Electron app the host executable is Electron.
- A probe must stay fast after the first start. Users won't wait for GPU probing on every launch.
- The `cuda` backend makes binaries large (90 MB release, static LLVM). A separate helper binary
  with every backend would be about as large again.

## Decision (proposed)

Three layers. Each one works without the next.

1. **Probe cache.** Probe results are stored per user (for example in the platform cache directory,
   overridable in `EngineConfig`) and keyed by device identity, driver version, Polypous version and
   backend. A device is probed again only when one of these changes. A normal start does no probing
   at all.

2. **Crash journal (default, always on).** Before probing a device, Polypous writes a journal entry
   ("probing device X with driver Y") to the cache directory and flushes it; after the probe it
   removes the entry. If an entry is still there at the next `Engine` creation, the previous probe
   crashed the process. That device is then marked `unavailable: crashed during probe (driver Y)`
   and never probed again until its driver version changes. The app crashes **at most once per
   device and driver version**, and from then on starts normally on the remaining devices.

3. **Out-of-process probing (optional, recommended for shipped apps).** If the app provides a probe
   helper, Polypous runs the probes in a child process and reads the results as JSON. A crashing
   driver then only kills the child, and the app never crashes. The helper can be:
   - the app itself: Rust apps call `polypous::probe::run_if_requested()` first thing in `main`,
     which handles a special command line and exits, so Polypous can re-launch the current
     executable for probing; or
   - a small `polypous-probe` executable shipped next to the app (needed for Electron/Node apps),
     configured with `EngineConfig::probe_helper`.

   If no helper is configured or it fails to start, Polypous falls back to layer 2 and logs a warning.

Results of all layers have the same shape (`DeviceInfo` with availability and reason), so apps and
the placement planner don't need to know which layer produced them.

## Alternatives considered

- **`catch_unwind` only (current plan):** handles Rust panics but not native crashes. Not enough.
- **Always probe out of process:** the most robust, but every app would have to ship a helper binary
  or adopt the re-exec hook, which conflicts with "easy to embed" (ADR-0008).
- **Driver blocklist only:** a curated list of known-bad driver versions. Useful as an addition
  later, but it only covers crashes someone has already reported. Data-driven, so it could live in a
  file updated with Polypous releases.
- **Skip integrated GPUs when a discrete GPU exists:** avoids this specific case on the reference
  machine, but not on iGPU-only laptops (the v1 headline target), where the iGPU is the only GPU.

## Consequences

- `placement.md` §1 gets a "crash safety" section describing the cache, journal and helper; the
  `EngineConfig` in `public-api.md` gets `probe_cache_dir`, `probe_helper` and a way to force
  re-probing.
- The wrapper guide (M7) explains how to ship `polypous-probe` with an Electron app.
- Issue #4 (device discovery) implements layers 1 and 2. Layer 3 becomes its own issue in M0 or M1.
- Crashes **after** probing (for example a driver that crashes only on certain kernels) are not
  covered. The same journal idea could later be extended to the first real use of a device.
- The journal needs care to stay correct with several `Engine`s or processes probing at once
  (file locking or one journal file per process).

## Open questions for the owner

1. Accept the three layers, or require out-of-process probing from the start?
2. Is a single crash per device and driver version acceptable for apps without a helper?
