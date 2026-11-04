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
| `status` | `available`, or `unavailable` with a reason (`crashed during probe`, `probe timed out`, `probe failed: …`, `library missing`) and whether the probe ran isolated |

**Implementation (M0, issue #4; `polypous_backend::device`).** The building blocks the prober runs:

- `list_devices()` / `list_backend_devices(backend)`: device ids from Burn's `Device::enumerate`
  per compiled-in backend, without opening devices. A missing runtime library means no devices.
  (`Device::enumerate_physical` skips devices without a card identity, such as the CPU and
  lavapipe, so it is not used directly; the same identity data is used for matching, see below.)
- `describe_device(id)`: opens the device (creates its CubeCL client, which loads the driver) and
  reads the fields above. `enumerate_devices()` = list + describe.
- `probe_device(id)`: `[[1,2],[3,4]] @ [[5,6],[7,8]] + 1` on the device, read back and compared
  exactly with `[20, 23, 44, 51]` (small integers, exact in f32/tf32/f16).
- None of them fails or panics. Errors and Rust panics (`catch_unwind`) become
  `unavailable: probe failed: …`; the in-process functions report `isolated: false`, the prober
  marks its results isolated. Calls are serialized by a process-wide lock.

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
| `status` | `{"state":"not_probed"}`, `{"state":"available","isolated":bool}`, `{"state":"unavailable","reason":{"code":"crashed_during_probe"\|"probe_timed_out"\|"probe_failed","message"\|"library_missing","library"},"isolated":bool}` | `not_probed` after `describe_device`; the probe sets the rest |

**`Engine::new` (M0, issue #6; `polypous::Engine`).** Until the prober process (#14) exists, the
engine discovers devices **in process**: it lists the devices of every compiled-in backend, sorts
them into **probe order** (CUDA, Metal, Vulkan discrete, Vulkan integrated, other Vulkan adapters,
Vulkan software rasterizers, CPU; the backend's order within each group), then describes and probes
each one (a device that could not be opened is not probed). The list keeps that order, and the
facade marks the result of `select_primary_gpu` as `primary` (public-api.md §1.2). A native driver
crash still takes the process down until #14; there is no crash journal yet. The discovery step is
one function from the config to the device list, so #14 replaces it without changing the engine.

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
