# Spec: Public API and Embedding Rules

**Status:** Draft. The facade grows from M1, the C ABI comes in M7. **Crates:** `polypous`, `polypous-capi`
**Decision:** [ADR-0008](../adr/0008-wrapper-friendly-api.md)

Primary users are app developers, often building Electron apps through a Node.js native addon they
write themselves. The Rust API must therefore be easy to wrap. Every public item follows the rules in §2.

## 1. Shape of the API (illustrative)

```rust
use polypous::{Engine, EngineConfig, ModelSource, LoadOptions, CancellationToken, ProbeOptions};

let engine = Engine::new(EngineConfig::default())?;          // device discovery: probe cache, else prober process
for d in engine.devices() { println!("{d:?}"); }

// Optional: probe again at any time, e.g. a "Re-scan hardware" button (ADR-0013).
let report = engine.refresh_devices(
    ProbeOptions { force: true, ..Default::default() },      // ignore the cache
    &CancellationToken::new(),
    |event| println!("{event:?}"),                           // ProbeStarted, DeviceProbed, ...
)?;

// Optional: inspect before loading.
let plan = engine.plan(&ModelSource::path("llama-3-8b.Q4_K_M.gguf"), &LoadOptions::default())?;
println!("{}", plan.summary());                              // memory per tier, speed estimate, warnings

let model = engine.load(
    ModelSource::path("llama-3-8b.Q4_K_M.gguf"),
    LoadOptions::default(),
    &CancellationToken::new(),
    |event| println!("{event:?}"),                           // LoadProgress { fraction, stage }
)?;

let chat = model.text_generation()?;                          // Err if the model's task differs
let mut session = chat.new_session(SessionOptions { max_context: 4096, ..Default::default() })?;

let result = session.generate(
    TextRequest::chat(vec![Message::user("Hello!")]),
    &CancellationToken::new(),
    |event| if let TextEvent::Token { text_delta, .. } = event { print!("{text_delta}") },
)?;
```

Builder/training side:

```rust
let graph = polypous::Graph::from_json(&json)?;               // or build programmatically
let report = graph.validate();                                 // all errors with node ids
let trainer = engine.trainer(&graph, TrainingConfig::from_json(&cfg)?)?;
let outcome = trainer.run(&cancel, |event| { /* Step, EpochFinished, ... */ })?;
outcome.export("model.graph.json", "model.safetensors")?;
```

## 2. Wrapper-friendliness rules (binding for all public items)

1. **No generics or lifetimes in public types.** No `Engine<B: Backend>`, no `Model<'a>`. Backend
   choice is a runtime value.
2. **Handles.** Public objects (`Engine`, `Model`, `Session`, `Trainer`) are cheap-to-clone handles
   (`Arc` inside), `Send + Sync + 'static`. Dropping the last handle frees resources; there is no
   required `close()`, but one is provided for deterministic release.
3. **Plain data across the boundary.** Configs, requests, results and events are serde structs/enums
   whose JSON form is stable and documented (and exported as JSON Schema). Every function that takes
   a config struct must be equivalent to one taking its JSON.
4. **Tensors** cross the boundary as `TensorData { dtype, shape: Vec<u64>, bytes: Vec<u8> }`
   (little-endian, row-major). There are no Burn types, and no zero-copy promises, in the public API.
   Element encodings: IEEE 754 for `f32`/`f16`, bfloat16 for `bf16`, two's complement for
   `i64`/`i32`, and one byte per element for `bool` (`0` false, `1` true; any other byte is an
   error). `bytes.len()` must equal the product of `shape` times the element size; a dimension may
   be `0`. Malformed data is an `InvalidInput` error, never a panic.
5. **Blocking calls + callbacks + cancellation.** Long operations block the calling thread, report
   through a callback, and take a `CancellationToken`. No async runtime is required. Wrappers call
   them from a worker thread.
6. **Callbacks may come from any thread**, but never concurrently for the same call, and always in
   order. Callbacks must not call back into the same session (documented; detected and errored).
7. **Errors** are `polypous::Error { kind: ErrorKind, message: String, context: ErrorContext }`.
   `ErrorKind` is a stable, non-exhaustive enum:
   `InvalidInput, InvalidGraph, UnsupportedModel, UnsupportedOperation, Io, OutOfMemory,
   BackendUnavailable, Backend, Cancelled, Internal`. Messages are suitable to show to end users,
   while `context` carries technical details (node id, file, tensor name, device).
8. **No panics across the API.** The facade catches panics at the boundary and turns them into `Internal` errors.
9. **No global state.** Several `Engine`s may coexist. Logging goes through `tracing`, and the app
   installs its own subscriber, or calls `polypous::logging::init_default()`.
10. **Strings are UTF-8**; paths are accepted as `&Path` and as UTF-8 strings.

## 3. C ABI (M7)

`polypous-capi` mirrors the facade 1:1 using the rules above:

```c
uint32_t   pp_abi_version(void);
pp_status  pp_engine_new(const char* config_json, pp_engine** out);
void       pp_engine_free(pp_engine*);
pp_status  pp_engine_devices_json(pp_engine*, char** out_json);
pp_status  pp_model_load(pp_engine*, const char* source_json, const char* options_json,
                         pp_cancel* cancel, pp_event_cb cb, void* user, pp_model** out);
pp_status  pp_text_session_new(pp_model*, const char* options_json, pp_text_session** out);
pp_status  pp_text_generate(pp_text_session*, const char* request_json, pp_cancel*,
                            pp_event_cb cb, void* user, char** out_result_json);
const char* pp_last_error_json(void);       /* thread-local, valid until next call on this thread */
void       pp_string_free(char*);
typedef void (*pp_event_cb)(void* user, const char* event_json);
```

- Everything structured is JSON (UTF-8). Strings returned to the caller are freed with `pp_string_free`.
- Handles are opaque pointers. Every `*_new` has a `*_free`.
- `pp_cancel` is created and triggered by the caller from any thread.
- The header is generated with `cbindgen`. ABI changes bump `pp_abi_version`.

### 1.1 Device probing ([ADR-0013](../adr/0013-crash-safe-device-probing.md))

| Item | Purpose |
|---|---|
| `polypous::probe::probe_devices(options, cancel, on_event) -> ProbeReport` | Run the probe without an `Engine` (prober process, cache, recovery) |
| `Engine::refresh_devices(options, cancel, on_event) -> ProbeReport` | Probe again and update a running engine's device list. Loaded models keep their placement |
| `polypous::probe::run_prober_if_requested()` | First line of `main` in Rust apps that link Polypous statically: lets Polypous use the app itself as the prober. Returns normally when the process was not started as a prober |
| `ProbeOptions` | `force` (ignore the cache), `devices` (subset), `isolation` (`auto` \| `out_of_process` \| `in_process`), `timeout_per_device` |
| `ProbeEvent` | `ProbeStarted`, `DeviceEnumerated`, `DeviceProbed`, `ProbeFinished` |
| `ProbeReport` | Devices with status and reason, whether probing was isolated, cache hits. Plain JSON data |
| `EngineConfig::probe_report` | Use a report the app produced itself; Polypous then starts no prober |
| `EngineConfig::prober_path`, `EngineConfig::probe_cache_dir` | Where the prober executable and the probe cache are |

C ABI (M7): `pp_probe_devices`, `pp_engine_refresh_devices`, and the entry point the tiny
`polypous-probe` executable calls.

### 1.2 Engine, devices and errors (M0, issue #6)

Implemented so far: `Engine::new`, `Engine::devices`, `Engine::primary_gpu`, `Engine::config`,
`EngineConfig` (with `from_json`/`to_json`), the device types below and `Error`. The CLI's
`polyp devices --json` prints exactly `Engine::devices()` as JSON.

| Item | Behavior |
|---|---|
| `Engine::new(config) -> Result<Engine, Error>` | Discovers and probes every device of every compiled-in backend (placement.md §1; in process until #14), then marks the primary GPU. A failing device is reported as unavailable, never as an error. A panic becomes `Internal` |
| `Engine` | `Arc` handle: `Clone + Send + Sync + 'static`, no generics or lifetimes. No `close()` yet: it holds no resources that need deterministic release until models can be loaded |
| `engine.devices() -> Vec<DeviceInfo>` | All devices in probe order (GPU priority order, the CPU last), the primary GPU with `primary: true` |
| `engine.primary_gpu() -> Option<DeviceInfo>` | The device with `primary: true`, or `None` (CPU only) |

**`EngineConfig` JSON.** An object; every field is optional, so `{}` is the defaults. Unknown
fields, and any JSON that is not an object, are an `InvalidInput` error (so a misspelled option never
goes unnoticed). `to_json` writes every field.

| Field | Type, default | Meaning |
|---|---|---|
| `software_rasterizers` | bool, `false` | Open and probe software rasterizers (Vulkan/Metal adapters of type `cpu`: lavapipe, SwiftShader, …). Off: they are listed with the reason `software_rasterizer_skipped` and never opened ([placement.md §1](placement.md#1-device-discovery)). They are never the primary GPU either way. `polyp devices` uses the default |

Later fields (`probe_report`, `prober_path`, `probe_cache_dir` with #14, placement options with M4)
are optional with documented defaults, so `{}` stays valid.

**Device JSON** (`DeviceInfo`, an array of them from `devices()`). The fields, names and values are
those of the backend's `DeviceInfo` in [placement.md §1](placement.md#1-device-discovery), plus
`primary`. The facade has its own types (`polypous::DeviceInfo`, `DeviceId`, `BackendKind`,
`AdapterKind`, `DeviceKind`, `PhysicalIdentity`, `DeviceCapabilities`, `DeviceStatus`,
`UnavailableReason`) and converts from `polypous_backend::device`, so the backend can change
without changing this JSON.

```jsonc
{
  "id": {"backend": "cuda", "index": 0},       // or {"backend":"vulkan"|"metal","adapter":"discrete_gpu"|"integrated_gpu"|"virtual_gpu"|"cpu"|"other","index":0}, {"backend":"cpu"}
  "backend": "cuda",                           // "cuda" | "metal" | "vulkan" | "cpu"
  "name": "NVIDIA GeForce RTX 3060 Laptop GPU",
  "vendor": "NVIDIA",                          // string or null
  "driver": null,                              // string or null
  "kind": "discrete",                          // "discrete" | "integrated" | "cpu" | "other"
  "dedicated_memory": 6441926656,              // bytes or null (null for the CPU)
  "free_memory": 5939134464,                   // bytes or null
  "unified_memory": false,
  "identity": {"pci_bus_id": "0000:01:00.0", "luid": null, "registry_id": null},
  "capabilities": {"f16": true, "bf16": true, "subgroup_size_min": 32, "subgroup_size_max": 32,
                   "max_buffer_size": 6441926656, "max_shared_memory": 49152, "compute_units": 30},
  "status": {"state": "available", "isolated": false},
      // or {"state":"not_probed"}, {"state":"unavailable","reason":{"code":"probe_failed","message":"…"},"isolated":false};
      // reason codes: crashed_during_probe, probe_timed_out, probe_failed (+message), library_missing (+library),
      // software_rasterizer_skipped (listed, not opened; EngineConfig.software_rasterizers opts in)
  "primary": true                              // the primary GPU (placement.md §1); at most one device, none for CPU only
}
```

Every field is always present (`null` when unknown). Readers must accept new fields and new enum
values (the Rust types are `#[non_exhaustive]`).

**Error JSON.** `{"kind":"invalid_input","message":"…","context":{"operation":"…","details":"…"}}`.
`kind` is the `snake_case` name of an `ErrorKind` from §2 rule 7 (`invalid_input`, `invalid_graph`,
`unsupported_model`, `unsupported_operation`, `io`, `out_of_memory`, `backend_unavailable`, `backend`,
`cancelled`, `internal`). `message` is for end users. `context` fields are optional and omitted when
absent: `operation` (the facade call that failed, for example `engine_new`) and `details` (the
underlying error or panic message). More context fields (node id, file, tensor name, device) are
added as needed.

**Logging.** Discovery logs through `tracing` under the `polypous` target: per-device enumeration,
description and probe results at `debug` (`RUST_LOG=polypous=debug`), unavailable devices at
`warn`, the primary GPU at `info`.

## 4. Events and threading

- Polypous runs heavy work on its own thread pool. The calling thread blocks until the operation
  finishes, and callbacks run on the calling thread *or* a worker thread (unspecified). Wrappers
  must marshal them to their event loop (for Node.js: N-API thread-safe functions).
- Within one call, events arrive in order, and a `Finished`-type event (or an error return) is
  always the last thing that happens.

## 5. Notes for wrapper authors (Electron)

Wrappers are out of scope for this repository, but must be easy to write. A guide will cover:
- building a Node addon with `napi-rs` over the Rust API (simplest) or over the C ABI (any language)
- running inference outside the renderer process (main or utility process)
- selecting backends via cargo features when building, so CUDA support does not require a CUDA
  installation on end-user machines (libraries load at runtime, with fallback)
- shipping `.polyp` files: built-ins are embedded, and extras are registered at runtime
- shipping the `polypous-probe` executable next to the Polypous library, signed like the app and
  outside the `asar` archive ([ADR-0013](../adr/0013-crash-safe-device-probing.md))

## 6. Stability

Before 1.0, the API may change between minor versions, and changes go in `CHANGELOG.md`. The JSON
shapes of configs, requests and events are treated as more stable than the Rust API, because wrappers depend on them.
