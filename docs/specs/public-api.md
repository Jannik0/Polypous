# Spec: Public API and Embedding Rules

**Status:** Draft. The facade grows from M1, the C ABI comes in M7. Binding: §1.1–§1.3 and §2 (implemented), §1.4 (M2). **Crates:** `polypous`, `polypous-capi`
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
let trainer = engine.trainer(&graph, TrainingConfig::from_json(&cfg)?, TrainerOptions::default(), &cancel)?;
let outcome = trainer.run(&cancel, |event| { /* started, step, epoch_finished, ... */ })?;
trainer.export("model.graph.json", "model.safetensors")?;     // §1.4
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

Implemented in M0 (issue #14). Behavior: [placement.md §1](placement.md#1-device-discovery), "The probe".

| Item | Purpose |
|---|---|
| `polypous::probe::probe_devices(options, cancel, on_event) -> Result<ProbeReport, Error>` | Run the probe without an `Engine` (cache, prober process, recovery, fallback) |
| `Engine::refresh_devices(options, cancel, on_event) -> Result<ProbeReport, Error>` | Probe again and update a running engine's device list (all devices, or only `options.devices`). Loaded models keep their placement. `prober_path` and `probe_cache_dir` default to the engine's; the other options are used as given |
| `Engine::probe_report() -> ProbeReport` | The report of the probe behind the current device list |
| `polypous::probe::run_prober_if_requested()` | First line of `main` in Rust apps that link Polypous: lets Polypous use the app itself as the prober. When started as `<app> --polypous-prober --request <json>` it runs the probe and exits; otherwise it returns normally |
| `polypous::probe::run_prober(request_json) -> i32` | Runs the prober in this process (stdout protocol) and returns the exit code; `polyp probe --json` uses it |
| `ProbeOptions` | `force` (ignore the cache), `devices` (subset), `isolation` (`auto` \| `out_of_process` \| `in_process`), `timeout_per_device_ms` (default 60000, 1 ms to one day), `software_rasterizers` (default false), `prober_path`, `probe_cache_dir`. JSON object, unknown fields rejected, `from_json`/`to_json` |
| `ProbeEvent` | JSON `{"event":"probe_started"}`, `{"event":"device_enumerated","id":…}`, `{"event":"device_probed","device":…,"cached":bool}` (`primary` is always false here), `{"event":"probe_finished","report":…}`; in that order, `probe_finished` last unless an error is returned |
| `ProbeReport` | `devices` (probe order, `primary` and `physical_gpu` set), `isolated` (every result in the report came from a prober process), `cache_hits` (ids), `prober_starts`, `fingerprint_known`, `warnings` (including why a compiled-in GPU backend has no device, placement.md §1). Plain JSON data |
| `CancellationToken` | `new`, `cancel`, `is_cancelled`; clones share one flag. Cancelling kills a running prober; the call returns `Cancelled` |
| `EngineConfig::probe_report` | Use a report the app produced itself; Polypous then starts no probe (`primary` is recomputed) |
| `EngineConfig::prober_path`, `EngineConfig::probe_cache_dir` | Where the prober executable and the probe cache are |
| `EngineConfig::probe_isolation`, `EngineConfig::probe_timeout_per_device_ms` | `isolation` and `timeout_per_device_ms` for `Engine::new` |

Errors: `InvalidInput` (timeout out of range), `Cancelled`, `BackendUnavailable` (`out_of_process`
without a working prober), `Internal` (caught panic). Device problems are statuses, never errors.

C ABI (M7): `pp_probe_devices`, `pp_engine_refresh_devices`, and the entry point the tiny
`polypous-probe` executable calls.

### 1.2 Engine, devices and errors (M0, issue #6)

Implemented so far: `Engine::new`, `Engine::devices`, `Engine::primary_gpu`, `Engine::config`,
`Engine::probe_report`, `Engine::refresh_devices`, `EngineConfig` (with `from_json`/`to_json`), the
device types below, the probe API of §1.1 and `Error`. The CLI's
`polyp devices --json` prints exactly `Engine::devices()` as JSON (`--json --report`:
`Engine::probe_report()`); `polyp devices` options `--software-rasterizers`, `--isolation`,
`--probe-cache-dir`, `--prober-path`, `--timeout-per-device-ms` set the `EngineConfig` fields.

| Item | Behavior |
|---|---|
| `Engine::new(config) -> Result<Engine, Error>` | Discovers and probes every device of every compiled-in backend (placement.md §1: probe cache, else prober process, else in process with crash journal), then marks the primary GPU. A failing device is reported as unavailable, never as an error. A panic becomes `Internal` |
| `Engine` | `Arc` handle: `Clone + Send + Sync + 'static`, no generics or lifetimes. No `close()` yet: it holds no resources that need deterministic release until models can be loaded |
| `engine.devices() -> Vec<DeviceInfo>` | All devices in probe order (GPU priority order, the CPU last), the primary GPU with `primary: true` |
| `engine.primary_gpu() -> Option<DeviceInfo>` | The device with `primary: true`, or `None` (CPU only) |

**`EngineConfig` JSON.** An object; every field is optional, so `{}` is the defaults. Unknown
fields, and any JSON that is not an object, are an `InvalidInput` error (so a misspelled option never
goes unnoticed). `to_json` writes every field.

| Field | Type, default | Meaning |
|---|---|---|
| `software_rasterizers` | bool, `false` | Open and probe software rasterizers (Vulkan/Metal adapters of type `cpu`: lavapipe, SwiftShader, …). Off: they are listed with the reason `software_rasterizer_skipped` and never opened ([placement.md §1](placement.md#1-device-discovery)). They are never the primary GPU either way. `polyp devices` uses the default |

| `probe_report` | `ProbeReport` or null, null | A report the app produced itself; no probe runs |
| `prober_path` | path string or null, null | The prober executable (else the current executable if it called `run_prober_if_requested`, else in-process probing) |
| `probe_cache_dir` | path string or null, null | Probe cache and crash journal directory (else the platform cache directory) |
| `probe_isolation` | `"auto"` \| `"out_of_process"` \| `"in_process"`, `"auto"` | Where `Engine::new` probes |
| `probe_timeout_per_device_ms` | integer, 60000 | Prober timeout per device |

Later fields (placement options with M4) are optional with documented defaults, so `{}` stays
valid.

**Device JSON** (`DeviceInfo`, an array of them from `devices()`). The fields, names and values are
those of the backend's `DeviceInfo` in [placement.md §1](placement.md#1-device-discovery), plus
`physical_gpu` and `primary`. The facade has its own types (`polypous::DeviceInfo`, `DeviceId`, `BackendKind`,
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
  "physical_gpu": 0,                           // index of the physical GPU, or null (CPU, software rasterizers)
  "primary": true                              // the primary GPU (placement.md §1); at most one device, none for CPU only
}
```

`physical_gpu` numbers the physical GPUs of the list, 0, 1, …, in the order they first appear.
Devices with the same number are one card reached through different backends (the RTX 3060 through
CUDA and Vulkan), matched by `identity` with the rule of placement.md §1. It is `null` for devices
of kind `cpu` (the CPU and software rasterizers). `polyp devices` shows it in its `GPU` column.

Every field is always present (`null` when unknown). Readers must accept new fields and new enum
values (the Rust types are `#[non_exhaustive]`).

**Error JSON.** `{"kind":"invalid_input","message":"…","context":{"operation":"…","details":"…"}}`.
`kind` is the `snake_case` name of an `ErrorKind` from §2 rule 7 (`invalid_input`, `invalid_graph`,
`unsupported_model`, `unsupported_operation`, `io`, `out_of_memory`, `backend_unavailable`, `backend`,
`cancelled`, `internal`). `message` is for end users. `context` fields are strings, optional and
omitted when absent:

| Field | Meaning |
|---|---|
| `operation` | The facade call that failed, for example `engine_new`, `engine_load_graph`, `model_run` |
| `details` | The underlying error or panic message (for an invalid graph: every diagnostic, one per line) |
| `node` | The node: a graph node id from a validation diagnostic, or the runtime's node path (`fc1`, `layers[3]/attn`, runtime.md §5) |
| `pointer` | JSON pointer (RFC 6901) into the graph document, from a validation diagnostic, or into a training config or checkpoint `state.json` (§1.4) (`""` is the whole document) |
| `tensor` | A weight (param), input, output or state entry name, or a tensor of a tensor file |
| `file` | The file concerned (graph, weight file or shard, tensor file), as given |
| `device` | The device, in its id syntax (`cpu`, `cuda:0`, `vulkan:discrete_gpu:0`, §1.3) |

Readers must accept new context fields.

**Logging.** Discovery logs through `tracing` under the `polypous` target: per-device enumeration,
description and probe results at `debug` (`RUST_LOG=polypous=debug`), unavailable devices at
`warn`, the primary GPU at `info`.

### 1.3 Graphs and models (M1, issue #45)

Load a graph document ([graph-ir.md](graph-ir.md)), validate it, load it with safetensors weights
on one device, run it and save it. The facade is a thin layer over `polypous-ir` (validation) and
`polypous-runtime` ([runtime.md](runtime.md): plan, weights, calls, cancellation, errors); it adds
no op, shape or weight logic of its own beyond the `TensorData` checks of §2 rule 4. Training and
parameter initialization (M2), `.polyp` and GGUF (M3) and placement options (M4) come later.

```rust
use polypous::{CancellationToken, Engine, EngineConfig, Graph, GraphLoadOptions, NamedTensor, tensor_file};

let engine = Engine::new(EngineConfig::default())?;
let graph = Graph::from_file("mlp.graph.json")?;              // InvalidGraph with node and pointer
let mut options = GraphLoadOptions::default();                 // primary GPU, else the CPU; f32
options.weights = Some("mlp.safetensors".into());
let model = engine.load_graph(&graph, options, &CancellationToken::new())?;
println!("{}", serde_json::to_string(&model.info())?);         // device, inputs, outputs, state

let inputs = tensor_file::read("image.safetensors")?;         // Vec<NamedTensor>
let out = model.run(inputs, vec![], &CancellationToken::new())?;
tensor_file::write("logits.safetensors", &out.outputs)?;
model.save("model.graph.json", "model.safetensors")?;
```

**Items.**

| Item | Behavior |
|---|---|
| `Graph` | A loaded, valid graph. `Arc` handle: `Clone + Send + Sync + 'static`, immutable |
| `Graph::from_json(json: &str) -> Result<Graph, Error>` | Loads and validates a document (graph-ir.md §4, compute dtype `f32`). Any diagnostic is `InvalidGraph` (below) |
| `Graph::from_file(path) -> Result<Graph, Error>` | The same for a file: `Io` if it can't be read; a file larger than the document limit (graph-ir.md §4.3, 64 MiB) is `InvalidGraph` (`document_too_large`) without reading the rest; bytes that are not UTF-8 are `invalid_json` |
| `graph.to_json() -> String` | The canonical JSON form (graph-ir.md §6). `Graph::from_json(&g.to_json())` equals `g` |
| `graph.validate(&GraphValidateOptions) -> ValidationReport` | Validates again, for example with another compute dtype |
| `Graph::validate_json(json, &GraphValidateOptions) -> ValidationReport` | Every diagnostic of a document that may not load (editors, `polyp graph validate`) |
| `Graph::validate_file(path, &GraphValidateOptions) -> Result<ValidationReport, Error>` | The same for a file; `Io` if it can't be read |
| `Engine::load_graph(&graph, GraphLoadOptions, &CancellationToken) -> Result<Model, Error>` | Compiles the graph for one device and loads its weights (runtime.md §1, §2). Blocking |
| `Model` | A loaded model. `Arc` handle: `Clone + Send + Sync + 'static`; calls on clones share the weights, and may run concurrently from several threads |
| `model.info() -> ModelInfo` | Device, compute dtype, inputs, outputs, state entries and params |
| `model.graph() -> Graph` | The graph it was loaded from |
| `model.run(inputs, state, &CancellationToken) -> Result<RunOutput, Error>` | One call (runtime.md §3). Blocking |
| `model.save(graph_path, weights_path) -> Result<(), Error>` | Writes `model.graph.json` and `model.safetensors` (below) |
| `model.close()` | Frees the weights now (§2 rule 2). Later `run` and `save` calls on any clone are `InvalidInput`; `info` and `graph` still work. Calls that are running finish first. Idempotent |
| `tensor_file::read(path) -> Result<Vec<NamedTensor>, Error>`, `tensor_file::write(path, &[NamedTensor])` | Named tensors from and to a safetensors file (below), for apps and the CLI |
| `DeviceId::parse(text) -> Result<DeviceId, Error>` (also `FromStr`) | The device id syntax that `Display` writes: `cpu`, `cuda:0`, `vulkan:discrete_gpu:0`, `metal:integrated_gpu:0` (adapters as in §1.2). Anything else is `InvalidInput` |

Paths are `impl AsRef<Path>`, so `&Path`, `PathBuf` and UTF-8 strings all work (§2 rule 10).

**`GraphValidateOptions` JSON.** `{"compute_dtype":"f32"}`: an object, every field optional,
unknown fields rejected (as `EngineConfig`, §1.2), `from_json`/`to_json`. `compute_dtype` is
`"f32"` (default), `"f16"` or `"bf16"` (`ComputeDtype`, graph-ir.md §2.2).

**`ValidationReport` JSON.** Plain data, the diagnostics in the order the checks ran:

```jsonc
{
  "valid": false,                                 // true iff diagnostics is empty
  "diagnostics": [
    {
      "code": "unknown_reference",                // stable code of graph-ir.md §4.2
      "node": "fc1",                              // node id as written (not qualified by repeats), or null
      "pointer": "/nodes/0/inputs/x",             // JSON pointer into the document; "" is the whole document
      "message": "unknown value `imgae`"          // human-readable
    }
  ]
}
```

For `graph.validate`, pointers point into the canonical form (`to_json`), which may differ from the
file the graph was loaded from. The validation functions never fail on a document: if validation
itself panics (a bug), the report has one diagnostic with the code `internal_error`, so a graph
never looks valid by accident.

**Invalid graphs.** `from_json` and `from_file` fail with `InvalidGraph`. `message` is
`The graph is not valid: <first diagnostic's message>` (with `(and N more errors)` when there are
more); `context.node` and `context.pointer` are the first diagnostic's (node omitted when it has
none), `context.details` lists every diagnostic, one per line, `context.file` the file. Use
`validate_json`/`validate_file` to get them as data. `load_graph` validates again with the
requested compute dtype; a graph that is valid with `f32` but not with that dtype is the same
`InvalidGraph` error.

**`GraphLoadOptions` JSON.** An object, every field optional, unknown fields rejected,
`from_json`/`to_json`:

| Field | Type, default | Meaning |
|---|---|---|
| `weights` | path string or null, null | A safetensors file, or a sharded checkpoint's `*.safetensors.index.json` (formats.md §1.5). Null: the graph must declare no params |
| `device` | device id (§1.2 JSON) or null, null | The device. Null: the engine's primary GPU, or the CPU if there is none |
| `compute_dtype` | `"f32"` \| `"f16"` \| `"bf16"`, `"f32"` | The compute dtype (graph-ir.md §2.2) |

`load_graph`, in this order:

1. **Device.** The device must be in `engine.devices()` with status `available`. A device that is
   unknown to the engine, unavailable (a failed probe, a skipped software rasterizer) or of a
   backend that isn't compiled in is `BackendUnavailable` with `context.device`. The default device
   needs no GPU: with no primary GPU it is the CPU.
2. **Plan** (runtime.md §1): validation with the compute dtype (`InvalidGraph`), then the device's
   dtype check. A compute dtype or value dtype the device lacks (`bf16` on Vulkan) is
   `UnsupportedOperation` with `context.device` (and `context.node` for a node output).
3. **Weights** (runtime.md §2): strict name match, storage dtype, shape, one rounding to the
   compute dtype. Errors name the tensor in `context.tensor` and the file in `context.file`.
4. **Cancellation** is checked before each step and after the last one; a cancelled load returns
   `Cancelled` and frees what it loaded. The runtime doesn't check the token while it uploads
   weights, so one step runs to its end.

The engine and the model are independent: a model keeps its device when the engine re-probes
(§1.1), and stays usable after the last `Engine` clone is dropped.

**`ModelInfo` JSON.** Plain data:

```jsonc
{
  "device": {"backend": "cpu"},                   // DeviceId JSON (§1.2)
  "compute_dtype": "f32",
  "inputs":  [ {"name": "image", "dtype": "f32", "shape": ["batch", 784]} ],    // declaration order
  "outputs": [ {"name": "logits", "dtype": "f32", "shape": ["batch", 10]} ],    // declaration order
  "state":   [ ],                                 // root-level state entries, expanded names, binding order
  "params":  [ {"name": "fc1.weight", "dtype": "f32", "shape": [128, 784]} ]   // declaration order
}
```

`dtype` is a value dtype (`f32`, `f16`, `bf16`, `i64`, `i32`, `bool`) with `float` resolved to the
compute dtype; params have the compute dtype. A `shape` entry is a number for a literal dim and a
string for a symbolic one: a symbol (`"batch"`) or a dim expression in its canonical form
(`"seq + 1"`, graph-ir.md §2.3), as validation inferred it.

**Tensors.** `TensorData { dtype, shape: Vec<u64>, bytes: Vec<u8> }` with the encodings of §2
rule 4, and `NamedTensor { name, tensor: TensorData }`. JSON:
`{"name":"x","tensor":{"dtype":"f32","shape":[2,3],"bytes":[0,0,128,63,…]}}` (`bytes` as an array
of numbers; wrappers should pass tensors natively, not as JSON). `TensorData::validate` checks a
value: the byte length must equal the product of `shape` times the element size, computed with
checked arithmetic (an overflow is an error, not a panic), and every `bool` byte must be `0` or `1`;
errors are `InvalidInput`.

**Running.** `model.run(inputs: Vec<NamedTensor>, state: Vec<NamedTensor>, cancel)`:

1. Every input and state value is checked with `TensorData::validate`, and a dimension of size `0`
   is rejected (M1 graphs have no zero-size dims, graph-ir.md §2.3): `InvalidInput` with
   `context.tensor`.
2. The facade uploads each value to the model's device (the runtime never moves values, runtime.md
   §3). A rank outside 1–6 is `InvalidInput`; a dtype the device can't hold (`bf16` on Vulkan) is
   `UnsupportedOperation`. Values are uploaded before the binding check, so this happens before a
   dtype mismatch with the graph is detected.
3. The runtime binds and runs the graph (runtime.md §3, §4). Binding errors are `InvalidInput` with
   the input or state entry in `context.tensor` (or the node in `context.node`); errors while running
   carry the node path in `context.node`; a set token is `Cancelled` with the node it stopped before.
4. `RunOutput { outputs, state }`: every graph output in declaration order and every root-level state
   entry (expanded names, the order of `ModelInfo::state`), read back as `TensorData`. A failed call
   returns nothing.

State is passed in explicitly on every call; the model keeps none between calls (sessions are M3).

**Saving.** `model.save(graph_path, weights_path)` writes the graph's canonical JSON to
`graph_path` and every declared param to `weights_path` as one safetensors file (formats.md §1.6,
by param name, no metadata). Params a node uses are read back from the device in the compute dtype
they are held in (`F32`, `F16` or `BF16` storage): an `f16` model saves `F16` tensors. Declared
params that no node uses are not on the device (runtime.md §2 step 1); they are copied byte for byte,
in their stored dtype, from the weight file the model was loaded from, which is opened again for
this: if it is gone or no longer holds the tensor with the declared shape and a float storage, save
fails (`Io` or `InvalidInput` naming the tensor and the file). The weights file is written to a
temporary file first and renamed (formats.md §1.6), so an existing file is never half overwritten;
the graph file is written after it. Loading the saved pair with the same options gives a model whose
outputs are identical.

**Tensor files.** `tensor_file::read` opens a safetensors file (formats.md §1: whole header
validated, sizes checked before allocating) and returns every tensor in name order. Tensors must be
stored as `F32`, `F16`, `BF16`, `I64`, `I32` or `BOOL` (bytes `0`/`1`); any other dtype is
`InvalidInput` naming the tensor. `tensor_file::write` validates every value first, then writes
them with the safetensors writer (temporary file and rename; a duplicate name is `InvalidInput`).
File errors are `Io` (I/O) or `InvalidInput` (malformed, truncated) with `context.file`.

**Error mapping.** Runtime errors map through `RuntimeError::kind()` (runtime.md §5) to the
`ErrorKind` of the same name. `message` is `Could not load the model: …`, `The model run failed: …`
or `Could not save the model: …` followed by the runtime's description, which names the node,
tensor, file or device; `context` carries them as fields (`node`, `tensor`, `file`, `device`) and
`details` the runtime's error text. A cancelled call's message is `The operation was cancelled.`,
and a broken invariant's (`Internal`) asks to report a bug, with the runtime's text in `details`.

| Failure | Kind | Context |
|---|---|---|
| Graph document does not parse or fails validation | `InvalidGraph` | `node`, `pointer`, `details`, `file` |
| Device unknown, unavailable or not compiled in | `BackendUnavailable` | `device` |
| Compute dtype or a value dtype unsupported by the device | `UnsupportedOperation` | `device`, `node` |
| Missing, extra, wrong-shape or non-float weight; malformed or truncated weight file | `InvalidInput` | `tensor`, `file` |
| Unsupported float storage (`F64`, `F8_*`) | `UnsupportedModel` | `tensor` |
| Weight, graph or tensor file can't be read or written | `Io` | `file` |
| Malformed `TensorData`, zero-size dim, wrong input or state names, dtypes or shapes | `InvalidInput` | `tensor` (or `node`) |
| Out-of-range index while running | `InvalidInput` | `node` |
| Cancelled | `Cancelled` | `node` (if a node was next) |
| A model used after `close()` | `InvalidInput` | |
| Backend failure | `Backend` | `node` |

**`CancellationToken`.** There is one token type for the whole API: `polypous::CancellationToken`
(§1.1, `new`, `cancel`, `is_cancelled`, `Clone`, `Default`, `Debug`) works for probing, loading and
running. It holds the runtime's token, and clones share one flag.

**CLI.** `polyp graph validate <file> [--json] [--compute-dtype D]` prints every diagnostic (code,
node, pointer, message) or, with `--json`, the `ValidationReport` JSON. `polyp run <graph>
[--weights FILE] --input NAME=FILE … --output FILE [--device ID] [--compute-dtype D] [--json]`
loads the graph through `load_graph`, reads each input from a safetensors file holding exactly one
tensor (of any name), runs it once, and writes every output, by name, to `--output`. `--device`
takes the id syntax above (`polyp devices` prints it). Graphs with state are rejected in M1 (the
initial state's shapes depend on the bound symbols); use `Model::run`. Exit codes: `0` success (or
a valid graph), `1` an error (`polypous::Error`: message and context on stderr, the error JSON with
`--json`), `2` a usage error, `3` `graph validate` found the graph invalid.

C ABI (M7): `pp_graph_from_json`, `pp_graph_validate_json`, `pp_engine_load_graph`,
`pp_model_info_json`, `pp_model_run`, `pp_model_save`.

### 1.4 Training (M2)

Train a graph from a training config ([training.md](training.md)), with events, cancellation,
checkpoints, resume and export. The facade is a thin layer over `polypous-train`; behavior,
config, events and checkpoints are specified in training.md, and this section gives the Rust and
JSON surface.

```rust
use polypous::{CancellationToken, Engine, EngineConfig, Graph, ResumeOptions, TrainerOptions, TrainingConfig};

let engine = Engine::new(EngineConfig::default())?;
let graph = Graph::from_file("cnn.graph.json")?;
let config = TrainingConfig::from_file("train.json")?;           // InvalidInput with a JSON pointer
let mut options = TrainerOptions::default();
options.base_dir = Some(".".into());                             // relative paths in the config
options.weights = Some("pretrained.safetensors".into());        // optional; other params use their init
let cancel = CancellationToken::new();
let trainer = engine.trainer(&graph, config, options, &cancel)?; // validates, reads the data, initializes

let outcome = trainer.run(&cancel, |event| println!("{}", event.to_json()))?;  // started, step, …, finished
println!("{:?} after {} steps", outcome.stop_reason, outcome.step);
trainer.export("model.graph.json", "model.safetensors")?;       // loads with Engine::load_graph

// Later, or in another process: continue from the newest checkpoint of the run directory.
let trainer = engine.resume_trainer("runs/mnist", ResumeOptions::default(), &cancel)?;
let outcome = trainer.run(&cancel, |_| {})?;
```

**Items.**

| Item | Behavior |
|---|---|
| `TrainingConfig` | The training config (training.md §2). `from_json(&str)`, `from_file(path)` (at most 1 MiB, `Io` if it can't be read), `to_json()` (canonical: every field, defaults filled in). Errors are `InvalidInput` with `context.pointer` (`precision` `f16`/`bf16`: `UnsupportedOperation`). Plain data; its fields are public |
| `TrainerOptions` | What a training run needs besides the config: `weights`, `base_dir`, `datasets` (below). `Default`, `from_json`/`to_json` |
| `InMemoryDataset` | `{ name: String, fields: Vec<NamedTensor> }`: a `tensors` dataset (training.md §3.5.1) |
| `Engine::trainer(&Graph, TrainingConfig, TrainerOptions, &CancellationToken) -> Result<Trainer, Error>` | Checks the config against the graph and the data, reads the datasets, loads the initial weights, computes the initial values and uploads the params, in the order of training.md §2.10. Blocking. Nothing trains yet |
| `Engine::resume_trainer(path, ResumeOptions, &CancellationToken) -> Result<Trainer, Error>` | A trainer restored from a checkpoint directory, or from the newest checkpoint of a run directory (training.md §5.4). Blocking |
| `ResumeOptions` | `device` (device id string or null: overrides the config's `device`), `base_dir` (path or null: overrides the checkpoint's), `datasets` (as in `TrainerOptions`), `discard_newer` (bool, `false`). `Default`, `from_json`/`to_json` |
| `Trainer` | A training run. `Arc` handle: `Clone + Send + Sync + 'static`, no generics or lifetimes. Holds the param store, the optimizer state and the datasets |
| `trainer.info() -> TrainerInfo` | Plain data: the `started` event's fields (training.md §4.1) with `step` the steps completed so far, `finished` (bool), and `trainable` (the trainable params as `{"name", "shape"}`, in declaration order). `to_json()` |
| `trainer.run(&CancellationToken, on_event: impl FnMut(TrainingEvent)) -> Result<TrainingOutcome, Error>` | Trains from the trainer's current step to the end, or until cancelled (training.md §1.2, §4). Blocking. Returns `Ok` also when cancelled (`stop_reason: "cancelled"`); calling `run` again continues. `InvalidInput` if the trainer is running (another clone, or its own callback), has finished, or has failed |
| `trainer.export(graph_path, weights_path) -> Result<(), Error>` | Writes the current model, loadable by `Engine::load_graph` (training.md §5.5). `InvalidInput` while running or after a failed run |
| `trainer.close()` | Frees the device memory and the datasets now (§2 rule 2). A running `run` finishes first. Later `run` and `export` calls on any clone are `InvalidInput`; `info` still works. Idempotent |
| `TrainingEvent` | The events of training.md §4.1, a `#[non_exhaustive]` enum. `to_json()`; JSON `{"event": "started", …}` |
| `TrainingOutcome` | Plain data (below), the value `run` returns and the `finished` event carries. `to_json()` |

Paths are `impl AsRef<Path>` (§2 rule 10). `TrainingConfig`, `TrainerOptions`, `ResumeOptions`,
`TrainingEvent`, `TrainingOutcome` and `TrainerInfo` are serde types whose JSON is the documented
form; every function that takes one is equivalent to taking its JSON (§2 rule 3).

**`TrainerOptions` JSON.** An object, every field optional, unknown fields rejected:

| Field | Type, default | Meaning |
|---|---|---|
| `weights` | path string or null, null | Initial weights: a safetensors file or a sharded checkpoint's index (formats.md §1.5). It may hold a subset of the params; the others use their `init` (training.md §2.10 step 7). Null: every param uses its `init` |
| `base_dir` | path string or null, null | The directory relative paths in the config are resolved against. Null: the process's current directory |
| `datasets` | array, `[]` | `InMemoryDataset`s, `{"name": "…", "fields": [NamedTensor JSON, …]}` (§1.3), at most 64, names unique. Wrappers should pass tensors natively, not as JSON |

**`TrainingOutcome` JSON.**

```jsonc
{
  "stop_reason": "completed",            // or "cancelled"
  "step": 4220,                          // optimizer steps completed
  "epochs_completed": 5,
  "final_metrics": {"train.loss": 0.021, "train.accuracy": 0.993, "val.loss": 0.031, "val.accuracy": 0.990},
                                         // the last finished epoch's metrics; {} if none finished
  "best": {"metric": "val.accuracy", "value": 0.9912, "epoch": 3, "step": 3376,
           "checkpoint": "runs/mnist/step-000000003376"},   // null without checkpoint.keep_best
  "history": [ {"epoch": 0, "step": 844, "metrics": {…}}, … ],   // every finished epoch (training.md §5.2)
  "last_checkpoint": "runs/mnist/step-000000004220",          // or null
  "elapsed_seconds": 812.5               // wall-clock time of every run of this training (informative)
}
```

Numbers follow training.md §4.1 (non-finite values as `"nan"`, `"inf"`, `"-inf"`). Readers must
accept new fields.

**Errors** are those of training.md §6, with `message` `Could not create the trainer: …`,
`Could not resume training: …`, `Training failed: …` or `Could not export the model: …` followed by
the description, and `context.operation` `engine_trainer`, `engine_resume_trainer`, `trainer_run`
or `trainer_export`.

**Threading.** `run` blocks the calling thread and runs the steps on Polypous's threads; the
callback runs on the calling thread or a worker thread (§4). One `run` at a time per trainer.
Several trainers, and models, may run at the same time on one engine; they don't share state (§2
rule 9), only the device.

**CLI.** `polyp train`:

```text
polyp train <graph> --config <file> [--weights FILE] [--device ID] [--json]
            [--export-graph FILE --export-weights FILE]
polyp train --resume <dir> [--device ID] [--discard-newer] [--json]
            [--export-graph FILE --export-weights FILE]
```

- A new run loads `<graph>` (`Graph::from_file`) and `--config` (`TrainingConfig::from_file`), with
  `TrainerOptions::base_dir` the config file's directory and `weights` from `--weights`. `--device`
  replaces the config's `device`.
- `--resume` takes a checkpoint or run directory (`Engine::resume_trainer`; `--device` and
  `--discard-newer` set `ResumeOptions`). `<graph>`, `--config` and `--weights` with `--resume` are a
  usage error. A config with a `tensors` dataset can't be used from the CLI (`InvalidInput`).
- Output: one line per event on stdout (epoch, step, loss, learning rate, metrics, checkpoint
  paths). With `--json`, each event's JSON on its own line (JSON Lines), the `finished` event last.
- `--export-graph` and `--export-weights` (both or neither) call `Trainer::export` after the run,
  also after a cancelled one.
- The first Ctrl-C (SIGINT; Ctrl-C or Ctrl-Break on Windows) cancels the run (training.md §4.3:
  the current step finishes, and a checkpoint is written if configured); a second one ends the
  process at once with exit code 130, without writing a checkpoint. A Ctrl-C while the trainer is
  still being created makes `Engine::trainer` or `Engine::resume_trainer` return `Cancelled`: an
  error, exit code 1.
- Exit codes: `0` training completed, `1` an error (message and context on stderr, the error JSON
  with `--json`; also a cancel during creation), `2` a usage error, `4` cancelled (the outcome was
  still printed and the export written), `130` a second Ctrl-C.

C ABI (M7): `pp_engine_trainer`, `pp_engine_resume_trainer`, `pp_trainer_run`,
`pp_trainer_export`, `pp_trainer_info_json`, `pp_trainer_free`.

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
