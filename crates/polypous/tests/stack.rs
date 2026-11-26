//! Stack use of training and inference through the facade (issue #114): a training step and an
//! inference call run on a thread with a [`STACK`] stack, in a debug build like every
//! `cargo test`. The facade and `polyp` run on the caller's thread, and the main thread of a
//! Windows program has 1 MB.
//!
//! The `#[ignore = "gpu"]` test does the same on every available GPU device (lavapipe with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, which the CI lavapipe job sets).
//!
//! A stack overflow aborts the process, so these tests have their own test binary.
#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "common/training_fixtures.rs"]
mod training_fixtures;

use std::path::PathBuf;

use polypous::{
    CancellationToken, DeviceId, Engine, EngineConfig, ErrorKind, Graph, GraphLoadOptions,
    TrainerOptions, TrainingConfig,
};
use serde_json::{Value, json};
use training_fixtures::*;

/// The stack of the test thread. Measured smallest stacks on Linux, debug build with the `vulkan`
/// feature, CPU device (KiB, 4 KiB steps):
///
/// | graph | training step | inference |
/// |---|---|---|
/// | `mlp(4, 16, 1)`, before #114 | 992 | 964 |
/// | `mlp(4, 16, 1)` | 388 | 352 |
/// | 8 nested `repeat`s | 388 | 396 |
///
/// On lavapipe (the GPU test) the same graphs need up to 476 KiB. 640 KiB leaves room for other compilers and
/// platforms below Windows' 1 MB main thread, and fails clearly if a large frame comes back.
const STACK: usize = 640 << 10;

/// The deepest `repeat` nesting a graph may have (graph-ir.md §4.3, `repeat_too_deep`).
const MAX_REPEAT_DEPTH: usize = 8;

#[test]
fn training_and_inference_fit_a_640_kib_stack() {
    let too_deep = Graph::from_json(&nested_json(MAX_REPEAT_DEPTH + 1).to_string()).unwrap_err();
    assert_eq!(too_deep.kind, ErrorKind::InvalidGraph);
    assert!(
        too_deep
            .context
            .details
            .is_some_and(|d| d.contains("repeat_too_deep")),
        "{} levels must be the limit",
        MAX_REPEAT_DEPTH
    );

    on_small_stack(&cpu_engine(), DeviceId::Cpu);
}

/// [`training_and_inference_fit_a_640_kib_stack`] on every available GPU device. Software
/// rasterizers only with `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (the CI lavapipe job sets it). Passes
/// with a SKIP line if there is no GPU device.
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn training_and_inference_fit_a_640_kib_stack_on_gpus() {
    let mut config = EngineConfig::default();
    config.software_rasterizers =
        std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    config.probe_cache_dir =
        Some(PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("facade-stack-probe-cache"));
    let engine = Engine::new(config).unwrap();
    let devices: Vec<DeviceId> = engine
        .devices()
        .into_iter()
        .filter(|d| d.backend != polypous::BackendKind::Cpu)
        .filter(|d| {
            let available = d.status.is_available();
            if !available {
                eprintln!("SKIP device {}: {:?}", d.id, d.status);
            }
            available
        })
        .map(|d| d.id)
        .collect();
    if devices.is_empty() {
        eprintln!("SKIP: no available GPU device");
    }
    for device in devices {
        on_small_stack(&engine, device);
    }
}

/// [`train_and_infer`] for the MLP and the 8 nested `repeat`s on `device`, each on a thread with
/// a [`STACK`] stack.
fn on_small_stack(engine: &Engine, device: DeviceId) {
    let graphs = [
        ("mlp", mlp(4, 16, 1)),
        (
            "nested",
            Graph::from_json(&nested_json(MAX_REPEAT_DEPTH).to_string()).unwrap(),
        ),
    ];
    for (name, graph) in graphs {
        let label = format!("{device}-{name}").replace(':', "-");
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("facade-stack-{label}"));
        let _ = std::fs::remove_dir_all(&dir);
        write_regression(&dir, 64, 5);
        let engine = engine.clone();
        std::thread::Builder::new()
            .name(format!("stack-{label}"))
            .stack_size(STACK)
            .spawn(move || train_and_infer(&engine, &graph, device, dir))
            .unwrap()
            .join()
            .unwrap_or_else(|_| panic!("{device}, {name}: the thread panicked"));
        eprintln!(
            "{device}: {name} trained and ran on a {} KiB stack",
            STACK >> 10
        );
    }
}

/// `Engine::trainer` and `Trainer::run` (one epoch) on `device`, then the export with
/// `Engine::load_graph` and `Model::run` on `device`.
fn train_and_infer(engine: &Engine, graph: &Graph, device: DeviceId, dir: PathBuf) {
    let cancel = CancellationToken::new();
    let config = regression_config(json!({ "epochs": 1, "device": device.to_string() }));
    let config = TrainingConfig::from_json(&config.to_string()).unwrap();
    let mut options = TrainerOptions::default();
    options.base_dir = Some(dir.clone());
    let trainer = engine.trainer(graph, config, options, &cancel).unwrap();
    assert_eq!(trainer.info().device, device);
    let outcome = trainer.run(&cancel, |_| {}).unwrap();
    assert!(outcome.step > 0);
    let (graph_path, weights_path) = (dir.join("model.graph.json"), dir.join("model.safetensors"));
    trainer.export(&graph_path, &weights_path).unwrap();

    let mut options = GraphLoadOptions::default();
    options.weights = Some(weights_path);
    options.device = Some(device);
    let exported = Graph::from_file(&graph_path).unwrap();
    let model = engine.load_graph(&exported, options, &cancel).unwrap();
    let x = regression_fields(64, 5).remove(0);
    let out = model.run(vec![x], vec![], &cancel).unwrap();
    assert_eq!(out.output("y").unwrap().shape, vec![64, 1]);
}

/// An engine with only the CPU, from a probe report: no probe runs.
fn cpu_engine() -> Engine {
    let cpu = json!({
        "id": { "backend": "cpu" }, "backend": "cpu", "name": "cpu", "vendor": null,
        "driver": null, "kind": "cpu", "dedicated_memory": null, "free_memory": null,
        "unified_memory": true,
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": { "state": "available", "isolated": true }, "physical_gpu": null,
        "primary": false
    });
    let config = json!({ "probe_report": { "devices": [cpu], "isolated": true } });
    Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap()
}

/// `x [batch, 4]` → relu → `levels` nested `repeat`s around `h + linear(h)` → linear →
/// `y [batch, 1]`, for the regression set. Every `repeat` has count 1 but the innermost (2); level
/// `d` has the carry `h{d}` and the params prefix `l{d}.{i}.`.
fn nested_json(levels: usize) -> Value {
    let last = levels - 1;
    let count = |d: usize| if d == last { 2 } else { 1 };
    let kaiming = json!({ "kind": "kaiming_uniform", "a": 2.23606797749979 });
    // The full names of the innermost body's params.
    let mut prefixes = vec![String::new()];
    for d in 0..levels {
        prefixes = prefixes
            .iter()
            .flat_map(|p| (0..count(d)).map(move |i| format!("{p}l{d}.{i}.")))
            .collect();
    }
    let mut params = Vec::new();
    for p in &prefixes {
        params.push(
            json!({ "name": format!("{p}w"), "dtype": "float", "shape": [4, 4],
            "init": kaiming }),
        );
        params.push(
            json!({ "name": format!("{p}b"), "dtype": "float", "shape": [4],
            "init": { "kind": "zeros" } }),
        );
    }
    params.push(json!({ "name": "out.w", "dtype": "float", "shape": [1, 4], "init": kaiming }));
    params.push(json!({ "name": "out.b", "dtype": "float", "shape": [1],
        "init": { "kind": "zeros" } }));

    let h = format!("h{last}");
    let mut body = json!({
        "nodes": [
            { "id": "lin", "op": "nn.linear", "inputs": { "x": h },
              "params": { "weight": "w", "bias": "b" } },
            { "id": "res", "op": "core.add", "inputs": { "a": h, "b": "lin" } }
        ],
        "carry_out": { h: "res" }
    });
    for d in (0..levels).rev() {
        body["params_prefix"] = json!(format!("l{d}.{{i}}."));
        let init = if d == 0 {
            "pre".to_owned()
        } else {
            format!("h{}", d - 1)
        };
        let repeat = json!({ "id": format!("r{d}"), "op": "repeat", "count": count(d),
            "carry": { format!("h{d}"): init }, "body": body });
        body = if d == 0 {
            repeat
        } else {
            json!({ "nodes": [repeat], "carry_out": { format!("h{}", d - 1): format!("r{d}:h{d}") } })
        };
    }
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 4] } ],
        "outputs": [ { "name": "y", "value": "out" } ],
        "params": params,
        "nodes": [
            { "id": "pre", "op": "nn.relu", "inputs": { "x": "x" } },
            body,
            { "id": "out", "op": "nn.linear", "inputs": { "x": "r0:h0" },
              "params": { "weight": "out.w", "bias": "out.b" } }
        ]
    })
}
