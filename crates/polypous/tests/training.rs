//! Training through the public facade (public-api.md §1.4, issue #83), on the CPU; the
//! `#[ignore = "gpu"]` test trains on every GPU device (lavapipe with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1`) for a loss threshold only.
//!
//! The engine of the CPU tests gets its device list from `EngineConfig::probe_report`, so no
//! probe runs, and the device tests can list devices that are unavailable, unknown to this build
//! or a primary GPU that doesn't exist. Behavior is `polypous-train`'s and tested there; these
//! tests check the facade's part: the types and their JSON, the device resolver, the error
//! mapping, panics at the boundary and the handle.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "common/training_fixtures.rs"]
mod training_fixtures;

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use polypous::training::StopReason;
use polypous::{
    CancellationToken, DeviceId, Engine, EngineConfig, Error, ErrorKind, Graph, GraphLoadOptions,
    InMemoryDataset, NamedTensor, ResumeOptions, Trainer, TrainerInfo, TrainerOptions,
    TrainingConfig, TrainingEvent, TrainingOutcome, tensor_file,
};
use serde_json::{Value, json};
use training_fixtures::*;

// ---- helpers ----

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("facade-training-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn device_json(id: Value, backend: &str, kind: &str, status: Value) -> Value {
    json!({
        "id": id, "backend": backend, "name": format!("test {backend}"), "vendor": null,
        "driver": null, "kind": kind, "dedicated_memory": null, "free_memory": null,
        "unified_memory": kind != "discrete",
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": status, "physical_gpu": null, "primary": false
    })
}

fn available() -> Value {
    json!({"state": "available", "isolated": true})
}

fn cpu_device() -> Value {
    device_json(json!({"backend": "cpu"}), "cpu", "cpu", available())
}

/// A skipped lavapipe: listed, unavailable.
fn skipped_lavapipe() -> Value {
    device_json(
        json!({"backend": "vulkan", "adapter": "cpu", "index": 0}),
        "vulkan",
        "cpu",
        json!({"state": "unavailable", "reason": {"code": "software_rasterizer_skipped"}, "isolated": true}),
    )
}

fn engine_with(devices: Vec<Value>) -> Engine {
    let config = json!({ "probe_report": { "devices": devices, "isolated": true } });
    Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap()
}

/// An available CPU and a skipped lavapipe; no GPU.
fn engine() -> Engine {
    engine_with(vec![skipped_lavapipe(), cpu_device()])
}

fn config(doc: &Value) -> TrainingConfig {
    TrainingConfig::from_json(&doc.to_string()).unwrap_or_else(|e| panic!("{e:?}"))
}

fn options_in(base: &Path) -> TrainerOptions {
    let mut options = TrainerOptions::default();
    options.base_dir = Some(base.to_path_buf());
    options
}

fn regression_trainer(engine: &Engine, base: &Path, extra: Value) -> Trainer {
    write_regression(base, 128, 5);
    engine
        .trainer(
            &mlp(4, 16, 1),
            config(&regression_config(extra)),
            options_in(base),
            &CancellationToken::new(),
        )
        .unwrap_or_else(|e| panic!("{e:?}"))
}

fn run_all(trainer: &Trainer) -> (TrainingOutcome, Vec<TrainingEvent>) {
    let mut events = Vec::new();
    let outcome = trainer
        .run(&CancellationToken::new(), |e| events.push(e))
        .unwrap_or_else(|e| panic!("{e:?}"));
    (outcome, events)
}

fn kind_of(event: &TrainingEvent) -> String {
    let json: Value = serde_json::from_str(&event.to_json()).unwrap();
    json["event"].as_str().unwrap().to_string()
}

#[track_caller]
fn assert_error(error: &Error, kind: ErrorKind, operation: &str, prefix: &str) {
    assert_eq!(error.kind, kind, "{error:?}");
    assert_eq!(
        error.context.operation.as_deref(),
        Some(operation),
        "{error:?}"
    );
    assert!(error.message.starts_with(prefix), "{error:?}");
    assert!(error.message.ends_with('.'), "{error:?}");
}

fn create_error(engine: &Engine, graph: &Graph, doc: &Value, base: &Path) -> Error {
    engine
        .trainer(
            graph,
            config(doc),
            options_in(base),
            &CancellationToken::new(),
        )
        .unwrap_err()
}

/// Every param of an exported weight file, by name.
fn exported(dir: &Path, name: &str, trainer: &Trainer) -> Vec<NamedTensor> {
    let graph = dir.join(format!("{name}.graph.json"));
    let weights = dir.join(format!("{name}.safetensors"));
    trainer.export(&graph, &weights).unwrap();
    tensor_file::read(&weights).unwrap()
}

// ---- training, events ----

#[test]
fn a_small_mlp_trains_from_a_safetensors_file() {
    let dir = dir("learns");
    let trainer = regression_trainer(&engine(), &dir, json!({}));
    let info = trainer.info();
    assert_eq!(info.device, DeviceId::Cpu);
    assert_eq!((info.train_samples, info.steps_per_epoch), (128, 8));
    assert_eq!(info.total_steps, 240);
    assert_eq!((info.step, info.finished), (0, false));
    assert_eq!(info.trainable_elements, 16 * 4 + 16 + 16 + 1);

    let (outcome, events) = run_all(&trainer);
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert_eq!((outcome.step, outcome.epochs_completed), (240, 30));
    let loss = outcome.final_metrics.get("train.loss").unwrap();
    eprintln!("cpu: final train.loss {loss}");
    assert!(loss < 0.01, "{loss}");
    assert!(trainer.info().finished);
    assert_eq!(trainer.info().step, 240);

    // Events: `started` first, `finished` last with the outcome; every kind's JSON has "event".
    assert_eq!(kind_of(&events[0]), "started");
    let TrainingEvent::Started { device, step, .. } = &events[0] else {
        panic!("{:?}", events[0]);
    };
    assert_eq!((*device, *step), (DeviceId::Cpu, 0));
    let started: Value = serde_json::from_str(&events[0].to_json()).unwrap();
    assert_eq!(
        started["device"],
        serde_json::to_value(DeviceId::Cpu).unwrap()
    );
    assert_eq!(started["device"], json!({"backend": "cpu"}));
    match events.last().unwrap() {
        TrainingEvent::Finished { outcome: last } => assert_eq!(**last, outcome),
        other => panic!("{other:?}"),
    }
    let mut kinds: Vec<String> = events.iter().map(kind_of).collect();
    kinds.dedup();
    for kind in ["epoch_started", "step", "epoch_finished"] {
        assert!(kinds.iter().any(|k| k == kind), "{kinds:?}");
    }
    assert_eq!(
        events.iter().filter(|e| kind_of(e) == "finished").count(),
        1
    );
    let finished: Value = serde_json::from_str(&events.last().unwrap().to_json()).unwrap();
    let outcome_json: Value = serde_json::from_str(&outcome.to_json()).unwrap();
    assert_eq!(finished["outcome"], outcome_json);
    assert_eq!(outcome_json["stop_reason"], "completed");
    assert_eq!(outcome_json["step"], 240);
    assert!(outcome_json["elapsed_seconds"].is_number());
    assert_eq!(outcome_json["best"], Value::Null);
    assert_eq!(outcome_json["history"].as_array().unwrap().len(), 30);
}

#[test]
fn an_in_memory_dataset_trains() {
    let dir = dir("in-memory");
    let mut options = options_in(&dir);
    options.datasets = vec![blobs("train", 240, 1), blobs("val", 60, 2)];
    let trainer = engine()
        .trainer(
            &mlp(4, 16, 3),
            config(&classification_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap();
    assert_eq!(trainer.info().validation_samples, 60);
    let (outcome, events) = run_all(&trainer);
    let last = &outcome.final_metrics;
    let names: Vec<&str> = last.0.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        ["train.loss", "train.accuracy", "val.loss", "val.accuracy"]
    );
    let loss = last.get("train.loss").unwrap();
    assert!(loss < 0.1, "{loss}");
    assert!(last.get("val.accuracy").unwrap() >= 0.95, "{last:?}");
    assert!(events.iter().any(|e| kind_of(e) == "validation_started"));
}

#[test]
fn malformed_in_memory_data_is_invalid_input_at_the_data_step() {
    let dir = dir("malformed-data");
    let mut bad = blobs("train", 30, 1);
    bad.fields[0].tensor.bytes.pop();
    let mut options = options_in(&dir);
    options.datasets = vec![bad, blobs("val", 9, 2)];
    let error = engine()
        .trainer(
            &mlp(4, 16, 3),
            config(&classification_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "engine_trainer",
        "Could not create the trainer: ",
    );
    assert_eq!(error.context.tensor.as_deref(), Some("x"), "{error:?}");
    assert!(error.message.contains("malformed"), "{error:?}");

    // The config is checked first (training.md §2.10 step 1), then the data (step 5).
    let mut bad = blobs("train", 30, 1);
    bad.fields[0].tensor.bytes.pop();
    let mut options = options_in(&dir);
    options.datasets = vec![bad];
    let mut c = config(&classification_config(json!({})));
    c.epochs = 0;
    let error = engine()
        .trainer(&mlp(4, 16, 3), c, options, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(
        error.context.pointer.as_deref(),
        Some("/epochs"),
        "{error:?}"
    );
}

// ---- cancellation, resume, export ----

#[test]
fn cancelling_from_the_callback_and_running_again() {
    let dir = dir("cancel");
    let engine = engine();
    let reference = regression_trainer(&engine, &dir.join("a"), json!({"epochs": 4}));
    run_all(&reference);

    let trainer = regression_trainer(&engine, &dir.join("b"), json!({"epochs": 4}));
    let cancel = CancellationToken::new();
    let mut events = Vec::new();
    let outcome = trainer
        .run(&cancel, |event| {
            if matches!(event, TrainingEvent::Step { .. }) {
                cancel.cancel();
            }
            events.push(event);
        })
        .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Cancelled);
    assert_eq!(outcome.step, 1);
    assert_eq!(trainer.info().step, 1);
    assert!(!trainer.info().finished);
    match events.last().unwrap() {
        TrainingEvent::Finished { outcome: last } => {
            assert_eq!(last.stop_reason, StopReason::Cancelled);
        }
        other => panic!("{other:?}"),
    }
    let json: Value = serde_json::from_str(&outcome.to_json()).unwrap();
    assert_eq!(json["stop_reason"], "cancelled");

    // A cancelled trainer exports, then continues to the same params as the uninterrupted run.
    exported(&dir, "mid", &trainer);
    let (outcome, events) = run_all(&trainer);
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert_eq!(outcome.step, 32);
    let TrainingEvent::Started { step, .. } = &events[0] else {
        panic!("{:?}", events[0]);
    };
    assert_eq!(*step, 1);
    assert_eq!(
        exported(&dir, "continued", &trainer),
        exported(&dir, "reference", &reference)
    );
}

#[test]
fn resume_reaches_the_params_of_an_uninterrupted_run() {
    let dir = dir("resume");
    let extra = json!({
        "epochs": 3,
        "data": { "train": { "kind": "safetensors", "path": "train.safetensors" },
                  "validation": { "kind": "split", "fraction": 0.25 } },
        "checkpoint": { "dir": "run", "every_steps": 5, "keep_last": null }
    });
    let reference = regression_trainer(&engine(), &dir.join("a"), extra.clone());
    let (expected, _) = run_all(&reference);

    // Cancelled mid-epoch after step 0 (the first `step` event is never throttled), which writes
    // the cancel checkpoint `step-1`; then resumed by a new engine.
    let base = dir.join("b");
    let trainer = regression_trainer(&engine(), &base, extra);
    let cancel = CancellationToken::new();
    let outcome = trainer
        .run(&cancel, |event| {
            if let TrainingEvent::Step { step: 0, .. } = event {
                cancel.cancel();
            }
        })
        .unwrap();
    assert_eq!(
        (outcome.stop_reason, outcome.step),
        (StopReason::Cancelled, 1)
    );
    let checkpoint = outcome.last_checkpoint.clone().unwrap();
    assert!(checkpoint.ends_with("step-000000000001"), "{checkpoint}");
    drop(trainer);

    let resumed = engine()
        .resume_trainer(
            base.join("run"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(resumed.info().step, 1);
    let (outcome, events) = run_all(&resumed);
    let TrainingEvent::Started { step, .. } = &events[0] else {
        panic!("{:?}", events[0]);
    };
    assert_eq!(*step, 1);
    assert_eq!(outcome.step, expected.step);
    assert_eq!(outcome.history, expected.history);
    assert_eq!(
        exported(&dir, "resumed", &resumed),
        exported(&dir, "reference", &reference)
    );

    // The final checkpoints' weights are the same bytes.
    let last = |o: &TrainingOutcome| {
        std::fs::read(Path::new(o.last_checkpoint.as_ref().unwrap()).join("weights.safetensors"))
            .unwrap()
    };
    assert_eq!(last(&outcome), last(&expected));

    // Newer checkpoints: refused, then discarded with `discard_newer`.
    let error = engine()
        .resume_trainer(
            base.join("run").join("step-000000000001"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "engine_resume_trainer",
        "Could not resume training: ",
    );
    assert!(error.context.file.is_some(), "{error:?}");
    let mut options = ResumeOptions::default();
    options.discard_newer = true;
    options.device = Some("cpu".into());
    let again = engine()
        .resume_trainer(
            base.join("run").join("step-000000000001"),
            options,
            &CancellationToken::new(),
        )
        .unwrap();
    let (outcome, _) = run_all(&again);
    assert_eq!(outcome.history, expected.history);
}

#[test]
fn export_loads_with_load_graph_and_runs() {
    let dir = dir("export");
    let engine = engine();
    let trainer = regression_trainer(&engine, &dir, json!({ "checkpoint": { "dir": "run" } }));
    let (outcome, _) = run_all(&trainer);
    let graph_path = dir.join("model.graph.json");
    let weights_path = dir.join("model.safetensors");
    trainer.export(&graph_path, &weights_path).unwrap();

    // The export is the trainer's state: the same bytes as the final checkpoint's weights, and
    // the graph unchanged.
    let checkpoint = PathBuf::from(outcome.last_checkpoint.as_ref().unwrap());
    assert_eq!(
        std::fs::read(&weights_path).unwrap(),
        std::fs::read(checkpoint.join("weights.safetensors")).unwrap()
    );
    let graph = Graph::from_file(&graph_path).unwrap();
    assert_eq!(graph, mlp(4, 16, 1));

    let load = |weights: PathBuf| {
        let mut options = GraphLoadOptions::default();
        options.weights = Some(weights);
        options.device = Some(DeviceId::Cpu);
        engine
            .load_graph(&graph, options, &CancellationToken::new())
            .unwrap()
    };
    let model = load(weights_path);
    let from_checkpoint = load(checkpoint.join("weights.safetensors"));
    let x = regression_fields(128, 5).remove(0);
    let out = model
        .run(vec![x.clone()], vec![], &CancellationToken::new())
        .unwrap();
    let reference = from_checkpoint
        .run(vec![x], vec![], &CancellationToken::new())
        .unwrap();
    assert_eq!(out, reference);

    // A sanity check of what was exported (not bit-exact: the trainer sums the loss in its own
    // order, and `train.loss` averages the epoch's batches while the params change): the exported
    // model fits the data as well as the last epoch's `train.loss` says.
    let predictions = read_f32(out.output("y").unwrap());
    let targets = read_f32(&regression_fields(128, 5)[1].tensor);
    let mse: f64 = predictions
        .iter()
        .zip(&targets)
        .map(|(p, t)| f64::from(p - t).powi(2))
        .sum::<f64>()
        / 128.0;
    let loss = outcome.final_metrics.get("train.loss").unwrap();
    eprintln!("exported model MSE {mse}, last train.loss {loss}");
    assert!(mse < 0.01 && loss < 0.01, "{mse} {loss}");
}

// ---- devices ----

#[test]
fn auto_is_the_primary_gpu_else_the_cpu() {
    let dir = dir("auto");
    // No GPU: the CPU.
    let trainer = regression_trainer(
        &engine(),
        &dir.join("a"),
        json!({"device": "auto", "epochs": 1}),
    );
    assert_eq!(trainer.info().device, DeviceId::Cpu);
    let (_, events) = run_all(&trainer);
    assert!(matches!(
        events[0],
        TrainingEvent::Started {
            device: DeviceId::Cpu,
            ..
        }
    ));

    // A primary GPU that isn't there (cuda:7): `auto` picks it, as `load_graph` does, and the
    // plan can't be compiled for it.
    let cuda = device_json(
        json!({"backend": "cuda", "index": 7}),
        "cuda",
        "discrete",
        available(),
    );
    let engine = engine_with(vec![cuda, cpu_device()]);
    assert_eq!(
        engine.primary_gpu().map(|d| d.id),
        Some(DeviceId::Cuda { index: 7 })
    );
    let base = dir.join("b");
    write_regression(&base, 128, 5);
    let error = create_error(
        &engine,
        &mlp(4, 16, 1),
        &regression_config(json!({"device": "auto"})),
        &base,
    );
    assert_error(
        &error,
        ErrorKind::BackendUnavailable,
        "engine_trainer",
        "Could not create the trainer: ",
    );
    assert_eq!(error.context.device.as_deref(), Some("cuda:7"), "{error:?}");
    // `load_graph` picks the same device.
    let load = engine
        .load_graph(
            &mlp(4, 16, 1),
            GraphLoadOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_eq!(load.context.device.as_deref(), Some("cuda:7"));
}

#[test]
fn unknown_and_unavailable_devices_after_config_and_graph_errors() {
    let dir = dir("devices");
    write_regression(&dir, 128, 5);
    let engine = engine();
    let graph = mlp(4, 16, 1);
    for (device, why) in [("cuda:3", "no such device"), ("vulkan:cpu:0", "software")] {
        let error = create_error(
            &engine,
            &graph,
            &regression_config(json!({"device": device})),
            &dir,
        );
        assert_error(
            &error,
            ErrorKind::BackendUnavailable,
            "engine_trainer",
            "Could not create the trainer: device ",
        );
        assert_eq!(error.context.device.as_deref(), Some(device), "{error:?}");
        assert!(error.message.contains(why), "{error:?}");

        // Step 1 (config) and step 2 (graph) come first.
        let mut c = config(&regression_config(json!({"device": device})));
        c.batch_size = 0;
        let error = engine
            .trainer(&graph, c, options_in(&dir), &CancellationToken::new())
            .unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{error:?}");
        assert_eq!(error.context.pointer.as_deref(), Some("/batch_size"));
        let state = Graph::from_json(
            &json!({
                "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
                "inputs": [ { "name": "x", "dtype": "float", "shape": [2] } ],
                "state": [ { "name": "acc", "dtype": "float", "shape": [2], "init": "zeros" } ],
                "outputs": [ { "name": "y", "value": "sum" } ],
                "nodes": [ { "id": "sum", "op": "core.add", "inputs": { "a": "acc", "b": "x" } } ],
                "state_updates": { "acc": "sum" }
            })
            .to_string(),
        )
        .unwrap();
        let error = create_error(
            &engine,
            &state,
            &regression_config(json!({"device": device})),
            &dir,
        );
        assert_eq!(error.kind, ErrorKind::UnsupportedOperation, "{error:?}");
        assert_eq!(error.context.tensor.as_deref(), Some("acc"));
        // Step 3 (device) comes before step 4 (references).
        let error = create_error(
            &engine,
            &graph,
            &regression_config(json!({"device": device, "inputs": {"z": "x"}})),
            &dir,
        );
        assert_eq!(error.kind, ErrorKind::BackendUnavailable, "{error:?}");
    }

    // On resume, `ResumeOptions::device`, also text that is no device id.
    let base = dir.join("resume");
    let trainer = regression_trainer(
        &engine,
        &base,
        json!({"epochs": 1, "checkpoint": {"dir": "run"}}),
    );
    run_all(&trainer);
    for (device, context) in [
        ("gpu0", "gpu0"),
        ("cuda:3", "cuda:3"),
        ("vulkan:cpu:0", "vulkan:cpu:0"),
    ] {
        let mut options = ResumeOptions::default();
        options.device = Some(device.into());
        let error = engine
            .resume_trainer(base.join("run"), options, &CancellationToken::new())
            .unwrap_err();
        assert_error(
            &error,
            ErrorKind::BackendUnavailable,
            "engine_resume_trainer",
            "Could not resume training: ",
        );
        assert_eq!(error.context.device.as_deref(), Some(context), "{error:?}");
    }
}

// ---- errors ----

#[test]
fn config_parse_errors_carry_the_pointer() {
    let doc =
        regression_config(json!({"optimizer": {"kind": "adam", "lr": 0.01, "betas": [0.9, 1.5]}}));
    let error = TrainingConfig::from_json(&doc.to_string()).unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "training_config_from_json",
        "The training config is not valid: /optimizer/betas/1",
    );
    assert_eq!(error.context.pointer.as_deref(), Some("/optimizer/betas/1"));

    for (extra, pointer) in [
        (json!({"epochs": 5.0}), "/epochs"),
        (json!({"epocs": 5}), "/epocs"),
        (json!({"epochs": null}), "/epochs"),
        (
            json!({"data": {"train": {"kind": "csv", "path": "a.csv"}}}),
            "/data/train/columns",
        ),
    ] {
        let doc = regression_config(extra);
        let error = TrainingConfig::from_json(&doc.to_string()).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{error:?}");
        assert_eq!(error.context.pointer.as_deref(), Some(pointer), "{error:?}");
    }
    for text in ["", "[]", "{", "null"] {
        let error = TrainingConfig::from_json(text).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{text}");
        assert_eq!(error.context.pointer.as_deref(), Some(""), "{error:?}");
    }

    let doc = regression_config(json!({"precision": "f16"}));
    let error = TrainingConfig::from_json(&doc.to_string()).unwrap_err();
    assert_error(
        &error,
        ErrorKind::UnsupportedOperation,
        "training_config_from_json",
        "The training config is not supported: /precision",
    );
    assert_eq!(error.context.pointer.as_deref(), Some("/precision"));

    // From a file: `context.file`; an unreadable file is `Io`; more than 1 MiB is refused.
    let dir = dir("config-file");
    let path = dir.join("train.json");
    std::fs::write(&path, regression_config(json!({"seed": -1})).to_string()).unwrap();
    let error = TrainingConfig::from_file(&path).unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "training_config_from_file",
        "The training config is not valid: /seed",
    );
    assert_eq!(error.context.file.as_deref(), Some(path.to_str().unwrap()));
    let missing = dir.join("missing.json");
    let error = TrainingConfig::from_file(&missing).unwrap_err();
    assert_error(
        &error,
        ErrorKind::Io,
        "training_config_from_file",
        "Could not open the config",
    );
    assert_eq!(
        error.context.file.as_deref(),
        Some(missing.to_str().unwrap())
    );
    std::fs::write(&path, vec![b' '; (1 << 20) + 1]).unwrap();
    let error = TrainingConfig::from_file(&path).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidInput);
    assert_eq!(error.context.pointer.as_deref(), Some(""));
    let good = regression_config(json!({}));
    std::fs::write(&path, good.to_string()).unwrap();
    assert_eq!(TrainingConfig::from_file(&path).unwrap(), config(&good));
}

#[test]
fn creation_errors_map_with_their_context() {
    let dir = dir("create-errors");
    write_regression(&dir, 128, 5);
    let engine = engine();
    let graph = mlp(4, 16, 1);
    let prefix = "Could not create the trainer: ";

    // §2.10 step 4: references, with pointer and tensor.
    let error = create_error(
        &engine,
        &graph,
        &regression_config(json!({"inputs": {"z": "x"}})),
        &dir,
    );
    assert_error(&error, ErrorKind::InvalidInput, "engine_trainer", prefix);
    assert!(
        error
            .context
            .pointer
            .as_deref()
            .unwrap()
            .starts_with("/inputs"),
        "{error:?}"
    );
    assert!(error.context.tensor.is_some(), "{error:?}");

    // Step 5: a missing dataset file is `Io` with the file; a malformed one `InvalidInput`.
    let error = create_error(
        &engine,
        &graph,
        &regression_config(
            json!({"data": {"train": {"kind": "safetensors", "path": "nope.safetensors"}}}),
        ),
        &dir,
    );
    assert_error(&error, ErrorKind::Io, "engine_trainer", prefix);
    assert!(
        error
            .context
            .file
            .as_deref()
            .unwrap()
            .ends_with("nope.safetensors"),
        "{error:?}"
    );
    std::fs::write(dir.join("bad.safetensors"), b"not a safetensors file").unwrap();
    let error = create_error(
        &engine,
        &graph,
        &regression_config(
            json!({"data": {"train": {"kind": "safetensors", "path": "bad.safetensors"}}}),
        ),
        &dir,
    );
    assert_error(&error, ErrorKind::InvalidInput, "engine_trainer", prefix);
    assert!(
        error
            .context
            .file
            .as_deref()
            .unwrap()
            .ends_with("bad.safetensors"),
        "{error:?}"
    );

    // A field that doesn't exist: pointer and tensor.
    let error = create_error(
        &engine,
        &graph,
        &regression_config(json!({"inputs": {"x": "nope"}})),
        &dir,
    );
    assert_error(&error, ErrorKind::InvalidInput, "engine_trainer", prefix);
    assert_eq!(
        error.context.pointer.as_deref(),
        Some("/inputs/x"),
        "{error:?}"
    );

    // Step 7: params with neither init nor weights, every one of them in details.
    let mut doc = mlp_json(4, 16, 1);
    for p in doc["params"].as_array_mut().unwrap() {
        p.as_object_mut().unwrap().remove("init");
    }
    let no_init = Graph::from_json(&doc.to_string()).unwrap();
    let error = create_error(&engine, &no_init, &regression_config(json!({})), &dir);
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "engine_trainer",
        "Could not create the trainer: These params have no init and no weights: ",
    );
    assert_eq!(error.context.tensor.as_deref(), Some("fc1.weight"));
    assert_eq!(
        error.context.details.as_deref(),
        Some("fc1.weight\nfc1.bias\nfc2.weight\nfc2.bias")
    );
    // ... but initial weights for all of them work; `weights` is used as given.
    let source = regression_trainer(&engine, &dir.join("w"), json!({"epochs": 1}));
    let weights = dir.join("w").join("init.safetensors");
    source
        .export(dir.join("w").join("init.graph.json"), &weights)
        .unwrap();
    let mut options = options_in(&dir);
    options.weights = Some(weights.clone());
    engine
        .trainer(
            &no_init,
            config(&regression_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap();
    let mut options = options_in(&dir);
    options.weights = Some(dir.join("missing.safetensors"));
    let error = engine
        .trainer(
            &graph,
            config(&regression_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(&error, ErrorKind::Io, "engine_trainer", prefix);
    assert!(
        error
            .context
            .file
            .as_deref()
            .unwrap()
            .ends_with("missing.safetensors"),
        "{error:?}"
    );

    // Step 8: a run directory that holds checkpoints.
    let with_run = json!({"epochs": 1, "checkpoint": {"dir": "run"}});
    run_all(&regression_trainer(
        &engine,
        &dir.join("r"),
        with_run.clone(),
    ));
    let error = create_error(
        &engine,
        &graph,
        &regression_config(with_run),
        &dir.join("r"),
    );
    assert_error(&error, ErrorKind::InvalidInput, "engine_trainer", prefix);
    assert!(error.context.file.is_some(), "{error:?}");

    // Cancelled while created.
    let cancel = CancellationToken::new();
    cancel.cancel();
    let error = engine
        .trainer(
            &graph,
            config(&regression_config(json!({}))),
            options_in(&dir),
            &cancel,
        )
        .unwrap_err();
    assert_error(&error, ErrorKind::Cancelled, "engine_trainer", prefix);

    // The datasets of the options: at most 64, names unique.
    let mut options = options_in(&dir);
    options.datasets = (0..65).map(|i| blobs(&format!("d{i}"), 3, 1)).collect();
    let error = engine
        .trainer(
            &graph,
            config(&regression_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(&error, ErrorKind::InvalidInput, "engine_trainer", prefix);
    assert!(error.message.contains("at most 64"), "{error:?}");
    let mut options = options_in(&dir);
    options.datasets = vec![blobs("d", 3, 1), blobs("d", 3, 2)];
    let error = engine
        .trainer(
            &graph,
            config(&regression_config(json!({}))),
            options,
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert!(error.message.contains("named `d`"), "{error:?}");
}

#[test]
fn resume_and_export_errors_map_with_their_context() {
    let dir = dir("resume-errors");
    let engine = engine();
    std::fs::create_dir_all(dir.join("empty")).unwrap();
    let error = engine
        .resume_trainer(
            dir.join("empty"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "engine_resume_trainer",
        "Could not resume training: ",
    );
    assert!(error.context.file.is_some(), "{error:?}");
    let error = engine
        .resume_trainer(
            dir.join("missing"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_eq!(
        error.context.operation.as_deref(),
        Some("engine_resume_trainer")
    );
    assert!(
        matches!(error.kind, ErrorKind::Io | ErrorKind::InvalidInput),
        "{error:?}"
    );

    // A checkpoint whose `state.json` is broken: InvalidInput with the file.
    let base = dir.join("b");
    run_all(&regression_trainer(
        &engine,
        &base,
        json!({"epochs": 1, "checkpoint": {"dir": "run"}}),
    ));
    let state = base
        .join("run")
        .join("step-000000000008")
        .join("state.json");
    let text = std::fs::read_to_string(&state).unwrap();
    std::fs::write(
        &state,
        text.replace("\"format_version\":1", "\"format_version\":2"),
    )
    .unwrap();
    let error = engine
        .resume_trainer(
            base.join("run"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "engine_resume_trainer",
        "Could not resume training: ",
    );
    assert!(
        error
            .context
            .file
            .as_deref()
            .unwrap()
            .ends_with("state.json"),
        "{error:?}"
    );

    // Export to a directory that doesn't exist: `Io` with the file.
    let trainer = regression_trainer(&engine, &dir.join("e"), json!({"epochs": 1}));
    let target = dir.join("no-such-dir").join("m.safetensors");
    let error = trainer
        .export(dir.join("m.graph.json"), &target)
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::Io,
        "trainer_export",
        "Could not export the model: ",
    );
    assert!(error.context.file.is_some(), "{error:?}");
}

// ---- the handle and its states ----

#[test]
fn the_state_machine_through_the_facade() {
    let dir = dir("states");
    let engine = engine();
    let trainer = regression_trainer(&engine, &dir.join("a"), json!({"epochs": 2}));
    let clone = trainer.clone();
    let inner: Arc<Mutex<Vec<Error>>> = Arc::default();
    let errors = Arc::clone(&inner);
    trainer
        .run(&CancellationToken::new(), move |event| {
            if matches!(event, TrainingEvent::Started { .. }) {
                let run = clone.run(&CancellationToken::new(), |_| {}).unwrap_err();
                let export = clone
                    .export(dir.join("a/x.graph.json"), dir.join("a/x.safetensors"))
                    .unwrap_err();
                assert_eq!(clone.info().step, 0);
                errors.lock().unwrap().extend([run, export]);
            }
        })
        .unwrap();
    let errors = inner.lock().unwrap();
    assert_error(
        &errors[0],
        ErrorKind::InvalidInput,
        "trainer_run",
        "Training failed: ",
    );
    assert!(errors[0].message.contains("running"), "{:?}", errors[0]);
    assert_error(
        &errors[1],
        ErrorKind::InvalidInput,
        "trainer_export",
        "Could not export the model: ",
    );

    // Finished: run is refused, export works.
    let error = trainer.run(&CancellationToken::new(), |_| {}).unwrap_err();
    assert_error(
        &error,
        ErrorKind::InvalidInput,
        "trainer_run",
        "Training failed: ",
    );
    assert!(error.message.contains("finished"), "{error:?}");
    let base = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("facade-training-states");
    trainer
        .export(base.join("f.graph.json"), base.join("f.safetensors"))
        .unwrap();

    // Closed: run and export refused on every clone, info works, close is idempotent.
    let clone = trainer.clone();
    trainer.close();
    trainer.close();
    for t in [&trainer, &clone] {
        let error = t.run(&CancellationToken::new(), |_| {}).unwrap_err();
        assert!(error.message.contains("closed"), "{error:?}");
        let error = t
            .export(base.join("g.graph.json"), base.join("g.safetensors"))
            .unwrap_err();
        assert_error(
            &error,
            ErrorKind::InvalidInput,
            "trainer_export",
            "Could not export the model: ",
        );
        assert_eq!(t.info().step, 16);
        assert!(t.info().finished);
    }

    // `close` from the callback takes effect when the run returns.
    let trainer = regression_trainer(&engine, &base.join("b"), json!({"epochs": 2}));
    let clone = trainer.clone();
    let outcome = trainer
        .run(&CancellationToken::new(), |event| {
            if matches!(event, TrainingEvent::Started { .. }) {
                clone.close();
            }
        })
        .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    let error = trainer.run(&CancellationToken::new(), |_| {}).unwrap_err();
    assert!(error.message.contains("closed"), "{error:?}");
}

#[test]
fn a_panicking_callback_is_internal_and_fails_only_its_trainer() {
    let dir = dir("panic");
    let engine = engine();
    let trainer = regression_trainer(&engine, &dir.join("a"), json!({"epochs": 2}));
    let other = regression_trainer(&engine, &dir.join("b"), json!({"epochs": 2}));
    let error = trainer
        .run(&CancellationToken::new(), |event| {
            if matches!(event, TrainingEvent::Step { .. }) {
                panic!("the app's callback broke");
            }
        })
        .unwrap_err();
    assert_error(
        &error,
        ErrorKind::Internal,
        "trainer_run",
        "Training failed: ",
    );
    assert_eq!(
        error.context.details.as_deref(),
        Some("panic: the app's callback broke")
    );
    // The trainer is failed: run and export refused; info and close still work.
    let again = trainer.run(&CancellationToken::new(), |_| {}).unwrap_err();
    assert_eq!(again.kind, ErrorKind::InvalidInput);
    assert!(again.message.contains("failed"), "{again:?}");
    let export = trainer
        .export(dir.join("x.graph.json"), dir.join("x.safetensors"))
        .unwrap_err();
    assert_eq!(export.kind, ErrorKind::InvalidInput);
    assert_eq!(trainer.info().step, 1);
    trainer.close();
    trainer.close();

    // Another trainer on the same engine, and a new one, are unaffected.
    let (outcome, _) = run_all(&other);
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    let third = regression_trainer(&engine, &dir.join("c"), json!({"epochs": 1}));
    assert_eq!(run_all(&third).0.stop_reason, StopReason::Completed);
}

#[test]
fn trainers_and_engines_share_no_state() {
    // Two engines and two trainers, run on two threads at once, give the same params.
    let dir = dir("threads");
    let handles: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|name| {
            let base = dir.join(name);
            std::thread::spawn(move || {
                let trainer = regression_trainer(&engine(), &base, json!({"epochs": 3}));
                run_all(&trainer);
                exported(&base, "m", &trainer)
            })
        })
        .collect();
    let results: Vec<Vec<NamedTensor>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    assert_eq!(results[0], results[1]);
}

#[test]
fn public_types_are_send_sync_static() {
    fn check<T: Send + Sync + 'static>() {}
    fn handle<T: Clone + Send + Sync + 'static>() {}
    handle::<Trainer>();
    check::<TrainingConfig>();
    check::<TrainerOptions>();
    check::<ResumeOptions>();
    check::<InMemoryDataset>();
    check::<TrainingEvent>();
    check::<TrainingOutcome>();
    check::<TrainerInfo>();
}

// ---- JSON ----

#[test]
fn training_config_json_is_equivalent() {
    let full = classification_config(json!({
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "split", "fraction": 0.25 }, "shuffle": false },
        "schedule": { "kind": "cosine", "warmup_steps": 3, "min_lr": 1e-5 },
        "gradient_clip": { "max_norm": 0.5 },
        "checkpoint": { "dir": "runs/x", "every_steps": 5, "keep_best": "val.accuracy" }
    }));
    for doc in [regression_config(json!({})), full] {
        let c = config(&doc);
        let text = c.to_json();
        assert_eq!(TrainingConfig::from_json(&text).unwrap(), c);
        // Serde and `to_json` write the same canonical document.
        assert_eq!(serde_json::to_string(&c).unwrap(), text);
        let back: TrainingConfig = serde_json::from_str(&text).unwrap();
        assert_eq!(back, c);
        let canonical: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(canonical["format"], "polypous.training");
        assert_eq!(canonical["format_version"], 1);
        // A change in Rust is a change of the JSON.
        let mut changed = c.clone();
        changed.seed = u64::MAX;
        changed.device = "vulkan:discrete_gpu:0".into();
        let again = TrainingConfig::from_json(&changed.to_json()).unwrap();
        assert_eq!(again, changed);
        assert_eq!(again.seed, u64::MAX);
    }
    // Unknown fields are rejected, also nested, through serde too.
    for extra in [
        json!({"extra": 1}),
        json!({"loss": {"kind": "mse", "prediction": "y", "target": "y", "weight": 2}}),
    ] {
        let doc = regression_config(extra).to_string();
        assert_eq!(
            TrainingConfig::from_json(&doc).unwrap_err().kind,
            ErrorKind::InvalidInput
        );
        assert!(serde_json::from_str::<TrainingConfig>(&doc).is_err());
    }
}

#[test]
fn options_json_is_equivalent() {
    assert_eq!(
        TrainerOptions::default().to_json(),
        r#"{"weights":null,"base_dir":null,"datasets":[]}"#
    );
    assert_eq!(
        ResumeOptions::default().to_json(),
        r#"{"device":null,"base_dir":null,"datasets":[],"discard_newer":false}"#
    );
    assert_eq!(
        TrainerOptions::from_json("{}"),
        Ok(TrainerOptions::default())
    );
    assert_eq!(ResumeOptions::from_json("{}"), Ok(ResumeOptions::default()));

    let mut options = TrainerOptions::default();
    options.weights = Some("pretrained.safetensors".into());
    options.base_dir = Some("/data".into());
    options.datasets = vec![blobs("train", 3, 1)];
    let text = options.to_json();
    assert_eq!(TrainerOptions::from_json(&text), Ok(options.clone()));
    let json: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(json["datasets"][0]["name"], "train");
    assert_eq!(json["datasets"][0]["fields"][1]["name"], "label");
    assert_eq!(json["datasets"][0]["fields"][1]["tensor"]["dtype"], "i64");

    let mut resume = ResumeOptions::default();
    resume.device = Some("cpu".into());
    resume.base_dir = Some("/elsewhere".into());
    resume.datasets = vec![blobs("val", 2, 3)];
    resume.discard_newer = true;
    assert_eq!(ResumeOptions::from_json(&resume.to_json()), Ok(resume));

    for text in [
        r#"{"weight":"w.safetensors"}"#,
        r#"{"datasets":[{"name":"a","fields":[],"extra":1}]}"#,
        r#"{"datasets":{}}"#,
        "[]",
        "null",
        "{",
    ] {
        let error = TrainerOptions::from_json(text).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{text}");
        assert_eq!(
            error.context.operation.as_deref(),
            Some("trainer_options_from_json")
        );
        let error = ResumeOptions::from_json(text).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidInput, "{text}");
        assert_eq!(
            error.context.operation.as_deref(),
            Some("resume_options_from_json")
        );
    }
    assert!(ResumeOptions::from_json(r#"{"discard_newer":"yes"}"#).is_err());
    assert!(ResumeOptions::from_json(r#"{"device":{"backend":"cpu"}}"#).is_err());

    // At most 64 datasets, unique names.
    let empty = |name: String| json!({"name": name, "fields": []});
    let many: Vec<Value> = (0..65).map(|i| empty(format!("d{i}"))).collect();
    let sixty_four: Vec<Value> = (0..64).map(|i| empty(format!("d{i}"))).collect();
    let twice = vec![empty("d".into()), empty("d".into())];
    for (list, ok) in [(many, false), (sixty_four, true), (twice, false)] {
        let text = json!({"datasets": list}).to_string();
        assert_eq!(TrainerOptions::from_json(&text).is_ok(), ok);
        assert_eq!(ResumeOptions::from_json(&text).is_ok(), ok);
        if !ok {
            let error = TrainerOptions::from_json(&text).unwrap_err();
            assert!(
                error
                    .message
                    .starts_with("The trainer options are not valid: "),
                "{error:?}"
            );
        }
    }
}

#[test]
fn info_and_events_json() {
    let dir = dir("info-json");
    let trainer = regression_trainer(
        &engine(),
        &dir,
        json!({"epochs": 1, "checkpoint": {"dir": "run"}}),
    );
    let info: Value = serde_json::from_str(&trainer.info().to_json()).unwrap();
    assert_eq!(
        info,
        json!({
            "device": {"backend": "cpu"}, "epochs": 1, "steps_per_epoch": 8, "total_steps": 8,
            "train_samples": 128, "validation_samples": 0, "trainable_elements": 97, "step": 0,
            "finished": false,
            "trainable": [
                {"name": "fc1.weight", "shape": [16, 4]}, {"name": "fc1.bias", "shape": [16]},
                {"name": "fc2.weight", "shape": [1, 16]}, {"name": "fc2.bias", "shape": [1]}
            ]
        })
    );
    let (_, events) = run_all(&trainer);
    let mut seen = Vec::new();
    for event in &events {
        let json: Value = serde_json::from_str(&event.to_json()).unwrap();
        assert_eq!(serde_json::to_value(event).unwrap(), json);
        let kind = json["event"].as_str().unwrap().to_string();
        match &kind[..] {
            "step" => {
                assert!(json["loss"].is_number() && json["lr"].is_number(), "{json}");
                assert_eq!(json["lr"], 0.01);
            }
            "epoch_finished" => assert!(json["metrics"]["train.loss"].is_number(), "{json}"),
            "checkpoint_saved" => {
                assert!(
                    json["path"]
                        .as_str()
                        .unwrap()
                        .ends_with("step-000000000008")
                );
                assert_eq!(json["step"], 8);
            }
            _ => {}
        }
        if !seen.contains(&kind) {
            seen.push(kind);
        }
    }
    assert_eq!(
        seen,
        [
            "started",
            "epoch_started",
            "step",
            "epoch_finished",
            "checkpoint_saved",
            "finished"
        ]
    );
}

// ---- GPU devices ----

/// The regression on every available GPU device to a loss threshold, a resume on it, and its
/// export loaded on it. Software rasterizers only with `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (the CI
/// lavapipe job sets it). MSE only: `cross_entropy`'s backward pass on lavapipe is exposed to the
/// Mesa crash M2-9.
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn trains_resumes_and_exports_on_gpus() {
    let dir = dir("gpu");
    let mut config = EngineConfig::default();
    config.software_rasterizers =
        std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    config.probe_cache_dir = Some(dir.join("probe-cache"));
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
    assert!(!devices.is_empty(), "no available GPU device");
    for device in devices {
        let base = dir.join(device.to_string().replace(':', "-"));
        let extra =
            json!({"device": device.to_string(), "checkpoint": {"dir": "run", "keep_last": null}});
        let trainer = regression_trainer(&engine, &base, extra);
        assert_eq!(trainer.info().device, device);
        // Cancelled after the first emitted step from step 50 on (`step` events are throttled).
        let cancel = CancellationToken::new();
        let mut first = None;
        let mut stopped = None;
        let outcome = trainer
            .run(&cancel, |event| {
                if let TrainingEvent::Started { device, .. } = &event {
                    first = Some(*device);
                }
                if let TrainingEvent::Step { step, .. } = event
                    && step >= 50
                    && stopped.is_none()
                {
                    stopped = Some(step + 1);
                    cancel.cancel();
                }
            })
            .unwrap();
        assert_eq!(first, Some(device));
        assert_eq!(outcome.stop_reason, StopReason::Cancelled);
        assert_eq!(Some(outcome.step), stopped);
        drop(trainer);
        let resumed = engine
            .resume_trainer(
                base.join("run"),
                ResumeOptions::default(),
                &CancellationToken::new(),
            )
            .unwrap_or_else(|e| panic!("{device}: {e:?}"));
        assert_eq!(resumed.info().device, device);
        assert_eq!(Some(resumed.info().step), stopped);
        let (outcome, _) = run_all(&resumed);
        let loss = outcome.final_metrics.get("train.loss").unwrap();
        eprintln!("{device}: final train.loss {loss} (resumed at step {stopped:?})");
        assert!(loss < 0.01, "{device}: {loss}");

        let graph_path = base.join("model.graph.json");
        let weights_path = base.join("model.safetensors");
        resumed.export(&graph_path, &weights_path).unwrap();
        let mut options = GraphLoadOptions::default();
        options.weights = Some(weights_path);
        options.device = Some(device);
        let model = engine
            .load_graph(
                &Graph::from_file(&graph_path).unwrap(),
                options,
                &CancellationToken::new(),
            )
            .unwrap();
        let x = regression_fields(128, 5).remove(0);
        let out = model
            .run(vec![x], vec![], &CancellationToken::new())
            .unwrap();
        let predictions = read_f32(out.output("y").unwrap());
        let targets = read_f32(&regression_fields(128, 5)[1].tensor);
        let mse: f64 = predictions
            .iter()
            .zip(&targets)
            .map(|(p, t)| f64::from(p - t).powi(2))
            .sum::<f64>()
            / 128.0;
        eprintln!("{device}: exported model MSE {mse}");
        assert!(mse < 0.01, "{device}: {mse}");
    }
}
