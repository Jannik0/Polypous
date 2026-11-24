//! The training loop, events, cancellation, the trainer's states and export
//! (`docs/specs/training.md` §1.2, §4, §5.5; public-api.md §1.4), on the CPU. `learns_gpu`
//! (`#[ignore = "gpu"]`) trains on every GPU device for a loss threshold only.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::sync::Arc;
use std::time::Duration;

use common::*;
use polypous_backend::DynTensor;
use polypous_backend::device::DeviceId;
use polypous_runtime::{
    CancellationToken, Model, Plan, PlanOptions, RunMode, RunOptions, WeightFile,
};
use polypous_train::{ManualClock, StopReason, TrainErrorKind, Trainer, TrainingEvent};
use serde_json::json;

#[test]
fn a_small_mlp_learns_on_the_cpu() {
    let dir = TempDir::new("learns");
    let config = classification_config(json!({
        "epochs": 8, "batch_size": 16,
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "tensors", "name": "val" } },
    }));
    let t = trainer(
        &mlp(4, 16, 3),
        config,
        options(
            dir.path(),
            vec![blobs("train", 240, 1), blobs("val", 60, 2)],
            None,
        ),
    );
    let (outcome, _) = run(&t, &CancellationToken::new());
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    let first = &outcome.history[0].metrics;
    let last = &outcome.final_metrics;
    let (l0, l1) = (
        first.get("train.loss").unwrap(),
        last.get("train.loss").unwrap(),
    );
    eprintln!(
        "train.loss {l0} → {l1}, val.accuracy {:?}",
        last.get("val.accuracy")
    );
    assert!(l1 < 0.1, "train.loss {l1} after 8 epochs");
    assert!(last.get("val.accuracy").unwrap() >= 0.95);
}

/// The regression task with MSE, for the GPU test (no indexing ops in the backward pass).
fn regression_run(device: DeviceId) -> f64 {
    let dir = TempDir::new("regression");
    let doc = json!({
        "format": "polypous.training", "format_version": 1, "seed": 3, "device": "auto",
        "epochs": 30, "batch_size": 16,
        "data": { "train": { "kind": "tensors", "name": "train" } },
        "inputs": { "x": "x" },
        "loss": { "kind": "mse", "prediction": "y", "target": "y" },
        "optimizer": { "kind": "adam", "lr": 0.01 }
    });
    let config = polypous_train::TrainingConfig::from_json(&doc.to_string()).unwrap();
    let t = Trainer::new(
        &graph(&mlp(4, 16, 1)),
        config,
        options(dir.path(), vec![regression("train", 128, 5)], None),
        &|_| Ok(device),
        &CancellationToken::new(),
    )
    .unwrap();
    let (outcome, _) = run(&t, &CancellationToken::new());
    outcome.final_metrics.get("train.loss").unwrap()
}

#[test]
fn regression_learns_on_the_cpu() {
    let loss = regression_run(DeviceId::Cpu);
    eprintln!("cpu: final train.loss {loss}");
    assert!(loss < 0.01, "{loss}");
}

#[test]
#[ignore = "gpu"]
fn learns_gpu() {
    use polypous_backend::device::{BackendKind, EnumerateOptions, list_devices};
    let options = EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    };
    let devices: Vec<DeviceId> = list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu && !options.skips(*id))
        .collect();
    assert!(
        !devices.is_empty(),
        "no GPU device found for the compiled-in GPU backends"
    );
    for device in devices {
        let loss = regression_run(device);
        eprintln!("{device}: final train.loss {loss}");
        assert!(loss < 0.01, "{device}: {loss}");
    }
}

/// The §3.8-style model with a split validation set, checkpoints by step and epoch.
fn events_config(extra: serde_json::Value) -> polypous_train::TrainingConfig {
    let mut doc = json!({
        "epochs": 2, "batch_size": 8,
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "split", "fraction": 0.25 } },
        "checkpoint": { "dir": "run", "every_steps": 4, "keep_last": 2 }
    });
    merge(&mut doc, extra);
    classification_config(doc)
}

#[test]
fn events_come_in_the_specified_order() {
    let dir = TempDir::new("events");
    let t = trainer(
        &bn_dropout(4, 6, 3),
        events_config(json!({})),
        options(dir.path(), vec![blobs("train", 48, 3)], None),
    );
    let info = t.info();
    let s = info.steps_per_epoch;
    assert_eq!(
        (info.train_samples, info.validation_samples, s),
        (36, 12, 5)
    );
    assert_eq!(info.total_steps, 10);
    assert_eq!(info.trainable_elements, 6 * 4 + 6 + 6 + 6 + 3 * 6 + 3 + 3);
    let names: Vec<&str> = info.trainable.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "fc1.weight",
            "fc1.bias",
            "bn.weight",
            "bn.bias",
            "fc2.weight",
            "fc2.bias",
            "unused"
        ]
    );
    let (outcome, events) = run(&t, &CancellationToken::new());
    // Without the removals: every step is emitted (the ticking clock); checkpoints after steps 4
    // and 8 (k + 1 a multiple of 4, not the last step of an epoch) and after every epoch.
    let kept: Vec<&str> = names_of(&events)
        .into_iter()
        .filter(|n| *n != "checkpoint_removed")
        .collect();
    let mut expected = vec!["started"];
    for e in 0..2u64 {
        expected.push("epoch_started");
        for b in 0..s {
            expected.push("step");
            let k = e * s + b;
            if (k + 1).is_multiple_of(4) && b != s - 1 {
                expected.push("checkpoint_saved");
            }
        }
        expected.push("validation_started");
        expected.push("epoch_finished");
        expected.push("checkpoint_saved");
    }
    expected.push("finished");
    assert_eq!(kept, expected);
    // Each removal follows a checkpoint_saved (or another removal).
    for (i, e) in events.iter().enumerate() {
        if e.name() == "checkpoint_removed" {
            assert!(matches!(
                events[i - 1].name(),
                "checkpoint_saved" | "checkpoint_removed"
            ));
        }
    }
    // Saved 4, 5, 8, 10; keep_last 2 removes 4 then 5.
    let saved: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            TrainingEvent::CheckpointSaved { step, path } => {
                assert!(path.ends_with(&format!("step-{step:012}")), "{path}");
                Some(*step)
            }
            _ => None,
        })
        .collect();
    assert_eq!(saved, [4, 5, 8, 10]);
    let removed: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            TrainingEvent::CheckpointRemoved { path } => Some(path.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(removed.len(), 2);
    assert!(removed[0].ends_with("step-000000000004") && removed[1].ends_with("step-000000000005"));
    // The fields.
    let steps: Vec<(u64, u64, u64)> = events
        .iter()
        .filter_map(|e| match e {
            TrainingEvent::Step {
                epoch,
                batch,
                step,
                loss,
                lr,
            } => {
                assert!(loss.is_finite() && *lr == 0.01);
                Some((*epoch, *batch, *step))
            }
            _ => None,
        })
        .collect();
    assert_eq!(
        steps,
        (0..10).map(|k| (k / s, k % s, k)).collect::<Vec<_>>()
    );
    let Some(TrainingEvent::EpochFinished {
        epoch: 1,
        step: 10,
        metrics,
    }) = events.iter().rev().find(|e| e.name() == "epoch_finished")
    else {
        panic!("no epoch_finished for epoch 1");
    };
    let reported: Vec<&str> = metrics.0.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        reported,
        ["train.loss", "train.accuracy", "val.loss", "val.accuracy"]
    );
    let TrainingEvent::Finished { outcome: last } = events.last().unwrap() else {
        panic!()
    };
    assert_eq!(**last, outcome);
    assert_eq!(outcome.step, 10);
    assert_eq!(outcome.epochs_completed, 2);
    assert_eq!(outcome.history.len(), 2);
    assert_eq!(&outcome.final_metrics, metrics);
    assert_eq!(
        outcome.last_checkpoint.as_deref(),
        Some(dir.join("run/step-000000000010").to_str().unwrap())
    );
    assert!(outcome.best.is_none());
    assert!(t.info().finished);
    let kept_dirs: Vec<String> = std::fs::read_dir(dir.join("run"))
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    let mut kept_dirs = kept_dirs;
    kept_dirs.sort();
    assert_eq!(kept_dirs, ["step-000000000008", "step-000000000010"]);
}

fn names_of(events: &[TrainingEvent]) -> Vec<&'static str> {
    events.iter().map(TrainingEvent::name).collect()
}

#[test]
fn step_events_are_throttled() {
    // 2 epochs of 5 steps. The fake clock moves only in the callback, by `advance` per `step`.
    let emitted = |advance: Duration| {
        let dir = TempDir::new("throttle");
        let clock = Arc::new(ManualClock::new());
        let t = trainer(
            &mlp(4, 6, 3),
            classification_config(json!({ "epochs": 2 })),
            options(dir.path(), vec![blobs("train", 40, 3)], Some(clock.clone())),
        );
        let mut steps = Vec::new();
        t.run(&CancellationToken::new(), |e| {
            if let TrainingEvent::Step { step, .. } = e {
                steps.push(step);
                clock.advance(advance);
            }
        })
        .unwrap();
        steps
    };
    // The first step of the run and the last of every epoch, always.
    assert_eq!(emitted(Duration::ZERO), [0, 4, 9]);
    assert_eq!(emitted(Duration::from_millis(99)), [0, 4, 9]);
    // At least 100 ms since the last emitted one.
    assert_eq!(
        emitted(Duration::from_millis(100)),
        (0..10).collect::<Vec<_>>()
    );
    // The first step of a continued run is emitted too.
    let dir = TempDir::new("throttle-2");
    let clock = Arc::new(ManualClock::new());
    let t = trainer(
        &mlp(4, 6, 3),
        classification_config(json!({ "epochs": 2 })),
        options(dir.path(), vec![blobs("train", 40, 3)], Some(clock)),
    );
    let cancel = CancellationToken::new();
    let mut first = Vec::new();
    t.run(&cancel, |e| {
        if let TrainingEvent::Step { step, .. } = e {
            first.push(step);
            cancel.cancel();
        }
    })
    .unwrap();
    let (_, events) = run(&t, &CancellationToken::new());
    let steps: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            TrainingEvent::Step { step, .. } => Some(*step),
            _ => None,
        })
        .collect();
    assert_eq!(first, [0]);
    assert_eq!(steps, [1, 4, 9]);
}

#[test]
fn running_statistics_are_written_back_and_unused_params_untouched() {
    // One step on the whole set, SGD with lr = 0: the params stay, the running statistics are
    // the runtime's training-mode result for step 0.
    let dir = TempDir::new("running-stats");
    let config = classification_config(json!({
        "epochs": 1, "batch_size": 16, "data": { "train": { "kind": "tensors", "name": "train" }, "shuffle": false },
        "optimizer": { "kind": "sgd", "lr": 0.0, "weight_decay": 0.1 }
    }));
    let doc = bn_dropout(4, 6, 3);
    let t = trainer(
        &doc,
        config,
        options(dir.path(), vec![blobs("train", 16, 4)], None),
    );
    let before = t.params().unwrap();
    run(&t, &CancellationToken::new());
    let after = t.params().unwrap();
    // The expected running statistics, from the runtime directly.
    let plan = Plan::new(&graph(&doc), PlanOptions::new(DeviceId::Cpu)).unwrap();
    let params: polypous_runtime::Params = before
        .iter()
        .map(|(n, d)| (n.clone(), DynTensor::from_data(d, DeviceId::Cpu).unwrap()))
        .collect();
    let x = blobs("train", 16, 4).fields[0].1.clone();
    let out = plan
        .run(
            vec![("x".into(), DynTensor::from_data(&x, DeviceId::Cpu).unwrap())],
            vec![],
            &params,
            RunMode::Training { seed: 7, step: 0 },
            &RunOptions::default(),
        )
        .unwrap();
    for (name, expected) in &out.running_stats {
        let got = &after.iter().find(|(n, _)| n == name).unwrap().1;
        assert_eq!(*got, expected.to_data().unwrap(), "{name}");
        let old = &before.iter().find(|(n, _)| n == name).unwrap().1;
        assert_ne!(got, old, "{name} changed");
    }
    assert_eq!(out.running_stats.len(), 2);
    // lr = 0: every trainable param is unchanged (SGD's weight decay only enters the gradient).
    for ((n, a), (_, b)) in before.iter().zip(&after) {
        if !n.starts_with("bn.running") {
            assert_eq!(a, b, "{n}");
        }
    }
    // SGD without momentum has no state.
    assert!(t.optimizer_state().unwrap().is_empty());

    // AdamW with decay: `unused` gets no gradient, so no decay, no update and no state.
    let dir = TempDir::new("unused");
    let t = trainer(
        &doc,
        classification_config(
            json!({ "optimizer": { "kind": "adamw", "lr": 0.1, "weight_decay": 0.5 } }),
        ),
        options(dir.path(), vec![blobs("train", 24, 4)], None),
    );
    run(&t, &CancellationToken::new());
    let params = t.params().unwrap();
    let unused = &params.iter().find(|(n, _)| n == "unused").unwrap().1;
    assert_eq!(read_f32(unused), [0.5, 0.5, 0.5]);
    let state = t.optimizer_state().unwrap();
    assert!(
        state
            .iter()
            .all(|(n, _)| !n.starts_with("unused:") && !n.starts_with("bn.running"))
    );
    assert_eq!(
        state.len(),
        2 * 6,
        "exp_avg and exp_avg_sq of the 6 params with a gradient"
    );
}

#[test]
fn trainer_states() {
    let dir = TempDir::new("states");
    let t = trainer(
        &mlp(4, 6, 3),
        classification_config(json!({ "epochs": 2, "checkpoint": { "dir": "run" } })),
        options(dir.path(), vec![blobs("train", 24, 4)], None),
    );
    // Cancelled before the first step: an outcome at step 0, and a checkpoint of step 0.
    let cancel = CancellationToken::new();
    cancel.cancel();
    let (outcome, events) = run(&t, &cancel);
    assert_eq!(
        (outcome.stop_reason, outcome.step),
        (StopReason::Cancelled, 0)
    );
    assert_eq!(
        names_of(&events),
        ["started", "checkpoint_saved", "finished"]
    );
    // Cancelled again at once: no second checkpoint of the same state.
    let (_, events) = run(&t, &cancel);
    assert_eq!(names_of(&events), ["started", "finished"]);
    // From its own callback: run and export are refused, close is deferred.
    let clone = t.clone();
    let mut inner = Vec::new();
    let outcome = t
        .run(&CancellationToken::new(), |e| {
            if let TrainingEvent::Step { step: 1, .. } = e {
                inner.push(clone.run(&CancellationToken::new(), |_| {}).unwrap_err());
                inner.push(
                    clone
                        .export(dir.join("g.json"), dir.join("w.safetensors"))
                        .unwrap_err(),
                );
                inner.push(clone.params().unwrap_err());
                assert_eq!(clone.info().step, 2);
            }
        })
        .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert_eq!(inner.len(), 3);
    for e in &inner {
        assert_eq!(e.kind(), TrainErrorKind::InvalidInput);
        assert!(e.message().contains("running"), "{e}");
    }
    // Finished: run is refused, export works.
    let e = t.run(&CancellationToken::new(), |_| {}).unwrap_err();
    assert!(e.message().contains("finished"), "{e}");
    t.export(dir.join("m.graph.json"), dir.join("m.safetensors"))
        .unwrap();
    assert!(t.info().finished);
    // Close: idempotent; run and export refused on every clone; info still works.
    t.close();
    t.close();
    for c in [&t, &clone] {
        assert!(
            c.run(&CancellationToken::new(), |_| {})
                .unwrap_err()
                .message()
                .contains("closed")
        );
        assert!(
            c.export(dir.join("x.json"), dir.join("x.safetensors"))
                .unwrap_err()
                .message()
                .contains("closed")
        );
        assert_eq!(c.info().step, t.info().total_steps);
    }

    // Close from the callback: the run finishes, then the trainer is closed.
    let dir2 = TempDir::new("states-2");
    let t = trainer(
        &mlp(4, 6, 3),
        classification_config(json!({})),
        options(dir2.path(), vec![blobs("train", 24, 4)], None),
    );
    let clone = t.clone();
    let outcome = t
        .run(&CancellationToken::new(), |e| {
            if e.name() == "started" {
                clone.close()
            }
        })
        .unwrap();
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert!(
        t.export(dir2.join("g"), dir2.join("w"))
            .unwrap_err()
            .message()
            .contains("closed")
    );

    // A failed run: the run directory becomes a file, so the checkpoint can't be written.
    let dir3 = TempDir::new("states-3");
    let t = trainer(
        &mlp(4, 6, 3),
        classification_config(json!({ "checkpoint": { "dir": "run" } })),
        options(dir3.path(), vec![blobs("train", 24, 4)], None),
    );
    let run_dir = dir3.join("run");
    let e = t
        .run(&CancellationToken::new(), |e| {
            if e.name() == "validation_started" || e.name() == "epoch_finished" {
                let _ = std::fs::remove_dir_all(&run_dir);
                std::fs::write(&run_dir, "x").unwrap();
            }
        })
        .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::Io, "{e}");
    assert!(
        t.run(&CancellationToken::new(), |_| {})
            .unwrap_err()
            .message()
            .contains("failed")
    );
    assert!(
        t.export(dir3.join("g"), dir3.join("w"))
            .unwrap_err()
            .message()
            .contains("failed")
    );
    let _ = t.info();
}

#[test]
fn export_round_trips_through_the_runtime() {
    let dir = TempDir::new("export");
    let doc = bn_dropout(4, 6, 3);
    let t = trainer(
        &doc,
        classification_config(json!({})),
        options(dir.path(), vec![blobs("train", 30, 4)], None),
    );
    run(&t, &CancellationToken::new());
    let (g, w) = (dir.join("model.graph.json"), dir.join("model.safetensors"));
    t.export(&g, &w).unwrap();
    // No temporary file is left next to them.
    let mut names: Vec<String> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    assert_eq!(names, ["model.graph.json", "model.safetensors"]);
    let loaded = polypous_ir::Graph::from_json(&std::fs::read(&g).unwrap()).unwrap();
    assert_eq!(
        loaded.to_json(),
        graph(&doc).to_json(),
        "the graph is written unchanged"
    );
    let model = Model::load(
        Plan::new(&loaded, PlanOptions::new(DeviceId::Cpu)).unwrap(),
        &WeightFile::open(&w).unwrap(),
    )
    .unwrap();
    // The trainer's inference-mode outputs: the same plan run on its param store.
    let plan = Plan::new(&graph(&doc), PlanOptions::new(DeviceId::Cpu)).unwrap();
    let params: polypous_runtime::Params = t
        .params()
        .unwrap()
        .iter()
        .map(|(n, d)| (n.clone(), DynTensor::from_data(d, DeviceId::Cpu).unwrap()))
        .collect();
    let x = blobs("x", 13, 9).fields[0].1.clone();
    let input = || {
        vec![(
            "x".to_owned(),
            DynTensor::from_data(&x, DeviceId::Cpu).unwrap(),
        )]
    };
    let expected = plan
        .run(
            input(),
            vec![],
            &params,
            RunMode::Inference,
            &RunOptions::default(),
        )
        .unwrap();
    let got = model.run(input(), vec![], &RunOptions::default()).unwrap();
    assert_eq!(
        got.output("y").unwrap().to_data().unwrap(),
        expected.output("y").unwrap().to_data().unwrap()
    );
    // Every declared param, running statistics included.
    let file = polypous_formats::safetensors::SafetensorsFile::open(&w).unwrap();
    assert_eq!(file.len(), 9);
    assert!(file.metadata().is_empty());
}

/// The resident memory of this process, in bytes (Linux).
#[cfg(target_os = "linux")]
fn rss() -> u64 {
    let statm = std::fs::read_to_string("/proc/self/statm").unwrap();
    let pages: u64 = statm.split_whitespace().nth(1).unwrap().parse().unwrap();
    pages * 4096
}

#[test]
#[cfg(target_os = "linux")]
fn memory_does_not_grow_per_step() {
    // A 512 × 512 linear layer (1 MiB of params, 2 MiB of AdamW state): after warm-up, 150 more
    // steps must not add memory in proportion to the steps.
    let dir = TempDir::new("memory");
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "nn": 1 }, "dims": { "batch": {} },
        "inputs": [ { "name": "x", "dtype": "float", "shape": ["batch", 512] } ],
        "outputs": [ { "name": "y", "value": "fc" } ],
        "params": [ { "name": "w", "dtype": "float", "shape": [512, 512], "init": { "kind": "normal", "mean": 0.0, "std": 0.01 } },
                    { "name": "b", "dtype": "float", "shape": [512], "init": { "kind": "zeros" } } ],
        "nodes": [ { "id": "fc", "op": "nn.linear", "inputs": { "x": "x" }, "params": { "weight": "w", "bias": "b" } } ]
    });
    let x = values(16 * 512, 1);
    let ds = polypous_train::InMemoryDataset {
        name: "train".into(),
        fields: vec![
            ("x".into(), f32_data(&[16, 512], &x)),
            ("y".into(), f32_data(&[16, 512], &x)),
        ],
    };
    let config = polypous_train::TrainingConfig::from_json(&json!({
        "format": "polypous.training", "format_version": 1, "epochs": 200, "batch_size": 16,
        "data": { "train": { "kind": "tensors", "name": "train" } }, "inputs": { "x": "x" },
        "loss": { "kind": "mse", "prediction": "y", "target": "y" }, "optimizer": { "kind": "adamw" }
    }).to_string()).unwrap();
    let t = trainer(&doc, config, options(dir.path(), vec![ds], None));
    let mut samples = Vec::new();
    t.run(&CancellationToken::new(), |e| {
        if let TrainingEvent::Step { step, .. } = e
            && (step == 50 || step == 199)
        {
            samples.push(rss());
        }
    })
    .unwrap();
    let growth = samples[1].saturating_sub(samples[0]);
    eprintln!(
        "RSS at step 50: {} MiB, at step 199: {} MiB",
        samples[0] >> 20,
        samples[1] >> 20
    );
    // One copy of the params per step would be 150 MiB; allow allocator noise of 16 MiB.
    assert!(
        growth < 16 << 20,
        "RSS grew by {growth} bytes over 150 steps"
    );
}

#[test]
fn every_step_has_its_own_dropout_masks() {
    // lr = 0 and one batch per epoch without shuffling: each step's loss differs only by its
    // dropout masks, which are keyed by the step k (§1.2 step 2). Each must equal the runtime's
    // training-mode result for that k.
    let dir = TempDir::new("dropout-steps");
    let doc = bn_dropout(4, 6, 3);
    let config = classification_config(json!({
        "epochs": 4, "batch_size": 16, "data": { "train": { "kind": "tensors", "name": "train" }, "shuffle": false },
        "optimizer": { "kind": "sgd", "lr": 0.0 }
    }));
    let t = trainer(
        &doc,
        config,
        options(dir.path(), vec![blobs("train", 16, 4)], None),
    );
    let params = t.params().unwrap();
    let (_, events) = run(&t, &CancellationToken::new());
    let losses: Vec<f64> = events
        .iter()
        .filter_map(|e| match e {
            TrainingEvent::Step { loss, .. } => Some(*loss),
            _ => None,
        })
        .collect();
    assert_eq!(losses.len(), 4);
    let plan = Plan::new(&graph(&doc), PlanOptions::new(DeviceId::Cpu)).unwrap();
    let ds = blobs("train", 16, 4);
    let loss = polypous_train::Loss::new(&polypous_train::LossConfig::CrossEntropy(
        polypous_train::CrossEntropyConfig {
            prediction: "y".into(),
            target: "label".into(),
            label_smoothing: 0.0,
        },
    ))
    .unwrap();
    for (k, got) in losses.iter().enumerate() {
        // The params are unchanged (lr = 0) except the running statistics, which training mode
        // doesn't read.
        let given: polypous_runtime::Params = params
            .iter()
            .map(|(n, d)| (n.clone(), DynTensor::from_data(d, DeviceId::Cpu).unwrap()))
            .collect();
        let out = plan
            .run(
                vec![(
                    "x".into(),
                    DynTensor::from_data(&ds.fields[0].1, DeviceId::Cpu).unwrap(),
                )],
                vec![],
                &given,
                RunMode::Training {
                    seed: 7,
                    step: k as u64,
                },
                &RunOptions::default(),
            )
            .unwrap();
        let l = loss
            .compute(
                out.output("y").unwrap(),
                &DynTensor::from_data(&ds.fields[1].1, DeviceId::Cpu).unwrap(),
            )
            .unwrap();
        let expected = f64::from(read_f32(&l.to_data().unwrap())[0]);
        assert_eq!(got.to_bits(), expected.to_bits(), "step {k}");
    }
    assert!(losses.windows(2).all(|w| w[0] != w[1]), "{losses:?}");
}
