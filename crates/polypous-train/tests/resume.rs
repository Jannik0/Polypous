//! Resume (`docs/specs/training.md` §5.4; testing-strategy.md §3.8 "Resume"), on the CPU: a small
//! model with `nn.batch_norm` and `nn.dropout`, AdamW, cosine with warmup, gradient clipping,
//! shuffling, a `split` validation set and `every_steps` checkpoints, for 3 epochs:
//!
//! - (a) uninterrupted;
//! - (b) cancelled in the middle of an epoch and resumed from the cancel checkpoint;
//! - (c) cancelled at an epoch boundary and continued by calling `run` again;
//! - (d) cancelled during the validation pass and resumed;
//! - (e) resumed from a mid-run `every_steps` checkpoint of (a).
//!
//! The final params, optimizer state, `history`, `best` and the bytes of the final checkpoint's
//! `weights.safetensors` and `optimizer.safetensors` must be identical (bit-exact, §2).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::path::{Path, PathBuf};

use common::*;
use polypous_backend::TensorData;
use polypous_runtime::CancellationToken;
use polypous_train::{
    InMemoryDataset, ResumeOptions, StopReason, Trainer, TrainingConfig, TrainingEvent,
};
use serde_json::json;

const EPOCHS: u64 = 3;

fn data() -> InMemoryDataset {
    blobs("train", 90, 11)
}

fn config(dir: &str) -> TrainingConfig {
    classification_config(json!({
        "seed": 1234, "epochs": EPOCHS, "batch_size": 8,
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "split", "fraction": 0.25 }, "shuffle": true },
        "optimizer": { "kind": "adamw", "lr": 0.02, "weight_decay": 0.05 },
        "schedule": { "kind": "cosine", "warmup_steps": 4, "min_lr": 0.001 },
        "gradient_clip": { "max_norm": 0.5 },
        "checkpoint": { "dir": dir, "every_steps": 5, "keep_last": null, "keep_best": "val.accuracy" }
    }))
}

/// Everything the guarantee covers, as comparable bytes and JSON.
#[derive(Debug, PartialEq)]
struct Result {
    params: Vec<(String, TensorData)>,
    optimizer: Vec<(String, TensorData)>,
    history: String,
    best: String,
    weights_file: Vec<u8>,
    optimizer_file: Vec<u8>,
}

fn result(t: &Trainer, last_checkpoint: &str) -> Result {
    let (history, best) = t.history().unwrap();
    let dir = PathBuf::from(last_checkpoint);
    Result {
        params: t.params().unwrap(),
        optimizer: t.optimizer_state().unwrap(),
        history: serde_json::to_string(&history).unwrap(),
        best: serde_json::to_string(&best).unwrap(),
        weights_file: std::fs::read(dir.join("weights.safetensors")).unwrap(),
        optimizer_file: std::fs::read(dir.join("optimizer.safetensors")).unwrap(),
    }
}

/// Compares field by field, naming the first difference.
fn assert_same(got: &Result, expected: &Result, what: &str) {
    for (i, ((n, a), (_, b))) in got.params.iter().zip(&expected.params).enumerate() {
        assert_eq!(a, b, "{what}: param {i} `{n}` differs");
    }
    assert_eq!(
        got.params.len(),
        expected.params.len(),
        "{what}: param count"
    );
    for ((n, a), (m, b)) in got.optimizer.iter().zip(&expected.optimizer) {
        assert_eq!(n, m, "{what}: optimizer state names");
        assert_eq!(a, b, "{what}: optimizer state `{n}` differs");
    }
    assert_eq!(
        got.optimizer.len(),
        expected.optimizer.len(),
        "{what}: state count"
    );
    assert_eq!(got.history, expected.history, "{what}: history");
    assert_eq!(got.best, expected.best, "{what}: best");
    assert!(
        got.weights_file == expected.weights_file,
        "{what}: weights.safetensors bytes"
    );
    assert!(
        got.optimizer_file == expected.optimizer_file,
        "{what}: optimizer.safetensors bytes"
    );
}

fn new_trainer(base: &Path) -> Trainer {
    trainer(
        &bn_dropout(4, 12, 3),
        config("run"),
        options(base, vec![data()], None),
    )
}

fn resume(path: &Path) -> Trainer {
    let options = ResumeOptions {
        datasets: vec![data()],
        clock: Some(ticking()),
        ..ResumeOptions::default()
    };
    Trainer::resume(path, options, &cpu, &CancellationToken::new())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// Runs `t` and cancels when `when` returns true for an event (checked after the callback).
fn run_until(
    t: &Trainer,
    when: impl Fn(&TrainingEvent) -> bool,
) -> polypous_train::TrainingOutcome {
    let cancel = CancellationToken::new();
    t.run(&cancel, |e| {
        if when(&e) {
            cancel.cancel();
        }
    })
    .unwrap()
}

fn uninterrupted(base: &Path) -> (Result, Trainer) {
    let t = new_trainer(base);
    let (outcome, _) = run(&t, &CancellationToken::new());
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    let r = result(&t, outcome.last_checkpoint.as_deref().unwrap());
    (r, t)
}

#[test]
fn resume_is_bit_exact() {
    let a = TempDir::new("resume-a");
    let (expected, ta) = uninterrupted(a.path());
    let info = ta.info();
    assert!(info.steps_per_epoch >= 8, "{info:?}");
    let s = info.steps_per_epoch;
    // The model is the §3.8 one: the unused param gets no state.
    assert!(
        expected
            .optimizer
            .iter()
            .all(|(n, _)| !n.starts_with("unused:"))
    );
    assert!(
        expected
            .optimizer
            .iter()
            .any(|(n, _)| n == "bn.weight:exp_avg")
    );

    // (b) Mid-epoch: cancel after step k = S + 2 (epoch 1, not a multiple of every_steps).
    let b = TempDir::new("resume-b");
    let t = new_trainer(b.path());
    let k = s + 2;
    let outcome = run_until(
        &t,
        |e| matches!(e, TrainingEvent::Step { step, .. } if *step == k),
    );
    assert_eq!(outcome.stop_reason, StopReason::Cancelled);
    assert_eq!(outcome.step, k + 1);
    drop(t);
    let t = resume(&b.join(&format!("run/step-{:012}", k + 1)));
    let (outcome, events) = run(&t, &CancellationToken::new());
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert!(matches!(events[0], TrainingEvent::Started { step, .. } if step == k + 1));
    assert!(matches!(events[1], TrainingEvent::EpochStarted { epoch: 1, batch } if batch == 3));
    assert_same(
        &result(&t, outcome.last_checkpoint.as_deref().unwrap()),
        &expected,
        "(b) mid-epoch",
    );

    // (c) Epoch boundary: cancel at epoch 0's `epoch_finished`, then `run` again.
    let c = TempDir::new("resume-c");
    let t = new_trainer(c.path());
    let outcome = run_until(&t, |e| {
        matches!(e, TrainingEvent::EpochFinished { epoch: 0, .. })
    });
    assert_eq!(outcome.stop_reason, StopReason::Cancelled);
    assert_eq!(outcome.step, s);
    assert_eq!(outcome.epochs_completed, 1);
    let (outcome, _) = run(&t, &CancellationToken::new());
    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert_same(
        &result(&t, outcome.last_checkpoint.as_deref().unwrap()),
        &expected,
        "(c) boundary",
    );

    // (d) During validation: the checkpoint has `batch = S`, and resume repeats the pass.
    let d = TempDir::new("resume-d");
    let t = new_trainer(d.path());
    let outcome = run_until(&t, |e| {
        matches!(e, TrainingEvent::ValidationStarted { epoch: 1 })
    });
    assert_eq!(outcome.step, 2 * s);
    assert_eq!(outcome.epochs_completed, 1);
    let state: serde_json::Value = serde_json::from_slice(
        &std::fs::read(d.join(&format!("run/step-{:012}/state.json", 2 * s))).unwrap(),
    )
    .unwrap();
    assert_eq!(state["epoch"], 1);
    assert_eq!(state["batch"], s);
    drop(t);
    let t = resume(&d.join("run"));
    let (outcome, events) = run(&t, &CancellationToken::new());
    assert!(matches!(events[1], TrainingEvent::EpochStarted { epoch: 1, batch } if batch == s));
    assert_eq!(names(&events)[2], "validation_started");
    assert_same(
        &result(&t, outcome.last_checkpoint.as_deref().unwrap()),
        &expected,
        "(d) validation",
    );

    // (e) From a mid-run `every_steps` checkpoint of (a), in a copy of (a)'s run directory: the
    // newer checkpoints are refused, then discarded with `discard_newer`. (`best` is defined over
    // the run directory's checkpoints, §5.3, so the run directory is copied whole.)
    let e = TempDir::new("resume-e");
    for entry in std::fs::read_dir(a.join("run")).unwrap() {
        let entry = entry.unwrap();
        copy_dir(&entry.path(), &e.join("run").join(entry.file_name()));
    }
    let from = e.join(&format!("run/step-{:012}", 15));
    let options = || ResumeOptions {
        datasets: vec![data()],
        clock: Some(ticking()),
        ..ResumeOptions::default()
    };
    let err = Trainer::resume(&from, options(), &cpu, &CancellationToken::new()).unwrap_err();
    assert!(err.message().contains("newer than step 15"), "{err}");
    let t = Trainer::resume(
        &from,
        ResumeOptions {
            discard_newer: true,
            ..options()
        },
        &cpu,
        &CancellationToken::new(),
    )
    .unwrap();
    let (outcome, events) = run(&t, &CancellationToken::new());
    let removed = events
        .iter()
        .filter(|e| matches!(e, TrainingEvent::CheckpointRemoved { .. }))
        .count();
    assert!(
        removed >= 3,
        "the newer checkpoints are discarded: {removed}"
    );
    assert_same(
        &result(&t, outcome.last_checkpoint.as_deref().unwrap()),
        &expected,
        "(e) every_steps",
    );
    let left: Vec<String> = std::fs::read_dir(e.join("run"))
        .unwrap()
        .map(|d| d.unwrap().file_name().into_string().unwrap())
        .collect();
    let mut left = left;
    left.sort();
    let mut in_a: Vec<String> = std::fs::read_dir(a.join("run"))
        .unwrap()
        .map(|d| d.unwrap().file_name().into_string().unwrap())
        .collect();
    in_a.sort();
    assert_eq!(left, in_a, "the same checkpoints as the uninterrupted run");

    // A second uninterrupted run is identical too (determinism, §1.4).
    let a2 = TempDir::new("resume-a2");
    assert_same(&uninterrupted(a2.path()).0, &expected, "second run");
}

pub fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// Resumes in a new process: the test binary runs itself with `POLYPOUS_RESUME_CHILD` set to a
/// checkpoint, and that process trains to the end. The final checkpoint's files must equal the
/// uninterrupted run's, so nothing the bit-exact guarantee needs lives only in this process.
#[test]
fn resume_in_another_process_is_bit_exact() {
    if let Ok(path) = std::env::var("POLYPOUS_RESUME_CHILD") {
        let t = resume(Path::new(&path));
        let (outcome, _) = run(&t, &CancellationToken::new());
        assert_eq!(outcome.stop_reason, StopReason::Completed);
        return;
    }
    let a = TempDir::new("process-a");
    let (expected, ta) = uninterrupted(a.path());
    let s = ta.info().steps_per_epoch;
    let b = TempDir::new("process-b");
    let t = new_trainer(b.path());
    let k = 2 * s + 1;
    run_until(
        &t,
        |e| matches!(e, TrainingEvent::Step { step, .. } if *step == k),
    );
    drop(t);
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "resume_in_another_process_is_bit_exact",
            "--nocapture",
        ])
        .env(
            "POLYPOUS_RESUME_CHILD",
            b.join(&format!("run/step-{:012}", k + 1)),
        )
        .status()
        .unwrap();
    assert!(status.success());
    let last = b.join(&format!("run/step-{:012}", 3 * s));
    let file = |dir: &Path, name: &str| std::fs::read(dir.join(name)).unwrap();
    assert!(file(&last, "weights.safetensors") == expected.weights_file);
    assert!(file(&last, "optimizer.safetensors") == expected.optimizer_file);
    let state = |dir: &Path| -> serde_json::Value {
        serde_json::from_slice(&file(dir, "state.json")).unwrap()
    };
    let (got, want) = (
        state(&last),
        state(&a.join(&format!("run/step-{:012}", 3 * s))),
    );
    assert_eq!(got["history"], want["history"]);
    assert_eq!(got["best"], want["best"]);
    assert!(!got["best"].is_null());
}

/// §5.3: only a checkpoint that §5.1's epoch rule writes has a `keep_best` value. 5 epochs of 4
/// steps; `every_epochs: 2` writes epoch checkpoints after epochs 1, 3 and 4 (the last), and
/// `every_steps: 5` writes mid-epoch checkpoints at steps 5, 10 and 15. With seed 4 and SGD at
/// `lr = 3` the loss diverges, so the best value is epoch 0's, which gets no epoch checkpoint.
fn best_config() -> TrainingConfig {
    classification_config(json!({
        "seed": 4, "epochs": 5, "batch_size": 8,
        "optimizer": { "kind": "sgd", "lr": 3.0, "momentum": 0.9 },
        "checkpoint": { "dir": "run", "every_epochs": 2, "every_steps": 5, "keep_last": null,
                        "keep_best": "train.loss" }
    }))
}

fn best_data() -> InMemoryDataset {
    blobs("train", 32, 6)
}

fn best_trainer(base: &Path) -> Trainer {
    trainer(
        &bn_dropout(4, 6, 3),
        best_config(),
        options(base, vec![best_data()], None),
    )
}

fn best_resume(path: &Path) -> Trainer {
    let options = ResumeOptions {
        datasets: vec![best_data()],
        clock: Some(ticking()),
        ..ResumeOptions::default()
    };
    Trainer::resume(path, options, &cpu, &CancellationToken::new())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// `best` of `t` as JSON (metric, value, epoch, step, checkpoint name).
fn best_of(t: &Trainer) -> String {
    serde_json::to_string(&t.history().unwrap().1).unwrap()
}

/// Every checkpoint's `state.json` `best`: null, or naming the epoch checkpoint at its step.
fn assert_every_best_is_an_epoch_checkpoint(run_dir: &Path) {
    for entry in std::fs::read_dir(run_dir).unwrap() {
        let path = entry.unwrap().path();
        let state: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("state.json")).unwrap()).unwrap();
        let best = &state["best"];
        if !best.is_null() {
            let step = best["step"].as_u64().unwrap();
            assert_eq!(
                best["checkpoint"].as_str().unwrap(),
                format!("step-{step:012}"),
                "{}: best names a checkpoint that isn't the epoch checkpoint of its entry",
                path.display()
            );
        }
    }
}

#[test]
fn best_counts_only_epoch_checkpoints() {
    // (a) Uninterrupted.
    let a = TempDir::new("best-a");
    let t = best_trainer(a.path());
    let (outcome, _) = run(&t, &CancellationToken::new());
    let losses: Vec<f64> = outcome
        .history
        .iter()
        .map(|h| h.metrics.get("train.loss").unwrap())
        .collect();
    let min_epoch = (0..losses.len())
        .min_by(|&i, &j| losses[i].total_cmp(&losses[j]))
        .unwrap();
    assert_eq!(
        min_epoch, 0,
        "the test needs the best value in epoch 0: {losses:?}"
    );
    assert!(
        a.join("run/step-000000000005").is_dir(),
        "a mid-epoch checkpoint after epoch 0"
    );
    let best = outcome.best.clone().unwrap();
    // Epoch 1's checkpoint, the best among the epoch checkpoints (epochs 1, 3, 4).
    assert_eq!((best.epoch, best.step), (1, 8), "{losses:?}");
    assert!(best.checkpoint.ends_with("step-000000000008"));
    assert_eq!(best.value.to_bits(), losses[1].to_bits());
    assert!(losses[1] < losses[3] && losses[1] < losses[4]);
    assert_every_best_is_an_epoch_checkpoint(&a.join("run"));
    let expected = best_of(&t);

    // (b) Cancelled mid-epoch at step k = 6: the cancel checkpoint (step 7) carries epoch 0's
    // history entry but has no value. Resumed from it.
    let b = TempDir::new("best-b");
    let t = best_trainer(b.path());
    run_until(&t, |e| matches!(e, TrainingEvent::Step { step: 6, .. }));
    let state: serde_json::Value =
        serde_json::from_slice(&std::fs::read(b.join("run/step-000000000007/state.json")).unwrap())
            .unwrap();
    assert_eq!(state["history"].as_array().unwrap().len(), 1);
    assert!(state["best"].is_null(), "no checkpoint has a value yet");
    drop(t);
    let t = best_resume(&b.join("run/step-000000000007"));
    run(&t, &CancellationToken::new());
    assert_eq!(best_of(&t), expected, "(b) mid-epoch");
    assert_every_best_is_an_epoch_checkpoint(&b.join("run"));

    // (c) Cancelled at the boundary after epoch 0, where no epoch checkpoint is due: the token
    // is next checked after step k = 4 (§4.3), and the cancel checkpoint (step 5, also an
    // `every_steps` one) carries epoch 0's entry but has no value. Continued with `run`.
    let c = TempDir::new("best-c");
    let t = best_trainer(c.path());
    let outcome = run_until(&t, |e| {
        matches!(e, TrainingEvent::EpochFinished { epoch: 0, .. })
    });
    assert_eq!(outcome.step, 5);
    assert!(outcome.best.is_none());
    run(&t, &CancellationToken::new());
    assert_eq!(best_of(&t), expected, "(c) undue boundary, run again");
    assert_every_best_is_an_epoch_checkpoint(&c.join("run"));

    // (c') Cancelled at the boundary after epoch 1, where the epoch checkpoint is due: it is
    // the cancel checkpoint, and has its value. Continued with `run`.
    let c2 = TempDir::new("best-c2");
    let t = best_trainer(c2.path());
    let outcome = run_until(&t, |e| {
        matches!(e, TrainingEvent::EpochFinished { epoch: 1, .. })
    });
    assert_eq!(outcome.step, 8);
    assert!(
        outcome
            .best
            .unwrap()
            .checkpoint
            .ends_with("step-000000000008")
    );
    run(&t, &CancellationToken::new());
    assert_eq!(best_of(&t), expected, "(c') due boundary, run again");

    // (d) Cancelled at the boundary after epoch 2 (not due either; it stops after step 12),
    // resumed from the run directory in a new trainer, which judges the earlier checkpoints
    // from their own state.json.
    let d = TempDir::new("best-d");
    let t = best_trainer(d.path());
    run_until(&t, |e| {
        matches!(e, TrainingEvent::EpochFinished { epoch: 2, .. })
    });
    drop(t);
    let t = best_resume(&d.join("run"));
    assert_eq!(t.info().step, 13);
    run(&t, &CancellationToken::new());
    assert_eq!(best_of(&t), expected, "(d) boundary, resumed");
    assert_every_best_is_an_epoch_checkpoint(&d.join("run"));
}
