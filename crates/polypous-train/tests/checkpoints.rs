//! Checkpoints (`docs/specs/training.md` §5.1–§5.4) on the CPU: `keep_last` and `keep_best`
//! with a NaN and a tie, leftovers of a crash, foreign files, and the loader of untrusted
//! checkpoints under truncation, bit flips and edited fields.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use common::*;
use polypous_runtime::CancellationToken;
use polypous_train::{ResumeOptions, TrainErrorKind, Trainer, TrainingConfig, TrainingEvent};
use serde_json::{Value, json};

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// A regression config with one step per epoch (16 samples, batch 16) and plain SGD.
fn sgd_config(lr: f64, checkpoint: Value) -> TrainingConfig {
    TrainingConfig::from_json(
        &json!({
            "format": "polypous.training", "format_version": 1, "epochs": 5, "batch_size": 16,
            "data": { "train": { "kind": "tensors", "name": "train" }, "shuffle": false },
            "inputs": { "x": "x" },
            "loss": { "kind": "mse", "prediction": "y", "target": "y" },
            "optimizer": { "kind": "sgd", "lr": lr },
            "checkpoint": checkpoint
        })
        .to_string(),
    )
    .unwrap()
}

#[test]
fn keep_best_with_nan() {
    // lr = 1e30 diverges: train.loss is finite in epoch 0, then NaN.
    let dir = TempDir::new("keep-nan");
    let t = trainer(
        &mlp(4, 6, 1),
        sgd_config(
            1e30,
            json!({ "dir": "run", "keep_last": 1, "keep_best": "train.loss" }),
        ),
        options(dir.path(), vec![regression("train", 16, 2)], None),
    );
    let (outcome, events) = run(&t, &CancellationToken::new());
    let losses: Vec<f64> = outcome
        .history
        .iter()
        .map(|h| h.metrics.get("train.loss").unwrap())
        .collect();
    eprintln!("{losses:?}");
    assert!(
        losses[0].is_finite() && losses.iter().any(|l| l.is_nan()),
        "{losses:?}"
    );
    let best = outcome.best.unwrap();
    assert_eq!((best.epoch, best.step), (0, 1));
    assert_eq!(best.value.to_bits(), losses[0].to_bits());
    assert!(best.checkpoint.ends_with("step-000000000001"));
    // keep_last 1 keeps the newest and the best.
    assert_eq!(
        listing(&dir.join("run")),
        ["step-000000000001", "step-000000000005"]
    );
    let removed = events
        .iter()
        .filter(|e| e.name() == "checkpoint_removed")
        .count();
    assert_eq!(removed, 3);
    // state.json writes the non-finite values as strings.
    let state = std::fs::read_to_string(dir.join("run/step-000000000005/state.json")).unwrap();
    assert!(state.contains("\"train.loss\":\"nan\""), "{state}");

    // Every value NaN: no best, nothing kept besides the newest.
    let dir = TempDir::new("keep-all-nan");
    let mut data = regression("train", 16, 2);
    data.fields[0].1 = f32_data(&[16, 4], &[f32::NAN; 64]);
    let t = trainer(
        &mlp(4, 6, 1),
        sgd_config(
            0.01,
            json!({ "dir": "run", "keep_last": 1, "keep_best": "train.loss" }),
        ),
        options(dir.path(), vec![data], None),
    );
    let (outcome, _) = run(&t, &CancellationToken::new());
    assert!(outcome.best.is_none());
    assert_eq!(listing(&dir.join("run")), ["step-000000000005"]);
}

#[test]
fn keep_best_tie_keeps_the_earlier() {
    // lr = 0: every epoch has the same train.loss, so the first checkpoint stays best.
    let dir = TempDir::new("keep-tie");
    let t = trainer(
        &mlp(4, 6, 1),
        sgd_config(
            0.0,
            json!({ "dir": "run", "keep_last": 2, "keep_best": "train.loss" }),
        ),
        options(dir.path(), vec![regression("train", 16, 2)], None),
    );
    let (outcome, _) = run(&t, &CancellationToken::new());
    let best = outcome.best.unwrap();
    assert_eq!((best.epoch, best.step), (0, 1));
    assert_eq!(
        listing(&dir.join("run")),
        [
            "step-000000000001",
            "step-000000000004",
            "step-000000000005"
        ]
    );
    // The same for a metric where larger is better.
    let dir = TempDir::new("keep-tie-acc");
    let t = trainer(
        &mlp(4, 6, 3),
        classification_config(json!({ "epochs": 4, "batch_size": 32,
            "optimizer": { "kind": "sgd", "lr": 0.0 },
            "checkpoint": { "dir": "run", "keep_last": 1, "keep_best": "train.accuracy" } })),
        options(dir.path(), vec![blobs("train", 32, 2)], None),
    );
    let (outcome, _) = run(&t, &CancellationToken::new());
    assert_eq!(outcome.best.unwrap().step, 1);
    assert_eq!(
        listing(&dir.join("run")),
        ["step-000000000001", "step-000000000004"]
    );
    // keep_last null keeps everything.
    let dir = TempDir::new("keep-all");
    let t = trainer(
        &mlp(4, 6, 1),
        sgd_config(0.0, json!({ "dir": "run", "keep_last": null })),
        options(dir.path(), vec![regression("train", 16, 2)], None),
    );
    run(&t, &CancellationToken::new());
    assert_eq!(listing(&dir.join("run")).len(), 5);
}

#[test]
fn leftovers_and_foreign_files() {
    let dir = TempDir::new("leftovers");
    let run_dir = dir.join("run");
    std::fs::create_dir_all(run_dir.join(".tmp-step-000000000003")).unwrap();
    std::fs::write(run_dir.join(".tmp-step-000000000003/state.json"), "{").unwrap();
    std::fs::create_dir_all(run_dir.join("step-5")).unwrap();
    std::fs::create_dir_all(run_dir.join(".tmp-other")).unwrap();
    std::fs::write(run_dir.join("notes.txt"), "mine").unwrap();
    std::fs::write(run_dir.join("step-000000000001.txt"), "mine too").unwrap();
    let t = trainer(
        &mlp(4, 6, 1),
        sgd_config(0.01, json!({ "dir": "run", "keep_last": 1 })),
        options(dir.path(), vec![regression("train", 16, 2)], None),
    );
    run(&t, &CancellationToken::new());
    assert_eq!(
        listing(&run_dir),
        [
            ".tmp-other",
            "notes.txt",
            "step-000000000001.txt",
            "step-000000000005",
            "step-5"
        ]
    );
    assert_eq!(
        std::fs::read_to_string(run_dir.join("notes.txt")).unwrap(),
        "mine"
    );
    // A checkpoint's four files.
    assert_eq!(
        listing(&run_dir.join("step-000000000005")),
        [
            "graph.json",
            "optimizer.safetensors",
            "state.json",
            "weights.safetensors"
        ]
    );
    // Resume ignores `.tmp-step-*` and foreign entries; a run directory without checkpoints is an
    // error.
    std::fs::create_dir_all(run_dir.join(".tmp-step-000000000009")).unwrap();
    let options = || ResumeOptions {
        datasets: vec![regression("train", 16, 2)],
        ..ResumeOptions::default()
    };
    let t = Trainer::resume(&run_dir, options(), &cpu, &CancellationToken::new()).unwrap();
    assert!(t.info().finished);
    assert_eq!(t.info().step, 5);
    let empty = dir.join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let e = Trainer::resume(&empty, options(), &cpu, &CancellationToken::new()).unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput);
    assert_eq!(e.file(), Some(empty.as_path()));
    // A checkpoint whose directory name doesn't match its step.
    copy_dir(
        &run_dir.join("step-000000000005"),
        &dir.join("x/step-000000000004"),
    );
    let e = Trainer::resume(
        dir.join("x/step-000000000004"),
        options(),
        &cpu,
        &CancellationToken::new(),
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
    // Data that changed since the checkpoint.
    let other = ResumeOptions {
        datasets: vec![regression("train", 16, 3)],
        ..ResumeOptions::default()
    };
    let e = Trainer::resume(&run_dir, other, &cpu, &CancellationToken::new()).unwrap_err();
    assert!(e.message().contains("the training data changed"), "{e}");
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

/// A deterministic byte stream (splitmix64).
fn rng(seed: u64) -> impl FnMut() -> u64 {
    let mut s = seed;
    move || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Every JSON pointer of `v` (objects' members and array elements).
fn pointers(v: &Value, at: String, out: &mut Vec<String>) {
    match v {
        Value::Object(m) => {
            for (k, c) in m {
                let p = format!("{at}/{}", k.replace('~', "~0").replace('/', "~1"));
                out.push(p.clone());
                pointers(c, p, out);
            }
        }
        Value::Array(a) => {
            for (i, c) in a.iter().enumerate() {
                let p = format!("{at}/{i}");
                out.push(p.clone());
                pointers(c, p, out);
            }
        }
        _ => {}
    }
}

#[test]
fn the_loader_survives_hostile_checkpoints() {
    // A real checkpoint, mid-epoch, with Adam state, `keep_best` and a validation set.
    let dir = TempDir::new("fuzz");
    let config = classification_config(json!({
        "epochs": 3, "batch_size": 8,
        "data": { "train": { "kind": "tensors", "name": "train" },
                  "validation": { "kind": "split", "fraction": 0.25 } },
        "optimizer": { "kind": "adam", "lr": 0.01 },
        "checkpoint": { "dir": "run", "every_steps": 5, "keep_last": null, "keep_best": "val.accuracy" }
    }));
    let t = trainer(
        &mlp(4, 5, 3),
        config,
        options(dir.path(), vec![blobs("train", 32, 5)], None),
    );
    let cancel = CancellationToken::new();
    t.run(&cancel, |e| {
        if matches!(e, TrainingEvent::Step { step: 4, .. }) {
            cancel.cancel();
        }
    })
    .unwrap();
    drop(t);
    let ckpt = dir.join("run/step-000000000005");
    let resume = || {
        let options = ResumeOptions {
            datasets: vec![blobs("train", 32, 5)],
            ..ResumeOptions::default()
        };
        Trainer::resume(&ckpt, options, &cpu, &CancellationToken::new())
    };
    resume().expect("the real checkpoint loads");

    let mut tried = 0usize;
    let mut errors = 0usize;
    let mut attempt = |file: &str, bytes: &[u8], what: &str, must_fail: bool| {
        let path = ckpt.join(file);
        let original = std::fs::read(&path).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let result = catch_unwind(AssertUnwindSafe(resume));
        std::fs::write(&path, &original).unwrap();
        let result = result.unwrap_or_else(|_| panic!("{file}: {what}: the loader panicked"));
        tried += 1;
        match result {
            Err(e) => {
                errors += 1;
                assert!(
                    matches!(
                        e.kind(),
                        TrainErrorKind::InvalidInput
                            | TrainErrorKind::Io
                            | TrainErrorKind::UnsupportedOperation
                    ),
                    "{file}: {what}: {e} ({:?})",
                    e.kind()
                );
                assert!(e.file().is_some(), "{file}: {what}: no file in {e}");
            }
            Ok(_) => assert!(!must_fail, "{file}: {what}: loaded"),
        }
    };
    let mut next = rng(42);
    for file in [
        "state.json",
        "graph.json",
        "weights.safetensors",
        "optimizer.safetensors",
    ] {
        let original = std::fs::read(ckpt.join(file)).unwrap();
        let n = original.len();
        // Every truncation (strided for the larger files): always an error.
        let stride = (n / 400).max(1);
        for len in (0..n).step_by(stride) {
            attempt(file, &original[..len], &format!("truncated to {len}"), true);
        }
        // A one-bit flip at every byte (strided): an error or a checkpoint that still loads.
        for i in (0..n).step_by(stride) {
            let mut b = original.clone();
            b[i] ^= 1 << (i % 8);
            attempt(
                file,
                &b,
                &format!("bit {} of byte {i} flipped", i % 8),
                false,
            );
        }
        // Random 4-byte corruptions.
        for _ in 0..150 {
            let mut b = original.clone();
            let i = (next() as usize) % n;
            for (j, byte) in b[i..(i + 4).min(n)].iter_mut().enumerate() {
                *byte = (next() >> (8 * j)) as u8;
            }
            attempt(file, &b, &format!("4 bytes at {i} replaced"), false);
        }
        // Grown: trailing bytes (a header that claims less than the file holds, or JSON garbage).
        let mut b = original.clone();
        b.extend_from_slice(&[0xFF; 64]);
        attempt(file, &b, "64 bytes appended", false);
    }
    // Edited fields of state.json: every value replaced by values of other types and ranges,
    // and every member removed (all are required: an error).
    let text = std::fs::read_to_string(ckpt.join("state.json")).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let mut all = Vec::new();
    pointers(&doc, String::new(), &mut all);
    let replacements = [
        json!(null),
        json!(-1),
        json!(0),
        json!(1),
        json!(u64::MAX),
        json!(1e300),
        json!(0.5),
        json!("x"),
        json!("nan"),
        json!([]),
        json!({}),
        json!(true),
    ];
    for p in &all {
        for r in &replacements {
            let mut d = doc.clone();
            *d.pointer_mut(p).unwrap() = r.clone();
            attempt(
                "state.json",
                d.to_string().as_bytes(),
                &format!("{p} = {r}"),
                false,
            );
        }
        let (parent, key) = p.rsplit_once('/').unwrap();
        let mut d = doc.clone();
        let removed = match d.pointer_mut(parent).unwrap() {
            Value::Object(m) => m
                .remove(&key.replace("~1", "/").replace("~0", "~"))
                .is_some(),
            _ => false,
        };
        if removed {
            // Removing a member of `metrics`, `correct`, `counted` or `inputs` changes names
            // instead of removing a required field, but must fail too (the names must match).
            attempt(
                "state.json",
                d.to_string().as_bytes(),
                &format!("{p} removed"),
                true,
            );
        }
    }
    // Counters that disagree, a fingerprint of other data, a huge state.json.
    for (p, v) in [
        ("/step", json!(6)),
        ("/batch", json!(2)),
        ("/epoch", json!(1)),
        ("/steps_per_epoch", json!(4)),
        ("/data_fingerprint", json!("0000000000000000")),
        ("/data_fingerprint", json!("ABCDEF0123456789")),
        ("/train_samples", json!(25)),
        ("/epoch_sums/samples", json!(31)),
        ("/config/epochs", json!(0)),
        (
            "/history",
            json!([{ "epoch": 0, "step": 3, "metrics": {} }]),
        ),
        (
            "/best",
            json!({
            "metric": "val.accuracy", "value": 1.0, "epoch": 0, "step": 3, "checkpoint": "step-000000000003" }),
        ),
        ("/format_version", json!(2)),
        ("/format", json!("polypous.training")),
        ("/config/precision", json!("f16")),
    ] {
        let mut d = doc.clone();
        *d.pointer_mut(p).unwrap() = v.clone();
        attempt(
            "state.json",
            d.to_string().as_bytes(),
            &format!("{p} = {v}"),
            true,
        );
    }
    let mut big = text.clone().into_bytes();
    big.resize((64 << 20) + 1, b' ');
    attempt("state.json", &big, "64 MiB + 1", true);
    eprintln!("{tried} hostile checkpoints, {errors} errors");
    assert!(
        errors * 10 >= tried * 9,
        "{errors} of {tried} gave an error"
    );
}

#[test]
fn optimizer_state_must_fit() {
    use polypous_formats::safetensors::{Dtype, SafetensorsFile, WriteTensor, write_file};
    use std::collections::BTreeMap;
    let dir = TempDir::new("opt-state");
    let config = classification_config(
        json!({ "epochs": 1, "optimizer": { "kind": "adam" }, "checkpoint": { "dir": "run" } }),
    );
    let t = trainer(
        &mlp(4, 5, 3),
        config,
        options(dir.path(), vec![blobs("train", 16, 5)], None),
    );
    run(&t, &CancellationToken::new());
    let ckpt = dir.join("run/step-000000000002");
    let path = ckpt.join("optimizer.safetensors");
    let original = std::fs::read(&path).unwrap();
    let file = SafetensorsFile::from_bytes(original.clone()).unwrap();
    let tensors: Vec<(String, Vec<usize>, Vec<u8>)> = file
        .names()
        .map(|n| {
            let v = file.tensor(n).unwrap();
            (n.to_owned(), v.shape().to_vec(), v.data().to_vec())
        })
        .collect();
    let resume = || {
        Trainer::resume(
            &ckpt,
            ResumeOptions {
                datasets: vec![blobs("train", 16, 5)],
                ..ResumeOptions::default()
            },
            &cpu,
            &CancellationToken::new(),
        )
    };
    let rewrite = |keep: &dyn Fn(&str) -> Option<String>| {
        let entries: Vec<(String, WriteTensor<'_>)> = tensors
            .iter()
            .filter_map(|(n, s, b)| {
                keep(n).map(|n| (n, WriteTensor::new(Dtype::F32, s, b).unwrap()))
            })
            .collect();
        write_file(&path, entries, &BTreeMap::new()).unwrap();
    };
    // Only one of a param's two states.
    rewrite(&|n| (n != "fc1.weight:exp_avg").then(|| n.to_owned()));
    let e = resume().unwrap_err();
    assert_eq!(e.tensor(), Some("fc1.weight"), "{e}");
    // A state name of another optimizer, a param that isn't trainable or doesn't exist.
    for (from, to) in [
        ("fc1.weight:exp_avg", "fc1.weight:momentum_buffer"),
        ("fc1.bias:exp_avg", "nope:exp_avg"),
        ("fc1.bias:exp_avg", "fc1.bias"),
    ] {
        rewrite(&|n| {
            Some(if n == from {
                to.to_owned()
            } else {
                n.to_owned()
            })
        });
        let e = resume().unwrap_err();
        assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
        assert_eq!(e.tensor(), Some(to), "{e}");
    }
    // No state at all at step > 0 loads (all or none per param), but is refused at the first
    // step (§5.4 step 4). The checkpoint above is of a finished run, so use one of a run of 2
    // epochs.
    std::fs::write(&path, &original).unwrap();
    let dir2 = TempDir::new("opt-state-2");
    let config = classification_config(
        json!({ "epochs": 2, "optimizer": { "kind": "adam" }, "checkpoint": { "dir": "run", "keep_last": null } }),
    );
    let t = trainer(
        &mlp(4, 5, 3),
        config,
        options(dir2.path(), vec![blobs("train", 16, 5)], None),
    );
    run(&t, &CancellationToken::new());
    let ckpt2 = dir2.join("run/step-000000000002");
    write_file(
        ckpt2.join("optimizer.safetensors"),
        Vec::<(String, WriteTensor<'_>)>::new(),
        &BTreeMap::new(),
    )
    .unwrap();
    let t = Trainer::resume(
        &ckpt2,
        ResumeOptions {
            datasets: vec![blobs("train", 16, 5)],
            discard_newer: true,
            ..ResumeOptions::default()
        },
        &cpu,
        &CancellationToken::new(),
    )
    .unwrap();
    let e = t.run(&CancellationToken::new(), |_| {}).unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
    assert_eq!(e.tensor(), Some("fc1.bias"), "{e}");
}
