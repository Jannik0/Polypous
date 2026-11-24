//! The checks of `docs/specs/training.md` §2.10 when a trainer is created: one test per step's
//! errors, and tests that the order holds (the first failing step wins).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::collections::BTreeMap;
use std::path::Path;

use common::*;
use polypous_backend::device::DeviceId;
use polypous_formats::safetensors::{Dtype, WriteTensor, write_file};
use polypous_runtime::CancellationToken;
use polypous_train::{
    InMemoryDataset, TrainError, TrainErrorKind, Trainer, TrainerOptions, TrainingConfig,
};
use serde_json::{Value, json};

fn create(
    graph_doc: &Value,
    config: TrainingConfig,
    options: TrainerOptions,
) -> Result<Trainer, TrainError> {
    Trainer::new(
        &graph(graph_doc),
        config,
        options,
        &cpu,
        &CancellationToken::new(),
    )
}

fn err(graph_doc: &Value, config: TrainingConfig, options: TrainerOptions) -> TrainError {
    create(graph_doc, config, options).expect_err("creating the trainer should fail")
}

fn base() -> TempDir {
    TempDir::new("validation")
}

fn blobs_opts(dir: &TempDir) -> TrainerOptions {
    options(dir.path(), vec![blobs("train", 30, 1)], None)
}

#[track_caller]
fn assert_err(e: &TrainError, kind: TrainErrorKind, pointer: Option<&str>, tensor: Option<&str>) {
    assert_eq!(e.kind(), kind, "{e}");
    assert_eq!(e.pointer(), pointer, "{e}");
    assert_eq!(e.tensor(), tensor, "{e}");
}

fn write_safetensors(path: &Path, tensors: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
    let entries: Vec<(&str, WriteTensor<'_>)> = tensors
        .iter()
        .map(|(n, d, s, b)| (*n, WriteTensor::new(*d, s, b).unwrap()))
        .collect();
    write_file(path, entries, &BTreeMap::new()).unwrap();
}

#[test]
fn step_1_config() {
    let dir = base();
    let mut c = classification_config(json!({}));
    c.batch_size = 0;
    assert_err(
        &err(&mlp(4, 8, 3), c, blobs_opts(&dir)),
        TrainErrorKind::InvalidInput,
        Some("/batch_size"),
        None,
    );
    let mut c = classification_config(json!({}));
    c.precision = "f16".into();
    assert_err(
        &err(&mlp(4, 8, 3), c, blobs_opts(&dir)),
        TrainErrorKind::UnsupportedOperation,
        Some("/precision"),
        None,
    );
}

fn state_graph() -> Value {
    json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1 },
        "inputs": [ { "name": "x", "dtype": "f32", "shape": [2] } ],
        "state": [ { "name": "s", "dtype": "f32", "shape": [2] } ],
        "outputs": [ { "name": "y", "value": "y" } ],
        "nodes": [ { "id": "y", "op": "core.add", "inputs": { "a": "s", "b": "x" } } ],
        "state_updates": { "s": "y" }
    })
}

#[test]
fn step_2_graph_with_state() {
    let dir = base();
    let e = err(
        &state_graph(),
        classification_config(json!({})),
        blobs_opts(&dir),
    );
    assert_err(&e, TrainErrorKind::UnsupportedOperation, None, Some("s"));
}

#[test]
fn step_3_device() {
    let dir = base();
    // The resolver's error is returned as it is.
    let refuse = |d: &str| -> Result<DeviceId, TrainError> {
        Err(TrainError::new(TrainErrorKind::BackendUnavailable, format!("no {d}")).with_device(d))
    };
    let e = Trainer::new(
        &graph(&mlp(4, 8, 3)),
        classification_config(json!({ "device": "cuda:3" })),
        blobs_opts(&dir),
        &refuse,
        &CancellationToken::new(),
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::BackendUnavailable);
    assert_eq!(e.device(), Some("cuda:3"));
    // A device that doesn't exist (or whose backend isn't compiled in).
    {
        let e = Trainer::new(
            &graph(&mlp(4, 8, 3)),
            classification_config(json!({})),
            blobs_opts(&dir),
            &|_| Ok(DeviceId::Cuda { index: 7 }),
            &CancellationToken::new(),
        )
        .unwrap_err();
        assert_eq!(e.kind(), TrainErrorKind::BackendUnavailable, "{e}");
        assert_eq!(e.device(), Some("cuda:7"), "{e}");
    }
}

#[test]
fn step_4_references() {
    let dir = base();
    let g = mlp(4, 8, 3);
    let cases = [
        (
            json!({ "inputs": { "x": "x", "z": "x" } }),
            "/inputs/z",
            Some("z"),
        ),
        (json!({ "inputs": {} }), "/inputs", Some("x")),
        (
            json!({ "loss": { "kind": "cross_entropy", "prediction": "logits", "target": "label" } }),
            "/loss/prediction",
            Some("logits"),
        ),
        (
            json!({ "metrics": [ { "kind": "accuracy", "prediction": "y", "target": "label" },
                              { "kind": "accuracy", "prediction": "p", "target": "label", "name": "b" } ] }),
            "/metrics/1/prediction",
            Some("p"),
        ),
        (
            json!({ "checkpoint": { "dir": "r", "keep_best": "train.acc" } }),
            "/checkpoint/keep_best",
            None,
        ),
        (
            json!({ "checkpoint": { "dir": "r", "keep_best": "val.accuracy" } }),
            "/checkpoint/keep_best",
            None,
        ),
    ];
    for (extra, pointer, tensor) in cases {
        let e = err(&g, classification_config(extra.clone()), blobs_opts(&dir));
        assert_err(&e, TrainErrorKind::InvalidInput, Some(pointer), tensor);
        if pointer == "/checkpoint/keep_best" && extra.to_string().contains("val.") {
            assert!(e.message().contains("needs a validation set"), "{e}");
        }
    }
    // keep_best with the reported names that exist.
    create(
        &g,
        classification_config(
            json!({ "checkpoint": { "dir": "r1", "keep_best": "train.accuracy" } }),
        ),
        blobs_opts(&dir),
    )
    .unwrap();
}

#[test]
fn step_5_data() {
    let dir = base();
    let g = mlp(4, 8, 3);
    let train_val = json!({ "data": { "train": { "kind": "tensors", "name": "train" },
                                     "validation": { "kind": "tensors", "name": "val" } } });
    // A field missing from the training set; from the validation set; a dtype that differs.
    let mut ds = blobs("train", 30, 1);
    ds.fields.push(("other".into(), i64_data(&[0; 30])));
    let e = err(
        &g,
        classification_config(json!({ "inputs": { "x": "image" } })),
        options(dir.path(), vec![ds.clone()], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("image"),
    );
    assert!(e.message().contains("training set"), "{e}");
    let mut val = blobs("val", 9, 2);
    val.fields.retain(|(n, _)| n != "label");
    let e = err(
        &g,
        classification_config(train_val.clone()),
        options(dir.path(), vec![ds.clone(), val], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/loss/target"),
        Some("label"),
    );
    assert!(e.message().contains("validation set"), "{e}");
    let mut val = blobs("val", 9, 2);
    val.fields[0].1 = f32_data(&[9, 2, 2], &[0.0; 36]);
    let e = err(
        &g,
        classification_config(train_val.clone()),
        options(dir.path(), vec![ds.clone(), val], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("x"),
    );
    // Dtypes that don't fit their use: an i64 field to a float input, f32 class indices.
    let e = err(
        &g,
        classification_config(json!({ "inputs": { "x": "other" } })),
        options(dir.path(), vec![ds.clone()], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("other"),
    );
    let mut bad = blobs("train", 30, 1);
    bad.fields[1].1 = f32_data(&[30], &[0.0; 30]);
    let e = err(
        &g,
        classification_config(json!({})),
        options(dir.path(), vec![bad], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/loss/target"),
        Some("label"),
    );
    // A dataset with none of the used fields: the first use is reported with its pointer.
    let none = InMemoryDataset {
        name: "train".into(),
        fields: vec![("a".into(), i64_data(&[0; 4]))],
    };
    let e = err(
        &g,
        classification_config(json!({})),
        options(dir.path(), vec![none], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("x"),
    );
    // File errors: Io and malformed, with the file.
    let missing = dir.join("missing.safetensors");
    let e = err(
        &g,
        classification_config(
            json!({ "data": { "train": { "kind": "safetensors", "path": "missing.safetensors" } } }),
        ),
        options(dir.path(), vec![], None),
    );
    assert_eq!(e.kind(), TrainErrorKind::Io, "{e}");
    assert_eq!(e.file(), Some(missing.as_path()));
    std::fs::write(dir.join("bad.safetensors"), b"not a safetensors file").unwrap();
    let e = err(
        &g,
        classification_config(
            json!({ "data": { "train": { "kind": "safetensors", "path": "bad.safetensors" } } }),
        ),
        options(dir.path(), vec![], None),
    );
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
    assert_eq!(e.file(), Some(dir.join("bad.safetensors").as_path()));
    // A `tensors` name the options don't have.
    let e = err(
        &g,
        classification_config(json!({})),
        options(dir.path(), vec![], None),
    );
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
}

#[test]
fn step_6_shapes_and_values() {
    let dir = base();
    // The sample shape doesn't bind: 4 features, the graph wants 5.
    let e = err(
        &mlp(5, 8, 3),
        classification_config(json!({})),
        blobs_opts(&dir),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("x"),
    );
    // The prediction doesn't fit the target: mse on [B, 3] against [B, 1].
    let mse = json!({ "loss": { "kind": "mse", "prediction": "y", "target": "y" }, "metrics": [] });
    let e = err(
        &mlp(4, 8, 3),
        classification_config(mse),
        options(dir.path(), vec![regression("train", 20, 1)], None),
    );
    assert_err(&e, TrainErrorKind::InvalidInput, Some("/loss"), Some("y"));
    // A class index out of range, on a validation sample.
    let mut val = blobs("val", 9, 2);
    val.fields[1].1 = i64_data(&[0, 1, 2, 0, 1, 3, 0, 1, 2]);
    let e = err(
        &mlp(4, 8, 3),
        classification_config(
            json!({ "data": { "train": { "kind": "tensors", "name": "train" },
                                               "validation": { "kind": "tensors", "name": "val" } } }),
        ),
        options(dir.path(), vec![blobs("train", 30, 1), val], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/loss/target"),
        Some("label"),
    );
    assert!(
        e.message().contains("validation")
            && e.message().contains("sample 5")
            && e.message().contains('3'),
        "{e}"
    );
    // `threshold` with class indices (MetricError::ThresholdWithMulticlass).
    let thr = json!({ "metrics": [ { "kind": "accuracy", "prediction": "y", "target": "label", "threshold": 0.5 } ] });
    let e = err(&mlp(4, 8, 3), classification_config(thr), blobs_opts(&dir));
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/metrics/0/threshold"),
        Some("label"),
    );
    // batch_norm with a last batch of one sample: 17 samples in batches of 8.
    let e = err(
        &bn_dropout(4, 6, 3),
        classification_config(json!({})),
        options(dir.path(), vec![blobs("train", 17, 1)], None),
    );
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput, "{e}");
    assert_eq!(e.node(), Some("bn"), "{e}");
    // 18 samples: the last batch has 2, which is fine; the validation set may have a batch of 1
    // (inference mode).
    create(
        &bn_dropout(4, 6, 3),
        classification_config(json!({})),
        options(dir.path(), vec![blobs("train", 18, 1)], None),
    )
    .unwrap();
    // A running statistic with two uses.
    let mut shared = bn_dropout(4, 6, 3);
    let nodes = shared["nodes"].as_array_mut().unwrap();
    nodes.insert(
        2,
        json!({ "id": "bn2", "op": "nn.batch_norm", "inputs": { "x": "bn" },
        "params": { "running_mean": "bn.running_mean", "running_var": "bn.running_var" } }),
    );
    nodes[3]["inputs"]["x"] = json!("bn2");
    let e = err(&shared, classification_config(json!({})), blobs_opts(&dir));
    assert_eq!(e.kind(), TrainErrorKind::UnsupportedOperation, "{e}");
    assert_eq!(e.tensor(), Some("bn.running_mean"), "{e}");
}

#[test]
fn step_7_params() {
    let dir = base();
    let g = mlp(4, 8, 3);
    // A subset of the params from a file (F16 widened), the others from their init.
    let w: Vec<u8> = (0..32)
        .flat_map(|i| {
            // f16 of i/8: exact.
            let x = i as f32 / 8.0;
            let bits = f16_bits(x);
            bits.to_le_bytes()
        })
        .collect();
    write_safetensors(
        &dir.join("w.safetensors"),
        &[("fc1.weight", Dtype::F16, vec![8, 4], w.clone())],
    );
    let mut opts = blobs_opts(&dir);
    opts.weights = Some(dir.join("w.safetensors"));
    let t = create(&g, classification_config(json!({})), opts).unwrap();
    let params = t.params().unwrap();
    let fc1 = read_f32(&params[0].1);
    assert_eq!(params[0].0, "fc1.weight");
    assert_eq!(fc1[..4], [0.0, 0.125, 0.25, 0.375]);
    // A tensor that no param declares.
    write_safetensors(
        &dir.join("x.safetensors"),
        &[("fc9.weight", Dtype::F32, vec![1], vec![0; 4])],
    );
    let mut opts = blobs_opts(&dir);
    opts.weights = Some(dir.join("x.safetensors"));
    let e = err(&g, classification_config(json!({})), opts);
    assert_err(&e, TrainErrorKind::InvalidInput, None, Some("fc9.weight"));
    assert_eq!(e.file(), Some(dir.join("x.safetensors").as_path()));
    // Wrong shape, unsupported storage.
    write_safetensors(
        &dir.join("s.safetensors"),
        &[("fc2.bias", Dtype::F32, vec![4], vec![0; 16])],
    );
    let mut opts = blobs_opts(&dir);
    opts.weights = Some(dir.join("s.safetensors"));
    assert_err(
        &err(&g, classification_config(json!({})), opts),
        TrainErrorKind::InvalidInput,
        None,
        Some("fc2.bias"),
    );
    write_safetensors(
        &dir.join("i.safetensors"),
        &[("fc2.bias", Dtype::I32, vec![3], vec![0; 12])],
    );
    let mut opts = blobs_opts(&dir);
    opts.weights = Some(dir.join("i.safetensors"));
    assert_err(
        &err(&g, classification_config(json!({})), opts),
        TrainErrorKind::InvalidInput,
        None,
        Some("fc2.bias"),
    );
    // Params without init and without weights: the specified message, all names in details.
    let mut no_init = mlp(4, 8, 3);
    for p in no_init["params"].as_array_mut().unwrap() {
        p.as_object_mut().unwrap().remove("init");
    }
    let mut opts = blobs_opts(&dir);
    opts.weights = Some(dir.join("w.safetensors"));
    let e = err(&no_init, classification_config(json!({})), opts);
    assert_eq!(
        e.message(),
        "These params have no init and no weights: fc1.bias, fc2.weight, fc2.bias"
    );
    assert_eq!(e.tensor(), Some("fc1.bias"));
    assert_eq!(e.details(), Some("fc1.bias\nfc2.weight\nfc2.bias"));
    // More than 10: the first 10, then `and N more`.
    let mut many = mlp(4, 8, 3);
    let params = many["params"].as_array_mut().unwrap();
    for i in 0..10 {
        params.push(json!({ "name": format!("extra{i:02}"), "dtype": "float", "shape": [1] }));
    }
    // Unused params need a value too; they are not used by a node (no init allowed is fine).
    let e = create(&many, classification_config(json!({})), blobs_opts(&dir)).unwrap_err();
    let names: Vec<String> = (0..10).map(|i| format!("extra{i:02}")).collect();
    assert_eq!(
        e.message(),
        format!(
            "These params have no init and no weights: {}",
            names.join(", ")
        )
    );
    for p in many["params"].as_array_mut().unwrap().iter_mut().take(2) {
        p.as_object_mut().unwrap().remove("init");
    }
    let e = create(&many, classification_config(json!({})), blobs_opts(&dir)).unwrap_err();
    assert!(e.message().ends_with("extra07 and 2 more"), "{e}");
    assert_eq!(e.details().unwrap().lines().count(), 12);
}

fn f16_bits(x: f32) -> u16 {
    // Exact for the multiples of 1/8 below 4 used here.
    if x == 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3ff;
    ((exp as u16) << 10) | mant as u16
}

#[test]
fn step_8_run_directory() {
    let dir = base();
    let g = mlp(4, 8, 3);
    let c = || classification_config(json!({ "checkpoint": { "dir": "runs/a" } }));
    // Created if missing.
    create(&g, c(), blobs_opts(&dir)).unwrap();
    assert!(dir.join("runs/a").is_dir());
    // Foreign entries are fine; a checkpoint is not.
    std::fs::write(dir.join("runs/a/notes.txt"), "x").unwrap();
    std::fs::create_dir(dir.join("runs/a/step-5")).unwrap();
    create(&g, c(), blobs_opts(&dir)).unwrap();
    std::fs::create_dir(dir.join("runs/a/step-000000000005")).unwrap();
    let e = err(&g, c(), blobs_opts(&dir));
    assert_eq!(e.kind(), TrainErrorKind::InvalidInput);
    assert_eq!(
        e.file(),
        Some(dir.join("runs/a/step-000000000005").as_path())
    );
    // A run directory that can't be created.
    std::fs::write(dir.join("runs/file"), "x").unwrap();
    let e = err(
        &g,
        classification_config(json!({ "checkpoint": { "dir": "runs/file/sub" } })),
        blobs_opts(&dir),
    );
    assert_eq!(e.kind(), TrainErrorKind::Io, "{e}");
}

#[test]
fn the_order_holds() {
    let dir = base();
    let unreadable =
        json!({ "data": { "train": { "kind": "safetensors", "path": "missing.safetensors" } } });
    // Step 4 (an unknown graph input) before step 5 (an unreadable dataset).
    let mut extra = unreadable.clone();
    merge(&mut extra, json!({ "inputs": { "x": "x", "z": "x" } }));
    let e = err(
        &mlp(4, 8, 3),
        classification_config(extra),
        options(dir.path(), vec![], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/z"),
        Some("z"),
    );
    // Step 2 (state) before step 3 (device) and step 4.
    let e = Trainer::new(
        &graph(&state_graph()),
        classification_config(json!({ "inputs": { "q": "x" } })),
        blobs_opts(&dir),
        &|_| {
            Err(TrainError::new(
                TrainErrorKind::BackendUnavailable,
                "no device",
            ))
        },
        &CancellationToken::new(),
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::UnsupportedOperation);
    // Step 3 (device) before step 4.
    let e = Trainer::new(
        &graph(&mlp(4, 8, 3)),
        classification_config(json!({ "inputs": { "q": "x" } })),
        blobs_opts(&dir),
        &|_| {
            Err(TrainError::new(
                TrainErrorKind::BackendUnavailable,
                "no device",
            ))
        },
        &CancellationToken::new(),
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::BackendUnavailable);
    // Step 5 before step 6: a missing field before a shape that doesn't bind.
    let e = err(
        &mlp(5, 8, 3),
        classification_config(json!({ "inputs": { "x": "nope" } })),
        blobs_opts(&dir),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/inputs/x"),
        Some("nope"),
    );
    // Step 6 before step 7: a bad target before params without a value.
    let mut no_init = mlp(4, 8, 3);
    no_init["params"][0].as_object_mut().unwrap().remove("init");
    let mut ds = blobs("train", 30, 1);
    ds.fields[1].1 = i64_data(&[5; 30]);
    let e = err(
        &no_init,
        classification_config(json!({})),
        options(dir.path(), vec![ds], None),
    );
    assert_err(
        &e,
        TrainErrorKind::InvalidInput,
        Some("/loss/target"),
        Some("label"),
    );
    // Step 7 before step 8.
    std::fs::create_dir_all(dir.join("r/step-000000000001")).unwrap();
    let e = err(
        &no_init,
        classification_config(json!({ "checkpoint": { "dir": "r" } })),
        blobs_opts(&dir),
    );
    assert!(e.message().starts_with("These params"), "{e}");
}

#[test]
fn cancellation_during_creation() {
    let dir = base();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let e = Trainer::new(
        &graph(&mlp(4, 8, 3)),
        classification_config(json!({ "checkpoint": { "dir": "never" } })),
        blobs_opts(&dir),
        &cpu,
        &cancel,
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::Cancelled);
    assert!(
        !dir.join("never").exists(),
        "a cancelled creation keeps nothing"
    );
    // Cancelled from the device resolver, i.e. between steps 3 and 4.
    let cancel = CancellationToken::new();
    let e = Trainer::new(
        &graph(&mlp(4, 8, 3)),
        classification_config(json!({})),
        blobs_opts(&dir),
        &|_| {
            cancel.cancel();
            Ok(DeviceId::Cpu)
        },
        &cancel,
    )
    .unwrap_err();
    assert_eq!(e.kind(), TrainErrorKind::Cancelled);
}
