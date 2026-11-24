//! Tests of the training config root (training.md §2.1–§2.9).

use serde_json::{Value, json};

use super::*;
use crate::error::TrainErrorKind;

/// The smallest valid config: only the required fields.
fn minimal() -> Value {
    json!({
        "format": "polypous.training", "format_version": 1,
        "epochs": 5, "batch_size": 64,
        "data": { "train": { "kind": "safetensors", "path": "train.safetensors" } },
        "inputs": { "image": "image" },
        "loss": { "kind": "cross_entropy", "prediction": "logits", "target": "label" },
        "optimizer": { "kind": "adamw" }
    })
}

/// The example of §2 (the first code block).
fn spec_example() -> Value {
    json!({
      "format": "polypous.training",
      "format_version": 1,
      "seed": 42,
      "device": "auto",
      "precision": "f32",
      "epochs": 5,
      "batch_size": 64,
      "data": {
        "train":      { "kind": "image_folder", "path": "data/train", "image_size": [28, 28], "channels": 1 },
        "validation": { "kind": "split", "from": "train", "fraction": 0.1 },
        "shuffle": true
      },
      "inputs":  { "image": "image" },
      "loss":    { "kind": "cross_entropy", "prediction": "logits", "target": "label", "label_smoothing": 0.0 },
      "metrics": [ { "kind": "accuracy", "prediction": "logits", "target": "label" } ],
      "optimizer": { "kind": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 },
      "schedule":  { "kind": "cosine", "warmup_steps": 100, "min_lr": 1e-5 },
      "gradient_clip": { "max_norm": 1.0 },
      "checkpoint": { "dir": "runs/mnist", "every_epochs": 1, "keep_last": 3, "keep_best": "val.accuracy" }
    })
}

fn parse(doc: &Value) -> Result<TrainingConfig, TrainError> {
    TrainingConfig::from_json(&doc.to_string())
}

/// `doc` with the value at `pointer` set to `value` (a new member of an object, or a replaced
/// array element).
fn with(doc: &Value, pointer: &str, value: Value) -> Value {
    let mut doc = doc.clone();
    let (parent, key) = pointer.rsplit_once('/').unwrap();
    let key = key.replace("~1", "/").replace("~0", "~");
    match doc.pointer_mut(parent).unwrap() {
        Value::Object(o) => {
            o.insert(key, value);
        }
        Value::Array(a) => a[key.parse::<usize>().unwrap()] = value,
        other => panic!("{parent} is {other}"),
    }
    doc
}

fn without(doc: &Value, pointer: &str) -> Value {
    let mut doc = doc.clone();
    let (parent, key) = pointer.rsplit_once('/').unwrap();
    doc.pointer_mut(parent)
        .unwrap()
        .as_object_mut()
        .unwrap()
        .remove(key);
    doc
}

#[track_caller]
fn fails_at(doc: &Value, pointer: &str) -> TrainError {
    let err = parse(doc).expect_err("should fail");
    assert_eq!(err.pointer(), Some(pointer), "{err}");
    assert_eq!(err.kind(), TrainErrorKind::InvalidInput, "{err}");
    err
}

#[test]
fn defaults_are_filled_in() {
    let c = parse(&minimal()).unwrap();
    assert_eq!(c.seed, 0);
    assert_eq!(c.device, "auto");
    assert_eq!(c.precision, "f32");
    assert!(c.metrics.is_empty());
    assert_eq!(c.schedule, ScheduleConfig::Constant {});
    assert_eq!(c.gradient_clip, None);
    assert_eq!(c.checkpoint, None);
    assert!(c.data.shuffle);
    assert_eq!(c.data.validation, None);
    assert_eq!(c.data.max_bytes, 1 << 32);
    let OptimizerConfig::Adamw(o) = c.optimizer else {
        panic!()
    };
    assert_eq!(o, AdamWConfig::default());
    // The checkpoint object's defaults (§2.9).
    let c = parse(&with(&minimal(), "/checkpoint", json!({ "dir": "runs/x" }))).unwrap();
    assert_eq!(c.checkpoint, Some(CheckpointConfig::new("runs/x")));
    let k = c.checkpoint.unwrap();
    assert_eq!(
        (
            k.every_epochs,
            k.every_steps,
            k.keep_last,
            k.keep_best,
            k.on_cancel
        ),
        (Some(1), None, Some(3), None, true)
    );
    // Null where the type says "or null".
    let c = parse(&with(
        &minimal(),
        "/checkpoint",
        json!({ "dir": "d", "every_epochs": null, "keep_last": null, "every_steps": 7 }),
    ))
    .unwrap();
    let k = c.checkpoint.unwrap();
    assert_eq!(
        (k.every_epochs, k.every_steps, k.keep_last),
        (None, Some(7), None)
    );
    let c = parse(&with(&minimal(), "/gradient_clip", Value::Null)).unwrap();
    assert_eq!(c.gradient_clip, None);
}

#[test]
fn the_spec_example_and_the_canonical_round_trip() {
    for doc in [minimal(), spec_example()] {
        let c = parse(&doc).unwrap();
        let text = c.to_json();
        let again = TrainingConfig::from_json(&text).unwrap();
        assert_eq!(again, c);
        assert_eq!(again.to_json(), text, "to_json is canonical");
    }
    // Every field, in table order.
    let text = parse(&minimal()).unwrap().to_json();
    let keys: Vec<String> = serde_json::from_str::<serde_json::Map<String, Value>>(&text)
        .unwrap()
        .keys()
        .cloned()
        .collect();
    // serde_json's Map sorts keys; check the order in the text instead.
    let order = [
        "\"format\"",
        "\"format_version\"",
        "\"seed\"",
        "\"device\"",
        "\"precision\"",
        "\"epochs\"",
        "\"batch_size\"",
        "\"data\"",
        "\"inputs\"",
        "\"loss\"",
        "\"metrics\"",
        "\"optimizer\"",
        "\"schedule\"",
        "\"gradient_clip\"",
        "\"checkpoint\"",
    ];
    let positions: Vec<usize> = order.iter().map(|k| text.find(k).unwrap()).collect();
    assert!(positions.windows(2).all(|w| w[0] < w[1]), "{text}");
    assert_eq!(keys.len(), order.len());
    assert!(
        text.contains("\"gradient_clip\":null,\"checkpoint\":null"),
        "{text}"
    );
    assert!(
        text.contains("\"schedule\":{\"kind\":\"constant\"}"),
        "{text}"
    );
    // The spec example's nested objects, with defaults filled in.
    let text = parse(&spec_example()).unwrap().to_json();
    assert!(text.contains(r#""checkpoint":{"dir":"runs/mnist","every_epochs":1,"every_steps":null,"keep_last":3,"keep_best":"val.accuracy","on_cancel":true}"#), "{text}");
    assert!(
        text.contains(r#""validation":{"kind":"split","from":"train","fraction":0.1}"#),
        "{text}"
    );
}

#[test]
fn numbers_are_the_nearest_f64_and_round_trip_bit_exactly() {
    for x in [
        "0.1",
        "1e-5",
        "0.00010000000000000003",
        "3.0000000000000004e-4",
    ] {
        let doc = with(
            &minimal(),
            "/optimizer",
            json!({ "kind": "sgd", "lr": serde_json::Value::from(0.5) }),
        );
        let text = doc.to_string().replace("0.5", x);
        let c = TrainingConfig::from_json(&text).unwrap();
        let want: f64 = x.parse().unwrap();
        assert_eq!(c.optimizer.lr().to_bits(), want.to_bits(), "{x}");
        let again = TrainingConfig::from_json(&c.to_json()).unwrap();
        assert_eq!(
            again.optimizer.lr().to_bits(),
            want.to_bits(),
            "{x} after to_json"
        );
    }
    // An integer field takes a JSON integer, not 5.0 or 5e0.
    for bad in [json!(5.0), json!(-1), json!("5"), json!(true)] {
        fails_at(&with(&minimal(), "/epochs", bad), "/epochs");
    }
    // A float field takes an integer too.
    let c = parse(&with(
        &minimal(),
        "/gradient_clip",
        json!({ "max_norm": 2 }),
    ))
    .unwrap();
    assert_eq!(c.gradient_clip.unwrap().max_norm, 2.0);
    // seed is a u64.
    let text = minimal()
        .to_string()
        .replace("\"epochs\"", "\"seed\":18446744073709551615,\"epochs\"");
    assert_eq!(TrainingConfig::from_json(&text).unwrap().seed, u64::MAX);
    let text = minimal()
        .to_string()
        .replace("\"epochs\"", "\"seed\":18446744073709551616,\"epochs\"");
    assert_eq!(
        TrainingConfig::from_json(&text).unwrap_err().pointer(),
        Some("/seed")
    );
}

#[test]
fn root_ranges() {
    let m = minimal();
    for (pointer, ok, bad) in [
        (
            "/epochs",
            vec![json!(1), json!(10_000)],
            vec![json!(0), json!(10_001)],
        ),
        (
            "/batch_size",
            vec![json!(1), json!(65_536)],
            vec![json!(0), json!(65_537)],
        ),
        (
            "/device",
            vec![
                json!("auto"),
                json!("cpu"),
                json!("cuda:0"),
                json!("vulkan:discrete_gpu:1"),
            ],
            vec![json!("gpu"), json!(""), json!("cuda:x"), json!(3)],
        ),
    ] {
        for v in ok {
            parse(&with(&m, pointer, v.clone())).unwrap_or_else(|e| panic!("{pointer} {v}: {e}"));
        }
        for v in bad {
            fails_at(&with(&m, pointer, v), pointer);
        }
    }
}

#[test]
fn precision_kinds() {
    let m = minimal();
    parse(&with(&m, "/precision", json!("f32"))).unwrap();
    for p in ["f16", "bf16"] {
        let err = parse(&with(&m, "/precision", json!(p))).unwrap_err();
        assert_eq!(err.kind(), TrainErrorKind::UnsupportedOperation, "{err}");
        assert_eq!(err.pointer(), Some("/precision"));
    }
    for p in [json!("f64"), json!("F32"), json!(""), json!(32)] {
        fails_at(&with(&m, "/precision", p), "/precision");
    }
    // In table order: an earlier invalid field wins over precision, precision over a later one.
    let doc = with(&with(&m, "/precision", json!("f16")), "/epochs", json!(0));
    assert_eq!(
        parse(&doc).unwrap_err().kind(),
        TrainErrorKind::UnsupportedOperation
    );
    let doc = with(&with(&m, "/precision", json!("f16")), "/device", json!("x"));
    assert_eq!(parse(&doc).unwrap_err().pointer(), Some("/device"));
}

#[test]
fn checkpoint_ranges() {
    let base = with(&minimal(), "/checkpoint", json!({ "dir": "runs" }));
    for (field, max, over) in [
        ("every_epochs", json!(10_000), json!(10_001)),
        ("every_steps", json!(1u64 << 32), json!((1u64 << 32) + 1)),
        ("keep_last", json!(10_000), json!(10_001)),
    ] {
        let at = format!("/checkpoint/{field}");
        parse(&with(&base, &at, json!(1))).unwrap();
        parse(&with(&base, &at, max)).unwrap();
        parse(&with(&base, &at, Value::Null)).unwrap();
        fails_at(&with(&base, &at, json!(0)), &at);
        fails_at(&with(&base, &at, over), &at);
        fails_at(&with(&base, &at, json!(1.5)), &at);
    }
    fails_at(
        &with(&base, "/checkpoint/dir", json!("")),
        "/checkpoint/dir",
    );
    fails_at(
        &with(&base, "/checkpoint/dir", json!("a".repeat(4097))),
        "/checkpoint/dir",
    );
    parse(&with(&base, "/checkpoint/dir", json!("a".repeat(4096)))).unwrap();
    fails_at(
        &with(&base, "/checkpoint/keep_best", json!("")),
        "/checkpoint/keep_best",
    );
    fails_at(
        &with(&base, "/checkpoint/on_cancel", json!(1)),
        "/checkpoint/on_cancel",
    );
    fails_at(&without(&base, "/checkpoint/dir"), "/checkpoint/dir");
    fails_at(
        &with(&base, "/checkpoint/every", json!(1)),
        "/checkpoint/every",
    );
    fails_at(
        &with(&minimal(), "/checkpoint", json!("runs")),
        "/checkpoint",
    );
}

#[test]
fn gradient_clip_and_inputs() {
    let m = minimal();
    for bad in [json!(0), json!(-1.0), json!(1e39)] {
        fails_at(
            &with(&m, "/gradient_clip", json!({ "max_norm": bad })),
            "/gradient_clip/max_norm",
        );
    }
    fails_at(
        &with(&m, "/gradient_clip", json!({})),
        "/gradient_clip/max_norm",
    );
    fails_at(
        &with(
            &m,
            "/gradient_clip",
            json!({ "max_norm": 1, "norm_type": 2 }),
        ),
        "/gradient_clip/norm_type",
    );
    // inputs: names 1–256 bytes, values strings; the key is escaped in the pointer.
    fails_at(&with(&m, "/inputs", json!({ "a/b": "" })), "/inputs/a~1b");
    fails_at(&with(&m, "/inputs", json!({ "x": 1 })), "/inputs/x");
    fails_at(
        &with(&m, "/inputs", json!({ "x": "a".repeat(257) })),
        "/inputs/x",
    );
    parse(&with(&m, "/inputs", json!({ "x": "a".repeat(256) }))).unwrap();
    let many: serde_json::Map<String, Value> =
        (0..4097).map(|i| (format!("i{i}"), json!("f"))).collect();
    fails_at(&with(&m, "/inputs", Value::Object(many)), "/inputs");
    fails_at(&with(&m, "/inputs", json!(["x"])), "/inputs");
}

#[test]
fn unknown_missing_and_mistyped_fields_have_full_pointers() {
    let m = spec_example();
    for (pointer, value) in [
        ("/extra", json!(1)),
        ("/data/extra", json!(1)),
        ("/data/train/extra", json!(1)),
        ("/data/validation/extra", json!(1)),
        ("/loss/momentum", json!(1)),
        ("/metrics/0/extra", json!(1)),
        ("/optimizer/momentum", json!(0.9)),
        ("/schedule/gamma", json!(0.1)),
        ("/checkpoint/extra", json!(1)),
    ] {
        fails_at(&with(&m, pointer, value), pointer);
    }
    for pointer in [
        "/epochs",
        "/batch_size",
        "/data",
        "/inputs",
        "/loss",
        "/optimizer",
        "/data/train",
        "/data/train/path",
        "/loss/kind",
        "/loss/target",
        "/metrics/0/prediction",
        "/optimizer/kind",
        "/format",
        "/format_version",
    ] {
        fails_at(&without(&m, pointer), pointer);
    }
    for (pointer, value) in [
        ("/seed", json!("1")),
        ("/data/shuffle", json!(1)),
        ("/data/train/image_size", json!([28])),
        ("/data/train/image_size/1", json!(28.5)),
        ("/optimizer/betas", json!([0.9])),
        ("/optimizer/betas/1", json!("0.999")),
        ("/optimizer/lr", Value::Null),
        ("/metrics", json!({})),
        ("/metrics/0/threshold", json!("0.5")),
        ("/loss/kind", json!("l1")),
        ("/optimizer/kind", json!("rmsprop")),
        ("/schedule/kind", json!("exponential")),
        ("/data/train/kind", json!("jpeg_folder")),
        ("/metrics/0/kind", json!("loss")),
    ] {
        fails_at(&with(&m, pointer, value), pointer);
    }
}

#[test]
fn nested_errors_get_their_prefix() {
    let m = spec_example();
    for (pointer, value) in [
        ("/data/max_bytes", json!(0)),
        ("/data/validation/fraction", json!(1.0)),
        ("/data/train/image_size/0", json!(0)),
        ("/data/train/channels", json!(2)),
        ("/loss/label_smoothing", json!(1.5)),
        ("/loss/prediction", json!("")),
        ("/metrics/0/name", json!("loss")),
        ("/metrics/0/threshold", json!(1e39)),
        ("/optimizer/betas/1", json!(1.0)),
        ("/optimizer/lr", json!(-1)),
        ("/schedule/min_lr", json!(0.01)),
        ("/schedule/warmup_steps", json!((1u64 << 32) + 1)),
    ] {
        fails_at(&with(&m, pointer, value), pointer);
    }
    // validate_metrics' own pointers: a repeated name, and more than 16.
    let two = with(
        &m,
        "/metrics",
        json!([
            { "kind": "accuracy", "prediction": "logits", "target": "label" },
            { "kind": "accuracy", "prediction": "logits", "target": "label" }
        ]),
    );
    fails_at(&two, "/metrics/1/name");
    let many: Vec<Value> = (0..17)
        .map(|i| json!({ "kind": "accuracy", "prediction": "p", "target": "t", "name": format!("a{i}") }))
        .collect();
    fails_at(&with(&m, "/metrics", Value::Array(many)), "/metrics");
    // A CSV columns key with a `/`.
    let csv = with(
        &m,
        "/data/train",
        json!({ "kind": "csv", "path": "t.csv",
        "columns": { "a/b": { "from": [], "dtype": "f32" } } }),
    );
    fails_at(&csv, "/data/train/columns/a~1b/from");
    let csv = with(
        &m,
        "/data/train",
        json!({ "kind": "csv", "path": "t.csv",
        "columns": { "x": { "from": "c", "dtype": "f64" } } }),
    );
    fails_at(&csv, "/data/train/columns/x/dtype");
}

#[test]
fn document_limits_and_shape() {
    for text in [
        "",
        "[]",
        "null",
        "{",
        "{\"format\": \"polypous.training\",}",
        "1",
    ] {
        let err = TrainingConfig::from_json(text).unwrap_err();
        assert_eq!(err.pointer(), Some(""), "{text:?}: {err}");
    }
    fails_at(
        &with(&minimal(), "/format", json!("polypous.graph")),
        "/format",
    );
    let err = fails_at(
        &with(&minimal(), "/format_version", json!(2)),
        "/format_version",
    );
    assert!(err.message().contains("newer"), "{err}");
    // Over 1 MiB (checked before parsing).
    let big = minimal().to_string().replace(
        "\"epochs\"",
        &format!("\"x\":\"{}\",\"epochs\"", "a".repeat(1 << 20)),
    );
    let err = TrainingConfig::from_json(&big).unwrap_err();
    assert!(err.message().contains("1 MiB"), "{err}");
    // Nesting depth 32 is allowed (the error is then the unknown field), 33 is not.
    let nested = |depth: usize| {
        let inner = format!("{}{}", "[".repeat(depth - 1), "]".repeat(depth - 1));
        minimal()
            .to_string()
            .replace("\"epochs\"", &format!("\"x\":{inner},\"epochs\""))
    };
    let err = TrainingConfig::from_json(&nested(32)).unwrap_err();
    assert_eq!(err.pointer(), Some("/x"), "{err}");
    let err = TrainingConfig::from_json(&nested(33)).unwrap_err();
    assert_eq!(err.pointer(), Some(""), "{err}");
    assert!(err.message().contains("nested"), "{err}");
    // A key given twice.
    let text = minimal()
        .to_string()
        .replace("\"epochs\":5", "\"epochs\":5,\"epochs\":6");
    assert_eq!(
        TrainingConfig::from_json(&text).unwrap_err().pointer(),
        Some("/epochs")
    );
    let text = spec_example()
        .to_string()
        .replace("\"lr\":0.001", "\"lr\":0.001,\"lr\":0.002");
    assert_eq!(
        TrainingConfig::from_json(&text).unwrap_err().pointer(),
        Some("/optimizer/lr")
    );
}

#[test]
fn from_file_reads_at_most_one_mib() {
    let dir = std::env::temp_dir().join(format!("polypous-config-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("train.json");
    std::fs::write(&path, minimal().to_string()).unwrap();
    assert_eq!(
        TrainingConfig::from_file(&path).unwrap(),
        parse(&minimal()).unwrap()
    );
    std::fs::write(&path, vec![b' '; (1 << 20) + 1]).unwrap();
    let err = TrainingConfig::from_file(&path).unwrap_err();
    assert_eq!(err.kind(), TrainErrorKind::InvalidInput);
    assert_eq!(err.file(), Some(path.as_path()));
    let err = TrainingConfig::from_file(dir.join("missing.json")).unwrap_err();
    assert_eq!(err.kind(), TrainErrorKind::Io);
    std::fs::write(
        &path,
        b"{\"format\": \"polypous.training\", \"x\": \"\xff\"}",
    )
    .unwrap();
    assert_eq!(
        TrainingConfig::from_file(&path).unwrap_err().pointer(),
        Some("")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_nested_readers_agree_with_the_serde_types() {
    // The nested objects are read by this module's walker; their serde derives (used by
    // to_json and by the earlier tests) must read the same values.
    let c = parse(&spec_example()).unwrap();
    let doc: Value = serde_json::from_str(&c.to_json()).unwrap();
    let opt: OptimizerConfig = serde_json::from_value(doc["optimizer"].clone()).unwrap();
    assert_eq!(opt, c.optimizer);
    let sched: ScheduleConfig = serde_json::from_value(doc["schedule"].clone()).unwrap();
    assert_eq!(sched, c.schedule);
    let loss: LossConfig = serde_json::from_value(doc["loss"].clone()).unwrap();
    assert_eq!(loss, c.loss);
    let metrics: Vec<MetricConfig> = serde_json::from_value(doc["metrics"].clone()).unwrap();
    assert_eq!(metrics, c.metrics);
    let data: DataConfig = serde_json::from_value(doc["data"].clone()).unwrap();
    assert_eq!(data, c.data);
    let csv = with(
        &minimal(),
        "/data/train",
        json!({ "kind": "csv", "path": "t.csv",
        "columns": { "x": { "from": ["a", "b"], "dtype": "f32" }, "y": { "from": "c", "dtype": "i32" } } }),
    );
    let c = parse(&csv).unwrap();
    let doc: Value = serde_json::from_str(&c.to_json()).unwrap();
    let data: DataConfig = serde_json::from_value(doc["data"].clone()).unwrap();
    assert_eq!(data, c.data);
    assert_eq!(TrainingConfig::from_json(&c.to_json()).unwrap(), c);
}

#[test]
fn reported_and_used_names() {
    let c = parse(&spec_example()).unwrap();
    assert_eq!(
        c.reported_names(),
        ["train.loss", "train.accuracy", "val.loss", "val.accuracy"]
    );
    assert_eq!(
        c.used_fields().into_iter().collect::<Vec<_>>(),
        ["image", "label"]
    );
    let c = parse(&minimal()).unwrap();
    assert_eq!(c.reported_names(), ["train.loss"]);
}

#[test]
fn validate_checks_a_config_built_in_rust() {
    let mut c = parse(&spec_example()).unwrap();
    c.validate().unwrap();
    c.batch_size = 0;
    assert_eq!(c.validate().unwrap_err().pointer(), Some("/batch_size"));
    c.batch_size = 1;
    c.precision = "bf16".into();
    assert_eq!(
        c.validate().unwrap_err().kind(),
        TrainErrorKind::UnsupportedOperation
    );
}
