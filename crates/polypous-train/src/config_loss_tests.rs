//! Unit tests of the `loss` and `metrics` config types (training.md §2.1, §2.6, §2.7): JSON form,
//! defaults, fields of other kinds, and every rule `validate` checks.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

fn loss(json: &str) -> LossConfig {
    serde_json::from_str(json).unwrap()
}

fn metric(json: &str) -> MetricConfig {
    serde_json::from_str(json).unwrap()
}

fn accuracy(name: &str, threshold: Option<f64>) -> MetricConfig {
    MetricConfig::Accuracy(AccuracyConfig {
        prediction: "logits".into(),
        target: "label".into(),
        name: name.into(),
        threshold,
    })
}

#[track_caller]
fn assert_pointer(result: Result<(), ConfigError>, pointer: &str) {
    let err = result.expect_err("expected an error");
    assert_eq!(err.pointer, pointer, "{err}");
}

#[track_caller]
fn assert_rejected<T: serde::de::DeserializeOwned + std::fmt::Debug>(json: &str, needle: &str) {
    let err = serde_json::from_str::<T>(json).expect_err(json).to_string();
    assert!(err.contains(needle), "{json}: {err}");
}

#[test]
fn loss_defaults() {
    assert_eq!(
        loss(r#"{"kind": "mse", "prediction": "y_hat", "target": "y"}"#),
        LossConfig::Mse(MseConfig {
            prediction: "y_hat".into(),
            target: "y".into()
        })
    );
    assert_eq!(
        loss(r#"{"kind": "cross_entropy", "prediction": "logits", "target": "label"}"#),
        LossConfig::CrossEntropy(CrossEntropyConfig {
            prediction: "logits".into(),
            target: "label".into(),
            label_smoothing: 0.0
        })
    );
    assert_eq!(
        loss(r#"{"kind": "binary_cross_entropy", "prediction": "p", "target": "y"}"#),
        LossConfig::BinaryCrossEntropy(BinaryCrossEntropyConfig {
            prediction: "p".into(),
            target: "y".into(),
            with_logits: false
        })
    );
}

#[test]
fn the_spec_example_parses() {
    // training.md §2's example.
    let l = loss(
        r#"{ "kind": "cross_entropy", "prediction": "logits", "target": "label", "label_smoothing": 0.0 }"#,
    );
    l.validate().unwrap();
    assert_eq!(
        (l.kind(), l.prediction(), l.target()),
        ("cross_entropy", "logits", "label")
    );
    let metrics: Vec<MetricConfig> = serde_json::from_str(
        r#"[ { "kind": "accuracy", "prediction": "logits", "target": "label" } ]"#,
    )
    .unwrap();
    validate_metrics(&metrics).unwrap();
    assert_eq!(metrics, [accuracy("accuracy", None)]);
    assert_eq!(
        (metrics[0].kind(), metrics[0].name()),
        ("accuracy", "accuracy")
    );
}

#[test]
fn json_round_trip_writes_every_field_in_table_order() {
    let cases = [
        (
            r#"{"kind":"mse","prediction":"a","target":"b"}"#,
            r#"{"kind":"mse","prediction":"a","target":"b"}"#,
        ),
        (
            r#"{"target":"b","kind":"cross_entropy","prediction":"a"}"#,
            r#"{"kind":"cross_entropy","prediction":"a","target":"b","label_smoothing":0.0}"#,
        ),
        (
            r#"{"kind":"binary_cross_entropy","with_logits":true,"prediction":"a","target":"b"}"#,
            r#"{"kind":"binary_cross_entropy","prediction":"a","target":"b","with_logits":true}"#,
        ),
    ];
    for (input, written) in cases {
        let c = loss(input);
        assert_eq!(serde_json::to_string(&c).unwrap(), written);
        assert_eq!(loss(written), c);
    }
    let m = metric(r#"{"kind":"accuracy","prediction":"p","target":"y"}"#);
    let written =
        r#"{"kind":"accuracy","prediction":"p","target":"y","name":"accuracy","threshold":null}"#;
    assert_eq!(serde_json::to_string(&m).unwrap(), written);
    assert_eq!(metric(written), m);
    let m =
        metric(r#"{"kind":"accuracy","prediction":"p","target":"y","name":"acc","threshold":0.0}"#);
    assert_eq!(
        serde_json::to_string(&m).unwrap(),
        r#"{"kind":"accuracy","prediction":"p","target":"y","name":"acc","threshold":0.0}"#
    );
}

#[test]
fn fields_of_other_kinds_and_unknown_fields_are_rejected() {
    // label_smoothing only with cross_entropy, with_logits only with binary_cross_entropy.
    for json in [
        r#"{"kind": "mse", "prediction": "a", "target": "b", "label_smoothing": 0.1}"#,
        r#"{"kind": "binary_cross_entropy", "prediction": "a", "target": "b", "label_smoothing": 0.1}"#,
    ] {
        assert_rejected::<LossConfig>(json, "unknown field `label_smoothing`");
    }
    for json in [
        r#"{"kind": "mse", "prediction": "a", "target": "b", "with_logits": true}"#,
        r#"{"kind": "cross_entropy", "prediction": "a", "target": "b", "with_logits": true}"#,
    ] {
        assert_rejected::<LossConfig>(json, "unknown field `with_logits`");
    }
    assert_rejected::<LossConfig>(
        r#"{"kind": "mse", "prediction": "a", "target": "b", "weight": 1}"#,
        "unknown field `weight`",
    );
    // "Later" kinds and fields (training.md §3) are unknown.
    assert_rejected::<LossConfig>(
        r#"{"kind": "l1", "prediction": "a", "target": "b"}"#,
        "unknown variant `l1`",
    );
    assert_rejected::<LossConfig>(
        r#"{"kind": "binary_cross_entropy", "prediction": "a", "target": "b", "pos_weight": 2.0}"#,
        "unknown field `pos_weight`",
    );
    assert_rejected::<MetricConfig>(
        r#"{"kind": "f1", "prediction": "a", "target": "b"}"#,
        "unknown variant `f1`",
    );
    assert_rejected::<MetricConfig>(
        r#"{"kind": "accuracy", "prediction": "a", "target": "b", "top_k": 5}"#,
        "unknown field `top_k`",
    );
    // Required fields.
    assert_rejected::<LossConfig>(
        r#"{"prediction": "a", "target": "b"}"#,
        "missing field `kind`",
    );
    assert_rejected::<LossConfig>(
        r#"{"kind": "mse", "target": "b"}"#,
        "missing field `prediction`",
    );
    assert_rejected::<LossConfig>(
        r#"{"kind": "cross_entropy", "prediction": "a"}"#,
        "missing field `target`",
    );
    assert_rejected::<MetricConfig>(
        r#"{"kind": "accuracy", "target": "b"}"#,
        "missing field `prediction`",
    );
    // Wrong JSON types.
    assert_rejected::<LossConfig>(
        r#"{"kind": "binary_cross_entropy", "prediction": "a", "target": "b", "with_logits": 1}"#,
        "invalid type",
    );
    assert_rejected::<LossConfig>(
        r#"{"kind": "cross_entropy", "prediction": "a", "target": "b", "label_smoothing": "0.1"}"#,
        "invalid type",
    );
    assert_rejected::<MetricConfig>(
        r#"{"kind": "accuracy", "prediction": "a", "target": "b", "name": 3}"#,
        "invalid type",
    );
    assert_rejected::<MetricConfig>(
        r#"{"kind": "accuracy", "prediction": "a", "target": "b", "threshold": "0.5"}"#,
        "invalid type",
    );
}

#[test]
fn the_loss_is_never_listed_as_a_metric() {
    // training.md §2.7: `{"kind": "loss"}` is InvalidInput.
    assert_rejected::<MetricConfig>(r#"{"kind": "loss"}"#, "unknown variant `loss`");
    assert_rejected::<MetricConfig>(
        r#"{"kind": "loss", "prediction": "a", "target": "b"}"#,
        "unknown variant `loss`",
    );
    assert_rejected::<Vec<MetricConfig>>(r#"[{"kind": "loss"}]"#, "unknown variant `loss`");
}

#[test]
fn label_smoothing_range() {
    let ce = |eps: f64| {
        LossConfig::CrossEntropy(CrossEntropyConfig {
            prediction: "a".into(),
            target: "b".into(),
            label_smoothing: eps,
        })
    };
    for ok in [0.0, 0.1, 0.5, 1.0] {
        ce(ok).validate().unwrap();
    }
    for bad in [-1e-9, 1.0 + 1e-9, -1.0, 2.0, f64::NAN, f64::INFINITY] {
        assert_pointer(ce(bad).validate(), "/label_smoothing");
    }
}

#[test]
fn names_are_1_to_256_bytes() {
    let long = "x".repeat(MAX_NAME_BYTES);
    let too_long = "x".repeat(MAX_NAME_BYTES + 1);
    let mse = |p: &str, t: &str| {
        LossConfig::Mse(MseConfig {
            prediction: p.into(),
            target: t.into(),
        })
    };
    mse(&long, &long).validate().unwrap();
    mse("ü", "y").validate().unwrap();
    assert_pointer(mse("", "y").validate(), "/prediction");
    assert_pointer(mse(&too_long, "y").validate(), "/prediction");
    assert_pointer(mse("p", "").validate(), "/target");
    assert_pointer(mse("p", &too_long).validate(), "/target");
    let acc = |p: &str, t: &str| {
        MetricConfig::Accuracy(AccuracyConfig {
            prediction: p.into(),
            target: t.into(),
            name: "accuracy".into(),
            threshold: None,
        })
    };
    acc(&long, &long).validate().unwrap();
    assert_pointer(acc("", "y").validate(), "/prediction");
    assert_pointer(acc("p", &too_long).validate(), "/target");
}

#[test]
fn metric_names() {
    for ok in [
        "accuracy",
        "a",
        "acc_2",
        "0",
        "_",
        &"a".repeat(MAX_METRIC_NAME_BYTES),
    ] {
        accuracy(ok, None).validate().unwrap();
    }
    for bad in [
        "",
        &"a".repeat(MAX_METRIC_NAME_BYTES + 1),
        "Accuracy",
        "acc-2",
        "acc.top",
        "acc 2",
        "ä",
        "loss",
    ] {
        assert_pointer(accuracy(bad, None).validate(), "/name");
    }
    // `losses` and `val_loss` are fine: only `loss` itself is reserved.
    accuracy("losses", None).validate().unwrap();
    accuracy("val_loss", None).validate().unwrap();
}

#[test]
fn threshold_must_round_to_a_finite_f32() {
    for ok in [0.5, 0.0, -3.0, 1e30, f64::from(f32::MAX)] {
        accuracy("a", Some(ok)).validate().unwrap();
    }
    for bad in [1e39, -1e39, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_pointer(accuracy("a", Some(bad)).validate(), "/threshold");
    }
}

#[test]
fn metric_lists() {
    validate_metrics(&[]).unwrap();
    validate_metrics(&[accuracy("accuracy", None), accuracy("acc_bin", Some(0.0))]).unwrap();
    // Unique names: the default name twice is a duplicate.
    let err = validate_metrics(&[
        accuracy("accuracy", None),
        accuracy("other", None),
        accuracy("accuracy", Some(0.5)),
    ])
    .unwrap_err();
    assert_eq!(err.pointer, "/2/name");
    assert!(err.message.contains("entry 0"), "{err}");
    // An entry's own error gets its index.
    assert_pointer(
        validate_metrics(&[accuracy("a", None), accuracy("b", Some(f64::NAN))]),
        "/1/threshold",
    );
    assert_pointer(validate_metrics(&[accuracy("loss", None)]), "/0/name");
    // At most 16 entries.
    let many: Vec<MetricConfig> = (0..MAX_METRICS)
        .map(|i| accuracy(&format!("a{i}"), None))
        .collect();
    validate_metrics(&many).unwrap();
    let mut too_many = many;
    too_many.push(accuracy("a16", None));
    assert_pointer(validate_metrics(&too_many), "");
}
