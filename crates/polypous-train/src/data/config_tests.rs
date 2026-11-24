//! Unit tests of the `data` config types: JSON form, defaults and every range check of
//! training.md §2.1, §2.3 and §2.4.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

fn data(json: &str) -> DataConfig {
    serde_json::from_str(json).unwrap()
}

fn data_err(json: &str) -> String {
    serde_json::from_str::<DataConfig>(json)
        .expect_err("expected a JSON error")
        .to_string()
}

#[track_caller]
fn assert_pointer(json: &str, pointer: &str) {
    let err = data(json).validate().expect_err("expected a range error");
    assert_eq!(err.pointer, pointer, "{err}");
}

const TRAIN: &str = r#""train": {"kind": "tensors", "name": "t"}"#;

#[test]
fn defaults() {
    let config = data(&format!("{{{TRAIN}}}"));
    assert_eq!(
        config,
        DataConfig {
            train: DatasetConfig::Tensors(TensorsDatasetConfig {
                name: "t".to_owned()
            }),
            validation: None,
            shuffle: true,
            max_bytes: 4_294_967_296,
        }
    );
    config.validate().unwrap();
    let image = data(
        r#"{"train": {"kind": "image_folder", "path": "d", "image_size": [28, 28]},
            "validation": {"kind": "split", "fraction": 0.1}}"#,
    );
    assert_eq!(
        image.train,
        DatasetConfig::ImageFolder(ImageFolderConfig {
            path: "d".to_owned(),
            image_size: [28, 28],
            channels: 1,
            image_field: "image".to_owned(),
            label_field: "label".to_owned(),
        })
    );
    assert_eq!(
        image.validation,
        Some(DatasetConfig::Split(SplitConfig {
            from: "train".to_owned(),
            fraction: 0.1
        }))
    );
    image.validate().unwrap();
}

/// The spec's example (§2) and a CSV config: JSON round trip with every field written, in table
/// order.
#[test]
fn round_trip() {
    let config = data(
        r#"{"train": {"kind": "image_folder", "path": "data/train", "image_size": [28, 28], "channels": 1},
            "validation": {"kind": "split", "from": "train", "fraction": 0.1},
            "shuffle": true}"#,
    );
    let json = serde_json::to_string(&config).unwrap();
    assert_eq!(
        json,
        r#"{"train":{"kind":"image_folder","path":"data/train","image_size":[28,28],"channels":1,"image_field":"image","label_field":"label"},"validation":{"kind":"split","from":"train","fraction":0.1},"shuffle":true,"max_bytes":4294967296}"#
    );
    assert_eq!(data(&json), config);
    let csv = data(
        r#"{"train": {"kind": "csv", "path": "a.csv",
                      "columns": {"x": {"from": ["a", "b"], "dtype": "f32"},
                                  "y": {"from": "label", "dtype": "i64"}}},
            "validation": {"kind": "safetensors", "path": "v.safetensors"},
            "shuffle": false, "max_bytes": 1}"#,
    );
    csv.validate().unwrap();
    let json = serde_json::to_string(&csv).unwrap();
    assert_eq!(
        json,
        r#"{"train":{"kind":"csv","path":"a.csv","columns":{"x":{"from":["a","b"],"dtype":"f32"},"y":{"from":"label","dtype":"i64"}}},"validation":{"kind":"safetensors","path":"v.safetensors"},"shuffle":false,"max_bytes":1}"#
    );
    assert_eq!(data(&json), csv);
    let DatasetConfig::Csv(c) = &csv.train else {
        panic!("csv")
    };
    assert_eq!(c.columns["x"].from.columns(), ["a", "b"]);
    assert_eq!(c.columns["y"].from.columns(), ["label"]);
    assert_eq!(c.columns["y"].dtype, CsvDtype::I64);
}

/// Strict objects (§2.1): unknown fields, fields of another kind, unknown kinds, wrong JSON types,
/// missing required fields, a column field given twice.
#[test]
fn strict_json() {
    for bad in [
        r#"{"train": {"kind": "tensors", "name": "t", "path": "p"}}"#,
        r#"{"train": {"kind": "tensors"}}"#,
        r#"{"train": {"kind": "jpeg_folder", "path": "p"}}"#,
        r#"{"train": {"kind": "safetensors"}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p"}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p", "image_size": [28]}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p", "image_size": [28, 28, 1]}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p", "image_size": [28.0, 28]}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p", "image_size": [28, 28], "channels": 1.0}}"#,
        r#"{"train": {"kind": "image_folder", "path": "p", "image_size": [28, 28], "resize": true}}"#,
        r#"{"train": {"kind": "csv", "path": "p"}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": "a"}}}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": "a", "dtype": "f64"}}}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": "a", "dtype": "bool"}}}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": 3, "dtype": "f32"}}}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": "a", "dtype": "f32", "scale": 2}}}}"#,
        r#"{"train": {"kind": "csv", "path": "p", "columns": []}}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "validation": {"kind": "split"}}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "validation": {"kind": "split", "fraction": 0.5, "seed": 1}}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "test": null}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "shuffle": 1}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "max_bytes": 1.5}"#,
        r#"{"train": {"kind": "tensors", "name": "t"}, "max_bytes": -1}"#,
        r#"{"validation": null}"#,
    ] {
        assert!(serde_json::from_str::<DataConfig>(bad).is_err(), "{bad}");
    }
    let twice = data_err(
        r#"{"train": {"kind": "csv", "path": "p", "columns": {"x": {"from": "a", "dtype": "f32"}, "x": {"from": "b", "dtype": "f32"}}}}"#,
    );
    assert!(twice.contains("given twice"), "{twice}");
    // `validation: null` is the default.
    assert_eq!(
        data(&format!(r#"{{{TRAIN}, "validation": null}}"#)).validation,
        None
    );
}

#[test]
fn max_bytes_range() {
    for ok in [1u64, 1 << 40] {
        data(&format!(r#"{{{TRAIN}, "max_bytes": {ok}}}"#))
            .validate()
            .unwrap();
    }
    for bad in [0u64, (1 << 40) + 1, u64::MAX] {
        assert_pointer(&format!(r#"{{{TRAIN}, "max_bytes": {bad}}}"#), "/max_bytes");
    }
}

#[test]
fn split_rules() {
    let v = |s: &str| format!(r#"{{{TRAIN}, "validation": {{"kind": "split", {s}}}}}"#);
    data(&v(r#""fraction": 0.5"#)).validate().unwrap();
    data(&v(r#""fraction": 1e-300"#)).validate().unwrap();
    data(&v(r#""fraction": 0.9999999999"#)).validate().unwrap();
    for bad in ["0", "1", "-0.1", "1.5", "0.0"] {
        assert_pointer(&v(&format!(r#""fraction": {bad}"#)), "/validation/fraction");
    }
    assert_pointer(
        &v(r#""from": "validation", "fraction": 0.5"#),
        "/validation/from",
    );
    assert_pointer(
        r#"{"train": {"kind": "split", "fraction": 0.5}}"#,
        "/train/kind",
    );
}

#[test]
fn names_and_paths() {
    let name = |n: usize| {
        format!(
            r#"{{"train": {{"kind": "tensors", "name": "{}"}}}}"#,
            "n".repeat(n)
        )
    };
    data(&name(1)).validate().unwrap();
    data(&name(256)).validate().unwrap();
    assert_pointer(&name(0), "/train/name");
    assert_pointer(&name(257), "/train/name");
    // Lengths are in bytes: 128 two-byte characters are 256 bytes, 129 are too many.
    let wide = |n: usize| {
        format!(
            r#"{{"train": {{"kind": "tensors", "name": "{}"}}}}"#,
            "é".repeat(n)
        )
    };
    data(&wide(128)).validate().unwrap();
    assert_pointer(&wide(129), "/train/name");
    let path = |kind: &str, n: usize, extra: &str| {
        format!(
            r#"{{"train": {{"kind": "tensors", "name": "t"}}, "validation": {{"kind": "{kind}", "path": "{}"{extra}}}}}"#,
            "p".repeat(n)
        )
    };
    let csv_extra = r#", "columns": {}"#;
    let image_extra = r#", "image_size": [1, 1]"#;
    for (kind, extra) in [
        ("safetensors", ""),
        ("csv", csv_extra),
        ("image_folder", image_extra),
    ] {
        data(&path(kind, 1, extra)).validate().unwrap();
        data(&path(kind, 4096, extra)).validate().unwrap();
        assert_pointer(&path(kind, 0, extra), "/validation/path");
        assert_pointer(&path(kind, 4097, extra), "/validation/path");
    }
}

#[test]
fn image_folder_ranges() {
    let image = |s: &str| {
        format!(
            r#"{{"train": {{"kind": "image_folder", "path": "p", "image_size": [28, 28]{s}}}}}"#
        )
    };
    let size = |h: u64, w: u64| {
        format!(r#"{{"train": {{"kind": "image_folder", "path": "p", "image_size": [{h}, {w}]}}}}"#)
    };
    data(&size(1, 4096)).validate().unwrap();
    data(&size(4096, 1)).validate().unwrap();
    assert_pointer(&size(0, 28), "/train/image_size/0");
    assert_pointer(&size(28, 0), "/train/image_size/1");
    assert_pointer(&size(4097, 28), "/train/image_size/0");
    assert_pointer(&size(28, 4097), "/train/image_size/1");
    data(&image(r#", "channels": 3"#)).validate().unwrap();
    for bad in ["0", "2", "4"] {
        assert_pointer(
            &image(&format!(r#", "channels": {bad}"#)),
            "/train/channels",
        );
    }
    assert_pointer(&image(r#", "image_field": """#), "/train/image_field");
    assert_pointer(
        &image(&format!(r#", "label_field": "{}""#, "l".repeat(257))),
        "/train/label_field",
    );
    data(&image(r#", "image_field": "x", "label_field": "y""#))
        .validate()
        .unwrap();
    assert_pointer(&image(r#", "label_field": "image""#), "/train/label_field");
}

#[test]
fn csv_columns() {
    let csv = |columns: &str| {
        format!(r#"{{"train": {{"kind": "csv", "path": "p", "columns": {{{columns}}}}}}}"#)
    };
    let list = |n: usize| {
        let names: Vec<String> = (0..n).map(|i| format!(r#""c{i}""#)).collect();
        names.join(",")
    };
    data(&csv(r#""x": {"from": "a", "dtype": "i32"}"#))
        .validate()
        .unwrap();
    data(&csv(&format!(
        r#""x": {{"from": [{}], "dtype": "f32"}}"#,
        list(16_384)
    )))
    .validate()
    .unwrap();
    assert_pointer(
        &csv(r#""x": {"from": [], "dtype": "f32"}"#),
        "/train/columns/x/from",
    );
    assert_pointer(
        &csv(&format!(
            r#""x": {{"from": [{}], "dtype": "f32"}}"#,
            list(16_385)
        )),
        "/train/columns/x/from",
    );
    // 16,384 columns in all, over two fields: 16,385 is too many.
    assert_pointer(
        &csv(&format!(
            r#""x": {{"from": [{}], "dtype": "f32"}}, "y": {{"from": "z", "dtype": "i64"}}"#,
            list(16_384)
        )),
        "/train/columns",
    );
    // 1,024 fields allowed, 1,025 not.
    let fields = |n: usize| {
        let f: Vec<String> = (0..n)
            .map(|i| format!(r#""f{i}": {{"from": "c", "dtype": "f32"}}"#))
            .collect();
        csv(&f.join(","))
    };
    data(&fields(1024)).validate().unwrap();
    assert_pointer(&fields(1025), "/train/columns");
    // Field names are 1–256 bytes; the pointer escapes `/` and `~` (RFC 6901).
    assert_pointer(
        &csv(r#""": {"from": "a", "dtype": "f32"}"#),
        "/train/columns/",
    );
    let long = "n".repeat(257);
    assert_pointer(
        &csv(&format!(r#""{long}": {{"from": "a", "dtype": "f32"}}"#)),
        &format!("/train/columns/{long}"),
    );
    let escaped = format!("a/b~{}", "c".repeat(260));
    assert_pointer(
        &csv(&format!(r#""{escaped}": {{"from": "a", "dtype": "f32"}}"#)),
        &format!("/train/columns/a~1b~0{}", "c".repeat(260)),
    );
}
