//! `polyp graph validate` and `polyp run` end to end: runs the built binary (issue #45).
//!
//! Weights and input files are written with `polypous::tensor_file`. `polyp run` creates an
//! engine, so each run uses a fresh probe cache directory; probing runs in a prober process
//! (`polyp` itself, re-launched).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use polypous::{Dtype, NamedTensor, TensorData, tensor_file};
use serde_json::Value;

const EXIT_ERROR: i32 = 1;
const EXIT_USAGE: i32 = 2;
const EXIT_INVALID: i32 = 3;

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-graphs-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn polyp(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(args)
        .env_remove("RUST_LOG")
        .env_remove("POLYPOUS_TEST_PROBER_FAULTS")
        .env_remove("POLYPOUS_TEST_FINGERPRINT")
        .output()
        .expect("polyp runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn f32_tensor(shape: &[u64], values: &[f32]) -> TensorData {
    TensorData::new(
        Dtype::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn path_str(path: &Path) -> &str {
    path.to_str().unwrap()
}

// ---- graph validate ----

#[test]
fn validate_a_valid_graph() {
    let file = data("mlp.graph.json");
    let output = polyp(&["graph", "validate", path_str(&file)]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(stdout(&output), format!("{}: valid\n", file.display()));

    let output = polyp(&["graph", "validate", path_str(&file), "--json"]);
    assert_eq!(output.status.code(), Some(0));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(
        report,
        serde_json::json!({"valid": true, "diagnostics": []})
    );
}

#[test]
fn validate_an_invalid_graph() {
    let file = data("invalid.graph.json");
    let output = polyp(&["graph", "validate", path_str(&file)]);
    assert_eq!(output.status.code(), Some(EXIT_INVALID));
    let text = stdout(&output);
    assert!(
        text.starts_with(&format!("{}: invalid, 3 errors\n", file.display())),
        "{text}"
    );
    assert!(
        text.contains("  unknown_reference (node `fc1`) at /nodes/0/inputs/x: "),
        "{text}"
    );
    assert!(
        text.contains("  unknown_op (node `act`) at /nodes/1/op: "),
        "{text}"
    );

    let output = polyp(&["graph", "validate", "--json", path_str(&file)]);
    assert_eq!(output.status.code(), Some(EXIT_INVALID));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["valid"], false);
    let diagnostics = report["diagnostics"].as_array().unwrap();
    assert_eq!(diagnostics.len(), 3);
    assert_eq!(diagnostics[0]["code"], "unknown_reference");
    assert_eq!(diagnostics[0]["node"], "fc1");
    assert_eq!(diagnostics[0]["pointer"], "/nodes/0/inputs/x");
    assert!(diagnostics[0]["message"].is_string());

    // A document that isn't JSON: one diagnostic without a node.
    let dir = dir("validate-garbage");
    let garbage = dir.join("garbage.json");
    std::fs::write(&garbage, "{ nope").unwrap();
    let output = polyp(&["graph", "validate", "--json", path_str(&garbage)]);
    assert_eq!(output.status.code(), Some(EXIT_INVALID));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["diagnostics"][0]["code"], "invalid_json");
    assert_eq!(report["diagnostics"][0]["node"], Value::Null);
}

#[test]
fn validate_a_missing_file() {
    let file = dir("validate-missing").join("missing.graph.json");
    let output = polyp(&["graph", "validate", path_str(&file)]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    assert!(output.stdout.is_empty());
    let text = stderr(&output);
    assert!(
        text.starts_with("error: Could not read the graph file"),
        "{text}"
    );
    assert!(text.contains("context: {"), "{text}");

    let output = polyp(&["graph", "validate", "--json", path_str(&file)]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "io");
    assert_eq!(error["context"]["file"], path_str(&file));
    assert_eq!(error["context"]["operation"], "graph_validate_file");
}

#[test]
fn usage_errors() {
    for args in [
        &["graph"][..],
        &["graph", "validate"],
        &["graph", "validate", "a.json", "--yaml"],
        &["run", "g.json"],
        &["run", "g.json", "--output", "o", "--device", "gpu"],
        &["run", "g.json", "--output", "o", "--input", "x"],
    ] {
        let output = polyp(args);
        assert_eq!(output.status.code(), Some(EXIT_USAGE), "{args:?}");
        assert!(stderr(&output).contains("Usage:"), "{args:?}");
    }
    let output = polyp(&["--help"]);
    let help = stdout(&output);
    for text in [
        "polyp graph validate",
        "polyp run",
        "--input NAME=FILE",
        "Exit codes",
    ] {
        assert!(help.contains(text), "{text}");
    }
}

// ---- run ----

/// Weights of the fixture MLP: fc1 = 4→3, fc2 = 3→2.
fn write_weights(dir: &Path) -> PathBuf {
    let w1: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) / 8.0).collect();
    let path = dir.join("mlp.safetensors");
    tensor_file::write(
        &path,
        &[
            NamedTensor::new("fc1.weight", f32_tensor(&[3, 4], &w1)),
            NamedTensor::new("fc1.bias", f32_tensor(&[3], &[0.5, -0.25, 0.0])),
            NamedTensor::new(
                "fc2.weight",
                f32_tensor(&[2, 3], &[1.0, 0.0, -1.0, 0.5, 0.5, 0.5]),
            ),
            NamedTensor::new("fc2.bias", f32_tensor(&[2], &[0.0, 1.0])),
        ],
    )
    .unwrap();
    path
}

/// The fixture MLP in plain Rust for one row.
fn reference(x: [f32; 4]) -> [f32; 2] {
    let w1: Vec<f32> = (0..12).map(|i| (i as f32 - 6.0) / 8.0).collect();
    let b1 = [0.5, -0.25, 0.0];
    let h: Vec<f32> = (0..3)
        .map(|o| ((0..4).map(|i| x[i] * w1[o * 4 + i]).sum::<f32>() + b1[o]).max(0.0))
        .collect();
    [h[0] - h[2], 0.5 * (h[0] + h[1] + h[2]) + 1.0]
}

#[test]
fn run_writes_every_output() {
    let dir = dir("run");
    let weights = write_weights(&dir);
    let input = dir.join("x.safetensors");
    let rows = [[1.0f32, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.0, 2.0]];
    let x: Vec<f32> = rows.iter().flatten().copied().collect();
    // The tensor's name in the file doesn't matter.
    tensor_file::write(
        &input,
        &[NamedTensor::new("anything", f32_tensor(&[2, 4], &x))],
    )
    .unwrap();
    let out = dir.join("out.safetensors");
    let cache = dir.join("probe-cache");
    let graph = data("mlp.graph.json");
    let input_arg = format!("x={}", input.display());
    let args = [
        "run",
        path_str(&graph),
        "--weights",
        path_str(&weights),
        "--input",
        &input_arg,
        "--output",
        path_str(&out),
        "--device",
        "cpu",
        "--probe-cache-dir",
        path_str(&cache),
    ];
    let output = polyp(&args);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    assert!(text.contains("on cpu (compute dtype f32)"), "{text}");
    assert!(text.contains("Wrote 1 output to"), "{text}");
    assert!(text.contains("  y  f32   [2, 2]"), "{text}");

    let outputs = tensor_file::read(&out).unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "y");
    assert_eq!(outputs[0].tensor.shape, vec![2, 2]);
    let y: Vec<f32> = outputs[0]
        .tensor
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let expected: Vec<f32> = rows.iter().flat_map(|r| reference(*r)).collect();
    for (a, e) in y.iter().zip(&expected) {
        assert!((a - e).abs() <= 1e-5, "{y:?} vs {expected:?}");
    }

    // --json: a summary.
    let mut json_args = args.to_vec();
    json_args.push("--json");
    let output = polyp(&json_args);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let summary: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(summary["device"], serde_json::json!({"backend": "cpu"}));
    assert_eq!(summary["compute_dtype"], "f32");
    assert_eq!(
        summary["outputs"],
        serde_json::json!([{"name": "y", "dtype": "f32", "shape": [2, 2]}])
    );
}

#[test]
fn run_rejects_bad_input_files() {
    let dir = dir("run-bad-input");
    let weights = write_weights(&dir);
    let out = dir.join("out.safetensors");
    let cache = dir.join("probe-cache");
    let graph = data("mlp.graph.json");
    let run = |input: &Path, json: bool| {
        let input_arg = format!("x={}", input.display());
        let mut args = vec![
            "run",
            path_str(&graph),
            "--weights",
            path_str(&weights),
            "--input",
            &input_arg,
            "--output",
            path_str(&out),
            "--probe-cache-dir",
            path_str(&cache),
        ];
        if json {
            args.push("--json");
        }
        polyp(&args)
    };

    // Two tensors in one input file.
    let two = dir.join("two.safetensors");
    tensor_file::write(
        &two,
        &[
            NamedTensor::new("a", f32_tensor(&[1, 4], &[0.0; 4])),
            NamedTensor::new("b", f32_tensor(&[1, 4], &[0.0; 4])),
        ],
    )
    .unwrap();
    let output = run(&two, false);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let text = stderr(&output);
    assert!(
        text.contains("holds 2 tensors; it must hold exactly one"),
        "{text}"
    );
    assert!(text.contains(&two.display().to_string()), "{text}");
    let output = run(&two, true);
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["file"], path_str(&two));

    // Not a safetensors file.
    let garbage = dir.join("garbage.safetensors");
    std::fs::write(&garbage, b"not a tensor file").unwrap();
    let output = run(&garbage, true);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["file"], path_str(&garbage));

    // A wrong shape: the facade's binding error names the input.
    let wide = dir.join("wide.safetensors");
    tensor_file::write(
        &wide,
        &[NamedTensor::new("x", f32_tensor(&[1, 5], &[0.0; 5]))],
    )
    .unwrap();
    let output = run(&wide, true);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["tensor"], "x");
    assert!(!out.exists());
}

#[test]
fn run_rejects_bad_devices_and_state() {
    let dir = dir("run-bad-device");
    let weights = write_weights(&dir);
    let input = dir.join("x.safetensors");
    tensor_file::write(
        &input,
        &[NamedTensor::new("x", f32_tensor(&[1, 4], &[0.0; 4]))],
    )
    .unwrap();
    let out = dir.join("out.safetensors");
    let cache = dir.join("probe-cache");
    let graph = data("mlp.graph.json");
    let input_arg = format!("x={}", input.display());
    let output = polyp(&[
        "run",
        path_str(&graph),
        "--weights",
        path_str(&weights),
        "--input",
        &input_arg,
        "--output",
        path_str(&out),
        "--device",
        "cuda:9",
        "--probe-cache-dir",
        path_str(&cache),
        "--json",
    ]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "backend_unavailable");
    assert_eq!(error["context"]["device"], "cuda:9");

    // A graph with state.
    let state_input = dir.join("x2.safetensors");
    tensor_file::write(
        &state_input,
        &[NamedTensor::new("x", f32_tensor(&[2], &[0.0; 2]))],
    )
    .unwrap();
    let state_arg = format!("x={}", state_input.display());
    let state_graph = data("state.graph.json");
    let output = polyp(&[
        "run",
        path_str(&state_graph),
        "--input",
        &state_arg,
        "--output",
        path_str(&out),
        "--probe-cache-dir",
        path_str(&cache),
    ]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let text = stderr(&output);
    assert!(text.contains("state entries (acc)"), "{text}");

    // An invalid graph.
    let output = polyp(&[
        "run",
        path_str(&data("invalid.graph.json")),
        "--output",
        path_str(&out),
        "--json",
    ]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error: Value = serde_json::from_str(stderr(&output).trim()).unwrap();
    assert_eq!(error["kind"], "invalid_graph");
    assert_eq!(error["context"]["node"], "fc1");
    assert_eq!(error["context"]["pointer"], "/nodes/0/inputs/x");
}

/// `polyp run --device vulkan:cpu:0 --software-rasterizers` (lavapipe in CI).
#[cfg(feature = "vulkan")]
#[test]
#[ignore = "gpu"]
fn run_on_lavapipe() {
    if !std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1") {
        eprintln!("SKIP: set POLYPOUS_TEST_SOFTWARE_VULKAN=1 to run on lavapipe");
        return;
    }
    let dir = dir("run-lavapipe");
    let weights = write_weights(&dir);
    let input = dir.join("x.safetensors");
    tensor_file::write(
        &input,
        &[NamedTensor::new(
            "x",
            f32_tensor(&[1, 4], &[1.0, 2.0, 3.0, 4.0]),
        )],
    )
    .unwrap();
    let out = dir.join("out.safetensors");
    let cache = dir.join("probe-cache");
    let graph = data("mlp.graph.json");
    let input_arg = format!("x={}", input.display());
    let output = polyp(&[
        "run",
        path_str(&graph),
        "--weights",
        path_str(&weights),
        "--input",
        &input_arg,
        "--output",
        path_str(&out),
        "--device",
        "vulkan:cpu:0",
        "--software-rasterizers",
        "--probe-cache-dir",
        path_str(&cache),
    ]);
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert!(stdout(&output).contains("on vulkan:cpu:0"));
    let y = &tensor_file::read(&out).unwrap()[0].tensor;
    let y: Vec<f32> = y
        .bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let expected = reference([1.0, 2.0, 3.0, 4.0]);
    for (a, e) in y.iter().zip(&expected) {
        assert!((a - e).abs() <= 1e-5, "{y:?} vs {expected:?}");
    }
}
