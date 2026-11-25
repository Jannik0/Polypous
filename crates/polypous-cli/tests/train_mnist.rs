//! The MNIST training test through `polyp train` (the built binary), on the CPU
//! (testing-strategy.md §3.8, issue #84). The facade side, the reference losses and resume are
//! `crates/polypous/tests/mnist.rs`.
//!
//! Needs the converted MNIST data in `fixtures/cache/` (`fetch_fixtures.py`) and takes minutes,
//! so it is `#[ignore = "mnist"]`: CI doesn't run it; the M2 exit check (#85) does, with
//! `cargo test --release -p polypous -p polypous-cli -- --ignored mnist`. Without the data it
//! fails at once with a message that says how to get it.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "../../polypous/tests/common/mnist.rs"]
mod mnist;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use mnist::*;
use serde_json::Value;

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-train-mnist-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `polyp train fixtures/small/mnist/cnn.graph.json --config <copy of train.json> --device cpu
/// --json`. The copy changes only the run directory (into this test's directory) and makes the
/// data paths absolute, because the CLI resolves relative paths against the config file's
/// directory and the copy lives elsewhere.
#[test]
#[ignore = "mnist"]
fn polyp_train_mnist_reaches_98_percent() {
    require_cache(MANIFEST_DIR);
    let dir = dir("accuracy");
    let config = with_absolute_data(
        MANIFEST_DIR,
        train_config_with_run_dir(MANIFEST_DIR, &dir.join("run")),
    );
    let config_path = dir.join("train.json");
    std::fs::write(&config_path, config.to_string()).unwrap();
    let graph = fixture_dir(MANIFEST_DIR).join("cnn.graph.json");

    let started = Instant::now();
    let output = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .arg("train")
        .arg(&graph)
        .arg("--config")
        .arg(&config_path)
        .args(["--device", "cpu", "--json", "--probe-cache-dir"])
        .arg(dir.join("probe-cache"))
        .env_remove("RUST_LOG")
        .env_remove("POLYPOUS_TEST_PROBER_FAULTS")
        .env_remove("POLYPOUS_TEST_FINGERPRINT")
        .output()
        .expect("polyp runs");
    let seconds = started.elapsed().as_secs_f64();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "polyp train failed ({:?}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let events: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).unwrap_or_else(|e| panic!("{line}: {e}")))
        .collect();
    for e in events.iter().filter(|e| e["event"] == "epoch_finished") {
        eprintln!("epoch {}: {}", e["epoch"], e["metrics"]);
    }
    let last = events.last().unwrap();
    assert_eq!(last["event"], "finished", "{last}");
    let outcome = &last["outcome"];
    assert_eq!(outcome["stop_reason"], "completed", "{outcome}");
    assert_eq!(outcome["epochs_completed"], 5, "{outcome}");
    let accuracies: Vec<f64> = outcome["history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| h["metrics"]["val.accuracy"].as_f64().unwrap())
        .collect();
    let best = &outcome["best"];
    assert_eq!(best["metric"], "val.accuracy", "{outcome}");
    let value = best["value"].as_f64().unwrap();
    eprintln!(
        "polyp train: {} steps in {seconds:.0} s, val.accuracy per epoch {accuracies:?}",
        outcome["step"]
    );
    assert!(
        value >= MIN_VAL_ACCURACY,
        "val.accuracy reached only {value} within 5 epochs (per epoch: {accuracies:?})"
    );
    // The best checkpoint is in the run directory and loads as an inference model (training.md §5.2).
    let checkpoint = Path::new(best["checkpoint"].as_str().unwrap());
    assert!(checkpoint.starts_with(dir.join("run")), "{best}");
    for file in ["graph.json", "weights.safetensors", "state.json"] {
        assert!(checkpoint.join(file).is_file(), "{}", checkpoint.display());
    }
}
