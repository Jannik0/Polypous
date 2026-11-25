//! The MNIST training test through the facade, on the CPU (testing-strategy.md §3.8, roadmap M2
//! exit, issue #84). The CLI side is `crates/polypous-cli/tests/train_mnist.rs`.
//!
//! - **Reference:** the reference graph (every dropout at `p = 0`) and config, from the generator's
//!   initial weights: the first 10 training losses and the inference-mode loss of the first 64
//!   test images after step 10, against PyTorch in f64 (`fixtures/small/mnist/reference.json`),
//!   within `rtol = 1e-3`.
//! - **Accuracy and resume:** `train.json` reaches `val.accuracy ≥ 0.98` within 5 epochs; a second
//!   run cancelled in the middle of an epoch and continued with `Engine::resume_trainer` ends with
//!   the same params, optimizer state (the final checkpoint's bytes), history and best checkpoint
//!   as the first, uninterrupted run (training.md §5.4).
//!
//! These tests need the converted MNIST data in `fixtures/cache/` (`fetch_fixtures.py`) and take
//! minutes, so they are `#[ignore = "mnist"]`: CI doesn't run them; the M2 exit check (#85) does,
//! with `cargo test --release -p polypous -p polypous-cli -- --ignored mnist`. Without the data
//! they fail at once with a message that says how to get it. Only the check of that message runs
//! in CI.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "common/mnist.rs"]
mod mnist;

use std::path::{Path, PathBuf};
use std::time::Instant;

use mnist::*;
use polypous::training::StopReason;
use polypous::{
    CancellationToken, DeviceId, Dtype, Graph, GraphLoadOptions, NamedTensor, ResumeOptions,
    TensorData, Trainer, TrainerOptions, TrainingConfig, TrainingEvent, TrainingOutcome,
    tensor_file,
};
use serde_json::Value;

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("mnist-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn options(weights: Option<PathBuf>) -> TrainerOptions {
    let mut options = TrainerOptions::default();
    options.base_dir = Some(fixture_dir(MANIFEST_DIR));
    options.weights = weights;
    options
}

fn trainer(graph: &str, config: &Value, weights: Option<PathBuf>) -> Trainer {
    let graph = Graph::from_file(fixture_dir(MANIFEST_DIR).join(graph)).unwrap();
    let config = TrainingConfig::from_json(&config.to_string()).unwrap();
    cpu_engine()
        .trainer(&graph, config, options(weights), &CancellationToken::new())
        .unwrap_or_else(|e| panic!("{e:?}"))
}

fn f32s(data: &TensorData) -> Vec<f32> {
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

fn i64s(data: &TensorData) -> Vec<i64> {
    let (chunks, _) = data.bytes.as_chunks::<8>();
    chunks.iter().map(|&b| i64::from_le_bytes(b)).collect()
}

/// The mean cross-entropy of `logits [n, c]` against `labels`, in f64 (`F.cross_entropy`).
fn cross_entropy(logits: &[f32], labels: &[i64], classes: usize) -> f64 {
    let rows = logits.chunks_exact(classes);
    let total: f64 = rows
        .zip(labels)
        .map(|(row, &label)| {
            let max = row
                .iter()
                .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
            let lse = max
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - max).exp())
                    .sum::<f64>()
                    .ln();
            lse - f64::from(row[usize::try_from(label).unwrap()])
        })
        .sum();
    total / labels.len() as f64
}

#[track_caller]
fn assert_close(what: &str, actual: f64, expected: f64, rtol: f64) -> f64 {
    let rel = (actual - expected).abs() / expected.abs();
    eprintln!("{what}: polypous {actual:.9}, reference {expected:.9}, relative error {rel:.2e}");
    assert!(
        (actual - expected).abs() <= rtol * expected.abs(),
        "{what}: {actual} vs the reference {expected}: relative error {rel:.3e} > rtol {rtol}"
    );
    rel
}

// ---- the cache check (runs in CI) ----

#[test]
fn a_missing_mnist_cache_is_a_clear_error() {
    let dir = dir("missing-cache");
    let manifest = repo(MANIFEST_DIR).join("fixtures/manifest.json");
    let error = find_cache(&manifest, &dir).unwrap_err();
    assert!(error.contains("is missing"), "{error}");
    assert!(error.contains("mnist-train"), "{error}");
    assert!(
        error.contains("python tools/reference/fetch_fixtures.py mnist-train mnist-test"),
        "{error}"
    );
    assert!(
        error.contains(&dir.join("mnist").display().to_string()),
        "{error}"
    );

    // A file of the wrong size (a partial copy) names both sizes.
    std::fs::create_dir_all(dir.join("mnist")).unwrap();
    std::fs::write(dir.join("mnist/mnist-train.safetensors"), b"partial").unwrap();
    let error = find_cache(&manifest, &dir).unwrap_err();
    assert!(error.contains("has 7 bytes"), "{error}");
    assert!(error.contains("fetch_fixtures.py"), "{error}");

    // A manifest without the entries.
    let empty = dir.join("manifest.json");
    std::fs::write(&empty, r#"{"schema_version": 1, "entries": []}"#).unwrap();
    let error = find_cache(&empty, &dir).unwrap_err();
    assert!(error.contains("no entry `mnist-train`"), "{error}");
}

// ---- reference ----

/// The first 10 training losses and the inference-mode loss after them, against PyTorch in f64.
/// Each loss comes from the `step` event of its own `run` call: the first step of a run is never
/// throttled (training.md §4.2), so the test cancels after it and calls `run` again (§4.3: the
/// same results as an uninterrupted run). The inference loss runs the exported model
/// (`Engine::load_graph`, training.md §5.5) on the first 64 test images, cross-entropy in f64.
#[test]
#[ignore = "mnist"]
fn mnist_reference_losses_match_pytorch() {
    let cache = require_cache(MANIFEST_DIR);
    let expected = fixture_json(MANIFEST_DIR, "reference.json");
    let rtol = expected["tolerance"]["rtol"].as_f64().unwrap();
    let steps = expected["steps"].as_u64().unwrap();
    let want: Vec<f64> = expected["losses"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(want.len() as u64, steps);

    let started = Instant::now();
    let config = fixture_json(MANIFEST_DIR, "reference_train.json");
    let weights = fixture_dir(MANIFEST_DIR).join("reference.safetensors");
    let trainer = trainer("reference_cnn.graph.json", &config, Some(weights));
    let mut worst = 0.0f64;
    for k in 0..steps {
        let cancel = CancellationToken::new();
        let mut loss = None;
        let outcome = trainer
            .run(&cancel, |event| {
                if let TrainingEvent::Step { step, loss: l, .. } = event {
                    assert_eq!(step, k, "the first step of this run");
                    loss.get_or_insert(l);
                    cancel.cancel();
                }
            })
            .unwrap_or_else(|e| panic!("{e:?}"));
        assert_eq!(
            (outcome.stop_reason, outcome.step),
            (StopReason::Cancelled, k + 1)
        );
        let loss = loss.unwrap_or_else(|| panic!("no step event in run {k}"));
        worst = worst.max(assert_close(
            &format!("loss of step {k}"),
            loss,
            want[usize::try_from(k).unwrap()],
            rtol,
        ));
    }

    let dir = dir("reference");
    let (graph_path, weights_path) = (dir.join("graph.json"), dir.join("weights.safetensors"));
    trainer.export(&graph_path, &weights_path).unwrap();
    let mut load = GraphLoadOptions::default();
    load.weights = Some(weights_path);
    load.device = Some(DeviceId::Cpu);
    let model = cpu_engine()
        .load_graph(
            &Graph::from_file(&graph_path).unwrap(),
            load,
            &CancellationToken::new(),
        )
        .unwrap();
    let first = usize::try_from(expected["inference"]["first"].as_u64().unwrap()).unwrap();
    let test = tensor_file::read(&cache.test).unwrap();
    let field = |name: &str| &test.iter().find(|t| t.name == name).unwrap().tensor;
    let (image, label) = (field("image"), field("label"));
    assert_eq!((image.dtype, label.dtype), (Dtype::F32, Dtype::I64));
    let batch = TensorData::new(
        Dtype::F32,
        vec![first as u64, 1, 28, 28],
        image.bytes[..first * 28 * 28 * 4].to_vec(),
    )
    .unwrap();
    let labels = &i64s(label)[..first];
    let outputs = model
        .run(
            vec![NamedTensor::new("image", batch)],
            vec![],
            &CancellationToken::new(),
        )
        .unwrap()
        .outputs;
    let logits = &outputs.iter().find(|t| t.name == "logits").unwrap().tensor;
    assert_eq!(logits.shape, vec![first as u64, 10]);
    let inference = cross_entropy(&f32s(logits), labels, 10);
    worst = worst.max(assert_close(
        "inference loss after step 10",
        inference,
        expected["inference"]["loss"].as_f64().unwrap(),
        rtol,
    ));
    eprintln!(
        "polypous (CPU) vs PyTorch f64: worst relative error {worst:.2e} (rtol {rtol}); {:.1} s",
        started.elapsed().as_secs_f64()
    );
}

// ---- accuracy and resume ----

fn val_accuracies(outcome: &TrainingOutcome) -> Vec<f64> {
    outcome
        .history
        .iter()
        .map(|h| h.metrics.get("val.accuracy").unwrap())
        .collect()
}

fn checkpoint_file(outcome: &TrainingOutcome, file: &str) -> Vec<u8> {
    let dir = Path::new(outcome.last_checkpoint.as_ref().unwrap());
    std::fs::read(dir.join(file)).unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

fn checkpoint_name(path: &str) -> String {
    Path::new(path)
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned()
}

/// `train.json` (only the run directory changed) trains to `val.accuracy ≥ 0.98` within its 5
/// epochs. Then the same config again, cancelled in the middle of epoch 2 (the cancel writes a
/// checkpoint), resumed by a new engine from the run directory: the final params and optimizer
/// state (the final checkpoint's `weights.safetensors` and `optimizer.safetensors`), the step,
/// the history and the best checkpoint equal the uninterrupted run's bit for bit.
#[test]
#[ignore = "mnist"]
fn mnist_trains_to_98_percent_and_resumes_bit_identically() {
    require_cache(MANIFEST_DIR);
    let dir = dir("accuracy");

    // (a) Uninterrupted.
    let started = Instant::now();
    let config = train_config_with_run_dir(MANIFEST_DIR, &dir.join("a"));
    let reference = trainer("cnn.graph.json", &config, None);
    let mut events = 0usize;
    let expected = reference
        .run(&CancellationToken::new(), |event| {
            events += 1;
            if let TrainingEvent::EpochFinished { epoch, metrics, .. } = &event {
                eprintln!(
                    "epoch {epoch}: {metrics:?} after {:.0} s",
                    started.elapsed().as_secs_f64()
                );
            }
        })
        .unwrap_or_else(|e| panic!("{e:?}"));
    let seconds_a = started.elapsed().as_secs_f64();
    let accuracies = val_accuracies(&expected);
    eprintln!(
        "uninterrupted: {} steps in {seconds_a:.0} s, val.accuracy per epoch {accuracies:?}",
        expected.step
    );
    assert_eq!(expected.stop_reason, StopReason::Completed);
    assert_eq!(expected.epochs_completed, 5);
    let best = expected.best.clone().unwrap();
    assert_eq!(best.metric, "val.accuracy");
    assert!(
        best.value >= MIN_VAL_ACCURACY,
        "val.accuracy reached only {} within 5 epochs (per epoch: {accuracies:?})",
        best.value
    );
    drop(reference);

    // (b) Cancelled in the middle of epoch 2, then resumed by a new engine.
    let started = Instant::now();
    let config = train_config_with_run_dir(MANIFEST_DIR, &dir.join("b"));
    let first = trainer("cnn.graph.json", &config, None);
    let cancel = CancellationToken::new();
    let stopped = first
        .run(&cancel, |event| {
            if let TrainingEvent::Step {
                epoch: 2, batch, ..
            } = event
                && batch >= 300
            {
                cancel.cancel();
            }
        })
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(stopped.stop_reason, StopReason::Cancelled);
    let steps_per_epoch = expected.step / 5;
    assert!(
        stopped.step > 2 * steps_per_epoch + 300 && stopped.step < 3 * steps_per_epoch,
        "cancelled at step {}",
        stopped.step
    );
    let checkpoint = stopped.last_checkpoint.clone().unwrap();
    assert_eq!(
        checkpoint_name(&checkpoint),
        format!("step-{:012}", stopped.step)
    );
    drop(first);

    let resumed = cpu_engine()
        .resume_trainer(
            dir.join("b"),
            ResumeOptions::default(),
            &CancellationToken::new(),
        )
        .unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!(resumed.info().step, stopped.step);
    let outcome = resumed
        .run(&CancellationToken::new(), |_| {})
        .unwrap_or_else(|e| panic!("{e:?}"));
    eprintln!(
        "cancelled at step {} and resumed: {:.0} s in all",
        stopped.step,
        started.elapsed().as_secs_f64()
    );

    assert_eq!(outcome.stop_reason, StopReason::Completed);
    assert_eq!(outcome.step, expected.step);
    assert_eq!(outcome.history, expected.history);
    assert_eq!(outcome.final_metrics, expected.final_metrics);
    let best_b = outcome.best.clone().unwrap();
    assert_eq!(
        (
            &best_b.metric,
            best_b.value.to_bits(),
            best_b.epoch,
            best_b.step
        ),
        (&best.metric, best.value.to_bits(), best.epoch, best.step)
    );
    assert_eq!(
        checkpoint_name(&best_b.checkpoint),
        checkpoint_name(&best.checkpoint)
    );
    assert_eq!(
        checkpoint_name(outcome.last_checkpoint.as_ref().unwrap()),
        checkpoint_name(expected.last_checkpoint.as_ref().unwrap())
    );
    for file in ["weights.safetensors", "optimizer.safetensors"] {
        assert!(
            checkpoint_file(&outcome, file) == checkpoint_file(&expected, file),
            "the final checkpoint's {file} differs from the uninterrupted run's"
        );
    }
    eprintln!("resume: final params, optimizer state, history and best are bit-identical");
}
