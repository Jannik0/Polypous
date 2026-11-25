//! `polyp train` end to end on the CPU (public-api.md §1.4, "CLI"; issue #83): runs the built
//! binary on a small MLP and a seeded synthetic safetensors dataset.
//!
//! Each run uses a fresh probe cache directory (`--probe-cache-dir`); probing runs in a prober
//! process (`polyp` itself). The Ctrl-C tests send SIGINT with the `kill` command and run on Unix
//! only; on Windows the same code path (the `ctrlc` handler, Ctrl-C and Ctrl-Break) is compiled
//! but not driven by a test, because a test can't send a console Ctrl-C to one child process.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "../../polypous/tests/common/training_fixtures.rs"]
mod training_fixtures;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use polypous::{
    CancellationToken, DeviceId, Engine, EngineConfig, Graph, GraphLoadOptions, NamedTensor,
    tensor_file,
};
use serde_json::{Value, json};
use training_fixtures::*;

const EXIT_ERROR: i32 = 1;
const EXIT_USAGE: i32 = 2;
const EXIT_CANCELLED: i32 = 4;

/// A fresh directory for one test's files.
fn dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("cli-train-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn command(dir: &Path, args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_polyp"));
    command
        .args(args)
        .args(["--probe-cache-dir", path_str(&dir.join("probe-cache"))])
        .env_remove("RUST_LOG")
        .env_remove("POLYPOUS_TEST_PROBER_FAULTS")
        .env_remove("POLYPOUS_TEST_FINGERPRINT");
    command
}

fn polyp(dir: &Path, args: &[&str]) -> Output {
    command(dir, args).output().expect("polyp runs")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn path_str(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// Writes the MLP graph, the regression data and a config (with `extra`) into `dir`; returns
/// the graph and config paths.
fn setup(dir: &Path, extra: Value) -> (PathBuf, PathBuf) {
    write_regression(dir, 128, 5);
    let graph = dir.join("mlp.graph.json");
    std::fs::write(&graph, mlp_json(4, 16, 1).to_string()).unwrap();
    let config = dir.join("train.json");
    std::fs::write(&config, regression_config(extra).to_string()).unwrap();
    (graph, config)
}

/// The JSON Lines of stdout; every line must be an event.
fn events(output: &Output) -> Vec<Value> {
    stdout(output)
        .lines()
        .map(|line| {
            let event: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{line}: {e}"));
            assert!(event["event"].is_string(), "{line}");
            event
        })
        .collect()
}

/// The error JSON on stderr (`--json`).
fn error_json(output: &Output) -> Value {
    let text = stderr(output);
    let line = text.lines().last().unwrap_or_else(|| panic!("no stderr"));
    serde_json::from_str(line).unwrap_or_else(|e| panic!("{text}: {e}"))
}

/// Runs the export with the facade on the CPU.
fn facade_run(graph: &Path, weights: &Path, x: NamedTensor) -> Vec<NamedTensor> {
    let config = json!({ "probe_report": { "devices": [{
        "id": {"backend": "cpu"}, "backend": "cpu", "name": "test cpu", "vendor": null,
        "driver": null, "kind": "cpu", "dedicated_memory": null, "free_memory": null,
        "unified_memory": true,
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": {"state": "available", "isolated": true}, "physical_gpu": null, "primary": false
    }], "isolated": true } });
    let engine = Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap();
    let mut options = GraphLoadOptions::default();
    options.weights = Some(weights.to_path_buf());
    options.device = Some(DeviceId::Cpu);
    let model = engine
        .load_graph(
            &Graph::from_file(graph).unwrap(),
            options,
            &CancellationToken::new(),
        )
        .unwrap();
    model
        .run(vec![x], vec![], &CancellationToken::new())
        .unwrap()
        .outputs
}

#[test]
fn a_new_run_with_json_and_export() {
    let dir = dir("new-json");
    let (graph, config) = setup(&dir, json!({"checkpoint": {"dir": "run"}}));
    let (eg, ew) = (dir.join("out.graph.json"), dir.join("out.safetensors"));
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
            "--export-graph",
            path_str(&eg),
            "--export-weights",
            path_str(&ew),
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let events = events(&output);
    assert_eq!(events[0]["event"], "started");
    assert_eq!(events[0]["device"], json!({"backend": "cpu"}));
    let last = events.last().unwrap();
    assert_eq!(last["event"], "finished");
    assert_eq!(last["outcome"]["stop_reason"], "completed");
    assert_eq!(last["outcome"]["step"], 240);
    let loss = last["outcome"]["final_metrics"]["train.loss"]
        .as_f64()
        .unwrap();
    assert!(loss < 0.01, "{loss}");
    // The config's relative paths are resolved against its directory.
    let checkpoint = last["outcome"]["last_checkpoint"].as_str().unwrap();
    assert!(
        Path::new(checkpoint).starts_with(dir.join("run")),
        "{checkpoint}"
    );
    assert!(events.iter().any(|e| e["event"] == "checkpoint_saved"));
    assert_eq!(
        events.iter().filter(|e| e["event"] == "finished").count(),
        1
    );

    // The export runs with `polyp run`, with the facade's outputs.
    let x = regression_fields(128, 5).remove(0);
    let input = dir.join("x.safetensors");
    tensor_file::write(&input, std::slice::from_ref(&x)).unwrap();
    let out = dir.join("y.safetensors");
    let run = polyp(
        &dir,
        &[
            "run",
            path_str(&eg),
            "--weights",
            path_str(&ew),
            "--input",
            &format!("x={}", path_str(&input)),
            "--output",
            path_str(&out),
            "--device",
            "cpu",
        ],
    );
    assert_eq!(run.status.code(), Some(0), "{}", stderr(&run));
    assert_eq!(tensor_file::read(&out).unwrap(), facade_run(&eg, &ew, x));
    // The export is the final checkpoint's weights.
    assert_eq!(
        std::fs::read(&ew).unwrap(),
        std::fs::read(Path::new(checkpoint).join("weights.safetensors")).unwrap()
    );
}

#[test]
fn the_human_output_has_one_line_per_event() {
    let dir = dir("human");
    let (graph, config) = setup(
        &dir,
        json!({"epochs": 2, "checkpoint": {"dir": "run", "keep_best": "train.loss"}}),
    );
    let (eg, ew) = (dir.join("out.graph.json"), dir.join("out.safetensors"));
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--export-graph",
            path_str(&eg),
            "--export-weights",
            path_str(&ew),
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let text = stdout(&output);
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[0].starts_with("started device=cpu epochs=2 steps_per_epoch=8 total_steps=16 "),
        "{text}"
    );
    assert_eq!(lines[1], "epoch_started epoch=0 batch=0");
    assert!(
        lines[2].starts_with("step epoch=0 batch=0 step=0 loss="),
        "{text}"
    );
    assert!(lines[2].contains(" lr=0.01"), "{text}");
    let finished = lines
        .iter()
        .position(|l| l.starts_with("finished "))
        .unwrap();
    let line = lines[finished];
    assert!(
        line.starts_with("finished stop_reason=completed step=16 epochs_completed=2 train.loss="),
        "{line}"
    );
    assert!(line.contains(" best=train.loss="), "{line}");
    assert!(line.contains(" last_checkpoint="), "{line}");
    assert!(line.contains(" elapsed_seconds="), "{line}");
    assert_eq!(
        lines[finished + 1],
        format!(
            "Exported the model to {} and {}.",
            eg.display(),
            ew.display()
        )
    );
    assert_eq!(lines.len(), finished + 2, "{text}");
    for line in &lines[..finished] {
        let name = line.split(' ').next().unwrap();
        assert!(
            [
                "started",
                "epoch_started",
                "step",
                "validation_started",
                "epoch_finished",
                "checkpoint_saved",
                "checkpoint_removed"
            ]
            .contains(&name),
            "{line}"
        );
    }
    let epoch = lines
        .iter()
        .find(|l| l.starts_with("epoch_finished epoch=0 step=8 train.loss="))
        .unwrap();
    assert!(!epoch.contains("val."), "{epoch}");
    assert!(
        lines
            .iter()
            .any(|l| l.starts_with("checkpoint_saved step=8 path="))
    );
}

#[test]
fn resume_from_a_run_directory() {
    let dir = dir("resume");
    let (graph, config) = setup(
        &dir,
        json!({"epochs": 3, "checkpoint": {"dir": "run", "every_steps": 5, "keep_last": null}}),
    );
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let expected = events(&output).pop().unwrap();

    // Newer checkpoints than the one resumed: an error, unless `--discard-newer`.
    let mid = dir.join("run").join("step-000000000005");
    let output = polyp(&dir, &["train", "--resume", path_str(&mid), "--json"]);
    assert_eq!(
        output.status.code(),
        Some(EXIT_ERROR),
        "{}",
        stderr(&output)
    );
    let error = error_json(&output);
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["operation"], "engine_resume_trainer");
    assert!(error["context"]["file"].is_string(), "{error}");
    assert!(stdout(&output).is_empty());

    let output = polyp(
        &dir,
        &[
            "train",
            "--resume",
            path_str(&mid),
            "--discard-newer",
            "--device",
            "cpu",
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    let resumed = events(&output);
    assert_eq!(resumed[0]["event"], "started");
    assert_eq!(resumed[0]["step"], 5);
    let last = resumed.last().unwrap();
    // On the CPU, the same history and steps as the uninterrupted run.
    assert_eq!(last["outcome"]["history"], expected["outcome"]["history"]);
    assert_eq!(last["outcome"]["step"], expected["outcome"]["step"]);
    assert!(resumed.iter().any(|e| e["event"] == "checkpoint_removed"));

    // The run directory's newest checkpoint is the final one: resuming it again is an error.
    let output = polyp(&dir, &["train", "--resume", path_str(&dir.join("run"))]);
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    assert!(
        stderr(&output).contains("training has finished"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn usage_errors_exit_with_2() {
    let dir = dir("usage");
    for args in [
        &["train", "--resume", "run", "g.json"][..],
        &["train", "--resume", "run", "--config", "t.json"][..],
        &["train", "--resume", "run", "--weights", "w.safetensors"][..],
        &[
            "train",
            "g.json",
            "--config",
            "t.json",
            "--export-graph",
            "m.json",
        ][..],
        &[
            "train",
            "g.json",
            "--config",
            "t.json",
            "--export-weights",
            "m.safetensors",
        ][..],
        &["train", "g.json"][..],
        &["train", "g.json", "--config", "t.json", "--discard-newer"][..],
        &["train", "g.json", "--config", "t.json", "--device", "gpu0"][..],
        &["train"][..],
    ] {
        let output = polyp(&dir, args);
        assert_eq!(output.status.code(), Some(EXIT_USAGE), "{args:?}");
        assert!(stderr(&output).starts_with("error: "), "{args:?}");
        assert!(
            stderr(&output).contains("polyp train"),
            "{args:?}: usage text"
        );
        assert!(stdout(&output).is_empty());
    }
}

#[test]
fn help_lists_train() {
    let dir = dir("help");
    let output = polyp(&dir, &["--help"]);
    assert_eq!(output.status.code(), Some(0));
    let text = stdout(&output);
    assert!(text.contains("polyp train GRAPH --config FILE"), "{text}");
    assert!(text.contains("polyp train --resume DIR"), "{text}");
    assert!(text.contains("4 `train` was cancelled"), "{text}");
}

#[test]
fn an_invalid_config_is_an_error_with_its_pointer() {
    let dir = dir("invalid-config");
    let (graph, config) = setup(&dir, json!({"epochs": 0}));
    let output = polyp(
        &dir,
        &["train", path_str(&graph), "--config", path_str(&config)],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let text = stderr(&output);
    assert!(
        text.starts_with("error: The training config is not valid: /epochs"),
        "{text}"
    );
    assert!(text.contains(r#""pointer":"/epochs""#), "{text}");
    assert!(stdout(&output).is_empty());

    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error = error_json(&output);
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["pointer"], "/epochs");
    assert_eq!(error["context"]["file"], path_str(&config));
    assert_eq!(error["context"]["operation"], "training_config_from_file");
    assert!(stdout(&output).is_empty());

    // A missing config file is `io`; an invalid graph is `invalid_graph`, before the config.
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&dir.join("nope.json")),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    assert_eq!(error_json(&output)["kind"], "io");
    std::fs::write(dir.join("bad.graph.json"), "{}").unwrap();
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&dir.join("bad.graph.json")),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    assert_eq!(error_json(&output)["kind"], "invalid_graph");

    // An error while creating the trainer (an unknown device): the facade's error JSON.
    let (graph, config) = setup(&dir, json!({"device": "cuda:3"}));
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error = error_json(&output);
    assert_eq!(error["kind"], "backend_unavailable");
    assert_eq!(error["context"]["device"], "cuda:3");
    assert_eq!(error["context"]["operation"], "engine_trainer");
    // `--device` replaces the config's.
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--device",
            "cpu",
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(0), "{}", stderr(&output));
    assert_eq!(events(&output)[0]["device"], json!({"backend": "cpu"}));
}

#[test]
fn a_tensors_dataset_is_an_error() {
    let dir = dir("tensors");
    let (graph, config) = setup(
        &dir,
        json!({"data": {"train": {"kind": "tensors", "name": "train"}}}),
    );
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let error = error_json(&output);
    assert_eq!(error["kind"], "invalid_input");
    assert_eq!(error["context"]["pointer"], "/data/train");
    assert_eq!(error["context"]["file"], path_str(&config));
    assert!(
        error["message"].as_str().unwrap().contains("`train`"),
        "{error}"
    );

    let (graph, config) = setup(
        &dir,
        json!({"data": {"train": {"kind": "safetensors", "path": "train.safetensors"},
                        "validation": {"kind": "tensors", "name": "val"}}}),
    );
    let output = polyp(
        &dir,
        &["train", path_str(&graph), "--config", path_str(&config)],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    let text = stderr(&output);
    assert!(text.contains(r#""pointer":"/data/validation""#), "{text}");
}

#[test]
fn a_run_that_fails_midway_keeps_stdout_valid_json_lines() {
    // Embedding ids from the data are checked while running (training.md §6: an out-of-range
    // index while running): the second batch holds id 9 of 4, so step 1 fails.
    let dir = dir("failed");
    let ids = [0i64, 1, 2, 3, 1, 9, 0, 2];
    tensor_file::write(
        dir.join("train.safetensors"),
        &[
            NamedTensor::new("ids", i64_tensor(&ids)),
            NamedTensor::new("y", f32_tensor(&[8, 1], &[0.5; 8])),
        ],
    )
    .unwrap();
    let graph = dir.join("emb.graph.json");
    let doc = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { "core": 1, "nn": 1 },
        "dims": { "batch": {} },
        "inputs": [ { "name": "ids", "dtype": "i64", "shape": ["batch"] } ],
        "outputs": [ { "name": "y", "value": "emb" } ],
        "params": [ { "name": "emb", "dtype": "float", "shape": [4, 1], "init": { "kind": "zeros" } } ],
        "nodes": [ { "id": "emb", "op": "nn.embedding", "inputs": { "ids": "ids" }, "params": { "weight": "emb" } } ]
    });
    std::fs::write(&graph, doc.to_string()).unwrap();
    let config = dir.join("train.json");
    let doc = regression_config(json!({
        "epochs": 1, "batch_size": 4, "inputs": {"ids": "ids"},
        "data": {"train": {"kind": "safetensors", "path": "train.safetensors"}, "shuffle": false}
    }));
    std::fs::write(&config, doc.to_string()).unwrap();
    let output = polyp(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    );
    assert_eq!(
        output.status.code(),
        Some(EXIT_ERROR),
        "{}",
        stderr(&output)
    );
    let seen = events(&output);
    assert_eq!(seen[0]["event"], "started");
    assert!(seen.iter().any(|e| e["event"] == "step"), "{seen:?}");
    assert!(!seen.iter().any(|e| e["event"] == "finished"), "{seen:?}");
    let error = error_json(&output);
    assert_eq!(error["kind"], "invalid_input", "{error}");
    assert_eq!(error["context"]["operation"], "trainer_run");
    assert_eq!(error["context"]["node"], "emb", "{error}");
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .starts_with("Training failed: "),
        "{error}"
    );

    // Without --json: the events as lines, then the message and context on stderr.
    let output = polyp(
        &dir,
        &["train", path_str(&graph), "--config", path_str(&config)],
    );
    assert_eq!(output.status.code(), Some(EXIT_ERROR));
    assert!(
        stdout(&output).starts_with("started "),
        "{}",
        stdout(&output)
    );
    let text = stderr(&output);
    assert!(text.starts_with("error: Training failed: "), "{text}");
    assert!(text.contains(r#""node":"emb""#), "{text}");
}

// ---- Ctrl-C ----

/// A run slow enough to interrupt: about half a second per step in a debug build, 64 epochs.
#[cfg(unix)]
fn slow_setup(dir: &Path) -> (PathBuf, PathBuf) {
    let n = 2048;
    let x = values(n * 64, 9);
    let y: Vec<f32> = (0..n).map(|i| x[i * 64..i * 64 + 4].iter().sum()).collect();
    tensor_file::write(
        dir.join("train.safetensors"),
        &[
            NamedTensor::new("x", f32_tensor(&[n as u64, 64], &x)),
            NamedTensor::new("y", f32_tensor(&[n as u64, 1], &y)),
        ],
    )
    .unwrap();
    let graph = dir.join("mlp.graph.json");
    std::fs::write(&graph, mlp_json(64, 2048, 1).to_string()).unwrap();
    let config = dir.join("train.json");
    let doc = regression_config(json!({
        "epochs": 64, "batch_size": 256, "checkpoint": {"dir": "run"}
    }));
    std::fs::write(&config, doc.to_string()).unwrap();
    (graph, config)
}

#[cfg(unix)]
fn sigint(child: &std::process::Child) {
    let status = Command::new("kill")
        .args(["-INT", &child.id().to_string()])
        .status()
        .expect("the kill command runs");
    assert!(status.success());
}

/// Lines of a child's pipe, on a thread.
#[cfg(unix)]
fn lines(pipe: impl std::io::Read + Send + 'static) -> std::sync::mpsc::Receiver<String> {
    use std::io::BufRead;
    let (send, receive) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(pipe).lines() {
            let Ok(line) = line else { break };
            if send.send(line).is_err() {
                break;
            }
        }
    });
    receive
}

#[cfg(unix)]
#[test]
fn ctrl_c_cancels_writes_a_checkpoint_and_exports() {
    use std::process::Stdio;
    use std::time::Duration;
    let dir = dir("ctrl-c");
    let (graph, config) = slow_setup(&dir);
    let (eg, ew) = (dir.join("out.graph.json"), dir.join("out.safetensors"));
    let mut child = command(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
            "--export-graph",
            path_str(&eg),
            "--export-weights",
            path_str(&ew),
        ],
    )
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let out = lines(child.stdout.take().unwrap());
    let err = lines(child.stderr.take().unwrap());
    let mut seen = Vec::new();
    loop {
        let line = out
            .recv_timeout(Duration::from_secs(120))
            .expect("a step event");
        let event: Value = serde_json::from_str(&line).unwrap();
        seen.push(event.clone());
        if event["event"] == "step" {
            break;
        }
    }
    sigint(&child);
    let status = child.wait().unwrap();
    seen.extend(
        out.iter()
            .map(|l| serde_json::from_str::<Value>(&l).unwrap()),
    );
    let stderr: Vec<String> = err.iter().collect();
    assert_eq!(status.code(), Some(EXIT_CANCELLED), "{stderr:?}");
    assert!(
        stderr.iter().any(|l| l.starts_with("Cancelling")),
        "{stderr:?}"
    );

    let last = seen.last().unwrap();
    assert_eq!(last["event"], "finished", "{seen:?}");
    let outcome = &last["outcome"];
    assert_eq!(outcome["stop_reason"], "cancelled");
    let step = outcome["step"].as_u64().unwrap();
    assert!((1..64 * 8).contains(&step), "{outcome}");
    // The cancel checkpoint and the export.
    let checkpoint = PathBuf::from(outcome["last_checkpoint"].as_str().unwrap());
    assert_eq!(checkpoint, dir.join("run").join(format!("step-{step:012}")));
    assert!(checkpoint.join("state.json").is_file());
    assert!(seen.iter().any(|e| e["event"] == "checkpoint_saved"));
    assert_eq!(
        std::fs::read(&ew).unwrap(),
        std::fs::read(checkpoint.join("weights.safetensors")).unwrap()
    );
    assert!(eg.is_file());
}

#[cfg(unix)]
#[test]
fn a_second_ctrl_c_ends_the_process_at_once() {
    use std::process::Stdio;
    use std::time::Duration;
    let dir = dir("ctrl-c-twice");
    let (graph, config) = slow_setup(&dir);
    let mut child = command(
        &dir,
        &[
            "train",
            path_str(&graph),
            "--config",
            path_str(&config),
            "--json",
        ],
    )
    .stdout(Stdio::piped())
    .stderr(Stdio::piped())
    .spawn()
    .unwrap();
    let out = lines(child.stdout.take().unwrap());
    let err = lines(child.stderr.take().unwrap());
    loop {
        let line = out
            .recv_timeout(Duration::from_secs(120))
            .expect("a step event");
        if line.contains(r#""event":"step""#) {
            break;
        }
    }
    sigint(&child);
    // The second only after the first was handled (signals of one kind don't queue).
    loop {
        let line = err
            .recv_timeout(Duration::from_secs(60))
            .expect("the cancel message");
        if line.starts_with("Cancelling") {
            break;
        }
    }
    sigint(&child);
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));
    let rest: Vec<String> = out.iter().collect();
    assert!(
        !rest.iter().any(|l| l.contains(r#""event":"finished""#)),
        "{rest:?}"
    );
    // No cancel checkpoint was written (the step in progress never finished).
    assert!(
        std::fs::read_dir(dir.join("run")).unwrap().next().is_none(),
        "a checkpoint was written"
    );
}
