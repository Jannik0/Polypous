//! End-to-end model reference tests (issue #46, roadmap M1 exit): an MLP and a small CNN written
//! as graph JSON by `tools/reference/gen_models.py`, loaded through the facade with their PyTorch
//! weights and compared with PyTorch's outputs (docs/testing-strategy.md §2 "Model end-to-end",
//! §3.1, §3.5).
//!
//! For every model, device and compute dtype of its cases, the graph is loaded **once**
//! (`Graph::from_file`, `Engine::load_graph` with `model.safetensors` and the dtype) and runs every
//! case of that dtype, batch 1 and batch 4, through the symbolic `batch` dim. A dtype the device
//! doesn't support (`bf16` on Vulkan) is skipped with a `SKIP` line; every model × device × dtype
//! gets a `SUMMARY` line. Run with `--nocapture` to see them.
//!
//! The CPU runs in a normal test. GPU devices run in a `#[ignore = "gpu"]` test with the `vulkan`
//! or `cuda` feature; software rasterizers (lavapipe) only with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`.
//! On CUDA the f32 cases are expected to fail until Burn can turn off TF32 in f32 matmul
//! (docs/status.md §4); the tolerance is not loosened for it.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "common/model_fixtures.rs"]
mod model_fixtures;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use model_fixtures::{ModelCase, compare_outputs, graph_path, load_cases, model_dir, weights_path};
use polypous::{
    CancellationToken, ComputeDtype, DeviceInfo, Dim, Engine, EngineConfig, Graph,
    GraphLoadOptions, TensorData, tensor_file,
};
use serde_json::json;

const MODELS: [&str; 2] = ["mlp", "cnn"];

fn fixture_dir(model: &str) -> PathBuf {
    model_dir(env!("CARGO_MANIFEST_DIR"), model)
}

/// An engine whose only device is an available CPU (f16 and bf16 supported). No probe runs.
fn cpu_engine() -> Engine {
    let cpu = json!({
        "id": {"backend": "cpu"}, "backend": "cpu", "name": "test cpu", "vendor": null,
        "driver": null, "kind": "cpu", "dedicated_memory": null, "free_memory": null,
        "unified_memory": true,
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": {"state": "available", "isolated": true}, "physical_gpu": null, "primary": false
    });
    let config = json!({ "probe_report": { "devices": [cpu], "isolated": true } });
    Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap()
}

fn supports(device: &DeviceInfo, dtype: ComputeDtype) -> bool {
    match dtype {
        ComputeDtype::F16 => device.capabilities.f16,
        ComputeDtype::Bf16 => device.capabilities.bf16,
        // f32, and any dtype added later: the load decides (an unsupported one fails the test).
        _ => true,
    }
}

#[derive(Default)]
struct Tally {
    passed: usize,
    skipped: usize,
    failed: usize,
}

/// Runs every case of `model` on every device in `devices`, from the weights in `weights`, and
/// returns the failures (one entry per failed case). Prints `SKIP` and `SUMMARY` lines.
fn run_model(engine: &Engine, devices: &[DeviceInfo], model: &str, weights: &Path) -> Vec<String> {
    let dir = fixture_dir(model);
    let graph = Graph::from_file(graph_path(&dir))
        .unwrap_or_else(|err| panic!("{model}: the graph does not load: {}", err.message));
    let cases = load_cases(&dir);
    let mut by_dtype: BTreeMap<&'static str, Vec<&ModelCase>> = BTreeMap::new();
    for case in &cases {
        assert_eq!(case.model, model, "{}", case.name);
        by_dtype
            .entry(case.compute_dtype.as_str())
            .or_default()
            .push(case);
    }
    let mut failures = Vec::new();
    let mut summary = String::new();
    for device in devices {
        for group in by_dtype.values() {
            let dtype = group[0].compute_dtype;
            let tally = run_group(engine, device, &graph, weights, group, &mut failures);
            let _ = writeln!(
                summary,
                "SUMMARY {model} {} {}: {} passed, {} skipped, {} failed",
                device.id,
                dtype.as_str(),
                tally.passed,
                tally.skipped,
                tally.failed
            );
        }
    }
    // One write, so that the lines of parallel tests don't interleave.
    eprint!("{summary}");
    failures
}

/// Loads the graph once for one device and compute dtype and runs every case of that dtype.
fn run_group(
    engine: &Engine,
    device: &DeviceInfo,
    graph: &Graph,
    weights: &Path,
    cases: &[&ModelCase],
    failures: &mut Vec<String>,
) -> Tally {
    let mut tally = Tally::default();
    let dtype = cases[0].compute_dtype;
    let model_name = &cases[0].model;
    if !supports(device, dtype) {
        eprintln!(
            "SKIP {model_name} on {}: compute dtype {} is not supported by the device",
            device.id,
            dtype.as_str()
        );
        tally.skipped = cases.len();
        return tally;
    }
    let mut options = GraphLoadOptions::default();
    options.weights = Some(weights.to_path_buf());
    options.device = Some(device.id);
    options.compute_dtype = dtype;
    let model = match engine.load_graph(graph, options, &CancellationToken::new()) {
        Ok(model) => model,
        Err(err) => {
            for case in cases {
                failures.push(format!(
                    "{model_name}/{} on {}: load failed: {} ({:?})",
                    case.name, device.id, err.message, err.context
                ));
            }
            tally.failed = cases.len();
            return tally;
        }
    };
    let info = model.info();
    assert_eq!(info.device, device.id);
    assert_eq!(info.compute_dtype, dtype);
    // One plan for every batch size: the batch dim stays symbolic.
    assert_eq!(info.inputs[0].shape[0], Dim::Symbolic("batch".into()));
    for case in cases {
        let label = format!("{model_name}/{} on {}", case.name, device.id);
        match model.run(case.inputs.clone(), vec![], &CancellationToken::new()) {
            Ok(out) => match compare_outputs(case, &out.outputs) {
                Ok(()) => tally.passed += 1,
                Err(message) => {
                    failures.push(format!("{label}:\n  {}", message.replace('\n', "\n  ")));
                    tally.failed += 1;
                }
            },
            Err(err) => {
                failures.push(format!(
                    "{label}: run failed: {} ({:?})",
                    err.message, err.context
                ));
                tally.failed += 1;
            }
        }
    }
    tally
}

fn assert_no_failures(failures: &[String]) {
    assert!(
        failures.is_empty(),
        "{} case(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn models_match_pytorch_on_the_cpu() {
    let engine = cpu_engine();
    let devices = engine.devices();
    let mut failures = Vec::new();
    for model in MODELS {
        failures.extend(run_model(
            &engine,
            &devices,
            model,
            &weights_path(&fixture_dir(model)),
        ));
    }
    assert_no_failures(&failures);
}

/// The negative check: the same comparison fails, naming the worst element, when one weight of
/// the MLP differs: `fc2.bias[3]` + 0.05 in a temporary copy of the weights, which moves
/// `logits[.., 3]` by 0.05 in every row, above the tolerance of every compute dtype.
#[test]
fn a_perturbed_weight_fails_the_comparison() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("models-perturbed");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut tensors = tensor_file::read(weights_path(&fixture_dir("mlp"))).unwrap();
    let bias = &mut tensors
        .iter_mut()
        .find(|t| t.name == "fc2.bias")
        .unwrap()
        .tensor;
    assert_eq!(bias.shape, vec![10]);
    let at = 3 * 4;
    let value = f32::from_le_bytes(bias.bytes[at..at + 4].try_into().unwrap()) + 0.05;
    bias.bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    let perturbed = dir.join("model.safetensors");
    tensor_file::write(&perturbed, &tensors).unwrap();

    let engine = cpu_engine();
    let failures = run_model(&engine, &engine.devices(), "mlp", &perturbed);
    eprintln!("{}", failures.join("\n"));
    assert_eq!(failures.len(), 6, "every case fails: {failures:?}");
    for failure in &failures {
        let logits = failure
            .lines()
            .find(|l| l.trim_start().starts_with("logits: "))
            .unwrap_or_else(|| panic!("no logits mismatch: {failure}"));
        assert!(
            logits.contains("elements outside the tolerance"),
            "{logits}"
        );
        assert!(
            logits.contains(", 3]: expected"),
            "worst element in column 3: {logits}"
        );
    }
}

/// The comparison itself: a NaN, an inf and a finite error are reported; a match passes.
#[test]
fn the_comparison_follows_the_sidecar_rule() {
    use model_fixtures::{Tolerance, compare};
    let tol = Tolerance {
        exact: false,
        atol: 1e-5,
        rtol: 1e-3,
    };
    let f32s = |v: &[f32]| {
        TensorData::new(
            polypous::Dtype::F32,
            vec![v.len() as u64],
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
        .unwrap()
    };
    let expected = f32s(&[1.0, -2.0, f32::NAN, f32::INFINITY]);
    assert!(
        compare(
            &expected,
            &f32s(&[1.0005, -2.001, f32::NAN, f32::INFINITY]),
            tol
        )
        .is_ok()
    );
    let err = compare(
        &expected,
        &f32s(&[1.01, -2.0, f32::NAN, f32::INFINITY]),
        tol,
    )
    .unwrap_err();
    assert!(
        err.contains("1 of 4 elements") && err.contains("worst at [0]"),
        "{err}"
    );
    let err = compare(&expected, &f32s(&[1.0, -2.0, 0.0, f32::INFINITY]), tol).unwrap_err();
    assert!(err.contains("worst at [2]"), "{err}");
    let err = compare(&expected, &f32s(&[1.0, -2.0, f32::NAN, f32::MAX]), tol).unwrap_err();
    assert!(err.contains("worst at [3]"), "{err}");
}

/// Every case on every available GPU device. Software rasterizers are probed only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (the CI lavapipe job sets it).
#[cfg(any(feature = "vulkan", feature = "cuda", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn models_match_pytorch_on_gpus() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("models-gpu");
    let _ = std::fs::remove_dir_all(&dir);
    let mut config = EngineConfig::default();
    config.software_rasterizers =
        std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    config.probe_cache_dir = Some(dir.join("probe-cache"));
    let engine = Engine::new(config).unwrap();
    let devices: Vec<DeviceInfo> = engine
        .devices()
        .into_iter()
        .filter(|d| d.backend != polypous::BackendKind::Cpu)
        .filter(|d| {
            let available = d.status.is_available();
            if !available {
                eprintln!("SKIP device {}: {:?}", d.id, d.status);
            }
            available
        })
        .collect();
    assert!(!devices.is_empty(), "no available GPU device");
    let mut failures = Vec::new();
    for model in MODELS {
        failures.extend(run_model(
            &engine,
            &devices,
            model,
            &weights_path(&fixture_dir(model)),
        ));
    }
    assert_no_failures(&failures);
}
