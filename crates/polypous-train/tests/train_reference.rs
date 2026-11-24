//! The optimizers, gradient clipping and schedules against the reference fixtures
//! (`fixtures/small/train/<component>/`, testing-strategy.md §3.7; `tools/reference/gen_train.py`).
//!
//! - **Optimizers:** 3 steps from the fixture's param and gradients, with `k = s − 1` and the
//!   config's constant `lr`; after each step the param and every state tensor (the names of
//!   training.md §5.2, exactly the set the fixture lists) within the sidecar's tolerance of
//!   PyTorch's own f32 result.
//! - **Gradient clipping:** the total norm and the clipped gradients, NaN where expected.
//! - **Schedules:** `lr(k)` for `k = 0 … T−1`, exact (bit for bit) or within the cosine's `atol`.
//!
//! `train_reference_cpu` runs on the CPU; `train_reference_gpu` (`#[ignore = "gpu"]`) runs the
//! optimizer and clipping fixtures on every GPU device (lavapipe with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1`). The worst errors are printed (`--nocapture`).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use polypous_backend::DynTensor;
use polypous_backend::device::DeviceId;
use polypous_formats::safetensors::SafetensorsFile;
use polypous_train::{
    GradientClipConfig, Optimizer, OptimizerConfig, Schedule, ScheduleConfig, TensorMap,
    clip_gradients,
};
use serde_json::Value;
use support::{Worst, assert_close, f32_tensor, gpu_devices, read_f32};

const OPTIMIZERS: [&str; 3] = ["optimizer.sgd", "optimizer.adam", "optimizer.adamw"];
const SCHEDULES: [&str; 4] = [
    "schedule.constant",
    "schedule.step",
    "schedule.cosine",
    "schedule.linear_warmup",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/small/train")
}

/// Every sidecar of a component, sorted. A component without cases is a failure.
fn sidecars(component: &str) -> Vec<(PathBuf, Value)> {
    let dir = root().join(component);
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "{}: no cases", dir.display());
    paths
        .into_iter()
        .map(|p| {
            let json = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
            (p, json)
        })
        .collect()
}

struct Tensors {
    file: SafetensorsFile,
}

impl Tensors {
    fn open(sidecar: &Path) -> Self {
        Self {
            file: SafetensorsFile::open(sidecar.with_extension("safetensors")).unwrap(),
        }
    }

    fn values(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let view = self
            .file
            .tensor(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let (chunks, _) = view.data().as_chunks::<4>();
        (
            view.shape().to_vec(),
            chunks.iter().map(|&b| f32::from_le_bytes(b)).collect(),
        )
    }

    fn tensor(&self, name: &str, device: DeviceId) -> DynTensor {
        let (shape, values) = self.values(name);
        f32_tensor(&shape, &values, device)
    }
}

fn tolerance(json: &Value) -> (bool, f64, f64) {
    let t = &json["tolerance"];
    (
        t["exact"].as_bool().unwrap(),
        t["atol"].as_f64().unwrap(),
        t["rtol"].as_f64().unwrap(),
    )
}

fn output_names(json: &Value) -> Vec<String> {
    json["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["name"].as_str().unwrap().to_owned())
        .collect()
}

/// Runs every optimizer fixture on `device`; returns the worst error per optimizer.
fn optimizers(device: DeviceId) -> BTreeMap<&'static str, Worst> {
    let mut worst = BTreeMap::new();
    for component in OPTIMIZERS {
        let w: &mut Worst = worst.entry(component).or_default();
        for (path, json) in sidecars(component) {
            let case = format!("{component}/{}", json["case"].as_str().unwrap());
            let config: OptimizerConfig = serde_json::from_value(json["config"].clone()).unwrap();
            let (_, atol, rtol) = tolerance(&json);
            let lr = config.lr();
            let mut opt = Optimizer::new(config).unwrap();
            let file = Tensors::open(&path);
            let mut params = TensorMap::new();
            params.insert("p".to_owned(), file.tensor("param", device));
            let outputs = output_names(&json);
            for s in 1..=3u64 {
                let mut grads = TensorMap::new();
                grads.insert("p".to_owned(), file.tensor(&format!("grad_{s}"), device));
                opt.step(s - 1, lr, &mut params, &grads).unwrap();
                let (_, want) = file.values(&format!("param_{s}"));
                w.merge(assert_close(
                    &format!("{case} on {device}: param_{s}"),
                    &read_f32(&params["p"]),
                    &want,
                    atol,
                    rtol,
                ));
                let mut expected_state: Vec<String> = outputs
                    .iter()
                    .filter(|n| n.ends_with(&format!("_{s}")) && !n.starts_with("param_"))
                    .cloned()
                    .collect();
                let mut got_state = Vec::new();
                for st in opt.export_state() {
                    assert_eq!(st.param, "p");
                    let name = format!("{}_{s}", st.state);
                    let (_, want) = file.values(&name);
                    w.merge(assert_close(
                        &format!("{case} on {device}: {name}"),
                        &read_f32(&st.tensor),
                        &want,
                        atol,
                        rtol,
                    ));
                    got_state.push(name);
                }
                expected_state.sort();
                got_state.sort();
                assert_eq!(
                    got_state, expected_state,
                    "{case}: state names after step {s}"
                );
            }
        }
    }
    worst
}

/// Runs every clipping fixture on `device`; returns the worst error.
fn clipping(device: DeviceId) -> Worst {
    let mut worst = Worst::default();
    for (path, json) in sidecars("gradient_clip") {
        let case = format!("gradient_clip/{}", json["case"].as_str().unwrap());
        let config: GradientClipConfig = serde_json::from_value(json["config"].clone()).unwrap();
        let (_, atol, rtol) = tolerance(&json);
        let file = Tensors::open(&path);
        let mut grads = TensorMap::new();
        for input in json["inputs"].as_array().unwrap() {
            let name = input["name"].as_str().unwrap();
            grads.insert(name.to_owned(), file.tensor(name, device));
        }
        let total = clip_gradients(&config, &mut grads).unwrap().unwrap();
        assert_eq!(total.shape(), [1]);
        assert_eq!(total.device(), device);
        let (_, want) = file.values("total_norm");
        worst.merge(assert_close(
            &format!("{case} on {device}: total_norm"),
            &read_f32(&total),
            &want,
            atol,
            rtol,
        ));
        for (i, input) in json["inputs"].as_array().unwrap().iter().enumerate() {
            let name = input["name"].as_str().unwrap();
            let (_, want) = file.values(&format!("clipped_{i}"));
            worst.merge(assert_close(
                &format!("{case} on {device}: clipped_{i}"),
                &read_f32(&grads[name]),
                &want,
                atol,
                rtol,
            ));
        }
    }
    worst
}

fn report(device: DeviceId, optimizers: &BTreeMap<&str, Worst>, clipping: Worst) {
    for (name, w) in optimizers {
        eprintln!(
            "{device} {name}: worst |error| {:.2e}, {} ulp, {:.3} of the tolerance",
            w.abs, w.ulp, w.ratio
        );
    }
    eprintln!(
        "{device} gradient_clip: worst |error| {:.2e}, {:.3} of the tolerance",
        clipping.abs, clipping.ratio
    );
}

#[test]
fn train_reference_cpu() {
    let device = DeviceId::Cpu;
    let opt = optimizers(device);
    let clip = clipping(device);
    report(device, &opt, clip);
}

#[test]
#[ignore = "gpu"]
fn train_reference_gpu() {
    for device in gpu_devices() {
        let opt = optimizers(device);
        let clip = clipping(device);
        report(device, &opt, clip);
    }
}

#[test]
fn schedules_match_the_reference() {
    let mut cases = 0;
    for component in SCHEDULES {
        for (path, json) in sidecars(component) {
            let case = format!("{component}/{}", json["case"].as_str().unwrap());
            assert!(
                !path.with_extension("safetensors").exists(),
                "{case}: schedule cases have no tensors"
            );
            let config = &json["config"];
            let lr = config["lr"].as_f64().unwrap();
            let total = config["total_steps"].as_u64().unwrap();
            let schedule: ScheduleConfig =
                serde_json::from_value(config["schedule"].clone()).unwrap();
            assert_eq!(format!("schedule.{}", schedule.kind()), component);
            let s = Schedule::new(schedule, lr, total).unwrap();
            let values = json["values"].as_array().unwrap();
            assert_eq!(values.len() as u64, total, "{case}: values for k = 0 … T−1");
            let (exact, atol, _) = tolerance(&json);
            for (k, want) in values.iter().enumerate() {
                let want = want.as_f64().unwrap();
                let got = s.lr(k as u64).unwrap();
                if exact {
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "{case}: lr({k}) = {got:e}, expected {want:e}"
                    );
                } else {
                    assert!(
                        (got - want).abs() <= atol,
                        "{case}: lr({k}) = {got:e}, expected {want:e} (atol {atol:e})"
                    );
                }
                // A pure function of k: asking again gives the same bits.
                assert_eq!(s.lr(k as u64).unwrap().to_bits(), got.to_bits());
            }
            // And a fresh schedule, asked in reverse order, too.
            let fresh = Schedule::new(schedule, lr, total).unwrap();
            for k in (0..total).rev() {
                assert_eq!(
                    fresh.lr(k).unwrap().to_bits(),
                    s.lr(k).unwrap().to_bits(),
                    "{case}: lr({k}) depends on the call order"
                );
            }
            cases += 1;
        }
    }
    eprintln!("{cases} schedule cases");
}
