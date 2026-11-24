//! The losses against the reference fixtures (`fixtures/small/train/loss.<kind>/`,
//! testing-strategy.md §3.7; `tools/reference/gen_train.py`).
//!
//! For every case: build the loss from the sidecar's `config` (with `prediction` and `target`
//! added, which the fixture leaves out), compute it on the tracked prediction, compare the value
//! and the gradient of `backward` with respect to the prediction (upstream gradient 1) under the
//! sidecar's tolerance (testing-strategy.md §2: `atol 1e-5`, `rtol 1e-3`, §3.2's rule), and check
//! that an untracked prediction gives the same value bit for bit. Class indices are stored as
//! `I64`; every `cross_entropy` case also runs with them converted to `i32` (training.md §3.3).
//!
//! `loss_reference_cpu` runs on the CPU; `loss_reference_gpu` (`#[ignore = "gpu"]`) on every GPU
//! device (lavapipe with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`). The worst errors per loss, absolute
//! and relative, are printed (`--nocapture`).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData};
use polypous_formats::safetensors::SafetensorsFile;
use polypous_train::{Loss, LossConfig};
use serde_json::Value;
use support::{Worst, assert_close, f32_tensor, gpu_devices, read_f32};

const LOSSES: [&str; 3] = [
    "loss.mse",
    "loss.cross_entropy",
    "loss.binary_cross_entropy",
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

/// The worst error of a loss over its cases: of the value and of the gradient, absolute and
/// relative (`|error| / |expected|`, over the elements with a non-zero expected value).
#[derive(Default)]
struct LossWorst {
    value: Worst,
    value_rel: f64,
    grad: Worst,
    grad_rel: f64,
}

fn relative(got: &[f32], want: &[f32]) -> f64 {
    got.iter()
        .zip(want)
        .filter(|(_, w)| w.is_finite() && **w != 0.0)
        .map(|(&g, &w)| ((f64::from(g) - f64::from(w)) / f64::from(w)).abs())
        .fold(0.0, f64::max)
}

/// The target of a case as a tensor on `device`: `f32`, or the `I64` class indices as `i64` or
/// converted to `i32`.
fn target(file: &SafetensorsFile, dtype: DType, device: DeviceId) -> DynTensor {
    let view = file.tensor("target").unwrap();
    let shape: Vec<u64> = view.shape().iter().map(|&d| d as u64).collect();
    let bytes = match dtype {
        DType::F32 | DType::I64 => view.data().to_vec(),
        DType::I32 => {
            let (chunks, _) = view.data().as_chunks::<8>();
            chunks
                .iter()
                .flat_map(|&b| i32::try_from(i64::from_le_bytes(b)).unwrap().to_le_bytes())
                .collect()
        }
        other => panic!("no target dtype {other}"),
    };
    DynTensor::from_data(&TensorData::new(dtype, shape, bytes).unwrap(), device).unwrap()
}

fn f32_values(file: &SafetensorsFile, name: &str) -> (Vec<usize>, Vec<f32>) {
    let view = file.tensor(name).unwrap_or_else(|e| panic!("{name}: {e}"));
    let (chunks, _) = view.data().as_chunks::<4>();
    (
        view.shape().to_vec(),
        chunks.iter().map(|&b| f32::from_le_bytes(b)).collect(),
    )
}

/// Runs every loss fixture on `device`; returns the worst errors per loss and the case count.
fn losses(device: DeviceId) -> (BTreeMap<&'static str, LossWorst>, usize) {
    let mut worst = BTreeMap::new();
    let mut runs = 0;
    for component in LOSSES {
        let w: &mut LossWorst = worst.entry(component).or_default();
        for (path, json) in sidecars(component) {
            let case = format!("{component}/{}", json["case"].as_str().unwrap());
            let mut config = json["config"].clone();
            config["prediction"] = "prediction".into();
            config["target"] = "target".into();
            let config: LossConfig = serde_json::from_value(config).unwrap();
            assert_eq!(format!("loss.{}", config.kind()), component);
            let loss = Loss::new(&config).unwrap();
            let t = &json["tolerance"];
            assert!(!t["exact"].as_bool().unwrap());
            let (atol, rtol) = (t["atol"].as_f64().unwrap(), t["rtol"].as_f64().unwrap());
            let file = SafetensorsFile::open(path.with_extension("safetensors")).unwrap();
            let (p_shape, p_values) = f32_values(&file, "prediction");
            let (_, want_loss) = f32_values(&file, "loss");
            let (_, want_grad) = f32_values(&file, "grad_prediction");
            let dtypes: &[DType] = if component == "loss.cross_entropy" {
                &[DType::I64, DType::I32]
            } else {
                &[DType::F32]
            };
            for &dtype in dtypes {
                let what = format!("{case} on {device} ({dtype} target)");
                let y = target(&file, dtype, device);
                let x = f32_tensor(&p_shape, &p_values, device);
                loss.check_shapes((x.dtype(), &x.shape()), (y.dtype(), &y.shape()))
                    .unwrap();
                loss.check_target_values(&p_shape, &y.to_data().unwrap())
                    .unwrap();
                let leaf = x.clone().require_grad().unwrap();
                let l = loss.compute(&leaf, &y).unwrap();
                assert_eq!(l.shape(), [1], "{what}");
                assert_eq!(l.device(), device, "{what}");
                let got_loss = read_f32(&l);
                let grads = l.backward().unwrap();
                let g = grads.get(&leaf).unwrap().expect("a gradient");
                let got_grad = read_f32(&g);
                w.value.merge(assert_close(
                    &format!("{what}: loss"),
                    &got_loss,
                    &want_loss,
                    atol,
                    rtol,
                ));
                w.value_rel = w.value_rel.max(relative(&got_loss, &want_loss));
                w.grad.merge(assert_close(
                    &format!("{what}: grad_prediction"),
                    &got_grad,
                    &want_grad,
                    atol,
                    rtol,
                ));
                w.grad_rel = w.grad_rel.max(relative(&got_grad, &want_grad));
                // An untracked prediction (a validation pass) gives the same value.
                let plain = read_f32(&loss.compute(&x, &y).unwrap());
                assert_eq!(
                    plain[0].to_bits(),
                    got_loss[0].to_bits(),
                    "{what}: tracked vs untracked value"
                );
                runs += 1;
            }
        }
    }
    (worst, runs)
}

fn report(device: DeviceId, worst: &BTreeMap<&str, LossWorst>, runs: usize) {
    for (name, w) in worst {
        eprintln!(
            "{device} {name}: loss worst |error| {:.2e} (relative {:.2e}, {:.1e} of the tolerance); \
             gradient worst |error| {:.2e} (relative {:.2e}, {:.1e} of the tolerance)",
            w.value.abs, w.value_rel, w.value.ratio, w.grad.abs, w.grad_rel, w.grad.ratio
        );
    }
    eprintln!("{device}: {runs} loss runs");
}

#[test]
fn loss_reference_cpu() {
    let (worst, runs) = losses(DeviceId::Cpu);
    report(DeviceId::Cpu, &worst, runs);
}

#[test]
#[ignore = "gpu"]
fn loss_reference_gpu() {
    for device in gpu_devices() {
        let (worst, runs) = losses(device);
        report(device, &worst, runs);
    }
}
