//! Unit tests of the losses on the CPU (training.md §3.3): the stable cases the spec names, the
//! gradients at the edges, tracked and untracked values, and every mismatch as an error. The
//! PyTorch comparisons on every fixture are `tests/loss_reference.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_backend::device::DeviceId;

use super::*;
use crate::config::{BinaryCrossEntropyConfig, CrossEntropyConfig, MseConfig};
use crate::targets::TargetValue;

const CPU: DeviceId = DeviceId::Cpu;

fn tensor(dtype: DType, shape: &[usize], bytes: Vec<u8>) -> DynTensor {
    let shape = shape.iter().map(|&d| d as u64).collect();
    DynTensor::from_data(&TensorData::new(dtype, shape, bytes).unwrap(), CPU).unwrap()
}

fn f32s(shape: &[usize], v: &[f32]) -> DynTensor {
    tensor(
        DType::F32,
        shape,
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

fn i64s(v: &[i64]) -> DynTensor {
    tensor(
        DType::I64,
        &[v.len()],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

fn i32s(v: &[i32]) -> DynTensor {
    tensor(
        DType::I32,
        &[v.len()],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
}

fn read(t: &DynTensor) -> Vec<f32> {
    let data = t.to_data().unwrap();
    assert_eq!(data.dtype, DType::F32);
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

fn mse() -> Loss {
    Loss::new(&LossConfig::Mse(MseConfig {
        prediction: "p".into(),
        target: "t".into(),
    }))
    .unwrap()
}

fn ce(label_smoothing: f64) -> Loss {
    Loss::new(&LossConfig::CrossEntropy(CrossEntropyConfig {
        prediction: "p".into(),
        target: "t".into(),
        label_smoothing,
    }))
    .unwrap()
}

fn bce(with_logits: bool) -> Loss {
    Loss::new(&LossConfig::BinaryCrossEntropy(BinaryCrossEntropyConfig {
        prediction: "p".into(),
        target: "t".into(),
        with_logits,
    }))
    .unwrap()
}

/// The loss value and its gradient with respect to the prediction.
fn value_and_grad(loss: &Loss, x: &DynTensor, y: &DynTensor) -> (f32, Vec<f32>) {
    let leaf = x.clone().require_grad().unwrap();
    let l = loss.compute(&leaf, y).unwrap();
    assert_eq!(l.shape(), [1]);
    assert_eq!(l.dtype(), DType::F32);
    assert!(l.is_tracked());
    let value = read(&l)[0];
    let grads = l.backward().unwrap();
    let g = grads
        .get(&leaf)
        .unwrap()
        .expect("the prediction has a gradient");
    assert!(!g.has_autodiff());
    // An untracked prediction (a validation pass) gives the same value, bit for bit.
    let plain = loss.compute(x, y).unwrap();
    assert!(!plain.is_tracked());
    assert_eq!(
        read(&plain)[0].to_bits(),
        value.to_bits(),
        "{}: tracked vs untracked",
        loss.kind()
    );
    (value, read(&g))
}

#[track_caller]
fn assert_near(got: &[f32], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let allowed = 1e-5 + 1e-3 * w.abs();
        assert!(
            (f64::from(g) - w).abs() <= allowed,
            "[{i}]: got {g:e}, expected {w:e}"
        );
    }
}

#[test]
fn cross_entropy_of_large_logits_is_finite() {
    // training.md §3.3: logits [1000, 0, −1000], class 2 → 2000, as in PyTorch.
    let x = f32s(&[1, 3], &[1000.0, 0.0, -1000.0]);
    let (v, g) = value_and_grad(&ce(0.0), &x, &i64s(&[2]));
    assert_eq!(v, 2000.0);
    assert_near(&g, &[1.0, 0.0, -1.0]);
    let (v, _) = value_and_grad(&ce(0.0), &x, &i64s(&[0]));
    assert_eq!(v, 0.0);
    // With smoothing 0.1: (0.9)·2000 + (0.1/3)·(0 + 1000 + 2000) = 1900.
    let (v, g) = value_and_grad(&ce(0.1), &x, &i64s(&[2]));
    assert_near(&[v], &[1900.0]);
    assert_near(&g, &[1.0 - 0.1 / 3.0, -0.1 / 3.0, -0.9 - 0.1 / 3.0]);
}

#[test]
fn cross_entropy_i32_and_i64_targets_agree() {
    let x = f32s(
        &[3, 4],
        &[
            0.1, -1.0, 2.0, 0.5, 3.0, 3.0, -2.0, 0.0, -0.5, 0.25, 1.5, -3.0,
        ],
    );
    for eps in [0.0, 0.1, 1.0] {
        let (v64, g64) = value_and_grad(&ce(eps), &x, &i64s(&[2, 0, 3]));
        let (v32, g32) = value_and_grad(&ce(eps), &x, &i32s(&[2, 0, 3]));
        assert_eq!(v64.to_bits(), v32.to_bits(), "ε = {eps}");
        assert_eq!(g64, g32, "ε = {eps}");
    }
}

#[test]
fn cross_entropy_with_one_class_is_zero() {
    // C = 1: log_softmax is 0, so the loss and its gradient are 0 for every ε.
    for eps in [0.0, 0.5, 1.0] {
        let (v, g) = value_and_grad(&ce(eps), &f32s(&[2, 1], &[3.0, -2.0]), &i64s(&[0, 0]));
        assert_eq!(v, 0.0, "ε = {eps}");
        assert_eq!(g, [0.0, 0.0], "ε = {eps}");
    }
}

#[test]
fn cross_entropy_out_of_range_index_is_an_error() {
    // Checked on the host before the trainer runs; at compute time, gather's own check catches it.
    let x = f32s(&[2, 3], &[0.0; 6]);
    for y in [i64s(&[0, 3]), i64s(&[-1, 0])] {
        let r = ce(0.0).compute(&x, &y);
        assert!(
            matches!(
                r,
                Err(LossError::Backend(TensorError::IndexOutOfRange { .. }))
            ),
            "{r:?}"
        );
    }
}

#[test]
fn bce_with_logits_is_stable() {
    // training.md §3.3: logits of ±200 give losses 200 and 0, not inf.
    let x = f32s(&[4], &[200.0, -200.0, 200.0, -200.0]);
    let y = f32s(&[4], &[0.0, 1.0, 1.0, 0.0]);
    for (i, want) in [200.0, 200.0, 0.0, 0.0].into_iter().enumerate() {
        let xi = f32s(&[1], &[read(&x)[i]]);
        let yi = f32s(&[1], &[read(&y)[i]]);
        let (v, g) = value_and_grad(&bce(true), &xi, &yi);
        assert_eq!(v, want, "element {i}");
        // σ(x) − y: 1, −1, 0, 0.
        assert_eq!(g, [[1.0, -1.0, 0.0, 0.0][i]], "element {i}");
    }
    let (v, g) = value_and_grad(&bce(true), &x, &y);
    assert_eq!(v, 100.0);
    assert_eq!(g, [0.25, -0.25, 0.0, 0.0]);
    // At 0: softplus's kink rule gives σ(0) = 1/2.
    let (v, g) = value_and_grad(&bce(true), &f32s(&[1], &[0.0]), &f32s(&[1], &[1.0]));
    assert_near(&[v], &[std::f64::consts::LN_2]);
    assert_eq!(g, [-0.5]);
}

#[test]
fn bce_at_zero_and_one_is_finite_with_pytorchs_gradient() {
    // x ∈ {0, 1} hit the −100 clamp; PyTorch's gradient (x − y) / max(x · (1 − x), 1e-12) / N.
    let x = f32s(&[6], &[0.0, 1.0, 0.0, 1.0, 1e-13, 0.5]);
    let y = f32s(&[6], &[0.0, 1.0, 1.0, 0.0, 0.3, 0.5]);
    let (v, g) = value_and_grad(&bce(false), &x, &y);
    // Losses: 0, 0, 100, 100, −(0.3·ln 1e-13 + 0.7·ln(1 − 1e-13)), ln 2.
    let want = (0.0 + 0.0 + 100.0 + 100.0 + -0.3 * (1e-13f64).ln() + std::f64::consts::LN_2) / 6.0;
    assert_near(&[v], &[want]);
    let e = 1e-12 * 6.0;
    assert_near(&g, &[0.0, 0.0, -1.0 / e, 1.0 / e, (1e-13 - 0.3) / e, 0.0]);
    assert!(g.iter().all(|v| v.is_finite()));
}

#[test]
fn mse_value_and_gradient() {
    let x = f32s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let y = f32s(&[2, 2], &[0.0, 2.0, 5.0, 3.0]);
    let (v, g) = value_and_grad(&mse(), &x, &y);
    assert_eq!(v, (1.0 + 0.0 + 4.0 + 1.0) / 4.0);
    assert_eq!(g, [0.5, 0.0, -1.0, 0.5]);
}

#[test]
fn a_batch_of_one() {
    let (v, g) = value_and_grad(&mse(), &f32s(&[1], &[3.0]), &f32s(&[1], &[1.0]));
    assert_eq!((v, g), (4.0, vec![4.0]));
    let (v, _) = value_and_grad(&ce(0.0), &f32s(&[1, 2], &[0.0, 0.0]), &i64s(&[1]));
    assert_near(&[v], &[std::f64::consts::LN_2]);
}

#[test]
fn mismatches_are_errors() {
    let x = f32s(&[4, 3], &[0.0; 12]);
    // mse / bce: same shape, f32 target, f32 prediction.
    for loss in [mse(), bce(false), bce(true)] {
        let cases = [
            (x.clone(), f32s(&[4], &[0.0; 4])),
            (x.clone(), f32s(&[4, 3, 1], &[0.0; 12])),
            (x.clone(), f32s(&[1, 3], &[0.0; 3])),
            (x.clone(), f32s(&[3, 4], &[0.0; 12])),
            (x.clone(), i64s(&[0; 12]).reshape(&[4, 3]).unwrap()),
            (
                x.clone().cast(DType::F16).unwrap(),
                f32s(&[4, 3], &[0.0; 12]),
            ),
        ];
        for (p, t) in cases {
            let r = loss.compute(&p, &t);
            assert!(
                matches!(r, Err(LossError::Target(_))),
                "{}: {:?} {:?}: {r:?}",
                loss.kind(),
                p.shape(),
                t.shape()
            );
        }
    }
    // cross_entropy: [B, C] and [B] class indices.
    let cases = [
        (x.clone(), i64s(&[0; 3])),
        (x.clone(), i64s(&[0; 12]).reshape(&[4, 3]).unwrap()),
        (x.clone(), f32s(&[4], &[0.0; 4])),
        (f32s(&[12], &[0.0; 12]), i64s(&[0; 12])),
        (x.clone().reshape(&[4, 3, 1]).unwrap(), i64s(&[0; 4])),
        (x.clone().cast(DType::F16).unwrap(), i64s(&[0; 4])),
    ];
    for (p, t) in cases {
        let r = ce(0.1).compute(&p, &t);
        assert!(
            matches!(r, Err(LossError::Target(_))),
            "{:?} {:?}: {r:?}",
            p.shape(),
            t.shape()
        );
    }
}

#[test]
fn config_errors() {
    let bad = LossConfig::CrossEntropy(CrossEntropyConfig {
        prediction: "p".into(),
        target: "t".into(),
        label_smoothing: 1.5,
    });
    assert!(
        matches!(Loss::new(&bad), Err(LossError::Config(e)) if e.pointer == "/label_smoothing")
    );
}

#[test]
fn target_value_checks() {
    let t = |v: &[i64]| {
        TensorData::new(
            DType::I64,
            vec![v.len() as u64],
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
        .unwrap()
    };
    ce(0.0)
        .check_target_values(&[8, 3], &t(&[0, 1, 2]))
        .unwrap();
    assert!(matches!(
        ce(0.0).check_target_values(&[8, 3], &t(&[0, 3, 2])),
        Err(TargetError::Value {
            sample: 1,
            value: TargetValue::Int(3),
            ..
        })
    ));
    assert!(matches!(
        ce(0.0).check_target_values(&[8], &t(&[0])),
        Err(TargetError::Shape { .. })
    ));
    let f = |v: &[f32]| {
        TensorData::new(
            DType::F32,
            vec![v.len() as u64],
            v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        )
        .unwrap()
    };
    bce(false)
        .check_target_values(&[8], &f(&[0.0, 0.5, 1.0]))
        .unwrap();
    bce(true)
        .check_target_values(&[8], &f(&[0.0, 0.5, 1.0]))
        .unwrap();
    for loss in [bce(false), bce(true)] {
        assert!(matches!(
            loss.check_target_values(&[8], &f(&[0.0, f32::NAN])),
            Err(TargetError::Value { sample: 1, .. })
        ));
        assert!(matches!(
            loss.check_target_values(&[8], &f(&[1.5])),
            Err(TargetError::Value { sample: 0, .. })
        ));
    }
    // mse takes any value, NaN included; the dtype is still checked.
    mse()
        .check_target_values(&[8], &f(&[f32::NAN, -1e30]))
        .unwrap();
    assert!(matches!(
        mse().check_target_values(&[8], &t(&[0])),
        Err(TargetError::TargetDType { .. })
    ));
    assert!(matches!(
        bce(false).check_target_values(&[8], &t(&[0])),
        Err(TargetError::TargetDType { .. })
    ));
    assert!(matches!(
        ce(0.0).check_target_values(&[8, 3], &f(&[0.0])),
        Err(TargetError::TargetDType { .. })
    ));
}
