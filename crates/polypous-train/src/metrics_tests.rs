//! Unit tests of the metrics (training.md §3.4). They are exact: every comparison is `==` (no
//! tolerance, testing-strategy.md §2).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_backend::device::DeviceId;

use super::*;
use crate::targets::TargetValue;

fn f32_data(shape: &[u64], v: &[f32]) -> TensorData {
    TensorData::new(
        DType::F32,
        shape.to_vec(),
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn i64_data(v: &[i64]) -> TensorData {
    TensorData::new(
        DType::I64,
        vec![v.len() as u64],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn i32_data(v: &[i32]) -> TensorData {
    TensorData::new(
        DType::I32,
        vec![v.len() as u64],
        v.iter().flat_map(|x| x.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn accuracy(threshold: Option<f64>) -> Accuracy {
    Accuracy::new(&AccuracyConfig {
        prediction: "p".into(),
        target: "t".into(),
        name: "accuracy".into(),
        threshold,
    })
    .unwrap()
}

const NAN: f32 = f32::NAN;

#[test]
fn argmax_is_torchs() {
    assert_eq!(argmax(&[1.0]), 0);
    assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
    // Ties go to the lowest index.
    assert_eq!(argmax(&[2.0, 5.0, 5.0, 1.0]), 1);
    assert_eq!(argmax(&[7.0, 7.0, 7.0]), 0);
    assert_eq!(argmax(&[0.0, -0.0]), 0);
    assert_eq!(argmax(&[-0.0, 0.0]), 0); // −0 == +0: a tie
    assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), 0);
    assert_eq!(argmax(&[1.0, f32::INFINITY, f32::INFINITY]), 1);
    // NaN is larger than everything, and the first NaN wins.
    assert_eq!(argmax(&[f32::INFINITY, NAN]), 1);
    assert_eq!(argmax(&[1.0, NAN, 9.0, NAN]), 1);
    assert_eq!(argmax(&[NAN, 9.0, NAN]), 0);
    assert_eq!(argmax(&[NAN, NAN]), 0);
}

#[test]
fn multiclass_counts_ties_and_nan_like_torch_argmax() {
    let mut acc = accuracy(None);
    #[rustfmt::skip]
    let x = f32_data(&[6, 3], &[
        1.0, 5.0, 5.0,   // tie → 1
        4.0, 4.0, 4.0,   // tie → 0
        0.0, NAN, 9.0,   // NaN → 1
        NAN, NAN, 1.0,   // first NaN → 0
        -1.0, -2.0, -0.5, // → 2
        3.0, 2.0, 1.0,   // → 0
    ]);
    // Correct: rows 0 (1), 1 (0), 2 (1), 3 (0), 4 is wrong (target 1), 5 is wrong (target 2).
    acc.update_host(&x, &i64_data(&[1, 0, 1, 0, 1, 2])).unwrap();
    assert_eq!((acc.correct(), acc.total()), (4, 6));
    // The second-lowest tie and a later NaN are not argmax.
    let mut acc = accuracy(None);
    acc.update_host(&x, &i64_data(&[2, 1, 2, 1, 2, 0])).unwrap();
    assert_eq!((acc.correct(), acc.total()), (2, 6));
    // i32 targets count the same.
    let mut acc32 = accuracy(None);
    acc32
        .update_host(&x, &i32_data(&[2, 1, 2, 1, 2, 0]))
        .unwrap();
    assert_eq!(acc32, acc);
    assert_eq!(acc.value(), Some(2.0 / 6.0));
}

#[test]
fn binary_threshold_and_nan() {
    let mut acc = accuracy(None);
    // (x > 0.5) = (y = 1). 0.5 itself is not above; NaN is never above.
    let x = f32_data(&[2, 4], &[0.9, 0.5, 0.500_000_06, NAN, 0.1, NAN, 1.0, 0.0]);
    let y = f32_data(&[2, 4], &[1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    // Correct: 0.9→1 ✓, 0.5→0 ✓, 0.50000006→1 ✓, NaN→0 vs 1 ✗, 0.1→0 ✓, NaN→0 ✓, 1.0→1 vs 0 ✗,
    // 0.0→0 vs 1 ✗.
    acc.update_host(&x, &y).unwrap();
    assert_eq!((acc.correct(), acc.total()), (5, 8));
    assert_eq!(acc.value(), Some(5.0 / 8.0));
    // Threshold 0 for logits; −0 is not above 0.
    let mut acc = accuracy(Some(0.0));
    let x = f32_data(&[5], &[-0.0, 0.0, 1e-30, -1e-30, NAN]);
    let y = f32_data(&[5], &[0.0, 0.0, 1.0, 0.0, 0.0]);
    acc.update_host(&x, &y).unwrap();
    assert_eq!((acc.correct(), acc.total()), (5, 5));
}

#[test]
fn binary_threshold_is_compared_in_f32() {
    // f32(0.1) = 0.100000001490116…, above the f64 0.1, but not above f32(0.1).
    let mut acc = accuracy(Some(0.1));
    acc.update_host(&f32_data(&[1], &[0.1]), &f32_data(&[1], &[0.0]))
        .unwrap();
    assert_eq!(acc.correct(), 1);
    // The next f32 is above it.
    let mut acc = accuracy(Some(0.1));
    let next = f32::from_bits(0.1f32.to_bits() + 1);
    acc.update_host(&f32_data(&[1], &[next]), &f32_data(&[1], &[1.0]))
        .unwrap();
    assert_eq!(acc.correct(), 1);
}

#[test]
fn counts_add_up_over_batches_and_the_epoch_divides_once() {
    let mut acc = accuracy(None);
    let x = f32_data(&[3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
    acc.update_host(&x, &i64_data(&[0, 1, 1])).unwrap(); // 2 of 3
    let x = f32_data(&[1, 2], &[0.0, 1.0]);
    acc.update_host(&x, &i64_data(&[1])).unwrap(); // 1 of 1, a smaller last batch
    assert_eq!((acc.correct(), acc.total()), (3, 4));
    assert_eq!(acc.value(), Some(0.75));
    acc.reset();
    assert_eq!((acc.correct(), acc.total(), acc.value()), (0, 0, None));
}

#[test]
fn a_bad_batch_counts_nothing() {
    let mut acc = accuracy(None);
    let x = f32_data(&[2, 3], &[0.0; 6]);
    acc.update_host(&x, &i64_data(&[0, 0])).unwrap();
    let before = acc.clone();
    for t in [
        i64_data(&[0, 3]),
        i64_data(&[-1, 0]),
        i64_data(&[0]),
        i64_data(&[0, 0, 0]),
    ] {
        assert!(acc.update_host(&x, &t).is_err());
        assert_eq!(acc, before);
    }
    // Binary: a target that isn't exactly 0 or 1, a shape mismatch, an integer prediction.
    let mut acc = accuracy(None);
    let x = f32_data(&[2], &[0.7, 0.2]);
    let r = acc.update_host(&x, &f32_data(&[2], &[1.0, 0.5]));
    assert!(
        matches!(
            r,
            Err(MetricError::Target(TargetError::Value { sample: 1, .. }))
        ),
        "{r:?}"
    );
    assert!(
        acc.update_host(&x, &f32_data(&[2, 1], &[1.0, 0.0]))
            .is_err()
    );
    assert!(
        acc.update_host(&i64_data(&[1, 0]), &f32_data(&[2], &[1.0, 0.0]))
            .is_err()
    );
    assert!(
        acc.update_host(
            &x,
            &TensorData {
                dtype: DType::F32,
                shape: vec![2],
                bytes: vec![0; 7]
            }
        )
        .is_err()
    );
    assert_eq!((acc.correct(), acc.total()), (0, 0));
}

#[test]
fn threshold_with_multiclass_is_an_error() {
    let acc = accuracy(Some(0.5));
    let r = acc.check_shapes((DType::F32, &[4, 3]), (DType::I64, &[4]));
    assert!(
        matches!(
            r,
            Err(MetricError::ThresholdWithMulticlass { target: DType::I64 })
        ),
        "{r:?}"
    );
    acc.check_shapes((DType::F32, &[4, 3]), (DType::F32, &[4, 3]))
        .unwrap();
    accuracy(None)
        .check_shapes((DType::F32, &[4, 3]), (DType::I32, &[4]))
        .unwrap();
    let mut acc = accuracy(Some(0.5));
    let r = acc.update_host(&f32_data(&[1, 2], &[0.0, 1.0]), &i64_data(&[1]));
    assert!(
        matches!(r, Err(MetricError::ThresholdWithMulticlass { .. })),
        "{r:?}"
    );
}

#[test]
fn target_value_checks() {
    let acc = accuracy(None);
    acc.check_target_values(&[4, 3], &i64_data(&[0, 2, 1]))
        .unwrap();
    assert!(matches!(
        acc.check_target_values(&[4, 3], &i32_data(&[0, 3])),
        Err(TargetError::Value {
            sample: 1,
            value: TargetValue::Int(3),
            ..
        })
    ));
    assert!(matches!(
        acc.check_target_values(&[4], &i64_data(&[0])),
        Err(TargetError::Shape { .. })
    ));
    acc.check_target_values(&[4], &f32_data(&[3], &[0.0, 1.0, 1.0]))
        .unwrap();
    assert!(matches!(
        acc.check_target_values(&[4], &f32_data(&[3], &[0.0, 1.0, 0.3])),
        Err(TargetError::Value { sample: 2, .. })
    ));
}

#[test]
fn update_reads_a_tracked_prediction_without_recording() {
    let p = f32_data(&[2, 2], &[0.0, 1.0, 1.0, 0.0]);
    let x = DynTensor::from_data(&p, DeviceId::Cpu)
        .unwrap()
        .require_grad()
        .unwrap();
    let mut acc = accuracy(None);
    acc.update(&x, &i64_data(&[1, 1])).unwrap();
    assert_eq!((acc.correct(), acc.total()), (1, 2));
    // The leaf is unchanged and still a leaf.
    assert!(x.is_grad_leaf());
}

#[test]
fn loss_metric_weights_by_batch_size() {
    let mut m = LossMetric::new();
    // Three full batches of 4 and a last batch of 1.
    for (loss, b) in [(0.5f32, 4), (0.25, 4), (1.0, 4), (2.0, 1)] {
        m.add(loss, b);
    }
    assert_eq!(m.samples(), 13);
    assert_eq!(
        m.value(),
        Some((0.5 * 4.0 + 0.25 * 4.0 + 1.0 * 4.0 + 2.0) / 13.0)
    );
    // Not the mean of the batch means (which would be 0.9375).
    assert_ne!(m.value(), Some(0.9375));
}

#[test]
fn loss_metric_sums_in_f64_in_batch_order_from_f32() {
    let losses = [0.1f32, 1e-8, 3.3, 7.0e7, 0.7];
    let sizes = [64, 64, 64, 64, 17];
    let mut m = LossMetric::new();
    let mut want = 0.0f64;
    for (&l, &b) in losses.iter().zip(&sizes) {
        m.add(l, b);
        want += f64::from(l) * b as f64;
    }
    let n: u64 = sizes.iter().sum::<usize>() as u64;
    assert_eq!(m.value().unwrap().to_bits(), (want / n as f64).to_bits());
    // The f32 values, not their decimal f64 neighbours: f64::from(0.1f32) ≠ 0.1.
    let mut one = LossMetric::new();
    one.add(0.1, 1);
    assert_eq!(one.value(), Some(f64::from(0.1f32)));
}

#[test]
fn loss_metric_edge_cases() {
    // An empty epoch has no value (not NaN): the module documentation says why.
    let mut m = LossMetric::new();
    assert_eq!(m.value(), None);
    // A batch of zero samples adds nothing, even with a NaN loss.
    m.add(f32::NAN, 0);
    assert_eq!((m.samples(), m.value()), (0, None));
    // A non-finite batch loss makes the epoch's loss non-finite, as in PyTorch.
    m.add(1.0, 2);
    m.add(f32::INFINITY, 2);
    assert_eq!(m.value(), Some(f64::INFINITY));
    m.add(f32::NAN, 1);
    assert!(m.value().unwrap().is_nan());
    m.reset();
    assert_eq!(m, LossMetric::new());
    // An empty accuracy epoch has no value either.
    assert_eq!(accuracy(None).value(), None);
}
