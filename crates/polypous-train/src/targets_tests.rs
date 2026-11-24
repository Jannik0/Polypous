//! Unit tests of the target checks (training.md §2.10 step 6, §3.3, §3.4): every rule, the sample
//! and element named in a value error, and malformed data. None of them may panic.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

fn i64_data(shape: &[u64], values: &[i64]) -> TensorData {
    TensorData::new(
        DType::I64,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn i32_data(shape: &[u64], values: &[i32]) -> TensorData {
    TensorData::new(
        DType::I32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn f32_data(shape: &[u64], values: &[f32]) -> TensorData {
    TensorData::new(
        DType::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

#[track_caller]
fn assert_value(result: Result<(), TargetError>, sample: u64, element: u64, value: TargetValue) {
    match result {
        Err(TargetError::Value {
            sample: s,
            element: e,
            value: v,
            ..
        }) => {
            assert_eq!((s, e), (sample, element));
            match (v, value) {
                (TargetValue::Float(a), TargetValue::Float(b)) if a.is_nan() && b.is_nan() => {}
                _ => assert_eq!(v, value),
            }
        }
        other => panic!("expected a value error, got {other:?}"),
    }
}

#[test]
fn same_shape_layout() {
    let ok = |p: &[usize], t: &[usize]| {
        check_layout(
            "mse",
            Layout::SameShapeF32,
            (DType::F32, p),
            (DType::F32, t),
        )
    };
    ok(&[4], &[4]).unwrap();
    ok(&[4, 3, 2], &[4, 3, 2]).unwrap();
    // No broadcasting, in either direction.
    for (p, t) in [
        (&[4, 3][..], &[4][..]),
        (&[4], &[4, 1]),
        (&[4, 1], &[4]),
        (&[4, 3], &[1, 3]),
        (&[4, 3], &[3, 4]),
        (&[4, 3], &[4, 3, 1]),
    ] {
        assert!(
            matches!(ok(p, t), Err(TargetError::Shape { .. })),
            "{p:?} {t:?}"
        );
    }
    // Empty tensors and rank 0.
    for s in [&[0][..], &[4, 0], &[]] {
        assert!(matches!(ok(s, s), Err(TargetError::Shape { .. })), "{s:?}");
    }
    // Dtypes.
    let r = check_layout(
        "mse",
        Layout::SameShapeF32,
        (DType::F16, &[4]),
        (DType::F32, &[4]),
    );
    assert!(
        matches!(
            r,
            Err(TargetError::PredictionDType {
                got: DType::F16,
                ..
            })
        ),
        "{r:?}"
    );
    for t in [DType::I64, DType::I32, DType::F16, DType::Bool] {
        let r = check_layout("mse", Layout::SameShapeF32, (DType::F32, &[4]), (t, &[4]));
        assert!(
            matches!(r, Err(TargetError::TargetDType { .. })),
            "{t}: {r:?}"
        );
    }
}

#[test]
fn class_index_layout() {
    let check = |p: &[usize], t: (DType, &[usize])| {
        check_layout("cross_entropy", Layout::ClassIndices, (DType::F32, p), t)
    };
    check(&[4, 3], (DType::I64, &[4])).unwrap();
    check(&[4, 3], (DType::I32, &[4])).unwrap();
    check(&[1, 1], (DType::I64, &[1])).unwrap(); // C = 1 and a batch of 1
    for (p, t) in [
        (&[4][..], &[4][..]),
        (&[4, 3, 2], &[4]),
        (&[4, 3], &[4, 1]),
        (&[4, 3], &[3]),
        (&[4, 3], &[]),
        (&[4, 0], &[4]),
        (&[0, 3], &[0]),
    ] {
        assert!(
            matches!(check(p, (DType::I64, t)), Err(TargetError::Shape { .. })),
            "{p:?} {t:?}"
        );
    }
    for t in [DType::F32, DType::Bool, DType::F16] {
        assert!(
            matches!(
                check(&[4, 3], (t, &[4])),
                Err(TargetError::TargetDType { .. })
            ),
            "{t}"
        );
    }
    let r = check_layout(
        "cross_entropy",
        Layout::ClassIndices,
        (DType::I64, &[4, 3]),
        (DType::I64, &[4]),
    );
    assert!(
        matches!(r, Err(TargetError::PredictionDType { .. })),
        "{r:?}"
    );
}

#[test]
fn class_indices_in_range() {
    check_values(
        "cross_entropy",
        Rule::ClassIndex,
        3,
        &i64_data(&[5], &[0, 1, 2, 2, 0]),
    )
    .unwrap();
    check_values(
        "cross_entropy",
        Rule::ClassIndex,
        3,
        &i32_data(&[5], &[0, 1, 2, 2, 0]),
    )
    .unwrap();
    check_values(
        "cross_entropy",
        Rule::ClassIndex,
        1,
        &i64_data(&[2], &[0, 0]),
    )
    .unwrap();
    // The first bad sample is named, with its value.
    let r = check_values(
        "cross_entropy",
        Rule::ClassIndex,
        3,
        &i64_data(&[5], &[0, 1, 3, -1, 0]),
    );
    assert_value(r, 2, 0, TargetValue::Int(3));
    let r = check_values(
        "cross_entropy",
        Rule::ClassIndex,
        3,
        &i32_data(&[5], &[0, 1, 2, -1, 9]),
    );
    assert_value(r, 3, 0, TargetValue::Int(-1));
    let r = check_values(
        "cross_entropy",
        Rule::ClassIndex,
        10,
        &i64_data(&[2], &[i64::MIN, 0]),
    );
    assert_value(r, 0, 0, TargetValue::Int(i64::MIN));
    let r = check_values(
        "cross_entropy",
        Rule::ClassIndex,
        10,
        &i64_data(&[2], &[0, i64::MAX]),
    );
    assert_value(r, 1, 0, TargetValue::Int(i64::MAX));
    let r = check_values("cross_entropy", Rule::ClassIndex, 1, &i32_data(&[1], &[1]));
    assert_value(r, 0, 0, TargetValue::Int(1));
    let msg = check_values("cross_entropy", Rule::ClassIndex, 3, &i64_data(&[1], &[7]))
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("sample 0") && msg.contains("target 7") && msg.contains("[0, 3)"),
        "{msg}"
    );
    // A float field is not a class-index field.
    let r = check_values(
        "cross_entropy",
        Rule::ClassIndex,
        3,
        &f32_data(&[1], &[1.0]),
    );
    assert!(matches!(r, Err(TargetError::TargetDType { .. })), "{r:?}");
}

#[test]
fn unit_interval_targets() {
    let ok = [0.0, 1.0, 0.5, -0.0, 1e-30, 0.999_999_9];
    check_values(
        "binary_cross_entropy",
        Rule::UnitInterval,
        0,
        &f32_data(&[6], &ok),
    )
    .unwrap();
    for (bad, at) in [
        (f32::NAN, 1),
        (-1e-30, 2),
        (1.000_000_1, 0),
        (f32::INFINITY, 3),
        (f32::NEG_INFINITY, 1),
        (2.0, 2),
    ] {
        let mut v = vec![0.25; 4];
        v[at] = bad;
        let r = check_values(
            "binary_cross_entropy",
            Rule::UnitInterval,
            0,
            &f32_data(&[4], &v),
        );
        assert_value(r, at as u64, 0, TargetValue::Float(bad));
    }
    // Samples with several elements: [N, 2, 3] names sample and element.
    let mut v = vec![0.0; 3 * 2 * 3];
    v[2 * 6 + 4] = 1.5;
    let r = check_values(
        "binary_cross_entropy",
        Rule::UnitInterval,
        0,
        &f32_data(&[3, 2, 3], &v),
    );
    assert_value(r, 2, 4, TargetValue::Float(1.5));
    let r = check_values(
        "binary_cross_entropy",
        Rule::UnitInterval,
        0,
        &i64_data(&[1], &[1]),
    );
    assert!(matches!(r, Err(TargetError::TargetDType { .. })), "{r:?}");
}

#[test]
fn zero_or_one_targets() {
    check_values(
        "accuracy",
        Rule::ZeroOrOne,
        0,
        &f32_data(&[4], &[0.0, 1.0, 1.0, -0.0]),
    )
    .unwrap();
    for bad in [
        0.5,
        0.999_999_9,
        1.000_000_1,
        1e-38,
        f32::NAN,
        -1.0,
        2.0,
        f32::INFINITY,
    ] {
        let r = check_values(
            "accuracy",
            Rule::ZeroOrOne,
            0,
            &f32_data(&[2, 2], &[1.0, 0.0, 0.0, bad]),
        );
        assert_value(r, 1, 1, TargetValue::Float(bad));
    }
}

#[test]
fn malformed_data_is_an_error() {
    // Byte length doesn't match the shape (the fields are public, so this can be built).
    let bad = TensorData {
        dtype: DType::I64,
        shape: vec![4],
        bytes: vec![0; 31],
    };
    for rule in [Rule::Any, Rule::ClassIndex] {
        let r = check_values("cross_entropy", rule, 3, &bad);
        assert!(matches!(r, Err(TargetError::Data { .. })), "{r:?}");
    }
    let huge = TensorData {
        dtype: DType::F32,
        shape: vec![u64::MAX, u64::MAX],
        bytes: vec![],
    };
    let r = check_values("binary_cross_entropy", Rule::UnitInterval, 0, &huge);
    assert!(matches!(r, Err(TargetError::Data { .. })), "{r:?}");
    // A field without a sample dim.
    let scalar = f32_data(&[], &[0.5]);
    let r = check_values("binary_cross_entropy", Rule::UnitInterval, 0, &scalar);
    assert!(matches!(r, Err(TargetError::Shape { .. })), "{r:?}");
    // Empty data has nothing to check.
    check_values("cross_entropy", Rule::ClassIndex, 3, &i64_data(&[0], &[])).unwrap();
    check_values(
        "binary_cross_entropy",
        Rule::UnitInterval,
        0,
        &f32_data(&[0, 5], &[]),
    )
    .unwrap();
}
