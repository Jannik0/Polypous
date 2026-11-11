//! Unit tests of `norm.rs`: argument checks (errors, never panics) and the formulas on small
//! inputs on the CPU (biased variance, `eps` inside the root, large means). Results against
//! PyTorch are in `tests/reference_nn.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, read_f32};
use super::*;

fn ones(shape: &[u64]) -> DynTensor {
    floats(shape, &vec![1.0; shape.iter().product::<u64>() as usize])
}

fn invalid(result: Result<DynTensor>, op: &str) {
    match result {
        Err(TensorError::InvalidArgument { op: got, .. }) if got == op => {}
        other => panic!("expected InvalidArgument from {op}, got {other:?}"),
    }
}

fn close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!(
            (a - e).abs() <= 1e-5 + 1e-4 * e.abs(),
            "{actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn formulas() {
    // layer_norm of [1, 3]: mean 2, biased variance 1, so ±1/sqrt(1 + eps).
    let x = || floats(&[1, 2], &[1.0, 3.0]);
    let s = 1.0 / (1.0f32 + 0.25).sqrt();
    close(
        &read_f32(x().layer_norm(&[2], None, None, 0.25).unwrap()),
        &[-s, s],
    );
    // rms_norm: mean(x²) = 5.
    let r = 1.0 / (5.0f32 + 0.25).sqrt();
    close(
        &read_f32(
            x().rms_norm(&[2], Some(floats(&[2], &[2.0, 1.0])), 0.25)
                .unwrap(),
        ),
        &[2.0 * r, 3.0 * r],
    );
    // A large mean: a one-pass variance would cancel to 0 or below in f32.
    let big = floats(&[1, 2], &[1.0e7, 1.0e7 + 2.0]);
    close(
        &read_f32(big.layer_norm(&[2], None, None, 1e-5).unwrap()),
        &[-1.0, 1.0],
    );
    // batch_norm: (x - mean) / sqrt(var + eps) · w + b per channel.
    let x = floats(&[1, 2, 1], &[3.0, 5.0]);
    let out = x
        .batch_norm(
            floats(&[2], &[1.0, 1.0]),
            floats(&[2], &[3.75, 0.75]),
            Some(floats(&[2], &[1.0, 2.0])),
            Some(floats(&[2], &[0.0, 1.0])),
            0.25,
        )
        .unwrap();
    close(&read_f32(out), &[1.0, 9.0]);
    // group_norm with 2 groups of 1 channel: each channel normalized on its own.
    let x = floats(&[1, 2, 2], &[1.0, 3.0, 10.0, 30.0]);
    let out = read_f32(x.group_norm(2, None, None, 1e-5).unwrap());
    close(&out, &[-1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn eps_and_shapes_are_checked() {
    for eps in [0.0, -1e-5, f64::NAN, f64::INFINITY] {
        invalid(
            ones(&[2, 3]).layer_norm(&[3], None, None, eps),
            "layer_norm",
        );
        invalid(ones(&[2, 3]).rms_norm(&[3], None, eps), "rms_norm");
        invalid(ones(&[2, 3]).group_norm(3, None, None, eps), "group_norm");
        invalid(
            ones(&[2, 3]).batch_norm(ones(&[3]), ones(&[3]), None, None, eps),
            "batch_norm",
        );
    }
    let x = || ones(&[2, 3, 4]);
    invalid(x().layer_norm(&[], None, None, 1e-5), "layer_norm");
    invalid(x().layer_norm(&[3], None, None, 1e-5), "layer_norm");
    invalid(
        x().layer_norm(&[1, 2, 3, 4], None, None, 1e-5),
        "layer_norm",
    );
    invalid(
        x().layer_norm(&[4], Some(ones(&[3])), None, 1e-5),
        "layer_norm",
    );
    invalid(
        x().layer_norm(&[3, 4], None, Some(ones(&[4])), 1e-5),
        "layer_norm",
    );
    invalid(x().rms_norm(&[4], Some(ones(&[1, 4])), 1e-5), "rms_norm");
    invalid(x().group_norm(0, None, None, 1e-5), "group_norm");
    invalid(x().group_norm(2, None, None, 1e-5), "group_norm");
    invalid(
        x().group_norm(3, Some(ones(&[4])), None, 1e-5),
        "group_norm",
    );
    invalid(ones(&[3]).group_norm(1, None, None, 1e-5), "group_norm");
    invalid(
        ones(&[3]).batch_norm(ones(&[3]), ones(&[3]), None, None, 1e-5),
        "batch_norm",
    );
    invalid(
        x().batch_norm(ones(&[4]), ones(&[3]), None, None, 1e-5),
        "batch_norm",
    );
    invalid(
        x().batch_norm(ones(&[3]), ones(&[3]), Some(ones(&[2])), None, 1e-5),
        "batch_norm",
    );
    assert_eq!(
        x().batch_norm(
            ones(&[3]),
            ones(&[3]).cast(DType::F16).unwrap(),
            None,
            None,
            1e-5
        )
        .unwrap_err(),
        TensorError::DTypeMismatch {
            op: "batch_norm",
            lhs: DType::F32,
            rhs: DType::F16
        }
    );
}

#[test]
fn half_precision_keeps_the_dtype() {
    for dtype in [DType::F16, DType::BF16] {
        let x = floats(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0])
            .cast(dtype)
            .unwrap();
        let w = ones(&[2]).cast(dtype).unwrap();
        assert_eq!(
            x.clone()
                .layer_norm(&[2], Some(w.clone()), None, 1e-5)
                .unwrap()
                .dtype(),
            dtype
        );
        assert_eq!(
            x.clone()
                .rms_norm(&[2], Some(w.clone()), 1e-5)
                .unwrap()
                .dtype(),
            dtype
        );
        assert_eq!(
            x.clone()
                .group_norm(1, Some(w.clone()), None, 1e-5)
                .unwrap()
                .dtype(),
            dtype
        );
        let out = x
            .batch_norm(w.clone(), w.clone(), Some(w.clone()), Some(w), 1e-5)
            .unwrap();
        assert_eq!(out.dtype(), dtype);
    }
}
