//! Unit tests of `pool.rs`: argument checks (errors, never panics), the NaN and `-inf` rules of
//! `max_pool2d`, and the divisors of `avg_pool2d` on the CPU. Results against PyTorch are in
//! `tests/reference_nn.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, read_f32};
use super::*;
use crate::tensor::DType;

fn ones(shape: &[u64]) -> DynTensor {
    floats(shape, &vec![1.0; shape.iter().product::<u64>() as usize])
}

fn invalid(result: Result<DynTensor>, op: &str) {
    match result {
        Err(TensorError::InvalidArgument { op: got, .. }) if got == op => {}
        other => panic!("expected InvalidArgument from {op}, got {other:?}"),
    }
}

#[test]
fn pool_errors() {
    let x = || ones(&[1, 1, 4, 4]);
    let mp = |k, s, p, d| x().max_pool2d(k, s, p, d);
    invalid(
        ones(&[1, 4, 4]).max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]),
        "max_pool2d",
    );
    invalid(mp([0, 2], [1, 1], [0, 0], [1, 1]), "max_pool2d");
    invalid(mp([2, 2], [0, 1], [0, 0], [1, 1]), "max_pool2d");
    invalid(mp([2, 2], [1, 1], [0, 0], [1, 0]), "max_pool2d");
    // padding at most kernel_size / 2 (integer division): k 3 allows 1, k 2 allows 1.
    invalid(mp([3, 3], [1, 1], [2, 0], [1, 1]), "max_pool2d");
    invalid(mp([1, 1], [1, 1], [0, 1], [1, 1]), "max_pool2d");
    assert!(mp([2, 2], [1, 1], [1, 1], [1, 1]).is_ok());
    // Output below 1.
    invalid(mp([3, 3], [1, 1], [0, 0], [3, 1]), "max_pool2d");
    let ap = |k, s, p| x().avg_pool2d(k, s, p, true);
    invalid(ap([5, 1], [1, 1], [0, 0]), "avg_pool2d");
    invalid(ap([2, 2], [1, 1], [2, 0]), "avg_pool2d");
    invalid(ap([2, 0], [1, 1], [0, 0]), "avg_pool2d");
    invalid(x().adaptive_avg_pool2d([0, 2]), "adaptive_avg_pool2d");
    invalid(
        ones(&[4, 4]).adaptive_avg_pool2d([2, 2]),
        "adaptive_avg_pool2d",
    );
    assert!(matches!(
        x().cast(DType::I32)
            .unwrap()
            .max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]),
        Err(TensorError::UnsupportedOperation { .. })
    ));
}

#[test]
fn max_pool_nan_and_neg_inf() {
    let nan = f32::NAN;
    let x = floats(&[1, 1, 2, 4], &[1.0, nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    let out = read_f32(x.max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]).unwrap());
    assert!(out[0].is_nan());
    assert_eq!(out[1], 7.0);
    // All-padding windows (dilation 3 skips the whole input) and padded windows of only -inf.
    let out = ones(&[1, 1, 2, 2])
        .max_pool2d([2, 1], [1, 1], [1, 0], [3, 1])
        .unwrap();
    assert_eq!(read_f32(out), [f32::NEG_INFINITY; 2]);
    let ninf = f32::NEG_INFINITY;
    let x = floats(&[1, 1, 2, 3], &[ninf, ninf, 1.0, ninf, ninf, 2.0]);
    let out = read_f32(x.max_pool2d([3, 3], [1, 1], [1, 1], [1, 1]).unwrap());
    assert_eq!(out, [ninf, 2.0, 2.0, ninf, 2.0, 2.0]);
}

#[test]
fn avg_pool_divisors() {
    let x = || ones(&[1, 1, 2, 2]);
    // One 3x3 window over the padded 2x2 input: 4 ones.
    let include = read_f32(x().avg_pool2d([3, 3], [1, 1], [1, 1], true).unwrap());
    assert!((include[0] - 4.0 / 9.0).abs() < 1e-6);
    let exclude = read_f32(x().avg_pool2d([3, 3], [1, 1], [1, 1], false).unwrap());
    assert_eq!(exclude, [1.0; 4]);
    let out = x().adaptive_avg_pool2d([3, 1]).unwrap();
    assert_eq!(out.shape(), [1, 1, 3, 1]);
    assert_eq!(read_f32(out), [1.0; 3]);
}
