//! Unit tests of `conv.rs`: the output size arithmetic and every argument check (errors, never
//! panics) on the CPU. Results against PyTorch are in `tests/reference_nn.rs`.

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
fn window_output_sizes() {
    let size = |n, k, s, p, d| window_output("t", "H", n, k, s, p, d);
    assert_eq!(size(10, 3, 1, 0, 1).unwrap(), 8);
    assert_eq!(size(11, 3, 2, 1, 1).unwrap(), 6);
    assert_eq!(size(12, 3, 1, 0, 2).unwrap(), 8);
    assert_eq!(size(3, 5, 1, 1, 1).unwrap(), 1);
    assert_eq!(size(2, 2, 1, 1, 3).unwrap(), 1);
    assert!(size(3, 5, 1, 0, 1).is_err());
    assert!(size(2, 2, 1, 0, 3).is_err());
    // Overflow is an error, not a panic or a wrap-around.
    assert!(size(1, usize::MAX, 1, 0, 2).is_err());
    assert!(size(1, 2, 1, usize::MAX, 1).is_err());
    assert!(size(usize::MAX, 1, 1, 1, 1).is_err());
}

#[test]
fn conv_shapes_and_values() {
    let out = ones(&[1, 2, 5]).conv1d(ones(&[3, 2, 3]), Some(ones(&[3])), 1, 0, 1, 1);
    let out = out.unwrap();
    assert_eq!(out.shape(), [1, 3, 3]);
    assert_eq!(read_f32(out), [7.0; 9]);
    let out = ones(&[2, 4, 5, 6])
        .conv2d(ones(&[6, 2, 3, 2]), None, [2, 1], [1, 0], [1, 2], 2)
        .unwrap();
    assert_eq!(out.shape(), [2, 6, 3, 4]);
    // f16 keeps its dtype (computed in f32).
    let out = ones(&[1, 1, 4])
        .cast(DType::F16)
        .unwrap()
        .conv1d(ones(&[1, 1, 2]).cast(DType::F16).unwrap(), None, 1, 0, 1, 1)
        .unwrap();
    assert_eq!(out.dtype(), DType::F16);
    assert_eq!(read_f32(out), [2.0; 3]);
}

#[test]
fn conv_errors() {
    let c1 = |x: &[u64], w: &[u64], b: Option<&[u64]>, s, p, d, g| {
        ones(x).conv1d(ones(w), b.map(ones), s, p, d, g)
    };
    // x rank, weight rank, C_in mismatch, groups not dividing, zero stride/dilation/groups.
    invalid(c1(&[2, 5], &[3, 2, 3], None, 1, 0, 1, 1), "conv1d");
    invalid(c1(&[1, 2, 5], &[3, 2], None, 1, 0, 1, 1), "conv1d");
    invalid(c1(&[1, 2, 5], &[3, 1, 3], None, 1, 0, 1, 1), "conv1d");
    invalid(c1(&[1, 4, 5], &[3, 2, 3], None, 1, 0, 1, 2), "conv1d");
    invalid(c1(&[1, 3, 5], &[4, 1, 3], None, 1, 0, 1, 2), "conv1d");
    invalid(c1(&[1, 2, 5], &[3, 2, 3], None, 0, 0, 1, 1), "conv1d");
    invalid(c1(&[1, 2, 5], &[3, 2, 3], None, 1, 0, 0, 1), "conv1d");
    invalid(c1(&[1, 2, 5], &[3, 2, 3], None, 1, 0, 1, 0), "conv1d");
    // Bias shape, output below 1, overflowing padding.
    invalid(c1(&[1, 2, 5], &[3, 2, 3], Some(&[2]), 1, 0, 1, 1), "conv1d");
    invalid(c1(&[1, 2, 2], &[3, 2, 3], None, 1, 0, 1, 1), "conv1d");
    invalid(
        c1(&[1, 2, 5], &[3, 2, 3], None, 1, usize::MAX, 1, 1),
        "conv1d",
    );
    let x = || ones(&[1, 2, 4, 4]);
    invalid(
        x().conv2d(ones(&[3, 2, 3]), None, [1, 1], [0, 0], [1, 1], 1),
        "conv2d",
    );
    invalid(
        x().conv2d(ones(&[3, 2, 5, 1]), None, [1, 1], [0, 0], [1, 1], 1),
        "conv2d",
    );
    invalid(
        x().conv2d(ones(&[3, 2, 3, 3]), None, [1, 0], [0, 0], [1, 1], 1),
        "conv2d",
    );
    assert_eq!(
        x().conv2d(
            ones(&[3, 2, 3, 3]).cast(DType::F16).unwrap(),
            None,
            [1, 1],
            [0, 0],
            [1, 1],
            1
        )
        .unwrap_err(),
        TensorError::DTypeMismatch {
            op: "conv2d",
            lhs: DType::F32,
            rhs: DType::F16
        }
    );
}
