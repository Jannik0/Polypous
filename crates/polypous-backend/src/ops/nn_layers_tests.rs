//! Unit tests of `nn_layers.rs`: argument checks of `linear`, `embedding` and `dropout` (errors,
//! never panics) and output shapes on the CPU. Results against PyTorch are in
//! `tests/reference_nn.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, ints, read_f32};
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

#[test]
fn linear_shapes() {
    let w = ones(&[4, 3]);
    let b = Some(ones(&[4]));
    assert_eq!(
        ones(&[3]).linear(w.clone(), b.clone()).unwrap().shape(),
        [4]
    );
    assert_eq!(
        ones(&[2, 1, 5, 3]).linear(w.clone(), None).unwrap().shape(),
        [2, 1, 5, 4]
    );
    assert_eq!(
        read_f32(ones(&[1, 3]).linear(w, b).unwrap()),
        [4.0, 4.0, 4.0, 4.0]
    );
}

#[test]
fn linear_errors() {
    let x = || ones(&[2, 3]);
    invalid(x().linear(ones(&[4, 2]), None), "linear");
    invalid(x().linear(ones(&[12]), None), "linear");
    invalid(x().linear(ones(&[4, 3]), Some(ones(&[3]))), "linear");
    invalid(x().linear(ones(&[4, 3]), Some(ones(&[1, 4]))), "linear");
    assert_eq!(
        x().linear(ones(&[4, 3]).cast(DType::F16).unwrap(), None)
            .unwrap_err(),
        TensorError::DTypeMismatch {
            op: "linear",
            lhs: DType::F32,
            rhs: DType::F16
        }
    );
    assert_eq!(
        x().linear(ones(&[4, 3]), Some(ones(&[4]).cast(DType::BF16).unwrap()))
            .unwrap_err(),
        TensorError::DTypeMismatch {
            op: "linear",
            lhs: DType::F32,
            rhs: DType::BF16
        }
    );
    assert_eq!(
        ints(DType::I64, &[3], &[1, 2, 3])
            .linear(ones(&[4, 3]), None)
            .unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "linear",
            dtype: DType::I64
        }
    );
}

#[test]
fn embedding_rows_and_errors() {
    let weight = || floats(&[3, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
    for dtype in [DType::I32, DType::I64] {
        let out = ints(dtype, &[2, 2], &[2, 0, 1, 2])
            .embedding(weight())
            .unwrap();
        assert_eq!(out.shape(), [2, 2, 2]);
        assert_eq!(
            read_f32(out),
            [20.0, 21.0, 0.0, 1.0, 10.0, 11.0, 20.0, 21.0]
        );
        for bad in [-1, 3, i64::from(i32::MAX)] {
            assert_eq!(
                ints(dtype, &[3], &[0, bad, 1])
                    .embedding(weight())
                    .unwrap_err(),
                TensorError::IndexOutOfRange {
                    op: "embedding",
                    index: bad,
                    size: 3
                }
            );
        }
    }
    // ids must be integers, the weight float and rank 2; rank-6 ids would give rank 7.
    assert_eq!(
        ones(&[2]).embedding(weight()).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "embedding",
            dtype: DType::F32
        }
    );
    assert_eq!(
        ints(DType::I64, &[1], &[0])
            .embedding(ints(DType::I64, &[2, 2], &[1, 2, 3, 4]))
            .unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "embedding",
            dtype: DType::I64
        }
    );
    invalid(
        ints(DType::I64, &[1], &[0]).embedding(ones(&[6])),
        "embedding",
    );
    invalid(
        ints(DType::I64, &[1, 1, 1, 1, 1, 1], &[0]).embedding(weight()),
        "embedding",
    );
}

#[test]
fn dropout_is_the_identity_and_checks_p() {
    let x = || floats(&[3], &[1.0, f32::NAN, -2.0]);
    for p in [0.0, 0.5, 0.999] {
        let out = read_f32(x().dropout(p).unwrap());
        assert_eq!(out[0], 1.0);
        assert!(out[1].is_nan());
        assert_eq!(out[2], -2.0);
    }
    for p in [-0.1, 1.0, 1.5, f64::NAN, f64::INFINITY] {
        invalid(x().dropout(p), "dropout");
    }
    assert!(matches!(
        ints(DType::I32, &[1], &[1]).dropout(0.5),
        Err(TensorError::UnsupportedOperation { .. })
    ));
}
