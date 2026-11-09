//! Unit tests of `shape.rs`: every bad argument is an error, never a panic, and views behave.
//! Numerical results are checked against PyTorch fixtures in `tests/reference_shape.rs`.

#![cfg(feature = "cpu")]

use super::*;
use crate::device::DeviceId;
use crate::tensor::TensorData;

fn arange(shape: &[u64]) -> DynTensor {
    let count: u64 = shape.iter().product();
    let bytes = (0..count as i64).flat_map(|v| v.to_le_bytes()).collect();
    let data = TensorData::new(DType::I64, shape.to_vec(), bytes).unwrap();
    DynTensor::from_data(&data, DeviceId::Cpu).unwrap()
}

fn values(t: DynTensor) -> Vec<i64> {
    let data = t.into_data().unwrap();
    let (chunks, _) = data.bytes.as_chunks::<8>();
    chunks.iter().map(|&b| i64::from_le_bytes(b)).collect()
}

fn invalid(result: Result<impl std::fmt::Debug>, op: &str) {
    match result {
        Err(TensorError::InvalidArgument { op: got, .. }) => assert_eq!(got, op),
        other => panic!("{op}: expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn reshape_checks_rank_count_and_zero_dims() {
    let t = arange(&[2, 3]);
    assert_eq!(
        t.clone().reshape(&[1; 7]).unwrap_err(),
        TensorError::UnsupportedRank { rank: 7 }
    );
    assert!(matches!(
        t.clone().reshape(&[4, 2]),
        Err(TensorError::ShapeMismatch { op: "reshape", .. })
    ));
    // An element count that overflows is a mismatch, not a panic.
    assert!(matches!(
        t.clone().reshape(&[usize::MAX, 2]),
        Err(TensorError::ShapeMismatch { .. })
    ));
    let empty = TensorData::new(DType::F32, vec![0, 3], vec![]).unwrap();
    let empty = DynTensor::from_data(&empty, DeviceId::Cpu).unwrap();
    invalid(empty.reshape(&[3, 0]), "reshape");
    assert_eq!(
        values(t.reshape(&[3, 2]).unwrap()),
        (0..6).collect::<Vec<_>>()
    );
}

#[test]
fn flatten_permute_transpose_errors() {
    let t = arange(&[2, 3, 4]);
    invalid(t.clone().flatten(2, 1), "flatten");
    invalid(t.clone().flatten(0, 3), "flatten");
    assert_eq!(t.clone().flatten(0, 2).unwrap().shape(), vec![24]);
    for dims in [&[0, 1][..], &[0, 1, 1], &[0, 1, 3], &[0, 1, 2, 3]] {
        invalid(t.clone().permute(dims), "permute");
    }
    invalid(t.clone().transpose(0, 3), "transpose");
    invalid(t.clone().transpose(5, 0), "transpose");
    assert_eq!(t.clone().transpose(1, 1).unwrap().shape(), vec![2, 3, 4]);
    let p = t.permute(&[2, 0, 1]).unwrap();
    assert_eq!(p.shape(), vec![4, 2, 3]);
    // A permuted view reads back in its logical order.
    assert_eq!(
        values(p.clone().slice(0, 1, 2, 1).unwrap()),
        vec![1, 5, 9, 13, 17, 21]
    );
}

#[test]
fn squeeze_and_unsqueeze_errors() {
    let t = arange(&[1, 3, 1]);
    invalid(t.clone().squeeze(&[1]), "squeeze"); // size 3
    invalid(t.clone().squeeze(&[0, 0]), "squeeze");
    invalid(t.clone().squeeze(&[3]), "squeeze");
    assert_eq!(t.clone().squeeze(&[]).unwrap().shape(), vec![1, 3, 1]);
    assert_eq!(t.clone().squeeze(&[2, 0]).unwrap().shape(), vec![3]);
    assert_eq!(arange(&[1, 1]).squeeze(&[0, 1]).unwrap().shape(), vec![1]);
    invalid(t.clone().unsqueeze(4), "unsqueeze");
    assert_eq!(t.unsqueeze(3).unwrap().shape(), vec![1, 3, 1, 1]);
    assert_eq!(
        arange(&[1; 6]).unsqueeze(0).unwrap_err(),
        TensorError::UnsupportedRank { rank: 7 }
    );
}

#[test]
fn expand_errors() {
    let t = arange(&[3, 1]);
    assert!(matches!(
        t.clone().expand(&[3, 2, 4]),
        Err(TensorError::ShapeMismatch { op: "expand", .. })
    ));
    assert!(matches!(
        t.clone().expand(&[4]),
        Err(TensorError::ShapeMismatch { op: "expand", .. })
    ));
    invalid(t.clone().expand(&[0, 3, 4]), "expand");
    assert_eq!(
        t.clone().expand(&[1; 7]).unwrap_err(),
        TensorError::UnsupportedRank { rank: 7 }
    );
    let e = t.expand(&[2, 3, 2]).unwrap();
    assert_eq!(values(e), vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2]);
}

#[test]
fn concat_errors() {
    let a = arange(&[2, 3]);
    invalid(DynTensor::concat(&[], 0), "concat");
    invalid(DynTensor::concat(std::slice::from_ref(&a), 2), "concat");
    let too_many = vec![a.clone(); MAX_CONCAT_SPLIT + 1];
    invalid(DynTensor::concat(&too_many, 0), "concat");
    assert!(matches!(
        DynTensor::concat(&[a.clone(), arange(&[3, 3])], 1),
        Err(TensorError::ShapeMismatch { op: "concat", .. })
    ));
    assert!(matches!(
        DynTensor::concat(&[a.clone(), arange(&[6])], 0),
        Err(TensorError::ShapeMismatch { op: "concat", .. })
    ));
    let f = a.clone().cast(DType::F32).unwrap();
    assert!(matches!(
        DynTensor::concat(&[a.clone(), f], 0),
        Err(TensorError::DTypeMismatch { op: "concat", .. })
    ));
    let max = vec![arange(&[1]); MAX_CONCAT_SPLIT];
    assert_eq!(
        DynTensor::concat(&max, 0).unwrap().shape(),
        vec![MAX_CONCAT_SPLIT]
    );
}

#[test]
fn split_and_slice_errors() {
    let t = arange(&[2, 5]);
    invalid(t.clone().split(1, &[]), "split");
    invalid(t.clone().split(1, &[2, 2]), "split");
    invalid(t.clone().split(1, &[5, 0]), "split");
    invalid(t.clone().split(2, &[2]), "split");
    invalid(t.clone().split(1, &[usize::MAX, 6]), "split");
    invalid(t.clone().split(1, &[1; MAX_CONCAT_SPLIT + 1]), "split");
    let parts = t.clone().split(1, &[2, 3]).unwrap();
    assert_eq!(values(parts[1].clone()), vec![2, 3, 4, 7, 8, 9]);

    invalid(t.clone().slice(1, 0, 5, 0), "slice");
    invalid(t.clone().slice(1, 3, 3, 1), "slice");
    invalid(t.clone().slice(1, 4, 2, 1), "slice");
    invalid(t.clone().slice(1, 0, 6, 1), "slice");
    invalid(t.clone().slice(2, 0, 1, 1), "slice");
    // A huge step selects only `start`.
    assert_eq!(
        values(t.clone().slice(1, 1, 5, usize::MAX).unwrap()),
        vec![1, 6]
    );
    assert_eq!(values(t.slice(1, 0, 5, 2).unwrap()), vec![0, 2, 4, 5, 7, 9]);
}

#[test]
fn pad_errors() {
    let t = arange(&[2, 2]);
    invalid(t.clone().pad(&[], Scalar::Int(0)), "pad");
    invalid(t.clone().pad(&[1], Scalar::Int(0)), "pad");
    invalid(t.clone().pad(&[1, 1, 1, 1, 1, 1], Scalar::Int(0)), "pad");
    invalid(t.clone().pad(&[1, usize::MAX], Scalar::Int(0)), "pad");
    // The value must fit the dtype (§3.1.7).
    invalid(t.clone().pad(&[1, 1], Scalar::Float(0.5)), "pad");
    invalid(t.clone().pad(&[1, 1], Scalar::Bool(true)), "pad");
    let i32s = t.clone().cast(DType::I32).unwrap();
    invalid(i32s.pad(&[1, 1], Scalar::Int(i64::MAX)), "pad");
    let bools = t.clone().cast(DType::Bool).unwrap();
    invalid(bools.pad(&[1, 1], Scalar::Int(0)), "pad");
    let padded = t.pad(&[1, 0, 0, 1], Scalar::Int(-1)).unwrap();
    assert_eq!(padded.shape(), vec![3, 3]);
    assert_eq!(values(padded), vec![-1, 0, 1, -1, 2, 3, -1, -1, -1]);
}

#[test]
fn errors_name_the_op() {
    let t = arange(&[2, 3]);
    assert!(matches!(
        t.clone().flatten(0, 5),
        Err(TensorError::InvalidArgument { op: "flatten", .. })
    ));
    // A shape derived from a valid argument that turns out invalid is reported under the op.
    let empty = TensorData::new(DType::F32, vec![0, 3], vec![]).unwrap();
    let empty = DynTensor::from_data(&empty, DeviceId::Cpu).unwrap();
    assert!(matches!(
        empty.flatten(0, 1),
        Err(TensorError::InvalidArgument { op: "flatten", .. })
    ));
}
