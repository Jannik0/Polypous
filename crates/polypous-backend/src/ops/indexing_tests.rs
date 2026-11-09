//! Unit tests of `indexing.rs`: argument checks, never a panic. Out-of-range indices on every
//! device and the numerical results are tested in `tests/reference_indexing.rs`.

#![cfg(feature = "cpu")]

use super::*;
use crate::device::DeviceId;
use crate::tensor::{DType, TensorData};

fn ints(dtype: DType, shape: &[u64], values: &[i64]) -> DynTensor {
    let bytes = match dtype {
        DType::I32 => values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
    };
    let data = TensorData::new(dtype, shape.to_vec(), bytes).unwrap();
    DynTensor::from_data(&data, DeviceId::Cpu).unwrap()
}

#[test]
fn index_dtype_must_be_an_integer() {
    let x = ints(DType::I64, &[2, 2], &[1, 2, 3, 4]);
    for dtype in [DType::F32, DType::Bool] {
        let index = ints(DType::I64, &[2, 2], &[0, 1, 1, 0])
            .cast(dtype)
            .unwrap();
        assert_eq!(
            x.clone().gather(0, index.clone()).unwrap_err(),
            TensorError::UnsupportedOperation {
                op: "gather",
                dtype
            }
        );
        let index = index.reshape(&[4]).unwrap();
        assert_eq!(
            x.clone().index_select(0, index).unwrap_err(),
            TensorError::UnsupportedOperation {
                op: "index_select",
                dtype
            }
        );
    }
}

#[test]
fn gather_shape_rules() {
    let x = ints(DType::I64, &[2, 3], &[1, 2, 3, 4, 5, 6]);
    // Different rank.
    let index = ints(DType::I64, &[2], &[0, 1]);
    assert!(matches!(
        x.clone().gather(1, index),
        Err(TensorError::ShapeMismatch { op: "gather", .. })
    ));
    // Larger than x outside `dim`.
    let index = ints(DType::I64, &[3, 1], &[0, 1, 0]);
    assert!(matches!(
        x.clone().gather(1, index),
        Err(TensorError::ShapeMismatch { op: "gather", .. })
    ));
    // Axis out of range.
    let index = ints(DType::I64, &[2, 3], &[0; 6]);
    assert!(matches!(
        x.clone().gather(2, index),
        Err(TensorError::InvalidArgument { op: "gather", .. })
    ));
    // Smaller outside `dim` reads the leading part (PyTorch's rule).
    let index = ints(DType::I32, &[1, 2], &[2, 0]);
    let out = x.gather(1, index).unwrap().into_data().unwrap();
    assert_eq!(out.shape, vec![1, 2]);
    assert_eq!(out.bytes[..8], 3i64.to_le_bytes());
    assert_eq!(out.bytes[8..], 1i64.to_le_bytes());
}

#[test]
fn index_select_shape_rules() {
    let x = ints(DType::I64, &[2, 3], &[1, 2, 3, 4, 5, 6]);
    let index = ints(DType::I64, &[1, 2], &[0, 1]);
    assert!(matches!(
        x.clone().index_select(0, index),
        Err(TensorError::InvalidArgument {
            op: "index_select",
            ..
        })
    ));
    let index = ints(DType::I64, &[2], &[0, 1]);
    assert!(matches!(
        x.clone().index_select(2, index),
        Err(TensorError::InvalidArgument {
            op: "index_select",
            ..
        })
    ));
    let index = ints(DType::I32, &[4], &[2, 2, 0, 1]);
    assert_eq!(x.index_select(1, index).unwrap().shape(), vec![2, 4]);
}

#[test]
fn out_of_range_is_checked_before_the_op() {
    let x = ints(DType::I64, &[2, 3], &[1, 2, 3, 4, 5, 6]);
    // flex would panic on these inside Burn.
    let index = ints(DType::I64, &[2, 3], &[0, 1, 2, 0, 1, 3]);
    assert_eq!(
        x.clone().gather(1, index).unwrap_err(),
        TensorError::IndexOutOfRange {
            op: "gather",
            index: 3,
            size: 3
        }
    );
    let index = ints(DType::I32, &[1], &[-1]);
    assert_eq!(
        x.index_select(0, index).unwrap_err(),
        TensorError::IndexOutOfRange {
            op: "index_select",
            index: -1,
            size: 2
        }
    );
    assert_eq!(
        TensorError::IndexOutOfRange {
            op: "gather",
            index: 7,
            size: 3
        }
        .to_string(),
        "index 7 is out of range in `gather`: valid indices are 0 to 3 (exclusive)"
    );
}
