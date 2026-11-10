//! Unit tests of `reduce.rs`: argument checks (never a panic), output shapes, and the NaN, tie and
//! wrap-around rules on the CPU. Results against PyTorch are in `tests/reference_reduce.rs`.

#![cfg(feature = "cpu")]

use super::*;
use crate::tensor::TensorData;

pub(crate) fn floats(shape: &[u64], values: &[f32]) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let data = TensorData::new(DType::F32, shape.to_vec(), bytes).unwrap();
    DynTensor::from_data(&data, DeviceId::Cpu).unwrap()
}

pub(crate) fn ints(dtype: DType, shape: &[u64], values: &[i64]) -> DynTensor {
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

pub(crate) fn read_f32(t: DynTensor) -> Vec<f32> {
    let t = t.cast(DType::F32).unwrap();
    let data = t.into_data().unwrap();
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

pub(crate) fn read_i64(t: DynTensor) -> Vec<i64> {
    let data = t.into_data().unwrap();
    match data.dtype {
        DType::I32 => {
            let (chunks, _) = data.bytes.as_chunks::<4>();
            chunks
                .iter()
                .map(|&b| i64::from(i32::from_le_bytes(b)))
                .collect()
        }
        _ => {
            let (chunks, _) = data.bytes.as_chunks::<8>();
            chunks.iter().map(|&b| i64::from_le_bytes(b)).collect()
        }
    }
}

#[test]
fn axes_are_checked() {
    let x = floats(&[2, 3], &[1.0; 6]);
    for dims in [&[][..], &[2], &[0, 0], &[1, 0, 1]] {
        for result in [
            x.clone().sum(Some(dims), false),
            x.clone().mean(Some(dims), false),
            x.clone().max(Some(dims), true),
            x.clone().min(Some(dims), false),
        ] {
            assert!(
                matches!(result, Err(TensorError::InvalidArgument { .. })),
                "{dims:?}: {result:?}"
            );
        }
    }
    for result in [
        x.clone().argmax(2, false, DType::I64),
        x.clone().argmin(5, false, DType::I64),
        x.clone().cumsum(2),
    ] {
        assert!(matches!(result, Err(TensorError::InvalidArgument { .. })));
    }
}

#[test]
fn dtype_classes_and_index_dtypes() {
    let b = floats(&[2, 2], &[1.0, 0.0, 1.0, 0.0])
        .cast(DType::Bool)
        .unwrap();
    let i = ints(DType::I64, &[2, 2], &[1, 2, 3, 4]);
    let unsupported = |op, dtype| TensorError::UnsupportedOperation { op, dtype };
    assert_eq!(
        b.clone().sum(None, false).unwrap_err(),
        unsupported("sum", DType::Bool)
    );
    assert_eq!(
        i.clone().mean(None, false).unwrap_err(),
        unsupported("mean", DType::I64)
    );
    assert_eq!(
        b.clone().max(None, false).unwrap_err(),
        unsupported("max", DType::Bool)
    );
    assert_eq!(
        b.clone().argmax(0, false, DType::I64).unwrap_err(),
        unsupported("argmax", DType::Bool)
    );
    assert_eq!(b.cumsum(0).unwrap_err(), unsupported("cumsum", DType::Bool));
    for dtype in [DType::F32, DType::Bool, DType::F16] {
        assert!(matches!(
            i.clone().argmin(0, false, dtype),
            Err(TensorError::InvalidArgument { op: "argmin", .. })
        ));
    }
    let out = i.argmax(1, false, DType::I32).unwrap();
    assert_eq!(out.dtype(), DType::I32);
    assert_eq!(read_i64(out), vec![1, 1]);
}

#[test]
fn output_shapes() {
    let x = floats(&[2, 3, 4], &[1.0; 24]);
    let shape = |r: Result<DynTensor>| r.unwrap().shape();
    assert_eq!(shape(x.clone().sum(None, false)), vec![1]);
    assert_eq!(shape(x.clone().sum(None, true)), vec![1, 1, 1]);
    assert_eq!(shape(x.clone().mean(Some(&[1]), false)), vec![2, 4]);
    assert_eq!(shape(x.clone().max(Some(&[2, 0]), true)), vec![1, 3, 1]);
    assert_eq!(shape(x.clone().min(Some(&[0, 1, 2]), false)), vec![1]);
    assert_eq!(shape(x.clone().argmax(1, true, DType::I64)), vec![2, 1, 4]);
    assert_eq!(shape(x.clone().argmin(2, false, DType::I64)), vec![2, 3]);
    assert_eq!(shape(x.cumsum(1)), vec![2, 3, 4]);
    let v = floats(&[5], &[1.0; 5]);
    assert_eq!(shape(v.clone().argmax(0, false, DType::I64)), vec![1]);
    assert_eq!(shape(v.sum(Some(&[0]), false)), vec![1]);
}

#[test]
fn nan_and_ties() {
    let nan = f32::NAN;
    let x = floats(&[2, 4], &[1.0, nan, 3.0, nan, 2.0, 5.0, 5.0, -1.0]);
    let max = read_f32(x.clone().max(Some(&[1]), false).unwrap());
    assert!(max[0].is_nan());
    assert_eq!(max[1], 5.0);
    let min = read_f32(x.clone().min(None, false).unwrap());
    assert!(min[0].is_nan());
    assert_eq!(
        read_i64(x.clone().argmax(1, false, DType::I64).unwrap()),
        vec![1, 1]
    );
    assert_eq!(
        read_i64(x.argmin(1, false, DType::I64).unwrap()),
        vec![1, 3]
    );
    // Only -inf: index 0.
    let inf = floats(&[3], &[f32::NEG_INFINITY; 3]);
    assert_eq!(read_i64(inf.argmax(0, false, DType::I64).unwrap()), vec![0]);
    // Ties along dim 0.
    let t = ints(DType::I32, &[3, 2], &[4, 1, 7, 1, 7, 0]);
    assert_eq!(
        read_i64(t.clone().argmax(0, false, DType::I64).unwrap()),
        vec![1, 0]
    );
    assert_eq!(
        read_i64(t.argmin(0, false, DType::I64).unwrap()),
        vec![0, 2]
    );
}

#[test]
fn integer_sum_and_cumsum_wrap_without_panicking() {
    let max64 = i64::MAX;
    let x = ints(DType::I64, &[2, 2], &[max64, 1, max64, max64]);
    assert_eq!(
        read_i64(x.clone().sum(Some(&[1]), false).unwrap()),
        vec![i64::MIN, -2]
    );
    assert_eq!(
        read_i64(x.clone().sum(None, false).unwrap()),
        vec![max64 - 1]
    );
    assert_eq!(read_i64(x.cumsum(0).unwrap()), vec![max64, 1, -2, i64::MIN]);
    let max32 = i64::from(i32::MAX);
    let y = ints(DType::I32, &[3], &[max32, 1, 1]);
    let sum = y.clone().sum(None, false).unwrap();
    assert_eq!(sum.dtype(), DType::I32);
    assert_eq!(read_i64(sum), vec![i64::from(i32::MIN) + 1]);
    let cumsum = y.cumsum(0).unwrap();
    assert_eq!(cumsum.dtype(), DType::I32);
    assert_eq!(
        read_i64(cumsum),
        vec![max32, i64::from(i32::MIN), i64::from(i32::MIN) + 1]
    );
}

#[test]
fn host_sum_reduces_any_axes() {
    let x = ints(DType::I64, &[2, 3, 2], &(0..12).collect::<Vec<_>>());
    assert_eq!(
        read_i64(x.clone().sum(Some(&[0, 2]), false).unwrap()),
        vec![14, 22, 30]
    );
    assert_eq!(
        read_i64(x.clone().sum(Some(&[1]), true).unwrap()),
        vec![6, 9, 24, 27]
    );
    assert_eq!(read_i64(x.cumsum(1).unwrap())[..6], [0, 1, 2, 4, 6, 9]);
}

#[test]
fn half_precision_accumulates_in_f32() {
    // 4096 ones: an f16 accumulator stops at 2048.
    let ones = floats(&[1, 4096], &[1.0; 4096]).cast(DType::F16).unwrap();
    let cumsum = read_f32(ones.clone().cumsum(1).unwrap());
    assert_eq!(cumsum[4095], 4096.0);
    assert_eq!(read_f32(ones.sum(None, false).unwrap()), vec![4096.0]);
    // The sum overflows f16, the mean doesn't.
    let big = floats(&[4096], &[1000.0; 4096]).cast(DType::F16).unwrap();
    assert_eq!(
        read_f32(big.clone().mean(None, false).unwrap()),
        vec![1000.0]
    );
    assert_eq!(read_f32(big.sum(None, false).unwrap()), vec![f32::INFINITY]);
}

#[test]
fn reduced_shape_and_axes_helpers() {
    assert_eq!(reduced_shape(&[2, 3], &[0, 1], false), vec![1]);
    assert_eq!(reduced_shape(&[2, 3], &[1], true), vec![2, 1]);
    assert_eq!(reduce_axes("sum", Some(&[2, 0]), 3), Ok(vec![0, 2]));
    assert_eq!(reduce_axes("sum", None, 2), Ok(vec![0, 1]));
    assert_eq!(
        cumsum_rows(&[1, 2, 3, 4], 2, 2, i32::wrapping_add),
        vec![1, 2, 4, 6]
    );
}
