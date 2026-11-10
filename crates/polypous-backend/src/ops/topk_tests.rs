//! Unit tests of `topk.rs`: argument checks, the order key, ties and NaN. Results against
//! PyTorch are in `tests/reference_reduce.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, ints, read_f32, read_i64};
use super::*;

#[test]
fn arguments_are_checked() {
    let x = floats(&[2, 3], &[1.0; 6]);
    for (k, dim) in [(0, 1), (4, 1), (3, 0), (1, 2)] {
        assert!(
            matches!(
                x.clone().top_k(k, dim, true, DType::I64),
                Err(TensorError::InvalidArgument { op: "top_k", .. })
            ),
            "k {k} dim {dim}"
        );
    }
    assert!(matches!(
        x.clone().top_k(1, 1, true, DType::F32),
        Err(TensorError::InvalidArgument { op: "top_k", .. })
    ));
    let i = ints(DType::I64, &[3], &[1, 2, 3]);
    assert_eq!(
        i.top_k(1, 0, true, DType::I64).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "top_k",
            dtype: DType::I64
        }
    );
}

#[test]
fn ties_keep_index_order_and_nan_is_largest() {
    let nan = f32::NAN;
    let x = floats(&[8], &[1.0, 3.0, nan, 3.0, -0.0, 0.0, 3.0, nan]);
    let (v, i) = x.clone().top_k(8, 0, true, DType::I32).unwrap();
    assert_eq!(i.dtype(), DType::I32);
    assert_eq!(read_i64(i), vec![2, 7, 1, 3, 6, 0, 4, 5]);
    let v = read_f32(v);
    assert!(v[0].is_nan() && v[1].is_nan());
    assert_eq!(&v[2..], &[3.0, 3.0, 3.0, 1.0, -0.0, 0.0]);
    let (v, i) = x.top_k(3, 0, false, DType::I64).unwrap();
    assert_eq!(read_i64(i), vec![4, 5, 0]);
    assert_eq!(read_f32(v), vec![-0.0, 0.0, 1.0]);
}

#[test]
fn along_an_inner_dim() {
    // [2, 3, 2], top 2 along dim 1.
    let x = floats(
        &[2, 3, 2],
        &[1.0, 6.0, 3.0, 5.0, 2.0, 4.0, 9.0, 0.0, 9.0, 1.0, 8.0, 2.0],
    );
    let (v, i) = x.top_k(2, 1, true, DType::I64).unwrap();
    assert_eq!(v.shape(), vec![2, 2, 2]);
    assert_eq!(read_f32(v), vec![3.0, 6.0, 2.0, 5.0, 9.0, 2.0, 9.0, 1.0]);
    assert_eq!(read_i64(i), vec![1, 0, 2, 1, 0, 2, 1, 1]);
}

#[test]
fn order_key_is_numeric_order() {
    let f32_key = |v: f32| order_key(DType::F32, &v.to_le_bytes());
    let values = [
        f32::NEG_INFINITY,
        -3.0e38,
        -1.0,
        -1.0e-38,
        0.0,
        1.0e-38,
        1.0,
        3.0e38,
        f32::INFINITY,
    ];
    for pair in values.windows(2) {
        assert!(f32_key(pair[0]) < f32_key(pair[1]), "{pair:?}");
    }
    assert_eq!(f32_key(-0.0), f32_key(0.0));
    assert_eq!(f32_key(f32::NAN), u32::MAX);
    assert_eq!(f32_key(-f32::NAN), u32::MAX);
    // f16: 0x7c00 = inf, 0x7e00 = NaN, 0xbc00 = -1; bf16: 0x7f80 = inf, 0x7fc0 = NaN.
    let h = |bits: u16| order_key(DType::F16, &bits.to_le_bytes());
    assert!(h(0xbc00) < h(0x0000) && h(0x3c00) < h(0x7c00) && h(0x7c00) < h(0x7e00));
    let b = |bits: u16| order_key(DType::BF16, &bits.to_le_bytes());
    assert!(b(0xbf80) < b(0x8000) && b(0x8000) == b(0x0000) && b(0x7f80) < b(0x7fc0));
    // 0x7c01 is a NaN in f16 but a finite number in bf16.
    assert_eq!(h(0x7c01), u32::MAX);
    assert!(b(0x7c01) < b(0x7f80));
}
