//! Unit tests of `softmax.rs`: argument checks, stability and the special rows. Results against
//! PyTorch are in `tests/reference_reduce.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, ints, read_f32};
use crate::tensor::{DType, TensorError};

#[test]
fn arguments_are_checked() {
    let x = floats(&[2, 3], &[1.0; 6]);
    assert!(matches!(
        x.clone().softmax(2),
        Err(TensorError::InvalidArgument { op: "softmax", .. })
    ));
    assert!(matches!(
        x.log_softmax(9),
        Err(TensorError::InvalidArgument {
            op: "log_softmax",
            ..
        })
    ));
    let i = ints(DType::I64, &[2], &[1, 2]);
    assert_eq!(
        i.softmax(0).unwrap_err(),
        TensorError::UnsupportedOperation {
            op: "softmax",
            dtype: DType::I64
        }
    );
}

#[test]
fn large_values_and_special_rows() {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let x = floats(
        &[5, 2],
        &[
            1000.0,
            1001.0,
            f32::NEG_INFINITY,
            0.0,
            nan,
            1.0,
            inf,
            1.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
        ],
    );
    let s = read_f32(x.clone().softmax(1).unwrap());
    assert!((s[0] - 0.268_941_43).abs() < 1e-6 && (s[1] - 0.731_058_6).abs() < 1e-6);
    assert_eq!(&s[2..4], &[0.0, 1.0]);
    assert!(s[4..].iter().all(|v| v.is_nan()), "{s:?}");
    let l = read_f32(x.log_softmax(1).unwrap());
    assert!((l[0] + 1.313_261_7).abs() < 1e-5 && (l[1] + 0.313_261_7).abs() < 1e-5);
    assert_eq!(&l[2..4], &[f32::NEG_INFINITY, 0.0]);
    assert!(l[4..].iter().all(|v| v.is_nan()), "{l:?}");
}

#[test]
fn half_precision_dtype_is_kept() {
    let x = floats(&[2, 2], &[1.0, 2.0, 3.0, 4.0])
        .cast(DType::BF16)
        .unwrap();
    let s = x.softmax(0).unwrap();
    assert_eq!(s.dtype(), DType::BF16);
    let s = read_f32(s);
    assert!((s[0] - 0.119_202_92).abs() < 1e-2, "{s:?}");
}
