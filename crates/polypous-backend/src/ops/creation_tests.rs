//! Unit tests of `creation.rs`: fill-value rules, argument checks (never a panic) and the
//! rounding of float fill values. Results are checked against PyTorch fixtures in
//! `tests/reference_creation.rs`.

use super::*;

#[test]
fn fill_values_follow_the_scalar_rule() {
    let op = "full";
    assert_eq!(
        fill_value(op, Scalar::Bool(true), DType::Bool),
        Ok(Fill::Bool(true))
    );
    assert_eq!(
        fill_value(op, Scalar::Int(i64::MIN), DType::I64),
        Ok(Fill::Int(i64::MIN))
    );
    assert_eq!(
        fill_value(op, Scalar::Int(-(1 << 31)), DType::I32),
        Ok(Fill::Int(-(1 << 31)))
    );
    assert!(fill_value(op, Scalar::Int(1 << 31), DType::I32).is_err());
    assert_eq!(
        fill_value(op, Scalar::Int(3), DType::F16),
        Ok(Fill::Float(3.0))
    );
    for bad in [
        (Scalar::Int(0), DType::Bool),
        (Scalar::Float(0.0), DType::Bool),
        (Scalar::Float(1.0), DType::I64),
        (Scalar::Bool(true), DType::I32),
        (Scalar::Bool(false), DType::F32),
    ] {
        assert!(
            matches!(
                fill_value(op, bad.0, bad.1),
                Err(TensorError::InvalidArgument { op: "full", .. })
            ),
            "{bad:?}"
        );
    }
}

#[test]
fn float_values_round_through_f32() {
    let fill = |v: f64, dtype| match fill_value("full", Scalar::Float(v), dtype) {
        Ok(Fill::Float(x)) => x,
        other => panic!("{other:?}"),
    };
    assert_eq!(fill(0.1, DType::F32), f64::from(0.1f32));
    // 1 + 2^-11 + 2^-40 is above the f16 tie between 1 and 1 + 2^-10, but rounds to the tie in
    // f32 first, then to even: 1.0, as in PyTorch.
    assert_eq!(fill(1.0 + 2f64.powi(-11) + 2f64.powi(-40), DType::F16), 1.0);
    assert_eq!(fill(1.0 + 2f64.powi(-8) + 2f64.powi(-40), DType::BF16), 1.0);
    assert_eq!(fill(65504.0, DType::F16), 65504.0);
    // Beyond the range: ±inf (IEEE 754).
    assert_eq!(fill(1e5, DType::F16), f64::INFINITY);
    assert_eq!(fill(-1e39, DType::F32), f64::NEG_INFINITY);
    assert_eq!(fill(-1e39, DType::BF16), f64::NEG_INFINITY);
    assert!(fill(f64::NAN, DType::BF16).is_nan());
    assert_eq!(fill(-0.0, DType::F16).to_bits(), (-0.0f64).to_bits());
}

#[cfg(feature = "cpu")]
mod cpu {
    use super::*;

    const CPU: DeviceId = DeviceId::Cpu;

    fn invalid(result: Result<DynTensor>, op: &str) {
        match result {
            Err(TensorError::InvalidArgument { op: got, .. }) => assert_eq!(got, op),
            other => panic!("{op}: expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn shapes_are_checked_before_allocating() {
        assert_eq!(
            DynTensor::zeros(&[], DType::F32, CPU).unwrap_err(),
            TensorError::UnsupportedRank { rank: 0 }
        );
        assert_eq!(
            DynTensor::ones(&[1; 7], DType::F32, CPU).unwrap_err(),
            TensorError::UnsupportedRank { rank: 7 }
        );
        invalid(DynTensor::zeros(&[2, 0], DType::F32, CPU), "zeros");
        invalid(
            DynTensor::full(&[usize::MAX, 2], Scalar::Int(0), DType::I64, CPU),
            "full",
        );
        invalid(
            DynTensor::full(&[1 << 62, 4], Scalar::Int(0), DType::I32, CPU),
            "full",
        );
    }

    #[test]
    fn constant_checks_count_limit_and_values() {
        invalid(
            DynTensor::constant(DType::I64, &[2, 2], &[Scalar::Int(1); 3], CPU),
            "constant",
        );
        let many = vec![Scalar::Int(0); MAX_CONSTANT_ELEMENTS + 1];
        invalid(
            DynTensor::constant(DType::I64, &[MAX_CONSTANT_ELEMENTS + 1], &many, CPU),
            "constant",
        );
        invalid(
            DynTensor::constant(DType::Bool, &[1], &[Scalar::Int(1)], CPU),
            "constant",
        );
        let max = vec![Scalar::Bool(true); MAX_CONSTANT_ELEMENTS];
        let t = DynTensor::constant(DType::Bool, &[256, 256], &max, CPU).unwrap();
        assert_eq!(t.shape(), vec![256, 256]);
    }

    #[test]
    fn arange_checks_its_range() {
        invalid(DynTensor::arange(0, 5, 0, DType::I64, CPU), "arange");
        invalid(DynTensor::arange(5, 5, 1, DType::I64, CPU), "arange");
        invalid(DynTensor::arange(6, 5, 1, DType::I64, CPU), "arange");
        assert_eq!(
            DynTensor::arange(0, 2, 1, DType::Bool, CPU).unwrap_err(),
            TensorError::UnsupportedOperation {
                op: "arange",
                dtype: DType::Bool
            }
        );
        // i32 values must fit; the last value counts, not `end`.
        let max = i32::MAX as usize;
        invalid(
            DynTensor::arange(max, max + 2, 1, DType::I32, CPU),
            "arange",
        );
        let last = DynTensor::arange(max - 4, max + 1, 4, DType::I32, CPU).unwrap();
        assert_eq!(last.shape(), vec![2]);
        let data = last.into_data().unwrap();
        assert_eq!(data.bytes[4..], i32::MAX.to_le_bytes());
        // i64: `end` may be past i64::MAX as long as the values are not.
        let big = (i64::MAX - 10) as usize;
        let t = DynTensor::arange(big, usize::MAX, usize::MAX, DType::I64, CPU).unwrap();
        assert_eq!(t.shape(), vec![1]);
        invalid(
            DynTensor::arange(i64::MAX as usize, usize::MAX, 1, DType::I64, CPU),
            "arange",
        );
    }

    #[test]
    fn full_like_keeps_shape_and_device() {
        let x = DynTensor::zeros(&[2, 3], DType::F16, CPU).unwrap();
        let y = x.full_like(Scalar::Int(-3), Some(DType::I32)).unwrap();
        assert_eq!(
            (y.shape(), y.dtype(), y.device()),
            (vec![2, 3], DType::I32, CPU)
        );
        let z = x.full_like(Scalar::Float(0.5), None).unwrap();
        assert_eq!(z.dtype(), DType::F16);
        invalid(x.full_like(Scalar::Bool(true), None), "full_like");
    }

    #[test]
    fn unavailable_devices_are_errors() {
        let gpu = DeviceId::Cuda { index: 99 };
        assert!(matches!(
            DynTensor::zeros(&[2], DType::F32, gpu),
            Err(TensorError::DeviceUnavailable { .. })
        ));
        assert!(matches!(
            DynTensor::arange(0, 2, 1, DType::I64, gpu),
            Err(TensorError::DeviceUnavailable { .. })
        ));
    }
}
