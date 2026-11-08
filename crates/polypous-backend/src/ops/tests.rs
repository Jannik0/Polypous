//! Unit tests of the ops module: broadcasting, dtype classes and every error path (bad arguments
//! are errors, never panics). Numerical results are checked against PyTorch fixtures in
//! `tests/reference_*.rs`; the CPU tests here use Burn's `flex` backend (the default `cpu`
//! feature).

use super::*;
use crate::tensor::TensorData;

#[test]
fn broadcast_shapes_follows_numpy_rules() {
    assert_eq!(broadcast_shapes(&[&[2, 3], &[2, 3]]), Some(vec![2, 3]));
    assert_eq!(broadcast_shapes(&[&[2, 3, 4], &[4]]), Some(vec![2, 3, 4]));
    assert_eq!(
        broadcast_shapes(&[&[2, 1, 4], &[1, 3, 1]]),
        Some(vec![2, 3, 4])
    );
    assert_eq!(broadcast_shapes(&[&[1], &[5, 2]]), Some(vec![5, 2]));
    assert_eq!(
        broadcast_shapes(&[&[3, 1], &[1, 4], &[2, 1, 1]]),
        Some(vec![2, 3, 4])
    );
    assert_eq!(broadcast_shapes(&[&[2, 3], &[3, 2]]), None);
    assert_eq!(broadcast_shapes(&[&[2, 3], &[6]]), None);
    assert_eq!(broadcast_shapes(&[]), None);
}

#[test]
fn dtype_classes() {
    assert!(DTypeClass::Numeric.check("x", DType::I32).is_ok());
    assert!(DTypeClass::Numeric.check("x", DType::Bool).is_err());
    assert!(DTypeClass::Float.check("x", DType::I64).is_err());
    assert!(DTypeClass::Float.check("x", DType::BF16).is_ok());
    assert!(DTypeClass::Bool.check("x", DType::F32).is_err());
    assert!(DTypeClass::Any.check("x", DType::Bool).is_ok());
}

#[cfg(feature = "cpu")]
mod cpu {
    use super::*;

    fn tensor(dtype: DType, shape: &[u64]) -> DynTensor {
        let count: u64 = shape.iter().product();
        let bytes = vec![0u8; count as usize * dtype.size_bytes()];
        let data = TensorData::new(dtype, shape.to_vec(), bytes).unwrap();
        DynTensor::from_data(&data, DeviceId::Cpu).unwrap()
    }

    fn i64s(values: &[i64]) -> DynTensor {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = TensorData::new(DType::I64, vec![values.len() as u64], bytes).unwrap();
        DynTensor::from_data(&data, DeviceId::Cpu).unwrap()
    }

    fn read_i64s(t: DynTensor) -> Vec<i64> {
        let data = t.into_data().unwrap();
        let (chunks, _) = data.bytes.as_chunks::<8>();
        chunks.iter().map(|&b| i64::from_le_bytes(b)).collect()
    }

    type Binary = fn(DynTensor, DynTensor) -> Result<DynTensor>;

    const BINARY: [(&str, Binary, DTypeClass); 15] = [
        ("add", DynTensor::add, DTypeClass::Numeric),
        ("sub", DynTensor::sub, DTypeClass::Numeric),
        ("mul", DynTensor::mul, DTypeClass::Numeric),
        ("div", DynTensor::div, DTypeClass::Float),
        ("pow", DynTensor::pow, DTypeClass::Float),
        ("maximum", DynTensor::maximum, DTypeClass::Numeric),
        ("minimum", DynTensor::minimum, DTypeClass::Numeric),
        ("eq", DynTensor::equal, DTypeClass::Any),
        ("ne", DynTensor::not_equal, DTypeClass::Any),
        ("lt", DynTensor::less, DTypeClass::Numeric),
        ("le", DynTensor::less_equal, DTypeClass::Numeric),
        ("gt", DynTensor::greater, DTypeClass::Numeric),
        ("ge", DynTensor::greater_equal, DTypeClass::Numeric),
        ("and", DynTensor::logical_and, DTypeClass::Bool),
        ("or", DynTensor::logical_or, DTypeClass::Bool),
    ];

    type Unary = fn(DynTensor) -> Result<DynTensor>;

    const UNARY: [(&str, Unary, DTypeClass); 10] = [
        ("neg", DynTensor::neg, DTypeClass::Numeric),
        ("abs", DynTensor::abs, DTypeClass::Numeric),
        ("exp", DynTensor::exp, DTypeClass::Float),
        ("log", DynTensor::log, DTypeClass::Float),
        ("sqrt", DynTensor::sqrt, DTypeClass::Float),
        ("rsqrt", DynTensor::rsqrt, DTypeClass::Float),
        ("sin", DynTensor::sin, DTypeClass::Float),
        ("cos", DynTensor::cos, DTypeClass::Float),
        ("tanh", DynTensor::tanh, DTypeClass::Float),
        ("not", DynTensor::logical_not, DTypeClass::Bool),
    ];

    #[test]
    fn binary_ops_check_dtype_class_and_shape() {
        for (op, f, class) in BINARY {
            for dtype in DType::ALL {
                let result = f(tensor(dtype, &[2, 3]), tensor(dtype, &[3]));
                if class.contains(dtype) {
                    let out = result.unwrap_or_else(|err| panic!("{op} {dtype}: {err}"));
                    assert_eq!(out.shape(), vec![2, 3], "{op} {dtype}");
                    let expected = if matches!(class, DTypeClass::Any)
                        || op.starts_with('l')
                        || op.starts_with('g')
                    {
                        DType::Bool
                    } else {
                        dtype
                    };
                    assert_eq!(out.dtype(), expected, "{op} {dtype}");
                } else {
                    assert_eq!(
                        result.unwrap_err(),
                        TensorError::UnsupportedOperation { op, dtype },
                        "{op} {dtype}"
                    );
                }
            }
            let (dtype, other) = match class {
                DTypeClass::Bool => (DType::Bool, DType::I32),
                _ => (DType::F32, DType::F16),
            };
            assert_eq!(
                f(tensor(dtype, &[2]), tensor(other, &[2])).unwrap_err(),
                TensorError::DTypeMismatch {
                    op,
                    lhs: dtype,
                    rhs: other
                },
                "{op}"
            );
            assert_eq!(
                f(tensor(dtype, &[2, 3]), tensor(dtype, &[2])).unwrap_err(),
                TensorError::ShapeMismatch {
                    op,
                    lhs: vec![2, 3],
                    rhs: vec![2]
                },
                "{op}"
            );
        }
    }

    #[test]
    fn unary_ops_check_dtype_class() {
        for (op, f, class) in UNARY {
            for dtype in DType::ALL {
                let result = f(tensor(dtype, &[2, 1, 3]));
                if class.contains(dtype) {
                    let out = result.unwrap_or_else(|err| panic!("{op} {dtype}: {err}"));
                    assert_eq!((out.dtype(), out.shape()), (dtype, vec![2, 1, 3]));
                } else {
                    assert_eq!(
                        result.unwrap_err(),
                        TensorError::UnsupportedOperation { op, dtype }
                    );
                }
            }
        }
    }

    #[test]
    fn broadcasting_reaches_rank_six() {
        let out = tensor(DType::F32, &[2, 1, 3, 1, 2, 1])
            .add(tensor(DType::F32, &[4]))
            .unwrap();
        assert_eq!(out.shape(), vec![2, 1, 3, 1, 2, 4]);
        let out = tensor(DType::F32, &[1])
            .mul(tensor(DType::F32, &[3, 2]))
            .unwrap();
        assert_eq!(out.shape(), vec![3, 2]);
    }

    #[test]
    fn i64_arithmetic_wraps_on_the_cpu() {
        let (max, min) = (i64::MAX, i64::MIN);
        let a = || i64s(&[max, min, max, -1]);
        let b = || i64s(&[1, -1, max, min]);
        assert_eq!(read_i64s(a().add(b()).unwrap()), [min, max, -2, max]);
        assert_eq!(read_i64s(a().sub(b()).unwrap()), [max - 1, min + 1, 0, max]);
        assert_eq!(read_i64s(a().mul(b()).unwrap()), [max, min, 1, min]);
        assert_eq!(read_i64s(a().neg().unwrap()), [min + 1, min, min + 1, 1]);
        assert_eq!(read_i64s(a().abs().unwrap()), [max, min, max, 1]);
        // Broadcast operands go through the host path too.
        let out = i64s(&[max]).add(i64s(&[1, 2])).unwrap();
        assert_eq!(read_i64s(out), [min, min + 1]);
    }

    #[test]
    fn clamp_rejects_bad_bounds() {
        let invalid = |result: Result<DynTensor>| {
            assert!(
                matches!(
                    result,
                    Err(TensorError::InvalidArgument { op: "clamp", .. })
                ),
                "{result:?}"
            );
        };
        let f = || tensor(DType::F32, &[3]);
        let i = || tensor(DType::I32, &[3]);
        invalid(f().clamp(None, None));
        invalid(f().clamp(Some(Scalar::Float(1.0)), Some(Scalar::Float(0.0))));
        invalid(f().clamp(Some(Scalar::Bool(true)), None));
        invalid(i().clamp(Some(Scalar::Float(0.5)), None));
        invalid(i().clamp(None, Some(Scalar::Int(i64::from(i32::MAX) + 1))));
        invalid(i().clamp(Some(Scalar::Int(5)), Some(Scalar::Int(4))));
        assert!(
            f().clamp(Some(Scalar::Int(-1)), Some(Scalar::Float(1.0)))
                .is_ok()
        );
        assert!(i().clamp(Some(Scalar::Int(-1)), None).is_ok());
        assert!(
            f().clamp(Some(Scalar::Float(f64::NAN)), Some(Scalar::Float(0.0)))
                .is_ok()
        );
        assert_eq!(
            tensor(DType::Bool, &[3])
                .clamp(Some(Scalar::Int(0)), None)
                .unwrap_err(),
            TensorError::UnsupportedOperation {
                op: "clamp",
                dtype: DType::Bool
            }
        );
    }

    #[test]
    fn where_checks_its_operands() {
        let cond = || tensor(DType::Bool, &[2, 1]);
        let out =
            DynTensor::where_(cond(), tensor(DType::I32, &[3]), tensor(DType::I32, &[1])).unwrap();
        assert_eq!((out.dtype(), out.shape()), (DType::I32, vec![2, 3]));
        assert_eq!(
            DynTensor::where_(
                tensor(DType::F32, &[2]),
                tensor(DType::F32, &[2]),
                tensor(DType::F32, &[2])
            )
            .unwrap_err(),
            TensorError::UnsupportedOperation {
                op: "where",
                dtype: DType::F32
            }
        );
        assert_eq!(
            DynTensor::where_(cond(), tensor(DType::F32, &[2]), tensor(DType::F16, &[2]))
                .unwrap_err(),
            TensorError::DTypeMismatch {
                op: "where",
                lhs: DType::F32,
                rhs: DType::F16
            }
        );
        // `cond` is the odd one out.
        assert_eq!(
            DynTensor::where_(
                tensor(DType::Bool, &[3]),
                tensor(DType::F32, &[2]),
                tensor(DType::F32, &[2])
            )
            .unwrap_err(),
            TensorError::ShapeMismatch {
                op: "where",
                lhs: vec![3],
                rhs: vec![2]
            }
        );
        assert!(matches!(
            DynTensor::where_(cond(), tensor(DType::F32, &[3]), tensor(DType::F32, &[4])),
            Err(TensorError::ShapeMismatch { op: "where", .. })
        ));
    }

    #[test]
    fn device_support_is_reported() {
        for dtype in DType::ALL {
            assert_eq!(DynTensor::device_supports(DeviceId::Cpu, dtype), Ok(true));
        }
        let missing = DeviceId::Cuda { index: 4096 };
        assert!(matches!(
            DynTensor::device_supports(missing, DType::F32),
            Err(TensorError::DeviceUnavailable { .. })
        ));
    }
}
