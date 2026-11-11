//! Unit tests of `activation.rs`: dtype classes and attribute checks (errors, never panics), and
//! the NaN, `-0.0` and large-input rules on the CPU. Results against PyTorch are in
//! `tests/reference_nn.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, ints, read_f32};
use super::*;
use crate::tensor::DType;

type Act = fn(DynTensor) -> Result<DynTensor>;

const ALL: [(&str, Act); 8] = [
    ("relu", DynTensor::relu),
    ("leaky_relu", |x| x.leaky_relu(0.01)),
    ("gelu", DynTensor::gelu),
    ("gelu_tanh", DynTensor::gelu_tanh),
    ("silu", DynTensor::silu),
    ("sigmoid", DynTensor::sigmoid),
    ("softplus", |x| x.softplus(1.0, 20.0)),
    ("mish", DynTensor::mish),
];

#[test]
fn float_only() {
    for (name, f) in ALL {
        for dtype in [DType::I64, DType::I32] {
            assert_eq!(
                f(ints(dtype, &[2], &[1, 2])).unwrap_err(),
                TensorError::UnsupportedOperation { op: name, dtype },
                "{name}"
            );
        }
    }
}

#[test]
fn attributes_are_checked() {
    let x = || floats(&[2], &[1.0, -1.0]);
    for slope in [f64::NAN, f64::INFINITY] {
        assert!(matches!(
            x().leaky_relu(slope),
            Err(TensorError::InvalidArgument {
                op: "leaky_relu",
                ..
            })
        ));
    }
    for (beta, threshold) in [
        (0.0, 20.0),
        (-1.0, 20.0),
        (f64::NAN, 20.0),
        (1.0, 0.0),
        (1.0, f64::INFINITY),
        (1.0, -3.0),
    ] {
        assert!(
            matches!(
                x().softplus(beta, threshold),
                Err(TensorError::InvalidArgument { op: "softplus", .. })
            ),
            "beta {beta}, threshold {threshold}"
        );
    }
    // A negative slope is allowed, as in PyTorch.
    assert_eq!(read_f32(x().leaky_relu(-0.5).unwrap()), [1.0, 0.5]);
}

#[test]
fn nan_propagates_and_signs_are_kept() {
    for (name, f) in ALL {
        let out = read_f32(f(floats(&[2], &[f32::NAN, 1.0])).unwrap());
        assert!(out[0].is_nan(), "{name}(NaN) = {}", out[0]);
    }
    let out = read_f32(
        floats(&[3], &[-0.0, -2.0, f32::NEG_INFINITY])
            .relu()
            .unwrap(),
    );
    assert!(
        out[0] == 0.0 && out[0].is_sign_negative(),
        "relu(-0.0) = -0.0"
    );
    assert_eq!(&out[1..], [0.0, 0.0]);
}

#[test]
fn large_inputs_give_no_nan() {
    let x = || floats(&[4], &[-1e30, -100.0, 100.0, 1e30]);
    assert_eq!(read_f32(x().sigmoid().unwrap()), [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(read_f32(x().silu().unwrap()), [-0.0, -0.0, 100.0, 1e30]);
    assert_eq!(
        read_f32(x().gelu_tanh().unwrap()),
        [-0.0, -0.0, 100.0, 1e30]
    );
    // mish(-100) = -3.8e-42 is subnormal: backend-defined (graph-ir.md §3.1.8), only its size.
    let mish = read_f32(x().mish().unwrap());
    assert_eq!([mish[0], mish[2], mish[3]], [-0.0, 100.0, 1e30]);
    assert!(mish[1] <= 0.0 && mish[1] > -1e-37);
    let softplus = read_f32(x().softplus(1.0, 20.0).unwrap());
    assert_eq!(softplus[2..], [100.0, 1e30]);
    assert!(softplus[..2].iter().all(|v| *v >= 0.0 && *v < 1e-30));
    // Linear above the threshold: beta·x > threshold.
    assert_eq!(
        read_f32(floats(&[2], &[2.6, 30.0]).softplus(2.0, 5.0).unwrap()),
        [2.6, 30.0]
    );
}

#[test]
fn half_precision_keeps_the_dtype() {
    for (name, f) in ALL {
        for dtype in [DType::F16, DType::BF16] {
            let x = floats(&[3], &[-1.0, 0.5, 2.0]).cast(dtype).unwrap();
            assert_eq!(f(x).unwrap().dtype(), dtype, "{name}");
        }
    }
}
