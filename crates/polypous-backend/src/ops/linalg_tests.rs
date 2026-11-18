//! Unit tests of `linalg.rs`: `matmul` shape rules and batch broadcasting across ranks. Results
//! against PyTorch are in `tests/reference_reduce.rs`; dtype and device errors in
//! `tensor/dyn_tensor_tests.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, read_f32};
use super::*;
use crate::device::DeviceId;

#[test]
fn shape_rules() {
    let a = floats(&[2, 3], &[1.0; 6]);
    let shape = |r: Result<DynTensor>| r.unwrap().shape();
    // Different batch ranks broadcast; the output has the larger rank.
    assert_eq!(
        shape(a.clone().matmul(floats(&[4, 3, 5], &[1.0; 60]))),
        vec![4, 2, 5]
    );
    assert_eq!(
        shape(floats(&[2, 1, 2, 3], &[1.0; 12]).matmul(floats(&[5, 3, 1], &[1.0; 15]))),
        vec![2, 5, 2, 1]
    );
    // k mismatch, rank 1, incompatible batch dims.
    for (lhs, rhs) in [
        (vec![2, 3], vec![2, 3]),
        (vec![3], vec![3, 2]),
        (vec![2, 3], vec![3]),
        (vec![2, 2, 3], vec![3, 3, 2]),
    ] {
        let l = floats(
            &lhs.iter().map(|&d| d as u64).collect::<Vec<_>>(),
            &vec![1.0; lhs.iter().product()],
        );
        let r = floats(
            &rhs.iter().map(|&d| d as u64).collect::<Vec<_>>(),
            &vec![1.0; rhs.iter().product()],
        );
        assert_eq!(
            l.matmul(r).unwrap_err(),
            TensorError::ShapeMismatch {
                op: "matmul",
                lhs,
                rhs
            }
        );
    }
}

#[test]
fn broadcast_batch_values() {
    // [2, 2, 2] @ [2, 1]: the rank-2 rhs applies to both batches.
    let a = floats(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = floats(&[2, 1], &[3.0, 4.0]);
    let c = a.matmul(b).unwrap();
    assert_eq!(c.shape(), vec![2, 2, 1]);
    assert_eq!(read_f32(c), vec![3.0, 4.0, 6.0, 8.0]);
}

/// `[2, k] @ [k, 3]` with `k = 4096` in `dtype` (`f16` or `bf16`) on `device`, and the expected
/// output: the exact sum (in `f64`) rounded once to `dtype`.
///
/// `a` is `±1`, `b` is `±(1 + j · 2^-p)` with `p` the mantissa bits of `dtype` (10 or 7), signs
/// and `j` pseudo-random. Every product is exact, and every partial sum, in any order, is a
/// multiple of `2^-p` below `2 · 4096` in magnitude, so exact in `f32` (at most 23 significant
/// bits). An `f32` accumulator therefore gives the exact sum in any order; an `f16`/`bf16`
/// accumulator drops low bits once a partial sum passes `2^(p - 1)` or so.
fn half_case(dtype: DType, device: DeviceId) -> Result<(DynTensor, DynTensor, Vec<f32>)> {
    const K: usize = 4096;
    let bits = if dtype == DType::F16 { 10 } else { 7 };
    let mut state = 0x2545_f491_u32;
    let mut next = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state >> 8
    };
    let sign = |r: u32| if r & 1 == 0 { 1.0 } else { -1.0 };
    let a: Vec<f32> = (0..2 * K).map(|_| sign(next())).collect();
    let b: Vec<f32> = (0..K * 3)
        .map(|_| {
            let r = next();
            let j = (r >> 1) % (1 << bits);
            sign(r) * (1.0 + j as f32 / (1u32 << bits) as f32)
        })
        .collect();
    let mut expected = Vec::with_capacity(6);
    for i in 0..2 {
        for col in 0..3 {
            let exact: f64 = (0..K)
                .map(|t| f64::from(a[i * K + t]) * f64::from(b[t * 3 + col]))
                .sum();
            expected.push(if dtype == DType::F16 {
                burn::tensor::f16::from_f64(exact).to_f32()
            } else {
                burn::tensor::bf16::from_f64(exact).to_f32()
            });
        }
    }
    let upload = |shape: &[u64], values: &[f32]| {
        let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let data = crate::tensor::TensorData::new(DType::F32, shape.to_vec(), bytes)?;
        DynTensor::from_data(&data, device)?.cast(dtype)
    };
    Ok((
        upload(&[2, K as u64], &a)?,
        upload(&[K as u64, 3], &b)?,
        expected,
    ))
}

#[test]
fn half_precision_computes_in_f32_and_rounds_once() {
    for dtype in [DType::F16, DType::BF16] {
        let (a, b, expected) = half_case(dtype, DeviceId::Cpu).unwrap();
        let c = a.matmul(b).unwrap();
        assert_eq!(c.dtype(), dtype);
        assert_eq!(read_f32(c), expected, "{dtype:?}");
    }
}

/// The same check on every GPU device that `list_devices()` returns (software rasterizers only
/// with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, as in `tensor/dyn_tensor_tests.rs`). A dtype the
/// device lacks (`bf16` on lavapipe) is skipped.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn half_precision_computes_in_f32_and_rounds_once_gpu() {
    use crate::device::{BackendKind, EnumerateOptions, list_devices};
    let options = EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    };
    let devices: Vec<DeviceId> = list_devices()
        .into_iter()
        .filter(|id| id.backend() != BackendKind::Cpu && !options.skips(*id))
        .collect();
    if devices.is_empty() {
        eprintln!(
            "no GPU device to test (software rasterizers need POLYPOUS_TEST_SOFTWARE_VULKAN=1)"
        );
        return;
    }
    for gpu in devices {
        for dtype in [DType::F16, DType::BF16] {
            let (a, b, expected) = match half_case(dtype, gpu) {
                Err(TensorError::UnsupportedDType { .. }) => {
                    eprintln!("{gpu}: {dtype:?} unsupported, skipped");
                    continue;
                }
                other => other.unwrap(),
            };
            let c = a.matmul(b).unwrap();
            assert_eq!(read_f32(c), expected, "{gpu} {dtype:?}");
            eprintln!("{gpu}: {dtype:?} matmul rounds once");
        }
    }
}
