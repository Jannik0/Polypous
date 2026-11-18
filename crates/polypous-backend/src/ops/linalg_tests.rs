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
/// Every element is `j · 2^-s` with a pseudo-random integer `|j| ≤ J`: `J = 63`, `s = 6` for
/// `f16`, `J = 31`, `s = 5` for `bf16`, so the inputs are exact in `dtype`. A product needs up to
/// 12 (10) significant bits, more than `f16` (`bf16`) holds, but is exact in `f32`. Every partial
/// sum, in any order, is a multiple of `2^-2s` below `2^12` in magnitude (`|product| < 1`), so it
/// needs at most 24 (22) bits and is exact in `f32` too: an `f32` accumulator gives the exact sum
/// in any order. Rounding products or partial sums to `dtype` loses bits, and the random signs
/// make the outputs cancel.
fn half_case(dtype: DType, device: DeviceId) -> Result<(DynTensor, DynTensor, Vec<f32>)> {
    const K: usize = 4096;
    let (max, scale) = if dtype == DType::F16 {
        (63, 64.0)
    } else {
        (31, 32.0)
    };
    let mut state = 0x2545_f491_u32;
    let mut value = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let j = (state >> 8) % (2 * max + 1);
        (j as f32 - max as f32) / scale
    };
    let a: Vec<f32> = (0..2 * K).map(|_| value()).collect();
    let b: Vec<f32> = (0..K * 3).map(|_| value()).collect();
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
