//! Reference tests of the creation ops (`core.constant arange zeros ones full full_like`) against
//! the PyTorch fixtures in `fixtures/small/ops/` (graph-ir.md §3.2, testing-strategy.md §6). The
//! harness is `tests/reference/common.rs`; these ops get the device under test as an argument.
//!
//! ```text
//! cargo test -p polypous-backend --test reference_creation                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_creation -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorError};

/// A `dim[]`/`int[]` shape attribute.
fn shape(case: &Case) -> Vec<usize> {
    let dims = case.ints("shape").expect("shape");
    dims.into_iter().map(|d| d as usize).collect()
}

/// A `dtype` attribute with its default.
fn dtype_or(case: &Case, default: DType) -> DType {
    if case.attributes.contains_key("dtype") {
        case.dtype("dtype")
    } else {
        default
    }
}

fn run(
    case: &Case,
    inputs: Vec<DynTensor>,
    device: DeviceId,
) -> Result<Vec<DynTensor>, TensorError> {
    let mut inputs = common::feed_layout(case, inputs)?;
    let out = match case.op.as_str() {
        "core.constant" => DynTensor::constant(
            case.dtype("dtype"),
            &shape(case),
            &case.scalars("values").expect("values"),
            device,
        )?,
        "core.arange" => {
            let start = case.int("start").unwrap_or(0) as usize;
            let end = case.int("end").expect("end") as usize;
            let step = case.int("step").unwrap_or(1) as usize;
            DynTensor::arange(start, end, step, dtype_or(case, DType::I64), device)?
        }
        "core.zeros" => DynTensor::zeros(&shape(case), dtype_or(case, DType::F32), device)?,
        "core.ones" => DynTensor::ones(&shape(case), dtype_or(case, DType::F32), device)?,
        "core.full" => DynTensor::full(
            &shape(case),
            case.scalar("value").expect("value"),
            dtype_or(case, DType::F32),
            device,
        )?,
        "core.full_like" => {
            let x = inputs.remove(0);
            let dtype = case
                .attributes
                .contains_key("dtype")
                .then(|| case.dtype("dtype"));
            x.full_like(case.scalar("value").expect("value"), dtype)?
        }
        other => panic!("no runner for {other}"),
    };
    Ok(vec![out])
}

macro_rules! reference_tests {
    ($($test:ident => $op:literal),* $(,)?) => {
        #[cfg(feature = "cpu")]
        mod cpu {
            $(
                #[test]
                fn $test() {
                    super::common::run_cpu_on($op, super::run);
                }
            )*
        }

        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        mod gpu {
            $(
                #[test]
                #[ignore = "gpu"]
                fn $test() {
                    super::common::run_gpu_on($op, super::run);
                }
            )*
        }
    };
}

reference_tests! {
    core_constant => "core.constant",
    core_arange => "core.arange",
    core_zeros => "core.zeros",
    core_ones => "core.ones",
    core_full => "core.full",
    core_full_like => "core.full_like",
}

/// Creation ops refuse a dtype the device doesn't support (graph-ir.md §2.2) with
/// `UnsupportedDType`, and create every dtype it supports on that device.
fn dtype_support_is_checked(device: DeviceId) {
    for dtype in DType::ALL {
        let supported = DynTensor::device_supports(device, dtype).unwrap();
        let one = match dtype {
            DType::Bool => polypous_backend::Scalar::Bool(true),
            d if d.is_int() => polypous_backend::Scalar::Int(1),
            _ => polypous_backend::Scalar::Float(1.0),
        };
        let results = [
            DynTensor::zeros(&[2], dtype, device),
            DynTensor::ones(&[2], dtype, device),
            DynTensor::full(&[2], one, dtype, device),
            DynTensor::constant(dtype, &[1], &[one], device),
        ];
        for result in results {
            match (supported, result) {
                (true, Ok(t)) => assert_eq!((t.dtype(), t.device()), (dtype, device)),
                (false, Err(TensorError::UnsupportedDType { dtype: d, .. })) => {
                    assert_eq!(d, dtype);
                }
                (supported, result) => {
                    panic!("{dtype} on {device} (supported: {supported}): {result:?}")
                }
            }
        }
        if dtype != DType::Bool {
            match (supported, DynTensor::arange(0, 4, 1, dtype, device)) {
                (true, Ok(t)) => assert_eq!(t.shape(), vec![4]),
                (false, Err(TensorError::UnsupportedDType { .. })) => {}
                (supported, result) => {
                    panic!("arange {dtype} on {device} (supported: {supported}): {result:?}")
                }
            }
        }
    }
}

#[cfg(feature = "cpu")]
#[test]
fn dtype_support_cpu() {
    dtype_support_is_checked(DeviceId::Cpu);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn dtype_support_gpu() {
    for device in common::gpu_devices() {
        dtype_support_is_checked(device);
    }
}
