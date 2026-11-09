//! Reference tests of the indexing ops (`core.gather`, `core.index_select`) against the PyTorch
//! fixtures in `fixtures/small/ops/` (graph-ir.md §3.2, testing-strategy.md §6), and the
//! out-of-range rule of graph-ir.md §3.1.8 on every device: PyTorch raises for those, so they
//! can't be fixtures and are tested here directly. The harness is `tests/reference/common.rs`.
//!
//! ```text
//! cargo test -p polypous-backend --test reference_indexing                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_indexing -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData, TensorError};

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let inputs = common::feed_layout(case, inputs)?;
    let [x, index]: [DynTensor; 2] = inputs.try_into().expect("x and index");
    let rank = x.rank() as i64;
    let dim = case.int("dim").expect("dim");
    let dim = if dim < 0 { dim + rank } else { dim } as usize;
    let out = match case.op.as_str() {
        "core.gather" => x.gather(dim, index)?,
        "core.index_select" => x.index_select(dim, index)?,
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
                    super::common::run_cpu($op, super::run);
                }
            )*
        }

        #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
        mod gpu {
            $(
                #[test]
                #[ignore = "gpu"]
                fn $test() {
                    super::common::run_gpu($op, super::run);
                }
            )*
        }
    };
}

reference_tests! {
    core_gather => "core.gather",
    core_index_select => "core.index_select",
}

// ---- Out-of-range indices (graph-ir.md §3.1.8) ----

fn upload(dtype: DType, shape: &[u64], values: &[i64], device: DeviceId) -> DynTensor {
    let bytes = match dtype {
        DType::I32 => values
            .iter()
            .flat_map(|&v| i32::try_from(v).unwrap().to_le_bytes())
            .collect(),
        DType::I64 => values.iter().flat_map(|&v| v.to_le_bytes()).collect(),
        DType::F32 => values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
        other => panic!("no test data for {other}"),
    };
    let data = TensorData::new(dtype, shape.to_vec(), bytes).unwrap();
    DynTensor::from_data(&data, device).unwrap()
}

/// Every out-of-range index is an `IndexOutOfRange` error naming the op and the index, with
/// `i32` and `i64` indices, for every axis; nothing panics or crashes (index `-1` crashed the
/// lavapipe process inside Burn before the check). Valid neighbours of each bad index still work.
fn out_of_range_indices_are_errors(device: DeviceId) {
    let x = upload(DType::F32, &[2, 3], &[1, 2, 3, 4, 5, 6], device);
    let mut checked = 0;
    for dtype in [DType::I32, DType::I64] {
        if !DynTensor::device_supports(device, dtype).unwrap() {
            eprintln!("SKIP out-of-range {dtype} indices on {device}: dtype not supported");
            continue;
        }
        for (dim, size) in [(0usize, 2i64), (1, 3)] {
            for bad in [-1, size, size + 5, i64::from(i32::MIN), i64::from(i32::MAX)] {
                // gather: index shape [2, 2] (≤ x outside `dim`), one bad entry last.
                let index = upload(dtype, &[2, 2], &[0, 1, 1, bad], device);
                let err = x.clone().gather(dim, index).unwrap_err();
                assert_eq!(
                    err,
                    TensorError::IndexOutOfRange {
                        op: "gather",
                        index: bad,
                        size: size as usize
                    },
                    "gather dim {dim}, {dtype} index {bad} on {device}"
                );
                let index = upload(dtype, &[3], &[bad, 0, 1], device);
                let err = x.clone().index_select(dim, index).unwrap_err();
                assert_eq!(
                    err,
                    TensorError::IndexOutOfRange {
                        op: "index_select",
                        index: bad,
                        size: size as usize
                    },
                    "index_select dim {dim}, {dtype} index {bad} on {device}"
                );
                checked += 2;
            }
            // The largest valid index works.
            let last = upload(dtype, &[2, 2], &[size - 1; 4], device);
            assert!(x.clone().gather(dim, last).is_ok());
            let last = upload(dtype, &[1], &[size - 1], device);
            assert!(x.clone().index_select(dim, last).is_ok());
        }
        // Both ends out of range: the negative one is reported.
        let index = upload(dtype, &[3], &[7, 0, -2], device);
        assert!(matches!(
            x.clone().index_select(1, index),
            Err(TensorError::IndexOutOfRange { index: -2, .. })
        ));
    }
    eprintln!("SUMMARY out-of-range indices {device}: {checked} cases passed");
}

#[cfg(feature = "cpu")]
#[test]
fn out_of_range_indices_cpu() {
    out_of_range_indices_are_errors(DeviceId::Cpu);
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn out_of_range_indices_gpu() {
    for device in common::gpu_devices() {
        out_of_range_indices_are_errors(device);
    }
}
