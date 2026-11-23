//! Reference tests of the `nn` ops (graph-ir.md §3.3: layers, normalization, activations) against
//! the PyTorch fixtures in `fixtures/small/ops/` (testing-strategy.md §6), and the out-of-range
//! rule of graph-ir.md §3.1.8 for `nn.embedding` on every device: PyTorch raises for those, so
//! they can't be fixtures and are tested here directly. The harness is `tests/reference/common.rs`.
//!
//! The ops run through the runner shared with the gradient test (`tests/reference/run_nn.rs`),
//! which fills in the attributes' defaults and builds the permuted views.
//!
//! ```text
//! cargo test -p polypous-backend --test reference_nn                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_nn -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;
#[path = "reference/run_nn.rs"]
mod runner;

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData, TensorError};
use runner::run;

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
    nn_linear => "nn.linear",
    nn_embedding => "nn.embedding",
    nn_conv1d => "nn.conv1d",
    nn_conv2d => "nn.conv2d",
    nn_max_pool2d => "nn.max_pool2d",
    nn_avg_pool2d => "nn.avg_pool2d",
    nn_adaptive_avg_pool2d => "nn.adaptive_avg_pool2d",
    nn_dropout => "nn.dropout",
    nn_batch_norm => "nn.batch_norm",
    nn_layer_norm => "nn.layer_norm",
    nn_group_norm => "nn.group_norm",
    nn_rms_norm => "nn.rms_norm",
    nn_relu => "nn.relu",
    nn_leaky_relu => "nn.leaky_relu",
    nn_gelu => "nn.gelu",
    nn_gelu_tanh => "nn.gelu_tanh",
    nn_silu => "nn.silu",
    nn_sigmoid => "nn.sigmoid",
    nn_softplus => "nn.softplus",
    nn_mish => "nn.mish",
}

fn upload(dtype: DType, shape: &[u64], values: &[i64], device: DeviceId) -> DynTensor {
    let bytes = match dtype {
        DType::I64 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::I32 => values
            .iter()
            .flat_map(|&v| i32::try_from(v).unwrap().to_le_bytes())
            .collect(),
        DType::F32 => values
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
        other => panic!("no test upload for {other}"),
    };
    DynTensor::from_data(
        &TensorData::new(dtype, shape.to_vec(), bytes).unwrap(),
        device,
    )
    .unwrap()
}

/// `nn.embedding` with ids outside `0..num` (negative ones included) is
/// [`TensorError::IndexOutOfRange`] on `device`, never a panic or a read out of bounds, for `i32`
/// and `i64` ids of rank 1 and 2. Returns the number of checked cases.
fn out_of_range_ids_are_errors(device: DeviceId) -> usize {
    let num = 4i64;
    let weight = upload(DType::F32, &[4, 3], &(0..12).collect::<Vec<_>>(), device);
    let mut checked = 0;
    for dtype in [DType::I32, DType::I64] {
        if !DynTensor::device_supports(device, dtype).unwrap() {
            eprintln!("SKIP out-of-range {dtype} ids on {device}: dtype not supported");
            continue;
        }
        let mut bad_ids = vec![-1, num, num + 5, i64::from(i32::MIN), i64::from(i32::MAX)];
        if dtype == DType::I64 {
            bad_ids.extend([i64::MIN, i64::MAX]);
        }
        for bad in bad_ids {
            for (shape, values) in [
                (vec![3u64], vec![0, bad, 1]),
                (vec![2u64, 2], vec![0, 1, 3, bad]),
            ] {
                let ids = upload(dtype, &shape, &values, device);
                let err = ids.embedding(weight.clone()).unwrap_err();
                assert_eq!(
                    err,
                    TensorError::IndexOutOfRange {
                        op: "embedding",
                        index: bad,
                        size: num as usize
                    },
                    "embedding, {dtype} ids {values:?} on {device}"
                );
                checked += 1;
            }
        }
        // The edges 0 and num - 1 work and pick the right rows.
        let ids = upload(dtype, &[2], &[num - 1, 0], device);
        let out = ids.embedding(weight.clone()).unwrap().into_data().unwrap();
        let (chunks, _) = out.bytes.as_chunks::<4>();
        let rows: Vec<f32> = chunks.iter().map(|&b| f32::from_le_bytes(b)).collect();
        assert_eq!(rows, [9.0, 10.0, 11.0, 0.0, 1.0, 2.0], "edges on {device}");
    }
    checked
}

#[cfg(feature = "cpu")]
#[test]
fn out_of_range_ids_cpu() {
    let checked = out_of_range_ids_are_errors(DeviceId::Cpu);
    eprintln!("SUMMARY nn.embedding out-of-range ids cpu: {checked} passed");
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn out_of_range_ids_gpu() {
    for device in common::gpu_devices() {
        let checked = out_of_range_ids_are_errors(device);
        eprintln!("SUMMARY nn.embedding out-of-range ids {device}: {checked} passed");
    }
}
