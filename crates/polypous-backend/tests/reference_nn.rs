//! Reference tests of the `nn` ops (graph-ir.md §3.3: layers, normalization, activations) against
//! the PyTorch fixtures in `fixtures/small/ops/` (testing-strategy.md §6), and the out-of-range
//! rule of graph-ir.md §3.1.8 for `nn.embedding` on every device: PyTorch raises for those, so
//! they can't be fixtures and are tested here directly. The harness is `tests/reference/common.rs`.
//!
//! Params come from the sidecar's `params` list (testing-strategy.md §3.2), after the inputs. The
//! sidecars hold the attributes as written in a graph (defaults left out, `normalized_shape`
//! possibly left out); [`run`] fills them in the way `polypous_ir::Analysis` does before the
//! runtime calls the backend. Cases named `permuted_…` or `transposed_…` get their first input as
//! a non-contiguous view (`common::feed_layout`).
//!
//! ```text
//! cargo test -p polypous-backend --test reference_nn                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_nn -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use std::collections::BTreeMap;

use common::Case;
use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor, TensorData, TensorError};

/// A float attribute with its default (JSON integers are accepted too).
fn float(case: &Case, name: &str, default: Option<f64>) -> f64 {
    match case.attributes.get(name) {
        Some(value) => value.as_f64().unwrap_or_else(|| {
            panic!(
                "{}/{}: attribute {name} is not a number: {value}",
                case.op, case.name
            )
        }),
        None => default.unwrap_or_else(|| {
            panic!(
                "{}/{}: required attribute {name} missing",
                case.op, case.name
            )
        }),
    }
}

fn int(case: &Case, name: &str, default: usize) -> usize {
    case.int(name)
        .map_or(default, |v| usize::try_from(v).unwrap())
}

/// An `int[2]` attribute, or `default`.
fn pair(case: &Case, name: &str, default: Option<[usize; 2]>) -> [usize; 2] {
    match case.ints(name) {
        Some(values) => {
            let values: Vec<usize> = values
                .into_iter()
                .map(|v| usize::try_from(v).unwrap())
                .collect();
            values.try_into().unwrap_or_else(|v| {
                panic!("{}/{}: {name} needs 2 entries: {v:?}", case.op, case.name)
            })
        }
        None => default.unwrap_or_else(|| {
            panic!(
                "{}/{}: required attribute {name} missing",
                case.op, case.name
            )
        }),
    }
}

fn flag(case: &Case, name: &str, default: bool) -> bool {
    case.attributes
        .get(name)
        .map_or(default, |v| v.as_bool().expect("a bool attribute"))
}

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let mut inputs = common::feed_layout(case, inputs)?;
    let params_start = case.inputs.len();
    assert_eq!(
        inputs.len(),
        params_start + case.params.len(),
        "{}/{}: inputs and params",
        case.op,
        case.name
    );
    let mut params: BTreeMap<String, DynTensor> = case
        .params
        .iter()
        .map(|(name, _)| name.clone())
        .zip(inputs.split_off(params_start))
        .collect();
    let mut param = |name: &str| params.remove(name);
    assert_eq!(inputs.len(), 1, "{}/{}: one input", case.op, case.name);
    let x = inputs.remove(0);
    // `normalized_shape` defaults to `weight`'s shape (graph-ir.md §3.3).
    let normalized_shape = |weight: Option<&DynTensor>| -> Vec<usize> {
        match case.ints("normalized_shape") {
            Some(dims) => dims.into_iter().map(|d| d as usize).collect(),
            None => weight.expect("normalized_shape or weight").shape(),
        }
    };
    let eps = float(case, "eps", Some(1e-5));
    let out = match case.op.as_str() {
        "nn.linear" => x.linear(param("weight").unwrap(), param("bias"))?,
        "nn.embedding" => x.embedding(param("weight").unwrap())?,
        "nn.conv1d" => x.conv1d(
            param("weight").unwrap(),
            param("bias"),
            int(case, "stride", 1),
            int(case, "padding", 0),
            int(case, "dilation", 1),
            int(case, "groups", 1),
        )?,
        "nn.conv2d" => x.conv2d(
            param("weight").unwrap(),
            param("bias"),
            pair(case, "stride", Some([1, 1])),
            pair(case, "padding", Some([0, 0])),
            pair(case, "dilation", Some([1, 1])),
            int(case, "groups", 1),
        )?,
        "nn.max_pool2d" | "nn.avg_pool2d" => {
            let kernel = pair(case, "kernel_size", None);
            let stride = pair(case, "stride", Some(kernel));
            let padding = pair(case, "padding", Some([0, 0]));
            if case.op == "nn.max_pool2d" {
                x.max_pool2d(
                    kernel,
                    stride,
                    padding,
                    pair(case, "dilation", Some([1, 1])),
                )?
            } else {
                x.avg_pool2d(
                    kernel,
                    stride,
                    padding,
                    flag(case, "count_include_pad", true),
                )?
            }
        }
        "nn.adaptive_avg_pool2d" => x.adaptive_avg_pool2d(pair(case, "output_size", None))?,
        "nn.dropout" => x.dropout(float(case, "p", Some(0.5)))?,
        "nn.batch_norm" => x.batch_norm(
            param("running_mean").unwrap(),
            param("running_var").unwrap(),
            param("weight"),
            param("bias"),
            eps,
        )?,
        "nn.layer_norm" => {
            let (weight, bias) = (param("weight"), param("bias"));
            let shape = normalized_shape(weight.as_ref());
            x.layer_norm(&shape, weight, bias, eps)?
        }
        "nn.group_norm" => x.group_norm(
            int(case, "num_groups", 0),
            param("weight"),
            param("bias"),
            eps,
        )?,
        "nn.rms_norm" => {
            let weight = param("weight");
            let shape = normalized_shape(weight.as_ref());
            x.rms_norm(&shape, weight, float(case, "eps", None))?
        }
        "nn.relu" => x.relu()?,
        "nn.leaky_relu" => x.leaky_relu(float(case, "negative_slope", Some(0.01)))?,
        "nn.gelu" => x.gelu()?,
        "nn.gelu_tanh" => x.gelu_tanh()?,
        "nn.silu" => x.silu()?,
        "nn.sigmoid" => x.sigmoid()?,
        "nn.softplus" => x.softplus(
            float(case, "beta", Some(1.0)),
            float(case, "threshold", Some(20.0)),
        )?,
        "nn.mish" => x.mish()?,
        other => panic!("no runner for {other}"),
    };
    assert!(
        params.is_empty(),
        "{}/{}: unused params {:?}",
        case.op,
        case.name,
        params.keys()
    );
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
