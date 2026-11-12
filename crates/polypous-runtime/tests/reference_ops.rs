//! Every M1 op through the interpreter, as a one-node graph against its op fixture
//! (`fixtures/small/ops/`, testing-strategy.md §6; runtime.md §3).
//!
//! The fixture harness is the backend's (`crates/polypous-backend/tests/reference/common.rs`,
//! included with `#[path]`), so cases load, skip (`SKIP …`: a dtype the device lacks), compare
//! with the sidecar tolerance and summarize (`SUMMARY …`) exactly as in the backend's reference
//! tests. Per case, [`run`] builds the graph from the sidecar the way
//! `crates/polypous-ir/tests/catalog.rs` does (sidecar inputs → graph inputs, `params` → graph
//! params, attributes as written, one graph output per op output), compiles a [`Plan`] for the
//! device and the case's compute dtype, loads the params from a safetensors file holding the
//! case's params (written in memory with `polypous-formats`), and runs it. Cases named
//! `permuted_…` etc. get their first input as a non-contiguous view, as in the backend tests.
//!
//! ```text
//! cargo test -p polypous-runtime --test reference_ops -- --nocapture                       # CPU
//! cargo test -p polypous-runtime --features vulkan --test reference_ops -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "../../polypous-backend/tests/reference/common.rs"]
mod common;

use std::collections::BTreeMap;

use common::Case;
use polypous_backend::device::DeviceId;
use polypous_backend::{DynTensor, TensorError};
use polypous_formats::safetensors::{
    Dtype as StorageDtype, SafetensorsFile, WriteTensor, serialize,
};
use polypous_ir::catalog::{self, InDtype, Outputs};
use polypous_ir::{ComputeDtype, Dtype, ValidateOptions};
use polypous_runtime::{Model, Plan, PlanOptions, RunOptions, RuntimeError, WeightFile};
use serde_json::{Value, json};

fn sidecar(case: &Case) -> Value {
    let path = common::fixture_dir(&case.op).join(format!("{}.json", case.name));
    serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap()
}

fn dtype_of(s: &str) -> Dtype {
    Dtype::parse(s).unwrap_or_else(|| panic!("unknown dtype {s}"))
}

/// The one-node graph of a sidecar and its compute dtype (as in `polypous-ir/tests/catalog.rs`).
fn graph_for(sidecar: &Value) -> (Value, ComputeDtype) {
    let op_name = sidecar["op"].as_str().unwrap();
    let op = catalog::lookup(op_name).unwrap_or_else(|| panic!("{op_name} is not in the catalog"));
    let mut inputs = Vec::new();
    let mut params = Vec::new();
    let mut node_inputs = serde_json::Map::new();
    let mut node_params = serde_json::Map::new();
    let mut variadic = Vec::new();
    let mut compute = None;
    for t in sidecar["inputs"].as_array().unwrap() {
        let name = t["name"].as_str().unwrap();
        assert!(
            op.param(name).is_none(),
            "{op_name}/{}: param `{name}` listed under inputs",
            sidecar["case"]
        );
        inputs.push(json!({ "name": name, "dtype": t["dtype"], "shape": t["shape"] }));
        match op.inputs.iter().find(|i| i.name == name) {
            Some(i) if i.variadic => variadic.push(json!(name)),
            Some(i) => {
                if i.dtype == InDtype::Compute {
                    compute.get_or_insert(dtype_of(t["dtype"].as_str().unwrap()));
                }
                node_inputs.insert(name.into(), json!(name));
            }
            None if op.inputs.iter().any(|i| i.variadic) => variadic.push(json!(name)),
            None => panic!("{op_name}/{}: unknown input `{name}`", sidecar["case"]),
        }
    }
    for t in sidecar
        .get("params")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let name = t["name"].as_str().unwrap();
        params.push(json!({ "name": name, "dtype": "float", "shape": t["shape"] }));
        node_params.insert(name.into(), json!(name));
        compute.get_or_insert(dtype_of(t["dtype"].as_str().unwrap()));
    }
    if let Some(v) = op.inputs.iter().find(|i| i.variadic) {
        node_inputs.insert(v.name.into(), json!(variadic));
    }
    let outputs: Vec<Value> = sidecar["outputs"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(k, o)| {
            let port = match op.outputs {
                Outputs::Fixed(outs) => outs[k].name.to_owned(),
                Outputs::Split => k.to_string(),
            };
            json!({ "name": o["name"], "value": format!("n:{port}") })
        })
        .collect();
    let graph = json!({
        "format": "polypous.graph", "format_version": 1, "opsets": { op.opset(): 1 },
        "inputs": inputs, "outputs": outputs, "params": params,
        "nodes": [ { "id": "n", "op": op_name, "inputs": node_inputs, "params": node_params,
                     "attrs": sidecar["attributes"] } ]
    });
    let compute = match compute {
        Some(Dtype::F16) => ComputeDtype::F16,
        Some(Dtype::Bf16) => ComputeDtype::Bf16,
        _ => ComputeDtype::F32,
    };
    (graph, compute)
}

/// The case's params as an in-memory safetensors file.
fn weight_file(case: &Case) -> WeightFile {
    let tensors: Vec<(String, StorageDtype, Vec<usize>, &[u8])> = case
        .params
        .iter()
        .map(|(name, data)| {
            let dtype = StorageDtype::from_name(&data.dtype.name().to_uppercase()).unwrap();
            let shape = data.shape.iter().map(|&d| d as usize).collect();
            (name.clone(), dtype, shape, data.bytes.as_slice())
        })
        .collect();
    let writes = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                WriteTensor::new(*dtype, shape, bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    let bytes = serialize(writes, &BTreeMap::new()).unwrap();
    WeightFile::from(SafetensorsFile::from_bytes(bytes).unwrap())
}

/// Maps an interpreter error back to the backend error the harness understands: an op's own
/// error stays as it is (so an `UnsupportedDType` from the op is a skip, as in the backend
/// tests); the plan's device check is the same skip; anything else is a failure.
fn to_tensor_error(err: RuntimeError) -> TensorError {
    match err {
        RuntimeError::Node { source, .. } => source,
        RuntimeError::UnsupportedDtype { dtype, device, .. } => TensorError::UnsupportedDType {
            dtype: polypous_backend::DType::from_name(dtype.as_str()).unwrap(),
            device,
        },
        other => TensorError::Backend {
            message: format!("interpreter: {other} (kind {:?})", other.kind()),
        },
    }
}

fn run(
    case: &Case,
    tensors: Vec<DynTensor>,
    device: DeviceId,
) -> Result<Vec<DynTensor>, TensorError> {
    let mut tensors = common::feed_layout(case, tensors)?;
    tensors.truncate(case.inputs.len()); // The params come from the weight file.
    let sidecar = sidecar(case);
    let (doc, compute_dtype) = graph_for(&sidecar);
    let bytes = serde_json::to_vec(&doc).unwrap();
    let options = ValidateOptions { compute_dtype };
    let outcome = polypous_ir::load(&bytes, &options);
    assert!(
        outcome.report.is_ok(),
        "{}/{}: {}",
        case.op,
        case.name,
        outcome.report
    );
    let graph = outcome.graph.unwrap();
    let plan = Plan::new(
        &graph,
        PlanOptions {
            device,
            compute_dtype,
        },
    )
    .map_err(to_tensor_error)?;
    let model = if case.params.is_empty() {
        Model::without_weights(plan)
    } else {
        Model::load(plan, &weight_file(case))
    }
    .map_err(to_tensor_error)?;
    let names = sidecar["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned());
    let inputs = names.zip(tensors).collect();
    let out = model
        .run(inputs, Vec::new(), &RunOptions::default())
        .map_err(to_tensor_error)?;
    assert!(out.state.is_empty());
    Ok(out.outputs.into_iter().map(|(_, t)| t).collect())
}

macro_rules! op_tests {
    ($($cpu:ident, $gpu:ident => $op:literal;)*) => {
        $(
            #[cfg(feature = "cpu")]
            #[test]
            fn $cpu() {
                common::run_cpu_on($op, run);
            }

            #[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
            #[test]
            #[ignore = "gpu"]
            fn $gpu() {
                common::run_gpu_on($op, run);
            }
        )*

        /// Every catalog op has a test here and a fixture directory.
        #[test]
        fn every_catalog_op_is_tested() {
            let tested = [$($op),*];
            let catalog: Vec<&str> = catalog::OPS.iter().map(|o| o.name).collect();
            assert_eq!(tested.to_vec(), catalog);
            for op in catalog {
                assert!(common::fixture_dir(op).is_dir(), "no fixtures for {op}");
            }
        }
    };
}

op_tests! {
    core_add_cpu, core_add_gpu => "core.add";
    core_sub_cpu, core_sub_gpu => "core.sub";
    core_mul_cpu, core_mul_gpu => "core.mul";
    core_div_cpu, core_div_gpu => "core.div";
    core_pow_cpu, core_pow_gpu => "core.pow";
    core_maximum_cpu, core_maximum_gpu => "core.maximum";
    core_minimum_cpu, core_minimum_gpu => "core.minimum";
    core_neg_cpu, core_neg_gpu => "core.neg";
    core_abs_cpu, core_abs_gpu => "core.abs";
    core_exp_cpu, core_exp_gpu => "core.exp";
    core_log_cpu, core_log_gpu => "core.log";
    core_sqrt_cpu, core_sqrt_gpu => "core.sqrt";
    core_rsqrt_cpu, core_rsqrt_gpu => "core.rsqrt";
    core_sin_cpu, core_sin_gpu => "core.sin";
    core_cos_cpu, core_cos_gpu => "core.cos";
    core_tanh_cpu, core_tanh_gpu => "core.tanh";
    core_clamp_cpu, core_clamp_gpu => "core.clamp";
    core_where_cpu, core_where_gpu => "core.where";
    core_eq_cpu, core_eq_gpu => "core.eq";
    core_ne_cpu, core_ne_gpu => "core.ne";
    core_lt_cpu, core_lt_gpu => "core.lt";
    core_le_cpu, core_le_gpu => "core.le";
    core_gt_cpu, core_gt_gpu => "core.gt";
    core_ge_cpu, core_ge_gpu => "core.ge";
    core_and_cpu, core_and_gpu => "core.and";
    core_or_cpu, core_or_gpu => "core.or";
    core_not_cpu, core_not_gpu => "core.not";
    core_reshape_cpu, core_reshape_gpu => "core.reshape";
    core_flatten_cpu, core_flatten_gpu => "core.flatten";
    core_permute_cpu, core_permute_gpu => "core.permute";
    core_transpose_cpu, core_transpose_gpu => "core.transpose";
    core_squeeze_cpu, core_squeeze_gpu => "core.squeeze";
    core_unsqueeze_cpu, core_unsqueeze_gpu => "core.unsqueeze";
    core_expand_cpu, core_expand_gpu => "core.expand";
    core_concat_cpu, core_concat_gpu => "core.concat";
    core_split_cpu, core_split_gpu => "core.split";
    core_slice_cpu, core_slice_gpu => "core.slice";
    core_pad_cpu, core_pad_gpu => "core.pad";
    core_gather_cpu, core_gather_gpu => "core.gather";
    core_index_select_cpu, core_index_select_gpu => "core.index_select";
    core_constant_cpu, core_constant_gpu => "core.constant";
    core_arange_cpu, core_arange_gpu => "core.arange";
    core_zeros_cpu, core_zeros_gpu => "core.zeros";
    core_ones_cpu, core_ones_gpu => "core.ones";
    core_full_cpu, core_full_gpu => "core.full";
    core_full_like_cpu, core_full_like_gpu => "core.full_like";
    core_sum_cpu, core_sum_gpu => "core.sum";
    core_mean_cpu, core_mean_gpu => "core.mean";
    core_max_cpu, core_max_gpu => "core.max";
    core_min_cpu, core_min_gpu => "core.min";
    core_argmax_cpu, core_argmax_gpu => "core.argmax";
    core_argmin_cpu, core_argmin_gpu => "core.argmin";
    core_matmul_cpu, core_matmul_gpu => "core.matmul";
    core_cast_cpu, core_cast_gpu => "core.cast";
    core_softmax_cpu, core_softmax_gpu => "core.softmax";
    core_log_softmax_cpu, core_log_softmax_gpu => "core.log_softmax";
    core_cumsum_cpu, core_cumsum_gpu => "core.cumsum";
    core_top_k_cpu, core_top_k_gpu => "core.top_k";
    nn_linear_cpu, nn_linear_gpu => "nn.linear";
    nn_embedding_cpu, nn_embedding_gpu => "nn.embedding";
    nn_conv1d_cpu, nn_conv1d_gpu => "nn.conv1d";
    nn_conv2d_cpu, nn_conv2d_gpu => "nn.conv2d";
    nn_max_pool2d_cpu, nn_max_pool2d_gpu => "nn.max_pool2d";
    nn_avg_pool2d_cpu, nn_avg_pool2d_gpu => "nn.avg_pool2d";
    nn_adaptive_avg_pool2d_cpu, nn_adaptive_avg_pool2d_gpu => "nn.adaptive_avg_pool2d";
    nn_dropout_cpu, nn_dropout_gpu => "nn.dropout";
    nn_batch_norm_cpu, nn_batch_norm_gpu => "nn.batch_norm";
    nn_layer_norm_cpu, nn_layer_norm_gpu => "nn.layer_norm";
    nn_group_norm_cpu, nn_group_norm_gpu => "nn.group_norm";
    nn_rms_norm_cpu, nn_rms_norm_gpu => "nn.rms_norm";
    nn_relu_cpu, nn_relu_gpu => "nn.relu";
    nn_leaky_relu_cpu, nn_leaky_relu_gpu => "nn.leaky_relu";
    nn_gelu_cpu, nn_gelu_gpu => "nn.gelu";
    nn_gelu_tanh_cpu, nn_gelu_tanh_gpu => "nn.gelu_tanh";
    nn_silu_cpu, nn_silu_gpu => "nn.silu";
    nn_sigmoid_cpu, nn_sigmoid_gpu => "nn.sigmoid";
    nn_softplus_cpu, nn_softplus_gpu => "nn.softplus";
    nn_mish_cpu, nn_mish_gpu => "nn.mish";
}
