//! Runner of the `nn` ops (graph-ir.md §3.3: layers, normalization, activations), shared by the
//! forward reference test (`tests/reference_nn.rs`) and the gradient reference test
//! (`tests/grad_reference_nn.rs`).
//!
//! Params come from the sidecar's `params` list (testing-strategy.md §3.2), after the inputs. The
//! sidecars hold the attributes as written in a graph (defaults left out, `normalized_shape`
//! possibly left out); [`run`] fills them in the way `polypous_ir::Analysis` does before the
//! runtime calls the backend. Cases named `permuted_…` or `transposed_…` get their first input as
//! a non-contiguous view (`common::feed_layout`).
//!
//! Training-mode cases (`case.mode == "training"`, schema 2, testing-strategy.md §3.6):
//!
//! - `nn.batch_norm` runs [`DynTensor::batch_norm_training`] (`momentum` default `0.1`) and
//!   returns `out`, then the new `running_mean` and `running_var`, as the harness expects.
//! - `nn.dropout` gets the case's `mask` (an `f32` tensor of `0` and `1`) after the inputs, and
//!   runs [`DynTensor::dropout_training`] with it as a `bool` keep mask (`1` → kept).

use std::collections::BTreeMap;

use super::common::{self, Case};
use polypous_backend::{DType, DynTensor, TensorError};

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

pub fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let mut inputs = common::feed_layout(case, inputs)?;
    let params_start = case.inputs.len();
    let training = case.mode == "training";
    // The keep mask of a training-mode dropout case comes last (`grads.rs`).
    let mask = if case.mask.is_some() {
        Some(inputs.pop().expect("the mask"))
    } else {
        None
    };
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
        "nn.dropout" if training => {
            // `1` kept, `0` dropped; the float-to-bool cast maps every nonzero to true.
            let keep = mask.expect("a training-mode dropout case has a mask");
            x.dropout_training(float(case, "p", Some(0.5)), keep.cast(DType::Bool)?)?
        }
        "nn.dropout" => x.dropout(float(case, "p", Some(0.5)))?,
        "nn.batch_norm" if training => {
            let (out, running_mean, running_var) = x.batch_norm_training(
                param("running_mean").unwrap(),
                param("running_var").unwrap(),
                param("weight"),
                param("bias"),
                eps,
                float(case, "momentum", Some(0.1)),
            )?;
            return Ok(vec![out, running_mean, running_var]);
        }
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
