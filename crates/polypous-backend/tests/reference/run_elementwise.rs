//! Runner of the elementwise, comparison and logic ops, shared by the forward
//! reference test (`tests/reference_elementwise.rs`) and the gradient reference test
//! (`tests/grad_reference_elementwise.rs`).

use super::common::Case;
use polypous_backend::{DynTensor, TensorError};

/// The inputs of a case as an array (the sidecar lists exactly the op's inputs).
fn take<const N: usize>(case: &Case, inputs: Vec<DynTensor>) -> [DynTensor; N] {
    let count = inputs.len();
    inputs
        .try_into()
        .unwrap_or_else(|_| panic!("{}/{}: {count} inputs, expected {N}", case.op, case.name))
}

pub fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
    let out = match case.op.as_str() {
        "core.add" | "core.sub" | "core.mul" | "core.div" | "core.pow" | "core.maximum"
        | "core.minimum" | "core.eq" | "core.ne" | "core.lt" | "core.le" | "core.gt"
        | "core.ge" | "core.and" | "core.or" => {
            let [a, b] = take(case, inputs);
            match case.op.as_str() {
                "core.add" => a.add(b),
                "core.sub" => a.sub(b),
                "core.mul" => a.mul(b),
                "core.div" => a.div(b),
                "core.pow" => a.pow(b),
                "core.maximum" => a.maximum(b),
                "core.minimum" => a.minimum(b),
                "core.eq" => a.equal(b),
                "core.ne" => a.not_equal(b),
                "core.lt" => a.less(b),
                "core.le" => a.less_equal(b),
                "core.gt" => a.greater(b),
                "core.ge" => a.greater_equal(b),
                "core.and" => a.logical_and(b),
                _ => a.logical_or(b),
            }
        }
        "core.where" => {
            let [cond, a, b] = take(case, inputs);
            DynTensor::where_(cond, a, b)
        }
        "core.clamp" => {
            let [x] = take(case, inputs);
            x.clamp(case.scalar("min"), case.scalar("max"))
        }
        op => {
            let [x] = take(case, inputs);
            match op {
                "core.neg" => x.neg(),
                "core.abs" => x.abs(),
                "core.exp" => x.exp(),
                "core.log" => x.log(),
                "core.sqrt" => x.sqrt(),
                "core.rsqrt" => x.rsqrt(),
                "core.sin" => x.sin(),
                "core.cos" => x.cos(),
                "core.tanh" => x.tanh(),
                "core.not" => x.logical_not(),
                other => panic!("no runner for {other}"),
            }
        }
    }?;
    Ok(vec![out])
}
