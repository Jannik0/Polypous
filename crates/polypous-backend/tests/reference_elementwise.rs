//! Reference tests of the elementwise, comparison and logic ops against the PyTorch fixtures in
//! `fixtures/small/ops/` (graph-ir.md §3.2, testing-strategy.md §6). The harness is
//! `tests/reference/common.rs`.
//!
//! ```text
//! cargo test -p polypous-backend --test reference_elementwise                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_elementwise -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::Case;
use polypous_backend::{DynTensor, TensorError};

/// The inputs of a case as an array (the sidecar lists exactly the op's inputs).
fn take<const N: usize>(case: &Case, inputs: Vec<DynTensor>) -> [DynTensor; N] {
    let count = inputs.len();
    inputs
        .try_into()
        .unwrap_or_else(|_| panic!("{}/{}: {count} inputs, expected {N}", case.op, case.name))
}

fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
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

/// One CPU test and one GPU test (`#[ignore = "gpu"]`) per op.
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
    core_add => "core.add",
    core_sub => "core.sub",
    core_mul => "core.mul",
    core_div => "core.div",
    core_pow => "core.pow",
    core_maximum => "core.maximum",
    core_minimum => "core.minimum",
    core_neg => "core.neg",
    core_abs => "core.abs",
    core_exp => "core.exp",
    core_log => "core.log",
    core_sqrt => "core.sqrt",
    core_rsqrt => "core.rsqrt",
    core_sin => "core.sin",
    core_cos => "core.cos",
    core_tanh => "core.tanh",
    core_clamp => "core.clamp",
    core_where => "core.where",
    core_eq => "core.eq",
    core_ne => "core.ne",
    core_lt => "core.lt",
    core_le => "core.le",
    core_gt => "core.gt",
    core_ge => "core.ge",
    core_and => "core.and",
    core_or => "core.or",
    core_not => "core.not",
}
