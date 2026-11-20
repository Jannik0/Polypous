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

#[path = "reference/run_elementwise.rs"]
mod runner;

use runner::run;

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
