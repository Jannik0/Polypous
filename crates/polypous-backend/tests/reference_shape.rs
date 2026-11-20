//! Reference tests of the shape ops (`core.reshape flatten permute transpose squeeze unsqueeze
//! expand concat split slice pad`) against the PyTorch fixtures in `fixtures/small/ops/`
//! (graph-ir.md §3.2, testing-strategy.md §6). The harness is `tests/reference/common.rs`.
//!
//! The sidecars hold the attributes as written in a graph (negative axes, `-1`, negative `slice`
//! bounds, defaults left out). [`run`] normalizes them the way `polypous_ir::Analysis` does
//! before the runtime calls the backend, and cases named `permuted_…`, `transposed_…`,
//! `expanded_…` or `strided_…` get their first input as a non-contiguous view
//! (`common::feed_layout`).
//!
//! ```text
//! cargo test -p polypous-backend --test reference_shape                     # CPU
//! cargo test -p polypous-backend --features vulkan --test reference_shape -- --include-ignored --nocapture
//! ```

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

#[path = "reference/run_shape.rs"]
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
    core_reshape => "core.reshape",
    core_flatten => "core.flatten",
    core_permute => "core.permute",
    core_transpose => "core.transpose",
    core_squeeze => "core.squeeze",
    core_unsqueeze => "core.unsqueeze",
    core_expand => "core.expand",
    core_concat => "core.concat",
    core_split => "core.split",
    core_slice => "core.slice",
    core_pad => "core.pad",
}

#[test]
fn layouts_are_chosen_by_prefix() {
    use common::Layout;
    assert_eq!(Layout::of("permuted_rank3_f32"), Layout::Permuted);
    assert_eq!(Layout::of("transposed_i64"), Layout::Transposed);
    assert_eq!(Layout::of("expanded_bool"), Layout::Expanded);
    assert_eq!(Layout::of("strided_f16"), Layout::Strided);
    assert_eq!(Layout::of("rank1_f32"), Layout::Contiguous);
    assert_eq!(Layout::of("my_permuted_f32"), Layout::Contiguous);
}
