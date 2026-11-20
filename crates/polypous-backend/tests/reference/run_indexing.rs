//! Runner of `core.gather` and `core.index_select`, shared by the forward reference test
//! (`tests/reference_indexing.rs`) and the gradient reference test
//! (`tests/grad_reference_indexing.rs`).

use super::common::{self, Case};
use polypous_backend::{DynTensor, TensorError};

pub fn run(case: &Case, inputs: Vec<DynTensor>) -> Result<Vec<DynTensor>, TensorError> {
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
