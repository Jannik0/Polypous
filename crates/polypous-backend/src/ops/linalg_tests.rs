//! Unit tests of `linalg.rs`: `matmul` shape rules and batch broadcasting across ranks. Results
//! against PyTorch are in `tests/reference_reduce.rs`; dtype and device errors in
//! `tensor/dyn_tensor_tests.rs`.

#![cfg(feature = "cpu")]

use super::super::reduce::tests::{floats, read_f32};
use super::*;

#[test]
fn shape_rules() {
    let a = floats(&[2, 3], &[1.0; 6]);
    let shape = |r: Result<DynTensor>| r.unwrap().shape();
    // Different batch ranks broadcast; the output has the larger rank.
    assert_eq!(
        shape(a.clone().matmul(floats(&[4, 3, 5], &[1.0; 60]))),
        vec![4, 2, 5]
    );
    assert_eq!(
        shape(floats(&[2, 1, 2, 3], &[1.0; 12]).matmul(floats(&[5, 3, 1], &[1.0; 15]))),
        vec![2, 5, 2, 1]
    );
    // k mismatch, rank 1, incompatible batch dims.
    for (lhs, rhs) in [
        (vec![2, 3], vec![2, 3]),
        (vec![3], vec![3, 2]),
        (vec![2, 3], vec![3]),
        (vec![2, 2, 3], vec![3, 3, 2]),
    ] {
        let l = floats(
            &lhs.iter().map(|&d| d as u64).collect::<Vec<_>>(),
            &vec![1.0; lhs.iter().product()],
        );
        let r = floats(
            &rhs.iter().map(|&d| d as u64).collect::<Vec<_>>(),
            &vec![1.0; rhs.iter().product()],
        );
        assert_eq!(
            l.matmul(r).unwrap_err(),
            TensorError::ShapeMismatch {
                op: "matmul",
                lhs,
                rhs
            }
        );
    }
}

#[test]
fn broadcast_batch_values() {
    // [2, 2, 2] @ [2, 1]: the rank-2 rhs applies to both batches.
    let a = floats(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
    let b = floats(&[2, 1], &[3.0, 4.0]);
    let c = a.matmul(b).unwrap();
    assert_eq!(c.shape(), vec![2, 2, 1]);
    assert_eq!(read_f32(c), vec![3.0, 4.0, 6.0, 8.0]);
}
