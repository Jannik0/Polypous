//! Tests of the autodiff API ([`DynTensor::require_grad`], [`DynTensor::backward`],
//! [`Gradients`], `detach`, `without_autodiff`): every misuse is an error, gradients have the
//! leaf's dtype, shape and device, the ops that fix Burn's gradients (graph-ir.md §3.1.11) and
//! the training modes never panic on non-finite inputs, the gradients the fixtures don't cover
//! match their formulas (`core` and `nn` ops), and `argmax`/`argmin` record nothing. The CPU tests use `flex`; the GPU test is
//! `#[ignore = "gpu"]` and runs on every GPU device (lavapipe only with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1`).
//!
//! An integration test of its own, so that its GPU work doesn't run in parallel with the
//! backend's other GPU unit tests in one process: on lavapipe that combination crashed with
//! SIGSEGV inside lavapipe's kernel code (#75, docs/status.md §4).
#![cfg(feature = "cpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

#[path = "reference/common.rs"]
mod common;

use common::same_values;
use polypous_backend::device::DeviceId;
use polypous_backend::tensor::Result;
use polypous_backend::{DType, DynTensor, Scalar, TensorData, TensorError};

fn f32s(values: &[f32], shape: &[u64], device: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    DynTensor::from_data(
        &TensorData::new(DType::F32, shape.to_vec(), bytes).unwrap(),
        device,
    )
    .unwrap()
}

fn i64s(values: &[i64], shape: &[u64], device: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    DynTensor::from_data(
        &TensorData::new(DType::I64, shape.to_vec(), bytes).unwrap(),
        device,
    )
    .unwrap()
}

fn read(t: &DynTensor) -> Vec<f32> {
    let t = if t.dtype() == DType::F32 {
        t.clone()
    } else {
        t.clone().cast(DType::F32).unwrap()
    };
    let data = t.into_data().unwrap();
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

/// `Err` unless `actual` holds the same values as `expected` by [`same_values`] (bit for bit,
/// any NaN matching any NaN).
fn same(expected: &DynTensor, actual: &DynTensor) -> std::result::Result<(), String> {
    same_values(&expected.to_data().unwrap(), &actual.to_data().unwrap())
}

/// `sum(t)` as a `[1]` tensor.
fn total(t: DynTensor) -> DynTensor {
    t.sum(None, true).unwrap().reshape(&[1]).unwrap()
}

fn scalar(value: f64, like: &DynTensor) -> DynTensor {
    DynTensor::full(&[1], Scalar::Float(value), like.dtype(), like.device()).unwrap()
}

fn invalid(result: Result<impl std::fmt::Debug>, op: &str) {
    match result {
        Err(TensorError::InvalidArgument { op: got, .. }) => assert_eq!(got, op),
        other => panic!("expected InvalidArgument from {op}, got {other:?}"),
    }
}

/// Every rank and float dtype: `loss = sum(x·x + 3x)` has the gradient `2x + 3`, in the leaf's
/// dtype, shape and device, without autodiff context.
fn leaves_of_every_rank_and_dtype(device: DeviceId) {
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        if !DynTensor::device_supports(device, dtype).unwrap() {
            eprintln!("SKIP {dtype} leaves on {device}: dtype not supported");
            continue;
        }
        for rank in 1..=6 {
            let shape: Vec<u64> = [2, 3, 1, 2, 1, 2][..rank].to_vec();
            let n: u64 = shape.iter().product();
            let values: Vec<f32> = (0..n).map(|i| i as f32 - 2.5).collect();
            let x = f32s(&values, &shape, device).cast(dtype).unwrap();
            assert!(!x.is_tracked() && !x.is_grad_leaf() && !x.has_autodiff());
            let leaf = x.require_grad().unwrap();
            assert!(leaf.is_tracked() && leaf.is_grad_leaf());
            assert_eq!(leaf.device(), device);
            let three = scalar(3.0, &leaf);
            let y = leaf
                .clone()
                .mul(leaf.clone())
                .unwrap()
                .add(leaf.clone().mul(three).unwrap())
                .unwrap();
            assert!(y.is_tracked() && !y.is_grad_leaf());
            let grads = total(y).backward().unwrap();
            let grad = grads.get(&leaf).unwrap().unwrap();
            assert_eq!(grad.dtype(), dtype);
            assert_eq!(grad.shape(), leaf.shape());
            assert_eq!(grad.device(), device);
            assert!(!grad.is_tracked() && !grad.has_autodiff());
            let want: Vec<f32> = values.iter().map(|v| 2.0 * v + 3.0).collect();
            assert_eq!(read(&grad), want, "{dtype} rank {rank} on {device}");
        }
    }
}

/// Every misuse is a `TensorError`, and the graph state survives a rejected second backward.
fn misuse_is_an_error(device: DeviceId) {
    let x = f32s(&[1.0, 2.0], &[2], device);
    let ints = i64s(&[1, 2], &[2], device);
    let bools = x.clone().greater(f32s(&[1.5], &[1], device)).unwrap();
    for t in [ints.clone(), bools] {
        assert!(matches!(
            t.clone().require_grad(),
            Err(TensorError::UnsupportedOperation {
                op: "require_grad",
                ..
            })
        ));
        assert!(matches!(
            t.backward(),
            Err(TensorError::UnsupportedOperation { op: "backward", .. })
        ));
    }
    // Untracked, and more than one element.
    invalid(total(x.clone()).backward(), "backward");
    let leaf = x.clone().require_grad().unwrap();
    invalid(
        leaf.clone().mul(leaf.clone()).unwrap().backward(),
        "backward",
    );

    // A second backward through the same graph.
    let loss = total(leaf.clone().mul(leaf.clone()).unwrap());
    let grads = loss.backward().unwrap();
    assert_eq!(read(&grads.get(&leaf).unwrap().unwrap()), vec![2.0, 4.0]);
    invalid(loss.backward(), "backward");
    invalid(loss.clone().backward(), "backward");
    // A fresh forward from the same leaf still works.
    let again = total(leaf.clone().mul(scalar(5.0, &leaf)).unwrap()).backward();
    assert_eq!(
        read(&again.unwrap().get(&leaf).unwrap().unwrap()),
        vec![5.0, 5.0]
    );

    // Gradients only of leaves.
    let y = leaf.clone().exp().unwrap();
    let mut grads = total(y.clone()).backward().unwrap();
    invalid(grads.get(&y), "gradient");
    invalid(grads.get(&x), "gradient");
    invalid(grads.get(&ints), "gradient");
    invalid(grads.take(&y), "gradient");
    // A leaf the loss doesn't depend on, and `take`.
    let other = x.clone().require_grad().unwrap();
    assert!(grads.get(&other).unwrap().is_none());
    assert!(grads.take(&leaf).unwrap().is_some());
    assert!(grads.take(&leaf).unwrap().is_none());
    assert!(format!("{grads:?}").starts_with("Gradients"));
}

/// `detach`, `without_autodiff`, `require_grad` on a tracked non-leaf, and gradients adding up
/// over uses and broadcast dims.
fn detach_and_gradient_flow(device: DeviceId) {
    let x = f32s(&[1.0, -2.0, 3.0], &[3], device);
    let leaf = x.clone().require_grad().unwrap();
    // sum(detach(x) · x): the detached factor is a constant.
    let loss = total(leaf.clone().detach().mul(leaf.clone()).unwrap());
    let grad = loss.backward().unwrap().get(&leaf).unwrap().unwrap();
    assert_eq!(read(&grad), vec![1.0, -2.0, 3.0]);
    // detach of untracked and non-float tensors: unchanged.
    same(&x, &x.clone().detach()).unwrap();
    assert!(!leaf.clone().detach().is_tracked());
    assert!(leaf.clone().detach().has_autodiff());
    let plain = leaf.clone().without_autodiff();
    assert!(!plain.is_tracked() && !plain.has_autodiff());
    same(&x, &plain).unwrap();
    // without_autodiff on int and bool tensors of a tracked graph.
    let ints = leaf.clone().cast(DType::I64).unwrap();
    assert!(!ints.is_tracked() && ints.has_autodiff());
    assert!(!ints.without_autodiff().has_autodiff());

    // A new leaf from a tracked non-leaf cuts the earlier ops.
    let y = leaf.clone().mul(scalar(2.0, &leaf)).unwrap();
    let z = y.require_grad().unwrap();
    let grads = total(z.clone().mul(z.clone()).unwrap()).backward().unwrap();
    assert_eq!(
        read(&grads.get(&z).unwrap().unwrap()),
        vec![4.0, -8.0, 12.0]
    );
    assert!(grads.get(&leaf).unwrap().is_none());

    // Broadcasting: a [3] leaf added to a [2, 3] tensor gets the sum over dim 0; a leaf used
    // twice gets both uses.
    let rows = f32s(&[0.0; 6], &[2, 3], device);
    let up = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device);
    let y = leaf.clone().add(rows).unwrap().mul(up).unwrap();
    let y = y.add(leaf.clone()).unwrap();
    let grad = total(y).backward().unwrap().get(&leaf).unwrap().unwrap();
    assert_eq!(read(&grad), vec![7.0, 9.0, 11.0]);
}

/// Tracked floats mix with bool masks, int indices and constants; the index checks leave the
/// graph intact; `full_like` passes no gradient; `to_device` keeps the graph.
///
/// `where` and `index_select` are back-propagated separately: their combination crashes lavapipe
/// (see `lavapipe_where_select_backward_crash` in `src/tensor/autodiff_tests.rs`).
fn mixing_and_index_checks(device: DeviceId, other: DeviceId) {
    let x = f32s(&[1.0, -2.0, 3.0, 0.5], &[4], device);
    let leaf = x.require_grad().unwrap();
    let zeros = leaf.full_like(Scalar::Float(0.0), None).unwrap();
    assert!(!zeros.is_tracked(), "full_like passes no gradient");
    let positive = leaf.clone().greater(zeros.clone()).unwrap();
    assert!(!positive.is_tracked());
    let relu = DynTensor::where_(positive, leaf.clone(), zeros).unwrap();
    let grad = total(relu).backward().unwrap().get(&leaf).unwrap().unwrap();
    assert_eq!(read(&grad), vec![1.0, 0.0, 1.0, 1.0]);

    // An out-of-range index is an error before the op runs; the graph is unchanged.
    let doubled = leaf.clone().mul(scalar(2.0, &leaf)).unwrap();
    let bad = i64s(&[0, 4], &[2], device);
    assert!(matches!(
        doubled.clone().gather(0, bad),
        Err(TensorError::IndexOutOfRange { .. })
    ));
    let index = i64s(&[2, 2, 0], &[3], device);
    let picked = doubled.index_select(0, index).unwrap();
    let moved = picked.to_device(other).unwrap();
    assert!(moved.is_tracked());
    assert_eq!(moved.device(), other);
    let grad = total(moved)
        .backward()
        .unwrap()
        .get(&leaf)
        .unwrap()
        .unwrap();
    assert_eq!(grad.device(), device);
    assert_eq!(read(&grad), vec![2.0, 0.0, 4.0, 0.0]);

    // A bool mask of a tracked tensor moved between backends (through the host, status.md §4).
    let mask = leaf.clone().greater(scalar(0.75, &leaf)).unwrap();
    let mask = mask.to_device(other).unwrap().to_device(device).unwrap();
    let kept = DynTensor::where_(mask, leaf.clone(), scalar(0.0, &leaf)).unwrap();
    let grad = total(kept).backward().unwrap().get(&leaf).unwrap().unwrap();
    assert_eq!(read(&grad), vec![1.0, 0.0, 1.0, 0.0]);
}

/// The ops that fix Burn's gradients complete a backward pass with NaN, `±inf` and `±0` inputs
/// (their gradients are unspecified there, graph-ir.md §3.1.11), and keep their forward value bit
/// for bit, except NaN payloads ([`same_values`]: the GPUs return their canonical NaN from the
/// tracked path's addition). Returns every failure, `"<op> on <device>: …"`, so that one failing
/// op doesn't hide the others.
fn fixed_ops_survive_non_finite_inputs(device: DeviceId) -> Vec<String> {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let a = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.5, nan, inf, 2.0];
    let b = [1.0, inf, inf, -0.0, 0.0, nan, -inf, nan, -inf, 0.5];
    let mut failures = Vec::new();
    let mut check = |name: &str, result: std::result::Result<(), String>| {
        if let Err(message) = result {
            failures.push(format!("{name} on {device}: {message}"));
        }
    };
    type Binary = fn(DynTensor, DynTensor) -> Result<DynTensor>;
    let binary: [(&str, Binary); 3] = [
        ("maximum", DynTensor::maximum),
        ("minimum", DynTensor::minimum),
        ("pow", DynTensor::pow),
    ];
    for (name, op) in binary {
        let run = || -> std::result::Result<(), String> {
            let (x, y) = (f32s(&a, &[10], device), f32s(&b, &[10], device));
            let plain = op(x.clone(), y.clone()).map_err(|e| e.to_string())?;
            let (lx, ly) = (x.require_grad().unwrap(), y.require_grad().unwrap());
            let out = op(lx.clone(), ly.clone()).map_err(|e| e.to_string())?;
            same(&plain, &out).map_err(|e| format!("forward: {e}"))?;
            let grads = total(out).backward().map_err(|e| e.to_string())?;
            for (input, leaf) in [("a", &lx), ("b", &ly)] {
                if grads.get(leaf).map_err(|e| e.to_string())?.is_none() {
                    return Err(format!("no gradient for {input}"));
                }
            }
            Ok(())
        };
        check(name, run());
    }
    type Unary = fn(DynTensor) -> Result<DynTensor>;
    let clamp_both: Unary = |x| x.clamp(Some(Scalar::Float(-1.0)), Some(Scalar::Float(1.0)));
    let clamp_nan: Unary = |x| x.clamp(Some(Scalar::Float(f64::NAN)), None);
    let unary: [(&str, Unary); 4] = [
        ("sqrt", DynTensor::sqrt),
        ("rsqrt", DynTensor::rsqrt),
        ("clamp", clamp_both),
        ("clamp NaN bound", clamp_nan),
    ];
    for (name, op) in unary {
        let run = || -> std::result::Result<(), String> {
            let x = f32s(&a, &[10], device);
            let plain = op(x.clone()).map_err(|e| e.to_string())?;
            let leaf = x.require_grad().unwrap();
            let out = op(leaf.clone()).map_err(|e| e.to_string())?;
            same(&plain, &out).map_err(|e| format!("forward: {e}"))?;
            if !out.is_tracked() {
                return Err("the result is not tracked".into());
            }
            let grads = total(out).backward().map_err(|e| e.to_string())?;
            if grads.get(&leaf).map_err(|e| e.to_string())?.is_none() {
                return Err("no gradient".into());
            }
            Ok(())
        };
        check(name, run());
    }
    // The reductions with a tie fix (#76) and `top_k`'s values, over every dim, some dims and
    // with keepdim. Rows of `a`: `[NaN, inf, -inf, 0, -0]` and `[1, -1.5, NaN, inf, 2]`.
    let max_all: Unary = |x| x.max(None, false);
    let max_dim1: Unary = |x| x.max(Some(&[1]), false);
    let max_dim0_keepdim: Unary = |x| x.max(Some(&[0]), true);
    let min_all: Unary = |x| x.min(None, true);
    let min_dim1: Unary = |x| x.min(Some(&[1]), false);
    let min_dims: Unary = |x| x.min(Some(&[0, 1]), false);
    let top_k_largest: Unary = |x| Ok(x.top_k(3, 1, true, DType::I64)?.0);
    let top_k_smallest: Unary = |x| Ok(x.top_k(2, 0, false, DType::I32)?.0);
    let reductions: [(&str, Unary); 8] = [
        ("max over every dim", max_all),
        ("max over dim 1", max_dim1),
        ("max over dim 0, keepdim", max_dim0_keepdim),
        ("min over every dim, keepdim", min_all),
        ("min over dim 1", min_dim1),
        ("min over dims 0 and 1", min_dims),
        ("top_k largest", top_k_largest),
        ("top_k smallest", top_k_smallest),
    ];
    for (name, op) in reductions {
        let run = || -> std::result::Result<(), String> {
            let x = f32s(&a, &[2, 5], device);
            let plain = op(x.clone()).map_err(|e| e.to_string())?;
            let leaf = x.require_grad().unwrap();
            let out = op(leaf.clone()).map_err(|e| e.to_string())?;
            same(&plain, &out).map_err(|e| format!("forward: {e}"))?;
            if !out.is_tracked() {
                return Err("the result is not tracked".into());
            }
            let grads = total(out).backward().map_err(|e| e.to_string())?;
            let grad = grads
                .get(&leaf)
                .map_err(|e| e.to_string())?
                .ok_or("no gradient")?;
            if grad.shape() != [2, 5] {
                return Err(format!("gradient shape {:?}", grad.shape()));
            }
            Ok(())
        };
        check(name, run());
    }
    // The `nn` ops with a gradient fix or a training mode (#77), on rank-4 `[1, 2, 1, 5]` and
    // `[2, 5]` views of `a`: activations, `max_pool2d` (NaN, `±inf` and padding in its windows),
    // `dropout` in training mode (NaN and `inf` at dropped elements) and `batch_norm` in training
    // mode (NaN and `inf` in a channel), whose running statistics must match too.
    let keep: Vec<i64> = (0..10).map(|i| i64::from(i % 3 != 0)).collect();
    type Nn = fn(DynTensor, DeviceId, &[i64]) -> Result<Vec<DynTensor>>;
    let nn_ops: [(&str, Nn); 9] = [
        ("relu", |x, _, _| Ok(vec![x.relu()?])),
        ("leaky_relu", |x, _, _| Ok(vec![x.leaky_relu(0.25)?])),
        ("sigmoid", |x, _, _| Ok(vec![x.sigmoid()?])),
        ("silu", |x, _, _| Ok(vec![x.silu()?])),
        ("softplus", |x, _, _| Ok(vec![x.softplus(2.0, 1.0)?])),
        ("max_pool2d", |x, _, _| {
            Ok(vec![x.reshape(&[1, 2, 1, 5])?.max_pool2d(
                [1, 2],
                [1, 1],
                [0, 1],
                [1, 1],
            )?])
        }),
        ("max_pool2d, all-padding windows", |x, _, _| {
            Ok(vec![x.reshape(&[1, 1, 2, 5])?.max_pool2d(
                [2, 1],
                [1, 1],
                [1, 0],
                [3, 1],
            )?])
        }),
        ("dropout_training", |x, device, keep| {
            let mask = i64s(keep, &[2, 5], device).cast(DType::Bool)?;
            Ok(vec![x.dropout_training(0.3, mask)?])
        }),
        ("batch_norm_training", |x, device, _| {
            let stats = f32s(&[0.5, 2.0], &[2], device);
            let (out, mean, var) = x.reshape(&[1, 2, 5])?.batch_norm_training(
                stats.clone(),
                stats.clone(),
                Some(stats.clone()),
                Some(stats),
                1e-5,
                0.1,
            )?;
            Ok(vec![out, mean, var])
        }),
    ];
    for (name, op) in nn_ops {
        let run = || -> std::result::Result<(), String> {
            let x = f32s(&a, &[2, 5], device);
            let plain = op(x.clone(), device, &keep).map_err(|e| e.to_string())?;
            let leaf = x.require_grad().unwrap();
            let outs = op(leaf.clone(), device, &keep).map_err(|e| e.to_string())?;
            for (p, o) in plain.iter().zip(&outs) {
                same(p, o).map_err(|e| format!("forward: {e}"))?;
            }
            let out = outs.into_iter().next().ok_or("no output")?;
            if !out.is_tracked() {
                return Err("the result is not tracked".into());
            }
            let grads = total(out).backward().map_err(|e| e.to_string())?;
            let grad = grads
                .get(&leaf)
                .map_err(|e| e.to_string())?
                .ok_or("no gradient")?;
            if grad.shape() != [2, 5] {
                return Err(format!("gradient shape {:?}", grad.shape()));
            }
            grad.to_data().map_err(|e| e.to_string())?;
            Ok(())
        };
        check(name, run());
    }
    failures
}

/// `argmax` and `argmin` aren't differentiable (graph-ir.md §3.2): on a tracked input they return
/// an untracked integer tensor, which works as an index into the tracked input (`gather`), and
/// a loss that uses both the input and its `argmax` back-propagates with the gradient of the
/// float path only.
fn non_differentiable_ops_record_nothing(device: DeviceId) {
    let x = f32s(&[3.0, 1.0, 3.0, -2.0, 5.0, 0.5], &[2, 3], device)
        .require_grad()
        .unwrap();
    for (name, index) in [
        ("argmax", x.clone().argmax(1, true, DType::I64).unwrap()),
        ("argmin", x.clone().argmin(1, true, DType::I32).unwrap()),
    ] {
        assert!(!index.is_tracked(), "{name}: the index is tracked");
        assert!(!index.is_grad_leaf(), "{name}: the index is a leaf");
        // loss = sum(x) + sum(gather(x, argmax)) · 2: d/dx = 1 + 2 at the picked element.
        let picked = x.clone().gather(1, index.clone()).unwrap();
        let loss = total(x.clone())
            .add(total(picked).mul(scalar(2.0, &x)).unwrap())
            .unwrap();
        let grads = loss.backward().unwrap();
        let grad = read(&grads.get(&x).unwrap().expect("a gradient"));
        let expected: [f32; 6] = if name == "argmax" {
            [3.0, 1.0, 1.0, 1.0, 3.0, 1.0] // the first 3 of row 0 (lowest index), the 5 of row 1
        } else {
            [1.0, 3.0, 1.0, 3.0, 1.0, 1.0]
        };
        assert_eq!(grad, expected, "{name}: gradient on {device}");
        assert_eq!(index.shape(), [2, 1], "{name}: shape");
    }
}

/// `Err` unless `got` is within `atol 1e-5`, `rtol 1e-3` of `want` (testing-strategy.md §2,
/// gradients), elementwise.
fn close(name: &str, want: &[f64], got: &[f32]) -> std::result::Result<(), String> {
    if want.len() != got.len() {
        return Err(format!(
            "{name}: {} elements, expected {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if (f64::from(g) - w).abs() > 1e-5 + 1e-3 * w.abs() {
            return Err(format!("{name}[{i}]: expected {w:e}, got {g:e}"));
        }
    }
    Ok(())
}

/// Deterministic test values in `[-1, 1)`.
fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed.wrapping_mul(40_503));
            (h % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// `∂/∂leaf` of `sum(out · g)` for each leaf.
fn grads_of(out: DynTensor, g: &[f32], leaves: &[&DynTensor]) -> Vec<Vec<f32>> {
    let shape: Vec<u64> = out.shape().iter().map(|&d| d as u64).collect();
    let upstream = f32s(g, &shape, out.device());
    let grads = total(out.mul(upstream).unwrap()).backward().unwrap();
    leaves
        .iter()
        .map(|leaf| read(&grads.get(leaf).unwrap().expect("a gradient")))
        .collect()
}

/// The gradients the fixtures don't cover, against their formulas in f64 on the host
/// (graph-ir.md §3.1.11: PyTorch's): `matmul` with a rank-2 operand against a rank-4 one (both
/// ways: the gradient of the rank-2 operand sums over both broadcast batch dims), `softmax` and
/// `log_softmax` at logits near `±1000` with a tied row maximum, and `cumsum` on a transposed
/// view (the reversed cumulative sum of `g`). Returns every failure.
fn gradients_match_their_formulas(device: DeviceId) -> Vec<String> {
    let mut failures = Vec::new();
    let mut check = |result: std::result::Result<(), String>| {
        if let Err(message) = result {
            failures.push(format!("{message} on {device}"));
        }
    };
    // matmul: a [3, 4] @ b [2, 3, 4, 5] = [2, 3, 3, 5], and b' [2, 3, 5, 4] @ a' [4, 2].
    let (n, k, m, batch) = (3usize, 4usize, 5usize, 6usize);
    let (av, bv) = (values(n * k, 1), values(batch * k * m, 2));
    let g = values(batch * n * m, 3);
    let a = f32s(&av, &[3, 4], device).require_grad().unwrap();
    let b = f32s(&bv, &[2, 3, 4, 5], device).require_grad().unwrap();
    let out = a.clone().matmul(b.clone()).unwrap();
    let [ga, gb]: [Vec<f32>; 2] = grads_of(out, &g, &[&a, &b]).try_into().unwrap();
    let (mut want_a, mut want_b) = (vec![0.0f64; n * k], vec![0.0f64; batch * k * m]);
    for t in 0..batch {
        for i in 0..n {
            for j in 0..m {
                let gij = f64::from(g[(t * n + i) * m + j]);
                for p in 0..k {
                    want_a[i * k + p] += gij * f64::from(bv[(t * k + p) * m + j]);
                    want_b[(t * k + p) * m + j] += f64::from(av[i * k + p]) * gij;
                }
            }
        }
    }
    check(close("matmul rank 2 @ rank 4: grad_a", &want_a, &ga));
    check(close("matmul rank 2 @ rank 4: grad_b", &want_b, &gb));
    // The other way round: [2, 3, 5, 4] @ [4, 2].
    let (n, k, m) = (5usize, 4usize, 2usize);
    let (av, bv) = (values(batch * n * k, 4), values(k * m, 5));
    let g = values(batch * n * m, 6);
    let a = f32s(&av, &[2, 3, 5, 4], device).require_grad().unwrap();
    let b = f32s(&bv, &[4, 2], device).require_grad().unwrap();
    let out = a.clone().matmul(b.clone()).unwrap();
    let [ga, gb]: [Vec<f32>; 2] = grads_of(out, &g, &[&a, &b]).try_into().unwrap();
    let (mut want_a, mut want_b) = (vec![0.0f64; batch * n * k], vec![0.0f64; k * m]);
    for t in 0..batch {
        for i in 0..n {
            for j in 0..m {
                let gij = f64::from(g[(t * n + i) * m + j]);
                for p in 0..k {
                    want_a[(t * n + i) * k + p] += gij * f64::from(bv[p * m + j]);
                    want_b[p * m + j] += f64::from(av[(t * n + i) * k + p]) * gij;
                }
            }
        }
    }
    check(close("matmul rank 4 @ rank 2: grad_a", &want_a, &ga));
    check(close("matmul rank 4 @ rank 2: grad_b", &want_b, &gb));

    // softmax / log_softmax along dim 1 at large logits; row 0 has a tied maximum.
    let x = [
        1000.0f32, 1001.0, 999.5, 1001.0, -1000.0, -1003.0, -999.0, -1000.5,
    ];
    let g = [0.5f32, -0.25, 0.75, 1.0, -0.5, 0.3, 0.9, -0.8];
    let (mut soft, mut log) = (vec![0.0f64; 8], vec![0.0f64; 8]);
    for row in 0..2 {
        let xs: Vec<f64> = x[row * 4..row * 4 + 4]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let gs: Vec<f64> = g[row * 4..row * 4 + 4]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = xs.iter().map(|v| (v - max).exp()).sum();
        let y: Vec<f64> = xs.iter().map(|v| (v - max).exp() / total).collect();
        let gy: f64 = gs.iter().zip(&y).map(|(g, y)| g * y).sum();
        let gsum: f64 = gs.iter().sum();
        for i in 0..4 {
            soft[row * 4 + i] = y[i] * (gs[i] - gy);
            log[row * 4 + i] = gs[i] - y[i] * gsum;
        }
    }
    for (name, want, log_form) in [("softmax", &soft, false), ("log_softmax", &log, true)] {
        let leaf = f32s(&x, &[2, 4], device).require_grad().unwrap();
        let out = if log_form {
            leaf.clone().log_softmax(1)
        } else {
            leaf.clone().softmax(1)
        }
        .unwrap();
        let [got]: [Vec<f32>; 1] = grads_of(out, &g, &[&leaf]).try_into().unwrap();
        check(close(&format!("{name} at large logits"), want, &got));
    }

    // cumsum along dim 1 of a transposed view: x [4, 3] stored, viewed as [3, 4].
    let xv = values(12, 7);
    let g = values(12, 8);
    let leaf = f32s(&xv, &[4, 3], device)
        .transpose(0, 1)
        .unwrap()
        .require_grad()
        .unwrap();
    let out = leaf.clone().cumsum(1).unwrap();
    let [got]: [Vec<f32>; 1] = grads_of(out, &g, &[&leaf]).try_into().unwrap();
    let mut want = vec![0.0f64; 12];
    for row in 0..3 {
        let mut acc = 0.0f64;
        for j in (0..4).rev() {
            acc += f64::from(g[row * 4 + j]);
            want[row * 4 + j] = acc;
        }
    }
    check(close("cumsum of a transposed view", &want, &got));
    failures
}

/// `Err` unless every element of `got` is within `rtol` of `want` (no `atol`: for gradients far
/// below `1e-5`), with `0` matching `±0` exactly.
fn close_rel(name: &str, want: &[f64], got: &[f32], rtol: f64) -> std::result::Result<(), String> {
    if want.len() != got.len() {
        return Err(format!(
            "{name}: {} elements, expected {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        if (f64::from(g) - w).abs() > rtol * w.abs() {
            return Err(format!("{name}[{i}]: expected {w:e}, got {g:e}"));
        }
    }
    Ok(())
}

/// The `nn` gradients and training modes the fixtures don't cover (graph-ir.md §3.1.11, §3.3),
/// against their formulas on the host:
///
/// - the kinks: `relu'(±0) = 0`, `leaky_relu'(±0) = negative_slope`, `softplus'` at `β·x = 0`
///   (`1/2`), just below and above `threshold` (`σ(β·x)` and `1`);
/// - the tails of `sigmoid` and `silu` at `x = -80` and `-60` (gradients near `1e-35` and
///   `1e-24`, relative tolerance `1e-3` without `atol`);
/// - `max_pool2d` with windows that lie entirely in the padding: output `-inf`, no gradient; and
///   at `0`/`-0` ties: the tracked forward equals the untracked one bit for bit, and `g` goes to
///   the first of the tied elements;
/// - `dropout` in training mode: kept elements are `x · s` and get `g · s` exactly (bit for bit,
///   with `s` from the spec's `f32` formula), dropped ones `0` (NaN for NaN and `±inf` inputs);
///   `p = 0` is the identity.
fn nn_gradients_match_their_formulas(device: DeviceId) -> Vec<String> {
    let mut failures = Vec::new();
    let mut check = |result: std::result::Result<(), String>| {
        if let Err(message) = result {
            failures.push(format!("{message} on {device}"));
        }
    };
    let x = [-0.0f32, 0.0, -1.5, 2.0, 0.25, -0.125];
    let g = [0.75f32, -1.25, 0.5, 2.0, -0.5, 1.5];
    let gradient = |op: &dyn Fn(DynTensor) -> Result<DynTensor>, x: &[f32], g: &[f32]| {
        let leaf = f32s(x, &[x.len() as u64], device).require_grad().unwrap();
        let out = op(leaf.clone()).unwrap();
        let [got]: [Vec<f32>; 1] = grads_of(out, g, &[&leaf]).try_into().unwrap();
        got
    };
    let relu = |v: f32| if v > 0.0 { 1.0 } else { 0.0 };
    let want: Vec<f64> = x
        .iter()
        .zip(&g)
        .map(|(&v, &g)| f64::from(g) * relu(v))
        .collect();
    check(close("relu at ±0", &want, &gradient(&|t| t.relu(), &x, &g)));
    let want: Vec<f64> = x
        .iter()
        .zip(&g)
        .map(|(&v, &g)| f64::from(g) * if v > 0.0 { 1.0 } else { 0.25 })
        .collect();
    check(close(
        "leaky_relu at ±0",
        &want,
        &gradient(&|t| t.leaky_relu(0.25), &x, &g),
    ));
    // softplus with beta 2, threshold 1: z = 2x is 0, 0, -3, 4 (> 1), 0.5, -0.25.
    let sp: Vec<f64> = x
        .iter()
        .zip(&g)
        .map(|(&v, &g)| {
            let z = 2.0 * f64::from(v);
            let d = if z > 1.0 {
                1.0
            } else {
                1.0 / (1.0 + (-z).exp())
            };
            f64::from(g) * d
        })
        .collect();
    check(close(
        "softplus at 0 and around the threshold",
        &sp,
        &gradient(&|t| t.softplus(2.0, 1.0), &x, &g),
    ));
    // Tails: sigmoid'(x) = s(1 - s), silu'(x) = s(1 + x(1 - s)).
    let tail = [-80.0f32, -60.0, -30.0];
    let ones = [1.0f32; 3];
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    let want: Vec<f64> = tail
        .iter()
        .map(|&v| {
            let s = sig(f64::from(v));
            s * (1.0 - s)
        })
        .collect();
    check(close_rel(
        "sigmoid tail",
        &want,
        &gradient(&|t| t.sigmoid(), &tail, &ones),
        1e-3,
    ));
    let want: Vec<f64> = tail
        .iter()
        .map(|&v| {
            let (v, s) = (f64::from(v), sig(f64::from(v)));
            s * (1.0 + v * (1.0 - s))
        })
        .collect();
    check(close_rel(
        "silu tail",
        &want,
        &gradient(&|t| t.silu(), &tail, &ones),
        1e-3,
    ));

    // max_pool2d, H = 2, kh = 2, p = 1, dl = 3: each window reads rows -1 and 2, both padding.
    let xs = values(6, 11);
    let leaf = f32s(&xs, &[1, 1, 2, 3], device).require_grad().unwrap();
    let out = leaf
        .clone()
        .max_pool2d([2, 1], [1, 1], [1, 0], [3, 1])
        .unwrap();
    let forward = read(&out);
    if forward != [f32::NEG_INFINITY; 3] {
        check(Err(format!(
            "max_pool2d all-padding windows: forward {forward:?}"
        )));
    }
    let [got]: [Vec<f32>; 1] = grads_of(out, &[1.0, -2.0, 3.0], &[&leaf])
        .try_into()
        .unwrap();
    check(close("max_pool2d all-padding windows", &[0.0; 6], &got));

    // max_pool2d at 0/-0 ties: the tracked forward is the untracked one bit for bit (lavapipe's
    // two pool kernels returned different zeros), and g goes to the first of the tied elements.
    let ties = [0.0f32, -0.0, -0.0, 0.0, 1.0, 1.0, -0.0, -0.0];
    let run = || -> std::result::Result<(), String> {
        let x = f32s(&ties, &[1, 1, 2, 4], device);
        let pool = |t: DynTensor| t.max_pool2d([1, 2], [1, 2], [0, 0], [1, 1]).unwrap();
        let plain = pool(x.clone());
        let leaf = x.require_grad().unwrap();
        let out = pool(leaf.clone());
        same(&plain, &out).map_err(|e| format!("forward: {e}"))?;
        let [got]: [Vec<f32>; 1] = grads_of(out, &[1.0, 2.0, 3.0, 4.0], &[&leaf])
            .try_into()
            .unwrap();
        let want = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0];
        if got != want {
            return Err(format!("gradient {got:?}, expected {want:?}"));
        }
        Ok(())
    };
    check(run().map_err(|e| format!("max_pool2d at 0/-0 ties: {e}")));

    // dropout in training mode, p = 0.3: s = 1 / f32(1 - 0.3).
    let p = 0.3f64;
    #[allow(clippy::cast_possible_truncation)]
    let s = 1.0f32 / ((1.0 - p) as f32);
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let xs = [1.5f32, -0.3, 7.0, inf, nan, -inf, 0.1, 3.3e-3, -2.25, 1e30];
    let keep = [1i64, 1, 0, 0, 0, 1, 1, 0, 1, 1];
    let gs = values(10, 12);
    let run = || -> std::result::Result<(), String> {
        let mask = i64s(&keep, &[2, 5], device).cast(DType::Bool).unwrap();
        let leaf = f32s(&xs, &[2, 5], device).require_grad().unwrap();
        let out = leaf.clone().dropout_training(p, mask.clone()).unwrap();
        let forward = read(&out);
        let [got]: [Vec<f32>; 1] = grads_of(out, &gs, &[&leaf]).try_into().unwrap();
        for i in 0..10 {
            let (want_out, want_grad) = if keep[i] == 1 {
                (xs[i] * s, gs[i] * s)
            } else {
                (xs[i] * 0.0, gs[i] * 0.0)
            };
            let same_bits =
                |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
            if !same_bits(forward[i], want_out) {
                return Err(format!(
                    "dropout out[{i}]: expected {want_out:e}, got {:e}",
                    forward[i]
                ));
            }
            if !same_bits(got[i], want_grad) {
                return Err(format!(
                    "dropout grad[{i}]: expected {want_grad:e}, got {:e}",
                    got[i]
                ));
            }
        }
        // p = 0: the identity, whatever the mask.
        let leaf = f32s(&xs, &[2, 5], device).require_grad().unwrap();
        let out = leaf.clone().dropout_training(0.0, mask).unwrap();
        same(&leaf, &out)?;
        let [got]: [Vec<f32>; 1] = grads_of(out, &gs, &[&leaf]).try_into().unwrap();
        if got != gs {
            return Err(format!("dropout p = 0: gradient {got:?}"));
        }
        Ok(())
    };
    check(run().map_err(|e| format!("dropout_training: {e}")));
    failures
}

/// Tracked `f16` inputs of the fixed activations, both training modes, the reductions, `matmul`,
/// `softmax`, `log_softmax`, `cumsum` and `top_k` (gradients are `f32` only in M2, graph-ir.md §3.1.11, but a graph that casts may
/// reach them): the backward pass completes, without a panic, with an `f16` gradient of the
/// leaf's shape.
fn half_precision_backward_completes(device: DeviceId) -> Vec<String> {
    let mut failures = Vec::new();
    if !DynTensor::device_supports(device, DType::F16).unwrap() {
        eprintln!("SKIP half_precision_backward_completes on {device}: no f16");
        return failures;
    }
    type Unary = fn(DynTensor) -> Result<DynTensor>;
    let ops: [(&str, Unary); 16] = [
        ("relu", DynTensor::relu),
        ("leaky_relu", |x| x.leaky_relu(0.1)),
        ("sigmoid", DynTensor::sigmoid),
        ("silu", DynTensor::silu),
        ("softplus", |x| x.softplus(1.0, 20.0)),
        ("dropout_training", |x| {
            let keep = x.clone().greater(x.full_like(Scalar::Float(0.0), None)?)?;
            x.dropout_training(0.5, keep)
        }),
        ("batch_norm_training", |x| {
            let c = x.clone().sum(Some(&[0]), false)?;
            let x = x.unsqueeze(2)?;
            Ok(
                x.batch_norm_training(c.clone(), c.clone(), Some(c.clone()), Some(c), 1e-5, 0.1)?
                    .0,
            )
        }),
        ("sum", |x| x.sum(Some(&[1]), false)),
        ("mean", |x| x.mean(None, false)),
        ("max", |x| x.max(Some(&[0]), true)),
        ("min", |x| x.min(None, false)),
        ("matmul", |x| {
            let t = x.clone().transpose(0, 1)?;
            x.matmul(t)
        }),
        ("softmax", |x| x.softmax(1)),
        ("log_softmax", |x| x.log_softmax(0)),
        ("cumsum", |x| x.cumsum(1)),
        ("top_k", |x| Ok(x.top_k(2, 1, true, DType::I64)?.0)),
    ];
    for (name, op) in ops {
        let run = || -> std::result::Result<(), String> {
            let x = f32s(&[1.0, 2.0, 2.0, -1.0, 0.5, 3.0], &[2, 3], device)
                .cast(DType::F16)
                .unwrap()
                .require_grad()
                .unwrap();
            let out = op(x.clone()).map_err(|e| e.to_string())?;
            let loss = total(out).cast(DType::F32).map_err(|e| e.to_string())?;
            let grads = loss.backward().map_err(|e| e.to_string())?;
            let grad = grads
                .get(&x)
                .map_err(|e| e.to_string())?
                .ok_or("no gradient")?;
            if grad.dtype() != DType::F16 || grad.shape() != [2, 3] {
                return Err(format!("gradient {} {:?}", grad.dtype(), grad.shape()));
            }
            Ok(())
        };
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
            Ok(Ok(())) => {}
            Ok(Err(message)) => failures.push(format!("{name} f16 on {device}: {message}")),
            Err(_) => failures.push(format!("{name} f16 on {device}: panicked")),
        }
    }
    failures
}

/// Runs `check`, turning a panic (a failed assertion) into a failure message.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
fn caught(name: &str, device: DeviceId, check: impl FnOnce()) -> Vec<String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(check)) {
        Ok(()) => Vec::new(),
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "panic".into());
            vec![format!("{name} on {device}: {message}")]
        }
    }
}

/// Every check on `device` (with `to_device` to and from `other`); all failures, none hidden.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
fn all(device: DeviceId, other: DeviceId) -> Vec<String> {
    let mut failures = Vec::new();
    failures.extend(caught("leaves_of_every_rank_and_dtype", device, || {
        leaves_of_every_rank_and_dtype(device);
    }));
    failures.extend(caught("misuse_is_an_error", device, || {
        misuse_is_an_error(device);
    }));
    failures.extend(caught("detach_and_gradient_flow", device, || {
        detach_and_gradient_flow(device);
    }));
    failures.extend(caught("mixing_and_index_checks", device, || {
        mixing_and_index_checks(device, other);
    }));
    failures.extend(caught(
        "mixing_and_index_checks (from the CPU)",
        device,
        || {
            mixing_and_index_checks(other, device);
        },
    ));
    failures.extend(caught(
        "non_differentiable_ops_record_nothing",
        device,
        || {
            non_differentiable_ops_record_nothing(device);
        },
    ));
    let mut formulas = Vec::new();
    failures.extend(caught("gradients_match_their_formulas", device, || {
        formulas = gradients_match_their_formulas(device);
    }));
    failures.extend(formulas);
    let mut nn = Vec::new();
    failures.extend(caught("nn_gradients_match_their_formulas", device, || {
        nn = nn_gradients_match_their_formulas(device);
    }));
    failures.extend(nn);
    let mut half = Vec::new();
    failures.extend(caught("half_precision_backward_completes", device, || {
        half = half_precision_backward_completes(device);
    }));
    failures.extend(half);
    let mut fixed = Vec::new();
    failures.extend(caught(
        "fixed_ops_survive_non_finite_inputs",
        device,
        || {
            fixed = fixed_ops_survive_non_finite_inputs(device);
        },
    ));
    failures.extend(fixed);
    failures
}

#[test]
fn leaves_of_every_rank_and_dtype_cpu() {
    leaves_of_every_rank_and_dtype(DeviceId::Cpu);
}

#[test]
fn misuse_is_an_error_cpu() {
    misuse_is_an_error(DeviceId::Cpu);
}

#[test]
fn detach_and_gradient_flow_cpu() {
    detach_and_gradient_flow(DeviceId::Cpu);
}

#[test]
fn mixing_and_index_checks_cpu() {
    mixing_and_index_checks(DeviceId::Cpu, DeviceId::Cpu);
}

#[test]
fn non_differentiable_ops_record_nothing_cpu() {
    non_differentiable_ops_record_nothing(DeviceId::Cpu);
}

#[test]
fn gradients_match_their_formulas_cpu() {
    let failures = gradients_match_their_formulas(DeviceId::Cpu);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn nn_gradients_match_their_formulas_cpu() {
    let failures = nn_gradients_match_their_formulas(DeviceId::Cpu);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn half_precision_backward_completes_cpu() {
    let failures = half_precision_backward_completes(DeviceId::Cpu);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn fixed_ops_survive_non_finite_inputs_cpu() {
    let failures = fixed_ops_survive_non_finite_inputs(DeviceId::Cpu);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every test on every GPU device, with `to_device` between it and the CPU. Runs every check on
/// every device and fails once at the end with the full list (`SUMMARY autodiff <device>: …`
/// per device), so that one failure can't hide the rest.
///
/// Lavapipe (a software rasterizer) only with `POLYPOUS_TEST_LAVAPIPE_AUTODIFF=1` as well: there
/// this test crashes the process with SIGSEGV in about 1 of 4 runs, in `select`'s backward. That is
/// a Mesa (lavapipe) bug, not a Burn or CubeCL one, and hardware GPUs aren't affected (#92,
/// docs/status.md §4; reproductions in `src/tensor/autodiff_tests.rs`). The skip stays as a
/// documented limitation (owner decision, 2026-11-20). The gradient reference tests still run on
/// lavapipe.
#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
#[test]
#[ignore = "gpu"]
fn autodiff_gpu() {
    use polypous_backend::device::{BackendKind, EnumerateOptions, list_devices};
    let flag = |name: &str| std::env::var(name).is_ok_and(|v| v == "1");
    let options = EnumerateOptions {
        software_rasterizers: flag("POLYPOUS_TEST_SOFTWARE_VULKAN"),
    };
    let hardware_only = EnumerateOptions {
        software_rasterizers: false,
    };
    let mut devices = Vec::new();
    for id in list_devices() {
        if id.backend() == BackendKind::Cpu || options.skips(id) {
            continue;
        }
        if hardware_only.skips(id) && !flag("POLYPOUS_TEST_LAVAPIPE_AUTODIFF") {
            eprintln!(
                "SKIP autodiff_gpu on {id}: software rasterizer; a Mesa lavapipe bug crashes \
                 this test intermittently (#92, docs/status.md §4). Set \
                 POLYPOUS_TEST_LAVAPIPE_AUTODIFF=1 to run it"
            );
            continue;
        }
        devices.push(id);
    }
    if devices.is_empty() {
        eprintln!("SKIP autodiff_gpu: no GPU device to test");
    }
    let mut failures = Vec::new();
    for device in devices {
        let found = all(device, DeviceId::Cpu);
        if found.is_empty() {
            eprintln!("SUMMARY autodiff {device}: passed");
        } else {
            eprintln!("SUMMARY autodiff {device}: {} failed", found.len());
        }
        failures.extend(found);
    }
    assert!(
        failures.is_empty(),
        "{} autodiff check(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
