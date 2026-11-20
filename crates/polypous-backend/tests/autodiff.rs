//! Tests of the autodiff API ([`DynTensor::require_grad`], [`DynTensor::backward`],
//! [`Gradients`], `detach`, `without_autodiff`): every misuse is an error, gradients have the
//! leaf's dtype, shape and device, and the ops that fix Burn's gradients (graph-ir.md §3.1.11)
//! never panic on non-finite inputs. The CPU tests use `flex`; the GPU test is
//! `#[ignore = "gpu"]` and runs on every GPU device (lavapipe only with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1`).
//!
//! An integration test of its own, so that its GPU work doesn't run in parallel with the
//! backend's other GPU unit tests in one process: on lavapipe that combination crashed with
//! SIGSEGV inside lavapipe's kernel code (#75, docs/status.md §4).
#![cfg(feature = "cpu")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

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

fn bits(t: &DynTensor) -> Vec<u32> {
    read(t).iter().map(|v| v.to_bits()).collect()
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
    assert_eq!(bits(&x.clone().detach()), bits(&x));
    assert!(!leaf.clone().detach().is_tracked());
    assert!(leaf.clone().detach().has_autodiff());
    let plain = leaf.clone().without_autodiff();
    assert!(!plain.is_tracked() && !plain.has_autodiff());
    assert_eq!(bits(&plain), bits(&x));
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
/// for bit.
fn fixed_ops_survive_non_finite_inputs(device: DeviceId) {
    let (inf, nan) = (f32::INFINITY, f32::NAN);
    let a = [nan, inf, -inf, 0.0, -0.0, 1.0, -1.5, nan, inf, 2.0];
    let b = [1.0, inf, inf, -0.0, 0.0, nan, -inf, nan, -inf, 0.5];
    type Binary = fn(DynTensor, DynTensor) -> Result<DynTensor>;
    let binary: [(&str, Binary); 3] = [
        ("maximum", DynTensor::maximum),
        ("minimum", DynTensor::minimum),
        ("pow", DynTensor::pow),
    ];
    for (name, op) in binary {
        let (x, y) = (f32s(&a, &[10], device), f32s(&b, &[10], device));
        let plain = op(x.clone(), y.clone()).unwrap();
        let (lx, ly) = (x.require_grad().unwrap(), y.require_grad().unwrap());
        let out = op(lx.clone(), ly.clone()).unwrap();
        assert_eq!(bits(&out), bits(&plain), "{name} on {device}");
        let grads = total(out).backward().unwrap();
        assert!(grads.get(&lx).unwrap().is_some(), "{name}");
        assert!(grads.get(&ly).unwrap().is_some(), "{name}");
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
        let x = f32s(&a, &[10], device);
        let plain = op(x.clone()).unwrap();
        let leaf = x.require_grad().unwrap();
        let out = op(leaf.clone()).unwrap();
        assert_eq!(bits(&out), bits(&plain), "{name} on {device}");
        assert!(out.is_tracked(), "{name}");
        let grads = total(out).backward().unwrap();
        assert!(grads.get(&leaf).unwrap().is_some(), "{name}");
    }
}

#[cfg(any(feature = "cuda", feature = "vulkan", feature = "metal"))]
fn all(device: DeviceId, other: DeviceId) {
    leaves_of_every_rank_and_dtype(device);
    misuse_is_an_error(device);
    detach_and_gradient_flow(device);
    mixing_and_index_checks(device, other);
    fixed_ops_survive_non_finite_inputs(device);
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
fn fixed_ops_survive_non_finite_inputs_cpu() {
    fixed_ops_survive_non_finite_inputs(DeviceId::Cpu);
}

/// Every test on every GPU device, with `to_device` between it and the CPU.
///
/// Lavapipe (a software rasterizer) only with `POLYPOUS_TEST_LAVAPIPE_AUTODIFF=1` as well: there
/// this test crashed the process with SIGSEGV inside lavapipe's kernel code in about 1 of 3 runs
/// after the CPU tests of this file (each part alone passed 10 of 10), a Burn 0.22.1 / CubeCL
/// finding with a standalone reproduction (`lavapipe_where_select_backward_crash` in
/// `src/tensor/autodiff_tests.rs`, docs/status.md §4). The gradient reference tests still run on
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
                "SKIP autodiff_gpu on {id}: software rasterizer; the test crashes lavapipe \
                 intermittently (docs/status.md §4). Set POLYPOUS_TEST_LAVAPIPE_AUTODIFF=1 to run it"
            );
            continue;
        }
        devices.push(id);
    }
    if devices.is_empty() {
        eprintln!("SKIP autodiff_gpu: no GPU device to test");
    }
    for device in devices {
        all(device, DeviceId::Cpu);
        mixing_and_index_checks(DeviceId::Cpu, device);
        eprintln!("SUMMARY autodiff {device}: passed");
    }
}
