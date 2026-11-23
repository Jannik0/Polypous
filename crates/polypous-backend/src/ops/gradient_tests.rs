//! Tests of [`with_gradient`] on the CPU (`flex`): the forward value stays bit for bit except NaN
//! payloads, and the gradient is `g · d` for finite, infinite and NaN `d`.
#![cfg(feature = "cpu")]

use burn::tensor::{Device, Tensor, TensorData};

use super::*;
use crate::device::DeviceId;
use crate::tensor::resolve;

fn device() -> Device {
    resolve(DeviceId::Cpu).unwrap()
}

fn t(values: &[f32]) -> Tensor<1> {
    Tensor::from_data(TensorData::new(values.to_vec(), [values.len()]), &device())
}

fn read(t: Tensor<1>) -> Vec<f32> {
    t.without_autodiff()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap()
}

/// Whether two f32 slices hold the same values by testing-strategy.md §3.2's exact rule: every
/// element bit for bit (`-0` and `+0` differ), except that any NaN matches any NaN (NaN payloads
/// aren't part of op semantics; the GPUs return their canonical NaN `0x7FFFFFFF`).
fn same_values(expected: &[f32], actual: &[f32]) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(e, a)| e.to_bits() == a.to_bits() || (e.is_nan() && a.is_nan()))
}

/// `∂/∂x` of `sum(with_gradient(out, x, d) · g)`.
fn gradient(out: &[f32], x: &[f32], d: &[f32], g: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let leaf = t(x).autodiff().require_grad();
    let y = with_gradient(t(out).autodiff(), [(leaf.clone(), t(d))]);
    let forward = read(y.clone());
    let grads = (y * t(g).autodiff()).sum().backward();
    (forward, read(leaf.grad(&grads).unwrap()))
}

#[test]
fn the_forward_value_is_kept_bit_for_bit() {
    let out = [
        -0.0,
        0.0,
        1.5,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(1), // the smallest subnormal
        f32::MAX,
    ];
    let x = [
        1.0,
        -1.0,
        0.0,
        -0.0,
        2.0,
        f32::NAN,
        f32::INFINITY,
        3.0,
        -4.0,
    ];
    let d = [
        2.0,
        -3.0,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e30,
        -1e-30,
    ];
    let (forward, _) = gradient(&out, &x, &d, &[1.0; 9]);
    assert!(same_values(&out, &forward), "{forward:?}");
    // Two terms (the binary ops) keep it too.
    let a = t(&x).autodiff().require_grad();
    let b = t(&x).autodiff().require_grad();
    let y = with_gradient(t(&out).autodiff(), [(a, t(&d)), (b, t(&d))]);
    assert!(same_values(&out, &read(y)));
}

#[test]
fn the_gradient_is_g_times_d() {
    let x = [1.0f32; 8];
    let out = [0.5f32; 8];
    let d = [
        2.0,
        -3.0,
        0.0,
        -0.0,
        0.1,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for g in [1.5f32, -2.0] {
        let (_, grad) = gradient(&out, &x, &d, &[g; 8]);
        for i in 0..5 {
            assert_eq!(grad[i], g * d[i], "d = {}, g = {g}", d[i]);
        }
        assert_eq!(grad[5], f32::INFINITY * g.signum());
        assert_eq!(grad[6], f32::NEG_INFINITY * g.signum());
        assert!(grad[7].is_nan());
    }
}

#[test]
fn non_finite_inputs_get_no_nan_in_the_forward_pass() {
    // x = ±inf or NaN: `x - x` would be NaN; the forward value stays, the gradient is 0.
    let x = [f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let (forward, grad) = gradient(&[1.0, -0.0, 3.0], &x, &[2.0, 2.0, 2.0], &[1.0; 3]);
    assert!(same_values(&[1.0, -0.0, 3.0], &forward), "{forward:?}");
    assert_eq!(grad, vec![0.0; 3]);
}

#[test]
fn untracked_inputs_are_skipped() {
    let plain = t(&[1.0, 2.0]);
    assert!(!any_tracked(&[&plain]));
    let y = with_gradient(t(&[3.0, 4.0]), [(plain, t(&[1.0, 1.0]))]);
    assert!(!y.is_tracked());
    assert_eq!(read(y), vec![3.0, 4.0]);
}

#[test]
fn same_values_ignores_only_nan_payloads() {
    let quiet = f32::from_bits(0x7FC0_0000);
    let canonical_gpu = f32::from_bits(0x7FFF_FFFF);
    assert!(same_values(
        &[quiet, 0.0, -0.0],
        &[canonical_gpu, 0.0, -0.0]
    ));
    assert!(same_values(&[quiet], &[-quiet]));
    // Negative checks: signed zeros, one ulp, and NaN against a number all differ.
    assert!(!same_values(&[0.0], &[-0.0]));
    assert!(!same_values(
        &[1.5],
        &[f32::from_bits(1.5f32.to_bits() + 1)]
    ));
    assert!(!same_values(&[quiet], &[1.0]));
    assert!(!same_values(&[1.0], &[quiet]));
    assert!(!same_values(&[1.0, 2.0], &[1.0]));
}

#[test]
fn largest_finite_values() {
    use burn::tensor::{bf16, f16};
    assert_eq!(largest_finite(BurnDType::F32), f64::from(f32::MAX));
    assert_eq!(largest_finite(BurnDType::F16), f16::MAX.to_f64());
    assert_eq!(largest_finite(BurnDType::BF16), bf16::MAX.to_f64());
}

/// [`with_reduced_gradient`] keeps `out` bit for bit (also `-0`, which a plain sum of the `-0`
/// terms could turn into `+0`), and broadcasts `g · d` over the reduced axes.
#[test]
fn reduced_gradient_keeps_out_and_broadcasts() {
    let d2 = |values: &[f32]| -> Tensor<2> {
        Tensor::from_data(TensorData::new(values.to_vec(), [2, 3]), &device())
    };
    let leaf = d2(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        .autodiff()
        .require_grad();
    let d = d2(&[0.5, 0.0, 0.5, f32::INFINITY, -1.0, 2.0]);
    let out: Tensor<2> =
        Tensor::from_data(TensorData::new(vec![-0.0f32, f32::NAN], [2, 1]), &device()).autodiff();
    let y = with_reduced_gradient(out, leaf.clone(), d, &[1]);
    let forward: Vec<f32> = y
        .clone()
        .without_autodiff()
        .into_data()
        .try_to_vec()
        .unwrap();
    assert_eq!(forward[0].to_bits(), (-0.0f32).to_bits());
    assert!(forward[1].is_nan());
    let g: Tensor<2> =
        Tensor::from_data(TensorData::new(vec![2.0f32, 0.5], [2, 1]), &device()).autodiff();
    let grads = (y * g).sum().backward();
    let grad: Vec<f32> = leaf.grad(&grads).unwrap().into_data().try_to_vec().unwrap();
    assert_eq!(grad, [1.0, 0.0, 1.0, f32::INFINITY, -0.5, 1.0]);
    // Untracked `x`: the constant `out`.
    let plain = with_reduced_gradient(
        d2(&[0.0; 6]).autodiff().sum_dim(1),
        d2(&[1.0; 6]),
        d2(&[1.0; 6]),
        &[1],
    );
    assert!(!plain.is_tracked());
}
