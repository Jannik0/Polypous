//! Initial values (`docs/specs/graph-ir.md` §2.4.1): bounds, means and variances of every kind
//! over large samples, and determinism. Thresholds as in testing-strategy.md §3.8: each statistic
//! within 5 standard errors of its expectation. With fixed seeds the tests are deterministic; the
//! thresholds catch wrong formulas (a missing `sqrt(3)`, a wrong gain, a wrong fan), not bad luck.
//! The values against the Python reference are in `init_reference.rs`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_ir::{Fans, Init};
use polypous_train::initial_values;

const SEED: u64 = 42;
const N_UNIFORM: usize = 1 << 20;
const N_NORMAL: usize = 1 << 19;

fn mean_var(v: &[f32]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
    let var = v
        .iter()
        .map(|&x| (f64::from(x) - mean).powi(2))
        .sum::<f64>()
        / n;
    (mean, var)
}

#[track_caller]
fn within_5_se(what: &str, value: f64, expected: f64, se: f64) {
    let z = (value - expected) / se;
    eprintln!(
        "{what}: {value:.6e} (expected {expected:.6e}, {:.2} SE)",
        z.abs()
    );
    assert!(
        z.abs() <= 5.0,
        "{what}: {value} is {z:.2} SE from {expected}"
    );
}

/// Uniform on `[low, high]`: every value inside (after rounding to f32), mean `(low + high) / 2`
/// (SE `w / sqrt(12 n)`), variance `w² / 12` (SE `sqrt((μ4 − σ⁴) / n)`, `μ4 = w⁴ / 80`).
#[track_caller]
fn check_uniform(what: &str, v: &[f32], low: f64, high: f64) {
    let (lo, hi) = (low as f32, high as f32);
    assert!(
        v.iter().all(|&x| (lo..=hi).contains(&x)),
        "{what}: a value is outside [{lo}, {hi}]"
    );
    let n = v.len() as f64;
    let w = high - low;
    let (mean, var) = mean_var(v);
    within_5_se(
        &format!("{what} mean"),
        mean,
        (low + high) / 2.0,
        w / (12.0 * n).sqrt(),
    );
    let sigma2 = w * w / 12.0;
    let mu4 = w.powi(4) / 80.0;
    within_5_se(
        &format!("{what} variance"),
        var,
        sigma2,
        ((mu4 - sigma2 * sigma2) / n).sqrt(),
    );
    // The extremes come close to the bounds (a halved range would fail here).
    let (min, max) = v
        .iter()
        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(
        f64::from(min) - low < w * 1e-4 && high - f64::from(max) < w * 1e-4,
        "{what}: {min} {max}"
    );
}

/// Normal: mean (SE `std / sqrt(n)`) and variance (SE `σ² sqrt(2 / n)`).
#[track_caller]
fn check_normal(what: &str, v: &[f32], mean: f64, std: f64) {
    let n = v.len() as f64;
    let (m, var) = mean_var(v);
    within_5_se(&format!("{what} mean"), m, mean, std / n.sqrt());
    within_5_se(
        &format!("{what} variance"),
        var,
        std * std,
        std * std * (2.0 / n).sqrt(),
    );
}

fn fans(fan_in: u64, fan_out: u64) -> Option<Fans> {
    Some(Fans { fan_in, fan_out })
}

#[test]
fn constant_kinds() {
    for (init, value) in [
        (Init::Zeros, 0.0f32),
        (Init::Ones, 1.0),
        (Init::Constant { value: 0.1 }, 0.1),
        (Init::Constant { value: -0.0 }, -0.0),
    ] {
        let v = initial_values(SEED, "c", &[3, 5], &init, None).unwrap();
        assert_eq!(v.len(), 15);
        assert!(v.iter().all(|x| x.to_bits() == value.to_bits()), "{init:?}");
    }
}

#[test]
fn uniform() {
    let init = Init::Uniform {
        low: -0.3,
        high: 0.7,
    };
    let v = initial_values(SEED, "u", &[N_UNIFORM], &init, None).unwrap();
    check_uniform("uniform", &v, -0.3, 0.7);
    // low = high gives a constant.
    let c = initial_values(
        SEED,
        "u",
        &[8],
        &Init::Uniform {
            low: 2.5,
            high: 2.5,
        },
        None,
    )
    .unwrap();
    assert!(c.iter().all(|&x| x == 2.5));
}

#[test]
fn normal() {
    let init = Init::Normal {
        mean: 0.5,
        std: 2.0,
    };
    let v = initial_values(SEED, "n", &[N_NORMAL], &init, None).unwrap();
    check_normal("normal", &v, 0.5, 2.0);
    // std = 0 gives a constant.
    let c = initial_values(
        SEED,
        "n",
        &[8],
        &Init::Normal {
            mean: -1.0,
            std: 0.0,
        },
        None,
    )
    .unwrap();
    assert!(c.iter().all(|&x| x == -1.0));
}

#[test]
fn kaiming_uniform() {
    // PyTorch's nn.Linear default, a = sqrt(5): bound = 1 / sqrt(fan_in).
    for (a, written) in [(5.0f64.sqrt(), Some(5.0f64.sqrt())), (0.0, None)] {
        let init = Init::KaimingUniform { a: written };
        let v = initial_values(SEED, "k", &[1024, 1024], &init, fans(256, 4096)).unwrap();
        let gain = (2.0 / (1.0 + a * a)).sqrt();
        let bound = 3.0f64.sqrt() * gain / 16.0;
        check_uniform(&format!("kaiming_uniform a={a:.3}"), &v, -bound, bound);
    }
}

#[test]
fn kaiming_normal() {
    let v = initial_values(SEED, "k", &[512, 1024], &Init::KaimingNormal, fans(64, 9)).unwrap();
    // gain = sqrt(2), std = gain / sqrt(64).
    check_normal("kaiming_normal", &v, 0.0, 2.0f64.sqrt() / 8.0);
}

#[test]
fn xavier_uniform() {
    let v = initial_values(
        SEED,
        "x",
        &[N_UNIFORM],
        &Init::XavierUniform,
        fans(100, 300),
    )
    .unwrap();
    let bound = 3.0f64.sqrt() * (2.0f64 / 400.0).sqrt();
    check_uniform("xavier_uniform", &v, -bound, bound);
}

#[test]
fn xavier_normal() {
    let v = initial_values(SEED, "x", &[N_NORMAL], &Init::XavierNormal, fans(30, 70)).unwrap();
    check_normal("xavier_normal", &v, 0.0, (2.0f64 / 100.0).sqrt());
}

#[test]
fn values_are_deterministic_and_independent() {
    let init = Init::Normal {
        mean: 0.0,
        std: 1.0,
    };
    let get =
        |seed, name: &str, shape: &[usize]| initial_values(seed, name, shape, &init, None).unwrap();
    let a = get(SEED, "layers.0.w", &[4, 6]);
    // Identical for a fixed seed, in any order and next to any other param.
    let b = get(SEED, "layers.1.w", &[100]);
    assert_eq!(get(SEED, "layers.0.w", &[4, 6]), a);
    assert_eq!(get(SEED, "layers.1.w", &[100]), b);
    // Element i depends only on i: the shape doesn't change the values.
    assert_eq!(get(SEED, "layers.0.w", &[24]), a);
    assert_eq!(get(SEED, "layers.0.w", &[2, 3, 4])[..], a[..]);
    assert_eq!(get(SEED, "layers.0.w", &[50])[..24], a[..]);
    // Another name or seed is another stream.
    assert_ne!(get(SEED, "layers.1.w", &[24]), a);
    assert_ne!(get(SEED + 1, "layers.0.w", &[24]), a);
    // The fan-based kinds use the same streams.
    let u = initial_values(
        SEED,
        "w",
        &[16],
        &Init::KaimingUniform { a: None },
        fans(4, 4),
    )
    .unwrap();
    let w = initial_values(
        SEED,
        "w",
        &[16],
        &Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        None,
    )
    .unwrap();
    let bound = 3.0f64.sqrt() * (2.0f64).sqrt() / 2.0;
    for (x, y) in u.iter().zip(&w) {
        // kaiming_uniform is uniform(-bound, bound): the same U(i), scaled.
        assert!((f64::from(*x) - bound * f64::from(*y)).abs() < 1e-6);
    }
}
