//! Tests of the generator against graph-ir.md §3.1.12's known values (bit-exact; `N` within one
//! `f64` ulp, its `ln` and `cos` being the platform's) and the statistical tests of
//! testing-strategy.md §3.8 that concern `u24`, dropout and `N`, with their thresholds. Each
//! statistic is printed (`STAT …`, `--nocapture`) next to the Python value of §3.8's table.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

const DROP_LABEL: &str = "dropout:layers[3]/drop";

#[test]
fn known_values() {
    assert_eq!(draw(0, 0), 0xE220_A839_7B1D_CDAF);
    assert_eq!(fnv1a("a"), 0xAF63_DC4C_8601_EC8C);
    let key = stream_key(42, 7, DROP_LABEL);
    assert_eq!(key, 0x559F_E239_6A2B_E18C);
    let draws: Vec<u64> = (0..4).map(|i| draw(key, i)).collect();
    assert_eq!(
        draws,
        [
            0xDEAF_BD20_49BB_A567,
            0x97B9_C0C7_F5A3_5D08,
            0xC5DA_3B22_6895_E44F,
            0x0940_7FF2_0490_E439
        ]
    );
    let uniform: Vec<f64> = draws.iter().map(|&z| u24(z)).collect();
    assert_eq!(
        uniform,
        [
            0.869_869_053_363_8,
            0.592_678_070_068_359_4,
            0.772_861_182_689_666_7,
            0.036_140_382_289_886_475
        ]
    );
    // The u24 values are exact in f32.
    for &u in &uniform {
        #[allow(clippy::cast_possible_truncation)]
        let narrow = f64::from(u as f32);
        assert_eq!(narrow, u);
    }

    let init = stream_key(42, 0, "init:fc1.weight");
    assert_eq!(init, 0x4A64_D327_A844_95D0);
    let uniform: Vec<f64> = (0..3).map(|i| u53(draw(init, i))).collect();
    assert_eq!(
        uniform,
        [
            0.941_849_421_737_909_3,
            0.952_777_646_315_502,
            0.218_862_624_243_094_22
        ]
    );
    let expected: [f64; 3] = [
        0.331_024_684_653_918_16,
        1.195_187_756_646_760_8,
        0.224_155_764_401_435_8,
    ];
    for (i, want) in expected.into_iter().enumerate() {
        let got = normal(init, i as u64);
        let ulps = got.to_bits().abs_diff(want.to_bits());
        assert!(
            ulps <= 1,
            "N({i}): {got:e}, expected {want:e} ({ulps} ulps)"
        );
    }
}

#[test]
fn dropout_masks_follow_the_rule() {
    // Known values: u24 = 0.87, 0.59, 0.77, 0.036 for "layers[3]/drop", seed 42, step 7.
    let mask = dropout_mask(42, 7, "layers[3]/drop", &[4], 0.5).unwrap();
    assert_eq!(mask, [true, true, true, false]);
    let mask = dropout_mask(42, 7, "layers[3]/drop", &[2, 2], 0.8).unwrap();
    assert_eq!(mask, [true, false, false, false]);
    // The comparison is ≥: p equal to a grid value keeps that element.
    let p = 0.592_678_070_068_359_4;
    assert_eq!(
        dropout_mask(42, 7, "layers[3]/drop", &[4], p).unwrap(),
        [true, true, true, false]
    );
    // Row-major index at the shape: element i is draw i, for every rank.
    let key = stream_key(42, 7, DROP_LABEL);
    let mask = dropout_mask(42, 7, "layers[3]/drop", &[3, 1, 5, 2], 0.3).unwrap();
    assert_eq!(mask.len(), 30);
    for (i, &kept) in mask.iter().enumerate() {
        assert_eq!(kept, dropout_keep(key, i as u64, 0.3), "element {i}");
    }
    // p = 0 keeps everything; a scalar has one element.
    assert_eq!(dropout_mask(1, 2, "d", &[2, 3], 0.0).unwrap(), [true; 6]);
    assert_eq!(dropout_mask(1, 2, "d", &[], 0.0).unwrap(), [true]);
    assert_eq!(
        dropout_mask(1, 2, "d", &[0, 4], 0.5).unwrap(),
        Vec::<bool>::new()
    );
}

#[test]
fn dropout_mask_rejects_bad_arguments() {
    for p in [1.0, 1.5, -0.1, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                dropout_mask(0, 0, "d", &[2], p),
                Err(RngError::InvalidProbability { .. })
            ),
            "p = {p}"
        );
    }
    let huge = [usize::MAX, 2];
    assert_eq!(
        dropout_mask(0, 0, "d", &huge, 0.5),
        Err(RngError::TooLarge {
            shape: huge.to_vec()
        })
    );
    // Fits usize, but can't be allocated: an error, not an abort.
    let unallocatable = [usize::MAX / 2];
    assert_eq!(
        dropout_mask(0, 0, "d", &unallocatable, 0.5),
        Err(RngError::TooLarge {
            shape: unallocatable.to_vec()
        })
    );
}

/// The masks of two node paths, and of two steps of one node path, differ.
#[test]
fn dropout_masks_of_different_streams_differ() {
    let shape = [64, 64];
    let base = dropout_mask(42, 7, "layers[3]/drop", &shape, 0.5).unwrap();
    let other_path = dropout_mask(42, 7, "layers[4]/drop", &shape, 0.5).unwrap();
    let other_step = dropout_mask(42, 8, "layers[3]/drop", &shape, 0.5).unwrap();
    let other_seed = dropout_mask(43, 7, "layers[3]/drop", &shape, 0.5).unwrap();
    assert_ne!(base, other_path);
    assert_ne!(base, other_step);
    assert_ne!(base, other_seed);
    // The same stream gives the same mask.
    assert_eq!(
        base,
        dropout_mask(42, 7, "layers[3]/drop", &shape, 0.5).unwrap()
    );
    // A dropout label never equals an init label (`:` in neither paths nor names).
    assert_ne!(
        stream_key(42, 0, "dropout:fc1.weight"),
        stream_key(42, 0, "init:fc1.weight")
    );
}

/// 2^20 draws of `stream_key(42, 7, "dropout:layers[3]/drop")`.
fn sample() -> Vec<f64> {
    let key = stream_key(42, 7, DROP_LABEL);
    (0..1u64 << 20).map(|i| u24(draw(key, i))).collect()
}

/// `|value - expected| / se`, printed and checked against 5 standard errors.
fn within_5_se(name: &str, value: f64, expected: f64, se: f64, python: &str) {
    let z = (value - expected).abs() / se;
    eprintln!("STAT {name}: {value:.6} ({z:.2} SE; Python {python})");
    assert!(z <= 5.0, "{name}: {value} is {z:.2} SE from {expected}");
}

#[test]
fn u24_is_uniform() {
    let values = sample();
    let n = values.len() as f64;
    let mut buckets = [0u64; 256];
    for &u in &values {
        // u < 1, so the bucket is below 256.
        buckets[(u * 256.0) as usize] += 1;
    }
    let expected = n / 256.0;
    let chi2: f64 = buckets
        .iter()
        .map(|&c| (c as f64 - expected).powi(2) / expected)
        .sum();
    let bound = 255.0 + 5.0 * (2.0f64 * 255.0).sqrt();
    eprintln!("STAT u24 chi-square, 256 buckets: {chi2:.1} (≤ {bound:.1}; Python 241.8)");
    assert!(chi2 <= 367.9 && chi2 <= bound, "chi-square {chi2}");

    let mean = values.iter().sum::<f64>() / n;
    within_5_se("u24 mean", mean, 0.5, (1.0 / 12.0 / n).sqrt(), "0.73 SE");
}

#[test]
fn dropout_keep_rate_and_independence() {
    let key = stream_key(42, 7, DROP_LABEL);
    let count = 1u64 << 20;
    let n = count as f64;
    for p in [0.1, 0.25, 0.5, 0.9] {
        let kept = (0..count).filter(|&i| dropout_keep(key, i, p)).count() as f64;
        within_5_se(
            &format!("dropout keep rate, p = {p}"),
            kept / n,
            1.0 - p,
            (p * (1.0 - p) / n).sqrt(),
            "≤ 0.58 SE",
        );
    }
    // The same node at steps 7 and 8, p = 0.5: kept in both a quarter of the time.
    let step8 = stream_key(42, 8, DROP_LABEL);
    let both = (0..count)
        .filter(|&i| dropout_keep(key, i, 0.5) && dropout_keep(step8, i, 0.5))
        .count() as f64;
    within_5_se(
        "dropout kept at steps 7 and 8, p = 0.5",
        both / n,
        0.25,
        (0.25 * 0.75 / n).sqrt(),
        "0.63 SE",
    );
}

#[test]
fn normal_moments() {
    let key = stream_key(42, 0, "init:fc1.weight");
    let values: Vec<f64> = (0..1u64 << 19).map(|i| normal(key, i)).collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let central = |k: i32| values.iter().map(|v| (v - mean).powi(k)).sum::<f64>() / n;
    let var = central(2);
    let skew = central(3) / var.powf(1.5);
    let kurt = central(4) / (var * var) - 3.0;
    within_5_se("N mean", mean, 0.0, (1.0 / n).sqrt(), "≤ 1.33 SE");
    within_5_se("N variance", var, 1.0, (2.0 / n).sqrt(), "≤ 1.33 SE");
    within_5_se("N skewness", skew, 0.0, (6.0 / n).sqrt(), "≤ 1.33 SE");
    within_5_se(
        "N excess kurtosis",
        kurt,
        0.0,
        (24.0 / n).sqrt(),
        "≤ 1.33 SE",
    );
}
