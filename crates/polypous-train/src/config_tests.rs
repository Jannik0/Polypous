//! Unit tests of the config types: JSON form, defaults, and every range check of training.md §2.8.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

fn optimizer(json: &str) -> OptimizerConfig {
    serde_json::from_str(json).unwrap()
}

fn schedule(json: &str) -> ScheduleConfig {
    serde_json::from_str(json).unwrap()
}

#[track_caller]
fn assert_pointer(result: Result<(), ConfigError>, pointer: &str) {
    let err = result.expect_err("expected a range error");
    assert_eq!(err.pointer, pointer, "{err}");
}

#[test]
fn defaults_are_pytorchs() {
    assert_eq!(
        optimizer(r#"{"kind": "sgd"}"#),
        OptimizerConfig::Sgd(SgdConfig {
            lr: 1e-3,
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            weight_decay: 0.0,
        })
    );
    assert_eq!(
        optimizer(r#"{"kind": "adam"}"#),
        OptimizerConfig::Adam(AdamConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: 0.0,
        })
    );
    assert_eq!(
        optimizer(r#"{"kind": "adamw"}"#),
        OptimizerConfig::Adamw(AdamWConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: 0.01,
        })
    );
    assert_eq!(
        schedule(r#"{"kind": "step", "every": 3}"#),
        ScheduleConfig::Step {
            every: 3,
            gamma: 0.1
        }
    );
    assert_eq!(
        schedule(r#"{"kind": "cosine"}"#),
        ScheduleConfig::Cosine {
            warmup_steps: 0,
            min_lr: 0.0
        }
    );
    assert_eq!(ScheduleConfig::default(), ScheduleConfig::Constant {});
}

#[test]
fn the_spec_example_parses() {
    // training.md §2, the example config.
    let opt = optimizer(
        r#"{ "kind": "adamw", "lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01 }"#,
    );
    assert_eq!(opt, OptimizerConfig::Adamw(AdamWConfig::default()));
    opt.validate().unwrap();
    let s = schedule(r#"{ "kind": "cosine", "warmup_steps": 100, "min_lr": 1e-5 }"#);
    s.validate(opt.lr()).unwrap();
    let clip: GradientClipConfig = serde_json::from_str(r#"{ "max_norm": 1.0 }"#).unwrap();
    clip.validate().unwrap();
}

#[test]
fn json_round_trip_writes_every_field_in_table_order() {
    let text = serde_json::to_string(&OptimizerConfig::Sgd(SgdConfig::default())).unwrap();
    assert_eq!(
        text,
        r#"{"kind":"sgd","lr":0.001,"momentum":0.0,"dampening":0.0,"nesterov":false,"weight_decay":0.0}"#
    );
    let text = serde_json::to_string(&OptimizerConfig::Adam(AdamConfig::default())).unwrap();
    assert_eq!(
        text,
        r#"{"kind":"adam","lr":0.001,"betas":[0.9,0.999],"eps":1e-8,"weight_decay":0.0}"#
    );
    let text = serde_json::to_string(&ScheduleConfig::Cosine {
        warmup_steps: 5,
        min_lr: 0.0,
    })
    .unwrap();
    assert_eq!(text, r#"{"kind":"cosine","warmup_steps":5,"min_lr":0.0}"#);
    assert_eq!(
        serde_json::to_string(&ScheduleConfig::Constant {}).unwrap(),
        r#"{"kind":"constant"}"#
    );
    for s in [
        ScheduleConfig::Constant {},
        ScheduleConfig::Step {
            every: 7,
            gamma: 0.5,
        },
        ScheduleConfig::Cosine {
            warmup_steps: 3,
            min_lr: 1e-5,
        },
        ScheduleConfig::LinearWarmup { warmup_steps: 9 },
    ] {
        assert_eq!(schedule(&serde_json::to_string(&s).unwrap()), s);
    }
    for o in [
        OptimizerConfig::Sgd(SgdConfig::default()),
        OptimizerConfig::Adam(AdamConfig::default()),
        OptimizerConfig::Adamw(AdamWConfig::default()),
    ] {
        assert_eq!(optimizer(&serde_json::to_string(&o).unwrap()), o);
    }
}

#[test]
fn unknown_and_mistyped_fields_are_rejected() {
    // §2.1: a field the selected kind doesn't have is an unknown field.
    for bad in [
        r#"{"kind": "adam", "momentum": 0.9}"#,
        r#"{"kind": "sgd", "betas": [0.9, 0.99]}"#,
        r#"{"kind": "adamw", "amsgrad": true}"#,
        r#"{"kind": "rmsprop"}"#,
        r#"{"lr": 0.1}"#,
        r#"{"kind": "sgd", "lr": "0.1"}"#,
        r#"{"kind": "adam", "betas": [0.9]}"#,
        r#"{"kind": "adam", "betas": [0.9, 0.99, 0.5]}"#,
        r#"{"kind": "sgd", "nesterov": 1}"#,
    ] {
        assert!(
            serde_json::from_str::<OptimizerConfig>(bad).is_err(),
            "accepted {bad}"
        );
    }
    for bad in [
        r#"{"kind": "constant", "gamma": 0.1}"#,
        r#"{"kind": "step"}"#,
        r#"{"kind": "step", "every": 5.0}"#,
        r#"{"kind": "step", "every": -1}"#,
        r#"{"kind": "step", "every": 5, "warmup_steps": 1}"#,
        r#"{"kind": "linear_warmup"}"#,
        r#"{"kind": "cosine", "warmup_steps": 1.5}"#,
        r#"{"kind": "exponential", "gamma": 0.9}"#,
    ] {
        assert!(
            serde_json::from_str::<ScheduleConfig>(bad).is_err(),
            "accepted {bad}"
        );
    }
    for bad in [r#"{}"#, r#"{"max_norm": 1.0, "norm_type": 2}"#] {
        assert!(
            serde_json::from_str::<GradientClipConfig>(bad).is_err(),
            "accepted {bad}"
        );
    }
    // Float fields take JSON integers (§2.1, "Numbers").
    assert_eq!(optimizer(r#"{"kind": "sgd", "lr": 1}"#).lr(), 1.0);
}

#[test]
fn sgd_ranges() {
    let ok = SgdConfig {
        lr: 0.0,
        momentum: 1.0,
        dampening: 1.0,
        nesterov: false,
        weight_decay: 0.0,
    };
    ok.validate().unwrap();
    let with = |f: fn(&mut SgdConfig)| {
        let mut c = ok;
        f(&mut c);
        c.validate()
    };
    assert_pointer(with(|c| c.lr = -1e-3), "/lr");
    assert_pointer(with(|c| c.lr = f64::NAN), "/lr");
    assert_pointer(with(|c| c.lr = 1e300), "/lr"); // rounds to inf in f32
    assert_pointer(with(|c| c.momentum = -0.1), "/momentum");
    assert_pointer(with(|c| c.momentum = 1.5), "/momentum");
    assert_pointer(with(|c| c.dampening = -0.1), "/dampening");
    assert_pointer(with(|c| c.dampening = 1.01), "/dampening");
    assert_pointer(with(|c| c.weight_decay = -1.0), "/weight_decay");
    assert_pointer(with(|c| c.weight_decay = f64::INFINITY), "/weight_decay");
}

#[test]
fn nesterov_needs_momentum_and_no_dampening() {
    let base = SgdConfig {
        lr: 0.1,
        momentum: 0.9,
        dampening: 0.0,
        nesterov: true,
        weight_decay: 0.0,
    };
    base.validate().unwrap();
    let mut c = base;
    c.momentum = 0.0;
    assert_pointer(c.validate(), "/nesterov");
    let mut c = base;
    c.dampening = 0.1;
    assert_pointer(c.validate(), "/nesterov");
}

#[test]
fn adam_ranges() {
    let ok = AdamConfig {
        lr: 0.0,
        betas: [0.0, 0.0],
        eps: 0.0,
        weight_decay: 0.0,
    };
    ok.validate().unwrap();
    let with = |f: fn(&mut AdamConfig)| {
        let mut c = ok;
        f(&mut c);
        c.validate()
    };
    assert_pointer(with(|c| c.lr = -1.0), "/lr");
    assert_pointer(with(|c| c.betas[0] = 1.0), "/betas/0");
    assert_pointer(with(|c| c.betas[0] = -0.1), "/betas/0");
    assert_pointer(with(|c| c.betas[1] = 1.0), "/betas/1");
    assert_pointer(with(|c| c.betas[1] = f64::NAN), "/betas/1");
    assert_pointer(with(|c| c.eps = -1e-8), "/eps");
    assert_pointer(with(|c| c.weight_decay = -0.01), "/weight_decay");
    // AdamW has the same ranges.
    let w = AdamWConfig {
        betas: [0.9, 1.0],
        ..AdamWConfig::default()
    };
    assert_pointer(w.validate(), "/betas/1");
    let w = AdamWConfig {
        weight_decay: -1.0,
        ..AdamWConfig::default()
    };
    assert_pointer(OptimizerConfig::Adamw(w).validate(), "/weight_decay");
}

#[test]
fn schedule_ranges() {
    let lr = 0.1;
    ScheduleConfig::Constant {}.validate(lr).unwrap();
    let step = |every, gamma| ScheduleConfig::Step { every, gamma }.validate(lr);
    step(1, 0.0).unwrap();
    step(MAX_SCHEDULE_STEPS, 1.0).unwrap();
    assert_pointer(step(0, 0.1), "/every");
    assert_pointer(step(MAX_SCHEDULE_STEPS + 1, 0.1), "/every");
    assert_pointer(step(5, -0.1), "/gamma");
    assert_pointer(step(5, 1.5), "/gamma");
    assert_pointer(step(5, f64::NAN), "/gamma");
    let cosine = |warmup_steps, min_lr| {
        ScheduleConfig::Cosine {
            warmup_steps,
            min_lr,
        }
        .validate(lr)
    };
    cosine(0, 0.0).unwrap();
    cosine(MAX_SCHEDULE_STEPS, lr).unwrap();
    assert_pointer(cosine(MAX_SCHEDULE_STEPS + 1, 0.0), "/warmup_steps");
    assert_pointer(cosine(0, -1e-5), "/min_lr");
    assert_pointer(cosine(0, f64::NAN), "/min_lr");
    assert_pointer(cosine(0, 0.2), "/min_lr"); // min_lr > lr: the cosine would rise
    let warmup = |warmup_steps| ScheduleConfig::LinearWarmup { warmup_steps }.validate(lr);
    warmup(1).unwrap();
    warmup(MAX_SCHEDULE_STEPS).unwrap();
    assert_pointer(warmup(0), "/warmup_steps");
    assert_pointer(warmup(MAX_SCHEDULE_STEPS + 1), "/warmup_steps");
}

#[test]
fn gradient_clip_ranges() {
    GradientClipConfig { max_norm: 1.0 }.validate().unwrap();
    GradientClipConfig { max_norm: 1e-30 }.validate().unwrap();
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e300] {
        assert_pointer(GradientClipConfig { max_norm: bad }.validate(), "/max_norm");
    }
}
