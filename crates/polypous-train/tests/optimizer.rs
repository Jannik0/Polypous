//! Optimizer behavior beyond the reference values (training.md §1.1, §3.1, §5.2, §5.4): state
//! export and import (bit-identical continuation on the CPU), the checks of an import, params
//! without a gradient, `lr = 0`, errors, and that a failed step changes nothing.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod support;

use polypous_backend::device::DeviceId;
use polypous_backend::{DType, DynTensor};
use polypous_train::optimizer::{EXP_AVG, EXP_AVG_SQ, MOMENTUM_BUFFER};
use polypous_train::{
    AdamConfig, AdamWConfig, GradientClipConfig, Optimizer, OptimizerConfig, OptimizerError,
    SgdConfig, StateTensor, TensorMap, clip_gradients,
};
use support::{f32_tensor, read_f32, values};

const CPU: DeviceId = DeviceId::Cpu;

fn configs() -> Vec<(&'static str, OptimizerConfig)> {
    vec![
        (
            "sgd_momentum",
            OptimizerConfig::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.1,
                weight_decay: 0.01,
                ..SgdConfig::default()
            }),
        ),
        (
            "sgd_nesterov",
            OptimizerConfig::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                nesterov: true,
                ..SgdConfig::default()
            }),
        ),
        (
            "sgd_plain",
            OptimizerConfig::Sgd(SgdConfig {
                lr: 0.1,
                weight_decay: 0.01,
                ..SgdConfig::default()
            }),
        ),
        (
            "adam",
            OptimizerConfig::Adam(AdamConfig {
                lr: 0.01,
                weight_decay: 0.01,
                ..AdamConfig::default()
            }),
        ),
        (
            "adamw",
            OptimizerConfig::Adamw(AdamWConfig {
                lr: 0.01,
                betas: [0.3, 0.99],
                ..AdamWConfig::default()
            }),
        ),
    ]
}

/// Two params of different shapes.
fn params(device: DeviceId) -> TensorMap {
    let mut p = TensorMap::new();
    p.insert(
        "fc.weight".into(),
        f32_tensor(&[3, 4], &values(12, 1, 2.0), device),
    );
    p.insert(
        "fc.bias".into(),
        f32_tensor(&[3], &values(3, 2, 2.0), device),
    );
    p
}

/// The gradients of step `k`.
fn grads(k: u64, device: DeviceId) -> TensorMap {
    let mut g = TensorMap::new();
    g.insert(
        "fc.weight".into(),
        f32_tensor(&[3, 4], &values(12, 100 + k, 1.0), device),
    );
    g.insert(
        "fc.bias".into(),
        f32_tensor(&[3], &values(3, 200 + k, 1.0), device),
    );
    g
}

fn bits(t: &DynTensor) -> Vec<u32> {
    read_f32(t).iter().map(|v| v.to_bits()).collect()
}

fn map_bits(m: &TensorMap) -> Vec<(String, Vec<u32>)> {
    m.iter().map(|(k, v)| (k.clone(), bits(v))).collect()
}

fn state_bits(opt: &Optimizer) -> Vec<(String, String, Vec<u32>)> {
    opt.export_state()
        .into_iter()
        .map(|s| (s.param, s.state, bits(&s.tensor)))
        .collect()
}

/// A copy through host bytes, as a checkpoint file would give it.
fn through_host(t: &DynTensor) -> DynTensor {
    DynTensor::from_data(&t.to_data().unwrap(), t.device()).unwrap()
}

#[test]
fn export_import_continues_bit_identically() {
    for (name, config) in configs() {
        let lr = config.lr();
        // Uninterrupted: steps 0, 1, 2.
        let mut opt = Optimizer::new(config.clone()).unwrap();
        let mut p = params(CPU);
        for k in 0..3 {
            opt.step(k, lr, &mut p, &grads(k, CPU)).unwrap();
        }
        // Interrupted after step 0 (and, separately, after step 1): export, a new optimizer, import.
        for stop in 1..3 {
            let mut first = Optimizer::new(config.clone()).unwrap();
            let mut q = params(CPU);
            for k in 0..stop {
                first.step(k, lr, &mut q, &grads(k, CPU)).unwrap();
            }
            let exported: Vec<StateTensor> = first
                .export_state()
                .into_iter()
                .map(|s| StateTensor {
                    tensor: through_host(&s.tensor),
                    ..s
                })
                .collect();
            let names: Vec<&str> = exported.iter().map(|s| s.state.as_str()).collect();
            match &config {
                OptimizerConfig::Sgd(c) if c.momentum == 0.0 => assert!(names.is_empty()),
                OptimizerConfig::Sgd(_) => assert_eq!(names, [MOMENTUM_BUFFER; 2]),
                _ => assert_eq!(names, [EXP_AVG, EXP_AVG_SQ, EXP_AVG, EXP_AVG_SQ]),
            }
            let mut q: TensorMap = q
                .iter()
                .map(|(k, v)| (k.clone(), through_host(v)))
                .collect();
            let mut second = Optimizer::new(config.clone()).unwrap();
            second.import_state(&q, exported).unwrap();
            assert_eq!(state_bits(&second), state_bits(&first), "{name}: import");
            for k in stop..3 {
                second.step(k, lr, &mut q, &grads(k, CPU)).unwrap();
            }
            assert_eq!(
                map_bits(&q),
                map_bits(&p),
                "{name}: params after resume at {stop}"
            );
            assert_eq!(
                state_bits(&second),
                state_bits(&opt),
                "{name}: state after resume at {stop}"
            );
        }
    }
}

fn adam_after_one_step() -> (Optimizer, TensorMap) {
    let mut opt = Optimizer::new(OptimizerConfig::Adam(AdamConfig::default())).unwrap();
    let mut p = params(CPU);
    opt.step(0, 1e-3, &mut p, &grads(0, CPU)).unwrap();
    (opt, p)
}

#[track_caller]
fn assert_invalid_state(result: Result<(), OptimizerError>, name: &str) {
    match result {
        Err(OptimizerError::InvalidState { name: got, reason }) => {
            assert_eq!(got, name, "{reason}");
        }
        other => panic!("expected InvalidState for {name}, got {other:?}"),
    }
}

#[test]
fn import_checks_names_shapes_dtypes_and_completeness() {
    let (mut opt, p) = adam_after_one_step();
    let before = state_bits(&opt);
    let good = opt.export_state();
    let with = |f: &dyn Fn(&mut Vec<StateTensor>)| {
        let mut e = good.clone();
        f(&mut e);
        e
    };
    // An unknown state name, and one of another optimizer.
    let e = with(&|e| e[0].state = "max_exp_avg_sq".into());
    assert_invalid_state(opt.import_state(&p, e), "fc.bias:max_exp_avg_sq");
    let e = with(&|e| e[0].state = MOMENTUM_BUFFER.into());
    assert_invalid_state(opt.import_state(&p, e), "fc.bias:momentum_buffer");
    // A param that isn't trainable (not in the map).
    let e = with(&|e| e[0].param = "fc.other".into());
    assert_invalid_state(opt.import_state(&p, e), "fc.other:exp_avg");
    // A wrong shape: the bias's state on the weight.
    let e = with(&|e| e[0].param = "fc.weight".into());
    assert_invalid_state(opt.import_state(&p, e), "fc.weight:exp_avg");
    // A wrong dtype.
    let e = with(&|e| e[1].tensor = e[1].tensor.clone().cast(DType::F16).unwrap());
    assert_invalid_state(opt.import_state(&p, e), "fc.bias:exp_avg_sq");
    // Twice the same name.
    let e = with(&|e| {
        let dup = e[0].clone();
        e.push(dup);
    });
    assert_invalid_state(opt.import_state(&p, e), "fc.bias:exp_avg");
    // Only some of a param's state.
    let e = with(&|e| {
        e.remove(1);
    });
    assert_invalid_state(opt.import_state(&p, e), "fc.bias");
    // None of the failures changed the state.
    assert_eq!(state_bits(&opt), before);
    // SGD without momentum has no state at all.
    let mut sgd = Optimizer::new(OptimizerConfig::Sgd(SgdConfig::default())).unwrap();
    assert!(sgd.state_names().is_empty());
    assert_invalid_state(sgd.import_state(&p, good.clone()), "fc.bias:exp_avg");
    sgd.import_state(&p, Vec::new()).unwrap();
    // A param with no state at all is fine at import (and an error at its first step k > 0).
    let only_weight: Vec<StateTensor> = good
        .into_iter()
        .filter(|s| s.param == "fc.weight")
        .collect();
    opt.import_state(&p, only_weight).unwrap();
    assert!(opt.has_state("fc.weight") && !opt.has_state("fc.bias"));
    let mut q = p.clone();
    match opt.step(1, 1e-3, &mut q, &grads(1, CPU)) {
        Err(OptimizerError::MissingState { name, step: 1 }) => assert_eq!(name, "fc.bias"),
        other => panic!("expected MissingState, got {other:?}"),
    }
    assert_eq!(map_bits(&q), map_bits(&p), "a failed step changes no param");
}

#[test]
fn a_param_without_a_gradient_is_untouched() {
    for (name, config) in configs() {
        let lr = config.lr();
        let mut opt = Optimizer::new(config).unwrap();
        let mut p = params(CPU);
        let bias_before = bits(&p["fc.bias"]);
        for k in 0..3 {
            let mut g = grads(k, CPU);
            g.remove("fc.bias");
            opt.step(k, lr, &mut p, &g).unwrap();
        }
        assert_eq!(bits(&p["fc.bias"]), bias_before, "{name}: value");
        assert!(!opt.has_state("fc.bias"), "{name}: no state");
        assert!(
            opt.export_state().iter().all(|s| s.param == "fc.weight"),
            "{name}: exported state"
        );
        assert_ne!(bits(&p["fc.weight"]), bits(&params(CPU)["fc.weight"]));
    }
}

#[test]
fn zero_lr_leaves_params_unchanged_and_updates_state() {
    for (name, config) in configs() {
        let mut opt = Optimizer::new(config.clone()).unwrap();
        let mut p = params(CPU);
        for k in 0..3 {
            opt.step(k, 0.0, &mut p, &grads(k, CPU)).unwrap();
        }
        // Compared as values: p + (−0) · x keeps every non-zero p bit for bit (none is ±0 here).
        assert_eq!(
            read_f32(&p["fc.weight"]),
            read_f32(&params(CPU)["fc.weight"]),
            "{name}"
        );
        assert_eq!(
            read_f32(&p["fc.bias"]),
            read_f32(&params(CPU)["fc.bias"]),
            "{name}"
        );
        // The state still follows the gradients.
        let state = opt.export_state();
        match config {
            OptimizerConfig::Sgd(c) if c.momentum == 0.0 => assert!(state.is_empty()),
            _ => {
                assert!(!state.is_empty(), "{name}");
                for s in state {
                    assert!(
                        read_f32(&s.tensor).iter().any(|&v| v != 0.0),
                        "{name}: {}",
                        s.state
                    );
                }
            }
        }
    }
}

#[test]
fn step_errors_change_nothing() {
    let config = OptimizerConfig::Adamw(AdamWConfig::default());
    let mut opt = Optimizer::new(config).unwrap();
    let mut p = params(CPU);
    opt.step(0, 1e-3, &mut p, &grads(0, CPU)).unwrap();
    let (p_before, s_before) = (map_bits(&p), state_bits(&opt));
    let check = |opt: &mut Optimizer, p: &mut TensorMap, g: TensorMap, lr: f64| {
        let err = opt.step(1, lr, p, &g).expect_err("expected an error");
        assert_eq!(map_bits(p), p_before, "{err}");
        assert_eq!(state_bits(opt), s_before, "{err}");
        err
    };
    // A gradient of a name that isn't a param. (`fc.bias` sorts first and is valid.)
    let mut g = grads(1, CPU);
    g.insert("zz".into(), f32_tensor(&[3], &[1.0; 3], CPU));
    assert!(matches!(
        check(&mut opt, &mut p, g, 1e-3),
        OptimizerError::UnknownParam { name } if name == "zz"
    ));
    // A gradient with the wrong shape.
    let mut g = grads(1, CPU);
    g.insert("fc.weight".into(), f32_tensor(&[4, 3], &[1.0; 12], CPU));
    assert!(matches!(
        check(&mut opt, &mut p, g, 1e-3),
        OptimizerError::Mismatch { name, .. } if name == "fc.weight"
    ));
    // A gradient that isn't f32.
    let mut g = grads(1, CPU);
    let half = g["fc.weight"].clone().cast(DType::F16).unwrap();
    g.insert("fc.weight".into(), half);
    assert!(matches!(
        check(&mut opt, &mut p, g, 1e-3),
        OptimizerError::Mismatch { .. }
    ));
    // Learning rates out of range.
    for lr in [-1e-3, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            check(&mut opt, &mut p, grads(1, CPU), lr),
            OptimizerError::LearningRate { .. }
        ));
    }
    // An invalid config never builds an optimizer.
    let bad = OptimizerConfig::Sgd(SgdConfig {
        nesterov: true,
        ..SgdConfig::default()
    });
    assert!(matches!(
        Optimizer::new(bad),
        Err(OptimizerError::Config(e)) if e.pointer == "/nesterov"
    ));
}

#[test]
fn autodiff_context_is_dropped() {
    let mut opt = Optimizer::new(OptimizerConfig::Adam(AdamConfig::default())).unwrap();
    let mut p = params(CPU);
    for v in p.values_mut() {
        *v = v.clone().require_grad().unwrap();
    }
    opt.step(0, 1e-3, &mut p, &grads(0, CPU)).unwrap();
    for v in p.values() {
        assert!(!v.has_autodiff());
    }
    for s in opt.export_state() {
        assert!(!s.tensor.has_autodiff());
    }
}

#[test]
fn clipping_edge_cases() {
    let config = GradientClipConfig { max_norm: 1.0 };
    let mut empty = TensorMap::new();
    assert!(clip_gradients(&config, &mut empty).unwrap().is_none());
    // A total far below max_norm: c = 1, the values are unchanged.
    let mut g = grads(0, CPU);
    let before = map_bits(&g);
    let total = clip_gradients(&GradientClipConfig { max_norm: 1e6 }, &mut g)
        .unwrap()
        .unwrap();
    assert!(read_f32(&total)[0] > 0.0);
    assert_eq!(map_bits(&g), before);
    // Errors change nothing.
    let mut g = grads(0, CPU);
    let half = g["fc.weight"].clone().cast(DType::F16).unwrap();
    g.insert("fc.weight".into(), half);
    assert!(matches!(
        clip_gradients(&config, &mut g),
        Err(OptimizerError::Mismatch { .. })
    ));
    assert_eq!(bits(&g["fc.bias"]), bits(&grads(0, CPU)["fc.bias"]));
    let mut g = grads(0, CPU);
    assert!(matches!(
        clip_gradients(&GradientClipConfig { max_norm: 0.0 }, &mut g),
        Err(OptimizerError::Config(_))
    ));
    assert_eq!(map_bits(&g), map_bits(&grads(0, CPU)));
}

/// The expected bits after 3 steps of training.md §3.1's op sequence, one rounding per op and no
/// FMA: NumPy f32 (`emulate_optimizer` of `tools/reference/gen_train.py`) on the inputs of
/// [`spec_sequence_is_bit_exact_on_the_cpu`]. The CPU's elementwise f32 ops round each op once,
/// so they must match bit for bit. `beta1 = 0.4` takes ATen's second lerp formula, and for it
/// `f32(1 − f32(1 − β1)) ≠ f32(β1)`: both choices of §3.1 change bits here, which the PyTorch
/// fixtures (tolerance 1e-6) can't see.
fn spec_bits(case: &str) -> &'static [(&'static str, &'static [u32])] {
    match case {
        "adam" => &[
            (
                "param",
                &[
                    0x3fbd5952, 0xbdc5de53, 0x3e66882a, 0x3f707327, 0x3f291a06, 0x3fd49b49,
                    0xbf4b3824, 0xbe9c898f, 0x3ed8b4e0, 0xbf3a7e2d, 0x3fbaf4b0, 0xbf51a641,
                    0xc009811d, 0x3f309a9a, 0x3f983013, 0x3ebbe83b,
                ],
            ),
            (
                "exp_avg",
                &[
                    0xbeef818e, 0x3f370c1e, 0x3f114d62, 0x3e50591a, 0x3ec6a798, 0x3f421dc4,
                    0x3e92124c, 0xbe4e32b9, 0x3df4652b, 0x3e22b326, 0xbaf43e00, 0xbcb222e8,
                    0xbe91ddea, 0xbedbf561, 0xbd3d5e48, 0x3f011da3,
                ],
            ),
            (
                "exp_avg_sq",
                &[
                    0x3dcd2677, 0x3e1445ea, 0x3ddbd921, 0x3ddd281d, 0x3d83cc6a, 0x3e5e92bc,
                    0x3dca68e6, 0x3da1c106, 0x3dcedfb9, 0x3c2d785d, 0x3e42304c, 0x3dd2f8f2,
                    0x3e01584c, 0x3dcc048c, 0x3d952dc8, 0x3de6af64,
                ],
            ),
        ],
        "adamw" => &[
            (
                "param",
                &[
                    0x3fbd93e3, 0xbdce432d, 0x3e58edb2, 0x3f6c7623, 0x3f23620e, 0x3fd0aba5,
                    0xbf49bd07, 0xbe9c4f88, 0x3ed8dce0, 0xbf48eb65, 0x3fb76e38, 0xbf51a102,
                    0xc00791c4, 0x3f31083a, 0x3fb0068d, 0x3eb40def,
                ],
            ),
            (
                "exp_avg",
                &[
                    0xbf193663, 0x3f35bc7a, 0x3f088180, 0x3ddcd99c, 0x3ea0ccf3, 0x3f177955,
                    0x3eb7f67b, 0xbe2bd27e, 0x3d9fb208, 0x3e61c0a8, 0xbe127908, 0x3d65775f,
                    0xbdac62b0, 0xbefa35d4, 0xbe1e1062, 0x3eea860d,
                ],
            ),
            (
                "exp_avg_sq",
                &[
                    0x3e056ae8, 0x3e107a24, 0x3dbfa778, 0x3db49feb, 0x3d355919, 0x3e1abc18,
                    0x3dcd74e3, 0x3d99f4e2, 0x3dcebcaf, 0x3c602f26, 0x3e24cd4c, 0x3dc43e5f,
                    0x3e0fb48c, 0x3ddc1067, 0x3db0574f, 0x3dc1737d,
                ],
            ),
        ],
        "sgd" => &[
            (
                "param",
                &[
                    0x3fc492e1, 0xbe1e0834, 0x3e4a0700, 0x3f7e88da, 0x3f335fe3, 0x3fc87cd1,
                    0xbf6a536e, 0xbea01032, 0x3f054bed, 0xbf45c63d, 0x3fb29ea0, 0xbf4245a8,
                    0xc00e5a14, 0x3f2c031b, 0x3fac89ab, 0x3e98dbc7,
                ],
            ),
            (
                "momentum_buffer",
                &[
                    0xbf96e6e8, 0x3fe71644, 0x3fbf1f5a, 0x3f34c79a, 0x3f829fb3, 0x3fc4ec0f,
                    0x3e7f783c, 0xbe85eef0, 0xbe81a8b1, 0x3e945844, 0x3f273195, 0xbe9aaba8,
                    0x3f186524, 0xbf351232, 0xbf2e56a1, 0x3fbc634e,
                ],
            ),
        ],
        "nesterov" => &[
            (
                "param",
                &[
                    0x3fd3ba1a, 0xbeb3ad9f, 0x3d15f174, 0x3f68a8ba, 0x3f17b21d, 0x3fb4e590,
                    0xbf6ffd5c, 0xbe88b1fa, 0x3f06dcce, 0xbf4ed033, 0x3fa983c1, 0xbf3f33a2,
                    0xc0129dd9, 0x3f3df9d5, 0x3fb74a91, 0x3e121a15,
                ],
            ),
            (
                "momentum_buffer",
                &[
                    0xbfa0ad3e, 0x3ff94e4c, 0x3fcce0d4, 0x3f454a79, 0x3f8adcc7, 0x3fcf4258,
                    0x3e8edeea, 0xbeac8ab4, 0xbe481a93, 0x3eaa4faa, 0x3f2af533, 0xbe808f02,
                    0x3f1d73c0, 0xbf436734, 0xbf401ed6, 0x3fc83ac4,
                ],
            ),
        ],
        other => panic!("no expected bits for {other}"),
    }
}

#[test]
fn spec_sequence_is_bit_exact_on_the_cpu() {
    let cases = [
        (
            "adam",
            OptimizerConfig::Adam(AdamConfig {
                lr: 0.1,
                betas: [0.4, 0.9],
                eps: 1e-4,
                weight_decay: 0.1,
            }),
        ),
        (
            "adamw",
            OptimizerConfig::Adamw(AdamWConfig {
                lr: 0.1,
                betas: [0.4, 0.9],
                eps: 1e-4,
                weight_decay: 0.1,
            }),
        ),
        (
            "sgd",
            OptimizerConfig::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.1,
                weight_decay: 0.01,
                ..SgdConfig::default()
            }),
        ),
        (
            "nesterov",
            OptimizerConfig::Sgd(SgdConfig {
                lr: 0.1,
                momentum: 0.9,
                nesterov: true,
                weight_decay: 0.01,
                ..SgdConfig::default()
            }),
        ),
    ];
    for (name, config) in cases {
        let lr = config.lr();
        let mut opt = Optimizer::new(config).unwrap();
        let mut p = TensorMap::new();
        p.insert("p".into(), f32_tensor(&[16], &values(16, 1, 2.0), CPU));
        for k in 0..3 {
            let mut g = TensorMap::new();
            g.insert(
                "p".into(),
                f32_tensor(&[16], &values(16, 100 + k, 1.0), CPU),
            );
            opt.step(k, lr, &mut p, &g).unwrap();
        }
        for &(state, want) in spec_bits(name) {
            let got = if state == "param" {
                bits(&p["p"])
            } else {
                let s = opt.export_state();
                bits(&s.iter().find(|s| s.state == state).unwrap().tensor)
            };
            assert_eq!(got, want, "{name}: {state}");
        }
    }
}
