//! The experiments of the autodiff spike (#71) that later tasks still use: the optimizers of
//! design (b) with their PyTorch comparison (`step_b`, Q3: kept for #79), the cost of host-side
//! randomness (Q4) and the MNIST-sized CNN step (Q5). `#[ignore = "spike"]`, and they do nothing
//! unless `POLYPOUS_SPIKE=1` is set, so neither the normal test run nor CI's `--include-ignored`
//! runs them: they are measurements, not checks (Q3 needs PyTorch's results; Q4 and Q5 allocate
//! MNIST-sized tensors and take minutes in a debug build). The spike's Q1 (API) and Q2 (op
//! gradients) experiments were replaced by the API tests in `tests/autodiff.rs` and the gradient
//! reference harness (`tests/grad_reference_*`, #75).
//!
//! ```text
//! POLYPOUS_SPIKE=1 cargo test -p polypous-backend --lib spike -- --ignored --nocapture --test-threads 1
//! # lavapipe as well:
//! POLYPOUS_SPIKE=1 VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json POLYPOUS_TEST_SOFTWARE_VULKAN=1 \
//!   cargo test -p polypous-backend --features vulkan --lib spike -- --ignored --nocapture --test-threads 1
//! # Q5 timings need a release build:
//! POLYPOUS_SPIKE=1 cargo test --release -p polypous-backend --lib spike_q5 -- --ignored --nocapture
//! ```
//!
//! `spike_q3_*` read PyTorch's results from `spike/out/q3_torch.json`, which
//! `spike/compare_torch.py q3` writes.

use std::time::Instant;

/// Whether `POLYPOUS_SPIKE=1` asks for the experiments (see the module documentation).
fn spike_enabled() -> bool {
    let enabled = std::env::var("POLYPOUS_SPIKE").is_ok_and(|v| v == "1");
    if !enabled {
        eprintln!("SKIP spike experiment: set POLYPOUS_SPIKE=1 to run it");
    }
    enabled
}

use crate::device::DeviceId;
use crate::ops::Scalar;
use crate::tensor::{DType, DynTensor, Result, TensorData};

// ---------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------

fn devices() -> Vec<DeviceId> {
    #[allow(unused_mut)]
    let mut out = vec![DeviceId::Cpu];
    #[cfg(feature = "vulkan")]
    {
        use crate::device::{BackendKind, EnumerateOptions, list_devices};
        let options = EnumerateOptions {
            software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
                .is_ok_and(|v| v == "1"),
        };
        out.extend(
            list_devices()
                .into_iter()
                .filter(|id| id.backend() != BackendKind::Cpu && !options.skips(*id)),
        );
    }
    out
}

fn ft(values: &[f32], shape: &[usize], dev: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = shape.iter().map(|&d| d as u64).collect();
    let data = TensorData::new(DType::F32, shape, bytes).unwrap();
    DynTensor::from_data(&data, dev).unwrap()
}

fn it(values: &[i64], shape: &[usize], dev: DeviceId) -> DynTensor {
    let bytes = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let shape = shape.iter().map(|&d| d as u64).collect();
    let data = TensorData::new(DType::I64, shape, bytes).unwrap();
    DynTensor::from_data(&data, dev).unwrap()
}

fn read(t: &DynTensor) -> Vec<f32> {
    let t = t.clone().without_autodiff();
    let t = if t.dtype() == DType::F32 {
        t
    } else {
        t.cast(DType::F32).unwrap()
    };
    let data = t.into_data().unwrap();
    let (chunks, _) = data.bytes.as_chunks::<4>();
    chunks.iter().map(|&b| f32::from_le_bytes(b)).collect()
}

fn out_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spike/out");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------------------------
// Q3: optimizers of design (b), own optimizers on DynTensor ops (ADR-0014), against PyTorch
// (spike/out/q3_torch.json from compare_torch.py). Design (a), Burn's optimizers on a hand-built
// Module, was removed after ADR-0014 was accepted; its numbers are in PR #86.
// ---------------------------------------------------------------------------------------------

/// The optimizer configs of the comparison (the same in compare_torch.py).
#[derive(Clone, Copy, Debug)]
enum Opt {
    Sgd {
        lr: f64,
        momentum: f64,
        dampening: f64,
        nesterov: bool,
        weight_decay: f64,
    },
    Adam {
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        decoupled: bool,
    },
}

fn q3_configs() -> Vec<(&'static str, Opt)> {
    vec![
        (
            "sgd_nesterov",
            Opt::Sgd {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.0,
                nesterov: true,
                weight_decay: 0.01,
            },
        ),
        (
            "sgd_momentum_dampening",
            Opt::Sgd {
                lr: 0.1,
                momentum: 0.9,
                dampening: 0.1,
                nesterov: false,
                weight_decay: 0.0,
            },
        ),
        (
            "adam",
            Opt::Adam {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.01,
                decoupled: false,
            },
        ),
        (
            "adamw",
            Opt::Adam {
                lr: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                eps: 1e-8,
                weight_decay: 0.1,
                decoupled: true,
            },
        ),
    ]
}

/// Per-param optimizer state of design (b): plain `DynTensor`s and a step count.
#[derive(Default, Clone)]
struct OptState {
    step: u32,
    m: Option<DynTensor>,
    v: Option<DynTensor>,
}

/// One optimizer step of design (b), in PyTorch's single-tensor order (`torch/optim/sgd.py`,
/// `adam.py`; `foreach=False`, which is PyTorch's CPU default). Scalars are computed in f64
/// like PyTorch's Python floats and rounded once to the tensor dtype.
fn step_b(opt: Opt, p: DynTensor, g: DynTensor, st: &mut OptState) -> Result<DynTensor> {
    let p = p.without_autodiff();
    let (dtype, device) = (p.dtype(), p.device());
    let k = |v: f64| DynTensor::full(&[1], Scalar::Float(v), dtype, device).unwrap();
    st.step += 1;
    match opt {
        Opt::Sgd {
            lr,
            momentum,
            dampening,
            nesterov,
            weight_decay,
        } => {
            let mut g = g;
            if weight_decay != 0.0 {
                g = g.add(p.clone().mul(k(weight_decay))?)?;
            }
            if momentum != 0.0 {
                let buf = match st.m.take() {
                    None => g.clone(),
                    Some(buf) => buf
                        .mul(k(momentum))?
                        .add(g.clone().mul(k(1.0 - dampening))?)?,
                };
                g = if nesterov {
                    g.add(buf.clone().mul(k(momentum))?)?
                } else {
                    buf.clone()
                };
                st.m = Some(buf);
            }
            p.add(g.mul(k(-lr))?)
        }
        Opt::Adam {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            decoupled,
        } => {
            let mut g = g;
            let mut p = p;
            if weight_decay != 0.0 {
                if decoupled {
                    p = p.clone().mul(k(1.0 - lr * weight_decay))?;
                } else {
                    g = g.add(p.clone().mul(k(weight_decay))?)?;
                }
            }
            let zeros = || p.full_like(Scalar::Float(0.0), None);
            let m = st.m.take().map_or_else(zeros, Ok)?;
            let v = st.v.take().map_or_else(zeros, Ok)?;
            // exp_avg.lerp_(grad, 1 - beta1): m + w * (g - m) for w < 0.5.
            let m = m.clone().add(g.clone().sub(m)?.mul(k(1.0 - beta1))?)?;
            // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            let v = v
                .mul(k(beta2))?
                .add(g.clone().mul(g)?.mul(k(1.0 - beta2))?)?;
            let t = f64::from(st.step);
            let bc1 = 1.0 - beta1.powf(t);
            let bc2 = 1.0 - beta2.powf(t);
            let step_size = lr / bc1;
            let denom = v.clone().sqrt()?.div(k(bc2.sqrt()))?.add(k(eps))?;
            let p = p.add(m.clone().div(denom)?.mul(k(-step_size))?)?;
            st.m = Some(m);
            st.v = Some(v);
            Ok(p)
        }
    }
}

struct TorchQ3 {
    w0: Vec<f32>,
    b0: Vec<f32>,
    x: Vec<f32>,
    y: Vec<f32>,
    /// Per config: per step, (w, b, grad_w, grad_b).
    steps: std::collections::HashMap<String, Vec<[Vec<f32>; 4]>>,
}

fn load_torch_q3() -> Option<TorchQ3> {
    let path = out_dir().join("q3_torch.json");
    let text = std::fs::read_to_string(&path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let vec = |v: &serde_json::Value| -> Vec<f32> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect()
    };
    let mut steps = std::collections::HashMap::new();
    for (name, list) in v["steps"].as_object().unwrap() {
        let list = list
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                [
                    vec(&s["w"]),
                    vec(&s["b"]),
                    vec(&s["grad_w"]),
                    vec(&s["grad_b"]),
                ]
            })
            .collect();
        steps.insert(name.clone(), list);
    }
    Some(TorchQ3 {
        w0: vec(&v["w0"]),
        b0: vec(&v["b0"]),
        x: vec(&v["x"]),
        y: vec(&v["y"]),
        steps,
    })
}

fn ulps(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    let (ia, ib) = (a.to_bits() as i32, b.to_bits() as i32);
    let fix = |i: i32| if i < 0 { i32::MIN - i } else { i };
    fix(ia).abs_diff(fix(ib))
}

fn compare(label: &str, got: &[f32], want: &[f32]) -> (f32, f32, u32) {
    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut max_ulp = 0u32;
    for (&g, &w) in got.iter().zip(want) {
        let abs = (g - w).abs();
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(abs / w.abs().max(1e-30));
        max_ulp = max_ulp.max(ulps(g, w));
    }
    let _ = label;
    (max_abs, max_rel, max_ulp)
}

/// MSE loss of the tiny `nn.linear` model: mean((x Wᵀ + b - y)²).
fn mse_linear(x: &DynTensor, w: &DynTensor, b: &DynTensor, y: &DynTensor) -> Result<DynTensor> {
    let pred = x.clone().linear(w.clone(), Some(b.clone()))?;
    let d = pred.sub(y.clone())?;
    d.clone().mul(d)?.mean(None, true)?.reshape(&[1])
}

#[test]
#[ignore = "spike"]
fn spike_q3_optimizers_design_b() {
    if !spike_enabled() {
        return;
    }
    let Some(torch) = load_torch_q3() else {
        eprintln!("spike/out/q3_torch.json missing: run spike/compare_torch.py q3 first");
        return;
    };
    for dev in devices() {
        let x = ft(&torch.x, &[2, 4], dev);
        let y = ft(&torch.y, &[2, 3], dev);
        for (name, opt) in q3_configs() {
            let mut w = ft(&torch.w0, &[3, 4], dev);
            let mut b = ft(&torch.b0, &[3], dev);
            let (mut sw, mut sb) = (OptState::default(), OptState::default());
            for (step, want) in torch.steps[name].iter().enumerate() {
                let wl = w.clone().require_grad().unwrap();
                let bl = b.clone().require_grad().unwrap();
                let loss = mse_linear(&x, &wl, &bl, &y).unwrap();
                let grads = loss.backward().unwrap();
                let gw = grads.get(&wl).unwrap().unwrap();
                let gb = grads.get(&bl).unwrap().unwrap();
                let cgw = compare("grad_w", &read(&gw), &want[2]);
                let cgb = compare("grad_b", &read(&gb), &want[3]);
                w = step_b(opt, wl, gw, &mut sw).unwrap();
                b = step_b(opt, bl, gb, &mut sb).unwrap();
                assert!(!w.has_autodiff());
                let cw = compare("w", &read(&w), &want[0]);
                let cb = compare("b", &read(&b), &want[1]);
                eprintln!(
                    "(b) {dev} {name:<24} step {}: grad max ulp {}/{}, params max abs {:.3e}/{:.3e} max rel {:.3e}/{:.3e} max ulp {}/{}",
                    step + 1,
                    cgw.2,
                    cgb.2,
                    cw.0,
                    cb.0,
                    cw.1,
                    cb.1,
                    cw.2,
                    cb.2
                );
            }
        }
    }
}

/// Optimizer math only: PyTorch's gradients of every step go into the optimizer, so the
/// comparison doesn't include the gradient's own rounding (as #79's optimizer fixtures will).
#[test]
#[ignore = "spike"]
fn spike_q3_optimizer_only_design_b() {
    if !spike_enabled() {
        return;
    }
    let Some(torch) = load_torch_q3() else {
        return;
    };
    for dev in devices() {
        for (name, opt) in q3_configs() {
            let mut w = ft(&torch.w0, &[3, 4], dev);
            let mut b = ft(&torch.b0, &[3], dev);
            let (mut sw, mut sb) = (OptState::default(), OptState::default());
            let mut line = Vec::new();
            for want in &torch.steps[name] {
                w = step_b(opt, w, ft(&want[2], &[3, 4], dev), &mut sw).unwrap();
                b = step_b(opt, b, ft(&want[3], &[3], dev), &mut sb).unwrap();
                let cw = compare("w", &read(&w), &want[0]);
                let cb = compare("b", &read(&b), &want[1]);
                line.push(format!(
                    "{}/{} ulp (rel {:.1e})",
                    cw.2,
                    cb.2,
                    cw.1.max(cb.1)
                ));
            }
            eprintln!(
                "(b, torch grads) {dev} {name:<24} steps 1-3: {}",
                line.join(", ")
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Q4: host-side randomness (init, dropout masks) at MNIST size
// ---------------------------------------------------------------------------------------------

/// SplitMix64 (Steele, Lea, Flood 2014), used as a counter-based hash: value `i` of a stream is
/// `mix(key + (i + 1) · γ)`. A stream is keyed by seed, step and node path, so masks don't depend
/// on the device, on the order ops run in, or on threads.
fn splitmix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn stream_key(seed: u64, step: u64, path: &str) -> u64 {
    // FNV-1a over the path, then mixed with seed and step.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in path.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
    }
    splitmix(splitmix(seed ^ 0x9E37_79B9_7F4A_7C15) ^ splitmix(step) ^ h)
}

/// Uniform f32 in [0, 1) from the top 24 bits.
fn uniform(key: u64, i: u64) -> f32 {
    let z = splitmix(key.wrapping_add((i + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
    (z >> 40) as f32 * (1.0 / (1u64 << 24) as f32)
}

/// Dropout mask scaled by 1/(1-p): values 0 or 1/(1-p).
fn dropout_mask(key: u64, n: usize, p: f32) -> Vec<f32> {
    let scale = 1.0 / (1.0 - p);
    (0..n as u64)
        .map(|i| if uniform(key, i) < p { 0.0 } else { scale })
        .collect()
}

#[test]
#[ignore = "spike"]
fn spike_q4_host_rng_cost() {
    if !spike_enabled() {
        return;
    }
    // Statistics of the generator: mean of 1M uniforms and the dropout rate.
    let key = stream_key(42, 7, "block1/dropout");
    let n = 1_000_000u64;
    let mean = (0..n).map(|i| f64::from(uniform(key, i))).sum::<f64>() / n as f64;
    let mask = dropout_mask(key, n as usize, 0.25);
    let dropped = mask.iter().filter(|&&m| m == 0.0).count() as f64 / n as f64;
    eprintln!("uniform mean {mean:.5} (0.5), dropout rate {dropped:.5} (0.25)");
    assert_ne!(
        stream_key(42, 7, "block1/dropout"),
        stream_key(42, 8, "block1/dropout")
    );

    for dev in devices() {
        for (label, shape) in [
            ("fc1 output [64, 128]", vec![64usize, 128]),
            ("pooled conv [64, 64, 12, 12]", vec![64, 64, 12, 12]),
            ("conv2 output [64, 64, 24, 24]", vec![64, 64, 24, 24]),
            ("init of fc1.weight [128, 9216]", vec![128, 9216]),
        ] {
            let n: usize = shape.iter().product();
            let reps = 20;
            let t0 = Instant::now();
            let mut gen_time = 0.0;
            for step in 0..reps {
                let g0 = Instant::now();
                let m = dropout_mask(stream_key(42, step, "x"), n, 0.25);
                gen_time += g0.elapsed().as_secs_f64();
                let t = ft(&m, &shape, dev);
                // Use it so the upload completes.
                let _ = read(&t.sum(None, false).unwrap());
            }
            let total = t0.elapsed().as_secs_f64() / reps as f64;
            eprintln!(
                "{dev}: {label} ({n} values): generate {:.2} ms, generate + upload + sync {:.2} ms per step",
                gen_time / reps as f64 * 1e3,
                total * 1e3
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Q5: speed and memory of an MNIST-sized CNN step (forward, backward, AdamW of design (b))
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "spike"]
fn spike_q5_mnist_cnn_step() {
    if !spike_enabled() {
        return;
    }
    let batch: usize = std::env::var("SPIKE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64);
    let steps: usize = std::env::var("SPIKE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);
    let only: Option<String> = std::env::var("SPIKE_DEVICE").ok();
    for dev in devices() {
        if only.as_deref().is_some_and(|o| o != dev.to_string()) {
            continue;
        }
        // Params, Kaiming-uniform-like init from the host generator.
        let init = |path: &str, shape: &[usize], fan_in: usize| {
            let n: usize = shape.iter().product();
            let bound = 1.0 / (fan_in as f32).sqrt();
            let key = stream_key(1, 0, path);
            let v: Vec<f32> = (0..n as u64)
                .map(|i| (uniform(key, i) * 2.0 - 1.0) * bound)
                .collect();
            ft(&v, shape, dev)
        };
        let mut params = vec![
            init("conv1.weight", &[32, 1, 3, 3], 9),
            init("conv1.bias", &[32], 9),
            init("conv2.weight", &[64, 32, 3, 3], 288),
            init("conv2.bias", &[64], 288),
            init("fc1.weight", &[128, 9216], 9216),
            init("fc1.bias", &[128], 9216),
            init("fc2.weight", &[10, 128], 128),
            init("fc2.bias", &[10], 128),
        ];
        let n_params: usize = params
            .iter()
            .map(|p| p.shape().iter().product::<usize>())
            .sum();
        let mut states = vec![OptState::default(); params.len()];
        let opt = Opt::Adam {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            decoupled: true,
        };
        let mut times = Vec::new();
        let mut phases = [0f64; 4];
        for step in 0..steps + 2 {
            // A batch of "images" and labels from the host generator.
            let key = stream_key(2, step as u64, "data");
            let images: Vec<f32> = (0..(batch * 784) as u64).map(|i| uniform(key, i)).collect();
            let labels: Vec<i64> = (0..batch as u64)
                .map(|i| (uniform(key ^ 1, i) * 10.0) as i64 % 10)
                .collect();
            let t0 = Instant::now();
            let x = ft(&images, &[batch, 1, 28, 28], dev);
            let target = it(&labels, &[batch, 1], dev);
            let mask1 = ft(
                &dropout_mask(stream_key(3, step as u64, "dropout1"), batch * 9216, 0.25),
                &[batch, 9216],
                dev,
            );
            let mask2 = ft(
                &dropout_mask(stream_key(3, step as u64, "dropout2"), batch * 128, 0.5),
                &[batch, 128],
                dev,
            );
            let leaves: Vec<DynTensor> = params
                .iter()
                .map(|p| p.clone().require_grad().unwrap())
                .collect();
            let t1 = Instant::now();
            let h = x
                .conv2d(
                    leaves[0].clone(),
                    Some(leaves[1].clone()),
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    1,
                )
                .unwrap()
                .relu()
                .unwrap();
            let h = h
                .conv2d(
                    leaves[2].clone(),
                    Some(leaves[3].clone()),
                    [1, 1],
                    [0, 0],
                    [1, 1],
                    1,
                )
                .unwrap()
                .relu()
                .unwrap();
            let h = h.max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]).unwrap();
            let h = h.flatten(1, 3).unwrap().mul(mask1).unwrap();
            let h = h
                .linear(leaves[4].clone(), Some(leaves[5].clone()))
                .unwrap()
                .relu()
                .unwrap()
                .mul(mask2)
                .unwrap();
            let logits = h
                .linear(leaves[6].clone(), Some(leaves[7].clone()))
                .unwrap();
            let logp = logits.log_softmax(1).unwrap();
            let loss = logp
                .gather(1, target)
                .unwrap()
                .mean(None, true)
                .unwrap()
                .reshape(&[1])
                .unwrap()
                .neg()
                .unwrap();
            let loss_value = read(&loss)[0];
            let t2 = Instant::now();
            let grads = loss.backward().unwrap();
            let gs: Vec<DynTensor> = leaves
                .iter()
                .map(|l| grads.get(l).unwrap().unwrap())
                .collect();
            // Sync so that the backward time isn't hidden in the update.
            for g in &gs {
                let _ = read(&g.clone().sum(None, false).unwrap());
            }
            let t3 = Instant::now();
            params = leaves
                .into_iter()
                .zip(gs)
                .zip(states.iter_mut())
                .map(|((p, g), st)| step_b(opt, p, g, st).unwrap())
                .collect();
            for p in &params {
                let _ = read(&p.clone().sum(None, false).unwrap());
            }
            let t4 = Instant::now();
            if step >= 2 {
                times.push((t4 - t0).as_secs_f64());
                phases[0] += (t1 - t0).as_secs_f64();
                phases[1] += (t2 - t1).as_secs_f64();
                phases[2] += (t3 - t2).as_secs_f64();
                phases[3] += (t4 - t3).as_secs_f64();
            }
            eprintln!(
                "{dev} step {step}: loss {loss_value:.4}, {:.1} ms",
                (t4 - t0).as_secs_f64() * 1e3
            );
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let per_epoch = 60_000usize.div_ceil(batch) as f64 * mean;
        let n = times.len() as f64;
        eprintln!(
            "{dev}: {n_params} params, batch {batch}: {:.1} ms per step (upload {:.1}, forward {:.1}, backward {:.1}, AdamW {:.1}); one epoch ≈ {:.1} min, 5 epochs ≈ {:.1} min",
            mean * 1e3,
            phases[0] / n * 1e3,
            phases[1] / n * 1e3,
            phases[2] / n * 1e3,
            phases[3] / n * 1e3,
            per_epoch / 60.0,
            per_epoch * 5.0 / 60.0
        );
    }
}
