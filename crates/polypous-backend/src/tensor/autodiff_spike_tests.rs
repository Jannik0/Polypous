//! Experiments of the autodiff spike (#71). Not part of the normal test run: they need the
//! `autodiff` feature and are `#[ignore = "spike"]`.
//!
//! ```text
//! cargo test -p polypous-backend --features autodiff --lib spike -- --ignored --nocapture --test-threads 1
//! # lavapipe as well:
//! VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json POLYPOUS_TEST_SOFTWARE_VULKAN=1 \
//!   cargo test -p polypous-backend --features autodiff,vulkan --lib spike -- --ignored --nocapture --test-threads 1
//! # Q5 timings need a release build:
//! cargo test --release -p polypous-backend --features autodiff --lib spike_q5 -- --ignored --nocapture
//! ```
//!
//! `spike_q2_*` and `spike_q3_*` write JSON next to this crate (`spike/out/`); the PyTorch side
//! is `spike/compare_torch.py`.

use std::time::Instant;

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

fn scalar_like(v: f64, like: &DynTensor) -> DynTensor {
    DynTensor::full(&[1], Scalar::Float(v), like.dtype(), like.device()).unwrap()
}

/// Sum to one element (keepdim, then flattened to `[1]`).
fn total(t: DynTensor) -> Result<DynTensor> {
    t.sum(None, true)?.reshape(&[1])
}

fn json_f32(values: &[f32]) -> serde_json::Value {
    serde_json::Value::Array(
        values
            .iter()
            .map(|&v| {
                if v.is_finite() {
                    serde_json::json!(v)
                } else {
                    serde_json::json!(v.to_string())
                }
            })
            .collect(),
    )
}

fn out_dir() -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("spike/out");
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn device_tag(dev: DeviceId) -> String {
    dev.to_string().replace(':', "_")
}

fn catch<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> std::result::Result<T, String> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(f).map_err(|e| {
        e.downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "panic".into())
    });
    std::panic::set_hook(hook);
    r
}

// ---------------------------------------------------------------------------------------------
// Q1: autodiff device, ranks, dtypes, API, to_device, bool/int
// ---------------------------------------------------------------------------------------------

#[test]
#[ignore = "spike"]
fn spike_q1_ranks_dtypes_api() {
    for dev in devices() {
        eprintln!("== Q1 on {dev}");
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            if !DynTensor::device_supports(dev, dtype).unwrap() {
                eprintln!("  {dtype}: not supported on {dev}, skipped");
                continue;
            }
            let mut ok = Vec::new();
            for rank in 1..=6 {
                let shape = &[2usize, 3, 1, 2, 1, 2][..rank];
                let n: usize = shape.iter().product();
                let values: Vec<f32> = (0..n).map(|i| i as f32 * 0.5 - 1.0).collect();
                let x = ft(&values, shape, dev)
                    .cast(dtype)
                    .unwrap()
                    .require_grad()
                    .unwrap();
                assert!(x.is_tracked() && x.is_autodiff());
                assert_eq!(x.device(), dev, "device id survives require_grad");
                // loss = sum(x * x + 3 * x); d/dx = 2x + 3. Uses mul, add, a broadcast scalar
                // (an untracked tensor of the plain device), sum and reshape.
                let three = scalar_like(3.0, &x);
                assert!(!three.is_autodiff());
                let y = x.clone().mul(x.clone()).unwrap();
                let y = y.add(x.clone().mul(three).unwrap()).unwrap();
                let loss = total(y).unwrap();
                assert!(loss.is_tracked());
                let grads = loss.backward().unwrap();
                let g = x.grad(&grads).expect("gradient of the leaf");
                assert_eq!(g.dtype(), dtype);
                assert_eq!(g.shape(), shape);
                assert_eq!(g.device(), dev);
                assert!(!g.is_tracked());
                let expected: Vec<f32> = values.iter().map(|v| 2.0 * v + 3.0).collect();
                assert_eq!(read(&g), expected, "{dev} {dtype} rank {rank}");
                ok.push(rank);
            }
            eprintln!("  {dtype}: ranks {ok:?} record and backprop, grad = 2x + 3 exactly");
        }

        // detach: cuts the graph, keeps the autodiff context.
        let x = ft(&[1.0, 2.0], &[2], dev).require_grad().unwrap();
        let d = x.clone().mul(x.clone()).unwrap().detach();
        assert!(d.is_autodiff() && !d.is_tracked());
        let loss = total(d.mul(x.clone()).unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        assert_eq!(read(&x.grad(&grads).unwrap()), vec![1.0, 4.0]);
        eprintln!("  detach: d(sum(detach(x*x) * x))/dx = x*x: ok");

        // A leaf that is not part of the loss has no gradient.
        let unused = ft(&[1.0], &[1], dev).require_grad().unwrap();
        assert!(unused.grad(&grads).is_none());
        eprintln!("  grad of a leaf outside the loss: None");

        // Errors instead of Burn panics.
        assert!(it(&[1, 2], &[2], dev).require_grad().is_err());
        assert!(ft(&[1.0], &[1], dev).backward().is_err());
        let x = ft(&[1.0, 2.0], &[2], dev).require_grad().unwrap();
        assert!(x.clone().mul(x.clone()).unwrap().backward().is_err());
        eprintln!("  require_grad(int), backward(untracked), backward(2 elements): errors");

        // Backward twice on the same graph: Burn panics (no retain_graph).
        let x = ft(&[1.0, 2.0], &[2], dev).require_grad().unwrap();
        let loss = total(x.clone().mul(x.clone()).unwrap()).unwrap();
        let _ = loss.backward().unwrap();
        let second = catch(std::panic::AssertUnwindSafe(|| loss.backward().is_ok()));
        eprintln!("  second backward on a consumed graph: {second:?}");

        // A fresh forward from the same leaf after a backward works (training loop shape).
        let loss = total(x.clone().mul(x.clone()).unwrap()).unwrap();
        let g = x.grad(&loss.backward().unwrap()).unwrap();
        assert_eq!(read(&g), vec![2.0, 4.0]);
        eprintln!("  second forward + backward from the same leaf: ok");

        // without_autodiff: ops are not recorded (optimizer updates).
        let p = x.clone().without_autodiff();
        assert!(!p.is_autodiff() && !p.is_tracked());
        let updated = p.sub(g.clone()).unwrap();
        assert!(!updated.is_autodiff());
        let x2 = updated.require_grad().unwrap();
        assert!(x2.is_tracked());
        eprintln!("  without_autodiff → update → require_grad: ok");

        // Bool and int tensors on an autodiff tensor's device.
        let x = ft(&[-1.0, 2.0, 0.5, -3.0], &[4], dev)
            .require_grad()
            .unwrap();
        let zeros = DynTensor::zeros(&[4], DType::F32, dev).unwrap();
        let mask = x.clone().greater(zeros.clone()).unwrap();
        eprintln!(
            "  mask = x > 0: dtype {}, autodiff {}, values {:?}",
            mask.dtype(),
            mask.is_autodiff(),
            mask.to_data().unwrap().bytes
        );
        let y = DynTensor::where_(mask.clone(), x.clone(), zeros.clone()).unwrap();
        let g = x.grad(&total(y).unwrap().backward().unwrap()).unwrap();
        assert_eq!(read(&g), vec![0.0, 1.0, 1.0, 0.0]);
        let not_mask = mask.logical_not().unwrap();
        eprintln!(
            "  where(x > 0, x, 0) grad ok; not(mask) autodiff {}",
            not_mask.is_autodiff()
        );
        let idx = x.clone().argmax(0, true, DType::I64).unwrap();
        eprintln!(
            "  argmax(x): dtype {}, autodiff {}",
            idx.dtype(),
            idx.is_autodiff()
        );
        let picked = x.clone().gather(0, idx).unwrap();
        let g = x.grad(&total(picked).unwrap().backward().unwrap()).unwrap();
        assert_eq!(read(&g), vec![0.0, 1.0, 0.0, 0.0]);
        eprintln!("  gather(x, argmax(x)) grad ok");
        let as_int = x.clone().cast(DType::I64).unwrap();
        eprintln!(
            "  cast(x, i64): autodiff {} tracked {}",
            as_int.is_autodiff(),
            as_int.is_tracked()
        );

        // to_device of a tracked tensor.
        for target in devices() {
            let x = ft(&[1.0, 2.0, 3.0], &[3], dev).require_grad().unwrap();
            let moved = x.clone().mul(x.clone()).unwrap().to_device(target);
            match moved {
                Ok(m) => {
                    let tracked = m.is_tracked();
                    let autodiff = m.is_autodiff();
                    let r = if tracked {
                        let w = ft(&[1.0, 1.0, 1.0], &[3], target);
                        catch(std::panic::AssertUnwindSafe(|| {
                            let loss = total(m.mul(w).unwrap()).unwrap();
                            x.grad(&loss.backward().unwrap()).map(|g| read(&g))
                        }))
                    } else {
                        Ok(None)
                    };
                    eprintln!(
                        "  to_device({dev} → {target}) of x*x: autodiff {autodiff}, tracked {tracked}, grad of x = {r:?}"
                    );
                }
                Err(e) => eprintln!("  to_device({dev} → {target}): error {e}"),
            }
            // Bool across backends goes through the host (M1 workaround).
            let b = ft(&[1.0, -1.0], &[2], dev)
                .require_grad()
                .unwrap()
                .greater(DynTensor::zeros(&[2], DType::F32, dev).unwrap())
                .unwrap();
            let moved = b
                .to_device(target)
                .map(|m| (m.is_autodiff(), m.to_data().unwrap().bytes));
            eprintln!("  bool to_device({dev} → {target}): {moved:?}");
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Q2: gradients through every differentiable op (compared to PyTorch by spike/compare_torch.py)
// ---------------------------------------------------------------------------------------------

type Forward = fn(&[DynTensor], DeviceId) -> Result<DynTensor>;

struct Case {
    name: &'static str,
    inputs: Vec<(Vec<usize>, Vec<f32>)>,
    forward: Forward,
    /// Uses no float constants of its own, so it also runs with f16/bf16 inputs.
    half: bool,
}

fn case(name: &'static str, inputs: Vec<(Vec<usize>, Vec<f32>)>, forward: Forward) -> Case {
    Case {
        name,
        inputs,
        forward,
        half: false,
    }
}

fn half(mut c: Case) -> Case {
    c.half = true;
    c
}

/// Deterministic, non-symmetric values in about [-2, 2].
fn wave(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = ((i as f32 * 1.7 + phase).sin() * 2.0 * 64.0).round() / 64.0;
            if v == 0.0 { 0.25 } else { v }
        })
        .collect()
}

fn s(x: f64) -> Option<Scalar> {
    Some(Scalar::Float(x))
}

#[allow(clippy::too_many_lines)]
fn q2_cases() -> Vec<Case> {
    let x4 = || vec![(vec![4], vec![-1.5, 0.0, 0.5, 2.0])];
    vec![
        // Elementwise
        half(case(
            "add_broadcast",
            vec![(vec![2, 3], wave(6, 0.0)), (vec![3], wave(3, 1.0))],
            |x, _| x[0].clone().add(x[1].clone()),
        )),
        half(case(
            "sub",
            vec![(vec![4], wave(4, 0.0)), (vec![4], wave(4, 2.0))],
            |x, _| x[0].clone().sub(x[1].clone()),
        )),
        half(case(
            "mul_broadcast",
            vec![(vec![2, 3], wave(6, 0.0)), (vec![2, 1], wave(2, 1.0))],
            |x, _| x[0].clone().mul(x[1].clone()),
        )),
        half(case(
            "div",
            vec![
                (vec![4], wave(4, 0.0)),
                (vec![4], vec![2.0, -1.0, 0.5, 1.5]),
            ],
            |x, _| x[0].clone().div(x[1].clone()),
        )),
        half(case(
            "pow",
            vec![
                (vec![4], vec![0.5, 1.5, 2.0, 3.0]),
                (vec![4], vec![2.0, -1.0, 0.5, 1.5]),
            ],
            |x, _| x[0].clone().pow(x[1].clone()),
        )),
        case(
            "pow_special",
            vec![
                (vec![5], vec![0.0, 0.0, 2.0, -2.0, 1.0]),
                (vec![5], vec![2.0, 0.5, 0.0, 2.0, 3.0]),
            ],
            |x, _| x[0].clone().pow(x[1].clone()),
        ),
        case(
            "maximum_ties",
            vec![
                (vec![4], vec![1.0, 2.0, 3.0, 3.0]),
                (vec![4], vec![1.0, 3.0, 2.0, 3.0]),
            ],
            |x, _| x[0].clone().maximum(x[1].clone()),
        ),
        case(
            "minimum_ties",
            vec![
                (vec![4], vec![1.0, 2.0, 3.0, 3.0]),
                (vec![4], vec![1.0, 3.0, 2.0, 3.0]),
            ],
            |x, _| x[0].clone().minimum(x[1].clone()),
        ),
        case(
            "maximum_nan",
            vec![
                (vec![3], vec![f32::NAN, 1.0, 2.0]),
                (vec![3], vec![0.0, f32::NAN, 1.0]),
            ],
            |x, _| x[0].clone().maximum(x[1].clone()),
        ),
        half(case("neg", x4(), |x, _| x[0].clone().neg())),
        half(case("abs", x4(), |x, _| x[0].clone().abs())),
        half(case("exp", x4(), |x, _| x[0].clone().exp())),
        half(case("log", vec![(vec![3], vec![0.5, 1.0, 2.0])], |x, _| {
            x[0].clone().log()
        })),
        half(case(
            "sqrt",
            vec![(vec![3], vec![0.0, 0.25, 4.0])],
            |x, _| x[0].clone().sqrt(),
        )),
        half(case("rsqrt", vec![(vec![2], vec![0.25, 4.0])], |x, _| {
            x[0].clone().rsqrt()
        })),
        half(case("sin", x4(), |x, _| x[0].clone().sin())),
        half(case("cos", x4(), |x, _| x[0].clone().cos())),
        half(case("tanh", x4(), |x, _| x[0].clone().tanh())),
        case(
            "clamp_bounds",
            vec![(vec![5], vec![-2.0, -1.0, 0.0, 1.0, 2.0])],
            |x, _| x[0].clone().clamp(s(-1.0), s(1.0)),
        ),
        case(
            "clamp_min_nan",
            vec![(vec![4], vec![-1.0, 0.0, 1.0, f32::NAN])],
            |x, _| x[0].clone().clamp(s(0.0), None),
        ),
        case(
            "where",
            vec![(vec![4], wave(4, 0.0)), (vec![4], wave(4, 3.0))],
            |x, dev| {
                let cond = x[0]
                    .clone()
                    .greater(DynTensor::zeros(&[4], DType::F32, dev)?)?;
                DynTensor::where_(cond, x[0].clone(), x[1].clone())
            },
        ),
        // Shape
        half(case("reshape", vec![(vec![2, 3], wave(6, 0.0))], |x, _| {
            x[0].clone().reshape(&[3, 2])
        })),
        half(case(
            "flatten",
            vec![(vec![2, 3, 2], wave(12, 0.0))],
            |x, _| x[0].clone().flatten(1, 2),
        )),
        half(case(
            "permute",
            vec![(vec![2, 3, 2], wave(12, 0.0))],
            |x, _| x[0].clone().permute(&[2, 0, 1]),
        )),
        half(case(
            "transpose",
            vec![(vec![2, 3], wave(6, 0.0))],
            |x, _| x[0].clone().transpose(0, 1),
        )),
        half(case(
            "squeeze",
            vec![(vec![2, 1, 3], wave(6, 0.0))],
            |x, _| x[0].clone().squeeze(&[1]),
        )),
        half(case(
            "unsqueeze",
            vec![(vec![2, 3], wave(6, 0.0))],
            |x, _| x[0].clone().unsqueeze(1),
        )),
        half(case("expand", vec![(vec![2, 1], wave(2, 0.0))], |x, _| {
            x[0].clone().expand(&[2, 3])
        })),
        half(case(
            "concat",
            vec![(vec![2, 2], wave(4, 0.0)), (vec![2, 1], wave(2, 1.0))],
            |x, _| DynTensor::concat(&[x[0].clone(), x[1].clone()], 1),
        )),
        half(case(
            "split_second",
            vec![(vec![2, 5], wave(10, 0.0))],
            |x, _| Ok(x[0].clone().split(1, &[2, 3])?.remove(1)),
        )),
        half(case(
            "slice_step",
            vec![(vec![2, 6], wave(12, 0.0))],
            |x, _| x[0].clone().slice(1, 1, 6, 2),
        )),
        case("pad", vec![(vec![2, 3], wave(6, 0.0))], |x, _| {
            x[0].clone().pad(&[1, 2, 0, 1], Scalar::Float(0.5))
        }),
        // Indexing and creation
        case("gather", vec![(vec![2, 3], wave(6, 0.0))], |x, dev| {
            x[0].clone()
                .gather(1, it(&[2, 0, 2, 1, 1, 0], &[2, 3], dev))
        }),
        case(
            "index_select",
            vec![(vec![3, 2], wave(6, 0.0))],
            |x, dev| x[0].clone().index_select(0, it(&[2, 0, 2], &[3], dev)),
        ),
        case("full_like_mul", vec![(vec![3], wave(3, 0.0))], |x, _| {
            let c = x[0].full_like(Scalar::Float(2.5), None)?;
            x[0].clone().mul(c)
        }),
        case(
            "cast_f16_roundtrip",
            vec![(vec![3], wave(3, 0.0))],
            |x, _| x[0].clone().cast(DType::F16)?.cast(DType::F32),
        ),
        // Reductions
        half(case("sum_dim", vec![(vec![2, 3], wave(6, 0.0))], |x, _| {
            x[0].clone().sum(Some(&[1]), false)
        })),
        half(case(
            "mean_dim",
            vec![(vec![2, 3], wave(6, 0.0))],
            |x, _| x[0].clone().mean(Some(&[0]), true),
        )),
        case(
            "max_all_ties",
            vec![(vec![4], vec![1.0, 3.0, 3.0, 2.0])],
            |x, _| x[0].clone().max(None, false),
        ),
        case(
            "max_dim_ties",
            vec![(vec![2, 3], vec![1.0, 3.0, 3.0, 2.0, 2.0, 1.0])],
            |x, _| x[0].clone().max(Some(&[1]), false),
        ),
        case(
            "min_dim_ties",
            vec![(vec![2, 3], vec![1.0, 3.0, 1.0, 2.0, 0.5, 0.5])],
            |x, _| x[0].clone().min(Some(&[1]), false),
        ),
        case(
            "max_dim_nan",
            vec![(vec![2, 3], vec![1.0, f32::NAN, 3.0, 2.0, 5.0, 1.0])],
            |x, _| x[0].clone().max(Some(&[1]), false),
        ),
        half(case("cumsum", vec![(vec![2, 3], wave(6, 0.0))], |x, _| {
            x[0].clone().cumsum(1)
        })),
        half(case(
            "matmul_broadcast",
            vec![(vec![2, 2, 3], wave(12, 0.0)), (vec![3, 2], wave(6, 1.0))],
            |x, _| x[0].clone().matmul(x[1].clone()),
        )),
        half(case("softmax", vec![(vec![2, 3], wave(6, 0.0))], |x, _| {
            x[0].clone().softmax(1)
        })),
        half(case(
            "log_softmax",
            vec![(vec![2, 3], wave(6, 0.0))],
            |x, _| x[0].clone().log_softmax(1),
        )),
        case(
            "top_k",
            vec![(vec![2, 4], vec![1.0, 5.0, 3.0, 4.0, 2.0, 2.0, 0.5, 1.0])],
            |x, _| Ok(x[0].clone().top_k(2, 1, true, DType::I64)?.0),
        ),
        // nn
        half(case(
            "linear",
            vec![
                (vec![2, 3], wave(6, 0.0)),
                (vec![2, 3], wave(6, 1.0)),
                (vec![2], wave(2, 2.0)),
            ],
            |x, _| x[0].clone().linear(x[1].clone(), Some(x[2].clone())),
        )),
        case("embedding", vec![(vec![4, 2], wave(8, 0.0))], |x, dev| {
            it(&[1, 3, 1], &[3], dev).embedding(x[0].clone())
        }),
        half(case(
            "conv1d_groups_stride",
            vec![
                (vec![1, 2, 6], wave(12, 0.0)),
                (vec![2, 1, 3], wave(6, 1.0)),
                (vec![2], wave(2, 2.0)),
            ],
            |x, _| {
                x[0].clone()
                    .conv1d(x[1].clone(), Some(x[2].clone()), 2, 1, 1, 2)
            },
        )),
        half(case(
            "conv2d_groups_dilation",
            vec![
                (vec![1, 2, 5, 5], wave(50, 0.0)),
                (vec![4, 1, 2, 2], wave(16, 1.0)),
                (vec![4], wave(4, 2.0)),
            ],
            |x, _| {
                x[0].clone()
                    .conv2d(x[1].clone(), Some(x[2].clone()), [1, 2], [1, 0], [2, 1], 2)
            },
        )),
        case(
            "max_pool2d_ties",
            vec![(
                vec![1, 1, 4, 4],
                vec![
                    1.0, 3.0, 2.0, 2.0, //
                    3.0, 0.0, 2.0, 2.0, //
                    0.5, 0.5, 1.0, 4.0, //
                    0.5, 0.5, 4.0, 1.0,
                ],
            )],
            |x, _| x[0].clone().max_pool2d([2, 2], [2, 2], [0, 0], [1, 1]),
        ),
        case(
            "max_pool2d_overlap_pad",
            vec![(vec![1, 1, 4, 4], wave(16, 0.0))],
            |x, _| x[0].clone().max_pool2d([3, 3], [2, 2], [1, 1], [1, 1]),
        ),
        half(case(
            "avg_pool2d_include_pad",
            vec![(vec![1, 1, 4, 4], wave(16, 0.0))],
            |x, _| x[0].clone().avg_pool2d([3, 3], [2, 2], [1, 1], true),
        )),
        half(case(
            "avg_pool2d_exclude_pad",
            vec![(vec![1, 1, 4, 4], wave(16, 0.0))],
            |x, _| x[0].clone().avg_pool2d([3, 3], [2, 2], [1, 1], false),
        )),
        half(case(
            "adaptive_avg_pool2d",
            vec![(vec![1, 1, 5, 5], wave(25, 0.0))],
            |x, _| x[0].clone().adaptive_avg_pool2d([3, 3]),
        )),
        half(case(
            "dropout_inference",
            vec![(vec![4], wave(4, 0.0))],
            |x, _| x[0].clone().dropout(0.5),
        )),
        case(
            "batch_norm_inference",
            vec![
                (vec![2, 3, 2], wave(12, 0.0)),
                (vec![3], wave(3, 1.0)),
                (vec![3], wave(3, 2.0)),
            ],
            |x, dev| {
                let rm = ft(&[0.1, -0.2, 0.3], &[3], dev);
                let rv = ft(&[0.5, 1.5, 2.0], &[3], dev);
                x[0].clone()
                    .batch_norm(rm, rv, Some(x[1].clone()), Some(x[2].clone()), 1e-5)
            },
        ),
        half(case(
            "layer_norm",
            vec![
                (vec![2, 4], wave(8, 0.0)),
                (vec![4], wave(4, 1.0)),
                (vec![4], wave(4, 2.0)),
            ],
            |x, _| {
                x[0].clone()
                    .layer_norm(&[4], Some(x[1].clone()), Some(x[2].clone()), 1e-5)
            },
        )),
        half(case(
            "group_norm",
            vec![
                (vec![2, 4, 2], wave(16, 0.0)),
                (vec![4], wave(4, 1.0)),
                (vec![4], wave(4, 2.0)),
            ],
            |x, _| {
                x[0].clone()
                    .group_norm(2, Some(x[1].clone()), Some(x[2].clone()), 1e-5)
            },
        )),
        half(case(
            "rms_norm",
            vec![(vec![2, 4], wave(8, 0.0)), (vec![4], wave(4, 1.0))],
            |x, _| x[0].clone().rms_norm(&[4], Some(x[1].clone()), 1e-6),
        )),
        // Activations
        half(case("relu", x4(), |x, _| x[0].clone().relu())),
        half(case("leaky_relu", x4(), |x, _| {
            x[0].clone().leaky_relu(0.1)
        })),
        half(case(
            "gelu",
            vec![(vec![5], vec![-3.0, -0.5, 0.0, 0.5, 3.0])],
            |x, _| x[0].clone().gelu(),
        )),
        half(case(
            "gelu_tanh",
            vec![(vec![5], vec![-3.0, -0.5, 0.0, 0.5, 3.0])],
            |x, _| x[0].clone().gelu_tanh(),
        )),
        half(case("silu", x4(), |x, _| x[0].clone().silu())),
        half(case("sigmoid", x4(), |x, _| x[0].clone().sigmoid())),
        half(case(
            "softplus",
            vec![(vec![8], vec![-20.0, -1.0, 0.0, 1.0, 19.5, 20.0, 20.5, 30.0])],
            |x, _| x[0].clone().softplus(1.0, 20.0),
        )),
        half(case(
            "softplus_beta2",
            vec![(vec![5], vec![-30.0, -1.0, 2.5, 3.0, 4.0])],
            |x, _| x[0].clone().softplus(2.0, 6.0),
        )),
        half(case(
            "mish",
            vec![(vec![6], vec![-20.0, -1.0, 0.0, 1e-4, 1.0, 30.0])],
            |x, _| x[0].clone().mish(),
        )),
    ]
}

fn run_case(c: &Case, dev: DeviceId, dtype: DType) -> serde_json::Value {
    let leaves: Vec<DynTensor> = c
        .inputs
        .iter()
        .map(|(shape, values)| {
            ft(values, shape, dev)
                .cast(dtype)
                .unwrap()
                .require_grad()
                .unwrap()
        })
        .collect();
    let out = match (c.forward)(&leaves, dev) {
        Ok(out) => out,
        Err(e) => return serde_json::json!({ "name": c.name, "error": e.to_string() }),
    };
    let n: usize = out.shape().iter().product();
    let upstream_values: Vec<f32> = (0..n).map(|i| 1.0 + 0.125 * (i % 8) as f32).collect();
    let upstream = ft(&upstream_values, &out.shape(), dev).cast(dtype).unwrap();
    let out_values = read(&out);
    let out_shape = out.shape();
    let tracked = out.is_tracked();
    let grads: Vec<serde_json::Value> = if tracked {
        let loss = total(out.mul(upstream).unwrap()).unwrap();
        let grads = loss.backward().unwrap();
        leaves
            .iter()
            .map(|l| match l.grad(&grads) {
                Some(g) => json_f32(&read(&g)),
                None => serde_json::Value::Null,
            })
            .collect()
    } else {
        leaves.iter().map(|_| serde_json::Value::Null).collect()
    };
    serde_json::json!({
        "name": c.name,
        "dtype": dtype.name(),
        "inputs": c.inputs.iter().map(|(s, v)| serde_json::json!({"shape": s, "values": json_f32(v)})).collect::<Vec<_>>(),
        "upstream": json_f32(&upstream_values),
        "out_shape": out_shape,
        "out": json_f32(&out_values),
        "out_tracked": tracked,
        "grads": grads,
    })
}

#[test]
#[ignore = "spike"]
fn spike_q2_op_gradients() {
    for dev in devices() {
        for dtype in [DType::F32, DType::F16] {
            if !DynTensor::device_supports(dev, dtype).unwrap() {
                continue;
            }
            let mut results = Vec::new();
            for c in q2_cases() {
                if dtype != DType::F32 && !c.half {
                    continue;
                }
                let r = catch(std::panic::AssertUnwindSafe(|| run_case(&c, dev, dtype)))
                    .unwrap_or_else(|p| serde_json::json!({ "name": c.name, "panic": p }));
                let status = if r.get("panic").is_some() {
                    "PANIC".to_string()
                } else if let Some(e) = r.get("error") {
                    format!("error {e}")
                } else if r["grads"].as_array().unwrap().iter().all(|g| g.is_null()) {
                    "no gradient (cut)".into()
                } else {
                    "gradient".into()
                };
                eprintln!("{dev} {dtype} {:<26} {status}", c.name);
                results.push(r);
            }
            let path = out_dir().join(format!("q2_{}_{}.json", device_tag(dev), dtype.name()));
            std::fs::write(&path, serde_json::to_string_pretty(&results).unwrap()).unwrap();
            eprintln!("wrote {}", path.display());
        }
    }
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
                let gw = wl.grad(&grads).unwrap();
                let gb = bl.grad(&grads).unwrap();
                let cgw = compare("grad_w", &read(&gw), &want[2]);
                let cgb = compare("grad_b", &read(&gb), &want[3]);
                w = step_b(opt, wl, gw, &mut sw).unwrap();
                b = step_b(opt, bl, gb, &mut sb).unwrap();
                assert!(!w.is_autodiff());
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
            let gs: Vec<DynTensor> = leaves.iter().map(|l| l.grad(&grads).unwrap()).collect();
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
