//! Initial values against the reference fixtures (`fixtures/small/init/<kind>/`,
//! testing-strategy.md §3.8 "Initial values", §3.9): every `init` kind on params of each fan role,
//! from a Python implementation of graph-ir.md §2.4.1 (`tools/reference/polyrng.py`, f64 rounded
//! once). The fans come from the IR's table (`polypous_ir::params::fans_of`) and must equal the
//! sidecar's. Uniform-based kinds must be bit-exact; normal-based kinds within the sidecar's
//! `ulp` (1, for the platform's `ln` and `cos`, graph-ir.md §3.1.12).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use std::path::{Path, PathBuf};

use polypous_formats::safetensors::SafetensorsFile;
use polypous_ir::params::fans_of;
use polypous_ir::{Fans, Init};
use polypous_train::initial_values;
use serde_json::Value;

const KINDS: [&str; 9] = [
    "zeros",
    "ones",
    "constant",
    "uniform",
    "normal",
    "kaiming_uniform",
    "kaiming_normal",
    "xavier_uniform",
    "xavier_normal",
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/small/init")
}

fn init_of(v: &Value) -> Init {
    let f = |k: &str| v[k].as_f64().unwrap_or_else(|| panic!("init field {k}"));
    match v["kind"].as_str().unwrap() {
        "zeros" => Init::Zeros,
        "ones" => Init::Ones,
        "constant" => Init::Constant { value: f("value") },
        "uniform" => Init::Uniform {
            low: f("low"),
            high: f("high"),
        },
        "normal" => Init::Normal {
            mean: f("mean"),
            std: f("std"),
        },
        "kaiming_uniform" => Init::KaimingUniform {
            a: v.get("a").and_then(Value::as_f64),
        },
        "kaiming_normal" => Init::KaimingNormal,
        "xavier_uniform" => Init::XavierUniform,
        "xavier_normal" => Init::XavierNormal,
        other => panic!("unknown init kind {other}"),
    }
}

/// The distance in ulps between two f32 values (0 for equal bits; NaN never occurs here).
fn ulps(a: f32, b: f32) -> u64 {
    let key = |x: f32| {
        let bits = x.to_bits();
        if bits >> 31 == 1 {
            -i64::from(bits & 0x7FFF_FFFF)
        } else {
            i64::from(bits)
        }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn initial_values_match_the_reference() {
    let mut total = 0;
    let mut off_by_one = 0;
    for kind in KINDS {
        let dir = root().join(kind);
        let mut sidecars: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        sidecars.sort();
        assert!(!sidecars.is_empty(), "{}: no cases", dir.display());
        for path in sidecars {
            let json: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let case = format!("{kind}/{}", json["case"].as_str().unwrap());
            let param = &json["param"];
            let name = param["name"].as_str().unwrap();
            let shape: Vec<usize> = param["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| usize::try_from(d.as_u64().unwrap()).unwrap())
                .collect();
            let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            let fans = match &param["role"] {
                Value::Null => None,
                role => fans_of(
                    role["op"].as_str().unwrap(),
                    role["param"].as_str().unwrap(),
                    &dims,
                ),
            };
            let want_fans = match &param["fans"] {
                Value::Null => None,
                f => Some(Fans {
                    fan_in: f["fan_in"].as_u64().unwrap(),
                    fan_out: f["fan_out"].as_u64().unwrap(),
                }),
            };
            assert_eq!(fans, want_fans, "{case}: fans");
            let seed = json["seed"].as_u64().unwrap();
            let got = initial_values(seed, name, &shape, &init_of(&param["init"]), fans).unwrap();
            let file = SafetensorsFile::open(path.with_extension("safetensors")).unwrap();
            let view = file.tensor("values").unwrap();
            assert_eq!(view.shape(), shape.as_slice(), "{case}: stored shape");
            let (chunks, _) = view.data().as_chunks::<4>();
            let want: Vec<f32> = chunks.iter().map(|&b| f32::from_le_bytes(b)).collect();
            assert_eq!(got.len(), want.len(), "{case}");
            let allowed = json["tolerance"]["ulp"].as_u64().unwrap();
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                let d = ulps(*g, *w);
                assert!(
                    d <= allowed,
                    "{case}[{i}]: {g:e} vs {w:e} ({d} ulp, allowed {allowed})"
                );
                if d == 1 {
                    off_by_one += 1;
                }
            }
            total += 1;
        }
    }
    eprintln!("SUMMARY init_reference: {total} cases passed, {off_by_one} elements 1 ulp off");
    assert_eq!(total, 45, "every committed case ran");
}
