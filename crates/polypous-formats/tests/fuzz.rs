//! Fuzz-style robustness test of the safetensors reader (`docs/specs/formats.md` §1.2): the parser
//! runs on truncated and randomly mutated files and must return `Ok` or `Err`, never panic, hang
//! or over-allocate.
//!
//! Deterministic (fixed seed, own PRNG) and bounded: every prefix of each seed file plus
//! `MUTATIONS` mutated files. Accepted files are read completely (every view, `values`,
//! `to_f32_vec`). Longer fuzzing campaigns run outside CI (docs/testing-strategy.md §1).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use std::collections::BTreeMap;
use std::time::Instant;

use polypous_formats::safetensors::{Dtype, SafetensorsFile, WriteTensor, serialize};

const SEED: u64 = 0x5afe_7e45_0125_0037;
const MUTATIONS: usize = 20_000;

/// SplitMix64: small, deterministic, good enough for test-case generation.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn seed_files() -> Vec<Vec<u8>> {
    let python = std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/python_written.safetensors"),
    )
    .unwrap();
    let f32s: Vec<u8> = [1.0f32, -2.0, 0.5, 8.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let bools = [1u8, 0, 1];
    let halves = [0u8, 0x3c, 0, 0x7c];
    let meta = BTreeMap::from([("format".to_owned(), "pt".to_owned())]);
    let written = serialize(
        [
            ("w", WriteTensor::new(Dtype::F32, &[2, 2], &f32s).unwrap()),
            ("mask", WriteTensor::new(Dtype::Bool, &[3], &bools).unwrap()),
            ("h", WriteTensor::new(Dtype::F16, &[2], &halves).unwrap()),
            ("bf", WriteTensor::new(Dtype::BF16, &[2], &halves).unwrap()),
            ("e", WriteTensor::new(Dtype::I64, &[0], &[]).unwrap()),
            ("f4", WriteTensor::new(Dtype::F4, &[2], &[0x12]).unwrap()),
        ],
        &meta,
    )
    .unwrap();
    let empty = serialize(Vec::<(&str, WriteTensor<'_>)>::new(), &BTreeMap::new()).unwrap();
    vec![python, written, empty]
}

/// Parses `bytes` and, if accepted, reads every tensor. Returns whether it was accepted.
fn exercise(bytes: Vec<u8>) -> bool {
    let Ok(file) = SafetensorsFile::from_bytes(bytes) else {
        return false;
    };
    let names: Vec<String> = file.names().map(str::to_owned).collect();
    for name in &names {
        let view = file.tensor(name).expect("listed tensor must resolve");
        let info = file.info(name).expect("listed tensor has info");
        assert_eq!(view.data().len(), info.byte_len());
        let _ = view.values();
        let _ = view.to_f32_vec();
    }
    let _ = file.metadata().len();
    true
}

/// Characters that change JSON structure or numbers.
const JSON_BYTES: &[u8] = b"{}[]\":,0123456789-+eE. \\_azFIUB";

fn mutate(rng: &mut Rng, base: &[u8]) -> Vec<u8> {
    let mut b = base.to_vec();
    // Most of the structure is in the header: bias positions towards it.
    let header_end = b
        .first_chunk::<8>()
        .map(|p| u64::from_le_bytes(*p) as usize + 8)
        .unwrap_or(b.len())
        .min(b.len());
    // A position in `b` (0 if empty), three times out of four inside the original header.
    let pos = |rng: &mut Rng, b: &[u8]| -> usize {
        let p = if rng.below(4) != 0 && header_end > 8 {
            8 + rng.below(header_end - 8)
        } else {
            rng.below(b.len().max(1))
        };
        p.min(b.len().saturating_sub(1))
    };
    let rounds = 1 + rng.below(4);
    for _ in 0..rounds {
        match rng.below(10) {
            // Truncate.
            0 => {
                let len = rng.below(b.len() + 1);
                b.truncate(len);
            }
            // Flip random bits.
            1 => {
                if !b.is_empty() {
                    let p = pos(rng, &b);
                    if let Some(x) = b.get_mut(p) {
                        *x ^= 1 << rng.below(8);
                    }
                }
            }
            // Replace a byte with a JSON-significant one.
            2 => {
                if !b.is_empty() {
                    let p = pos(rng, &b);
                    let c = JSON_BYTES[rng.below(JSON_BYTES.len())];
                    if let Some(x) = b.get_mut(p) {
                        *x = c;
                    }
                }
            }
            // Insert a byte (shifts everything after it).
            3 => {
                let p = pos(rng, &b).min(b.len());
                b.insert(p, JSON_BYTES[rng.below(JSON_BYTES.len())]);
            }
            // Delete a range.
            4 => {
                if !b.is_empty() {
                    let p = pos(rng, &b).min(b.len() - 1);
                    let n = 1 + rng.below(8);
                    b.drain(p..(p + n).min(b.len()));
                }
            }
            // Overwrite the header length: random, near the real one, or huge.
            5 => {
                if b.len() >= 8 {
                    let real = u64::from_le_bytes(b[..8].try_into().unwrap());
                    let v = match rng.below(3) {
                        0 => rng.next(),
                        1 => real.wrapping_add(rng.below(17) as u64).wrapping_sub(8),
                        _ => u64::MAX - rng.below(4) as u64,
                    };
                    b[..8].copy_from_slice(&v.to_le_bytes());
                }
            }
            // Duplicate a range (repeats keys, digits, structures).
            6 => {
                if !b.is_empty() {
                    let p = pos(rng, &b).min(b.len() - 1);
                    let n = 1 + rng.below(32);
                    let chunk: Vec<u8> = b[p..(p + n).min(b.len())].to_vec();
                    let at = pos(rng, &b).min(b.len());
                    b.splice(at..at, chunk);
                }
            }
            // Change digits in the header: the JSON stays valid, so shapes and offsets change and
            // the size and contiguity rules are exercised.
            7 | 8 => {
                let digits: Vec<usize> = (8..header_end.min(b.len()))
                    .filter(|&i| b[i].is_ascii_digit())
                    .collect();
                for _ in 0..1 + rng.below(3) {
                    if !digits.is_empty() {
                        let i = digits[rng.below(digits.len())];
                        b[i] = b'0' + rng.below(10) as u8;
                    }
                }
            }
            // Replace a byte with a large number.
            _ => {
                let p = pos(rng, &b);
                let big = [
                    "18446744073709551615",
                    "99999999999999999999",
                    "4294967296",
                    "-1",
                ];
                let s = big[rng.below(big.len())].as_bytes();
                let end = (p + 1).min(b.len());
                b.splice(p.min(b.len())..end, s.iter().copied());
            }
        }
    }
    // Sometimes keep the length prefix consistent with an edited header, so mutations reach the
    // JSON and validation stages instead of failing at rule 3.
    if rng.below(2) == 0 && b.len() >= 8 {
        let delta = b.len() as i64 - base.len() as i64;
        let real = u64::from_le_bytes(base[..8].try_into().unwrap()) as i64;
        let fixed = (real + delta).max(0) as u64;
        b[..8].copy_from_slice(&fixed.to_le_bytes());
    }
    b
}

#[test]
fn parser_survives_truncated_and_mutated_files() {
    let start = Instant::now();
    let seeds = seed_files();
    for s in &seeds {
        assert!(exercise(s.clone()), "seed files must be valid");
    }

    let mut truncated = 0;
    for s in &seeds {
        for len in 0..s.len() {
            // A strict prefix is never valid: the data buffer must be covered exactly.
            assert!(
                !exercise(s[..len].to_vec()),
                "prefix of length {len} accepted"
            );
            truncated += 1;
        }
    }

    let mut rng = Rng(SEED);
    let mut accepted = 0;
    for _ in 0..MUTATIONS {
        let base = &seeds[rng.below(seeds.len())];
        if exercise(mutate(&mut rng, base)) {
            accepted += 1;
        }
    }
    assert!(accepted > 0, "mutations should not all be rejected");
    eprintln!(
        "fuzz: {truncated} truncations + {MUTATIONS} mutations ({accepted} accepted) in {:?}",
        start.elapsed()
    );
}
