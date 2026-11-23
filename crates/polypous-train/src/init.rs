//! Initial values of params (`docs/specs/graph-ir.md` §2.4.1, §3.1.12; training.md §1.3,
//! §2.10 step 9), computed on the host, never on a device.
//!
//! Each param has its own stream of the Polypous generator, keyed by the training seed and its
//! name: `stream_key(seed, 0, "init:" + name)` ([`polypous_ir::rng`]). Element `i` (row-major)
//! uses `U(i)` ([`rng::u53`] of draw `i`) or `N(i)` ([`rng::normal`]), so the values don't
//! depend on the device, on other params or on the order params are initialized in. Every value
//! is computed in `f64` exactly as the spec's table writes it, in the same order of operations as
//! `torch.nn.init`, and rounded once to `f32` (round to nearest, ties to even).
//!
//! `zeros`, `ones`, `constant`, `uniform`, `kaiming_uniform` and `xavier_uniform` are
//! bit-identical on every platform. `normal`, `kaiming_normal` and `xavier_normal` use the
//! platform's `ln` and `cos`, and may differ by 1 ulp in rare elements (§3.1.12, "Exactness").

use polypous_ir::rng;
use polypous_ir::{Fans, Init};
use thiserror::Error;

/// The prefix of an init stream's label (training.md §1.3): `"init:"` + the param name.
pub const INIT_LABEL_PREFIX: &str = "init:";

/// Why [`initial_values`] could not compute a param's values.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum InitError {
    /// A fan-based kind (`kaiming_*`, `xavier_*`) without fans. Validation guarantees the fans of
    /// every param with such an `init` ([`polypous_ir::Analysis::param_fans`]), so this means the
    /// caller passed none.
    #[error("param `{name}`: init `{kind}` needs the param's fans")]
    MissingFans {
        /// The param name.
        name: String,
        /// The init kind.
        kind: &'static str,
    },
    /// A fan is zero (`kaiming_*` divide by `sqrt(fan_in)`, `xavier_*` by `fan_in + fan_out`).
    /// Validated params have no zero dims, so their fans are never zero.
    #[error("param `{name}`: init `{kind}` needs non-zero fans, got {fans}")]
    ZeroFans {
        /// The param name.
        name: String,
        /// The init kind.
        kind: &'static str,
        /// The fans given.
        fans: Fans,
    },
    /// The element count of the shape overflows, or the values can't be allocated.
    #[error("param `{name}`: shape {shape:?} is too large for its initial values")]
    TooLarge {
        /// The param name.
        name: String,
        /// The shape.
        shape: Vec<usize>,
    },
}

/// A fan as `f64`: round to nearest (§2.4.1). Fans are far below `2^53`, so this is exact in
/// practice.
#[allow(
    clippy::cast_precision_loss,
    reason = "the spec converts fans to f64 with round to nearest"
)]
fn fan(v: u64) -> f64 {
    v as f64
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the single rounding to f32 of graph-ir.md §2.4.1"
)]
fn round(v: f64) -> f32 {
    v as f32
}

/// The distribution of a param's elements, with its parameters already computed in `f64`.
enum Dist {
    Constant(f64),
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std: f64 },
}

/// The `f32` initial values of the param `name` with `shape` and `init`, for the training `seed`
/// (graph-ir.md §2.4.1), in row-major order. `fans` are the param's fans from the analysis
/// ([`polypous_ir::Analysis::param_fans`]); only the fan-based kinds read them.
///
/// | Kind | Element `i` (in `f64`, rounded once) |
/// |---|---|
/// | `zeros`, `ones`, `constant{value}` | `0`, `1`, `value` |
/// | `uniform{low, high}` | `low + (high - low) · U(i)` |
/// | `normal{mean, std}` | `mean + std · N(i)` |
/// | `kaiming_uniform{a}` | `uniform(-bound, bound)`, `gain = sqrt(2 / (1 + a · a))`, `std = gain / sqrt(fan_in)`, `bound = sqrt(3) · std` |
/// | `kaiming_normal` | `normal(0, gain / sqrt(fan_in))`, `gain = sqrt(2 / (1 + 0))` |
/// | `xavier_uniform` | `uniform(-bound, bound)`, `std = 1 · sqrt(2 / (fan_in + fan_out))`, `bound = sqrt(3) · std` |
/// | `xavier_normal` | `normal(0, 1 · sqrt(2 / (fan_in + fan_out)))` |
///
/// `fan_in + fan_out` is summed as integers and then converted to `f64`, as
/// `torch.nn.init._calculate_fan_in_and_fan_out` and `xavier_*_` do. The element count is
/// computed with checked arithmetic and the values are allocated fallibly: an overflowing or
/// unallocatable shape is [`InitError::TooLarge`], never a panic.
pub fn initial_values(
    seed: u64,
    name: &str,
    shape: &[usize],
    init: &Init,
    fans: Option<Fans>,
) -> Result<Vec<f32>, InitError> {
    let kind = init.kind();
    let needs_fans = || -> Result<Fans, InitError> {
        let f = fans.ok_or_else(|| InitError::MissingFans {
            name: name.to_owned(),
            kind,
        })?;
        if f.fan_in == 0 || f.fan_out == 0 {
            return Err(InitError::ZeroFans {
                name: name.to_owned(),
                kind,
                fans: f,
            });
        }
        Ok(f)
    };
    let sqrt3 = 3.0f64.sqrt();
    let dist = match *init {
        Init::Zeros => Dist::Constant(0.0),
        Init::Ones => Dist::Constant(1.0),
        Init::Constant { value } => Dist::Constant(value),
        Init::Uniform { low, high } => Dist::Uniform { low, high },
        Init::Normal { mean, std } => Dist::Normal { mean, std },
        Init::KaimingUniform { a } => {
            let f = needs_fans()?;
            let a = a.unwrap_or(0.0);
            let gain = (2.0 / (1.0 + a * a)).sqrt();
            let std = gain / fan(f.fan_in).sqrt();
            let bound = sqrt3 * std;
            Dist::Uniform {
                low: -bound,
                high: bound,
            }
        }
        Init::KaimingNormal => {
            let f = needs_fans()?;
            let gain = (2.0f64 / (1.0 + 0.0)).sqrt();
            Dist::Normal {
                mean: 0.0,
                std: gain / fan(f.fan_in).sqrt(),
            }
        }
        Init::XavierUniform | Init::XavierNormal => {
            let f = needs_fans()?;
            let sum = f
                .fan_in
                .checked_add(f.fan_out)
                .ok_or_else(|| InitError::TooLarge {
                    name: name.to_owned(),
                    shape: shape.to_vec(),
                })?;
            let std = 1.0 * (2.0 / fan(sum)).sqrt();
            if matches!(init, Init::XavierUniform) {
                let bound = sqrt3 * std;
                Dist::Uniform {
                    low: -bound,
                    high: bound,
                }
            } else {
                Dist::Normal { mean: 0.0, std }
            }
        }
    };

    let too_large = || InitError::TooLarge {
        name: name.to_owned(),
        shape: shape.to_vec(),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(too_large)?;
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| too_large())?;
    let key = rng::stream_key(seed, 0, &format!("{INIT_LABEL_PREFIX}{name}"));
    // usize → u64 is lossless on every supported target (at most 64-bit pointers).
    let indices = 0..count as u64;
    match dist {
        Dist::Constant(v) => out.resize(count, round(v)),
        Dist::Uniform { low, high } => {
            out.extend(indices.map(|i| round(low + (high - low) * rng::u53(rng::draw(key, i)))));
        }
        Dist::Normal { mean, std } => {
            out.extend(indices.map(|i| round(mean + std * rng::normal(key, i))));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_table_case() {
        // graph-ir.md §3.1.12, "Known values": kaiming_uniform (a = 0) of fc1.weight
        // [128, 784], seed 42, elements 0–2.
        let fans = Fans {
            fan_in: 784,
            fan_out: 128,
        };
        let v = initial_values(
            42,
            "fc1.weight",
            &[128, 784],
            &Init::KaimingUniform { a: None },
            Some(fans),
        )
        .unwrap();
        assert_eq!(v.len(), 128 * 784);
        assert_eq!(v[..3], [0.077_307_545, 0.079_219_59, -0.049_188_793]);
    }

    #[test]
    fn errors() {
        let w = Init::XavierUniform;
        assert_eq!(
            initial_values(0, "w", &[2, 2], &w, None),
            Err(InitError::MissingFans {
                name: "w".into(),
                kind: "xavier_uniform"
            })
        );
        let zero = Fans {
            fan_in: 0,
            fan_out: 2,
        };
        assert!(matches!(
            initial_values(0, "w", &[2, 2], &Init::KaimingNormal, Some(zero)),
            Err(InitError::ZeroFans { .. })
        ));
        assert!(matches!(
            initial_values(0, "w", &[usize::MAX, 2], &Init::Zeros, None),
            Err(InitError::TooLarge { .. })
        ));
        // Fits the count but not the memory: an allocation error, not an abort.
        assert!(matches!(
            initial_values(0, "w", &[usize::MAX / 4], &Init::Ones, None),
            Err(InitError::TooLarge { .. })
        ));
        let huge = Fans {
            fan_in: u64::MAX,
            fan_out: 1,
        };
        assert!(matches!(
            initial_values(0, "w", &[1], &Init::XavierNormal, Some(huge)),
            Err(InitError::TooLarge { .. })
        ));
    }
}
