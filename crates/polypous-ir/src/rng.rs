//! The Polypous random generator (graph-ir.md §3.1.12): SplitMix64 used as a counter-based hash,
//! on the host. Value `i` of a stream is a hash of the stream's key and `i`, so no generator state
//! is kept; the values are the same on every device, backend and platform, and don't depend on
//! threads or on the order in which nodes run (owner decision 2 for M2, ADR-0014).
//!
//! - [`mix`], [`fnv1a`], [`stream_key`], [`draw`]: the integer core, exactly as the spec writes
//!   it (wrapping `u64` arithmetic).
//! - [`u24`] and [`u53`]: uniform values in `[0, 1)` from a draw, exact in `f64`.
//! - [`normal`]: the standard normal value `N(i)` of a stream (Box–Muller). Its `ln` and `cos`
//!   come from the platform's math library, so it may differ between platforms in the last bit of
//!   the `f64` result (§3.1.12, "Exactness").
//! - [`dropout_keep`] and [`dropout_mask`]: the keep rule of `nn.dropout` in training mode
//!   (graph-ir.md §3.3): element `i` is kept if `u24(draw(key, i)) ≥ p`, compared in `f64`.
//!
//! The initializers of §2.4.1 and the shuffle (training.md) build on these and live with their
//! users.
//!
//! ```
//! use polypous_ir::rng;
//!
//! // graph-ir.md §3.1.12, "Known values".
//! assert_eq!(rng::draw(0, 0), 0xE220_A839_7B1D_CDAF);
//! let key = rng::stream_key(42, 7, "dropout:layers[3]/drop");
//! assert_eq!(key, 0x559F_E239_6A2B_E18C);
//! let keep = rng::dropout_mask(42, 7, "layers[3]/drop", &[2, 2], 0.5).unwrap();
//! assert_eq!(keep, [true, true, true, false]); // u24: 0.87, 0.59, 0.77, 0.036
//! ```

use thiserror::Error;

/// SplitMix64's increment, the golden-ratio constant `0x9E3779B97F4A7C15`.
pub const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The prefix of a dropout stream's label (§3.1.12, "Streams"): `"dropout:"` + the node path.
pub const DROPOUT_LABEL_PREFIX: &str = "dropout:";

/// `2^-24`.
const TWO_POW_M24: f64 = 1.0 / (1u64 << 24) as f64;
/// `2^-53`.
const TWO_POW_M53: f64 = 1.0 / (1u64 << 53) as f64;

/// SplitMix64's finalizer.
#[must_use]
pub const fn mix(z: u64) -> u64 {
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The 64-bit FNV-1a hash of `label`'s bytes.
#[must_use]
pub fn fnv1a(label: &str) -> u64 {
    label.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    })
}

/// The key of a stream: `mix(mix(seed ^ GAMMA) ^ mix(step) ^ fnv1a(label))`.
#[must_use]
pub fn stream_key(seed: u64, step: u64, label: &str) -> u64 {
    mix(mix(seed ^ GAMMA) ^ mix(step) ^ fnv1a(label))
}

/// Draw `i` of the stream `key`: `mix(key + (i + 1) · GAMMA)`, wrapping.
#[must_use]
pub const fn draw(key: u64, i: u64) -> u64 {
    mix(key.wrapping_add(i.wrapping_add(1).wrapping_mul(GAMMA)))
}

/// `(z >> 40) · 2^-24`: uniform in `[0, 1)` on a grid of `2^-24`, exact in `f32` and `f64`.
/// Dropout uses it.
#[must_use]
pub fn u24(z: u64) -> f64 {
    // Below 2^24: exact in f64.
    #[allow(clippy::cast_precision_loss, reason = "the value is below 2^24")]
    let top = (z >> 40) as f64;
    top * TWO_POW_M24
}

/// `(z >> 11) · 2^-53`: uniform in `[0, 1)` on a grid of `2^-53`, exact in `f64` (the spec's
/// `U(i)` for `z = draw(key, i)`).
#[must_use]
pub fn u53(z: u64) -> f64 {
    #[allow(clippy::cast_precision_loss, reason = "the value is below 2^53")]
    let top = (z >> 11) as f64;
    top * TWO_POW_M53
}

/// `N(i)` of the stream `key`, standard normal by the Box–Muller transform (§3.1.12):
/// `sqrt(-2 · ln(v1)) · cos(TAU · v2)` with `v1 = ((draw(key, 2i) >> 11) + 1) · 2^-53` in
/// `(0, 1]` and `v2 = u53(draw(key, 2i + 1))`. `ln` and `cos` come from the platform's math
/// library. Indices of allocatable tensors are far below `2^63`, so `2i + 1` doesn't wrap (it
/// would wrap silently, never panic).
#[must_use]
pub fn normal(key: u64, i: u64) -> f64 {
    let first = i.wrapping_mul(2);
    #[allow(clippy::cast_precision_loss, reason = "the value is at most 2^53")]
    let v1 = ((draw(key, first) >> 11) + 1) as f64 * TWO_POW_M53;
    let v2 = u53(draw(key, first.wrapping_add(1)));
    (-2.0 * v1.ln()).sqrt() * (std::f64::consts::TAU * v2).cos()
}

/// Whether element `i` of a dropout stream is kept: `u24(draw(key, i)) ≥ p`, compared in `f64`
/// (graph-ir.md §3.3). `p = 0` keeps every element.
#[must_use]
pub fn dropout_keep(key: u64, i: u64, p: f64) -> bool {
    u24(draw(key, i)) >= p
}

/// Why [`dropout_mask`] could not build a mask.
#[derive(Clone, Debug, Error, PartialEq)]
#[non_exhaustive]
pub enum RngError {
    /// `p` is outside `[0, 1)` (NaN included).
    #[error("dropout `p` must be in [0, 1), got {p}")]
    InvalidProbability {
        /// The rejected probability.
        p: f64,
    },
    /// The element count of `shape` overflows, or the mask can't be allocated.
    #[error("a dropout mask of shape {shape:?} is too large")]
    TooLarge {
        /// The rejected shape.
        shape: Vec<usize>,
    },
}

/// The keep mask of `nn.dropout` in training mode for a tensor of `shape` (graph-ir.md §3.3,
/// §3.1.12): element `i` (row-major at `shape`) is `true` (kept) if
/// `u24(draw(key, i)) ≥ p`, with `key = stream_key(seed, step, "dropout:" + node_path)`.
/// `node_path` is the node path of runtime.md §5 (`drop1`, `layers[3]/drop`). `p = 0` keeps every
/// element and draws nothing. A shape without dims (a scalar) has one element.
///
/// The shape is untrusted: its element count is computed with checked arithmetic, and the mask
/// is allocated with [`Vec::try_reserve_exact`], so an overflowing or unallocatable shape is
/// [`RngError::TooLarge`], never a panic or an abort. `p` outside `[0, 1)` is
/// [`RngError::InvalidProbability`].
pub fn dropout_mask(
    seed: u64,
    step: u64,
    node_path: &str,
    shape: &[usize],
    p: f64,
) -> Result<Vec<bool>, RngError> {
    if !(0.0..1.0).contains(&p) {
        return Err(RngError::InvalidProbability { p });
    }
    let too_large = || RngError::TooLarge {
        shape: shape.to_vec(),
    };
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(too_large)?;
    let mut mask = Vec::new();
    mask.try_reserve_exact(count).map_err(|_| too_large())?;
    if p == 0.0 {
        mask.resize(count, true);
        return Ok(mask);
    }
    let key = stream_key(seed, step, &format!("{DROPOUT_LABEL_PREFIX}{node_path}"));
    // usize → u64 is lossless on every supported target (at most 64-bit pointers).
    mask.extend((0..count as u64).map(|i| dropout_keep(key, i, p)));
    Ok(mask)
}

#[cfg(test)]
#[path = "rng_tests.rs"]
mod tests;
