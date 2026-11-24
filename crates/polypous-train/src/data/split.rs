//! The `split` validation set (`docs/specs/training.md` §3.5.5).

use polypous_ir::rng::stream_key;

use super::batch::{SPLIT_LABEL, gather, permutation, sample_bytes};
use super::{DataError, Dataset};

/// Splits `train` into the training and validation sets of a `split{fraction}` (§3.5.5):
/// `n_val = ⌊n · fraction⌋` (in `f64`) samples, `1 ≤ n_val ≤ n − 1`, chosen as `π(0) … π(n_val−1)`
/// of the permutation with key `stream_key(seed, 0, "split:validation")`. Both sets keep the
/// training set's dataset order.
///
/// The validation samples are copied out (fallibly), then the training fields are compacted in
/// place, so the peak memory is `n + n_val` samples and the result holds `n`.
///
/// # Errors
/// `InvalidInput` if `n_val` is out of range or `fraction` isn't in `(0, 1)`; `OutOfMemory`.
pub fn split(train: Dataset, fraction: f64, seed: u64) -> Result<(Dataset, Dataset), DataError> {
    let n = train.len();
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(DataError::invalid(format!(
            "`fraction` must be in (0, 1), got {fraction}"
        )));
    }
    // n ≤ 2^32 is exact in f64, and the product is in [0, n), so the floor fits u64.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        reason = "0 ≤ n · fraction < n ≤ 2^32"
    )]
    let n_val = (n as f64 * fraction).floor() as u64;
    if n_val < 1 || n_val > n - 1 {
        return Err(DataError::invalid(format!(
            "a split with `fraction` {fraction} of {n} training samples gives {n_val} validation \
             samples; it must give 1 to n − 1 = {}",
            n - 1
        )));
    }
    let mut order = permutation(stream_key(seed, 0, SPLIT_LABEL), n)?;
    // n_val < n = order.len().
    #[allow(clippy::cast_possible_truncation, reason = "n_val < order.len()")]
    let (val, rest) = order.split_at_mut(n_val as usize);
    val.sort_unstable();
    rest.sort_unstable();
    let validation = gather(&train, val)?;
    let mut fields = train.into_fields();
    for data in fields.values_mut() {
        let sample = sample_bytes(data, n);
        for (to, &from) in rest.iter().enumerate() {
            // `rest` is ascending, so `from ≥ to` and the copy only moves data forward.
            let src = from as usize * sample;
            data.bytes.copy_within(src..src + sample, to * sample);
        }
        data.bytes.truncate(rest.len() * sample);
        data.bytes.shrink_to_fit();
        if let Some(first) = data.shape.first_mut() {
            *first = n - n_val;
        }
    }
    Ok((
        Dataset::from_checked(n - n_val, fields),
        Dataset::from_checked(n_val, validation),
    ))
}
