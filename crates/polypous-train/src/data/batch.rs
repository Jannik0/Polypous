//! Shuffling and batching (`docs/specs/training.md` §3.6).
//!
//! The permutation is a Fisher–Yates shuffle driven by the Polypous generator
//! ([`polypous_ir::rng::draw`]), with the rejection rule of §3.6, in integer arithmetic only, so
//! it is the same on every OS and device. Epoch `e`'s order depends on `seed` and `e` alone, and
//! batch `b` covers fixed positions of it, so a resumed run computes the order of its epoch
//! directly ([`Batcher::order`], [`Batcher::positions`]) without replaying earlier epochs, and
//! needs no saved data state (§5.4).

use std::collections::BTreeMap;
use std::ops::Range;

use polypous_backend::TensorData;
use polypous_ir::rng::{draw, stream_key};

use super::{DataError, Dataset, MAX_SAMPLES, try_bytes};

/// The label of the training order's stream: `stream_key(seed, e, "shuffle:train")` (§1.3).
pub const SHUFFLE_LABEL: &str = "shuffle:train";
/// The label of the `split` stream: `stream_key(seed, 0, "split:validation")` (§1.3).
pub const SPLIT_LABEL: &str = "split:validation";

/// The permutation of `0 … n−1` for `key` (§3.6, "Permutation"): position `i` holds
/// `order[i]`. `n` is at most `2^32`, so every index fits in `u32`.
///
/// # Errors
/// `InvalidInput` if `n > 2^32`; `OutOfMemory` if the `4 · n` bytes can't be reserved.
pub fn permutation(key: u64, n: u64) -> Result<Vec<u32>, DataError> {
    if n > MAX_SAMPLES {
        return Err(DataError::invalid(format!(
            "cannot shuffle {n} samples; a dataset has at most 2^32"
        )));
    }
    let len = usize::try_from(n).map_err(|_| DataError::out_of_memory(n * 4, "an order"))?;
    let mut order: Vec<u32> = Vec::new();
    order
        .try_reserve_exact(len)
        .map_err(|_| DataError::out_of_memory(n * 4, "an order"))?;
    // Every index is below n ≤ 2^32, so the conversion never fails; `take(len)` keeps the loop
    // panic-free regardless.
    order.extend((0..=u32::MAX).take(len));
    let mut c = 0u64;
    fisher_yates(&mut order, || {
        let z = draw(key, c);
        c += 1;
        z
    });
    Ok(order)
}

/// The loop of §3.6 on `order`, with `next` giving `draw(key, c)` for `c = 0, 1, …`.
fn fisher_yates(order: &mut [u32], mut next: impl FnMut() -> u64) {
    for i in (1..order.len()).rev() {
        let m = i as u64 + 1;
        // (2^64 − m) mod m = 2^64 mod m.
        let t = m.wrapping_neg() % m;
        let z = loop {
            let z = next();
            if z >= t {
                break z;
            }
        };
        // `z mod m ≤ i < order.len()`, so the cast and the index are in range.
        #[allow(clippy::cast_possible_truncation, reason = "z mod m < m ≤ order.len()")]
        let j = (z % m) as usize;
        order.swap(i, j);
    }
}

/// The batches of a dataset of `len` samples, `batch_size` samples per batch (the last one
/// possibly smaller), and the epoch orders of §3.6.
///
/// It holds no data and no state besides its four numbers: [`Batcher::order`] computes epoch
/// `e`'s order from `seed` and `e`, and [`Batcher::positions`] gives batch `b`'s positions in
/// it, so any batch of any epoch can be computed directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Batcher {
    len: u64,
    batch_size: u64,
    shuffle: bool,
    seed: u64,
}

impl Batcher {
    /// A batcher for the training set: shuffled every epoch if `shuffle` (`data.shuffle`).
    ///
    /// # Errors
    /// `InvalidInput` if `len` is not 1 to `2^32`, or `batch_size` is 0.
    pub fn new(len: u64, batch_size: u64, shuffle: bool, seed: u64) -> Result<Self, DataError> {
        if !(1..=MAX_SAMPLES).contains(&len) {
            return Err(DataError::invalid(format!(
                "a dataset has 1 to 2^32 samples, got {len}"
            )));
        }
        if batch_size == 0 {
            return Err(DataError::invalid("`batch_size` must be at least 1"));
        }
        Ok(Self {
            len,
            batch_size,
            shuffle,
            seed,
        })
    }

    /// A batcher for the validation set, which is never shuffled (§3.6).
    ///
    /// # Errors
    /// As [`Batcher::new`].
    pub fn validation(len: u64, batch_size: u64) -> Result<Self, DataError> {
        Self::new(len, batch_size, false, 0)
    }

    /// The sample count `n`.
    #[must_use]
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Always `false` (a batcher has at least one sample).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// `S = ⌈n / batch_size⌉`, the batches (steps) per epoch.
    #[must_use]
    pub fn steps_per_epoch(&self) -> u64 {
        self.len.div_ceil(self.batch_size)
    }

    /// The key of epoch `e`'s order: `stream_key(seed, e, "shuffle:train")`.
    #[must_use]
    pub fn epoch_key(&self, epoch: u64) -> u64 {
        stream_key(self.seed, epoch, SHUFFLE_LABEL)
    }

    /// Epoch `e`'s order: position `i` holds dataset sample `order[i]`. The permutation of
    /// [`Batcher::epoch_key`] with shuffling, else the dataset order.
    ///
    /// # Errors
    /// `OutOfMemory` if the order can't be allocated.
    pub fn order(&self, epoch: u64) -> Result<Vec<u32>, DataError> {
        if self.shuffle {
            permutation(self.epoch_key(epoch), self.len)
        } else {
            permutation_identity(self.len)
        }
    }

    /// The positions of batch `b`: `b · batch_size` to `min((b + 1) · batch_size, n) − 1`.
    /// `None` if `b ≥ S`.
    #[must_use]
    pub fn positions(&self, batch: u64) -> Option<Range<u64>> {
        if batch >= self.steps_per_epoch() {
            return None;
        }
        // b < S = ⌈n / B⌉, so b · B < n ≤ 2^32 and nothing overflows.
        let start = batch * self.batch_size;
        Some(start..self.len.min(start + self.batch_size))
    }

    /// Batch `b` of the epoch whose order is `order` ([`Batcher::order`]), gathered from
    /// `dataset`: each field as `[B] + sample shape` ([`gather`]).
    ///
    /// # Errors
    /// `InvalidInput` if `b ≥ S` or `order` doesn't fit the dataset; `OutOfMemory`.
    pub fn batch(
        &self,
        dataset: &Dataset,
        order: &[u32],
        batch: u64,
    ) -> Result<BTreeMap<String, TensorData>, DataError> {
        if dataset.len() != self.len || order.len() as u64 != self.len {
            return Err(DataError::invalid(format!(
                "the order has {} entries and the dataset {} samples, but the batcher was made \
                 for {}",
                order.len(),
                dataset.len(),
                self.len
            )));
        }
        let range = self.positions(batch).ok_or_else(|| {
            DataError::invalid(format!(
                "batch {batch} doesn't exist: an epoch has {} batches",
                self.steps_per_epoch()
            ))
        })?;
        // The range is within 0..len = 0..order.len(), which fits usize.
        #[allow(clippy::cast_possible_truncation, reason = "end ≤ order.len()")]
        let indices = order
            .get(range.start as usize..range.end as usize)
            .ok_or_else(|| DataError::invalid("batch positions out of range"))?;
        gather(dataset, indices)
    }
}

fn permutation_identity(n: u64) -> Result<Vec<u32>, DataError> {
    let len = usize::try_from(n).map_err(|_| DataError::out_of_memory(n * 4, "an order"))?;
    let mut order: Vec<u32> = Vec::new();
    order
        .try_reserve_exact(len)
        .map_err(|_| DataError::out_of_memory(n * 4, "an order"))?;
    order.extend((0..=u32::MAX).take(len));
    Ok(order)
}

/// The samples `indices` of `dataset`, in that order, stacked per field into a [`TensorData`]
/// of shape `[indices.len()] + sample shape` (§3.5: a batch).
///
/// # Errors
/// `InvalidInput` if an index is not below the dataset's length; `OutOfMemory`.
pub fn gather(
    dataset: &Dataset,
    indices: &[u32],
) -> Result<BTreeMap<String, TensorData>, DataError> {
    if let Some(&bad) = indices.iter().find(|&&i| u64::from(i) >= dataset.len()) {
        return Err(DataError::invalid(format!(
            "sample {bad} doesn't exist: the dataset has {} samples",
            dataset.len()
        )));
    }
    let mut out = BTreeMap::new();
    for (name, data) in dataset.fields() {
        let sample = sample_bytes(data, dataset.len());
        let total = (sample as u64)
            .checked_mul(indices.len() as u64)
            .ok_or_else(|| DataError::out_of_memory(u64::MAX, "a batch"))?;
        let mut bytes = try_bytes(total, "a batch")?;
        for &i in indices {
            let start = i as usize * sample;
            // `i < n` and the field has `n · sample` bytes (a validated dataset field).
            let chunk = data
                .bytes
                .get(start..start + sample)
                .ok_or_else(|| DataError::invalid("a field is shorter than its sample count"))?;
            bytes.extend_from_slice(chunk);
        }
        let mut shape = data.shape.clone();
        if let Some(first) = shape.first_mut() {
            *first = indices.len() as u64;
        }
        out.insert(
            name.to_owned(),
            TensorData {
                dtype: data.dtype,
                shape,
                bytes,
            },
        );
    }
    Ok(out)
}

/// The bytes of one sample of a field of `n` samples.
pub(crate) fn sample_bytes(data: &TensorData, n: u64) -> usize {
    // A dataset field has `n · sample` bytes with n ≥ 1, so the division is exact.
    usize::try_from(data.bytes.len() as u64 / n.max(1)).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// training.md §3.6, "Known values".
    #[test]
    fn known_values() {
        let key = stream_key(42, 0, SHUFFLE_LABEL);
        assert_eq!(key, 0xB220_B916_172E_A32D);
        assert_eq!(draw(key, 0), 0x1C96_72AD_B157_0786);
        assert_eq!(draw(key, 1), 0x9AB7_1BAF_68C6_2E49);
        assert_eq!(draw(key, 2), 0x6647_5AAA_DA25_2698);
        assert_eq!(
            permutation(key, 10).unwrap(),
            [8, 3, 4, 1, 5, 7, 2, 9, 6, 0]
        );
        let key1 = stream_key(42, 1, SHUFFLE_LABEL);
        assert_eq!(key1, 0x3105_47A3_638A_4BFA);
        assert_eq!(
            permutation(key1, 10).unwrap(),
            [8, 5, 4, 6, 3, 0, 1, 2, 9, 7]
        );
        let split = stream_key(42, 0, SPLIT_LABEL);
        assert_eq!(split, 0x412B_830F_89CE_0DDF);
        assert_eq!(
            permutation(split, 10).unwrap(),
            [7, 4, 8, 2, 3, 5, 0, 1, 9, 6]
        );
        // The batcher gives the same orders, epoch 1 directly.
        let batcher = Batcher::new(10, 4, true, 42).unwrap();
        assert_eq!(batcher.order(1).unwrap(), [8, 5, 4, 6, 3, 0, 1, 2, 9, 7]);
        assert_eq!(batcher.order(0).unwrap(), [8, 3, 4, 1, 5, 7, 2, 9, 6, 0]);
    }

    /// An independent transcription of §3.6's pseudocode with u128 arithmetic for `2^64 mod m`.
    fn reference(key: u64, n: u64) -> Vec<u32> {
        let mut order: Vec<u32> = (0..n as u32).collect();
        let mut c = 0;
        for i in (1..n).rev() {
            let m = i + 1;
            let t = ((1u128 << 64) % u128::from(m)) as u64;
            let mut z = draw(key, c);
            c += 1;
            while z < t {
                z = draw(key, c);
                c += 1;
            }
            order.swap(i as usize, (z % m) as usize);
        }
        order
    }

    #[test]
    fn matches_the_pseudocode() {
        for n in [1, 2, 3, 7, 100, 1000] {
            for e in 0..5 {
                let key = stream_key(3, e, SHUFFLE_LABEL);
                assert_eq!(permutation(key, n).unwrap(), reference(key, n), "n = {n}");
            }
        }
        assert_eq!(permutation(5, 0).unwrap(), Vec::<u32>::new());
        assert_eq!(permutation(5, 1).unwrap(), [0]);
        assert!(permutation(5, MAX_SAMPLES + 1).is_err());
    }

    /// The rejection rule: `t = 2^64 mod m` (1 for `m = 3`, 0 for `m = 2`, 4 for `m = 6`), and a
    /// draw `z < t` is skipped, consuming a counter value. Real draws are rejected with
    /// probability below `m / 2^64`, so a scripted draw sequence tests it.
    #[test]
    fn rejection_rule() {
        let mut draws = [0u64, 5, 0, 9].into_iter();
        let mut order = [0, 1, 2];
        // i = 2 (m = 3, t = 1): 0 rejected, 5 accepted, j = 2. i = 1 (m = 2, t = 0): 0, j = 0.
        fisher_yates(&mut order, || draws.next().unwrap());
        assert_eq!(order, [1, 0, 2]);
        assert_eq!(draws.next(), Some(9), "exactly three draws used");
        assert_eq!(3u64.wrapping_neg() % 3, 1);
        assert_eq!(5u64.wrapping_neg() % 5, 1);
        assert_eq!(6u64.wrapping_neg() % 6, 4);
        // m = 6 (t = 4): 3 rejected, 4 accepted, j = 4. m = 5 (t = 1): 7, j = 2. m = 4 (t = 0):
        // 1, j = 1. m = 3 (t = 1): 1 accepted, j = 1. m = 2: 1, j = 1.
        let mut draws = [3u64, 4, 7, 1, 1, 1, 99].into_iter();
        let mut order = [0, 1, 2, 3, 4, 5];
        fisher_yates(&mut order, || draws.next().unwrap());
        assert_eq!(order, [0, 5, 3, 1, 2, 4]);
        assert_eq!(draws.next(), Some(99));
    }

    /// testing-strategy.md §3.8: n = 4, 24,000 epochs of seed 7, chi-square over the 24
    /// permutations ≤ 56.9 (23 degrees of freedom). The Python implementation gives 20.2.
    #[test]
    fn shuffle_uniformity() {
        let batcher = Batcher::new(4, 1, true, 7).unwrap();
        let mut counts = BTreeMap::<Vec<u32>, u64>::new();
        for e in 0..24_000 {
            *counts.entry(batcher.order(e).unwrap()).or_default() += 1;
        }
        assert_eq!(counts.len(), 24);
        let expected = 1000.0;
        let chi2: f64 = counts
            .values()
            .map(|&c| (c as f64 - expected).powi(2) / expected)
            .sum();
        assert!(chi2 <= 56.9, "chi-square {chi2}");
        // The value is deterministic; the spec's Python value is 20.2 (to one decimal).
        assert!(
            (chi2 - 20.2).abs() < 0.05,
            "chi-square {chi2}, Python gives 20.2"
        );
    }

    #[test]
    fn batches_keep_the_last_smaller_one() {
        let b = Batcher::new(10, 4, false, 0).unwrap();
        assert_eq!(b.steps_per_epoch(), 3);
        assert_eq!(b.positions(0), Some(0..4));
        assert_eq!(b.positions(1), Some(4..8));
        assert_eq!(b.positions(2), Some(8..10));
        assert_eq!(b.positions(3), None);
        assert_eq!(b.order(5).unwrap(), (0..10).collect::<Vec<u32>>());
        let exact = Batcher::new(8, 4, true, 0).unwrap();
        assert_eq!(exact.steps_per_epoch(), 2);
        assert_eq!(exact.positions(1), Some(4..8));
        let one = Batcher::new(1, 64, true, 0).unwrap();
        assert_eq!(one.steps_per_epoch(), 1);
        assert_eq!(one.positions(0), Some(0..1));
        let big = Batcher::new(MAX_SAMPLES, 1, true, 0).unwrap();
        assert_eq!(big.steps_per_epoch(), MAX_SAMPLES);
        assert_eq!(
            big.positions(MAX_SAMPLES - 1),
            Some(MAX_SAMPLES - 1..MAX_SAMPLES)
        );
        assert!(Batcher::new(0, 1, true, 0).is_err());
        assert!(Batcher::new(MAX_SAMPLES + 1, 1, true, 0).is_err());
        assert!(Batcher::new(5, 0, true, 0).is_err());
    }

    #[test]
    fn validation_is_never_shuffled() {
        let b = Batcher::validation(10, 3).unwrap();
        for e in 0..3 {
            assert_eq!(b.order(e).unwrap(), (0..10).collect::<Vec<u32>>());
        }
    }
}
