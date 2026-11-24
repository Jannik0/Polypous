//! Unit tests of [`read_datasets`] with `tensors` datasets, the split, the fingerprint and
//! batching (training.md §3.5.1, §3.5.5, §3.6, §5.2).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;

fn f32_field(shape: &[u64], values: &[f32]) -> TensorData {
    TensorData::new(
        DType::F32,
        shape.to_vec(),
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn i64_field(values: &[i64]) -> TensorData {
    TensorData::new(
        DType::I64,
        vec![values.len() as u64],
        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .unwrap()
}

fn f32s(t: &TensorData) -> Vec<f32> {
    let (chunks, _) = t.bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn i64s(t: &TensorData) -> Vec<i64> {
    let (chunks, _) = t.bytes.as_chunks::<8>();
    chunks.iter().map(|c| i64::from_le_bytes(*c)).collect()
}

fn config(json: &str) -> DataConfig {
    serde_json::from_str(json).unwrap()
}

/// A dataset `d` of `n` samples: `x` = `[i, 10 i]` (f32, sample shape [2]), `y` = `i` (i64).
fn numbered(n: i64) -> InMemoryDataset {
    let x: Vec<f32> = (0..n).flat_map(|i| [i as f32, 10.0 * i as f32]).collect();
    let y: Vec<i64> = (0..n).collect();
    InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![
            ("x".to_owned(), f32_field(&[n as u64, 2], &x)),
            ("y".to_owned(), i64_field(&y)),
        ],
    }
}

fn read(json: &str, in_memory: Vec<InMemoryDataset>, seed: u64) -> Result<Datasets, DataError> {
    read_with(json, in_memory, FieldSelection::All, seed)
}

fn read_with(
    json: &str,
    in_memory: Vec<InMemoryDataset>,
    fields: FieldSelection,
    seed: u64,
) -> Result<Datasets, DataError> {
    read_datasets(
        &config(json),
        seed,
        DataSources {
            base_dir: PathBuf::new(),
            in_memory,
            fields,
        },
        &CancellationToken::new(),
    )
}

const TENSORS: &str = r#"{"train": {"kind": "tensors", "name": "d"}}"#;

#[test]
fn tensors_dataset() {
    let sets = read(TENSORS, vec![numbered(3)], 0).unwrap();
    assert!(sets.validation.is_none());
    let train = &sets.train;
    assert_eq!(train.len(), 3);
    assert_eq!(train.field_info("x"), Some((DType::F32, &[2u64][..])));
    assert_eq!(train.field_info("y"), Some((DType::I64, &[][..])));
    assert_eq!(
        f32s(train.field("x").unwrap()),
        [0.0, 0.0, 1.0, 10.0, 2.0, 20.0]
    );
    assert_eq!(train.byte_len(), 3 * 8 + 3 * 8);
    assert_eq!(
        train.fields().map(|(n, _)| n).collect::<Vec<_>>(),
        ["x", "y"]
    );
}

#[test]
fn tensors_errors() {
    let err = read(
        r#"{"train": {"kind": "tensors", "name": "other"}}"#,
        vec![numbered(3)],
        0,
    )
    .unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::InvalidInput);
    assert_eq!(err.dataset(), Some("train"));
    assert!(err.message().contains("no dataset named `other`"), "{err}");

    let mut twice = numbered(3);
    twice
        .fields
        .push(("x".to_owned(), f32_field(&[3], &[0.0; 3])));
    let err = read(TENSORS, vec![twice], 0).unwrap_err();
    assert!(err.message().contains("two fields named `x`"), "{err}");

    let mut count = numbered(3);
    count.fields.push(("z".to_owned(), i64_field(&[1, 2])));
    let err = read(TENSORS, vec![count], 0).unwrap_err();
    assert_eq!(err.field(), Some("z"));
    assert!(err.message().contains("same sample count"), "{err}");

    let scalar = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![("s".to_owned(), f32_field(&[], &[1.0]))],
    };
    let err = read(TENSORS, vec![scalar], 0).unwrap_err();
    assert!(err.message().contains("rank 0"), "{err}");
    let rank8 = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![("s".to_owned(), f32_field(&[1; 8], &[1.0]))],
    };
    assert!(
        read(TENSORS, vec![rank8], 0)
            .unwrap_err()
            .message()
            .contains("rank 8")
    );
    let rank7 = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![("s".to_owned(), f32_field(&[1; 7], &[1.0]))],
    };
    assert_eq!(read(TENSORS, vec![rank7], 0).unwrap().train.len(), 1);

    let empty = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![("s".to_owned(), f32_field(&[0, 2], &[]))],
    };
    assert!(
        read(TENSORS, vec![empty], 0)
            .unwrap_err()
            .message()
            .contains("1 to 2^32")
    );

    // Malformed data (public-api.md §2 rule 4): wrong byte length, a bool byte of 2.
    let bad = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![(
            "s".to_owned(),
            TensorData {
                dtype: DType::F32,
                shape: vec![2],
                bytes: vec![0; 7],
            },
        )],
    };
    assert!(
        read(TENSORS, vec![bad], 0)
            .unwrap_err()
            .message()
            .contains("malformed")
    );
    let bad_bool = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![(
            "b".to_owned(),
            TensorData {
                dtype: DType::Bool,
                shape: vec![2],
                bytes: vec![1, 2],
            },
        )],
    };
    assert!(read(TENSORS, vec![bad_bool], 0).is_err());

    let none = read_with(
        TENSORS,
        vec![numbered(3)],
        FieldSelection::Only(["q".to_owned()].into()),
        0,
    )
    .unwrap_err();
    assert!(none.message().contains("none of the fields"), "{none}");
}

#[test]
fn tensors_widen_f16_and_bf16_and_skip_unused_fields() {
    // f16: 1.0, -2.0, the smallest subnormal 2^-24, 65504; bf16: 1.0, -0.5.
    let f16 = TensorData::new(
        DType::F16,
        vec![4],
        [0x3C00u16, 0xC000, 0x0001, 0x7BFF]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .unwrap();
    let bf16 = TensorData::new(
        DType::BF16,
        vec![4, 1],
        [0x3F80u16, 0xBF00, 0, 0x8000]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    )
    .unwrap();
    // An unused malformed field is ignored.
    let bad = TensorData {
        dtype: DType::F32,
        shape: vec![9],
        bytes: vec![],
    };
    let ds = InMemoryDataset {
        name: "d".to_owned(),
        fields: vec![
            ("h".to_owned(), f16),
            ("b".to_owned(), bf16),
            ("bad".to_owned(), bad),
        ],
    };
    let sets = read_with(
        TENSORS,
        vec![ds],
        FieldSelection::Only(["h".to_owned(), "b".to_owned()].into()),
        0,
    )
    .unwrap();
    let h = sets.train.field("h").unwrap();
    assert_eq!(h.dtype, DType::F32);
    assert_eq!(f32s(h), [1.0, -2.0, 2f32.powi(-24), 65504.0]);
    let b = sets.train.field("b").unwrap();
    assert_eq!((b.dtype, b.shape.as_slice()), (DType::F32, &[4u64, 1][..]));
    assert_eq!(f32s(b), [1.0, -0.5, 0.0, -0.0]);
    assert!(f32s(b)[3].is_sign_negative());
    assert!(sets.train.field("bad").is_none());
}

#[test]
fn a_tensors_dataset_used_twice_is_copied() {
    let json = r#"{"train": {"kind": "tensors", "name": "d"},
                   "validation": {"kind": "tensors", "name": "d"}}"#;
    let sets = read(json, vec![numbered(4)], 0).unwrap();
    assert_eq!(Some(&sets.train), sets.validation.as_ref());
}

#[test]
fn max_bytes_covers_both_sets() {
    let both = |max: u64| {
        let json = format!(
            r#"{{"train": {{"kind": "tensors", "name": "d"}},
                "validation": {{"kind": "tensors", "name": "v"}}, "max_bytes": {max}}}"#
        );
        let mut v = numbered(2);
        v.name = "v".to_owned();
        read(&json, vec![numbered(3), v], 0)
    };
    // train: 3 · (8 + 8) = 48 bytes, validation: 2 · 16 = 32.
    assert!(both(80).is_ok());
    let err = both(79).unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::InvalidInput);
    assert_eq!(err.dataset(), Some("validation"));
    assert!(err.message().contains("max_bytes"), "{err}");
    let err = both(47).unwrap_err();
    assert_eq!(err.dataset(), Some("train"));
}

/// training.md §3.6's known split: seed 42, n = 10, fraction 0.3.
#[test]
fn split_known_values() {
    let json = r#"{"train": {"kind": "tensors", "name": "d"},
                   "validation": {"kind": "split", "fraction": 0.3}}"#;
    let sets = read(json, vec![numbered(10)], 42).unwrap();
    let val = sets.validation.as_ref().unwrap();
    assert_eq!(i64s(val.field("y").unwrap()), [4, 7, 8]);
    assert_eq!(
        f32s(val.field("x").unwrap()),
        [4.0, 40.0, 7.0, 70.0, 8.0, 80.0]
    );
    assert_eq!(val.field("x").unwrap().shape, [3, 2]);
    assert_eq!(i64s(sets.train.field("y").unwrap()), [0, 1, 2, 3, 5, 6, 9]);
    assert_eq!(sets.train.field("x").unwrap().shape, [7, 2]);
    assert_eq!(sets.train.len(), 7);
    assert_eq!(
        f32s(sets.train.field("x").unwrap()),
        [
            0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 5.0, 50.0, 6.0, 60.0, 9.0, 90.0
        ]
    );
    // The same split every time (and after a resume): a pure function of the seed.
    assert_eq!(read(json, vec![numbered(10)], 42).unwrap(), sets);
    // The fingerprint is of the data before the split.
    let whole = read(TENSORS, vec![numbered(10)], 42).unwrap();
    assert_eq!(whole.fingerprint, sets.fingerprint);
}

#[test]
fn split_counts() {
    let split = |n: i64, fraction: f64| {
        let json = format!(
            r#"{{"train": {{"kind": "tensors", "name": "d"}},
                "validation": {{"kind": "split", "fraction": {fraction}}}}}"#
        );
        read(&json, vec![numbered(n)], 1).map(|s| (s.train.len(), s.validation.unwrap().len()))
    };
    assert_eq!(split(10, 0.1).unwrap(), (9, 1));
    assert_eq!(split(10, 0.99).unwrap(), (1, 9));
    assert_eq!(split(2, 0.5).unwrap(), (1, 1));
    // ⌊n · fraction⌋ in f64: 0.3 · 10 = 3.0000000000000004 → 3; 0.7 · 10 = 7 → 7.
    assert_eq!(split(10, 0.7).unwrap(), (3, 7));
    let err = split(9, 0.1).unwrap_err();
    assert_eq!(err.dataset(), Some("validation"));
    assert!(
        err.message().contains("gives 0 validation samples"),
        "{err}"
    );
    assert!(split(1, 0.5).is_err());
}

/// §5.2: FNV-1a over field name, 0, dtype name, 0, shape (u64 LE), data, in name order.
#[test]
fn fingerprint() {
    let sets = read(TENSORS, vec![numbered(2)], 0).unwrap();
    let mut bytes = Vec::new();
    for (name, dtype, shape, data) in [
        (
            "x",
            "f32",
            vec![2u64, 2],
            [0f32, 0.0, 1.0, 10.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        ),
        (
            "y",
            "i64",
            vec![2u64],
            [0i64, 1].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ] {
        bytes.extend_from_slice(name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(dtype.as_bytes());
        bytes.push(0);
        shape
            .iter()
            .for_each(|d| bytes.extend_from_slice(&d.to_le_bytes()));
        bytes.extend_from_slice(&data);
    }
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x100_0000_01B3);
    }
    assert_eq!(sets.fingerprint, h);
    assert_eq!(sets.fingerprint_hex(), format!("{h:016x}"));
    assert_eq!(sets.fingerprint_hex().len(), 16);
    // The bytes-based hash agrees with polypous-ir's string hash.
    let mut f = Fnv1a::new();
    f.update(b"shuffle:train");
    assert_eq!(f.finish(), polypous_ir::rng::fnv1a("shuffle:train"));
    // One changed value changes it.
    let mut changed = numbered(2);
    changed.fields[1].1 = i64_field(&[0, 2]);
    assert_ne!(read(TENSORS, vec![changed], 0).unwrap().fingerprint, h);
}

#[test]
fn cancelled_before_reading() {
    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = read_datasets(
        &config(TENSORS),
        0,
        DataSources {
            in_memory: vec![numbered(2)],
            ..DataSources::default()
        },
        &cancel,
    )
    .unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::Cancelled);
}

#[test]
fn invalid_config_is_rejected() {
    let err = read(r#"{"train": {"kind": "tensors", "name": ""}}"#, vec![], 0).unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::InvalidInput);
    assert!(err.message().contains("/train/name"), "{err}");
}

/// Batches of epoch `e` are computed directly: they equal the samples at the epoch order's
/// positions, the last one is smaller, and together they cover every sample once.
#[test]
fn batches_of_any_epoch() {
    let sets = read(TENSORS, vec![numbered(10)], 0).unwrap();
    let train = &sets.train;
    let batcher = Batcher::new(train.len(), 4, true, 42).unwrap();
    assert_eq!(batcher.steps_per_epoch(), 3);
    // Epoch 1 directly, without computing epoch 0 (§3.6's known order).
    let order = batcher.order(1).unwrap();
    assert_eq!(order, [8, 5, 4, 6, 3, 0, 1, 2, 9, 7]);
    let last = batcher.batch(train, &order, 2).unwrap();
    assert_eq!(i64s(&last["y"]), [9, 7]);
    assert_eq!(last["y"].shape, [2]);
    assert_eq!(last["x"].shape, [2, 2]);
    assert_eq!(f32s(&last["x"]), [9.0, 90.0, 7.0, 70.0]);
    let first = batcher.batch(train, &order, 0).unwrap();
    assert_eq!(i64s(&first["y"]), [8, 5, 4, 6]);
    assert_eq!(first["x"].shape, [4, 2]);
    for e in [0, 3, 1000, u64::MAX] {
        let order = batcher.order(e).unwrap();
        let mut seen: Vec<i64> = (0..3)
            .flat_map(|b| i64s(&batcher.batch(train, &order, b).unwrap()["y"]))
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..10).collect::<Vec<_>>(), "epoch {e}");
    }
    assert!(batcher.batch(train, &order, 3).is_err());
    assert!(batcher.batch(train, &order[..9], 0).is_err());
    assert!(gather(train, &[10]).is_err());
    // shuffle: false is the dataset order in every epoch.
    let fixed = Batcher::new(10, 4, false, 42).unwrap();
    let b = fixed.batch(train, &fixed.order(7).unwrap(), 1).unwrap();
    assert_eq!(i64s(&b["y"]), [4, 5, 6, 7]);
}
