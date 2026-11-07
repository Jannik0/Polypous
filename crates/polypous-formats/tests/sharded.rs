//! Sharded checkpoints (`docs/specs/formats.md` §1.5).

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::TestDir;
use polypous_formats::safetensors::{
    Dtype, ErrorKind, MAX_INDEX_LEN, ShardedSafetensors, WriteTensor, serialize,
};

/// Writes a shard holding `names`, each a one-byte U8 tensor with the value `value`.
fn shard(dir: &TestDir, file: &str, names: &[&str], value: u8) {
    let data = [value];
    let tensors: Vec<_> = names
        .iter()
        .map(|n| (*n, WriteTensor::new(Dtype::U8, &[1], &data).unwrap()))
        .collect();
    dir.write(file, &serialize(tensors, &BTreeMap::new()).unwrap());
}

fn index(dir: &TestDir, body: &str) -> PathBuf {
    dir.write("model.safetensors.index.json", body.as_bytes())
}

fn two_shards(dir: &TestDir) {
    shard(dir, "model-00001-of-00002.safetensors", &["a", "b"], 1);
    shard(dir, "model-00002-of-00002.safetensors", &["c"], 2);
}

const VALID_INDEX: &str = r#"{
  "metadata": {"total_size": 3, "format": "pt"},
  "weight_map": {
    "a": "model-00001-of-00002.safetensors",
    "b": "model-00001-of-00002.safetensors",
    "c": "model-00002-of-00002.safetensors"
  }
}"#;

#[test]
fn opens_a_sharded_checkpoint() {
    let dir = TestDir::new("sharded-valid");
    two_shards(&dir);
    let path = index(&dir, VALID_INDEX);
    let ckpt = ShardedSafetensors::open(&path).unwrap();
    assert_eq!(ckpt.index_path(), path.as_path());
    assert_eq!(ckpt.total_size(), Some(3));
    assert_eq!(ckpt.names().collect::<Vec<_>>(), ["a", "b", "c"]);
    assert_eq!(ckpt.len(), 3);
    // Each shard is opened once.
    assert_eq!(ckpt.shards().count(), 2);
    assert!(ckpt.shards().all(|s| s.is_memory_mapped()));
    assert_eq!(ckpt.shard_of("c"), Some("model-00002-of-00002.safetensors"));
    assert_eq!(ckpt.tensor("a").unwrap().data(), &[1]);
    assert_eq!(ckpt.tensor("c").unwrap().data(), &[2]);
    assert_eq!(ckpt.info("b").unwrap().dtype(), Dtype::U8);
    let err = ckpt.tensor("zzz").unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::TensorNotFound));
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn rejects_shard_paths_that_leave_the_directory() {
    let dir = TestDir::new("sharded-paths");
    two_shards(&dir);
    for (i, shard_path) in [
        "../model-00001-of-00002.safetensors",
        "/etc/passwd",
        "sub/model-00001-of-00002.safetensors",
        "..\\\\model-00001-of-00002.safetensors",
        "C:\\\\x.safetensors",
        "..",
        "",
    ]
    .iter()
    .enumerate()
    {
        let body = format!(r#"{{"weight_map":{{"a":"{shard_path}"}}}}"#);
        let path = dir.write(&format!("bad{i}.index.json"), body.as_bytes());
        let err = ShardedSafetensors::open(&path).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::InvalidShardPath { .. }),
            "{shard_path}: {err}"
        );
        assert_eq!(err.tensor(), Some("a"));
        assert_eq!(err.path(), Some(path.as_path()));
    }
}

#[test]
fn rejects_a_tensor_missing_from_its_shard() {
    let dir = TestDir::new("sharded-missing");
    two_shards(&dir);
    let body = VALID_INDEX.replace(
        r#""c": "model-00002-of-00002.safetensors""#,
        r#""c": "model-00002-of-00002.safetensors", "d": "model-00002-of-00002.safetensors""#,
    );
    let err = ShardedSafetensors::open(index(&dir, &body)).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::MissingInShard { .. }),
        "{err}"
    );
    assert_eq!(err.tensor(), Some("d"));
    assert_eq!(
        err.path(),
        Some(dir.join("model-00002-of-00002.safetensors").as_path())
    );
}

#[test]
fn rejects_a_tensor_in_more_than_one_shard() {
    let dir = TestDir::new("sharded-dup");
    shard(&dir, "model-00001-of-00002.safetensors", &["a", "b"], 1);
    shard(&dir, "model-00002-of-00002.safetensors", &["b", "c"], 2);
    let err = ShardedSafetensors::open(index(&dir, VALID_INDEX)).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::InMultipleShards { mapped, found }
            if mapped == "model-00001-of-00002.safetensors"
            && found == "model-00002-of-00002.safetensors"),
        "{err}"
    );
    assert_eq!(err.tensor(), Some("b"));
}

#[test]
fn rejects_a_tensor_the_index_does_not_list() {
    let dir = TestDir::new("sharded-unlisted");
    shard(
        &dir,
        "model-00001-of-00002.safetensors",
        &["a", "b", "x"],
        1,
    );
    shard(&dir, "model-00002-of-00002.safetensors", &["c"], 2);
    let err = ShardedSafetensors::open(index(&dir, VALID_INDEX)).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::UnlistedInShard { .. }),
        "{err}"
    );
    assert_eq!(err.tensor(), Some("x"));
}

#[test]
fn rejects_an_index_over_the_size_limit() {
    let dir = TestDir::new("sharded-big");
    let path = dir.join("model.safetensors.index.json");
    let file = std::fs::File::create(&path).unwrap();
    // Sparse on most file systems: no 64 MiB write needed.
    file.set_len(MAX_INDEX_LEN + 1).unwrap();
    drop(file);
    let err = ShardedSafetensors::open(&path).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::IndexTooLarge { .. }));
}

#[test]
fn rejects_malformed_indices() {
    let dir = TestDir::new("sharded-malformed");
    two_shards(&dir);
    for (i, body) in [
        "",
        "[]",
        r#"{"metadata":{}}"#,
        r#"{"weight_map":[]}"#,
        r#"{"weight_map":{"a":1}}"#,
        r#"{"weight_map":{},"weight_map":{}}"#,
        r#"{"weight_map":{"a":"model-00001-of-00002.safetensors","a":"model-00001-of-00002.safetensors"}}"#,
    ]
    .iter()
    .enumerate()
    {
        let path = dir.write(&format!("bad{i}.index.json"), body.as_bytes());
        let err = ShardedSafetensors::open(&path).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::InvalidIndex(_) | ErrorKind::DuplicateKey { .. }),
            "{body}: {err}"
        );
        assert_eq!(err.path(), Some(path.as_path()));
    }
    let path = dir.write("utf8.index.json", &[b'{', 0xff, b'}']);
    assert!(matches!(
        ShardedSafetensors::open(&path).unwrap_err().kind(),
        ErrorKind::InvalidIndex(_)
    ));
}

#[test]
fn a_broken_shard_is_reported_with_its_path() {
    let dir = TestDir::new("sharded-broken");
    shard(&dir, "model-00001-of-00002.safetensors", &["a", "b"], 1);
    dir.write("model-00002-of-00002.safetensors", b"\x02\0\0\0\0\0\0\0{}x");
    let err = ShardedSafetensors::open(index(&dir, VALID_INDEX)).unwrap_err();
    assert!(
        matches!(err.kind(), ErrorKind::DataLengthMismatch { .. }),
        "{err}"
    );
    assert_eq!(
        err.path(),
        Some(dir.join("model-00002-of-00002.safetensors").as_path())
    );

    let dir = TestDir::new("sharded-absent");
    shard(&dir, "model-00001-of-00002.safetensors", &["a", "b"], 1);
    let err = ShardedSafetensors::open(index(&dir, VALID_INDEX)).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Io(_)), "{err}");
}

#[test]
fn an_index_with_an_empty_weight_map_opens_no_shards() {
    let dir = TestDir::new("sharded-empty");
    let ckpt = ShardedSafetensors::open(index(&dir, r#"{"weight_map":{}}"#)).unwrap();
    assert!(ckpt.is_empty());
    assert_eq!(ckpt.total_size(), None);
    assert_eq!(ckpt.shards().count(), 0);
}
