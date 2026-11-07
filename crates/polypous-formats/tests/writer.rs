//! safetensors writer (`docs/specs/formats.md` §1.6): output layout, round trips and rejections.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use std::collections::BTreeMap;

use common::{TestDir, f32_bytes};
use polypous_formats::safetensors::{
    Dtype, ErrorKind, SafetensorsFile, TensorValues, WriteTensor, serialize, write, write_file,
};

fn metadata() -> BTreeMap<String, String> {
    BTreeMap::from([
        ("format".to_owned(), "pt".to_owned()),
        ("note".to_owned(), "quote \" and ünïcode".to_owned()),
    ])
}

#[test]
fn output_layout_matches_the_spec() {
    let a = f32_bytes(&[1.0, 2.0]);
    let b = [1u8, 0, 1];
    let tensors = vec![
        ("b", WriteTensor::new(Dtype::Bool, &[3], &b).unwrap()),
        ("a", WriteTensor::new(Dtype::F32, &[1, 2], &a).unwrap()),
    ];
    let bytes = serialize(tensors, &BTreeMap::new()).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    assert_eq!(n % 8, 0, "data buffer must start 8-byte aligned");
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    let expected = r#"{"a":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"b":{"dtype":"BOOL","shape":[3],"data_offsets":[8,11]}}"#;
    assert_eq!(header.trim_end_matches(' '), expected);
    assert!(header.len() - expected.len() < 8);
    assert_eq!(&bytes[8 + n..8 + n + 8], &a[..]);
    assert_eq!(&bytes[8 + n + 8..], &b[..]);
}

#[test]
fn round_trips_through_the_reader() {
    let f32s = f32_bytes(&[0.5, -1.0, 3.25, f32::INFINITY, -0.0, 1e-30]);
    let i64s: Vec<u8> = [i64::MIN, 0, i64::MAX]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let f16s: Vec<u8> = [0x3c00u16, 0x7c00]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let u8s = [1u8, 2, 3];
    let tensors = vec![
        (
            "weights.f32",
            WriteTensor::new(Dtype::F32, &[2, 3], &f32s).unwrap(),
        ),
        ("ids", WriteTensor::new(Dtype::I64, &[3], &i64s).unwrap()),
        ("half", WriteTensor::new(Dtype::F16, &[2], &f16s).unwrap()),
        ("raw", WriteTensor::new(Dtype::U8, &[3], &u8s).unwrap()),
        (
            "empty",
            WriteTensor::new(Dtype::BF16, &[0, 7], &[]).unwrap(),
        ),
        (
            "scalar",
            WriteTensor::new(Dtype::I32, &[], &[1, 0, 0, 0]).unwrap(),
        ),
        ("", WriteTensor::new(Dtype::F4, &[2], &[0x21]).unwrap()),
    ];
    let bytes = serialize(tensors.clone(), &metadata()).unwrap();
    let file = SafetensorsFile::from_bytes(bytes).unwrap();
    assert_eq!(file.metadata(), &metadata());
    assert_eq!(file.len(), tensors.len());
    for (name, t) in &tensors {
        let view = file.tensor(name).unwrap();
        assert_eq!(view.dtype(), t.dtype(), "{name}");
        assert_eq!(view.shape(), t.shape(), "{name}");
        assert_eq!(view.data(), t.data(), "{name}");
    }
    assert_eq!(
        file.tensor("ids").unwrap().values().unwrap(),
        TensorValues::I64(vec![i64::MIN, 0, i64::MAX])
    );
    assert_eq!(
        file.tensor("half").unwrap().to_f32_vec().unwrap(),
        vec![1.0, f32::INFINITY]
    );
}

#[test]
fn output_is_independent_of_input_order() {
    let x = [1u8, 2];
    let y = [3u8];
    let t1 = vec![
        (
            "x".to_owned(),
            WriteTensor::new(Dtype::U8, &[2], &x).unwrap(),
        ),
        (
            "y".to_owned(),
            WriteTensor::new(Dtype::U8, &[1], &y).unwrap(),
        ),
    ];
    let mut t2 = t1.clone();
    t2.reverse();
    assert_eq!(
        serialize(t1, &metadata()).unwrap(),
        serialize(t2, &metadata()).unwrap()
    );
}

#[test]
fn write_to_a_writer_equals_serialize() {
    let x = f32_bytes(&[4.0]);
    let tensors = [("x", WriteTensor::new(Dtype::F32, &[1], &x).unwrap())];
    let mut out = Vec::new();
    write(&mut out, tensors, &metadata()).unwrap();
    assert_eq!(out, serialize(tensors, &metadata()).unwrap());
}

#[test]
fn write_file_round_trips_and_replaces_an_unmapped_file() {
    let dir = TestDir::new("writer-file");
    let path = dir.join("out.safetensors");
    let x = f32_bytes(&[1.0, 2.0]);
    write_file(
        &path,
        [("x", WriteTensor::new(Dtype::F32, &[2], &x).unwrap())],
        &metadata(),
    )
    .unwrap();
    {
        let file = SafetensorsFile::open(&path).unwrap();
        assert_eq!(
            file.tensor("x").unwrap().to_f32_vec().unwrap(),
            vec![1.0, 2.0]
        );
        // `file` (and its map) is dropped here, before the file is replaced: Windows can't
        // replace a mapped file.
    }
    let y = [7u8];
    write_file(
        &path,
        [("y", WriteTensor::new(Dtype::U8, &[1], &y).unwrap())],
        &BTreeMap::new(),
    )
    .unwrap();
    let file = SafetensorsFile::open(&path).unwrap();
    assert_eq!(file.names().collect::<Vec<_>>(), ["y"]);
    drop(file);
    // No temporary files are left behind.
    let entries: Vec<_> = std::fs::read_dir(dir.path()).unwrap().collect();
    assert_eq!(entries.len(), 1);
}

#[test]
fn write_file_into_a_missing_directory_fails_cleanly() {
    let dir = TestDir::new("writer-missing");
    let path = dir.join("no/such/dir/out.safetensors");
    let err = write_file(
        &path,
        Vec::<(&str, WriteTensor<'_>)>::new(),
        &BTreeMap::new(),
    )
    .unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Io(_)));
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn rejects_duplicate_and_reserved_names() {
    let x = [1u8];
    let t = WriteTensor::new(Dtype::U8, &[1], &x).unwrap();
    let err = serialize([("a", t), ("a", t)], &BTreeMap::new()).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::DuplicateKey { key } if key == "a"));
    let err = serialize([("__metadata__", t)], &BTreeMap::new()).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::DuplicateKey { .. }));
}

#[test]
fn write_tensor_checks_the_byte_length() {
    assert!(matches!(
        WriteTensor::new(Dtype::F32, &[2], &[0; 7])
            .unwrap_err()
            .kind(),
        ErrorKind::LengthMismatch {
            expected: 8,
            actual: 7
        }
    ));
    assert!(matches!(
        WriteTensor::new(Dtype::F4, &[3], &[0; 2])
            .unwrap_err()
            .kind(),
        ErrorKind::PartialByte { .. }
    ));
    assert!(matches!(
        WriteTensor::new(Dtype::U8, &[usize::MAX, 2], &[])
            .unwrap_err()
            .kind(),
        ErrorKind::SizeOverflow
    ));
}

#[test]
fn rejects_a_header_over_the_limit() {
    // One metadata value just over 100 MB.
    let big = "x".repeat(100_000_000);
    let meta = BTreeMap::from([("k".to_owned(), big)]);
    let err = serialize(Vec::<(&str, WriteTensor<'_>)>::new(), &meta).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::HeaderTooLarge { .. }));
}

#[test]
fn every_tensor_is_aligned_to_its_element_size() {
    // Names chosen so that name order alone would misalign the wider tensors.
    let b = [1u8, 0, 1];
    let h = [0u8; 6];
    let f = [0u8; 12];
    let d = [0u8; 16];
    let l = [0u8; 8];
    let tensors = vec![
        ("a_bool", WriteTensor::new(Dtype::Bool, &[3], &b).unwrap()),
        ("b_f16", WriteTensor::new(Dtype::F16, &[3], &h).unwrap()),
        ("c_f32", WriteTensor::new(Dtype::F32, &[3], &f).unwrap()),
        ("d_f64", WriteTensor::new(Dtype::F64, &[2], &d).unwrap()),
        ("e_i64", WriteTensor::new(Dtype::I64, &[1], &l).unwrap()),
        ("f_u8", WriteTensor::new(Dtype::U8, &[3], &b).unwrap()),
        ("g_f4", WriteTensor::new(Dtype::F4, &[2], &[0]).unwrap()),
    ];
    let bytes = serialize(tensors, &BTreeMap::new()).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let data_start = 8 + n;
    assert_eq!(data_start % 8, 0);
    let file = SafetensorsFile::from_bytes(bytes).unwrap();
    let mut order: Vec<(usize, &str)> = file
        .names()
        .map(|name| (file.info(name).unwrap().data_offsets().0, name))
        .collect();
    order.sort();
    let order: Vec<&str> = order.into_iter().map(|(_, n)| n).collect();
    // Descending element size, then name; 1-byte and sub-byte dtypes last.
    assert_eq!(
        order,
        ["d_f64", "e_i64", "c_f32", "b_f16", "a_bool", "f_u8", "g_f4"]
    );
    for name in file.names() {
        let info = file.info(name).unwrap();
        let size = info.dtype().alignment();
        let (begin, _) = info.data_offsets();
        assert_eq!(
            begin % size,
            0,
            "{name}: offset {begin}, element size {size}"
        );
        assert_eq!((data_start + begin) % size, 0, "{name}: file offset");
    }
}
