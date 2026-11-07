//! safetensors reader: acceptance and one test per rejection rule of `docs/specs/formats.md` §1.2.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

mod common;

use common::{TestDir, f32_bytes, raw_file};
use polypous_formats::safetensors::{
    Dtype, ErrorKind, MAX_HEADER_LEN, MAX_RANK, SafetensorsFile, TensorValues,
};

fn open(bytes: Vec<u8>) -> Result<SafetensorsFile, polypous_formats::safetensors::Error> {
    SafetensorsFile::from_bytes(bytes)
}

fn kind(bytes: Vec<u8>) -> ErrorKind {
    open(bytes).unwrap_err().into_kind()
}

#[test]
fn reads_a_valid_file_from_disk_memory_mapped() {
    let dir = TestDir::new("reader-valid");
    let header = r#"{"__metadata__":{"format":"pt"},"b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"a":{"dtype":"I32","shape":[],"data_offsets":[8,12]}}"#;
    let mut data = f32_bytes(&[1.5, -2.0]);
    data.extend_from_slice(&7i32.to_le_bytes());
    let path = dir.write("m.safetensors", &raw_file(header, &data));

    let file = SafetensorsFile::open(&path).unwrap();
    assert!(file.is_memory_mapped());
    assert_eq!(file.path(), Some(path.as_path()));
    assert_eq!(
        file.metadata().get("format").map(String::as_str),
        Some("pt")
    );
    assert_eq!(file.names().collect::<Vec<_>>(), ["a", "b"]);
    assert_eq!(file.len(), 2);

    let b = file.tensor("b").unwrap();
    assert_eq!(b.dtype(), Dtype::F32);
    assert_eq!(b.shape(), &[2]);
    assert_eq!(b.numel(), 2);
    assert_eq!(b.data(), &data[..8]);
    assert_eq!(b.values().unwrap(), TensorValues::F32(vec![1.5, -2.0]));
    assert_eq!(b.to_f32_vec().unwrap(), vec![1.5, -2.0]);

    let a = file.tensor("a").unwrap();
    assert_eq!(a.shape(), &[] as &[usize]);
    assert_eq!(a.numel(), 1);
    assert_eq!(a.values().unwrap(), TensorValues::I32(vec![7]));
    assert_eq!(file.info("a").unwrap().data_offsets(), (8, 12));

    let err = file.tensor("missing").unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::TensorNotFound));
    assert_eq!(err.tensor(), Some("missing"));
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn empty_header_and_no_tensors() {
    let file = open(raw_file("{}", &[])).unwrap();
    assert!(file.is_empty());
    assert!(file.metadata().is_empty());
    // Whitespace padding after the object is allowed.
    assert!(open(raw_file("{}      ", &[])).is_ok());
}

#[test]
fn zero_element_tensors_are_allowed_and_may_share_an_offset() {
    let header = r#"{"e1":{"dtype":"F32","shape":[0,3],"data_offsets":[0,0]},"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"e2":{"dtype":"F16","shape":[4,0],"data_offsets":[1,1]},"e3":{"dtype":"BF16","shape":[0],"data_offsets":[1,1]}}"#;
    let file = open(raw_file(header, &[9])).unwrap();
    let e1 = file.tensor("e1").unwrap();
    assert_eq!(e1.numel(), 0);
    assert!(e1.data().is_empty());
    assert_eq!(e1.to_f32_vec().unwrap(), Vec::<f32>::new());
    assert_eq!(file.tensor("x").unwrap().data(), &[9]);
}

#[test]
fn every_dtype_is_recognized_and_only_six_decode() {
    for dtype in Dtype::ALL {
        // 8 elements is a whole number of bytes for every dtype (also 4- and 6-bit ones).
        let len = 8 * dtype.bits() / 8;
        let header = format!(
            r#"{{"t":{{"dtype":"{}","shape":[2,4],"data_offsets":[0,{len}]}}}}"#,
            dtype.name()
        );
        let file = open(raw_file(&header, &vec![0u8; len])).unwrap();
        let view = file.tensor("t").unwrap();
        assert_eq!(view.dtype(), dtype);
        assert_eq!(view.data().len(), len);
        let values = view.values();
        if dtype.is_readable_as_values() {
            assert!(values.is_ok(), "{dtype}");
        } else {
            let err = values.unwrap_err();
            assert!(
                matches!(err.kind(), ErrorKind::UnsupportedDtype { dtype: d } if *d == dtype),
                "{dtype}: {err}"
            );
            assert_eq!(err.tensor(), Some("t"));
        }
        let floats = matches!(dtype, Dtype::F32 | Dtype::F16 | Dtype::BF16);
        assert_eq!(view.to_f32_vec().is_ok(), floats, "{dtype}");
    }
}

#[test]
fn decodes_every_readable_dtype() {
    let header = r#"{"f16":{"dtype":"F16","shape":[2],"data_offsets":[0,4]},"bf16":{"dtype":"BF16","shape":[2],"data_offsets":[4,8]},"i64":{"dtype":"I64","shape":[1],"data_offsets":[8,16]},"bool":{"dtype":"BOOL","shape":[3],"data_offsets":[16,19]}}"#;
    let mut data = Vec::new();
    data.extend_from_slice(&0x3c00u16.to_le_bytes()); // 1.0
    data.extend_from_slice(&0xc000u16.to_le_bytes()); // -2.0
    data.extend_from_slice(&0x3f80u16.to_le_bytes()); // 1.0
    data.extend_from_slice(&0x4040u16.to_le_bytes()); // 3.0
    data.extend_from_slice(&(-5i64).to_le_bytes());
    data.extend_from_slice(&[1, 0, 1]);
    let file = open(raw_file(header, &data)).unwrap();
    let get = |n| file.tensor(n).unwrap();
    assert_eq!(
        get("f16").values().unwrap(),
        TensorValues::F16(vec![0x3c00, 0xc000])
    );
    assert_eq!(get("f16").to_f32_vec().unwrap(), vec![1.0, -2.0]);
    assert_eq!(
        get("bf16").values().unwrap(),
        TensorValues::BF16(vec![0x3f80, 0x4040])
    );
    assert_eq!(get("bf16").to_f32_vec().unwrap(), vec![1.0, 3.0]);
    assert_eq!(get("i64").values().unwrap(), TensorValues::I64(vec![-5]));
    assert_eq!(
        get("bool").values().unwrap(),
        TensorValues::Bool(vec![true, false, true])
    );
}

#[test]
fn rejects_bool_bytes_other_than_0_and_1() {
    let header = r#"{"b":{"dtype":"BOOL","shape":[3],"data_offsets":[0,3]}}"#;
    let file = open(raw_file(header, &[1, 2, 0])).unwrap();
    let err = file.tensor("b").unwrap().values().unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::InvalidBoolValue { index: 1, value: 2 }
    ));
}

// ----- One test per rejection rule (§1.2) -----

#[test]
fn rule_1_file_too_small() {
    for len in 0..8 {
        assert!(matches!(kind(vec![0; len]), ErrorKind::FileTooSmall { .. }));
    }
    let dir = TestDir::new("reader-small");
    let path = dir.write("empty.safetensors", &[]);
    let err = SafetensorsFile::open(&path).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::FileTooSmall { len: 0 }));
    assert_eq!(err.path(), Some(path.as_path()));
}

#[test]
fn rule_2_header_too_large() {
    let mut bytes = (MAX_HEADER_LEN + 1).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(kind(bytes), ErrorKind::HeaderTooLarge { .. }));
    let bytes = u64::MAX.to_le_bytes().to_vec();
    assert!(matches!(kind(bytes), ErrorKind::HeaderTooLarge { .. }));
}

#[test]
fn rule_3_header_out_of_bounds() {
    let mut bytes = 100u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(kind(bytes), ErrorKind::HeaderOutOfBounds { .. }));
}

#[test]
fn rule_4_header_not_utf8() {
    let mut bytes = 4u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[b'{', 0xff, 0xfe, b'}']);
    assert!(matches!(kind(bytes), ErrorKind::HeaderNotUtf8));
}

#[test]
fn rule_5_header_not_a_json_object() {
    for header in ["", "[]", "\"x\"", "{", "{} {}", "{}\0", "null", "{\"a\":}"] {
        assert!(
            matches!(kind(raw_file(header, &[])), ErrorKind::InvalidHeader(_)),
            "{header:?}"
        );
    }
}

#[test]
fn rule_6_metadata_not_string_to_string() {
    for header in [
        r#"{"__metadata__":{"a":1}}"#,
        r#"{"__metadata__":{"a":null}}"#,
        r#"{"__metadata__":["a"]}"#,
        r#"{"__metadata__":"a"}"#,
    ] {
        assert!(
            matches!(kind(raw_file(header, &[])), ErrorKind::InvalidMetadata(_)),
            "{header}"
        );
    }
}

#[test]
fn rule_7_tensor_entry_structure() {
    for header in [
        r#"{"t":[]}"#,
        r#"{"t":{"shape":[],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[]}}"#,
        r#"{"t":{"dtype":1,"shape":[],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[1.0],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[-1],"data_offsets":[0,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[],"data_offsets":[0]}}"#,
        r#"{"t":{"dtype":"F32","shape":[],"data_offsets":[0,4,4]}}"#,
        r#"{"t":{"dtype":"F32","shape":[],"data_offsets":[-1,4]}}"#,
    ] {
        let err = open(raw_file(header, &[0; 4])).unwrap_err();
        assert!(
            matches!(err.kind(), ErrorKind::InvalidHeader(_)),
            "{header}: {err}"
        );
        assert_eq!(err.tensor(), Some("t"), "{header}");
    }
    // Unknown fields are ignored, as in the reference implementation.
    let header = r#"{"t":{"dtype":"F32","shape":[],"data_offsets":[0,4],"x":{"y":[1,2]}}}"#;
    assert!(open(raw_file(header, &[0; 4])).is_ok());
}

#[test]
fn rule_8_duplicate_keys() {
    for header in [
        r#"{"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"t":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}"#,
        r#"{"__metadata__":{},"__metadata__":{}}"#,
        r#"{"__metadata__":{"k":"a","k":"b"}}"#,
        r#"{"t":{"dtype":"U8","shape":[1],"shape":[1],"data_offsets":[0,1]}}"#,
    ] {
        assert!(
            matches!(kind(raw_file(header, &[0])), ErrorKind::DuplicateKey { .. }),
            "{header}"
        );
    }
}

#[test]
fn rule_9_unknown_dtype() {
    let header = r#"{"t":{"dtype":"Q4_K","shape":[],"data_offsets":[0,4]}}"#;
    let err = open(raw_file(header, &[0; 4])).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::UnknownDtype { name } if name == "Q4_K"));
    assert_eq!(err.tensor(), Some("t"));
    let header = r#"{"t":{"dtype":"f32","shape":[],"data_offsets":[0,4]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 4])),
        ErrorKind::UnknownDtype { .. }
    ));
}

#[test]
fn rule_10_rank_too_large() {
    let dims = vec!["1"; MAX_RANK + 1].join(",");
    let header = format!(r#"{{"t":{{"dtype":"U8","shape":[{dims}],"data_offsets":[0,1]}}}}"#);
    let err = open(raw_file(&header, &[0])).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::RankTooLarge { .. }));
    assert_eq!(err.tensor(), Some("t"));
}

#[test]
fn rule_11_size_overflow() {
    let big = u64::MAX;
    for header in [
        format!(r#"{{"t":{{"dtype":"U8","shape":[{big},{big}],"data_offsets":[0,0]}}}}"#),
        format!(
            r#"{{"t":{{"dtype":"F64","shape":[{}],"data_offsets":[0,0]}}}}"#,
            usize::MAX / 2
        ),
        r#"{"t":{"dtype":"U8","shape":[18446744073709551616],"data_offsets":[0,0]}}"#.to_owned(),
    ] {
        let k = kind(raw_file(&header, &[]));
        // A number above u64::MAX is a JSON type error; the others overflow.
        assert!(
            matches!(k, ErrorKind::SizeOverflow | ErrorKind::InvalidHeader(_)),
            "{header}: {k}"
        );
    }
    let header = format!(r#"{{"t":{{"dtype":"U8","shape":[{big},2],"data_offsets":[0,0]}}}}"#);
    assert!(matches!(
        kind(raw_file(&header, &[])),
        ErrorKind::SizeOverflow
    ));
}

#[test]
fn rule_12_partial_byte() {
    let header = r#"{"t":{"dtype":"F4","shape":[3],"data_offsets":[0,1]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0])),
        ErrorKind::PartialByte { .. }
    ));
    let header = r#"{"t":{"dtype":"F4","shape":[4],"data_offsets":[0,2]}}"#;
    assert!(open(raw_file(header, &[0, 0])).is_ok());
}

#[test]
fn rule_13_length_mismatch_and_inverted_offsets() {
    let header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 4])),
        ErrorKind::LengthMismatch {
            expected: 8,
            actual: 4
        }
    ));
    let header = r#"{"t":{"dtype":"F32","shape":[0],"data_offsets":[4,0]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 4])),
        ErrorKind::InvalidOffsets { begin: 4, end: 0 }
    ));
}

#[test]
fn rule_14_overlap_and_holes() {
    // Overlap.
    let header = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[1,3]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 3])),
        ErrorKind::NonContiguous {
            expected: 2,
            begin: 1
        }
    ));
    // Two tensors on the same bytes.
    let header = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 2])),
        ErrorKind::NonContiguous { .. }
    ));
    // Hole between tensors.
    let header = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#;
    assert!(matches!(
        kind(raw_file(header, &[0; 3])),
        ErrorKind::NonContiguous { .. }
    ));
    // Hole at the start.
    let header = r#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}"#;
    let err = open(raw_file(header, &[0; 2])).unwrap_err();
    assert!(matches!(
        err.kind(),
        ErrorKind::NonContiguous {
            expected: 0,
            begin: 1
        }
    ));
    assert_eq!(err.tensor(), Some("a"));
}

#[test]
fn rule_15_data_buffer_not_covered_exactly() {
    let header = r#"{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}"#;
    // Trailing bytes.
    assert!(matches!(
        kind(raw_file(header, &[0; 3])),
        ErrorKind::DataLengthMismatch {
            covered: 2,
            data_len: 3
        }
    ));
    // Truncated data.
    assert!(matches!(
        kind(raw_file(header, &[0; 1])),
        ErrorKind::DataLengthMismatch {
            covered: 2,
            data_len: 1
        }
    ));
    // No tensors but data.
    assert!(matches!(
        kind(raw_file("{}", &[0])),
        ErrorKind::DataLengthMismatch { .. }
    ));
}

#[test]
fn errors_from_open_name_the_file() {
    let dir = TestDir::new("reader-errpath");
    let header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let path = dir.write("bad.safetensors", &raw_file(header, &[0; 4]));
    let err = SafetensorsFile::open(&path).unwrap_err();
    assert_eq!(err.path(), Some(path.as_path()));
    assert_eq!(err.tensor(), Some("t"));
    let msg = err.to_string();
    assert!(
        msg.contains("bad.safetensors") && msg.contains("\"t\""),
        "{msg}"
    );

    let err = SafetensorsFile::open(dir.join("does-not-exist.safetensors")).unwrap_err();
    assert!(matches!(err.kind(), ErrorKind::Io(_)));
}
