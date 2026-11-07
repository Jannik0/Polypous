//! Reads a file written by Python `safetensors` (reference implementation): `tests/data/
//! python_written.safetensors`, produced by `tests/data/generate_python_fixture.py`.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous_formats::safetensors::{
    Dtype, SafetensorsFile, TensorValues, WriteTensor, f16_to_f32, serialize,
};

fn fixture() -> SafetensorsFile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/python_written.safetensors");
    SafetensorsFile::open(path).unwrap()
}

#[test]
fn reads_every_tensor_python_wrote() {
    let file = fixture();
    assert_eq!(
        file.metadata().get("format").map(String::as_str),
        Some("np")
    );
    assert_eq!(
        file.names().collect::<Vec<_>>(),
        [
            "bool",
            "empty",
            "f16",
            "f16_as_f32",
            "f32",
            "f64",
            "i32",
            "i64",
            "scalar",
            "u8"
        ]
    );
    let get = |n: &str| file.tensor(n).unwrap();

    let f32s = get("f32");
    assert_eq!(f32s.shape(), &[2, 3]);
    assert_eq!(
        f32s.values().unwrap(),
        TensorValues::F32(vec![0.5, -1.25, 3.0, 1e-7, -0.0, 65504.0])
    );
    assert_eq!(
        get("i64").values().unwrap(),
        TensorValues::I64(vec![i64::MIN, -1, 0, 1, i64::MAX])
    );
    let i32s = get("i32");
    assert_eq!(i32s.shape(), &[3, 1]);
    assert_eq!(
        i32s.values().unwrap(),
        TensorValues::I32(vec![i32::MIN, 7, i32::MAX])
    );
    assert_eq!(
        get("bool").values().unwrap(),
        TensorValues::Bool(vec![true, false, true])
    );
    assert_eq!(get("u8").dtype(), Dtype::U8);
    assert_eq!(get("u8").data(), &[0, 1, 2, 3]);
    assert_eq!(get("f64").dtype(), Dtype::F64);
    assert_eq!(get("f64").data(), &1.5f64.to_le_bytes());
    let scalar = get("scalar");
    assert_eq!(scalar.shape(), &[] as &[usize]);
    assert_eq!(scalar.to_f32_vec().unwrap(), vec![2.5]);
    let empty = get("empty");
    assert_eq!(empty.shape(), &[0, 3]);
    assert!(empty.data().is_empty());
}

/// numpy's f16 → f32 conversion (in the fixture) is the reference for `f16_to_f32`, compared
/// bit for bit: subnormals, ±0, ±inf, max, and a quiet NaN.
#[test]
fn f16_decoding_matches_numpy_bit_for_bit() {
    let file = fixture();
    let ours = file.tensor("f16").unwrap().to_f32_vec().unwrap();
    let numpy = file.tensor("f16_as_f32").unwrap().to_f32_vec().unwrap();
    assert_eq!(ours.len(), 12);
    for (i, (a, b)) in ours.iter().zip(&numpy).enumerate() {
        assert_eq!(a.to_bits(), b.to_bits(), "element {i}: {a} vs numpy {b}");
    }
    let TensorValues::F16(bits) = file.tensor("f16").unwrap().values().unwrap() else {
        panic!("expected F16 values");
    };
    assert_eq!(bits[2], 0x3c00);
    assert_eq!(f16_to_f32(bits[2]), 1.0);
}

/// Python files are re-serialized by our writer with the same tensors and metadata, and read
/// back identically. (The Python-side check of files we write is in the PR description.)
#[test]
fn rewriting_a_python_file_preserves_it() {
    let file = fixture();
    let shapes: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = file
        .names()
        .map(|n| {
            let v = file.tensor(n).unwrap();
            (
                n.to_owned(),
                v.dtype(),
                v.shape().to_vec(),
                v.data().to_vec(),
            )
        })
        .collect();
    let tensors = shapes
        .iter()
        .map(|(n, d, s, b)| (n.as_str(), WriteTensor::new(*d, s, b).unwrap()));
    let copy = SafetensorsFile::from_bytes(serialize(tensors, file.metadata()).unwrap()).unwrap();
    assert_eq!(copy.metadata(), file.metadata());
    for (n, d, s, b) in &shapes {
        let v = copy.tensor(n).unwrap();
        assert_eq!((v.dtype(), v.shape(), v.data()), (*d, &s[..], &b[..]));
    }
}
