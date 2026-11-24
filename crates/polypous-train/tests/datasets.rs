//! Dataset files (training.md §3.5.2–§3.5.4): `image_folder`, `csv` and `safetensors` read
//! through [`read_datasets`] from files generated in a temporary directory (owner decision 9: no
//! committed image corpus), with their limits and errors, and fuzz-style loops over malformed
//! files (truncated, bit-flipped). Host-only.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use png::{BitDepth, ColorType, Encoder};
use polypous_backend::{DType, TensorData};
use polypous_formats::safetensors::{Dtype, WriteTensor, write_file};
use polypous_runtime::CancellationToken;
use polypous_train::data::{DataConfig, DataSources, Datasets, FieldSelection};
use polypous_train::{DataError, DataErrorKind, read_datasets};

// --- Helpers ---------------------------------------------------------------------------------

/// A fresh directory, removed on drop.
struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        // Digits only: `SystemTime`'s Debug form contains `:`, which Windows paths don't allow.
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let dir = std::env::temp_dir().join(format!(
            "polypous-datasets-{name}-{}-{nanos}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn write(&self, rel: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
        path
    }

    fn mkdir(&self, rel: &str) {
        fs::create_dir_all(self.0.join(rel)).unwrap();
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn read_in(dir: &Path, json: &str, fields: FieldSelection) -> Result<Datasets, DataError> {
    let config: DataConfig = serde_json::from_str(json).unwrap();
    read_datasets(
        &config,
        0,
        DataSources {
            base_dir: dir.to_path_buf(),
            in_memory: Vec::new(),
            fields,
        },
        &CancellationToken::new(),
    )
}

fn read(dir: &Path, json: &str) -> Result<Datasets, DataError> {
    read_in(dir, json, FieldSelection::All)
}

fn only(names: &[&str]) -> FieldSelection {
    FieldSelection::Only(names.iter().map(|s| (*s).to_owned()).collect())
}

fn f32s(t: &TensorData) -> Vec<f32> {
    assert_eq!(t.dtype, DType::F32);
    let (chunks, _) = t.bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn i64s(t: &TensorData) -> Vec<i64> {
    assert_eq!(t.dtype, DType::I64);
    let (chunks, _) = t.bytes.as_chunks::<8>();
    chunks.iter().map(|c| i64::from_le_bytes(*c)).collect()
}

fn i32s(t: &TensorData) -> Vec<i32> {
    assert_eq!(t.dtype, DType::I32);
    let (chunks, _) = t.bytes.as_chunks::<4>();
    chunks.iter().map(|c| i32::from_le_bytes(*c)).collect()
}

#[track_caller]
fn invalid(result: Result<Datasets, DataError>, needle: &str) -> DataError {
    let err = result.expect_err("expected an error");
    assert_eq!(err.kind(), DataErrorKind::InvalidInput, "{err}");
    assert!(err.to_string().contains(needle), "`{needle}` not in: {err}");
    err
}

/// An 8-bit PNG of `h × w` pixels (`color` gray or RGB) whose samples are `value`.
fn png(h: u32, w: u32, color: ColorType, value: u8) -> Vec<u8> {
    let samples = match color {
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::GrayscaleAlpha => 2,
        _ => 1,
    };
    let mut buf = Vec::new();
    let mut enc = Encoder::new(&mut buf, w, h);
    enc.set_color(color);
    enc.set_depth(BitDepth::Eight);
    let mut writer = enc.write_header().unwrap();
    writer
        .write_image_data(&vec![value; (h * w) as usize * samples])
        .unwrap();
    writer.finish().unwrap();
    buf
}

fn gray(value: u8) -> Vec<u8> {
    png(2, 3, ColorType::Grayscale, value)
}

const FOLDER: &str = r#"{"train": {"kind": "image_folder", "path": "imgs", "image_size": [2, 3]}}"#;

fn folder(extra: &str) -> String {
    format!(
        r#"{{"train": {{"kind": "image_folder", "path": "imgs", "image_size": [2, 3]{extra}}}}}"#
    )
}

// --- image_folder ----------------------------------------------------------------------------

/// Classes and files in byte order (`Zebra` before `cat`, `B.PNG` before `a.png`; names that
/// differ only in case would collide on Windows and macOS), labels by class index, the
/// `[C, H, W]` tensor, and ignored entries.
#[test]
fn image_folder_order_and_labels() {
    let dir = TempDir::new("order");
    dir.write("imgs/cat/c.png", &gray(2));
    dir.write("imgs/cat/B.PNG", &gray(0));
    dir.write("imgs/cat/a.png", &gray(1));
    dir.write("imgs/Zebra/z.Png", &gray(10));
    dir.write("imgs/dog/x.png", &gray(20));
    // Ignored: other files, a nested directory (even with PNGs inside), a directory named
    // `.png`, and files directly in `path`.
    dir.write("imgs/cat/.DS_Store", b"junk");
    dir.write("imgs/cat/Thumbs.db", b"junk");
    dir.write("imgs/cat/photo.jpg", b"junk");
    dir.write("imgs/cat/png", b"junk");
    dir.write("imgs/cat/nested/inner.png", &gray(99));
    dir.mkdir("imgs/cat/dir.png");
    dir.write("imgs/README.txt", b"junk");
    dir.write("imgs/top.png", &gray(99));
    let sets = read(dir.path(), FOLDER).unwrap();
    let train = &sets.train;
    assert_eq!(train.len(), 5);
    assert_eq!(i64s(train.field("label").unwrap()), [0, 1, 1, 1, 2]);
    let image = train.field("image").unwrap();
    assert_eq!(image.shape, [5, 1, 2, 3]);
    let values = f32s(image);
    let firsts: Vec<f32> = values.chunks(6).map(|c| c[0]).collect();
    // Zebra/z, cat/B.PNG, cat/a.png, cat/c.png, dog/x.
    assert_eq!(
        firsts,
        [10.0 / 255.0, 0.0, 1.0 / 255.0, 2.0 / 255.0, 20.0 / 255.0]
    );
    assert!(values.chunks(6).all(|c| c.iter().all(|&v| v == c[0])));
    // Other field names.
    let sets = read(
        dir.path(),
        &folder(r#", "image_field": "pixels", "label_field": "class""#),
    )
    .unwrap();
    assert!(sets.train.field("pixels").is_some() && sets.train.field("class").is_some());
}

#[test]
fn image_folder_channels() {
    let dir = TempDir::new("channels");
    dir.write("imgs/a/gray.png", &gray(51));
    dir.write("imgs/b/rgb.png", &png(2, 3, ColorType::Rgb, 102));
    // channels: 3 replicates gray.
    let sets = read(dir.path(), &folder(r#", "channels": 3"#)).unwrap();
    let image = sets.train.field("image").unwrap();
    assert_eq!(image.shape, [2, 3, 2, 3]);
    let v = f32s(image);
    assert!(v[..18].iter().all(|&x| x == 51.0 / 255.0));
    assert!(v[18..].iter().all(|&x| x == 102.0 / 255.0));
    // channels: 1 with an RGB image is an error naming the file.
    let err = invalid(read(dir.path(), FOLDER), "`channels` is 1");
    assert!(err.file().unwrap().ends_with("rgb.png"), "{err}");
    assert_eq!(err.field(), Some("image"));
}

#[test]
fn image_folder_errors() {
    let dir = TempDir::new("errors");
    // The folder doesn't exist: Io.
    let err = read(dir.path(), FOLDER).unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::Io);
    assert!(err.file().unwrap().ends_with("imgs"));
    // No classes.
    dir.mkdir("imgs");
    dir.write("imgs/loose.png", &gray(1));
    invalid(read(dir.path(), FOLDER), "no class folders");
    // An empty class, and one with only JPEGs.
    dir.write("imgs/good/1.png", &gray(1));
    dir.mkdir("imgs/empty");
    let err = invalid(read(dir.path(), FOLDER), "`empty` has no PNG image");
    assert!(err.file().unwrap().ends_with("empty"));
    fs::remove_dir(dir.path().join("imgs/empty")).unwrap();
    dir.write("imgs/jpegs/1.jpg", b"jpeg");
    invalid(read(dir.path(), FOLDER), "`jpegs` has no PNG image");
    fs::remove_dir_all(dir.path().join("imgs/jpegs")).unwrap();
    assert!(read(dir.path(), FOLDER).is_ok());
    // The wrong size, named with both sizes.
    let wrong = dir.write("imgs/good/2.png", &png(3, 2, ColorType::Grayscale, 0));
    let err = invalid(read(dir.path(), FOLDER), "3×2");
    assert!(err.message().contains("[2, 3]"), "{err}");
    assert_eq!(err.file(), Some(wrong.as_path()));
    // Not a PNG, and a truncated one.
    fs::write(&wrong, b"not a png").unwrap();
    let err = invalid(read(dir.path(), FOLDER), "not a valid PNG");
    assert_eq!(err.file(), Some(wrong.as_path()));
    let file = gray(7);
    fs::write(&wrong, &file[..file.len() - 20]).unwrap();
    invalid(read(dir.path(), FOLDER), "not a valid PNG");
    // A file over 64 MiB is rejected from its size, before it is read.
    let big = fs::File::create(&wrong).unwrap();
    big.set_len((64 << 20) + 1).unwrap();
    drop(big);
    let err = invalid(read(dir.path(), FOLDER), "64 MiB");
    assert_eq!(err.file(), Some(wrong.as_path()));
    let exact = fs::File::create(&wrong).unwrap();
    exact.set_len(64 << 20).unwrap();
    drop(exact);
    invalid(read(dir.path(), FOLDER), "not a valid PNG");
}

/// `count · channels · H · W · 4` bytes within `data.max_bytes`, checked before decoding: the
/// broken file is never decoded when the budget is too small.
#[test]
fn image_folder_memory_limit() {
    let dir = TempDir::new("memory");
    dir.write("imgs/a/1.png", &gray(1));
    dir.write("imgs/a/2.png", b"broken");
    // 2 images · 1 · 2 · 3 · 4 = 48 bytes of image, 16 bytes of labels.
    let err = invalid(read(dir.path(), &folder_max(63)), "max_bytes");
    assert_eq!(err.dataset(), Some("train"));
    invalid(read(dir.path(), &folder_max(64)), "not a valid PNG");
    // Only the labels: the images are neither counted nor read.
    let sets = read_in(dir.path(), &folder_max(16), only(&["label"])).unwrap();
    assert_eq!(i64s(sets.train.field("label").unwrap()), [0, 0]);
    assert!(sets.train.field("image").is_none());
}

fn folder_max(max: u64) -> String {
    format!(
        r#"{{"train": {{"kind": "image_folder", "path": "imgs", "image_size": [2, 3]}}, "max_bytes": {max}}}"#
    )
}

#[test]
fn image_folder_class_limit() {
    let dir = TempDir::new("classes");
    for i in 0..10_000 {
        dir.mkdir(&format!("imgs/c{i:05}"));
    }
    // 10,000 classes are allowed (the empty first class then fails), 10,001 are not.
    invalid(read(dir.path(), FOLDER), "`c00000` has no PNG image");
    dir.mkdir("imgs/one_more");
    invalid(read(dir.path(), FOLDER), "more than 10000 class folders");
}

#[cfg(unix)]
#[test]
fn image_folder_symlinks_are_followed() {
    use std::os::unix::fs::symlink;
    let dir = TempDir::new("symlinks");
    dir.write("store/real.png", &gray(5));
    dir.write("store/class/x.png", &gray(6));
    dir.mkdir("imgs/a");
    symlink(
        dir.path().join("store/real.png"),
        dir.path().join("imgs/a/link.png"),
    )
    .unwrap();
    symlink(dir.path().join("store/class"), dir.path().join("imgs/b")).unwrap();
    // A dangling link is not a regular file: ignored.
    symlink(
        dir.path().join("missing.png"),
        dir.path().join("imgs/a/dangling.png"),
    )
    .unwrap();
    let sets = read(dir.path(), FOLDER).unwrap();
    assert_eq!(i64s(sets.train.field("label").unwrap()), [0, 1]);
    let v = f32s(sets.train.field("image").unwrap());
    assert_eq!((v[0], v[6]), (5.0 / 255.0, 6.0 / 255.0));
}

/// A class folder name that isn't UTF-8 is `InvalidInput` (Linux allows any bytes in names;
/// macOS and Windows file systems don't, so the test runs on Linux only).
#[cfg(target_os = "linux")]
#[test]
fn image_folder_non_utf8_names() {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let dir = TempDir::new("utf8");
    dir.write("imgs/ok/1.png", &gray(1));
    let bad = dir.path().join("imgs").join(OsStr::from_bytes(b"bad\xff"));
    fs::create_dir(&bad).unwrap();
    invalid(read(dir.path(), FOLDER), "not valid UTF-8");
    fs::remove_dir(&bad).unwrap();
    // An image file name that isn't UTF-8 too; an ignored file may have any name.
    fs::write(
        dir.path()
            .join("imgs/ok")
            .join(OsStr::from_bytes(b"x\xfe.txt")),
        b"",
    )
    .unwrap();
    assert!(read(dir.path(), FOLDER).is_ok());
    fs::write(
        dir.path()
            .join("imgs/ok")
            .join(OsStr::from_bytes(b"x\xfe.png")),
        gray(1),
    )
    .unwrap();
    invalid(read(dir.path(), FOLDER), "not valid UTF-8");
}

#[test]
fn image_folder_cancel() {
    let dir = TempDir::new("cancel");
    dir.write("imgs/a/1.png", &gray(1));
    let config: DataConfig = serde_json::from_str(FOLDER).unwrap();
    let cancel = CancellationToken::new();
    cancel.cancel();
    let err = read_datasets(
        &config,
        0,
        DataSources {
            base_dir: dir.path().to_path_buf(),
            ..DataSources::default()
        },
        &cancel,
    )
    .unwrap_err();
    assert_eq!(err.kind(), DataErrorKind::Cancelled);
}

// --- csv -------------------------------------------------------------------------------------

fn csv_config(columns: &str) -> String {
    format!(r#"{{"train": {{"kind": "csv", "path": "d.csv", "columns": {{{columns}}}}}}}"#)
}

const XY: &str =
    r#""x": {"from": ["a", "b"], "dtype": "f32"}, "y": {"from": "label", "dtype": "i64"}"#;

#[test]
fn csv_quotes_crlf_bom_and_last_line() {
    let dir = TempDir::new("csv");
    let text = "\u{feff}a,\"b\",label,\"note, with \"\"quotes\"\"\"\r\n\
                1,-0.5,3,\"multi\r\nline, text\"\r\n\
                \"2.5\",1e-3,\"-7\",plain\r\n";
    dir.write("d.csv", text.as_bytes());
    let note = r#", "n": {"from": "note, with \"quotes\"", "dtype": "i32"}"#;
    let sets = read_in(
        dir.path(),
        &csv_config(&format!("{XY}{note}")),
        only(&["x", "y"]),
    )
    .unwrap();
    let train = &sets.train;
    assert_eq!(train.len(), 2);
    let x = train.field("x").unwrap();
    assert_eq!(x.shape, [2, 2]);
    assert_eq!(f32s(x), [1.0, -0.5, 2.5, 1e-3]);
    let y = train.field("y").unwrap();
    assert_eq!(y.shape, [2]);
    assert_eq!(i64s(y), [3, -7]);
    // The unused field `n` (text in an i32 column) is not parsed.
    assert!(train.field("n").is_none());
    // LF line breaks, no BOM, no final line break; `from` order is kept; i32.
    dir.write("d.csv", b"a,b,label\n1,2,3\n4,5,6");
    let sets = read(
        dir.path(),
        &csv_config(
            r#""x": {"from": ["b", "a"], "dtype": "f32"}, "y": {"from": "label", "dtype": "i32"}"#,
        ),
    )
    .unwrap();
    assert_eq!(f32s(sets.train.field("x").unwrap()), [2.0, 1.0, 5.0, 4.0]);
    assert_eq!(i32s(sets.train.field("y").unwrap()), [3, 6]);
    // An empty last line is allowed; a second one is a record with one empty field.
    dir.write("d.csv", b"a,b,label\n1,2,3\n");
    assert_eq!(read(dir.path(), &csv_config(XY)).unwrap().train.len(), 1);
    dir.write("d.csv", b"a,b,label\n1,2,3\n\n");
    let err = invalid(read(dir.path(), &csv_config(XY)), "record 3");
    assert_eq!(err.record(), Some(3));
}

/// A CSV file, a part of the expected message, the record and the column.
type ErrorCase = (
    &'static [u8],
    &'static str,
    Option<u64>,
    Option<&'static str>,
);

/// Every error case names the record (the header is record 1) and, where there is one, the
/// column.
#[test]
fn csv_errors() {
    let dir = TempDir::new("csv-errors");
    let cases: &[ErrorCase] = &[
        (b"a,b,label\n1,,3\n", "empty value", Some(2), Some("b")),
        (
            b"a,b,label\n1,2,3\n1,x,3\n",
            "not an f32",
            Some(3),
            Some("b"),
        ),
        (
            b"a,b,label\n1,inf,3\n",
            "not a finite f32",
            Some(2),
            Some("b"),
        ),
        (
            b"a,b,label\n1,-Infinity,3\n",
            "not a finite f32",
            Some(2),
            Some("b"),
        ),
        (
            b"a,b,label\nNaN,2,3\n",
            "not a finite f32",
            Some(2),
            Some("a"),
        ),
        (
            b"a,b,label\n1,1e39,3\n",
            "not a finite f32",
            Some(2),
            Some("b"),
        ),
        (
            b"a,b,label\n1,-3.5e38,3\n",
            "not a finite f32",
            Some(2),
            Some("b"),
        ),
        (b"a,b,label\n1, 2,3\n", "not an f32", Some(2), Some("b")),
        (
            b"a,b,label\n1,2,9223372036854775808\n",
            "out of the i64 range",
            Some(2),
            Some("label"),
        ),
        (
            b"a,b,label\n1,2,3.0\n",
            "not a decimal integer",
            Some(2),
            Some("label"),
        ),
        (
            b"a,b,label\n1,2\n",
            "has 2 fields, but the header has 3",
            Some(2),
            None,
        ),
        (b"a,b,label\n1,2,3,4\n", "has 4 fields", Some(2), None),
        (
            b"a,b,label\n1,2,\"3\n",
            "unterminated",
            Some(2),
            Some("label"),
        ),
        (
            b"a,b,label\n1,2\"x,3\n",
            "unquoted field",
            Some(2),
            Some("b"),
        ),
        (
            b"a,b,label\n\"1\"2,2,3\n",
            "closing quote",
            Some(2),
            Some("a"),
        ),
        (
            b"a,b,label\n1,2,3\n1,2,\xff\n",
            "not valid UTF-8",
            Some(3),
            Some("label"),
        ),
        (b"a,b,a\n1,2,3\n", "appears twice", Some(1), Some("a")),
        (
            b"a,,label\n1,2,3\n",
            "header column 2 is empty",
            Some(1),
            None,
        ),
        (b"", "empty", None, None),
        (b"\xef\xbb\xbf", "empty", None, None),
        (b"a,b,label\n", "no records", None, None),
        (b"a,b,label\r\n", "no records", None, None),
    ];
    for &(text, needle, record, column) in cases {
        let path = dir.write("d.csv", text);
        let err = invalid(read(dir.path(), &csv_config(XY)), needle);
        assert_eq!(err.record(), record, "{err}");
        assert_eq!(err.column(), column, "{err}");
        assert_eq!(err.file(), Some(path.as_path()), "{err}");
        if let (Some(r), Some(c)) = (record, column) {
            assert!(
                err.message().contains(&format!("record {r}, column `{c}`")),
                "{err}"
            );
        }
    }
    // Invalid UTF-8 in a column no field uses is still an error (the file must be UTF-8).
    dir.write("d.csv", b"a,b,label,extra\n1,2,3,\xc3\n");
    invalid(
        read(dir.path(), &csv_config(XY)),
        "column `extra`: the field is not valid UTF-8",
    );
    // An unknown column, also for an unused field.
    dir.write("d.csv", b"a,b,label\n1,2,3\n");
    let err = invalid(
        read_in(
            dir.path(),
            &csv_config(r#""x": {"from": ["a", "c"], "dtype": "f32"}"#),
            only(&["nothing"]),
        ),
        "no column `c`",
    );
    assert_eq!(err.field(), Some("x"));
    // i32 range.
    dir.write("d.csv", b"a,b,label\n1,2,-2147483649\n");
    invalid(
        read(
            dir.path(),
            &csv_config(r#""y": {"from": "label", "dtype": "i32"}"#),
        ),
        "out of the i32 range",
    );
    // A missing file is Io.
    fs::remove_file(dir.path().join("d.csv")).unwrap();
    assert_eq!(
        read(dir.path(), &csv_config(XY)).unwrap_err().kind(),
        DataErrorKind::Io
    );
}

#[test]
fn csv_limits() {
    let dir = TempDir::new("csv-limits");
    // A record over 1 MiB.
    let mut text = b"a\n".to_vec();
    text.extend(std::iter::repeat_n(b'1', (1 << 20) + 1));
    text.push(b'\n');
    dir.write("d.csv", &text);
    let a = r#""a": {"from": "a", "dtype": "f32"}"#;
    invalid(
        read(dir.path(), &csv_config(a)),
        "record 2, column `a`: the record is longer than 1 MiB",
    );
    // More than 16,384 columns in the header.
    let header: Vec<String> = (0..16_385).map(|i| format!("c{i}")).collect();
    dir.write("d.csv", format!("{}\n", header.join(",")).as_bytes());
    let c0 = r#""a": {"from": "c0", "dtype": "f32"}"#;
    invalid(read(dir.path(), &csv_config(c0)), "more than 16384 fields");
    // max_bytes: 3 records of x (2 · f32) and y (i64) are 48 bytes.
    dir.write("d.csv", b"a,b,label\n1,2,3\n4,5,6\n7,8,9\n");
    let max = |m: u64| {
        format!(
            r#"{{"train": {{"kind": "csv", "path": "d.csv", "columns": {{{XY}}}}}, "max_bytes": {m}}}"#
        )
    };
    assert!(read(dir.path(), &max(48)).is_ok());
    let err = invalid(read(dir.path(), &max(47)), "max_bytes");
    assert_eq!(err.record(), Some(4));
}

/// The 4 GiB limit is checked from the file's size before anything is read (a sparse file, so
/// it takes no disk space; Linux only).
#[cfg(target_os = "linux")]
#[test]
fn csv_file_limit() {
    let dir = TempDir::new("csv-big");
    let path = dir.write("d.csv", b"");
    let file = fs::File::options().write(true).open(&path).unwrap();
    file.set_len((1 << 32) + 1).unwrap();
    drop(file);
    invalid(read(dir.path(), &csv_config(XY)), "at most 4 GiB");
}

// --- safetensors -----------------------------------------------------------------------------

fn st_config(max: u64) -> String {
    format!(
        r#"{{"train": {{"kind": "safetensors", "path": "d.safetensors"}}, "max_bytes": {max}}}"#
    )
}

fn le<T: Copy, const N: usize>(values: &[T], f: impl Fn(T) -> [u8; N]) -> Vec<u8> {
    values.iter().flat_map(|&v| f(v)).collect()
}

fn write_st(dir: &TempDir, tensors: &[(&str, Dtype, &[usize], Vec<u8>)]) -> PathBuf {
    let path = dir.path().join("d.safetensors");
    let views: Vec<(String, WriteTensor<'_>)> = tensors
        .iter()
        .map(|(n, d, s, b)| ((*n).to_owned(), WriteTensor::new(*d, s, b).unwrap()))
        .collect();
    write_file(&path, views, &BTreeMap::new()).unwrap();
    path
}

#[test]
fn safetensors_fields_and_widening() {
    let dir = TempDir::new("st");
    // f16: 1, -2, 2^-24 (subnormal), 65504, -0; bf16: 1, -0.5, 2^-133 (subnormal in f32), 3.
    let f16 = le(
        &[0x3C00u16, 0xC000, 0x0001, 0x7BFF, 0x8000],
        u16::to_le_bytes,
    );
    let bf16 = le(&[0x3F80u16, 0xBF00, 0x0001, 0x4040, 0], u16::to_le_bytes);
    write_st(
        &dir,
        &[
            ("h", Dtype::F16, &[5], f16),
            ("b", Dtype::BF16, &[5, 1], bf16),
            (
                "f",
                Dtype::F32,
                &[5],
                le(&[1.5f32, 2.0, 3.0, 4.0, 5.0], f32::to_le_bytes),
            ),
            (
                "i",
                Dtype::I64,
                &[5],
                le(&[1i64, -2, 3, 4, i64::MAX], i64::to_le_bytes),
            ),
            (
                "j",
                Dtype::I32,
                &[5, 1, 1],
                le(&[1i32, 2, 3, 4, i32::MIN], i32::to_le_bytes),
            ),
            ("m", Dtype::Bool, &[5], vec![0, 1, 1, 0, 1]),
            // Unused fields: a U8 tensor, one with another sample count, a scalar.
            ("u8", Dtype::U8, &[2], vec![1, 2]),
            ("other_count", Dtype::F32, &[3], vec![0; 12]),
            ("scalar", Dtype::F32, &[], vec![0; 4]),
        ],
    );
    let used = only(&["h", "b", "f", "i", "j", "m"]);
    let sets = read_in(dir.path(), &st_config(1 << 20), used.clone()).unwrap();
    let t = &sets.train;
    assert_eq!(t.len(), 5);
    let h = t.field("h").unwrap();
    assert_eq!(f32s(h), [1.0, -2.0, 2f32.powi(-24), 65504.0, -0.0]);
    assert!(f32s(h)[4].is_sign_negative());
    let b = t.field("b").unwrap();
    assert_eq!(b.shape, [5, 1]);
    assert_eq!(f32s(b), [1.0, -0.5, f32::from_bits(0x0001_0000), 3.0, 0.0]);
    assert_eq!(f32s(t.field("f").unwrap()), [1.5, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(i64s(t.field("i").unwrap()), [1, -2, 3, 4, i64::MAX]);
    let j = t.field("j").unwrap();
    assert_eq!(
        (j.shape.as_slice(), i32s(j)),
        (&[5u64, 1, 1][..], vec![1, 2, 3, 4, i32::MIN])
    );
    let m = t.field("m").unwrap();
    assert_eq!(
        (m.dtype, m.bytes.as_slice()),
        (DType::Bool, &[0u8, 1, 1, 0, 1][..])
    );
    assert!(t.field("u8").is_none());
    // Memory from the header: 5 · (4 + 4 + 4 + 8 + 4 + 1) = 125 bytes (f16/bf16 widened).
    assert!(read_in(dir.path(), &st_config(125), used.clone()).is_ok());
    invalid(read_in(dir.path(), &st_config(124), used), "max_bytes");
    // Each unused tensor's problem, once used.
    let err = invalid(
        read_in(dir.path(), &st_config(1 << 20), only(&["u8"])),
        "U8",
    );
    assert_eq!(err.field(), Some("u8"));
    let err = invalid(
        read_in(dir.path(), &st_config(1 << 20), only(&["f", "other_count"])),
        "same sample count",
    );
    assert_eq!(err.field(), Some("other_count"));
    invalid(
        read_in(dir.path(), &st_config(1 << 20), only(&["scalar"])),
        "rank 0",
    );
    invalid(
        read_in(dir.path(), &st_config(1 << 20), only(&["none"])),
        "none of the tensors",
    );
}

/// Unused tensors are not read: a used tensor next to an unused one with invalid BOOL bytes and
/// an unsupported dtype reads fine; the bad BOOL is an error once used.
#[test]
fn safetensors_unused_tensors_are_not_read() {
    let dir = TempDir::new("st-unused");
    write_st(
        &dir,
        &[
            ("x", Dtype::F32, &[2], le(&[1.0f32, 2.0], f32::to_le_bytes)),
            ("bad_bool", Dtype::Bool, &[2], vec![1, 7]),
            ("f64", Dtype::F64, &[2], vec![0; 16]),
        ],
    );
    let sets = read_in(dir.path(), &st_config(1 << 20), only(&["x"])).unwrap();
    assert_eq!(f32s(sets.train.field("x").unwrap()), [1.0, 2.0]);
    let err = invalid(
        read_in(dir.path(), &st_config(1 << 20), only(&["bad_bool"])),
        "must be 0 or 1",
    );
    assert_eq!(err.field(), Some("bad_bool"));
    invalid(read(dir.path(), &st_config(1 << 20)), "F64");
    // A missing file is Io; a file that isn't safetensors is InvalidInput.
    fs::remove_file(dir.path().join("d.safetensors")).unwrap();
    assert_eq!(
        read(dir.path(), &st_config(1)).unwrap_err().kind(),
        DataErrorKind::Io
    );
    dir.write("d.safetensors", b"\x02\0\0\0\0\0\0\0{}x");
    invalid(
        read(dir.path(), &st_config(1)),
        "not a valid safetensors file",
    );
}

// --- Fuzz-style tests ------------------------------------------------------------------------

/// Every truncation and a one-bit flip at every byte position of `valid`, written to `rel` and
/// read with `json`. Each must give an error or a valid result, never a panic; the small
/// `max_bytes` bounds what a corrupted size could make the reader allocate.
fn fuzz(dir: &TempDir, rel: &str, valid: &[u8], json: &str, fields: &FieldSelection) -> (u32, u32) {
    let (mut ok, mut errors) = (0, 0);
    let mut check = |bytes: &[u8]| {
        dir.write(rel, bytes);
        match read_in(dir.path(), json, fields.clone()) {
            Ok(sets) => {
                ok += 1;
                for (_, data) in sets.train.fields() {
                    data.validate().unwrap();
                    assert_eq!(data.shape[0], sets.train.len());
                }
            }
            Err(e) => {
                assert_ne!(e.kind(), DataErrorKind::Cancelled);
                errors += 1;
            }
        }
    };
    for len in 0..valid.len() {
        check(&valid[..len]);
    }
    for (i, bit) in (0..valid.len()).zip((0..8).cycle()) {
        let mut bad = valid.to_vec();
        bad[i] ^= 1 << bit;
        check(&bad);
    }
    let mut state = 7u64;
    for _ in 0..200 {
        let mut bad = valid.to_vec();
        for _ in 0..3 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let at = (state >> 33) as usize % bad.len();
            bad[at] = (state >> 24) as u8;
        }
        check(&bad);
    }
    (ok, errors)
}

#[test]
fn fuzz_png_files() {
    let dir = TempDir::new("fuzz-png");
    let valid = png(2, 3, ColorType::GrayscaleAlpha, 9);
    let json = r#"{"train": {"kind": "image_folder", "path": "imgs", "image_size": [2, 3], "channels": 3}, "max_bytes": 4096}"#;
    let (ok, errors) = fuzz(&dir, "imgs/a/x.png", &valid, json, &FieldSelection::All);
    assert!(errors > 0 && ok > 0, "{ok} ok, {errors} errors");
}

#[test]
fn fuzz_csv_files() {
    let dir = TempDir::new("fuzz-csv");
    let valid = b"\xef\xbb\xbfa,\"b\",label\r\n1,\"-0.5\",3\r\n2.5,1e-3,\"-7\"\n";
    let (ok, errors) = fuzz(&dir, "d.csv", valid, &csv_config(XY), &FieldSelection::All);
    assert!(errors > 0 && ok > 0, "{ok} ok, {errors} errors");
}

#[test]
fn fuzz_safetensors_files() {
    let dir = TempDir::new("fuzz-st");
    let path = write_st(
        &dir,
        &[
            (
                "x",
                Dtype::F16,
                &[3, 2],
                vec![0x00, 0x3C, 0x00, 0xC0, 0x01, 0x00, 0, 0, 0, 0, 0, 0],
            ),
            ("y", Dtype::I64, &[3], le(&[1i64, 2, 3], i64::to_le_bytes)),
            ("m", Dtype::Bool, &[3], vec![0, 1, 0]),
        ],
    );
    let valid = fs::read(&path).unwrap();
    let (ok, errors) = fuzz(
        &dir,
        "d.safetensors",
        &valid,
        &st_config(4096),
        &FieldSelection::All,
    );
    assert!(errors > 0 && ok > 0, "{ok} ok, {errors} errors");
}
