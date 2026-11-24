//! PNG decoding tests (training.md §3.5.4) with small images generated here: every color type
//! and bit depth, palettes with `tRNS`, alpha, Adam7 interlacing, ancillary chunks, the size and
//! channel rules, and malformed files (truncated, bit-flipped). Expected values are computed by
//! hand from the samples written: `v / (2^d − 1)` as one f32 division.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use super::*;
use png::Encoder;

/// The f32 values of a decoded image.
fn f32s(bytes: &[u8]) -> Vec<f32> {
    let (chunks, _) = bytes.as_chunks::<4>();
    chunks.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn decode_to(file: &[u8], h: u32, w: u32, channels: usize) -> Result<Vec<f32>, String> {
    let mut out = vec![0u8; channels * (h * w) as usize * 4];
    let mut scratch = Vec::new();
    decode(
        file,
        &Expected {
            size: [h, w],
            channels,
        },
        &mut scratch,
        &mut out,
    )?;
    Ok(f32s(&out))
}

fn depth(d: u32) -> BitDepth {
    match d {
        1 => BitDepth::One,
        2 => BitDepth::Two,
        4 => BitDepth::Four,
        8 => BitDepth::Eight,
        _ => BitDepth::Sixteen,
    }
}

fn samples_per_pixel(color: ColorType) -> usize {
    match color {
        ColorType::Grayscale | ColorType::Indexed => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
    }
}

/// Packs rows of samples (`row_samples` per row) at `d` bits, PNG order (big-endian, MSB first,
/// rows padded to whole bytes).
fn pack(samples: &[u16], row_samples: usize, d: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for row in samples.chunks(row_samples) {
        match d {
            16 => row
                .iter()
                .for_each(|v| out.extend_from_slice(&v.to_be_bytes())),
            8 => row.iter().for_each(|&v| out.push(v as u8)),
            _ => {
                let mut acc = 0u8;
                let mut bits = 0;
                for &v in row {
                    acc |= (v as u8) << (8 - d - bits);
                    bits += d;
                    if bits == 8 {
                        out.push(acc);
                        acc = 0;
                        bits = 0;
                    }
                }
                if bits > 0 {
                    out.push(acc);
                }
            }
        }
    }
    out
}

/// A test image: `h × w` pixels of `color` at depth `d`, samples row-major, pixel-interleaved.
struct Image {
    h: u32,
    w: u32,
    color: ColorType,
    d: u32,
    samples: Vec<u16>,
    palette: Option<Vec<u8>>,
    trns: Option<Vec<u8>>,
}

impl Image {
    /// Deterministic samples covering 0 and `2^d − 1`.
    fn new(h: u32, w: u32, color: ColorType, d: u32) -> Self {
        let s = samples_per_pixel(color);
        let max = (1u32 << d) - 1;
        let n = (h * w) as usize * s;
        let samples = (0..n)
            .map(|i| match i % 5 {
                0 => 0,
                1 => max as u16,
                _ => ((i as u64 * 2_654_435_761) % u64::from(max + 1)) as u16,
            })
            .collect();
        Self {
            h,
            w,
            color,
            d,
            samples,
            palette: None,
            trns: None,
        }
    }

    fn with_palette(mut self, entries: usize) -> Self {
        self.palette = Some((0..entries * 3).map(|i| (i * 37 % 256) as u8).collect());
        self.trns = Some(vec![0, 128, 255]);
        self
    }

    fn raw(&self) -> Vec<u8> {
        pack(
            &self.samples,
            self.w as usize * samples_per_pixel(self.color),
            self.d,
        )
    }

    /// Encoded with the `png` crate, optionally with extra chunks before the image data.
    fn encode(&self, extra: bool) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut enc = Encoder::new(&mut buf, self.w, self.h);
        enc.set_color(self.color);
        enc.set_depth(depth(self.d));
        if let Some(p) = &self.palette {
            enc.set_palette(p.clone());
        }
        if let Some(t) = &self.trns {
            enc.set_trns(t.clone());
        }
        if extra {
            enc.set_source_gamma(png::ScaledFloat::new(0.3));
            enc.add_text_chunk("Comment".into(), "hello".into())
                .unwrap();
            enc.add_ztxt_chunk("Big".into(), "z".repeat(10_000))
                .unwrap();
        }
        let mut writer = enc.write_header().unwrap();
        if extra {
            writer
                .write_chunk(png::chunk::ChunkType(*b"prVt"), b"private ancillary data")
                .unwrap();
        }
        writer.write_image_data(&self.raw()).unwrap();
        writer.finish().unwrap();
        buf
    }

    /// The expected tensor, `[C, H, W]`, computed from the samples.
    fn expected(&self, channels: usize) -> Vec<f32> {
        let s = samples_per_pixel(self.color);
        let plane = (self.h * self.w) as usize;
        let max = ((1u32 << self.d) - 1) as f32;
        let mut out = vec![0.0f32; channels * plane];
        for i in 0..plane {
            let px = &self.samples[i * s..(i + 1) * s];
            let values: Vec<f32> = match self.color {
                ColorType::Grayscale | ColorType::GrayscaleAlpha => {
                    vec![f32::from(px[0]) / max; channels]
                }
                ColorType::Rgb | ColorType::Rgba => {
                    (0..3).map(|c| f32::from(px[c]) / max).collect()
                }
                ColorType::Indexed => {
                    let p = self.palette.as_ref().unwrap();
                    let k = px[0] as usize * 3;
                    (0..3).map(|c| f32::from(p[k + c]) / 255.0).collect()
                }
            };
            for (c, v) in values.into_iter().enumerate() {
                out[c * plane + i] = v;
            }
        }
        out
    }
}

/// Every color type and bit depth of the PNG standard.
const COMBOS: [(ColorType, u32); 15] = [
    (ColorType::Grayscale, 1),
    (ColorType::Grayscale, 2),
    (ColorType::Grayscale, 4),
    (ColorType::Grayscale, 8),
    (ColorType::Grayscale, 16),
    (ColorType::Rgb, 8),
    (ColorType::Rgb, 16),
    (ColorType::Indexed, 1),
    (ColorType::Indexed, 2),
    (ColorType::Indexed, 4),
    (ColorType::Indexed, 8),
    (ColorType::GrayscaleAlpha, 8),
    (ColorType::GrayscaleAlpha, 16),
    (ColorType::Rgba, 8),
    (ColorType::Rgba, 16),
];

fn image(color: ColorType, d: u32) -> Image {
    // Odd sizes, so sub-byte rows end in padding bits.
    let img = Image::new(3, 5, color, d);
    if color == ColorType::Indexed {
        img.with_palette(1 << d)
    } else {
        img
    }
}

fn is_gray(color: ColorType) -> bool {
    matches!(color, ColorType::Grayscale | ColorType::GrayscaleAlpha)
}

#[test]
fn every_color_type_and_bit_depth() {
    for (color, d) in COMBOS {
        let img = image(color, d);
        for extra in [false, true] {
            let file = img.encode(extra);
            if is_gray(color) {
                assert_eq!(
                    decode_to(&file, 3, 5, 1).unwrap(),
                    img.expected(1),
                    "{color:?} {d}"
                );
            } else {
                let err = decode_to(&file, 3, 5, 1).unwrap_err();
                assert!(err.contains("`channels` is 1"), "{color:?} {d}: {err}");
            }
            // channels: 3 replicates gray; RGB and palette are as stored, alpha dropped.
            assert_eq!(
                decode_to(&file, 3, 5, 3).unwrap(),
                img.expected(3),
                "{color:?} {d}"
            );
        }
    }
}

/// Hand-computed values: `v / (2^d − 1)` as one f32 division.
#[test]
fn values_are_one_f32_division() {
    let mut img = Image::new(1, 4, ColorType::Grayscale, 16);
    img.samples = vec![0, 1, 32_768, 65_535];
    assert_eq!(
        decode_to(&img.encode(false), 1, 4, 1).unwrap(),
        [0.0, 1.0f32 / 65_535.0, 32_768.0f32 / 65_535.0, 1.0]
    );
    let mut img = Image::new(1, 3, ColorType::Grayscale, 2);
    img.samples = vec![1, 2, 3];
    assert_eq!(
        decode_to(&img.encode(false), 1, 3, 1).unwrap(),
        [1.0f32 / 3.0, 2.0f32 / 3.0, 1.0]
    );
    let mut img = Image::new(1, 2, ColorType::Rgba, 8);
    img.samples = vec![1, 2, 3, 0, 255, 128, 0, 7];
    // Alpha 0 and 7 are dropped, not composited.
    assert_eq!(
        decode_to(&img.encode(false), 1, 2, 3).unwrap(),
        [
            1.0 / 255.0,
            1.0,
            2.0 / 255.0,
            128.0 / 255.0,
            3.0 / 255.0,
            0.0
        ]
    );
    // 1-bit gray: 0 and 1.
    let mut img = Image::new(1, 9, ColorType::Grayscale, 1);
    img.samples = vec![1, 0, 1, 1, 0, 0, 0, 1, 1];
    assert_eq!(
        decode_to(&img.encode(false), 1, 9, 1).unwrap(),
        [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn palette_with_trns_and_bad_index() {
    let mut img = Image::new(1, 3, ColorType::Indexed, 8);
    img.palette = Some(vec![10, 20, 30, 40, 50, 60]);
    img.trns = Some(vec![0]);
    img.samples = vec![1, 0, 1];
    assert_eq!(
        decode_to(&img.encode(false), 1, 3, 3).unwrap(),
        [
            40.0 / 255.0,
            10.0 / 255.0,
            40.0 / 255.0,
            50.0 / 255.0,
            20.0 / 255.0,
            50.0 / 255.0,
            60.0 / 255.0,
            30.0 / 255.0,
            60.0 / 255.0
        ]
    );
    img.samples = vec![0, 2, 1];
    let err = decode_to(&img.encode(false), 1, 3, 3).unwrap_err();
    assert!(err.contains("palette index 2"), "{err}");
}

#[test]
fn size_is_checked_from_the_header() {
    let file = image(ColorType::Grayscale, 8).encode(false);
    let err = decode_to(&file, 5, 3, 1).unwrap_err();
    assert!(err.contains("3×5") && err.contains("[5, 3]"), "{err}");
    assert!(decode_to(&file, 3, 6, 1).is_err());
}

// --- Adam7: the `png` encoder writes no interlaced data, so a minimal encoder writes it here. --

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let mut body = kind.to_vec();
    body.extend_from_slice(data);
    out.extend_from_slice(&body);
    out.extend_from_slice(&crc32(&body).to_be_bytes());
}

/// zlib with stored (uncompressed) deflate blocks.
fn zlib_stored(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65_535).collect()
    };
    for (i, block) in blocks.iter().enumerate() {
        out.push(u8::from(i + 1 == blocks.len()));
        let len = block.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + u32::from(x)) % 65_521;
        b = (b + a) % 65_521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn color_code(color: ColorType) -> u8 {
    match color {
        ColorType::Grayscale => 0,
        ColorType::Rgb => 2,
        ColorType::Indexed => 3,
        ColorType::GrayscaleAlpha => 4,
        ColorType::Rgba => 6,
    }
}

/// The image Adam7-interlaced (filter type 0 on every row).
fn encode_adam7(img: &Image) -> Vec<u8> {
    const PASSES: [(u32, u32, u32, u32); 7] = [
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ];
    let s = samples_per_pixel(img.color);
    let mut raw = Vec::new();
    for (x0, y0, dx, dy) in PASSES {
        let cols: Vec<u32> = (x0..img.w).step_by(dx as usize).collect();
        let rows: Vec<u32> = (y0..img.h).step_by(dy as usize).collect();
        if cols.is_empty() || rows.is_empty() {
            continue;
        }
        for &y in &rows {
            let mut row = Vec::new();
            for &x in &cols {
                let at = ((y * img.w + x) as usize) * s;
                row.extend_from_slice(&img.samples[at..at + s]);
            }
            raw.push(0);
            raw.extend(pack(&row, row.len(), img.d));
        }
    }
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&img.w.to_be_bytes());
    ihdr.extend_from_slice(&img.h.to_be_bytes());
    ihdr.extend_from_slice(&[img.d as u8, color_code(img.color), 0, 0, 1]);
    chunk(&mut out, b"IHDR", &ihdr);
    if let Some(p) = &img.palette {
        chunk(&mut out, b"PLTE", p);
    }
    if let Some(t) = &img.trns {
        chunk(&mut out, b"tRNS", t);
    }
    chunk(&mut out, b"IDAT", &zlib_stored(&raw));
    chunk(&mut out, b"IEND", &[]);
    out
}

#[test]
fn interlaced_images_have_the_same_pixels() {
    for (color, d) in COMBOS {
        // 11 × 13 covers every Adam7 pass with partial blocks.
        let img = Image::new(11, 13, color, d);
        let img = if color == ColorType::Indexed {
            img.with_palette(1 << d)
        } else {
            img
        };
        let file = encode_adam7(&img);
        assert_eq!(
            decode_to(&file, 11, 13, 3).unwrap(),
            img.expected(3),
            "{color:?} {d}"
        );
        // The same image written without interlacing decodes to the same values.
        assert_eq!(
            decode_to(&img.encode(false), 11, 13, 3).unwrap(),
            img.expected(3),
            "{color:?} {d}"
        );
    }
    // Tiny images where some passes are empty.
    for (h, w) in [(1, 1), (1, 2), (2, 1), (3, 3)] {
        let img = Image::new(h, w, ColorType::Grayscale, 8);
        assert_eq!(
            decode_to(&encode_adam7(&img), h, w, 1).unwrap(),
            img.expected(1)
        );
    }
}

/// The widest allowed row (4,096 RGBA samples of 16 bits) decodes within the decoder limit.
#[test]
fn widest_image_fits_the_decoder_limit() {
    let img = Image::new(2, 4096, ColorType::Rgba, 16);
    assert_eq!(
        decode_to(&img.encode(false), 2, 4096, 3).unwrap(),
        img.expected(3)
    );
    let img = Image::new(4096, 1, ColorType::Rgba, 16);
    assert_eq!(
        decode_to(&encode_adam7(&img), 4096, 1, 3).unwrap(),
        img.expected(3)
    );
}

#[test]
fn a_huge_compressed_text_chunk_is_not_inflated() {
    // 64 MiB of text deflates to a small chunk; the decoder skips text chunks.
    let img = Image::new(1, 1, ColorType::Grayscale, 8);
    let mut buf = Vec::new();
    let mut enc = Encoder::new(&mut buf, 1, 1);
    enc.set_color(ColorType::Grayscale);
    enc.set_depth(BitDepth::Eight);
    enc.add_ztxt_chunk("Bomb".into(), "a".repeat(64 << 20))
        .unwrap();
    let mut writer = enc.write_header().unwrap();
    writer.write_image_data(&img.raw()).unwrap();
    writer.finish().unwrap();
    assert!(buf.len() < 1 << 20);
    assert_eq!(decode_to(&buf, 1, 1, 1).unwrap(), img.expected(1));
}

/// Fuzz-style: every truncation and a bit flip at every byte of valid files, for several color
/// types and interlacing. Each result is an error or a valid image of the right size, never a
/// panic.
#[test]
fn malformed_files_are_errors_not_panics() {
    let files = [
        (image(ColorType::Grayscale, 1).encode(true), 3, 5),
        (image(ColorType::Indexed, 4).encode(false), 3, 5),
        (image(ColorType::Rgba, 16).encode(false), 3, 5),
        (image(ColorType::GrayscaleAlpha, 8).encode(false), 3, 5),
        (encode_adam7(&Image::new(11, 13, ColorType::Rgb, 8)), 11, 13),
        (encode_adam7(&image(ColorType::Indexed, 2)), 3, 5),
    ];
    let mut errors = 0;
    let mut total = 0;
    for (file, h, w) in &files {
        for len in 0..file.len() {
            total += 1;
            if decode_to(&file[..len], *h, *w, 3).is_err() {
                errors += 1;
            }
        }
        for (i, bit) in (0..file.len()).zip((0..8).cycle()) {
            let mut bad = file.clone();
            bad[i] ^= 1 << bit;
            total += 1;
            match decode_to(&bad, *h, *w, 3) {
                Ok(values) => assert!(values.iter().all(|v| (0.0..=1.0).contains(v))),
                Err(_) => errors += 1,
            }
        }
        // Seeded random multi-byte corruption.
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..300 {
            let mut bad = file.clone();
            for _ in 0..4 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                let at = (state >> 33) as usize % bad.len();
                bad[at] = (state >> 20) as u8;
            }
            total += 1;
            if decode_to(&bad, *h, *w, 3).is_err() {
                errors += 1;
            }
        }
    }
    // Every truncation fails (IEND is needed after the image data at least by the CRC'd chunks),
    // and most flips are caught by CRC-32.
    assert!(
        errors * 10 > total * 9,
        "{errors} of {total} malformed files were errors"
    );
}
