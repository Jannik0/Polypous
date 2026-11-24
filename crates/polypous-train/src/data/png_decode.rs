//! PNG decoding for the `image_folder` dataset (`docs/specs/training.md` §3.5.4), with the `png`
//! crate.
//!
//! The decoder runs without transformations (`Transformations::IDENTITY`): it returns the raw
//! samples, de-interlaced, and this module does the rest exactly as §3.5.4 says, so no conversion
//! of the crate's (gamma, alpha, 16-to-8-bit) can creep in:
//!
//! - sub-byte gray and palette samples are unpacked, most significant bits first;
//! - a palette index is looked up in `PLTE` (8-bit RGB entries); an index past its end is an
//!   invalid file;
//! - alpha (`tRNS` and the alpha sample of gray+alpha and RGBA) is dropped, not composited;
//! - gray gives 1 channel, RGB and palette 3; with `channels: 3` gray is replicated, with
//!   `channels: 1` an RGB or palette image is an error;
//! - each value is `v / (2^d − 1)`, one `f32` division of exact integers (`d = 8` for palette).
//!
//! The size and the color type are checked from the header before any pixel data is decoded.
//! Text and ICC chunks are skipped (`set_ignore_text_chunk`, `set_ignore_iccp_chunk`), so a
//! compressed chunk can't make the decoder inflate it, and the decoder's own allocations are
//! capped by [`DECODER_LIMIT`]. CRC and Adler-32 checksums are verified.

use std::io::Cursor;

use png::{BitDepth, ColorType, DecodeOptions, Decoder, Limits};

/// The cap on the `png` decoder's own buffers (ancillary chunks it keeps, row buffers). Pixel
/// rows of the largest allowed image (4,096 RGBA samples of 16 bits: 32 KiB) fit easily; the
/// output frame is our buffer and isn't counted.
pub(crate) const DECODER_LIMIT: usize = 16 << 20;

/// What the caller asks for.
pub(crate) struct Expected {
    /// `[H, W]`.
    pub(crate) size: [u32; 2],
    /// `1` or `3`.
    pub(crate) channels: usize,
}

/// Decodes `file` into `out` (`channels · H · W` little-endian `f32` values, channel first).
/// `scratch` is the reusable frame buffer. The error is a message for an `InvalidInput` naming
/// the file.
pub(crate) fn decode(
    file: &[u8],
    expected: &Expected,
    scratch: &mut Vec<u8>,
    out: &mut [u8],
) -> Result<(), String> {
    let mut options = DecodeOptions::default();
    options.set_ignore_text_chunk(true);
    options.set_ignore_iccp_chunk(true);
    options.set_ignore_adler32(false);
    let mut decoder = Decoder::new_with_options(Cursor::new(file), options);
    decoder.set_limits(Limits {
        bytes: DECODER_LIMIT,
    });
    let invalid = |e: png::DecodingError| format!("not a valid PNG file: {e}");
    let header = decoder.read_header_info().map_err(invalid)?;
    let (width, height) = (header.width, header.height);
    let [h, w] = expected.size;
    if (height, width) != (h, w) {
        return Err(format!(
            "the image is {height}×{width} (height × width), but `image_size` is [{h}, {w}]; \
             images are not resized"
        ));
    }
    let color = header.color_type;
    if expected.channels == 1
        && matches!(color, ColorType::Rgb | ColorType::Rgba | ColorType::Indexed)
    {
        return Err(format!(
            "the image is {} ({color:?}), but `channels` is 1; there is no color conversion",
            if color == ColorType::Indexed {
                "a palette (RGB) image"
            } else {
                "an RGB image"
            }
        ));
    }
    let mut reader = decoder.read_info().map_err(invalid)?;
    let len = reader
        .output_buffer_size()
        .ok_or_else(|| "the image is too large to decode".to_owned())?;
    scratch.clear();
    scratch
        .try_reserve(len)
        .map_err(|_| format!("could not reserve {len} bytes to decode the image"))?;
    scratch.resize(len, 0);
    let frame = reader.next_frame(scratch).map_err(invalid)?;
    if (frame.width, frame.height) != (width, height) {
        return Err(format!(
            "not a valid PNG file: the first frame is {}×{}, the header says {height}×{width}",
            frame.height, frame.width
        ));
    }
    let (color, depth) = (frame.color_type, frame.bit_depth);
    let d = match depth {
        BitDepth::One => 1,
        BitDepth::Two => 2,
        BitDepth::Four => 4,
        BitDepth::Eight => 8,
        BitDepth::Sixteen => 16,
    };
    let samples = match color {
        ColorType::Grayscale | ColorType::Indexed => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
    };
    let palette = match color {
        ColorType::Indexed => Some(
            reader
                .info()
                .palette
                .as_deref()
                .ok_or_else(|| "a palette image without a PLTE chunk".to_owned())?
                .to_vec(),
        ),
        _ => None,
    };
    let (w, h) = (width as usize, height as usize);
    let plane = w * h;
    let line = frame.line_size;
    if line.checked_mul(h).is_none_or(|n| n > scratch.len())
        || out.len() != expected.channels * plane * 4
    {
        return Err("internal: unexpected decoded frame size".to_owned());
    }
    let gray = matches!(color, ColorType::Grayscale | ColorType::GrayscaleAlpha);
    // 2^d − 1, exact in f32 (≤ 65,535).
    let max = f32::from(u16::try_from((1u32 << d) - 1).unwrap_or(u16::MAX));
    let mut put = |c: usize, i: usize, v: f32| {
        let at = (c * plane + i) * 4;
        out[at..at + 4].copy_from_slice(&v.to_le_bytes());
    };
    for y in 0..h {
        let row = &scratch[y * line..(y + 1) * line];
        for x in 0..w {
            let i = y * w + x;
            let first = x * samples;
            if let Some(palette) = &palette {
                let index = usize::from(sample(row, first, d));
                let Some(rgb) = palette.get(3 * index..3 * index + 3) else {
                    return Err(format!(
                        "not a valid PNG file: palette index {index} at pixel ({y}, {x}) is past \
                         the {}-entry palette",
                        palette.len() / 3
                    ));
                };
                for (c, &v) in rgb.iter().enumerate() {
                    put(c, i, f32::from(v) / 255.0);
                }
            } else if gray {
                let v = f32::from(sample(row, first, d)) / max;
                for c in 0..expected.channels {
                    put(c, i, v);
                }
            } else {
                for c in 0..3 {
                    put(c, i, f32::from(sample(row, first + c, d)) / max);
                }
            }
        }
    }
    Ok(())
}

/// Sample `k` of a row of `d`-bit samples (PNG order: big-endian, most significant bits first).
fn sample(row: &[u8], k: usize, d: u32) -> u16 {
    match d {
        16 => u16::from_be_bytes([row[2 * k], row[2 * k + 1]]),
        8 => u16::from(row[k]),
        _ => {
            let bit = k * d as usize;
            let shift = 8 - d as usize - bit % 8;
            u16::from((row[bit / 8] >> shift) & ((1u8 << d) - 1))
        }
    }
}

#[cfg(test)]
#[path = "png_decode_tests.rs"]
mod tests;
