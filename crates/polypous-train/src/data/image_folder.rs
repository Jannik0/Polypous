//! The `image_folder` dataset (`docs/specs/training.md` §3.5.4): one class per sub-directory,
//! PNG images, decoded by [`super::png_decode`].
//!
//! Listing comes first and allocates only names, bounded by the limits it counts as it goes
//! (10,000 classes, 1,000,000 images). Then the memory of the decoded fields is checked against
//! `data.max_bytes` and reserved, and only then are the files read, one at a time, each checked
//! against the 64 MiB file limit before it is read and against `image_size` from its header
//! before it is decoded.
//!
//! Names: classes and images are sorted by the bytes of their UTF-8 names. A class folder whose
//! name isn't UTF-8 is `InvalidInput` (§3.5.4); so is an image file (a regular file ending in
//! `.png`) whose name isn't UTF-8, since its position in the byte order of UTF-8 names would be
//! undefined. Ignored entries may have any name.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use polypous_backend::{DType, TensorData};
use polypous_runtime::CancellationToken;

use super::config::ImageFolderConfig;
use super::png_decode::{self, Expected};
use super::{Budget, DataError, Dataset, FieldSelection, try_bytes};

/// At most 10,000 classes (§3.5.4).
pub(crate) const MAX_CLASSES: usize = 10_000;
/// At most 1,000,000 images (§3.5.4).
pub(crate) const MAX_IMAGES: usize = 1_000_000;
/// Each file at most 64 MiB (§3.5.4).
pub(crate) const MAX_PNG_BYTES: u64 = 64 << 20;

/// Whether `name` ends in `.png`, ASCII case-insensitive.
fn is_png_name(name: &[u8]) -> bool {
    name.len() >= 4 && name[name.len() - 4..].eq_ignore_ascii_case(b".png")
}

/// What an entry is, following symbolic links. A link whose target doesn't exist is neither.
enum Entry {
    Dir,
    File,
    Other,
}

fn entry_kind(path: &Path) -> Result<Entry, DataError> {
    match fs::metadata(path) {
        Ok(m) if m.is_dir() => Ok(Entry::Dir),
        Ok(m) if m.is_file() => Ok(Entry::File),
        Ok(_) => Ok(Entry::Other),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Entry::Other),
        Err(e) => Err(DataError::io(e, path, "read the directory entry")),
    }
}

pub(crate) fn read(
    root: &Path,
    config: &ImageFolderConfig,
    fields: &FieldSelection,
    budget: &mut Budget,
    cancel: &CancellationToken,
) -> Result<Dataset, DataError> {
    let classes = list_classes(root)?;
    let mut images: Vec<(PathBuf, i64)> = Vec::new();
    for (label, class) in classes.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err(DataError::cancelled());
        }
        let dir = root.join(class);
        let names = list_images(&dir, class, MAX_IMAGES - images.len())?;
        images
            .try_reserve(names.len())
            .map_err(|_| DataError::out_of_memory(names.len() as u64, "the image list"))?;
        // label < MAX_CLASSES, exact in i64.
        let label = i64::try_from(label).unwrap_or(i64::MAX);
        images.extend(names.into_iter().map(|n| (dir.join(n), label)));
    }
    let count = images.len() as u64;
    let want_image = fields.wants(&config.image_field);
    let want_label = fields.wants(&config.label_field);
    if !want_image && !want_label {
        return Err(DataError::invalid(format!(
            "neither `{}` nor `{}` is used by the training",
            config.image_field, config.label_field
        ))
        .with_file(root));
    }
    // image_size and channels are validated (1–4,096, 1 or 3), so these products fit easily.
    let [h, w] = config.image_size;
    let channels = config.channels;
    let image_bytes = channels * h * w * 4;
    let mut out = BTreeMap::new();
    if want_label {
        budget
            .charge(count * 8)
            .map_err(|e| e.with_file(root).with_field(&config.label_field))?;
    }
    let mut pixels = if want_image {
        let total = count * image_bytes;
        budget
            .charge(total)
            .map_err(|e| e.with_file(root).with_field(&config.image_field))?;
        let mut v = try_bytes(total, "the decoded images")?;
        // `try_bytes` reserved exactly `total`; filling it allocates nothing.
        #[allow(
            clippy::cast_possible_truncation,
            reason = "total was reserved, so it fits"
        )]
        v.resize(total as usize, 0);
        Some(v)
    } else {
        None
    };
    if want_label {
        let mut labels = try_bytes(count * 8, "the labels")?;
        for (_, label) in &images {
            labels.extend_from_slice(&label.to_le_bytes());
        }
        out.insert(
            config.label_field.clone(),
            TensorData {
                dtype: DType::I64,
                shape: vec![count],
                bytes: labels,
            },
        );
    }
    if let Some(pixels) = &mut pixels {
        #[allow(
            clippy::cast_possible_truncation,
            reason = "validated: ≤ 4,096 and 1 or 3"
        )]
        let expected = Expected {
            size: [h as u32, w as u32],
            channels: channels as usize,
        };
        #[allow(clippy::cast_possible_truncation, reason = "≤ 3 · 4,096² · 4 bytes")]
        let image_bytes = image_bytes as usize;
        let mut file = Vec::new();
        let mut scratch = Vec::new();
        for ((path, _), slot) in images.iter().zip(pixels.chunks_exact_mut(image_bytes)) {
            if cancel.is_cancelled() {
                return Err(DataError::cancelled());
            }
            read_file(path, &mut file)?;
            png_decode::decode(&file, &expected, &mut scratch, slot).map_err(|m| {
                DataError::invalid(m)
                    .with_file(path)
                    .with_field(&config.image_field)
            })?;
        }
    }
    if let Some(pixels) = pixels {
        out.insert(
            config.image_field.clone(),
            TensorData {
                dtype: DType::F32,
                shape: vec![count, channels, h, w],
                bytes: pixels,
            },
        );
    }
    Ok(Dataset::from_checked(count, out))
}

/// The class folders directly inside `root`, sorted by the bytes of their names.
fn list_classes(root: &Path) -> Result<Vec<String>, DataError> {
    let entries =
        fs::read_dir(root).map_err(|e| DataError::io(e, root, "list the image folder"))?;
    let mut classes = Vec::new();
    let mut ignored = 0u64;
    for entry in entries {
        let entry = entry.map_err(|e| DataError::io(e, root, "list the image folder"))?;
        let path = entry.path();
        match entry_kind(&path)? {
            Entry::Dir => {
                let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                    return Err(DataError::invalid(format!(
                        "the class folder name {:?} is not valid UTF-8",
                        entry.file_name()
                    ))
                    .with_file(&path));
                };
                if classes.len() == MAX_CLASSES {
                    return Err(DataError::invalid(format!(
                        "the image folder has more than {MAX_CLASSES} class folders"
                    ))
                    .with_file(root));
                }
                classes.push(name);
            }
            Entry::File | Entry::Other => ignored += 1,
        }
    }
    if ignored > 0 {
        tracing::warn!(
            path = %root.display(),
            ignored,
            "image folder: ignored entries that are not class folders"
        );
    }
    if classes.is_empty() {
        return Err(DataError::invalid(
            "the image folder has no class folders (one sub-directory per class)",
        )
        .with_file(root));
    }
    classes.sort_unstable();
    Ok(classes)
}

/// The PNG images directly inside the class folder `dir`, sorted by the bytes of their names.
/// At most `room` images may be added before the 1,000,000 limit.
fn list_images(dir: &Path, class: &str, room: usize) -> Result<Vec<String>, DataError> {
    let entries = fs::read_dir(dir).map_err(|e| DataError::io(e, dir, "list the class folder"))?;
    let mut names = Vec::new();
    let mut ignored = 0u64;
    for entry in entries {
        let entry = entry.map_err(|e| DataError::io(e, dir, "list the class folder"))?;
        let name = entry.file_name();
        let path = entry.path();
        if !is_png_name(name.as_encoded_bytes()) || !matches!(entry_kind(&path)?, Entry::File) {
            ignored += 1;
            continue;
        }
        let Some(name) = name.to_str().map(str::to_owned) else {
            return Err(DataError::invalid(format!(
                "the image file name {name:?} in class `{class}` is not valid UTF-8"
            ))
            .with_file(&path));
        };
        if names.len() == room {
            return Err(DataError::invalid(format!(
                "the image folder has more than {MAX_IMAGES} images"
            ))
            .with_file(dir));
        }
        names
            .try_reserve(1)
            .map_err(|_| DataError::out_of_memory(1, "the image list"))?;
        names.push(name);
    }
    if ignored > 0 {
        tracing::warn!(
            path = %dir.display(),
            class,
            ignored,
            "image folder: ignored entries of a class folder that are not .png files"
        );
    }
    if names.is_empty() {
        return Err(DataError::invalid(format!(
            "the class folder `{class}` has no PNG image (only files ending in .png are read)"
        ))
        .with_file(dir));
    }
    names.sort_unstable();
    Ok(names)
}

/// Reads `path` into `buf` after checking its size, never more than the limit.
fn read_file(path: &Path, buf: &mut Vec<u8>) -> Result<(), DataError> {
    let mut file =
        fs::File::open(path).map_err(|e| DataError::io(e, path, "open the image file"))?;
    let len = file
        .metadata()
        .map_err(|e| DataError::io(e, path, "read the image file"))?
        .len();
    if len > MAX_PNG_BYTES {
        return Err(DataError::invalid(format!(
            "the image file is {len} bytes; at most 64 MiB ({MAX_PNG_BYTES} bytes) are allowed"
        ))
        .with_file(path));
    }
    buf.clear();
    #[allow(clippy::cast_possible_truncation, reason = "len ≤ 64 MiB")]
    buf.try_reserve_exact(len as usize)
        .map_err(|_| DataError::out_of_memory(len, "an image file"))?;
    (&mut file)
        .take(MAX_PNG_BYTES + 1)
        .read_to_end(buf)
        .map_err(|e| DataError::io(e, path, "read the image file"))?;
    if buf.len() as u64 > MAX_PNG_BYTES {
        return Err(
            DataError::invalid("the image file grew past 64 MiB while it was read").with_file(path),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_names() {
        for yes in [".png", "a.png", "a.PNG", "a.pNg", "x.y.png", "é.png"] {
            assert!(is_png_name(yes.as_bytes()), "{yes}");
        }
        for no in ["png", "a.pn", "a.png.txt", "a_png", "a.jpg", "", "a.png "] {
            assert!(!is_png_name(no.as_bytes()), "{no}");
        }
    }

    /// The 1,000,000-image limit is counted while listing: `list_images` fails as soon as the
    /// room left is exceeded (a small room stands in for a million files).
    #[test]
    fn image_limit_is_counted_while_listing() {
        let dir = std::env::temp_dir().join(format!("polypous-image-limit-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        for name in ["a.png", "b.png", "c.png", "notes.txt"] {
            fs::write(dir.join(name), b"").unwrap();
        }
        assert_eq!(
            list_images(&dir, "c", 3).unwrap(),
            ["a.png", "b.png", "c.png"]
        );
        let err = list_images(&dir, "c", 2).unwrap_err();
        assert!(err.message().contains("more than 1000000 images"), "{err}");
        let err = list_images(&dir, "c", 0).unwrap_err();
        assert!(err.message().contains("more than 1000000 images"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
