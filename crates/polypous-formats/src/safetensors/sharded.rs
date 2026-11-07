//! Sharded checkpoints: `*.safetensors.index.json` with a `weight_map`
//! (`docs/specs/formats.md` §1.5).

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use super::error::truncate_for_message;
use super::json;
use super::{Error, ErrorKind, MAX_INDEX_LEN, SafetensorsFile, TensorInfo, TensorView};

/// A sharded safetensors checkpoint: an index file and the shards it names, all validated.
///
/// Each shard is opened (memory-mapped) once. Every tensor resolves to exactly one shard.
#[derive(Debug)]
pub struct ShardedSafetensors {
    index_path: PathBuf,
    total_size: Option<u64>,
    /// Shards sorted by file name.
    shards: Vec<(String, SafetensorsFile)>,
    /// Tensor name to index into `shards`.
    tensors: BTreeMap<String, usize>,
}

impl ShardedSafetensors {
    /// Opens the index file at `index_path` and every shard it names, and checks that the index
    /// and the shards agree (§1.5). Shards are resolved relative to the index's directory.
    pub fn open(index_path: impl AsRef<Path>) -> Result<ShardedSafetensors, Error> {
        let index_path = index_path.as_ref();
        let raw = read_index(index_path).map_err(|e| e.with_path(index_path))?;
        let dir = match index_path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p,
            _ => Path::new("."),
        };

        // Shard file name to index into `shards`, sorted by name.
        let mut shard_ids = BTreeMap::<&str, usize>::new();
        for (tensor, shard) in &raw.weight_map {
            if !is_plain_file_name(shard) {
                return Err(Error::from(ErrorKind::InvalidShardPath {
                    path: truncate_for_message(shard, 256),
                })
                .with_tensor(tensor)
                .with_path(index_path));
            }
            shard_ids.insert(shard.as_str(), 0);
        }
        let mut shards = Vec::with_capacity(shard_ids.len());
        for (i, (name, id)) in shard_ids.iter_mut().enumerate() {
            *id = i;
            let file = SafetensorsFile::open(dir.join(name))?;
            shards.push(((*name).to_owned(), file));
        }

        let mut tensors = BTreeMap::new();
        for (tensor, shard) in &raw.weight_map {
            // Every shard name was inserted into `shard_ids` above.
            let Some(&id) = shard_ids.get(shard.as_str()) else {
                continue;
            };
            tensors.insert(tensor.clone(), id);
        }

        for (id, (shard_name, file)) in shards.iter().enumerate() {
            let shard_path = || dir.join(shard_name);
            for name in file.names() {
                match tensors.get(name) {
                    Some(&mapped) if mapped == id => {}
                    Some(&mapped) => {
                        let mapped = shards
                            .get(mapped)
                            .map(|(n, _)| n.clone())
                            .unwrap_or_default();
                        return Err(Error::from(ErrorKind::InMultipleShards {
                            mapped,
                            found: shard_name.clone(),
                        })
                        .with_tensor(name)
                        .with_path(&shard_path()));
                    }
                    None => {
                        return Err(Error::from(ErrorKind::UnlistedInShard {
                            shard: shard_name.clone(),
                        })
                        .with_tensor(name)
                        .with_path(&shard_path()));
                    }
                }
            }
        }
        for (tensor, &id) in &tensors {
            if let Some((shard_name, file)) = shards.get(id)
                && !file.contains(tensor)
            {
                return Err(Error::from(ErrorKind::MissingInShard {
                    shard: shard_name.clone(),
                })
                .with_tensor(tensor)
                .with_path(&dir.join(shard_name)));
            }
        }

        Ok(ShardedSafetensors {
            index_path: index_path.to_path_buf(),
            total_size: raw.total_size,
            shards,
            tensors,
        })
    }

    /// The index file's path.
    pub fn index_path(&self) -> &Path {
        &self.index_path
    }

    /// The index's `metadata.total_size`, if it is a non-negative integer. Not checked against
    /// the shards.
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint has no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Tensor names, sorted byte-wise.
    pub fn names(&self) -> impl Iterator<Item = &str> + '_ {
        self.tensors.keys().map(String::as_str)
    }

    /// Whether a tensor with this name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// The file name of the shard that holds the named tensor.
    pub fn shard_of(&self, name: &str) -> Option<&str> {
        self.shard(name).map(|(n, _)| n.as_str())
    }

    /// Metadata of the named tensor, or `None` if it doesn't exist.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.shard(name).and_then(|(_, f)| f.info(name))
    }

    /// A view of the named tensor, borrowed from its shard.
    ///
    /// Fails with [`ErrorKind::TensorNotFound`] if no tensor has this name.
    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>, Error> {
        match self.shard(name) {
            Some((_, file)) => file.tensor(name),
            None => Err(Error::from(ErrorKind::TensorNotFound)
                .with_tensor(name)
                .with_path(&self.index_path)),
        }
    }

    /// The shards, sorted by file name.
    pub fn shards(&self) -> impl Iterator<Item = &SafetensorsFile> + '_ {
        self.shards.iter().map(|(_, f)| f)
    }

    fn shard(&self, name: &str) -> Option<&(String, SafetensorsFile)> {
        self.tensors.get(name).and_then(|&id| self.shards.get(id))
    }
}

/// Reads and parses the index, bounded by [`MAX_INDEX_LEN`].
fn read_index(path: &Path) -> Result<json::RawIndex, Error> {
    let too_large = || ErrorKind::IndexTooLarge { max: MAX_INDEX_LEN };
    let file = File::open(path).map_err(ErrorKind::Io)?;
    let len = file.metadata().map_err(ErrorKind::Io)?.len();
    if len > MAX_INDEX_LEN {
        return Err(too_large().into());
    }
    // The length check above can race with a growing file, so the read is bounded as well.
    let mut buf = Vec::new();
    file.take(MAX_INDEX_LEN + 1)
        .read_to_end(&mut buf)
        .map_err(ErrorKind::Io)?;
    if buf.len() as u64 > MAX_INDEX_LEN {
        return Err(too_large().into());
    }
    let text = String::from_utf8(buf)
        .map_err(|_| ErrorKind::InvalidIndex("index file is not valid UTF-8".into()))?;
    json::parse_index(&text)
}

/// Whether `name` is a plain file name: no directory part, no Windows drive or stream syntax.
fn is_plain_file_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains(['/', '\\', ':', '\0'])
}

#[cfg(test)]
mod tests {
    use super::is_plain_file_name;

    #[test]
    fn plain_file_names() {
        assert!(is_plain_file_name("model-00001-of-00002.safetensors"));
        assert!(is_plain_file_name("..safetensors"));
        for bad in [
            "",
            ".",
            "..",
            "../x.safetensors",
            "a/b.safetensors",
            "a\\b.safetensors",
            "/abs.safetensors",
            "C:x.safetensors",
            "C:\\x.safetensors",
            "x.safetensors:stream",
            "x\0.safetensors",
        ] {
            assert!(!is_plain_file_name(bad), "{bad:?}");
        }
    }
}
