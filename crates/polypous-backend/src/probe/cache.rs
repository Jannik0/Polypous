//! The probe cache (ADR-0013 §2; placement.md §1, "Probe cache").
//!
//! One JSON file per Polypous version and set of compiled-in backends in the cache directory,
//! holding the last result of every device with what identifies it: stable key, enumeration
//! fingerprint and the OS driver fingerprint it was probed under. The file is untrusted (size
//! limit, a parse failure means "no cache"), written atomically (temporary file, then rename), and
//! a failure to read or write it is logged, never an error.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::device::{BackendKind, DeviceId, DeviceInfo, DeviceStatus, UnavailableReason};

/// Format version of the cache file.
pub(crate) const CACHE_FORMAT: u32 = 1;

/// Largest cache file read.
const MAX_CACHE_BYTES: u64 = 4 * 1024 * 1024;

/// One cached device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CacheEntry {
    /// The device's last description and status.
    pub(crate) device: DeviceInfo,
    /// Stable key from enumeration (`pci:<address>`), if any.
    pub(crate) key: Option<String>,
    /// Enumeration fingerprint, if any.
    pub(crate) enum_fingerprint: Option<String>,
    /// OS driver fingerprint the device was probed under; `None` if it was unknown.
    pub(crate) fingerprint: Option<String>,
}

impl CacheEntry {
    /// Whether the cached status is a crash or a timeout: such a device is not opened again until
    /// its fingerprint changes or a forced, isolated re-probe.
    pub(crate) fn is_crash(&self) -> bool {
        matches!(
            self.device.status,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::CrashedDuringProbe | UnavailableReason::ProbeTimedOut,
                ..
            }
        )
    }

    /// Whether the entry only says "software rasterizer, not opened".
    pub(crate) fn is_skipped_rasterizer(&self) -> bool {
        matches!(
            self.device.status,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::SoftwareRasterizerSkipped,
                ..
            }
        )
    }
}

/// The cache file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CacheFile {
    pub(crate) format: u32,
    pub(crate) polypous_version: String,
    pub(crate) backends: Vec<BackendKind>,
    /// GPU keys of the OS fingerprint at the last save (empty if unknown). A different set means
    /// a GPU was added or removed.
    pub(crate) gpu_keys: Vec<String>,
    /// Whether `devices` is the result of a probe of all devices that ran to the end. An
    /// in-process probe that crashed the app saved only part of the devices; a probe whose
    /// enumeration failed for a backend misses that backend's devices. Only a complete cache can
    /// be a cache hit for every device.
    pub(crate) complete: bool,
    pub(crate) devices: Vec<CacheEntry>,
}

impl CacheFile {
    /// An empty cache for this build.
    pub(crate) fn empty() -> Self {
        CacheFile {
            format: CACHE_FORMAT,
            polypous_version: env!("CARGO_PKG_VERSION").to_string(),
            backends: BackendKind::compiled_in(),
            gpu_keys: Vec::new(),
            complete: false,
            devices: Vec::new(),
        }
    }

    /// The entry of `id`.
    pub(crate) fn entry(&self, id: DeviceId) -> Option<&CacheEntry> {
        self.devices.iter().find(|entry| entry.device.id == id)
    }

    /// Stores `entry`, replacing the entry with the same id or the same stable key. A "software
    /// rasterizer skipped" result never replaces a probed result of the same device: it says
    /// nothing new, and the probed one is needed when the caller opts in again.
    pub(crate) fn upsert(&mut self, entry: CacheEntry) {
        if entry.is_skipped_rasterizer()
            && self
                .entry(entry.device.id)
                .is_some_and(|old| !old.is_skipped_rasterizer())
        {
            return;
        }
        self.devices.retain(|old| {
            old.device.id != entry.device.id && (old.key.is_none() || old.key != entry.key)
        });
        self.devices.push(entry);
    }
}

/// The cache file name for this build: `probe-cache-v1-<backends>.json`, so builds with other
/// backends (a CPU-only test binary next to a CUDA build) don't share entries.
pub(crate) fn file_name() -> String {
    let backends: Vec<&str> = BackendKind::compiled_in()
        .into_iter()
        .map(BackendKind::as_str)
        .collect();
    format!("probe-cache-v{CACHE_FORMAT}-{}.json", backends.join("-"))
}

/// The platform cache directory for Polypous: `$XDG_CACHE_HOME/polypous` or
/// `~/.cache/polypous` (Linux and other Unix), `~/Library/Caches/polypous` (macOS),
/// `%LOCALAPPDATA%\polypous\cache` (Windows). `None` if the variables it needs are missing.
pub fn default_cache_dir() -> Option<PathBuf> {
    let from = |name: &str| {
        std::env::var_os(name)
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
    };
    if cfg!(windows) {
        from("LOCALAPPDATA").map(|dir| dir.join("polypous").join("cache"))
    } else if cfg!(target_os = "macos") {
        from("HOME").map(|home| home.join("Library").join("Caches").join("polypous"))
    } else {
        from("XDG_CACHE_HOME")
            .or_else(|| from("HOME").map(|home| home.join(".cache")))
            .map(|dir| dir.join("polypous"))
    }
}

/// Reads the cache at `path`. Missing, unreadable, oversized, malformed or written by another
/// version or backend set: `None`.
pub(crate) fn load(path: &Path) -> Option<CacheFile> {
    let file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) => {
            tracing::debug!(path = %path.display(), %error, "no probe cache");
            return None;
        }
    };
    let mut text = String::new();
    if file
        .take(MAX_CACHE_BYTES + 1)
        .read_to_string(&mut text)
        .is_err()
        || text.len() as u64 > MAX_CACHE_BYTES
    {
        tracing::warn!(path = %path.display(), "ignoring an unreadable or oversized probe cache");
        return None;
    }
    let cache: CacheFile = match serde_json::from_str(&text) {
        Ok(cache) => cache,
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "ignoring a malformed probe cache");
            return None;
        }
    };
    let expected = CacheFile::empty();
    if cache.format != expected.format
        || cache.polypous_version != expected.polypous_version
        || cache.backends != expected.backends
    {
        tracing::debug!(path = %path.display(), "probe cache is from another Polypous build");
        return None;
    }
    Some(cache)
}

/// Writes the cache atomically. Errors are logged.
pub(crate) fn save(path: &Path, cache: &CacheFile) {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let temporary = path.with_extension(format!(
            "tmp-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let json = serde_json::to_vec_pretty(cache).map_err(std::io::Error::other)?;
        let mut file = fs::File::create(&temporary)?;
        file.write_all(&json)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path).inspect_err(|_| {
            let _ = fs::remove_file(&temporary);
        })
    })();
    match result {
        Ok(()) => tracing::debug!(path = %path.display(), "saved the probe cache"),
        Err(error) => {
            tracing::warn!(path = %path.display(), %error, "could not save the probe cache");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::unprobed_device;

    pub(crate) fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "polypous-test-{name}-{}-{:?}",
            std::process::id(),
            std::time::SystemTime::now()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    fn entry(id: DeviceId, status: DeviceStatus, key: Option<&str>) -> CacheEntry {
        CacheEntry {
            device: DeviceInfo {
                status,
                ..unprobed_device(id)
            },
            key: key.map(Into::into),
            enum_fingerprint: None,
            fingerprint: Some("fp".into()),
        }
    }

    #[test]
    fn round_trip_and_rejections() {
        let dir = temp_dir("cache");
        let path = dir.join(file_name());
        assert!(load(&path).is_none());
        let mut cache = CacheFile::empty();
        cache.upsert(entry(
            DeviceId::Cpu,
            DeviceStatus::Available { isolated: true },
            None,
        ));
        save(&path, &cache);
        assert_eq!(load(&path), Some(cache.clone()));

        fs::write(&path, "{not json").unwrap();
        assert!(load(&path).is_none());

        let mut other = cache.clone();
        other.polypous_version = "999.0.0".into();
        fs::write(&path, serde_json::to_string(&other).unwrap()).unwrap();
        assert!(load(&path).is_none());

        let huge = vec![b' '; usize::try_from(MAX_CACHE_BYTES).unwrap() + 10];
        fs::write(&path, huge).unwrap();
        assert!(load(&path).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn upsert_replaces_by_id_or_key_and_keeps_probed_rasterizers() {
        let lavapipe = crate::probe::session::tests::lavapipe();
        let mut cache = CacheFile::empty();
        let crashed = DeviceStatus::Unavailable {
            reason: UnavailableReason::CrashedDuringProbe,
            isolated: true,
        };
        cache.upsert(entry(lavapipe, crashed.clone(), None));
        cache.upsert(entry(
            lavapipe,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::SoftwareRasterizerSkipped,
                isolated: false,
            },
            None,
        ));
        assert_eq!(cache.entry(lavapipe).unwrap().device.status, crashed);
        assert!(cache.entry(lavapipe).unwrap().is_crash());

        let gpu = |index| DeviceId::Vulkan {
            adapter: crate::device::AdapterKind::DiscreteGpu,
            index,
        };
        let available = DeviceStatus::Available { isolated: true };
        cache.upsert(entry(gpu(0), available.clone(), Some("pci:0000:01:00.0")));
        // The same card under a new index replaces the old entry.
        cache.upsert(entry(gpu(1), available, Some("pci:0000:01:00.0")));
        assert!(cache.entry(gpu(0)).is_none());
        assert!(cache.entry(gpu(1)).is_some());
        assert_eq!(cache.devices.len(), 2);
    }

    #[test]
    fn file_name_names_the_backends() {
        let name = file_name();
        assert!(name.starts_with("probe-cache-v1-"), "{name}");
        assert!(name.ends_with(".json"));
    }
}
