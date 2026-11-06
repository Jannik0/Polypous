//! The crash journal of the in-process fallback (ADR-0013 §5; placement.md §1, "Fallback").
//!
//! Before each in-process probe, the device's cache entry (with the status it gets if the probe
//! kills the process: `crashed during probe`, not isolated) is written and flushed to a journal
//! file in the cache directory; after the probe the file is emptied. A journal left non-empty by a
//! dead process means that probe crashed it.
//!
//! Each probe call has its own file, `probe-journal-<pid>-<n>.json`, held with an exclusive file
//! lock while it is in use. [`recover`] only reads files it can lock, so a journal of another
//! process that is still probing is never mistaken for a crash.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use super::cache::CacheEntry;

const PREFIX: &str = "probe-journal-";
const SUFFIX: &str = ".json";
/// Largest journal file read.
const MAX_JOURNAL_BYTES: u64 = 64 * 1024;

/// An open, locked journal file. Dropping it removes the file.
pub(crate) struct Journal {
    file: Option<File>,
    path: PathBuf,
}

impl Journal {
    /// Creates and locks a new journal file in `dir`.
    pub(crate) fn create(dir: &Path) -> io::Result<Journal> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        fs::create_dir_all(dir)?;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let path = dir.join(format!(
            "{PREFIX}{}-{nanos}-{}{SUFFIX}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.lock()?;
        Ok(Journal {
            file: Some(file),
            path,
        })
    }

    /// Records that `entry`'s device is about to be probed, and flushes it to disk.
    pub(crate) fn begin(&mut self, entry: &CacheEntry) -> io::Result<()> {
        let json = serde_json::to_vec(entry).map_err(io::Error::other)?;
        let file = self.file()?;
        file.set_len(0)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&json)?;
        file.sync_all()
    }

    /// Records that the probe returned.
    pub(crate) fn end(&mut self) -> io::Result<()> {
        let file = self.file()?;
        file.set_len(0)?;
        file.sync_all()
    }

    fn file(&mut self) -> io::Result<&mut File> {
        self.file
            .as_mut()
            .ok_or_else(|| io::Error::other("the journal is closed"))
    }
}

impl Drop for Journal {
    fn drop(&mut self) {
        // Closing the file releases the lock.
        self.file = None;
        let _ = fs::remove_file(&self.path);
    }
}

/// Reads and removes the journals in `dir` that no running process holds: each non-empty one is
/// the entry of a probe that crashed its process.
pub(crate) fn recover(dir: &Path) -> Vec<CacheEntry> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut crashed = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(PREFIX) || !name.ends_with(SUFFIX) {
            continue;
        }
        let path = entry.path();
        let Ok(file) = OpenOptions::new().read(true).write(true).open(&path) else {
            continue;
        };
        // A journal another process holds is in use, not left over.
        if file.try_lock().is_err() {
            continue;
        }
        let mut text = String::new();
        let read = (&file).take(MAX_JOURNAL_BYTES).read_to_string(&mut text);
        drop(file);
        if read.is_ok() && !text.trim().is_empty() {
            match serde_json::from_str::<CacheEntry>(&text) {
                Ok(entry) => {
                    tracing::warn!(
                        device = %entry.device.id,
                        journal = %path.display(),
                        "a previous in-process probe crashed the process: marking the device \
                         crashed during probe"
                    );
                    crashed.push(entry);
                }
                Err(error) => {
                    tracing::warn!(journal = %path.display(), %error, "ignoring a malformed crash journal");
                }
            }
        }
        let _ = fs::remove_file(&path);
    }
    crashed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceId, DeviceInfo, DeviceStatus, UnavailableReason, unprobed_device};

    fn entry() -> CacheEntry {
        CacheEntry {
            device: DeviceInfo {
                status: DeviceStatus::Unavailable {
                    reason: UnavailableReason::CrashedDuringProbe,
                    isolated: false,
                },
                ..unprobed_device(DeviceId::Cuda { index: 0 })
            },
            key: None,
            enum_fingerprint: None,
            fingerprint: Some("fp".into()),
        }
    }

    fn dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "polypous-test-journal-{name}-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn held_journals_are_not_leftovers_and_dropped_ones_vanish() {
        let dir = dir("held");
        let mut journal = Journal::create(&dir).unwrap();
        journal.begin(&entry()).unwrap();
        // Another "process" (here: a second handle) must not take it while it is locked.
        assert!(recover(&dir).is_empty());
        journal.end().unwrap();
        drop(journal);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 0);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn leftover_entries_are_recovered_once() {
        let dir = dir("leftover");
        fs::create_dir_all(&dir).unwrap();
        // What a crashed process leaves: a non-empty, unlocked journal.
        let path = dir.join(format!("{PREFIX}1-2-3{SUFFIX}"));
        fs::write(&path, serde_json::to_vec(&entry()).unwrap()).unwrap();
        fs::write(dir.join(format!("{PREFIX}4-5-6{SUFFIX}")), "").unwrap();
        fs::write(dir.join(format!("{PREFIX}7-8-9{SUFFIX}")), "garbage").unwrap();
        fs::write(dir.join("unrelated.json"), "x").unwrap();
        assert_eq!(recover(&dir), vec![entry()]);
        assert!(recover(&dir).is_empty());
        assert!(dir.join("unrelated.json").exists());
        let _ = fs::remove_dir_all(&dir);
    }
}
