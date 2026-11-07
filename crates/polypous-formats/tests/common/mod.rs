//! Helpers shared by the integration tests.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(dead_code)] // Each test binary uses a subset of the helpers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// A fresh directory under the system temp dir, removed on drop (best effort).
///
/// Declare it before any `SafetensorsFile` that maps a file in it, so the map is dropped first
/// (Windows can't delete a mapped file).
pub struct TestDir(PathBuf);

impl TestDir {
    pub fn new(label: &str) -> TestDir {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "polypous-formats-{label}-{}-{n}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TestDir(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    /// Writes a new file. Never used to overwrite a file that may still be mapped.
    pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.join(name);
        assert!(!path.exists(), "test helper must not overwrite {path:?}");
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A file from a header string and a data buffer; the length prefix is the header's length.
pub fn raw_file(header: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Little-endian bytes of f32 values.
pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
