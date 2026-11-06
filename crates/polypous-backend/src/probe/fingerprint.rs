//! Driver fingerprints from operating-system sources that load no GPU driver (ADR-0013 §2;
//! placement.md §1, "Probe cache").
//!
//! One interface per OS: [`current`] returns a stable key and a driver fingerprint per GPU, plus a
//! fingerprint for devices that have no stable key, or `None` ("unknown"). Unknown makes the probe
//! take the enumerate-only path.
//!
//! - **Linux** (this module): GPUs are the PCI display controllers in `/sys/bus/pci/devices`
//!   (class `0x03xxxx`), keyed by PCI address. A GPU's fingerprint has its PCI vendor/device ids,
//!   kernel driver, the driver module's version (`/sys/module/<driver>/version`) or else the
//!   kernel release, the NVIDIA driver version (`/proc/driver/nvidia/version`), and the installed
//!   Vulkan drivers (ICD manifests and the size and modification time of the libraries they name,
//!   plus the loader variables that select drivers). Nothing here opens a device node or loads a
//!   library.
//! - **Windows** (#28) and **macOS**: unknown for now.
//!
//! LUIDs change on reboot and are never a key.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Driver fingerprints of this machine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemFingerprint {
    /// Per GPU: stable key (`pci:<address>`, lower case) → fingerprint.
    pub gpus: BTreeMap<String, String>,
    /// Fingerprint of devices without a stable key (the CPU, software rasterizers, a CUDA device
    /// that crashed before its PCI address was known). Includes every driver source and the set of
    /// GPU keys, because those devices are matched by id, which depends on the device set.
    pub shared: String,
}

impl SystemFingerprint {
    /// The GPU keys, sorted.
    pub fn gpu_keys(&self) -> Vec<String> {
        self.gpus.keys().cloned().collect()
    }

    /// The fingerprint of one device: by its enumeration key if it has one, else by the PCI
    /// address its description reported (plus the GPU key set, since such a device is matched by
    /// id), else [`SystemFingerprint::shared`]. `None` if the device's GPU is no longer present.
    pub fn for_device(&self, key: Option<&str>, pci_bus_id: Option<&str>) -> Option<String> {
        if let Some(key) = key {
            return self.gpus.get(&key.to_ascii_lowercase()).cloned();
        }
        if let Some(pci) = pci_bus_id {
            let key = format!("pci:{}", pci.to_ascii_lowercase());
            return self
                .gpus
                .get(&key)
                .map(|gpu| format!("{gpu};gpus={}", self.gpu_keys().join(",")));
        }
        Some(self.shared.clone())
    }
}

/// The fingerprint of this machine, or `None` if it can't be determined on this OS.
pub fn current() -> Option<SystemFingerprint> {
    #[cfg(feature = "test-fault-injection")]
    if let Ok(value) = std::env::var(super::faults::FINGERPRINT_VAR) {
        tracing::warn!(fingerprint = %value, "test-only fingerprint override is active");
        return (value != "unknown").then(|| SystemFingerprint {
            gpus: BTreeMap::new(),
            shared: value,
        });
    }
    os::current()
}

#[cfg(target_os = "linux")]
mod os {
    use std::collections::BTreeMap;
    use std::fs;
    use std::io::Read;
    use std::path::{Path, PathBuf};
    use std::time::UNIX_EPOCH;

    use super::SystemFingerprint;

    /// Largest file read here (sysfs attributes, `/proc` files, ICD manifests).
    const MAX_FILE: u64 = 64 * 1024;
    /// At most this many PCI devices and ICD manifests are looked at.
    const MAX_ENTRIES: usize = 4096;

    pub(super) fn current() -> Option<SystemFingerprint> {
        let kernel = read_small("/proc/sys/kernel/osrelease")?;
        let nvidia = read_small("/proc/driver/nvidia/version")
            .and_then(|text| text.lines().next().map(str::trim).map(str::to_string))
            .unwrap_or_default();
        let vulkan = vulkan_fingerprint();

        let mut gpus = BTreeMap::new();
        let entries = fs::read_dir("/sys/bus/pci/devices").ok()?;
        for entry in entries.flatten().take(MAX_ENTRIES) {
            let path = entry.path();
            let Some(address) = entry.file_name().to_str().map(str::to_ascii_lowercase) else {
                continue;
            };
            // Display controllers: PCI class 0x03xxxx (VGA, 3D, display).
            let is_display = read_small(path.join("class"))
                .is_some_and(|class| class.to_ascii_lowercase().starts_with("0x03"));
            if !is_display {
                continue;
            }
            let driver = fs::read_link(path.join("driver"))
                .ok()
                .and_then(|link| {
                    link.file_name()
                        .and_then(|name| name.to_str())
                        .map(str::to_string)
                })
                .unwrap_or_default();
            let module = if driver.is_empty() {
                String::new()
            } else {
                read_small(format!("/sys/module/{driver}/version"))
                    .unwrap_or_else(|| kernel.clone())
            };
            let fingerprint = format!(
                "vendor={};device={};driver={driver};module={module};nvidia={};vulkan={vulkan}",
                read_small(path.join("vendor")).unwrap_or_default(),
                read_small(path.join("device")).unwrap_or_default(),
                if driver == "nvidia" {
                    nvidia.as_str()
                } else {
                    ""
                },
            );
            gpus.insert(format!("pci:{address}"), fingerprint);
        }
        let keys: Vec<&str> = gpus.keys().map(String::as_str).collect();
        let shared = format!(
            "kernel={kernel};nvidia={nvidia};vulkan={vulkan};gpus={}",
            keys.join(",")
        );
        Some(SystemFingerprint { gpus, shared })
    }

    /// A trimmed small text file, `None` if missing, unreadable, too large or empty.
    fn read_small(path: impl AsRef<Path>) -> Option<String> {
        let file = fs::File::open(path).ok()?;
        let mut text = String::new();
        file.take(MAX_FILE).read_to_string(&mut text).ok()?;
        let text = text.trim();
        (!text.is_empty()).then(|| text.to_string())
    }

    /// `name:size:mtime` of a file, or `name:missing`.
    fn stat(path: &Path) -> String {
        let name = path.display();
        match fs::metadata(path) {
            Ok(metadata) => {
                let mtime = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |since| since.as_secs());
                format!("{name}:{}:{mtime}", metadata.len())
            }
            Err(_) => format!("{name}:missing"),
        }
    }

    /// The installed Vulkan drivers: the loader variables that select drivers, every ICD manifest
    /// in the standard directories, and the library each manifest names.
    fn vulkan_fingerprint() -> String {
        let mut parts = Vec::new();
        for variable in ["VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES"] {
            if let Ok(value) = std::env::var(variable) {
                parts.push(format!("{variable}={value}"));
            }
        }
        let mut directories = vec![
            PathBuf::from("/etc/vulkan/icd.d"),
            PathBuf::from("/usr/share/vulkan/icd.d"),
            PathBuf::from("/usr/local/share/vulkan/icd.d"),
        ];
        if let Some(home) = std::env::var_os("HOME") {
            directories.push(PathBuf::from(home).join(".local/share/vulkan/icd.d"));
        }
        let mut manifests: Vec<PathBuf> = directories
            .iter()
            .filter_map(|directory| fs::read_dir(directory).ok())
            .flat_map(|entries| entries.flatten().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
            .take(MAX_ENTRIES)
            .collect();
        manifests.sort();
        for manifest in manifests {
            parts.push(stat(&manifest));
            if let Some(library) = read_small(&manifest).and_then(|text| icd_library(&text)) {
                parts.push(stat(&library));
            }
        }
        parts.join("|")
    }

    /// The library an ICD manifest names, as an absolute path where it can be resolved.
    fn icd_library(manifest: &str) -> Option<PathBuf> {
        let value: serde_json::Value = serde_json::from_str(manifest).ok()?;
        let library = value.get("ICD")?.get("library_path")?.as_str()?;
        let path = PathBuf::from(library);
        if path.is_absolute() {
            return Some(path);
        }
        let multiarch = format!("/usr/lib/{}-linux-gnu", std::env::consts::ARCH);
        [
            multiarch.as_str(),
            "/usr/lib64",
            "/usr/lib",
            "/usr/local/lib",
        ]
        .iter()
        .map(|directory| Path::new(directory).join(&path))
        .find(|candidate| candidate.exists())
        .or(Some(path))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn icd_library_paths() {
            assert_eq!(
                icd_library(r#"{"ICD":{"library_path":"/usr/lib/libvulkan_lvp.so"}}"#),
                Some(PathBuf::from("/usr/lib/libvulkan_lvp.so"))
            );
            assert!(icd_library(r#"{"ICD":{}}"#).is_none());
            assert!(icd_library("not json").is_none());
        }

        #[test]
        fn linux_fingerprint_is_known_and_stable() {
            // Every Linux machine (CI included) has /proc/sys/kernel/osrelease and sysfs.
            let first = current().expect("a Linux fingerprint");
            assert!(first.shared.contains("kernel="));
            assert_eq!(current(), Some(first));
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod os {
    use super::SystemFingerprint;

    /// Windows (#28) and macOS have no fingerprint yet: unknown, so the enumerate-only path runs.
    pub(super) fn current() -> Option<SystemFingerprint> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn machine() -> SystemFingerprint {
        let mut gpus = BTreeMap::new();
        gpus.insert("pci:0000:01:00.0".to_string(), "nvidia-615".to_string());
        gpus.insert("pci:0000:00:02.0".to_string(), "i915".to_string());
        SystemFingerprint {
            gpus,
            shared: "shared".into(),
        }
    }

    #[test]
    fn device_fingerprints() {
        let machine = machine();
        assert_eq!(
            machine.for_device(Some("pci:0000:01:00.0"), None),
            Some("nvidia-615".into())
        );
        assert_eq!(
            machine.for_device(Some("PCI:0000:01:00.0"), None),
            Some("nvidia-615".into())
        );
        // A GPU that is gone has no fingerprint.
        assert_eq!(machine.for_device(Some("pci:0000:02:00.0"), None), None);
        // Matched by id: includes the GPU set.
        assert_eq!(
            machine.for_device(None, Some("0000:01:00.0")),
            Some("nvidia-615;gpus=pci:0000:00:02.0,pci:0000:01:00.0".into())
        );
        assert_eq!(machine.for_device(None, None), Some("shared".into()));
    }
}
