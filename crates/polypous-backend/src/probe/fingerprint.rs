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
//! - **Windows** ([`registry`]): GPUs are the present display adapters among the PCI device
//!   instances in the registry, keyed by the PCI address in their `LocationInformation`. A GPU's
//!   fingerprint has its instance's hardware id and its driver key's version, date, provider and
//!   matching device id; the shared one adds the Windows build and the Vulkan loader. Nothing here
//!   loads a GPU driver.
//! - **macOS**: unknown for now.
//!
//! LUIDs change on reboot and are never a key. Adapters whose graphics API reports no PCI address
//! (Intel iGPUs through Vulkan on Windows) get their key by PCI vendor/device id
//! ([`SystemFingerprint::key_for_pci_ids`]).

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
    /// Per GPU: stable key → PCI vendor and device id as `vvvv:dddd` (lower-case hex), where known.
    #[serde(default)]
    pub pci_ids: BTreeMap<String, String>,
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

    /// The key of the one GPU with this PCI vendor and device id, for adapters whose graphics API
    /// reports no PCI address. `None` if no GPU or several have these ids.
    pub fn key_for_pci_ids(&self, vendor: u32, device: u32) -> Option<String> {
        let wanted = pci_ids_text(vendor, device);
        let mut matching = self.pci_ids.iter().filter(|(_, ids)| **ids == wanted);
        let (key, _) = matching.next()?;
        matching.next().is_none().then(|| key.clone())
    }
}

/// `vvvv:dddd`, the form of [`SystemFingerprint::pci_ids`].
fn pci_ids_text(vendor: u32, device: u32) -> String {
    format!("{vendor:04x}:{device:04x}")
}

/// The fingerprint of this machine, or `None` if it can't be determined on this OS.
pub fn current() -> Option<SystemFingerprint> {
    #[cfg(feature = "test-fault-injection")]
    if let Ok(value) = std::env::var(super::faults::FINGERPRINT_VAR) {
        tracing::warn!(fingerprint = %value, "test-only fingerprint override is active");
        return (value != "unknown").then(|| SystemFingerprint {
            gpus: BTreeMap::new(),
            shared: value,
            pci_ids: BTreeMap::new(),
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
        let mut pci_ids = BTreeMap::new();
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
            let vendor = read_small(path.join("vendor")).unwrap_or_default();
            let device = read_small(path.join("device")).unwrap_or_default();
            if let (Some(vendor), Some(device)) = (hex_id(&vendor), hex_id(&device)) {
                pci_ids.insert(
                    format!("pci:{address}"),
                    super::pci_ids_text(vendor, device),
                );
            }
            let fingerprint = format!(
                "vendor={vendor};device={device};driver={driver};module={module};nvidia={};vulkan={vulkan}",
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
        Some(SystemFingerprint {
            gpus,
            shared,
            pci_ids,
        })
    }

    /// A sysfs id such as `0x10de`.
    fn hex_id(text: &str) -> Option<u32> {
        u32::from_str_radix(text.trim().strip_prefix("0x")?, 16).ok()
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

/// The Windows fingerprint from registry data (placement.md §1, "Driver fingerprint"). Pure
/// functions over recorded values, so they are tested on every OS; [`os::current`] reads the values
/// on Windows.
#[cfg_attr(
    not(any(windows, test)),
    expect(dead_code, reason = "the registry is only read on Windows")
)]
pub(crate) mod registry {
    use std::collections::BTreeMap;

    use super::SystemFingerprint;

    /// The device setup class of display adapters.
    pub(crate) const DISPLAY_CLASS: &str = "{4d36e968-e325-11ce-bfc1-08002be10318}";

    /// One present display adapter as the registry describes it.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub(crate) struct Adapter {
        /// The hardware part of the device instance id below `Enum\PCI`, for example
        /// `VEN_8086&DEV_46A6&SUBSYS_201B1A58&REV_0C`.
        pub(crate) hardware_id: String,
        /// `LocationInformation` of the device instance.
        pub(crate) location: Option<String>,
        /// Values of the driver key (`Control\Class\<Driver>`); empty if missing.
        pub(crate) driver_version: String,
        pub(crate) driver_date: String,
        pub(crate) provider: String,
        pub(crate) matching_device_id: String,
    }

    /// Machine-wide values of the shared fingerprint.
    #[derive(Debug, Clone, Default, PartialEq, Eq)]
    pub(crate) struct Machine {
        /// `CurrentBuild.UBR`, for example `26200.7019`.
        pub(crate) build: String,
        /// Size and modification time of `vulkan-1.dll`, or `missing`.
        pub(crate) vulkan_loader: String,
        /// The Vulkan loader variables that are set, as `NAME=value`.
        pub(crate) loader_variables: Vec<String>,
    }

    /// The PCI address `0000:bb:dd.f` from a `LocationInformation` value: either the indirect form
    /// `@System32\drivers\pci.sys,#65536;PCI bus %1, device %2, function %3;(1,0,0)` or the plain
    /// `PCI bus 1, device 0, function 0`. Windows doesn't store the PCI segment; 0 is assumed.
    pub(crate) fn pci_address(location: &str) -> Option<String> {
        let numbers: Vec<u32> = match location
            .trim()
            .strip_suffix(')')
            .and_then(|rest| rest.rsplit_once('('))
        {
            Some((_, triple)) => triple
                .split(',')
                .map(|part| part.trim().parse().ok())
                .collect::<Option<_>>()?,
            None => ["bus", "device", "function"]
                .iter()
                .map(|label| number_after(location, label))
                .collect::<Option<_>>()?,
        };
        let [bus, device, function] = numbers[..] else {
            return None;
        };
        (bus <= 0xff && device <= 0x1f && function <= 7)
            .then(|| format!("0000:{bus:02x}:{device:02x}.{function:x}"))
    }

    /// The decimal number after `<label> ` in `text`.
    fn number_after(text: &str, label: &str) -> Option<u32> {
        let lower = text.to_ascii_lowercase();
        let start = lower.find(&format!("{label} "))? + label.len() + 1;
        let digits: String = lower[start..]
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        digits.parse().ok()
    }

    /// PCI vendor and device id from a hardware id such as `VEN_8086&DEV_46A6&…`.
    pub(crate) fn pci_ids(hardware_id: &str) -> Option<(u32, u32)> {
        let field = |prefix: &str| {
            hardware_id.split('&').find_map(|part| {
                let value = part
                    .get(..prefix.len())?
                    .eq_ignore_ascii_case(prefix)
                    .then(|| &part[prefix.len()..])?;
                u32::from_str_radix(value, 16).ok()
            })
        };
        Some((field("VEN_")?, field("DEV_")?))
    }

    /// The fingerprint of a machine with these adapters. Adapters without a readable PCI address
    /// have no key but still count in the shared fingerprint.
    pub(crate) fn fingerprint(adapters: &[Adapter], machine: &Machine) -> SystemFingerprint {
        let mut gpus = BTreeMap::new();
        let mut ids_by_key = BTreeMap::new();
        let mut drivers = Vec::new();
        for adapter in adapters {
            let fingerprint = format!(
                "hardware={};version={};date={};provider={};matching={}",
                adapter.hardware_id.to_ascii_lowercase(),
                adapter.driver_version.trim(),
                adapter.driver_date.trim(),
                adapter.provider.trim(),
                adapter.matching_device_id.trim().to_ascii_lowercase()
            );
            drivers.push(fingerprint.clone());
            let Some(address) = adapter.location.as_deref().and_then(pci_address) else {
                continue;
            };
            let key = format!("pci:{address}");
            if let Some((vendor, device)) = pci_ids(&adapter.hardware_id) {
                ids_by_key.insert(key.clone(), super::pci_ids_text(vendor, device));
            }
            gpus.insert(key, fingerprint);
        }
        drivers.sort();
        let keys: Vec<&str> = gpus.keys().map(String::as_str).collect();
        let shared = format!(
            "build={};vulkan_loader={};variables={};gpus={};drivers={}",
            machine.build,
            machine.vulkan_loader,
            machine.loader_variables.join(","),
            keys.join(","),
            drivers.join("|")
        );
        SystemFingerprint {
            gpus,
            shared,
            pci_ids: ids_by_key,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The reference machine's registry (Windows 11, NVIDIA 617.42, Intel 32.0.101.7092).
        fn reference_adapters() -> Vec<Adapter> {
            vec![
                Adapter {
                    hardware_id: "VEN_10DE&DEV_2560&SUBSYS_201B1A58&REV_A1".into(),
                    location: Some(
                        r"@System32\drivers\pci.sys,#65536;PCI bus %1, device %2, function %3;(1,0,0)"
                            .into(),
                    ),
                    driver_version: "32.0.16.1742".into(),
                    driver_date: "9-30-2026".into(),
                    provider: "NVIDIA".into(),
                    matching_device_id: r"pci\ven_10de&dev_2560&subsys_201b1a58".into(),
                },
                Adapter {
                    hardware_id: "VEN_8086&DEV_46A6&SUBSYS_201B1A58&REV_0C".into(),
                    location: Some(
                        r"@System32\drivers\pci.sys,#65536;PCI bus %1, device %2, function %3;(0,2,0)"
                            .into(),
                    ),
                    driver_version: "32.0.101.7092".into(),
                    driver_date: "9-3-2026".into(),
                    provider: "Intel Corporation".into(),
                    matching_device_id: r"PCI\VEN_8086&DEV_46A6".into(),
                },
            ]
        }

        fn machine() -> Machine {
            Machine {
                build: "26200.7019".into(),
                vulkan_loader: "1234567:1790000000".into(),
                loader_variables: Vec::new(),
            }
        }

        #[test]
        fn pci_addresses_from_location_information() {
            let indirect =
                r"@System32\drivers\pci.sys,#65536;PCI bus %1, device %2, function %3;(1,0,0)";
            assert_eq!(pci_address(indirect).as_deref(), Some("0000:01:00.0"));
            assert_eq!(
                pci_address("PCI bus 0, device 2, function 0").as_deref(),
                Some("0000:00:02.0")
            );
            assert_eq!(
                pci_address("PCI bus 200, device 31, function 7").as_deref(),
                Some("0000:c8:1f.7")
            );
            for bad in [
                "",
                "PCI bus 1, device 0",
                "(1,0)",
                "(1,0,0,0)",
                "(256,0,0)",
                "(0,32,0)",
                "(0,0,8)",
                "(a,b,c)",
                "Internal",
            ] {
                assert_eq!(pci_address(bad), None, "{bad}");
            }
        }

        #[test]
        fn pci_ids_from_hardware_ids() {
            assert_eq!(
                pci_ids("VEN_10DE&DEV_2560&SUBSYS_201B1A58&REV_A1"),
                Some((0x10de, 0x2560))
            );
            assert_eq!(pci_ids("ven_8086&dev_46a6"), Some((0x8086, 0x46a6)));
            assert_eq!(pci_ids("VEN_8086"), None);
            assert_eq!(pci_ids("VEN_XYZ&DEV_46A6"), None);
            assert_eq!(pci_ids(""), None);
        }

        #[test]
        fn reference_machine_fingerprint() {
            let fingerprint = fingerprint(&reference_adapters(), &machine());
            assert_eq!(
                fingerprint.gpu_keys(),
                vec!["pci:0000:00:02.0", "pci:0000:01:00.0"]
            );
            assert_eq!(
                fingerprint.gpus["pci:0000:01:00.0"],
                "hardware=ven_10de&dev_2560&subsys_201b1a58&rev_a1;version=32.0.16.1742;\
                 date=9-30-2026;provider=NVIDIA;matching=pci\\ven_10de&dev_2560&subsys_201b1a58"
            );
            // The Iris Xe has no PCI address through Vulkan on Windows: found by its PCI ids.
            assert_eq!(
                fingerprint.key_for_pci_ids(0x8086, 0x46a6).as_deref(),
                Some("pci:0000:00:02.0")
            );
            assert_eq!(
                fingerprint.key_for_pci_ids(0x10de, 0x2560).as_deref(),
                Some("pci:0000:01:00.0")
            );
            assert_eq!(fingerprint.key_for_pci_ids(0x1002, 0x73df), None);
            // CUDA devices are matched by the PCI address their description reports.
            assert!(
                fingerprint
                    .for_device(None, Some("0000:01:00.0"))
                    .is_some_and(|text| text.contains("32.0.16.1742"))
            );
            assert!(fingerprint.shared.contains("build=26200.7019"));
            assert!(fingerprint.shared.contains("32.0.101.7092"));
        }

        #[test]
        fn a_driver_update_changes_only_that_gpu() {
            let before = fingerprint(&reference_adapters(), &machine());
            let mut adapters = reference_adapters();
            adapters[1].driver_version = "32.0.101.7200".into();
            let after = fingerprint(&adapters, &machine());
            assert_eq!(
                before.gpus["pci:0000:01:00.0"],
                after.gpus["pci:0000:01:00.0"]
            );
            assert_ne!(
                before.gpus["pci:0000:00:02.0"],
                after.gpus["pci:0000:00:02.0"]
            );
            // Devices without a key (the CPU) see every driver change.
            assert_ne!(before.shared, after.shared);
            // The same registry data gives the same fingerprint (stable across reboots: no LUID).
            assert_eq!(before, fingerprint(&reference_adapters(), &machine()));
        }

        #[test]
        fn ambiguous_or_unlocated_adapters_have_no_key() {
            let mut adapters = reference_adapters();
            // A second identical NVIDIA card: its PCI ids no longer identify one GPU.
            let mut twin = adapters[0].clone();
            twin.location = Some("PCI bus 2, device 0, function 0".into());
            adapters.push(twin);
            // An adapter without a readable location: no key, but its driver counts.
            adapters.push(Adapter {
                hardware_id: "VEN_1002&DEV_73DF".into(),
                location: None,
                driver_version: "31.0.1".into(),
                ..Adapter::default()
            });
            let fingerprint = fingerprint(&adapters, &machine());
            assert_eq!(fingerprint.gpus.len(), 3);
            assert_eq!(fingerprint.key_for_pci_ids(0x10de, 0x2560), None);
            assert_eq!(fingerprint.key_for_pci_ids(0x1002, 0x73df), None);
            assert!(fingerprint.shared.contains("version=31.0.1"));
        }
    }
}

#[cfg(windows)]
mod os {
    use std::path::PathBuf;
    use std::time::UNIX_EPOCH;

    use windows_registry::{Key, LOCAL_MACHINE};

    use super::SystemFingerprint;
    use super::registry::{Adapter, DISPLAY_CLASS, Machine, fingerprint};

    /// At most this many PCI hardware ids, and instances per id, are looked at.
    const MAX_ENTRIES: usize = 4096;

    /// The Windows fingerprint, or unknown if the PCI device list can't be read.
    pub(super) fn current() -> Option<SystemFingerprint> {
        let adapters = display_adapters()?;
        let machine = Machine {
            build: windows_build(),
            vulkan_loader: vulkan_loader(),
            loader_variables: ["VK_DRIVER_FILES", "VK_ICD_FILENAMES", "VK_ADD_DRIVER_FILES"]
                .iter()
                .filter_map(|name| {
                    std::env::var(name)
                        .ok()
                        .map(|value| format!("{name}={value}"))
                })
                .collect(),
        };
        Some(fingerprint(&adapters, &machine))
    }

    /// The present display adapters among the PCI device instances.
    fn display_adapters() -> Option<Vec<Adapter>> {
        let pci = LOCAL_MACHINE
            .open(r"SYSTEM\CurrentControlSet\Enum\PCI")
            .ok()?;
        let mut adapters = Vec::new();
        for hardware_id in pci.keys().ok()?.take(MAX_ENTRIES) {
            let Ok(hardware) = pci.open(&hardware_id) else {
                continue;
            };
            let Ok(instances) = hardware.keys() else {
                continue;
            };
            for instance_id in instances.take(MAX_ENTRIES) {
                let Ok(instance) = hardware.open(&instance_id) else {
                    continue;
                };
                let class = instance.get_string("ClassGUID").unwrap_or_default();
                // Present: Windows creates the volatile `Control` subkey while the device is
                // started; instances of removed devices stay in the registry without it.
                if !class.eq_ignore_ascii_case(DISPLAY_CLASS) || instance.open("Control").is_err() {
                    continue;
                }
                let driver = instance.get_string("Driver").ok().and_then(|driver| {
                    LOCAL_MACHINE
                        .open(format!(r"SYSTEM\CurrentControlSet\Control\Class\{driver}"))
                        .ok()
                });
                let value = |name: &str| string(driver.as_ref(), name);
                adapters.push(Adapter {
                    hardware_id: hardware_id.clone(),
                    location: instance.get_string("LocationInformation").ok(),
                    driver_version: value("DriverVersion"),
                    driver_date: value("DriverDate"),
                    provider: value("ProviderName"),
                    matching_device_id: value("MatchingDeviceId"),
                });
            }
        }
        Some(adapters)
    }

    fn string(key: Option<&Key>, name: &str) -> String {
        key.and_then(|key| key.get_string(name).ok())
            .unwrap_or_default()
    }

    /// `CurrentBuild.UBR`, for example `26200.7019`.
    fn windows_build() -> String {
        let key = LOCAL_MACHINE
            .open(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion")
            .ok();
        let build = string(key.as_ref(), "CurrentBuild");
        let revision = key
            .and_then(|key| key.get_u32("UBR").ok())
            .map_or_else(String::new, |ubr| ubr.to_string());
        format!("{build}.{revision}")
    }

    /// Size and modification time of the Vulkan loader, or `missing`.
    fn vulkan_loader() -> String {
        let Some(root) = std::env::var_os("SystemRoot") else {
            return "missing".into();
        };
        let path = PathBuf::from(root).join("System32").join("vulkan-1.dll");
        match std::fs::metadata(&path) {
            Ok(metadata) => {
                let mtime = metadata
                    .modified()
                    .ok()
                    .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
                    .map_or(0, |since| since.as_secs());
                format!("{}:{mtime}", metadata.len())
            }
            Err(_) => "missing".into(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// Every Windows machine with a display adapter (CI runners included) has one in the
        /// registry, and reading twice gives the same result.
        #[test]
        fn windows_fingerprint_is_known_and_stable() {
            let first = current().expect("a Windows fingerprint");
            assert!(first.shared.contains("build="), "{}", first.shared);
            assert_eq!(current(), Some(first));
        }
    }
}

#[cfg(not(any(target_os = "linux", windows)))]
mod os {
    use super::SystemFingerprint;

    /// macOS has no fingerprint yet: unknown, so the enumerate-only path runs.
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
            pci_ids: BTreeMap::new(),
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
