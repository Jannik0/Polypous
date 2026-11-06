//! Why a compiled-in GPU backend has no device (placement.md §1, "A compiled-in GPU backend
//! without devices"): a `cuda` build on a machine without usable CUDA explains itself in the probe
//! report instead of silently listing no CUDA device.
//!
//! Nothing here loads a library or opens a device: driver libraries are looked for as files.

use std::path::{Path, PathBuf};

use crate::device::{BackendKind, DeviceInfo};

/// One warning per compiled-in GPU backend that has no device in `devices`, except the backends in
/// `failed` (their enumeration failed, which has its own warning). Only meaningful for a report of
/// all devices.
pub(crate) fn notes(devices: &[DeviceInfo], failed: &[BackendKind]) -> Vec<String> {
    notes_with(
        devices,
        failed,
        &BackendKind::compiled_in(),
        &|name| std::env::var(name).ok(),
        &|path| path.is_file(),
    )
}

/// [`notes`] with the compiled-in backends, the environment and the file system injected (tests).
fn notes_with(
    devices: &[DeviceInfo],
    failed: &[BackendKind],
    compiled_in: &[BackendKind],
    variable: &dyn Fn(&str) -> Option<String>,
    exists: &dyn Fn(&Path) -> bool,
) -> Vec<String> {
    compiled_in
        .iter()
        .filter(|&&backend| backend != BackendKind::Cpu)
        .filter(|backend| !failed.contains(backend))
        .filter(|&&backend| !devices.iter().any(|device| device.backend == backend))
        .map(|&backend| {
            format!(
                "{backend}: compiled in, but no device found: {}",
                reason(backend, variable, exists)
            )
        })
        .collect()
}

/// The first reason that applies: a selecting variable is set, the driver library is missing, or
/// no device was reported.
fn reason(
    backend: BackendKind,
    variable: &dyn Fn(&str) -> Option<String>,
    exists: &dyn Fn(&Path) -> bool,
) -> String {
    let (variables, library, nothing): (&[&str], Option<&str>, &str) = match backend {
        BackendKind::Cuda => (
            &["CUDA_VISIBLE_DEVICES"],
            cuda_library(),
            "no CUDA device was reported",
        ),
        BackendKind::Vulkan => (
            &["VK_DRIVER_FILES", "VK_ICD_FILENAMES"],
            vulkan_library(),
            "no Vulkan device was reported",
        ),
        BackendKind::Metal => (&[], None, "no Metal device was reported"),
        BackendKind::Cpu => (&[], None, "no CPU device was reported"),
    };
    if let Some((name, value)) = variables
        .iter()
        .find_map(|name| variable(name).map(|value| (name, value)))
    {
        return format!("{name} is set to {value:?}");
    }
    if let Some(library) = library {
        let directories = library_directories(variable);
        if !directories.is_empty()
            && !directories
                .iter()
                .any(|directory| exists(&directory.join(library)))
        {
            return format!("the driver library {library} is not installed");
        }
    }
    nothing.to_string()
}

/// The CUDA driver library, if this OS has one Polypous looks for.
fn cuda_library() -> Option<&'static str> {
    if cfg!(windows) {
        Some("nvcuda.dll")
    } else if cfg!(target_os = "linux") {
        Some("libcuda.so.1")
    } else {
        None
    }
}

/// The Vulkan loader library, if this OS has one Polypous looks for.
fn vulkan_library() -> Option<&'static str> {
    if cfg!(windows) {
        Some("vulkan-1.dll")
    } else if cfg!(target_os = "linux") {
        Some("libvulkan.so.1")
    } else {
        None
    }
}

/// Where the driver libraries are looked for. Empty if that can't be determined (then no
/// "not installed" reason is given).
fn library_directories(variable: &dyn Fn(&str) -> Option<String>) -> Vec<PathBuf> {
    if cfg!(windows) {
        return variable("SystemRoot")
            .map(|root| vec![PathBuf::from(root).join("System32")])
            .unwrap_or_default();
    }
    let arch = format!("{}-linux-gnu", std::env::consts::ARCH);
    let mut directories: Vec<PathBuf> = variable("LD_LIBRARY_PATH")
        .map(|paths| {
            paths
                .split(':')
                .filter(|path| !path.is_empty())
                .map(PathBuf::from)
                .collect()
        })
        .unwrap_or_default();
    directories.extend(
        [
            format!("/usr/lib/{arch}"),
            format!("/lib/{arch}"),
            "/usr/lib64".to_string(),
            "/usr/lib".to_string(),
            "/usr/local/lib".to_string(),
            "/usr/lib/wsl/lib".to_string(),
        ]
        .into_iter()
        .map(PathBuf::from),
    );
    directories
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceId, unprobed_device};

    const GPU_BACKENDS: [BackendKind; 3] =
        [BackendKind::Cuda, BackendKind::Vulkan, BackendKind::Cpu];

    fn vulkan_gpu() -> DeviceInfo {
        unprobed_device(DeviceId::Vulkan {
            adapter: crate::device::AdapterKind::DiscreteGpu,
            index: 0,
        })
    }

    fn system_root(name: &str) -> Option<String> {
        (name == "SystemRoot").then(|| r"C:\Windows".to_string())
    }

    #[test]
    fn backends_with_devices_or_failed_enumeration_have_no_note() {
        let devices = [
            unprobed_device(DeviceId::Cuda { index: 0 }),
            vulkan_gpu(),
            unprobed_device(DeviceId::Cpu),
        ];
        assert!(notes_with(&devices, &[], &GPU_BACKENDS, &|_| None, &|_| true).is_empty());
        // Vulkan's enumeration failed: that has its own warning.
        let devices = [unprobed_device(DeviceId::Cuda { index: 0 })];
        assert!(
            notes_with(
                &devices,
                &[BackendKind::Vulkan],
                &GPU_BACKENDS,
                &|_| None,
                &|_| true
            )
            .is_empty()
        );
        // Backends that are not compiled in are never mentioned.
        assert!(notes_with(&[], &[], &[BackendKind::Cpu], &|_| None, &|_| false).is_empty());
    }

    #[test]
    fn a_selecting_variable_comes_first() {
        let devices = [vulkan_gpu()];
        let variable = |name: &str| match name {
            "CUDA_VISIBLE_DEVICES" => Some("-1".to_string()),
            other => system_root(other),
        };
        assert_eq!(
            notes_with(&devices, &[], &GPU_BACKENDS, &variable, &|_| false),
            vec![r#"cuda: compiled in, but no device found: CUDA_VISIBLE_DEVICES is set to "-1""#]
        );
        let variable = |name: &str| (name == "VK_DRIVER_FILES").then(|| "/nowhere.json".into());
        assert_eq!(
            notes_with(&[], &[], &[BackendKind::Vulkan], &variable, &|_| true),
            vec![
                r#"vulkan: compiled in, but no device found: VK_DRIVER_FILES is set to "/nowhere.json""#
            ]
        );
    }

    #[test]
    fn a_missing_driver_library_is_named() {
        let devices = [vulkan_gpu()];
        let library = cuda_library();
        let notes = notes_with(&devices, &[], &GPU_BACKENDS, &system_root, &|_| false);
        match library {
            Some(library) => assert_eq!(
                notes,
                vec![format!(
                    "cuda: compiled in, but no device found: the driver library {library} is not \
                     installed"
                )]
            ),
            None => assert_eq!(
                notes,
                vec!["cuda: compiled in, but no device found: no CUDA device was reported"]
            ),
        }
    }

    #[test]
    fn an_installed_library_without_devices_says_so() {
        let devices = [vulkan_gpu()];
        assert_eq!(
            notes_with(&devices, &[], &GPU_BACKENDS, &system_root, &|_| true),
            vec!["cuda: compiled in, but no device found: no CUDA device was reported"]
        );
    }

    #[test]
    fn library_directories_per_os() {
        let directories = library_directories(&|name| match name {
            "SystemRoot" => Some(r"C:\Windows".into()),
            "LD_LIBRARY_PATH" => Some("/opt/cuda/lib64::/opt/other".into()),
            _ => None,
        });
        if cfg!(windows) {
            assert_eq!(directories, vec![PathBuf::from(r"C:\Windows\System32")]);
        } else {
            assert_eq!(directories[0], PathBuf::from("/opt/cuda/lib64"));
            assert_eq!(directories[1], PathBuf::from("/opt/other"));
            assert!(directories.contains(&PathBuf::from("/usr/lib/wsl/lib")));
        }
    }
}
