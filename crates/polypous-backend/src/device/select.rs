//! Primary GPU selection and physical-device grouping (placement.md §1, "Primary GPU selection").

use super::info::{BackendKind, DeviceInfo, DeviceKind};

/// Groups devices that are the same physical device reached through different backends (for
/// example an RTX 3060 through CUDA and Vulkan). Returns indices into `devices`, each group in
/// list order, groups ordered by their first member. Devices without a physical identity (the CPU,
/// software rasterizers) form a group of their own.
pub fn group_physical_devices(devices: &[DeviceInfo]) -> Vec<Vec<usize>> {
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (index, device) in devices.iter().enumerate() {
        let existing = groups.iter_mut().find(|group| {
            group
                .iter()
                .any(|&member| devices[member].is_same_physical_device(device))
        });
        match existing {
            Some(group) => group.push(index),
            None => groups.push(vec![index]),
        }
    }
    groups
}

/// The primary-GPU rank of a device: lower is better, `None` means "never primary".
///
/// CUDA = 1, Metal = 2, Vulkan discrete = 3, Vulkan integrated = 4. Only devices whose probe
/// passed count. Vulkan software rasterizers (lavapipe, kind `Cpu`) and virtual/unknown adapters are not
/// GPUs for this purpose: the CPU backend is faster than lavapipe.
fn rank(device: &DeviceInfo) -> Option<u8> {
    if !device.status.is_available() {
        return None;
    }
    match (device.backend, device.kind) {
        (BackendKind::Cuda, DeviceKind::Discrete | DeviceKind::Integrated) => Some(1),
        (BackendKind::Metal, DeviceKind::Discrete | DeviceKind::Integrated) => Some(2),
        (BackendKind::Vulkan, DeviceKind::Discrete) => Some(3),
        (BackendKind::Vulkan, DeviceKind::Integrated) => Some(4),
        _ => None,
    }
}

/// Chooses the one GPU v1 uses, or `None` for CPU only (placement.md §1).
///
/// Priority: CUDA → Metal → Vulkan discrete → Vulkan integrated → none. Within a rank the device
/// with the most dedicated memory wins; remaining ties go to the earlier device in the list, so
/// the result is deterministic. A physical GPU reached through several backends counts once
/// ([`group_physical_devices`]) and is represented by its best-ranked available backend, so CUDA
/// wins over Vulkan for an NVIDIA card, and Vulkan is used if only the CUDA probe failed.
pub fn select_primary_gpu(devices: &[DeviceInfo]) -> Option<&DeviceInfo> {
    // (rank, memory, index): the best candidate of each physical device, then the best overall.
    let key = |index: usize| {
        rank(&devices[index]).map(|rank| (rank, devices[index].dedicated_memory.unwrap_or(0)))
    };
    let better = |a: (u8, u64), b: (u8, u64)| a.0 < b.0 || (a.0 == b.0 && a.1 > b.1);

    let mut best: Option<(usize, (u8, u64))> = None;
    for group in group_physical_devices(devices) {
        let mut group_best: Option<(usize, (u8, u64))> = None;
        for index in group {
            if let Some(k) = key(index)
                && group_best.is_none_or(|(_, current)| better(k, current))
            {
                group_best = Some((index, k));
            }
        }
        if let Some((index, k)) = group_best
            && best.is_none_or(|(current_index, current)| {
                better(k, current) || (k == current && index < current_index)
            })
        {
            best = Some((index, k));
        }
    }
    best.map(|(index, _)| &devices[index])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::info::{
        AdapterKind, Capabilities, DeviceId, DeviceStatus, PhysicalIdentity, UnavailableReason,
    };

    const GB: u64 = 1 << 30;

    fn available() -> DeviceStatus {
        DeviceStatus::Available { isolated: true }
    }

    fn failed() -> DeviceStatus {
        DeviceStatus::Unavailable {
            reason: UnavailableReason::ProbeFailed {
                message: "test".into(),
            },
            isolated: true,
        }
    }

    fn device(
        id: DeviceId,
        name: &str,
        kind: DeviceKind,
        memory: Option<u64>,
        pci: Option<&str>,
    ) -> DeviceInfo {
        DeviceInfo {
            id,
            backend: id.backend(),
            name: name.into(),
            vendor: None,
            driver: None,
            kind,
            dedicated_memory: memory,
            free_memory: None,
            unified_memory: kind != DeviceKind::Discrete,
            identity: PhysicalIdentity {
                pci_bus_id: pci.map(Into::into),
                ..Default::default()
            },
            capabilities: Capabilities::default(),
            status: available(),
        }
    }

    fn cuda(index: usize, memory: u64, pci: &str) -> DeviceInfo {
        device(
            DeviceId::Cuda { index },
            "cuda gpu",
            DeviceKind::Discrete,
            Some(memory),
            Some(pci),
        )
    }

    fn vulkan(adapter: AdapterKind, index: usize, memory: u64, pci: Option<&str>) -> DeviceInfo {
        device(
            DeviceId::Vulkan { adapter, index },
            "vulkan gpu",
            adapter.device_kind(),
            Some(memory),
            pci,
        )
    }

    fn cpu() -> DeviceInfo {
        device(DeviceId::Cpu, "cpu", DeviceKind::Cpu, None, None)
    }

    /// The reference machine (docs/hardware-targets.md, T1): RTX 3060 Laptop via CUDA and Vulkan,
    /// Intel Iris Xe via Vulkan, CPU. Vulkan lists the RTX first (ADR-0007 follow-up notes).
    fn reference_machine() -> Vec<DeviceInfo> {
        vec![
            cuda(0, 6 * GB, "0000:01:00.0"),
            vulkan(AdapterKind::DiscreteGpu, 0, 6 * GB, Some("0000:01:00.0")),
            vulkan(AdapterKind::IntegratedGpu, 0, 7 * GB, Some("0000:00:02.0")),
            cpu(),
        ]
    }

    #[test]
    fn reference_machine_picks_rtx_via_cuda() {
        let devices = reference_machine();
        let primary = select_primary_gpu(&devices).unwrap();
        assert_eq!(primary.id, DeviceId::Cuda { index: 0 });
    }

    #[test]
    fn reference_machine_rtx_is_one_physical_device() {
        let devices = reference_machine();
        assert_eq!(
            group_physical_devices(&devices),
            vec![vec![0, 1], vec![2], vec![3]]
        );
    }

    #[test]
    fn reference_machine_order_does_not_matter() {
        let mut devices = reference_machine();
        devices.reverse();
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Cuda { index: 0 }
        );
    }

    #[test]
    fn reference_machine_cuda_failed_uses_rtx_via_vulkan() {
        let mut devices = reference_machine();
        devices[0].status = failed();
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Vulkan {
                adapter: AdapterKind::DiscreteGpu,
                index: 0
            }
        );
    }

    #[test]
    fn reference_machine_without_nvidia_uses_iris_xe() {
        // For example a build without `cuda` and the NVIDIA Vulkan probe failed.
        let mut devices = reference_machine();
        devices.remove(0);
        devices[0].status = failed();
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Vulkan {
                adapter: AdapterKind::IntegratedGpu,
                index: 0
            }
        );
    }

    #[test]
    fn iris_xe_crashed_is_ignored_and_rtx_still_wins() {
        let mut devices = reference_machine();
        devices[2].status = DeviceStatus::Unavailable {
            reason: UnavailableReason::CrashedDuringProbe,
            isolated: true,
        };
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Cuda { index: 0 }
        );
    }

    #[test]
    fn all_gpus_unavailable_means_cpu_only() {
        let mut devices = reference_machine();
        for device in &mut devices[..3] {
            device.status = failed();
        }
        assert!(select_primary_gpu(&devices).is_none());
    }

    #[test]
    fn not_probed_devices_are_not_chosen() {
        let mut devices = reference_machine();
        for device in &mut devices {
            device.status = DeviceStatus::NotProbed;
        }
        assert!(select_primary_gpu(&devices).is_none());
    }

    #[test]
    fn cpu_only_and_empty() {
        assert!(select_primary_gpu(&[]).is_none());
        assert!(select_primary_gpu(&[cpu()]).is_none());
    }

    #[test]
    fn lavapipe_is_not_a_gpu() {
        let devices = vec![vulkan(AdapterKind::Cpu, 0, 2 * GB, None), cpu()];
        assert!(select_primary_gpu(&devices).is_none());
    }

    #[test]
    fn skipped_lavapipe_changes_nothing() {
        // Debian 13 with stock Mesa: lavapipe is listed and skipped (EnumerateOptions default).
        let mut devices = reference_machine();
        devices.insert(
            3,
            crate::device::skipped_device(DeviceId::Vulkan {
                adapter: AdapterKind::Cpu,
                index: 0,
            }),
        );
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Cuda { index: 0 }
        );
        assert_eq!(
            group_physical_devices(&devices),
            vec![vec![0, 1], vec![2], vec![3], vec![4]]
        );
        // Lavapipe and the CPU alone: CPU only, whether lavapipe was skipped or probed.
        let lavapipe = devices.remove(3);
        assert!(select_primary_gpu(&[lavapipe, cpu()]).is_none());
    }

    #[test]
    fn virtual_gpu_is_not_chosen() {
        let devices = vec![vulkan(AdapterKind::VirtualGpu, 0, 8 * GB, None)];
        assert!(select_primary_gpu(&devices).is_none());
    }

    #[test]
    fn metal_beats_vulkan() {
        let metal = device(
            DeviceId::Metal {
                adapter: AdapterKind::IntegratedGpu,
                index: 0,
            },
            "apple m2",
            DeviceKind::Integrated,
            Some(10 * GB),
            None,
        );
        let devices = vec![
            vulkan(AdapterKind::DiscreteGpu, 0, 16 * GB, Some("0000:03:00.0")),
            metal,
        ];
        assert_eq!(
            select_primary_gpu(&devices).unwrap().backend,
            BackendKind::Metal
        );
    }

    #[test]
    fn vulkan_discrete_beats_bigger_integrated() {
        let devices = vec![
            vulkan(AdapterKind::IntegratedGpu, 0, 16 * GB, Some("0000:00:02.0")),
            vulkan(AdapterKind::DiscreteGpu, 0, 4 * GB, Some("0000:01:00.0")),
        ];
        assert_eq!(
            select_primary_gpu(&devices).unwrap().kind,
            DeviceKind::Discrete
        );
    }

    #[test]
    fn same_rank_most_memory_wins() {
        let devices = vec![
            cuda(0, 8 * GB, "0000:01:00.0"),
            cuda(1, 24 * GB, "0000:02:00.0"),
            cuda(2, 24 * GB, "0000:03:00.0"),
        ];
        // 24 GB wins; the tie between 1 and 2 goes to the earlier device.
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Cuda { index: 1 }
        );
    }

    #[test]
    fn unknown_memory_counts_as_zero() {
        let mut small = vulkan(AdapterKind::DiscreteGpu, 0, 0, Some("0000:01:00.0"));
        small.dedicated_memory = None;
        let devices = vec![
            small,
            vulkan(AdapterKind::DiscreteGpu, 1, 2 * GB, Some("0000:02:00.0")),
        ];
        assert_eq!(
            select_primary_gpu(&devices).unwrap().id,
            DeviceId::Vulkan {
                adapter: AdapterKind::DiscreteGpu,
                index: 1
            }
        );
    }

    #[test]
    fn devices_without_identity_are_separate() {
        let devices = vec![
            vulkan(AdapterKind::DiscreteGpu, 0, 4 * GB, None),
            vulkan(AdapterKind::DiscreteGpu, 1, 4 * GB, None),
        ];
        assert_eq!(group_physical_devices(&devices), vec![vec![0], vec![1]]);
    }
}
