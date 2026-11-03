//! Device discovery on real hardware (issue #4). Prints every device with memory, identity and
//! probe status; see the PR's "Needs verification on the reference machine".
//!
//! ```text
//! cargo test -p polypous-backend --features cuda,vulkan --test devices -- --include-ignored --nocapture
//! ```
//!
//! This runs probes in the test process, not in a prober process (that is #14), so a native
//! driver crash takes the test down.

use polypous_backend::device::{
    BackendKind, DeviceStatus, describe_device, list_backend_devices, probe_device,
    select_primary_gpu,
};

fn gib(bytes: Option<u64>) -> String {
    match bytes {
        Some(bytes) => format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64),
        None => "-".to_string(),
    }
}

#[test]
#[ignore = "gpu"]
fn print_discovered_devices() {
    println!(
        "compiled-in backends: {:?}, os={}, arch={}",
        BackendKind::compiled_in(),
        std::env::consts::OS,
        std::env::consts::ARCH
    );

    let mut devices = Vec::new();
    for backend in BackendKind::compiled_in() {
        match list_backend_devices(backend) {
            Ok(ids) => println!("{backend}: {} device(s) {ids:?}", ids.len()),
            Err(err) => println!("{err}"),
        }
        for id in list_backend_devices(backend).unwrap_or_default() {
            let mut info = describe_device(id);
            if info.status == DeviceStatus::NotProbed {
                info.status = probe_device(id);
            }
            devices.push(info);
        }
    }

    for info in &devices {
        println!();
        println!("[{}] {}", info.id, info.name);
        println!(
            "  kind={:?} vendor={} driver={}",
            info.kind,
            info.vendor.as_deref().unwrap_or("-"),
            info.driver.as_deref().unwrap_or("-")
        );
        println!(
            "  dedicated_memory={} free_memory={} unified_memory={}",
            gib(info.dedicated_memory),
            gib(info.free_memory),
            info.unified_memory
        );
        println!(
            "  pci_bus_id={} luid={}",
            info.identity.pci_bus_id.as_deref().unwrap_or("-"),
            info.identity.luid.as_deref().unwrap_or("-")
        );
        println!("  capabilities={:?}", info.capabilities);
        println!("  status: {}", info.status);
        println!(
            "  json: {}",
            serde_json::to_string(info).expect("DeviceInfo serializes")
        );
    }

    println!();
    match select_primary_gpu(&devices) {
        Some(primary) => println!("primary GPU: [{}] {}", primary.id, primary.name),
        None => println!("primary GPU: none (CPU only)"),
    }

    assert!(
        devices
            .iter()
            .all(|info| info.status != DeviceStatus::NotProbed),
        "every device has a probe result"
    );
    if BackendKind::Cpu.is_compiled_in() {
        assert!(
            devices
                .iter()
                .any(|info| info.backend == BackendKind::Cpu && info.status.is_available()),
            "the CPU backend is always available"
        );
    }
}
