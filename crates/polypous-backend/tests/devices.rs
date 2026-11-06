//! Device discovery on real hardware (issue #4). Prints every device with memory, identity and
//! probe status; see the PR's "Needs verification on the reference machine".
//!
//! ```text
//! cargo test -p polypous-backend --features cuda,vulkan --test devices -- --include-ignored --nocapture
//! ```
//!
//! This runs probes in the test process, not in a prober process (ADR-0013), so a native
//! driver crash takes the test down.
//!
//! Software rasterizers (lavapipe, …) are listed but not opened, as in discovery's default
//! (placement.md §1), unless the test-only environment variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1`
//! is set (the CI lavapipe job sets it). The library itself never reads environment variables.

use polypous_backend::device::{
    BackendKind, DeviceStatus, EnumerateOptions, UnavailableReason, describe_device,
    list_backend_devices, probe_device, select_primary_gpu, skipped_device,
};

/// Discovery options for the GPU tests: software rasterizers only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1`.
fn test_options() -> EnumerateOptions {
    EnumerateOptions {
        software_rasterizers: std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN")
            .is_ok_and(|v| v == "1"),
    }
}

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

    let options = test_options();
    println!("options: {options:?} (POLYPOUS_TEST_SOFTWARE_VULKAN=1 opts in)");

    let mut devices = Vec::new();
    for backend in BackendKind::compiled_in() {
        match list_backend_devices(backend) {
            Ok(ids) => println!("{backend}: {} device(s) {ids:?}", ids.len()),
            Err(err) => println!("{err}"),
        }
        for id in list_backend_devices(backend).unwrap_or_default() {
            if options.skips(id) {
                println!("skipped {id}: software rasterizer, not opened");
                devices.push(skipped_device(id));
                continue;
            }
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
    for info in &devices {
        let skipped = matches!(
            info.status,
            DeviceStatus::Unavailable {
                reason: UnavailableReason::SoftwareRasterizerSkipped,
                ..
            }
        );
        assert_eq!(skipped, options.skips(info.id), "{}", info.id);
    }
    if BackendKind::Cpu.is_compiled_in() {
        assert!(
            devices
                .iter()
                .any(|info| info.backend == BackendKind::Cpu && info.status.is_available()),
            "the CPU backend is always available"
        );
    }
}
