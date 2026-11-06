//! `polyp devices` end to end: runs the built binary (issue #6).
//!
//! These run in CI with the default (`cpu`) features, and in the lavapipe job with `vulkan`
//! compiled in. `polyp` probes devices in a prober process (itself, re-launched; #14), so results
//! are isolated. Each run uses a fresh probe cache directory.
//!
//! `polyp devices` always uses the default config, so software rasterizers (lavapipe, …) are
//! listed as skipped. The GPU test of the engine itself opts in to them only with the test-only
//! environment variable `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (the CI lavapipe job sets it); the
//! library never reads environment variables.

use std::process::Command;

use serde_json::Value;

/// A fresh probe cache directory for one test.
fn cache_dir(name: &str) -> std::path::PathBuf {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("devices-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

// Test helper: failing to start the binary under test should fail the test.
#[allow(clippy::expect_used)]
fn polyp(args: &[&str]) -> std::process::Output {
    let name = args.join("-").replace("--", "");
    Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(args)
        .arg("--probe-cache-dir")
        .arg(cache_dir(&name))
        .env_remove("RUST_LOG")
        .env_remove("POLYPOUS_TEST_PROBER_FAULTS")
        .env_remove("POLYPOUS_TEST_FINGERPRINT")
        .output()
        .expect("polyp runs")
}

#[test]
fn devices_json_lists_the_cpu() {
    let output = polyp(&["devices", "--json"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let devices: Value = serde_json::from_slice(&output.stdout).expect("stdout is JSON");
    let devices = devices.as_array().expect("a JSON array");

    let cpu = devices
        .iter()
        .find(|device| device["id"] == serde_json::json!({"backend": "cpu"}))
        .expect("the CPU device is listed");
    assert_eq!(cpu["backend"], "cpu");
    assert_eq!(cpu["kind"], "cpu");
    assert_eq!(cpu["unified_memory"], true);
    assert_eq!(cpu["dedicated_memory"], Value::Null);
    // Probed in the prober process.
    assert_eq!(
        cpu["status"],
        serde_json::json!({"state": "available", "isolated": true})
    );
    assert_eq!(cpu["primary"], false);
    assert!(!cpu["name"].as_str().unwrap_or_default().is_empty());

    // Every device has the documented fields, a probe result, and at most one is primary.
    let fields = [
        "id",
        "backend",
        "name",
        "vendor",
        "driver",
        "kind",
        "dedicated_memory",
        "free_memory",
        "unified_memory",
        "identity",
        "capabilities",
        "status",
        "physical_gpu",
        "primary",
    ];
    for device in devices {
        let object = device.as_object().expect("each device is an object");
        let mut keys: Vec<&str> = object.keys().map(String::as_str).collect();
        let mut expected = fields.to_vec();
        keys.sort_unstable();
        expected.sort_unstable();
        assert_eq!(keys, expected);
        assert_ne!(device["status"]["state"], "not_probed");
    }
    let primaries: Vec<&Value> = devices
        .iter()
        .filter(|device| device["primary"] == true)
        .collect();
    assert!(primaries.len() <= 1);
    for primary in primaries {
        assert_eq!(primary["status"]["state"], "available");
        assert!(primary["kind"] == "discrete" || primary["kind"] == "integrated");
    }
}

#[test]
fn devices_table_lists_the_cpu() {
    let output = polyp(&["devices"]);
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let table = String::from_utf8(output.stdout).expect("UTF-8 output");
    let mut lines = table.lines();
    let header = lines.next().expect("a header line");
    for column in [
        "GPU",
        "BACKEND",
        "NAME",
        "KIND",
        "DEDICATED",
        "UNIFIED",
        "STATUS",
    ] {
        assert!(header.contains(column), "{header}");
    }
    assert!(
        table.lines().any(|line| {
            // The CPU row: no primary marker, no physical GPU (`-`), backend `cpu`.
            let cells: Vec<&str> = line.split_whitespace().collect();
            cells.starts_with(&["-", "cpu"]) && line.contains("available")
        }),
        "{table}"
    );
    assert!(
        table.contains("* primary GPU: ") || table.contains("No usable GPU"),
        "{table}"
    );
}

#[test]
fn unknown_arguments_fail_with_usage() {
    let output = polyp(&["devices", "--yaml"]);
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--yaml"), "{stderr}");
    assert!(stderr.contains("Usage:"), "{stderr}");
}

/// On a machine with a GPU (the reference machine, with `--features cuda,vulkan`), the table
/// marks exactly one primary GPU. Run with `--include-ignored`; see the PR for expected output.
#[test]
#[ignore = "gpu"]
fn devices_marks_a_primary_gpu_when_one_is_usable() {
    let output = polyp(&["devices", "--json"]);
    assert!(output.status.success());
    let devices: Vec<Value> = serde_json::from_slice(&output.stdout).expect("JSON array");
    let usable_gpu = devices.iter().any(|device| {
        device["status"]["state"] == "available"
            && (device["kind"] == "discrete" || device["kind"] == "integrated")
    });
    let primaries = devices
        .iter()
        .filter(|device| device["primary"] == true)
        .count();
    // A usable GPU means exactly one primary; none (CPU only, or lavapipe only) means zero.
    assert_eq!(primaries, usize::from(usable_gpu));
    // The CLI keeps the default: software rasterizers are listed, not probed.
    for device in devices
        .iter()
        .filter(|device| is_software_rasterizer(device))
    {
        assert_eq!(
            device["status"],
            serde_json::json!({
                "state": "unavailable",
                "reason": {"code": "software_rasterizer_skipped"},
                "isolated": false
            }),
            "{device}"
        );
        eprintln!("skipped {}: software rasterizer", device["id"]);
    }
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
}

/// A Vulkan or Metal adapter of type `cpu` in the device JSON.
fn is_software_rasterizer(device: &Value) -> bool {
    (device["backend"] == "vulkan" || device["backend"] == "metal")
        && device["id"]["adapter"] == "cpu"
}

/// `Engine::new` with `software_rasterizers` from `POLYPOUS_TEST_SOFTWARE_VULKAN=1`: skipped
/// without it, opened and probed with it, never the primary GPU either way.
#[test]
#[ignore = "gpu"]
fn engine_skips_software_rasterizers_unless_opted_in() {
    let opt_in = std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    let mut config = polypous::EngineConfig::default();
    config.software_rasterizers = opt_in;
    config.probe_cache_dir = Some(cache_dir("engine-software-rasterizers"));
    let engine = polypous::Engine::new(config).expect("engine");
    let devices: Vec<Value> = engine
        .devices()
        .iter()
        .map(|device| serde_json::to_value(device).expect("DeviceInfo serializes"))
        .collect();
    let software: Vec<&Value> = devices
        .iter()
        .filter(|device| is_software_rasterizer(device))
        .collect();
    eprintln!(
        "software_rasterizers={opt_in}: {} software rasterizer(s)",
        software.len()
    );
    for device in software {
        let skipped = device["status"]["reason"]["code"] == "software_rasterizer_skipped";
        eprintln!("{}: {}", device["id"], device["status"]);
        assert_eq!(skipped, !opt_in, "{device}");
        assert_ne!(device["status"]["state"], "not_probed");
        assert_eq!(device["primary"], false);
    }
}
