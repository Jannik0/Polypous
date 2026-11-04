//! `polyp devices` end to end: runs the built binary (issue #6).
//!
//! These run in CI with the default (`cpu`) features, and in the lavapipe job with `vulkan`
//! compiled in. They probe devices in the test's child process, not in a prober process (#14).

use std::process::Command;

use serde_json::Value;

// Test helper: failing to start the binary under test should fail the test.
#[allow(clippy::expect_used)]
fn polyp(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(args)
        .env_remove("RUST_LOG")
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
    assert_eq!(
        cpu["status"],
        serde_json::json!({"state": "available", "isolated": false})
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
    for column in ["BACKEND", "NAME", "KIND", "DEDICATED", "UNIFIED", "STATUS"] {
        assert!(header.contains(column), "{header}");
    }
    assert!(
        table
            .lines()
            .any(|line| line.trim_start().starts_with("cpu ") && line.contains("available")),
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
    eprintln!("{}", String::from_utf8_lossy(&output.stdout));
}
