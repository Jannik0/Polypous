//! The prober, probe cache and crash journal end to end (issue #14, "Tests").
//!
//! Each test runs the built `polyp` binary, which starts itself as the prober. The test-only fault
//! injection of `polypous-backend` (feature `test-fault-injection`, turned on by this crate's
//! dev-dependency) replaces the real devices by fakes and injects crashes, hangs and malformed
//! output through `POLYPOUS_TEST_PROBER_FAULTS`; `POLYPOUS_TEST_FINGERPRINT` replaces the OS driver
//! fingerprint so cache behavior is the same on every OS. They run on Linux and Windows CI.
//! Every test uses its own probe cache directory.

// Test code: a failed unwrap is a failed test.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

const FAULTS: &str = "POLYPOUS_TEST_PROBER_FAULTS";
const FINGERPRINT: &str = "POLYPOUS_TEST_FINGERPRINT";

/// Fake devices of the reference machine (no driver is loaded).
const FAKES: &str = "fake_devices=cuda:0,vulkan:discrete_gpu:0,vulkan:integrated_gpu:0,cpu";

fn cache_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("prober-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Runs `polyp devices --json --report` with the given cache directory, faults, fingerprint and
/// extra arguments.
fn devices(dir: &Path, faults: &str, fingerprint: &str, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_polyp"));
    command
        .args(["devices", "--json", "--report", "--probe-cache-dir"])
        .arg(dir)
        .args(extra)
        .env_remove("RUST_LOG")
        .env(FAULTS, faults)
        .env(FINGERPRINT, fingerprint);
    command.output().expect("polyp runs")
}

/// The report of a successful run.
fn report(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "polyp failed ({:?}); stderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("the report is JSON")
}

fn id(text: &str) -> Value {
    match text {
        "cpu" => json!({"backend": "cpu"}),
        "cuda:0" => json!({"backend": "cuda", "index": 0}),
        other => {
            let parts: Vec<&str> = other.split(':').collect();
            json!({"backend": parts[0], "adapter": parts[1], "index": parts[2].parse::<u64>().unwrap()})
        }
    }
}

/// The status of a device in a report.
fn status(report: &Value, device: &str) -> Value {
    report["devices"]
        .as_array()
        .expect("devices")
        .iter()
        .find(|entry| entry["id"] == id(device))
        .unwrap_or_else(|| panic!("{device} is in the report: {report:#}"))["status"]
        .clone()
}

fn available(isolated: bool) -> Value {
    json!({"state": "available", "isolated": isolated})
}

fn unavailable(code: &str, isolated: bool) -> Value {
    json!({"state": "unavailable", "reason": {"code": code}, "isolated": isolated})
}

fn warnings(report: &Value) -> String {
    report["warnings"].to_string()
}

#[test]
fn crash_on_a_device_marks_it_and_the_others_are_reported() {
    let dir = cache_dir("crash");
    let faults = format!("{FAKES};crash=vulkan:integrated_gpu:0");
    let report = report(&devices(&dir, &faults, "fp-a", &[]));
    assert_eq!(
        status(&report, "vulkan:integrated_gpu:0"),
        unavailable("crashed_during_probe", true)
    );
    for device in ["cuda:0", "vulkan:discrete_gpu:0", "cpu"] {
        assert_eq!(status(&report, device), available(true), "{device}");
    }
    // Restarted once for the CPU, which comes after the crashing device in probe order.
    assert_eq!(report["prober_starts"], 2);
    assert_eq!(report["isolated"], true);
    assert!(
        warnings(&report).contains("while probing vulkan:integrated_gpu:0"),
        "{report:#}"
    );
    // The fake RTX via CUDA is the primary GPU.
    let primary: Vec<&Value> = report["devices"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|device| device["primary"] == true)
        .collect();
    assert_eq!(primary.len(), 1);
    assert_eq!(primary[0]["id"], id("cuda:0"));
}

#[test]
fn crashed_device_stays_unavailable_until_the_fingerprint_changes() {
    let dir = cache_dir("fingerprint");
    let crash = format!("{FAKES};crash=vulkan:integrated_gpu:0");
    let first = report(&devices(&dir, &crash, "fp-a", &[]));
    assert_eq!(first["prober_starts"], 2);

    // Same fingerprint, no fault: a complete cache hit, no prober, the device stays crashed.
    let cached = report(&devices(
        &dir,
        &format!("{FAKES};exit_before_header"),
        "fp-a",
        &[],
    ));
    assert_eq!(cached["prober_starts"], 0, "{cached:#}");
    assert_eq!(cached["cache_hits"].as_array().unwrap().len(), 4);
    assert_eq!(
        status(&cached, "vulkan:integrated_gpu:0"),
        unavailable("crashed_during_probe", true)
    );

    // A new driver fingerprint: everything is probed again, and the device works now.
    let fresh = report(&devices(&dir, FAKES, "fp-b", &[]));
    assert_eq!(fresh["prober_starts"], 1);
    assert_eq!(fresh["cache_hits"], json!([]));
    assert_eq!(status(&fresh, "vulkan:integrated_gpu:0"), available(true));
}

#[test]
fn hang_times_out_kills_the_prober_and_recovers() {
    let dir = cache_dir("hang");
    let pid_file = dir.with_extension("pid");
    let _ = std::fs::remove_file(&pid_file);
    let faults = format!("{FAKES};hang=cuda:0;pid_file={}", pid_file.display());
    let started = Instant::now();
    let report = report(&devices(
        &dir,
        &faults,
        "fp-a",
        &["--timeout-per-device-ms", "1500"],
    ));
    assert!(started.elapsed() < Duration::from_secs(60));
    assert_eq!(
        status(&report, "cuda:0"),
        unavailable("probe_timed_out", true)
    );
    for device in ["vulkan:discrete_gpu:0", "vulkan:integrated_gpu:0", "cpu"] {
        assert_eq!(status(&report, device), available(true), "{device}");
    }
    assert_eq!(report["prober_starts"], 2);
    assert!(warnings(&report).contains("was killed"), "{report:#}");

    // Both probers (the hung one included) are gone: killed or exited, and reaped.
    #[cfg(target_os = "linux")]
    {
        let pids = std::fs::read_to_string(&pid_file).expect("the probers wrote their pids");
        let pids: Vec<&str> = pids.lines().collect();
        assert_eq!(pids.len(), 2, "{pids:?}");
        for pid in pids {
            assert!(
                !Path::new(&format!("/proc/{pid}")).exists(),
                "prober {pid} still exists"
            );
        }
    }
}

#[test]
fn unknown_fingerprint_compares_the_enumeration() {
    let dir = cache_dir("unknown");
    let first = report(&devices(&dir, FAKES, "unknown", &[]));
    assert_eq!(first["fingerprint_known"], false);
    assert_eq!(first["prober_starts"], 1);
    assert_eq!(first["cache_hits"], json!([]));

    // Enumerate-only: one prober run that opens nothing (a probe would crash it here).
    let crash_all =
        format!("{FAKES};crash=cuda:0,vulkan:discrete_gpu:0,vulkan:integrated_gpu:0,cpu");
    let second = report(&devices(&dir, &crash_all, "unknown", &[]));
    assert_eq!(second["prober_starts"], 1);
    assert_eq!(second["cache_hits"].as_array().unwrap().len(), 4);
    assert_eq!(status(&second, "cpu"), available(true));

    // A changed driver (enumeration fingerprint) means a new probe.
    let third = report(&devices(
        &dir,
        &format!("{FAKES};fake_driver=2.0"),
        "unknown",
        &[],
    ));
    assert_eq!(third["cache_hits"], json!([]));
    assert_eq!(status(&third, "cuda:0"), available(true));
}

#[test]
fn malformed_oversized_and_endless_output_is_a_clean_failure() {
    for (fault, message) in [
        ("garbage", "malformed line"),
        ("oversized", "longer than"),
        // Any extra line while a device is in progress breaks the protocol order before the
        // line limit is reached; the limit itself is unit-tested in the supervisor.
        ("spam", "while probing"),
    ] {
        let dir = cache_dir(fault);
        let faults = format!("{FAKES};{fault}=vulkan:discrete_gpu:0");
        let report = report(&devices(&dir, &faults, "fp-a", &[]));
        let failed = status(&report, "vulkan:discrete_gpu:0");
        assert_eq!(failed["state"], "unavailable", "{fault}: {report:#}");
        assert_eq!(failed["reason"]["code"], "probe_failed", "{fault}");
        let text = failed["reason"]["message"].as_str().unwrap_or_default();
        assert!(text.contains("invalid output"), "{fault}: {text}");
        assert!(text.contains(message), "{fault}: {text}");
        for device in ["cuda:0", "vulkan:integrated_gpu:0", "cpu"] {
            assert_eq!(
                status(&report, device),
                available(true),
                "{fault}: {device}"
            );
        }
    }
}

#[test]
fn crash_while_enumerating_skips_that_backend() {
    let dir = cache_dir("enumeration");
    let faults = format!("{FAKES};crash_enumeration=vulkan");
    let report = report(&devices(&dir, &faults, "fp-a", &[]));
    let ids: Vec<Value> = report["devices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|device| device["id"].clone())
        .collect();
    assert_eq!(ids, vec![id("cuda:0"), id("cpu")]);
    assert_eq!(report["prober_starts"], 2);
    assert!(
        warnings(&report).contains("while enumerating vulkan devices"),
        "{report:#}"
    );
}

#[test]
fn journal_leftover_marks_the_device_in_the_fallback() {
    let dir = cache_dir("journal");
    let faults = format!("{FAKES};crash=vulkan:integrated_gpu:0");
    let in_process = ["--isolation", "in_process"];

    // In process, the crash takes polyp itself down.
    let crashed = devices(&dir, &faults, "fp-a", &in_process);
    assert!(!crashed.status.success(), "{crashed:?}");

    // The next start finds the journal and doesn't open the device again (the fault is still
    // set: opening it would crash again).
    let report = report(&devices(&dir, &faults, "fp-a", &in_process));
    assert_eq!(
        status(&report, "vulkan:integrated_gpu:0"),
        unavailable("crashed_during_probe", false)
    );
    for device in ["cuda:0", "vulkan:discrete_gpu:0", "cpu"] {
        assert_eq!(status(&report, device), available(false), "{device}");
    }
    assert_eq!(report["isolated"], false);
    assert_eq!(report["prober_starts"], 0);
    assert!(warnings(&report).contains("crashed this app"), "{report:#}");
    let journals = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .filter(|entry| entry.file_name().to_string_lossy().contains("journal"))
        .count();
    assert_eq!(journals, 0);

    // The table says that probing was not isolated.
    let table = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(["devices", "--isolation", "in_process", "--probe-cache-dir"])
        .arg(&dir)
        .env(FAULTS, &faults)
        .env(FINGERPRINT, "fp-a")
        .output()
        .unwrap();
    let table = String::from_utf8_lossy(&table.stdout);
    assert!(table.contains("NOT isolated"), "{table}");
    assert!(
        table.contains("crashed during probe, not isolated"),
        "{table}"
    );
}

#[test]
fn prober_that_cannot_start_falls_back_or_fails() {
    let dir = cache_dir("no-header");
    let faults = format!("{FAKES};exit_before_header");
    let report = report(&devices(&dir, &faults, "fp-a", &[]));
    assert_eq!(report["isolated"], false);
    assert_eq!(status(&report, "cpu"), available(false));
    assert!(
        warnings(&report).contains("could not be used"),
        "{report:#}"
    );

    let dir = cache_dir("no-header-strict");
    let strict = devices(&dir, &faults, "fp-a", &["--isolation", "out_of_process"]);
    assert!(!strict.status.success());
    let stderr = String::from_utf8_lossy(&strict.stderr);
    assert!(
        stderr.contains("backend_unavailable") || stderr.contains("separate process"),
        "{stderr}"
    );
}

#[test]
fn supervised_prober_exits_when_its_stdin_closes() {
    // A prober hanging on a device, whose parent is gone (stdin at end of file at once).
    let started = Instant::now();
    let status = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(["--polypous-prober", "--request", "{}"])
        .env(FAULTS, "fake_devices=cpu;hang=cpu")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(3));
    assert!(started.elapsed() < Duration::from_secs(30));
}

#[test]
fn probe_json_writes_the_protocol() {
    let output = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(["probe", "--json"])
        .env(FAULTS, "fake_devices=vulkan:cpu:0,cpu")
        .output()
        .unwrap();
    assert!(output.status.success());
    let lines: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    // Fake devices enumerate every backend (cuda, metal, vulkan, cpu), so leave out the
    // `enumerating` lines.
    let types: Vec<&str> = lines
        .iter()
        .map(|line| line["type"].as_str().unwrap())
        .filter(|kind| *kind != "enumerating")
        .collect();
    assert_eq!(
        types,
        [
            "header",
            "enumerated",
            "enumerated",
            "probing",
            "probed",
            "probing",
            "probed",
            "done"
        ]
    );
    assert_eq!(lines[0]["protocol"], 1);
    let probed: Vec<&Value> = lines
        .iter()
        .filter(|line| line["type"] == "probed")
        .collect();
    // Lavapipe without the opt-in: skipped, not opened.
    assert_eq!(
        probed[0]["device"]["status"],
        unavailable("software_rasterizer_skipped", false)
    );
    assert_eq!(probed[1]["device"]["status"], available(true));

    let bad = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(["probe", "--json", "--request", r#"{"protocol":42}"#])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(2));
}
