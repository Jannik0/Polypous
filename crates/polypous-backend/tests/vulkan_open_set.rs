//! The up-front open of the Vulkan open set on the first Vulkan open (placement.md §1, "Remaining
//! window"; issue #118), on real Vulkan devices.
//!
//! The step runs once per process, so every scenario runs in a fresh process: the test re-runs
//! this test binary with only itself selected and `POLYPOUS_TEST_OPEN_SET_SCENARIO` set, and the
//! child runs the body. The race the step removes doesn't reproduce here (it needs Debian 13's
//! Vulkan loader and a debug build computing on one GPU while another is opened); these tests
//! check what the step does: which devices it opens, how often, and what happens to a device
//! that fails to open.
//!
//! Software rasterizers (lavapipe) are used only with `POLYPOUS_TEST_SOFTWARE_VULKAN=1`, as in
//! the other GPU tests (lavapipe crashes on Debian 13, M0-9).
//!
//! ```text
//! cargo test -p polypous-backend --features vulkan,test-fault-injection --test vulkan_open_set -- --include-ignored
//! ```

#![cfg(feature = "vulkan")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use std::sync::Barrier;

use polypous_backend::device::{
    BackendKind, DeviceId, DeviceInfo, DeviceStatus, describe_device, list_backend_devices,
    note_probe_results, probe_device, unprobed_device, vulkan_open_step,
};
use polypous_backend::{DType, DynTensor, TensorData, TensorError};

const SCENARIO_VAR: &str = "POLYPOUS_TEST_OPEN_SET_SCENARIO";

/// Runs `body` in a fresh process: re-runs this test binary with only `name` selected and the
/// scenario variable set (plus `env`), and fails if the child fails. In the child, runs `body`.
fn in_fresh_process(name: &str, env: &[(&str, &str)], body: impl FnOnce()) {
    if std::env::var(SCENARIO_VAR).as_deref() == Ok(name) {
        body();
        return;
    }
    let exe = std::env::current_exe().expect("the test binary's path");
    let status = std::process::Command::new(exe)
        .args([name, "--exact", "--include-ignored", "--nocapture"])
        .env(SCENARIO_VAR, name)
        .envs(env.iter().copied())
        .status()
        .expect("re-running the test binary");
    assert!(
        status.success(),
        "scenario {name} failed in its own process"
    );
}

/// Every listed Vulkan device, and those the tests may use: software rasterizers only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1`. `None` (with a SKIP line) if the tests may use none.
fn vulkan_devices() -> Option<(Vec<DeviceId>, Vec<DeviceId>)> {
    let listed = list_backend_devices(BackendKind::Vulkan).expect("Vulkan enumeration");
    let software = std::env::var("POLYPOUS_TEST_SOFTWARE_VULKAN").is_ok_and(|v| v == "1");
    let usable: Vec<DeviceId> = listed
        .iter()
        .copied()
        .filter(|id| software || !id.is_software_rasterizer())
        .collect();
    if usable.is_empty() {
        eprintln!(
            "SKIP: no Vulkan device to test (listed: {listed:?}; set POLYPOUS_TEST_SOFTWARE_VULKAN=1 for lavapipe)"
        );
        return None;
    }
    Some((listed, usable))
}

fn one_float() -> TensorData {
    TensorData::new(DType::F32, vec![1], 1f32.to_le_bytes().to_vec()).unwrap()
}

/// A probe result for `id` with `status`.
fn probed(id: DeviceId, status: DeviceStatus) -> DeviceInfo {
    DeviceInfo {
        status,
        ..unprobed_device(id)
    }
}

/// `threads` threads make their first Vulkan call at the same moment, each on one of `devices`,
/// and compute on it.
fn first_calls_at_once(devices: &[DeviceId], threads: usize) {
    let barrier = Barrier::new(threads);
    std::thread::scope(|scope| {
        for thread in 0..threads {
            let id = devices[thread % devices.len()];
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                let tensor = DynTensor::from_data(&one_float(), id).expect("upload");
                let sum = tensor.clone().add(tensor).expect("add");
                let back = sum.into_data().expect("read back");
                assert_eq!(back.bytes, 2f32.to_le_bytes(), "{id}");
            });
        }
    });
}

/// Asserts that the step ran once and opened exactly `expected`, each once, in order.
fn assert_opened_once(expected: &[DeviceId]) {
    let step = vulkan_open_step().expect("the step ran on the first Vulkan call");
    assert_eq!(step.runs, 1, "{step:?}");
    assert_eq!(step.opened, expected, "{step:?}");
    assert!(step.failed.is_empty(), "{step:?}");
}

/// Without probe results (a direct backend user), the step opens every listed Vulkan device but
/// software rasterizers, once, however many threads make their first call at once.
#[test]
#[ignore = "gpu"]
fn default_set_is_opened_once_by_concurrent_first_calls() {
    in_fresh_process(
        "default_set_is_opened_once_by_concurrent_first_calls",
        &[],
        || {
            let Some((listed, usable)) = vulkan_devices() else {
                return;
            };
            assert_eq!(
                vulkan_open_step(),
                None,
                "nothing opened before the first call"
            );
            first_calls_at_once(&usable, 8);
            let expected: Vec<DeviceId> = listed
                .into_iter()
                .filter(|id| !id.is_software_rasterizer())
                .collect();
            assert_opened_once(&expected);
            // Later calls, describing included, don't run it again.
            first_calls_at_once(&usable, 4);
            for &id in &usable {
                assert_eq!(describe_device(id).status, DeviceStatus::NotProbed, "{id}");
            }
            assert_opened_once(&expected);
        },
    );
}

/// With probe results that mark the test's devices usable (as the facade passes them), the step
/// opens exactly those, a software rasterizer included, before the callers' devices are used:
/// every caller's device is in what the step opened.
#[test]
#[ignore = "gpu"]
fn usable_set_is_opened_once_before_the_callers_devices() {
    in_fresh_process(
        "usable_set_is_opened_once_before_the_callers_devices",
        &[],
        || {
            let Some((_, usable)) = vulkan_devices() else {
                return;
            };
            let report: Vec<DeviceInfo> = usable
                .iter()
                .map(|&id| probed(id, DeviceStatus::Available { isolated: true }))
                .collect();
            assert!(note_probe_results(&report));
            first_calls_at_once(&usable, 8);
            assert_opened_once(&usable);
            assert!(
                !note_probe_results(&report),
                "the set is fixed once the step ran"
            );
        },
    );
}

/// A device that fails to open in the step (fault injection: `POLYPOUS_TEST_FAIL_OPEN` makes its
/// open panic like CubeCL does) is remembered: every later use is `DeviceUnavailable` with the
/// reason, and it is never opened again. Without that, these uses would open the (healthy)
/// device and succeed.
#[cfg(feature = "test-fault-injection")]
#[test]
#[ignore = "gpu"]
fn failed_open_is_remembered_and_never_retried() {
    // The scenario's device: the last one the tests may use (lavapipe in the cloud).
    let Some((_, usable)) = vulkan_devices() else {
        return;
    };
    let broken = *usable.last().unwrap_or_else(|| unreachable!());
    let broken_text = broken.to_string();
    in_fresh_process(
        "failed_open_is_remembered_and_never_retried",
        &[("POLYPOUS_TEST_FAIL_OPEN", &broken_text)],
        || {
            let report: Vec<DeviceInfo> = usable
                .iter()
                .map(|&id| probed(id, DeviceStatus::Available { isolated: true }))
                .collect();
            assert!(note_probe_results(&report));
            for attempt in 0..3 {
                match DynTensor::from_data(&one_float(), broken) {
                    Err(TensorError::DeviceUnavailable { device, reason }) => {
                        assert_eq!(device, broken);
                        assert!(reason.contains("injected"), "{reason}");
                        assert!(reason.contains("not opened again"), "{reason}");
                    }
                    other => panic!("attempt {attempt}: expected DeviceUnavailable, got {other:?}"),
                }
            }
            let info = describe_device(broken);
            assert!(
                info.status.to_string().contains("not opened again"),
                "{}",
                info.status
            );
            assert!(!probe_device(broken).is_available());
            let healthy: Vec<DeviceId> =
                usable.iter().copied().filter(|&id| id != broken).collect();
            if !healthy.is_empty() {
                first_calls_at_once(&healthy, 4);
            }
            let step = vulkan_open_step().expect("the step ran");
            assert_eq!(step.runs, 1);
            assert_eq!(step.opened, healthy);
            assert_eq!(step.failed.len(), 1, "{step:?}");
            assert_eq!(step.failed[0].0, broken);
        },
    );
}
