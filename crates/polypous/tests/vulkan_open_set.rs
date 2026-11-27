//! The facade passes its probe results to the backend, so the up-front open of the Vulkan open
//! set (placement.md §1, "Remaining window"; issue #118) opens only devices a probe found usable
//! and none rejected (ADR-0013).
//!
//! Each scenario runs in a fresh process (the step runs once per process): the test re-runs this
//! test binary with only itself selected and `POLYPOUS_TEST_OPEN_SET_SCENARIO` set. The engines
//! get their probe reports through `EngineConfig::probe_report`, so no probe runs and nothing is
//! opened before the first Vulkan call. Software rasterizers (lavapipe) are used only with
//! `POLYPOUS_TEST_SOFTWARE_VULKAN=1` (M0-9).

#![cfg(feature = "vulkan")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.

use polypous::{Engine, EngineConfig};
use polypous_backend::device::{BackendKind, DeviceId, list_backend_devices, vulkan_open_step};
use polypous_backend::{DType, DynTensor, TensorData};
use serde_json::{Value, json};

const SCENARIO_VAR: &str = "POLYPOUS_TEST_OPEN_SET_SCENARIO";

/// Runs `body` in a fresh process (see the module docs); in the child, runs it.
fn in_fresh_process(name: &str, body: impl FnOnce()) {
    if std::env::var(SCENARIO_VAR).as_deref() == Ok(name) {
        body();
        return;
    }
    let exe = std::env::current_exe().expect("the test binary's path");
    let status = std::process::Command::new(exe)
        .args([name, "--exact", "--include-ignored", "--nocapture"])
        .env(SCENARIO_VAR, name)
        .status()
        .expect("re-running the test binary");
    assert!(
        status.success(),
        "scenario {name} failed in its own process"
    );
}

/// Listed Vulkan devices and those the test may use (software rasterizers only with
/// `POLYPOUS_TEST_SOFTWARE_VULKAN=1`); `None` with a SKIP line if none.
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

/// The facade's JSON form of a backend device id.
fn id_json(id: DeviceId) -> Value {
    match id {
        DeviceId::Vulkan { adapter, index } => {
            json!({"backend": "vulkan", "adapter": serde_json::to_value(adapter).unwrap(), "index": index})
        }
        other => panic!("not a Vulkan id: {other}"),
    }
}

/// A probe report device entry for `id` with `status` (public-api.md §1.2 JSON).
fn device_json(id: DeviceId, status: Value) -> Value {
    json!({
        "id": id_json(id),
        "backend": "vulkan",
        "name": id.to_string(),
        "vendor": null,
        "driver": null,
        "kind": if id.is_software_rasterizer() { "cpu" } else { "other" },
        "dedicated_memory": null,
        "free_memory": null,
        "unified_memory": true,
        "identity": {"pci_bus_id": null, "luid": null, "registry_id": null},
        "capabilities": {"f16": false, "bf16": false, "subgroup_size_min": null,
                         "subgroup_size_max": null, "max_buffer_size": null,
                         "max_shared_memory": null, "compute_units": null},
        "status": status,
        "physical_gpu": null,
        "primary": false
    })
}

fn available() -> Value {
    json!({"state": "available", "isolated": true})
}

fn crashed() -> Value {
    json!({"state": "unavailable", "reason": {"code": "crashed_during_probe"}, "isolated": true})
}

fn skipped() -> Value {
    json!({"state": "unavailable", "reason": {"code": "software_rasterizer_skipped"}, "isolated": false})
}

/// An engine whose devices come from a report: `status(id)` for each listed Vulkan device.
fn engine_with_report(listed: &[DeviceId], status: impl Fn(DeviceId) -> Value) -> Engine {
    let devices: Vec<Value> = listed
        .iter()
        .map(|&id| device_json(id, status(id)))
        .collect();
    let config = json!({"probe_report": {"devices": devices, "isolated": true}});
    let config = EngineConfig::from_json(&config.to_string()).expect("engine config");
    Engine::new(config).expect("engine")
}

/// The first Vulkan call of the process, on `id`.
fn first_vulkan_call(id: DeviceId) {
    let data = TensorData::new(DType::F32, vec![1], 1f32.to_le_bytes().to_vec()).unwrap();
    let tensor = DynTensor::from_data(&data, id).expect("upload");
    assert_eq!(tensor.into_data().unwrap().bytes, 1f32.to_le_bytes());
}

/// An engine whose probe found the test's devices usable (an opted-in software rasterizer
/// included) makes the step open exactly those; devices the test may not use are reported as
/// skipped, which gives no verdict.
#[test]
#[ignore = "gpu"]
fn the_engines_usable_devices_are_the_open_set() {
    in_fresh_process("the_engines_usable_devices_are_the_open_set", || {
        let Some((listed, usable)) = vulkan_devices() else {
            return;
        };
        let _engine = engine_with_report(&listed, |id| {
            if usable.contains(&id) {
                available()
            } else {
                skipped()
            }
        });
        assert_eq!(vulkan_open_step(), None, "Engine::new opened nothing");
        first_vulkan_call(usable[0]);
        let step = vulkan_open_step().expect("the step ran");
        assert_eq!(step.runs, 1);
        assert_eq!(step.opened, usable, "{step:?}");
        assert!(step.failed.is_empty(), "{step:?}");
    });
}

/// A device one engine's probe rejected is never opened by the step, even though another engine
/// found it usable (several engines: excluded if any probe rejected it).
#[test]
#[ignore = "gpu"]
fn a_device_any_engine_rejected_is_not_opened() {
    in_fresh_process("a_device_any_engine_rejected_is_not_opened", || {
        let Some((listed, usable)) = vulkan_devices() else {
            return;
        };
        // The rejected device: the last one the test may use (lavapipe in the cloud).
        let rejected = *usable.last().unwrap_or_else(|| unreachable!());
        let status = |id: DeviceId| {
            if usable.contains(&id) {
                available()
            } else {
                skipped()
            }
        };
        let _trusting = engine_with_report(&listed, status);
        let _rejecting = engine_with_report(&listed, |id| {
            if id == rejected {
                crashed()
            } else {
                status(id)
            }
        });
        // The first call goes to the first usable device (the rejected one itself if it is the
        // only one: it then opens alone, after the step).
        first_vulkan_call(usable[0]);
        let step = vulkan_open_step().expect("the step ran");
        let expected: Vec<DeviceId> = usable
            .iter()
            .copied()
            .filter(|&id| id != rejected)
            .collect();
        assert_eq!(step.runs, 1);
        assert_eq!(step.opened, expected, "{step:?}");
        assert!(!step.opened.contains(&rejected));
        assert!(step.failed.is_empty(), "{step:?}");
    });
}
