//! An app that links Polypous and serves as its own prober (ADR-0013 §3, option 2; issue #14).
//!
//! This test binary has its own `main` (`harness = false`). Like an app, it calls
//! `polypous::probe::run_prober_if_requested()` first; when Polypous re-launches it with the
//! prober arguments, that call runs the probe and exits. The rest of `main` is the "app": it probes
//! out of process, which only works if Polypous found and started this executable as the prober.

// Test binary: failing assertions and unexpected errors should panic.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use polypous::probe::{self, ProbeEvent};
use polypous::{CancellationToken, DeviceId, Engine, EngineConfig, Isolation, ProbeOptions};

fn main() {
    probe::run_prober_if_requested();

    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("embedded-prober");
    let _ = std::fs::remove_dir_all(&dir);

    // Out of process only: an error if this executable could not serve as the prober.
    let mut options = ProbeOptions::default();
    options.isolation = Isolation::OutOfProcess;
    options.probe_cache_dir = Some(dir.clone());
    let mut events = Vec::new();
    let report = probe::probe_devices(options.clone(), &CancellationToken::new(), |event| {
        events.push(event);
    })
    .expect("probing through the re-launched test binary");
    assert_eq!(report.prober_starts, 1, "{report:#?}");
    assert!(report.isolated, "{report:#?}");
    assert!(matches!(events.first(), Some(ProbeEvent::ProbeStarted)));
    assert!(matches!(
        events.last(),
        Some(ProbeEvent::ProbeFinished { .. })
    ));
    if cfg!(feature = "cpu") {
        let cpu = report
            .devices
            .iter()
            .find(|device| device.id == DeviceId::Cpu)
            .expect("the CPU");
        assert_eq!(
            serde_json::to_value(&cpu.status).unwrap(),
            serde_json::json!({"state": "available", "isolated": true})
        );
    }

    // `Engine::new` with the same cache directory: every device from the cache. With a known OS
    // driver fingerprint (Linux, Windows) no prober starts; with an unknown one (macOS) one prober
    // enumerates and compares, but opens nothing.
    let mut config = EngineConfig::default();
    config.probe_cache_dir = Some(dir.clone());
    let engine = Engine::new(config).expect("engine");
    let cached = engine.probe_report();
    let expected_starts = if report.fingerprint_known { 0 } else { 1 };
    assert_eq!(cached.prober_starts, expected_starts, "{cached:#?}");
    assert_eq!(cached.cache_hits.len(), report.devices.len(), "{cached:#?}");
    assert_eq!(engine.devices(), report.devices);

    // A forced refresh starts the prober again.
    let mut force = ProbeOptions::default();
    force.force = true;
    let refreshed = engine
        .refresh_devices(force, &CancellationToken::new(), |_| {})
        .expect("refresh");
    assert_eq!(refreshed.prober_starts, 1);
    assert!(refreshed.cache_hits.is_empty());

    let _ = std::fs::remove_dir_all(&dir);
    println!("embedded_prober: ok ({} devices)", report.devices.len());
}
