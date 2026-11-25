//! The MNIST training test's fixtures (testing-strategy.md §3.8, issue #84): the committed graphs,
//! configs and reference expectations in `fixtures/small/mnist/` (written by
//! `tools/reference/gen_mnist.py`) and the converted MNIST data in the fixture cache
//! (`fixtures/manifest.json` entries `mnist-train` and `mnist-test`).
//!
//! Shared by `crates/polypous/tests/mnist.rs` and `crates/polypous-cli/tests/train_mnist.rs`,
//! which include this file with `#[path]`. Only the public `polypous` API is used.

#![allow(
    dead_code,
    reason = "each test binary uses a different part of this module"
)]

use std::path::{Path, PathBuf};

use polypous::{Engine, EngineConfig};
use serde_json::{Value, json};

/// The repository root, from `CARGO_MANIFEST_DIR` (both crates sit in `crates/`).
pub fn repo(manifest_dir: &str) -> PathBuf {
    Path::new(manifest_dir).join("../..")
}

/// `fixtures/small/mnist/`.
pub fn fixture_dir(manifest_dir: &str) -> PathBuf {
    repo(manifest_dir).join("fixtures/small/mnist")
}

/// The command that fills the cache, named in every error about it.
pub const FETCH_HINT: &str = "run `python tools/reference/fetch_fixtures.py mnist-train mnist-test` \
(or `python tools/reference/convert_mnist.py`) from the repository root";

/// The converted MNIST files: `(train, test)`.
#[derive(Debug)]
pub struct MnistCache {
    pub train: PathBuf,
    pub test: PathBuf,
}

/// Finds the `mnist-train` and `mnist-test` entries of `manifest` in `cache_dir` and checks that
/// each file is there with the manifest's size. (Their SHA-256 is checked when they are fetched or
/// converted; the trainer itself validates every header and size of the files it reads.) The
/// error says what is missing and how to get it.
pub fn find_cache(manifest: &Path, cache_dir: &Path) -> Result<MnistCache, String> {
    let text = std::fs::read_to_string(manifest)
        .map_err(|e| format!("can't read {}: {e}", manifest.display()))?;
    let doc: Value =
        serde_json::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
    let entry = |name: &str| -> Result<PathBuf, String> {
        let entries = doc["entries"].as_array().map(Vec::as_slice).unwrap_or(&[]);
        let e = entries
            .iter()
            .find(|e| e["name"] == name)
            .ok_or_else(|| format!("{} has no entry `{name}`", manifest.display()))?;
        let rel = e["path"].as_str().unwrap_or_default();
        let size = e["size"].as_u64().unwrap_or_default();
        let path = cache_dir.join(rel);
        match std::fs::metadata(&path) {
            Ok(meta) if meta.is_file() && meta.len() == size => Ok(path),
            Ok(meta) => Err(format!(
                "the MNIST data file {} has {} bytes, the manifest entry `{name}` says {size}: {FETCH_HINT}",
                path.display(),
                meta.len()
            )),
            Err(_) => Err(format!(
                "the MNIST data file {} is missing (manifest entry `{name}`): {FETCH_HINT}",
                path.display()
            )),
        }
    };
    Ok(MnistCache {
        train: entry("mnist-train")?,
        test: entry("mnist-test")?,
    })
}

/// The cache of this checkout, or a test failure that says how to fill it.
pub fn require_cache(manifest_dir: &str) -> MnistCache {
    let root = repo(manifest_dir);
    find_cache(
        &root.join("fixtures/manifest.json"),
        &root.join("fixtures/cache"),
    )
    .unwrap_or_else(|message| panic!("The MNIST tests need the MNIST data: {message}"))
}

/// A committed JSON fixture of `fixtures/small/mnist/`.
pub fn fixture_json(manifest_dir: &str, name: &str) -> Value {
    let path = fixture_dir(manifest_dir).join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// `train.json` with only its run directory changed (to `run_dir`), so parallel or repeated runs
/// never share one; its data paths stay relative to `fixtures/small/mnist/`.
pub fn train_config_with_run_dir(manifest_dir: &str, run_dir: &Path) -> Value {
    let mut doc = fixture_json(manifest_dir, "train.json");
    doc["checkpoint"]["dir"] = json!(run_dir.to_str().unwrap());
    doc
}

/// `doc` with its dataset paths made absolute against `fixtures/small/mnist/`, for a copy of the
/// config written elsewhere (the CLI resolves relative paths against the config file).
pub fn with_absolute_data(manifest_dir: &str, mut doc: Value) -> Value {
    let base = fixture_dir(manifest_dir);
    for role in ["train", "validation"] {
        if let Some(path) = doc["data"][role]["path"].as_str() {
            let absolute = base.join(path);
            doc["data"][role]["path"] = json!(absolute.to_str().unwrap());
        }
    }
    doc
}

/// An engine whose only available device is the CPU (no probe runs).
pub fn cpu_engine() -> Engine {
    let config = json!({ "probe_report": { "devices": [{
        "id": {"backend": "cpu"}, "backend": "cpu", "name": "test cpu", "vendor": null,
        "driver": null, "kind": "cpu", "dedicated_memory": null, "free_memory": null,
        "unified_memory": true,
        "identity": { "pci_bus_id": null, "luid": null, "registry_id": null },
        "capabilities": { "f16": true, "bf16": true, "subgroup_size_min": null,
            "subgroup_size_max": null, "max_buffer_size": null, "max_shared_memory": null,
            "compute_units": null },
        "status": {"state": "available", "isolated": true}, "physical_gpu": null, "primary": false
    }], "isolated": true } });
    Engine::new(EngineConfig::from_json(&config.to_string()).unwrap()).unwrap()
}

/// The accuracy criterion of testing-strategy.md §3.8.
pub const MIN_VAL_ACCURACY: f64 = 0.98;
