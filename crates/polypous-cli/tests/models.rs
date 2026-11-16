//! `polyp run` on the end-to-end MLP fixture (issue #46, docs/testing-strategy.md §3.5): the
//! built binary runs `fixtures/small/models/mlp/model.graph.json` with its PyTorch weights, and
//! its output file is compared with the reference outputs of the case, with the case's tolerance.
//! The facade tests (`crates/polypous/tests/models.rs`) cover every case, device and dtype.

#![allow(clippy::unwrap_used, clippy::expect_used)] // Test code: a panic is a test failure.
#![allow(clippy::result_large_err)] // `polypous::Error` is large by design (public-api.md §2 rule 7).

#[path = "../../polypous/tests/common/model_fixtures.rs"]
mod model_fixtures;

use std::path::{Path, PathBuf};
use std::process::Command;

use model_fixtures::{compare_outputs, graph_path, load_cases, model_dir, weights_path};
use polypous::tensor_file;

fn path_str(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn polyp_run_matches_pytorch_on_the_mlp() {
    let fixtures = model_dir(env!("CARGO_MANIFEST_DIR"), "mlp");
    let cases = load_cases(&fixtures);
    let case = cases
        .iter()
        .find(|c| c.name == "batch4_f32")
        .expect("the mlp fixture has a batch4_f32 case");
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("cli-models-mlp");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // One single-tensor file per graph input, as `--input NAME=FILE` takes them.
    let mut args: Vec<String> = vec![
        "run".into(),
        path_str(&graph_path(&fixtures)).into(),
        "--weights".into(),
        path_str(&weights_path(&fixtures)).into(),
    ];
    for input in &case.inputs {
        let file = dir.join(format!("{}.safetensors", input.name));
        tensor_file::write(&file, std::slice::from_ref(input)).unwrap();
        args.push("--input".into());
        args.push(format!("{}={}", input.name, file.display()));
    }
    let out = dir.join("out.safetensors");
    args.extend([
        "--output".into(),
        path_str(&out).into(),
        "--device".into(),
        "cpu".into(),
        "--compute-dtype".into(),
        case.compute_dtype.as_str().into(),
        "--probe-cache-dir".into(),
        path_str(&dir.join("probe-cache")).into(),
    ]);
    let output = Command::new(env!("CARGO_BIN_EXE_polyp"))
        .args(&args)
        .env_remove("RUST_LOG")
        .env_remove("POLYPOUS_TEST_PROBER_FAULTS")
        .env_remove("POLYPOUS_TEST_FINGERPRINT")
        .output()
        .expect("polyp runs");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(0), "{stderr}");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Wrote 2 outputs to"), "{stdout}");

    let outputs = tensor_file::read(&out).unwrap();
    if let Err(message) = compare_outputs(case, &outputs) {
        panic!(
            "polyp run output differs from PyTorch ({}):\n{message}",
            case.name
        );
    }
}
