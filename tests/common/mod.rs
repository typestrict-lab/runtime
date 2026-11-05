//! The test model: typestrict/TS-PRGuard at the revision `scripts/fetch-test-model.sh` pins.

use std::io::Write;
use std::path::PathBuf;

/// The model directory (`TYPESTRICT_TEST_MODEL`, else `test-model/`), or `None` with a message saying how to get it.
pub fn model_dir(test: &str) -> Option<PathBuf> {
    let dir = std::env::var_os("TYPESTRICT_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-model"));
    if dir.join("decision_ai.json").is_file() && dir.join("onnx/model_8bit.onnx").is_file() {
        Some(dir)
    } else {
        // Straight to stderr: the test harness captures eprintln!, and a skip must be seen.
        let _ = writeln!(
            std::io::stderr(),
            "SKIPPED {test}: no test model in {} (run scripts/fetch-test-model.sh, or set TYPESTRICT_TEST_MODEL)",
            dir.display()
        );
        None
    }
}
