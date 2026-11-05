//! The Rust reader against the typestrict.ai playground's: `tests/fixtures/golden.json` holds what the playground's
//! `reader.ts` (run in Node by `scripts/golden/make-golden.mts`, with onnxruntime-web) fed the model and answered for
//! each question. Token ids, option positions, question type and padding must be identical; probabilities and
//! confidence within 1e-4 (ONNX Runtime native against WebAssembly).
//!
//! Needs the test model (`scripts/fetch-test-model.sh`); skipped with a message without it.

mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use typestrict_serve::manifest::Kind;
use typestrict_serve::reader::{Encoder, Model, Question};

const TOLERANCE: f64 = 1e-4;

fn golden() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// A model directory with the case's manifest patch (top-level keys replaced, as the generator does), the other
/// files linked to the test model's.
fn patched(model: &Path, case: &str, patch: &Value) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("typestrict-serve-golden-{}-{case}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut m: Value = serde_json::from_str(&std::fs::read_to_string(model.join("decision_ai.json")).unwrap()).unwrap();
    m.as_object_mut().unwrap().extend(patch.as_object().unwrap().clone());
    std::fs::write(dir.join("decision_ai.json"), m.to_string()).unwrap();
    for f in ["tokenizer.json", "onnx"] {
        #[cfg(unix)]
        std::os::unix::fs::symlink(model.join(f).canonicalize().unwrap(), dir.join(f)).unwrap();
        #[cfg(not(unix))]
        panic!("the golden test links {f}; run it on a Unix system");
    }
    dir
}

fn question(q: &Value) -> Question {
    let kind = serde_json::from_value::<Kind>(q["type"].clone()).unwrap();
    let opt = |v: &Value| Some(v.clone()).filter(|v| !v.is_null());
    Question {
        kind,
        instructions: opt(&q["instructions"]),
        state: opt(&q["state"]),
        options: q["options"].as_array().unwrap().iter().map(opt).collect(),
    }
}

fn ints(v: &Value) -> Vec<i64> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

#[test]
fn the_reader_matches_the_playground() {
    let Some(model) = common::model_dir("the_reader_matches_the_playground") else { return };
    let g = golden();
    let mut loaded: Option<(String, Model)> = None;
    let mut worst = 0.0f64;
    let mut checked = 0;
    eprintln!("{:<32} {:>6} {:>8} {:>12} {:>12}", "case", "tokens", "ids", "max |dp|", "|dconf|");
    for case in g["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let patch = &case["manifest_patch"];
        let key = patch.to_string();
        if loaded.as_ref().is_none_or(|(k, _)| *k != key) {
            let dir = if patch.is_null() { model.clone() } else { patched(&model, name, patch) };
            let enc = Arc::new(Encoder::load(&dir).unwrap());
            loaded = Some((key, Model::load(&dir, enc, 1).unwrap()));
        }
        let m = &mut loaded.as_mut().unwrap().1;
        let q = question(&case["question"]);

        let e = m.enc.encode(&q).unwrap();
        let ids: Vec<i64> = e.ids.iter().map(|&i| i64::from(i)).collect();
        let want = ints(&case["ids"]);
        if ids != want {
            let at = ids.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(ids.len().min(want.len()));
            panic!("{name}: token ids differ at {at} (rust {} ids, playground {})", ids.len(), want.len());
        }
        let positions: Vec<i64> = e.option_positions.iter().map(|&p| p as i64).collect();
        assert_eq!(positions, ints(&case["option_positions"]), "{name}: option positions");
        assert_eq!(e.question_type, case["question_type"].as_i64().unwrap(), "{name}: question type");
        assert_eq!(e.padded_length as u64, case["padded_length"].as_u64().unwrap(), "{name}: padded length");
        assert_eq!(e.ids.len() as u64, case["tokens"].as_u64().unwrap(), "{name}: tokens");

        let Some(want_p) = case["probabilities"].as_array() else {
            // The playground could not run it (onnxruntime-web at 4,096 tokens): ids only, and the native run works.
            let d = m.run(&e).unwrap();
            assert!((d.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            eprintln!("{name:<32} {:>6} {:>8} {:>12} {:>12}", e.ids.len(), "equal", "(ids only)", "");
            continue;
        };
        let d = m.run(&e).unwrap();
        assert_eq!(d.probabilities.len(), want_p.len(), "{name}: option count");
        let dp = d.probabilities.iter().zip(want_p).map(|(a, b)| (a - b.as_f64().unwrap()).abs()).fold(0.0, f64::max);
        let dc = match (d.confidence, case["confidence"].as_f64()) {
            (Some(a), Some(b)) => (a - b).abs(),
            (None, None) => 0.0,
            (a, b) => panic!("{name}: confidence {a:?} against the playground's {b:?}"),
        };
        eprintln!("{name:<32} {:>6} {:>8} {dp:>12.2e} {dc:>12.2e}", e.ids.len(), "equal");
        assert!(dp < TOLERANCE, "{name}: probabilities differ by {dp:e}: {:?} against {want_p:?}", d.probabilities);
        assert!(dc < TOLERANCE, "{name}: confidence differs by {dc:e}");
        assert_eq!(d.best as u64, case["best"].as_u64().unwrap(), "{name}: best option");
        worst = worst.max(dp).max(dc);
        checked += 1;
    }
    eprintln!("{checked} answers within {TOLERANCE:e} of the playground's (worst {worst:.2e})");
    assert!(checked >= 10);
}

/// Questions batched in one ONNX run answer as they do alone: the pad and option masks keep the padding of the shorter
/// rows (tokens and options) out of every row.
#[test]
fn a_batch_answers_as_its_questions_alone() {
    let Some(model) = common::model_dir("a_batch_answers_as_its_questions_alone") else { return };
    let g = golden();
    let enc = Arc::new(Encoder::load(&model).unwrap());
    let mut m = Model::load(&model, enc.clone(), 1).unwrap();
    let batch: Vec<_> = g["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["manifest_patch"].is_null() && !c["probabilities"].is_null())
        .map(|c| enc.encode(&question(&c["question"])).unwrap())
        .collect();
    assert!(batch.len() >= 8, "rows of different lengths and option counts");
    let together = m.run_batch(&batch).unwrap();
    for (e, d) in batch.iter().zip(&together) {
        let alone = m.run(e).unwrap();
        let dp = alone.probabilities.iter().zip(&d.probabilities).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(dp < 1e-5, "{dp:e}");
        let dc = (alone.confidence.unwrap_or(0.0) - d.confidence.unwrap_or(0.0)).abs();
        assert!(dc < 1e-5, "confidence {dc:e}");
    }
}
