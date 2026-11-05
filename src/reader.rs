//! The option reader (the Dinah protocol), ported line by line from the typestrict.ai playground's `reader.ts`:
//!
//! `[CLS] instructions [SEP] state [SEP] [OPT] option_1 ... [OPT] option_n`, with the cuts of training (the trainer's
//! Collator): each option to `max_criterion_tokens`, the state to what the rest leaves of `max_length`. Padded to a
//! multiple of `pad_multiple`; one logit per option, then the temperature for (question type, option count) and a
//! softmax; the confidence logit through Platt scaling.
//!
//! [`Encoder`] does the text and token work (no ONNX, shareable across threads); [`Model`] owns one ONNX Runtime
//! session. `tests/golden.rs` checks both against vectors the playground's own code wrote.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::Tensor;
use tokenizers::Tokenizer;

use crate::manifest::{Kind, Manifest};
use crate::render::{Content, render};

/// The true option of a yes/no question with neither a true criterion nor instructions (`reader.ts` `optionsOf`).
pub const NOUL_TRUE_DEFAULT: &str = "A afirmação se aplica.";

/// A question as the reader takes it. For `noul`, `options` is `[false, true]`, either of which may be missing.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    pub kind: Kind,
    pub instructions: Option<Content>,
    pub state: Option<Content>,
    pub options: Vec<Option<Content>>,
}

/// A question the model cannot take as asked: the client's fault, answered with 422.
#[derive(Debug, Clone, PartialEq)]
pub struct Unanswerable(pub String);

impl std::fmt::Display for Unanswerable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Unanswerable {}

/// What the model is fed for one question.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// The sequence, without padding.
    pub ids: Vec<u32>,
    pub option_positions: Vec<usize>,
    pub question_type: i64,
    /// Length after padding to `pad_multiple`.
    pub padded_length: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Decision {
    pub probabilities: Vec<f64>,
    /// Index of the most likely option (the first one on a tie).
    pub best: usize,
    /// Platt-scaled confidence logit, when the model has that output.
    pub confidence: Option<f64>,
    /// Tokens read, without padding.
    pub tokens: usize,
    /// The option logits as the model returned them, before calibration.
    pub logits: Vec<f32>,
}

struct SpecialIds {
    cls: u32,
    sep: u32,
    pad: u32,
    opt: u32,
}

/// Manifest plus tokenizer: everything but the ONNX session.
pub struct Encoder {
    pub manifest: Manifest,
    tok: Tokenizer,
    ids: SpecialIds,
}

impl Encoder {
    /// Reads `decision_ai.json` and the tokenizer it names from a model directory.
    pub fn load(dir: &Path) -> Result<Self> {
        let manifest = Manifest::load(dir)?;
        let path = dir.join(&manifest.tokenizer_file);
        let tok = Tokenizer::from_file(&path).map_err(|e| anyhow!("loading {}: {e}", path.display()))?;
        Self::new(manifest, tok)
    }

    pub fn new(manifest: Manifest, tok: Tokenizer) -> Result<Self> {
        let id = |t: &str| tok.token_to_id(t).ok_or_else(|| anyhow!("special token {t} is not in the tokenizer"));
        let ids = SpecialIds {
            cls: id(&manifest.tokens.cls)?,
            sep: id(&manifest.tokens.sep)?,
            pad: id(&manifest.tokens.pad)?,
            opt: id(&manifest.tokens.option)?,
        };
        Ok(Self { manifest, tok, ids })
    }

    fn encode_text(&self, text: &str, limit: Option<usize>) -> Result<Vec<u32>> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        let mut ids = self.tok.encode(text, false).map_err(|e| anyhow!("tokenizer: {e}"))?.get_ids().to_vec();
        if let Some(l) = limit {
            ids.truncate(l);
        }
        Ok(ids)
    }

    /// The option texts of a question: a noul question is false against true (`format.py` `noul()`).
    pub fn options_of(&self, q: &Question) -> Vec<Content> {
        let given = |i: usize| q.options.get(i).cloned().flatten().filter(|v| !v.is_null());
        match q.kind {
            Kind::Noul => vec![
                given(0).unwrap_or_else(|| Content::String(self.manifest.noul_false_default.clone())),
                given(1)
                    .or_else(|| q.instructions.clone().filter(|v| !v.is_null()))
                    .unwrap_or_else(|| Content::String(NOUL_TRUE_DEFAULT.into())),
            ],
            _ => q.options.iter().map(|o| o.clone().unwrap_or(Content::Null)).collect(),
        }
    }

    /// The sequence for a question. `Err` with [`Unanswerable`] inside when the question cannot be asked of this model.
    pub fn encode(&self, q: &Question) -> Result<Encoded> {
        let m = &self.manifest;
        let options = self.options_of(q);
        if options.len() < 2 {
            return Err(Unanswerable("a question needs at least 2 options".into()).into());
        }
        let question_type = *m
            .question_types
            .get(q.kind.as_str())
            .ok_or_else(|| Unanswerable(format!("this model does not answer {} questions", q.kind.as_str())))?;
        let crit = options
            .iter()
            .map(|o| self.encode_text(&render(Some(o)), m.max_criterion_tokens))
            .collect::<Result<Vec<_>>>()?;
        let instr = self.encode_text(&render(q.instructions.as_ref()), None)?;
        let reserved = 3 + instr.len() + crit.iter().map(|c| 1 + c.len()).sum::<usize>();
        if reserved + 1 > m.max_length {
            return Err(Unanswerable(format!("the options do not fit in {} tokens", m.max_length)).into());
        }
        let state = self.encode_text(&render(q.state.as_ref()), Some(m.max_length - reserved))?;
        let mut ids = Vec::with_capacity(reserved + state.len());
        ids.push(self.ids.cls);
        ids.extend(&instr);
        ids.push(self.ids.sep);
        ids.extend(&state);
        ids.push(self.ids.sep);
        let mut option_positions = Vec::with_capacity(crit.len());
        for c in &crit {
            option_positions.push(ids.len());
            ids.push(self.ids.opt);
            ids.extend(c);
        }
        let k = m.pad_multiple.max(1);
        let padded_length = ids.len().div_ceil(k) * k;
        Ok(Encoded { ids, option_positions, question_type, padded_length })
    }
}

/// Divides the logits by the temperature for (question type, option count), as the playground does with the trainer's
/// table (`temperature[type][min(n, 9) - 2]`, ignored when missing or not positive); a `decision-ai/model@1` manifest
/// may give one temperature per option count or a single one instead.
pub fn calibrate(logits: &[f64], question_type: i64, m: &Manifest) -> Vec<f64> {
    let n = logits.len();
    let c = &m.calibration;
    let t = match &c.table {
        Some(table) => usize::try_from(question_type)
            .ok()
            .and_then(|q| table.get(q))
            .and_then(|row| row.get(n.min(9).saturating_sub(2)))
            .copied()
            .flatten(),
        None => c.per_option_count.get(&n).copied().or(c.scalar),
    };
    match t {
        Some(t) if t > 0.0 => logits.iter().map(|x| x / t).collect(),
        _ => logits.to_vec(),
    }
}

pub fn platt(z: f64, m: &Manifest) -> f64 {
    let (a, b) = m.calibration.platt.unwrap_or((1.0, 0.0));
    1.0 / (1.0 + (-(a * z + b)).exp())
}

pub fn softmax(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let s: f64 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Index of the largest value, the first one on a tie (JavaScript's `indexOf(Math.max(...))`, Python's `max(range)`).
pub fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

/// One ONNX Runtime session over a shared [`Encoder`]. A session runs one question at a time.
pub struct Model {
    pub enc: Arc<Encoder>,
    session: Session,
}

impl Model {
    /// Loads the ONNX the manifest names, with `threads` intra-op threads.
    pub fn load(dir: &Path, enc: Arc<Encoder>, threads: usize) -> Result<Self> {
        let path = dir.join(&enc.manifest.model_file);
        // The builder's errors carry the builder, which is not Sync: they become text.
        let session = Session::builder()
            .map_err(|e| anyhow!("{e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow!("{e}"))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow!("{e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("{e}"))?
            .commit_from_file(&path)
            .with_context(|| format!("loading {}", path.display()))?;
        Ok(Self { enc, session })
    }

    /// Answers one question. `Err` with [`Unanswerable`] inside when the question cannot be asked of this model.
    pub fn decide(&mut self, q: &Question) -> Result<Decision> {
        let e = self.enc.encode(q)?;
        self.run(&e)
    }

    /// Runs an encoded question.
    pub fn run(&mut self, e: &Encoded) -> Result<Decision> {
        let m = &self.enc.manifest;
        let t = e.padded_length;
        let mut ids = vec![i64::from(self.enc.ids.pad); t];
        let mut pad_mask = vec![false; t];
        for (j, &id) in e.ids.iter().enumerate() {
            ids[j] = i64::from(id);
            pad_mask[j] = true;
        }
        let n = e.option_positions.len();
        let positions: Vec<i64> = e.option_positions.iter().map(|&p| p as i64).collect();
        let (logits_name, conf_name) = (m.logits_output(), m.confidence_output());
        let out = self.session.run(ort::inputs![
            m.input("ids") => Tensor::from_array(([1, t], ids))?,
            m.input("pad_mask") => Tensor::from_array(([1, t], pad_mask))?,
            m.input("option_positions") => Tensor::from_array(([1, n], positions))?,
            m.input("option_mask") => Tensor::from_array(([1, n], vec![true; n]))?,
            m.input("question_type") => Tensor::from_array(([1], vec![e.question_type]))?,
        ])?;
        let logits: Vec<f32> = out
            .get(&logits_name)
            .ok_or_else(|| anyhow!("the model has no output {logits_name}"))?
            .try_extract_tensor::<f32>()?
            .1
            .iter()
            .take(n)
            .copied()
            .collect();
        anyhow::ensure!(logits.len() == n, "the model returned {} logits for {n} options", logits.len());
        let conf = match out.get(&conf_name) {
            Some(v) => v.try_extract_tensor::<f32>()?.1.first().map(|&z| platt(f64::from(z), m)),
            None => None,
        };
        let wide: Vec<f64> = logits.iter().map(|&l| f64::from(l)).collect();
        let probabilities = softmax(&calibrate(&wide, e.question_type, m));
        Ok(Decision { best: argmax(&probabilities), probabilities, confidence: conf, tokens: e.ids.len(), logits })
    }
}

/// How long `f` takes, in milliseconds.
pub fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let v = f();
    (v, t0.elapsed().as_secs_f64() * 1000.0)
}

#[cfg(test)]
mod tests {
    //! The playground's `reader.test.ts`, with its fake tokenizer as a word-level tokenizer: special tokens have fixed
    //! ids and every other word encodes to 10 + its length.
    use super::*;
    use serde_json::json;

    fn tokenizer() -> Tokenizer {
        let mut vocab = serde_json::Map::new();
        for (t, id) in [("[PAD]", 0), ("[CLS]", 1), ("[SEP]", 2), ("[OPT]", 3), ("[UNK]", 99)] {
            vocab.insert(t.into(), json!(id));
        }
        for w in ["a", "b", "c", "x", "y", "w", "no", "yes", "way", "pick", "one", "long", "option"] {
            vocab.insert(w.into(), json!(10 + w.len()));
        }
        let spec = json!({
            "version": "1.0",
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
        });
        Tokenizer::from_bytes(serde_json::to_vec(&spec).unwrap()).unwrap()
    }

    fn manifest(extra: serde_json::Value) -> Manifest {
        let mut base = json!({
            "format": "decision-ai/option-reader@1", "name": "m",
            "files": {"model": "onnx/model.onnx", "tokenizer": "tokenizer.json"},
            "max_length": 32, "pad_multiple": 8,
            "tokens": {"cls": "[CLS]", "sep": "[SEP]", "pad": "[PAD]", "option": "[OPT]"},
            "question_types": {"choice": 0, "noul": 1, "score": 2},
            "noul_false_default": "no",
            "inputs": {"ids": "input_ids"}, "outputs": {"logits": "logits"}
        });
        base.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        Manifest::parse(&base.to_string()).unwrap()
    }

    fn q(kind: Kind, instructions: Option<&str>, state: Option<&str>, options: &[Option<&str>]) -> Question {
        Question {
            kind,
            instructions: instructions.map(|s| json!(s)),
            state: state.map(|s| json!(s)),
            options: options.iter().map(|o| o.map(|s| json!(s))).collect(),
        }
    }

    #[test]
    fn lays_out_cls_instructions_sep_state_sep_and_options() {
        let enc = Encoder::new(manifest(json!({})), tokenizer()).unwrap();
        let e = enc.encode(&q(Kind::Choice, Some("pick one"), Some("a b c"), &[Some("yes"), Some("no way")])).unwrap();
        assert_eq!(e.ids, [1, 14, 13, 2, 11, 11, 11, 2, 3, 13, 3, 12, 13]);
        assert_eq!(e.padded_length, 16);
        assert_eq!(e.option_positions, [8, 10]);
        assert_eq!(e.question_type, 0);
    }

    #[test]
    fn applies_the_trainers_cuts() {
        let m = manifest(json!({"pad_multiple": 0, "truncation": {"max_length": 9, "max_criterion_tokens": 1}}));
        let enc = Encoder::new(m, tokenizer()).unwrap();
        let e = enc.encode(&q(Kind::Noul, None, Some("w w w w w"), &[Some("long option"), Some("x y")])).unwrap();
        // 9 tokens: [CLS] [SEP] state (cut to the 2 left) [SEP] [OPT] long [OPT] x
        assert_eq!(e.ids, [1, 2, 11, 11, 2, 3, 14, 3, 11]);
        assert_eq!(e.padded_length, 9);
        assert_eq!(e.question_type, 1);
    }

    #[test]
    fn noul_is_false_against_true_with_defaults() {
        let enc = Encoder::new(manifest(json!({})), tokenizer()).unwrap();
        let opts = |q: &Question| enc.options_of(q);
        assert_eq!(opts(&q(Kind::Noul, None, None, &[])), [json!("no"), json!(NOUL_TRUE_DEFAULT)]);
        assert_eq!(opts(&q(Kind::Noul, Some("it applies"), None, &[])), [json!("no"), json!("it applies")]);
        assert_eq!(opts(&q(Kind::Noul, Some("i"), None, &[None, Some("t")])), [json!("no"), json!("t")]);
        assert_eq!(opts(&q(Kind::Noul, None, None, &[Some("f"), Some("t")])), [json!("f"), json!("t")]);
        assert_eq!(opts(&q(Kind::Choice, None, None, &[Some("a"), Some("b")])), [json!("a"), json!("b")]);
    }

    #[test]
    fn refuses_questions_with_fewer_than_2_options_or_that_do_not_fit() {
        let enc = Encoder::new(manifest(json!({"max_length": 6})), tokenizer()).unwrap();
        assert_eq!(err_of(&enc, q(Kind::Score, None, None, &[Some("a")])), "a question needs at least 2 options");
        assert_eq!(
            err_of(&enc, q(Kind::Choice, None, None, &[Some("a b"), Some("c")])),
            "the options do not fit in 6 tokens"
        );
        let enc = Encoder::new(manifest(json!({"question_types": {"choice": 0}})), tokenizer()).unwrap();
        assert_eq!(
            err_of(&enc, q(Kind::Score, None, None, &[Some("a"), Some("b")])),
            "this model does not answer score questions"
        );
    }

    fn err_of(enc: &Encoder, q: Question) -> String {
        enc.encode(&q).unwrap_err().downcast::<Unanswerable>().unwrap().0
    }

    #[test]
    fn special_tokens_must_be_in_the_tokenizer() {
        let m = manifest(json!({"tokens": {"cls": "[CLS]", "sep": "[SEP]", "pad": "[PAD]", "option": "[NOPE]"}}));
        assert_eq!(
            Encoder::new(m, tokenizer()).err().unwrap().to_string(),
            "special token [NOPE] is not in the tokenizer"
        );
    }

    #[test]
    fn temperature_per_type_and_option_count_then_platt() {
        let m = manifest(json!({"calibration": {"temperature": [[], [2]], "platt": [2, 1]}}));
        // noul is type 1 here: logits [0, 4] over 2 -> softmax of [0, 2].
        let p = softmax(&calibrate(&[0.0, 4.0], 1, &m));
        assert!((p[1] - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
        assert!((platt(0.5, &m) - 1.0 / (1.0 + (-2.0f64).exp())).abs() < 1e-12);
        // A missing row, a missing entry and a non-positive temperature leave the logits alone.
        assert_eq!(calibrate(&[1.0, 0.0], 0, &m), [1.0, 0.0]);
        assert_eq!(calibrate(&[1.0, 0.0, 2.0], 1, &m), [1.0, 0.0, 2.0]);
        let zero = manifest(json!({"calibration": {"temperature": [[0]], "platt": null}}));
        assert_eq!(calibrate(&[1.0, 0.0], 0, &zero), [1.0, 0.0]);
        assert_eq!(platt(0.0, &zero), 0.5);
        // More than 9 options use the 9-option temperature.
        let wide = manifest(json!({"calibration": {"temperature": [[1, 1, 1, 1, 1, 1, 1, 4]]}}));
        assert_eq!(calibrate(&[4.0; 12], 0, &wide), [1.0; 12]);
    }

    #[test]
    fn argmax_takes_the_first_of_a_tie() {
        assert_eq!(argmax(&[0.2, 0.4, 0.4]), 1);
        assert_eq!(argmax(&[0.5, 0.5]), 0);
    }
}
