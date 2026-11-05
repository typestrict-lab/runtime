//! `decision_ai.json`, the manifest typestrict.ai ships with every model, for the option reader (the Dinah protocol).
//!
//! Two formats, as the `decision_ai` package reads them:
//!
//! - `decision-ai/option-reader@1` (what typestrict.ai writes today): every setting at the top level. Read exactly as the
//!   playground's `reader.ts` reads it.
//! - `decision-ai/model@1`: `files`, `max_length` and `calibration` at the top level, the reader's settings under
//!   `reader_config`, with `"reader": "option-reader"`. Missing settings take the `decision_ai` package's defaults.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};

pub const OPTION_READER_FORMAT: &str = "decision-ai/option-reader@1";
pub const MODEL_FORMAT: &str = "decision-ai/model@1";

/// The three question types of the API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Noul,
    Choice,
    Score,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Noul => "noul",
            Kind::Choice => "choice",
            Kind::Score => "score",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tokens {
    pub cls: String,
    pub sep: String,
    pub pad: String,
    pub option: String,
}

/// The task a single-task model was trained on, with the option texts exactly as training rendered them.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Task {
    pub kind: Kind,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    #[serde(default)]
    pub options: Vec<String>,
}

/// One question of a multi-task model; `options` is null when every example brought its own.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ListedTask {
    #[serde(default)]
    pub name: String,
    pub kind: Kind,
    #[serde(default)]
    pub question: Option<String>,
    #[serde(default)]
    pub options: Option<Vec<String>>,
}

/// Temperatures before the softmax.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Calibration {
    /// (question type index) x (option count 2..9), as the trainer fits it (`calibrate.py`); entries may be null.
    pub table: Option<Vec<Vec<Option<f64>>>>,
    /// `decision-ai/model@1`: one temperature for every question...
    pub scalar: Option<f64>,
    /// ...or one per option count.
    pub per_option_count: BTreeMap<usize, f64>,
    /// Platt scaling of the confidence logit, `[a, b]`; identity when absent.
    pub platt: Option<(f64, f64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub format: String,
    pub name: String,
    pub description: Option<String>,
    pub release_date: Option<String>,
    pub license: Option<String>,
    pub model_file: String,
    pub tokenizer_file: String,
    /// The context the reader fills: `truncation.max_length` when the manifest has it, else `max_length`.
    pub max_length: usize,
    /// Each option is cut to this many tokens (`truncation.max_criterion_tokens`).
    pub max_criterion_tokens: Option<usize>,
    pub pad_multiple: usize,
    pub tokens: Tokens,
    pub question_types: BTreeMap<String, i64>,
    pub noul_false_default: String,
    pub inputs: BTreeMap<String, String>,
    pub outputs: BTreeMap<String, String>,
    pub calibration: Calibration,
    pub task: Option<Task>,
    pub tasks: Vec<ListedTask>,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("decision_ai.json");
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("in {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let json: Map<String, Value> = serde_json::from_str(text).context("decision_ai.json is not a JSON object")?;
        let format = str_at(&json, "format").unwrap_or(OPTION_READER_FORMAT).to_string();
        // Settings: the top level (option-reader@1) or reader_config over it (model@1).
        let (cfg, legacy) = match format.as_str() {
            OPTION_READER_FORMAT => (json.clone(), true),
            MODEL_FORMAT => {
                let reader = str_at(&json, "reader").unwrap_or("");
                ensure!(reader == "option-reader", "reader {reader:?} is not supported (only option-reader)");
                let mut cfg = json.clone();
                if let Some(Value::Object(rc)) = json.get("reader_config") {
                    cfg.extend(rc.clone());
                }
                (cfg, false)
            }
            other => bail!("unsupported manifest format {other:?} (expected {OPTION_READER_FORMAT} or {MODEL_FORMAT})"),
        };

        let files = json.get("files").and_then(Value::as_object).ok_or_else(|| anyhow!("no `files`"))?;
        let file = |role: &str| {
            files.get(role).and_then(Value::as_str).map(str::to_string).ok_or_else(|| anyhow!("no `files.{role}`"))
        };
        let usize_at = |m: &Map<String, Value>, k: &str| m.get(k).and_then(Value::as_u64).map(|v| v as usize);
        let truncation = cfg.get("truncation").and_then(Value::as_object);
        let max_length = truncation
            .and_then(|t| usize_at(t, "max_length"))
            .or_else(|| usize_at(&json, "max_length"))
            .ok_or_else(|| anyhow!("no `max_length`"))?;
        let max_criterion_tokens = truncation.and_then(|t| usize_at(t, "max_criterion_tokens"));

        let strings = |k: &str| -> BTreeMap<String, String> {
            cfg.get(k)
                .and_then(Value::as_object)
                .map(|m| m.iter().filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string()))).collect())
                .unwrap_or_default()
        };
        let (mut inputs, mut outputs) = (strings("inputs"), strings("outputs"));
        if !legacy {
            // The decision_ai package's OptionReader defaults (the playground's would be the role name itself).
            for (role, name) in [
                ("ids", "ids"),
                ("pad_mask", "pad_mask"),
                ("option_positions", "opt_positions"),
                ("option_mask", "opt_mask"),
                ("question_type", "qtype"),
            ] {
                inputs.entry(role.into()).or_insert_with(|| name.into());
            }
            outputs.entry("logits".into()).or_insert_with(|| "logits".into());
            outputs.entry("confidence_logit".into()).or_insert_with(|| "conf_logit".into());
        }

        let tok = cfg.get("tokens").and_then(Value::as_object);
        let token = |role: &str, default: &str| {
            tok.and_then(|t| t.get(role)).and_then(Value::as_str).unwrap_or(default).to_string()
        };
        let question_types = match cfg.get("question_types").and_then(Value::as_object) {
            Some(m) => m.iter().filter_map(|(k, v)| v.as_i64().map(|i| (k.clone(), i))).collect(),
            None => [("noul", 0), ("choice", 1), ("score", 2)].into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        };

        let cal = json.get("calibration").or_else(|| cfg.get("calibration")).and_then(Value::as_object);
        let calibration = match cal {
            None => Calibration::default(),
            Some(c) => Calibration {
                table: match c.get("temperature") {
                    Some(Value::Array(rows)) => Some(
                        rows.iter()
                            .map(|row| {
                                row.as_array().map(|r| r.iter().map(Value::as_f64).collect()).unwrap_or_default()
                            })
                            .collect(),
                    ),
                    _ => None,
                },
                scalar: c.get("temperature").and_then(Value::as_f64),
                per_option_count: c
                    .get("per_option_count")
                    .and_then(Value::as_object)
                    .map(|m| m.iter().filter_map(|(k, v)| Some((k.parse().ok()?, v.as_f64()?))).collect())
                    .unwrap_or_default(),
                platt: match c.get("platt").and_then(Value::as_array).map(Vec::as_slice) {
                    Some([a, b]) => Some((a.as_f64().unwrap_or(1.0), b.as_f64().unwrap_or(0.0))),
                    _ => None,
                },
            },
        };

        let task = match json.get("task") {
            Some(t) if !t.is_null() => Some(serde_json::from_value(t.clone()).context("`task`")?),
            _ => None,
        };
        let tasks = match json.get("tasks") {
            Some(t) if !t.is_null() => serde_json::from_value(t.clone()).context("`tasks`")?,
            _ => Vec::new(),
        };

        let pad_multiple = usize_at(&cfg, "pad_multiple").unwrap_or(1);
        Ok(Manifest {
            name: str_at(&json, "name").unwrap_or("unnamed model").to_string(),
            description: str_at(&json, "description").map(str::to_string),
            release_date: str_at(&json, "release_date").map(str::to_string),
            license: str_at(&json, "license").map(str::to_string),
            model_file: file("model")?,
            tokenizer_file: file("tokenizer")?,
            max_length,
            max_criterion_tokens,
            pad_multiple,
            tokens: Tokens {
                cls: token("cls", "[CLS]"),
                sep: token("sep", "[SEP]"),
                pad: token("pad", "[PAD]"),
                option: token("option", "[OPT]"),
            },
            question_types,
            noul_false_default: str_at(&cfg, "noul_false_default")
                .unwrap_or(if legacy { "A afirmação não se aplica." } else { "The statement above is false." })
                .to_string(),
            inputs,
            outputs,
            calibration,
            task,
            tasks,
            format,
        })
    }

    /// The tensor name of an input role (`ids`, `pad_mask`, `option_positions`, `option_mask`, `question_type`).
    pub fn input(&self, role: &str) -> String {
        self.inputs.get(role).cloned().unwrap_or_else(|| role.to_string())
    }

    pub fn logits_output(&self) -> String {
        self.outputs.get("logits").cloned().unwrap_or_else(|| "logits".into())
    }

    pub fn confidence_output(&self) -> String {
        self.outputs.get("confidence_logit").cloned().unwrap_or_else(|| "conf_logit".into())
    }
}

fn str_at<'a>(m: &'a Map<String, Value>, k: &str) -> Option<&'a str> {
    m.get(k).and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEGACY: &str = r#"{
      "format": "decision-ai/option-reader@1", "name": "ticket-router",
      "files": {"model": "onnx/model_8bit.onnx", "tokenizer": "tokenizer.json"},
      "max_length": 8192, "pad_multiple": 64,
      "tokens": {"cls": "<cls>", "sep": "<sep>", "pad": "<pad>", "option": "[OPT]"},
      "question_types": {"noul": 0, "choice": 1, "score": 2},
      "noul_false_default": "No.",
      "inputs": {"ids": "ids", "pad_mask": "pad_mask"},
      "outputs": {"logits": "logits", "confidence_logit": "conf_logit"},
      "calibration": {"temperature": [[1.5, null], [2.0]], "platt": [1.2, -0.3]},
      "truncation": {"max_length": 2048, "max_criterion_tokens": 160},
      "task": {"kind": "choice", "question": "Which team?", "labels": ["billing", "support"],
               "options": ["billing: Payments", "support: Everything else"]}
    }"#;

    #[test]
    fn reads_the_option_reader_format_as_the_playground_does() {
        let m = Manifest::parse(LEGACY).unwrap();
        assert_eq!(m.name, "ticket-router");
        assert_eq!(m.model_file, "onnx/model_8bit.onnx");
        assert_eq!(m.max_length, 2048, "truncation.max_length wins over max_length");
        assert_eq!(m.max_criterion_tokens, Some(160));
        assert_eq!(m.pad_multiple, 64);
        assert_eq!(m.tokens.cls, "<cls>");
        assert_eq!(m.noul_false_default, "No.");
        assert_eq!(m.input("ids"), "ids");
        assert_eq!(m.input("option_positions"), "option_positions", "the role name when the manifest has none");
        assert_eq!(m.calibration.table, Some(vec![vec![Some(1.5), None], vec![Some(2.0)]]));
        assert_eq!(m.calibration.platt, Some((1.2, -0.3)));
        let t = m.task.unwrap();
        assert_eq!((t.kind, t.labels.len()), (Kind::Choice, 2));
    }

    #[test]
    fn reads_the_model_format_with_reader_config_and_package_defaults() {
        let m = Manifest::parse(
            r#"{"format": "decision-ai/model@1", "name": "m", "reader": "option-reader",
                "files": {"model": "onnx/model.onnx", "tokenizer": "tokenizer.json"}, "max_length": 512,
                "calibration": {"temperature": 1.07, "per_option_count": {"2": 5.0}},
                "reader_config": {"pad_multiple": 8, "question_types": {"choice": 0, "noul": 1, "score": 2}}}"#,
        )
        .unwrap();
        assert_eq!((m.max_length, m.pad_multiple, m.max_criterion_tokens), (512, 8, None));
        assert_eq!(m.question_types["noul"], 1);
        assert_eq!(m.input("option_positions"), "opt_positions");
        assert_eq!(m.confidence_output(), "conf_logit");
        assert_eq!(m.noul_false_default, "The statement above is false.");
        assert_eq!(m.calibration.scalar, Some(1.07));
        assert_eq!(m.calibration.per_option_count[&2], 5.0);
        assert_eq!(m.tokens.option, "[OPT]");
    }

    #[test]
    fn refuses_other_formats_and_readers() {
        assert!(Manifest::parse(r#"{"format": "decision-ai/model@2"}"#).is_err());
        let e = Manifest::parse(r#"{"format": "decision-ai/model@1", "reader": "laya", "files": {}}"#).unwrap_err();
        assert!(e.to_string().contains("laya"));
        assert!(Manifest::parse(r#"{"format": "decision-ai/option-reader@1", "files": {"model": "x"}}"#).is_err());
    }
}
