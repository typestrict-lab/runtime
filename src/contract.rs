//! Jev's decision API (`contract/typesafe-openapi.json` in the typestrict lab, version 0.2.0): the request as the client
//! wrote it, each question as the reader takes it, and the answer in the exact shape the contract defines.
//!
//! Two rules govern this module:
//!
//! - **Order is the client's.** `questions` and the `criteria` of a `choice` are JSON objects whose key order is the
//!   order of the answers and of the options the model reads. `serde_json`'s `Map` sorts keys (on purpose: `render()`
//!   needs it), so these two are read by [`InOrder`] and [`Criteria`], in document order.
//! - **Never add a field to the response.** A client that validates by schema breaks on one extra field; usage goes in
//!   `usage`, latency in a header.

use std::fmt;
use std::marker::PhantomData;

use serde::de::{Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Serialize, Serializer};
use serde_json::{Value, json};

use crate::manifest::{Kind, Manifest};
use crate::reader::{Decision, NOUL_TRUE_DEFAULT, Question};
use crate::render::{Content, render};

// ------------------------------------------------------------------------------------------------------ request --

/// A JSON object read in document order. A repeated key keeps the last value at the first one's place, as a Python
/// `dict` does.
#[derive(Debug)]
pub struct InOrder<V>(pub Vec<(String, V)>);

fn read_map<'de, A: MapAccess<'de>, V: Deserialize<'de>>(mut m: A) -> Result<Vec<(String, V)>, A::Error> {
    let mut v: Vec<(String, V)> = Vec::new();
    while let Some((k, x)) = m.next_entry::<String, V>()? {
        match v.iter_mut().find(|(c, _)| *c == k) {
            Some(pair) => pair.1 = x,
            None => v.push((k, x)),
        }
    }
    Ok(v)
}

impl<'de, V: Deserialize<'de>> Deserialize<'de> for InOrder<V> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Vis<V>(PhantomData<V>);
        impl<'de, V: Deserialize<'de>> Visitor<'de> for Vis<V> {
            type Value = InOrder<V>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON object")
            }
            fn visit_map<A: MapAccess<'de>>(self, m: A) -> Result<Self::Value, A::Error> {
                read_map(m).map(InOrder)
            }
        }
        d.deserialize_map(Vis(PhantomData))
    }
}

/// `criteria`: an object (noul, choice) or an ordered list (score).
#[derive(Debug)]
pub enum Criteria {
    Map(Vec<(String, Value)>),
    List(Vec<Value>),
    Other,
}

impl<'de> Deserialize<'de> for Criteria {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Vis;
        impl<'de> Visitor<'de> for Vis {
            type Value = Criteria;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object or a list of criteria")
            }
            fn visit_map<A: MapAccess<'de>>(self, m: A) -> Result<Criteria, A::Error> {
                read_map(m).map(Criteria::Map)
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut s: A) -> Result<Criteria, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = s.next_element::<Value>()? {
                    v.push(x);
                }
                Ok(Criteria::List(v))
            }
            // A string, a number or a boolean: the question explains what is wrong with it.
            fn visit_str<E>(self, _: &str) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
            fn visit_bool<E>(self, _: bool) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
            fn visit_i64<E>(self, _: i64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
            fn visit_u64<E>(self, _: u64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
            fn visit_f64<E>(self, _: f64) -> Result<Criteria, E> {
                Ok(Criteria::Other)
            }
        }
        d.deserialize_any(Vis)
    }
}

#[derive(Debug, Deserialize)]
pub struct RawQuestion {
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub instructions: Option<Value>,
    #[serde(default)]
    pub criteria: Option<Criteria>,
}

/// `POST /v1/systemone`. `state` is required by the contract but read leniently: a missing state is no state.
#[derive(Debug, Deserialize)]
pub struct Request {
    pub model: String,
    #[serde(default)]
    pub state: Option<Value>,
    pub questions: InOrder<RawQuestion>,
}

/// One entry of a 422's `detail`, shaped as the contract's `ValidationError`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ValidationError {
    pub loc: Vec<Value>,
    pub msg: String,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ValidationError {
    pub fn new(loc: Vec<Value>, msg: impl Into<String>, kind: &str) -> Self {
        Self { loc, msg: msg.into(), kind: kind.into() }
    }

    fn question(name: &str, field: Option<&str>, msg: impl Into<String>) -> Self {
        let mut loc = vec![json!("body"), json!("questions"), json!(name)];
        loc.extend(field.map(|f| json!(f)));
        Self::new(loc, msg, "value_error")
    }
}

/// Python's `x or default`: null, false, zero, an empty string and an empty collection count as absent.
pub fn filled(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// A question of the request, ready for the reader, with what its answer needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Prepared {
    pub name: String,
    pub question: Question,
    /// `choice`: the answer's keys, in the client's order.
    pub labels: Vec<String>,
    /// `score`: the criteria as the client sent them, for the legend.
    pub levels: Vec<Content>,
}

/// Turns a question of the request into the reader's question (`montar()` of the lab's reference API, on the
/// `format.py` rules):
///
/// - `noul`: options `[false, true]`; an absent or empty side falls to the reader's defaults (the manifest's
///   `noul_false_default`; the instructions, or "A afirmação se aplica.").
/// - `choice`: one option per criterion, in the client's order: its description, or its name when it has none.
/// - `score`: the criteria list, in order (the position is the score).
///
/// When the question is the one the model was trained on (see [`trained_options`]), the options are the manifest's
/// option texts, exactly as training rendered them.
pub fn prepare(m: &Manifest, state: &Option<Value>, name: &str, q: RawQuestion) -> Result<Prepared, ValidationError> {
    let kind = match q.kind.as_deref() {
        Some("noul") => Kind::Noul,
        Some("choice") => Kind::Choice,
        Some("score") => Kind::Score,
        Some(other) => {
            return Err(ValidationError::question(
                name,
                Some("type"),
                format!("unknown question type {other:?}: use noul, choice or score"),
            ));
        }
        None => {
            return Err(ValidationError::question(
                name,
                Some("type"),
                "a question needs a type: noul, choice or score",
            ));
        }
    };
    let present = |v: &Value| Some(v.clone()).filter(filled);
    let (options, labels, levels): (Vec<Option<Content>>, Vec<String>, Vec<Content>) = match (kind, q.criteria) {
        (Kind::Noul, None) => (vec![None, None], vec![], vec![]),
        (Kind::Noul, Some(Criteria::Map(c))) => {
            let side = |k: &str| c.iter().find(|(key, _)| key == k).and_then(|(_, v)| present(v));
            (vec![side("false"), side("true")], vec![], vec![])
        }
        (Kind::Noul, Some(_)) => {
            return Err(ValidationError::question(
                name,
                Some("criteria"),
                "noul criteria are an object with true and false",
            ));
        }
        (Kind::Choice, Some(Criteria::Map(c))) if c.len() >= 2 => {
            let options = c.iter().map(|(label, d)| Some(present(d).unwrap_or_else(|| json!(label)))).collect();
            (options, c.into_iter().map(|(k, _)| k).collect(), vec![])
        }
        (Kind::Choice, _) => {
            return Err(ValidationError::question(
                name,
                Some("criteria"),
                "choice needs an object with at least 2 named criteria",
            ));
        }
        (Kind::Score, Some(Criteria::List(l))) if l.len() >= 2 => (l.iter().cloned().map(Some).collect(), vec![], l),
        (Kind::Score, _) => {
            return Err(ValidationError::question(
                name,
                Some("criteria"),
                "score needs an ordered list of at least 2 criteria",
            ));
        }
    };
    let mut options = trained_options(m, kind, q.instructions.as_ref(), &labels, &options)
        .map(|t| t.into_iter().map(|s| Some(json!(s))).collect())
        .unwrap_or(options);
    // `true or (instructions or "A afirmação se aplica.")`: empty instructions are no instructions, as in training.
    if kind == Kind::Noul && options[1].is_none() && !q.instructions.as_ref().is_some_and(filled) {
        options[1] = Some(json!(NOUL_TRUE_DEFAULT));
    }
    Ok(Prepared {
        name: name.to_string(),
        question: Question { kind, instructions: q.instructions, state: state.clone(), options },
        labels,
        levels,
    })
}

/// The option texts the model was trained on, when the client asks that question with no other wording:
///
/// - a single-task model (`task` in the manifest): same type and the same labels in the same order (`choice`: the
///   criteria names; `score`: the criteria themselves), each description absent or the trained one. This is what the
///   playground does when the visitor keeps the trained labels.
/// - a multi-task model (`tasks`): a `noul` whose instructions are a listed question, each side absent or the trained
///   one.
fn trained_options(
    m: &Manifest,
    kind: Kind,
    instructions: Option<&Value>,
    labels: &[String],
    options: &[Option<Content>],
) -> Option<Vec<String>> {
    let same = |given: &Option<Content>, trained: &str, label: Option<&str>| match given {
        None => true,
        Some(v) => {
            let text = render(Some(v));
            text == trained || label.is_some_and(|l| l == text || format!("{l}: {text}") == trained)
        }
    };
    if let Some(t) = &m.task {
        let n = options.len();
        if t.kind == kind && t.options.len() == n && t.labels.len() == n {
            let matches = match kind {
                Kind::Choice => {
                    labels == t.labels.as_slice()
                        && options.iter().zip(&t.options).zip(labels).all(|((o, tr), l)| same(o, tr, Some(l)))
                }
                Kind::Score => options.iter().zip(&t.options).zip(&t.labels).all(|((o, tr), l)| {
                    o.as_ref().is_some_and(|v| {
                        let text = render(Some(v));
                        text == *l || text == *tr
                    })
                }),
                Kind::Noul => false,
            };
            if matches {
                return Some(t.options.clone());
            }
        }
    }
    if kind == Kind::Noul {
        let asked = render(instructions);
        for t in &m.tasks {
            if let (Kind::Noul, Some(q), Some(trained)) = (t.kind, &t.question, &t.options)
                && trained.len() == 2
                && *q == asked
                && options.iter().zip(trained).all(|(o, tr)| same(o, tr, None))
            {
                return Some(trained.clone());
            }
        }
    }
    None
}

// ----------------------------------------------------------------------------------------------------- response --

/// A map serialized in insertion order: `probabilities` and `legend` follow the client's order, not sorted keys.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ordered<V>(pub Vec<(String, V)>);

impl<V: Serialize> Serialize for Ordered<V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

/// An answer. The shape differs by type on purpose: in `noul` the probability is the answer, so it has no confidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
    Noul { noul: f64 },
    Choice { choice: String, confidence: f64, probabilities: Ordered<f64> },
    Score { score: f64, confidence: f64, legend: Ordered<String>, probabilities: Ordered<f64> },
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Response {
    pub model: String,
    pub answers: Ordered<Answer>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    pub release_date: String,
}

/// `format.py` `to_answer()`: the model's distribution as the answer the API returns.
///
/// - `noul`: `p[1]`, the probability of true.
/// - `choice`: the most likely label (the first on a tie), the confidence (the model's confidence head when it has
///   one, else the winning probability) and every label's probability.
/// - `score`: the expectation over the levels (1.7 out of a distribution whose mode is 2), not the argmax.
pub fn to_answer(p: &Prepared, d: &Decision) -> Answer {
    let probs = &d.probabilities;
    match p.question.kind {
        Kind::Noul => Answer::Noul { noul: probs[1] },
        Kind::Choice => Answer::Choice {
            choice: p.labels[d.best].clone(),
            confidence: d.confidence.unwrap_or(probs[d.best]),
            probabilities: Ordered(p.labels.iter().cloned().zip(probs.iter().copied()).collect()),
        },
        Kind::Score => Answer::Score {
            score: probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum(),
            confidence: d.confidence.unwrap_or_else(|| probs.iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            legend: Ordered(p.levels.iter().enumerate().map(|(i, c)| (i.to_string(), render(Some(c)))).collect()),
            probabilities: Ordered(probs.iter().enumerate().map(|(i, p)| (i.to_string(), *p)).collect()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(extra: &str) -> Manifest {
        Manifest::parse(&format!(
            r#"{{"format": "decision-ai/option-reader@1", "name": "m", "max_length": 512,
                "files": {{"model": "m.onnx", "tokenizer": "t.json"}}, "noul_false_default": "Não se aplica."{extra}}}"#
        ))
        .unwrap()
    }

    fn raw(text: &str) -> RawQuestion {
        serde_json::from_str(text).unwrap()
    }

    fn prep(m: &Manifest, text: &str) -> Result<Prepared, ValidationError> {
        prepare(m, &None, "q", raw(text))
    }

    fn opts(p: &Prepared) -> Vec<Option<Value>> {
        p.question.options.clone()
    }

    #[test]
    fn questions_keep_the_document_order() {
        let r: Request = serde_json::from_str(
            r#"{"model":"m","state":{},"questions":{"zeta":{"type":"noul"},"alfa":{"type":"noul"},"meio":{"type":"noul"},"alfa":{"type":"score"}}}"#,
        )
        .unwrap();
        let keys: Vec<_> = r.questions.0.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["zeta", "alfa", "meio"]);
        assert_eq!(r.questions.0[1].1.kind.as_deref(), Some("score"), "a repeated key keeps the last value");
    }

    /// From the raw text: `json!` would build an already sorted map before the deserializer saw the order.
    #[test]
    fn choice_labels_keep_the_document_order_and_a_bare_name_is_its_own_description() {
        let p = prep(&manifest(""), r#"{"type": "choice", "criteria": {"z": "last", "a": null, "m": ""}}"#).unwrap();
        assert_eq!(p.labels, ["z", "a", "m"]);
        assert_eq!(opts(&p), [Some(json!("last")), Some(json!("a")), Some(json!("m"))]);
    }

    #[test]
    fn noul_is_false_then_true_and_empty_sides_fall_to_the_defaults() {
        let m = manifest("");
        let p = prep(&m, r#"{"type": "noul", "criteria": {"true": "goes", "false": ""}}"#).unwrap();
        assert_eq!(opts(&p), [None, Some(json!("goes"))]);
        let p = prep(&m, r#"{"type": "noul", "instructions": "Is it spam?", "criteria": null}"#).unwrap();
        assert_eq!(opts(&p), [None, None]);
        assert_eq!(p.question.instructions, Some(json!("Is it spam?")));
        let p = prep(&m, r#"{"type": "noul", "instructions": ""}"#).unwrap();
        assert_eq!(opts(&p), [None, Some(json!(NOUL_TRUE_DEFAULT))]);
    }

    #[test]
    fn contract_refusals() {
        let m = manifest("");
        let err = |t: &str| prep(&m, t).unwrap_err();
        assert_eq!(
            err(r#"{"type": "choice", "criteria": {"only": "one"}}"#).loc,
            [json!("body"), json!("questions"), json!("q"), json!("criteria")]
        );
        err(r#"{"type": "choice", "criteria": ["a", "b"]}"#);
        err(r#"{"type": "choice"}"#);
        err(r#"{"type": "score", "criteria": ["one"]}"#);
        err(r#"{"type": "score", "criteria": {"a": "b", "c": "d"}}"#);
        err(r#"{"type": "noul", "criteria": ["no", "yes"]}"#);
        err(r#"{"type": "noul", "criteria": "yes"}"#);
        assert_eq!(err(r#"{"type": "other"}"#).loc.last(), Some(&json!("type")));
        assert_eq!(err(r#"{"instructions": "x"}"#).loc.last(), Some(&json!("type")));
    }

    #[test]
    fn the_trained_option_texts_replace_the_labels_of_the_trained_task() {
        let m = manifest(
            r#", "task": {"kind": "choice", "question": "Which team?", "labels": ["billing", "support"],
                 "options": ["billing: Payments and invoices", "support: Everything else"]}"#,
        );
        let trained = [Some(json!("billing: Payments and invoices")), Some(json!("support: Everything else"))];
        let p = prep(&m, r#"{"type": "choice", "criteria": {"billing": null, "support": null}}"#).unwrap();
        assert_eq!(opts(&p), trained);
        let p = prep(&m, r#"{"type": "choice", "criteria": {"billing": "Payments and invoices", "support": "support: Everything else"}}"#)
            .unwrap();
        assert_eq!(opts(&p), trained);
        // Another description, another order or another label set: the client's wording.
        let p = prep(&m, r#"{"type": "choice", "criteria": {"billing": "Money", "support": null}}"#).unwrap();
        assert_eq!(opts(&p), [Some(json!("Money")), Some(json!("support"))]);
        let p = prep(&m, r#"{"type": "choice", "criteria": {"support": null, "billing": null}}"#).unwrap();
        assert_eq!(opts(&p), [Some(json!("support")), Some(json!("billing"))]);
        let p = prep(&m, r#"{"type": "score", "criteria": ["billing", "support"]}"#).unwrap();
        assert_eq!(opts(&p), [Some(json!("billing")), Some(json!("support"))], "another type");
    }

    #[test]
    fn a_listed_yes_no_question_reads_its_trained_sides() {
        let m = manifest(
            r#", "tasks": [{"name": "vul", "kind": "noul", "question": "Is it vulnerable?", "options": ["No, safe.", "Yes, vulnerable."]},
                          {"name": "free", "kind": "choice", "question": "Which?", "options": null}]"#,
        );
        let trained = [Some(json!("No, safe.")), Some(json!("Yes, vulnerable."))];
        assert_eq!(opts(&prep(&m, r#"{"type": "noul", "instructions": " Is it vulnerable? "}"#).unwrap()), trained);
        assert_eq!(
            opts(
                &prep(
                    &m,
                    r#"{"type": "noul", "instructions": "Is it vulnerable?", "criteria": {"true": "Yes, vulnerable."}}"#
                )
                .unwrap()
            ),
            trained
        );
        assert_eq!(
            opts(
                &prep(&m, r#"{"type": "noul", "instructions": "Is it vulnerable?", "criteria": {"true": "Yes."}}"#)
                    .unwrap()
            ),
            [None, Some(json!("Yes."))]
        );
        assert_eq!(opts(&prep(&m, r#"{"type": "noul", "instructions": "Is it broken?"}"#).unwrap()), [None, None]);
    }

    fn decision(p: &[f64], confidence: Option<f64>) -> Decision {
        Decision { probabilities: p.to_vec(), best: crate::reader::argmax(p), confidence, tokens: 1, logits: vec![] }
    }

    #[test]
    fn answer_shapes_follow_the_contract() {
        let m = manifest("");
        let noul = prep(&m, r#"{"type": "noul"}"#).unwrap();
        assert_eq!(
            serde_json::to_value(to_answer(&noul, &decision(&[0.2, 0.8], Some(0.9)))).unwrap(),
            json!({"type": "noul", "noul": 0.8})
        );

        let choice = prep(&m, r#"{"type": "choice", "criteria": {"z": "a", "a": "b"}}"#).unwrap();
        let a = serde_json::to_string(&to_answer(&choice, &decision(&[0.5, 0.5], None))).unwrap();
        assert_eq!(a, r#"{"type":"choice","choice":"z","confidence":0.5,"probabilities":{"z":0.5,"a":0.5}}"#);

        let score = prep(&m, r#"{"type": "score", "criteria": [" low ", {"b": 1, "a": 2}, "high"]}"#).unwrap();
        let a = serde_json::to_value(to_answer(&score, &decision(&[0.1, 0.1, 0.8], Some(0.7)))).unwrap();
        assert_eq!(a["legend"], json!({"0": "low", "1": "{\"a\":2,\"b\":1}", "2": "high"}));
        assert!((a["score"].as_f64().unwrap() - 1.7).abs() < 1e-12, "the expectation, not the argmax");
        assert_eq!(a["confidence"], json!(0.7));
        let keys: Vec<_> = a.as_object().unwrap().keys().cloned().collect();
        assert_eq!(keys.len(), 5, "type, score, confidence, legend, probabilities and nothing else: {keys:?}");
    }
}
