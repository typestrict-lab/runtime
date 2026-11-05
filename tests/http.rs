//! The HTTP surface over the test model: the order of questions and options, the exact answer shapes, the 4xx
//! answers, the bearer token, `/v1/models` and `/healthz`. Skipped with a message without the test model
//! (`scripts/fetch-test-model.sh`).

mod common;

use std::sync::{Arc, OnceLock};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;
use typestrict_serve::contract::ModelMetadata;
use typestrict_serve::reader::{Encoder, Model};
use typestrict_serve::server::{App, LATENCY_HEADER, router};

const NAME: &str = "pr-guard-test";

fn meta() -> ModelMetadata {
    ModelMetadata {
        name: NAME.into(),
        description: "TS-PRGuard, for the tests".into(),
        release_date: "2026-10-29".into(),
    }
}

fn encoder(test: &str) -> Option<Arc<Encoder>> {
    static ENC: OnceLock<Option<Arc<Encoder>>> = OnceLock::new();
    let dir = common::model_dir(test)?;
    ENC.get_or_init(|| Some(Arc::new(Encoder::load(&dir).unwrap()))).clone()
}

/// One app with one loaded session, shared by the tests that need answers.
fn app(test: &str) -> Option<Arc<App>> {
    static APP: OnceLock<Arc<App>> = OnceLock::new();
    let enc = encoder(test)?;
    let dir = common::model_dir(test)?;
    Some(
        APP.get_or_init(|| {
            let app = App::new(enc.clone(), meta(), None);
            app.add_model(Model::load(&dir, enc, 1).unwrap());
            app
        })
        .clone(),
    )
}

struct Reply {
    status: StatusCode,
    headers: axum::http::HeaderMap,
    text: String,
}

impl Reply {
    fn json(&self) -> Value {
        serde_json::from_str(&self.text).unwrap_or_else(|e| panic!("{e}: {}", self.text))
    }
}

async fn call(app: &Arc<App>, method: &str, path: &str, body: Option<&str>, auth: Option<&str>) -> Reply {
    let mut req = Request::builder().method(method).uri(path);
    if let Some(a) = auth {
        req = req.header(header::AUTHORIZATION, a);
    }
    if body.is_some() {
        req = req.header(header::CONTENT_TYPE, "application/json");
    }
    let req = req.body(body.map(|b| Body::from(b.to_string())).unwrap_or_else(Body::empty)).unwrap();
    let res = router(app.clone()).oneshot(req).await.unwrap();
    let (parts, body) = res.into_parts();
    let bytes = body.collect().await.unwrap().to_bytes();
    Reply { status: parts.status, headers: parts.headers, text: String::from_utf8(bytes.to_vec()).unwrap() }
}

async fn ask(app: &Arc<App>, body: &str) -> Reply {
    call(app, "POST", "/v1/systemone", Some(body), None).await
}

const DIFF: &str = "--- a/src/auth.py\n+++ b/src/auth.py\n@@ -1,3 +1,3 @@\n def find_user(db, name):\n-    return db.execute(\"SELECT * FROM users WHERE name = ?\", (name,))\n+    return db.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n";

fn golden(case: &str) -> Value {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/golden.json");
    let g: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    g["cases"].as_array().unwrap().iter().find(|c| c["name"] == case).unwrap().clone()
}

fn close(a: &Value, b: &Value) -> bool {
    (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() < 1e-4
}

#[tokio::test]
async fn answers_keep_the_request_order_and_the_contract_shapes() {
    let Some(app) = app("answers_keep_the_request_order_and_the_contract_shapes") else { return };
    // Raw text: the order of `questions` and of the choice criteria is the document's, not sorted.
    let body = format!(
        r#"{{"model": "{NAME}", "state": {state}, "questions": {{
            "zeta": {{"type": "score", "instructions": "How risky is this change?", "criteria": ["none", "low", {{"level": "high"}}]}},
            "alfa": {{"type": "noul", "instructions": "Does this code change introduce a security vulnerability?"}},
            "meio": {{"type": "choice", "instructions": "What kind of change is this diff?",
                      "criteria": {{"refactor": "Refactoring without behavior change", "bug": "Bug fix", "feature": null}}}}
        }}}}"#,
        state = serde_json::to_string(DIFF).unwrap()
    );
    let r = ask(&app, &body).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    assert!(r.headers.get(LATENCY_HEADER).unwrap().to_str().unwrap().parse::<f64>().unwrap() > 0.0);
    let v = r.json();
    let top: Vec<_> = v.as_object().unwrap().keys().cloned().collect();
    assert_eq!(top, ["answers", "model", "usage"], "no field beyond the contract's");
    assert_eq!(v["model"], NAME);
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 100);
    assert_eq!(v["usage"]["output_tokens"], 0);

    // Key order as served: serde_json's Map sorts, so read the text.
    let order: Vec<usize> = ["\"zeta\"", "\"alfa\"", "\"meio\""].iter().map(|k| r.text.find(k).unwrap()).collect();
    assert!(order.windows(2).all(|w| w[0] < w[1]), "answers in the request's order: {}", r.text);
    let labels: Vec<usize> =
        ["\"refactor\"", "\"bug\"", "\"feature\""].iter().map(|k| r.text.rfind(k).unwrap()).collect();
    assert!(labels.windows(2).all(|w| w[0] < w[1]), "choice probabilities in the criteria's order: {}", r.text);

    let a = &v["answers"];
    let keys = |x: &Value| {
        let mut k: Vec<_> = x.as_object().unwrap().keys().cloned().collect();
        k.sort();
        k
    };
    assert_eq!(keys(&a["alfa"]), ["noul", "type"]);
    assert_eq!(keys(&a["meio"]), ["choice", "confidence", "probabilities", "type"]);
    assert_eq!(keys(&a["zeta"]), ["confidence", "legend", "probabilities", "score", "type"]);

    // noul: the trained question reads its trained sides, so it is the golden noul_task_options answer.
    let g = golden("noul_task_options");
    assert!(
        close(&a["alfa"]["noul"], &g["probabilities"][1]),
        "{} against {}",
        a["alfa"]["noul"],
        g["probabilities"][1]
    );

    let choice = &a["meio"];
    let p = choice["probabilities"].as_object().unwrap();
    assert!((p.values().map(|x| x.as_f64().unwrap()).sum::<f64>() - 1.0).abs() < 1e-9);
    let best = p.iter().max_by(|x, y| x.1.as_f64().partial_cmp(&y.1.as_f64()).unwrap()).unwrap().0;
    assert_eq!(choice["choice"], json!(best));

    let score = &a["zeta"];
    assert_eq!(score["legend"], json!({"0": "none", "1": "low", "2": "{\"level\":\"high\"}"}));
    let expected: f64 = (0..3).map(|i| i as f64 * score["probabilities"][i.to_string()].as_f64().unwrap()).sum();
    assert!((score["score"].as_f64().unwrap() - expected).abs() < 1e-9, "score is the expectation over the levels");
}

#[tokio::test]
async fn a_choice_answer_is_the_playgrounds() {
    let Some(app) = app("a_choice_answer_is_the_playgrounds") else { return };
    let g = golden("choice_5_task");
    let labels = ["trivial", "small", "medium", "large", "very_large"];
    let criteria: serde_json::Map<String, Value> = labels
        .iter()
        .zip(g["question"]["options"].as_array().unwrap())
        .map(|(l, o)| (l.to_string(), o.clone()))
        .collect();
    // serde_json's Map sorts keys: write the object by hand to keep the order.
    let criteria =
        format!("{{{}}}", labels.iter().map(|l| format!("\"{l}\": {}", criteria[*l])).collect::<Vec<_>>().join(", "));
    let body = format!(
        r#"{{"model": "{NAME}", "state": {}, "questions": {{"c": {{"type": "choice", "instructions": {}, "criteria": {criteria}}}}}}}"#,
        g["question"]["state"], g["question"]["instructions"]
    );
    let r = ask(&app, &body).await;
    assert_eq!(r.status, StatusCode::OK, "{}", r.text);
    let a = &r.json()["answers"]["c"];
    for (i, l) in labels.iter().enumerate() {
        assert!(
            close(&a["probabilities"][l], &g["probabilities"][i]),
            "{l}: {} against {}",
            a["probabilities"][l],
            g["probabilities"][i]
        );
    }
    assert_eq!(a["choice"], json!(labels[g["best"].as_u64().unwrap() as usize]));
    assert!(close(&a["confidence"], &g["confidence"]), "the confidence head, Platt-scaled");
    assert_eq!(r.json()["usage"]["input_tokens"], g["tokens"]);
}

#[tokio::test]
async fn invalid_requests_get_4xx_with_a_detail() {
    let Some(app) = app("invalid_requests_get_4xx_with_a_detail") else { return };
    let q = |questions: &str| format!(r#"{{"model": "{NAME}", "state": "x", "questions": {questions}}}"#);

    let r = ask(&app, "{not json").await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r.json()["detail"][0]["type"], "json_invalid");
    assert_eq!(r.json()["detail"][0]["loc"], json!(["body"]));

    let r = ask(&app, r#"{"state": "x", "questions": {"a": {"type": "noul"}}}"#).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY, "no model: {}", r.text);

    let r = ask(&app, &q("{}")).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r.json()["detail"][0]["loc"], json!(["body", "questions"]));

    let r = ask(&app, &q(r#"{"ok": {"type": "noul"}, "bad": {"type": "choice", "criteria": {"one": "only"}}, "worse": {"type": "maybe"}}"#)).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    let d = r.json()["detail"].clone();
    assert_eq!(d[0]["loc"], json!(["body", "questions", "bad", "criteria"]));
    assert_eq!(d[1]["loc"], json!(["body", "questions", "worse", "type"]));
    for e in d.as_array().unwrap() {
        assert!(e["msg"].is_string() && e["type"].is_string());
    }

    for bad in
        [r#"{"s": {"type": "score", "criteria": ["one"]}}"#, r#"{"n": {"type": "noul", "criteria": ["no", "yes"]}}"#]
    {
        assert_eq!(ask(&app, &q(bad)).await.status, StatusCode::UNPROCESSABLE_ENTITY, "{bad}");
    }

    // 30 options of 160 tokens (each cut there) need more than the model's 4,096.
    let long = "word ".repeat(300);
    let criteria = (0..30).map(|i| format!("\"o{i}\": \"{long}\"")).collect::<Vec<_>>().join(",");
    let r = ask(&app, &q(&format!(r#"{{"wide": {{"type": "choice", "criteria": {{{criteria}}}}}}}"#))).await;
    assert_eq!(r.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(r.json()["detail"][0]["msg"], "the options do not fit in 4096 tokens");

    assert_eq!(call(&app, "GET", "/v1/nothing", None, None).await.status, StatusCode::NOT_FOUND);
    assert_eq!(call(&app, "GET", "/v1/systemone", None, None).await.status, StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn a_multi_question_request_answers_as_single_questions_do() {
    let test = "a_multi_question_request_answers_as_single_questions_do";
    let (Some(dir), Some(enc)) = (common::model_dir(test), encoder(test)) else { return };
    // Three sessions, so the questions of one request really run in parallel.
    let app = App::new(enc.clone(), meta(), None);
    for _ in 0..3 {
        app.add_model(Model::load(&dir, enc.clone(), 1).unwrap());
    }
    let state = serde_json::to_string(DIFF).unwrap();
    // The model's five trained questions, plus a score: different lengths, option counts and types in one request.
    let tasks: Value = serde_json::from_str(
        &std::fs::read_to_string(common::model_dir("").unwrap().join("decision_ai.json")).unwrap(),
    )
    .unwrap();
    let mut questions: Vec<(String, String)> = tasks["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let q = if t["kind"] == "noul" {
                json!({"type": "noul", "instructions": t["question"]}).to_string()
            } else {
                let criteria = t["options"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(i, o)| format!("\"o{i}\": {o}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(r#"{{"type": "choice", "instructions": {}, "criteria": {{{criteria}}}}}"#, t["question"])
            };
            (t["name"].as_str().unwrap().to_string(), q)
        })
        .collect();
    questions.push(("risk".into(), r#"{"type": "score", "criteria": ["none", "some", "high"]}"#.into()));
    let body = |qs: &[(String, String)]| {
        let qs = qs.iter().map(|(k, q)| format!("\"{k}\": {q}")).collect::<Vec<_>>().join(", ");
        format!(r#"{{"model": "{NAME}", "state": {state}, "questions": {{{qs}}}}}"#)
    };
    let all = ask(&app, &body(&questions)).await;
    assert_eq!(all.status, StatusCode::OK, "{}", all.text);
    let all = all.json();
    let mut tokens = 0;
    for q in &questions {
        let one = ask(&app, &body(std::slice::from_ref(q))).await.json();
        tokens += one["usage"]["input_tokens"].as_u64().unwrap();
        let (a, b) = (&all["answers"][&q.0], &one["answers"][&q.0]);
        assert_eq!(a["type"], b["type"], "{}", q.0);
        let probs = |x: &Value| -> Vec<f64> {
            match x.get("probabilities") {
                Some(p) => p.as_object().unwrap().values().map(|v| v.as_f64().unwrap()).collect(),
                None => vec![x["noul"].as_f64().unwrap()],
            }
        };
        for (x, y) in probs(a).iter().zip(probs(b)) {
            assert!((x - y).abs() < 1e-5, "{}: {x} in the request against {y} alone", q.0);
        }
        assert_eq!(a.get("choice"), b.get("choice"), "{}", q.0);
    }
    assert_eq!(all["usage"]["input_tokens"].as_u64().unwrap(), tokens, "usage is the sum of the questions' tokens");
}

#[tokio::test]
async fn any_model_name_is_answered_and_echoed() {
    let Some(app) = app("any_model_name_is_answered_and_echoed") else { return };
    for name in ["jev-latest", "", NAME] {
        let body = format!(r#"{{"model": "{name}", "state": "x", "questions": {{"a": {{"type": "noul"}}}}}}"#);
        let r = ask(&app, &body).await;
        assert_eq!(r.status, StatusCode::OK, "{name:?}: {}", r.text);
        assert_eq!(r.json()["model"], name, "the requested name comes back");
    }
    // /v1/models still lists the model by its own name.
    assert_eq!(call(&app, "GET", "/v1/models", None, None).await.json()["models"][0]["name"], NAME);
}

#[tokio::test]
async fn the_body_is_json_whatever_the_content_type() {
    let Some(app) = app("the_body_is_json_whatever_the_content_type") else { return };
    let body = format!(
        r#"{{"model": "{NAME}", "state": "x", "questions": {{"a": {{"type": "noul", "instructions": "Is it?"}}}}}}"#
    );
    let req = Request::builder().method("POST").uri("/v1/systemone").body(Body::from(body)).unwrap();
    let res = router(app).oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}

#[tokio::test]
async fn models_lists_the_model_inside() {
    let Some(app) = app("models_lists_the_model_inside") else { return };
    let r = call(&app, "GET", "/v1/models", None, None).await;
    assert_eq!(r.status, StatusCode::OK);
    assert_eq!(
        r.json(),
        json!({"models": [{"name": NAME, "description": "TS-PRGuard, for the tests", "release_date": "2026-10-29"}]})
    );
}

#[tokio::test]
async fn healthz_says_ok_once_a_session_is_loaded() {
    let Some(dir) = common::model_dir("healthz_says_ok_once_a_session_is_loaded") else { return };
    let enc = encoder("healthz_says_ok_once_a_session_is_loaded").unwrap();
    let app = App::new(enc.clone(), meta(), None);
    let r = call(&app, "GET", "/healthz", None, None).await;
    assert_eq!((r.status, r.text.as_str()), (StatusCode::SERVICE_UNAVAILABLE, "loading"));
    let body = format!(r#"{{"model": "{NAME}", "state": "x", "questions": {{"a": {{"type": "noul"}}}}}}"#);
    assert_eq!(ask(&app, &body).await.status, StatusCode::SERVICE_UNAVAILABLE);
    app.add_model(Model::load(&dir, enc, 1).unwrap());
    let r = call(&app, "GET", "/healthz", None, None).await;
    assert_eq!((r.status, r.text.as_str()), (StatusCode::OK, "ok"));
}

#[tokio::test]
async fn a_token_guards_everything_but_healthz() {
    let Some(_) = common::model_dir("a_token_guards_everything_but_healthz") else { return };
    let enc = encoder("a_token_guards_everything_but_healthz").unwrap();
    // No session needed: the token is checked first, and /v1/models does not run the model.
    let app = App::new(enc, meta(), Some("s3cret-token".into()));
    let body = format!(r#"{{"model": "{NAME}", "state": "x", "questions": {{"a": {{"type": "noul"}}}}}}"#);
    for auth in
        [None, Some("Bearer wrong"), Some("Bearer s3cret-toke"), Some("s3cret-token"), Some("Basic s3cret-token")]
    {
        let r = call(&app, "POST", "/v1/systemone", Some(&body), auth).await;
        assert_eq!(r.status, StatusCode::UNAUTHORIZED, "{auth:?}");
        assert_eq!(r.headers.get(header::WWW_AUTHENTICATE).unwrap(), "Bearer");
        assert!(r.json()["detail"].is_string());
        assert_eq!(call(&app, "GET", "/v1/models", None, auth).await.status, StatusCode::UNAUTHORIZED, "{auth:?}");
    }
    assert_eq!(call(&app, "GET", "/v1/models", None, Some("Bearer s3cret-token")).await.status, StatusCode::OK);
    // Past the token: the model is still loading.
    let r = call(&app, "POST", "/v1/systemone", Some(&body), Some("Bearer s3cret-token")).await;
    assert_eq!(r.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(call(&app, "GET", "/healthz", None, None).await.status, StatusCode::SERVICE_UNAVAILABLE);
}

#[test]
fn tokens_compare_in_constant_time_and_correctly() {
    use typestrict_serve::server::constant_time_eq;
    assert!(constant_time_eq(b"abc", b"abc"));
    assert!(!constant_time_eq(b"abc", b"abd"));
    assert!(!constant_time_eq(b"abc", b"ab"));
}
