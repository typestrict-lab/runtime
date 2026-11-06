//! The HTTP surface: `POST /v1/systemone`, `GET /v1/models`, `GET /healthz`, and `GET /`, a page for a browser.
//!
//! - **A pool of 1-thread sessions**, not one N-thread session: on CPU a 1-thread session costs only ~1.5x the time of a
//!   4-thread one, so N 1-thread workers answer ~2.5x more questions per CPU (measured in the typestrict lab). Each
//!   question of a request takes the next free session on its own.
//! - **The order is the client's**: questions are answered in the order sent, options read in the order sent.
//! - **Nothing added to the body**: the time from dispatch to the last answer goes in the
//!   `x-typestrict-latency-ms` header.
//! - **`TYPESTRICT_TOKEN`**: when set, every request but `/healthz` needs `Authorization: Bearer <token>`, compared in
//!   constant time.
//!
//! No database, no network calls, no telemetry: requests and data stay in the process.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Request as HttpRequest, State};
use axum::http::header::{AUTHORIZATION, CONTENT_TYPE, WWW_AUTHENTICATE};
use axum::http::{HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::sync::{Mutex, mpsc};

use crate::contract::{self, ModelMetadata, Ordered, Request, Usage, ValidationError};
use crate::reader::{Encoder, Model, Unanswerable};

/// Requests larger than this are refused with 413. The model reads at most `max_length` tokens of a state, so this only
/// bounds the JSON a client may send.
pub const BODY_LIMIT: usize = 16 * 1024 * 1024;

/// Header with the time the model took for the request, in milliseconds.
pub const LATENCY_HEADER: &str = "x-typestrict-latency-ms";

pub struct App {
    pub enc: Arc<Encoder>,
    pub meta: ModelMetadata,
    token: Option<String>,
    /// Free sessions: taking one is waiting for the next to come back.
    free: Mutex<mpsc::UnboundedReceiver<Model>>,
    back: mpsc::UnboundedSender<Model>,
    ready: AtomicBool,
}

impl App {
    /// An app with no session yet: `/healthz` answers 503 until [`App::add_model`] gives it one.
    pub fn new(enc: Arc<Encoder>, meta: ModelMetadata, token: Option<String>) -> Arc<Self> {
        let (back, free) = mpsc::unbounded_channel();
        Arc::new(Self {
            enc,
            meta,
            token: token.filter(|t| !t.is_empty()),
            free: Mutex::new(free),
            back,
            ready: AtomicBool::new(false),
        })
    }

    pub fn add_model(&self, m: Model) {
        let _ = self.back.send(m);
        self.ready.store(true, Ordering::Release);
    }

    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

pub fn router(app: Arc<App>) -> Router {
    let api = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(systemone))
        .route_layer(middleware::from_fn_with_state(app.clone(), auth));
    Router::new()
        .route("/healthz", get(healthz))
        .route("/", get(page))
        .merge(api)
        .fallback(|| async { detail(StatusCode::NOT_FOUND, "Not Found") })
        .layer(DefaultBodyLimit::max(BODY_LIMIT))
        .with_state(app)
}

fn detail(status: StatusCode, msg: impl Into<String>) -> Response {
    (status, Json(json!({ "detail": msg.into() }))).into_response()
}

/// 422 with the contract's `HTTPValidationError` body.
fn invalid(errors: Vec<ValidationError>) -> Response {
    (StatusCode::UNPROCESSABLE_ENTITY, Json(json!({ "detail": errors }))).into_response()
}

/// Whether two byte strings are equal, in a time that does not depend on where they differ.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(app): State<Arc<App>>, req: HttpRequest, next: Next) -> Response {
    if let Some(token) = &app.token {
        let given = req
            .headers()
            .get(AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer ")))
            .map(str::trim)
            .unwrap_or("");
        if !constant_time_eq(given.as_bytes(), token.as_bytes()) {
            let mut r =
                detail(StatusCode::UNAUTHORIZED, "Not authenticated: send Authorization: Bearer <TYPESTRICT_TOKEN>");
            r.headers_mut().insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
            return r;
        }
    }
    next.run(req).await
}

async fn healthz(State(app): State<Arc<App>>) -> Response {
    if app.is_ready() {
        (StatusCode::OK, "ok").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "loading").into_response()
    }
}

async fn page(State(app): State<Arc<App>>) -> Response {
    let html = crate::page::render(&app.meta, &app.enc.manifest, app.is_ready(), app.token.is_some());
    ([(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"))], html).into_response()
}

async fn models(State(app): State<Arc<App>>) -> Json<Value> {
    Json(json!({ "models": [app.meta] }))
}

async fn systemone(State(app): State<Arc<App>>, body: Bytes) -> Response {
    // The body is JSON whatever its Content-Type says, as FastAPI reads it.
    let req: Request = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            let kind = if e.is_syntax() || e.is_eof() { "json_invalid" } else { "model_attributes_type" };
            return invalid(vec![ValidationError::new(vec![json!("body")], e.to_string(), kind)]);
        }
    };
    // The container holds one model and answers whatever name the client sends, echoing it back as Jev does: a
    // client switches from Jev by changing only its base URL.
    let req_model = req.model.clone();
    if req.questions.0.is_empty() {
        return invalid(vec![ValidationError::new(
            vec![json!("body"), json!("questions")],
            "questions needs at least 1 question",
            "too_short",
        )]);
    }
    let mut prepared = Vec::with_capacity(req.questions.0.len());
    let mut errors = Vec::new();
    for (name, q) in req.questions.0 {
        match contract::prepare(&app.enc.manifest, &req.state, &name, q) {
            Ok(p) => prepared.push(p),
            Err(e) => errors.push(e),
        }
    }
    if !errors.is_empty() {
        return invalid(errors);
    }
    // Tokens and cuts before taking a session: a question that does not fit never waits for one.
    let mut encoded = Vec::with_capacity(prepared.len());
    for p in &prepared {
        match app.enc.encode(&p.question) {
            Ok(e) => encoded.push(e),
            Err(e) => return question_error(&p.name, e),
        }
    }
    if !app.is_ready() {
        return detail(StatusCode::SERVICE_UNAVAILABLE, "the model is still loading");
    }

    // Each question waits for the next free session on its own and runs there (batch 1, as the playground runs
    // them): an idle server answers the questions of a request in parallel, a busy one queues them fairly (tokio's
    // Mutex is FIFO), and no session idles while a question waits. Measured against running them one after another
    // on one session and against batching them in one ONNX run: see "Multi-question requests" in README.md.
    let t0 = std::time::Instant::now();
    let tasks: Vec<_> = encoded
        .into_iter()
        .map(|e| {
            let app = app.clone();
            tokio::spawn(async move {
                let mut model = app.free.lock().await.recv().await.expect("the session pool is never closed");
                let (model, d) = tokio::task::spawn_blocking(move || {
                    let d = model.run(&e);
                    (model, d)
                })
                .await
                .map_err(|e| anyhow::anyhow!("inference panicked: {e}"))?;
                app.add_model(model);
                d
            })
        })
        .collect();
    let mut decisions = Vec::with_capacity(tasks.len());
    for t in tasks {
        match t.await {
            Ok(Ok(d)) => decisions.push(d),
            Ok(Err(e)) => return detail(StatusCode::INTERNAL_SERVER_ERROR, format!("inference failed: {e:#}")),
            Err(e) => return detail(StatusCode::INTERNAL_SERVER_ERROR, format!("inference panicked: {e}")),
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    let tokens = decisions.iter().map(|d| d.tokens as u64).sum();
    let answers = prepared.iter().zip(&decisions).map(|(p, d)| (p.name.clone(), contract::to_answer(p, d))).collect();
    let mut r = Json(contract::Response {
        model: req_model,
        answers: Ordered(answers),
        usage: Usage { input_tokens: tokens, output_tokens: 0 },
    })
    .into_response();
    if let Ok(v) = HeaderValue::from_str(&format!("{ms:.2}")) {
        r.headers_mut().insert(LATENCY_HEADER, v);
    }
    r
}

fn question_error(name: &str, e: anyhow::Error) -> Response {
    match e.downcast_ref::<Unanswerable>() {
        Some(u) => invalid(vec![ValidationError::new(
            vec![json!("body"), json!("questions"), json!(name)],
            u.0.clone(),
            "value_error",
        )]),
        None => detail(StatusCode::INTERNAL_SERVER_ERROR, format!("{name}: {e:#}")),
    }
}
