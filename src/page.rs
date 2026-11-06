//! `GET /`: a page for whoever opens the container in a browser. The typestrict.ai wordmark and mosaic, the model inside,
//! the questions it was trained on and a `curl` that asks the first of them.
//!
//! - **Self-contained**: the SVGs are in the binary (`page/`, built by the brandkit's `repos/build.py`), the CSS inline,
//!   the system fonts. The page loads nothing from anywhere.
//! - **The questions only without a token**: with `TYPESTRICT_TOKEN` set the API is private, and so are the questions
//!   (they say what the model decides); the page then shows the model's name and how to send the token.

use std::fmt::Write;

use crate::contract::ModelMetadata;
use crate::manifest::{Kind, Manifest};

const WORDMARK: &str = include_str!("page/wordmark.svg");
const MOSAIC: &str = include_str!("page/mosaic.svg");

/// One question the model was trained on, as the page lists it and the example asks it.
struct Trained {
    name: String,
    kind: Kind,
    question: Option<String>,
    /// `(name, description)` of each criterion: the name before a `: ` in the trained option text, when it has one.
    criteria: Vec<(Option<String>, String)>,
}

fn trained(m: &Manifest, model: &str) -> Vec<Trained> {
    if !m.tasks.is_empty() {
        return m
            .tasks
            .iter()
            .map(|t| Trained {
                name: t.name.clone(),
                kind: t.kind,
                question: t.question.clone(),
                criteria: t.options.iter().flatten().map(|o| split(o, None)).collect(),
            })
            .collect();
    }
    let Some(t) = &m.task else { return vec![] };
    let criteria = t.options.iter().enumerate().map(|(i, o)| split(o, t.labels.get(i).map(String::as_str))).collect();
    vec![Trained { name: model.to_string(), kind: t.kind, question: t.question.clone(), criteria }]
}

/// "Trivial: a few lines" as `(Some("Trivial"), "a few lines")`, the way training renders a named criterion.
fn split(option: &str, label: Option<&str>) -> (Option<String>, String) {
    if let Some(l) = label {
        let rest = option.strip_prefix(&format!("{l}: ")).unwrap_or(option);
        return (Some(l.to_string()), rest.to_string());
    }
    match option.split_once(": ") {
        Some((name, rest)) if !name.is_empty() && name.chars().count() <= 32 => (Some(name.into()), rest.into()),
        _ => (None, option.into()),
    }
}

/// The request body of the example: the first trained question, keys in the order a person reads them.
fn example(model: &str, t: &Trained) -> String {
    let s = |v: &str| serde_json::to_string(v).unwrap();
    let mut q = format!("{{\n      \"type\": {}", s(t.kind.as_str()));
    if let Some(text) = &t.question {
        let _ = write!(q, ",\n      \"instructions\": {}", s(text));
    }
    match t.kind {
        Kind::Noul => {}
        Kind::Choice => {
            let items: Vec<String> = t
                .criteria
                .iter()
                .enumerate()
                .map(|(i, (name, desc))| {
                    format!("        {}: {}", s(&name.clone().unwrap_or_else(|| format!("option_{}", i + 1))), s(desc))
                })
                .collect();
            let _ = write!(q, ",\n      \"criteria\": {{\n{}\n      }}", items.join(",\n"));
        }
        Kind::Score => {
            let items: Vec<String> = t
                .criteria
                .iter()
                .map(|(name, desc)| {
                    format!("        {}", s(&name.as_ref().map_or(desc.clone(), |n| format!("{n}: {desc}"))))
                })
                .collect();
            let _ = write!(q, ",\n      \"criteria\": [\n{}\n      ]", items.join(",\n"));
        }
    }
    format!(
        "{{\n  \"model\": {},\n  \"state\": \"The text the model reads.\",\n  \"questions\": {{\n    {}: {q}\n    }}\n  }}\n}}",
        s(model),
        s(&t.name)
    )
}

fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// The page. `ready` is whether a session is loaded; `token` whether the API needs one.
pub fn render(meta: &ModelMetadata, m: &Manifest, ready: bool, token: bool) -> String {
    let tasks = trained(m, &meta.name);
    let mut body = String::new();
    let status = if ready { ("ready", "Ready") } else { ("loading", "Loading the model") };
    let _ = write!(
        body,
        r#"<section class="model"><p class="eyebrow">The model inside</p><h1>{}</h1><p class="lead">{}</p>
<p class="facts"><span class="dot {}"></span>{} · released {} · runtime {}</p></section>"#,
        esc(&meta.name),
        esc(&meta.description),
        status.0,
        status.1,
        esc(&meta.release_date),
        env!("CARGO_PKG_VERSION")
    );
    let auth = if token { " \\\n  -H \"Authorization: Bearer $TYPESTRICT_TOKEN\"" } else { "" };
    if token {
        body.push_str(
            r#"<section><h2>Private</h2><p>This container answers only with <code>Authorization: Bearer &lt;TYPESTRICT_TOKEN&gt;</code>, so it keeps the model's questions to itself. <code>GET /v1/models</code> with the token names the model.</p></section>"#,
        );
    } else if !tasks.is_empty() {
        body.push_str(r#"<section><h2>What it decides</h2><ol class="tasks">"#);
        for t in &tasks {
            let _ = write!(
                body,
                r#"<li><div class="task-head"><code>{}</code><span class="kind">{}</span></div>"#,
                esc(&t.name),
                t.kind.as_str()
            );
            if let Some(q) = &t.question {
                let _ = write!(body, "<p>{}</p>", esc(q));
            }
            if !t.criteria.is_empty() {
                body.push_str("<ul>");
                for (name, desc) in &t.criteria {
                    match name {
                        Some(n) => {
                            let _ = write!(body, "<li><b>{}</b> {}</li>", esc(n), esc(desc));
                        }
                        None => {
                            let _ = write!(body, "<li>{}</li>", esc(desc));
                        }
                    }
                }
                body.push_str("</ul>");
            }
            body.push_str("</li>");
        }
        body.push_str("</ol><p class=\"note\">Any question is answered; these are the ones it was trained on, where it is at its best.</p></section>");
    }
    let request = match tasks.first().filter(|_| !token) {
        Some(t) => example(&meta.name, t),
        None => format!(
            "{{\n  \"model\": {},\n  \"state\": \"The text the model reads.\",\n  \"questions\": {{\n    \"urgent\": {{ \"type\": \"noul\", \"instructions\": \"Is this urgent?\" }}\n  }}\n}}",
            serde_json::to_string(&meta.name).unwrap()
        ),
    };
    let _ = write!(
        body,
        r#"<section><h2>Ask it</h2><p>The same request and answer as Jev's <code>POST /v1/systemone</code>: switching is changing the base URL.</p>
<pre><code>curl http://localhost:8080/v1/systemone \
  -H "Content-Type: application/json"{} \
  -d '{}'</code></pre>
<table><tr><td><code>POST /v1/systemone</code></td><td>Answers named questions about a <code>state</code>.</td></tr>
<tr><td><code>GET /v1/models</code></td><td>The model inside, by name.</td></tr>
<tr><td><code>GET /healthz</code></td><td><code>ok</code> once the model is loaded.</td></tr></table></section>"#,
        auth,
        esc(&request.replace('\'', "'\\''"))
    );
    format!(
        r#"<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} · typestrict-runtime</title><style>{CSS}</style></head><body>
<header><div class="mosaic">{MOSAIC}</div><div class="bar"><a class="wordmark" href="https://typestrict.ai">{WORDMARK}</a><span>runtime</span></div></header>
<main>{body}</main>
<footer>No network calls and no telemetry: requests and data stay in this container. <a href="https://github.com/typestrict-lab/runtime">typestrict-runtime</a>, Apache-2.0. Trained on <a href="https://typestrict.ai">typestrict.ai</a>.</footer>
</body></html>"#,
        name = esc(&meta.name)
    )
}

const CSS: &str = r#"
:root{--ink:#1A1806;--yellow:#FFD400;--bg:#FFFDF2;--fg:#1A1806;--muted:#1A18068C;--line:#1A18061F;--card:#FFF8D6;--code:#1A1806;--codefg:#FFE766}
@media (prefers-color-scheme:dark){:root{--bg:#11100A;--fg:#F4F1E2;--muted:#F4F1E28C;--line:#F4F1E21F;--card:#1A1806;--code:#000000}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{background:var(--yellow);padding:24px 16px 20px}
.mosaic,.bar,main,footer{max-width:880px;margin:0 auto}
.mosaic svg{display:block;width:100%;height:72px}
.bar{display:flex;align-items:flex-end;gap:12px;margin-top:20px}
.wordmark svg{display:block;height:40px;width:auto}.wordmark .w{fill:var(--ink)}.wordmark .ai{fill:#1A18068C}
.bar span{color:var(--ink);font-weight:700;font-size:20px;line-height:1;padding-bottom:3px}
main{padding:8px 16px 32px}
section{padding:28px 0;border-bottom:1px solid var(--line)}
.eyebrow{margin:0;color:var(--muted);font-size:13px;text-transform:uppercase;letter-spacing:.06em}
h1{margin:4px 0 6px;font-size:32px;line-height:1.15;letter-spacing:-.02em;overflow-wrap:anywhere}
h2{margin:0 0 12px;font-size:20px;letter-spacing:-.01em}
.lead{margin:0;color:var(--muted)}
.facts{margin:12px 0 0;font-size:14px;color:var(--muted);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.dot{width:10px;height:10px;border-radius:50%;background:#C9A400}.dot.ready{background:#2FA84F}
.tasks{list-style:none;margin:0;padding:0;display:grid;gap:12px}
.tasks>li{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.task-head{display:flex;justify-content:space-between;gap:12px;align-items:center}
.kind{font-size:12px;font-weight:700;background:var(--yellow);color:var(--ink);border-radius:6px;padding:2px 8px}
.tasks p{margin:8px 0 0}.tasks ul{margin:8px 0 0;padding-left:18px;color:var(--muted);font-size:14px}
.note{color:var(--muted);font-size:14px;margin:12px 0 0}
code{font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;overflow-wrap:anywhere}
pre{background:var(--code);color:var(--codefg);border-radius:12px;padding:16px;overflow-x:auto;margin:12px 0 16px}
pre code{overflow-wrap:normal;white-space:pre}
table{border-collapse:collapse;width:100%;font-size:14px}td{padding:8px 0;border-top:1px solid var(--line);vertical-align:top}td:first-child{width:1%;padding-right:24px;white-space:nowrap}
a{color:inherit}
footer{padding:24px 16px 40px;color:var(--muted);font-size:13px}
@media (max-width:560px){h1{font-size:26px}.wordmark svg{height:30px}td{display:block;width:auto}td:first-child{width:auto;padding:10px 0 0}td+td{border-top:0;padding-top:2px}}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(tasks: &str) -> Manifest {
        Manifest::parse(&format!(
            r#"{{"format":"decision-ai/option-reader@1","name":"m","files":{{"model":"onnx/model_8bit.onnx","tokenizer":"tokenizer.json"}},
            "max_length":512,"tokens":{{"cls":"[CLS]","sep":"[SEP]","pad":"[PAD]","option":"[OPT]"}},
            "question_types":{{"noul":0,"choice":1,"score":2}},"tasks":{tasks}}}"#
        ))
        .unwrap()
    }

    fn meta() -> ModelMetadata {
        ModelMetadata { name: "pr<guard>".into(), description: "Reviews PRs".into(), release_date: "2026-11-06".into() }
    }

    const TASKS: &str = r#"[{"name":"size","kind":"choice","question":"How big?","options":["Trivial: a few lines","Large: many files"]},
        {"name":"breaks","kind":"noul","question":"Does it break users?","options":["No.","Yes."]}]"#;

    #[test]
    fn the_page_lists_the_trained_questions_and_asks_the_first() {
        let page = render(&meta(), &manifest(TASKS), true, false);
        assert!(page.contains("pr&lt;guard&gt;") && !page.contains("pr<guard>"), "names are escaped");
        assert!(page.contains("How big?") && page.contains("Does it break users?"));
        assert!(page.contains("<b>Trivial</b> a few lines"));
        // The example is a valid request whose choice keeps the trained names, in order.
        let start = page.find("-d '").unwrap() + 4;
        let json = page[start..page[start..].find("'</code>").unwrap() + start].replace("&quot;", "\"");
        let req: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(req["questions"]["size"]["criteria"]["Trivial"], "a few lines");
        assert_eq!(req["questions"]["size"]["type"], "choice");
        assert!(json.find("Trivial").unwrap() < json.find("Large").unwrap());
        assert!(page.contains("dot ready"));
    }

    #[test]
    fn with_a_token_the_questions_stay_private() {
        let page = render(&meta(), &manifest(TASKS), false, true);
        assert!(!page.contains("How big?") && !page.contains("Trivial"));
        assert!(page.contains("Authorization: Bearer $TYPESTRICT_TOKEN"));
        assert!(page.contains("Loading the model"));
    }

    #[test]
    fn the_page_loads_nothing_from_outside() {
        let page = render(&meta(), &manifest(TASKS), true, false);
        assert!(!page.contains("<script") && !page.contains("<link") && !page.contains("src=\"http"));
    }
}
