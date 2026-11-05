//! `typestrict-serve [--model <dir>] [--host <addr>] [--port <n>] [--workers <n>] [--threads <n>]`
//!
//! Serves the typestrict.ai model in `<dir>` (default `/model`) over HTTP. See README.md.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use typestrict_serve::contract::ModelMetadata;
use typestrict_serve::reader::{Encoder, Model};
use typestrict_serve::server::{self, App};

const USAGE: &str = "\
typestrict-serve: a typestrict.ai model on CPU, with Jev's decision API (POST /v1/systemone)

Usage: typestrict-serve [options]

Options:
  --model <dir>     model directory with decision_ai.json, tokenizer.json and the ONNX it names
                    [env TYPESTRICT_MODEL_DIR, default /model]
  --host <addr>     address to listen on [env HOST, default 0.0.0.0]
  --port <n>        port to listen on [env PORT, default 8080]
  --workers <n>     ONNX Runtime sessions, one request each at a time
                    [env TYPESTRICT_WORKERS, default: the CPU count, at most 4; 1 on AWS Lambda]
  --threads <n>     threads per session [env TYPESTRICT_THREADS, default 1; the CPU count on AWS Lambda]
  --name <name>     the model name the API answers to [env TYPESTRICT_MODEL_NAME, default: the manifest's name]
  -h, --help        this help
  -V, --version     the version

Environment:
  TYPESTRICT_TOKEN  when set, every request but GET /healthz needs Authorization: Bearer <token>
";

struct Config {
    model: PathBuf,
    host: String,
    port: u16,
    workers: usize,
    threads: usize,
    name: Option<String>,
    token: Option<String>,
}

fn env(k: &str) -> Option<String> {
    std::env::var(k).ok().filter(|v| !v.is_empty())
}

fn parse<T: std::str::FromStr>(what: &str, v: &str) -> Result<T> {
    v.parse().map_err(|_| anyhow::anyhow!("{what}: {v:?} is not valid"))
}

impl Config {
    fn read() -> Result<Option<Self>> {
        let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        // Lambda sends an execution environment one request at a time: one session with every vCPU serves it best,
        // and a second session would only add its memory (up to ~2 GB each at 4,096 tokens).
        let lambda = env("AWS_LAMBDA_FUNCTION_NAME").is_some();
        let (workers, threads) = if lambda { (1, cpus) } else { (cpus.min(4), 1) };
        let mut c = Config {
            model: env("TYPESTRICT_MODEL_DIR").unwrap_or_else(|| "/model".into()).into(),
            host: env("HOST").unwrap_or_else(|| "0.0.0.0".into()),
            port: env("PORT").map(|p| parse("PORT", &p)).transpose()?.unwrap_or(8080),
            workers: env("TYPESTRICT_WORKERS").map(|w| parse("TYPESTRICT_WORKERS", &w)).transpose()?.unwrap_or(workers),
            threads: env("TYPESTRICT_THREADS").map(|t| parse("TYPESTRICT_THREADS", &t)).transpose()?.unwrap_or(threads),
            name: env("TYPESTRICT_MODEL_NAME"),
            token: env("TYPESTRICT_TOKEN"),
        };
        let mut args = std::env::args().skip(1);
        while let Some(a) = args.next() {
            let mut value = || args.next().with_context(|| format!("{a} needs a value"));
            match a.as_str() {
                "--model" => c.model = value()?.into(),
                "--host" => c.host = value()?,
                "--port" => c.port = parse("--port", &value()?)?,
                "--workers" => c.workers = parse("--workers", &value()?)?,
                "--threads" => c.threads = parse("--threads", &value()?)?,
                "--name" => c.name = Some(value()?),
                "-h" | "--help" => {
                    print!("{USAGE}");
                    return Ok(None);
                }
                "-V" | "--version" => {
                    println!("typestrict-serve {}", env!("CARGO_PKG_VERSION"));
                    return Ok(None);
                }
                other => bail!("unknown argument {other:?} (see --help)"),
            }
        }
        c.workers = c.workers.max(1);
        c.threads = c.threads.max(1);
        Ok(Some(c))
    }
}

/// The model directory as a per-model image lays it out, or a message that says what is missing.
fn check_model_dir(dir: &Path) -> Result<()> {
    if !dir.join("decision_ai.json").is_file() {
        bail!(
            "no model in {}: expected decision_ai.json, tokenizer.json and the ONNX the manifest names \
             (onnx/model_8bit.onnx). The typestrict-serve image carries no model: build an image FROM it that copies \
             the model files to /model, mount them there (docker run -v <model dir>:/model:ro ...), or pass --model <dir>",
            dir.display()
        );
    }
    Ok(())
}

/// `YYYY-MM-DD` of a time, in UTC.
fn date(t: SystemTime) -> String {
    let days = t.duration_since(UNIX_EPOCH).map(|d| d.as_secs() / 86_400).unwrap_or(0) as i64;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Ctrl-C, or the SIGTERM a container platform sends before it stops the process.
async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
}

async fn run(c: Config) -> Result<()> {
    check_model_dir(&c.model)?;
    // The manifest and the tokenizer now (fast, and where a broken package fails); the ONNX sessions once the port is
    // open, so the process is up while /healthz still says "loading".
    let enc = Arc::new(Encoder::load(&c.model)?);
    let m = &enc.manifest;
    if !c.model.join(&m.model_file).is_file() {
        bail!("{}: the manifest names {}, which is not there", c.model.display(), m.model_file);
    }
    let meta = ModelMetadata {
        name: c.name.clone().unwrap_or_else(|| m.name.clone()),
        description: m.description.clone().unwrap_or_else(|| format!("{}, a typestrict.ai decision model", m.name)),
        release_date: m.release_date.clone().unwrap_or_else(|| {
            date(std::fs::metadata(c.model.join("decision_ai.json")).and_then(|f| f.modified()).unwrap_or(UNIX_EPOCH))
        }),
    };
    let app = App::new(enc.clone(), meta, c.token.clone());

    let addr: SocketAddr =
        format!("{}:{}", c.host, c.port).parse().with_context(|| format!("address {}:{}", c.host, c.port))?;
    let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("listening on {addr}"))?;
    eprintln!(
        "typestrict-serve {} on http://{addr}: model {:?} from {} ({} worker(s) x {} thread(s), token {})",
        env!("CARGO_PKG_VERSION"),
        app.meta.name,
        c.model.display(),
        c.workers,
        c.threads,
        if c.token.is_some() { "required" } else { "not required" }
    );

    let loader = app.clone();
    let (dir, workers, threads) = (c.model.clone(), c.workers, c.threads);
    std::thread::spawn(move || {
        for i in 0..workers {
            let t0 = Instant::now();
            match Model::load(&dir, enc.clone(), threads) {
                Ok(m) => {
                    loader.add_model(m);
                    let ready = if i == 0 { ", ready" } else { "" };
                    eprintln!(
                        "session {}/{workers} loaded in {:.0} ms{ready}",
                        i + 1,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                }
                Err(e) => {
                    eprintln!("typestrict-serve: {e:#}");
                    std::process::exit(1);
                }
            }
        }
    });

    axum::serve(listener, server::router(app)).with_graceful_shutdown(shutdown()).await.context("server")?;
    Ok(())
}

fn main() -> ExitCode {
    let c = match Config::read() {
        Ok(Some(c)) => c,
        Ok(None) => return ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("typestrict-serve: {e:#}");
            return ExitCode::from(2);
        }
    };
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    match rt.block_on(run(c)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("typestrict-serve: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn dates_are_utc_calendar_days() {
        assert_eq!(date(UNIX_EPOCH), "1970-01-01");
        assert_eq!(date(UNIX_EPOCH + Duration::from_secs(951_782_400)), "2000-02-29");
        assert_eq!(date(UNIX_EPOCH + Duration::from_secs(1_793_836_800)), "2026-11-05");
    }

    #[test]
    fn an_empty_model_dir_says_what_is_missing() {
        let e = check_model_dir(Path::new("/nonexistent-model-dir")).unwrap_err().to_string();
        assert!(e.contains("no model in /nonexistent-model-dir") && e.contains("decision_ai.json"), "{e}");
    }
}
