//! How to run the questions of one request: measures the strategies on a model directory.
//!
//!   cargo run --release --example strategies -- <A|B|C|D> <short|long> [model dir]
//!
//! A: one after another on one session. B: fanned out to the free sessions. C: one batched ONNX run on one session.
//! D: B when sessions are free, C on the sessions it got. Q: B per question, each question queued for the next free
//! session on its own. Four sessions of one thread. Prints the latency of one
//! request alone, then latency and questions/s at concurrency 4 and 16, and the largest probability difference from
//! single-question runs. Run it under `/usr/bin/time -l` for the peak RSS.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use serde_json::json;
use typestrict_serve::manifest::Kind;
use typestrict_serve::reader::{Decision, Encoded, Encoder, Model, Question};

const WORKERS: usize = 4;

struct Pool {
    free: Mutex<Vec<Model>>,
    cv: Condvar,
}

impl Pool {
    /// Waits for one session, then takes up to `max` that are free.
    fn take(&self, max: usize) -> Vec<Model> {
        let mut free = self.cv.wait_while(self.free.lock().unwrap(), |f| f.is_empty()).unwrap();
        let k = max.min(free.len());
        let at = free.len() - k;
        free.split_off(at)
    }
    fn give(&self, ms: Vec<Model>) {
        self.free.lock().unwrap().extend(ms);
        self.cv.notify_all();
    }
}

fn answer(pool: &Pool, strategy: &str, qs: &[Encoded]) -> Vec<Decision> {
    match strategy {
        "A" => {
            let mut m = pool.take(1);
            let d = qs.iter().map(|e| m[0].run(e).unwrap()).collect();
            pool.give(m);
            d
        }
        "C" => {
            let mut m = pool.take(1);
            let d = m[0].run_batch(qs).unwrap();
            pool.give(m);
            d
        }
        "Q" => {
            // Each question waits for the next free session on its own: no session idles while a question waits.
            let mut out: Vec<Option<Decision>> = vec![None; qs.len()];
            std::thread::scope(|s| {
                let hs: Vec<_> = qs
                    .iter()
                    .map(|e| {
                        s.spawn(move || {
                            let mut m = pool.take(1);
                            let d = m[0].run(e).unwrap();
                            pool.give(m);
                            d
                        })
                    })
                    .collect();
                for (i, h) in hs.into_iter().enumerate() {
                    out[i] = Some(h.join().unwrap());
                }
            });
            out.into_iter().map(Option::unwrap).collect()
        }
        "B" | "D" => {
            let mut ms = pool.take(qs.len());
            let k = ms.len();
            let batch = strategy == "D";
            // Question i goes to session i % k; each session answers its share in order (or batched, for D).
            let mut out: Vec<Option<Decision>> = vec![None; qs.len()];
            std::thread::scope(|s| {
                let hs: Vec<_> = ms
                    .iter_mut()
                    .enumerate()
                    .map(|(w, m)| {
                        let mine: Vec<usize> = (w..qs.len()).step_by(k).collect();
                        s.spawn(move || {
                            let es: Vec<Encoded> = mine.iter().map(|&i| qs[i].clone()).collect();
                            let d = if batch && es.len() > 1 {
                                m.run_batch(&es).unwrap()
                            } else {
                                es.iter().map(|e| m.run(e).unwrap()).collect()
                            };
                            mine.into_iter().zip(d).collect::<Vec<_>>()
                        })
                    })
                    .collect();
                for h in hs {
                    for (i, d) in h.join().unwrap() {
                        out[i] = Some(d);
                    }
                }
            });
            pool.give(ms);
            out.into_iter().map(Option::unwrap).collect()
        }
        other => panic!("unknown strategy {other}"),
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let strategy = args.first().map(String::as_str).unwrap_or("A").to_string();
    let size = args.get(1).map(String::as_str).unwrap_or("short").to_string();
    let dir = PathBuf::from(args.get(2).cloned().unwrap_or_else(|| "test-model".into()));
    let enc = Arc::new(Encoder::load(&dir).unwrap());

    // A pull request as pr-guard sends it: the description and the diff, under the model's five trained questions.
    let source = include_str!("../src/contract.rs");
    let lines: Vec<&str> = source.lines().collect();
    let take = if size == "long" { lines.len() } else { 10 };
    let diff = format!(
        "--- /dev/null\n+++ b/src/contract.rs\n@@ -0,0 +1,{take} @@\n{}",
        lines[..take].iter().map(|l| format!("+{l}")).collect::<Vec<_>>().join("\n")
    );
    let state = json!({"description": "Add the contract module: request parsing in document order and the answer shapes.", "diff": diff});
    let qs: Vec<Encoded> = enc
        .manifest
        .tasks
        .iter()
        .map(|t| {
            let q = Question {
                kind: t.kind,
                instructions: t.question.clone().map(|s| json!(s)),
                state: Some(state.clone()),
                options: t.options.clone().unwrap_or_default().into_iter().map(|o| Some(json!(o))).collect(),
            };
            assert!(matches!(q.kind, Kind::Noul | Kind::Choice));
            enc.encode(&q).unwrap()
        })
        .collect();
    let tokens: Vec<usize> = qs.iter().map(|e| e.ids.len()).collect();

    let t0 = Instant::now();
    let pool = Pool {
        free: Mutex::new((0..WORKERS).map(|_| Model::load(&dir, enc.clone(), 1).unwrap()).collect()),
        cv: Condvar::new(),
    };
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Reference: each question alone, batch of one.
    let reference: Vec<Decision> = {
        let mut m = pool.take(1);
        let d = qs.iter().map(|e| m[0].run(e).unwrap()).collect();
        pool.give(m);
        d
    };
    let got = answer(&pool, &strategy, &qs);
    let diff_max = reference
        .iter()
        .zip(&got)
        .flat_map(|(a, b)| a.probabilities.iter().zip(&b.probabilities).map(|(x, y)| (x - y).abs()))
        .fold(0.0f64, f64::max);

    let solo_n = if size == "long" { 5 } else { 20 };
    let mut solo: Vec<f64> = (0..solo_n)
        .map(|_| {
            let t = Instant::now();
            answer(&pool, &strategy, &qs);
            t.elapsed().as_secs_f64() * 1000.0
        })
        .collect();
    println!(
        "{strategy} {size} tokens={tokens:?} load={load_ms:.0}ms max|dp|={diff_max:.2e} solo p50={:.0}ms p95={:.0}ms",
        pct(&mut solo, 0.5),
        pct(&mut solo, 0.95)
    );

    let total = if size == "long" { 16 } else { 80 };
    for conc in [4, 16] {
        let next = Mutex::new(0usize);
        let lat = Mutex::new(Vec::new());
        let t = Instant::now();
        std::thread::scope(|s| {
            for _ in 0..conc {
                s.spawn(|| {
                    loop {
                        {
                            let mut n = next.lock().unwrap();
                            if *n >= total {
                                break;
                            }
                            *n += 1;
                        }
                        let t = Instant::now();
                        answer(&pool, &strategy, &qs);
                        lat.lock().unwrap().push(t.elapsed().as_secs_f64() * 1000.0);
                    }
                });
            }
        });
        let secs = t.elapsed().as_secs_f64();
        let mut l = lat.into_inner().unwrap();
        println!(
            "{strategy} {size} c{conc}: {:.2} questions/s, request p50={:.0}ms p95={:.0}ms ({total} requests in {secs:.1}s)",
            (total * qs.len()) as f64 / secs,
            pct(&mut l, 0.5),
            pct(&mut l, 0.95)
        );
    }
}
