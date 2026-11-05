# typestrict-serve

The HTTP runtime that serves a [typestrict.ai](https://typestrict.ai) model on CPU with the same decision API as Jev:
`POST /v1/systemone`. Switching from Jev to your own model is changing a base URL.

- One static binary (Rust, ONNX Runtime linked in), no GPU, no database.
- No network calls and no telemetry: requests and data stay in your infrastructure.
- Published as the image `ghcr.io/typestrict-lab/serve` (linux/amd64, linux/arm64). The image carries no model: a
  model's image is this one plus one layer with the model files at `/model`.
- Runs unchanged on Docker, Kubernetes, Cloud Run and AWS Lambda (the
  [AWS Lambda Web Adapter](https://github.com/awslabs/aws-lambda-web-adapter) is in the image).

## Run it

A model directory holds what typestrict.ai ships for each model:

```
model/
  decision_ai.json        the manifest: reader settings, calibration, file names
  tokenizer.json
  onnx/model_8bit.onnx    the ONNX the manifest's files.model names
```

With Docker, mount it at `/model`:

```bash
docker run -p 8080:8080 -v "$PWD/model:/model:ro" ghcr.io/typestrict-lab/serve
```

or bake it into an image of its own:

```dockerfile
FROM ghcr.io/typestrict-lab/serve:0.1
COPY model/ /model/
```

From source (Rust 1.88 or newer):

```bash
cargo build --release
./target/release/typestrict-serve --model ./model
```

`GET /healthz` answers `503 loading` while the model loads, then `200 ok`. Without a model in the directory the
process exits at once with a message that says what is missing.

## API

| Endpoint | What it does |
| --- | --- |
| `POST /v1/systemone` | Answers named questions about a `state`, in Jev's request and response format. |
| `GET /v1/models` | The model inside, by name: `{"models": [{"name", "description", "release_date"}]}`. |
| `GET /healthz` | `ok` once the model is loaded, for health and readiness checks. Never needs the token. |

```bash
curl http://localhost:8080/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "ticket-router",
  "state": "I was charged twice. Please help.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team handles the ticket?",
      "criteria": { "billing": "Payments and invoices", "support": "Everything else" }
    },
    "urgent": { "type": "noul", "instructions": "Is this urgent?" },
    "severity": { "type": "score", "criteria": ["Can wait", "This week", "Today"] }
  }
}'
```

```json
{
  "model": "ticket-router",
  "answers": {
    "team": { "type": "choice", "choice": "billing", "confidence": 0.97, "probabilities": { "billing": 0.97, "support": 0.03 } },
    "urgent": { "type": "noul", "noul": 0.81 },
    "severity": { "type": "score", "score": 1.62, "confidence": 0.7, "legend": { "0": "Can wait", "1": "This week", "2": "Today" }, "probabilities": { "0": 0.08, "1": 0.22, "2": 0.7 } }
  },
  "usage": { "input_tokens": 118, "output_tokens": 0 }
}
```

- The container holds one model, so any `model` value is answered, and the response echoes the requested `model` back,
  as Jev does: a client switches by changing only its base URL. `GET /v1/models` lists the model by its own name (the
  manifest's `name`, or `TYPESTRICT_MODEL_NAME`).
- Answers come in the order of the questions, and a choice's probabilities in the order of its criteria.
- `noul` is the probability of yes. `choice` returns the most likely name, its confidence and every probability.
  `score` is the expected level (the probability-weighted average of the levels, so it can fall between them).
- The response has exactly the contract's fields. The time the model took is in the `x-typestrict-latency-ms` header.
- Invalid requests get `422` with `{"detail": [{"loc", "msg", "type"}]}`, as the contract's `HTTPValidationError`;
  a question whose options do not fit in the model's context is one of them.

## Configuration

| Variable | Flag | Default | What it does |
| --- | --- | --- | --- |
| `PORT` | `--port` | `8080` | The port to listen on (Cloud Run and Lambda set it). |
| `TYPESTRICT_TOKEN` | | none | When set, every request but `/healthz` needs `Authorization: Bearer <token>` (compared in constant time). Prefer your platform's own auth (Cloud Run IAM, Lambda IAM) when you have it. |
| `TYPESTRICT_MODEL_DIR` | `--model` | `/model` | The model directory. |
| `TYPESTRICT_MODEL_NAME` | `--name` | the manifest's `name` | The model name the API answers to. |
| `TYPESTRICT_WORKERS` | `--workers` | min(CPU count, 4, memory limit / 2.5 GB), at least 1 (1 on Lambda) | ONNX Runtime sessions; each answers one request at a time. |
| `TYPESTRICT_THREADS` | `--threads` | 1 (the CPU count on Lambda) | Threads per session. |
| `HOST` | `--host` | `0.0.0.0` | The address to listen on. |

Several 1-thread sessions answer more requests per CPU than one session with many threads; more threads per session
cut the latency of a single request. Memory is per session: about 400 MB idle, growing up to about 2.3 GB once it has
read a 4,096-token input (a 150M-parameter model), and ONNX Runtime keeps that memory. So without `TYPESTRICT_WORKERS`
the runtime reads the container's memory limit (cgroup v2 `memory.max`, or v1 `memory.limit_in_bytes`) and starts at
most one session per 2.5 GB of it: a 2 GB Cloud Run or Kubernetes container gets one session. The startup log says how
many sessions it chose and why. Set `TYPESTRICT_WORKERS` explicitly if your inputs are short.

## AWS Lambda

The image includes the AWS Lambda Web Adapter as a Lambda extension, with its readiness check on `/healthz`
(`AWS_LWA_READINESS_CHECK_PATH=/healthz`, healthy on `200` only), so the same image runs as a Lambda container
function with no change. Lambda gives an execution environment one request at a time, so on Lambda the runtime
defaults to one session using every vCPU. CPU grows with memory on Lambda: 2048 MB is a good start. The first request
after a cold start waits for the model to load.

## How it reads a model

The runtime is a port of the typestrict.ai playground's option reader, the code that runs the same model files in the
browser, and of `typestrict/format.py`, the format the models were trained on:

- the sequence `[CLS] instructions [SEP] state [SEP] [OPT] option_1 ... [OPT] option_n`, with the training cuts: each
  option to `truncation.max_criterion_tokens`, the state to what is left of `truncation.max_length`;
- content rendered as training rendered it: strings stripped like Python's `str.strip()`, anything else as
  `json.dumps(sort_keys=True, ensure_ascii=False, separators=(",", ":"))`, floats as Python writes them;
- padding to `pad_multiple`, input and output names from the manifest, the temperature for (question type, option
  count), and Platt scaling of the confidence head;
- a `noul` question is read as false against true: its `criteria.false` or the manifest's `noul_false_default`, its
  `criteria.true` or its instructions;
- when a question is the one the model was trained on (same labels for a single-task model, same instructions for a
  yes/no task of a multi-task model) and the client gives no other wording, the options are the manifest's trained
  option texts.

Both manifest formats are read: `decision-ai/option-reader@1` (what typestrict.ai writes) and `decision-ai/model@1`
with `"reader": "option-reader"` and a `reader_config`.

`tests/golden.rs` checks this against vectors the playground's own reader wrote, on a published model: identical
token ids, option positions and padding, and probabilities within 1e-4 (in practice within 2e-6).

## Development

```bash
scripts/fetch-test-model.sh     # typestrict/TS-PRGuard at a pinned revision, each file checked by SHA-256 (170 MB)
cargo test                      # unit, golden and HTTP tests; the last two are skipped, with a message, without the model
cargo clippy --all-targets -- -D warnings
docker build -t typestrict-serve . && scripts/smoke.sh typestrict-serve test-model
```

### Golden vectors

`tests/fixtures/golden.json` is written by `scripts/golden/make-golden.sh`, which runs the playground's `reader.ts`
unchanged in Node (onnxruntime-web, WebAssembly) over the test model and records, for each question, the token ids, the
inputs fed to the model and the answer. It needs a checkout of the typestrict web app with its dependencies installed:

```bash
TYPESTRICT_WEB=../typestrict/apps/web scripts/golden/make-golden.sh
```

Regenerate it whenever the test model's revision in `scripts/fetch-test-model.sh` changes.

## License

Apache-2.0. See [LICENSE](LICENSE).
