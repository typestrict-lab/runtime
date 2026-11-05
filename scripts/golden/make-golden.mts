// Writes tests/fixtures/golden.json: what the typestrict.ai playground's option reader (apps/web/src/lib/engine/reader.ts,
// run unchanged in Node with onnxruntime-web's WebAssembly build) feeds the model and answers for a set of questions
// over the test model. tests/golden.rs checks that the Rust reader produces the same token ids and probabilities.
//
//   TYPESTRICT_WEB=~/Project/typestrict/apps/web scripts/golden/make-golden.sh
//
// See "Golden vectors" in README.md.
import { readFileSync, writeFileSync } from "node:fs"
import { join, resolve } from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const web = process.env.TYPESTRICT_WEB ?? fail("set TYPESTRICT_WEB to the typestrict apps/web directory")
const modelDir = resolve(process.env.MODEL_DIR ?? "test-model")
const out = resolve(process.env.OUT ?? "tests/fixtures/golden.json")

type Content = string | number | boolean | null | Content[] | { [k: string]: Content }
type Question = { type: "choice" | "noul" | "score"; instructions?: Content; state?: Content; options: Content[] }
type Case = { name: string; manifest_patch?: Record<string, unknown>; question: Question }

const { Reader } = await import(join(web, "src/lib/engine/reader.ts"))
// The same module instance the reader imports. In a browser the reader points ORT at the CDN; in Node, at the installed
// dist folder, single-threaded (the playground's default on a page without cross-origin isolation).
const ort = await import(join(web, "node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs"))
ort.env.wasm.wasmPaths = `${pathToFileURL(join(web, "node_modules/onnxruntime-web/dist"))}/`
ort.env.wasm.numThreads = 1

// The reader fetches its files; here they come from the model directory, with the case's manifest patch applied.
let patch: Record<string, unknown> = {}
globalThis.fetch = (async (input: string | URL) => {
  const url = String(input)
  // onnxruntime-web's .wasm, from its dist folder (Node's fetch refuses file: URLs).
  if (url.startsWith("file:")) return new Response(readFileSync(fileURLToPath(url)), { headers: { "content-type": "application/wasm" } })
  const file = url.replace(/^model:\/\//, "")
  if (file === "decision_ai.json") {
    const m = JSON.parse(readFileSync(join(modelDir, file), "utf8"))
    return new Response(JSON.stringify({ ...m, ...patch }))
  }
  const bytes = readFileSync(join(modelDir, file))
  return new Response(bytes, { headers: { "content-length": String(bytes.length) } })
}) as typeof fetch

const diff = `--- a/src/auth.py
+++ b/src/auth.py
@@ -1,3 +1,3 @@
 def find_user(db, name):
-    return db.execute("SELECT * FROM users WHERE name = ?", (name,))
+    return db.execute(f"SELECT * FROM users WHERE name = '{name}'")
`
const m = JSON.parse(readFileSync(join(modelDir, "decision_ai.json"), "utf8"))
const task = (name: string) => m.tasks.find((t: { name: string }) => t.name === name)
const long = Array.from({ length: 1500 }, (_, i) => `+    total_${i} = compute(values[${i}]) * 0.5  # line ${i}`).join("\n")

const cases: Case[] = [
  { name: "noul_task_options", question: { type: "noul", instructions: task("prg-vul").question, state: diff, options: task("prg-vul").options } },
  { name: "noul_defaults", question: { type: "noul", instructions: "The change touches database code.", state: diff, options: [] } },
  { name: "noul_only_true", question: { type: "noul", instructions: "Is it risky?", state: diff, options: [null, "Yes, it is risky."] } },
  {
    name: "choice_2",
    question: { type: "choice", instructions: "What kind of change is this diff?", state: diff, options: ["Bug fix", "New feature"] },
  },
  {
    name: "choice_5_task",
    question: { type: "choice", instructions: task("prg-complexidade").question, state: diff, options: task("prg-complexidade").options },
  },
  {
    name: "choice_9",
    question: {
      type: "choice",
      instructions: "What kind of change is this diff?",
      state: diff,
      options: ["Bug fix", "New feature", "Refactoring without behavior change", "Performance improvement", "Tests only",
        "Build system, dependencies or continuous integration", "Maintenance or tooling", "Formatting or style only", "Documentation only"],
    },
  },
  {
    name: "score_object_state",
    question: {
      type: "score",
      instructions: { task: "How urgent is this ticket?", lang: "en" },
      state: { subject: "Duplicate charge", message: "I was charged twice.\tPlease help!", amount: 19.9, tiny: 0.00001234, count: 2, paid: true, tags: ["billing", null], "ação": "reembolso" },
      options: ["Can wait", "Needs attention this week", "Needs attention today", { level: "critical", note: "page someone" }],
    },
  },
  {
    // At the package's own cut (4,096 tokens) onnxruntime-web fails in WebAssembly ("Integer overflow" in OrtRun): the
    // ids are still checked, the answer is not.
    name: "long_state_cut_4096_ids_only",
    question: { type: "noul", instructions: task("prg-quebra").question, state: `--- a/calc.py\n+++ b/calc.py\n${long}`, options: task("prg-quebra").options },
  },
  {
    name: "long_state_cut_1536",
    manifest_patch: { truncation: { max_length: 1536, max_criterion_tokens: 160 } },
    question: { type: "noul", instructions: task("prg-quebra").question, state: `--- a/calc.py\n+++ b/calc.py\n${long}`, options: task("prg-quebra").options },
  },
  {
    name: "long_option_cut",
    question: { type: "choice", instructions: "Which description fits?", state: "short", options: ["word ".repeat(400), "other ".repeat(300), "x"] },
  },
  {
    name: "unicode",
    question: {
      type: "choice",
      instructions: "  Qual é o tom da mensagem? 🤔　",
      state: "  Olá! Ünïcödé façade 日本語のテキスト 👩‍💻 مرحبا é \u0085",
      options: ["Irritado 😠", "Calmo: neutro ou educado", "Animado ✨ (entusiasmado)"],
    },
  },
  {
    name: "calibrated",
    manifest_patch: { calibration: { temperature: [[1.5, 2, 2, 2, 2, 2, 2, 2], [0.8, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8], [2, 2, 2, 2, 2, 2, 2, 2]], platt: [1.3, -0.2] } },
    question: { type: "choice", instructions: task("prg-complexidade").question, state: diff, options: task("prg-complexidade").options },
  },
  {
    name: "calibrated_noul_short_cuts",
    manifest_patch: { calibration: { temperature: [[0.7]], platt: null }, truncation: { max_length: 96, max_criterion_tokens: 8 } },
    question: { type: "noul", instructions: task("prg-vul").question, state: diff, options: task("prg-vul").options },
  },
]

const results = []
let reader: InstanceType<typeof Reader> | null = null
let loadedPatch = ""
for (const c of cases) {
  const key = JSON.stringify(c.manifest_patch ?? {})
  if (!reader || key !== loadedPatch) {
    patch = c.manifest_patch ?? {}
    reader = await Reader.load((f: string) => `model://${f}`)
    loadedPatch = key
  }
  // What the reader feeds the session and what it gets back, around its own call.
  const session = (reader as unknown as { session: { run: (f: Record<string, { data: ArrayLike<number | bigint>; dims: number[] }>) => Promise<Record<string, { data: Float32Array }>> } }).session
  const run = session.run.bind(session)
  let feeds: Record<string, { data: ArrayLike<number | bigint>; dims: number[] }> = {}
  let outputs: Record<string, { data: Float32Array }> = {}
  session.run = async (f) => {
    feeds = f
    outputs = await run(f)
    return outputs
  }
  let a: { probabilities: number[] | null; best: number | null; confidence: number | null; tokens: number }
  try {
    a = await reader.decide(c.question)
  } catch (e) {
    if (!c.name.endsWith("_ids_only")) throw e
    const ones = Array.from(feeds[reader.manifest.inputs.pad_mask].data, Number).reduce((s, v) => s + v, 0)
    a = { probabilities: null, best: null, confidence: null, tokens: ones }
    outputs = {}
  }
  session.run = run
  const names = reader.manifest.inputs
  const nums = (x: ArrayLike<number | bigint>) => Array.from(x, Number)
  const ids = nums(feeds[names.ids].data)
  const mask = nums(feeds[names.pad_mask].data)
  results.push({
    name: c.name,
    manifest_patch: c.manifest_patch ?? null,
    question: c.question,
    ids: ids.slice(0, a.tokens),
    padded_length: ids.length,
    pad_mask_ones: mask.reduce((s, v) => s + v, 0),
    option_positions: nums(feeds[names.option_positions].data),
    question_type: nums(feeds[names.question_type].data)[0],
    logits: outputs[reader.manifest.outputs.logits] ? Array.from(outputs[reader.manifest.outputs.logits].data) : null,
    probabilities: a.probabilities,
    best: a.best,
    confidence: a.confidence,
    tokens: a.tokens,
  })
  console.log(`${c.name}: ${a.tokens} tokens, best ${a.best}, p=${a.probabilities?.map((p: number) => p.toFixed(4)).join(" ")}`)
}

const ortVersion = (await import(join(web, "node_modules/onnxruntime-web/package.json"), { with: { type: "json" } })).default.version
writeFileSync(
  out,
  `${JSON.stringify({ generator: "scripts/golden/make-golden.mts", reader: "typestrict apps/web/src/lib/engine/reader.ts", onnxruntime_web: ortVersion, cases: results }, null, 1)}\n`,
)
console.log(`wrote ${out}`)

function fail(msg: string): never {
  throw new Error(msg)
}
