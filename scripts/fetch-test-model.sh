#!/bin/sh
# Downloads the model the golden and HTTP tests run: typestrict/TS-PRGuard from Hugging Face (decision_ai.json,
# tokenizer.json, onnx/model_8bit.onnx, 170 MB), at a pinned revision, each file checked by SHA-256.
#
#   scripts/fetch-test-model.sh [dir]     # default: test-model/ at the repository root (ignored by git)
#
# The golden vectors in tests/fixtures/golden.json were made from these exact files: bump REVISION and the three
# hashes only together with a new tests/fixtures/golden.json (README.md, "Golden vectors").
set -eu

dest="${1:-$(cd "$(dirname "$0")/.." && pwd)/test-model}"
repo="typestrict/TS-PRGuard"
REVISION="2179dee21c568da8c75304c0fd6dd3c97a079b3c"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d ' ' -f 1
  else shasum -a 256 "$1" | cut -d ' ' -f 1; fi
}

fetch() { # <path in the repo> <sha256>
  out="$dest/$1"
  mkdir -p "$(dirname "$out")"
  if [ -f "$out" ] && [ "$(sha256 "$out")" = "$2" ]; then return 0; fi
  curl -fsSL --retry 3 -o "$out.part" "https://huggingface.co/$repo/resolve/$REVISION/$1"
  actual="$(sha256 "$out.part")"
  if [ "$actual" != "$2" ]; then
    rm -f "$out.part"
    echo "fetch-test-model: $1 has SHA-256 $actual, expected $2" >&2
    exit 1
  fi
  mv "$out.part" "$out"
}

fetch decision_ai.json c33899c0ba2de2d7929abfb97e7dd72b2b16e9f19e8895c9e4c0a04055cc03f8
fetch tokenizer.json 858fe1cd418807d6c71c01e24b6d3239a224a1e3c995472fef20841e09eab056
fetch onnx/model_8bit.onnx 5f61cf8beeb8cdc2f2d112549f0a66ff42f75e05fc216f4c8e2d0766d9815891
echo "TS-PRGuard @ $REVISION in $dest"
