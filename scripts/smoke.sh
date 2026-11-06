#!/bin/sh
# Smoke test of an image: runs it with a model mounted at /model and asks one question of each type.
#
#   scripts/smoke.sh <image> <model dir>
set -eu
image="${1:?usage: smoke.sh <image> <model dir>}"
model="$(cd "${2:?usage: smoke.sh <image> <model dir>}" && pwd)"
name="typestrict-runtime-smoke-$$"
port="${SMOKE_PORT:-18080}"

docker run -d --rm --name "$name" -p "$port:8080" -e TYPESTRICT_TOKEN=smoke -v "$model:/model:ro" "$image" >/dev/null
trap 'docker logs "$name" 2>&1 | sed "s/^/  | /"; docker stop "$name" >/dev/null 2>&1 || true' EXIT

i=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/healthz")" = 200 ]; do
  i=$((i + 1))
  if [ "$i" -gt 240 ]; then echo "smoke: /healthz never answered 200" >&2; exit 1; fi
  sleep 0.5
done

code="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$port/v1/models")"
[ "$code" = 401 ] || { echo "smoke: /v1/models without the token answered $code, not 401" >&2; exit 1; }

model_name="$(curl -fsS -H 'Authorization: Bearer smoke' "http://localhost:$port/v1/models" | sed -n 's/.*"name":"\([^"]*\)".*/\1/p')"
answer="$(curl -fsS -H 'Authorization: Bearer smoke' -H 'Content-Type: application/json' "http://localhost:$port/v1/systemone" -d "{
  \"model\": \"$model_name\",
  \"state\": \"I was charged twice. Please help.\",
  \"questions\": {
    \"billing\": {\"type\": \"noul\", \"instructions\": \"Is this message about billing?\"},
    \"team\": {\"type\": \"choice\", \"criteria\": {\"billing\": \"Payments and invoices\", \"support\": \"Everything else\"}},
    \"urgency\": {\"type\": \"score\", \"criteria\": [\"Can wait\", \"This week\", \"Today\"]}
  }
}")"
echo "$answer"
for k in '"billing":{"type":"noul"' '"team":{"type":"choice"' '"urgency":{"type":"score"' '"usage":{"input_tokens":'; do
  case "$answer" in *"$k"*) ;; *) echo "smoke: $k missing from the answer" >&2; exit 1 ;; esac
done
trap 'docker stop "$name" >/dev/null 2>&1 || true' EXIT
echo "smoke: ok"
