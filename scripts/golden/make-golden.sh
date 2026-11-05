#!/bin/sh
# Regenerates tests/fixtures/golden.json with the playground's reader. Needs a checkout of the typestrict web app with its
# dependencies installed (pnpm install), and the test model (scripts/fetch-test-model.sh).
#
#   TYPESTRICT_WEB=~/Project/typestrict/apps/web scripts/golden/make-golden.sh
set -eu
root="$(cd "$(dirname "$0")/../.." && pwd)"
web="${TYPESTRICT_WEB:?set TYPESTRICT_WEB to the typestrict apps/web directory}"
cd "$root"
TYPESTRICT_WEB="$(cd "$web" && pwd)" "$web/node_modules/.bin/tsx" scripts/golden/make-golden.mts
