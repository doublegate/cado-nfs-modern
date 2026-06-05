#!/usr/bin/env bash
# Deployment integration: run a real factorization where cado-nfs.py starts only
# the server + driver (slaves.nrclients=0, no Python clients) and EXTERNAL Rust
# clients (cado-nfs-client-rs) do all the distributed work -- polyselect and
# sieve work-units -- end to end, until the server signals 410.
#
# Run from the repo root:  bash rust/deploy-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
N="${1:-90377629292003121684002147101760858109247336549001090677693}"  # 59-digit
PY="${PY:-cado-nfs.venv/bin/python3}"
CLIENT="rust/target/release/cado-nfs-client-rs"
[ -x "$CLIENT" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }
T="$(mktemp -d /tmp/cado-deploy.XXXXXX)"; LOG="$T/cado.log"

echo "## starting cado-nfs.py with slaves.nrclients=0 (no Python clients) ..."
# server.whitelist=localhost: allow our external clients (cado-nfs.py would
# normally whitelist the client hosts it spawns; with nrclients=0 it spawns none).
"$PY" ./cado-nfs.py "$N" server.ssl=no slaves.nrclients=0 server.whitelist=localhost -t 2 >"$LOG" 2>&1 &
CADO=$!
trap 'kill $CADO 2>/dev/null; pkill -P $$ 2>/dev/null' EXIT

URL=""
for _ in $(seq 1 120); do
  URL=$(grep -oE 'http://[0-9a-zA-Z_.:-]+' "$LOG" 2>/dev/null | grep -oE 'http://[^/ ]+' | head -1)
  [ -n "$URL" ] && break
  kill -0 $CADO 2>/dev/null || { echo "cado-nfs.py exited early:"; tail -8 "$LOG"; exit 1; }
  sleep 1
done
echo "## server URL: ${URL:-NONE}"; [ -z "$URL" ] && { tail -15 "$LOG"; exit 1; }

echo "## launching 2 external Rust clients (loop until 410) ..."
for i in 1 2; do
  "$CLIENT" --server "$URL" --dldir "$T/dl$i" --workdir "$T/w$i" \
       --clientid "rust-client-$i" >"$T/client$i.log" 2>&1 &
done

echo "## waiting for the factorization to finish (Rust clients do all WU work) ..."
wait $CADO; RC=$?
pkill -P $$ 2>/dev/null  # stop any still-looping clients
echo "## cado-nfs.py exit code: $RC"
echo "## factors line:"; grep -E '^[0-9]+ [0-9]+( [0-9]+)*$' "$LOG" | tail -1
echo "## Rust client WU activity:"
grep -hcE '# got workunit' "$T"/client*.log 2>/dev/null | paste -sd+ | bc | xargs -I{} echo "  {} work-units processed by Rust clients"
[ "$RC" -eq 0 ] && echo "## PASS: full factorization completed via external Rust clients" || echo "## FAIL"
exit "$RC"
