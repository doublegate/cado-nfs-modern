#!/usr/bin/env bash
# In-process server swap: run a real factorization where cado-nfs.py uses the
# Rust work-unit server (cado-wu-server-rs) in place of the in-process Flask
# api_server.py, over the same wudb SQLite database, with the stock Python
# clients. Triggered by the CADO_RUST_WU_SERVER environment variable, which
# cadotask.py honours.
#
# Run from the repo root:  bash rust/server-swap-test.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
N="${1:-90377629292003121684002147101760858109247336549001090677693}"  # 59-digit
PY="${PY:-cado-nfs.venv/bin/python3}"
export CADO_RUST_WU_SERVER="$ROOT/rust/target/release/cado-wu-server-rs"
[ -x "$CADO_RUST_WU_SERVER" ] || { echo "build first: (cd rust && cargo build --release)"; exit 1; }
T="$(mktemp -d /tmp/cado-swap.XXXXXX)"; LOG="$T/cado.log"

echo "## CADO_RUST_WU_SERVER=$CADO_RUST_WU_SERVER"
echo "## running cado-nfs.py with the Rust server swapped in (Python clients) ..."
"$PY" ./cado-nfs.py "$N" server.ssl=no -t 2 >"$LOG" 2>&1
RC=$?

echo "## cado-nfs.py exit code: $RC"
echo "## shim launched the Rust server?"
grep -m1 "Launching Rust work-unit server" "$LOG" | sed 's/^/   /' || echo "   (NOT FOUND -- swap did not engage)"
echo "## Rust server activity (assignments / results):"
python3 -c "
import sys
a=r=0
for l in open('$LOG',encoding='utf-8',errors='replace'):
    if 'rust-wu-server:' in l and 'assigned wu' in l: a+=1
    if 'rust-wu-server:' in l and 'recorded result' in l: r+=1
print(f'   {a} work-units assigned, {r} results recorded by the Rust server')
"
echo "## factors:"; grep -E '^[0-9]+ [0-9]+( [0-9]+)*$' "$LOG" | tail -1 | sed 's/^/   /'
[ "$RC" -eq 0 ] && echo "## PASS: factorization completed with the Rust server in-process" || { echo "## FAIL"; tail -15 "$LOG"; }
exit "$RC"
