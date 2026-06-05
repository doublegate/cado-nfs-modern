#!/usr/bin/env bash
#
# gpu-cofac-batch-bench.sh -- correctness + throughput of the batched GPU ECM
# drain (CADO_GPU_ECM=batch). Runs the same deterministic c120 sieve slice
# CPU-only vs batch and reports wall time, relations, rel/s, and the diff.
#
# Expectation (see docs/gpu-cofactorization.md item 4):
#   - correctness: batch is a valid SUPERSET (lost=0); it may emit extra valid
#     relations because the GPU resolves cofactors facul would give up on.
#   - throughput: on c120 this box, batch is NOT a net win (cofactoring is only
#     ~8% of las CPU time -> Amdahl ceiling; the blanket pre-pass runs full ECM
#     on every cofactor). Recorded honestly so the regime-shift follow-on has a
#     baseline.
#
# USAGE: bench/gpu-cofac-batch-bench.sh [las-binary] [threads] [q1-upper]
set -eu
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAS="${1:-$HERE/build-gpu/sieve/las}"
T="${2:-1}"; Q1="${3:-603000}"
POLY="$HERE/tests/sieve/c120.poly"; FB="${FB:-/tmp/cado-bench/c120.fb1}"
MAKEFB="$(dirname "$LAS")/makefb"; OUT="$(mktemp -d /tmp/cado-gpu-bb.XXXXXX)"
ARGS=(-poly "$POLY" -fb1 "$FB" -lim0 2500000 -lim1 3400000
      -lpb0 27 -lpb1 28 -mfb0 52 -mfb1 54 -I 12
      -q0 600000 -q1 "$Q1" -sqside 1 -t "$T")
[ -s "$FB" ] || "$MAKEFB" -poly "$POLY" -lim 3400000 -maxbits 12 -side 1 -t "$(nproc)" -out "$FB" >/dev/null 2>&1

run() {
  local label="$1" mode="$2" t0 t1 w rel
  t0=$(date +%s.%N)
  if [ -z "$mode" ]; then env -u CADO_GPU_ECM "$LAS" "${ARGS[@]}" 2>"$OUT/$label.err"
  else CADO_GPU_ECM="$mode" "$LAS" "${ARGS[@]}" 2>"$OUT/$label.err"; fi \
      | grep '^[^#]' | sort > "$OUT/rel.$label"
  t1=$(date +%s.%N); w=$(echo "$t1-$t0"|bc); rel=$(wc -l < "$OUT/rel.$label")
  printf "%-6s -t%-3s wall=%6.1fs  relations=%-6s rel/s=%-5.0f " "$label" "$T" "$w" "$rel" "$(echo "$rel/$w"|bc -l)"
  grep -i 'GPU ECM cofac hook' "$OUT/$label.err" || echo "(hook off)"
}

echo "### c120 600000..$Q1, -t $T   las=$LAS"
run off   ""
run batch batch
echo "### correctness"
echo "batch vs off: +$(comm -13 "$OUT/rel.off" "$OUT/rel.batch"|wc -l) extra, -$(comm -23 "$OUT/rel.off" "$OUT/rel.batch"|wc -l) lost (lost must be 0)"
echo "# artifacts in $OUT"
