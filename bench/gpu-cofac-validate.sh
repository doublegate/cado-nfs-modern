#!/usr/bin/env bash
#
# gpu-cofac-validate.sh -- prove the GPU ECM cofactorization hook is correct.
#
# Runs the SAME deterministic c120 sieve (fixed poly + factor base + special-q
# slice, -t 1) three ways through one CUDA-enabled `las`:
#
#   off      CADO_GPU_ECM unset  -> stock CPU cofactoring (hook not entered)
#   shadow   CADO_GPU_ECM=shadow -> GPU ECM runs on the real leftover cofactors
#                                   but never changes facul's verdict
#   salvage  CADO_GPU_ECM=salvage-> GPU retries facul give-ups (FACUL_MAYBE)
#
# Expectation (see docs/gpu-cofactorization.md):
#   - off vs shadow : byte-for-byte IDENTICAL relations (identity preserved),
#     with the GPU reporting it split real survivor cofactors.
#   - off vs salvage: a valid SUPERSET (lost=0); extra relations only when the
#     GPU fully splits a cofactor facul gave up on.
#
# USAGE: bench/gpu-cofac-validate.sh [las-binary] [q1-upper-bound]
#   las-binary defaults to build-gpu/sieve/las (must be built -DENABLE_GPU=ON).
set -eu

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LAS="${1:-$HERE/build-gpu/sieve/las}"
Q1="${2:-603000}"
POLY="$HERE/tests/sieve/c120.poly"
FB="${FB:-/tmp/cado-bench/c120.fb1}"
MAKEFB="$(dirname "$LAS")/makefb"
OUT="$(mktemp -d /tmp/cado-gpu-val.XXXXXX)"

# c120 sieve parameters (parameters/factor/params.c120)
ARGS=(-poly "$POLY" -fb1 "$FB" -lim0 2500000 -lim1 3400000
      -lpb0 27 -lpb1 28 -mfb0 52 -mfb1 54 -I 12
      -q0 600000 -q1 "$Q1" -sqside 1 -t 1)

mkdir -p "$(dirname "$FB")"
[ -s "$FB" ] || "$MAKEFB" -poly "$POLY" -lim 3400000 -maxbits 12 -side 1 \
                          -t "$(nproc)" -out "$FB" >/dev/null 2>&1

run() {  # $1 label, $2 mode ("" = unset)
  local label="$1" mode="$2"
  if [ -z "$mode" ]; then
    env -u CADO_GPU_ECM "$LAS" "${ARGS[@]}" 2>"$OUT/$label.err"
  else
    CADO_GPU_ECM="$mode" "$LAS" "${ARGS[@]}" 2>"$OUT/$label.err"
  fi | grep '^[^#]' | sort > "$OUT/rel.$label"
  printf "%-8s relations=%-6s " "$label" "$(wc -l < "$OUT/rel.$label")"
  grep -i 'GPU ECM cofac hook' "$OUT/$label.err" || echo "(hook off)"
}

echo "### las: $LAS   special-q 600000..$Q1   c120   -t 1"
run off     ""
run shadow  shadow
run salvage salvage

echo "### result"
if diff -q "$OUT/rel.off" "$OUT/rel.shadow" >/dev/null; then
  echo "PASS off==shadow : $(wc -l < "$OUT/rel.off") relations byte-identical (identity preserved)"
else
  echo "FAIL off!=shadow : shadow mode must not change the relation set"; diff "$OUT/rel.off" "$OUT/rel.shadow" | head
fi
echo "salvage vs off  : +$(comm -13 "$OUT/rel.off" "$OUT/rel.salvage" | wc -l) extra, -$(comm -23 "$OUT/rel.off" "$OUT/rel.salvage" | wc -l) lost (lost must be 0)"
echo "# artifacts in $OUT"
