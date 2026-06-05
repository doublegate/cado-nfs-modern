#!/usr/bin/env bash
#
# las-microbench.sh — deterministic, low-variance siever microbenchmark.
#
# WHY: end-to-end ./cado-nfs.py timings vary ~15-20% run-to-run because
# polynomial selection is randomized — far too noisy to measure a compiler
# flag, SIMD, or code change worth a few percent. This harness pins the
# polynomial + factor base and uses `las --random-sample N --seed S`, so the
# sieving workload is identical every run. Total-CPU variance is then <1%,
# which makes ~5% kernel changes clearly measurable.
#
# USAGE:
#   bench/las-microbench.sh <las-binary> [reps] [num_q] [seed]
# e.g.
#   bench/las-microbench.sh build/$(hostname)/sieve/las 3
#
# Reports each run's total CPU time and the sieving sub-time, plus the mean.
# Compare two builds by pointing it at each one's las binary.
set -eu

LAS="${1:?usage: las-microbench.sh <las-binary> [reps] [num_q] [seed]}"
REPS="${2:-3}"
NQ="${3:-100}"
SEED="${4:-1}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"     # repo root
POLY="$HERE/tests/sieve/c120.poly"            # shipped fixed degree-5 c120 poly
FB="${FB:-/tmp/cado-nfs-bench/c120.fb1}"      # algebraic-side factor base (built once)
MAKEFB="$(dirname "$LAS")/makefb"

# c120 sieve parameters (from parameters/factor/params.c120)
LIM0=2500000 LIM1=3400000 LPB0=27 LPB1=28 MFB0=52 MFB1=54 I=12
Q0=600000 Q1=1200000 SQSIDE=1

mkdir -p "$(dirname "$FB")"
if [ ! -s "$FB" ]; then
  echo "# building factor base (once) -> $FB"
  "$MAKEFB" -poly "$POLY" -lim "$LIM1" -maxbits "$I" -side "$SQSIDE" -t "$(nproc)" -out "$FB" >/dev/null 2>&1
fi

echo "# las        : $LAS"
"$LAS" 2>&1 | grep -m1 "Compilation flags (C++)" || true
echo "# workload   : c120 poly, ${NQ} sampled special-q (seed $SEED), -t 1, ${REPS} reps"
echo "# ----------------------------------------------------------------"

total=0
for i in $(seq 1 "$REPS"); do
  line=$("$LAS" -poly "$POLY" -fb1 "$FB" -lim0 $LIM0 -lim1 $LIM1 -lpb0 $LPB0 -lpb1 $LPB1 \
        -mfb0 $MFB0 -mfb1 $MFB1 -I $I -q0 $Q0 -q1 $Q1 -sqside $SQSIDE \
        -random-sample "$NQ" -seed "$SEED" -t 1 2>/dev/null | grep "Total cpu time")
  cpu=$(echo "$line" | sed -E 's/.*Total cpu time ([0-9.]+)s.*/\1/')
  sieve=$(echo "$line" | sed -E 's/.*sieving ([0-9.]+) .*/\1/')
  printf "run %d: cpu=%ss  sieving=%ss\n" "$i" "$cpu" "$sieve"
  total=$(awk "BEGIN{print $total + $cpu}")
done
awk "BEGIN{printf \"mean cpu = %.2fs over %d reps\n\", $total/$REPS, $REPS}"
