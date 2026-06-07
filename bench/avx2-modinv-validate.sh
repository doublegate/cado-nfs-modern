#!/usr/bin/env bash
#
# avx2-modinv-validate.sh — compile the AVX2 8-way batched 32-bit modular inverse
# (the vectorizable arithmetic core of the siever's per-prime lattice setup,
# Roadmap B4) and bit-exactly validate it against GMP, then print the measured
# scalar-vs-AVX2 speedup. Unlike the AVX-512 sibling (B1), this RUNS NATIVELY on
# any AVX2 host (no Intel SDE needed) — the reference box (Comet Lake) has AVX2.
#
#     bench/avx2-modinv-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/avx2-modinv.c"
BIN="${TMPDIR:-/tmp}/avx2-modinv"

echo "# compiling $SRC with -mavx2"
${CC:-gcc} -O2 -mavx2 -Wall "$SRC" -lgmp -o "$BIN"

if ! grep -qiE 'avx2|^flags.*\bavx2\b' /proc/cpuinfo 2>/dev/null; then
    echo "# NOTE: this host reports no AVX2; the binary would SIGILL here."
    echo "# Built OK; run it on an AVX2 host."
    exit 0
fi

echo "# running natively (AVX2 present)"
"$BIN"
