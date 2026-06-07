#!/usr/bin/env bash
#
# avx512-modinv-validate.sh — compile the AVX-512 16-way batched 32-bit modular
# inverse (the vectorizable arithmetic core of the siever's per-prime lattice
# setup, Roadmap B1) and bit-exactly validate it against GMP under Intel SDE
# (the dev box is Comet Lake = no AVX-512). perf is gated on AVX-512 silicon.
#
#     bench/avx512-modinv-validate.sh                 # auto-detects sde64
#     SDE=/path/to/sde64 bench/avx512-modinv-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/avx512-modinv.c"
BIN="${TMPDIR:-/tmp}/avx512-modinv"
SDE="${SDE:-$(command -v sde64 || command -v sde \
    || ls /opt/intel-sde/sde64 2>/dev/null \
    || ls /opt/sde*/sde64 2>/dev/null | head -1 || true)}"

echo "# compiling $SRC with -mavx512f -mavx512cd"
${CC:-gcc} -O2 -mavx512f -mavx512cd -Wall "$SRC" -lgmp -o "$BIN"

if [ -z "$SDE" ]; then
    echo "# NOTE: Intel SDE not found (set \$SDE or install it)."
    echo "# Binary built OK; it cannot run on a non-AVX-512 host (would SIGILL)."
    echo "# Verify AVX-512 is present instead:"
    objdump -d "$BIN" | grep -m2 -iE "zmm" || echo "  (no zmm found — unexpected)"
    exit 0
fi

echo "# running under: $SDE -future"
"$SDE" -future -- "$BIN"
