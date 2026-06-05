#!/usr/bin/env bash
#
# vpclmul-validate.sh — compile the VPCLMULQDQ gf2x mul_1_n kernel and
# bit-exactly validate it against the scalar reference under Intel SDE
# (emulating a future AVX-512 CPU, since the dev box is Comet Lake = no AVX-512).
#
# Install SDE first (e.g. `paru -S intel-sde`), then:
#     bench/vpclmul-validate.sh                 # auto-detects sde64 on PATH
#     SDE=/path/to/sde64 bench/vpclmul-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/vpclmul-mul1n.c"
BIN="${TMPDIR:-/tmp}/vpclmul-test"
SDE="${SDE:-$(command -v sde64 || command -v sde || true)}"

echo "# compiling $SRC with -mavx512f -mvpclmulqdq"
${CC:-gcc} -O2 -mavx512f -mvpclmulqdq -Wall -Wextra "$SRC" -o "$BIN"

if [ -z "$SDE" ]; then
    echo "# NOTE: Intel SDE not found (set \$SDE or install it)."
    echo "# Binary built OK; it cannot run on a non-AVX-512 host (would SIGILL)."
    echo "# Verify the instruction is present instead:"
    objdump -d "$BIN" | grep -m2 -iE "vpclmul" || echo "  (no vpclmul found — unexpected)"
    exit 0
fi

echo "# running under: $SDE -future"
# -future emulates a CPU with the newest ISA (incl. AVX-512 + VPCLMULQDQ).
"$SDE" -future -- "$BIN"
