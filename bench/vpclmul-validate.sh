#!/usr/bin/env bash
#
# vpclmul-validate.sh — compile the VPCLMULQDQ gf2x kernels and bit-exactly
# validate them against the scalar reference under Intel SDE (emulating a future
# AVX-512 CPU, since the dev box is Comet Lake = no AVX-512). Covers:
#   - vpclmul-mul1n.c : the variable-length mul_1_n / addmul_1_n base case (3.1.0)
#   - vpclmul-muln.c  : the fixed-size gf2x_mul2 / mul3 / mul4 kernels (3.2.0, B2)
#
# Install SDE first (e.g. `paru -S intel-sde`), then:
#     bench/vpclmul-validate.sh                 # auto-detects sde64 on PATH
#     SDE=/path/to/sde64 bench/vpclmul-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SDE="${SDE:-$(command -v sde64 || command -v sde \
    || ls /opt/intel-sde/sde64 2>/dev/null \
    || ls /opt/sde*/sde64 2>/dev/null | head -1 || true)}"

rc=0
for name in vpclmul-mul1n vpclmul-muln; do
    SRC="$HERE/$name.c"
    BIN="${TMPDIR:-/tmp}/$name-test"
    echo "# compiling $SRC with -mavx512f -mvpclmulqdq"
    ${CC:-gcc} -O2 -mavx512f -mvpclmulqdq -Wall -Wextra "$SRC" -o "$BIN"
    if [ -z "$SDE" ]; then
        echo "# NOTE: Intel SDE not found (set \$SDE or install it)."
        echo "# Binary built OK; it cannot run on a non-AVX-512 host (would SIGILL)."
        echo "# Verify the instruction is present instead:"
        objdump -d "$BIN" | grep -m2 -iE "vpclmul" || echo "  (no vpclmul found — unexpected)"
        continue
    fi
    echo "# running under: $SDE -future  ($name)"
    # -future emulates a CPU with the newest ISA (incl. AVX-512 + VPCLMULQDQ).
    "$SDE" -future -- "$BIN" || rc=1
done
exit $rc
