#!/usr/bin/env bash
#
# ifma-validate.sh — compile the AVX-512 IFMA GF(p) kernels and bit-exactly
# validate them against GMP under Intel SDE (emulating a CPU with AVX-512-IFMA,
# since the dev box is Comet Lake = no IFMA). Covers:
#   - ifma-modmul.c : the 8-way Montgomery modmul primitive (3.1.0, Track 1.4)
#   - ifma-gfp.c    : plain-representation GF(p) ops (plain_mul, vec_add_dotprod
#                     shape) matching the arith-modp BWC backend (3.2.0, B3)
#
# Install SDE first (e.g. `paru -S intel-sde`), then:
#     bench/ifma-validate.sh                 # auto-detects sde64
#     SDE=/path/to/sde64 bench/ifma-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
SDE="${SDE:-$(command -v sde64 || command -v sde \
    || ls /opt/intel-sde/sde64 2>/dev/null \
    || ls /opt/sde*/sde64 2>/dev/null | head -1 || true)}"

rc=0
for name in ifma-modmul ifma-gfp; do
    SRC="$HERE/$name.c"
    BIN="${TMPDIR:-/tmp}/$name"
    echo "# compiling $SRC with -mavx512f -mavx512ifma"
    ${CC:-gcc} -O2 -mavx512f -mavx512ifma -Wall "$SRC" -lgmp -o "$BIN"
    if [ -z "$SDE" ]; then
        echo "# NOTE: Intel SDE not found (set \$SDE or install it)."
        echo "# Binary built OK; it cannot run on a non-IFMA host (would SIGILL)."
        echo "# Verify the IFMA instruction is present instead:"
        objdump -d "$BIN" | grep -m2 -iE "vpmadd52" || echo "  (no vpmadd52 found — unexpected)"
        continue
    fi
    echo "# running under: $SDE -future  ($name)"
    "$SDE" -future -- "$BIN" || rc=1
done
exit $rc
