# 3.0.0-modern build config — Phase 1 conclusion.
# Microbenchmarked (bench/las-microbench.sh) on the c120 siever workload:
#   stock -O2            : 12.57s  (baseline)
#   -O2 -march=native    : 11.79s  (-6.2%)   <- the real win is host-ISA codegen
#   -O3 -march=native    : 11.66s  (-7.2%)   <- adopted (fastest, builds clean)
#   + LTO                : 11.65s  (0%, and breaks -Werror)        -> rejected
#   + PGO                : 11.99s  (+2.8% SLOWER)                  -> rejected
CFLAGS="-O3 -march=native -mtune=native -fcommon"
CXXFLAGS="-O3 -march=native -mtune=native"
# 3.0.0 needs the Python flask/requests modules at configure time. Create the
# venv once with `bash scripts/setup-venv.sh`, then point cmake at it. local.sh
# is sourced from the source root, so $PWD resolves portably.
CMAKE_EXTRA_ARGS="-DPYTHON_EXECUTABLE=$PWD/cado-nfs.venv/bin/python3"
