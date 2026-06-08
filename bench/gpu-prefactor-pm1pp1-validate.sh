#!/usr/bin/env bash
#
# gpu-prefactor-pm1pp1-validate.sh — compile and run the bit-exact validation of
# the GPU Pollard P-1 / Williams P+1 kernels (Roadmap C7), the two methods added to
# the GPU pre-NFS factoring front-end beside the batched ECM. Each kernel is checked
# bit-exact vs the host (__host__ __device__) code AND vs an independent GMP
# reference (mpz_powm for P-1, a GMP Lucas chain for P+1), plus functional factor
# recovery on crafted composites. Needs nvcc + an NVIDIA GPU (the reference box has
# an RTX 3090, sm_86) + GMP.
#
#     bench/gpu-prefactor-pm1pp1-validate.sh
#
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$HERE/gpu-prefactor-pm1pp1.cu"
BIN="${TMPDIR:-/tmp}/gpu-prefactor-pm1pp1"
ARCH="${CUDA_ARCH:-sm_86}"

if ! command -v nvcc >/dev/null 2>&1; then
    echo "# NOTE: no nvcc found; skipping (build on a CUDA host)."
    exit 0
fi

echo "# compiling $SRC (-arch=$ARCH)"
nvcc -arch="$ARCH" -O3 -I "$ROOT/misc/gpu_prefactor" "$SRC" -lgmp -o "$BIN"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    echo "# NOTE: built OK but no GPU visible; run it on a host with an NVIDIA GPU."
    exit 0
fi

echo "# running on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
"$BIN"
