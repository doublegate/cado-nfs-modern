# Benchmarks

Reference performance of **CADO-NFS 3.2.0-modern** on a single desktop —
CPU factorization, the deterministic siever microbenchmark, and the GPU /
AVX-512 work through 3.2.0. These numbers characterize this build on this class of
hardware; they are not a comparison against other NFS implementations. Every
factorization is verified (`product == N`, factors prime); every GPU/SIMD kernel is
validated bit-exact against a CPU/GMP reference (or, for AVX-512, under Intel SDE).

_CPU/GPU-linalg/prefactor/AVX-512 numbers re-confirmed **2026-06-06** (3.1.0); the
**3.2.0 GPU + algorithm additions** (§6) measured **2026-06-07** on the same
machine. CPU factorization is unchanged from 3.0.0/3.1.0-modern — **3.2.0 adds no
CPU-path change** (all new work is GPU / AVX-512-SDE / orchestration) — so §1
carries forward, re-confirmed here._

## Test machine

| Component | Detail |
|-----------|--------|
| CPU | Intel Core i9-10850K — 10 cores / 20 threads, 3.6 GHz base / 5.2 GHz boost (Comet Lake) |
| GPU | NVIDIA GeForce RTX 3090 (24 GiB, sm_86) |
| RAM | 64 GiB DDR4 |
| OS / kernel | CachyOS, Linux 7.0.10 |
| Compiler | GCC 16.1.1, **`-O3 -march=native -mtune=native`** (this fork's `local.sh`); CUDA 13.3 (nvcc, `-O3`, `sm_86`) |
| Libraries | GMP 6.3.0, hwloc 2.13.0 |
| Orchestration | Python 3.14.5 (Flask work-unit server) |
| AVX-512 emulation | Intel SDE 10.8.0 (`-future`) — Comet Lake has no AVX-512, so VPCLMULQDQ/IFMA are validated under SDE |
| CADO-NFS | 3.1.0-modern (this fork; rebased on upstream 3.0.0) |

---

## 1. CPU factorization (default build)

Factoring balanced (RSA-like) semiprimes `n = p·q` with `p`, `q` of equal digit
length — the hardest case for NFS. The exact inputs are the **same seeded inputs**
used for the 2.3.1-modern benchmarks (listed in §8), so the generational
comparison is apples-to-apples on identical numbers.

- **Command:** `cado-nfs.venv/bin/python3 ./cado-nfs.py <n> server.ssl=no -t 20`
  — all 20 logical threads; TLS disabled to isolate compute.
- **Timing:** wall-clock around the process; CPU/elapsed as reported by CADO-NFS.
  `parallel` = total CPU ÷ elapsed.
- **Runs:** one per size; NFS timing has inherent ~±15-20 % variance, mostly from
  randomized polynomial selection — treat as representative, not exact.

| Digits | Bits | Wall time | CADO CPU | Parallel | Status |
|-------:|-----:|----------:|---------:|---------:|:------:|
| 60 | ~199 | 18.5 s | 31.1 s | 1.7× | `product == N` |
| 70 | ~232 | 27.2 s | 108.5 s | 4.1× | `product == N` |
| 80 | ~265 | 74.4 s | 406.0 s | 5.5× | `product == N` |
| 90 | ~299 | 197.9 s | 1604.2 s | 8.1× | `product == N` |

### Per-phase CPU time (seconds)

CADO-reported CPU per stage (aggregate CPU-seconds across threads). Polynomial
selection is shown as CPU (CADO 3.0.0 reports it as an aggregate "Total time"),
comparable to the other columns.

| Digits | Polyselect | Lattice sieving | Filtering | Linear algebra | Square root |
|-------:|-----------:|----------------:|----------:|---------------:|------------:|
| 60 |  4.2 |   18.0 |   4.0 |    2.1 |   2.0 |
| 70 |  8.9 |   52.6 |  38.2 |    3.9 |   3.0 |
| 80 | 19.2 |  263.5 |  57.9 |   39.7 |  22.8 |
| 90 | 51.7 | 1069.6 | 100.1 |  237.6 | 134.0 |

("Filtering" sums dup1/dup2/purge/merge/replay; its run-to-run spread is large —
the merge step in particular varies with the matrix that filtering produces.)

### Versus 2.3.1-modern (same seeded inputs)

The prior `2.3.1-modern` fork (upstream 2.3.0 base, `-O2 -fcommon`) on the same
machine and the **same seeded inputs**, versus the current 3.x-modern line:

| Digits | Wall: 2.3.1 → 3.x | CADO CPU: 2.3.1 → 3.x | Parallel: 2.3.1 → 3.x |
|-------:|-------------------|------------------------|------------------------|
| 60 | 30.6 s → **18.5 s** | 57.8 → **31.1** (−46 %) | 1.9× → 1.7× |
| 70 | 35.4 s → **27.2 s** | 121.8 → **108.5** (−11 %) | 3.5× → 4.1× |
| 80 | 73.9 s → **70–74 s** | 558.0 → **406.0** (−27 %) | 7.6× → 5.5× |
| 90 | 175.3 s → ~198 s | 1942.7 → **1604.2** (−17 %) | 11.1× → 8.1× |

**The robust, repeatable signal is reduced total CPU work** — driven by upstream
3.0.0's Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16`
sieving, compounded by this fork's `-O3 -march=native` (~7 % on the siever).
Per-run CPU swings within ±15-20 % (the c60 −46 % and c70 −11 % this run bracket
the ~−25-32 % seen across repeated runs); wall-time gains shrink with size and
fall inside the polyselect noise by c80-c90. **Parallel efficiency drops** as a
*consequence* of less sieve work (the sequential phases — linear algebra, square
root, orchestration — become a larger fraction), not a regression.

### Observations & projections

- **Sieving dominates** (≈45-67 % of CPU) and is the embarrassingly-parallel
  phase; its share falls at the largest sizes as **linear algebra grows the
  fastest** (2.1 → 237.6 CPU-s c60→c90, ~110×, vs ~60× for sieving) — the classic
  NFS trade-off and the motivation for the GPU linalg work in §4.
- Wall-time roughly doubles per +10 digits in this range (sub-exponential
  `L(1/3)`). Order-of-magnitude envelope on this desktop: **≤c75 interactive ·
  c80-c95 a few minutes · ~c100 ≈ 10 min · ~c110 ≈ 1 hr · ≥c130 wants distributed
  mode.** RAM (64 GiB) is not the limit in this range.

---

## 2. Siever microbenchmark (deterministic)

`bench/las-microbench.sh` pins the polynomial + factor base and uses
`las --random-sample` so the sieving workload is identical every run (total-CPU
variance < 1 %) — far less noisy than an end-to-end factorization, which makes a
few-percent code/flag change measurable.

| Build | Mean CPU (3 reps, c120 poly, 100 sampled-q, `-t 1`) | sieving |
|-------|---:|---:|
| `-O3 -march=native` (this fork) | **11.67 s** | 8.2 s |

This matches the figure `local.sh` records for the adopted flags, confirming
**3.1.0 introduces no CPU-path change**: the PGO retry was re-measured at **+3.0 %
(rejected)** and no safe hot-scalar micro-opt was found (the hot loops —
`fill_in_buckets`, `apply_buckets_inner`, `plattice_info`, `invmod_redc_32` — are
already SIMD/unrolled/prefetched). See `CHANGELOG.md` (Tracks 1.2/1.3).

---

## 3. GPU pre-factoring ECM — CPU vs GPU (Track 2.1)

Throughput of the multi-precision ECM (stage-1 + stage-2 BSGS) behind the GPU
pre-NFS factoring front-end (`misc/gpu_prefactor`, `cado-nfs.py --gpu-prefactor`).
The **same** `__host__ __device__ ecm_run2` runs on both sides — the CPU side
across all 20 threads — so it is an apples-to-apples algorithm comparison.
`B1=50000`, `B2=5e6`, RTX 3090 vs the full i9-10850K (`bench/gpu-prefactor-bench.cu`):

| Modulus width | GPU (curves/s) | CPU, 20 threads (curves/s) | GPU speedup |
|---|---:|---:|---:|
| 128-bit (≤ ~38-digit N) | 16 899 | 347 | **48.7×** |
| 256-bit (≤ ~77-digit N) | 3 072 | 121 | **25.4×** |
| 512-bit (≤ ~154-digit N) | 332 | 32 | **10.5×** |

The edge shrinks at wider moduli (more registers/limbs per lane, lower occupancy)
but stays an order of magnitude even at 512-bit. Unlike in-sieve GPU
cofactorization (a documented Amdahl no-win, `docs/gpu-cofactorization.md`),
pre-factoring is a *separate* stage with no Amdahl ceiling, so this is a real
single-machine win when `N` has a findable factor. End-to-end, a 90-digit N with
a 14-digit factor is fully resolved by `--gpu-prefactor` in seconds, skipping NFS.
ECM math validated bit-exact (`bench/gpu-ecm-mp.cu`); see `docs/gpu-prefactor.md`.

---

## 4. GPU linear algebra (BWC SpMV) — Track 2.2

The fastest-growing NFS phase (linalg ~110× c60→c90, §1). A real `mm_impl=gpu`
BWC backend (`linalg/bwc/matmul-gpu.cu`) with both M and Mᵀ resident as CSR and a
coalesced warp-per-row kernel; bit-exact vs the CPU loop at every size.

### Scaling sweep (b64, ~30 nnz/row) — the win grows with N

`bench/gpu-spmv-bench.cu`; warp PASS (0 wrong words) at every size. CPU column is
the threaded reference SpMV loop (20 threads). Rows span c100→c120 linalg scale:

| ~size | rows | nnz | GPU warp (Gnz/s) | CPU ref loop, 20 thr (Gnz/s) | GPU ÷ CPU loop |
|---|---:|---:|---:|---:|---:|
| c100 | 1.0 M | 30 M | 30.6 | 2.25 | 14× |
| c110 | 2.0 M | 60 M | 12.6 | 1.37 | 9× |
| c115 | 4.0 M | 120 M | 9.3 | 0.37 | 25× |
| c120 | 8.0 M | 240 M | 8.1 | 0.23 | **35×** |

**Honest reading.** Two effects compound: the GPU's *absolute* throughput falls
as the matrix grows (30.6→8.1 Gnz/s — cache-resident at 1 M rows, memory-latency
bound on uncoalesced gathers at 8 M), and the CPU reference loop falls *much*
faster (random-CSR cache thrash). So the **relative win widens with N** — the
regime where large-N linear algebra lives. Against CADO's *tuned* `bucket`
backend (which reorders columns for locality and **saturates at ~1.8 Gnz/s** on
this CPU), the GPU warp kernel at c120-scale is a steadier **~4.5×** — a real
single-machine win and a floor (the kernel realises a fraction of the 3090's
~936 GB/s; aggregate multi-GPU/multi-node bandwidth is where it compounds).

### End-to-end (full vector residency, `product == N`)

A 90-digit GNFS run with `tasks.linalg.bwc.mm_impl=gpu` +
`CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1` drives the whole BWC pipeline
(krylov → lingen → mksol → gather) through the GPU backend and returns the correct
factors — **bwc total 8.18 s** (krylov 2.65 s / 2300 iters, lingen 2.14 s, mksol
1.72 s). At c90 the matrix is small, so the end-to-end linalg is immaterial; the
point is end-to-end correctness of the GPU path (full vector residency eliminates
the per-iteration PCIe transfers — the measured ~60 % of SpMV time at c70/c80),
with the size-driven win quantified by the sweep above. Intra-node multi-GPU
partition (`CADO_GPU_NPART`) is validated at N=1; multi-node residency is a
documented design (HW-gated). See `docs/gpu-linalg.md`.

---

## 5. AVX-512 kernels (correctness-only, CI-gated under SDE)

Comet Lake has no AVX-512, so these are validated **bit-exact under Intel SDE**
(`-future`); the ~39 % perf gain is gated on real AVX-512 silicon. CI runs the
same checks (`.github/workflows/avx512-validate.yml`, objdump fallback if SDE is
absent).

| Kernel | Validation | Result |
|--------|------------|--------|
| gf2x **VPCLMULQDQ** `mul_1_n` / `addmul_1_n` (`bench/vpclmul-mul1n.c`) | bit-exact vs scalar, 200 000 trials | **PASS** |
| **IFMA** GF(p) Montgomery modmul, 8-way radix-2⁵² (`bench/ifma-modmul.c`) | bit-exact vs GMP, 260-bit, 8 lanes | **PASS** (0 / 32 000) |
| **(3.2.0, B2)** gf2x VPCLMULQDQ `mul2`/`mul3`/`mul4` (`bench/vpclmul-muln.c`) | bit-exact vs scalar GF(2)[x], 200 000 trials each | **PASS** (0 / 200 000 ×3) |
| **(3.2.0, B3)** IFMA GF(p) plain-rep `plain_mul` + `vec_add_dotprod` (`bench/ifma-gfp.c`) | bit-exact vs GMP, 260-bit, 8-way | **PASS** (0 / 32 000 ×2) |
| **(3.2.0, B1)** AVX-512 16-way batched 32-bit modular inverse (`bench/avx512-modinv.c`) | bit-exact vs GMP, 640 000 trials | **PASS** (0 / 640 000) |

The gf2x VPCLMULQDQ backend is auto-detected by `configure` (run-test-gated, so a
non-AVX-512 host safely keeps the pclmul backend). B1 (the siever's per-prime
modular inverse), B2 (the small Karatsuba gf2x kernels), and B3 (the `arith-modp`
GF(p) BWC backend) are the 3.2.0 AVX-512 additions — correctness-only here, perf
gated on real AVX-512 silicon. All in `.github/workflows/avx512-validate.yml`. See
`docs/avx512-sieving-b1.md`, `docs/ifma-gfp-b3.md`, `CHANGELOG.md`.

---

## 6. v3.2.0 GPU + algorithm additions

Measured **2026-06-07** (RTX 3090 + i9-10850K). Each is validated bit-exact /
`product == N`; see the per-track docs for full analysis + honest scope.

### 6.1 GPU mixed-representation ECM — twisted-Edwards vs the ladder (A2)

`bench/gpu-ecm-edwards.cu`: an `a=−1` twisted-Edwards stage-1 (double-and-add and
wNAF) vs the Montgomery XZ ladder, **bit-exact through the birational map**
(0/8192 at every width). curves/s, `B1=2000`:

| width | ladder | Edwards wNAF | speedup | Edwards d&a |
|-------|-------:|-------------:|:-------:|:-----------:|
| 128-bit | 299 K | 778 K | **2.60×** | 1.74× |
| 256-bit | 132 K | 177 K | **1.34×** | 0.88× |
| 512-bit | 24 K | 58 K | **2.36×** | 1.91× |

wNAF wins (~1.3–2.6× here, growing with width; small-batch run-to-run variance);
plain double-and-add is ≈ break-even (theory 12.5 vs 11 mm/bit). The CPU `facul`
path already uses mixed-rep (the upstream "mishmash"); this is the GPU half.

### 6.2 Adaptive GPU SpMV — sub-warp vec16 (C1)

`bench/gpu-spmv-bench.cu`, bit-exact at every size. In the cache-resident regime
the sub-warp CSR-vector kernel (vec16) beats the warp kernel; the backend
dispatches adaptively (vec16 when L2-resident, warp otherwise):

| ~size | rows | GPU warp | GPU vec16 | vec16 ÷ warp | CPU (20 thr) |
|-------|-----:|---------:|----------:|:------------:|-------------:|
| c100 (b64) | 0.8 M | 32.8 | **43.5** Gnz/s | **1.32×** | 2.05 |
| c115 (b64) | 2.0 M | 11.6 | 10.8 (warp picked) | — | 0.75 |

### 6.3 GPU batch-smoothness leaf (C3) & GPU sieving feasibility (C4)

- **C3** (`bench/gpu-batch-smooth.cu`): the Bernstein batch-smoothness *leaf*
  extraction (`gcd(R, (P mod R)^{2^e} mod R)`, reusing the A2 montmul), **bit-exact
  vs GMP** (0/8192 at 128/256/512-bit), **14–24 Mleaf/s** (0.04–0.26 µs/leaf).
  Honest: the leaf is the only fixed-width-arithmetic fit; the batch-smoothness
  bottleneck is the big-integer remainder tree, which stays CPU/GMP.
- **C4** (`bench/gpu-sieve-scatter.cu`, measured negative): GPU atomic scatter (the
  core sieve op) is **~6.4× a full CPU socket** in the cache-resident regime (16
  KiB: 6960 vs 1082 M upd/s) — but byte-atomic granularity, on-GPU update
  generation, and capacity make a GPU siever unsolved; keep GPU on cofactorization
  / linalg / polyselect.

---

## 7. v3.3.0 additions (measured 2026-06-07)

The honest v3.3.0 frame: single-machine *speed* is tapped out, so the measurable
wins are the operator experience (Track E, not timed here) plus two algorithm
kernels that run on **this** hardware. C5/C6/B5 are research/HW-gated — measured
kernels, honest non-wins on one desktop.

### 7.1 AVX2 batched modular inverse — measured on the silicon (B4)

The first fork SIMD kernel that runs **natively** (Comet Lake has AVX2). The
siever's per-prime 32-bit modular inverse, 8-way AVX2 masked binary-GCD
(`bench/avx2-modinv.c`):

| | ns / inverse | over 2^20 inverses |
|---|---|---|
| scalar binary-GCD | ~193 | 0.203 s |
| **AVX2 8-way** | **~42** | 0.044 s |
| **speedup** | **~4.6×** | bit-exact vs GMP (0/320000) |

Honest: Amdahl-bounded end-to-end — the siever's byte-scatter majority (~29 %)
stays scalar, so the whole-siever ceiling is ~1.05–1.10× even at 4.6× on this slice.

### 7.2 GPU root-sieve (C5) & GPU GF(p) lingen NTT (C6)

- **C5** (`bench/gpu-ropt-stage2.cu`): the stage-2 root sieve
  (`rootsieve_run_line`) as an int32-accumulate scatter, **bit-exact vs the int16
  CPU reference (0 wrong)** over a 4 M-cell line; ~1.7× on the raw apply step but an
  **honest wash** at testable sizes (real `ropt` sieves small per-rotation arrays →
  PCIe/launch-bound). Win is large-N only.
- **C6** (`bench/gpu-lingen-ntt.cu`): iterative Cooley–Tukey NTT polynomial multiply
  over a 31-bit NTT prime, **bit-exact vs schoolbook (0/1199)**, ~0.5 ms for a
  degree-2^16 × degree-2^16 product (NTT size 2^17). The single-prime inner
  transform of a multi-modular GF(p) lingen; lingen is ~3–8 % of BWC so <1 %
  single-machine net — multi-GPU/cluster DLP only.

### 7.3 AVX2/Galois/IFMA correctness (gated)

- **B5** (`bench/ifma-gfp.c`, under Intel SDE): the IFMA GF(p) routing path —
  radix-2^64↔2^52 bridge + the `vec_add_dotprod` `+w` addend — **bit-exact vs GMP,
  0/32000** (260-bit, 8-way). Perf HW-gated (no IFMA silicon) + repack-sensitive.
- **A5** (`scripts/cadofactor/galois.py`): exact automorphism detection,
  cross-validated against CADO's `tests/sieve/galois.poly` (`autom2.2`); the
  matrix/sieve reduction is CADO's upstream `--galois`.

---

## 8. Reproducing

```bash
# one-time: create the Flask/requests venv the 3.0.0 orchestrator needs
bash scripts/setup-venv.sh
PY=cado-nfs.venv/bin/python3

# 1. CPU factorization (default build) — exact seeded inputs used above
$PY ./cado-nfs.py 218874463111634589510199972681714178136600659532376772034259 server.ssl=no -t 20            # c60
$PY ./cado-nfs.py 1303194040226516848020750679655954294201842794411873471660917321538889 server.ssl=no -t 20  # c70
$PY ./cado-nfs.py 13719034522081971984611388445022948804646613410389445937337686473743642720735633 server.ssl=no -t 20  # c80
$PY ./cado-nfs.py 298486368711190085093354660667346905640598024729851236663480292985820275742836529265711663 server.ssl=no -t 20  # c90

# 2. siever microbench (deterministic)
bash bench/las-microbench.sh build/$(hostname)/sieve/las 3

# 3-4. GPU kernels (need CUDA). Standalone — no CADO build required:
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-prefactor-bench.cu -lgmp -o gpu-prefactor-bench && ./gpu-prefactor-bench
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu -o gpu-spmv-bench && ./gpu-spmv-bench
# end-to-end GPU linalg (needs an -DENABLE_GPU=ON build):
CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1 $PY ./cado-nfs.py <c90> tasks.linalg.bwc.mm_impl=gpu -t 8

# 5. AVX-512 kernels under Intel SDE (auto-detects /opt/intel-sde/sde64)
bash bench/vpclmul-validate.sh      # gf2x mul_1_n + mul2/3/4 (3.1.0 + B2)
bash bench/ifma-validate.sh         # IFMA modmul + GF(p) plain-rep (3.1.0 + B3)
bash bench/avx512-modinv-validate.sh  # B1 siever batched modular inverse

# 6. v3.2.0 GPU additions (need CUDA; standalone, no CADO build):
nvcc -arch=sm_86 -O3 bench/gpu-ecm-edwards.cu -o gpu-ecm-edwards && ./gpu-ecm-edwards          # A2
nvcc -arch=sm_86 -O3 bench/gpu-batch-smooth.cu -lgmp -o gpu-batch-smooth && ./gpu-batch-smooth  # C3
nvcc -arch=sm_86 -O3 -Xcompiler -fopenmp bench/gpu-sieve-scatter.cu -o gpu-sieve-scatter && ./gpu-sieve-scatter  # C4
# multi-GPU partition end-to-end (needs -DENABLE_GPU=ON), bit-exact product==N:
CADO_GPU_NPART=2 $PY ./cado-nfs.py <c90> tasks.linalg.bwc.mm_impl=gpu -t 8       # D1

# 7. v3.3.0 additions
bash bench/avx2-modinv-validate.sh                                              # B4 (native, AVX2)
nvcc -arch=sm_86 -O3 bench/gpu-ropt-stage2.cu -o gpu-ropt-stage2 && ./gpu-ropt-stage2     # C5
nvcc -arch=sm_86 -O3 bench/gpu-lingen-ntt.cu -o gpu-lingen-ntt && ./gpu-lingen-ntt        # C6
bash bench/ifma-validate.sh                                                     # B5 (extended; SDE)
$PY ./cado-nfs.py --doctor <N>            # E5 preflight; --galois-detect FILE   # A5
```

_CPU/3.1.0 numbers re-confirmed 2026-06-06; the §6 3.2.0 additions measured
2026-06-07 (CADO-NFS 3.2.0-modern) on the machine above. Re-run on your own
hardware to recalibrate; `-t <n>` sets the thread count, `-DENABLE_GPU=ON` in
`local.sh` builds the GPU backend._
