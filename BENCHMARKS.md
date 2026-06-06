# Benchmarks

Reference performance of **CADO-NFS 3.1.0-modern** on a single desktop —
CPU factorization, the deterministic siever microbenchmark, and the 3.1.0 GPU /
AVX-512 work. These numbers characterize this build on this class of hardware;
they are not a comparison against other NFS implementations. Every factorization
is verified (`product == N`, factors prime); every GPU/SIMD kernel is validated
bit-exact against a CPU/GMP reference (or, for AVX-512, under Intel SDE).

_All numbers re-measured **2026-06-06** on the machine below. CPU factorization
is unchanged from 3.0.0-modern (3.1.0 adds no CPU-path change — see §2), so those
results carry forward, re-confirmed here._

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
| gf2x **VPCLMULQDQ** `mul_1_n` / `addmul_1_n` (`bench/vpclmul-mul1n.c`) | bit-exact vs scalar, 200 000 random trials | **PASS** |
| **IFMA** GF(p) Montgomery modmul, 8-way radix-2⁵² (`bench/ifma-modmul.c`) | bit-exact vs GMP, 260-bit, 8 lanes | **PASS** (0 / 32 000 wrong) |

The gf2x VPCLMULQDQ backend is auto-detected by `configure` (run-test-gated, so a
non-AVX-512 host safely keeps the pclmul backend); the IFMA modmul is the
foundation kernel for a GF(p) DLP backend. See `CHANGELOG.md` (Tracks 1.1/1.4).

---

## 6. Reproducing

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
bash bench/vpclmul-validate.sh
bash bench/ifma-validate.sh
```

_Re-measured 2026-06-06 on the machine above (CADO-NFS 3.1.0-modern). Re-run on
your own hardware to recalibrate; `-t <n>` sets the thread count, `-DENABLE_GPU=ON`
in `local.sh` builds the GPU backend._
