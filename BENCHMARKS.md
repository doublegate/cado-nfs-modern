# Benchmarks

Reference performance of **CADO-NFS 3.0.0-modern** factoring balanced
(RSA-like) semiprimes on a single desktop. These numbers characterize this
build on this class of hardware; they are not a comparison against other NFS
implementations.

## Test machine

| Component | Detail |
|-----------|--------|
| CPU | Intel Core i9-10850K — 10 cores / 20 threads, 3.6 GHz base / 5.2 GHz boost (Comet Lake) |
| RAM | 64 GiB DDR4 |
| OS / kernel | CachyOS, Linux 7.0.10 |
| Compiler | GCC 16.1.1, **`-O3 -march=native -mtune=native`** (this fork's `local.sh`) |
| Libraries | GMP 6.3.0, hwloc 2.13.0 |
| Orchestration | Python 3.14.5 (Flask work-unit server) |
| CADO-NFS | 3.0.0-modern (this fork; rebased on upstream 3.0.0) |

## Methodology

- **Inputs:** balanced semiprimes `n = p·q` with `p`, `q` of equal digit length
  (the hardest case for NFS). The exact inputs are listed under *Reproducing* —
  they are the **same seeded inputs** used for the 2.3.1-modern benchmarks, so
  the comparison below is apples-to-apples on identical numbers.
- **Command:** `cado-nfs.venv/bin/python3 ./cado-nfs.py <n> server.ssl=no -t 20`
  — all 20 logical threads; TLS transport disabled to isolate compute (it adds
  <1 s of cert setup and is irrelevant to the result either way).
- **Timing:** wall-clock measured around the process; CPU and elapsed times as
  reported by CADO-NFS (`Total cpu/elapsed time for entire ...`). `parallel`
  = total CPU ÷ elapsed (effective speedup across the whole run).
- **Runs:** one run per size. NFS timing has inherent ~±15-20 % variance, mostly
  from randomized polynomial selection; treat these as representative, not exact.
- Every factorization was verified: the reported factors multiply back to `n`
  and each is prime.

## Results

| Digits | Bits | Wall time | CADO CPU | Parallel | Status |
|-------:|-----:|----------:|---------:|---------:|:------:|
| 60 | ~199 | 19.2 s | 40.0 s | 2.1× | verified |
| 70 | ~232 | 26.0 s | 87.5 s | 3.4× | verified |
| 80 | ~265 | 70.3 s | 379.1 s | 5.4× | verified |
| 90 | ~299 | 184.5 s | 1465.0 s | 8.0× | verified |

### Per-phase CPU time (seconds)

CADO-reported CPU per stage (all aggregate CPU-seconds across threads). These
sum to ~99 % of the headline CPU; the small remainder is factor-base generation,
free relations, and un-itemized setup. Unlike the 2.3.1-modern table, polynomial
selection is shown as CPU here (CADO 3.0.0 reports it as an aggregate "Total
time"), so it is comparable to the other columns.

| Digits | Polyselect | Lattice sieving | Filtering | Linear algebra | Square root |
|-------:|-----------:|----------------:|----------:|---------------:|------------:|
| 60 |  4.2 |   17.9 |  9.5 |   2.1 |  5.4 |
| 70 |  8.9 |   53.5 | 12.4 |   4.0 |  6.7 |
| 80 | 19.3 |  273.5 | 29.7 |  31.9 | 21.5 |
| 90 | 50.4 | 1080.6 | 69.0 | 231.5 | 22.2 |

("Filtering" sums dup1/dup2/purge/merge/replay.)

## Versus 2.3.1-modern (same inputs)

The prior `2.3.1-modern` fork (upstream 2.3.0 base, `-O2 -fcommon`) on the same
machine and the **same seeded inputs**:

| Digits | Wall: 2.3.1 → 3.0.0 | CADO CPU: 2.3.1 → 3.0.0 | Parallel: 2.3.1 → 3.0.0 |
|-------:|---------------------|--------------------------|--------------------------|
| 60 | 30.6 s → **19.2 s** (−37 %) | 57.8 → **40.0** (−31 %) | 1.9× → 2.1× |
| 70 | 35.4 s → **26.0 s** (−27 %) | 121.8 → **87.5** (−28 %) | 3.5× → 3.4× |
| 80 | 73.9 s → **70.3 s** (−5 %)  | 558.0 → **379.1** (−32 %) | 7.6× → 5.4× |
| 90 | 175.3 s → 184.5 s (+5 %)    | 1942.7 → **1465.0** (−25 %) | 11.1× → 8.0× |

**The robust, repeatable signal is total CPU work, down ~25-32 % across the
board.** Two upstream-3.0.0 changes drive most of it — the Bouvier–Imbert batch
cofactorization (eprint 2018/669) and `I>16` sieving — compounded by this fork's
`-O3 -march=native` (~7 % on the siever; see `CHANGELOG.md`).

Two honest caveats:

- **Wall-time gains shrink with size and fall inside the ±15-20 % polyselect
  noise at c80-c90** (c90 even reads +5 %). The small-size wall wins
  (c60 −37 %, c70 −27 %) partly reflect that variance; the CPU reduction is the
  number to trust.
- **Parallel efficiency *drops*** (e.g. c90 11.1× → 8.0×) precisely *because*
  total CPU fell: when the embarrassingly-parallel sieve does less work, the
  sequential phases (linear algebra, square root, Python orchestration) become a
  larger fraction of the run, so the CPU-÷-elapsed ratio declines even though the
  wall clock is comparable or better. Lower parallel speedup here is a *good*
  sign — it means less wasted sieving, not a regression.

## Observations

- **Sieving still dominates** (45-74 % of CPU) and is the embarrassingly-parallel
  phase. Its share now falls at the largest size as linear algebra grows.
- **Linear algebra grows the fastest** of any phase: 2.1 → 4.0 → 31.9 → 231.5
  CPU-s from c60 to c90 (~110×, vs ~60× for sieving over the same span). Block
  Wiedemann scales worse than sieving and is the clear emerging second
  bottleneck — the classic NFS trade-off.
- **Growth rate.** Wall-time roughly doubles per +10 digits in this range; CPU
  work grows ~3.5-4× per decade, matching the sub-exponential `L(1/3)`
  complexity of NFS.

## Rough projections (this machine)

Extrapolated from the measured CPU growth and rising parallel efficiency.
**Order-of-magnitude only** — real numbers depend on parameter tuning, memory
bandwidth, and luck in polynomial selection.

| Digits | Projected wall time | Practicality |
|-------:|--------------------:|--------------|
| ≤ 75 | < 30 s | interactive |
| 80-95 | 1-4 min | coffee break |
| 100 | ~8-12 min | single sitting |
| 110 | ~40-80 min | long session |
| 120 | a few hours | overnight |
| ≥ 130 | a day or more | wants a cluster / distributed mode |

For this 10-core desktop, **up to ~c105-c110 is comfortable in one session**;
beyond ~c120, use multiple machines (CADO-NFS distributed mode) or expect
multi-day runs. RAM (64 GiB) is not the limit in this range.

## Reproducing

```bash
# one-time: create the Flask/requests venv the 3.0.0 orchestrator needs
bash scripts/setup-venv.sh
PY=cado-nfs.venv/bin/python3

# exact seeded inputs used above
$PY ./cado-nfs.py 218874463111634589510199972681714178136600659532376772034259 server.ssl=no -t 20            # c60
$PY ./cado-nfs.py 1303194040226516848020750679655954294201842794411873471660917321538889 server.ssl=no -t 20  # c70
$PY ./cado-nfs.py 13719034522081971984611388445022948804646613410389445937337686473743642720735633 server.ssl=no -t 20  # c80
$PY ./cado-nfs.py 298486368711190085093354660667346905640598024729851236663480292985820275742836529265711663 server.ssl=no -t 20  # c90
```

_Re-measured 2026-06-05 on the 3.0.0-modern build (the machine above). Re-run on
your own hardware to recalibrate; `-t <n>` sets the thread count._

---

## GPU pre-factoring ECM — CPU vs GPU (v3.1.0-modern, Track 2.1)

Throughput of the multi-precision ECM (stage-1 + stage-2 BSGS) that powers the
GPU pre-NFS factoring front-end (`misc/gpu_prefactor`). The **same**
`__host__ __device__` `ecm_run2` runs on both sides — the CPU side parallelized
across all 20 threads with `std::thread` — so this is an apples-to-apples
algorithm comparison, not a comparison against a different ECM. `B1=50000`,
`B2=5e6`, **RTX 3090 vs the full i9-10850K** (`bench/gpu-prefactor-bench.cu`).

| Modulus width | GPU (curves/s) | CPU, 20 threads (curves/s) | GPU speedup |
|---|---:|---:|---:|
| 128-bit (≤~38 digit N) | 16 844 | 342 | **49.3×** |
| 256-bit (≤~77 digit N) | 3 058 | 120 | **25.5×** |
| 512-bit (≤~154 digit N) | 332 | 31 | **10.8×** |

**Honest reading:** the GPU's advantage **shrinks as the modulus widens** —
wider K-limb arithmetic uses more registers/local memory per thread, lowering GPU
occupancy. For NFS-sized inputs (≥85 digits → 512-bit width) it is ~11× the whole
CPU; for the smaller cofactors that arise after partial stripping, 25–49×. Either
way the pre-factoring stage is a real GPU win (unlike in-sieve cofactorization —
see `docs/gpu-cofactorization.md`), because it is a *separate* stage with no
Amdahl ceiling. End-to-end, a 90-digit N with a 14-digit factor is fully
resolved by `cado-nfs.py --gpu-prefactor` in seconds, skipping NFS entirely.

```bash
# build the GPU pre-factoring tool + benchmark (needs CUDA)
nvcc -arch=sm_86 -O3 misc/gpu_prefactor/gpu-prefactor.cu -lgmp -o gpu-prefactor
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-prefactor-bench.cu -lgmp -o gpu-prefactor-bench
./gpu-prefactor <N> staged 30        # escalating-B1 schedule up to ~30-digit factors
./gpu-prefactor-bench                 # the table above
```

_Measured 2026-06-05 on the RTX 3090 + i9-10850K. The pre-factoring ECM math is
validated bit-exact (`bench/gpu-ecm-mp.cu`)._

## GPU linear algebra (BWC SpMV) at scale (v3.1.0-modern, Track 2.2)

The GPU GF(2) SpMV backend (`mm_impl=gpu`, `linalg/bwc/matmul-gpu.cu`) is the
kernel Block Wiedemann runs thousands of times, and the fastest-growing NFS phase
(linalg ~110× c60→c90). The thesis of this fork's GPU-linalg work is that the win
**grows with N** — so the headline is a scaling sweep, not a single point.

**Scaling sweep** (`bench/gpu-spmv-bench.cu`, b64 = 64 vectors, ~30 nnz/row,
bit-exact warp kernel `PASS` at every size; RTX 3090 vs the i9-10850K). Rows are
representative of c100–c120 linalg matrices:

| ~size | rows | nnz | GPU warp (Gnz/s) | CPU ref loop, 20 thr (Gnz/s) | GPU vs ref loop |
|---|---:|---:|---:|---:|---:|
| c100 | 1.0 M | 30 M | 28.9 | 1.58 | 18× |
| c110 | 2.0 M | 60 M | 12.3 | 0.76 | 16× |
| c115 | 4.0 M | 120 M | 9.2 | 0.24 | 38× |
| c120 | 8.0 M | 240 M | 7.9 | 0.19 | **41×** |

**Honest reading.** Two effects compound. (1) The GPU's absolute throughput
*falls* as the matrix grows (28.9→7.9 Gnz/s) — at 1 M rows it largely fits cache;
at 8 M rows / 240 M nnz it is memory-latency-bound on uncoalesced `src[col]`
gathers off random CSR. (2) The CPU reference loop falls *much faster*
(1.58→0.19 Gnz/s) because random-CSR SpMV thrashes cache once the matrix exceeds
LLC. So the GPU's **relative** advantage widens with N — the central claim. The
CPU column here is the threaded reference loop; against CADO's *tuned* `bucket`
backend (which reorders columns for locality and **saturates at ~1.8 Gnz/s** on
this CPU, measured separately in `docs/gpu-linalg.md`), the GPU warp kernel at
c120-scale is a **steadier ~4.4×** — a real single-machine win, and a floor (the
kernel realises only a fraction of the 3090's ~936 GB/s; the biggest gains are at
aggregate multi-GPU/multi-node bandwidth and out-of-core matrices).

**End-to-end anchor (real factorization, `product == N`).** A 90-digit GNFS run
with `tasks.linalg.bwc.mm_impl=gpu` + full vector residency
(`CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1`, `-t 8`) drove the whole BWC pipeline
(krylov → lingen → mksol → gather) through the GPU backend and returned the
correct factors. At c90 the matrix is small — bwc total **8.18 s real** (krylov
2.65 s / 2300 iters, lingen 2.14 s, mksol 1.72 s) out of a ~349 s factorization
(sieving-dominated) — so GPU vs CPU linalg is immaterial *at this size*; the point
of the anchor is end-to-end correctness of the GPU path, and the sweep above is
where the size-driven win lives. (A standalone c59 GPU run also returns
`product == N`, used as the fast residency regression.)

```bash
# GPU SpMV scaling sweep (needs CUDA; matrices are synthetic, generated in-process)
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu -o gpu-spmv-bench
./gpu-spmv-bench
# real end-to-end GPU linalg (needs -DENABLE_GPU=ON build)
CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1 \
  cado-nfs.venv/bin/python3 ./cado-nfs.py <c90> tasks.linalg.bwc.mm_impl=gpu -t 8
```

_Measured 2026-06-06 on the RTX 3090 + i9-10850K (CUDA 13.3, sm_86). SpMV kernel
bit-exact vs the CPU reference at every size; end-to-end `product == N`._
