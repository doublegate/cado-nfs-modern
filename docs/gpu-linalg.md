# GPU linear algebra (Block Wiedemann SpMV) — v3.1.0-modern, Track 2.2

The linear-algebra (matrix) step is the **fastest-growing** phase of NFS: in the
3.0.0-modern benchmarks its CPU time grows ~110× from c60 to c90 (vs ~60× for
sieving; see `BENCHMARKS.md`), so it becomes the second bottleneck as numbers
grow and is the natural target for new compute. Its kernel is a **sparse
matrix × block-of-vectors product (SpMV)** over GF(2), run thousands of times by
Block Wiedemann (`linalg/bwc`). This track puts that kernel on the GPU.

## The operation (matched bit-exact)

From `linalg/bwc/matmul-basic.cpp`, the SpMV for a non-transposed matrix×vector
is, per output row `i`, an XOR-accumulate over that row's nonzero columns:

```
dst[i] = XOR over { j : (i,j) nonzero }  of  src[j]
```

Each vector element is a **bitsliced block of K 64-bit limbs** — `b64` (K=1, 64
vectors at once) or `b128` (K=2, 128 vectors). The matrix is rows of column
indices (implicit 1 coefficients over GF(2)); `matmul-basic`'s `q` array is
`[len₀, col, …, len₁, col, …]`, equivalent to CSR (`rowptr`, `col`).

## Validated GPU kernel + benchmark

`bench/gpu-spmv-bench.cu` implements the GPU SpMV (one thread per output row,
XOR-accumulating `src[col]` over the row, K limbs per element) and **validates it
bit-exact** against the same CPU loop, then benchmarks both (CPU parallelized
across all cores). The matrix stays resident on the GPU (BWC reuses it across
thousands of iterations), so only the kernel is timed. On an RTX 3090 vs a
20-thread i9-10850K, synthetic matrices (~30 nonzeros/row):

| Block | Matrix | Validation | GPU | CPU (20 thr) | Speedup |
|---|---|---|---:|---:|---:|
| b64 (64 vec) | 2.0 M rows, 60 M nnz | **PASS** | 7.9 Gnz/s | 1.27 Gnz/s | 6.2× |
| b128 (128 vec) | 2.0 M rows, 60 M nnz | **PASS** | 5.1 Gnz/s | 0.33 Gnz/s | 15.4× |
| b256 (256 vec) | 0.5 M rows, 30 M nnz | **PASS** | 5.2 Gnz/s | 1.09 Gnz/s | 4.8× |

```bash
nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu -o gpu-spmv-bench && ./gpu-spmv-bench
```

## Measured against CADO's real `bucket` backend (the honest comparison)

The table above compares the GPU to the **naive `matmul-basic` loop**, which
overstates the win. Measured directly with `bench_matcache` on a **real
1M×1M GF(2) matrix (30M nonzeros)** from `random_matrix`, **single CPU thread**:

| backend | ns/coeff | Gnz/s | vs naive |
|---|---:|---:|---:|
| `basic` (naive loop) | ~2.8 | 0.35 | 1.0× |
| `sliced` | ~1.5 | 0.67 | 1.9× |
| **`bucket` (production default)** | ~1.58 | **0.63** | **1.8×** |

So CADO's production `bucket` is ~1.8× the naive loop single-threaded. And it
**barely scales with cores**, because SpMV is memory-bandwidth-bound — measured
with `bench_matcache --nthreads N` (one ~6M-nnz submatrix per thread) on the
i9-10850K (10C/20T):

| threads | aggregate Gnz/s | vs 1 thread |
|---:|---:|---:|
| 1 | 0.85 | 1.0× |
| 2 | 1.26 | 1.5× |
| 4 | 1.51 | 1.8× |
| 8 | 1.59 | 1.9× |
| 16 | 1.81 | 2.1× |
| **20** | **1.81** | **2.1×** |

**Full-CPU `bucket` saturates at ~1.8 Gnz/s** (only ~2.1× the single thread —
bandwidth-bound by ~N=4). Against that, the GPU b64 kernel at **7.9 Gnz/s is a
measured ~4.4× the full production CPU** — a real single-machine win, well below
the inflated "6–15× vs naive" but solidly above bandwidth parity. And it is a
*floor*: the un-tuned kernel realizes only ~10% of the 3090's ~936 GB/s, so a
coalesced kernel should widen the gap.

**The GPU's biggest advantage is still at *scale*:** aggregate bandwidth across
many GPUs/nodes and matrices too large for one machine's RAM (the multi-GPU/MPI
path below) — but even on one desktop the kernel is ~4× the tuned CPU backend.

## Other caveats

- **Synthetic matrix** for the GPU table: random CSR has worse locality than a
  real filtered/balanced BWC matrix (columns reordered for cache reuse). The real
  comparison can move either way; the `bucket` numbers above *are* on a real
  matrix.
- **Kernel un-tuned.** One-thread-per-row hits ~10% of the 3090's ~936 GB/s peak
  (uncoalesced `src[col]` gather). ELL/sliced formats, column sorting for reuse,
  and shared-memory `src` caching are known wins not yet applied — GPU headroom.
- **Memory.** The matrix must fit in GPU memory (24 GB on a 3090 → roughly up to
  ~c150-scale); larger needs the multi-GPU/multi-node path below.

## The `matmul_bNN_gpu` backend (implemented + validated)

`linalg/bwc/matmul-gpu.cu` is a real GPU backend plugged into BWC's matmul
dispatch — selected with `mm_impl=gpu`, registered through the same
`CONFIGURE_MATMUL_LIB` / `COOKED_BWC_BACKENDS` machinery as `basic`/`bucket`,
built only with `-DENABLE_GPU=ON` (compiled by nvcc; `b64` and `b128`). It mirrors
`matmul-basic`'s cache format, keeps **both the matrix and its transpose resident
on the device as CSR** (so both BWC directions are fast one-thread-per-row
gathers), and copies only the `src`/`dst` vectors per call.

- **Bit-exact:** passes `bench_matcache`'s consistency check
  (`(M·v₁)·v₂ == (Mᵀ·v₂)·v₁`) — **all 4 checks pass** for both directions on the
  real 1M×1M matrix.
- **Measured (real backend, incl. per-call vector transfers):** BWC reuses the
  same host vector buffers every iteration, so the backend **page-locks (pins)
  each one the first time it sees it** — all subsequent H2D/D2H transfers then run
  at full PCIe bandwidth. That lifted it from 4.95 → **6.76 Gnz/s** on the RTX
  3090 — **~11× a single `bucket` thread** and **~3.75× the full-CPU `bucket`**
  (1.8 Gnz/s). The residual ~1 ms/iter is the unavoidable `src`+`dst` (~16 MB)
  still crossing PCIe each call; only **full vector residency** (vectors never
  leaving the GPU, exchanged only across MPI — an `mmt_vec`-layer change above the
  backend) removes it, toward the ~7.9 Gnz/s kernel-only ceiling.

## Next increments

1. **Pinned host transfers — done.** The backend pins each reused BWC vector once,
   so transfers run at full PCIe speed (4.95 → 6.76 Gnz/s). The remaining win
   needs **full vector residency** (hold the `mmt_vec` data on the GPU across all
   iterations, copy out only for the MPI exchange) — an `mmt_vec`-layer change
   above the backend, removing the last ~1 ms/iter.
2. **Kernel tuning**: coalesced/ELL layout, column sorting, shared-memory `src`
   caching (the kernel realizes only ~10% of peak bandwidth today).
3. **Multi-GPU / multi-node**: BWC already splits the matrix across an `nh×nv`
   MPI grid (`balancing_workhorse`), each rank owning a submatrix; the GPU backend
   slots in at each rank's local `mm->mul()`. One GPU per rank → multi-GPU on a
   node and multi-node via the unchanged MPI comm layer — the natural HPC
   scale-out and where the GPU's aggregate-bandwidth advantage compounds.

## Status

- **Done & validated:**
  - the GF(2) SpMV GPU kernel (b64/b128/b256), bit-exact vs the CPU reference
    (`bench/gpu-spmv-bench.cu`);
  - the honest full-CPU baseline — `bucket` saturates at ~1.8 Gnz/s (20 threads,
    bandwidth-bound), so the kernel-only GPU win is ~4.4×;
  - **a real `matmul_bNN_gpu` backend** (`linalg/bwc/matmul-gpu.cu`) plugged into
    BWC's dispatch (`mm_impl=gpu`), passing `bench_matcache`'s bit-exact check
    (4/4, both directions), with **pinned host-vector transfers** — **6.76 Gnz/s,
    ~3.75× the full-CPU `bucket`**.
- **Next:** full vector residency (eliminate the residual per-call transfer →
  toward the ~7.9 Gnz/s kernel ceiling), kernel tuning, then multi-GPU/MPI wiring
  where the GPU's aggregate-bandwidth advantage at scale lives.
