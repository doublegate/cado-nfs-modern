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

## Honest caveats (read before believing the table)

- **The CPU baseline is the *naive* `matmul-basic` loop, not CADO's production
  `bucket` backend.** `bucket` is cache-blocked and bucket-sorted and is
  materially faster than `basic`, so the speedup vs CADO's *actual* CPU linear
  algebra is **lower** than the table above. The honest next step is to benchmark
  the GPU kernel against `bench_matcache --impl bucket` on a real BWC matrix —
  until then, treat these numbers as "GPU vs the reference loop", an upper bound
  on the production win.
- **Synthetic matrix.** Random CSR has worse locality than a real
  filtered/balanced BWC matrix (whose columns are reordered for cache reuse), but
  also lacks the structure a tuned CPU backend exploits — the real comparison can
  move either way and must be measured.
- **Bandwidth-bound, kernel un-tuned.** SpMV is memory-bound; the one-thread-per-
  row kernel hits ~10% of the 3090's ~936 GB/s peak because the `src[col]` gather
  is uncoalesced. ELL/sliced formats, sorting columns for reuse, and caching `src`
  in shared memory are known wins not yet applied — so the GPU number has
  headroom too.
- **Memory.** The matrix must fit in GPU memory (24 GB on a 3090 → roughly up to
  ~c150-scale); larger matrices need the multi-GPU/multi-node path below.

## Integration path (next increments)

1. **A `matmul_bNN_gpu` backend** implementing `matmul_interface`
   (`build_cache`/`reload_cache`/`mul`) behind `matmul_interface::create`
   (`linalg/bwc/matmul.cpp` dispatch), keeping the CSR matrix **resident on the
   device** and copying only the `src`/`dst` vectors per iteration (or keeping
   them resident too and exchanging only across MPI). Selected with
   `mm_impl=gpu`.
2. **Bit-exact gate** via the existing `bench_matcache` check
   ((M·v₁)·v₂ == (Mᵀ·v₂)·v₁) plus a real-matrix run vs `bucket`.
3. **Multi-GPU / multi-node**: BWC already splits the matrix across an `nh×nv`
   MPI grid (`balancing_workhorse`), each rank owning a submatrix. The GPU
   backend slots in at each rank's local `mm->mul()`; one GPU per rank gives
   multi-GPU on a node and multi-node via the unchanged MPI comm layer — the
   natural HPC scale-out.

## Status

- **Done & validated:** the GF(2) SpMV GPU kernel (b64/b128/b256), bit-exact vs
  the CPU reference, with a benchmark (`bench/gpu-spmv-bench.cu`).
- **Next:** benchmark vs the real `bucket` backend on a true BWC matrix; then the
  `matmul_bNN_gpu` backend + the multi-GPU/MPI wiring.
