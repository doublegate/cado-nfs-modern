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
- **End-to-end: validated in a real factorization.** A full
  `cado-nfs.py … tasks.linalg.bwc.mm_impl=gpu` run (59-digit, `thr=2x2`) drives
  the entire BWC pipeline (krylov → lingen → mksol → gather, balanced matrix,
  multi-thread comm) through the GPU backend and returns the correct factors
  (**product == N**) — confirmed by the `mm_impl=gpu` bwc command line and the
  per-instance `matmul-gpu split` timing in the bwc logs. The backend is
  production-usable, not just a bench.
- **Measured (real backend, incl. per-call vector transfers):** BWC reuses the
  same host vector buffers every iteration, so the backend **page-locks (pins)
  each one the first time it sees it** — all subsequent H2D/D2H transfers then run
  at full PCIe bandwidth. That lifted it from 4.95 → **6.76 Gnz/s** on the RTX
  3090 — **~11× a single `bucket` thread** and **~3.75× the full-CPU `bucket`**
  (1.8 Gnz/s). The residual ~1 ms/iter is the unavoidable `src`+`dst` (~16 MB)
  still crossing PCIe each call; only **full vector residency** (vectors never
  leaving the GPU, exchanged only across MPI — an `mmt_vec`-layer change above the
  backend) removes it, toward the ~7.9 Gnz/s kernel-only ceiling.

## Where the time goes (measured), and the residency decision

Split-timed in the backend (`CADO_GPU_TIMING=1`, pinned, b64, 30M nnz, RTX 3090):

| segment | per-SpMV | share |
|---|---:|---:|
| H2D `src` | 1.81 ms | 36% |
| **kernel** | 2.63 ms | 52% |
| D2H `dst` | 0.64 ms | 12% |
| — transfers total | 2.45 ms | **48%** |

So even after pinning, **~half the time is the per-call `src`/`dst` transfers** —
full vector residency is a genuine **~2× opportunity** (~13 Gnz/s → ~7× the full
CPU). The catch is *what* it takes, established by reading the BWC vector layer:

**Measured residency-win ceiling at scale (real factorizations, `-t 8`, coalesced
kernel, RTX 3090).** Aggregating the `CADO_GPU_TIMING` split over the *main krylov
loop* (the highest-count SpMV class, i.e. the steady iterations that dominate):

| N | main-loop SpMV | H2D / kernel / D2H (ms) | transfer share | SpMV speedup if resident |
|---|---:|---|---:|---:|
| c70 | 3584 | 0.066 / 0.069 / 0.041 | **61%** | **2.55×** |
| c80 | 8192 | 0.099 / 0.092 / 0.050 | **62%** | **2.61×** |

The transfer share is **stable-to-rising with N** (per-thread submatrices shrink, so
the kernel falls while the vector transfers hold), so residency — which removes the
H2D+D2H of the steady iterations — is a measured **~2.6× on the SpMV hot loop**,
growing with problem size. (Wall-clock impact on a whole factorization scales with
linalg's share of the run — small for c70/c80 where sieving dominates, large for
c100+/DLP where linalg dominates; that large-N/DLP regime is the target.)

- **Every SpMV is bracketed by host-memory work that can't be skipped at the
  backend level.** After each `matmul_top_mul_cpu` (the `mm->mul` seam,
  `matmul_top.cpp:762`) comes `mmt_vec_allreduce` / `matmul_top_mul_comm`
  (`matmul_top_comm.cpp`), whose reduce/broadcast call `MPI_Allreduce` /
  `MPI_Reduce_scatter_block` / `MPI_Allgather` **directly on `vec->v` host
  memory** and do thread-local `vec_add_and_reduce`/`vec_set` on host. And every
  inner iteration of `krylov.cpp` calls `x_dotprod(... ymy[0] ...)` which **reads
  the vector on the host** before the SpMV.
- Therefore a device-resident vector must, every iteration, be on the host for
  the dot-product and for the comm — so **transfers can only be removed by moving
  the dot-product, the thread reduction, the broadcast, and (for >1 node) the MPI
  exchange onto the GPU too.** That is a port of BWC's vector layer, not a backend
  tweak, and any missed sync silently corrupts the result.

**Decision:** this is worth ~2× but is a deliberate multi-step project, not a safe
single increment. The honest sequence:

1. **A device-resident `mmt_vec` shadow** + `matmul_interface` sync hooks
   (`to_device`/`from_device`), backend keeps the authoritative device copy.
2. **GPU `x_dotprod`** (sparse `x` · device vector) so the per-iteration dot
   product no longer pulls the vector to the host. **Kernel built + validated
   bit-exact** (`bench/gpu-xdotprod-bench.cu`, ALL PASS) — ready to wire in.
3. **GPU intra-node reduction/broadcast** for the single-node multi-thread comm
   (the common single-machine case), keeping vectors on-device across the inner
   loop; sync to host only per-interval (twist/save) and for real MPI. **Kernels
   built + validated bit-exact** (`bench/gpu-vecreduce-bench.cu`: GF(2) XOR-reduce
   + broadcast, ALL PASS, ~440 GB/s) — ready to wire in.
4. **Multi-node**: GPU-direct (CUDA-aware) MPI, or host-staged exchange.

Approach: each device-side primitive is built and validated **standalone first**
(no risk to real runs), then wired into the resident loop, with a full verified
factorization (`product == N`) as the gate at integration. **All three primitives
are ready and bit-exact**: the coalesced SpMV kernel, `x_dotprod`, and the
intra-node reduce/broadcast.

### Wiring — the device-residency control plane (done, `product == N`)

`matmul_interface` gains a `host_vector_modified(hostptr)` hook (no-op for CPU
backends). The GPU backend keeps a **persistent per-host-vector device buffer
pool** with a `current` flag; with `CADO_GPU_VECRESIDENT=1` a current buffer lets
`mul()` skip the H2D upload, and `host_vector_modified` clears the flag.
`matmul_top` calls the hook at every host write — after the comm
(`mmt_vec_allreduce` / `matmul_top_mul_comm`) and after `mmt_vec_twist/untwist`.
**Both modes verified by a real factorization** (`tasks.linalg.bwc.mm_impl=gpu`,
59-digit): default mode is bit-identical, and **resident-skip mode also returns
`product == N`** — proving the invalidation coverage is complete for the krylov
path (no missed host write).

But the skip is **correct yet inert**: the comm runs after *every* SpMV and
invalidates the src, so it is always re-uploaded — the H2D share does not drop.
This empirically confirms the analysis: **the win requires moving the comm itself
onto the device**, so the vector stays device-authoritative through the comm and
the (already-proven-correct) skip finally triggers.

### Comm-on-device — design, what is built, and the honest blocker

Reading the comm makes the shape precise. `mmt_vec_allreduce`
(`matmul_top_comm.cpp:738`) and `matmul_top_mul_comm` reduce **across sibling
threads' *host* buffers** (`v.sibling(k).v`, chunked by `thread_chunk`,
`vec_add_and_reduce` = XOR, then `vec_set` broadcast), in `bwc_base` — a layer
with **no CUDA and no `mm` handle**. So this is a small rearchitecture, not a
surgical edit:

1. **Process-global device registry** (replaces the per-`mm` pool) — **done,
   `product == N`** (`matmul-gpu.cu`, `g_pool`/`g_pin`/`g_invalidate`): a
   thread-safe `host_ptr → {device buffer, current, host_dirty}` map, since the
   comm touches *siblings'* buffers and all threads share the CUDA context. The
   SpMV uses it as today.
2. **A registered CUDA hook** — **built + validated** (`matmul-gpu-hooks.h`,
   `cado_gpu_comm_reduce_bcast` / `cado_gpu_sync_to_host`; installed by the GPU
   backend's ctor so `bwc_base` keeps no hard CUDA dependency). The hook runs the
   GF(2) XOR-reduce + broadcast **on the device-resident sibling buffers**
   (`vecreduce_inplace` + `vecbroadcast_n`, bit-exact with the host path and with
   `bench/gpu-vecreduce-bench.cu`). It is wired into `mmt_vec_allreduce` behind
   `CADO_GPU_DEVCOMM` and gives **`product == N`** on the c60 (default,
   `DEVCOMM`, and `VECRESIDENT+DEVCOMM`).
3. **`from_device(host_ptr)` syncs** — `cado_gpu_sync_to_host` plumbing is in
   place (the `host_dirty` flag), to be called at the host-read points the inner
   loop keeps: `x_dotprod`, the online `check`, `mmt_vec_save`, `twist/untwist`.
4. **MPI**: for >1 job, stage through host (gated to `njobs == 1` for now).

**Two honest findings from wiring (2):**

- **`allreduce` is *not* the per-iteration hot comm for factoring.** A 1-matrix
  factorization (the common case) goes `matmul_top_mul` → `matmul_top_mul_comm`
  = `mmt_vec_reduce` + `mmt_vec_broadcast` (direction-changing, with the
  reduce-scatter "pack at the beginning" repack), **not** `mmt_vec_allreduce`.
  `allreduce` is hit only by the twist and the prep/secure rank checks. So the
  validated hook proves the device-comm *mechanism* end-to-end, but the
  transfer-saving port must target `reduce`+`broadcast`.
- **`reduce`+`broadcast` is a 2D transpose, not a 1D reduction — and it is now
  ported and validated.** `mmt_vec_allreduce` reduces across the siblings of
  **one** communicator (`wr[v.d]`) and every sibling ends equal — a 1D XOR-reduce.
  `matmul_top_mul_comm`, by contrast, **reduce-scatters along `w`'s direction
  `wr[w.d]` and all-gathers along the *perpendicular* direction `wr[v.d]`**
  (`mmt_vec_reduce_inner` packs into `sibling(0)`; `mmt_vec_broadcast` all-gathers
  over `xwrpals` in the other axis) — the "shuffled product" the code comments
  warn is "not the identity", coupling *all* grid threads' data. Rather than
  re-derive the transpose, `matmul_top_mul_comm_gpu` (#3, `matmul_top_comm.cpp`)
  **mirrors the host algorithm op-for-op at identical byte offsets** on the device
  copies, so the result is bit-for-bit the host comm's by construction: five
  barriered phases per thread (upload `w`; `reduce_inner` XOR via `xor_block`;
  `mmt_vec_reduce` repack via device copy at `mmt_my_own_offset_in_items`; the
  `mmt_vec_broadcast` `own_vec_set2`+`full_vec_set` copies — collapsing to a no-op
  for the common `THREAD_SHARED` source vector; download `v`). Gated on
  `CADO_GPU_DEVCOMM`, single-node only. **Validated `product == N` on the c60
  across `-t 4` (2×2 square) and `-t 8` (2×4 rectangular), 10/10 each, in default,
  `DEVCOMM`, and `VECRESIDENT+DEVCOMM` modes; `compute-sanitizer` memcheck on
  `prep` reports 0 errors.**
  - *Bug found and fixed in the process* (compute-sanitizer): `mul()` pins host
    buffers (`cudaHostRegister`) at *its* size, but the comm copies the full
    vector (larger); CUDA enforces the registered region size, which corrupts the
    context (a flaky "invalid argument" / "Copy larger than memobj size"). Pinning
    is a transfer-speed optimisation that residency makes moot, so `g_pin` now
    skips when `DEVCOMM` is active (copies go pageable — correct; the default path
    keeps pinning and its measured speedup).
- **Full vector residency — done: the steady loop now skips H2D/D2H.** With
  `CADO_GPU_VECRESIDENT` + `CADO_GPU_DEVCOMM`, the krylov inner loop keeps the
  vectors device-resident across `mul → comm → mul`: `mul()` skips its H2D (device
  src current) and D2H (dst left on device, `host_dirty`); the 2D comm skips its
  host upload of `w` and, instead of writing `v` back, marks it device-resident
  (`cado_gpu_dev_mark_resident`). The catch that had to be fixed: `matmul_top_mul`
  invalidates the comm's output after the comm (correct for the host-writing
  no-trust path) — in residency that would discard the device result and force a
  re-upload, so it is now skipped when residency is active.
  - **Scoped to the krylov inner loop** via `cado_gpu_residency_active`
    (set/cleared by `krylov.cpp` around the loop), so prep/secure/twist — which
    overwrite host buffers without invalidation — stay host-authoritative and are
    untouched. The one per-iteration host read (`x_dotprod`) and the loop boundary
    call `cado_gpu_sync_to_host` to materialise the vector; `twist`/`untwist`
    already invalidate the device copy, so the next block re-seeds correctly.
  - **Validated:** `product == N` 45/45 across default, `DEVCOMM`, and
    `VECRESIDENT+DEVCOMM` × `-t 2/4/8`; `compute-sanitizer` memcheck on krylov in
    residency mode reports 0 errors. Transfer counters (`CADO_GPU_STATS`) confirm
    the steady loop skips **D2H 100%** and **H2D ~99%** (only the per-block re-seed
    after the twist remains) — i.e. the per-iteration PCIe transfers (the measured
    ~60% of SpMV time at c70/c80, growing with N) are eliminated, realising the
    ~2.6× SpMV-hot-loop win where linalg dominates (large N / DLP).

So the GPU SpMV **and** its comm now run entirely on device buffers across the
steady iteration. The follow-on pieces are all landed too: **GPU `x_dotprod`**
(the lone surviving per-iteration D2H, now a device gather off the resident
vector) gives the krylov steady loop **zero** per-iteration host transfers; the
same accumulator treatment extends to **mksol** and **secure** (device
`addmul_tiny`). Residency is **gated to the single-node case**
(`pi->wr[0]->njobs == 1 && pi->wr[1]->njobs == 1`) in all three drivers: the
device comm only handles `njobs==1` and the sync-based MPI fallback yields no
transfer win, so under MPI residency cleanly disables and the run takes the
validated GPU-SpMV + host-comm path. The default and DEVCOMM-only paths are
unchanged and bit-exact (the default path keeps host-vector pinning and its
measured speedup).

## Kernel tuning — done (coalesced warp-per-row)

The one-thread-per-row kernel was uncoalesced. A **warp-per-row** kernel — lanes
stride the row's nonzeros (so `col[]` reads coalesce), `src` gathered through the
read-only cache (`__ldg`), then the K-limb accumulator warp-reduced — is
**bit-exact** (validated in `bench/gpu-spmv-bench.cu`: warp PASS, 0 words; and the
backend still passes `bench_matcache` 4/4) and **1.8–3.1× faster** standalone
(b64 6.9→12.6 Gnz/s, b256 3.9→12.3). In the backend it cut the kernel from
2.63 → **0.97 ms** (2.7×), lifting the end-to-end SpMV to **8.96 Gnz/s (~5× the
full-CPU `bucket`)**.

Crucially this **flips the bottleneck**: the split is now H2D 1.81 ms / kernel
0.97 ms / D2H 0.62 ms — **72% transfers**. So the residency port above is now the
clearly dominant remaining single-machine win (~3× more headroom), and further
kernel tuning (ELL, column sorting, shared-mem `src`) is secondary.
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
    (4/4, both directions), with **pinned host-vector transfers** and a
    **coalesced warp-per-row kernel** — **8.96 Gnz/s, ~5× the full-CPU `bucket`**
    (kernel cut 2.7×; the split is now 72% transfers / 28% kernel).
  - **comm-on-device foundation**: the process-global device registry, the
    `bwc_base`↔GPU hook ABI (`matmul-gpu-hooks.h`), and a bit-exact device GF(2)
    reduce/broadcast wired into `mmt_vec_allreduce` (behind `CADO_GPU_DEVCOMM`) —
    `product == N` in default, `DEVCOMM`, and `VECRESIDENT+DEVCOMM` modes. This is
    validated *plumbing*, not yet a transfer saver (it uploads+writes-back for
    correctness; see "Comm-on-device" above).
  - **full vector residency** — the 2D hot comm (`mmt_vec_reduce` +
    `mmt_vec_broadcast`, not `allreduce`) ported to the device, plus GPU
    `x_dotprod` and device `addmul_tiny` for mksol/secure, so the steady krylov
    iteration runs entirely on device buffers with **zero** per-iteration host
    transfers (D2H 100% / H2D ~99% skipped). `product == N` across
    default/`DEVCOMM`/`VECRESIDENT+DEVCOMM` × `-t 2/4/8`; compute-sanitizer clean.
    Gated to single-node (`njobs==1` on both grid axes).
  - **multi-node MPI + per-rank multi-GPU**: the backend builds and runs under
    MPI (the `--mpi` compile marker is applied only to non-CUDA languages so
    `matmul-gpu.cu` compiles under nvcc), and `gpu_select_device()` binds one GPU
    per node-local rank. `product == N` validated under `mpi=1x2`/`2x1` with GPU
    SpMV; residency disables under MPI (host comm), so multi-rank runs stay
    correct on a single GPU and round-robin on multi-GPU hardware.
- **Next:** multi-node residency with a real transfer win — split the
  local-device reduce from the MPI data exchange so device-resident vectors are
  exchanged only across MPI (needs CUDA-aware MPI and multi-GPU HW to validate);
  further kernel tuning (ELL, column sorting, shared-mem `src`) is secondary.
