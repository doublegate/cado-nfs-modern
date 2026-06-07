# GPU lattice sieving — measured feasibility study (Roadmap C4)

Per the roadmap, C4 is a **measured feasibility study, not a promised feature**:
is GPU NFS lattice sieving worth pursuing? Short answer, grounded in a
microbenchmark + the literature: **no — the GPU NFS win is cofactorization (ECM,
done in A2/3.1.0), not the siever.** GPU random scatter is ~5× a full CPU socket
*for the apply step in isolation*, but that step is only a slice of sieving and
the surrounding problems (byte-atomic granularity, on-GPU update generation,
memory capacity, pipeline integration) are why no production GPU siever exists.

## The core sieve operation

NFS lattice sieving spends its time scattering byte updates into the sieve array
— `S[x] -= log p` for every hit of every factor-base prime — via
`fill_in_buckets`/`apply_buckets` and `sieve_small_bucket_region`. The bucket
region is sized to fit cache by design, and CADO applies updates as **scalar,
race-free** writes within a single-thread region. The question for GPU is whether
this random read-modify-write scatter goes faster on a GPU.

## The measurement (`bench/gpu-sieve-scatter.cu`, RTX 3090 + i9-10850K)

4 M random updates `S[off] += v`; CPU 1-thread, CPU all-core (per-thread private
regions, the siever's model), and GPU atomic (the correct GPU sieve op), across
region sizes from cache-resident to DRAM. Updates/s:

| region | CPU 1-thread | CPU all-core | GPU atomic | GPU ÷ CPU all-core |
|--------|-------------:|-------------:|-----------:|:------------------:|
| 16 KiB | 630 M | 1377 M | 7378 M | **5.4×** |
| 256 KiB | 625 M | 1365 M | 7382 M | **5.4×** |
| 4 MiB | 333 M | 181 M | 2965 M | 16× |
| 64 MiB | 85 M | 135 M | 599 M | 4.5× |

**Honest reading.** In the cache-resident regime where real bucket regions live
(≤ a few hundred KiB), GPU atomic scatter is **~5.4×** a full CPU socket. The CPU
all-core plateaus at ~1.4 G updates/s because it is bound by **streaming the
update arrays** (`off`,`v` are 32 MB) over ~50 GB/s DDR4, not by the writes; the
GPU's ~5× edge is mostly its ~936 GB/s HBM for that stream. So the GPU does have a
real advantage *on this one step*.

## Why that doesn't make GPU sieving a win (the honest scope)

The ~5× scatter advantage does not survive the rest of the siever:

1. **Byte granularity.** The real sieve array is `uint8`; **AVX-512-style and CUDA
   atomics are 32/64-bit** — there is no 8-bit atomic. Correct GPU byte updates
   need packed 32-bit CAS loops, which under scatter *conflict* are far slower than
   the int-cell figure above (this probe used int cells, i.e. an optimistic 4×
   memory and no byte-CAS penalty).
2. **The updates must be generated on-GPU.** This probe is handed `off`,`v`. In a
   real siever those come from the **per-prime lattice/root arithmetic + bucket
   fill** — itself roughly half the work and scalar/scatter-heavy (the
   `plattice_info`/`invmod_redc_32` arithmetic, ~20%, and `fill_in_buckets`
   scatter, ~12%). Keeping the whole produce→consume pipeline on the GPU is the
   unsolved part, not the apply step measured here.
3. **Memory capacity & the rest of the pipeline.** A real factorization's factor
   base + bucket arrays are many GB; norm initialization, survivor extraction, and
   cofactorization must also live somewhere. Cofactorization is the one piece that
   *does* map to GPU — and the fork already does it (GPU ECM, A2/3.1.0).
4. **Empirical reality.** GPUs have been used heavily for NFS **cofactorization**
   for over a decade, yet **no production GPU lattice siever exists**. The
   "GPU/tensor-core lattice sieving" literature is about **lattice reduction**
   (SVP/CVP for cryptanalysis), a different problem from NFS relation collection.

## Verdict

- **Don't pursue a GPU NFS siever.** The dominant scatter is ~5× faster on GPU in
  isolation, but byte-atomic granularity, on-GPU update generation, memory
  capacity, and pipeline integration are unsolved — consistent with the absence of
  any production GPU siever. This is recorded as a measured negative (like the
  in-sieve GPU-cofactorization Amdahl negative, `docs/gpu-cofactorization.md`).
- **Keep GPU effort where it pays:** cofactorization (GPU ECM + the A2 mixed-rep
  win), GPU linear algebra (C1 SpMV, D1), and GPU polynomial-selection collisions
  (C2). The siever's own CPU envelope is the AVX-512 arithmetic slice (B1) plus its
  cache-resident scatter, which it already exploits well.

## Reproducing

```bash
nvcc -arch=sm_86 -O3 -Xcompiler -fopenmp bench/gpu-sieve-scatter.cu -o gpu-sieve-scatter
./gpu-sieve-scatter
```

## Sources

- The siever profile + the byte-scatter wall: `docs/avx512-sieving-b1.md`,
  CADO-NFS `sieve/las-apply-buckets.hpp`, `sieve/bucket.cpp`.
- GPU NFS effort is cofactorization, not sieving: `docs/gpu-cofactorization.md`,
  `docs/gpu-prefactor.md`, `docs/gpu-ecm-mixedrep.md`.
- "GPU/tensor-core lattice sieving" = lattice reduction (SVP), not NFS — e.g. the
  G6K-GPU line of work (Ducas et al.); distinct problem.
