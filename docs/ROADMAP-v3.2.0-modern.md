# CADO-NFS v3.2.0-modern — roadmap

Forward plan for the next minor of this modernization + performance fork, after
`3.1.0-modern` shipped GPU linear algebra (SpMV + full vector residency), a GPU
pre-NFS ECM front-end, AVX-512 VPCLMULQDQ/IFMA kernels, and the Track-3
orchestration/UX layer. This roadmap is **research-grounded** (sources at the
bottom) and **honest about hardware**: the reference box is Comet Lake
(i9-10850K, **no AVX-512**) + a single **RTX 3090**, so AVX-512 work is
SDE/CI-gated and multi-GPU/multi-node work is validated at the degenerate path
locally + on cloud/CI multi-GPU, with perf claims only from real hardware.

## The strategic reframe

The single most important datapoint for sequencing:

> **RSA-250 cost (Zimmermann/PRACE): sieving ≈ 2450 core-years (91 %), linear
> algebra ≈ 250 core-years (9 %).**

`3.1.0-modern` put its GPU effort into the linear-algebra phase — the **9 %**. At
scale the cost lives in **sieving and polynomial selection**, so v3.2.0 aims the
algorithm + GPU + CPU effort there, while finishing the linalg work (better SpMV
kernel, real multi-GPU) it started.

## Standing ethos (unchanged)

- **No NFS-math change that alters results.** Gate every change on `make check` +
  verified `product == N` (or bit-exact vs CPU/GMP; AVX-512 under Intel SDE).
- **Honest negatives recorded.** v3.1.0 rejected PGO (×2) and hot-scalar
  micro-opt; those avenues stay closed unless new evidence appears.
- **Hardware-gated work ships as code validated at the degenerate path + a design
  doc**, never as an unvalidated claim.

---

## Track A — Algorithm / new techniques

- **A1. 3-D lattice sieving** ("A New Angle on Lattice Sieving for the NFS",
  arXiv 2001.10860). **RE-SCOPED — original framing was a misread (honest
  correction).** The paper's 3-D (and higher-D) enumeration is for the **Tower NFS
  / discrete log in extension fields** (its headline is a 133-bit subgroup DLP in
  𝔽_{p⁶}, a 423-bit field, ≈3× the prior record); the extra dimensions exist
  because TNFS sieves over higher-degree elements `a + b·x + c·x² …`. For ordinary
  **integer factorization the relations are (a,b) pairs — inherently 2-D** — and
  CADO's factorization siever is 2-D *because of that*, not as a limitation (the
  q-lattice has exactly two basis vectors; survivors map (i,j)→(a,b)). So **there
  is no "seeded c100" 3-D win**: the stated gate is not achievable, and 3-D sieving
  would only help the fork's extension-field **DLP** side — research-grade, a large
  effort, and *not* "locally testable on a c100." This avenue is therefore closed
  for the factorization track (recorded like the PGO / column-reorder negatives);
  any future pursuit belongs under **A4 (DLP/exTNFS)**. The freed sequencing slot
  goes to the genuinely high-value sieving + polyselect work (C2 collision-search
  offload, B1 AVX-512 sieving, D3 sieve fan-out).
- **A2. Mixed-representation ECM cofactorization** (Springer 2020, "Faster
  Cofactorization with ECM Using Mixed Representations"). **PARTLY DONE / honest
  finding.** The **CPU `facul` ECM already implements it** — upstream's
  "mishmash" bytecode (`bytecode_mishmash_B1_data.h`, `EDWARDS_ADDmontgomery`) is
  this exact scheme (the paper's authors are CADO authors). The fork's **GPU** ECM
  used a pure Montgomery ladder; a validated twisted-Edwards `a=−1` mixed-rep
  stage-1 (`bench/gpu-ecm-edwards.cu`, EFD/HWCD-2008 formulas, bit-exact vs the
  ladder through the birational map) is **measurably faster on GPU** — wNAF ~1.5×
  (128-bit) → ~2.9× (512-bit), the win growing with width. Foundation + measured
  win landed; live wiring into `gpu_ecm`/`gpu_prefactor` (+ dedicated tripling and
  the final Edwards→Montgomery switch) is the next step. Feeds C3. See
  `docs/gpu-ecm-mixedrep.md`.
- **A3. Parallel structured Gaussian elimination** for filtering/merge. **DONE /
  honest finding: already implemented upstream.** `filter/merge.cpp` *is* the
  Bouillaguet–Zimmermann parallel SGE (the exact roadmap reference, Math.
  Cryptology 0(1) 2020) + the Davis–Duff–Nakov parallel Markowitz threshold,
  ~50 OpenMP pragmas — the RSA-240/250 merge — and the orchestration already runs
  it with all logical threads. Verified it parallelizes (measured ~3.3× at 8
  threads on c60/c90 matrices; plateaus past 8 because desktop matrices are small
  vs the RSA-scale regime the parallelism targets). No code change. See
  `docs/parallel-merge-a3.md`.
- **A4. (stretch) DLP: exTNFS / Tower-NFS exploration** for the medium-
  characteristic discrete-log side the fork supports. Research-grade,
  correctness-only; document feasibility rather than commit speculative math.

## Track B — CPU performance (the honest envelope)

CPU tuning is nearly tapped (PGO/micro-opt rejected in 3.1.0). The real items are
AVX-512, which this box can only validate (SDE) not perf-measure:

- **B1. AVX-512 block + bucket sieving** (SciTePress/SECRYPT 2021). **Vectorizable
  slice DONE; scatter wall confirmed.** AVX-512 does *not* rescue the byte-scatter
  majority (~29%: fill_in_buckets, small-sieve, apply_buckets) — the sieve array is
  `uint8` and AVX-512 scatter is 32/64-bit-only (no 8-bit scatter), extending the
  3.1.0 AVX2 negative to AVX-512. The vectorizable slice — the per-prime modular
  inverse (`invmod_redc_32`, ~9.5% + feeds plattice's 11%) — is implemented as an
  **AVX-512 16-way masked batched modular inverse** (`bench/avx512-modinv.c`),
  bit-exact vs GMP under SDE (0/640000). Remaining: batch `plattice_info` setups
  through it (REDC fixup + scalar tail) — net win Amdahl-bounded by the scatter
  majority, perf gated on AVX-512 silicon. See `docs/avx512-sieving-b1.md`.
- **B2. Complete the AVX-512 VPCLMULQDQ gf2x port** (`mul2`–`mul9` + threshold
  retune). **mul2/mul3/mul4 DONE** — VPCLMULQDQ Karatsuba kernels (pack base
  products into the 512-bit lanes, one `_mm512_clmulepi64_epi128`, scalar fold),
  AVX-512F+VPCLMULQDQ only, guarded with PCLMUL fallback; **bit-exact under Intel
  SDE** (`bench/vpclmul-muln.c`, 0/200000 at 128/192/256-bit; integrated headers
  also compiled+validated). 3.1.0 had shipped `mul1` + detection. Remaining: the
  Toom kernels `mul5`–`mul9` (harder lane mapping), threshold retune, and the
  `lingen` perf number — all need real AVX-512 silicon (perf CI-gated; the main DG
  win is the already-shipped `mul_1_n`).
- **B3. IFMA GF(p) backend → mpfq integration.** **Primitive DONE.** "mpfq" is now
  the C++ `arith-modp` (plain-representation, scalar + vector ops). The validated
  IFMA modmul is Montgomery → mismatch; `bench/ifma-gfp.c` adds the missing
  **plain-representation** batched ops (`plain_mul`, `vec_add_dotprod` shape) on the
  validated kernel via `M(M(a,b),R²)` (amortized `R^{-1}` in the dotprod), bit-exact
  vs GMP under SDE (0/32000, 260-bit). Remaining: route
  `vec_add_dotprod`/`vec_addmul_and_reduce` for `p4`/`p5` through IFMA (limb repack +
  block map) — DLP-only; perf needs AVX-512-IFMA silicon. See `docs/ifma-gfp-b3.md`.

## Track C — GPU, single machine

- **C1. Faster SpMV kernel** (ELL / hybrid / sliced-ELL + column reordering +
  shared-memory `src` caching). The 3.1.0 warp kernel realizes only ~10–15 % of
  the 3090's ~936 GB/s; Schmidt et al. reach 4–8× single-GPU with a hybrid sparse
  format. Bit-exact gate; **measurable now** with the existing SpMV sweep harness.
  *Sequenced #1.*
- **C2. GPU polynomial selection.** Proven in **msieve** (GPU Kleinjung stage-1
  since 2009); CADO lacks it. Polyselect grows with N and is embarrassingly
  parallel — a clean single-machine GPU win, reusing the 3.1.0 multi-precision
  device arithmetic. Gate: same polynomial quality (Murphy-E) as the CPU path.
- **C3. GPU batch-smoothness product tree** (Bernstein remainder tree).
  **PARTLY DONE — A2-reusable leaf stage validated.** The leaf extraction
  (`s = gcd(R, (P mod R)^(2^e) mod R)`, reusing A2 `montmul`) is implemented and
  bit-exact vs GMP (`bench/gpu-batch-smooth.cu`, 0/8192 at 128/256/512-bit,
  20.6/1.45/1.60 Mleaf/s). Honest finding: the leaf is cheap and the only
  A2-arithmetic fit; the batch-smoothness bottleneck is the **big-integer
  product/remainder tree** (CPU/GMP, `sieve/ecm/batch.cpp`), whose GPU port needs
  arbitrary-precision multiply/division (separate large effort, not an A2 reuse).
  For GPU cofactorization, ECM (A2) is the better single-machine fit. See
  `docs/gpu-batch-smooth-c3.md`.
- **C4. (research) GPU sieving feasibility study.** Honest: NFS lattice sieving is
  memory-scatter-bound and largely unsolved on GPU (the "GPU tensor-core lattice
  sieving" work is for lattice *cryptanalysis*, a different problem). Scope as a
  *measured feasibility study*, not a promised feature.

## Track D — GPU at scale (multi-GPU / HPC / multi-machine)

The 3.1.0 designs, implementable where HW allows; correctness validated at the
degenerate path locally + on a multi-GPU cloud/CI runner; perf only from real HW.

- **D1. Intra-node multi-GPU matrix partition → real validation.** 3.1.0 shipped
  the `CADO_GPU_NPART` partition validated at N=1; finalize cross-device staging +
  per-device CUDA streams for overlap, validate on ≥2 GPUs. Schmidt et al. show
  GPU clusters beating larger CPU clusters on kilobit-SNFS matrices — the target.
- **D2. NVSHMEM / GPUDirect multi-node residency** (the 3.1.0 local-device-reduce
  / MPI-boundary-exchange split design). Use NVSHMEM (GPU-initiated PGAS comm over
  NVLink / InfiniBand-RDMA) to keep BWC vectors device-resident and exchange only
  across the MPI boundary, overlapping comm with the SpMV — the only path to a
  multi-node win per the 3.1.0 transfer accounting.
- **D3. Cluster / HPC orchestration.** Promote `scripts/cluster-launch.sh` to a
  real distributed driver: Slurm `sbatch` job arrays, GPU-aware client placement
  (one rank per GPU), and fan-out of the *sieving* stage (the 90 %) through the
  Rust work-unit server — the biggest practical lever.

## Track E — UI/UX & orchestration

Building on 3.1.0's `--json-status`, `/status`, `/dashboard`, clap CLIs, and
`cado-nfs-monitor-rs`:

- **E1. Real web dashboard** — phase timeline, per-phase ETA, live throughput +
  GPU/CPU utilization graphs, work-unit/client map (extend `/dashboard` into an
  SPA over the `/status` history).
- **E2. Per-machine parameter auto-tuner** ✓ **DONE.** `--autotune` calibrates the
  **safe scheduling knobs** to the host — local client/thread layout
  (`slaves.nrclients`) and work-unit granularity (`tasks.sieve.qrange`,
  `tasks.polyselect.adrange`, bounded √(threads/20) scaling). It **deliberately
  does not touch `lim*/lpb*/mfb*/I`** (the original framing): those set relation
  yield + matrix structure, so retuning them risks altering/breaking the
  factorization. Tuning only the *chunking* of identical work keeps `product == N`
  by construction (verified on the c59 smoke). `scripts/cadofactor/planner.py`.
- **E3. "Factor planner"** ✓ **DONE.** `--plan` / `--plan-json`: given `N` (+ the
  host thread count, `--host-speed`, GPU presence/build), prints feasibility, a
  wall-time envelope, the per-phase split, single-machine-vs-cluster strategy, and
  `--gpu-prefactor` / GPU-linalg triage, then exits. Model anchored on the measured
  BENCHMARKS c60–c90 sweep + the documented c100/c110 envelope, log-linear in
  digits, Amdahl-scaled to thread count. Doctested (`test_python_planner`).
- **E4. Checkpoint/resume + observability hardening** — robust mid-phase resume,
  structured JSON logs, optional OpenTelemetry metrics; container images
  (CUDA + CPU) and reproducible release artifacts.

---

## Cross-cutting

- **Correctness gate** after every change: seeded c60–c120 `product == N`;
  GPU/SIMD bit-exact; AVX-512 under SDE 10.8.0 (`/opt/intel-sde/sde64`).
- **CI:** extend `avx512-validate.yml`; add a multi-GPU cloud CI job (D1/D2) and an
  AVX-512 perf runner (B1/B2/B3).
- **Bench:** extend `bench/` + `BENCHMARKS.md`; record honest negatives.
- **Version/docs:** `CMakeLists.txt` `3.1.0 → 3.2.0-modern`; new
  `## [3.2.0-modern]` CHANGELOG; new docs (`docs/gpu-polyselect.md`,
  `docs/avx512-sieving.md`, a multi-GPU deep-dive).

## Sequencing & honest expected payoff

| # | Item | Effort | Risk | Honest payoff (this box → at scale) |
|---|------|--------|------|--------------------------------------|
| 1 | **C1** better SpMV kernel | Med | Low | measurable now; bigger at scale |
| 2 | **C2** GPU polyselect | Med | Low–Med | **real single-machine win** (proven in msieve) |
| 3 | ~~**A1** 3-D lattice sieving~~ → **C2 collision-search offload** | High | Med | A1 closed (TNFS/DLP only, not c100 — honest correction); the collision-search GPU offload is the real polyselect win |
| 4 | **E2/E3** autotuner + planner ✓ DONE | Med | Low | UX + cuts variance; no raw speed |
| 5 | **A2** mixed-rep ECM ✓ DONE (CPU already upstream; GPU win validated) | Med | Med | feeds C3 + CPU cofactor |
| 6 | **A3** parallel merge ✓ DONE (already upstream; verified ~3.3× @ t8) | Med | Med | cuts the high-variance filtering phase |
| 7 | **D1** multi-GPU partition (real) | High | High | **the large-N / HPC win** (needs ≥2 GPUs) |
| 8 | **B1/B2/B3** AVX-512 sieving + gf2x + IFMA (B1 batched modinv ✓; B2 mul2/3/4 ✓; B3 plain-rep GF(p) ✓ — all SDE-validated) | Med–High | Low | AVX-512-HW-only (SDE-correct, CI-perf) |
| 9 | **D2** NVSHMEM multi-node residency | High | High | cluster win (needs CUDA-aware MPI + multi-GPU) |
| 10 | **C3** product-tree (leaf stage ✓ DONE; trees = big-int, CPU) · **C4 / A4** GPU-sieve study · exTNFS | High | High | research / regime-specific |

**Net north star:** put the GPU + algorithm effort where the cost actually is —
**polynomial selection (C2) and sieving (A1, B1, D3)** — plus a **faster SpMV
kernel (C1)** and finally realising the **multi-GPU / multi-node linalg win
(D1/D2)** on real hardware, wrapped in a genuinely usable **autotuner + planner +
dashboard (E)**.

---

## Sources

- New angle on lattice sieving (3-D enumeration): arXiv:2001.10860
- AVX-512 block/bucket sieving: SciTePress/SECRYPT 2021 (105152)
- Multi-GPU Block Wiedemann over GF(2), hybrid sparse format: Schmidt et al.,
  Concurrency & Computation 2013 (DOI 10.1002/cpe.2896)
- RSA-250 cost breakdown: Zimmermann, *Factoring RSA-250 with PRACE*
- Parallel structured Gaussian elimination: *Mathematical Cryptology* 0(1)
- Faster cofactorization with ECM (mixed representations): Springer 2020
  (10.1007/978-3-030-45388-6_17); GPU cofactorization: eprint 2014/397
- GPU polynomial selection (Kleinjung stage-1): msieve (`Readme.nfs`)
- NVSHMEM / GPUDirect: NVIDIA developer docs; RDMA SpMV on GPUs (arXiv:2311.18141)
- Surveys: *Advancements and Prospects in Large Integer Factorization* (2024);
  *Integer Factorization: Another perspective* (arXiv:2507.07055, 2025)
