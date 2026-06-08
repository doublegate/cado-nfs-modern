# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CADO-NFS is an implementation of the Number Field Sieve (NFS) for integer factorization and discrete logarithms. C (C99) + C++ (**C++20 required** as of 3.0.0) for the core, Python 3 (with **Flask**/requests) for orchestration (`cado-nfs.py`).

**This repo is `doublegate/cado-nfs-modern`** (renamed 2026-06-06 from `cado-nfs-3.0.0-modern`; GitHub auto-redirects the old URL) — a modernization + performance fork of upstream [CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) 3.0.0 (LGPL-2.1). Internal version is **`3.4.0-modern`** (`CMakeLists.txt`: numeric `3.4.0` + `-modern` suffix — the fork carries its own minor line; upstream NFS algorithms/parameters are unchanged). On top of the 3.1.0-modern GPU-linalg / GPU-prefactor / AVX-512 / orchestration tracks (themselves on the 3.0.0-modern build/SIMD/GPU-cofactor/Rust base), **3.2.0-modern** adds: a validated **GPU twisted-Edwards mixed-rep ECM** (A2), **adaptive GPU SpMV** + **GPU polyselect collision offload** (C1/C2), **AVX-512 VPCLMULQDQ `mul2/3/4` + IFMA GF(p) + batched modular-inverse** kernels (B1/B2/B3, SDE-validated), real **multi-GPU partition** (per-device streams, `product==N` at NPART=2) + **multi-node NVSHMEM/GPUDirect design** + **cluster sieving orchestration** (D1/D2/D3), a **factor planner + autotuner** (E2/E3), a **GPU batch-smoothness leaf** (C3), plus honest findings that the parallel merge (A3) and CPU mixed-rep ECM are already optimal upstream and that exTNFS-DLP (A4) / GPU-sieving (C4) are research-grade / measured-negative. On top of that, **3.3.0-modern** — opened on the now-thrice-confirmed premise that single-machine *speed* is tapped out — adds a shippable **operator-experience core** (live dashboard + trend-ETA / `--doctor` preflight / shell completions + man page / `--checkpoint-interval` / Slurm+PBS integration; E4–E8), the fork's **first measured-on-silicon SIMD** (AVX2 batched modular inverse, ~4.6×, B4), **Galois automorphism auto-detection** (`--galois-detect`, A5), and an honestly-gated research track (GPU root-sieve C5, GPU GF(p) lingen NTT C6, IFMA→arith-modp routing B5, exTNFS skeleton A6). On top of that, **3.4.0-modern** — same tapped-out premise — lands the one materially new opportunity the codebase exploration surfaced: the GPU pre-NFS factoring front-end is the **one stage with no Amdahl ceiling**, so it gets the cycle's **headline measured win**: GPU **Pollard P-1 / Williams P+1** beside the batched ECM with adaptive escalating-B1 (C7, 14/14 bit-exact vs GMP, ~3.3× faster time-to-strip); around it a shippable **observability/usability core** (completion **notifications** E9, NDJSON **event log** + Prometheus **`/metrics`** E10, **run-history DB** + `--list-runs`/`--compare-runs` E11, **per-phase + all-phases ETA** + `--wizard` + path-aware completions E12), a **data-driven autotuner** (`--calibrate` + regression cost model on `runs.db`, A7 — number-theoretic bounds untouched), and the carry-forward research track (GPU root-sieve launch threshold C5+, GPU GF(p) lingen NTT multi-modular CRT C6+). `main` tracks the latest release (`v3.4.0-modern`); `v3.3.0-modern`, `v3.2.0-modern`, `v3.1.0-modern`, `v3.0.0-modern`, `v2.3.1-modern` are preserved under their tags. See `CHANGELOG.md` for the full patch list, `README.md` for attribution, and `docs/` for the deep-dives.

## Build

```bash
bash scripts/setup-venv.sh   # ONE-TIME: create cado-nfs.venv (Flask + requests)
make            # configures + builds via scripts/call_cmake.sh (out-of-tree)
make cmake      # re-run cmake configuration (do this after editing local.sh)
make tidy       # DANGER: deletes the entire build tree (interactive y/N prompt)
```

- **3.0.0 needs Flask + requests at *configure* time.** The committed `local.sh` points cmake at `$PWD/cado-nfs.venv/bin/python3`, so create that venv once with `scripts/setup-venv.sh` before the first `make`. Without it, cmake configuration fails on the missing interpreter.
- The top-level `Makefile` is a thin wrapper over `scripts/call_cmake.sh`; it does **not** build directly. Do not run `cmake` by hand expecting `local.sh` to be read.
- Build output goes to `build/<hostname>/` (named by hostname, so one source tree serves multiple machines). There is no `build/` until you build.
- **Configuration is via `local.sh`**, not direct cmake flags: copy `local.sh.example` to `local.sh`, edit (`CC`, `CXX`, `CFLAGS`, `CXXFLAGS`, `GMP`, `MPI`, `PREFIX`, `build_tree`, ...), then `make cmake` to reconfigure. `local.sh` is sourced by `call_cmake.sh`, **not** by cmake itself — a plain `cmake /path` build ignores it. This fork's committed `local.sh` sets `-O3 -march=native -mtune=native` (Phase 1: ~7% on the siever; see `CHANGELOG.md`) plus the venv `PYTHON_EXECUTABLE`.
- **GMP (v5+) is mandatory and must be built with `--enable-shared`** or compilation fails. Optional: MPI, hwloc, curl. Locate GMP via `GMP` / `GMP_LIBDIR` / `GMP_INCDIR`.
- **Toolchain floor (3.0.0):** C99 + C++20; GCC >= 10 / LLVM Clang >= 12 / Apple Clang >= 16 / Intel ICX >= 2023; CMake >= 3.18.

## Test

```bash
make check                          # run the test suite (ctest under the hood)
make check ARGS="-j"                # parallel
make check ARGS="-R test_memusage"  # run tests matching a regex
```

Expensive tests are opt-in: `export CHECKS_EXPENSIVE=yes && make cmake && make check`. Tests live in `tests/`, mirror the source tree, and are named `test_*.c` / `test_*.sh`.

### Verifying a change fast (do this after editing C/C++)

The full suite is ~795 tests. When you touch one subsystem, run only its tests with `-R <regex>` instead of the whole suite — each test has a `builddep_<name>` companion that ctest builds automatically, so a targeted run also recompiles just what it needs. Two ways to invoke:

```bash
make check ARGS="-R 'bwc|bitlinalg|lingen'"        # from the source root (rebuilds deps first)
cd build/$(hostname) && ctest -R 'bwc' --output-on-failure   # faster, if the tree is already built
```

Map of what to run after patching each directory (ctest has **no labels** — selection is purely by test-name regex):

| You patched | Run `make check ARGS="-R '<regex>'"` | Matches |
|-------------|--------------------------------------|---------|
| `sieve/` | `sievetest\|F9_` | `sievetest_I`, `F9_sievetest*`, `F9_makefbtest`, `F9_dupsuptest`, `F9_fakereltest` |
| `linalg/` (incl. `linalg/bwc/`) | `bwc\|bitlinalg\|lingen\|matmul` | `test-bwc-*`, `test_bitlinalg_*`, `bwc_staged_krylov`, `dispatch-matmul-*`, `lingen` |
| `polyselect/` | `polyselect` | polynomial-selection tests |
| `sqrt/` | `sqrt\|testsm` | square-root tests |
| `numbertheory/` | `numbertheory` | `numbertheory-*` |
| `gf2x/`, `linalg/m4ri`-style low-level | `mpfq\|bitlinalg` | `mpfq_test_*`, matrix-op tests |

`linalg/bwc/cpubinding.cpp` (hwloc CPU-binding) is exercised by the `bwc` tests **and** by any real factorization's Linear Algebra phase — a `./cado-nfs.py` smoke test is the surest end-to-end check for it. For the whole pipeline, a 59-digit smoke factorization (`cado-nfs.venv/bin/python3 ./cado-nfs.py 90377629292003121684002147101760858109247336549001090677693 -t 4`) runs in ~15 s.

## Run

Main entry point is `./cado-nfs.py`. It needs the Python `sqlite3` module **and** Flask/requests (the work-unit server is Flask now), so run it under the venv: `cado-nfs.venv/bin/python3 ./cado-nfs.py ...`. It deduces the `build/<hostname>/` binary dir automatically — invoke it from the source root, do not call binaries by path.

```bash
cado-nfs.venv/bin/python3 ./cado-nfs.py <N>          # factor N on all local cores (default -t all)
cado-nfs.venv/bin/python3 ./cado-nfs.py <N> -t 2     # cap at 2 threads
cado-nfs.venv/bin/python3 ./cado-nfs.py /path/XXX.parameters_snapshot.YYY   # resume an interrupted run
```

Optimized for numbers > 85 digits; < 60 digits is unsupported. Strip small prime factors (trial division / P-1 / P+1 / ECM) before using it. Parameter presets per size live in `parameters/`. **Arg-order quirk:** `key=value` params must come *before* `-t`/other flags (`./cado-nfs.py <N> server.ssl=no -t 4`).

## Directory map (NFS stages)

| Dir | Stage / contents |
|-----|------------------|
| `polyselect/` | Polynomial selection (first NFS stage) |
| `sieve/` | Lattice siever (`las`), relation collection, factor base |
| `filter/` | Filtering: merge / purge / balance the relation matrix |
| `linalg/` | Linear algebra — Block Wiedemann (BWC), MPI-capable |
| `sqrt/` | Square root in the number field (final stage) |
| `numbertheory/`, `utils/`, `misc/` | Number-theory helpers, generic utilities, profiling |
| `scripts/cadofactor/` | Python orchestration: task scheduling, work-unit distribution (Flask `api_server.py`, `cadotask.py`, `wudb.py`) |
| `rust/` | Phase-4 Rust orchestration: static-binary client + async work-unit server (see below) |
| `config/` | CMake compiler/dependency detection |
| `parameters/` | Per-size factorization parameter presets (c60, c90, ...) |
| `gf2x/` | GF(2)[x] arithmetic (separate configure) |
| `docs/` | Theory explainers + the GPU / Rust deep-dives (see below) |

## Gotchas

- Some ancient GCC versions miscompile CADO-NFS (4.1.2 / 4.2.0–4.2.2). Irrelevant on a modern box, but noted upstream.
- GMP <= 6.0 + multi-threaded sqrt: pass `tasks.sqrt.threads=1` or upgrade to GMP >= 6.1.0.
- Numbers > 200 digits need `FLAGS_SIZE` set in `local.sh` to enable 64-bit counters.
- Distributed/server mode needs SSH public-key auth and `localhost` -> 127.0.0.1; see `README` for the SSH config block.
- No `.clang-format` or enforced style exists. Match the style of surrounding code; do not bulk-reformat. (Fork-specific files like `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `docs/*`, `rust/*` were added by this fork.)
- `-march=native` in the committed `local.sh` is host-specific; for a distributable build, override it in your own `local.sh`.

## This fork (rebased on upstream 3.0.0)

3.0.0 already subsumes the 2.3.x portability fixes the prior `2.3.1-modern` fork carried (hwloc 2.x, Python 3 stdlib moves, OpenSSL 3.x SSL) and brings, for free, the Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16` sieving. What *this* fork adds on top, validated and measured (full detail in `CHANGELOG.md`):

**3.0.0-modern tracks (foundation):**

- **Build (Phase 1):** `local.sh` sets `-O3 -march=native` (the only real CPU win — LTO/PGO were measured and rejected). Builds on CMake 4.x / GCC 16 / Python 3.14.
- **SIMD (Phase 2):** AVX2 on the siever was ruled out by profiling; an AVX-512 VPCLMULQDQ gf2x base-case kernel is validated bit-exact under Intel SDE (perf-gated on real AVX-512 silicon). See `gf2x/already_tuned/x86_64_vpclmul/`.
- **GPU cofactorization (Phase 3):** a complete batched ECM cofactorization backend (`sieve/ecm/gpu/`, CMake CUDA), validated bit-exact vs the CPU path. Honest finding: cofactorization is Amdahl-bounded (~8% of siever time) → **no net single-machine speedup** at these sizes. See `docs/gpu-cofactorization.md`.
- **Rust orchestration (Phase 4):** the `rust/` workspace — `cado-nfs-client-rs` (static-binary work-unit client) and `cado-wu-server-rs` (async axum/tokio server over the same `wudb` SQLite schema). Keeps the exact HTTP/JSON protocol, so it interoperates with an unmodified `cado-nfs.py`. The Rust server can be **swapped in** for the Flask `api_server.py` in a live run by setting `CADO_RUST_WU_SERVER=<path to cado-wu-server-rs>` (a Python shim, `scripts/cadofactor/external_api_server.py`, launches it; `cadotask.py` switches to it). TLS + IP whitelist + cert-pinning all work through the swap. `cadotask.py` (the high-level task DAG) is deliberately left in Python. See `docs/rust-orchestration.md` and the `rust/*-test.sh` validators.

**3.1.0-modern additions (every shipped change gated by `product == N` or a bit-exact validation; HW-blocked items are documented designs, not committed unvalidated code):**

- **GPU linear algebra:** a real `mm_impl=gpu` BWC SpMV backend (`linalg/bwc/matmul-gpu.cu`, b64/b128) with M+Mᵀ resident as CSR, a coalesced warp-per-row kernel, and **full vector residency** — the steady krylov/mksol/secure loop runs entirely on device buffers (2D comm + GPU `x_dotprod` + device `addmul_tiny` ported). Env: `CADO_GPU_VECRESIDENT=1 CADO_GPU_DEVCOMM=1`. The win grows with N (~7.9 Gnz/s at c120-scale, ~4.4× the tuned `bucket`). Intra-node multi-GPU partition via `CADO_GPU_NPART` (default 1; validated at N=1, true multi-GPU HW-gated); multi-node residency is a documented design. See `docs/gpu-linalg.md`. Hook definitions live in `linalg/bwc/matmul-gpu-hooks.cpp` (in `matmul_common`, so `bench_matcache --impl gpu` links).
- **GPU pre-NFS factoring:** `cado-nfs.py --gpu-prefactor` (`misc/gpu_prefactor/`, `scripts/cadofactor/gpu_prefactor.py`) strips factors with batched multi-precision GPU ECM before NFS — a separate stage with no Amdahl ceiling (~49×/26×/12× the full CPU at 128/256/512-bit). See `docs/gpu-prefactor.md`.
- **AVX-512 (correctness-only, CI-gated under SDE):** gf2x VPCLMULQDQ auto-detection (`gf2x/config/features.m4` `CHECK_VPCLMUL_SUPPORT`, run-test-gated so non-AVX-512 hosts keep pclmul — safe) and an IFMA GF(p) Montgomery modmul kernel (`bench/ifma-modmul.c`). Both bit-exact under Intel SDE (`bench/{vpclmul,ifma}-validate.sh`; SDE at `/opt/intel-sde/sde64`), gated by `.github/workflows/avx512-validate.yml`.
- **Orchestration/UX:** parameter interpolation + `--suggest-params` (`toplevel.py`); `--json-status`/`--progress` (`status.py`) + `GET /status` & `/dashboard` on both servers; `clap` CLIs for the Rust binaries; `scripts/cluster-launch.sh` (SSH/Slurm); `cado-nfs-monitor-rs` (ratatui).
- **Honest negatives (measured, recorded):** siever-trained PGO retry (+3%, rejected); no safe hot-scalar micro-opt (hot loops already SIMD/unrolled/prefetched).
- **Versioning:** `CADO_VERSION_STRING` is `3.4.0-modern` (`CMakeLists.txt`: `CADO_VERSION_MINOR 4` + `-modern` suffix). The numeric minor **deliberately diverges** from upstream 3.0.0 to mark the fork's substantial original work; no test depends on the string.

**3.2.0-modern additions (same gate: `product == N` / bit-exact / Intel-SDE; HW-blocked items are documented designs or correctness-validated kernels):**

- **GPU mixed-rep ECM (A2):** twisted-Edwards `a=−1` extended-coordinate stage-1 (`bench/gpu-ecm-edwards.cu`), bit-exact vs the Montgomery ladder via the birational map; wNAF ~1.5–2.9×. CPU `facul` already does mixed-rep (upstream "mishmash"). See `docs/gpu-ecm-mixedrep.md`.
- **GPU C1/C2/C3:** adaptive sub-warp SpMV (`spmv_vec`, `CADO_GPU_SPMV={vec,warp}`); GPU polyselect collision offload (`--gpu-polyselect`, size-gated); GPU batch-smoothness leaf (`bench/gpu-batch-smooth.cu`). See `docs/gpu-polyselect.md`, `docs/gpu-batch-smooth-c3.md`.
- **AVX-512 (B1/B2/B3, SDE):** VPCLMULQDQ gf2x `mul2/3/4` (`bench/vpclmul-muln.c`); IFMA GF(p) plain-rep ops (`bench/ifma-gfp.c`); 16-way batched modular inverse for the siever (`bench/avx512-modinv.c`). All in `avx512-validate.yml`. See `docs/avx512-sieving-b1.md`, `docs/ifma-gfp-b3.md`.
- **Multi-GPU/HPC (D1/D2/D3):** per-device CUDA streams for `CADO_GPU_NPART` (validated `product==N` at NPART=2, c90); NVSHMEM/GPUDirect multi-node design (`docs/multinode-residency-d2.md`); `scripts/cluster-launch.sh` Slurm `--sbatch` job arrays + `--gpus-per-node` placement.
- **UX (E2/E3):** `cado-nfs.py --plan`/`--plan-json` factor planner + `--autotune` per-host scheduling calibration (`scripts/cadofactor/planner.py`, doctested).
- **Honest findings (recorded):** parallel merge (A3) is already the Bouillaguet–Zimmermann RSA-record code (verified ~3.3× @ t8); exTNFS-DLP (A4) is research-grade (`docs/extnfs-a4.md`); GPU lattice sieving (C4) is a measured negative (`docs/gpu-sieving-c4.md`).

**3.3.0-modern additions (premise: single-machine speed is tapped out; split into a measurable usability core + an honestly-gated research track; same gate: `product == N` / bit-exact / Intel-SDE):**

- **Usability core (E4–E8):** live `cado-nfs-monitor-rs` trend-ETA + throughput + host CPU/GPU, mirrored into the `/dashboard` HTML; `cado-nfs.py --doctor`/`--doctor-json` preflight (`scripts/cadofactor/doctor.py`, doctested); bash/zsh/fish completions (`scripts/build-completions.py` → `misc/completions/`) + `--completions <shell>` on the Rust binaries + `misc/man/cado-nfs.1`; `--checkpoint-interval`; `scripts/cluster-launch.sh --pbs` + `--suggest-{slurm,pbs}-config`. See `docs/usability-v330.md`.
- **AVX2 modinv (B4):** AVX2 8-way masked binary-GCD modular inverse (`bench/avx2-modinv.c`, `bench/avx2-modinv-validate.sh`) — the fork's first *measured-on-silicon* SIMD, **~4.6×** scalar, bit-exact vs GMP (native, no SDE). Amdahl-bounded whole-siever. See `docs/avx2-simd-b4.md`.
- **Galois auto-detect (A5):** `scripts/cadofactor/galois.py` (Möbius-invariance detector, doctested, `test_python_galois`) + `cado-nfs.py --galois-detect POLYFILE`; cross-validated vs `tests/sieve/galois.poly`. Advisory; the reduction is CADO's upstream `--galois`. See `docs/galois-auto-a5.md`.
- **Research track (gated, bit-exact, honest non-wins):** GPU stage-2 root-sieve (`bench/gpu-ropt-stage2.cu`, C5 — per-rotation wash); GPU GF(p) lingen NTT (`bench/gpu-lingen-ntt.cu`, C6 — multi-GPU/DLP); IFMA→arith-modp routing bridge (`bench/ifma-gfp.c` block 3, B5 — HW-gated + repack-sensitive); exTNFS interface skeleton (`docs/extnfs-a4.md` §A6 — documented design). See `docs/{gpu-polyselect-ropt-c5,gpu-lingen-ntt-c6}.md`.

**3.4.0-modern additions (same tapped-out premise; headline is the no-Amdahl prefactor stage; same gate: `product == N` / bit-exact / Intel-SDE):**

- **C7 — GPU prefactor P-1/P+1 (headline, measured):** GPU Pollard **P-1** + Williams **P+1** (stage-1 + stage-2 BSGS) added to the GPU pre-factoring front-end (`misc/gpu_prefactor/gpu_pm1_pp1.cuh`, integrated in `gpu-prefactor.cu` via `pm1pp1_pass`/`run_stage_K`), reusing the bit-exact Montgomery core, with **adaptive escalating-B1** (ECM skipped once cofactor prime/1; `CADO_PREFACTOR_NOPM1PP1=1` disables). **14/14 bit-exact** vs CPU & GMP (`bench/gpu-prefactor-pm1pp1.cu` + `.sh`, RTX 3090), `f|N` re-verified, **~3.3× faster time-to-strip**. One sequence = one lane (coverage, not throughput — that stays ECM). See `docs/gpu-prefactor-pm1pp1-c7.md`.
- **Usability/observability core (E9–E12):** completion/failure **notifications** (`scripts/cadofactor/notify.py`, `--notify`; ntfy/Slack/Discord/webhook/email/desktop; secrets via env/`[notifications]`, never the snapshot); NDJSON **event log** (`--json-log`) + Prometheus **`/metrics`** on the Flask + Rust servers; **run-history DB** (`scripts/cadofactor/runs.py` → `~/.cado-nfs/runs.db`, `--list-runs`/`--compare-runs`); **per-phase + all-phases ETA** in `monitor.rs` + `/dashboard`, `--wizard` (`scripts/cadofactor/wizard.py`), path-aware completions + man EXAMPLES. All wired through `status.py` finish hooks; `notify`/`runs`/`wizard` doctested. See `docs/usability-v340.md`.
- **A7 — data-driven autotuner:** `planner.calibrate_host_speed` + `planner.regression_estimate` (log-linear OLS on `runs.db`), `--calibrate`, empirical refinement folded into `--plan`. Number-theoretic bounds untouched (prediction only). Seeded validation: 1.500× recovered, R²=0.978.
- **Research track (C5+/C6+, bit-exact, honest):** GPU root-sieve **conditional-launch threshold** (`bench/gpu-ropt-threshold-c5plus.cu` — bit-exact every size, routes to the measured-faster path; crossover ~16M-cell lines, ~3.9× large-N); GPU GF(p) lingen NTT **multi-modular CRT wrapper** (`bench/gpu-lingen-ntt-crt-c6plus.cu` — CRT == `__int128` convolution, 0/2999). `bench/gpu-research-v340-validate.sh`. See `docs/{gpu-polyselect-ropt-c5,gpu-lingen-ntt-c6}.md`.

### Docs

- `docs/number-field-sieve.md` — in-depth NFS mathematics explainer.
- `docs/number-field-sieve-plain-english.md` — the same in layman's terms.
- `docs/gpu-cofactorization.md` — GPU ECM cofactorization: measured results + honest Amdahl analysis (+ 3.1.0 cofactor scale-out / product-tree designs).
- `docs/gpu-prefactor.md` — (3.1.0) GPU pre-NFS ECM factoring front-end: why it sidesteps Amdahl, the multi-precision Montgomery ECM, measured CPU-vs-GPU.
- `docs/gpu-linalg.md` — (3.1.0) GPU BWC SpMV backend + full vector residency: kernel, transfer analysis, at-scale sweep, multi-GPU partition + multi-node residency design.
- `docs/gpu-ecm-mixedrep.md` — (3.2.0, A2) mixed-representation ECM: the CPU path already does it (upstream "mishmash"); a validated twisted-Edwards `a=−1` GPU stage-1 (bit-exact vs the ladder via the birational map) is ~1.5–2.9× the Montgomery ladder, growing with modulus width.
- `docs/parallel-merge-a3.md` — (3.2.0, A3) parallel structured Gaussian elimination (merge) is already upstream (Bouillaguet–Zimmermann, the RSA-240/250 code) and runs with all logical threads; measured ~3.3× at 8 threads on c60/c90 matrices (plateaus past 8 — desktop matrices are small vs the RSA-scale regime).
- `docs/gpu-batch-smooth-c3.md` — (3.2.0, C3) GPU batch-smoothness leaf extraction (`s = gcd(R, (P mod R)^(2^e) mod R)`, reusing A2 `montmul`), bit-exact vs GMP at 128/256/512-bit. Honest scope: the leaf is the only A2-arithmetic fit; the batch bottleneck is the big-integer product/remainder tree (CPU/GMP), so ECM (A2) stays the better GPU-cofactorization fit.
- `docs/multinode-residency-d2.md` — (3.2.0, D2) NVSHMEM/GPUDirect multi-node device-resident BWC design: keep vectors on-device through the MPI grid collectives (Allgather/Reduce_scatter/Allreduce in `matmul_top_comm.cpp`) via CUDA-aware MPI / NVSHMEM, overlap with the SpMV (reuses D1 streams). HW-gated design; no unvalidated code; single-rank degenerate path already validated.
- `docs/extnfs-a4.md` — (3.2.0, A4) exTNFS/Tower-NFS DLP feasibility: CADO does classic NFS-DLP (GF(p)/GF(p²)/small-k, 2-D siever); exTNFS needs a tower (≥3-D siever = the A1 paper, tower polyselect, tower-ideal relations) — research-grade, documented not committed.
- `docs/gpu-sieving-c4.md` — (3.2.0, C4) GPU lattice-sieving measured feasibility (`bench/gpu-sieve-scatter.cu`): GPU scatter ~5.4× a CPU socket on the apply step alone, but byte-atomics/on-GPU-generation/capacity unsolved → no GPU siever; keep GPU on cofactorization/linalg/polyselect. Measured negative.
- `docs/avx512-sieving-b1.md` — (3.2.0, B1) AVX-512 sieving: the byte-scatter majority (~29%) doesn't vectorize (no 8-bit AVX-512 scatter — extends the 3.1.0 AVX2 negative); the vectorizable slice (per-prime modular inverse) is an AVX-512 16-way masked batched modinv (`bench/avx512-modinv.c`), bit-exact vs GMP under SDE (0/640000). Integration + perf are AVX-512-HW-gated.
- `docs/ifma-gfp-b3.md` — (3.2.0, B3) AVX-512 IFMA GF(p) for the BWC backend: "mpfq" is now `arith-modp` (plain representation). `bench/ifma-gfp.c` adds plain-representation batched `plain_mul` + `vec_add_dotprod`-shape ops on the validated Montgomery IFMA kernel (`M(M(a,b),R²)`, amortized `R^{-1}`), bit-exact vs GMP under SDE (0/32000, 260-bit). **(3.3.0, B5)** the doc now also covers the validated routing bridge — radix-2^64 (arith-modp storage) ↔ radix-2^52 (IFMA) repack + the `vec_add_dotprod` `+w` addend, bit-exact under SDE; in-tree routing is DLP-only, HW-gated (no IFMA silicon) *and* repack-sensitive, documented not committed.
- `docs/rust-orchestration.md` — the Phase-4 Rust client/server, the protocol, and the in-process swap.
- `docs/ROADMAP-v3.3.0-modern.md` — (3.3.0) the cycle anchor: the "speed is tapped out" premise, the usability core vs honestly-gated research split, the full track map + sequencing.
- `docs/usability-v330.md` — (3.3.0, E4–E8) the operator-experience core: live dashboard + trend-ETA, `--doctor` preflight, completions + man page, `--checkpoint-interval`, Slurm/PBS integration.
- `docs/avx2-simd-b4.md` — (3.3.0, B4) the AVX2 8-way batched modular inverse: the fork's first *measured-on-silicon* SIMD (~4.6×, bit-exact, native), honestly Amdahl-scoped for the whole siever. Montgomery's trick does not apply (per-prime moduli).
- `docs/galois-auto-a5.md` — (3.3.0, A5) exact Galois-automorphism detection (Möbius-invariance + the orbit guard) and `--galois-detect`; cross-validated vs CADO's `galois.poly`. Detection/advisory only; the reduction is CADO's upstream `--galois`.
- `docs/gpu-polyselect-ropt-c5.md` — (3.3.0, C5) GPU stage-2 root-sieve core (bit-exact vs int16 CPU); ~1.7× raw apply but an honest per-rotation wash at testable sizes (large-N only).
- `docs/gpu-lingen-ntt-c6.md` — (3.3.0, C6; +3.4.0 C6+) GPU GF(p) lingen NTT (bit-exact vs schoolbook); the single-prime inner transform of a multi-modular GF(p) lingen; the C6+ section adds the multi-modular CRT wrapper (CRT == `__int128` convolution, 0/2999). <1% single-machine net, multi-GPU/DLP play.
- `docs/gpu-prefactor-pm1pp1-c7.md` — (3.4.0, C7 — headline) GPU Pollard P-1 / Williams P+1 + adaptive escalating-B1 on the no-Amdahl prefactor front-end: 14/14 bit-exact vs CPU & GMP, ~3.3× faster time-to-strip; coverage (one lane each), not GPU throughput.
- `docs/usability-v340.md` — (3.4.0, E9–E12 + A7) notifications, NDJSON event log + Prometheus `/metrics`, run-history DB (`--list-runs`/`--compare-runs`), per-phase + all-phases ETA + `--wizard`, the `--calibrate`/regression data-driven autotuner.
- `docs/ROADMAP-v3.4.0-modern.md` — (3.4.0) the cycle anchor: the tapped-out premise, the one new opportunity (no-Amdahl prefactor stage), the track map + sequencing + gates.
- The C5 doc (`docs/gpu-polyselect-ropt-c5.md`) gains a C5+ section: the conditional-launch threshold (bit-exact every size; routes to the measured-faster path; crossover ~16M-cell lines).
- `BENCHMARKS.md` — performance sweep + methodology (§7 = 3.3.0; §8 = 3.4.0 additions).
- Upstream: `@README` (build/run/distributed), `@README.dlp` (discrete log), `@README.Python` (orchestration internals).
