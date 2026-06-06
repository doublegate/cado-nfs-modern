# CADO-NFS 3.1.0-modern

A **modernization + performance fork** of
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs) — a complete
implementation in C / C++ / Python of the **Number Field Sieve (NFS)**, the
fastest known algorithm for factoring large integers and computing discrete
logarithms in finite fields.

> **What this fork is.** It rebases the upstream 3.0.0 codebase onto a current
> bleeding-edge toolchain (**CMake 4.x, GCC 16, CUDA 13, Python 3.14**) and adds
> several tracks of *rigorously measured* performance and orchestration work.
> From **3.1.0-modern** the fork carries its own minor line — still upstream
> 3.0.0's NFS algorithms, numerics, and parameters (**unchanged**), the bump
> reflecting substantial original work: **GPU linear algebra** with full vector
> residency, a **GPU pre-NFS ECM factoring front-end**, **AVX-512 VPCLMULQDQ +
> IFMA** kernels (SDE-validated), and an expanded **orchestration / UX** layer,
> on top of the 3.0.0-modern build/SIMD/GPU-cofactor/Rust tracks. Every change is
> a portability or throughput/robustness optimization, gated on `make check` +
> verified `product == N` factorizations (or a bit-exact validation); hardware-
> blocked items (multi-GPU/multi-node) are shipped as documented designs, not
> unvalidated code. Results are reported honestly, including the parts that did
> **not** pay off.
>
> This is **not** the official CADO-NFS. For releases, ongoing development, and
> support, use the upstream project (links at the bottom). The earlier
> 2.3.0-based release (`2.3.1-modern`) is preserved under the `v2.3.1-modern`
> tag, and `3.0.0-modern` under `v3.0.0-modern`.

[![CI](https://github.com/doublegate/cado-nfs-modern/actions/workflows/ci.yml/badge.svg)](https://github.com/doublegate/cado-nfs-modern/actions/workflows/ci.yml)
[![AVX-512 validation](https://github.com/doublegate/cado-nfs-modern/actions/workflows/avx512-validate.yml/badge.svg)](https://github.com/doublegate/cado-nfs-modern/actions/workflows/avx512-validate.yml)
[![License: LGPL-2.1](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](COPYING)
[![Upstream: 3.0.0](https://img.shields.io/badge/upstream-CADO--NFS%203.0.0-informational.svg)](https://gitlab.inria.fr/cado-nfs/cado-nfs)
[![Release: 3.1.0-modern](https://img.shields.io/badge/release-3.1.0--modern-success.svg)](https://github.com/doublegate/cado-nfs-modern/releases/tag/v3.1.0-modern)

**Jump to:** [Quick start](#quick-start) · [Performance](#performance) ·
[What this fork changes](#what-this-fork-changes-vs-upstream-300) ·
[How it works](#how-it-works) · [Documentation](#documentation) ·
[License](#license) · [Credits](#credits-and-attribution)

## Quick start

### Dependencies

- A C/C++ compiler conforming to **C99 + C++20** — GCC ≥ 10, LLVM Clang ≥ 12,
  Apple Clang ≥ 16, or Intel ICX ≥ 2023 (tested through GCC 16)
- [GMP](https://gmplib.org/) ≥ 5, **built with `--enable-shared`**
- CMake ≥ 3.18 (CMake 4.x works)
- Python 3 with **Flask** and **requests** (the work-unit server is Flask in
  3.0.0) — install them into a venv with the helper below
- Optional: `hwloc` (CPU pinning), MPI (clusters), `curl`, CUDA (the GPU backend)

On Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential cmake libgmp-dev libhwloc-dev \
                        python3 python3-venv python3-pip
```

### Build

```bash
bash scripts/setup-venv.sh   # ONE-TIME: create cado-nfs.venv (Flask + requests)
make                         # out-of-tree build into build/<hostname>/
make check                   # run the test suite (ctest); ARGS="-R <regex>" to filter
```

> **Why the venv first?** CADO-NFS 3.0.0 needs Flask/requests at *configure*
> time, and the committed [`local.sh`](local.sh) points cmake at
> `cado-nfs.venv/bin/python3`. `local.sh` also carries this fork's performance
> flags (`-O3 -march=native -mtune=native`). Edit it and run `make cmake` to
> reconfigure; `-march=native` is host-specific, so override it for a
> distributable build.

### Factor a number

```bash
cado-nfs.venv/bin/python3 ./cado-nfs.py \
  90377629292003121684002147101760858109247336549001090677693 -t 4
```

This runs the full pipeline (polynomial selection → sieving → filtering →
linear algebra → square root) over a local work-unit server and prints the
prime factors; the 59-digit demo finishes in ~15 s on a modern desktop.
CADO-NFS targets numbers **> 85 digits**; **< 60 digits is unsupported**, and
you should strip small factors first.

> **CLI note:** `key=value` parameters must come *before* flags like `-t`
> (e.g. `./cado-nfs.py <N> server.ssl=no -t 4`). Pass `server.ssl=no` for plain
> HTTP instead of TLS.

For larger and distributed factorizations, discrete logarithms, and the full
parameter reference, see the preserved upstream guide
([`README.upstream.md`](README.upstream.md)),
[`README.dlp`](README.dlp), and [`README.Python`](README.Python).

## Performance

Reference timings factoring balanced (RSA-like) semiprimes on an
**Intel i9-10850K (10 cores / 20 threads), 64 GiB RAM, CachyOS**, GCC 16.1.1
`-O3 -march=native`, GMP 6.3.0, all 20 threads. Same seeded inputs as the prior
`2.3.1-modern` benchmarks, so the comparison is apples-to-apples. Every result
was verified (factors multiply back to the input and are prime).

| Digits | Bits | Wall time | CADO CPU | Parallel | vs 2.3.1-modern (CPU) |
|-------:|-----:|----------:|---------:|---------:|----------------------:|
| 60 | ~199 | 19.2 s | 40.0 s | 2.1× | −31 % |
| 70 | ~232 | 26.0 s | 87.5 s | 3.4× | −28 % |
| 80 | ~265 | 70.3 s | 379.1 s | 5.4× | −32 % |
| 90 | ~299 | 184.5 s | 1465.0 s | 8.0× | −25 % |

**Key findings**

- **Total CPU work is down ~25-32 %** versus the 2.3.0-based fork on identical
  inputs — mostly from upstream 3.0.0's Bouvier–Imbert batch cofactorization
  (eprint 2018/669) and `I>16` sieving, compounded by this fork's
  `-O3 -march=native` (~7 % on the siever). This is the robust, repeatable
  signal; wall-time gains shrink with size and fall inside the ±15-20 %
  polynomial-selection noise by c90.
- **Parallel efficiency *drops* (e.g. c90 11.1×→8.0×) as a consequence of the
  CPU reduction**, not a regression: with less embarrassingly-parallel sieve
  work, the sequential phases (linear algebra, square root, orchestration)
  become a larger fraction of the run.
- **Sieving dominates** (45-74 % of CPU) and is the parallel phase; **linear
  algebra grows the fastest** (~110× from c60 to c90) and is the emerging second
  bottleneck — the classic NFS trade-off. Wall-time roughly doubles per +10
  digits, matching the sub-exponential `L(1/3)` complexity of NFS.
- **Practical envelope on this desktop:** ≤c75 interactive · c80-c95 a few
  minutes · ~c100 ≈ 10 min · ~c110 ≈ 1 hr · ≥c130 wants distributed mode.

**GPU (3.1.0-modern, RTX 3090).** The GPU pre-factoring ECM front-end runs
**~49× / 26× / 12×** the full 20-thread CPU at 128/256/512-bit moduli, and the
GPU BWC SpMV holds **~7.9 Gnz/s at c120-scale (240 M nonzeros, bit-exact)** —
~4.4× the tuned CPU `bucket` backend, with the advantage **widening as the matrix
grows** (the CPU loop is memory-bound; this is where large-N linear algebra
lives). Full vector residency removes the per-iteration PCIe transfers in the
steady krylov loop. These are GPU-build (`-DENABLE_GPU=ON`) results; the CPU
table above is the default build.

Full methodology, per-phase breakdown, the 2.3.1→3.0.0 comparison, the GPU
sweeps, projections, and seeded reproducible inputs:
[**`BENCHMARKS.md`**](BENCHMARKS.md).

## What this fork changes (vs upstream 3.0.0)

Upstream 3.0.0 already subsumes the toolchain-portability fixes the prior
`2.3.1-modern` fork carried (hwloc 2.x, Python-3 stdlib moves, OpenSSL 3.x) and
brings, for free, the batch cofactorization and `I>16` sieving above. On top of
that, this fork adds four independently-validated tracks:

| Track | What | Result |
|-------|------|--------|
| **1 · Build / compiler** | `local.sh`: `-O3 -march=native -mtune=native`; LTO and PGO evaluated | **~7 % on the siever** (host-ISA codegen). LTO/PGO were *measured and rejected* (0 % / −2.8 %). |
| **2 · SIMD** | AVX2 on the siever profiled; an AVX-512 **VPCLMULQDQ** gf2x base-case kernel added | AVX2 ruled out (the hot path is scatter + scalar modarith, not vectorizable). The gf2x kernel is **validated bit-exact** under Intel SDE; perf is gated on real AVX-512 silicon (the reference box is Comet Lake). |
| **3 · GPU ECM cofactorization** | A batched CUDA ECM backend (`sieve/ecm/gpu/`) behind `facul`/`las-cofactor`, validated bit-exact vs the CPU path | The GPU modmul primitive is ~39× a 20-core CPU — but cofactorization is only ~8 % of siever time, so **honestly, no net single-machine speedup at these sizes** (Amdahl-bounded). Documented as a measured negative, not an unsubstantiated win. |
| **4 · Rust orchestration** | The `rust/` workspace — a static-binary work-unit **client** and an async **server**, same HTTP/JSON protocol + `wudb` SQLite schema | Interoperates with an unmodified `cado-nfs.py`; the Rust server can be **swapped in** for the Flask server live (`CADO_RUST_WU_SERVER=…`), with TLS, IP-whitelist, and cert-pinning. For multi-client robustness, not single-machine speed. |

### New in 3.1.0-modern

Building on that base, 3.1.0 aims the GPU at the *linear-algebra* phase (where
the structural headroom actually is), adds a separate GPU factoring stage, and
rounds out the orchestration/UX layer:

| Track | What | Result |
|-------|------|--------|
| **GPU linear algebra** | A real `mm_impl=gpu` BWC SpMV backend (`linalg/bwc/matmul-gpu.cu`) with M+Mᵀ resident as CSR, a coalesced warp-per-row kernel, and **full vector residency** (the steady krylov/mksol/secure loop runs entirely on device buffers — 2D comm, `x_dotprod`, and `addmul_tiny` all ported) | Bit-exact (`bench_matcache` 4/4) and end-to-end `product == N`. The win **grows with N**: GPU warp SpMV holds **7.9 Gnz/s at c120-scale** while the CPU reference loop collapses — ~4.4× the tuned `bucket` backend, 18×→41× the reference loop across 1M→8M rows. Intra-node multi-GPU partition (`CADO_GPU_NPART`); multi-node residency is a documented design (HW-gated). |
| **GPU pre-NFS factoring** | `cado-nfs.py --gpu-prefactor`: batched multi-precision GPU ECM (stage-1 + stage-2 BSGS + Suyama, multi-GPU, escalating-`B1`) strips factors *before* NFS — a separate stage with no Amdahl ceiling | **~49× / 26× / 12×** the full CPU at 128/256/512-bit on an RTX 3090; end-to-end `product == N`. Unlike in-sieve cofactorization (Track 3), this is a real single-machine win when `N` has a findable factor. |
| **AVX-512 (VPCLMULQDQ + IFMA)** | gf2x VPCLMULQDQ auto-detection (run-test-gated, safe on non-AVX-512 hosts) + an IFMA GF(p) Montgomery modmul kernel | Both **bit-exact under Intel SDE** (`mul1`; IFMA modmul 0/32000 vs GMP), CI-gated (`avx512-validate.yml`). Correctness-only; the ~39 % perf gain is AVX-512-hardware-gated. |
| **Orchestration / UX** | Parameter interpolation + `--suggest-params`; a `/status` endpoint + `/dashboard` on both servers; `clap` CLIs, a `cluster-launch` helper (SSH/Slurm), and a `ratatui` terminal monitor (`cado-nfs-monitor-rs`) | Off-preset sizes get interpolated params instead of an error (`product == N` on a real c45); live phase/progress/ETA/factors; one-command cluster fan-out. |

The honest through-line: **CPU-side tuning is nearly tapped out** on this
hand-optimized codebase (PGO was re-measured at +3 % and rejected again; the hot
loops are already SIMD/unrolled/prefetched). The structural headroom is in new
compute resources — and 3.1.0 confirms it: the **GPU linear-algebra win grows
with N** (the fastest-growing NFS phase), and the **GPU pre-factoring front-end**
is a clean single-machine win, while AVX-512 is correctness-validated and
perf-gated on hardware. Full rationale and measurement methodology are in
[`CHANGELOG.md`](CHANGELOG.md); the GPU, GPU-linalg, GPU-prefactor, and Rust work
each have a dedicated deep-dive (see [Documentation](#documentation)).

## How it works

New to the Number Field Sieve? Two companion explainers, by background:

- **Plain English, no math** —
  [`docs/number-field-sieve-plain-english.md`](docs/number-field-sieve-plain-english.md):
  a friendly, analogy-driven tour for any curious reader (why factoring is hard,
  what the program actually does, and why it matters for online security).
- **The mathematics** —
  [`docs/number-field-sieve.md`](docs/number-field-sieve.md): the rigorous
  version — congruence of squares, the two-polynomial number-field
  construction, smoothness and lattice sieving, the $\mathbb{F}_2$ linear
  algebra, the algebraic square root, complexity, and how each phase maps to the
  directories in this tree.

The pipeline, stage by stage (each maps to a top-level directory):
**polynomial selection** (`polyselect/`) → **lattice sieving** (`sieve/`) →
**filtering** (`filter/`) → **linear algebra / Block Wiedemann** (`linalg/`) →
**algebraic square root** (`sqrt/`), orchestrated by `cado-nfs.py` /
`scripts/cadofactor/` (or the Rust layer in `rust/`).

## Documentation

| Document | What it covers |
|----------|----------------|
| [`README.upstream.md`](README.upstream.md) | Upstream 3.0.0's full build / usage / distributed / troubleshooting guide |
| [`docs/number-field-sieve-plain-english.md`](docs/number-field-sieve-plain-english.md) | The NFS, in layman's terms |
| [`docs/number-field-sieve.md`](docs/number-field-sieve.md) | The NFS mathematics, in depth |
| [`docs/gpu-linalg.md`](docs/gpu-linalg.md) | GPU BWC SpMV backend + full vector residency: kernel, transfer analysis, at-scale sweep, multi-GPU partition + multi-node residency design |
| [`docs/gpu-prefactor.md`](docs/gpu-prefactor.md) | GPU pre-NFS ECM factoring front-end: why it sidesteps Amdahl, the multi-precision Montgomery ECM, measured CPU-vs-GPU |
| [`docs/gpu-cofactorization.md`](docs/gpu-cofactorization.md) | GPU ECM cofactorization: measured results + honest Amdahl analysis (+ cofactor scale-out / product-tree designs) |
| [`docs/rust-orchestration.md`](docs/rust-orchestration.md) | The Rust client/server, the work-unit protocol, and the in-process swap |
| [`BENCHMARKS.md`](BENCHMARKS.md) | Performance sweep, per-phase breakdown, methodology, projections (incl. GPU pre-factoring + GPU linalg at scale) |
| [`CHANGELOG.md`](CHANGELOG.md) | Everything this fork changed, with rationale (3.1.0-modern · 3.0.0-modern · 2.3.1-modern) |
| [`CLAUDE.md`](CLAUDE.md) | Build/test/run notes and the fork's internal map |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) · [`SECURITY.md`](SECURITY.md) | How to contribute here vs upstream; security-reporting policy |
| [`README.dlp`](README.dlp) · [`README.Python`](README.Python) | Discrete logarithms; Python orchestration internals (upstream) |

## License

CADO-NFS is free software under the **GNU Lesser General Public License,
version 2.1** — see [`COPYING`](COPYING). This fork preserves that license and
all upstream copyright and authorship; the original authors are credited in
[`AUTHORS`](AUTHORS). The modifications described above are released under the
same LGPL-2.1 terms.

## Credits and attribution

- **Original work:** the CADO-NFS development team (INRIA / LORIA, Nancy,
  France) — Shi Bai, Cyril Bouvier, Pierrick Gaudry, Alexander Kruppa,
  Emmanuel Thomé, Paul Zimmermann, and many others (see [`AUTHORS`](AUTHORS)).
- **Upstream project:** <https://gitlab.inria.fr/cado-nfs/cado-nfs> ·
  homepage <http://cado-nfs.inria.fr/> · GitHub mirror
  <https://github.com/cado-nfs/cado-nfs>
- **Citation:** *CADO-NFS, An Implementation of the Number Field Sieve
  Algorithm.* See [`AUTHORS`](AUTHORS) for the BibTeX entry; per upstream's own
  note, set the release number to the version you actually used (here, 3.0.0).
- **This modernization + performance fork:** maintained by
  [@doublegate](https://github.com/doublegate).

If you use CADO-NFS in academic work, please cite the **upstream** project, not
this fork.
