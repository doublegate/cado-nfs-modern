# Roadmap — v3.4.0-modern

This is the planning anchor for the `3.4.0-modern` development cycle, the successor
to [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md). It records *why* this
cycle is shaped the way it is, the per-track scope, and the honesty gates each item
must pass. Per-track deep-dives are linked as they land.

## The honest premise (now four times confirmed)

CADO-NFS factors a large integer `N` in five stages — **polynomial selection**,
**lattice sieving**, **filtering**, **linear algebra** (Block Wiedemann), and the
**algebraic square root**. Wall-clock is dominated by sieving (~91 % at RSA-250
scale) and, far behind, linear algebra (~9 %).

Across four revisions this fork has measured, and re-measured, the same wall: **on
the reference box — an Intel i9-10850K (Comet Lake: AVX2, *no* AVX-512) and a single
RTX 3090 (sm_86) — single-machine NFS *speed* is essentially tapped out.** A fresh
June-2026 internet survey (upstream CADO on INRIA GitLab; the sethtroisi/MichaelBell
forks; msieve, YAFU, GGNFS; GMP-ECM / ecmongpu / CGBN; pstach/gls; the RSA-250 /
DLP-240 / F_p^4–F_p^6 record papers; 2021–2026 eprint/arxiv) reconfirms two things:
**(1)** no published technique since ~2010 delivers a >5 % single-machine NFS
speedup — GPU lattice sieving remains a measured negative (pstach/gls is slower than
the CPU siever; the Ducas–Stevens GPU "sieving" is *lattice reduction*, not NFS),
AVX-512 is silicon-gated and memory-bound anyway, ML-polyselect has no production
code, and tower-NFS is DLP-only research with no approachable open-source siever; and
**(2)** this fork is already *ahead of every public implementation* on GPU and SIMD
modernization. No integer-factoring record has moved since RSA-250 (Feb 2020).

## The one materially new opportunity

The v3.4.0 codebase exploration surfaced a leverage point the prior cycles had only
partly used: **the GPU pre-NFS factoring front-end (`misc/gpu_prefactor/`) is the one
place in the whole pipeline with no Amdahl ceiling.** It strips factors *before* NFS
starts, so it is a *separate stage*, not a fraction of sieve time — which is exactly
why it measured 49×/26×/12× (128/256/512-bit) in 3.1.0-modern while in-sieve GPU
cofactorization, Amdahl-capped at ~8 % of sieve, netted <1 %. That front-end runs
**ECM only** today. Extending it is the cycle's headline measured-on-silicon win.

## The shape this dictates

Like v3.3.0, this cycle splits transparently into a **shippable, measurable core**
and an **honestly-gated, carry-forward research track**, and covers both integer
factorization and discrete logarithm (DLP). What is new is that the headline item is
itself measured (not a wash or HW-gated design): the GPU prefactor extension.

## Track map

A = number-field / cofactor math, B = SIMD, C = GPU, D = multi-GPU/HPC, E = UX /
orchestration. v3.4.0 continues each.

| Code | Item | Class | Honest payoff on this hardware | Doc |
|------|------|-------|--------------------------------|-----|
| C7 | GPU prefactor: Pollard **P-1/P+1** (stage-1 + stage-2 BSGS) + **adaptive escalating-B1** | GPU | **Headline MEASURED win** — no Amdahl ceiling; finds factors ECM misses | [gpu-prefactor-pm1pp1-c7](gpu-prefactor-pm1pp1-c7.md) |
| E9 | Completion/failure **notifications** (ntfy / Slack / Discord / webhook / email / desktop) | Usability | Real, here-and-now | [usability-v340](usability-v340.md) |
| E10 | Structured **JSON event log** + Prometheus **`/metrics`** (Flask + Rust servers) | Observability | Real (Grafana/alerting) | [usability-v340](usability-v340.md) |
| E11 | **Multi-run history DB** + `--list-runs` / `--compare-runs` | Usability | Real (campaign tracking; feeds A7) | [usability-v340](usability-v340.md) |
| E12 | **Per-phase ETA** + `--wizard` param TUI + dynamic completions | Usability | Real UX polish | [usability-v340](usability-v340.md) |
| A7 | Data-driven autotuner: `--calibrate` + regression cost model on `runs.db` | Heuristic | **Real & measurable** (better `--plan`) | [usability-v340](usability-v340.md) |
| C5+ | GPU root-sieve **conditional-launch threshold** heuristic | GPU | Cheap; unlocks the v3.3.0 C5 kernel at large N | [gpu-polyselect-ropt-c5](gpu-polyselect-ropt-c5.md) |
| C6+ | GPU GF(p) lingen NTT **multi-prime CRT wrapper** | GPU/DLP | Research; cluster/DLP play | [gpu-lingen-ntt-c6](gpu-lingen-ntt-c6.md) |
| B5 | IFMA GF(p) → `arith-modp` routing | SIMD/DLP | Carry-forward; HW-gated, documented | [ifma-gfp-b3](ifma-gfp-b3.md) |
| A6 | exTNFS / Tower-NFS skeleton | Research | Carry-forward; documented design | [extnfs-a4](extnfs-a4.md) |

## Sequencing

1. **E9** notifications + **E12c** completions/man — immediate UX, runs here.
2. **E10** event log + `/metrics`, **E11** runs.db — observability + campaign tracking.
3. **C7** GPU P-1/P+1 + adaptive B1 — the headline measured GPU win.
4. **A7** calibrate + regression autotuner — measurable planning accuracy.
5. **E12a/b** per-phase ETA + wizard — UX polish.
6. **C5+** root-sieve threshold heuristic — unlocks the v3.3.0 C5 kernel at large N.
7. **C6+**, **B5**, **A6** — DLP/cluster + research; HW/cluster-gated, documented.

## Gates (unchanged fork ethos)

- No changes to the core NFS algorithms or their parameters.
- After every change: full `make check` + seeded c60–c100 factorizations verified
  `product == N`; GPU/SIMD kernels re-verified bit-exact vs the CPU/GMP path (under
  Intel SDE for any AVX-512 piece); every prefactor-recovered factor re-verified
  `f | N`.
- Measured results reported honestly — **including negatives and HW-gated designs** —
  in `docs/` and `BENCHMARKS.md`.
- Do not bulk-reformat upstream C/C++/Python. Commit / push / release only when asked.

## Net target

The headline *measured-on-silicon* value is the stronger GPU prefactor front-end
(**C7** — P-1/P+1 + adaptive B1, the one non-Amdahl stage), plus a markedly better
operator experience (**E9–E12**) and a data-driven autotuner (**A7**). The
DLP/cluster research (**C5+, C6+, B5, A6**) is carried under the validation gate and
reported honestly. **No dishonest single-machine "speed win" is promised.**
