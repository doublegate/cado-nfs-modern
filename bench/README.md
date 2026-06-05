# bench/ — performance harnesses for cado-nfs 3.0.0-modern

Tooling added by the optimization effort (see the project CHANGELOG). Two parts:
a rigorous siever microbenchmark, and the AVX-512 VPCLMULQDQ gf2x kernel.

## las-microbench.sh — deterministic siever benchmark

End-to-end `./cado-nfs.py` timings vary ~15–20 % run-to-run because polynomial
selection is randomized — useless for measuring a change worth a few percent.
This harness pins the shipped c120 polynomial + factor base and uses
`las --random-sample N --seed S`, so the sieving workload is identical every run
(**<1 % total-CPU variance**). A/B two builds by pointing it at each `las`:

```bash
bench/las-microbench.sh build/$(hostname)/sieve/las 3
```

Used to settle Phase 1 (build flags): `-march=native` gives ~6 % on the siever,
`-O3` ~1 % more, **LTO 0 %** (and breaks `-Werror`), **PGO −3 %** (regresses).
Final config: `-O3 -march=native -mtune=native` (see `local.sh`).

## VPCLMULQDQ gf2x kernel (Phase 2, AVX-512)

`gf2x`'s base-case schoolbook multiply (`gf2x_mul_1_n`/`addmul_1_n`, the hot path
of GF(2)[x] multiplication used by CADO-NFS's GF(2) linear algebra / lingen)
does **one** 64×64 carryless multiply per `_mm_clmulepi64_si128` (128-bit). The
AVX-512 `_mm512_clmulepi64_epi128` (VPCLMULQDQ) does **four per instruction**;
processing 4 operand words per step is reported at up to ~39 % faster
(Drucker & Gueron, arXiv:2201.10473).

- **`vpclmul-mul1n.c`** — self-contained VPCLMULQDQ `mul_1_n`/`addmul_1_n` plus a
  scalar reference and a 200 000-trial bit-exactness self-test. Decoupled from
  the gf2x autotools build so the *math* can be validated independently first.
- **`vpclmul-validate.sh`** — compile + validate under Intel SDE.

### Status

- ✅ Kernel written; **compiles** with `-mavx512f -mvpclmulqdq`; disassembly
  confirms the real 512-bit `vpclmullqlqdq %zmm…` is emitted.
- ⏳ **Correctness validation pending Intel SDE** (the reference box is Comet
  Lake = no AVX-512, so the code cannot run natively — it would `SIGILL`).
  Install SDE (`paru -S intel-sde`), then:
  ```bash
  bench/vpclmul-validate.sh        # runs the 200k-trial test under `sde64 -future`
  ```
- ⏳ **Performance** deferred to real AVX-512 silicon (SDE is functional-only).

### Remaining gf2x integration (after SDE correctness sign-off)

1. Add `gf2x/lowlevel/mul1vcl.c` — the same kernel using gf2x's
   `GF2X_STORAGE_CLASS_*` / `_gf2x_mm_*` macros (model: `mul1cl.c`), guarded by a
   new `GF2X_HAVE_VPCLMUL_SUPPORT`.
2. `gf2x/configure.ac` — add a `gf2x_cv_cc_supports_vpclmul` check and an
   `x86_64_vpclmul` `hwdir` (model: the existing `x86_64_pclmul` selection at
   lines ~154–186).
3. Runtime CPUID dispatch so the VPCLMULQDQ path is chosen only on capable CPUs.
4. Re-run gf2x's threshold tuning (a faster base case shifts the Karatsuba/Toom
   crossovers), then benchmark `lingen` on real AVX-512 hardware.
