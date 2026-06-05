# x86_64_vpclmul gf2x backend — integration status & completion guide

AVX-512 **VPCLMULQDQ** lowlevel backend for gf2x, used by CADO-NFS's GF(2)
linear algebra (`lingen`). `_mm512_clmulepi64_epi128` does 4 carryless multiplies
per instruction, so the base-case `gf2x_mul_1_n`/`addmul_1_n` process 4 operand
words per step (vs 2 per 128-bit `_mm_clmulepi64_si128` in `x86_64_pclmul`).
Ref: Drucker & Gueron, arXiv:2201.10473 (~39% on GF(2)[x] mul).

## Done

- **`gf2x_mul1.h`** — VPCLMULQDQ `gf2x_mul_1_n` / `gf2x_addmul_1_n` (the hot base
  case). The exact 4-wide fold logic is **validated bit-exact over 200 000 random
  trials under Intel SDE** (`bench/vpclmul-mul1n.c` / `bench/vpclmul-validate.sh`,
  `PASS`). Compiles to the real 512-bit `vpclmullqlqdq %zmm…`.
- The rest of this directory (`gf2x_mul2.h … gf2x_mul9.h`, tuning tables) is
  copied from `x86_64_pclmul`, so the backend is **complete and buildable**: the
  hottest multiplier (`mul1`) is VPCLMULQDQ-accelerated, the larger ones fall
  back to the proven PCLMUL code until they're ported too.

## Remaining (requires AVX-512 silicon for the perf payoff)

gf2x selects ONE `hwdir` at **configure time** (no runtime dispatch in this
1.1-era gf2x), which fits CADO's build-from-source-per-machine model: build on
the AVX-512 host with `-march=native` and the VPCLMULQDQ backend is selected.

1. **`configure.ac`** — add VPCLMULQDQ detection mirroring pclmul:
   - Find the m4 macro that sets `gf2x_cv_cc_supports_pclmul` (a
     `GF2X_CHECK_*`-style macro in the bundled m4) and clone it as
     `gf2x_cv_cc_supports_vpclmul` — its test program uses `__m512i` +
     `_mm512_clmulepi64_epi128`, compiled with `-mavx512f -mvpclmulqdq`.
   - In the `core2|opteron|x86_64|…` case (≈ line 164), prefer the new backend:
     ```
     if   test "$gf2x_cv_cc_supports_vpclmul" = yes; then hwdir=x86_64_vpclmul
     elif test "$gf2x_cv_cc_supports_pclmul"  = yes; then hwdir=x86_64_pclmul
     else hwdir=x86_64 ; fi
     ```
   - Emit `GF2X_HAVE_VPCLMUL_SUPPORT` (same mechanism as
     `GF2X_HAVE_PCLMUL_SUPPORT`) and ensure `-mvpclmulqdq` reaches the lowlevel
     compile flags for this hwdir.
2. **Regenerate** `configure` (`autoreconf -i` in `gf2x/`), since CADO builds gf2x
   from the committed `configure`.
3. **Build-test under SDE**: force the backend
   (`./configure --enable-hardware-specific-code` on a host where the vpclmul
   test passes — or temporarily hard-set `hwdir`), build with
   `-mavx512f -mvpclmulqdq`, and run gf2x's `tests/` multiply checks under
   `sde64 -future`.
4. **Port `mul2.h … mul9.h`** to VPCLMULQDQ for the full speedup (the paper's
   bigger gains are at 256/512-bit operands), then re-run gf2x's **threshold
   tuning** (a faster base case shifts the Karatsuba/Toom crossovers).
5. **Benchmark `lingen`** on real AVX-512 hardware — the only place an actual
   speedup number can be obtained (this dev box is Comet Lake; SDE is
   functional-only, not a performance model).
