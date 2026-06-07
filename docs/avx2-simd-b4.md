# AVX2 batched modular inverse on real silicon (Roadmap B4)

> **Status: in progress (v3.3.0-modern).** The fork's **first SIMD kernel that
> actually runs on the reference CPU** (Comet Lake i9-10850K — AVX2, no AVX-512),
> so it yields a *measured* number rather than an SDE-only correctness proof. See
> [`ROADMAP-v3.3.0-modern.md`](ROADMAP-v3.3.0-modern.md).

## Why AVX2, and why only the modinv

The v3.2.0 B-series (B1 AVX-512 batched modinv, B2 VPCLMULQDQ gf2x, B3 IFMA GF(p))
is bit-exact under Intel SDE but **silicon-gated**: this CPU has no AVX-512, so none
of it produces a measured speedup here. Of the three, **only B1's modular inverse
has a meaningful AVX2 analog**:

- The siever's per-prime lattice setup computes `invmod_redc_32` (a 32-bit modular
  inverse) per factor-base prime (`sieve/las-arith.h`). At 32 bits, an AVX2 register
  (256-bit) holds **8 lanes** — an 8-way batched binary-extended-GCD modular
  inverse. **Montgomery's batch-inversion trick does _not_ apply here:** it compresses
  `n` inversions in **one ring** to a single inversion plus `3n` multiplies, but each
  factor-base prime is a **different modulus**, so the lanes must run independent
  inverses. (The AVX-512 B1 kernel notes the same.)
- **B2 (VPCLMULQDQ) and B3 (IFMA) have no AVX2 path** — carry-less `VPCLMULQDQ` on
  YMM and `VPMADD52` (IFMA) are AVX-512-era features absent on Comet Lake. They stay
  SDE-gated; this doc does not claim otherwise.

## What B4 delivers

`bench/avx2-modinv.c` is an AVX2 8-way batched 32-bit modular inverse: the binary
extended-GCD as a **per-lane masked state machine**, ported from the AVX-512 B1
kernel. AVX2 has no k-mask registers, so the masking uses vector masks
(`_mm256_cmpgt_epi32` / `_mm256_cmpeq_epi32`, all-ones per lane) and
`_mm256_blendv_epi8` for conditional moves. Because `U, V, X1, X2` all stay below
`2^31` (the modulus `b < 2^31`, the cofactors live in `[0, B)`), the high bit is
always clear and **signed AVX2 compares coincide with the unsigned compares** the
algorithm needs — sidestepping AVX2's lack of unsigned 32-bit comparison.

## Measured — on the silicon (no SDE)

The reference box (Comet Lake i9-10850K) **runs this natively**, so it yields the
fork's first *measured* batched-modinv SIMD result (`bench/avx2-modinv-validate.sh`):

| | ns / inverse | over 2^20 inverses |
|---|---|---|
| scalar binary-GCD | ~193 | 0.203 s |
| **AVX2 8-way** | **~42** | 0.044 s |
| **speedup** | **~4.6×** | bit-exact (0 mismatches) |

Correctness: **PASS, 0 wrong over 320 000 trials vs GMP `mpz_invert`**, and 0
scalar-vs-AVX2 mismatches over the 2^20-inverse timing run.

## Honest scope

This is the *kernel* result. End-to-end siever impact is **Amdahl-bounded**: the
per-prime modular inverse is ~9.5 % (plus part of `plattice`) of siever self-time,
and the byte-scatter majority (~29 %) stays scalar (no 8-bit SIMD scatter — the C4
wall). A ~4.6× on the modinv slice therefore caps the whole-siever gain at roughly
1.05–1.10× even if the rest were free; the real integrated delta must be *measured*,
not assumed, when wired into the live `plattice` setup. The committed deliverable is
the validated, measured kernel; the in-siever integration (behind an AVX2 runtime
guard with the scalar path as fallback, `product == N` preserved) is the next step.
