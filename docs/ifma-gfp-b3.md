# AVX-512 IFMA GF(p) for the BWC backend (Roadmap B3)

This documents roadmap item **B3**, *"finish the IFMA GF(p) backend → mpfq
integration"* — wiring the v3.1.0 validated AVX-512 IFMA modmul toward CADO's
GF(p) linear-algebra backend. The deliverable is the validated, representation-
compatible **batched primitive** plus an honest map of where it can and cannot
plug in.

## Background

3.1.0 (Track 1.4) shipped `bench/ifma-modmul.c`: an 8-way batched **Montgomery**
modular multiply in radix 2^52 (`_mm512_madd52lo/hi_epu64`), 260-bit, bit-exact
vs GMP under Intel SDE. AVX-512-IFMA does 8 independent 52×52→104 multiply-adds
per instruction, so the natural shape is *8 independent GF(p) modmuls at once*.

## What "mpfq" is now, and the two facts that decide the integration

CADO 3.0.0 **replaced mpfq** with the C++ `arith-modp` backend
(`linalg/bwc/arith-modp*.hpp`) — the per-size `p1`…`p8` GF(p) fields (64…512-bit)
used by the **DLP Block Wiedemann**. Reading it settles the integration shape:

1. **`arith-modp` stores elements PLAIN** — reduced in `[0,p)`, schoolbook
   `mul_ur` + a Barrett-style `reduce` (`arith-modp-main.hpp`). It is **not**
   Montgomery. The validated IFMA kernel is Montgomery (`a·b·R^{-1} mod p`), so it
   **cannot be dropped in as-is** — a representation bridge is required.
2. **The batched full-GF(p)-modmul sites are the vector ops** in
   `arith-generic.hpp`: `vec_add_dotprod` (`Σ aᵢ·bᵢ`) and
   `vec_addmul_and_reduce`. The scalar `mul()` (one element — wastes 7 of 8 lanes)
   and the SpMV (multiply-by-small-coefficient, not a full GF(p)×GF(p)) are **not**
   where an 8-wide modmul helps.

## The primitive (`bench/ifma-gfp.c`) — plain-representation, validated

Built on the validated Montgomery kernel, with **no per-element domain churn**
(per-element to/from-Montgomery conversion would cost as much as the work it
saves), using the identity `M(x,y)=x·y·R^{-1}`:

- **`plain_mul(a,b)` = `M(M(a,b), R²)` = `a·b mod p`** — plain in / plain out
  (two Montgomery muls; `R²` precomputed per field). This is exactly
  `arith-modp`'s `mul` semantics.
- **`dotprod(a[],b[]) = Σ aᵢ·bᵢ mod p`** — `acc = Σ M(aᵢ,bᵢ) = (Σ aᵢbᵢ)·R^{-1}`,
  then **one** final `M(acc, R²)`. The common `R^{-1}` amortizes: **n montmuls +
  1**, not 2n — the efficient shape for `arith-generic`'s `vec_add_dotprod`.

Both are plain-in/plain-out (matching `arith-modp`) and 8-way (one independent
GF(p) field per 512-bit lane — i.e. 8 RHS/independent fields processed together,
the BWC block-width / multi-sequence dimension).

### Validation — bit-exact vs GMP under Intel SDE

Comet Lake has no AVX-512-IFMA, so validated for correctness under
`sde64 -future` (same method as the gf2x VPCLMULQDQ work), wired into
`bench/ifma-validate.sh` + the `avx512-validate` CI:

| op | reference (GMP) | result |
|----|-----------------|--------|
| `plain_mul` (`a·b mod p`) | `mpz_mul; mpz_mod` | **PASS** 0/32000 |
| `dotprod` (`Σ aᵢbᵢ mod p`, 12 terms) | `mpz_mul; mpz_add; mpz_mod` | **PASS** 0/32000 |

`mpz_mul; mpz_mod` *is* `arith-modp`'s `mul` semantics, so this is a direct proof
that the IFMA path computes what the backend's GF(p) mul/dotprod compute, in the
backend's plain representation, at the 260-bit (`p4`/`p5`) field range.

## Status & honest scope

- **Done:** the representation-compatible batched IFMA GF(p) primitives
  (`plain_mul`, `dotprod`), validated bit-exact vs GMP under SDE — the missing
  piece the existing Montgomery-only kernel did not provide.
- **Remaining (the actual `arith-modp` code change):** route
  `vec_add_dotprod` / `vec_addmul_and_reduce` for the `p4`/`p5` fields through an
  IFMA path (limb repack 64-bit↔52-bit at the vector boundary, `R²` per field, the
  8-lane block mapping). That is a real change to the templated `arith-modp`
  vector ops and is **DLP-only**.
- **Perf is gated on real AVX-512-IFMA silicon** (Ice Lake / Sapphire Rapids+);
  SDE is functional-only. Honest expectation: `plain_mul` pays 2 montmuls per
  result (the price of plain representation), so the win concentrates in
  `dotprod`/`vec_addmul` (1 montmul/term, amortized `R^{-1}`) where 8-lane IFMA
  throughput can beat 8 scalar `arith-modp` muls.

## Reproducing

```bash
gcc -O2 -mavx512f -mavx512ifma bench/ifma-gfp.c -lgmp -o ifma-gfp
/opt/intel-sde/sde64 -future -- ./ifma-gfp        # or: bash bench/ifma-validate.sh
```

## Sources

- AVX-512-IFMA GF(p): the v3.1.0 kernel `bench/ifma-modmul.c` (Track 1.4).
- CADO-NFS `linalg/bwc/arith-modp*.hpp`, `arith-generic.hpp` — the GF(p) BWC
  backend that replaced mpfq.
- Montgomery multiplication: Koç, Acar, Kaliski, *Analyzing and comparing
  Montgomery multiplication algorithms* (CIOS).
