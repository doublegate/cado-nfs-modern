# AVX-512 sieving — batched modular inverse (Roadmap B1)

This documents roadmap item **B1**, *"AVX-512 block + bucket sieving … vectorize
the sieve-index calculation, sieve-array updates, and bucket insertion."* The
honest result: most of the siever is **scatter-bound and does not vectorize on
AVX-512 either** (confirming + extending the 3.1.0 AVX2 negative); the one
genuinely vectorizable hot slice — the **per-prime modular inverse** of the
lattice setup — is implemented as an AVX-512 16-way kernel and validated bit-exact.

## Where the siever spends time (measured, v3.1.0 Track 1.3)

A `perf` profile of the c120 microbench:

| routine | self-time | nature |
|---------|----------:|--------|
| `fill_in_buckets` | ~12% | **scatter** (push updates to bucket regions) |
| `plattice_info` ctor | ~11% | modular arithmetic (per-prime lattice reduction) |
| `sieve_small_bucket_region` | ~10% | **byte scatter** (sieve[r], r+p, …) |
| `invmod_redc_32` | ~9.5% | **modular inverse** (per prime) |
| `apply_buckets_inner` | ~7% | **scatter** (already 16×-unrolled SIMD) |

## Why the scatter loops don't vectorize on AVX-512 (extends the 3.1.0 finding)

3.1.0 ruled out AVX2 on the siever ("the hot path is scatter plus scalar modular
arithmetic, not vectorizable loops"). AVX-512 does add scatter
(`vpscatterdd`/`vpscatterqd`) and conflict detection (`vpconflictd`) that AVX2
lacks — but **the sieve array is `uint8` (log approximations), and AVX-512 scatter
is 32/64-bit-element only**: there is *no 8-bit scatter*. So `fill_in_buckets`,
`sieve_small_bucket_region`, and `apply_buckets_inner` — the byte-scatter majority
(~29%) — stay scalar on AVX-512 too. `apply_buckets_inner` is already
16×-unrolled with batched 64-bit reads + prefetch and a cache-resident target by
design. This is an honest scatter wall, not a missing optimization.

## The vectorizable slice — per-prime modular inverse

The arithmetic part — `invmod_redc_32` (~9.5%), which feeds `plattice_info`
(~11%) — is pure 32-bit integer work. `invmod_redc_32` is a **binary
extended-GCD inverse mod p with a DIFFERENT modulus per prime**, so Montgomery's
batch-inversion trick does *not* apply (it needs one shared ring). But **16
independent inverses map cleanly onto AVX-512 32-bit lanes**: run the binary-GCD
inverse as a **masked per-lane state machine**, looping until every lane is done.
This is the published AVX-512 sieve-index angle (SECRYPT 2021), genuinely
distinct from the ruled-out AVX2 path (which had no efficient per-lane masking).

`bench/avx512-modinv.c` implements `modinv16`: 16 lanes, each computes
`a⁻¹ mod b` (b odd, `gcd(a,b)=1`) via the binary extended GCD, with per-lane
masks selecting one primitive step (halve-`u` / halve-`v` / subtract-and-update)
per iteration, `halve_mod`/`submod` done with `_mm512_mask_*`. It computes the
**plain** inverse `a⁻¹ mod b`; the REDC `2⁻³²` normalization `invmod_redc_32`
adds is a cheap deterministic per-lane fixup folded in at integration. Supports
`b < 2³¹` (covers factor-base primes in this regime; keeps the arithmetic in
`uint32`, since `x,b < 2³¹ ⇒ x+b < 2³²`).

### Validation — bit-exact vs GMP under Intel SDE

Comet Lake has no AVX-512, so validated for correctness under `sde64 -future`
(same method as the gf2x VPCLMULQDQ / IFMA work), wired into
`bench/avx512-modinv-validate.sh` + the `avx512-validate` CI:

> **AVX-512 16-way batched 32-bit modular inverse vs GMP: PASS, 0 wrong /
> 640 000 trials.**

## Honest scope

- **Done + validated:** the AVX-512 16-way batched modular inverse — the siever's
  one vectorizable hot slice (~9.5% + part of the 11% lattice setup).
- **Confirmed un-vectorizable:** the byte-scatter majority (~29%:
  `fill_in_buckets`, `sieve_small_bucket_region`, `apply_buckets_inner`) — no
  8-bit AVX-512 scatter exists; this is a hard memory wall, extending the 3.1.0
  AVX2 negative to AVX-512.
- **Remaining (integration):** route `plattice_info`'s per-prime inverse through
  `modinv16` by batching 16 primes' lattice setups, with the REDC `2⁻³²` fixup and
  the `b ≥ 2³¹` tail on the scalar path. 3.1.0 flagged this as an invasive
  restructuring of per-prime lattice setup; the kernel here de-risks the
  arithmetic half. **Net siever speedup is Amdahl-bounded** (the scatter majority
  is fixed) and **gated on real AVX-512 silicon** (SDE is functional-only).

## Reproducing

```bash
gcc -O2 -mavx512f -mavx512cd bench/avx512-modinv.c -lgmp -o avx512-modinv
/opt/intel-sde/sde64 -future -- ./avx512-modinv   # or: bash bench/avx512-modinv-validate.sh
```

## Sources

- AVX-512 sieving / sieve-index vectorization: SciTePress/SECRYPT 2021 (105152).
- CADO-NFS `sieve/las-arith.hpp` (`invmod_redc_32`), `sieve/las-plattice.hpp`
  (`plattice_info`); the v3.1.0 profile (CHANGELOG Track 1.3).
