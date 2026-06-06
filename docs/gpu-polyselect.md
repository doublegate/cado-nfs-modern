# GPU polynomial selection (v3.2.0-modern, Track C2)

Polynomial selection is the first NFS stage and, per the v3.2.0 reframe, one of
the two highest-leverage phases (the other is sieving). It is also **proven
GPU-friendly**: msieve has driven Kleinjung stage-1 on the GPU since 2009. CADO's
polyselect is CPU-only — this track adds a GPU path. Like the GPU ECM / linalg
work, it starts from a profile and a bit-exact foundation kernel, not a guess.

## Where the time goes (measured)

A `perf` profile of CADO `polyselect` stage-1 (collision search, c90-sized N,
degree 5, `-P 7000`) on the i9-10850K, top self-time:

| function | self | what it is |
|---|---:|---|
| `modredcul_intinv` | **16 %** | modular inverse mod the small primes p |
| `double_poly_compute_roots` | 13 % | real-root finding for `L2_skewness` (size scoring) |
| `modul_poly_div_r` | 6 % | polynomial division mod p |
| `modul_poly_xpowmod_ui` | 5 % | `x^p mod f` (root finding mod p) |
| `polyselect_proots_dispatch_to_shash_flat` | 5 % | collision hash dispatch |
| `invert_q2_mod_all_p2` + `modcalc` subtasks | 6 % | the q²-mod-p² collision arithmetic |

Two clusters dominate:

1. **Per-prime modular root finding** (`modul_poly_roots` → `xpowmod` + `div_r` +
   `modredcul_intinv`): ~30–40 % of stage-1. For each of thousands of small primes
   p, find the roots of the algebraic polynomial mod p. **Independent per prime →
   the GPU sweet spot** (exactly the shape of the batched GPU ECM).
2. **Size scoring** (`L2_skewness` → `double_poly_compute_roots`): ~15 %, a
   floating-point Newton/root computation per candidate — also batchable.

The **collision search** (hash dispatch, ~10 %) is the memory-bound part msieve
keeps on GPU too, but is the harder/secondary target.

## Done — foundation kernel (bit-exact)

`bench/gpu-polyselect-modinv.cu`: a GPU batched **single-word modular inverse**
(the 16 % hottest leaf), validated **bit-exact vs GMP** over 200 000 (a, p) pairs
(0 wrong; 469 M inv/s on an RTX 3090). The modular inverse is mathematically
unique, so vs-GMP is the meaningful gate (independent of CADO's REDC
representation). This proves the per-prime modular arithmetic runs correctly on
the GPU and gives the building block the root-finding kernel needs.

## Plan — the GPU polyselect path

1. **Batched per-prime root-finding kernel** (the ~30–40 % target): for a batch of
   primes p, compute the roots of `f mod p` via `gcd(x^p − x, f) mod p` (repeated
   squaring for `x^p mod f`, then root extraction), reusing the single-word REDC /
   modinv arithmetic validated above. Bit-exact vs CADO's `modul_poly_roots` over a
   prime batch.
2. **Feed the collision search**: stream the GPU `proots` into the existing
   `shash` collision machinery (keep the hash/match on the side that wins — likely
   CPU first, GPU later), so the GPU computes roots while the CPU collides.
3. **GPU size scoring** (`L2_skewness`): batch the skewness/`double_poly_compute_
   roots` evaluation for candidate polynomials.
4. **Integration + gate**: wire behind a `--gpu-polyselect` flag (default off,
   like `--gpu-prefactor`); the gate is **same polynomial quality** — the selected
   polynomial's Murphy-E (and the resulting relation yield) must match the CPU
   path, validated end-to-end by a `product == N` factorization that uses the
   GPU-selected polynomial.

## Honest scoping

- The full path (steps 1–4) is a substantial module — a new GPU kernel for
  polynomial arithmetic mod p plus careful integration with CADO's collision
  search and the `ropt` (stage-2 root optimization, the BBKZ algorithm, the other
  large chunk of polyselect time). It lands incrementally, each step gated.
- This is a single-machine win (polyselect is embarrassingly parallel and grows
  with N), complementary to distributing polyselect across the work-unit clients
  (Track D / the existing orchestration).
