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

## Done — foundation kernels (bit-exact)

1. **Modular inverse** (`bench/gpu-polyselect-modinv.cu`): a GPU batched
   single-word modular inverse (the 16 % hottest leaf), validated **bit-exact vs
   GMP** over 200 000 (a, p) pairs (0 wrong; 469 M inv/s on an RTX 3090). The
   inverse is mathematically unique, so vs-GMP is the meaningful gate (independent
   of CADO's REDC representation).
2. **Per-prime root finding** (`bench/gpu-polyselect-roots.cu`): for a fixed
   degree-d polynomial f and a batch of primes p, find all roots of f mod p — one
   thread per prime, direct Horner evaluation over F_p. Exactly correct by
   construction (roots = {a : f(a) ≡ 0}); validated **bit-exact vs a CPU reference
   + a self-check** (every root satisfies f(r) ≡ 0): 0 mismatch / 0 self-check-bad
   over 5133 primes (deg 6). **GPU 45.9 ms vs CPU 20-thread 277.6 ms = 6.0×.**
   *Honest limitation:* direct evaluation is **O(p) per prime**, so it is a win
   only in the small-prime regime — one-thread-per-prime is load-imbalanced and
   slow for large p. The asymptotically-better method (next sub-step) is
   `gcd(x^p − x, f) mod p` (O(d² log p), independent of p's magnitude), built from
   polynomial arithmetic mod f reusing the validated modular inverse — **done, #3.**
3. **gcd-based root finding** (`bench/gpu-polyselect-roots-gcd.cu`): the
   asymptotically-better method. Per prime: `h = x^p mod f` by binary
   exponentiation (polynomial multiply mod f), `g = gcd(h − x, f)` (whose roots are
   exactly f's roots in F_p), then **Cantor–Zassenhaus** split of g into linear
   factors (`(x+δ)^((p−1)/2) mod g`, iterated δ). All over F_p with the validated
   single-word modular arithmetic; `__host__ __device__` so GPU and CPU run
   identical code. **Validated bit-exact vs direct-eval** (full root multiset):
   0 mismatch / 0 self-check-bad over 3245 primes (deg 6, p < 30 000). **Key win —
   p-magnitude-independent:** 5000 primes near **10⁹** in **27.8 ms** (4981 roots,
   0 self-check-bad), where direct evaluation's O(p) (~10⁹ steps/prime) is
   hopeless. This is the production root-finder for the full prime range.

Together these prove the per-prime modular arithmetic **and** root finding (both
small- and large-prime regimes) run correctly on the GPU — the building blocks the
collision-feed needs.

## Plan — the GPU polyselect path

1. **Batched per-prime root-finding kernel** (the ~30–40 % target): **done** —
   direct-eval for the small-prime regime (6×) and the gcd(x^p − x, f) + CZ method
   for the full range (p-magnitude-independent), both validated bit-exact vs
   direct-eval. Remaining: validate against CADO's exact `modul_poly_roots` output
   over a prime batch and wire it as a drop-in for that call.
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
