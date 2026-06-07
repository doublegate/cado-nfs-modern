# GPU batch-smoothness leaf extraction (Roadmap C3)

This documents roadmap item **C3**, *"GPU batch-smoothness product tree …
flag-gated alternative to per-cofactor ECM in heavy-`mfb` regimes; reuses the
validated device arithmetic."* It delivers the **A2-arithmetic-reusable part**
of the algorithm, validated bit-exact, plus an honest measurement of where the
batch-smoothness cost actually sits.

## What batch smoothness is (for newcomers)

After sieving, each surviving relation leaves a **cofactor** — the part of the
norm the factor base did not divide out. A relation is usable only if its
cofactor is **smooth** (all prime factors ≤ the large-prime bound `2^lpb`).
Instead of factoring each cofactor separately (per-cofactor ECM), **batch
smoothness** (Bernstein, *How to find small factors of integers*, Alg. 7.1)
tests *many cofactors at once*:

1. `P` = product of **all** primes ≤ `2^lpb` (one huge integer).
2. **Product tree** of the cofactors `R[j]`: leaves `R[j]`, root = `∏ R[j]`.
3. **Remainder tree**: descend computing `P mod node`, giving `P mod R[j]` at
   every leaf in one shared pass.
4. **Leaf extraction**: from `y = P mod R`, the smooth part of `R` is
   `gcd(R, y^(2^e) mod R)` (the powering folds in prime multiplicities);
   `R` is smooth ⇔ smooth part `== R`.

CADO already implements all of this on CPU (OpenMP) in `sieve/ecm/batch.cpp`
(the Bouvier–Imbert batch cofactorization upstream 3.0.0 ships).

## What maps onto the GPU (and what doesn't)

- **Steps 2–3 (the trees) are big-integer** — the product-tree nodes grow toward
  a multi-million-bit root, and the remainder tree needs arbitrary-precision
  multiply and division. This is GMP/CPU territory; it is **not** a fit for the
  fixed-K-limb Montgomery arithmetic from the GPU ECM (A2 / `bench/gpu-ecm-mp.cu`).
- **Step 4 (leaf extraction) fans out to `n` independent, bounded-width
  cofactors** and is exactly the fixed-width Montgomery regime A2 excels at. Using
  Bernstein's powering variant, each leaf is `e` modular squarings + one gcd —
  **`montmul` is the A2 kernel**, and since `gcd(R, y·2^{64K}) = gcd(R, y)` for
  odd `R`, the gcd runs directly on the Montgomery-form result (no leave-Montgomery,
  no big-integer division at all).

So C3's GPU-amenable, A2-reusing core is the **leaf extraction**, implemented and
validated here in `bench/gpu-batch-smooth.cu`.

## Validation — bit-exact vs GMP

For each width K∈{2,4,8} (128/256/512-bit cofactors), 8192 cofactors are built
with known structure (half smooth = products of primes ≤ `B=2^20`; half rough =
one prime `> B`), `y = P mod R` is computed with GMP, and the GPU computes the
smooth part `gcd(R, y^(2^e) mod R)`. Checked against an **independent GMP ground
truth** (iterated `gcd`-with-`P`): both the **smooth part** (bit-exact) and the
**smooth/rough classification** must match.

| width | smooth-part vs GMP | classification vs GMP | throughput |
|-------|:------------------:|:---------------------:|-----------:|
| 128-bit | **PASS** 0/8192 | **PASS** 0/8192 | 20.6 Mleaf/s |
| 256-bit | **PASS** 0/8192 | **PASS** 0/8192 | 1.45 Mleaf/s |
| 512-bit | **PASS** 0/8192 | **PASS** 0/8192 | 1.60 Mleaf/s |

(`e` = 7/8/9 Montgomery squarings at 128/256/512-bit, i.e. `2^e ≥ bits(R)` to
cover any prime multiplicity.) The leaf extraction is **correct and very fast**:
0.05 µs/leaf at 128-bit, ~0.6–0.7 µs/leaf at 256/512-bit.

## Honest reading — the leaf is correct but not the bottleneck

The leaf extraction is *cheap by design*: ~`e` squarings + a gcd over a
bounded-width cofactor. The dominant cost of batch smoothness is the
**big-integer remainder tree** (step 3). For scale, a naive un-amortized
`P mod R` (P is 1.5 M bits at `B=2^20`) is ~170 µs per cofactor; the remainder
tree amortizes that big-integer work to ~`O(M(N) log N / n)` per leaf, but it is
**still arbitrary-precision and stays on CPU/GMP** — it is not expressible in
A2's fixed-K-limb arithmetic.

Therefore:

- The **A2-reusable part of C3 is done and validated** — a correct, fast GPU leaf
  extractor (`bench/gpu-batch-smooth.cu`).
- A **full** GPU batch-smoothness path would require porting the product/remainder
  **trees** to GPU big-integer arithmetic (Karatsuba/FFT multiply + a
  remainder-tree division) — a separate, substantial effort, *not* a reuse of A2.
  This is recorded as the remaining piece (a documented design, not committed
  unvalidated code), consistent with the fork's HW/scope-gating ethos.
- For **GPU cofactorization overall, ECM (A2) remains the better single-machine
  fit**: it is entirely fixed-width (no big-integer trees), embarrassingly
  parallel, and was just made ~1.5–2.9× faster (see `docs/gpu-ecm-mixedrep.md`).
  Batch smoothness is the CPU/GMP complement (`sieve/ecm/batch.cpp`), with its
  one GPU-friendly stage validated here.

## Reproducing

```bash
nvcc -arch=sm_86 -O3 bench/gpu-batch-smooth.cu -lgmp -o gpu-batch-smooth && ./gpu-batch-smooth
```

## Sources

- Bernstein. *How to find small factors of integers* (Algorithm 7.1),
  cr.yp.to/papers.html#sf.
- Bernstein. *How to find smooth parts of integers.*
- CADO-NFS `sieve/ecm/batch.cpp` — the upstream CPU batch cofactorization
  (product/remainder trees, OpenMP).
- A2 device arithmetic: `bench/gpu-ecm-mp.cu`, `docs/gpu-ecm-mixedrep.md`.
