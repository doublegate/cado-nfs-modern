# GPU ECM cofactorization (Phase 3)

CADO-NFS factors the small "cofactors" of sieve survivors with ECM
(`sieve/ecm/`, entered via `facul_all(std::vector<cxx_mpz>, strategies)` — a
**batch** of small moduli, each run through ~11–14 ECM curves). ECM is
**modular-arithmetic-bound** on small (≤ mfb ≈ 52–96 bit) moduli, and the batch
is thousands of independent curves — the ideal GPU workload. Offloading it frees
CPU cores to sieve (Bos & Kleinjung; eprint 2014/397).

## Measured on this box (RTX 3090, sm_86, vs i9-10850K)

`bench/gpu-modmul-bench.cu` — batched 64-bit Montgomery modmul (ECM's inner
primitive), 1 M independent moduli, **correctness PASS vs CPU**:

| | throughput |
|---|---|
| RTX 3090 | **279 G modmul/s** |
| CPU, 1 core | 0.36 G/s |
| CPU, 20 cores (est.) | 7.2 G/s |
| **GPU ÷ 20-core CPU** | **~39×** |

So the GPU does ECM's core op ~39× faster than the whole CPU.

## What that does and doesn't mean (honest Amdahl analysis)

- The **cofactorization phase** can be ~39× cheaper — but in the c120 siever
  profile, cofactorization (`factor`) is only **~8 %** of `las` CPU time
  (`sieving` dominates at ~70 %). So *naively* offloading the current
  cofactorization frees ~8 % of CPU for more sieving — a real but modest
  end-to-end gain.
- **The bigger win is a parameter-regime shift.** CADO uses modest `ncurves`
  (11–14) and `mfb` partly *because* cofactorization is expensive on CPU. With a
  ~39×-cheaper GPU cofactorization you can afford **many more curves / larger
  `mfb` / an extra large prime**, which raises the relations-per-special-q yield
  and rebalances sieve-vs-cofactor — potentially a larger net throughput gain,
  and the main reason GPU cofactorization matters in practice (and more so for
  DLP, where cofactorization is a larger fraction).

## Working GPU ECM stage-1 (`bench/gpu-ecm.cu`)

A complete batched ECM stage-1 — Montgomery-curve XZ ladder for `[∏ p^e]P`
(prime powers ≤ B1), on-device binary gcd — runs one curve per thread. The same
`ecm_stage1` compiles for host and device, so the GPU is checked bit-exact:

```
validation   : PASS (0/2048 GPU lanes differ from CPU)
factors found: 256/256 composites cracked by >=1 curve   (B1=2000, 64-bit n)
throughput   : ~2.75M curves/s on the RTX 3090
```

At CADO's ~13 curves/cofactor that is ~210 k cofactors/s. This is the core of the
backend; the remaining work is integration + refinement.

## Integration roadmap (remaining)

1. ~~GPU ECM stage-1 kernel~~ ✅ done (`bench/gpu-ecm.cu`): validated bit-exact,
   finds factors, benchmarked. Refinements:
   - ✅ **Suyama-σ parametrization** (`bench/gpu-ecm-suyama.cu`) — curves with
     guaranteed torsion 12; device modular inverse (with lucky-factor-in-setup);
     ~1.08× more composites cracked at a fixed low curve budget vs the POC curve.
   - ✅ **128-bit (2-limb) Montgomery modmul** (`bench/gpu-mont128.cu`) — CIOS
     REDC, validated bit-exact on CPU and GPU over 200 000 trials; the core for
     larger-`mfb` cofactors. Full 128-bit ECM = the validated 64-bit ladder/curve
     structure over this modmul.
   - ✅ **stage 2 (BSGS continuation)** (`bench/gpu-ecm-stage2.cu`) — baby steps
     `[r]Q`, giant steps `[mW]Q`, accumulated cross-differences with one final
     gcd; validated bit-exact GPU vs CPU, and finds **+64% more factors per
     curve** (1011 → 1663 / 2048 composites at B1=2000, B2=50000).
2. ✅ **Library module** `sieve/ecm/gpu_ecm.{hpp,cu}` — packages the validated
   stage-1 + stage-2 kernel behind a plain-C++ batch API
   (`gpu_ecm::factor_batch(moduli, ncurves, B1, B2, factor)` +
   `gpu_ecm::available()`). Standalone-tested: **1000/1000 moduli cracked, every
   returned factor divides its modulus.** `gpu_ecm_stub.cpp` is the non-CUDA
   fallback so the CPU path is unaffected when CUDA is off.
3. ✅ **CMake CUDA build** — `config/cuda.cmake` (`option(ENABLE_GPU)`,
   `check_language(CUDA)`, `CUDAToolkit`, default `sm_86`) wired into the `facul`
   library (`sieve/ecm/CMakeLists.txt`). **Configures cleanly with
   `-DENABLE_GPU=ON`** (CUDA 13.3 detected; pass `-DCMAKE_CUDA_ARCHITECTURES=86`
   for the 3090). CPU build is unchanged when the flag is off.
4. **Survivor-batch hook.** Architectural finding: `facul_all(N, …)` receives the
   **2 cofactors of one relation**, so the GPU must hook one level up, where
   survivors already accumulate. CADO 3.0.0 collects all survivors in
   **`las.survivors.L`** (a `list<(special_q, list<cofac_candidate>)>`, each
   `cofac_candidate` = `{a, b, vector<cxx_mpz> cofactor}`), drained for
   cofactoring at `sieve/las.cpp:~997` (batch mode).
   - ✅ **Bridge done** — `sieve/ecm/gpu_cofac.{hpp,cpp}`: a `cxx_mpz`↔`uint64`
     survivor-batch flush (`gpu_ecm::cofac_batch(cofactors, ncurves, B1, B2)`)
     that gathers the eligible single-word cofactors, runs ONE GPU launch, and
     returns a found factor per cofactor (1 = none → CPU path). Plain C++ (GMP),
     calls the validated `gpu_ecm::factor_batch`; **syntax-checks clean against
     CADO headers**; built into the `facul` library.
   - ✅ **Live hook wired** — `factor_leftover_norms` (`sieve/las-cofactor.cpp`,
     the single per-survivor cofactoring chokepoint reached by the default
     `detached_cofac` path) now calls the GPU ECM batch when the `CADO_GPU_ECM`
     environment variable is set and `gpu_ecm::available()`. **Default OFF**, so
     the stock path is untouched. Two modes:
     - `shadow` (`CADO_GPU_ECM=1|shadow`) — **identity-preserving**: runs GPU ECM
       on the real leftover cofactors but does *not* change `facul`'s verdict, so
       the relation set is byte-for-byte identical. The safe validation hook.
     - `salvage` (`CADO_GPU_ECM=salvage`) — retries only `facul` give-ups
       (`FACUL_MAYBE`); upgrades to a smooth relation **only** when the GPU fully
       splits the cofactor into two primes within the large-prime bound. Emits a
       *valid superset* of the CPU-only relations, never a wrong one (the
       product==norm invariant holds; `relation::compress()` sorts each side, and
       factorization is unique, so discovery order cannot change a relation).
   - ✅ **Validated relations-identical** (c120 poly, special-q 600000–603000,
     `-t 1`): **17986 relations** in every mode. `shadow` is **byte-identical** to
     CPU-only while the GPU split **37023 real survivor cofactors** across 29953
     cofactoring calls; `salvage` gave +0/−0 (no `MAYBE` give-ups arose). This
     proves the CUDA-enabled `las` offloads real cofactors to the RTX 3090 with
     **zero change to emitted relations**.
   - ✅ **Batched drain implemented** (`CADO_GPU_ECM=batch`) —
     `cofactoring_sync` (`sieve/las-process-bucket-region.cpp`) now collects a
     whole bucket region's async survivors, issues **one GPU ECM launch per
     region** over all their leftover cofactors, and stores a per-side factor
     hint in `cofac_standalone::gpu_hint`. `factor_leftover_norms` then **divides
     the hint out** (when it is a prime ≤ 2^lpb) so `facul` factors the smaller
     remainder, re-attaching the prime so `product==norm` holds. Correctness
     validated (c120, special-q 600000–603000): **26645 relations vs 17986
     CPU-only, +8659 extra, −0 lost** — a clean valid superset (the GPU resolves
     cofactors `facul` would give up on; `bench/gpu-cofac-validate.sh` /
     `validate-batch`).
   - ⚠️ **Honest throughput result: not a net win in this regime.** Same c120
     slice, RTX 3090 + i9-10850K: batch **95 s vs 25 s** wall (`-t 1` and `-t 20`
     both), i.e. **278 vs 718 rel/s**. Two compounding reasons, both predicted by
     the Amdahl analysis above: (1) cofactorization is only ~8 % of `las` CPU
     time, so the offload ceiling is ~8 % even with perfect overlap; (2) the
     blanket pre-pass runs a full 16-curve, B2=50000 ECM on **every** eligible
     cofactor — including the majority `facul` cracks in microseconds — so the
     added GPU work and the +48 % extra relations to emit dominate the small
     cofactoring saving. (The narrow slice is also fixed-cost/FB-load bound and
     does not thread-scale, so it understates overlap; but Amdahl caps the win
     regardless.)
   - ⏳ **To make it a win** (the real follow-on): (a) **target only hard
     cofactors** — send to the GPU just the upper-bitsize cofactors near `mfb`
     where `facul`'s curve budget tends to fail, not the easy ones; (b) the
     **parameter regime-shift** — raise `mfb`/`ncurves`/add a large prime so
     cofactoring becomes a large enough CPU fraction to be worth offloading, then
     re-benchmark; (c) a truly **async GPU stream** so the launch overlaps
     sieving instead of blocking the region thread.
5. ✅ **Full in-CADO CUDA build works.** With `-DENABLE_GPU=ON`
   `-DCMAKE_CUDA_ARCHITECTURES=86`, `gpu_ecm.cu` compiles under `nvcc` inside
   CADO's C++20/`-Werror` build (no flag conflicts), `gpu_cofac.cpp` compiles,
   and the **entire suite** (`polyselect`, `las`, `makefb`, `purge`, `merge`,
   `bwc`, `sqrt`, …) builds; `las` links `libcudart` with the
   `gpu_ecm::{available,factor_batch,cofac_batch}` symbols present and the hook
   live (per-call `shadow`/`salvage` and the per-region `batch` drain). ⏳
   Remaining: make `batch` a net win via hard-cofactor targeting + the parameter
   regime-shift + async overlap (item 4).

## Status

- ✅ CUDA 13.3 + RTX 3090 verified end-to-end; CADO had no prior GPU code.
- ✅ **Full GPU ECM engine built & validated**: stage-1, Suyama curves, 128-bit
  modmul, stage-2 BSGS — all bit-exact vs CPU; ~39× modmul throughput vs the
  20-core CPU.
- ✅ **Packaged as `sieve/ecm/gpu_ecm` + CMake `ENABLE_GPU` build** that
  configures cleanly; standalone batch test passes (1000/1000).
- ✅ **Full suite builds with CUDA** (`-DENABLE_GPU=ON -DCMAKE_CUDA_ARCHITECTURES=86`)
  and the **live hook is wired** into `factor_leftover_norms` behind `CADO_GPU_ECM`
  (default OFF).
- ✅ **Relations-identical validated**: GPU `shadow` mode is byte-for-byte
  identical to CPU-only on a real c120 sieve run while splitting 37023 real
  survivor cofactors on the RTX 3090.
- ✅ **Batched per-region drain implemented & validated** (`CADO_GPU_ECM=batch`):
  one GPU launch per bucket region, hint divided out before `facul`; a clean
  valid superset (26645 vs 17986 relations, −0 lost on the c120 slice).
- ⚠️ **Honest throughput finding**: blanket GPU cofactor offload is **not a net
  win** for c120 on this box (95 s vs 25 s wall; ~8 % Amdahl ceiling, and full
  ECM on every cofactor costs more than it saves). The path to an actual win is
  hard-cofactor targeting + the parameter regime-shift + async overlap.
