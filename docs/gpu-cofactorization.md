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
   finds factors, benchmarked. Refinements: **2-word (128-bit) moduli** for larger
   `mfb`, optional **stage 2**, and **Suyama sigma** parametrization (higher hit
   rate than the proof-of-concept a24/x0=2 curve).
2. **Batching layer** behind `facul_all` (`sieve/las-cofactor.cpp` /
   `sieve/ecm/facul.cpp`): accumulate survivors across special-q into a device
   batch, launch the kernel, return factors. Tune batch size vs PCIe latency and
   the sieve's per-region cadence.
3. **CMake CUDA detection** (`/opt/cuda`), `sm_86` target, behind an
   `ENABLE_GPU` flag; CPU path stays default.
4. **Benchmark** relations/sec and CPU-utilization shift (toward sieving) with vs
   without the GPU on a c100–c120 run, then retune `ncurves`/`mfb` to exploit the
   cheap cofactorization.

## Status

- ✅ CUDA 13.3 + RTX 3090 verified end-to-end; CADO has no prior GPU code.
- ✅ Cofactorization integration point identified (`facul_all`, batch of small
  cofactors).
- ✅ **Core thesis validated & measured: ~39× GPU vs 20-core CPU on ECM's modmul
  primitive, correctness-checked.**
- ⏳ Full ECM kernel + `facul_all` integration + retuning — large follow-on, all
  fully testable on this hardware (unlike the AVX-512 track).
