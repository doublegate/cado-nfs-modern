# GPU Pollard P-1 / Williams P+1 in the pre-factoring front-end (v3.4.0, Track C7)

This extends the GPU pre-NFS factoring front-end ([gpu-prefactor.md](gpu-prefactor.md),
v3.1.0 Track 2.1) — which stripped factors with batched **ECM** before NFS — with two
more classical methods, **Pollard P-1** and **Williams P+1**, sharing the same
bit-exact-validated K-limb Montgomery arithmetic. The point is *coverage*: P-1 and
P+1 find prime factors `p` of `N` whose `p-1` (resp. `p+1`) is smooth — a different
class than ECM's random elliptic curves — and they do it cheaply enough to run as a
pre-pass at every B1 level under an adaptive escalating-B1 schedule.

## Why add them (and why the front-end, not the sieve cofactorization)

The front-end is the **one stage in the whole pipeline with no Amdahl ceiling**: it
runs *before* NFS, so any factor it strips is pure upside (the 3.1.0 measurement was
49×/26×/12× the full CPU at 128/256/512-bit, precisely because it is a separate
stage — unlike in-sieve GPU cofactorization, which is ~8 % of sieve time and nets
<1 %). ECM alone leaves a gap: a factor `p` with `p-1` very smooth but `p` large is
exactly the case Pollard P-1 was invented for, and ECM only reaches it by escalating
B1 and burning curves. Adding P-1/P+1 closes that gap.

## The methods (all on the validated Montgomery core)

`misc/gpu_prefactor/gpu_pm1_pp1.cuh` adds, on top of the `montmul`/`addmod`/`submod`
primitives from `gpu_ecm_mp.cuh`:

- **Pollard P-1.** Stage 1 is a single Montgomery exponentiation `a = base^E` where
  `E = lcm` of the prime powers `≤ B1`; the factor is `gcd(a-1, N)`. Stage 2 is a
  baby-step/giant-step continuation in the multiplicative group — identical in shape
  to the ECM stage-2 BSGS, but with a scalar `montmul` where ECM uses a curve
  addition. Implemented as `montpow` + `pm1_run`.
- **Williams P+1.** Uses Lucas sequences `V_n` (Chebyshev): `V_0=2`, `V_1=seed`,
  `V_{2n}=V_n²-2`, `V_{m+n}=V_m V_n - V_{m-n}`. Stage 1 evaluates `V_E(seed)` by a
  **Lucas ladder** (the additive analogue of the Montgomery ladder), via the
  composition identity `V_{ab}(x)=V_a(V_b(x))`; the factor is `gcd(V_E - 2, N)`.
  Stage 2 is the Lucas BSGS, advancing giant steps by `V_{(k+1)W}=V_W V_{kW}-V_{(k-1)W}`.
  Implemented as `lucas_chain` + `pp1_run`.

Both are `__host__ __device__`, so the exact device code runs on the CPU as the
self-check reference.

## Honest scope — coverage, not throughput

On a **single N**, each P-1/P+1 run is *one sequence* = one GPU lane. It does **not**
benefit from the GPU's thousands-of-curves parallelism the way ECM does — there is no
throughput story here. Its value is twofold and measured below:

1. **Coverage** — it strips p-/+1-smooth factors the ECM curve count can miss.
2. **Cheaper time-to-strip** — run as a one-lane pre-pass before the ECM batch, and
   skip the ECM batch entirely at a B1 level once the cofactor is prime/1.

The front-end's *throughput* win remains ECM's. (Where P-1/P+1 *would* parallelise is
batch cofactorization during sieving — thousands of cofactors — but that subsystem is
Amdahl-capped and was deliberately not chosen; see
[gpu-batch-smooth-c3.md](gpu-batch-smooth-c3.md).) The few one-lane kernels run only
on GPU 0; the multi-GPU split stays an ECM-only concern.

## Adaptive escalating-B1 schedule

`run_stage_K<K>` now runs, at each B1 level: **P-1 → P+1 → (maybe) ECM**. If the cheap
P-1/P+1 pre-passes already reduce the cofactor to a prime or 1, the expensive ECM
batch at that level is **skipped** (`"P-1/P+1 reduced the cofactor to prime/1;
skipping ECM at this B1"`). P-1 uses 4 distinct bases `{2,3,5,7}`; P+1 uses 6 distinct
seeds `{3,5,7,11,13,17}` (the seed sets the Lucas group order class, so a handful
raises the hit rate; seed 2 is degenerate and skipped). Set
`CADO_PREFACTOR_NOPM1PP1=1` to disable the pre-passes (ECM only).

## Correctness — bit-exact, and `product == N`

`bench/gpu-prefactor-pm1pp1.cu` (run via `bench/gpu-prefactor-pm1pp1-validate.sh`)
checks, for K ∈ {2,4} (128/256-bit moduli):

1. **GPU vs CPU**, bit-exact: the kernels vs `pm1_run`/`pp1_run` on the host (same
   `__host__ __device__` code) — stage-1 **and** stage-2 limbs identical.
2. **GPU vs an independent GMP reference**: P-1 stage-1 `base^E` via `mpz_powm_ui`;
   P+1 stage-1 `V_E(seed)` via a GMP Lucas chain — residues identical.
3. **Functional**: crafted composites `n=p·q` with `p-1` (resp. `p+1`) B1-powersmooth
   are cracked by P-1 (resp. P+1) stage 1, and a `p` with `p-1 = smooth · prime∈(B1,B2]`
   is cracked by P-1 stage 2.

Measured on the reference RTX 3090 (sm_86), CUDA 13.3:

```
=== pm1/pp1 validation: 14 pass, 0 fail ===
```

Per-run the integrated binary also self-checks every pass (`pm1/pp1 selfcheck: PASS
(0/4|0/6 GPU lanes differ from CPU)`), and every recovered factor is re-verified to
divide N before it is reported, so the end-to-end result stays `product == N`.

## Measured (RTX 3090, i9-10850K reference box)

**The coverage + time-to-strip win** — `N = p·q`, `p` a 21-digit factor with `p-1`
2000-powersmooth, `q` a 25-digit prime (`staged` schedule):

| Run | What happens | Wall |
|-----|--------------|------|
| **With P-1/P+1** (default) | P-1 strips `p` at B1=2000; ECM **skipped** | **0.52 s** |
| ECM only (`CADO_PREFACTOR_NOPM1PP1=1`) | B1=2000 finds nothing → escalate to B1=11000, ECM finds it | 1.73 s |

→ **~3.3× faster** time-to-strip on this p-1-smooth case, with `product == N` both ways.

**The honest cost when they find nothing** — `N` with a general 12-digit factor (not
p±1-smooth), steady-state of 3 runs:

| Run | Wall |
|-----|------|
| ECM only | ~0.34 s |
| With P-1/P+1 (find nothing, then ECM strips it) | ~0.37 s |

→ the P-1/P+1 pre-passes cost **~30 ms** here (a few one-lane kernels + the
bit-exact self-check) — negligible against the ECM batch, and bounded because they
are one lane each.

## Status

- **Done & validated:** GPU P-1 (stage 1 + stage-2 BSGS) and P+1 (Lucas stage 1 +
  stage-2 BSGS), bit-exact vs CPU and vs GMP at K ∈ {2,4}; integrated into the staged
  schedule with the adaptive ECM-skip; per-run self-check; `CADO_PREFACTOR_NOPM1PP1`
  toggle; `cado-nfs.py --gpu-prefactor` carries them transparently (the
  `factors stripped:` output is unchanged).
- **Honest non-win:** on a single N these are one-lane methods — coverage, not GPU
  throughput. The measured time win is real but case-dependent (it appears when a
  factor is p±1-smooth); on general inputs the cost is the ~30 ms above.
