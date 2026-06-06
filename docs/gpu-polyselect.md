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

**Validated on the *exact* polyselect polynomial.** CADO's per-prime root step is
`roots_mod_uint64(rp, Ñ mod p, d, p)` — solve **`x^d ≡ Ñ (mod p)`** (`utils/roots_mod.cpp`,
"solve x^k = a mod p"), i.e. the d-th roots of `a = Ñ mod p`, not an arbitrary
polynomial. Re-ran the gcd kernel with `f = x^d − a` (the polyselect case, d=6):
**0 mismatch / 0 self-check-bad** over 3245 primes (p < 30 000) and 5000 primes
near 10⁹ — so the GPU kernel's root set matches `roots_mod_uint64` across the full
prime range. (CADO uses a specialised d-th-root algorithm; the gcd kernel returns
the same *set*, which is what the collision search consumes.)

## Live integration — DONE (and the honest performance verdict)

The full live `--gpu-polyselect` wiring is implemented, validated, and **shipped
behind a default-off flag** — together with an honest measurement that it is a
**net slowdown at the sizes testable on this hardware**, for reasons that are
fundamental rather than implementation bugs. The capability is real and correct;
the speedup is not there yet. Both facts are recorded here.

**What was built (all committed):**

- `polyselect/polyselect-gpu.cu` — the device backend. The validated gcd + Cantor–
  Zassenhaus `roots_gcd` (host/device-identical) ported verbatim from the bench;
  one thread per prime builds `f = x^d − a_i` and solves it mod `p_i`. The host
  entry batches a thread's whole prime range into **one launch** using
  **persistent per-thread device + pinned-host buffers** (allocated once, grown on
  demand, never freed) so the thousands of small per-ad-value calls do not pay
  `cudaMalloc`/`cudaFree` latency every time.
- `polyselect/polyselect-gpu-hooks.{h,cpp}` + `polyselect-gpu-stub.cpp` — the hook
  ABI. The function pointer `cado_gpu_polyselect_roots` is defined in
  `polyselect_common` (the dependency-free leaf), so the CUDA-free
  `polyselect_proots.cpp` calls through it with no circular static-link dependency
  (the `matmul-gpu-hooks` lesson). `cado_gpu_polyselect_init()` is defined in the
  `.cu` (GPU build, installs the pointer) **or** the stub (CPU build, no-op) —
  exactly one is linked, so there is no duplicate symbol.
- `polyselect/polyselect_proots.cpp` — the injection. Gated on `CADO_GPU_POLYSELECT`
  **and** a non-null hook: gather `(p_i, a_i = Ñ mod p_i)` for the thread's
  `[i0,i1)` range, one device call, then per-prime `roots_lift` +
  `polyselect_proots_add` (byte-identical downstream). Returns to the per-prime CPU
  loop on any failure; the CPU loop is otherwise completely unchanged.
- `cado-nfs.py --gpu-polyselect` (`toplevel.py`) — sets `CADO_GPU_POLYSELECT=1` so
  every spawned `polyselect` worker uses the device path; documented as
  experimental.

**Correctness gate — passed.** The GPU path produces a **bit-identical polynomial
set** to the CPU path: direct `polyselect` head-to-head, sorted poly lines, 0 diff
at three scales (198, 136, and 7 kept polynomials; `d=4` and `d=5`, `P` up to
250000). Device-absent → clean `"no CUDA device; using CPU root-finding"` fallback.
**End-to-end `product == N`**: the 59-digit smoke factors correctly with
`--gpu-polyselect` (and with `CADO_GPU_POLYSELECT=1` propagated to subprocesses),
`260938498861057 · 588120598053661 · 760926063870977 · 773951836515617 == N`.

**Performance gate — honest negative.** Measured on the i9-10850K + RTX 3090
(`d=5`, `P=50000`, `admax=60000`, `-t 4`): **CPU 3.2 s vs GPU 4.1 s** — a net
slowdown (persistent buffers already cut it from 4.7 s; per-call `cudaMalloc` churn
was the first culprit). Two fundamental reasons, neither fixable inside the
root-finding offload:

1. **Amdahl.** Root-finding is ~30–40 % of stage-1; the collision search and size
   optimization (the majority) stay on the CPU. Even *free* root-finding caps the
   whole-stage speedup at ~1.5×.
2. **The CPU baseline is already fast.** CADO's `roots_mod_uint64` is a
   *specialised d-th-root* algorithm; it finds every small prime's roots for the
   whole table in well under the 0.01 s the phase reports. There is essentially no
   compute to amortise a PCIe round-trip + kernel-launch + sync against — so per
   ad-value the device path loses on latency, exactly the small-batch GPU
   anti-pattern.

This mirrors the **GPU-cofactorization** finding (3.0.0: correct, validated, but
Amdahl-bound → no net single-machine win). The honest conclusion: **offloading
root-finding alone does not speed up polyselect at these sizes.** The msieve-style
win comes from offloading the **collision search** (the memory-bound hash/match
bulk of stage-1) at much larger `N` — which this validated root-finder and hook ABI
are the foundation for, and which is the documented next step (below). The flag and
`.cu` ship so that work, and larger-`N` / multi-GPU experiments, start from a
correct, integrated base rather than a prototype.

## Integration into the collision search (original design notes)

With the root set validated, the wiring is well-defined. The per-prime CPU loop in
`polyselect_proots_compute_subtask` (`polyselect/polyselect_proots.cpp`) is:

```
for each prime p_i:  rp = roots_mod_uint64(Ñ mod p_i, d, p_i)   # <-- GPU target
                     roots_lift(rp, …, m0, p_i)                  # cheap, stays CPU
                     polyselect_proots_add(R, |rp|, rp, i)       # stores R->roots[i]
```

and the collision search (`polyselect.cpp` / `polyselect_shash`) consumes
`R->roots`. So the offload is a clean batch-replace of the `roots_mod_uint64` call:

1. **Gather** `(p_i, a_i = Ñ mod p_i)` for the whole primes table.
2. **One GPU launch** computing `x^d ≡ a_i mod p_i` for all i (the validated
   kernel), returning each prime's root set + count.
3. **Scatter**: per prime, `roots_lift` (CPU, cheap) + `polyselect_proots_add` —
   so `R` is byte-identical to the CPU path and the **shash collision search is
   unchanged**.
4. **Build**: a `polyselect/polyselect-gpu.cu` built only under `-DENABLE_GPU=ON`,
   reached from the CUDA-free `polyselect_proots.cpp` through a hook ABI (the
   `matmul-gpu-hooks` pattern), so the CPU path is untouched when no GPU backend
   is loaded.
5. **Flag + gate**: `cado-nfs.py --gpu-polyselect` (default off). Because the root
   *set* is identical, the candidate polynomials — and thus the best one — are the
   same; the gate is **matching Murphy-E of the selected polynomial + an
   end-to-end `product == N`** factorization using the GPU-selected polynomial.

**Status update.** This design was fully implemented and validated — see "Live
integration — DONE" above. Each thread issues its own batched launch over its prime
sub-range (no thread-team restructure needed), the CUDA build was added to
`polyselect/` behind `HAVE_GPU_ECM`, and the bit-identical-poly-set +
`product == N` gates pass. The remaining open item is the **performance** win,
which root-finding offload alone does not deliver (the honest negative above).

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
