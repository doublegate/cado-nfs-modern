# Parallel structured Gaussian elimination — merge (Roadmap A3)

This documents the v3.2.0-modern investigation of roadmap item **A3**, *"parallel
structured Gaussian elimination for filtering/merge (the algorithm used for
RSA-240/250)."* Like A2's CPU half, the finding is an honest **"already done
upstream"** — with an added empirical verification that it actually parallelizes.

## What merge is (for newcomers)

After sieving and duplicate removal, NFS has a big set of **relations**, each a
product of primes (on two sides). **Filtering** turns these into the sparse matrix
that linear algebra (Block Wiedemann) consumes:

- **purge** removes *singletons* (a prime occurring in only one relation can't be
  part of a dependency) — this cascades and shrinks the set;
- **merge** is **structured Gaussian elimination**: it repeatedly eliminates
  primes of small weight (occurring in few relations) by adding relations
  together, shrinking the matrix dimension while controlling *fill-in* (density
  growth) via a Markowitz-style cost heuristic.

merge is the phase `BENCHMARKS.md` flagged as high-variance (its cost depends on
the run-dependent matrix that filtering produces).

## Finding — A3 is already implemented upstream (the exact paper)

CADO's `filter/merge.cpp` **is** parallel structured Gaussian elimination. Its
header cites precisely the algorithm the roadmap names:

> [1] *Parallel Structured Gaussian Elimination for the Number Field Sieve*,
> Charles Bouillaguet and Paul Zimmermann, **Mathematical Cryptology, volume 0,
> number 1, pages 22–39, 2020.**
> [2] *Design and Implementation of a Parallel Markowitz Threshold Algorithm*,
> Davis, Duff, Nakov, SIAM J. Matrix Anal. Appl. 41(2), 2020.

The implementation carries **~50 OpenMP pragmas** (`#pragma omp parallel for`,
`parallel`, `for schedule(...)`, `atomic`, `barrier`, `single`) across the
weight computation, the Markowitz cost passes, the apply/eliminate passes, and
the output — comments even tune the schedule per RSA-512 / RSA-240 thread counts
(`merge.cpp:1082`). It is the same code that performed the RSA-240/250 record
factorizations. CADO 3.0.0 already subsumes this; **there is nothing to add.**

### It is already run with all logical threads

The orchestration wires merge's thread count automatically. `merge` takes `-t`
(`omp_set_num_threads`); the toplevel `set_threads_and_client_threads()` lets
`tasks.filter.merge.threads` inherit `tasks.threads` (= all **logical** cores),
*distinct from* `tasks.linalg.bwc.threads`, which is deliberately held to
**physical** cores — its own docstring notes *"for merge, using all hyperthreads
is beneficial"* and the doctest asserts `merge.threads = 32` vs `bwc.threads = 16`
on a 16-physical / 32-logical host. So on the reference box merge already runs
with 20 threads, no configuration needed.

## Empirical verification — merge thread-scaling (measured)

To turn "it has pragmas" into a measured fact, `filter/merge` was run on real
purged matrices at varying `-t` (wall-clock, `target_density=170`, RTX 3090 box's
i9-10850K, 10C/20T):

| matrix (purged) | rows × cols | -t 1 | -t 2 | -t 4 | -t 8 | -t 16 | -t 20 | best |
|-----------------|------------:|-----:|-----:|-----:|-----:|------:|------:|:----:|
| c60 | 20 K × 67 K | 0.34 s | 0.25 s | 0.17 s | 0.10 s | — | 0.17 s | **3.4× @ t8** |
| c90 | 303 K × 1.13 M | 7.27 s | 4.30 s | 2.76 s | 2.20 s | 2.27 s | 2.65 s | **3.3× @ t8** |

**Honest reading.** merge **does** parallelize — both matrices reach ~3.3–3.4×
at 8 threads (c90: 7.27 s → 2.20 s). But the matrices produced by *desktop-scale*
factorizations are small (c90's is 0.3 M rows), so beyond ~8 threads the
parallel/scheduling overhead outweighs the remaining work and wall-time plateaus,
then slightly regresses at 16–20 threads. This is expected: the
Bouillaguet–Zimmermann parallelism is designed for **RSA-scale** matrices (tens
of millions of rows for RSA-240/250), where it scales to 100+ threads; at c60–c90
the absolute time is already a few seconds and overhead-bound past 8 threads. The
fork's box can't reach the regime where the parallelism's full benefit shows, but
the implementation and the all-threads wiring are present and correct — and even
here it cuts merge wall-time ~3.3×.

## Conclusion

A3 is **complete upstream** and already enabled by default. Recorded as an honest
"already done" finding (like A2's CPU `facul` path and the 3.1.0 PGO/micro-opt
negatives). No code change; the contribution here is the verification + this
record. The freed effort stays with the genuinely open items (the GPU and
algorithm tracks).

## Sources

- Bouillaguet, Zimmermann. *Parallel Structured Gaussian Elimination for the
  Number Field Sieve.* Mathematical Cryptology 0(1):22–39, 2020.
- Davis, Duff, Nakov. *Design and Implementation of a Parallel Markowitz
  Threshold Algorithm.* SIAM J. Matrix Anal. Appl. 41(2), 2020.
- CADO-NFS `filter/merge.cpp` (Bouillaguet–Zimmermann, the RSA-240/250 merge).
