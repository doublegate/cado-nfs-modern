# Changelog

All notable changes to this fork are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
loosely follows [Semantic Versioning](https://semver.org/).

This is a downstream **modernization + performance fork** of upstream
[CADO-NFS](https://gitlab.inria.fr/cado-nfs/cado-nfs). The `3.0.x-modern` line is
rebased onto upstream **3.0.0**; only the changes introduced by this fork are
listed. For the upstream history see [`NEWS`](NEWS). The earlier `2.3.1-modern`
release (rebased on upstream 2.3.0) lives on the `main` branch.

## [3.0.0-modern] — 2026-06-05

Rebases the modernization effort onto upstream CADO-NFS **3.0.0** and adds four
tracks of rigorously-measured performance / robustness work. **No NFS
mathematics is changed** — every track is an implementation, build, or
orchestration optimization, and correctness (verified `product == N`
factorizations + `make check`) is the gate after each phase. Results are
reported honestly, including the parts that did *not* pay off.

Reference box for all measurements: **i9-10850K (10C/20T, Comet Lake → no
AVX-512) + RTX 3090 (Ampere, `sm_86`), 64 GiB, CachyOS.**

### Phase 0 — rebase onto upstream 3.0.0

- Builds cleanly on a current box (**CMake 4.x, GCC 16, C++20, Python 3.14**).
  3.0.0 already subsumes the 2.3.x portability fixes and brings, for free, the
  Bouvier–Imbert batch cofactorization (eprint 2018/669) and `I>16` sieving.
- The Flask/`requests` orchestration deps live in a venv
  (`scripts/setup-venv.sh`, pointed at by `local.sh`) rather than the system.
- Baseline established with a verified smoke factorization.

### Phase 1 — build & compiler (measured, not guessed)

- `bench/las-microbench.sh`: a deterministic siever microbenchmark (pinned c120
  polynomial + `las --random-sample N --seed S`, <1% CPU variance) that makes
  few-percent changes measurable despite the ~15–20% end-to-end polyselect noise.
- Result on the c120 siever workload (mean of 3, <1% variance):

  | flags | time | vs stock |
  |---|---|---|
  | stock `-O2` | 12.57 s | — |
  | `-O2 -march=native` | 11.79 s | −6.2% |
  | **`-O3 -march=native`** | **11.66 s** | **−7.2% (adopted)** |
  | `+LTO` | — | 0% (and breaks `-Werror`) → rejected |
  | `+PGO` | — | +2.8% (regresses) → rejected |

  `-march=native` (AVX2/BMI2/FMA host codegen) is the only real win; LTO and PGO
  are rejected on evidence. Final flags adopted in `local.sh`.

### Phase 2 — SIMD modernization

- **AVX2 on the siever: ruled out by profiling.** The hot path is scatter
  (`fill_in_buckets`, small-sieve) plus scalar modular arithmetic
  (`invmod_redc_32`, `plattice_info`), not vectorizable loops; the explicitly
  SSE2 survivor code is cold. Widening it would not move the needle.
- **AVX-512 VPCLMULQDQ gf2x base-case multiply** (the GF(2)[x] hot path used by
  `lingen`): `bench/vpclmul-mul1n.c` implements `mul_1_n`/`addmul_1_n` at four
  carryless multiplies per instruction (vs one per 128-bit `pclmul`), validated
  **bit-exact over 200 000 random trials under Intel SDE**
  (`bench/vpclmul-validate.sh`). `gf2x/already_tuned/x86_64_vpclmul/` carries the
  gf2x-integrated backend plus an `INTEGRATION.md` completion guide. Performance
  is gated on real AVX-512 silicon — the reference box is Comet Lake, so this is
  validated correct-only here.

### Phase 3 — GPU ECM cofactorization (RTX 3090) — validated, honest negative

- `bench/gpu-modmul-bench.cu`: batched 64-bit Montgomery modmul (ECM's inner
  primitive) — **~279 G modmul/s on the 3090 vs ~7.2 G across 20 CPU cores
  (~39×)**, correctness-checked.
- `bench/gpu-ecm.cu`: a complete batched ECM stage-1 (Montgomery-curve XZ ladder,
  on-device binary gcd). The same `__host__ __device__` code runs on CPU and GPU,
  so the GPU path is **bit-exact**; it cracks 256/256 test composites at
  ~2.75 M curves/s (B1=2000). Extended with Suyama-σ curves, a 128-bit Montgomery
  modulus (CIOS REDC), and a stage-2 BSGS continuation (+64% yield/curve).
- Integrated into CADO behind `facul`/`las-cofactor` with a CMake CUDA build and
  a live batched-drain hook (relations validated identical to the CPU path).
- **Honest conclusion (`docs/gpu-cofactorization.md`):** cofactorization is only
  ~8% of siever time (Amdahl ceiling), and a mid-effort "throughput win" turned
  out to be a correctness bug (a single-factor batch emitting relations with
  primes > `lpb`; fixed in `c675f82` with a complete-factorization check). With
  correctness enforced there is **no net single-machine speedup** from the GPU at
  these sizes — the value would be a cheaper-cofactorization *parameter-regime
  shift* on much larger inputs. Documented as a measured negative result rather
  than an unsubstantiated win.

### Phase 4 — Rust orchestration (robustness / scalability)

Ports the **network/DB substrate** of the distributed layer to async Rust while
keeping the **exact existing HTTP/JSON work-unit protocol**, so Rust binaries
interoperate with an unmodified `cado-nfs.py` during migration. This buys
robustness (no GIL, connection pooling, a single static client binary), not
single-machine factoring speed. Full details in `docs/rust-orchestration.md`.

- **`cado-nfs-client-rs`** (`rust/cado-nfs-client`, reqwest + rustls, no OpenSSL):
  the complete client loop — fetch work-unit, download + checksum-verify
  (sha1/256/3-256) inputs, run commands (argv split, no shell, exactly as the
  Python client), upload results. Plus failover across multiple `--server` URLs,
  TLS cert-pinning (`--certsha1`), `--niceness`, download `flock`, and `STDIN<n>`
  redirection (which the stock Python client never implemented).
- **`cado-wu-server-rs`** (`rust/cado-wu-server`, axum + tokio + bundled
  rusqlite): the five endpoints over the same `wudb` SQLite schema and status
  codes, replicating the `WuAccess` assign/result lifecycle, with stale-work
  timeout reassignment, `410`-after-finish, an r2d2/WAL connection pool, TLS
  (`--cert/--key`), and IP-whitelist enforcement (`--whitelist`, IPv4/IPv6
  exact + CIDR, matching `api_server.py`'s `api_limit_remote_addr`).
- **In-process swap:** `scripts/cadofactor/external_api_server.py` (`ExternalApiServer`)
  implements the `ApiServer` interface the driver uses by launching the Rust
  server over the same database; `cadotask.py` switches to it when
  `CADO_RUST_WU_SERVER` is set. TLS is wired (the shim regenerates the same
  self-signed cert and reports its SHA1 so clients pin it) and the whitelist is
  passed through.
- **`cadotask.py` is deliberately left in Python.** Phase 4's scope is the
  substrate, not the scheduler DAG; porting ~7.5K LOC of orchestration logic has
  no robustness/throughput payoff and a large correctness surface, so it stays
  behind the stable `ApiServer` seam.
- **Validation:** `rust/interop-test.sh` (Rust client ↔ stock Python server),
  `rust/server-interop-test.sh` (full lifecycle, Rust server + client),
  `rust/robustness-test.sh` (**13/13**), `rust/deploy-test.sh` (a real 59-digit
  factorization run entirely by external Rust clients), `rust/server-swap-test.sh`
  (Rust server swapped into a live `cado-nfs.py` run, plain HTTP) and
  `rust/server-swap-tls-test.sh` (the same over TLS with cert-pinning + whitelist).

### Net finding

CPU-side tuning is nearly tapped out on this hand-optimized codebase (Phase 1
~7%, Phase 2 AVX2 nil). The structural headroom is in *new compute resources* —
the GPU (validated ~39× on the modmul primitive, though Amdahl-bounded at these
sizes) and, where available, AVX-512 VPCLMULQDQ (validated correct) — and in the
Rust orchestration substrate for multi-client robustness.

### Versioning

- `CADO_VERSION_STRING` is `3.0.0-modern` (`CMakeLists.txt`); the numeric
  components stay at upstream 3.0.0, only the display string is suffixed. No test
  depends on the version string.
