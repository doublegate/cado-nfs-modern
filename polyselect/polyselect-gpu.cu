/* GPU polynomial-selection root-finding backend (v3.2.0-modern, Track C2).
 *
 * Installs cado_gpu_polyselect_roots: for a batch of primes p_i and targets a_i,
 * solve x^d == a_i (mod p_i) on the device -- the per-prime root step that
 * dominates polyselect stage-1 (~30-40% with the modular inverse). One thread per
 * prime. The root-finder is gcd(x^p - x, f) + Cantor-Zassenhaus over F_p with
 * f = x^d - a_i, ported verbatim (host/device identical code) from the validated
 * bench kernel bench/gpu-polyselect-roots-gcd.cu -- which was checked bit-exact
 * (full root multiset) against direct evaluation over 3245 primes (p<30000) and
 * 5000 primes near 1e9, on the exact polyselect polynomial f = x^d - a.
 *
 * O(d^2 log p) per prime, independent of p's magnitude -- so it covers the full
 * polyselect prime range, unlike the O(p) direct-eval kernel. The returned root
 * SET matches utils/roots_mod.cpp's roots_mod_uint64 (CADO uses a specialised
 * d-th-root algorithm; the collision search consumes the set, not an order), so
 * roots_lift + polyselect_proots_add downstream are byte-identical to the CPU path.
 *
 * Built only under -DENABLE_GPU=ON (HAVE_GPU_ECM); otherwise polyselect-gpu-stub.cpp
 * provides a no-op cado_gpu_polyselect_init() and the per-prime CPU path runs. The
 * whole device path is additionally gated at the call site on CADO_GPU_POLYSELECT. */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "polyselect-gpu-hooks.h"

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;
#define D 8                       /* max poly degree we handle (polyselect d <= 7) */

namespace {

struct Poly { u64 c[2 * D + 2]; int deg; };   /* deg = highest nonzero index, -1 if zero */

__host__ __device__ static u64 mmul(u64 a, u64 b, u64 p) { return (a * b) % p; }
__host__ __device__ static u64 madd(u64 a, u64 b, u64 p) { a += b; return a >= p ? a - p : a; }
__host__ __device__ static u64 msub(u64 a, u64 b, u64 p) { return a >= b ? a - b : a + p - b; }
__host__ __device__ static u64 minv(u64 a, u64 m)
{
    long long t0 = 0, t1 = 1; u64 r0 = m, r1 = a % m;
    while (r1) { u64 q = r0 / r1, r2 = r0 - q * r1; long long t2 = t0 - (long long) q * t1; r0 = r1; r1 = r2; t0 = t1; t1 = t2; }
    if (r0 != 1) return 0; long long i = t0 % (long long) m; if (i < 0) i += m; return (u64) i;
}
__host__ __device__ static void pnorm(Poly & a) { a.deg = -1; for (int i = 2 * D + 1; i >= 0; i--) if (a.c[i]) { a.deg = i; break; } }
__host__ __device__ static void pmonic(Poly & a, u64 p)
{
    if (a.deg < 0) return; u64 iv = minv(a.c[a.deg], p);
    for (int i = 0; i <= a.deg; i++) a.c[i] = mmul(a.c[i], iv, p);
}
/* a mod b (b monic), schoolbook; a modified in place, result deg < b.deg */
__host__ __device__ static void prem(Poly & a, const Poly & b, u64 p)
{
    if (b.deg <= 0) { a.deg = -1; for (int i = 0; i < 2 * D + 2; i++) a.c[i] = 0; return; }
    while (a.deg >= b.deg) { u64 lc = a.c[a.deg]; int sh = a.deg - b.deg;
        for (int i = 0; i <= b.deg; i++) a.c[i + sh] = msub(a.c[i + sh], mmul(lc, b.c[i], p), p);
        a.c[a.deg] = 0; pnorm(a); if (a.deg < 0) break; }
}
/* gcd(a,b) monic */
__host__ __device__ static Poly pgcd(Poly a, Poly b, u64 p)
{
    pnorm(a); pnorm(b);
    while (b.deg >= 0) { Poly bb = b; pmonic(bb, p); prem(a, bb, p); Poly t = a; a = b; b = t; }
    pmonic(a, p); return a;
}
/* (base)^e mod m, polys; m monic-ized internally */
__host__ __device__ static Poly ppow(Poly base, u64 e, Poly m, u64 p)
{
    pmonic(m, p);
    Poly r; for (int i = 0; i < 2 * D + 2; i++) r.c[i] = 0; r.c[0] = 1; r.deg = 0;
    prem(base, m, p);
    while (e) {
        if (e & 1) { Poly t; for (int i = 0; i < 2 * D + 2; i++) t.c[i] = 0;     /* t = r*base */
            for (int i = 0; i <= (r.deg < 0 ? 0 : r.deg); i++) if (r.c[i]) for (int j = 0; j <= (base.deg < 0 ? 0 : base.deg); j++) if (base.c[j]) t.c[i + j] = madd(t.c[i + j], mmul(r.c[i], base.c[j], p), p);
            pnorm(t); prem(t, m, p); r = t; }
        Poly s; for (int i = 0; i < 2 * D + 2; i++) s.c[i] = 0;                  /* base = base^2 */
        for (int i = 0; i <= (base.deg < 0 ? 0 : base.deg); i++) if (base.c[i]) for (int j = 0; j <= (base.deg < 0 ? 0 : base.deg); j++) if (base.c[j]) s.c[i + j] = madd(s.c[i + j], mmul(base.c[i], base.c[j], p), p);
        pnorm(s); prem(s, m, p); base = s; e >>= 1;
    }
    return r;
}
/* exact division a/b when b|a (b monic), returns quotient */
__host__ __device__ static Poly pdiv(Poly a, Poly b, u64 p)
{
    Poly q; for (int i = 0; i < 2 * D + 2; i++) q.c[i] = 0; q.deg = -1;
    pmonic(b, p);
    while (a.deg >= b.deg && a.deg >= 0) { u64 lc = a.c[a.deg]; int sh = a.deg - b.deg; q.c[sh] = lc;
        for (int i = 0; i <= b.deg; i++) a.c[i + sh] = msub(a.c[i + sh], mmul(lc, b.c[i], p), p);
        a.c[a.deg] = 0; pnorm(a); }
    pnorm(q); return q;
}
/* roots of f (degree fdeg) mod p; write up to D roots into out, return count */
__host__ __device__ static int roots_gcd(const u64 * fc, int fdeg, u64 p, u32 * out)
{
    if (p < 3) { int k = 0; for (u64 a = 0; a < p; a++) { u64 acc = fc[fdeg] % p; for (int i = fdeg - 1; i >= 0; i--) acc = (acc * a + fc[i]) % p; if (acc % p == 0) out[k++] = (u32) a; } return k; }
    Poly f; for (int i = 0; i < 2 * D + 2; i++) f.c[i] = 0; for (int i = 0; i <= fdeg; i++) f.c[i] = fc[i] % p; pnorm(f);
    if (f.deg <= 0) return 0;             /* constant: 0 roots */
    pmonic(f, p);
    Poly x; for (int i = 0; i < 2 * D + 2; i++) x.c[i] = 0; x.c[1] = 1; x.deg = 1;
    Poly h = ppow(x, p, f, p);            /* x^p mod f */
    h.c[1] = msub(h.c[1], 1, p); pnorm(h);   /* h - x */
    Poly g = pgcd(f, h, p);               /* roots = roots of g (squarefree, splits to linears) */
    if (g.deg <= 0) return 0;
    /* Cantor-Zassenhaus: split g into linear factors */
    Poly stack[D + 2]; int sp = 0; stack[sp++] = g; int k = 0;
    while (sp > 0 && k < D) {
        Poly h2 = stack[--sp]; pnorm(h2); if (h2.deg <= 0) continue;
        if (h2.deg == 1) { pmonic(h2, p); out[k++] = (u32) msub(0, h2.c[0], p); continue; }  /* x + c -> root -c */
        Poly fac; bool split = false;
        for (u64 dd = 0; dd < 64 && !split; dd++) {
            Poly base; for (int i = 0; i < 2 * D + 2; i++) base.c[i] = 0; base.c[0] = dd % p; base.c[1] = 1; base.deg = 1; /* x+dd */
            Poly b = ppow(base, (p - 1) / 2, h2, p);
            b.c[0] = msub(b.c[0], 1, p); pnorm(b);                      /* b - 1 */
            fac = pgcd(h2, b, p);
            if (fac.deg > 0 && fac.deg < h2.deg) split = true;
        }
        if (!split) {   /* fallback: peel the r==0 factor, else give up on this branch (rare) */
            Poly base; for (int i = 0; i < 2 * D + 2; i++) base.c[i] = 0; base.c[1] = 1; base.deg = 1; base.c[0] = 0;
            Poly b = ppow(base, (p - 1) / 2, h2, p); fac = pgcd(h2, b, p);
            if (!(fac.deg > 0 && fac.deg < h2.deg)) continue;
        }
        Poly other = pdiv(h2, fac, p);
        stack[sp++] = fac; stack[sp++] = other;
    }
    return k;
}

/* For prime p[i] and target a[i], build f = x^d - a[i] and solve f == 0 mod p[i]. */
__global__ void k_polyroots(const u64 * a, const u32 * p, unsigned int n, int d,
                            u32 * out, u32 * cnt)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    u32 pp = p[i];
    if (pp == 0) { cnt[i] = 0; return; }            /* skipped prime */
    u64 fc[D + 1];
    for (int j = 0; j <= d; j++) fc[j] = 0;
    fc[d] = 1;                                       /* x^d */
    u64 am = a[i] % pp;
    fc[0] = (pp - am) % pp;                          /* -a mod p */
    u32 r[D];
    int k = roots_gcd(fc, d, pp, r);
    cnt[i] = k;
    for (int j = 0; j < k; j++) out[(size_t) i * D + j] = r[j];
}

/* Per-thread persistent device + pinned-host buffers. polyselect calls the hook
 * once per ad-value per thread (thousands of small batches), so allocating and
 * freeing device memory every call swamps the actual root-finding compute with
 * cudaMalloc/cudaFree latency. Each polyselect worker thread keeps its own buffers
 * (CUDA contexts are shared across threads of a process; thread_local avoids any
 * cross-thread race), grown on demand and never freed -- the OS reclaims them at
 * process exit. Pinned host staging makes the D2H copies fast and async-capable. */
struct GpuBufs {
    unsigned int cap = 0;
    u64 * da = nullptr; u32 * dp = nullptr; u32 * dout = nullptr; u32 * dcnt = nullptr;
    u32 * hcnt = nullptr; u32 * hout = nullptr;
    bool ensure(unsigned int n) {
        if (n <= cap) return true;
        if (da) { cudaFree(da); cudaFree(dp); cudaFree(dout); cudaFree(dcnt);
                  cudaFreeHost(hcnt); cudaFreeHost(hout); }
        cap = 0; da = nullptr;
        if (cudaMalloc(&da, (size_t) n * sizeof(u64))) return false;
        if (cudaMalloc(&dp, (size_t) n * sizeof(u32))) return false;
        if (cudaMalloc(&dout, (size_t) n * D * sizeof(u32))) return false;
        if (cudaMalloc(&dcnt, (size_t) n * sizeof(u32))) return false;
        if (cudaMallocHost(&hcnt, (size_t) n * sizeof(u32))) return false;
        if (cudaMallocHost(&hout, (size_t) n * D * sizeof(u32))) return false;
        cap = n;
        return true;
    }
};
thread_local GpuBufs g_bufs;

/* Host entry installed into the hook pointer. Computes the whole batch in one
 * launch using the calling thread's persistent buffers. Returns 1 if handled on
 * the device, 0 to fall back to the CPU path (alloc/launch failure or a degree we
 * do not handle). */
int gpu_polyselect_roots_impl(const uint64_t * a, const uint32_t * p,
                              unsigned int n, int d,
                              uint64_t * roots, unsigned int * nr)
{
    if (n == 0) return 1;
    if (d < 1 || d > D) return 0;                    /* unsupported degree -> CPU */
    if (!g_bufs.ensure(n)) {
        fprintf(stderr, "polyselect GPU: device alloc failed; using CPU root-finding\n");
        return 0;
    }

    cudaError_t e;
    e = cudaMemcpy(g_bufs.da, a, (size_t) n * sizeof(u64), cudaMemcpyHostToDevice); if (e) goto fail;
    e = cudaMemcpy(g_bufs.dp, p, (size_t) n * sizeof(u32), cudaMemcpyHostToDevice); if (e) goto fail;

    {
        int tpb = 64;
        unsigned int blk = (n + tpb - 1) / tpb;
        k_polyroots<<<blk, tpb>>>(g_bufs.da, g_bufs.dp, n, d, g_bufs.dout, g_bufs.dcnt);
        e = cudaGetLastError(); if (e) goto fail;
    }

    e = cudaMemcpy(g_bufs.hcnt, g_bufs.dcnt, (size_t) n * sizeof(u32), cudaMemcpyDeviceToHost); if (e) goto fail;
    e = cudaMemcpy(g_bufs.hout, g_bufs.dout, (size_t) n * D * sizeof(u32), cudaMemcpyDeviceToHost); if (e) goto fail;

    for (unsigned int i = 0; i < n; i++) {
        unsigned int k = g_bufs.hcnt[i];
        if (k > (unsigned) d) k = d;                 /* never exceed the caller's d-slot buffer */
        nr[i] = k;
        for (unsigned int j = 0; j < k; j++)
            roots[(size_t) i * d + j] = g_bufs.hout[(size_t) i * D + j];
    }
    return 1;

fail:
    fprintf(stderr, "polyselect GPU: %s; falling back to CPU root-finding\n",
            cudaGetErrorString(e));
    return 0;
}

/* ===================== GPU collision search (Track C2 cont.) ===================== */

typedef int64_t i64;

/* Per-(prime,root) entry emission count, matching the CADO dispatch loop exactly:
 *   for (u = u0;       u <  umax;     u += ppl) emit;     // k >= 0
 *   for (u = u0 - ppl; u + umax >= 0; u -= ppl) emit;     // k >= 1   */
__host__ __device__ static inline u64 emit_count(i64 u0, i64 ppl, i64 umax)
{
    u64 npos = (u0 < umax) ? (u64) ((umax - 1 - u0) / ppl) + 1 : 0;
    u64 nneg = (u0 + umax >= 0) ? (u64) ((u0 + umax) / ppl) : 0;
    return npos + nneg;
}

/* expand prime table -> per-entry (ppl, prime tag); roots_flat is already in entry
 * order (prime 0's roots, then prime 1's, ...). off[i] = sum_{<i} nr[i]. */
__global__ void k_expand(const u32 * primes, const u8 * nr, const u32 * off,
                         u32 lenPrimes, i64 * ent_ppl, u32 * ent_p)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= lenPrimes) return;
    i64 pp = (i64) primes[i] * (i64) primes[i];
    u32 base = off[i], cnt = nr[i];
    for (u32 j = 0; j < cnt; j++) { ent_ppl[base + j] = pp; ent_p[base + j] = primes[i]; }
}
__global__ void k_ccount(const i64 * ent_root, const i64 * ent_ppl, u32 nent,
                         i64 umax, u64 * cnt)
{
    u32 e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nent) return;
    cnt[e] = emit_count(ent_root[e], ent_ppl[e], umax);
}
__global__ void k_cwrite(const i64 * ent_root, const i64 * ent_ppl, const u32 * ent_p,
                         u32 nent, i64 umax, const u64 * off, u64 * U, u32 * P)
{
    u32 e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nent) return;
    i64 u0 = ent_root[e], pp = ent_ppl[e]; u32 tag = ent_p[e]; u64 o = off[e];
    for (i64 u = u0; u < umax; u += pp) { U[o] = (u64) u; P[o] = tag; o++; }
    for (i64 u = u0 - pp; u + umax >= 0; u -= pp) { U[o] = (u64) u; P[o] = tag; o++; }
}
/* U sorted; one thread per run-start emits ALL pairs in the run (matches the CPU
 * shash, which collides each element with every earlier equal one). Runs of length
 * >2 are astronomically rare, but handled for bit-exactness. */
__global__ void k_cdetect(const u64 * U, const u32 * P, u64 m, u32 cap,
                          u64 * oU, u32 * oP1, u32 * oP2, u32 * ncoll)
{
    u64 t = (u64) blockIdx.x * blockDim.x + threadIdx.x; if (t >= m) return;
    if (t > 0 && U[t - 1] == U[t]) return;          /* not a run start */
    u64 e = t + 1; while (e < m && U[e] == U[t]) e++;
    for (u64 a = t; a < e; a++)
        for (u64 b = a + 1; b < e; b++) {
            if (P[a] == P[b]) continue;             /* same prime cannot truly collide */
            u32 idx = atomicAdd(ncoll, 1u);
            if (idx < cap) {
                oU[idx] = U[a];
                oP1[idx] = P[a] < P[b] ? P[a] : P[b];
                oP2[idx] = P[a] < P[b] ? P[b] : P[a];
            }
        }
}

struct CollBufs {
    u32 ecap = 0;                         /* entry-array capacity */
    u64 ucap = 0;                         /* u-array capacity */
    u32 ccap = 0;                         /* collision-output capacity */
    i64 *d_root = nullptr, *d_ppl = nullptr; u32 *d_p = nullptr;
    u8 *d_nr = nullptr; u32 *d_primes = nullptr, *d_off32 = nullptr; u64 *d_cnt = nullptr, *d_off = nullptr;
    u64 *d_U = nullptr; u32 *d_P = nullptr;
    u64 *d_oU = nullptr; u32 *d_oP1 = nullptr, *d_oP2 = nullptr, *d_ncoll = nullptr;
    u64 *h_oU = nullptr; u32 *h_oP1 = nullptr, *h_oP2 = nullptr;

    bool ensure_entries(u32 lenPrimes, u32 nent) {
        if (nent <= ecap && lenPrimes <= ecap) return true;
        u32 c = lenPrimes > nent ? lenPrimes : nent; if (c < 1) c = 1;
        if (d_root) { cudaFree(d_root); cudaFree(d_ppl); cudaFree(d_p); cudaFree(d_nr);
                      cudaFree(d_primes); cudaFree(d_off32); cudaFree(d_cnt); cudaFree(d_off); }
        d_root = nullptr;
        if (cudaMalloc(&d_root, (size_t) c * 8)) return false;
        if (cudaMalloc(&d_ppl, (size_t) c * 8)) return false;
        if (cudaMalloc(&d_p, (size_t) c * 4)) return false;
        if (cudaMalloc(&d_nr, (size_t) c)) return false;
        if (cudaMalloc(&d_primes, (size_t) c * 4)) return false;
        if (cudaMalloc(&d_off32, (size_t) c * 4)) return false;
        if (cudaMalloc(&d_cnt, (size_t) c * 8)) return false;
        if (cudaMalloc(&d_off, (size_t) (c + 1) * 8)) return false;
        ecap = c; return true;
    }
    bool ensure_u(u64 total) {
        if (total <= ucap) return true;
        if (d_U) { cudaFree(d_U); cudaFree(d_P); }
        d_U = nullptr;
        if (cudaMalloc(&d_U, total * 8)) return false;
        if (cudaMalloc(&d_P, total * 4)) return false;
        ucap = total; return true;
    }
    bool ensure_coll(u32 cap) {
        if (cap <= ccap) return true;
        if (d_oU) { cudaFree(d_oU); cudaFree(d_oP1); cudaFree(d_oP2);
                    cudaFreeHost(h_oU); cudaFreeHost(h_oP1); cudaFreeHost(h_oP2); }
        d_oU = nullptr;
        if (cudaMalloc(&d_oU, (size_t) cap * 8)) return false;
        if (cudaMalloc(&d_oP1, (size_t) cap * 4)) return false;
        if (cudaMalloc(&d_oP2, (size_t) cap * 4)) return false;
        if (!d_ncoll && cudaMalloc(&d_ncoll, 4)) return false;
        if (cudaMallocHost(&h_oU, (size_t) cap * 8)) return false;
        if (cudaMallocHost(&h_oP1, (size_t) cap * 4)) return false;
        if (cudaMallocHost(&h_oP2, (size_t) cap * 4)) return false;
        ccap = cap; return true;
    }
};
/* The GPU is a shared resource and polyselect may call this from several team
 * leaders at once; serialize device work with a global mutex (each call is short).
 * Because the mutex makes collision work strictly single-threaded-at-a-time, the
 * device buffers are SHARED (not thread_local) — one set, reused across all teams,
 * so the multi-GB u-array is allocated 1x rather than once per team (which OOMed),
 * and thrust never runs from two threads at once. Per-stream concurrency with
 * per-thread buffers is a possible future optimization. */
CollBufs g_cbufs;
std::mutex g_coll_mutex;

#define COLL_CAP (1u << 18)               /* max collisions returned before CPU fallback */

int gpu_polyselect_collisions_impl(
        const u32 * primes, const u8 * nr, const i64 * roots_flat,
        u32 lenPrimes, u32 nent, i64 umax,
        const u64 ** out_u, const u32 ** out_p1, const u32 ** out_p2, u32 * out_ncoll)
{
    if (lenPrimes == 0 || nent == 0) { *out_ncoll = 0; return 1; }
    std::lock_guard<std::mutex> lock(g_coll_mutex);
    /* Each polyselect worker thread that reaches here must have device 0 current;
     * raw cudaMalloc auto-selects it, but thrust's execution policy needs it set
     * explicitly (else exclusive_scan throws cudaErrorInvalidDevice). */
    if (cudaSetDevice(0) != cudaSuccess) return 0;
    CollBufs & B = g_cbufs;
    if (!B.ensure_entries(lenPrimes, nent) || !B.ensure_coll(COLL_CAP)) return 0;

    cudaError_t e = cudaSuccess;
  try {
    e = cudaMemcpy(B.d_primes, primes, (size_t) lenPrimes * 4, cudaMemcpyHostToDevice); if (e) goto fail;
    e = cudaMemcpy(B.d_nr, nr, (size_t) lenPrimes, cudaMemcpyHostToDevice); if (e) goto fail;
    e = cudaMemcpy(B.d_root, roots_flat, (size_t) nent * 8, cudaMemcpyHostToDevice); if (e) goto fail;

    {
        int tpb = 128;
        /* per-prime offsets into the entry arrays (exclusive scan of nr). NOTE: nr
         * is uint8_t, so the scan MUST accumulate in a wider type -- pass an
         * unsigned init so the accumulator is 32-bit, else it overflows at 256 and
         * the offsets (hence the whole expansion) are garbage. */
        thrust::device_ptr<u8> nrp(B.d_nr);
        thrust::device_ptr<u32> offp(B.d_off32);
        thrust::exclusive_scan(thrust::device, nrp, nrp + lenPrimes, offp, (unsigned int) 0);
        k_expand<<<(lenPrimes + tpb - 1) / tpb, tpb>>>(B.d_primes, B.d_nr, B.d_off32,
                lenPrimes, B.d_ppl, B.d_p);
        e = cudaGetLastError(); if (e) goto fail;

        /* per-entry emission counts -> exclusive scan -> total */
        k_ccount<<<(nent + tpb - 1) / tpb, tpb>>>(B.d_root, B.d_ppl, nent, umax, B.d_cnt);
        e = cudaGetLastError(); if (e) goto fail;
        thrust::device_ptr<u64> cntp(B.d_cnt), offu(B.d_off);
        thrust::exclusive_scan(thrust::device, cntp, cntp + nent, offu);
        u64 total = 0, last_off = 0, last_cnt = 0;
        e = cudaMemcpy(&last_off, B.d_off + (nent - 1), 8, cudaMemcpyDeviceToHost); if (e) goto fail;
        e = cudaMemcpy(&last_cnt, B.d_cnt + (nent - 1), 8, cudaMemcpyDeviceToHost); if (e) goto fail;
        total = last_off + last_cnt;
        if (getenv("CADO_GPU_POLYSELECT_DEBUG"))
            fprintf(stderr, "polyselect GPU coll: lenPrimes=%u nent=%u umax=%lld total_u=%llu\n",
                    lenPrimes, nent, (long long) umax, (unsigned long long) total);
        if (total == 0) { *out_ncoll = 0; return 1; }
        if (!B.ensure_u(total)) goto fail;

        k_cwrite<<<(nent + tpb - 1) / tpb, tpb>>>(B.d_root, B.d_ppl, B.d_p, nent, umax,
                B.d_off, B.d_U, B.d_P);
        e = cudaGetLastError(); if (e) goto fail;

        /* sort u-values, carrying the prime tag, then detect collisions */
        thrust::device_ptr<u64> Up(B.d_U); thrust::device_ptr<u32> Pp(B.d_P);
        thrust::sort_by_key(thrust::device, Up, Up + total, Pp);
        e = cudaMemset(B.d_ncoll, 0, 4); if (e) goto fail;
        u64 mblk = (total + tpb - 1) / tpb;
        k_cdetect<<<(u32) mblk, tpb>>>(B.d_U, B.d_P, total, COLL_CAP,
                B.d_oU, B.d_oP1, B.d_oP2, B.d_ncoll);
        e = cudaGetLastError(); if (e) goto fail;

        u32 nc = 0;
        e = cudaMemcpy(&nc, B.d_ncoll, 4, cudaMemcpyDeviceToHost); if (e) goto fail;
        if (nc > COLL_CAP) {       /* overflow: too many collisions, let the CPU do it */
            fprintf(stderr, "polyselect GPU: %u collisions exceed cap %u; CPU fallback\n", nc, COLL_CAP);
            return 0;
        }
        e = cudaMemcpy(B.h_oU, B.d_oU, (size_t) nc * 8, cudaMemcpyDeviceToHost); if (e) goto fail;
        e = cudaMemcpy(B.h_oP1, B.d_oP1, (size_t) nc * 4, cudaMemcpyDeviceToHost); if (e) goto fail;
        e = cudaMemcpy(B.h_oP2, B.d_oP2, (size_t) nc * 4, cudaMemcpyDeviceToHost); if (e) goto fail;
        *out_u = B.h_oU; *out_p1 = B.h_oP1; *out_p2 = B.h_oP2; *out_ncoll = nc;
    }
    return 1;
  } catch (const std::exception & ex) {
    fprintf(stderr, "polyselect GPU: thrust error (%s); falling back to CPU collision search\n",
            ex.what());
    return 0;
  }

fail:
    fprintf(stderr, "polyselect GPU: %s; falling back to CPU collision search\n",
            cudaGetErrorString(e));
    return 0;
}

} // namespace

void cado_gpu_polyselect_init(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "polyselect GPU: no CUDA device; using CPU root-finding\n");
        return;
    }
    cado_gpu_polyselect_roots = gpu_polyselect_roots_impl;
    cado_gpu_polyselect_collisions = gpu_polyselect_collisions_impl;
}
