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
#include <cuda_runtime.h>

#include "polyselect-gpu-hooks.h"

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

} // namespace

void cado_gpu_polyselect_init(void)
{
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        fprintf(stderr, "polyselect GPU: no CUDA device; using CPU root-finding\n");
        return;
    }
    cado_gpu_polyselect_roots = gpu_polyselect_roots_impl;
}
