/*
 * gpu-polyselect-roots.cu — GPU batched polynomial root-finding mod primes, the
 * second building block for GPU polynomial selection (v3.2.0-modern, Track C2).
 *
 * For a fixed degree-d polynomial f and a batch of primes p, find all roots of
 * f mod p (the per-prime root finding that dominates polyselect stage-1, ~25-40 %
 * with the modular inverse). One thread per prime, direct Horner evaluation over
 * F_p. Exactly correct by construction (roots = {a : f(a) == 0 mod p}); validated
 * bit-exact vs a CPU reference plus a self-check that every root satisfies f(r)==0.
 *
 * Honest scope: direct evaluation is O(p) per prime, so it is fast only in the
 * SMALL-prime regime (load-imbalanced and slow for large p with one thread/prime).
 * The asymptotically-better method polyselect ultimately needs is
 * gcd(x^p - x, f) mod p (O(d^2 log p), independent of p's magnitude) — the next
 * C2 sub-step (poly arithmetic mod f, reusing the validated modular inverse).
 *
 *   nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-polyselect-roots.cu -o gpu-polyselect-roots
 *   ./gpu-polyselect-roots
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
typedef uint32_t u32; typedef uint64_t u64;
#define DMAX 8

/* f(a) mod p by Horner; c[0..deg], c[deg] leading. p < 2^31 => a*acc < 2^62. */
__host__ __device__ static u64 evalmod(const u64 * c, int deg, u64 a, u64 p)
{
    u64 acc = c[deg] % p;
    for (int i = deg - 1; i >= 0; i--) acc = (acc * a + c[i]) % p;
    return acc;
}
__global__ void k_roots(const u64 * c, int deg, const u32 * p, int n, u32 * roots, u32 * cnt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    u64 pp = p[i]; int k = 0;
    for (u64 a = 0; a < pp && k < DMAX; a++)
        if (evalmod(c, deg, a, pp) == 0) roots[(size_t) i * DMAX + k++] = (u32) a;
    cnt[i] = k;
}
static void cpu_roots(const u64 * c, int deg, u32 p, u32 * roots, u32 * cnt)
{
    int k = 0;
    for (u64 a = 0; a < p && k < DMAX; a++)
        if (evalmod(c, deg, a, p) == 0) roots[k++] = (u32) a;
    *cnt = k;
}

int main()
{
    const int deg = 6;
    u64 c[DMAX + 1];
    u64 s = 0x5151ULL;
    auto xr = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int i = 0; i <= deg; i++) c[i] = xr() % 2000000000ULL;   /* random degree-6 f */

    const int BOUND = 50000;                 /* small-prime regime (polyselect's P-ish) */
    std::vector<u32> primes;
    { std::vector<char> sv(BOUND + 1, 1);
      for (int i = 2; i <= BOUND; i++) if (sv[i]) { primes.push_back(i);
          for (int j = 2 * i; j <= BOUND; j += i) sv[j] = 0; } }
    int n = (int) primes.size();

    u64 * dc; u32 *dp, *droots, *dcnt;
    cudaMalloc(&dc, (deg + 1) * 8); cudaMalloc(&dp, n * 4);
    cudaMalloc(&droots, (size_t) n * DMAX * 4); cudaMalloc(&dcnt, n * 4);
    cudaMemcpy(dc, c, (deg + 1) * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dp, primes.data(), n * 4, cudaMemcpyHostToDevice);
    int tpb = 128, blk = (n + tpb - 1) / tpb;
    k_roots<<<blk, tpb>>>(dc, deg, dp, n, droots, dcnt); cudaDeviceSynchronize();
    auto g0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 20; it++) k_roots<<<blk, tpb>>>(dc, deg, dp, n, droots, dcnt);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::steady_clock::now();
    std::vector<u32> roots((size_t) n * DMAX), cnt(n);
    cudaMemcpy(roots.data(), droots, (size_t) n * DMAX * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(cnt.data(), dcnt, n * 4, cudaMemcpyDeviceToHost);
    double gsec = std::chrono::duration<double>(g1 - g0).count() / 20;

    /* CPU reference, all cores (same direct-eval) */
    std::vector<u32> croots((size_t) n * DMAX), ccnt(n);
    int nthr = (int) std::thread::hardware_concurrency(); if (nthr < 1) nthr = 1;
    auto c0 = std::chrono::steady_clock::now();
    { std::vector<std::thread> th;
      for (int t = 0; t < nthr; t++) th.emplace_back([&, t] {
          for (int i = t; i < n; i += nthr) cpu_roots(c, deg, primes[i], &croots[(size_t) i * DMAX], &ccnt[i]);
      });
      for (auto & x : th) x.join(); }
    auto c1 = std::chrono::steady_clock::now();
    double csec = std::chrono::duration<double>(c1 - c0).count();

    long mism = 0, selfbad = 0, totroots = 0;
    for (int i = 0; i < n; i++) {
        if (cnt[i] != ccnt[i]) { mism++; continue; }
        for (int k = 0; k < (int) cnt[i]; k++) {
            if (roots[(size_t) i * DMAX + k] != croots[(size_t) i * DMAX + k]) mism++;
            if (evalmod(c, deg, roots[(size_t) i * DMAX + k], primes[i]) != 0) selfbad++;
        }
        totroots += cnt[i];
    }
    printf("%s: GPU poly root-finding mod %d primes (deg %d, p<%d): %ld mismatch, "
           "%ld self-check-bad, %ld roots\n",
           (mism == 0 && selfbad == 0) ? "PASS" : "FAIL", n, deg, BOUND, mism, selfbad, totroots);
    printf("        GPU %.2f ms | CPU(%d thr) %.2f ms | %.1fx (direct-eval is O(p): small-p regime; "
           "gcd(x^p-x,f) is the next step for large p)\n",
           gsec * 1e3, nthr, csec * 1e3, csec > 0 ? csec / gsec : 0);
    cudaFree(dc); cudaFree(dp); cudaFree(droots); cudaFree(dcnt);
    return (mism || selfbad) != 0;
}
