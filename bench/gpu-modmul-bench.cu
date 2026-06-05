/*
 * gpu-modmul-bench.cu — batched 64-bit Montgomery modular-multiply throughput,
 * GPU (RTX 3090, sm_86) vs CPU. This is the inner primitive of ECM, which is
 * what CADO-NFS's cofactorization (sieve/ecm, facul_all over a batch of small
 * cofactors) spends its time on. The GPU/CPU throughput ratio measured here
 * bounds the achievable speedup of offloading cofactorization to the GPU
 * (cf. Bos & Kleinjung; eprint 2014/397).
 *
 * Each lane runs an independent chain of Montgomery modmuls on its own modulus
 * (mirroring ECM's per-curve modmul-bound inner loop). Correctness is checked
 * against a CPU reference; throughput is reported in modmuls/sec for both.
 *
 * Build & run:
 *   nvcc -arch=sm_86 -O3 bench/gpu-modmul-bench.cu -o /tmp/gpu-modmul && /tmp/gpu-modmul
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>

/* -n^{-1} mod 2^64 via Newton iteration (n must be odd) */
static uint64_t montinv(uint64_t n)
{
    uint64_t x = n;                 /* x = n^{-1} mod 2^3 holds for odd n */
    for (int i = 0; i < 5; i++) x *= 2 - n * x;   /* 3 -> 6 -> 12 -> 24 -> 48 -> 96 bits */
    return (uint64_t)(0) - x;       /* negate: -n^{-1} mod 2^64 */
}

/* Montgomery multiply: (a*b*R^{-1}) mod n, R = 2^64. Correct for n < 2^63. */
#ifdef __CUDACC__
__host__ __device__
#endif
static inline uint64_t montmul(uint64_t a, uint64_t b, uint64_t n, uint64_t np)
{
    unsigned __int128 T = (unsigned __int128)a * b;
    uint64_t m = (uint64_t)T * np;
    unsigned __int128 s = T + (unsigned __int128)m * n;   /* divisible by 2^64 */
    uint64_t t = (uint64_t)(s >> 64);
    if (t >= n) t -= n;
    return t;
}

/* one lane: CHAIN modmuls, x <- montmul(x, c, n) */
#define CHAIN 4096

__global__ void modmul_kernel(const uint64_t *n_, const uint64_t *np_,
                              const uint64_t *c_, uint64_t *out, int lanes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lanes) return;
    uint64_t n = n_[i], np = np_[i], c = c_[i], x = c;
    for (int k = 0; k < CHAIN; k++) x = montmul(x, c, n, np);
    out[i] = x;
}

static void cpu_chain(const uint64_t *n_, const uint64_t *np_,
                      const uint64_t *c_, uint64_t *out, int lanes)
{
    for (int i = 0; i < lanes; i++) {
        uint64_t n = n_[i], np = np_[i], c = c_[i], x = c;
        for (int k = 0; k < CHAIN; k++) x = montmul(x, c, n, np);
        out[i] = x;
    }
}

static uint64_t rnd(uint64_t *s) { *s ^= *s<<13; *s ^= *s>>7; *s ^= *s<<17; return *s; }

int main()
{
    const int LANES = 1 << 20;              /* ~1M independent moduli/curves */
    std::vector<uint64_t> n(LANES), np(LANES), c(LANES), og(LANES), oc(LANES);
    uint64_t s = 0x123456789abcdefULL;
    for (int i = 0; i < LANES; i++) {
        uint64_t m = (rnd(&s) % ((1ULL<<52) - 3)) | 1ULL;   /* odd, ~52-bit (cofactor-sized) */
        if (m < 3) m = 3;
        n[i] = m; np[i] = montinv(m); c[i] = rnd(&s) % m;
    }

    /* ---- GPU ---- */
    uint64_t *dn,*dnp,*dc,*dout;
    cudaMalloc(&dn,LANES*8); cudaMalloc(&dnp,LANES*8); cudaMalloc(&dc,LANES*8); cudaMalloc(&dout,LANES*8);
    cudaMemcpy(dn,n.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dnp,np.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dc,c.data(),LANES*8,cudaMemcpyHostToDevice);
    int tpb = 256, blocks = (LANES + tpb - 1) / tpb;
    modmul_kernel<<<blocks,tpb>>>(dn,dnp,dc,dout,LANES);      /* warmup */
    cudaDeviceSynchronize();
    auto g0 = std::chrono::steady_clock::now();
    const int ITERS = 20;
    for (int r = 0; r < ITERS; r++) modmul_kernel<<<blocks,tpb>>>(dn,dnp,dc,dout,LANES);
    cudaDeviceSynchronize();
    auto g1 = std::chrono::steady_clock::now();
    cudaError_t e = cudaGetLastError();
    cudaMemcpy(og.data(),dout,LANES*8,cudaMemcpyDeviceToHost);
    double gsec = std::chrono::duration<double>(g1-g0).count();
    double gmodmuls = (double)LANES * CHAIN * ITERS;

    /* ---- CPU (1 thread; scale note below) ---- */
    int CPU_LANES = LANES / 64;             /* smaller so it finishes quickly */
    auto c0 = std::chrono::steady_clock::now();
    cpu_chain(n.data(),np.data(),c.data(),oc.data(),CPU_LANES);
    auto c1 = std::chrono::steady_clock::now();
    double csec = std::chrono::duration<double>(c1-c0).count();
    double cmodmuls = (double)CPU_LANES * CHAIN;

    /* ---- correctness: GPU vs CPU on the CPU_LANES subset ---- */
    long mism = 0;
    for (int i = 0; i < CPU_LANES; i++) if (og[i] != oc[i]) mism++;

    double gtp = gmodmuls / gsec, ctp1 = cmodmuls / csec;
    printf("GPU status     : %s\n", cudaGetErrorString(e));
    printf("correctness    : %s (%ld/%d lanes mismatched vs CPU)\n",
           mism==0?"PASS":"FAIL", mism, CPU_LANES);
    printf("GPU throughput : %.2f G modmul/s  (%d lanes x %d chain x %d iters in %.3fs)\n",
           gtp/1e9, LANES, CHAIN, ITERS, gsec);
    printf("CPU throughput : %.3f G modmul/s  (1 core)\n", ctp1/1e9);
    printf("CPU throughput : %.3f G modmul/s  (est. 20 cores)\n", ctp1*20/1e9);
    printf("GPU vs 1 core  : %.0fx\n", gtp/ctp1);
    printf("GPU vs 20 core : %.1fx\n", gtp/(ctp1*20));
    cudaFree(dn);cudaFree(dnp);cudaFree(dc);cudaFree(dout);
    return mism != 0;
}
