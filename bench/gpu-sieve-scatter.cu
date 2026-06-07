/*
 * gpu-sieve-scatter.cu — measured feasibility probe for GPU NFS lattice sieving
 * (Roadmap C4). The core sieve operation is a random read-modify-write SCATTER
 * into the sieve array (S[x] -= log p for each hit). This benchmarks that pattern
 * on the GPU vs the CPU to quantify, honestly, why GPU sieving is unsolved.
 *
 * NFS lattice sieving spends its time in:
 *   - fill_in_buckets / apply_buckets : push/apply scattered byte updates,
 *   - sieve_small_bucket_region       : strided byte updates S[r], S[r+p], ...,
 * i.e. RANDOM SCATTER into a cache-resident byte array (the bucket region is
 * sized to fit L1/L2 by design). CADO's apply does these as scalar updates within
 * a single-thread bucket region (no races). On the GPU, many threads scattering
 * into the same region need ATOMICS (conflicts), and byte granularity isn't
 * directly atomic — both fundamental mismatches. This probe measures:
 *   (1) CPU scalar scatter into a cache-resident region (the real siever's op),
 *   (2) GPU atomic scatter into the same region (correct, conflict-bound),
 *   (3) GPU coalesced sequential add (the friendly upper bound, NOT the sieve),
 * across region sizes spanning cache-resident -> DRAM.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-sieve-scatter.cu -o gpu-sieve-scatter && ./gpu-sieve-scatter
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef uint32_t u32;
typedef unsigned long long u64;

static u64 xrnd(u64 *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* GPU: atomic scatter — each thread applies one update S[off[i]] += v[i] */
__global__ void k_atomic_scatter(int *S, const u32 *off, const u32 *v, int U) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < U) atomicAdd(&S[off[i]], (int)v[i]);
}
/* GPU friendly upper bound: coalesced sequential accumulate (NOT the sieve) */
__global__ void k_coalesced(int *S, const u32 *v, int U, int R) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < U) S[i % R] += (int)v[i];   /* near-sequential within a warp */
}

static double cpu_scatter(std::vector<int>&S, const std::vector<u32>&off,
                          const std::vector<u32>&v, int reps) {
    int U = off.size();
    auto t0 = std::chrono::steady_clock::now();
    volatile int sink = 0;
    for (int r=0;r<reps;r++)
        for (int i=0;i<U;i++) S[off[i]] += (int)v[i];
    auto t1 = std::chrono::steady_clock::now();
    sink = S[off[0]]; (void)sink;
    return std::chrono::duration<double>(t1-t0).count() / reps;
}
/* multi-threaded CPU: each thread owns a PRIVATE region and a slice of updates —
 * exactly the siever's model (one bucket region per thread, no atomics). */
static double cpu_scatter_mt(int R, const std::vector<u32>&off,
                             const std::vector<u32>&v, int reps, int *nthreads) {
    int U = off.size();
    double secs = 0;
#ifdef _OPENMP
    *nthreads = omp_get_max_threads();
#else
    *nthreads = 1;
#endif
    /* allocate one private region per thread ONCE (reused across reps), so the
     * timing measures the scatter, not allocation — the siever reuses its buffer. */
    #ifdef _OPENMP
    #pragma omp parallel
    #endif
    {
        std::vector<int> Spriv(R, 0);
        volatile int sink = 0;
        #ifdef _OPENMP
        #pragma omp barrier
        #endif
        auto t0 = std::chrono::steady_clock::now();
        for (int r=0;r<reps;r++) {
            #ifdef _OPENMP
            #pragma omp for schedule(static)
            #endif
            for (int i=0;i<U;i++) Spriv[off[i]] += (int)v[i];
        }
        auto t1 = std::chrono::steady_clock::now();
        sink = Spriv[off[0]]; (void)sink;
        #ifdef _OPENMP
        #pragma omp master
        #endif
        secs = std::chrono::duration<double>(t1-t0).count() / reps;
    }
    return secs;
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    printf("GPU vs CPU sieve-scatter probe (C4). Updates/s for S[off]+=v, the core sieve op.\n");
    printf("region = bucket-region size (CPU sieve keeps it cache-resident by design).\n\n");
    printf("%-12s %13s %13s %13s %9s %12s\n",
           "region", "CPU 1-thread", "CPU all-core", "GPU atomic", "GPU/CPU*", "GPUcoalesced");
    printf("(* GPU vs CPU all-core = the honest full-socket comparison)\n");

    const int U = 1<<22;           /* 4M updates per measurement */
    u64 seed = 0xC0FFEE;
    /* region sizes (in int cells): 16KiB, 256KiB, 4MiB, 64MiB (cells*4 bytes) */
    int regionsKB[] = {16, 256, 4096, 65536};
    for (int ri=0; ri<4; ri++) {
        int R = regionsKB[ri]*1024/4;   /* int cells */
        std::vector<u32> off(U), v(U);
        for (int i=0;i<U;i++){ off[i] = (u32)(xrnd(&seed) % R); v[i] = (u32)(xrnd(&seed)&0xff); }
        std::vector<int> Sc(R, 0);
        double cpu = cpu_scatter(Sc, off, v, 4);
        double cpu_ups = U / cpu;
        int nthr = 1;
        double cpu_mt = cpu_scatter_mt(R, off, v, 4, &nthr);
        double cpu_mt_ups = U / cpu_mt;

        int *dS; u32 *doff,*dv; cudaMalloc(&dS,(size_t)R*4); cudaMalloc(&doff,(size_t)U*4); cudaMalloc(&dv,(size_t)U*4);
        cudaMemset(dS,0,(size_t)R*4);
        cudaMemcpy(doff,off.data(),(size_t)U*4,cudaMemcpyHostToDevice);
        cudaMemcpy(dv,v.data(),(size_t)U*4,cudaMemcpyHostToDevice);
        int TPB=256, B=(U+TPB-1)/TPB;
        /* warmup */ k_atomic_scatter<<<B,TPB>>>(dS,doff,dv,U); cudaDeviceSynchronize();
        int REPS=20;
        auto t0=std::chrono::steady_clock::now();
        for(int r=0;r<REPS;r++) k_atomic_scatter<<<B,TPB>>>(dS,doff,dv,U);
        cudaDeviceSynchronize(); auto t1=std::chrono::steady_clock::now();
        double gA = std::chrono::duration<double>(t1-t0).count()/REPS;
        double gA_ups = U/gA;
        k_coalesced<<<B,TPB>>>(dS,dv,U,R); cudaDeviceSynchronize();
        auto t2=std::chrono::steady_clock::now();
        for(int r=0;r<REPS;r++) k_coalesced<<<B,TPB>>>(dS,dv,U,R);
        cudaDeviceSynchronize(); auto t3=std::chrono::steady_clock::now();
        double gC_ups = U/(std::chrono::duration<double>(t3-t2).count()/REPS);
        cudaFree(dS);cudaFree(doff);cudaFree(dv);

        char rk[16]; snprintf(rk,sizeof rk,"%dKiB", regionsKB[ri]);
        printf("%-12s %11.0fM %11.0fM %11.0fM %8.2fx %10.0fM\n",
               rk, cpu_ups/1e6, cpu_mt_ups/1e6, gA_ups/1e6, gA_ups/cpu_mt_ups, gC_ups/1e6);
    }
    printf("\nCPU all-core = per-thread private regions (the siever's model, no atomics).\n");
    printf("GPU 'atomic' is the correct sieve op (conflict-bound); 'coalesced' is a\n");
    printf("non-sieve friendly upper bound. NB: GPU uses int cells (4B); the real sieve\n");
    printf("array is uint8 (no 8-bit GPU atomic) and the updates must also be GENERATED\n");
    printf("on-GPU (lattice arithmetic + bucket fill) — costs this probe excludes.\n");
    return 0;
}
