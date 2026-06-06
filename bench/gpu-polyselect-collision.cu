/*
 * gpu-polyselect-collision.cu — GPU collision search for polyselect stage-1
 * (v3.2.0-modern, Track C2 continuation: "offload the collision search at large N").
 *
 * CADO's collision search (polyselect_collisions.cpp + polyselect_shash.cpp) is the
 * memory-bound bulk of Kleinjung stage-1 and the part that grows with N. For each
 * prime p with a lifted root r it emits EVERY u == r (mod p^2) in [-umax, umax) (an
 * arithmetic progression, step ppl = p^2), then detects two equal u from different
 * primes -- a collision that yields a candidate polynomial. The CPU does this with a
 * two-level open-addressing hash (shash); the GPU-friendly reformulation (msieve's
 * model, on the GPU since 2009) is generate -> radix-sort -> detect adjacent equal,
 * with the huge intermediate u-array RESIDENT ON DEVICE so only the small (p,r) table
 * goes in and only the few collisions come back. That fusion is what sidesteps the
 * Amdahl/PCIe trap that made offloading root-finding alone a net loss (see
 * docs/gpu-polyselect.md): at large N the generation + sort dominate and never leave
 * the device.
 *
 * This is the validated FOUNDATION kernel for that offload. It builds the EXACT same
 * u-multiset as CADO's dispatch loop, sorts it on the GPU, and detects collisions
 * (equal adjacent u). The gate is bit-exact vs a CPU reference (std::sort + adjacent
 * compare) on a realistic workload: identical sorted multiset and identical collision
 * set (u-value + the two source-prime indices). Correct regardless of how many
 * collisions occur, so it does not rely on the birthday paradox firing.
 *
 *   nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-polyselect-collision.cu -o gpu-polyselect-collision
 *   ./gpu-polyselect-collision
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef int64_t  i64;

#define CK(call) do { cudaError_t e_ = (call); if (e_) { \
    printf("CUDA error %s at %d: %s\n", #call, __LINE__, cudaGetErrorString(e_)); return 2; } } while (0)

/* Exact per-(p,r) emission count, matching polyselect_proots_dispatch_to_shash_flat:
 *   for (u = u0;       u <  umax;       u += ppl) emit;   // k >= 0
 *   for (u = u0 - ppl; u + umax >= 0;   u -= ppl) emit;   // k >= 1
 * with u0 = r (an int64), ppl = p^2 > 0. */
__host__ __device__ static inline u64 emit_count(i64 u0, i64 ppl, i64 umax)
{
    u64 npos = (u0 < umax) ? (u64) ((umax - 1 - u0) / ppl) + 1 : 0;
    u64 nneg = (u0 + umax >= 0) ? (u64) ((u0 + umax) / ppl) : 0;   /* k = 1 .. floor */
    return npos + nneg;
}

/* count kernel: one thread per (p,r) entry, writes its emission count */
__global__ void k_count(const i64 * r, const i64 * ppl, u32 n, i64 umax, u64 * cnt)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    cnt[i] = emit_count(r[i], ppl[i], umax);
}

/* write kernel: one thread per (p,r) entry; fills its u-values at off[i].. and tags
 * each with the source entry index so a collision can name its two primes. */
__global__ void k_write(const i64 * r, const i64 * ppl, u32 n, i64 umax,
                        const u64 * off, u64 * U, u32 * SRC)
{
    u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    i64 u0 = r[i], pp = ppl[i];
    u64 o = off[i];
    for (i64 u = u0; u < umax; u += pp) { U[o] = (u64) u; SRC[o] = i; o++; }
    for (i64 u = u0 - pp; u + umax >= 0; u -= pp) { U[o] = (u64) u; SRC[o] = i; o++; }
}

/* mark kernel: U sorted by key; flag[t]=1 iff U[t]==U[t+1] and they come from
 * different source entries (a real collision, not two points of one progression --
 * which cannot be equal anyway, but the src check makes that explicit/robust). */
__global__ void k_mark(const u64 * U, const u32 * SRC, u64 m, u32 * flag)
{
    u64 t = (u64) blockIdx.x * blockDim.x + threadIdx.x; if (t + 1 >= m) return;
    flag[t] = (U[t] == U[t + 1] && SRC[t] != SRC[t + 1]) ? 1u : 0u;
}

int main()
{
    /* --- workload: primes in [PMIN,PMAX], one lifted root each, large umax so the
     * u-multiset is in the tens of millions (a realistic stage-1 stress). --- */
    const u32 PMIN = 50000, PMAX = 100000;
    const i64 UMAX = 25000000000000LL;            /* 2.5e13 */
    u64 s = 0xC0111DE5ULL;
    auto xr = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };

    std::vector<u32> primes;
    { std::vector<char> sv(PMAX + 1, 1);
      for (u32 i = 2; i <= PMAX; i++) if (sv[i]) { if (i >= PMIN) primes.push_back(i);
          for (u32 j = 2 * i; j <= PMAX; j += i) sv[j] = 0; } }
    u32 n = (u32) primes.size();

    std::vector<i64> r(n), ppl(n);
    for (u32 i = 0; i < n; i++) {
        i64 pp = (i64) primes[i] * (i64) primes[i];
        ppl[i] = pp;
        r[i] = (i64) (xr() % (u64) pp);           /* root in [0, p^2) */
    }
    /* inject a handful of guaranteed collisions: force a few primes to share a u. */
    int injected = 0;
    for (u32 i = 5; i + 1 < n && injected < 8; i += (n / 9) + 1) {
        i64 target = (i64) (xr() % (u64) UMAX);   /* a u in range */
        r[i]     = ((target % ppl[i])     + ppl[i])     % ppl[i];
        r[i + 1] = ((target % ppl[i + 1]) + ppl[i + 1]) % ppl[i + 1];
        injected++;
    }

    /* ---------------- GPU: count -> scan -> write -> sort -> mark ---------------- */
    i64 *dr, *dppl; u64 *dcnt, *doff;
    CK(cudaMalloc(&dr, (size_t) n * 8)); CK(cudaMalloc(&dppl, (size_t) n * 8));
    CK(cudaMalloc(&dcnt, (size_t) n * 8)); CK(cudaMalloc(&doff, (size_t) (n + 1) * 8));
    CK(cudaMemcpy(dr, r.data(), (size_t) n * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dppl, ppl.data(), (size_t) n * 8, cudaMemcpyHostToDevice));

    int tpb = 128; u32 blk = (n + tpb - 1) / tpb;
    k_count<<<blk, tpb>>>(dr, dppl, n, UMAX, dcnt); CK(cudaGetLastError());
    /* exclusive scan -> offsets; total = off[n] */
    thrust::exclusive_scan(thrust::device, thrust::device_ptr<u64>(dcnt),
                           thrust::device_ptr<u64>(dcnt) + n, thrust::device_ptr<u64>(doff));
    u64 total = 0, last_cnt = 0, last_off = 0;
    CK(cudaMemcpy(&last_off, doff + (n - 1), 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&last_cnt, dcnt + (n - 1), 8, cudaMemcpyDeviceToHost));
    total = last_off + last_cnt;
    CK(cudaMemcpy(doff + n, &total, 8, cudaMemcpyHostToDevice));
    printf("workload: %u primes [%u,%u], umax=%lld -> %llu u-values (%.1f MB), %d injected collisions\n",
           n, PMIN, PMAX, (long long) UMAX, (unsigned long long) total,
           total * 12.0 / 1e6, injected);

    u64 * dU; u32 * dSRC;
    CK(cudaMalloc(&dU, total * 8)); CK(cudaMalloc(&dSRC, total * 4));
    k_write<<<blk, tpb>>>(dr, dppl, n, UMAX, doff, dU, dSRC); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    auto g0 = std::chrono::steady_clock::now();
    /* sort u-values, carrying the source index (sort_by_key) */
    thrust::sort_by_key(thrust::device, thrust::device_ptr<u64>(dU),
                        thrust::device_ptr<u64>(dU) + total, thrust::device_ptr<u32>(dSRC));
    u32 * dflag; CK(cudaMalloc(&dflag, total * 4)); CK(cudaMemset(dflag, 0, total * 4));
    u64 mblk = (total + tpb - 1) / tpb;
    k_mark<<<(u32) mblk, tpb>>>(dU, dSRC, total, dflag); CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    auto g1 = std::chrono::steady_clock::now();
    double gsec = std::chrono::duration<double>(g1 - g0).count();

    /* pull GPU collisions: (u, srcA, srcB) for each flagged adjacent pair */
    std::vector<u64> hU(total); std::vector<u32> hSRC(total), hflag(total);
    CK(cudaMemcpy(hU.data(), dU, total * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hSRC.data(), dSRC, total * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hflag.data(), dflag, total * 4, cudaMemcpyDeviceToHost));
    std::vector<std::pair<u64, std::pair<u32, u32>>> gpu_coll;
    for (u64 t = 0; t + 1 < total; t++) if (hflag[t]) {
        u32 a = hSRC[t], b = hSRC[t + 1]; if (a > b) std::swap(a, b);
        gpu_coll.push_back({hU[t], {a, b}});
    }

    /* ---------------- CPU reference: same multiset, std::sort, adjacent ---------- */
    auto c0 = std::chrono::steady_clock::now();
    std::vector<std::pair<u64, u32>> cpu(total); u64 w = 0;
    for (u32 i = 0; i < n; i++) {
        i64 u0 = r[i], pp = ppl[i];
        for (i64 u = u0; u < UMAX; u += pp) cpu[w++] = {(u64) u, i};
        for (i64 u = u0 - pp; u + UMAX >= 0; u -= pp) cpu[w++] = {(u64) u, i};
    }
    std::sort(cpu.begin(), cpu.end(),
              [](const std::pair<u64,u32>& a, const std::pair<u64,u32>& b){ return a.first < b.first; });
    auto c1 = std::chrono::steady_clock::now();
    double csec = std::chrono::duration<double>(c1 - c0).count();
    std::vector<std::pair<u64, std::pair<u32, u32>>> cpu_coll;
    for (u64 t = 0; t + 1 < total; t++)
        if (cpu[t].first == cpu[t + 1].first && cpu[t].second != cpu[t + 1].second) {
            u32 a = cpu[t].second, b = cpu[t + 1].second; if (a > b) std::swap(a, b);
            cpu_coll.push_back({cpu[t].first, {a, b}});
        }

    /* ---------------- compare: sorted multiset + collision set ------------------- */
    long mism = 0;
    for (u64 t = 0; t < total; t++) if (hU[t] != cpu[t].first) { mism++; if (mism <= 3) printf("  sort mism at %llu: gpu=%llu cpu=%llu\n", (unsigned long long)t,(unsigned long long)hU[t],(unsigned long long)cpu[t].first); }
    auto norm = [](std::vector<std::pair<u64,std::pair<u32,u32>>>& v){ std::sort(v.begin(), v.end()); };
    norm(gpu_coll); norm(cpu_coll);
    bool coll_ok = (gpu_coll.size() == cpu_coll.size());
    if (coll_ok) for (size_t i = 0; i < gpu_coll.size(); i++) if (gpu_coll[i] != cpu_coll[i]) { coll_ok = false; break; }

    printf("%s: sorted multiset %s (%ld mism); collisions GPU=%zu CPU=%zu %s\n",
           (mism == 0 && coll_ok) ? "PASS" : "FAIL",
           mism == 0 ? "identical" : "DIFFER", mism,
           gpu_coll.size(), cpu_coll.size(), coll_ok ? "identical" : "DIFFER");
    printf("        GPU gen+sort+detect %.1f ms | CPU sort %.1f ms | %.1fx  (%.0f Mu/s GPU)\n",
           gsec * 1e3, csec * 1e3, csec > 0 ? csec / gsec : 0, total / gsec / 1e6);

    cudaFree(dr); cudaFree(dppl); cudaFree(dcnt); cudaFree(doff);
    cudaFree(dU); cudaFree(dSRC); cudaFree(dflag);
    return (mism == 0 && coll_ok) ? 0 : 1;
}
