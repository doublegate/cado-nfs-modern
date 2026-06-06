/* test_gpu_vecreduce.cu — ctest for the GPU GF(2) intra-node reduce+broadcast
 * that backs the comm-on-device hook (matmul-gpu.cu: vecreduce_inplace /
 * vecbroadcast_n), v3.1.0-modern Track 2.2.
 *
 * The single-node BWC comm must turn the T per-core "sibling" vectors into the
 * XOR-sum of them all (reduce), then copy that back over every sibling
 * (broadcast). Over F_2 the reduction is element-wise XOR. This test runs the
 * same in-place reduce-into-sibling[0] + broadcast that the hook uses and checks
 * it bit-for-bit against a CPU reference, over several widths T and sizes.
 *
 * Built only with -DENABLE_GPU=ON. Exits 0 on pass OR when no CUDA device is
 * present (so it is a no-op on GPU-less CI), nonzero only on a real mismatch. */
#include <cstdio>
#include <cstdint>
#include <vector>
typedef uint64_t u64;

static u64 xr(u64 * s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* in-place reduce: p0[g] = XOR over t<T of p[t][g] (p[0] is the accumulator). */
__global__ void reduce_inplace(u64 ** p, unsigned T, size_t words) {
    size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= words) return;
    u64 acc = 0;
    for (unsigned t = 0; t < T; t++) acc ^= p[t][g];
    p[0][g] = acc;
}
/* broadcast: p[t][g] = p[0][g] for all t>0. */
__global__ void broadcast_n(u64 ** p, unsigned T, size_t words) {
    size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= words) return;
    u64 v = p[0][g];
    for (unsigned t = 1; t < T; t++) p[t][g] = v;
}

static int run(unsigned T, size_t words) {
    std::vector<u64> in((size_t) T * words), cpu(words);
    u64 st = 0xC0FFEEULL ^ (u64) words ^ ((u64) T << 32); if (!st) st = 1;
    for (auto & x : in) x = xr(&st);
    for (size_t g = 0; g < words; g++) { u64 a = 0; for (unsigned t = 0; t < T; t++) a ^= in[(size_t) t * words + g]; cpu[g] = a; }

    /* T device buffers, each seeded with sibling t's data */
    std::vector<u64 *> hd(T);
    for (unsigned t = 0; t < T; t++) {
        if (cudaMalloc(&hd[t], words * 8) != cudaSuccess) { printf("  malloc fail\n"); return 2; }
        cudaMemcpy(hd[t], in.data() + (size_t) t * words, words * 8, cudaMemcpyHostToDevice);
    }
    u64 ** dp = nullptr;
    cudaMalloc(&dp, T * sizeof(u64 *));
    cudaMemcpy(dp, hd.data(), T * sizeof(u64 *), cudaMemcpyHostToDevice);

    int tpb = 256; size_t blk = (words + tpb - 1) / tpb;
    reduce_inplace<<<blk, tpb>>>(dp, T, words);
    broadcast_n<<<blk, tpb>>>(dp, T, words);
    cudaDeviceSynchronize();

    /* every sibling must now equal the CPU XOR-sum */
    long mis = 0;
    std::vector<u64> back(words);
    for (unsigned t = 0; t < T; t++) {
        cudaMemcpy(back.data(), hd[t], words * 8, cudaMemcpyDeviceToHost);
        for (size_t g = 0; g < words; g++) if (back[g] != cpu[g]) mis++;
    }
    for (unsigned t = 0; t < T; t++) cudaFree(hd[t]);
    cudaFree(dp);
    printf("  T=%u words=%zu : %s (%ld differ)\n", T, words, mis == 0 ? "PASS" : "FAIL", mis);
    return mis != 0;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("test_gpu_vecreduce: no CUDA device, skipping\n");
        return 0;       /* not a failure on GPU-less machines */
    }
    printf("GPU GF(2) reduce+broadcast (comm-on-device core) — bit-exact vs CPU\n");
    int f = 0;
    f += run(2, 65536);     /* b64, T=2 */
    f += run(3, 65536);     /* odd width */
    f += run(4, 131072);    /* T=4 (e.g. -t 8 grid) */
    f += run(2, 262144);    /* b128-sized words */
    printf("%s\n", f == 0 ? "ALL PASS" : "FAILURES");
    return f != 0;
}
