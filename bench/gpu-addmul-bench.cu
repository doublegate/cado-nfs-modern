/*
 * gpu-addmul-bench.cu — GPU addmul_tiny for the Block Wiedemann mksol inner loop,
 * the compute primitive that lets mksol keep its accumulator device-resident
 * (v3.1.0-modern, Track 2.2, full-residency port — mksol).
 *
 * mksol.cpp's inner loop does, per coefficient k, a host
 *   addmul_tiny(ymy[0].own, vi[i].own, ff_slice, eblock)
 * (linalg/bwc/arith-cross.cpp): w (n times 64L-bit) += u (n times 64K-bit) times
 * the 64K x 64L GF(2) matrix v. Bit-for-bit (non-SSE2 reference):
 *   w[j*L+l] ^= XOR over k<K, i<64 of  ( (u[j*K+k]>>i)&1 ? v[(k*64+i)*L+l] : 0 )
 * Today that runs on the host every iteration, forcing the accumulator back to the
 * CPU. This kernel does the same gather on the GPU, reading a device-resident u
 * (vi[i]) and writing a device-resident w (ymy[0]).
 *
 * Same __host__ __device__ reference on CPU and GPU => validated bit-exact.
 *   nvcc -arch=sm_86 -O3 bench/gpu-addmul-bench.cu -o /tmp/gpu-addmul && /tmp/gpu-addmul
 */
#include <cstdio>
#include <cstdint>
#include <vector>
typedef uint64_t u64;
typedef uint32_t u32;

static u64 xr(u64 * s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* w[j*L+l] ^= XOR over k<K,i<64 of ((u[j*K+k]>>i)&1 ? v[(k*64+i)*L+l] : 0). */
__global__ void addmul_kernel(u64 * w, const u64 * u, const u64 * v,
                              unsigned n, unsigned K, unsigned L)
{
    size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    unsigned j = (unsigned) (idx / L), l = (unsigned) (idx % L);
    if (j >= n) return;
    u64 rx = 0;
    for (unsigned k = 0; k < K; k++) {
        u64 a = u[(size_t) j * K + k];
        const u64 * vv = v + (size_t) (k * 64) * L + l;
        for (unsigned i = 0; i < 64; i++) {
            rx ^= vv[0] & (~(u64) 0 * (a & 1));   /* mask = all-ones if bit set */
            a >>= 1;
            vv += L;
        }
    }
    w[(size_t) j * L + l] ^= rx;
}

/* CPU reference (mirrors arith-cross.cpp non-SSE2 addmul_tiny exactly). */
static void cpu_addmul(u64 * w, const u64 * u, const u64 * v, unsigned n, unsigned K, unsigned L) {
    for (unsigned j = 0; j < n; j++)
        for (unsigned l = 0; l < L; l++) {
            u64 rx = 0;
            for (unsigned k = 0; k < K; k++) {
                u64 a = u[(size_t) j * K + k];
                const u64 * vv = v + (size_t) (k * 64) * L + l;
                for (unsigned i = 0; i < 64; i++) { rx ^= vv[0] & (~(u64) 0 * (a & 1)); a >>= 1; vv += L; }
            }
            w[(size_t) j * L + l] ^= rx;
        }
}

static int run(const char * label, unsigned n, unsigned K, unsigned L) {
    u64 st = 0xADD0ULL + (K << 8) + L; if (!st) st = 1;
    std::vector<u64> u((size_t) n * K), v((size_t) 64 * K * L), w0((size_t) n * L), w1;
    for (auto & x : u) x = xr(&st);
    for (auto & x : v) x = xr(&st);
    for (auto & x : w0) x = xr(&st);           /* addmul accumulates onto existing w */
    w1 = w0;

    u64 *du, *dv, *dw;
    if (cudaMalloc(&du, u.size() * 8) != cudaSuccess) { printf("  [%s] malloc fail\n", label); return 2; }
    cudaMalloc(&dv, v.size() * 8); cudaMalloc(&dw, w0.size() * 8);
    cudaMemcpy(du, u.data(), u.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), v.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w0.data(), w0.size() * 8, cudaMemcpyHostToDevice);
    int tpb = 64; size_t blk = ((size_t) n * L + tpb - 1) / tpb;
    addmul_kernel<<<blk, tpb>>>(dw, du, dv, n, K, L);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    std::vector<u64> wg(w0.size());
    cudaMemcpy(wg.data(), dw, w0.size() * 8, cudaMemcpyDeviceToHost);
    cudaFree(du); cudaFree(dv); cudaFree(dw);

    cpu_addmul(w1.data(), u.data(), v.data(), n, K, L);
    long mis = 0; for (size_t i = 0; i < w1.size(); i++) if (w1[i] != wg[i]) mis++;
    printf("  [%s] n=%u K=%u L=%u : %s (%ld/%zu differ)%s\n", label, n, K, L,
           mis == 0 ? "PASS" : "FAIL", mis, w1.size(), e ? "  CUDAERR" : "");
    return mis != 0;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        printf("gpu-addmul: no CUDA device, skipping\n"); return 0;
    }
    printf("GPU addmul_tiny (mksol accumulator) — validated bit-exact vs CPU reference\n");
    int f = 0;
    f += run("b64 ", 100000, 1, 1);   /* Av=As=64 */
    f += run("b128", 100000, 2, 2);   /* Av=As=128 */
    f += run("mix ", 50000, 2, 1);    /* Av=128, As=64 */
    printf("%s\n", f == 0 ? "ALL PASS" : "FAILURES");
    return f != 0;
}
