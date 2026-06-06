/*
 * gpu-polyselect-modinv.cu — GPU batched single-word modular inverse, the
 * foundation kernel for GPU polynomial selection (v3.2.0-modern, Track C2).
 *
 * A `perf` profile of CADO's polyselect stage-1 (collision search) puts the
 * hottest leaf in `modredcul_intinv` (~16 % self) — the modular inverse mod the
 * small primes p, called inside the per-prime root finding (`modul_poly_roots`
 * -> `modul_poly_xpowmod_ui` / `modul_poly_div_r`, another ~25 %). Root finding
 * over thousands of independent primes is embarrassingly parallel — the GPU
 * sweet spot (cf. the batched GPU ECM behind --gpu-prefactor).
 *
 * This validates the building block: a GPU binary-extended-Euclid modular inverse
 * over many (a, p) pairs, bit-exact vs GMP. The modular inverse is mathematically
 * unique, so "bit-exact vs GMP" is the meaningful correctness gate (independent of
 * CADO's internal REDC representation). The next C2 step is the full per-prime
 * root-finding kernel (x^p mod f, gcd) feeding the collision search.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-polyselect-modinv.cu -lgmp -o gpu-polyselect-modinv
 *   ./gpu-polyselect-modinv
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <gmp.h>
#include <cuda_runtime.h>
typedef uint64_t u64;

/* a^{-1} mod m via the extended Euclidean algorithm (m prime here, as in the
 * polyselect projective-root computation). Returns 0 if not invertible. */
__host__ __device__ static u64 modinv(u64 a, u64 m)
{
    if (a == 0) return 0;
    long long t0 = 0, t1 = 1;
    u64 r0 = m, r1 = a % m;
    while (r1) {
        u64 q = r0 / r1, r2 = r0 - q * r1;
        long long t2 = t0 - (long long) q * t1;
        r0 = r1; r1 = r2; t0 = t1; t1 = t2;
    }
    if (r0 != 1) return 0;
    long long inv = t0 % (long long) m;
    if (inv < 0) inv += m;
    return (u64) inv;
}

__global__ void k_modinv(const u64 * a, const u64 * m, u64 * out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = modinv(a[i], m[i]);
}

int main()
{
    const int N = 200000;
    std::vector<u64> a(N), m(N), out(N), primes;
    primes.reserve(N);
    mpz_t p; mpz_init_set_ui(p, 7000);          /* small primes, like polyselect's P */
    for (int i = 0; i < N; i++) {
        mpz_nextprime(p, p);
        if (mpz_sizeinbase(p, 2) > 40) { mpz_set_ui(p, 100003); mpz_nextprime(p, p); }
        primes.push_back(mpz_get_ui(p));
    }
    u64 s = 0xC2C2ULL;
    auto xr = [&] { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; };
    for (int i = 0; i < N; i++) { m[i] = primes[i]; a[i] = 1 + xr() % (m[i] - 1); }

    u64 *da, *dm, *dout;
    cudaMalloc(&da, N * 8); cudaMalloc(&dm, N * 8); cudaMalloc(&dout, N * 8);
    cudaMemcpy(da, a.data(), N * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dm, m.data(), N * 8, cudaMemcpyHostToDevice);
    int tpb = 128, blk = (N + tpb - 1) / tpb;
    k_modinv<<<blk, tpb>>>(da, dm, dout, N); cudaDeviceSynchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < 50; it++) k_modinv<<<blk, tpb>>>(da, dm, dout, N);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    cudaMemcpy(out.data(), dout, N * 8, cudaMemcpyDeviceToHost);
    double sec = std::chrono::duration<double>(t1 - t0).count() / 50;

    mpz_t ga, gm, gi; mpz_inits(ga, gm, gi, NULL);
    long wrong = 0;
    for (int i = 0; i < N; i++) {
        mpz_set_ui(ga, a[i]); mpz_set_ui(gm, m[i]);
        mpz_invert(gi, ga, gm);
        if (mpz_get_ui(gi) != out[i]) wrong++;
    }
    printf("%s: GPU batched modular inverse bit-exact vs GMP (%ld/%d wrong); %.0f Minv/s\n",
           wrong == 0 ? "PASS" : "FAIL", wrong, N, N / sec / 1e6);
    mpz_clears(ga, gm, gi, p, NULL);
    cudaFree(da); cudaFree(dm); cudaFree(dout);
    return wrong != 0;
}
