/*
 * gpu-ecm.cu — batched ECM stage-1 on the GPU (Montgomery curves, XZ ladder),
 * for 64-bit moduli (the common cofactor size, mfb <= ~62). This is the core of
 * a GPU cofactorization backend for CADO-NFS (sieve/ecm): each GPU thread runs
 * one ECM curve on one cofactor, so a batch of survivors x curves = tens of
 * thousands of independent curves run in parallel.
 *
 * The same `ecm_stage1` runs on host and device (deterministic), so the GPU is
 * validated bit-exact against the CPU. We then (1) show it FINDS factors on
 * crafted composites and (2) report curves/sec on the RTX 3090.
 *
 * Build & run:
 *   nvcc -arch=sm_86 -O3 bench/gpu-ecm.cu -o /tmp/gpu-ecm && /tmp/gpu-ecm
 *
 * Curve setup uses a24 directly with x0=2 (valid XZ-ECM curve, random torsion).
 * Suyama sigma parametrization (better hit rate) is the production refinement;
 * see docs/gpu-cofactorization.md.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <chrono>

typedef uint64_t u64;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* ---- Montgomery arithmetic mod n (R = 2^64), correct for n < 2^63 ---- */
HD static inline u64 mm(u64 a, u64 b, u64 n, u64 np) {        /* a*b*R^{-1} mod n */
    unsigned __int128 T = (unsigned __int128)a * b;
    u64 m = (u64)T * np;
    unsigned __int128 s = T + (unsigned __int128)m * n;
    u64 t = (u64)(s >> 64);
    return t >= n ? t - n : t;
}
HD static inline u64 ad(u64 a, u64 b, u64 n) { u64 s = a + b; return s >= n ? s - n : s; }
HD static inline u64 sb(u64 a, u64 b, u64 n) { return a >= b ? a - b : a + n - b; }

HD static inline int ctz64(u64 x) {          /* count trailing zeros, x != 0 */
#ifdef __CUDA_ARCH__
    return __ffsll((long long)x) - 1;        /* device intrinsic */
#else
    return __builtin_ctzll(x);
#endif
}
HD static u64 bgcd(u64 a, u64 b) {           /* binary gcd */
    if (!a) return b; if (!b) return a;
    int sh = ctz64(a | b);
    a >>= ctz64(a);
    do { b >>= ctz64(b); if (a > b) { u64 t = a; a = b; b = t; } b -= a; } while (b);
    return a << sh;
}

/* ---- Montgomery curve XZ ops (all operands in Montgomery form) ---- */
struct PT { u64 X, Z; };

HD static inline PT dbl(PT p, u64 a24, u64 n, u64 np) {
    u64 A = mm(ad(p.X,p.Z,n), ad(p.X,p.Z,n), n, np);   /* (X+Z)^2 */
    u64 B = mm(sb(p.X,p.Z,n), sb(p.X,p.Z,n), n, np);   /* (X-Z)^2 */
    u64 C = sb(A, B, n);                               /* 4XZ */
    PT r; r.X = mm(A, B, n, np);
    r.Z = mm(C, ad(B, mm(a24, C, n, np), n), n, np);
    return r;
}
HD static inline PT dadd(PT p1, PT p2, PT pd, u64 n, u64 np) {   /* (p1+p2), diff pd */
    u64 DA = mm(sb(p1.X,p1.Z,n), ad(p2.X,p2.Z,n), n, np);
    u64 CB = mm(ad(p1.X,p1.Z,n), sb(p2.X,p2.Z,n), n, np);
    u64 s = ad(DA, CB, n), d = sb(DA, CB, n);
    PT r; r.X = mm(pd.Z, mm(s, s, n, np), n, np);
    r.Z = mm(pd.X, mm(d, d, n, np), n, np);
    return r;
}
/* Montgomery ladder: [k]P, k>=1 */
HD static PT ladder(PT P, u64 k, u64 a24, u64 n, u64 np) {
    if (k == 1) return P;
    PT R0 = P, R1 = dbl(P, a24, n, np);
    int b = 63; while (!((k >> b) & 1)) b--;            /* top set bit */
    for (b--; b >= 0; b--) {
        if ((k >> b) & 1) { R0 = dadd(R0, R1, P, n, np); R1 = dbl(R1, a24, n, np); }
        else              { R1 = dadd(R0, R1, P, n, np); R0 = dbl(R0, a24, n, np); }
    }
    return R0;
}

/* ECM stage 1: run curve `seed` on n; multipliers s[] = prime powers <= B1.
 * Returns gcd(Z, n) after Q = [prod s]P  (a nontrivial factor, or 1, or n). */
HD static u64 ecm_stage1(u64 n, u64 np, u64 R1, u64 R2, u64 seed,
                         const u64 *s, int ns) {
    u64 a24 = seed % n; if (a24 < 2) a24 = 2;            /* 2 <= a24 < n */
    u64 two = ad(R1, R1, n);                             /* 2 in Montgomery form */
    PT P; P.X = two; P.Z = R1;                           /* x0=2, z0=1 */
    u64 a24m = mm(a24, R2, n, np);                       /* a24 -> Montgomery */
    for (int i = 0; i < ns; i++) P = ladder(P, s[i], a24m, n, np);
    u64 z = mm(P.Z, 1, n, np);                           /* leave Montgomery form */
    return bgcd(z, n);
}

/* ---- batch kernel: one thread per (cofactor, curve) lane ---- */
__global__ void ecm_kernel(const u64 *n_, const u64 *np_, const u64 *R1_, const u64 *R2_,
                           const u64 *seed_, const u64 *s, int ns, u64 *g, int lanes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= lanes) return;
    g[i] = ecm_stage1(n_[i], np_[i], R1_[i], R2_[i], seed_[i], s, ns);
}

/* ---- host helpers ---- */
static u64 montinv(u64 n) { u64 x = n; for (int i=0;i<5;i++) x *= 2 - n*x; return (u64)0 - x; }
static u64 rmod(u64 n)  { return (u64)(((unsigned __int128)1 << 64) % n); }
static u64 r2mod(u64 n) { u64 r = rmod(n); return (u64)(((unsigned __int128)r * r) % n); }
static u64 rnd(u64 *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static bool isp(u64 n){ if(n<2)return false; for(u64 p=2;p*p<=n;p++) if(n%p==0) return false; return true; }
static u64 randprime(u64 lo, u64 hi, u64 *st){ for(;;){ u64 c=lo+rnd(st)%(hi-lo); c|=1; if(isp(c)) return c; } }

int main() {
    const u64 B1 = 2000;
    /* prime powers <= B1 (the stage-1 multiplier list) */
    std::vector<u64> s;
    std::vector<char> comp(B1 + 1, 0);
    for (u64 p = 2; p <= B1; p++) if (!comp[p]) {
        for (u64 q = p*p; q <= B1; q += p) comp[q] = 1;
        u64 pe = p; while (pe * p <= B1) pe *= p;        /* largest p^e <= B1 */
        s.push_back(pe);
    }
    int ns = (int)s.size();

    /* test composites n = p*q, p ~20-bit (findable at B1=2000), q ~30-bit */
    const int NCOMP = 256, CURVES = 256;
    const int LANES = NCOMP * CURVES;
    std::vector<u64> n(LANES), np(LANES), R1(LANES), R2(LANES), seed(LANES);
    std::vector<u64> pfac(NCOMP);
    u64 st = 0xC0FFEEULL;
    for (int c = 0; c < NCOMP; c++) {
        u64 p = randprime(1u<<19, 1u<<20, &st), q = randprime(1u<<29, 1u<<30, &st);
        u64 N = p * q; pfac[c] = p;                      /* N < 2^50, safe for 64-bit mont */
        for (int j = 0; j < CURVES; j++) {
            int i = c*CURVES + j;
            n[i]=N; np[i]=montinv(N); R1[i]=rmod(N); R2[i]=r2mod(N);
            seed[i] = rnd(&st) | 2;
        }
    }

    /* ---- GPU ---- */
    u64 *dn,*dnp,*dR1,*dR2,*ds,*dseed,*dg;
    cudaMalloc(&dn,LANES*8);cudaMalloc(&dnp,LANES*8);cudaMalloc(&dR1,LANES*8);
    cudaMalloc(&dR2,LANES*8);cudaMalloc(&dseed,LANES*8);cudaMalloc(&dg,LANES*8);cudaMalloc(&ds,ns*8);
    cudaMemcpy(dn,n.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dnp,np.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,R1.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,R2.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dseed,seed.data(),LANES*8,cudaMemcpyHostToDevice);
    cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);
    int tpb=128, blk=(LANES+tpb-1)/tpb;
    ecm_kernel<<<blk,tpb>>>(dn,dnp,dR1,dR2,dseed,ds,ns,dg,LANES); cudaDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now();
    ecm_kernel<<<blk,tpb>>>(dn,dnp,dR1,dR2,dseed,ds,ns,dg,LANES); cudaDeviceSynchronize();
    auto t1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    std::vector<u64> g(LANES); cudaMemcpy(g.data(),dg,LANES*8,cudaMemcpyDeviceToHost);
    double gsec=std::chrono::duration<double>(t1-t0).count();

    /* ---- CPU reference on a subset, validate bit-exact ---- */
    int CPU=LANES/32; long mism=0;
    for (int i=0;i<CPU;i++){ u64 gg=ecm_stage1(n[i],np[i],R1[i],R2[i],seed[i],s.data(),ns); if(gg!=g[i]) mism++; }

    /* ---- factor-finding: how many composites got cracked by >=1 curve ---- */
    int cracked=0;
    for (int c=0;c<NCOMP;c++){ bool ok=false; for(int j=0;j<CURVES;j++){ u64 gg=g[c*CURVES+j]; if(gg>1&&gg<n[c*CURVES]){ if(gg==pfac[c]) ok=true; } } cracked+=ok; }

    printf("GPU status   : %s\n", cudaGetErrorString(e));
    printf("validation   : %s (%ld/%d GPU lanes differ from CPU)\n", mism==0?"PASS":"FAIL", mism, CPU);
    printf("B1=%llu, %d prime-power multipliers, %d composites x %d curves = %d lanes\n",
           (unsigned long long)B1, ns, NCOMP, CURVES, LANES);
    printf("factors found: %d/%d composites cracked by >=1 curve\n", cracked, NCOMP);
    printf("throughput   : %.0f curves/s  (%d curves in %.4fs on RTX 3090)\n", LANES/gsec, LANES, gsec);
    return mism!=0;
}
