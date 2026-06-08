/*
 * gpu_pm1_pp1.cuh — Pollard P-1 and Williams P+1 on the GPU (v3.4.0 Track C7).
 *
 * Adds two classical factoring methods to the GPU pre-NFS factoring front-end,
 * BESIDE the batched ECM (gpu_ecm_mp.cuh). All three share the same bit-exact-
 * validated K-limb Montgomery arithmetic (montmul/addmod/submod from
 * gpu_ecm_mp.cuh), so P-1/P+1 inherit the fork's standing correctness gate.
 *
 *   - P-1 (Pollard 1974): finds a prime p | N when p-1 is B1-smooth (stage 1) or
 *     B1-smooth times one prime in (B1,B2] (stage 2). Stage 1 is a single
 *     Montgomery exponentiation a = base^E (E = lcm of prime powers <= B1); the
 *     factor is gcd(a-1, N). Stage 2 is a baby-step/giant-step continuation in the
 *     multiplicative group, identical in shape to the ECM stage-2 BSGS but with a
 *     scalar montmul where ECM uses a curve addition.
 *
 *   - P+1 (Williams 1982): finds p when p+1 is smooth, using Lucas sequences
 *     V_n (Chebyshev): V_0=2, V_1=seed, V_{2n}=V_n^2-2, V_{m+n}=V_m V_n - V_{m-n}.
 *     Stage 1 evaluates V_E(seed) by a Lucas ladder (the additive analogue of the
 *     Montgomery ladder); the factor is gcd(V_E - 2, N). Stage 2 is the Lucas BSGS.
 *
 * HONEST SCOPE (see docs/gpu-prefactor-pm1pp1-c7.md): on a SINGLE N each P-1/P+1
 * run is one sequence == one GPU lane, so it does NOT benefit from the GPU's
 * thousands-of-curves parallelism the way ECM does. Its value is *coverage* (it
 * catches p-/+1-smooth factors the ECM curve count can miss) and *cheap time-to-
 * strip* (one exponentiation each, run before the ECM batch under the adaptive
 * escalating-B1 schedule). The throughput win of the front-end remains ECM's.
 *
 * Same __host__ __device__ code runs on CPU and GPU, so the device math is
 * validated bit-exact (bench/gpu-prefactor-pm1pp1.cu).
 */
#ifndef CADO_GPU_PM1_PP1_CUH
#define CADO_GPU_PM1_PP1_CUH

#include "gpu_ecm_mp.cuh"

#define GPU_PM1_W 32            /* BSGS baby-step table size (bounds local memory) */

/* Montgomery exponentiation: rM = baseM^e (Montgomery in, Montgomery out).
 * oneM is R mod n (the Montgomery representation of 1). */
template<int K> HD void montpow(u64 *rM, const u64 *baseM, u64 e,
                                const u64 *n, u64 np, const u64 *oneM){
    u64 acc[K]; mp_copy<K>(acc, oneM);
    u64 b[K];   mp_copy<K>(b, baseM);
    while(e){
        if(e & 1ULL){ u64 t[K]; montmul<K>(t, acc, b, n, np); mp_copy<K>(acc, t); }
        e >>= 1;
        if(e){ u64 t[K]; montmul<K>(t, b, b, n, np); mp_copy<K>(b, t); }
    }
    mp_copy<K>(rM, acc);
}

/* ---- Pollard P-1 ----
 * base is a plain integer mod n (per lane). Returns:
 *   g1out = base^E mod n            (plain)  -> host gcds (g1out - 1) with n
 *   g2out = stage-2 cross-product   (plain)  -> host gcds g2out with n
 * s[] are the prime powers <= B1; pr[] the primes in (B1,B2]. */
template<int K> HD void pm1_run(u64 *g1out, u64 *g2out,
        const u64 *n, u64 np, const u64 *R1, const u64 *R2,
        const u64 *base, const u64 *s, int ns, const u64 *pr, int npr){
    u64 oneM[K]; mp_copy<K>(oneM, R1);                 /* 1 in Montgomery */
    u64 oneP[K]; mp_set0<K>(oneP); oneP[0]=1;          /* 1 plain (Mont->plain via montmul) */
    u64 Q[K]; montmul<K>(Q, base, R2, n, np);          /* base -> Montgomery */
    /* stage 1: Q = base^E, E = prod s[i]  (V_{ab}=... ; here plain composition of powers) */
    for(int i=0;i<ns;i++){ u64 t[K]; montpow<K>(t, Q, s[i], n, np, oneM); mp_copy<K>(Q, t); }
    montmul<K>(g1out, Q, oneP, n, np);                 /* base^E (plain) */
    /* stage 2 BSGS over pr[] (multiplicative group) */
    if(npr<=0){ mp_copy<K>(g2out, oneP); return; }     /* nothing to do; gcd(1,n)=1 */
    u64 T[GPU_PM1_W][K];
    mp_copy<K>(T[1], Q);
    for(int r=2;r<GPU_PM1_W;r++) montmul<K>(T[r], T[r-1], Q, n, np);   /* T[r] = Q^r (Mont) */
    u64 QW[K]; montpow<K>(QW, Q, (u64)GPU_PM1_W, n, np, oneM);         /* Q^W */
    int m=(int)((pr[0]+GPU_PM1_W-1)/GPU_PM1_W);
    u64 V[K]; montpow<K>(V, Q, (u64)m*GPU_PM1_W, n, np, oneM);         /* Q^{mW} */
    u64 g[K]; mp_copy<K>(g, oneM);                                     /* accumulator = 1 (Mont) */
    for(int k=0;k<npr;k++){
        u64 p=pr[k]; int mp=(int)((p+GPU_PM1_W-1)/GPU_PM1_W);
        while(m<mp){ u64 t[K]; montmul<K>(t, V, QW, n, np); mp_copy<K>(V, t); m++; }
        int r=m*GPU_PM1_W-(int)p;
        if(r<=0||r>=GPU_PM1_W) continue;
        u64 diff[K]; submod<K>(diff, V, T[r], n);                      /* Q^{mW} - Q^r */
        u64 t[K]; montmul<K>(t, g, diff, n, np); mp_copy<K>(g, t);
    }
    montmul<K>(g2out, g, oneP, n, np);
}

/* Lucas chain: outM = V_k(seedM) in Montgomery. V_0=2, V_1=seed.
 * Ladder over the bits of k maintaining (V_m, V_{m+1}); twoM is 2 in Montgomery. */
template<int K> HD void lucas_chain(u64 *outM, const u64 *seedM, u64 k,
        const u64 *n, u64 np, const u64 *twoM){
    if(k==0){ mp_copy<K>(outM, twoM); return; }
    if(k==1){ mp_copy<K>(outM, seedM); return; }
    int b=63; while(!((k>>b)&1ULL)) b--;
    u64 Vm[K], Vm1[K];
    mp_copy<K>(Vm, seedM);                                             /* V_1 */
    { u64 s2[K]; montmul<K>(s2, seedM, seedM, n, np); submod<K>(Vm1, s2, twoM, n); } /* V_2 = V_1^2 - 2 */
    for(b--; b>=0; b--){
        if((k>>b)&1ULL){
            /* (V_{2m+1}, V_{2m+2}) = (V_m V_{m+1} - V_1, V_{m+1}^2 - 2) */
            u64 pr2[K]; montmul<K>(pr2, Vm, Vm1, n, np); submod<K>(pr2, pr2, seedM, n);
            u64 sq[K];  montmul<K>(sq, Vm1, Vm1, n, np); submod<K>(sq, sq, twoM, n);
            mp_copy<K>(Vm, pr2); mp_copy<K>(Vm1, sq);
        } else {
            /* (V_{2m}, V_{2m+1}) = (V_m^2 - 2, V_m V_{m+1} - V_1) */
            u64 sq[K];  montmul<K>(sq, Vm, Vm, n, np); submod<K>(sq, sq, twoM, n);
            u64 pr2[K]; montmul<K>(pr2, Vm, Vm1, n, np); submod<K>(pr2, pr2, seedM, n);
            mp_copy<K>(Vm, sq); mp_copy<K>(Vm1, pr2);
        }
    }
    mp_copy<K>(outM, Vm);
}

/* ---- Williams P+1 ----
 * seed is a plain integer mod n (the Lucas V_1, per lane). Returns:
 *   g1out = (V_E(seed) - 2) mod n  (plain)  -> host gcds g1out with n
 *   g2out = stage-2 cross-product  (plain)  -> host gcds g2out with n  */
template<int K> HD void pp1_run(u64 *g1out, u64 *g2out,
        const u64 *n, u64 np, const u64 *R1, const u64 *R2,
        const u64 *seed, const u64 *s, int ns, const u64 *pr, int npr){
    u64 oneP[K]; mp_set0<K>(oneP); oneP[0]=1;
    u64 twoM[K]; addmod<K>(twoM, R1, R1, n);           /* 2 in Montgomery (1+1) */
    u64 Q[K]; montmul<K>(Q, seed, R2, n, np);          /* seed -> Montgomery (= V_1) */
    /* stage 1: Q = V_E(seed) by Chebyshev composition V_{ab}(x) = V_a(V_b(x)) */
    for(int i=0;i<ns;i++){ u64 t[K]; lucas_chain<K>(t, Q, s[i], n, np, twoM); mp_copy<K>(Q, t); }
    { u64 d[K]; submod<K>(d, Q, twoM, n); montmul<K>(g1out, d, oneP, n, np); }  /* V_E - 2 (plain) */
    /* stage 2 BSGS (Lucas): accumulate (V_{mW} - V_r) over primes p = mW - r */
    if(npr<=0){ mp_copy<K>(g2out, oneP); return; }
    u64 T[GPU_PM1_W][K];
    mp_copy<K>(T[0], twoM);                            /* V_0 = 2 */
    mp_copy<K>(T[1], Q);                               /* V_1 = Q */
    for(int r=2;r<GPU_PM1_W;r++){                      /* V_r = V_1 V_{r-1} - V_{r-2} */
        u64 t[K]; montmul<K>(t, Q, T[r-1], n, np); submod<K>(T[r], t, T[r-2], n); }
    u64 VW[K]; lucas_chain<K>(VW, Q, (u64)GPU_PM1_W, n, np, twoM);     /* V_W */
    int m=(int)((pr[0]+GPU_PM1_W-1)/GPU_PM1_W);
    u64 V[K], Vp[K];
    lucas_chain<K>(V,  Q, (u64)m*GPU_PM1_W,     n, np, twoM);          /* V_{mW} */
    lucas_chain<K>(Vp, Q, (u64)(m-1)*GPU_PM1_W, n, np, twoM);          /* V_{(m-1)W} */
    u64 g[K]; mp_copy<K>(g, R1);                                       /* accumulator = 1 (Mont) */
    for(int k=0;k<npr;k++){
        u64 p=pr[k]; int mp=(int)((p+GPU_PM1_W-1)/GPU_PM1_W);
        while(m<mp){                                                  /* V_{(m+1)W} = V_W V_{mW} - V_{(m-1)W} */
            u64 t[K]; montmul<K>(t, VW, V, n, np); submod<K>(t, t, Vp, n);
            mp_copy<K>(Vp, V); mp_copy<K>(V, t); m++; }
        int r=m*GPU_PM1_W-(int)p;
        if(r<=0||r>=GPU_PM1_W) continue;
        u64 diff[K]; submod<K>(diff, V, T[r], n);
        u64 t[K]; montmul<K>(t, g, diff, n, np); mp_copy<K>(g, t);
    }
    montmul<K>(g2out, g, oneP, n, np);
}

#ifdef __CUDACC__
template<int K> __global__ void pm1_kernel(const u64*N,const u64*NP,const u64*R1,
        const u64*R2,const u64*base,const u64*s,int ns,const u64*pr,int npr,
        u64*G1,u64*G2,int lanes){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    pm1_run<K>(G1+i*K,G2+i*K,N+i*K,NP[i],R1+i*K,R2+i*K,base+i*K,s,ns,pr,npr);
}
template<int K> __global__ void pp1_kernel(const u64*N,const u64*NP,const u64*R1,
        const u64*R2,const u64*seed,const u64*s,int ns,const u64*pr,int npr,
        u64*G1,u64*G2,int lanes){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    pp1_run<K>(G1+i*K,G2+i*K,N+i*K,NP[i],R1+i*K,R2+i*K,seed+i*K,s,ns,pr,npr);
}
#endif

#endif /* CADO_GPU_PM1_PP1_CUH */
