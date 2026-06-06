/*
 * gpu-prefactor-bench.cu — honest CPU-vs-GPU throughput for the multi-precision
 * ECM that powers the GPU pre-factoring front-end (misc/gpu_prefactor). The
 * SAME __host__ __device__ ecm_run2 (stage-1 + stage-2 BSGS) runs on both sides
 * — the CPU side parallelized across all cores with std::thread — so the
 * comparison is apples-to-apples (identical algorithm, identical code), unlike
 * comparing against a different CPU ECM implementation.
 *
 *   nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-prefactor-bench.cu -lgmp \
 *        -o /tmp/gpu-prefactor-bench && /tmp/gpu-prefactor-bench
 *
 * Reports curves/s for the GPU (RTX 3090) and the full CPU (all cores) at a few
 * widths, plus the speedup, at B1=50000 B2=5e6 (a representative pre-factoring
 * stage).
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>
#include <gmp.h>
#include "../misc/gpu_prefactor/gpu_ecm_mp.cuh"

static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static u64 xrnd(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static void to_limbs(u64*o,int K,const mpz_t v){ for(int i=0;i<K;i++)o[i]=0; size_t c=0; mpz_export(o,&c,-1,8,0,0,v); }

template<int K>
static void bench(const char*label,unsigned long B1,unsigned long B2,
                  int gpu_curves,int cpu_curves){
    /* random odd modulus of this width with 2-bit headroom */
    u64 st=0xBEEF1234ULL+K; u64 Nl[K];
    for(int i=0;i<K;i++) Nl[i]=xrnd(&st);
    Nl[K-1]>>=2; Nl[0]|=1;
    mpz_t N,t,R1m,R2m; mpz_inits(N,t,R1m,R2m,NULL);
    mpz_import(N,K,-1,8,0,0,Nl);
    mpz_setbit(t,64*K); mpz_mod(R1m,t,N);
    mpz_set_ui(t,0); mpz_setbit(t,128*K); mpz_mod(R2m,t,N);
    u64 R1[K],R2[K]; to_limbs(R1,K,R1m); to_limbs(R2,K,R2m); u64 np=ninv64(Nl[0]);

    /* sieve: stage-1 prime powers <= B1, stage-2 primes in (B1,B2] */
    std::vector<u64> spow,pr; std::vector<char> comp(B2+1,0);
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){ for(unsigned long q=p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ u64 pe=p; while(pe*p<=B1) pe*=p; spow.push_back(pe); } else pr.push_back(p); }
    int ns=(int)spow.size(), npr=(int)pr.size();

    /* POC curves: X0=2, Z0=1, a24=seed (plain; the ladder work is the same as Suyama) */
    auto fillcurve=[&](u64*X0,u64*Z0,u64*A24,int i){
        mp_set0<K>(X0); X0[0]=2; mp_set0<K>(Z0); Z0[0]=1; mp_set0<K>(A24); A24[0]=(u64)(7+i*2); };

    /* ---- GPU ---- */
    std::vector<u64> gX0(gpu_curves*K),gZ0(gpu_curves*K),gA24(gpu_curves*K);
    std::vector<u64> gN(gpu_curves*K),gR1(gpu_curves*K),gR2(gpu_curves*K),gNP(gpu_curves);
    for(int i=0;i<gpu_curves;i++){ fillcurve(&gX0[i*K],&gZ0[i*K],&gA24[i*K],i);
        mp_copy<K>(&gN[i*K],Nl); mp_copy<K>(&gR1[i*K],R1); mp_copy<K>(&gR2[i*K],R2); gNP[i]=np; }
    u64 *dN,*dNP,*dR1,*dR2,*dX0,*dZ0,*dA24,*ds,*dpr,*dZ1,*dG2;
    size_t cb=(size_t)gpu_curves*K*8;
    cudaMalloc(&dN,cb);cudaMalloc(&dNP,gpu_curves*8);cudaMalloc(&dR1,cb);cudaMalloc(&dR2,cb);
    cudaMalloc(&dX0,cb);cudaMalloc(&dZ0,cb);cudaMalloc(&dA24,cb);cudaMalloc(&dZ1,cb);cudaMalloc(&dG2,cb);
    cudaMalloc(&ds,ns*8);cudaMalloc(&dpr,npr*8);
    cudaMemcpy(dN,gN.data(),cb,cudaMemcpyHostToDevice);cudaMemcpy(dNP,gNP.data(),gpu_curves*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,gR1.data(),cb,cudaMemcpyHostToDevice);cudaMemcpy(dR2,gR2.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(dX0,gX0.data(),cb,cudaMemcpyHostToDevice);cudaMemcpy(dZ0,gZ0.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(dA24,gA24.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(ds,spow.data(),ns*8,cudaMemcpyHostToDevice);cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);
    int tpb=64, blk=(gpu_curves+tpb-1)/tpb;
    ecm_kernel2<K><<<blk,tpb>>>(dN,dNP,dR1,dR2,dX0,dZ0,dA24,ds,ns,dpr,npr,dZ1,dG2,gpu_curves); /* warm */
    cudaDeviceSynchronize();
    auto g0=std::chrono::steady_clock::now();
    ecm_kernel2<K><<<blk,tpb>>>(dN,dNP,dR1,dR2,dX0,dZ0,dA24,ds,ns,dpr,npr,dZ1,dG2,gpu_curves);
    cudaDeviceSynchronize();
    auto g1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    double gsec=std::chrono::duration<double>(g1-g0).count(), gcps=gpu_curves/gsec;
    cudaFree(dN);cudaFree(dNP);cudaFree(dR1);cudaFree(dR2);cudaFree(dX0);cudaFree(dZ0);
    cudaFree(dA24);cudaFree(dZ1);cudaFree(dG2);cudaFree(ds);cudaFree(dpr);

    /* ---- CPU, all cores ---- */
    int nthr=(int)std::thread::hardware_concurrency(); if(nthr<1) nthr=1;
    std::vector<u64> sink(nthr,0);   /* observable sink: prevents dead-code elimination */
    auto c0=std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for(int t=0;t<nthr;t++) th.emplace_back([&,t]{
        int lo=t*cpu_curves/nthr, hi=(t+1)*cpu_curves/nthr; u64 acc=0;
        for(int i=lo;i<hi;i++){ u64 X0[K],Z0[K],A24[K],z1[K],g2[K];
            fillcurve(X0,Z0,A24,i);
            ecm_run2<K>(z1,g2,Nl,np,R1,R2,X0,Z0,A24,spow.data(),ns,pr.data(),npr);
            acc ^= z1[0]^g2[0]; }
        sink[t]=acc;
    });
    for(auto&x:th) x.join();
    auto c1=std::chrono::steady_clock::now();
    double csec=std::chrono::duration<double>(c1-c0).count(), ccps=cpu_curves/csec;
    u64 chk=0; for(u64 v:sink) chk^=v;

    printf("  %-9s GPU %8.0f curves/s | CPU(%2d thr) %8.0f curves/s | speedup %5.1fx%s  [chk %016llx]\n",
           label, gcps, nthr, ccps, ccps>0?gcps/ccps:0, e?"  CUDAERR":"",
           (unsigned long long)chk);
    mpz_clears(N,t,R1m,R2m,NULL);
}

int main(){
    printf("CPU-vs-GPU throughput for the pre-factoring ECM (same ecm_run2; stage1+stage2)\n");
    printf("B1=50000 B2=5000000, RTX 3090 vs full CPU\n");
    bench<2>("128-bit", 50000,5000000, 16384, 512);
    bench<4>("256-bit", 50000,5000000,  8192, 256);
    bench<8>("512-bit", 50000,5000000,  4096, 128);
    return 0;
}
