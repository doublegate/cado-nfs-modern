/*
 * gpu-mont128.cu — 128-bit (2-limb) Montgomery modular multiply for GPU ECM on
 * larger cofactors (mfb up to ~126 bits). De-risks the 2-word Montgomery REDC
 * (the hard part of 128-bit ECM) by validating it bit-exact, on both CPU and
 * GPU, against an obviously-correct binary mulmod reference.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-mont128.cu -o /tmp/gpu-mont128 && /tmp/gpu-mont128
 */
#include <cstdio>
#include <cstdint>
#include <vector>
typedef uint64_t u64;
typedef unsigned __int128 u128;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* -n^{-1} mod 2^64 of the low limb (Newton) */
static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }

/* 2-limb CIOS Montgomery multiply: returns a*b*R^{-1} mod n, R=2^128.
 * Requires n odd, n < 2^126, a,b < n. np = -n^{-1} mod 2^64 (low limb). */
HD static u128 montmul128(u128 a, u128 b, u128 n, u64 np)
{
    u64 a0=(u64)a, a1=(u64)(a>>64);
    u64 b0=(u64)b, b1=(u64)(b>>64);
    u64 n0=(u64)n, n1=(u64)(n>>64);
    u64 A[2]={a0,a1}, B[2]={b0,b1}, N[2]={n0,n1};
    u64 t[4]={0,0,0,0};
    for(int i=0;i<2;i++){
        /* t += A * B[i] */
        u128 c=0;
        for(int j=0;j<2;j++){ u128 x=(u128)t[j] + (u128)A[j]*B[i] + c; t[j]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[2]+c; t[2]=(u64)x; t[3]=(u64)(x>>64); }
        /* m = t[0]*np;  t += m*N;  t >>= 64 */
        u64 m=t[0]*np;
        u128 x0=(u128)t[0] + (u128)m*N[0]; c=x0>>64;        /* low limb cancels */
        { u128 x=(u128)t[1] + (u128)m*N[1] + c; t[0]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[2] + c; t[1]=(u64)x; c=x>>64; }
        t[2]=t[3]+(u64)c; t[3]=0;
    }
    u128 r=((u128)t[1]<<64)|t[0];
    if(r>=n) r-=n;
    return r;
}

/* ---- obviously-correct reference: binary mulmod (a*b mod n), n<2^127 ---- */
HD static u128 addmod128(u128 a,u128 b,u128 n){ u128 s=a+b; if(s<a||s>=n) s-=n; return s; }
HD static u128 mulmod128(u128 a,u128 b,u128 n){ u128 r=0; a%=n; while(b){ if(b&1) r=addmod128(r,a,n); a=addmod128(a,a,n); b>>=1; } return r; }

/* R^2 mod n = 2^256 mod n (R = 2^128), via the binary reference (host setup) */
static u128 R2mod(u128 n){ u128 r=1%n; for(int i=0;i<256;i++) r=addmod128(r,r,n); return r; }

struct Case{ u128 a,b,n; u64 np; u128 R2; };

__global__ void kern(const Case* c, u128* outMont, u128* outRef, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    Case k=c[i];
    /* validate: fromMont(montmul(toMont(a),toMont(b))) == a*b mod n */
    u128 am=montmul128(k.a,k.R2,k.n,k.np); /* a -> Montgomery */
    u128 bm=montmul128(k.b,k.R2,k.n,k.np);     /* b -> Montgomery */
    u128 pm=montmul128(am,bm,k.n,k.np);        /* (ab)R mod n */
    outMont[i]=montmul128(pm,1,k.n,k.np);      /* leave Montgomery -> ab mod n */
    outRef[i]=mulmod128(k.a,k.b,k.n);
}

static u64 rnd(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

int main(){
    const int N=200000; std::vector<Case> cs(N); u64 s=0xABCDEF12345ULL;
    for(int i=0;i<N;i++){
        u128 n=(((u128)rnd(&s)<<64)|rnd(&s));
        n >>= 2;                             /* < 2^126 */
        n |= 1;                              /* odd (Montgomery requires gcd(n,2)=1) */
        if(n<3) n=3;
        u128 a=(((u128)rnd(&s)<<64)|rnd(&s))%n;
        u128 b=(((u128)rnd(&s)<<64)|rnd(&s))%n;
        cs[i]={a,b,n,ninv64((u64)n),R2mod(n)};
    }
    /* CPU */
    long cmis=0;
    for(int i=0;i<N;i++){ Case k=cs[i]; u128 am=montmul128(k.a,k.R2,k.n,k.np),bm=montmul128(k.b,k.R2,k.n,k.np),pm=montmul128(am,bm,k.n,k.np),mo=montmul128(pm,1,k.n,k.np); if(mo!=mulmod128(k.a,k.b,k.n)) cmis++; }
    /* GPU */
    Case* dc; u128 *dm,*dr; cudaMalloc(&dc,N*sizeof(Case)); cudaMalloc(&dm,N*16); cudaMalloc(&dr,N*16);
    cudaMemcpy(dc,cs.data(),N*sizeof(Case),cudaMemcpyHostToDevice);
    kern<<<(N+127)/128,128>>>(dc,dm,dr,N); cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError();
    std::vector<u128> hm(N),hr(N); cudaMemcpy(hm.data(),dm,N*16,cudaMemcpyDeviceToHost); cudaMemcpy(hr.data(),dr,N*16,cudaMemcpyDeviceToHost);
    long gmis=0; for(int i=0;i<N;i++) if(hm[i]!=hr[i]) gmis++;
    printf("GPU status   : %s\n", cudaGetErrorString(e));
    printf("CPU montmul128 : %s (%ld/%d wrong vs binary mulmod)\n", cmis==0?"PASS":"FAIL", cmis, N);
    printf("GPU montmul128 : %s (%ld/%d wrong vs binary mulmod)\n", gmis==0?"PASS":"FAIL", gmis, N);
    return (cmis||gmis)!=0;
}
