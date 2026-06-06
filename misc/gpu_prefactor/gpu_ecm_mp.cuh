/*
 * gpu_ecm_mp.cuh — multi-precision (K-limb) Montgomery arithmetic + ECM stage-1
 * for the GPU, shared by the standalone pre-factoring tool and its validation.
 *
 * The math is the bit-exact-validated 2-limb CIOS Montgomery (bench/gpu-mont128.cu)
 * and Montgomery-curve XZ ladder (bench/gpu-ecm.cu) generalized to K 64-bit limbs;
 * see bench/gpu-ecm-mp.cu for the bit-exact GPU-vs-CPU and GPU-vs-binary-mulmod
 * validation. Requires odd modulus n < 2^{64K-2} (pick K = ceil((bits+2)/64)).
 */
#ifndef CADO_GPU_ECM_MP_CUH
#define CADO_GPU_ECM_MP_CUH

#include <cstdint>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

template<int K> HD void mp_set0(u64 *r){ for(int i=0;i<K;i++) r[i]=0; }
template<int K> HD void mp_copy(u64 *r,const u64 *a){ for(int i=0;i<K;i++) r[i]=a[i]; }
template<int K> HD bool mp_geq(const u64 *a,const u64 *b){
    for(int i=K-1;i>=0;i--){ if(a[i]!=b[i]) return a[i]>b[i]; } return true; }
template<int K> HD void mp_sub(u64 *r,const u64 *a,const u64 *b){
    u128 br=0; for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; r[i]=(u64)x; br=(x>>64)&1; } }
template<int K> HD void addmod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 c=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]+(u128)b[i]+c; t[i]=(u64)x; c=x>>64; }
    if(c || mp_geq<K>(t,n)) mp_sub<K>(r,t,n); else mp_copy<K>(r,t); }
template<int K> HD void submod(u64 *r,const u64 *a,const u64 *b,const u64 *n){
    u64 t[K]; u128 br=0;
    for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; t[i]=(u64)x; br=(x>>64)&1; }
    if(br){ u128 c=0; for(int i=0;i<K;i++){ u128 x=(u128)t[i]+(u128)n[i]+c; r[i]=(u64)x; c=x>>64; } }
    else mp_copy<K>(r,t); }
template<int K> HD void montmul(u64 *r,const u64 *A,const u64 *B,const u64 *N,u64 np){
    u64 t[K+2]; for(int i=0;i<K+2;i++) t[i]=0;
    for(int i=0;i<K;i++){
        u128 c=0;
        for(int j=0;j<K;j++){ u128 x=(u128)t[j]+(u128)A[j]*B[i]+c; t[j]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K]=(u64)x; t[K+1]=(u64)(x>>64); }
        u64 m=t[0]*np;
        { u128 x=(u128)t[0]+(u128)m*N[0]; c=x>>64; }
        for(int j=1;j<K;j++){ u128 x=(u128)t[j]+(u128)m*N[j]+c; t[j-1]=(u64)x; c=x>>64; }
        { u128 x=(u128)t[K]+c; t[K-1]=(u64)x; c=x>>64; }
        t[K]=t[K+1]+(u64)c; t[K+1]=0;
    }
    if(mp_geq<K>(t,N)) mp_sub<K>(r,t,N); else mp_copy<K>(r,t);
}

template<int K> struct PT { u64 X[K], Z[K]; };
template<int K> HD void cdbl(PT<K>&r,const PT<K>&p,const u64*a24,const u64*n,u64 np){
    u64 apz[K],amz[K],A[K],B[K],C[K],t[K];
    addmod<K>(apz,p.X,p.Z,n); montmul<K>(A,apz,apz,n,np);
    submod<K>(amz,p.X,p.Z,n); montmul<K>(B,amz,amz,n,np);
    submod<K>(C,A,B,n); montmul<K>(r.X,A,B,n,np);
    montmul<K>(t,a24,C,n,np); addmod<K>(t,B,t,n); montmul<K>(r.Z,C,t,n,np);
}
template<int K> HD void cadd(PT<K>&r,const PT<K>&p1,const PT<K>&p2,const PT<K>&pd,
                             const u64*n,u64 np){
    u64 p1pz[K],p1mz[K],p2pz[K],p2mz[K],DA[K],CB[K],s[K],d[K],ss[K],dd[K];
    submod<K>(p1mz,p1.X,p1.Z,n); addmod<K>(p2pz,p2.X,p2.Z,n);
    addmod<K>(p1pz,p1.X,p1.Z,n); submod<K>(p2mz,p2.X,p2.Z,n);
    montmul<K>(DA,p1mz,p2pz,n,np); montmul<K>(CB,p1pz,p2mz,n,np);
    addmod<K>(s,DA,CB,n); submod<K>(d,DA,CB,n);
    montmul<K>(ss,s,s,n,np); montmul<K>(dd,d,d,n,np);
    montmul<K>(r.X,pd.Z,ss,n,np); montmul<K>(r.Z,pd.X,dd,n,np);
}
template<int K> HD void ladder(PT<K>&out,const PT<K>&P,u64 k,const u64*a24,
                               const u64*n,u64 np){
    if(k==1){ out=P; return; }
    PT<K> R0=P, R1; cdbl<K>(R1,P,a24,n,np);
    int b=63; while(!((k>>b)&1)) b--;
    for(b--;b>=0;b--){
        if((k>>b)&1){ PT<K> t; cadd<K>(t,R0,R1,P,n,np); R0=t; cdbl<K>(R1,R1,a24,n,np); }
        else        { PT<K> t; cadd<K>(t,R1,R0,P,n,np); R1=t; cdbl<K>(R0,R0,a24,n,np); }
    }
    out=R0;
}
/* ECM stage 1; returns leave-Montgomery Z of Q=[prod s]P (host gcds with n). */
template<int K> HD void ecm_stage1(u64 *zout,const u64 *n,u64 np,
                                   const u64 *R1,const u64 *R2,
                                   const u64 *a24,const u64 *s,int ns){
    PT<K> P;
    addmod<K>(P.X,R1,R1,n);          /* x0 = 2 */
    mp_copy<K>(P.Z,R1);              /* z0 = 1 */
    u64 a24m[K]; montmul<K>(a24m,a24,R2,n,np);
    for(int i=0;i<ns;i++){ PT<K> t; ladder<K>(t,P,s[i],a24m,n,np); P=t; }
    u64 one[K]; mp_set0<K>(one); one[0]=1;
    montmul<K>(zout,P.Z,one,n,np);
}

#ifdef __CUDACC__
template<int K> __global__ void ecm_kernel(const u64*N,const u64*NP,const u64*R1,
                const u64*R2,const u64*SEED,const u64*s,int ns,u64*Z,int lanes){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    ecm_stage1<K>(Z+i*K,N+i*K,NP[i],R1+i*K,R2+i*K,SEED+i*K,s,ns);
}
#endif

#endif /* CADO_GPU_ECM_MP_CUH */
