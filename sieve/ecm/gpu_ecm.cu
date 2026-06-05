/* CUDA implementation of the batched GPU ECM cofactorization backend declared
 * in gpu_ecm.hpp. The device ECM (Montgomery-curve XZ ladder, stage 1 + stage-2
 * BSGS, on-device binary gcd) is the bit-exact-validated code from
 * bench/gpu-ecm-stage2.cu, packaged as a library callable from facul_all().
 *
 * Scope: odd moduli < 2^62 (one word; the common cofactor size). Larger moduli
 * are skipped (factor[i]=0) and left to the CPU path; a 128-bit variant
 * (bench/gpu-mont128.cu, validated) is the documented extension.
 */
#include "gpu_ecm.hpp"
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

typedef uint64_t u64;
#define WBS 60                                 /* stage-2 baby-step table size */

__host__ __device__ static inline u64 mm(u64 a,u64 b,u64 n,u64 np){ unsigned __int128 T=(unsigned __int128)a*b; u64 m=(u64)T*np; unsigned __int128 s=T+(unsigned __int128)m*n; u64 t=(u64)(s>>64); return t>=n?t-n:t; }
__host__ __device__ static inline u64 ad(u64 a,u64 b,u64 n){ u64 s=a+b; return s>=n?s-n:s; }
__host__ __device__ static inline u64 sb(u64 a,u64 b,u64 n){ return a>=b?a-b:a+n-b; }
__device__ static inline int ctz64(u64 x){ return __ffsll((long long)x)-1; }
__device__ static u64 bgcd(u64 a,u64 b){ if(!a)return b; if(!b)return a; int sh=ctz64(a|b); a>>=ctz64(a); do{ b>>=ctz64(b); if(a>b){u64 t=a;a=b;b=t;} b-=a; }while(b); return a<<sh; }

struct PT{ u64 X,Z; };
__host__ __device__ static inline PT dbl(PT p,u64 a,u64 n,u64 np){ u64 A=mm(ad(p.X,p.Z,n),ad(p.X,p.Z,n),n,np),B=mm(sb(p.X,p.Z,n),sb(p.X,p.Z,n),n,np),C=sb(A,B,n); PT r; r.X=mm(A,B,n,np); r.Z=mm(C,ad(B,mm(a,C,n,np),n),n,np); return r; }
__host__ __device__ static inline PT dadd(PT p1,PT p2,PT pd,u64 n,u64 np){ u64 DA=mm(sb(p1.X,p1.Z,n),ad(p2.X,p2.Z,n),n,np),CB=mm(ad(p1.X,p1.Z,n),sb(p2.X,p2.Z,n),n,np),s=ad(DA,CB,n),d=sb(DA,CB,n); PT r; r.X=mm(pd.Z,mm(s,s,n,np),n,np); r.Z=mm(pd.X,mm(d,d,n,np),n,np); return r; }
__device__ static PT ladder(PT P,u64 k,u64 a,u64 n,u64 np){ if(k==1)return P; PT R0=P,R1=dbl(P,a,n,np); int b=63; while(!((k>>b)&1))b--; for(b--;b>=0;b--){ if((k>>b)&1){R0=dadd(R0,R1,P,n,np);R1=dbl(R1,a,n,np);} else {R1=dadd(R0,R1,P,n,np);R0=dbl(R0,a,n,np);} } return R0; }

__global__ void ecm_kernel(const u64* n_,const u64* np_,const u64* R1_,const u64* R2_,const u64* seed_,
                           const u64* s,int ns,const u64* pr,int npr,u64* fac,int lanes,int ncurves)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=lanes) return;
    u64 n=n_[i]; if(n==0){ fac[i]=0; return; }
    u64 np=np_[i],R1=R1_[i],R2=R2_[i];
    u64 a24=seed_[i]%n; if(a24<2)a24=2;
    PT P; P.X=ad(R1,R1,n); P.Z=R1;             /* x0=2 */
    u64 a=mm(a24,R2,n,np);
    for(int t=0;t<ns;t++) P=ladder(P,s[t],a,n,np);   /* stage 1 */
    PT Q=P;
    u64 g=bgcd(mm(Q.Z,1,n,np),n);
    if(g==1 && npr>0){                          /* stage 2 BSGS */
        PT T[WBS]; T[1]=Q; T[2]=dbl(Q,a,n,np);
        for(int r=3;r<WBS;r++) T[r]=dadd(T[r-1],Q,T[r-2],n,np);
        PT Wg=ladder(Q,WBS,a,n,np);
        int m=(int)((pr[0]+WBS-1)/WBS);
        PT V=ladder(Q,(u64)m*WBS,a,n,np), Vp=ladder(Q,(u64)(m-1)*WBS,a,n,np);
        u64 acc=R1;
        for(int k=0;k<npr;k++){ u64 p=pr[k]; int mp=(int)((p+WBS-1)/WBS);
            while(m<mp){ PT Vn=dadd(V,Wg,Vp,n,np); Vp=V; V=Vn; m++; }
            int r=m*WBS-(int)p; if(r<=0||r>=WBS) continue;
            acc=mm(acc, sb(mm(V.X,T[r].Z,n,np), mm(T[r].X,V.Z,n,np), n), n, np);
        }
        g=bgcd(mm(acc,1,n,np),n);
    }
    fac[i]=(g>1 && g<n)?g:0;
    (void)ncurves;
}

/* host helpers */
static u64 montinv(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static u64 rmod(u64 n){ return (u64)(((unsigned __int128)1<<64)%n); }
static u64 r2mod(u64 n){ u64 r=rmod(n); return (u64)(((unsigned __int128)r*r)%n); }

namespace gpu_ecm {

bool available()
{
    int dev=0; cudaError_t e=cudaGetDeviceCount(&dev);
    return e==cudaSuccess && dev>0;
}

void factor_batch(std::vector<uint64_t> const & moduli, int ncurves,
                  unsigned long B1, unsigned long B2, std::vector<uint64_t> & factor)
{
    int M=(int)moduli.size();
    factor.assign(M,0);
    if(M==0 || !available()) return;

    /* stage-1 prime powers <= B1, and primes in (B1,B2] for stage 2 */
    std::vector<char> comp(B2+1,0); std::vector<u64> s,pr;
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){ for(unsigned long q=(unsigned long)p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ unsigned long pe=p; while(pe*p<=B1) pe*=p; s.push_back(pe); } else pr.push_back(p); }
    int ns=(int)s.size(), npr=(int)pr.size();

    /* one lane per (modulus, curve); skip non-eligible moduli (lane n=0) */
    int L=M*ncurves;
    std::vector<u64> n(L),np(L),R1(L),R2(L),seed(L);
    u64 rs=0x9E3779B97F4A7C15ULL;
    for(int i=0;i<M;i++){ u64 N=moduli[i]; bool ok=(N>2)&&(N&1)&&(N<(1ULL<<62));
        for(int j=0;j<ncurves;j++){ int l=i*ncurves+j;
            if(ok){ n[l]=N; np[l]=montinv(N); R1[l]=rmod(N); R2[l]=r2mod(N);
                    rs^=rs<<13; rs^=rs>>7; rs^=rs<<17; seed[l]=(rs%N)|2; }
            else  { n[l]=0; np[l]=0; R1[l]=0; R2[l]=0; seed[l]=0; } } }

    u64 *dn,*dnp,*dR1,*dR2,*dse,*ds,*dpr,*dfac;
    cudaMalloc(&dn,L*8);cudaMalloc(&dnp,L*8);cudaMalloc(&dR1,L*8);cudaMalloc(&dR2,L*8);cudaMalloc(&dse,L*8);cudaMalloc(&dfac,L*8);
    cudaMalloc(&ds,(ns?ns:1)*8);cudaMalloc(&dpr,(npr?npr:1)*8);
    cudaMemcpy(dn,n.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dnp,np.data(),L*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,R1.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR2,R2.data(),L*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dse,seed.data(),L*8,cudaMemcpyHostToDevice);
    if(ns) cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);
    if(npr) cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);

    ecm_kernel<<<(L+63)/64,64>>>(dn,dnp,dR1,dR2,dse,ds,ns,dpr,npr,dfac,L,ncurves);
    cudaDeviceSynchronize();

    std::vector<u64> fl(L); cudaMemcpy(fl.data(),dfac,L*8,cudaMemcpyDeviceToHost);
    for(int i=0;i<M;i++) for(int j=0;j<ncurves;j++){ u64 f=fl[i*ncurves+j]; if(f){ factor[i]=f; break; } }

    cudaFree(dn);cudaFree(dnp);cudaFree(dR1);cudaFree(dR2);cudaFree(dse);cudaFree(dfac);cudaFree(ds);cudaFree(dpr);
}

} // namespace gpu_ecm
