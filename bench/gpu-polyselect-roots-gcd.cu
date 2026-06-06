/* GPU gcd-based polynomial root-finding mod primes (v3.2.0 C2): roots of f mod p
 * via g = gcd(x^p - x, f) then Cantor-Zassenhaus split. O(d^2 log p) per prime,
 * p-magnitude-independent (vs direct eval's O(p)). __host__ __device__ so GPU and
 * CPU run identical code; validated vs direct-eval (full root multiset). */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>
#include <cuda_runtime.h>
typedef uint32_t u32; typedef uint64_t u64;
#define D 8                 /* max poly degree we handle */
struct Poly { u64 c[2*D+2]; int deg; };  /* deg = highest nonzero index, -1 if zero */

__host__ __device__ static u64 mmul(u64 a,u64 b,u64 p){ return (a*b)%p; }
__host__ __device__ static u64 madd(u64 a,u64 b,u64 p){ a+=b; return a>=p?a-p:a; }
__host__ __device__ static u64 msub(u64 a,u64 b,u64 p){ return a>=b?a-b:a+p-b; }
__host__ __device__ static u64 minv(u64 a,u64 m){ long long t0=0,t1=1; u64 r0=m,r1=a%m;
    while(r1){u64 q=r0/r1,r2=r0-q*r1; long long t2=t0-(long long)q*t1; r0=r1;r1=r2;t0=t1;t1=t2;}
    if(r0!=1) return 0; long long i=t0%(long long)m; if(i<0)i+=m; return (u64)i; }
__host__ __device__ static void pnorm(Poly&a){ a.deg=-1; for(int i=2*D+1;i>=0;i--) if(a.c[i]){a.deg=i;break;} }
__host__ __device__ static void pmonic(Poly&a,u64 p){ if(a.deg<0)return; u64 iv=minv(a.c[a.deg],p);
    for(int i=0;i<=a.deg;i++) a.c[i]=mmul(a.c[i],iv,p); }
/* a mod b (b monic), schoolbook; a is modified in place, result deg < b.deg */
__host__ __device__ static void prem(Poly&a,const Poly&b,u64 p){
    if(b.deg<=0){ a.deg=-1; for(int i=0;i<2*D+2;i++)a.c[i]=0; return; }
    while(a.deg>=b.deg){ u64 lc=a.c[a.deg]; int sh=a.deg-b.deg;
        for(int i=0;i<=b.deg;i++) a.c[i+sh]=msub(a.c[i+sh], mmul(lc,b.c[i],p), p);
        a.c[a.deg]=0; pnorm(a); if(a.deg<0)break; }
}
/* gcd(a,b) monic */
__host__ __device__ static Poly pgcd(Poly a,Poly b,u64 p){
    pnorm(a);pnorm(b);
    while(b.deg>=0){ Poly bb=b; pmonic(bb,p); prem(a,bb,p); Poly t=a;a=b;b=t; }
    pmonic(a,p); return a;
}
/* (base)^e mod m, polys; m monic-ized internally */
__host__ __device__ static Poly ppow(Poly base,u64 e,Poly m,u64 p){
    pmonic(m,p);
    Poly r; for(int i=0;i<2*D+2;i++)r.c[i]=0; r.c[0]=1; r.deg=0;
    prem(base,m,p);
    while(e){
        if(e&1){ Poly t; for(int i=0;i<2*D+2;i++)t.c[i]=0;     /* t = r*base */
            for(int i=0;i<=(r.deg<0?0:r.deg);i++) if(r.c[i]) for(int j=0;j<=(base.deg<0?0:base.deg);j++) if(base.c[j]) t.c[i+j]=madd(t.c[i+j],mmul(r.c[i],base.c[j],p),p);
            pnorm(t); prem(t,m,p); r=t; }
        Poly s; for(int i=0;i<2*D+2;i++)s.c[i]=0;             /* base = base^2 */
        for(int i=0;i<=(base.deg<0?0:base.deg);i++) if(base.c[i]) for(int j=0;j<=(base.deg<0?0:base.deg);j++) if(base.c[j]) s.c[i+j]=madd(s.c[i+j],mmul(base.c[i],base.c[j],p),p);
        pnorm(s); prem(s,m,p); base=s; e>>=1;
    }
    return r;
}
/* exact division a/b when b|a (b monic), returns quotient */
__host__ __device__ static Poly pdiv(Poly a,Poly b,u64 p){
    Poly q; for(int i=0;i<2*D+2;i++)q.c[i]=0; q.deg=-1;
    pmonic(b,p);
    while(a.deg>=b.deg && a.deg>=0){ u64 lc=a.c[a.deg]; int sh=a.deg-b.deg; q.c[sh]=lc;
        for(int i=0;i<=b.deg;i++) a.c[i+sh]=msub(a.c[i+sh],mmul(lc,b.c[i],p),p);
        a.c[a.deg]=0; pnorm(a); }
    pnorm(q); return q;
}
/* find roots of f (degree df given by coeffs) mod p; write up to D roots, return count */
__host__ __device__ static int roots_gcd(const u64* fc,int fdeg,u64 p,u32* out){
    if(p<3) { int k=0; for(u64 a=0;a<p;a++){u64 acc=fc[fdeg]%p; for(int i=fdeg-1;i>=0;i--)acc=(acc*a+fc[i])%p; if(acc%p==0)out[k++]=(u32)a;} return k; }
    Poly f; for(int i=0;i<2*D+2;i++)f.c[i]=0; for(int i=0;i<=fdeg;i++)f.c[i]=fc[i]%p; pnorm(f);
    if(f.deg<=0) return 0;          /* constant: 0 roots (nonzero) */
    pmonic(f,p);
    Poly x; for(int i=0;i<2*D+2;i++)x.c[i]=0; x.c[1]=1; x.deg=1;
    Poly h=ppow(x,p,f,p);           /* x^p mod f */
    h.c[1]=msub(h.c[1],1,p); pnorm(h);   /* h - x */
    Poly g=pgcd(f,h,p);             /* roots = roots of g (squarefree, splits to linears) */
    if(g.deg<=0) return 0;
    /* Cantor-Zassenhaus: split g into linear factors */
    Poly stack[D+2]; int sp=0; stack[sp++]=g; int k=0;
    while(sp>0 && k<D){
        Poly h2=stack[--sp]; pnorm(h2); if(h2.deg<=0) continue;
        if(h2.deg==1){ pmonic(h2,p); out[k++]=(u32)msub(0,h2.c[0],p); continue; }  /* x + c -> root -c */
        Poly fac; bool split=false;
        for(u64 d=0; d<64 && !split; d++){
            Poly base; for(int i=0;i<2*D+2;i++)base.c[i]=0; base.c[0]=d%p; base.c[1]=1; base.deg=1; /* x+d */
            Poly b=ppow(base,(p-1)/2,h2,p);
            b.c[0]=msub(b.c[0],1,p); pnorm(b);                    /* b - 1 */
            fac=pgcd(h2,b,p);
            if(fac.deg>0 && fac.deg<h2.deg) split=true;
        }
        if(!split){ /* fallback: peel via gcd(h2, b) for the r+d==0 case, else bail */
            Poly base; for(int i=0;i<2*D+2;i++)base.c[i]=0; base.c[1]=1; base.deg=1; base.c[0]=0;
            Poly b=ppow(base,(p-1)/2,h2,p); fac=pgcd(h2,b,p);
            if(!(fac.deg>0&&fac.deg<h2.deg)) continue; /* give up on this factor (rare) */
        }
        Poly other=pdiv(h2,fac,p);
        stack[sp++]=fac; stack[sp++]=other;
    }
    return k;
}
__global__ void k_gcd(const u64* fc,int fdeg,const u32* p,int n,u32* out,u32* cnt){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    u32 r[D]; int k=roots_gcd(fc,fdeg,p[i],r); cnt[i]=k; for(int j=0;j<k;j++) out[(size_t)i*D+j]=r[j];
}
/* direct-eval reference */
static int ref_roots(const u64* c,int deg,u32 p,u32* out){ int k=0; for(u64 a=0;a<p;a++){u64 acc=c[deg]%p; for(int i=deg-1;i>=0;i--)acc=(acc*a+c[i])%p; if(acc==0)out[k++]=(u32)a;} return k; }
int main(){
    const int deg=6; u64 c[D+1]; u64 s=0x6cd6ULL; auto xr=[&]{s^=s<<13;s^=s>>7;s^=s<<17;return s;};
    for(int i=0;i<=deg;i++)c[i]=xr()%2000000000ULL;
    /* validation primes: up to 30000 (direct-eval feasible) */
    std::vector<u32> primes; {std::vector<char>sv(30001,1); for(int i=2;i<=30000;i++) if(sv[i]){primes.push_back(i);for(int j=2*i;j<=30000;j+=i)sv[j]=0;}}
    int n=primes.size();
    u64*dc; u32*dp,*dout,*dcnt; cudaMalloc(&dc,(deg+1)*8);cudaMalloc(&dp,n*4);cudaMalloc(&dout,(size_t)n*D*4);cudaMalloc(&dcnt,n*4);
    cudaMemcpy(dc,c,(deg+1)*8,cudaMemcpyHostToDevice);cudaMemcpy(dp,primes.data(),n*4,cudaMemcpyHostToDevice);
    int tpb=64,blk=(n+tpb-1)/tpb; k_gcd<<<blk,tpb>>>(dc,deg,dp,n,dout,dcnt); cudaDeviceSynchronize();
    cudaError_t e=cudaGetLastError(); if(e){printf("CUDA err: %s\n",cudaGetErrorString(e));return 1;}
    std::vector<u32> out((size_t)n*D),cnt(n); cudaMemcpy(out.data(),dout,(size_t)n*D*4,cudaMemcpyDeviceToHost);cudaMemcpy(cnt.data(),dcnt,n*4,cudaMemcpyDeviceToHost);
    long mism=0,selfbad=0,tot=0;
    for(int i=0;i<n;i++){ u32 rr[D]; int rc=ref_roots(c,deg,primes[i],rr);
        std::vector<u32> gv(out.begin()+(size_t)i*D, out.begin()+(size_t)i*D+cnt[i]); std::sort(gv.begin(),gv.end());
        if(rc!=(int)cnt[i]){mism++; continue;}
        for(int k=0;k<rc;k++){ if(gv[k]!=rr[k])mism++; u64 acc=c[deg]%primes[i]; for(int t=deg-1;t>=0;t--)acc=(acc*gv[k]+c[t])%primes[i]; if(acc)selfbad++; }
        tot+=rc; }
    printf("%s: GPU gcd-based root-finding vs direct-eval, %d primes (deg %d, p<30000): %ld mismatch, %ld self-bad, %ld roots\n",
           (mism==0&&selfbad==0)?"PASS":"FAIL",n,deg,mism,selfbad,tot);
    /* large-p demonstration: direct-eval is O(p) and infeasible here; the gcd
     * method is O(d^2 log p). Primes near 1e9, time + self-check every root. */
    {
        std::vector<u32> lp; u64 q=1000000007ULL;
        for(int i=0;i<5000;i++){ // pick 5000 primes near 1e9 (trial via simple test)
            while(true){ q+=2; bool pr=true; for(u64 d=3; d*d<=q; d+=2) if(q%d==0){pr=false;break;} if(pr)break; }
            lp.push_back((u32)q);
        }
        int m=lp.size(); u32 *dlp,*dlo,*dlc; cudaMalloc(&dlp,m*4);cudaMalloc(&dlo,(size_t)m*D*4);cudaMalloc(&dlc,m*4);
        cudaMemcpy(dlp,lp.data(),m*4,cudaMemcpyHostToDevice);
        int b2=(m+tpb-1)/tpb; k_gcd<<<b2,tpb>>>(dc,deg,dlp,m,dlo,dlc); cudaDeviceSynchronize();
        auto t0=std::chrono::steady_clock::now();
        for(int it=0;it<10;it++) k_gcd<<<b2,tpb>>>(dc,deg,dlp,m,dlo,dlc); cudaDeviceSynchronize();
        auto t1=std::chrono::steady_clock::now(); double sec=std::chrono::duration<double>(t1-t0).count()/10;
        std::vector<u32> lo((size_t)m*D),lc(m); cudaMemcpy(lo.data(),dlo,(size_t)m*D*4,cudaMemcpyDeviceToHost);cudaMemcpy(lc.data(),dlc,m*4,cudaMemcpyDeviceToHost);
        long bad=0,tot2=0; for(int i=0;i<m;i++) for(int k=0;k<(int)lc[i];k++){ u64 r=lo[(size_t)i*D+k],acc=c[deg]%lp[i]; for(int t=deg-1;t>=0;t--)acc=(acc*r+c[t])%lp[i]; if(acc)bad++; tot2+=1; }
        printf("        large-p: %d primes near 1e9, %.2f ms, %ld roots, %ld self-check-bad (direct-eval O(p) infeasible here)\n", m, sec*1e3, tot2, bad);
        cudaFree(dlp);cudaFree(dlo);cudaFree(dlc);
    }
    cudaFree(dc);cudaFree(dp);cudaFree(dout);cudaFree(dcnt);
    return (mism||selfbad)!=0;
}
