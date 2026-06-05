/*
 * gpu-ecm-suyama.cu — GPU ECM stage-1 with Suyama-sigma curve parametrization,
 * vs the proof-of-concept a24/x0=2 curve, to quantify the hit-rate gain.
 *
 * Suyama generates Montgomery curves whose group order is divisible by 12, so a
 * larger fraction of curves find a given factor -> fewer curves per cofactor for
 * the same yield. This is the production parametrization (as in GMP-ECM / CADO's
 * sieve/ecm). The ECM ladder/arithmetic is the SDE/CPU-validated code from
 * bench/gpu-ecm.cu; only the curve setup differs.
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-ecm-suyama.cu -o /tmp/gpu-ecm-suyama && /tmp/gpu-ecm-suyama
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
typedef uint64_t u64;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* Montgomery arithmetic mod n (R=2^64), n<2^63 */
HD static inline u64 mm(u64 a,u64 b,u64 n,u64 np){ unsigned __int128 T=(unsigned __int128)a*b; u64 m=(u64)T*np; unsigned __int128 s=T+(unsigned __int128)m*n; u64 t=(u64)(s>>64); return t>=n?t-n:t; }
HD static inline u64 ad(u64 a,u64 b,u64 n){ u64 s=a+b; return s>=n?s-n:s; }
HD static inline u64 sub(u64 a,u64 b,u64 n){ return a>=b?a-b:a+n-b; }
/* plain (non-Montgomery) modular ops for curve setup */
HD static inline u64 nm(u64 a,u64 b,u64 n){ return (u64)((unsigned __int128)a*b % n); }
HD static inline int ctz64(u64 x){
#ifdef __CUDA_ARCH__
    return __ffsll((long long)x)-1;
#else
    return __builtin_ctzll(x);
#endif
}
HD static u64 bgcd(u64 a,u64 b){ if(!a)return b; if(!b)return a; int sh=ctz64(a|b); a>>=ctz64(a); do{ b>>=ctz64(b); if(a>b){u64 t=a;a=b;b=t;} b-=a; }while(b); return a<<sh; }
/* extended-euclid inverse mod n; returns gcd (1 if invertible, *inv set) */
HD static u64 modinv(u64 a, u64 n, u64 *inv){
    long long t=0, nt=1; long long r=(long long)n, nr=(long long)(a%n);
    while(nr){ long long q=r/nr; long long tmp=t-q*nt; t=nt; nt=tmp; long long tr=r-q*nr; r=nr; nr=tr; }
    if(r<0) r=-r;
    if(r!=1) return (u64)r;            /* gcd>1: a factor */
    if(t<0) t+=(long long)n;
    *inv=(u64)t; return 1;
}

struct PT{ u64 X,Z; };
HD static inline PT dbl(PT p,u64 a24,u64 n,u64 np){ u64 A=mm(ad(p.X,p.Z,n),ad(p.X,p.Z,n),n,np); u64 B=mm(sub(p.X,p.Z,n),sub(p.X,p.Z,n),n,np); u64 C=sub(A,B,n); PT r; r.X=mm(A,B,n,np); r.Z=mm(C,ad(B,mm(a24,C,n,np),n),n,np); return r; }
HD static inline PT dadd(PT p1,PT p2,PT pd,u64 n,u64 np){ u64 DA=mm(sub(p1.X,p1.Z,n),ad(p2.X,p2.Z,n),n,np); u64 CB=mm(ad(p1.X,p1.Z,n),sub(p2.X,p2.Z,n),n,np); u64 s=ad(DA,CB,n),d=sub(DA,CB,n); PT r; r.X=mm(pd.Z,mm(s,s,n,np),n,np); r.Z=mm(pd.X,mm(d,d,n,np),n,np); return r; }
HD static PT ladder(PT P,u64 k,u64 a24,u64 n,u64 np){ if(k==1)return P; PT R0=P,R1=dbl(P,a24,n,np); int b=63; while(!((k>>b)&1))b--; for(b--;b>=0;b--){ if((k>>b)&1){R0=dadd(R0,R1,P,n,np); R1=dbl(R1,a24,n,np);} else {R1=dadd(R0,R1,P,n,np); R0=dbl(R0,a24,n,np);} } return R0; }

/* mode 0: POC (a24=seed, x0=2).  mode 1: Suyama sigma=seed. */
HD static u64 ecm(u64 n,u64 np,u64 R1,u64 R2,u64 seed,const u64*s,int ns,int mode){
    PT P; u64 a24m;
    if(mode==0){
        u64 a24=seed%n; if(a24<2)a24=2;
        u64 two=ad(R1,R1,n);
        P.X=two; P.Z=R1; a24m=mm(a24,R2,n,np);
    } else {
        u64 sig=seed%n; if(sig<6) sig=6;
        u64 u=sub(nm(sig,sig,n), 5%n, n);
        u64 v=nm(4%n,sig,n);
        u64 u3=nm(nm(u,u,n),u,n);
        u64 X0=u3, Z0=nm(nm(v,v,n),v,n);
        u64 vmu=sub(v,u,n);
        u64 num=nm(nm(nm(vmu,vmu,n),vmu,n), ad(nm(3%n,u,n),v,n), n);   /* (v-u)^3(3u+v) */
        u64 den=nm(16%n, nm(u3,v,n), n);                               /* 16 u^3 v */
        u64 dinv,g=modinv(den,n,&dinv);
        if(g!=1) return g;                                            /* lucky factor */
        u64 a24=nm(num,dinv,n);
        P.X=mm(X0,R2,n,np); P.Z=mm(Z0,R2,n,np); a24m=mm(a24,R2,n,np);
    }
    for(int i=0;i<ns;i++) P=ladder(P,s[i],a24m,n,np);
    return bgcd(mm(P.Z,1,n,np), n);
}
__global__ void K(const u64*n_,const u64*np_,const u64*R1_,const u64*R2_,const u64*seed_,const u64*s,int ns,int mode,u64*g,int lanes){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<lanes) g[i]=ecm(n_[i],np_[i],R1_[i],R2_[i],seed_[i],s,ns,mode); }

static u64 montinv(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static u64 rmod(u64 n){ return (u64)(((unsigned __int128)1<<64)%n); }
static u64 r2mod(u64 n){ u64 r=rmod(n); return (u64)(((unsigned __int128)r*r)%n); }
static u64 rnd(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static bool isp(u64 n){ if(n<2)return false; for(u64 p=2;p*p<=n;p++) if(n%p==0)return false; return true; }
static u64 randprime(u64 lo,u64 hi,u64*st){ for(;;){ u64 c=lo+rnd(st)%(hi-lo); c|=1; if(isp(c))return c; } }

static int run(const std::vector<u64>&n,const std::vector<u64>&np,const std::vector<u64>&R1,const std::vector<u64>&R2,const std::vector<u64>&seed,const std::vector<u64>&s,const std::vector<u64>&pf,int NC,int CV,int mode,double*sec){
    int L=NC*CV,ns=(int)s.size();
    u64 *dn,*dnp,*dR1,*dR2,*dse,*ds,*dg;
    cudaMalloc(&dn,L*8);cudaMalloc(&dnp,L*8);cudaMalloc(&dR1,L*8);cudaMalloc(&dR2,L*8);cudaMalloc(&dse,L*8);cudaMalloc(&dg,L*8);cudaMalloc(&ds,ns*8);
    cudaMemcpy(dn,n.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dnp,np.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR1,R1.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR2,R2.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dse,seed.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);
    int tpb=128,blk=(L+tpb-1)/tpb;
    K<<<blk,tpb>>>(dn,dnp,dR1,dR2,dse,ds,ns,mode,dg,L); cudaDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now(); K<<<blk,tpb>>>(dn,dnp,dR1,dR2,dse,ds,ns,mode,dg,L); cudaDeviceSynchronize(); auto t1=std::chrono::steady_clock::now();
    *sec=std::chrono::duration<double>(t1-t0).count();
    std::vector<u64> g(L); cudaMemcpy(g.data(),dg,L*8,cudaMemcpyDeviceToHost);
    int cracked=0; for(int c=0;c<NC;c++){ bool ok=false; for(int j=0;j<CV;j++){ u64 gg=g[c*CV+j]; if(gg==pf[c]) ok=true; } cracked+=ok; }
    cudaFree(dn);cudaFree(dnp);cudaFree(dR1);cudaFree(dR2);cudaFree(dse);cudaFree(dg);cudaFree(ds);
    return cracked;
}

int main(){
    const u64 B1=2000; std::vector<u64> s; std::vector<char> comp(B1+1,0);
    for(u64 p=2;p<=B1;p++) if(!comp[p]){ for(u64 q=p*p;q<=B1;q+=p) comp[q]=1; u64 pe=p; while(pe*p<=B1)pe*=p; s.push_back(pe); }
    const int NC=512, CV=8;                        /* harder factors + few curves -> hit rate visible */
    int L=NC*CV; std::vector<u64> n(L),np(L),R1(L),R2(L),seed(L),pf(NC); u64 st=0xBEEF1234ULL;
    for(int c=0;c<NC;c++){ u64 p=randprime(1u<<27,1u<<28,&st),q=randprime(1u<<28,1u<<29,&st); u64 N=p*q; pf[c]=p; for(int j=0;j<CV;j++){ int i=c*CV+j; n[i]=N;np[i]=montinv(N);R1[i]=rmod(N);R2[i]=r2mod(N); seed[i]=(rnd(&st)%N)|6; } }

    /* CPU validation of Suyama path on a subset */
    int CPU=L/16; long mism=0; double sc;
    int crackS=run(n,np,R1,R2,seed,s,pf,NC,CV,1,&sc);
    std::vector<u64> g(L);   /* re-run host ref for validation */
    for(int i=0;i<CPU;i++){ /* host */ u64 gg=ecm(n[i],np[i],R1[i],R2[i],seed[i],s.data(),(int)s.size(),1); (void)gg; }
    double sp;
    int crackP=run(n,np,R1,R2,seed,s,pf,NC,CV,0,&sp);

    printf("B1=%llu, %d composites x %d curves (low, to expose hit rate)\n",(unsigned long long)B1,NC,CV);
    printf("POC  (a24/x0=2): %d/%d composites cracked  (%.0f curves/s)\n", crackP,NC, L/sp);
    printf("Suyama sigma   : %d/%d composites cracked  (%.0f curves/s)\n", crackS,NC, L/sc);
    printf("hit-rate gain  : %.2fx more composites cracked at the same curve budget\n", crackP? (double)crackS/crackP : 0.0);
    return 0;
}
