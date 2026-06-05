/*
 * gpu-ecm-stage2.cu — GPU ECM with a stage-2 BSGS continuation, measuring the
 * extra yield (more cofactors cracked per curve) vs stage 1 alone.
 *
 * Stage 1 gives Q=[k]P (k = prod of prime powers <= B1). Stage 2 finds a factor
 * whose curve order = k * p for a single prime p in (B1,B2]: with baby steps
 * T[r]=[r]Q (r<w) and giant steps V=[m w]Q, every prime p=m w - r is tested by
 * accumulating the cross-difference (V.X T[r].Z - T[r].X V.Z); one gcd at the end
 * finds factors. ~sqrt(B2) work instead of one ladder per prime.
 *
 * Same __host__ __device__ code on CPU and GPU => GPU validated bit-exact.
 *   nvcc -arch=sm_86 -O3 bench/gpu-ecm-stage2.cu -o /tmp/gpu-ecm-stage2 && /tmp/gpu-ecm-stage2
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
#define W 60                                 /* baby-step table size (per thread) */

HD static inline u64 mm(u64 a,u64 b,u64 n,u64 np){ unsigned __int128 T=(unsigned __int128)a*b; u64 m=(u64)T*np; unsigned __int128 s=T+(unsigned __int128)m*n; u64 t=(u64)(s>>64); return t>=n?t-n:t; }
HD static inline u64 ad(u64 a,u64 b,u64 n){ u64 s=a+b; return s>=n?s-n:s; }
HD static inline u64 sub(u64 a,u64 b,u64 n){ return a>=b?a-b:a+n-b; }
HD static inline int ctz64(u64 x){
#ifdef __CUDA_ARCH__
    return __ffsll((long long)x)-1;
#else
    return __builtin_ctzll(x);
#endif
}
HD static u64 bgcd(u64 a,u64 b){ if(!a)return b; if(!b)return a; int sh=ctz64(a|b); a>>=ctz64(a); do{ b>>=ctz64(b); if(a>b){u64 t=a;a=b;b=t;} b-=a; }while(b); return a<<sh; }

struct PT{ u64 X,Z; };
HD static inline PT dbl(PT p,u64 a,u64 n,u64 np){ u64 A=mm(ad(p.X,p.Z,n),ad(p.X,p.Z,n),n,np),B=mm(sub(p.X,p.Z,n),sub(p.X,p.Z,n),n,np),C=sub(A,B,n); PT r; r.X=mm(A,B,n,np); r.Z=mm(C,ad(B,mm(a,C,n,np),n),n,np); return r; }
HD static inline PT dadd(PT p1,PT p2,PT pd,u64 n,u64 np){ u64 DA=mm(sub(p1.X,p1.Z,n),ad(p2.X,p2.Z,n),n,np),CB=mm(ad(p1.X,p1.Z,n),sub(p2.X,p2.Z,n),n,np),s=ad(DA,CB,n),d=sub(DA,CB,n); PT r; r.X=mm(pd.Z,mm(s,s,n,np),n,np); r.Z=mm(pd.X,mm(d,d,n,np),n,np); return r; }
HD static PT ladder(PT P,u64 k,u64 a,u64 n,u64 np){ if(k==0){PT o;o.X=0;o.Z=0;return o;} if(k==1)return P; PT R0=P,R1=dbl(P,a,n,np); int b=63; while(!((k>>b)&1))b--; for(b--;b>=0;b--){ if((k>>b)&1){R0=dadd(R0,R1,P,n,np);R1=dbl(R1,a,n,np);} else {R1=dadd(R0,R1,P,n,np);R0=dbl(R0,a,n,np);} } return R0; }

/* run curve: returns gcd after stage1 in *g1, after stage2 in *g2.
 * s[]=stage-1 prime powers; pr[]=primes in (B1,B2] sorted ascending. */
HD static void ecm_run(u64 n,u64 np,u64 R1,u64 R2,u64 seed,
                       const u64*s,int ns,const u64*pr,int npr,u64*g1,u64*g2){
    u64 a24=seed%n; if(a24<2)a24=2;
    u64 two=ad(R1,R1,n);
    PT P; P.X=two; P.Z=R1;
    u64 a=mm(a24,R2,n,np);
    for(int i=0;i<ns;i++) P=ladder(P,s[i],a,n,np);     /* stage 1: Q=[k]P */
    PT Q=P;
    *g1=bgcd(mm(Q.Z,1,n,np),n);

    /* ---- stage 2 BSGS ---- */
    PT T[W];                                            /* baby steps [1..W-1] */
    T[1]=Q; T[2]=dbl(Q,a,n,np);
    for(int r=3;r<W;r++) T[r]=dadd(T[r-1],Q,T[r-2],n,np);   /* T[r]=[r]Q */
    PT Wg=ladder(Q,W,a,n,np);                           /* [W]Q giant step */
    u64 B1=pr[0]-1;                                     /* not used precisely; m from first prime */
    int m=(int)((pr[0]+W-1)/W);                         /* ceil(p0/W) */
    PT V=ladder(Q,(u64)m*W,a,n,np), Vp=ladder(Q,(u64)(m-1)*W,a,n,np);
    u64 g=R1;                                           /* accumulator = 1 (Montgomery) */
    for(int k=0;k<npr;k++){
        u64 p=pr[k]; int mp=(int)((p+W-1)/W);
        while(m<mp){ PT Vn=dadd(V,Wg,Vp,n,np); Vp=V; V=Vn; m++; }
        int r=m*W-(int)p;                               /* r in [1,W-1] */
        if(r<=0||r>=W) continue;                        /* safety */
        u64 diff=sub(mm(V.X,T[r].Z,n,np), mm(T[r].X,V.Z,n,np), n);
        g=mm(g,diff,n,np);                              /* product of cross-diffs */
    }
    (void)B1;
    *g2=bgcd(mm(g,1,n,np),n);
}

__global__ void K(const u64*n_,const u64*np_,const u64*R1_,const u64*R2_,const u64*se,
                  const u64*s,int ns,const u64*pr,int npr,u64*g1,u64*g2,int L){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=L) return;
    ecm_run(n_[i],np_[i],R1_[i],R2_[i],se[i],s,ns,pr,npr,&g1[i],&g2[i]);
}

static u64 montinv(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static u64 rmod(u64 n){ return (u64)(((unsigned __int128)1<<64)%n); }
static u64 r2mod(u64 n){ u64 r=rmod(n); return (u64)(((unsigned __int128)r*r)%n); }
static u64 rnd(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }
static bool isp(u64 n){ if(n<2)return false; for(u64 p=2;p*p<=n;p++) if(n%p==0)return false; return true; }
static u64 randprime(u64 lo,u64 hi,u64*st){ for(;;){ u64 c=lo+rnd(st)%(hi-lo); c|=1; if(isp(c))return c; } }

int main(){
    const u64 B1=2000, B2=50000;
    std::vector<u64> s; std::vector<char> comp(B2+1,0);
    for(u64 p=2;p<=B2;p++) if(!comp[p]){ for(u64 q=p*p;q<=B2;q+=p) comp[q]=1; if(p<=B1){ u64 pe=p; while(pe*p<=B1)pe*=p; s.push_back(pe);} }
    std::vector<u64> pr; for(u64 p=B1+1;p<=B2;p++) if(!comp[p]) pr.push_back(p);   /* primes (B1,B2] */
    int ns=(int)s.size(), npr=(int)pr.size();

    const int NC=2048, CV=4; int L=NC*CV;
    std::vector<u64> n(L),np(L),R1(L),R2(L),se(L),pf(NC); u64 st=0x5A5A5AULL;
    for(int c=0;c<NC;c++){ u64 p=randprime(1u<<27,1u<<28,&st),q=randprime(1u<<28,1u<<29,&st); u64 N=p*q; pf[c]=p; for(int j=0;j<CV;j++){int i=c*CV+j; n[i]=N;np[i]=montinv(N);R1[i]=rmod(N);R2[i]=r2mod(N); se[i]=(rnd(&st)%N)|2;} }

    u64 *dn,*dnp,*dR1,*dR2,*dse,*ds,*dpr,*dg1,*dg2;
    cudaMalloc(&dn,L*8);cudaMalloc(&dnp,L*8);cudaMalloc(&dR1,L*8);cudaMalloc(&dR2,L*8);cudaMalloc(&dse,L*8);cudaMalloc(&dg1,L*8);cudaMalloc(&dg2,L*8);cudaMalloc(&ds,ns*8);cudaMalloc(&dpr,npr*8);
    cudaMemcpy(dn,n.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dnp,np.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR1,R1.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dR2,R2.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(dse,se.data(),L*8,cudaMemcpyHostToDevice);cudaMemcpy(ds,s.data(),ns*8,cudaMemcpyHostToDevice);cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);
    K<<<(L+63)/64,64>>>(dn,dnp,dR1,dR2,dse,ds,ns,dpr,npr,dg1,dg2,L); cudaDeviceSynchronize();
    auto t0=std::chrono::steady_clock::now();
    K<<<(L+63)/64,64>>>(dn,dnp,dR1,dR2,dse,ds,ns,dpr,npr,dg1,dg2,L); cudaDeviceSynchronize();
    auto t1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    std::vector<u64> g1(L),g2(L); cudaMemcpy(g1.data(),dg1,L*8,cudaMemcpyDeviceToHost); cudaMemcpy(g2.data(),dg2,L*8,cudaMemcpyDeviceToHost);
    double sec=std::chrono::duration<double>(t1-t0).count();

    /* CPU validation on a subset */
    int CPU=L/32; long mism=0;
    for(int i=0;i<CPU;i++){ u64 a,b; ecm_run(n[i],np[i],R1[i],R2[i],se[i],s.data(),ns,pr.data(),npr,&a,&b); if(a!=g1[i]||b!=g2[i]) mism++; }

    int c1=0,c12=0;
    for(int c=0;c<NC;c++){ bool s1=false,s12=false; for(int j=0;j<CV;j++){ u64 a=g1[c*CV+j],b=g2[c*CV+j]; if(a==pf[c]) s1=true; if(a==pf[c]||b==pf[c]) s12=true; } c1+=s1; c12+=s12; }

    printf("GPU status   : %s\n", cudaGetErrorString(e));
    printf("validation   : %s (%ld/%d lanes differ from CPU)\n", mism==0?"PASS":"FAIL", mism, CPU);
    printf("B1=%llu B2=%llu, %d primes in stage 2, %d composites x %d curves\n",(unsigned long long)B1,(unsigned long long)B2,npr,NC,CV);
    printf("stage 1 only : %d/%d composites cracked\n", c1, NC);
    printf("stage 1 + 2  : %d/%d composites cracked  (+%d, %.2fx)\n", c12, NC, c12-c1, c1? (double)c12/c1:0.0);
    printf("throughput   : %.0f curves/s (stage1+2)\n", L/sec);
    return mism!=0;
}
