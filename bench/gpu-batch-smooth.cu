/*
 * gpu-batch-smooth.cu — GPU batch-smoothness leaf extraction (Roadmap C3),
 * reusing the A2 / v3.1.0 K-limb Montgomery device arithmetic.
 *
 * Background.  CADO's sieve/ecm/batch.cpp implements Bernstein's batch
 * smoothness (Algorithm 7.1, "How to find small factors of integers"): build a
 * product tree of the cofactors R[j], a remainder tree giving T[0][j] = P mod
 * R[j] (P = product of all primes <= 2^lpb), then per leaf extract the smooth
 * part of R[j].  The trees are big-integer (CPU/GMP) and stay there; the **leaf
 * extraction fans out to n independent, bounded-width cofactors** and is the part
 * that maps cleanly onto the GPU with the fixed-K-limb Montgomery arithmetic from
 * the GPU ECM (bench/gpu-ecm-mp.cu).
 *
 * Leaf extraction, Bernstein's powering variant (no big-integer division):
 *   given R (the cofactor) and y0 = P mod R,
 *     y = y0^(2^e) mod R         with 2^e >= max prime multiplicity (e = ceil lg(bits R))
 *     s = gcd(R, y)              = the B-smooth part of R
 *     R is B-smooth  <=>  s == R
 * The powering folds in prime multiplicities; e modular squarings + one gcd per
 * leaf, all in K-limb Montgomery form — montmul is exactly the A2 kernel.  (We
 * gcd against the Montgomery-form y directly: gcd(R, y*2^{64K}) = gcd(R, y) since
 * R is odd, so no leave-Montgomery step is needed.)
 *
 *   nvcc -arch=sm_86 -O3 bench/gpu-batch-smooth.cu -lgmp -o gpu-batch-smooth && ./gpu-batch-smooth
 *
 * Validation: GPU smooth/rough classification + smooth part, bit-exact vs an
 * independent GMP ground truth (gcd-with-P), at 128/256/512-bit cofactors. We
 * also time the GMP P-mod-R (the remainder-tree stand-in) vs the GPU leaf
 * extraction to show, honestly, where the cost sits.
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <gmp.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

/* ---------- K-limb arithmetic (same code as gpu-ecm-mp.cu) ---------- */
template<int K> HD void mp_set0(u64 *r){ for(int i=0;i<K;i++) r[i]=0; }
template<int K> HD void mp_copy(u64 *r,const u64 *a){ for(int i=0;i<K;i++) r[i]=a[i]; }
template<int K> HD bool mp_geq(const u64 *a,const u64 *b){
    for(int i=K-1;i>=0;i--){ if(a[i]!=b[i]) return a[i]>b[i]; } return true; }
template<int K> HD bool mp_is0(const u64 *a){ for(int i=0;i<K;i++) if(a[i]) return false; return true; }
template<int K> HD bool mp_eq(const u64*a,const u64*b){ for(int i=0;i<K;i++) if(a[i]!=b[i]) return false; return true; }
template<int K> HD void mp_sub(u64 *r,const u64 *a,const u64 *b){
    u128 br=0; for(int i=0;i<K;i++){ u128 x=(u128)a[i]-(u128)b[i]-br; r[i]=(u64)x; br=(x>>64)&1; } }
template<int K> HD void mp_rshift1(u64 *a){
    for(int i=0;i<K-1;i++) a[i]=(a[i]>>1)|(a[i+1]<<63); a[K-1]>>=1; }
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
/* binary gcd, K-limb, HD (from gpu-ecm-mp.cu host_gcd) */
template<int K> HD void mp_gcd(u64 *g,const u64 *a0,const u64 *b0){
    u64 a[K],b[K]; mp_copy<K>(a,a0); mp_copy<K>(b,b0);
    if(mp_is0<K>(a)){ mp_copy<K>(g,b); return; }
    if(mp_is0<K>(b)){ mp_copy<K>(g,a); return; }
    while(!(a[0]&1)) mp_rshift1<K>(a);
    while(!mp_is0<K>(b)){
        while(!(b[0]&1)) mp_rshift1<K>(b);
        if(mp_geq<K>(a,b)){ u64 t[K]; mp_copy<K>(t,a); mp_copy<K>(a,b); mp_copy<K>(b,t); }
        mp_sub<K>(b,b,a);
    }
    mp_copy<K>(g,a);
}

/* ---------- the leaf operation (HD: runs on CPU for validation + GPU) ---------- */
/* given cofactor R (odd, K-limb), y0 = P mod R, np = -R^{-1} mod 2^64,
 * R2 = 2^{128K} mod R, and e squarings: return smooth = gcd(R, y0^(2^e) mod R). */
template<int K> HD void smooth_part(u64 *s,const u64 *R,const u64 *y0,
                                    const u64 *R2,u64 np,int e){
    u64 y[K]; montmul<K>(y,y0,R2,R,np);        /* -> Montgomery form */
    for(int i=0;i<e;i++) montmul<K>(y,y,y,R,np);  /* y = y0^(2^e), Montgomery */
    /* gcd(R, y_mont) = gcd(R, y) since R is odd (2^{64K} coprime to R) */
    mp_gcd<K>(s,R,y);
}

/* ---------- host helpers ---------- */
static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
template<int K> static void addmod_h(u64*r,const u64*a,const u64*b,const u64*n){
    u64 t[K]; u128 c=0; for(int i=0;i<K;i++){ u128 x=(u128)a[i]+(u128)b[i]+c; t[i]=(u64)x; c=x>>64; }
    if(c||mp_geq<K>(t,n)) mp_sub<K>(r,t,n); else mp_copy<K>(r,t); }
template<int K> static void compute_R2(u64 *R2,const u64 *n){
    u64 r[K]; mp_set0<K>(r); r[0]=1; if(mp_geq<K>(r,n)) mp_sub<K>(r,r,n);
    for(int i=0;i<128*K;i++) addmod_h<K>(r,r,r,n); mp_copy<K>(R2,r); }
/* mpz <-> K limbs (little-endian) */
template<int K> static void mpz_to_limbs(u64*o,const mpz_t a){
    mp_set0<K>(o); size_t cnt=0; mpz_export(o,&cnt,-1,8,0,0,a); }
template<int K> static void limbs_to_mpz(mpz_t a,const u64*o){ mpz_import(a,K,-1,8,0,0,o); }

/* ---------- GPU kernel ---------- */
template<int K> __global__ void k_smooth(const u64*R,const u64*Y0,const u64*R2,
                                         const u64*NP,int e,u64*S,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    smooth_part<K>(S+i*K,R+i*K,Y0+i*K,R2+i*K,NP[i],e);
}

static u64 rnd(u64 *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* ---------- per-width driver ---------- */
template<int K>
static int run_width(const char*label,unsigned long B,const mpz_t P,double pmod_ms_per)
{
    const int n=8192;
    /* e: need 2^e >= max multiplicity <= bits(R) <= 64K-? ; uniform per width */
    int e=0; while((1<<e) < 64*K) e++;     /* 2^e >= 64K */

    /* build n cofactors: half smooth (primes <= B), half rough (one prime > B) */
    std::vector<u64> R(n*K),Y0(n*K),R2(n*K),NP(n);
    std::vector<char> truth(n);            /* 1 = smooth by construction */
    u64 st=0xBA7C4ULL+K*97;
    mpz_t r,q,pmod,g,Pg; mpz_inits(r,q,pmod,g,Pg,NULL);
    /* small odd primes <= B */
    std::vector<unsigned long> sp; { std::vector<char> c(B+1,0);
        for(unsigned long p=3;p<=B;p+=2){ if(!c[p]){ sp.push_back(p); for(unsigned long m=p*p;m<=B;m+=2*p) c[m]=1; } } }
    unsigned long maxbits=64UL*K-2;
    for(int i=0;i<n;i++){
        bool smooth=(i&1)==0;
        mpz_set_ui(r,1);
        /* multiply odd primes <= B until close to the width (leave headroom) */
        for(;;){ unsigned long p=sp[rnd(&st)%sp.size()];
            mpz_mul_ui(q,r,p); if(mpz_sizeinbase(q,2) > maxbits-40) break; mpz_set(r,q); }
        if(!smooth){ /* throw in one prime > B (use ~ B*small .. random, made prime) */
            mpz_set_ui(q, B + 2 + (rnd(&st)% (B*4+7))); mpz_setbit(q,0); mpz_nextprime(q,q);
            mpz_t t2; mpz_init(t2); mpz_mul(t2,r,q);
            if(mpz_sizeinbase(t2,2) <= maxbits){ mpz_set(r,t2); } else smooth=true; /* didn't fit -> stays smooth */
            mpz_clear(t2);
        }
        if(mpz_even_p(r)) mpz_add_ui(r,r,1);     /* ensure odd (montgomery) */
        truth[i]=smooth?1:0;
        /* y0 = P mod r */
        mpz_mod(pmod,P,r);
        mpz_to_limbs<K>(&R[i*K],r); mpz_to_limbs<K>(&Y0[i*K],pmod);
        NP[i]=ninv64(((u64*)&R[i*K])[0]); compute_R2<K>(&R2[i*K],&R[i*K]);
    }

    /* ---- GPU extraction ---- */
    u64 *dR,*dY,*dR2,*dNP,*dS; cudaMalloc(&dR,n*K*8);cudaMalloc(&dY,n*K*8);
    cudaMalloc(&dR2,n*K*8);cudaMalloc(&dNP,n*8);cudaMalloc(&dS,n*K*8);
    cudaMemcpy(dR,R.data(),n*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dY,Y0.data(),n*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,R2.data(),n*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dNP,NP.data(),n*8,cudaMemcpyHostToDevice);
    auto t0=std::chrono::steady_clock::now();
    k_smooth<K><<<(n+127)/128,128>>>(dR,dY,dR2,dNP,e,dS,n);
    cudaDeviceSynchronize(); auto t1=std::chrono::steady_clock::now();
    cudaError_t err=cudaGetLastError();
    std::vector<u64> S(n*K); cudaMemcpy(S.data(),dS,n*K*8,cudaMemcpyDeviceToHost);
    double gpu_ms=std::chrono::duration<double,std::milli>(t1-t0).count();

    /* ---- validate vs GMP ground truth + classification vs construction ---- */
    long wrong_part=0, wrong_class=0;
    for(int i=0;i<n;i++){
        limbs_to_mpz<K>(r,&R[i*K]);
        /* GMP smooth part: iterate g=gcd(P,r); r/=g until g==1 -> rough; smooth part = r0/rough */
        mpz_set(g, r); /* will become rough part */
        mpz_t rough; mpz_init_set(rough,r);
        for(;;){ mpz_gcd(Pg,P,rough); if(mpz_cmp_ui(Pg,1)==0) break; mpz_divexact(rough,rough,Pg); }
        /* smooth part (GMP) = r / rough */
        mpz_t sp_gmp; mpz_init(sp_gmp); mpz_divexact(sp_gmp,r,rough);
        /* GPU smooth part */
        mpz_t sp_gpu; mpz_init(sp_gpu); limbs_to_mpz<K>(sp_gpu,&S[i*K]);
        if(mpz_cmp(sp_gmp,sp_gpu)!=0) wrong_part++;
        bool gpu_smooth = (mpz_cmp(sp_gpu,r)==0);
        bool gmp_smooth = (mpz_cmp_ui(rough,1)==0);
        if(gpu_smooth!=gmp_smooth) wrong_class++;
        mpz_clears(rough,sp_gmp,sp_gpu,NULL);
    }
    int nsm=0; for(int i=0;i<n;i++) if(truth[i]) nsm++;
    printf("  [%s] leaf extract: smooth-part %s (%ld/%d) classification %s (%ld/%d); "
           "%d smooth/%d; %.2f Mleaf/s%s\n", label,
           wrong_part==0?"PASS":"FAIL", wrong_part, n,
           wrong_class==0?"PASS":"FAIL", wrong_class, n,
           nsm, n, n/gpu_ms/1e3, err?" CUDAERR":"");
    printf("       GPU leaf-extract %.3f us/leaf (%d Mont-sq + gcd). For scale, a "
           "naive un-amortized P mod R is %.1f us/leaf; the remainder TREE amortizes\n"
           "       that big-integer work to ~O(M(N)logN/n) but it stays CPU/GMP "
           "(arbitrary-precision) — the leaf stage is the only A2-arithmetic fit.\n",
           gpu_ms*1e3/n, e, pmod_ms_per*1e3);

    cudaFree(dR);cudaFree(dY);cudaFree(dR2);cudaFree(dNP);cudaFree(dS);
    mpz_clears(r,q,pmod,g,Pg,NULL);
    return (wrong_part||wrong_class||err)?1:0;
}

int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    unsigned long B = 1UL<<20;       /* smoothness bound 2^20 */
    /* P = product of all primes <= B (GMP) — the prime product (remainder-tree input) */
    mpz_t P; mpz_init_set_ui(P,1);
    { std::vector<char> c(B+1,0); mpz_t pp; mpz_init(pp);
      for(unsigned long p=2;p<=B;p++){ if(!c[p]){ mpz_mul_ui(P,P,p); for(unsigned long m=p*p;m<=B;m+=p) c[m]=1; } }
      mpz_clear(pp); }
    printf("GPU batch-smoothness leaf extraction (C3) — B=2^20, P has %zu bits\n",
           mpz_sizeinbase(P,2));

    /* measure GMP P mod R cost per leaf (remainder-tree stand-in) at ~256-bit R */
    double pmod_ms_per; { mpz_t r,t; mpz_inits(r,t,NULL); mpz_set_ui(r,1);
        for(int i=0;i<6;i++) mpz_mul_ui(r,r, 4294967311UL); mpz_setbit(r,0);
        auto a=std::chrono::steady_clock::now(); int reps=2000;
        for(int i=0;i<reps;i++) mpz_mod(t,P,r);
        auto b=std::chrono::steady_clock::now();
        pmod_ms_per=std::chrono::duration<double,std::milli>(b-a).count()/reps;
        mpz_clears(r,t,NULL); }

    int fails=0;
    fails += run_width<2>("128-bit ",B,P,pmod_ms_per);
    fails += run_width<4>("256-bit ",B,P,pmod_ms_per);
    fails += run_width<8>("512-bit ",B,P,pmod_ms_per);
    printf("%s\n", fails==0?"ALL PASS":"FAILURES");
    mpz_clear(P);
    return fails!=0;
}
