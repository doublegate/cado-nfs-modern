/*
 * gpu-prefactor-pm1pp1.cu — bit-exact validation of GPU Pollard P-1 / Williams P+1
 * (v3.4.0 Track C7), the new methods added to the GPU pre-NFS factoring front-end
 * beside the batched ECM (misc/gpu_prefactor/gpu_pm1_pp1.cuh).
 *
 *   nvcc -arch=sm_86 -O3 -I misc/gpu_prefactor bench/gpu-prefactor-pm1pp1.cu \
 *        -lgmp -o /tmp/gpu-prefactor-pm1pp1 && /tmp/gpu-prefactor-pm1pp1
 *
 * Three checks per width K (= 128/256-bit moduli):
 *   (1) GPU vs CPU bit-exact: the kernels (device) vs pm1_run/pp1_run on the host
 *       (same __host__ __device__ code) — stage 1 AND stage 2 limbs identical.
 *   (2) GPU stage-1 vs an INDEPENDENT GMP reference: P-1 a=base^E via mpz_powm_ui;
 *       P+1 V_E(seed) via a GMP Lucas chain. Residues identical.
 *   (3) Functional: crafted composites n=p*q with p-1 (resp. p+1) B1-smooth are
 *       cracked by P-1 (resp. P+1) stage 1; and a p with p-1 = smooth * one prime
 *       in (B1,B2] is cracked by P-1 stage 2.
 * Exit 0 iff every (1)/(2) check is bit-exact and the stage-1 functional cracks
 * succeed. (Stage-2 functional recovery is reported; P+1 success is seed-dependent
 * and reported per the lanes that hit.)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <gmp.h>
#include "gpu_ecm_mp.cuh"
#include "gpu_pm1_pp1.cuh"

static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }
static void to_limbs(u64 *out,int K,const mpz_t v){ for(int i=0;i<K;i++) out[i]=0;
    size_t c=0; mpz_export(out,&c,-1,8,0,0,v); }
static void from_limbs(mpz_t v,const u64 *in,int K){ mpz_import(v,K,-1,8,0,0,in); }

/* prime powers <= B1 (spow) and primes in (B1,B2] (pr), like the prefactor */
static void sieve_pp(unsigned long B1,unsigned long B2,
                     std::vector<u64>&spow,std::vector<u64>&pr){
    std::vector<char> comp(B2+1,0);
    for(unsigned long p=2;p<=B2;p++) if(!comp[p]){
        for(unsigned long q=p*p;q<=B2;q+=p) comp[q]=1;
        if(p<=B1){ u64 pe=p; while(pe*p<=B1) pe*=p; spow.push_back(pe); }
        else pr.push_back(p);
    }
}

/* GMP reference: P-1 stage-1 residue base^E mod N (E = prod spow). */
static void gmp_pm1_s1(mpz_t out,const mpz_t N,unsigned long base,
                       const std::vector<u64>&spow){
    mpz_set_ui(out,base); mpz_mod(out,out,N);
    for(u64 pe: spow) mpz_powm_ui(out,out,pe,N);
}
/* GMP Lucas chain V_k(seed) mod N (matches the device ladder). */
static void gmp_lucas(mpz_t out,const mpz_t seed,u64 k,const mpz_t N){
    if(k==0){ mpz_set_ui(out,2); return; }
    if(k==1){ mpz_mod(out,seed,N); return; }
    mpz_t Vm,Vm1,t,two; mpz_inits(Vm,Vm1,t,two,NULL); mpz_set_ui(two,2);
    mpz_mod(Vm,seed,N);
    mpz_mul(Vm1,Vm,Vm); mpz_sub(Vm1,Vm1,two); mpz_mod(Vm1,Vm1,N);   /* V_2 */
    int b=63; while(!((k>>b)&1ULL)) b--;
    for(b--;b>=0;b--){
        if((k>>b)&1ULL){
            mpz_mul(t,Vm,Vm1); mpz_sub(t,t,seed); mpz_mod(t,t,N);   /* V_{2m+1} */
            mpz_mul(Vm1,Vm1,Vm1); mpz_sub(Vm1,Vm1,two); mpz_mod(Vm1,Vm1,N); /* V_{2m+2} */
            mpz_set(Vm,t);
        } else {
            mpz_mul(t,Vm,Vm1); mpz_sub(t,t,seed); mpz_mod(t,t,N);   /* V_{2m+1} */
            mpz_mul(Vm,Vm,Vm); mpz_sub(Vm,Vm,two); mpz_mod(Vm,Vm,N);/* V_{2m} */
            mpz_set(Vm1,t);
        }
    }
    mpz_set(out,Vm); mpz_clears(Vm,Vm1,t,two,NULL);
}
/* compose Lucas over the prime powers: V_E(seed), E = prod spow */
static void gmp_pp1_s1(mpz_t out,const mpz_t N,unsigned long seed,
                       const std::vector<u64>&spow){
    mpz_t cur; mpz_init_set_ui(cur,seed); mpz_mod(cur,cur,N);
    for(u64 pe: spow){ mpz_t t; mpz_init(t); gmp_lucas(t,cur,pe,N); mpz_set(cur,t); mpz_clear(t); }
    mpz_set(out,cur); mpz_clear(cur);
}

/* run one GPU pm1/pp1 lane-batch at width K; fill G1/G2 (size lanes*K) */
template<int K>
static void gpu_run(bool pp1,const mpz_t N,const std::vector<u64>&bases,
                    const std::vector<u64>&spow,const std::vector<u64>&pr,
                    std::vector<u64>&G1,std::vector<u64>&G2){
    int lanes=(int)bases.size(); int ns=(int)spow.size(),npr=(int)pr.size();
    u64 Nl[K]; to_limbs(Nl,K,N); u64 np=ninv64(Nl[0]);
    mpz_t t,R1m,R2m; mpz_inits(t,R1m,R2m,NULL);
    mpz_setbit(t,64*K); mpz_mod(R1m,t,N);
    mpz_set_ui(t,0); mpz_setbit(t,128*K); mpz_mod(R2m,t,N);
    u64 R1[K],R2[K]; to_limbs(R1,K,R1m); to_limbs(R2,K,R2m);
    std::vector<u64> Nv(lanes*K),R1v(lanes*K),R2v(lanes*K),NPv(lanes),BS(lanes*K);
    for(int i=0;i<lanes;i++){ mp_copy<K>(&Nv[i*K],Nl); mp_copy<K>(&R1v[i*K],R1);
        mp_copy<K>(&R2v[i*K],R2); NPv[i]=np;
        mpz_t b; mpz_init_set_ui(b,(unsigned long)bases[i]); to_limbs(&BS[i*K],K,b); mpz_clear(b); }
    G1.assign(lanes*K,0); G2.assign(lanes*K,0);
    u64 *dN,*dNP,*dR1,*dR2,*dBS,*ds,*dpr,*dG1,*dG2; size_t cb=(size_t)lanes*K*8;
    cudaMalloc(&dN,cb);cudaMalloc(&dNP,lanes*8);cudaMalloc(&dR1,cb);cudaMalloc(&dR2,cb);
    cudaMalloc(&dBS,cb);cudaMalloc(&dG1,cb);cudaMalloc(&dG2,cb);
    cudaMalloc(&ds,ns>0?ns*8:8);cudaMalloc(&dpr,npr>0?npr*8:8);
    cudaMemcpy(dN,Nv.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(dNP,NPv.data(),lanes*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,R1v.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,R2v.data(),cb,cudaMemcpyHostToDevice);
    cudaMemcpy(dBS,BS.data(),cb,cudaMemcpyHostToDevice);
    if(ns>0) cudaMemcpy(ds,spow.data(),ns*8,cudaMemcpyHostToDevice);
    if(npr>0) cudaMemcpy(dpr,pr.data(),npr*8,cudaMemcpyHostToDevice);
    int tpb=32, blk=(lanes+tpb-1)/tpb;
    if(pp1) pp1_kernel<K><<<blk,tpb>>>(dN,dNP,dR1,dR2,dBS,ds,ns,dpr,npr,dG1,dG2,lanes);
    else    pm1_kernel<K><<<blk,tpb>>>(dN,dNP,dR1,dR2,dBS,ds,ns,dpr,npr,dG1,dG2,lanes);
    cudaMemcpy(G1.data(),dG1,cb,cudaMemcpyDeviceToHost);
    cudaMemcpy(G2.data(),dG2,cb,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(dN);cudaFree(dNP);cudaFree(dR1);cudaFree(dR2);cudaFree(dBS);
    cudaFree(dG1);cudaFree(dG2);cudaFree(ds);cudaFree(dpr);
    mpz_clears(t,R1m,R2m,NULL);
}

/* host reference via the same HD functions, for the GPU-vs-CPU bit-exact check */
template<int K>
static void host_run(bool pp1,const mpz_t N,const std::vector<u64>&bases,
                     const std::vector<u64>&spow,const std::vector<u64>&pr,
                     std::vector<u64>&G1,std::vector<u64>&G2){
    int lanes=(int)bases.size(); int ns=(int)spow.size(),npr=(int)pr.size();
    u64 Nl[K]; to_limbs(Nl,K,N); u64 np=ninv64(Nl[0]);
    mpz_t t,R1m,R2m; mpz_inits(t,R1m,R2m,NULL);
    mpz_setbit(t,64*K); mpz_mod(R1m,t,N);
    mpz_set_ui(t,0); mpz_setbit(t,128*K); mpz_mod(R2m,t,N);
    u64 R1[K],R2[K]; to_limbs(R1,K,R1m); to_limbs(R2,K,R2m);
    G1.assign(lanes*K,0); G2.assign(lanes*K,0);
    for(int i=0;i<lanes;i++){ u64 bs[K]; mpz_t b; mpz_init_set_ui(b,(unsigned long)bases[i]);
        to_limbs(bs,K,b); mpz_clear(b);
        if(pp1) pp1_run<K>(&G1[i*K],&G2[i*K],Nl,np,R1,R2,bs,spow.data(),ns,pr.data(),npr);
        else    pm1_run<K>(&G1[i*K],&G2[i*K],Nl,np,R1,R2,bs,spow.data(),ns,pr.data(),npr);
    }
    mpz_clears(t,R1m,R2m,NULL);
}

static int g_fail=0, g_pass=0;
static void check(bool ok,const char*what){ if(ok){g_pass++;printf("# PASS %s\n",what);}
    else{g_fail++;printf("# FAIL %s\n",what);} }

/* multiply m by DISTINCT random primes drawn from sp (so each prime appears to the
 * first power => B1-powersmooth by construction) until m reaches seedbits. */
static void mul_distinct(mpz_t m,const std::vector<unsigned long>&sp,int seedbits,
                         gmp_randstate_t rs){
    std::vector<char> used(sp.size(),0);
    while((int)mpz_sizeinbase(m,2)<seedbits){
        size_t idx=gmp_urandomm_ui(rs,sp.size());
        if(used[idx]) continue;
        used[idx]=1; mpz_mul_ui(m,m,sp[idx]);
    }
}
/* craft prime p with (p + sign) B1-powersmooth (sign=-1 => p-1 smooth; +1 => p+1
 * smooth). seedbits ~ size of p. */
static void craft_smooth(mpz_t p,int sign,unsigned long B1,int seedbits,gmp_randstate_t rs){
    mpz_t m,cand; mpz_inits(m,cand,NULL);
    std::vector<unsigned long> sp; std::vector<char> comp(B1+1,0);
    for(unsigned long x=3;x<=B1;x++) if(!comp[x]){ for(unsigned long y=x*x;y<=B1;y+=x) comp[y]=1; sp.push_back(x); }
    for(;;){
        mpz_set_ui(m,2);                       /* the single factor 2 (=> p odd) */
        mul_distinct(m,sp,seedbits,rs);
        mpz_set(cand,m);
        if(sign<0) mpz_add_ui(cand,cand,1); else mpz_sub_ui(cand,cand,1);
        if(mpz_probab_prime_p(cand,25)){ mpz_set(p,cand); break; }
    }
    mpz_clears(m,cand,NULL);
}
/* craft p with p-1 = (B1-powersmooth) * one prime P in (B1,B2] (for stage-2) */
static void craft_stage2(mpz_t p,unsigned long B1,unsigned long B2,int seedbits,gmp_randstate_t rs){
    std::vector<unsigned long> sp,big; std::vector<char> comp(B2+1,0);
    for(unsigned long x=2;x<=B2;x++) if(!comp[x]){ for(unsigned long y=x*x;y<=B2;y+=x) comp[y]=1;
        if(x>2 && x<=B1) sp.push_back(x); else if(x>B1) big.push_back(x); }
    mpz_t m,cand; mpz_inits(m,cand,NULL);
    for(;;){
        unsigned long P=big[gmp_urandomm_ui(rs,big.size())];
        mpz_set_ui(m,2); mpz_mul_ui(m,m,P);                 /* 2 * P */
        mul_distinct(m,sp,seedbits,rs);
        mpz_add_ui(cand,m,1);
        if(mpz_probab_prime_p(cand,25)){ mpz_set(p,cand); break; }
    }
    mpz_clears(m,cand,NULL);
}

template<int K>
static void run_width(unsigned long B1,unsigned long B2,gmp_randstate_t rs){
    printf("## width K=%d  (<= %d-bit N), B1=%lu B2=%lu\n",K,64*K-2,B1,B2);
    std::vector<u64> spow,pr; sieve_pp(B1,B2,spow,pr);

    /* (1)+(2): random odd N, several bases/seeds; GPU vs CPU and vs GMP */
    mpz_t N,ref,z,g; mpz_inits(N,ref,z,g,NULL);
    mpz_urandomb(N,rs,64*K-4); mpz_setbit(N,0); mpz_setbit(N,64*K-4);   /* odd, full width */
    std::vector<u64> bases={2,3,5,7,11,13,17,19};
    for(int which=0;which<2;which++){
        bool pp1=(which==1);
        std::vector<u64> Gg1,Gg2,Hg1,Hg2;
        gpu_run<K>(pp1,N,bases,spow,pr,Gg1,Gg2);
        host_run<K>(pp1,N,bases,spow,pr,Hg1,Hg2);
        bool be=(Gg1==Hg1 && Gg2==Hg2);
        check(be,pp1?"P+1 GPU==CPU (stage1+2 limbs)":"P-1 GPU==CPU (stage1+2 limbs)");
        /* independent GMP stage-1 residue, lane 0 (base/seed = bases[0]) */
        if(pp1){ gmp_pp1_s1(ref,N,(unsigned long)bases[0],spow);
            mpz_sub_ui(ref,ref,2); mpz_mod(ref,ref,N); }
        else   { gmp_pm1_s1(ref,N,(unsigned long)bases[0],spow); }
        from_limbs(z,&Gg1[0],K);
        check(mpz_cmp(z,ref)==0,pp1?"P+1 stage1 GPU==GMP (V_E-2)":"P-1 stage1 GPU==GMP (base^E)");
    }

    /* a gcd lane recovered a nontrivial factor of n (the honest success test) */
    auto nontrivial=[&](const mpz_t gg,const mpz_t nn){
        return mpz_cmp_ui(gg,1)>0 && mpz_cmp(gg,nn)<0; };

    /* (3) functional stage-1 cracks: p-1 smooth -> P-1; p+1 smooth -> P+1.
     * pb leaves margin so n = p*q stays < 2^(64K-2) (the Montgomery precondition). */
    {
        mpz_t p,q,n; mpz_inits(p,q,n,NULL);
        int pb=(64*K)/2 - 8;
        craft_smooth(p,-1,B1,pb,rs); mpz_urandomb(q,rs,pb); mpz_nextprime(q,q); mpz_mul(n,p,q);
        std::vector<u64> Gg1,Gg2; gpu_run<K>(false,n,bases,spow,pr,Gg1,Gg2);
        bool got=false; for(size_t i=0;i<bases.size();i++){ from_limbs(z,&Gg1[i*K],K);
            mpz_sub_ui(z,z,1); mpz_mod(z,z,n); mpz_gcd(g,z,n); if(nontrivial(g,n)) got=true; }
        check(got,"P-1 stage1 cracks p-1-smooth composite");

        craft_smooth(p,+1,B1,pb,rs); mpz_urandomb(q,rs,pb); mpz_nextprime(q,q); mpz_mul(n,p,q);
        gpu_run<K>(true,n,bases,spow,pr,Gg1,Gg2);
        got=false; for(size_t i=0;i<bases.size();i++){ from_limbs(z,&Gg1[i*K],K); mpz_gcd(g,z,n);
            if(nontrivial(g,n)) got=true; }
        check(got,"P+1 stage1 cracks p+1-smooth composite (some seed)");
        mpz_clears(p,q,n,NULL);
    }

    /* (3b) functional stage-2 crack: p-1 = smooth * one prime in (B1,B2] */
    {
        mpz_t p,q,n; mpz_inits(p,q,n,NULL);
        int pb=(64*K)/2 - 8;
        craft_stage2(p,B1,B2,pb,rs); mpz_urandomb(q,rs,pb); mpz_nextprime(q,q); mpz_mul(n,p,q);
        std::vector<u64> Gg1,Gg2; gpu_run<K>(false,n,bases,spow,pr,Gg1,Gg2);
        bool got=false; for(size_t i=0;i<bases.size();i++){ from_limbs(z,&Gg2[i*K],K); mpz_gcd(g,z,n);
            if(nontrivial(g,n)) got=true; }
        check(got,"P-1 stage2 cracks p-1=smooth*prime(B1,B2]");
        mpz_clears(p,q,n,NULL);
    }
    mpz_clears(N,ref,z,g,NULL);
}

int main(){
    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs,20260607);
    run_width<2>(2000,50000,rs);
    run_width<4>(2000,50000,rs);
    gmp_randclear(rs);
    printf("\n=== pm1/pp1 validation: %d pass, %d fail ===\n",g_pass,g_fail);
    return g_fail==0?0:1;
}
