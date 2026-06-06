/*
 * gpu-prefactor.cu — standalone GPU ECM pre-factoring front-end (v3.1.0 Track 2.1).
 *
 * Strips small/medium prime factors from a large N on the GPU with batched ECM,
 * BEFORE handing the (reduced) cofactor to NFS — which CADO already advises users
 * to do by hand (README: "strip small prime factors ... before using it"). This
 * sidesteps the Amdahl wall that makes in-sieve GPU cofactorization a no-win
 * (docs/gpu-cofactorization.md): pre-factoring is a *separate* stage, so the GPU's
 * ~39x ECM throughput is pure upside when N has a findable factor.
 *
 * The device math is the bit-exact-validated multi-precision Montgomery ECM
 * (gpu_ecm_mp.cuh; see bench/gpu-ecm-mp.cu). GMP handles N parsing, the
 * per-modulus Montgomery setup (n^{-1}, R mod n, R^2 mod n), and gcd(Z,n).
 *
 *   nvcc -arch=sm_86 -O3 misc/gpu_prefactor/gpu-prefactor.cu -lgmp -o gpu-prefactor
 *   ./gpu-prefactor <N> [B1=50000] [curves=4096]
 *
 * Stage-1 only for now (stage-2 BSGS, Suyama-sigma, and multi-GPU are the next
 * Track 2.1 increments). Exit code 0 if at least one factor was stripped.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <gmp.h>
#include "gpu_ecm_mp.cuh"

static u64 ninv64(u64 n){ u64 x=n; for(int i=0;i<5;i++) x*=2-n*x; return (u64)0-x; }

/* export mpz -> exactly K 64-bit little-endian limbs (zero-padded) */
static void to_limbs(u64 *out, int K, const mpz_t v){
    for(int i=0;i<K;i++) out[i]=0;
    size_t count=0;
    mpz_export(out, &count, -1, 8, 0, 0, v);   /* little-endian, 8-byte words */
}
static void from_limbs(mpz_t v, const u64 *in, int K){
    mpz_import(v, K, -1, 8, 0, 0, in);
}

/* Launch `lanes` ECM curves on N (K limbs) on the GPU; return Z[lane*K..]. */
template<int K>
static bool run_curves(const u64 *Nl, u64 np, const u64 *R1, const u64 *R2,
                       const std::vector<u64> &seeds, const std::vector<u64> &spow,
                       std::vector<u64> &Zout, double &curves_per_s){
    int lanes=(int)seeds.size(), ns=(int)spow.size();
    std::vector<u64> N(lanes*K), R1v(lanes*K), R2v(lanes*K), SEED(lanes*K);
    std::vector<u64> NP(lanes);
    for(int i=0;i<lanes;i++){
        mp_copy<K>(&N[i*K], Nl); mp_copy<K>(&R1v[i*K], R1); mp_copy<K>(&R2v[i*K], R2);
        NP[i]=np;
        u64 *sd=&SEED[i*K]; mp_set0<K>(sd); sd[0]=seeds[i];
    }
    u64 *dN,*dNP,*dR1,*dR2,*dSEED,*ds,*dZ;
    if(cudaMalloc(&dN,(size_t)lanes*K*8)!=cudaSuccess) return false;
    cudaMalloc(&dNP,lanes*8); cudaMalloc(&dR1,(size_t)lanes*K*8);
    cudaMalloc(&dR2,(size_t)lanes*K*8); cudaMalloc(&dSEED,(size_t)lanes*K*8);
    cudaMalloc(&dZ,(size_t)lanes*K*8); cudaMalloc(&ds,ns*8);
    cudaMemcpy(dN,N.data(),(size_t)lanes*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dNP,NP.data(),lanes*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR1,R1v.data(),(size_t)lanes*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dR2,R2v.data(),(size_t)lanes*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(dSEED,SEED.data(),(size_t)lanes*K*8,cudaMemcpyHostToDevice);
    cudaMemcpy(ds,spow.data(),ns*8,cudaMemcpyHostToDevice);
    int tpb=64, blk=(lanes+tpb-1)/tpb;
    auto t0=std::chrono::steady_clock::now();
    ecm_kernel<K><<<blk,tpb>>>(dN,dNP,dR1,dR2,dSEED,ds,ns,dZ,lanes);
    cudaDeviceSynchronize();
    auto t1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    Zout.resize((size_t)lanes*K);
    cudaMemcpy(Zout.data(),dZ,(size_t)lanes*K*8,cudaMemcpyDeviceToHost);
    double sec=std::chrono::duration<double>(t1-t0).count();
    curves_per_s = sec>0 ? lanes/sec : 0;
    cudaFree(dN);cudaFree(dNP);cudaFree(dR1);cudaFree(dR2);cudaFree(dSEED);cudaFree(dZ);cudaFree(ds);
    return e==cudaSuccess;
}

/* One GPU ECM pass on N at the chosen width K; append any nontrivial gcd factors. */
template<int K>
static bool ecm_pass(const mpz_t N, unsigned long B1, int ncurves,
                     const std::vector<u64> &spow,
                     std::vector<std::string> &found, double &cps){
    (void)B1;
    u64 Nl[K]; to_limbs(Nl, K, N);
    u64 np = ninv64(Nl[0]);
    mpz_t t,R1m,R2m,g,z; mpz_inits(t,R1m,R2m,g,z,NULL);
    mpz_setbit(t, 64*K); mpz_mod(R1m, t, N);              /* R mod n */
    mpz_set_ui(t,0); mpz_setbit(t,128*K); mpz_mod(R2m, t, N); /* R^2 mod n */
    u64 R1[K],R2[K]; to_limbs(R1,K,R1m); to_limbs(R2,K,R2m);
    std::vector<u64> seeds(ncurves);
    for(int i=0;i<ncurves;i++) seeds[i]=(u64)(2+i*2654435761u % 1000003);  /* varied a24 */
    std::vector<u64> Z;
    bool ok = run_curves<K>(Nl,np,R1,R2,seeds,spow,Z,cps);
    bool any=false;
    if(ok){
        for(int i=0;i<ncurves;i++){
            from_limbs(z,&Z[(size_t)i*K],K);
            mpz_gcd(g,z,N);
            if(mpz_cmp_ui(g,1)>0 && mpz_cmp(g,N)<0){
                char *s=mpz_get_str(NULL,10,g);
                std::string fac(s); free(s);
                bool dup=false; for(auto&f:found) if(f==fac) dup=true;
                if(!dup){ found.push_back(fac); any=true; }
            }
        }
    }
    mpz_clears(t,R1m,R2m,g,z,NULL);
    return any;
}

int main(int argc, char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <N> [B1=50000] [curves=4096]\n",argv[0]); return 2; }
    mpz_t N; mpz_init(N);
    if(mpz_set_str(N, argv[1], 10)!=0){ fprintf(stderr,"bad N\n"); return 2; }
    unsigned long B1 = argc>2 ? strtoul(argv[2],0,10) : 50000;
    int ncurves      = argc>3 ? atoi(argv[3]) : 4096;

    size_t bits = mpz_sizeinbase(N,2);
    int needK = (int)((bits+2+63)/64);
    int K = needK<=2?2: needK<=4?4: needK<=8?8: needK<=16?16: 0;
    if(K==0){ fprintf(stderr,"N too large (%zu bits); supported up to 1022 bits (~307 digits)\n",bits); return 2; }
    if(mpz_even_p(N)){ fprintf(stderr,"N is even; strip factors of 2 first\n"); return 2; }

    /* prime powers <= B1 */
    std::vector<u64> spow; std::vector<char> comp(B1+1,0);
    for(unsigned long p=2;p<=B1;p++) if(!comp[p]){ for(unsigned long q=p*p;q<=B1;q+=p) comp[q]=1;
        u64 pe=p; while(pe*p<=B1) pe*=p; spow.push_back(pe); }

    printf("# GPU ECM pre-factor: %zu-bit N (~%zu digits), width K=%d (%d-bit), B1=%lu, curves=%d, %zu multipliers\n",
           bits, mpz_sizeinbase(N,10), K, 64*K, B1, ncurves, spow.size());

    std::vector<std::string> found; double cps=0; bool any=false;
    switch(K){
        case 2:  any=ecm_pass<2 >(N,B1,ncurves,spow,found,cps); break;
        case 4:  any=ecm_pass<4 >(N,B1,ncurves,spow,found,cps); break;
        case 8:  any=ecm_pass<8 >(N,B1,ncurves,spow,found,cps); break;
        case 16: any=ecm_pass<16>(N,B1,ncurves,spow,found,cps); break;
    }
    printf("# throughput: %.0f curves/s on the GPU\n", cps);

    /* divide out everything found; report factors + remaining cofactor */
    mpz_t cof; mpz_init_set(cof,N);
    for(auto &f: found){ mpz_t fz; mpz_init_set_str(fz,f.c_str(),10);
        while(mpz_divisible_p(cof,fz)) mpz_divexact(cof,cof,fz);
        mpz_clear(fz); }
    if(any){
        printf("factors stripped:");
        for(auto &f: found) printf(" %s", f.c_str());
        printf("\n");
        char *c=mpz_get_str(NULL,10,cof);
        printf("remaining cofactor: %s%s\n", c,
               mpz_probab_prime_p(cof,25)? "  (prime)" :
               (mpz_cmp_ui(cof,1)==0? "  (fully factored)" : "  (composite -> hand to NFS)"));
        free(c);
    } else {
        printf("no factor found at B1=%lu with %d curves (try a larger B1 / more curves)\n",
               B1, ncurves);
    }
    mpz_clear(cof); mpz_clear(N);
    return any?0:1;
}
