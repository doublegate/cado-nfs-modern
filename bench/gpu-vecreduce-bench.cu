/*
 * gpu-vecreduce-bench.cu — GPU intra-node vector reduction for Block Wiedemann,
 * the third (and last standalone) compute primitive of the full-vector-residency
 * port (v3.1.0-modern, Track 2.2, step 1).
 *
 * On a single node with T cores in a direction, the BWC comm
 * (mmt_vec_allreduce / mmt_vec_reduce in linalg/bwc/matmul_top_comm.cpp) reduces
 * the T per-core "sibling" vectors and broadcasts the result. For GF(2) the
 * reduction is `vec_add_and_reduce`, which arith-mod2.hpp implements as plain
 * `vec_add` = element-wise XOR (no carry/reduction over F_2). So:
 *     result[g] = XOR over t<T of sibling[t][g]          (reduce)
 *     sibling[t][g] = result[g] for all t                (broadcast)
 * Today this runs on the host every iteration; this is the device version, so a
 * device-resident vector's comm need not return to the CPU (single-node case).
 *
 * Same __host__ __device__ reduction on CPU and GPU => validated bit-exact.
 *   nvcc -arch=sm_86 -O3 bench/gpu-vecreduce-bench.cu -o /tmp/gpu-vecred && /tmp/gpu-vecred
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
typedef uint64_t u64;

static u64 xr(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* reduce: out[g] = XOR over t<T of in[t*words + g]. One thread per word g. */
__global__ void vecreduce(const u64 *in, u64 *out, unsigned T, size_t words){
    size_t g = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(g>=words) return;
    u64 acc=0;
    for(unsigned t=0;t<T;t++) acc ^= in[(size_t)t*words + g];
    out[g]=acc;
}
/* broadcast: in[t*words + g] = src[g] for all t. */
__global__ void vecbroadcast(u64 *inout, const u64 *src, unsigned T, size_t words){
    size_t g = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if(g>=words) return;
    u64 v=src[g];
    for(unsigned t=0;t<T;t++) inout[(size_t)t*words + g]=v;
}

static int run(const char* label, unsigned T, size_t words, int iters){
    std::vector<u64> in((size_t)T*words);
    u64 st=0xC0FFEEULL ^ (u64)words ^ (u64)T; if(!st) st=1;
    for(auto&x:in) x=xr(&st);
    std::vector<u64> outg(words), outc(words), bcg((size_t)T*words);

    u64 *din,*dout;
    if(cudaMalloc(&din,in.size()*8)!=cudaSuccess){ printf("  [%s] malloc fail\n",label); return 1; }
    cudaMalloc(&dout,words*8);
    cudaMemcpy(din,in.data(),in.size()*8,cudaMemcpyHostToDevice);
    int tpb=256; size_t blk=(words+tpb-1)/tpb;
    vecreduce<<<blk,tpb>>>(din,dout,T,words); cudaDeviceSynchronize();   /* warm */
    auto t0=std::chrono::steady_clock::now();
    for(int it=0;it<iters;it++) vecreduce<<<blk,tpb>>>(din,dout,T,words);
    cudaDeviceSynchronize();
    auto t1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    cudaMemcpy(outg.data(),dout,words*8,cudaMemcpyDeviceToHost);
    /* broadcast the reduced result back over the T siblings, check it too */
    vecbroadcast<<<blk,tpb>>>(din,dout,T,words); cudaDeviceSynchronize();
    cudaMemcpy(bcg.data(),din,bcg.size()*8,cudaMemcpyDeviceToHost);
    double sec=std::chrono::duration<double>(t1-t0).count()/iters;
    cudaFree(din);cudaFree(dout);

    /* CPU reference */
    for(size_t g=0;g<words;g++){ u64 a=0; for(unsigned t=0;t<T;t++) a^=in[(size_t)t*words+g]; outc[g]=a; }
    long mis=0; for(size_t g=0;g<words;g++) if(outc[g]!=outg[g]) mis++;
    long bmis=0; for(unsigned t=0;t<T;t++) for(size_t g=0;g<words;g++) if(bcg[(size_t)t*words+g]!=outc[g]) bmis++;

    double bytes = (double)(T+1)*words*8;     /* read T, write 1 */
    printf("  [%s] T=%u words=%zu : reduce %s, broadcast %s (%ld/%ld differ) ; %.0f GB/s%s\n",
           label, T, words, mis==0?"PASS":"FAIL", bmis==0?"PASS":"FAIL", mis, bmis,
           bytes/sec/1e9, e?"  CUDAERR":"");
    return (mis!=0)||(bmis!=0);
}

int main(){
    printf("GPU intra-node vector reduce+broadcast (GF(2) XOR) — validated bit-exact vs CPU\n");
    int f=0;
    f += run("2x (b64, 2M rows) ", 2, 2000000, 100);   /* 2 cores in the direction */
    f += run("4x (b64, 2M rows) ", 4, 2000000, 100);   /* 2x2 grid -> up to 4 */
    f += run("4x (b128, 2M rows)", 4, 4000000, 100);   /* b128: 2 limbs/elt */
    printf("%s\n", f==0?"ALL PASS":"FAILURES");
    return f!=0;
}
