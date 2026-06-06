/*
 * gpu-spmv-bench.cu — GF(2) sparse-matrix x block-of-vectors product on the GPU,
 * the core kernel of Block Wiedemann linear algebra (linalg/bwc), validated
 * bit-exact against a CPU reference and benchmarked vs the full CPU.
 *
 * This is the foundation for a GPU matmul backend (v3.1.0 Track 2.2). BWC's SpMV
 * (matmul-basic.cpp): the matrix is rows of column indices, and per nonzero
 * (i,j) the op is dst[i] ^= src[j], where each vector element is a bitsliced
 * block of K 64-bit limbs (b64: K=1 / 64 vectors, b128: K=2 / 128 vectors). It
 * runs thousands of SpMV iterations on a fixed matrix, so the matrix is resident
 * on the GPU and we time the kernel only (matrix transfer is one-time).
 *
 *   nvcc -arch=sm_86 -O3 -Xcompiler -pthread bench/gpu-spmv-bench.cu \
 *        -o /tmp/gpu-spmv-bench && /tmp/gpu-spmv-bench
 *
 * The matrix is synthetic (random CSR, ~avg_nnz nonzeros/row) but the SpMV
 * semantics are exactly matmul-basic's; a real BWC matrix has the same shape.
 */
#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>
#include <thread>
typedef uint64_t u64;
typedef uint32_t u32;

static u64 xr(u64*s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

/* GPU SpMV: one thread per output row. dst[i] = XOR_{j in row i} src[j], each a
 * K-limb GF(2) block. Matrix in CSR (rowptr[nrows+1], col[nnz]). */
template<int K>
__global__ void spmv(const u32* rowptr, const u32* col, const u64* src,
                     u64* dst, int nrows){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i>=nrows) return;
    u64 acc[K];
    #pragma unroll
    for(int k=0;k<K;k++) acc[k]=0;
    u32 lo=rowptr[i], hi=rowptr[i+1];
    for(u32 p=lo;p<hi;p++){
        const u64* s = src + (size_t)col[p]*K;
        #pragma unroll
        for(int k=0;k<K;k++) acc[k]^=s[k];
    }
    u64* d = dst + (size_t)i*K;
    #pragma unroll
    for(int k=0;k<K;k++) d[k]=acc[k];
}

/* CPU reference (also used multi-threaded for the throughput comparison) */
template<int K>
static void cpu_spmv(const u32* rowptr, const u32* col, const u64* src,
                     u64* dst, int r0, int r1){
    for(int i=r0;i<r1;i++){
        u64 acc[K]; for(int k=0;k<K;k++) acc[k]=0;
        for(u32 p=rowptr[i];p<rowptr[i+1];p++){
            const u64* s=src+(size_t)col[p]*K;
            for(int k=0;k<K;k++) acc[k]^=s[k];
        }
        for(int k=0;k<K;k++) dst[(size_t)i*K+k]=acc[k];
    }
}

template<int K>
static int run(const char* label, int n, int avg_nnz, int iters){
    /* ---- build a random CSR matrix (n x n, ~avg_nnz nonzeros/row) ---- */
    u64 st=0xA5A5F00DULL + K;
    std::vector<u32> rowptr(n+1), col;
    col.reserve((size_t)n*avg_nnz);
    rowptr[0]=0;
    for(int i=0;i<n;i++){
        int d = avg_nnz/2 + (int)(xr(&st)%(avg_nnz+1));   /* vary row weight */
        for(int t=0;t<d;t++) col.push_back((u32)(xr(&st)%n));
        rowptr[i+1]=(u32)col.size();
    }
    size_t nnz=col.size();
    std::vector<u64> src((size_t)n*K), dst_g((size_t)n*K), dst_c((size_t)n*K);
    for(auto& v:src) v=xr(&st);

    /* ---- GPU: matrix+src resident, time the kernel only ---- */
    u32 *drp,*dcol; u64 *dsrc,*ddst;
    if(cudaMalloc(&drp,(n+1)*4)!=cudaSuccess){ printf("  [%s] cudaMalloc failed\n",label); return 1; }
    cudaMalloc(&dcol,nnz*4); cudaMalloc(&dsrc,src.size()*8); cudaMalloc(&ddst,dst_g.size()*8);
    cudaMemcpy(drp,rowptr.data(),(n+1)*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dcol,col.data(),nnz*4,cudaMemcpyHostToDevice);
    cudaMemcpy(dsrc,src.data(),src.size()*8,cudaMemcpyHostToDevice);
    int tpb=128, blk=(n+tpb-1)/tpb;
    spmv<K><<<blk,tpb>>>(drp,dcol,dsrc,ddst,n); cudaDeviceSynchronize();  /* warm */
    auto g0=std::chrono::steady_clock::now();
    for(int it=0;it<iters;it++) spmv<K><<<blk,tpb>>>(drp,dcol,dsrc,ddst,n);
    cudaDeviceSynchronize();
    auto g1=std::chrono::steady_clock::now();
    cudaError_t e=cudaGetLastError();
    cudaMemcpy(dst_g.data(),ddst,dst_g.size()*8,cudaMemcpyDeviceToHost);
    double gsec=std::chrono::duration<double>(g1-g0).count()/iters;
    cudaFree(drp);cudaFree(dcol);cudaFree(dsrc);cudaFree(ddst);

    /* ---- CPU reference (single thread) for bit-exact validation ---- */
    cpu_spmv<K>(rowptr.data(),col.data(),src.data(),dst_c.data(),0,n);
    long mis=0; for(size_t i=0;i<dst_c.size();i++) if(dst_c[i]!=dst_g[i]) mis++;

    /* ---- CPU throughput, all cores ---- */
    int nthr=(int)std::thread::hardware_concurrency(); if(nthr<1) nthr=1;
    auto c0=std::chrono::steady_clock::now();
    for(int it=0;it<iters;it++){
        std::vector<std::thread> th;
        for(int t=0;t<nthr;t++) th.emplace_back([&,t]{
            int r0=(long)t*n/nthr, r1=(long)(t+1)*n/nthr;
            cpu_spmv<K>(rowptr.data(),col.data(),src.data(),dst_c.data(),r0,r1);
        });
        for(auto&x:th) x.join();
    }
    auto c1=std::chrono::steady_clock::now();
    double csec=std::chrono::duration<double>(c1-c0).count()/iters;

    double gnz_g=nnz/gsec/1e9, gnz_c=nnz/csec/1e9;
    printf("  [%s] n=%d nnz=%zu (avg %d/row): validation %s (%ld/%zu words)\n",
           label, n, nnz, avg_nnz, mis==0?"PASS":"FAIL", mis, dst_c.size());
    printf("        GPU %6.2f Gnz/s (%.2f ms) | CPU(%2d thr) %5.2f Gnz/s (%.2f ms) | speedup %4.1fx%s\n",
           gnz_g, gsec*1e3, nthr, gnz_c, csec*1e3, gnz_c>0?gnz_g/gnz_c:0, e?"  CUDAERR":"");
    return mis!=0;
}

int main(){
    printf("GF(2) sparse matrix x block-of-vectors (BWC SpMV) — RTX 3090 vs full CPU\n");
    int fails=0;
    fails += run<1>("b64 ", 2000000, 30, 50);   /* 64 vectors, ~c110-scale matrix */
    fails += run<2>("b128", 2000000, 30, 50);   /* 128 vectors */
    fails += run<4>("b256",  500000, 60, 50);   /* 256 vectors, denser */
    printf("%s\n", fails==0?"ALL PASS":"FAILURES");
    return fails!=0;
}
