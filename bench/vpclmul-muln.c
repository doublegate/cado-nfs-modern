/*
 * vpclmul-muln.c — AVX-512 VPCLMULQDQ ports of gf2x's small fixed-size kernels
 * gf2x_mul2 / gf2x_mul3 / gf2x_mul4, with scalar references and a bit-exactness
 * self-test (Roadmap B2; companion to bench/vpclmul-mul1n.c which did mul_1_n).
 *
 * gf2x_mulN(c, a, b): c[0..2N-1] = a[0..N-1] * b[0..N-1] over GF(2)[x] (each word
 * a degree-<64 polynomial). The PCLMUL versions (gf2x/already_tuned/x86_64_pclmul)
 * use one 64x64 carryless multiply per `_mm_clmulepi64_si128`. VPCLMULQDQ's
 * `_mm512_clmulepi64_epi128` does FOUR per instruction (one per 128-bit lane), so
 * the independent Karatsuba base products are packed into the lanes and produced
 * in a single instruction. Ref: Drucker & Gueron, arXiv:2201.10473.
 *
 * The honest win for these *fixed tiny* sizes is modest (few clmuls fused, vs the
 * cost of assembling the operand zmm) — the big DG win is the variable-length
 * mul_1_n (already shipped). These are validated for CORRECTNESS only; perf is
 * gated on real AVX-512 silicon (this box is Comet Lake — no AVX-512).
 *
 *   gcc -O2 -mavx512f -mvpclmulqdq bench/vpclmul-muln.c -o /tmp/vpclmul-muln
 *   sde64 -future -- /tmp/vpclmul-muln          # validate under emulation
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- scalar reference: schoolbook GF(2)[x] product (ground truth) ------- */
static void clmul64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi)
{
    uint64_t l = 0, h = 0;
    for (int i = 0; i < 64; i++)
        if ((b >> i) & 1ULL) { l ^= a << i; if (i) h ^= a >> (64 - i); }
    *lo = l; *hi = h;
}
static void ref_muln(uint64_t *c, const uint64_t *a, const uint64_t *b, int N)
{
    for (int i = 0; i < 2 * N; i++) c[i] = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            uint64_t lo, hi; clmul64(a[i], b[j], &lo, &hi);
            c[i + j] ^= lo; c[i + j + 1] ^= hi;
        }
}

/* ---------- AVX-512 VPCLMULQDQ implementations (AVX-512F + VPCLMULQDQ only) ----
 * These use only _mm512_set_epi64 / _mm512_clmulepi64_epi128 / _mm512_storeu_si512
 * (no AVX-512DQ/VL insert/extract), so they compile with the same
 * `-mavx512f -mvpclmulqdq` flags the gf2x vpclmul backend already adds. Each base
 * product is placed in the low 64 of one 128-bit lane and read back from the
 * stored 8-word buffer, exactly like gf2x_mul1.h's mul_1_n. */
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
#include <immintrin.h>

/* gf2x_mul2: 128x128 -> 256, Karatsuba (3 products) in ONE VPCLMULQDQ.
 * lanes: 0:a0*b0  1:a1*b1  2:(a0^a1)*(b0^b1) ; imm 0x00 = lane.a.lo * lane.b.lo */
static inline void vmul2(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
    __m512i A = _mm512_set_epi64(0,0,0,(long long)(a[0]^a[1]), 0,(long long)a[1], 0,(long long)a[0]);
    __m512i B = _mm512_set_epi64(0,0,0,(long long)(b[0]^b[1]), 0,(long long)b[1], 0,(long long)b[0]);
    uint64_t t[8]; _mm512_storeu_si512((void*)t, _mm512_clmulepi64_epi128(A, B, 0x00));
    uint64_t tklo = t[0]^t[2]^t[4], tkhi = t[1]^t[3]^t[5];   /* tk = P0^P1^P2 */
    c[0]=t[0];        c[1]=t[1]^tklo;       /* P0 ^ (tk<<64) */
    c[2]=t[2]^tkhi;   c[3]=t[3];            /* P1 ^ (tk>>64) */
}

/* gf2x_mul4: 256x256 -> 512, Karatsuba on top of mul2 (3 mul2 = 9 products). */
static inline void vmul4(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
    uint64_t L[4], H[4], Mt[4];
    uint64_t am[2] = { a[0]^a[2], a[1]^a[3] }, bm[2] = { b[0]^b[2], b[1]^b[3] };
    vmul2(L, a, b); vmul2(H, a+2, b+2); vmul2(Mt, am, bm);
    for (int i = 0; i < 4; i++) Mt[i] ^= L[i] ^ H[i];
    c[0]=L[0]; c[1]=L[1];
    c[2]=L[2]^Mt[0]; c[3]=L[3]^Mt[1];
    c[4]=H[0]^Mt[2]; c[5]=H[1]^Mt[3];
    c[6]=H[2]; c[7]=H[3];
}

/* gf2x_mul3: 192x192 -> 384, 6-product Karatsuba in two VPCLMULQDQ calls.
 * p0=a0b0,p1=a1b1,p2=a2b2, p01=(a0^a1)(b0^b1), p02=(a0^a2)(b0^b2), p12=(a1^a2)(b1^b2).
 * limbs (128-bit, staggered 64): e0=p0; e1=p0^p1^p01; e2=p0^p1^p2^p02;
 * e3=p1^p2^p12; e4=p2 ; c[j..j+1] ^= e_j. */
static inline void vmul3(uint64_t *c, const uint64_t *a, const uint64_t *b)
{
    uint64_t a0=a[0],a1=a[1],a2=a[2], b0=b[0],b1=b[1],b2=b[2];
    __m512i A1 = _mm512_set_epi64(0,(long long)(a0^a1), 0,(long long)a2, 0,(long long)a1, 0,(long long)a0);
    __m512i B1 = _mm512_set_epi64(0,(long long)(b0^b1), 0,(long long)b2, 0,(long long)b1, 0,(long long)b0);
    __m512i A2 = _mm512_set_epi64(0,0, 0,0, 0,(long long)(a1^a2), 0,(long long)(a0^a2));
    __m512i B2 = _mm512_set_epi64(0,0, 0,0, 0,(long long)(b1^b2), 0,(long long)(b0^b2));
    uint64_t r1[8], r2[8];
    _mm512_storeu_si512((void*)r1, _mm512_clmulepi64_epi128(A1, B1, 0x00));
    _mm512_storeu_si512((void*)r2, _mm512_clmulepi64_epi128(A2, B2, 0x00));
    /* r1: p0=r1[0,1] p1=r1[2,3] p2=r1[4,5] p01=r1[6,7] ; r2: p02=r2[0,1] p12=r2[2,3] */
    uint64_t e[5][2];
    for (int i=0;i<2;i++){
        uint64_t p0=r1[i], p1=r1[2+i], p2=r1[4+i], p01=r1[6+i], p02=r2[i], p12=r2[2+i];
        e[0][i]=p0;
        e[1][i]=p0^p1^p01;
        e[2][i]=p0^p1^p2^p02;
        e[3][i]=p1^p2^p12;
        e[4][i]=p2;
    }
    for (int i=0;i<6;i++) c[i]=0;
    for (int j=0;j<5;j++){ c[j]^=e[j][0]; c[j+1]^=e[j][1]; }
}
#endif

/* ---------- self-test ---------- */
static uint64_t rng(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

int main(void)
{
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
    const int TRIALS = 200000;
    uint64_t s = 0xC0FFEEULL;
    long bad2=0, bad3=0, bad4=0;
    for (int t = 0; t < TRIALS; t++) {
        uint64_t a[4], b[4], cref[8], cgot[8];
        for (int i = 0; i < 4; i++) { a[i]=rng(&s); b[i]=rng(&s); }
        ref_muln(cref, a, b, 2); vmul2(cgot, a, b);
        if (memcmp(cref, cgot, 4*8)) bad2++;
        ref_muln(cref, a, b, 3); vmul3(cgot, a, b);
        if (memcmp(cref, cgot, 6*8)) bad3++;
        ref_muln(cref, a, b, 4); vmul4(cgot, a, b);
        if (memcmp(cref, cgot, 8*8)) bad4++;
    }
    printf("VPCLMULQDQ gf2x small kernels vs scalar GF(2)[x] reference, %d trials:\n", TRIALS);
    printf("  mul2 (128x128): %s (%ld wrong)\n", bad2?"FAIL":"PASS", bad2);
    printf("  mul3 (192x192): %s (%ld wrong)\n", bad3?"FAIL":"PASS", bad3);
    printf("  mul4 (256x256): %s (%ld wrong)\n", bad4?"FAIL":"PASS", bad4);
    int fail = (bad2||bad3||bad4);
    printf("%s\n", fail?"FAILURES":"ALL PASS");
    return fail?1:0;
#else
    printf("vpclmul-muln: built without AVX-512 VPCLMULQDQ; nothing to test here.\n");
    printf("Build with -mavx512f -mvpclmulqdq -mavx512vl and run under sde64 -future.\n");
    return 0;
#endif
}
