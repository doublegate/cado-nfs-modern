/*
 * avx2-modinv.c — AVX2 8-way batched 32-bit modular inverse (Roadmap B4).
 *
 * The AVX2 sibling of the SDE-only AVX-512 kernel (bench/avx512-modinv.c, B1).
 * Same target: the siever's per-prime modular inverse (invmod_redc_32, feeding
 * plattice_info) — ~9.5% + part of plattice in a c120 profile — which is the one
 * genuinely vectorizable hot slice (the byte-scatter majority does not vectorize:
 * no 8-bit scatter on AVX2 or AVX-512). Each factor-base prime is a DIFFERENT
 * modulus, so Montgomery's batch-inversion trick (1 inverse + 3n muls in ONE ring)
 * does NOT apply; the lanes run independent binary-extended-GCD inverses instead.
 *
 * Why a separate AVX2 kernel at all: the reference box (Comet Lake i9-10850K) has
 * AVX2 but NO AVX-512, so B1's win is only ever SDE-validated there. This one
 * RUNS on the silicon, so it yields the fork's first *measured* batched-modinv
 * SIMD number — see the scalar-vs-AVX2 timing at the bottom.
 *
 * Port note: AVX-512 uses k-mask registers + masked ops; AVX2 has none, so the
 * per-lane state machine uses vector masks (all-ones/all-zero per lane) and
 * _mm256_blendv_epi8 for conditional moves. U,V,X1,X2 stay < 2^31 (b < 2^31, the
 * X's live in [0,B)), so the high bit is always clear and signed AVX2 compares
 * (_mm256_cmpgt_epi32) coincide with the unsigned compares the algorithm needs.
 *
 *   gcc -O2 -mavx2 bench/avx2-modinv.c -lgmp -o avx2-modinv && ./avx2-modinv
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <gmp.h>

#define LANES 8

/* scalar reference: binary extended-GCD inverse a^-1 mod b (b odd, gcd=1). */
static uint32_t modinv1(uint32_t a, uint32_t b)
{
    uint32_t U = a, V = b, B = b, X1 = 1, X2 = 0;
    while (U != 1 && V != 1) {
        if ((U & 1) == 0) {
            U >>= 1;
            X1 = (X1 & 1) ? (X1 + B) >> 1 : X1 >> 1;
        } else if ((V & 1) == 0) {
            V >>= 1;
            X2 = (X2 & 1) ? (X2 + B) >> 1 : X2 >> 1;
        } else if (U >= V) {
            U -= V;
            X1 = (X1 >= X2) ? X1 - X2 : X1 + B - X2;
        } else {
            V -= U;
            X2 = (X2 >= X1) ? X2 - X1 : X2 + B - X1;
        }
    }
    uint32_t res = (U == 1) ? X1 : X2;
    if (res >= B) res -= B;
    return res;
}

#if defined(__AVX2__)
#include <immintrin.h>

static inline __m256i is_odd(__m256i x, __m256i one)
{
    return _mm256_cmpeq_epi32(_mm256_and_si256(x, one), one);
}

/* 8-way batched modular inverse: r[L] = a[L]^-1 mod b[L], b odd, gcd=1, b<2^31. */
static void modinv8(uint32_t *r, const uint32_t *a, const uint32_t *b)
{
    const __m256i one  = _mm256_set1_epi32(1);
    const __m256i ones = _mm256_set1_epi32(-1);
    __m256i U = _mm256_loadu_si256((const __m256i *)a);
    __m256i V = _mm256_loadu_si256((const __m256i *)b);
    __m256i B = V;
    __m256i X1 = one, X2 = _mm256_setzero_si256();

    for (int it = 0; it < 4096; it++) {
        __m256i done = _mm256_or_si256(_mm256_cmpeq_epi32(U, one),
                                       _mm256_cmpeq_epi32(V, one));
        if ((unsigned)_mm256_movemask_epi8(done) == 0xFFFFFFFFu) break;
        __m256i act = _mm256_xor_si256(done, ones);          /* ~done */
        __m256i Uodd = is_odd(U, one);
        __m256i Vodd = is_odd(V, one);
        __m256i ueven = _mm256_andnot_si256(Uodd, act);       /* act & ~Uodd */
        __m256i veven = _mm256_and_si256(                     /* act & Uodd & ~Vodd */
            _mm256_andnot_si256(Vodd, act), Uodd);
        __m256i bothodd = _mm256_and_si256(act,
            _mm256_and_si256(Uodd, Vodd));

        /* branch 1: U even -> U>>=1 ; X1 = halve_mod(X1,B) */
        {
            __m256i Uh = _mm256_srli_epi32(U, 1);
            U = _mm256_blendv_epi8(U, Uh, ueven);
            __m256i x1odd = is_odd(X1, one);
            __m256i xe = _mm256_srli_epi32(X1, 1);
            __m256i xo = _mm256_srli_epi32(_mm256_add_epi32(X1, B), 1);
            __m256i xh = _mm256_blendv_epi8(xe, xo, x1odd);
            X1 = _mm256_blendv_epi8(X1, xh, ueven);
        }
        /* branch 2: U odd & V even -> V>>=1 ; X2 = halve_mod(X2,B) */
        {
            __m256i Vh = _mm256_srli_epi32(V, 1);
            V = _mm256_blendv_epi8(V, Vh, veven);
            __m256i x2odd = is_odd(X2, one);
            __m256i xe = _mm256_srli_epi32(X2, 1);
            __m256i xo = _mm256_srli_epi32(_mm256_add_epi32(X2, B), 1);
            __m256i xh = _mm256_blendv_epi8(xe, xo, x2odd);
            X2 = _mm256_blendv_epi8(X2, xh, veven);
        }
        /* both odd: subtract smaller from larger (U,V<2^31 -> signed cmp ok). */
        __m256i vgtU = _mm256_cmpgt_epi32(V, U);              /* V > U */
        __m256i ugeV = _mm256_and_si256(bothodd,
            _mm256_andnot_si256(vgtU, ones));                 /* bothodd & U>=V */
        __m256i vgt = _mm256_and_si256(bothodd, vgtU);        /* bothodd & V>U */
        /* branch U>=V: U-=V ; X1 = submod(X1,X2,B) */
        {
            U = _mm256_blendv_epi8(U, _mm256_sub_epi32(U, V), ugeV);
            __m256i d = _mm256_sub_epi32(X1, X2);
            __m256i dpb = _mm256_add_epi32(d, B);
            __m256i lt = _mm256_cmpgt_epi32(X2, X1);          /* X1 < X2 */
            __m256i sm = _mm256_blendv_epi8(d, dpb, lt);
            X1 = _mm256_blendv_epi8(X1, sm, ugeV);
        }
        /* branch V>U: V-=U ; X2 = submod(X2,X1,B) */
        {
            V = _mm256_blendv_epi8(V, _mm256_sub_epi32(V, U), vgt);
            __m256i d = _mm256_sub_epi32(X2, X1);
            __m256i dpb = _mm256_add_epi32(d, B);
            __m256i lt = _mm256_cmpgt_epi32(X1, X2);          /* X2 < X1 */
            __m256i sm = _mm256_blendv_epi8(d, dpb, lt);
            X2 = _mm256_blendv_epi8(X2, sm, vgt);
        }
    }
    __m256i uone = _mm256_cmpeq_epi32(U, one);
    __m256i res = _mm256_blendv_epi8(X2, X1, uone);
    __m256i ge = _mm256_andnot_si256(_mm256_cmpgt_epi32(B, res), ones); /* res>=B */
    res = _mm256_blendv_epi8(res, _mm256_sub_epi32(res, B), ge);
    _mm256_storeu_si256((__m256i *)r, res);
}
#endif

static uint64_t xrnd(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

static double now_s(void)
{
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

int main(void)
{
#if defined(__AVX2__)
    mpz_t ga, gb, gi; mpz_inits(ga, gb, gi, NULL);
    uint64_t seed = 0x5EED1234ULL;
    long trials = 0, wrong = 0;
    for (int rep = 0; rep < 40000; rep++) {
        uint32_t a[LANES], b[LANES], r[LANES];
        for (int L = 0; L < LANES; L++) {
            uint32_t bb, aa;
            do {
                bb = (uint32_t)(xrnd(&seed) & 0x7FFFFFFF) | 1u;   /* odd, < 2^31 */
                if (bb < 3) bb = 3;
                aa = (uint32_t)(xrnd(&seed) % bb);
                if (aa == 0) aa = 1;
                mpz_set_ui(ga, aa); mpz_set_ui(gb, bb);
            } while (mpz_invert(gi, ga, gb) == 0);   /* require gcd(a,b)==1 */
            a[L] = aa; b[L] = bb;
        }
        modinv8(r, a, b);
        for (int L = 0; L < LANES; L++) {
            mpz_set_ui(ga, a[L]); mpz_set_ui(gb, b[L]);
            mpz_invert(gi, ga, gb);
            uint32_t expect = (uint32_t) mpz_get_ui(gi);
            trials++;
            if (r[L] != expect) {
                if (wrong < 4)
                    printf("  MISMATCH lane %d: a=%u b=%u got=%u expect=%u\n",
                           L, a[L], b[L], r[L], expect);
                wrong++;
            }
        }
    }
    printf("AVX2 8-way batched 32-bit modular inverse vs GMP, %ld trials:\n", trials);
    printf("  %s (%ld wrong)\n", wrong ? "FAIL" : "PASS", wrong);

    /* --- measured speedup on real silicon: scalar vs AVX2 over the same data --- */
    const int N = 1 << 20;               /* ~1M inverses, multiple of LANES */
    uint32_t *A = malloc(N * 4), *Bb = malloc(N * 4), *Rs = malloc(N * 4),
             *Rv = malloc(N * 4);
    for (int i = 0; i < N; i++) {
        uint32_t bb = (uint32_t)(xrnd(&seed) & 0x7FFFFFFF) | 1u;
        if (bb < 3) bb = 3;
        uint32_t aa;
        for (;;) {
            aa = (uint32_t)(xrnd(&seed) % bb); if (aa == 0) aa = 1;
            mpz_set_ui(ga, aa); mpz_set_ui(gb, bb);
            if (mpz_invert(gi, ga, gb)) break;
        }
        A[i] = aa; Bb[i] = bb;
    }
    double t0 = now_s();
    for (int i = 0; i < N; i++) Rs[i] = modinv1(A[i], Bb[i]);
    double ts = now_s() - t0;
    double t1 = now_s();
    for (int i = 0; i < N; i += LANES) modinv8(&Rv[i], &A[i], &Bb[i]);
    double tv = now_s() - t1;
    long mism = 0;
    for (int i = 0; i < N; i++) if (Rs[i] != Rv[i]) mism++;
    printf("\nMeasured on this CPU (%d inverses):\n", N);
    printf("  scalar : %.1f ns/inv  (%.3f s)\n", ts * 1e9 / N, ts);
    printf("  AVX2 x8: %.1f ns/inv  (%.3f s)\n", tv * 1e9 / N, tv);
    printf("  speedup: %.2fx   scalar-vs-AVX2 mismatches: %ld\n", ts / tv, mism);

    free(A); free(Bb); free(Rs); free(Rv);
    mpz_clears(ga, gb, gi, NULL);
    printf("%s\n", (wrong || mism) ? "FAILURES" : "ALL PASS");
    return (wrong || mism) != 0;
#else
    (void) modinv1; (void) xrnd; (void) now_s;
    printf("avx2-modinv: built without AVX2; build with -mavx2.\n");
    return 0;
#endif
}
