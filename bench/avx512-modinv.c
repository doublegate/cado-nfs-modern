/*
 * avx512-modinv.c — AVX-512 16-way batched 32-bit modular inverse (Roadmap B1),
 * the vectorizable arithmetic core of the siever's per-prime lattice setup.
 *
 * WHY this and not "vectorize the sieve loop": a c120 perf profile (v3.1.0,
 * Track 1.3) puts the siever self-time in fill_in_buckets (~12%, scatter),
 * plattice_info ctor (~11%, modular arith), sieve_small_bucket_region (~10%,
 * byte scatter), invmod_redc_32 (~9.5%, modular inverse), apply_buckets_inner
 * (~7%, scatter). The scatter-bound loops do NOT vectorize on AVX-512 either:
 * the sieve array is uint8 and AVX-512 scatter is 32/64-bit-element only (no
 * 8-bit scatter) — confirming the 3.1.0 "AVX2-on-siever ruled out" finding.
 * The one genuinely vectorizable hot slice is the per-prime MODULAR INVERSE
 * (invmod_redc_32, feeding plattice_info) — pure 32-bit arithmetic, ~9.5% + part
 * of the 11%. invmod_redc_32 is a binary-GCD inverse mod p with a DIFFERENT
 * modulus per prime, so Montgomery's batch-inversion trick does NOT apply
 * (different rings); but 16 INDEPENDENT inverses map cleanly onto AVX-512 32-bit
 * lanes with per-lane masks. This is the published AVX-512 sieve-index angle
 * (SECRYPT 2021), distinct from the ruled-out AVX2 path.
 *
 * Kernel: each lane runs the binary extended-GCD inverse of a mod b (b odd,
 * gcd(a,b)=1) as a masked per-lane state machine; the loop runs until every lane
 * is done. Computes the PLAIN inverse a^-1 mod b (the REDC 2^-32 normalization
 * that invmod_redc_32 adds is a cheap deterministic per-lane fixup, folded in at
 * integration). Supports b < 2^31 (covers factor-base primes in this regime; the
 * arithmetic stays within uint32 since x,b < 2^31 => x+b < 2^32).
 *
 * Comet Lake has no AVX-512, so validated for CORRECTNESS bit-exact vs GMP under
 * Intel SDE (same method as the gf2x/IFMA work); perf gated on AVX-512 silicon.
 *
 *   gcc -O2 -mavx512f -mavx512cd bench/avx512-modinv.c -lgmp -o avx512-modinv
 *   /opt/intel-sde/sde64 -future -- ./avx512-modinv
 */
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>

#define LANES 16

#if defined(__AVX512F__)
#include <immintrin.h>

/* 16-way batched modular inverse: r[L] = a[L]^-1 mod b[L], b odd, gcd=1, b<2^31.
 * Binary extended GCD as a masked per-lane state machine. */
static void modinv16(uint32_t *r, const uint32_t *a, const uint32_t *b)
{
    const __m512i one  = _mm512_set1_epi32(1);
    const __m512i zero = _mm512_setzero_si512();
    __m512i U = _mm512_loadu_si512(a);
    __m512i V = _mm512_loadu_si512(b);
    __m512i B = V;
    __m512i X1 = one, X2 = zero;

    /* a step removes >=1 bit of total size or resolves a compare; 32-bit operands
     * need well under 4096 such primitive steps. We loop until all lanes done. */
    for (int it = 0; it < 4096; it++) {
        __mmask16 done = _kor_mask16(_mm512_cmpeq_epu32_mask(U, one),
                                     _mm512_cmpeq_epu32_mask(V, one));
        if (done == 0xFFFF) break;
        __mmask16 act = _knot_mask16(done);
        __mmask16 ueven = _mm512_test_epi32_mask(U, one);     /* (U&1)!=0 -> odd */
        ueven = _kandn_mask16(ueven, act);                    /* active & U even */
        __mmask16 veven = _mm512_test_epi32_mask(V, one);
        veven = _kandn_mask16(veven, act);
        veven = _kandn_mask16(ueven, veven);                  /* and not handling U-even */

        /* halve_mod(X,B): X even ? X>>1 : (X+B)>>1   (applied where we halve U or V) */
        /* --- branch 1: U even -> U>>=1 ; X1 = halve_mod(X1,B) --- */
        {
            __m512i Uh = _mm512_srli_epi32(U, 1);
            U = _mm512_mask_mov_epi32(U, ueven, Uh);
            __mmask16 x1odd = _mm512_test_epi32_mask(X1, one);
            __m512i xe = _mm512_srli_epi32(X1, 1);
            __m512i xo = _mm512_srli_epi32(_mm512_add_epi32(X1, B), 1);
            __m512i xh = _mm512_mask_blend_epi32(x1odd, xe, xo);
            X1 = _mm512_mask_mov_epi32(X1, ueven, xh);
        }
        /* --- branch 2: U odd, V even -> V>>=1 ; X2 = halve_mod(X2,B) --- */
        {
            __m512i Vh = _mm512_srli_epi32(V, 1);
            V = _mm512_mask_mov_epi32(V, veven, Vh);
            __mmask16 x2odd = _mm512_test_epi32_mask(X2, one);
            __m512i xe = _mm512_srli_epi32(X2, 1);
            __m512i xo = _mm512_srli_epi32(_mm512_add_epi32(X2, B), 1);
            __m512i xh = _mm512_mask_blend_epi32(x2odd, xe, xo);
            X2 = _mm512_mask_mov_epi32(X2, veven, xh);
        }
        /* --- both odd: subtract smaller from larger, update x via submod --- */
        __mmask16 bothodd = _kandn_mask16(ueven, _kandn_mask16(veven, act));
        __mmask16 uge = _kand_mask16(bothodd, _mm512_cmpge_epu32_mask(U, V));
        __mmask16 vgt = _kandn_mask16(uge, bothodd);
        /* submod(x,y,B) = x>=y ? x-y : x-y+B */
        /* branch U>=V: U-=V ; X1 = submod(X1,X2,B) */
        {
            U = _mm512_mask_sub_epi32(U, uge, U, V);
            __m512i d = _mm512_sub_epi32(X1, X2);
            __m512i dpb = _mm512_add_epi32(d, B);
            __mmask16 lt = _mm512_cmplt_epu32_mask(X1, X2);
            __m512i sm = _mm512_mask_blend_epi32(lt, d, dpb);
            X1 = _mm512_mask_mov_epi32(X1, uge, sm);
        }
        /* branch V>U: V-=U ; X2 = submod(X2,X1,B) */
        {
            V = _mm512_mask_sub_epi32(V, vgt, V, U);
            __m512i d = _mm512_sub_epi32(X2, X1);
            __m512i dpb = _mm512_add_epi32(d, B);
            __mmask16 lt = _mm512_cmplt_epu32_mask(X2, X1);
            __m512i sm = _mm512_mask_blend_epi32(lt, d, dpb);
            X2 = _mm512_mask_mov_epi32(X2, vgt, sm);
        }
    }
    /* result: where U==1 use X1 else X2; reduce mod B into [0,B) */
    __mmask16 uone = _mm512_cmpeq_epu32_mask(U, one);
    __m512i res = _mm512_mask_blend_epi32(uone, X2, X1);
    /* one conditional subtract (values already < B by construction, but be safe) */
    __mmask16 ge = _mm512_cmpge_epu32_mask(res, B);
    res = _mm512_mask_sub_epi32(res, ge, res, B);
    _mm512_storeu_si512(r, res);
}
#endif

static uint64_t xrnd(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

int main(void)
{
#if defined(__AVX512F__)
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
        modinv16(r, a, b);
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
    printf("AVX-512 16-way batched 32-bit modular inverse vs GMP, %ld trials:\n", trials);
    printf("  %s (%ld wrong)\n", wrong ? "FAIL" : "PASS", wrong);
    mpz_clears(ga, gb, gi, NULL);
    printf("%s\n", wrong ? "FAILURES" : "ALL PASS");
    return wrong != 0;
#else
    printf("avx512-modinv: built without AVX-512; build with -mavx512f -mavx512cd"
           " and run under sde64 -future.\n");
    return 0;
#endif
}
