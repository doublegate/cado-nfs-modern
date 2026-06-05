/*
 * vpclmul-mul1n.c — AVX-512 VPCLMULQDQ kernel for gf2x's mul_1_n / addmul_1_n,
 * with a scalar reference and a bit-exactness self-test.
 *
 * WHY: gf2x's base-case schoolbook multiply (the hot path of GF(2)[x]
 * multiplication, used by CADO-NFS's GF(2) linear algebra / lingen) loops over
 * `gf2x_mul_1_n`, doing ONE 64x64 carryless multiply per `_mm_clmulepi64_si128`
 * (128-bit). The AVX-512 VPCLMULQDQ instruction `_mm512_clmulepi64_epi128`
 * performs FOUR 64x64 carryless multiplies per instruction (one per 128-bit
 * lane of a 512-bit register), so we process 4 words of the operand per step.
 * Reference: Drucker & Gueron, "Faster multiplication over F2[X] using AVX512
 * and VPCLMULQDQ", arXiv:2201.10473 (reports up to ~39% on GF(2)[x] mul).
 *
 * HARDWARE: VPCLMULQDQ-512 needs AVX512F + VPCLMULQDQ (Ice Lake / Tiger Lake /
 * Sapphire Rapids / Zen4+). The CADO-NFS reference box is Comet Lake (no
 * AVX-512), so this is validated for CORRECTNESS under Intel SDE
 * (`sde64 -future -- ./vpclmul-test`) and benchmarked later on real AVX-512
 * silicon. Build:
 *     gcc -O2 -mavx512f -mvpclmulqdq bench/vpclmul-mul1n.c -o /tmp/vpclmul-test
 *     sde64 -future -- /tmp/vpclmul-test          # validates under emulation
 *
 * Semantics of mul_1_n(cp, bp, sb, a): cp[0..sb-1] = bp[0..sb-1] * a over
 * GF(2)[x] (each word a degree-<64 polynomial), returning the carry-out word.
 * Product of bp[i]*a is 128-bit: its low half lands in cp[i], its high half
 * carries into cp[i+1].
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---------- scalar reference (the ground truth) ---------- */

static void clmul64(uint64_t a, uint64_t b, uint64_t *lo, uint64_t *hi)
{
    uint64_t l = 0, h = 0;
    for (int i = 0; i < 64; i++) {
        if ((b >> i) & 1ULL) {
            l ^= a << i;
            if (i) h ^= a >> (64 - i);
        }
    }
    *lo = l; *hi = h;
}

static uint64_t ref_mul_1_n(uint64_t *cp, const uint64_t *bp, long sb, uint64_t a)
{
    uint64_t carry = 0;
    for (long i = 0; i < sb; i++) {
        uint64_t lo, hi;
        clmul64(bp[i], a, &lo, &hi);
        cp[i] = lo ^ carry;   /* low half + carry from previous word's high half */
        carry = hi;
    }
    return carry;             /* == cp[sb] */
}

static uint64_t ref_addmul_1_n(uint64_t *dp, const uint64_t *cp,
                               const uint64_t *bp, long sb, uint64_t a)
{
    uint64_t carry = 0;
    for (long i = 0; i < sb; i++) {
        uint64_t lo, hi;
        clmul64(bp[i], a, &lo, &hi);
        dp[i] = cp[i] ^ lo ^ carry;
        carry = hi;
    }
    return carry;
}

/* ---------- AVX-512 VPCLMULQDQ implementation ---------- */

#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
#include <immintrin.h>

/*
 * Process 4 words of bp per VPCLMULQDQ. We place bp[i+k] in the low 64 of lane
 * k (slots 0,2,4,6) and broadcast `a` to every slot; imm 0x00 selects, per
 * 128-bit lane, clmul(X.lo64, Y.lo64) = bp[i+k] * a. The 8 result words are
 *   [p0.lo p0.hi p1.lo p1.hi p2.lo p2.hi p3.lo p3.hi]
 * and fold (each product's high half overlaps the next word) as:
 *   cp[i]   = p0.lo ^ carry_in
 *   cp[i+1] = p0.hi ^ p1.lo
 *   cp[i+2] = p1.hi ^ p2.lo
 *   cp[i+3] = p2.hi ^ p3.lo
 *   carry_out = p3.hi
 */
static uint64_t vcl_mul_1_n(uint64_t *cp, const uint64_t *bp, long sb, uint64_t a)
{
    const __m512i y = _mm512_set1_epi64((long long)a);
    uint64_t carry = 0;
    long i = 0;
    for (; i + 4 <= sb; i += 4) {
        __m512i x = _mm512_set_epi64(0, (long long)bp[i+3], 0, (long long)bp[i+2],
                                     0, (long long)bp[i+1], 0, (long long)bp[i+0]);
        __m512i p = _mm512_clmulepi64_epi128(x, y, 0x00);
        uint64_t t[8];
        _mm512_storeu_si512((void *)t, p);
        cp[i+0] = t[0] ^ carry;
        cp[i+1] = t[1] ^ t[2];
        cp[i+2] = t[3] ^ t[4];
        cp[i+3] = t[5] ^ t[6];
        carry   = t[7];
    }
    for (; i < sb; i++) {       /* scalar tail for the last <4 words */
        uint64_t lo, hi;
        clmul64(bp[i], a, &lo, &hi);
        cp[i] = lo ^ carry;
        carry = hi;
    }
    return carry;
}

static uint64_t vcl_addmul_1_n(uint64_t *dp, const uint64_t *cp,
                               const uint64_t *bp, long sb, uint64_t a)
{
    const __m512i y = _mm512_set1_epi64((long long)a);
    uint64_t carry = 0;
    long i = 0;
    for (; i + 4 <= sb; i += 4) {
        __m512i x = _mm512_set_epi64(0, (long long)bp[i+3], 0, (long long)bp[i+2],
                                     0, (long long)bp[i+1], 0, (long long)bp[i+0]);
        __m512i p = _mm512_clmulepi64_epi128(x, y, 0x00);
        uint64_t t[8];
        _mm512_storeu_si512((void *)t, p);
        dp[i+0] = cp[i+0] ^ t[0] ^ carry;
        dp[i+1] = cp[i+1] ^ t[1] ^ t[2];
        dp[i+2] = cp[i+2] ^ t[3] ^ t[4];
        dp[i+3] = cp[i+3] ^ t[5] ^ t[6];
        carry   = t[7];
    }
    for (; i < sb; i++) {
        uint64_t lo, hi;
        clmul64(bp[i], a, &lo, &hi);
        dp[i] = cp[i] ^ lo ^ carry;
        carry = hi;
    }
    return carry;
}
#endif /* AVX512F && VPCLMULQDQ */

/* ---------- self-test ---------- */

static uint64_t rnd(void)
{
    /* xorshift64* — deterministic, no libc rand differences */
    static uint64_t s = 0x9e3779b97f4a7c15ULL;
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 0x2545F4914F6CDD1DULL;
}

int main(void)
{
#if defined(__AVX512F__) && defined(__VPCLMULQDQ__)
    enum { MAXN = 64, TRIALS = 200000 };
    uint64_t bp[MAXN], cref[MAXN + 1], cvcl[MAXN + 1];
    uint64_t din[MAXN], dref[MAXN + 1], dvcl[MAXN + 1];
    long fails = 0;

    for (long t = 0; t < TRIALS; t++) {
        long sb = 1 + (long)(rnd() % MAXN);   /* exercise all lengths incl. tails */
        uint64_t a = rnd();
        for (long i = 0; i < sb; i++) { bp[i] = rnd(); din[i] = rnd(); }

        memset(cref, 0, sizeof cref); memset(cvcl, 0, sizeof cvcl);
        uint64_t r1 = ref_mul_1_n(cref, bp, sb, a);
        uint64_t r2 = vcl_mul_1_n(cvcl, bp, sb, a);
        if (r1 != r2 || memcmp(cref, cvcl, sb * sizeof(uint64_t))) { fails++; continue; }

        uint64_t r3 = ref_addmul_1_n(dref, din, bp, sb, a);
        uint64_t r4 = vcl_addmul_1_n(dvcl, din, bp, sb, a);
        if (r3 != r4 || memcmp(dref, dvcl, sb * sizeof(uint64_t))) fails++;
    }

    if (fails == 0) {
        printf("PASS: VPCLMULQDQ mul_1_n/addmul_1_n bit-exact vs scalar (%d trials)\n", TRIALS);
        return 0;
    }
    printf("FAIL: %ld mismatches out of %d trials\n", fails, TRIALS);
    return 1;
#else
    printf("SKIP: built without AVX512F+VPCLMULQDQ "
           "(compile with -mavx512f -mvpclmulqdq, run under `sde64 -future`)\n");
    return 0;
#endif
}
