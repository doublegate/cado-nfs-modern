/*
 * ifma-gfp.c — AVX-512 IFMA GF(p) operations in arith-modp's PLAIN representation
 * (Roadmap B3: wire the v3.1.0 IFMA modmul toward the BWC GF(p) backend).
 *
 * CADO 3.0.0 replaced mpfq with the C++ `arith-modp` GF(p) backend
 * (linalg/bwc/arith-modp*.hpp, the per-size p1..p8 fields used by the DLP Block
 * Wiedemann). Two facts decide the integration:
 *   1. arith-modp stores elements PLAIN (reduced in [0,p), schoolbook mul +
 *      Barrett-style reduce) — NOT Montgomery. The validated IFMA kernel
 *      (bench/ifma-modmul.c) is Montgomery (a*b*R^-1 mod p). So the kernel cannot
 *      be dropped into arith-modp as-is — a representation bridge is needed.
 *   2. The hot batched full-GF(p)-modmul sites are the vector ops in
 *      arith-generic.hpp: vec_add_dotprod (sum a_i*b_i) and vec_addmul_and_reduce.
 *      The scalar mul() and the SpMV (multiply-by-small-coefficient) are NOT where
 *      an 8-wide modmul helps.
 *
 * This builds the missing PLAIN-representation primitives on the validated
 * Montgomery IFMA kernel, with NO per-element domain churn:
 *   - plain_mul(a,b)  = M(M(a,b), R^2)            // = a*b mod p, two montmuls
 *   - dotprod(a[],b[]) : acc = sum_i M(a_i,b_i) = (sum a_i b_i) R^-1 ;
 *                        result = M(acc, R^2) = sum a_i b_i mod p
 *     (the common R^-1 factor amortizes: n montmuls + 1, not 2n — the right shape
 *      for arith-generic's vec_add_dotprod).
 * Both are plain-in / plain-out, matching arith-modp, and 8-way (one independent
 * GF(p) field per 512-bit lane).
 *
 * Comet Lake has no AVX-512-IFMA, so this is validated for CORRECTNESS bit-exact
 * vs GMP under Intel SDE (the same method as the gf2x VPCLMULQDQ work); the
 * speedup is gated on real AVX-512-IFMA silicon (Ice Lake / Sapphire Rapids+).
 *
 *   gcc -O2 -mavx512f -mavx512ifma bench/ifma-gfp.c -lgmp -o ifma-gfp
 *   SDE=/opt/intel-sde/sde64 -future -- ./ifma-gfp
 */
#include <stdio.h>
#include <stdint.h>
#include <gmp.h>
#include <immintrin.h>

#define NLIMBS 5                 /* 5 * 52 = 260-bit capacity (>= 256-bit p, arith-modp p4/p5) */
#define RADIX  52
#define MASK52 ((1ULL << 52) - 1)
#define LANES  8

/* ---- 8-lane batched CIOS Montgomery multiply, radix 2^52 (from ifma-modmul.c,
 * validated bit-exact vs GMP). r = a*b*2^-(52*NLIMBS) mod m, per lane. ---- */
static void mont_mul_ifma(__m512i *r, const __m512i *a, const __m512i *b,
                          const __m512i *m, __m512i mp)
{
    const __m512i vmask = _mm512_set1_epi64(MASK52);
    __m512i t[NLIMBS + 2];
    for (int k = 0; k < NLIMBS + 2; k++) t[k] = _mm512_setzero_si512();
    for (int i = 0; i < NLIMBS; i++) {
        __m512i ai = a[i];
        __m512i C = _mm512_setzero_si512();
        for (int j = 0; j < NLIMBS; j++) {
            __m512i lo = _mm512_madd52lo_epu64(t[j], ai, b[j]);
            lo = _mm512_add_epi64(lo, C);
            t[j] = _mm512_and_si512(lo, vmask);
            C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo, RADIX), ai, b[j]);
        }
        t[NLIMBS] = _mm512_add_epi64(t[NLIMBS], C);
        __m512i q = _mm512_and_si512(_mm512_madd52lo_epu64(_mm512_setzero_si512(),
                                                           t[0], mp), vmask);
        __m512i lo0 = _mm512_madd52lo_epu64(t[0], q, m[0]);
        C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo0, RADIX), q, m[0]);
        for (int j = 1; j < NLIMBS; j++) {
            __m512i lo = _mm512_madd52lo_epu64(t[j], q, m[j]);
            lo = _mm512_add_epi64(lo, C);
            t[j - 1] = _mm512_and_si512(lo, vmask);
            C = _mm512_madd52hi_epu64(_mm512_srli_epi64(lo, RADIX), q, m[j]);
        }
        __m512i s = _mm512_add_epi64(t[NLIMBS], C);
        t[NLIMBS - 1] = _mm512_and_si512(s, vmask);
        t[NLIMBS] = _mm512_add_epi64(t[NLIMBS + 1], _mm512_srli_epi64(s, RADIX));
        t[NLIMBS + 1] = _mm512_setzero_si512();
    }
    __m512i borrow = _mm512_setzero_si512();
    __m512i d[NLIMBS];
    for (int j = 0; j < NLIMBS; j++) {
        __m512i diff = _mm512_sub_epi64(_mm512_sub_epi64(t[j], m[j]), borrow);
        d[j] = _mm512_and_si512(diff, vmask);
        borrow = _mm512_and_si512(_mm512_srli_epi64(diff, 63), _mm512_set1_epi64(1));
    }
    __mmask8 ge = _mm512_cmpeq_epi64_mask(borrow, _mm512_setzero_si512());
    for (int j = 0; j < NLIMBS; j++)
        r[j] = _mm512_mask_blend_epi64(ge, t[j], d[j]);
}

/* a += b mod m, per lane, plain 52-bit-limb representation (a,b < m) */
static void add_mod_ifma(__m512i *r, const __m512i *a, const __m512i *b, const __m512i *m)
{
    const __m512i vmask = _mm512_set1_epi64(MASK52);
    __m512i s[NLIMBS], C = _mm512_setzero_si512();
    for (int j = 0; j < NLIMBS; j++) {
        __m512i x = _mm512_add_epi64(_mm512_add_epi64(a[j], b[j]), C);
        s[j] = _mm512_and_si512(x, vmask);
        C = _mm512_srli_epi64(x, RADIX);
    }
    /* a,b < m < 2^251 so s = a+b < 2m < 2^260 fits in NLIMBS limbs (C==0) and
     * one conditional subtract suffices: subtract m if s >= m. */
    (void)C;
    __m512i borrow = _mm512_setzero_si512(), d[NLIMBS];
    for (int j = 0; j < NLIMBS; j++) {
        __m512i diff = _mm512_sub_epi64(_mm512_sub_epi64(s[j], m[j]), borrow);
        d[j] = _mm512_and_si512(diff, vmask);
        borrow = _mm512_and_si512(_mm512_srli_epi64(diff, 63), _mm512_set1_epi64(1));
    }
    __mmask8 ge = _mm512_cmpeq_epi64_mask(borrow, _mm512_setzero_si512()); /* borrow==0 -> s>=m */
    for (int j = 0; j < NLIMBS; j++) r[j] = _mm512_mask_blend_epi64(ge, s[j], d[j]);
}

/* PLAIN modmul: r = a*b mod m (plain in/out) = M(M(a,b), R2), R2 = R^2 mod m. */
static void plain_mul_ifma(__m512i *r, const __m512i *a, const __m512i *b,
                           const __m512i *m, __m512i mp, const __m512i *R2)
{
    __m512i t[NLIMBS];
    mont_mul_ifma(t, a, b, m, mp);     /* a*b*R^-1 */
    mont_mul_ifma(r, t, R2, m, mp);    /* (a*b*R^-1)*R^2*R^-1 = a*b   */
}

/* PLAIN dot product: r = sum_i a[i]*b[i] mod m (plain), the vec_add_dotprod shape.
 * acc = sum_i M(a_i,b_i) = (sum a_i b_i) R^-1 ; r = M(acc, R2) = sum a_i b_i. */
static void dotprod_ifma(__m512i *r, const __m512i a[][NLIMBS], const __m512i b[][NLIMBS],
                         int n, const __m512i *m, __m512i mp, const __m512i *R2)
{
    __m512i acc[NLIMBS]; for (int j = 0; j < NLIMBS; j++) acc[j] = _mm512_setzero_si512();
    for (int i = 0; i < n; i++) {
        __m512i t[NLIMBS];
        mont_mul_ifma(t, a[i], b[i], m, mp);      /* a_i b_i R^-1 */
        add_mod_ifma(acc, acc, t, m);
    }
    mont_mul_ifma(r, acc, R2, m, mp);             /* * R^2 * R^-1 = sum a_i b_i mod m */
}

/* ---- host helpers ---- */
static void to52(uint64_t out[NLIMBS], const mpz_t v) {
    mpz_t t; mpz_init_set(t, v);
    for (int i = 0; i < NLIMBS; i++) { out[i]=(uint64_t)(mpz_get_ui(t)&MASK52); mpz_fdiv_q_2exp(t,t,RADIX); }
    mpz_clear(t);
}
static void from52(mpz_t v, const uint64_t in[NLIMBS]) {
    mpz_set_ui(v, 0);
    for (int i = NLIMBS-1; i >= 0; i--) { mpz_mul_2exp(v,v,RADIX); mpz_add_ui(v,v,in[i]); }
}
static uint64_t xrnd(uint64_t *s){ *s^=*s<<13; *s^=*s>>7; *s^=*s<<17; return *s; }

int main(void) {
    uint64_t seed = 0x1F3A2B5CULL;
    mpz_t R, R2z, Rinv, mpz_, e, g; mpz_inits(R,R2z,Rinv,mpz_,e,g,NULL);
    mpz_set_ui(R,1); mpz_mul_2exp(R, R, RADIX*NLIMBS);          /* R = 2^260 */

    long mul_trials=0, mul_wrong=0, dot_trials=0, dot_wrong=0;
    const int VECN = 12;   /* dot-product length */

    for (int rep = 0; rep < 4000; rep++) {
        mpz_t ms[LANES]; uint64_t mpL[LANES];
        uint64_t mL[NLIMBS][LANES], R2L[NLIMBS][LANES];
        /* per-lane field: odd ~255-bit prime-ish modulus */
        for (int L = 0; L < LANES; L++) {
            mpz_init(ms[L]);
            mpz_set_ui(ms[L], 0);
            for (int i = 0; i < NLIMBS; i++) { mpz_mul_2exp(ms[L],ms[L],RADIX); mpz_add_ui(ms[L], ms[L], xrnd(&seed)&MASK52); }
            mpz_fdiv_q_2exp(ms[L], ms[L], 6);    /* headroom below R */
            mpz_setbit(ms[L], 250); mpz_setbit(ms[L], 0);   /* ~251-bit, odd */
            mpz_t base; mpz_init_set_ui(base,1); mpz_mul_2exp(base,base,RADIX);
            mpz_invert(mpz_, ms[L], base); mpz_sub(mpz_, base, mpz_); mpz_mod(mpz_, mpz_, base);
            mpL[L] = mpz_get_ui(mpz_) & MASK52; mpz_clear(base);
            mpz_mod(R2z, R, ms[L]); mpz_mul(R2z, R2z, R2z); mpz_mod(R2z, R2z, ms[L]); /* R^2 mod m */
            uint64_t tmp[NLIMBS];
            to52(tmp, ms[L]); for (int i=0;i<NLIMBS;i++) mL[i][L]=tmp[i];
            to52(tmp, R2z);   for (int i=0;i<NLIMBS;i++) R2L[i][L]=tmp[i];
        }
        __m512i mv[NLIMBS], R2v[NLIMBS];
        for (int i=0;i<NLIMBS;i++){ mv[i]=_mm512_loadu_si512(mL[i]); R2v[i]=_mm512_loadu_si512(R2L[i]); }
        __m512i mpv = _mm512_loadu_si512(mpL);

        /* ---- (1) plain_mul: a*b mod m ---- */
        {
            mpz_t as[LANES], bs[LANES]; uint64_t aL[NLIMBS][LANES], bL[NLIMBS][LANES];
            for (int L=0;L<LANES;L++){ mpz_inits(as[L],bs[L],NULL);
                mpz_set_ui(as[L],0); mpz_set_ui(bs[L],0);
                for(int i=0;i<NLIMBS;i++){ mpz_mul_2exp(as[L],as[L],RADIX); mpz_add_ui(as[L],as[L],xrnd(&seed)&MASK52);
                                           mpz_mul_2exp(bs[L],bs[L],RADIX); mpz_add_ui(bs[L],bs[L],xrnd(&seed)&MASK52); }
                mpz_mod(as[L],as[L],ms[L]); mpz_mod(bs[L],bs[L],ms[L]);
                uint64_t tmp[NLIMBS]; to52(tmp,as[L]); for(int i=0;i<NLIMBS;i++)aL[i][L]=tmp[i];
                to52(tmp,bs[L]); for(int i=0;i<NLIMBS;i++)bL[i][L]=tmp[i]; }
            __m512i av[NLIMBS],bv[NLIMBS],rv[NLIMBS];
            for(int i=0;i<NLIMBS;i++){ av[i]=_mm512_loadu_si512(aL[i]); bv[i]=_mm512_loadu_si512(bL[i]); }
            plain_mul_ifma(rv, av, bv, mv, mpv, R2v);
            uint64_t rL[NLIMBS][LANES]; for(int i=0;i<NLIMBS;i++)_mm512_storeu_si512(rL[i],rv[i]);
            for(int L=0;L<LANES;L++){ mpz_mul(e,as[L],bs[L]); mpz_mod(e,e,ms[L]);
                uint64_t tmp[NLIMBS]; for(int i=0;i<NLIMBS;i++)tmp[i]=rL[i][L]; from52(g,tmp);
                mul_trials++; if(mpz_cmp(g,e)){ if(mul_wrong<3) gmp_printf("  mul MISMATCH L%d got %Zd exp %Zd\n",L,g,e); mul_wrong++; }
                mpz_clears(as[L],bs[L],NULL); }
        }
        /* ---- (2) dotprod: sum_i a_i*b_i mod m, VECN terms ---- */
        {
            __m512i A[VECN][NLIMBS], B[VECN][NLIMBS]; mpz_t acc[LANES];
            for(int L=0;L<LANES;L++){ mpz_init_set_ui(acc[L],0); }
            for(int i=0;i<VECN;i++){
                uint64_t aL[NLIMBS][LANES],bL[NLIMBS][LANES];
                for(int L=0;L<LANES;L++){ mpz_t a,b; mpz_inits(a,b,NULL);
                    mpz_set_ui(a,0); mpz_set_ui(b,0);
                    for(int k=0;k<NLIMBS;k++){ mpz_mul_2exp(a,a,RADIX); mpz_add_ui(a,a,xrnd(&seed)&MASK52);
                                               mpz_mul_2exp(b,b,RADIX); mpz_add_ui(b,b,xrnd(&seed)&MASK52); }
                    mpz_mod(a,a,ms[L]); mpz_mod(b,b,ms[L]);
                    mpz_t t; mpz_init(t); mpz_mul(t,a,b); mpz_add(acc[L],acc[L],t); mpz_mod(acc[L],acc[L],ms[L]); mpz_clear(t);
                    uint64_t tmp[NLIMBS]; to52(tmp,a); for(int k=0;k<NLIMBS;k++)aL[k][L]=tmp[k];
                    to52(tmp,b); for(int k=0;k<NLIMBS;k++)bL[k][L]=tmp[k]; mpz_clears(a,b,NULL); }
                for(int k=0;k<NLIMBS;k++){ A[i][k]=_mm512_loadu_si512(aL[k]); B[i][k]=_mm512_loadu_si512(bL[k]); }
            }
            __m512i rv[NLIMBS]; dotprod_ifma(rv, A, B, VECN, mv, mpv, R2v);
            uint64_t rL[NLIMBS][LANES]; for(int i=0;i<NLIMBS;i++)_mm512_storeu_si512(rL[i],rv[i]);
            for(int L=0;L<LANES;L++){ uint64_t tmp[NLIMBS]; for(int i=0;i<NLIMBS;i++)tmp[i]=rL[i][L]; from52(g,tmp);
                dot_trials++; if(mpz_cmp(g,acc[L])){ if(dot_wrong<3) gmp_printf("  dot MISMATCH L%d got %Zd exp %Zd\n",L,g,acc[L]); dot_wrong++; }
                mpz_clear(acc[L]); }
        }
        for(int L=0;L<LANES;L++) mpz_clear(ms[L]);
    }
    printf("IFMA GF(p) plain-representation ops (arith-modp compatible), %d-bit, %d-way:\n", RADIX*NLIMBS, LANES);
    printf("  plain_mul  (a*b mod p)        : %s (%ld/%ld wrong)\n", mul_wrong?"FAIL":"PASS", mul_wrong, mul_trials);
    printf("  dotprod    (sum a_i b_i mod p): %s (%ld/%ld wrong)  [vec_add_dotprod shape, %d terms]\n",
           dot_wrong?"FAIL":"PASS", dot_wrong, dot_trials, VECN);
    int fail = (mul_wrong||dot_wrong);
    printf("%s\n", fail?"FAILURES":"ALL PASS");
    mpz_clears(R,R2z,Rinv,mpz_,e,g,NULL);
    return fail;
}
