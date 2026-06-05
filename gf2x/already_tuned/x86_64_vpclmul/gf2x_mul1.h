/* This file is part of the gf2x library.

   Copyright 2010, 2013, 2015
   Richard Brent, Pierrick Gaudry, Emmanuel Thome', Paul Zimmermann
   AVX-512 VPCLMULQDQ backend added 2026 (cado-nfs 3.0.0-modern).

   This program is free software; you can redistribute it and/or modify it
   under the terms of either:
    - If the archive contains a file named toom-gpl.c (not a trivial
    placeholder), the GNU General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.
    - If the archive contains a file named toom-gpl.c which is a trivial
    placeholder, the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

   This program is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.  See the license text for more details.

   You should have received a copy of the GNU General Public License as
   well as the GNU Lesser General Public License along with this program;
   see the files COPYING and COPYING.LIB.
*/

/*
 * AVX-512 VPCLMULQDQ base-case multiplier. `_mm512_clmulepi64_epi128` performs
 * four 64x64 carryless multiplies per instruction (one per 128-bit lane), so
 * gf2x_mul_1_n / gf2x_addmul_1_n process FOUR words of `bp` per step instead of
 * the two-per-128-bit-clmul of the pclmul backend. The fold pattern (each
 * 128-bit product's high half overlaps the next word) is identical to the
 * scalar/pclmul reference; this exact logic is validated bit-exact over 200000
 * random trials under Intel SDE (see bench/vpclmul-mul1n.c).
 * Ref: Drucker & Gueron, arXiv:2201.10473.
 */

#ifndef GF2X_MUL1_H_
#define GF2X_MUL1_H_

#include "gf2x.h"
/* All gf2x source files for lowlevel functions must include gf2x-small.h
 * This is mandatory for the tuning mechanism. */
#include "gf2x/gf2x-small.h"

#if GF2X_WORDSIZE != 64
#error "This code is for 64-bit only"
#endif

#ifndef GF2X_HAVE_VPCLMUL_SUPPORT
#error "This code needs AVX-512 VPCLMULQDQ support"
#endif

/* A single 1x1 product gains nothing from a 512-bit register; keep the optimal
 * 128-bit pclmul form (VPCLMULQDQ builds always include pclmul). */
GF2X_STORAGE_CLASS_mul1 void
gf2x_mul1 (unsigned long *c, unsigned long a, unsigned long b)
{
    __m128i aa = _gf2x_mm_setr_epi64(a, 0);
    __m128i bb = _gf2x_mm_setr_epi64(b, 0);
    _mm_storeu_si128((__m128i*)c, _mm_clmulepi64_si128(aa, bb, 0));
}

GF2X_STORAGE_CLASS_mul_1_n unsigned long
gf2x_mul_1_n (unsigned long *cp, const unsigned long *bp, long sb, unsigned long a)
{
    const __m512i y = _mm512_set1_epi64((long long) a);
    unsigned long carry = 0;
    long i = 0;

    /* four words of bp per VPCLMULQDQ: lane k = bp[i+k]*a, stored as
     * (lo,hi) in slots (2k,2k+1) of the 8x64 result. */
    for ( ; i + 4 <= sb; i += 4) {
        __m512i x = _mm512_set_epi64(0, (long long) bp[i+3], 0, (long long) bp[i+2],
                                     0, (long long) bp[i+1], 0, (long long) bp[i+0]);
        __m512i p = _mm512_clmulepi64_epi128(x, y, 0x00);
        unsigned long t[8];
        _mm512_storeu_si512((void *) t, p);
        cp[i+0] = t[0] ^ carry;
        cp[i+1] = t[1] ^ t[2];
        cp[i+2] = t[3] ^ t[4];
        cp[i+3] = t[5] ^ t[6];
        carry   = t[7];
    }
    /* scalar tail (<4 words) via 128-bit pclmul */
    for ( ; i < sb; i++) {
        __m128i x = _gf2x_mm_setr_epi64(bp[i], 0);
        __m128i yy = _gf2x_mm_setr_epi64(a, 0);
        union { __m128i s; unsigned long x[2]; } cc;
        cc.s = _mm_clmulepi64_si128(x, yy, 0);
        cp[i] = cc.x[0] ^ carry;
        carry = cc.x[1];
    }
    return carry;
}

GF2X_STORAGE_CLASS_addmul_1_n unsigned long
gf2x_addmul_1_n (unsigned long *dp, const unsigned long *cp, const unsigned long* bp, long sb, unsigned long a)
{
    const __m512i y = _mm512_set1_epi64((long long) a);
    unsigned long carry = 0;
    long i = 0;

    for ( ; i + 4 <= sb; i += 4) {
        __m512i x = _mm512_set_epi64(0, (long long) bp[i+3], 0, (long long) bp[i+2],
                                     0, (long long) bp[i+1], 0, (long long) bp[i+0]);
        __m512i p = _mm512_clmulepi64_epi128(x, y, 0x00);
        unsigned long t[8];
        _mm512_storeu_si512((void *) t, p);
        dp[i+0] = cp[i+0] ^ t[0] ^ carry;
        dp[i+1] = cp[i+1] ^ t[1] ^ t[2];
        dp[i+2] = cp[i+2] ^ t[3] ^ t[4];
        dp[i+3] = cp[i+3] ^ t[5] ^ t[6];
        carry   = t[7];
    }
    for ( ; i < sb; i++) {
        __m128i x = _gf2x_mm_setr_epi64(bp[i], 0);
        __m128i yy = _gf2x_mm_setr_epi64(a, 0);
        union { __m128i s; unsigned long x[2]; } dd;
        dd.s = _mm_clmulepi64_si128(x, yy, 0);
        dp[i] = cp[i] ^ dd.x[0] ^ carry;
        carry = dd.x[1];
    }
    return carry;
}

#endif   /* GF2X_MUL1_H_ */
