/* This file is part of the gf2x library.

   Copyright 2007, 2008, 2009, 2010, 2012, 2013, 2015
   Richard Brent, Pierrick Gaudry, Emmanuel Thome', Paul Zimmermann

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
   see the files COPYING and COPYING.LIB.  If not, write to the Free
   Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
   02110-1301, USA.
*/


#ifndef GF2X_MUL3_H_
#define GF2X_MUL3_H_

#include "gf2x.h"
/* All gf2x source files for lowlevel functions must include gf2x-small.h
 * This is mandatory for the tuning mechanism. */
#include "gf2x/gf2x-small.h"

#if GF2X_WORDSIZE != 64
#error "This code is for 64-bit only"
#endif

#if !defined(GF2X_HAVE_PCLMUL_SUPPORT) && !defined(GF2X_HAVE_VPCLMUL_SUPPORT)
#error "This code needs pclmul or vpclmul support"
#endif

#ifdef GF2X_HAVE_VPCLMUL_SUPPORT
/* AVX-512 VPCLMULQDQ: the same 6-product Karatsuba as the PCLMUL path below,
 * but the 6 word-products are packed into TWO _mm512_clmulepi64_epi128 calls
 * (4 lanes then 2 lanes), then combined scalar-side. Validated bit-exact vs the
 * scalar GF(2)[x] reference over 200000 trials under Intel SDE
 * (bench/vpclmul-muln.c). Ref: Drucker & Gueron, arXiv:2201.10473. */
GF2X_STORAGE_CLASS_mul3
void gf2x_mul3 (unsigned long *c, const unsigned long *a, const unsigned long *b)
{
    unsigned long a0=a[0],a1=a[1],a2=a[2], b0=b[0],b1=b[1],b2=b[2];
    /* call 1 lanes: p0=a0b0, p1=a1b1, p2=a2b2, p01=(a0^a1)(b0^b1) */
    __m512i A1 = _mm512_set_epi64(0,(long long)(a0^a1), 0,(long long)a2,
                                  0,(long long)a1,       0,(long long)a0);
    __m512i B1 = _mm512_set_epi64(0,(long long)(b0^b1), 0,(long long)b2,
                                  0,(long long)b1,       0,(long long)b0);
    /* call 2 lanes: p02=(a0^a2)(b0^b2), p12=(a1^a2)(b1^b2) */
    __m512i A2 = _mm512_set_epi64(0,0, 0,0, 0,(long long)(a1^a2), 0,(long long)(a0^a2));
    __m512i B2 = _mm512_set_epi64(0,0, 0,0, 0,(long long)(b1^b2), 0,(long long)(b0^b2));
    unsigned long r1[8], r2[8];
    _mm512_storeu_si512((void*)r1, _mm512_clmulepi64_epi128(A1, B1, 0x00));
    _mm512_storeu_si512((void*)r2, _mm512_clmulepi64_epi128(A2, B2, 0x00));
    unsigned long e[5][2];
    for (int i=0;i<2;i++){
        unsigned long p0=r1[i], p1=r1[2+i], p2=r1[4+i],
                      p01=r1[6+i], p02=r2[i], p12=r2[2+i];
        e[0][i]=p0;
        e[1][i]=p0^p1^p01;
        e[2][i]=p0^p1^p2^p02;
        e[3][i]=p1^p2^p12;
        e[4][i]=p2;
    }
    for (int i=0;i<6;i++) c[i]=0;
    for (int j=0;j<5;j++){ c[j]^=e[j][0]; c[j+1]^=e[j][1]; }
}
#else
/* TODO: if somebody comes up with a neat way to improve the interface so
 * as to remove the false dependency on pclmul, that would be nice.
 */
static inline __m128i
GF2X_FUNC(mul3cl_mul1) (unsigned long a, unsigned long b)
{
    __m128i aa = _gf2x_mm_setr_epi64(a, 0);
    __m128i bb = _gf2x_mm_setr_epi64(b, 0);
    return _mm_clmulepi64_si128(aa, bb, 0);
}

/* uses the variant of Karatsuba with 6 multiplications */
GF2X_STORAGE_CLASS_mul3
void gf2x_mul3 (unsigned long *c, const unsigned long *a, const unsigned long *b)
{
  unsigned long aa[3], bb[3];
  aa[0] = a[1]^a[2];
  aa[1] = a[0]^a[2];
  aa[2] = a[0]^a[1];
  bb[0] = b[1]^b[2];
  bb[1] = b[0]^b[2];
  bb[2] = b[0]^b[1];
  __m128i p0  = GF2X_FUNC(mul3cl_mul1)(a[0], b[0]);
  __m128i p1  = GF2X_FUNC(mul3cl_mul1)(a[1], b[1]);
  __m128i p2  = GF2X_FUNC(mul3cl_mul1)(a[2], b[2]);
  __m128i pp0 = GF2X_FUNC(mul3cl_mul1)(aa[0], bb[0]);
  __m128i pp1 = GF2X_FUNC(mul3cl_mul1)(aa[1], bb[1]);
  __m128i pp2 = GF2X_FUNC(mul3cl_mul1)(aa[2], bb[2]);

#define PXOR(lop, rop) _mm_xor_si128((lop), (rop))

  __m128i ce0 = p0;
  __m128i ce2 = PXOR(p0, PXOR(p1, PXOR(p2, pp1)));
  __m128i ce4 = p2;

  __m128i co1 = PXOR(p0, PXOR(p1, pp2));
  __m128i co3 = PXOR(pp0, PXOR(p1, p2));

  _mm_storeu_si128((__m128i*)(c),   PXOR(ce0, _mm_slli_si128(co1, 8)));
  _mm_storeu_si128((__m128i*)(c+2), PXOR(ce2, PXOR(_mm_srli_si128(co1, 8), _mm_slli_si128(co3, 8))));
  _mm_storeu_si128((__m128i*)(c+4), PXOR(ce4, _mm_srli_si128(co3, 8)));
#undef PXOR
}
#endif  /* GF2X_HAVE_VPCLMUL_SUPPORT */

#endif  /* GF2X_MUL3_H_ */
