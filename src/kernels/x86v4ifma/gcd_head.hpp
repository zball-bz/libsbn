/* hgcd2.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1996, 1998, 2000-2004, 2008, 2012, 2019 Free Software Foundation,
Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of either:

  * the GNU Lesser General Public License as published by the Free
    Software Foundation; either version 3 of the License, or (at your
    option) any later version.

or

  * the GNU General Public License as published by the Free Software
    Foundation; either version 2 of the License, or (at your option) any
    later version.

or both in parallel, as here.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received copies of the GNU General Public License and the
GNU Lesser General Public License along with the GNU MP Library.  If not,
see https://www.gnu.org/licenses/.  */


// Adapted from GNU MP hgcd2.c and hgcd2-div.h; upstream notices retained.
// Uses exact leading-word division with a proved one-step correction;
// bit-wise division remains the fallback for an unusually large quotient.
#pragma once
#include <stdint.h>
#include <assert.h>
#include "common/small_checks.h"
#define GCD_HEAD_ASSERT(x) do { if constexpr(SBN3_CHECK_SMALL) assert(x); } while(0)
namespace sbn::v3::gcd_small {
struct IgnoreQuotient {void operator()(unsigned,uint64_t)const noexcept {}};
struct Matrix {uint64_t u[2][2];};
struct QR {uint64_t d0,d1;};
inline QR div1(uint64_t n,uint64_t d){const uint64_t q=n/d;return {n-q*d,q};}
inline void sub2(uint64_t&h,uint64_t&l,uint64_t ah,uint64_t al,uint64_t bh,uint64_t bl){l=al-bl;h=ah-bh-(al<bl);}
#define sub_ddmmss(h,l,ah,al,bh,bl) sub2(h,l,ah,al,bh,bl)
// GMP hgcd2-div.h, bit-wise two-word division method 2. Notice below.
inline uint64_t div2_bits(uint64_t*r,uint64_t nh,uint64_t nl,uint64_t dh,uint64_t dl){
 unsigned s=unsigned(__builtin_clzll(dh)-__builtin_clzll(nh));
 __uint128_t n=(__uint128_t(nh)<<64)|nl,d=((__uint128_t(dh)<<64)|dl)<<s;uint64_t q=0;
 do{q<<=1;if(n>=d){n-=d;++q;}d>>=1;}while(s--);
 r[0]=uint64_t(n);r[1]=uint64_t(n>>64);return q;
}
template<bool Hardware>inline uint64_t div2(uint64_t*r,uint64_t nh,uint64_t nl,uint64_t dh,uint64_t dl){
 if constexpr(Hardware){const uint64_t q0=nh/dh;
  if(q0<=dh){uint64_t q=q0;__uint128_t p=__uint128_t(q)*dl;
   __uint128_t n=(__uint128_t(nh-q*dh)<<64)|nl;
   if(p>n){--q;n+=__uint128_t(dh)<<64;n+=dl;}
   n-=p;r[0]=uint64_t(n);r[1]=uint64_t(n>>64);return q;}}
 return div2_bits(r,nh,nl,dh,dl);
}
template<unsigned Stop=33,bool Hardware=false,class Observer=IgnoreQuotient> inline bool
head_matrix (uint64_t ah, uint64_t al, uint64_t bh, uint64_t bl,
	   Matrix *M, Observer observe={})
{
  uint64_t u00, u01, u10, u11;

  if (ah < 2 || bh < 2)
    return 0;

  if (ah > bh || (ah == bh && al > bl))
    {
      sub_ddmmss (ah, al, ah, al, bh, bl);
      if (ah < 2)
	return 0;

      u00 = u01 = u11 = 1;
      u10 = 0;
      observe(1,1);
    }
  else
    {
      sub_ddmmss (bh, bl, bh, bl, ah, al);
      if (bh < 2)
	return 0;

      u00 = u10 = u11 = 1;
      u01 = 0;
      observe(0,1);
    }

  if (ah < bh)
    goto subtract_a;

  for (;;)
    {
      GCD_HEAD_ASSERT(ah >= bh);
      if (ah == bh)
	goto done;

      if (ah < (uint64_t(1) << (32)))
	{
	  ah = (ah << (32) ) + (al >> (32));
	  bh = (bh << (32) ) + (bl >> (32));

	  break;
	}

      /* Subtract a -= q b, and multiply M from the right by (1 q ; 0
	 1), affecting the second column of M. */
      GCD_HEAD_ASSERT(ah > bh);
      sub_ddmmss (ah, al, ah, al, bh, bl);

      if (ah < 2)
	goto done;

      if (ah <= bh)
	{
	  /* Use q = 1 */
	  u01 += u00;
	  u11 += u10; observe(1,1);
	}
      else
	{
	  uint64_t r[2];
	  uint64_t q = div2<Hardware> (r, ah, al, bh, bl);
	  al = r[0]; ah = r[1];
	  if (ah < 2)
	    {
	      /* A is too small, but q is correct. */
	      u01 += q * u00;
	      u11 += q * u10; observe(1,q);
	      goto done;
	    }
	  q++;
	  u01 += q * u00;
	  u11 += q * u10; observe(1,q);
	}
    subtract_a:
      GCD_HEAD_ASSERT(bh >= ah);
      if (ah == bh)
	goto done;

      if (bh < (uint64_t(1) << (32)))
	{
	  ah = (ah << (32) ) + (al >> (32));
	  bh = (bh << (32) ) + (bl >> (32));

	  goto subtract_a1;
	}

      /* Subtract b -= q a, and multiply M from the right by (1 0 ; q
	 1), affecting the first column of M. */
      sub_ddmmss (bh, bl, bh, bl, ah, al);

      if (bh < 2)
	goto done;

      if (bh <= ah)
	{
	  /* Use q = 1 */
	  u00 += u01;
	  u10 += u11; observe(0,1);
	}
      else
	{
	  uint64_t r[2];
	  uint64_t q = div2<Hardware> (r, bh, bl, ah, al);
	  bl = r[0]; bh = r[1];
	  if (bh < 2)
	    {
	      /* B is too small, but q is correct. */
	      u00 += q * u01;
	      u10 += q * u11; observe(0,q);
	      goto done;
	    }
	  q++;
	  u00 += q * u01;
	  u10 += q * u11; observe(0,q);
	}
    }

  /* NOTE: Since we discard the least significant half limb, we don't get a
     truly maximal M (corresponding to |a - b| < 2^{GMP_LIMB_BITS +1}). */
  /* Single precision loop */
  for (;;)
    {
      GCD_HEAD_ASSERT(ah >= bh);

      ah -= bh;
      if (ah < (uint64_t(1) << (Stop)))
	break;

      if (ah <= bh)
	{
	  /* Use q = 1 */
	  u01 += u00;
	  u11 += u10; observe(1,1);
	}
      else
	{
	  QR rq = div1 (ah, bh);
	  uint64_t q = rq.d1;
	  ah = rq.d0;

	  if (ah < (uint64_t(1) << (Stop)))
	    {
	      /* A is too small, but q is correct. */
	      u01 += q * u00;
	      u11 += q * u10; observe(1,q);
	      break;
	    }
	  q++;
	  u01 += q * u00;
	  u11 += q * u10; observe(1,q);
	}
    subtract_a1:
      GCD_HEAD_ASSERT(bh >= ah);

      bh -= ah;
      if (bh < (uint64_t(1) << (Stop)))
	break;

      if (bh <= ah)
	{
	  /* Use q = 1 */
	  u00 += u01;
	  u10 += u11; observe(0,1);
	}
      else
	{
	  QR rq = div1 (bh, ah);
	  uint64_t q = rq.d1;
	  bh = rq.d0;

	  if (bh < (uint64_t(1) << (Stop)))
	    {
	      /* B is too small, but q is correct. */
	      u00 += q * u01;
	      u10 += q * u11; observe(0,q);
	      break;
	    }
	  q++;
	  u00 += q * u01;
	  u10 += q * u11; observe(0,q);
	}
    }

 done:
  M->u[0][0] = u00; M->u[0][1] = u01;
  M->u[1][0] = u10; M->u[1][1] = u11;

  return 1;
}


#undef sub_ddmmss
#undef GCD_HEAD_ASSERT
}
