/* mpn_modexact_1c_odd -- mpn by limb exact division style remainder.

   THE FUNCTIONS IN THIS FILE ARE FOR INTERNAL USE ONLY.  THEY'RE ALMOST
   CERTAIN TO BE SUBJECT TO INCOMPATIBLE CHANGES OR DISAPPEAR COMPLETELY IN
   FUTURE GNU MP RELEASES.

Copyright 2000-2004 Free Software Foundation, Inc.

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


// The scalar cancellation recurrence is adapted from GNU MP mode1o.c.
// Four independent equal-length chunks and their recombination are libsbn's
// extension. This returns a GCD-equivalent residue, NOT a public remainder.
#pragma once
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include "common/small_checks.h"
namespace sbn::v3::gcd_small {
static inline uint64_t odd_step(uint64_t x,uint64_t c,uint64_t d,uint64_t inverse){
    const uint64_t low=x-c,q=low*inverse;
    return uint64_t((__uint128_t(q)*d)>>64)+uint64_t(x<c);
}
static inline uint64_t reduce_product(__uint128_t x,uint64_t d){
    uint64_t lo=uint64_t(x),hi=uint64_t(x>>64);
    // Caller proves x<d*2^64, so the hardware quotient fits one word.
    if constexpr(SBN3_CHECK_SMALL)assert(hi<d);
    __asm__("divq %2" : "+a"(lo),"+d"(hi) : "r"(d) : "cc");
    return hi;
}
static uint64_t radix_power(size_t exponent,uint64_t d){
    uint64_t b=UINT64_MAX%d+1;if(b==d)b=0;uint64_t r=1;
    while(exponent){if(exponent&1)r=reduce_product(__uint128_t(r)*b,d);exponent>>=1;
        if(exponent)b=reduce_product(__uint128_t(b)*b,d);}
    return r;
}
static uint64_t odd_gcd_residue(const uint64_t*a,size_t n,uint64_t d){
    if(d==1)return 0;
    uint64_t inverse=(3*d)^2;
    inverse*=2-d*inverse;inverse*=2-d*inverse;inverse*=2-d*inverse;inverse*=2-d*inverse;
    if(n<32){uint64_t c=0;for(size_t j=0;j<n;++j)c=odd_step(a[j],c,d,inverse);return c;}
    const size_t length=(n+3)/4,last=n-3*length;uint64_t c0=0,c1=0,c2=0,c3=0;
    size_t j=0;
    for(;j<last;++j){c0=odd_step(a[j],c0,d,inverse);c1=odd_step(a[length+j],c1,d,inverse);
        c2=odd_step(a[2*length+j],c2,d,inverse);c3=odd_step(a[3*length+j],c3,d,inverse);}
    for(;j<length;++j){c0=odd_step(a[j],c0,d,inverse);c1=odd_step(a[length+j],c1,d,inverse);
        c2=odd_step(a[2*length+j],c2,d,inverse);c3=odd_step(0,c3,d,inverse);}
    const uint64_t power=radix_power(length,d);
    uint64_t r=reduce_product(__uint128_t(c3)*power+c2,d);
    r=reduce_product(__uint128_t(r)*power+c1,d);
    return reduce_product(__uint128_t(r)*power+c0,d);
}
}
