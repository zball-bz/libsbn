/* jacobi_2.c

   THE FUNCTIONS IN THIS FILE ARE INTERNAL WITH MUTABLE INTERFACES.  IT IS ONLY
   SAFE TO REACH THEM THROUGH DOCUMENTED INTERFACES.  IN FACT, IT IS ALMOST
   GUARANTEED THAT THEY'LL CHANGE OR DISAPPEAR IN A FUTURE GNU MP RELEASE.

Copyright 1996, 1998, 2000-2004, 2008, 2010 Free Software Foundation, Inc.

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

// Half-odd binary representation adapted from GNU MP jacobi_2 methods 1 and 2.
// Generic unsigned-width loop and word narrowing are libsbn adaptations.
#pragma once
#include <stdint.h>
#include "gcd_odd_reduce.hpp"
namespace sbn::v3::gcd_small {
template<class T>inline unsigned jacobi_zeros(T a){
    if constexpr(sizeof(T)==8)return __builtin_ctzll(a);
    else return uint64_t(a)?unsigned(__builtin_ctzll(uint64_t(a))):64+unsigned(__builtin_ctzll(uint64_t(a>>64)));
}
template<class T>static int jacobi_half(T a,T b,unsigned e){
    // Values denote the positive odd integers 2*a+1 and 2*b+1.
    // They occupy at most width-1 bits, so the difference's high bit
    // provides a full-width swap mask without an extra comparison.
    if(!a||!b)return 1-2*int(e&1);
    do{
        if(a==b)return 0;
        if constexpr(sizeof(T)>8)if(!((a|b)>>63))return jacobi_half(uint64_t(a),uint64_t(b),e);
        const T delta=a-b,mask=T(0)-(delta>>(8*sizeof(T)-1));
        e^=unsigned(mask&a&b);b+=delta&mask;
        const unsigned shift=jacobi_zeros(delta)+1;
        e^=shift&unsigned(b^(b>>1));
        a=((delta^mask)-mask)>>shift;
    }while(a);
    return 1-2*int(e&1);
}
template<class T>static int jacobi_binary(T a,T b){
    if(!a)return b==1;
    const unsigned shift=jacobi_zeros(a);
    const unsigned e=shift&unsigned((b>>1)^(b>>2));
    return jacobi_half((a>>shift)>>1,b>>1,e);
}
// The two-word loop follows the directional binary reduction of GMP
// jacobi_2 method 2. A nonzero low word proves shift<64, avoiding the
// general 128-bit variable shift on the common path.
static __uint128_t strip128(__uint128_t a,uint64_t b,unsigned &e){
    const uint64_t lo=a,hi=a>>64;unsigned shift;
    if(lo){shift=__builtin_ctzll(lo);a>>=shift;}
    else {shift=__builtin_ctzll(hi);a=hi>>shift;}
    e^=shift&unsigned((b>>1)^(b>>2));return a;
}
static int jacobi128(__uint128_t a,__uint128_t b){
    if(!a)return b==1;
    if(b==1)return 1;
    unsigned e=0;a=strip128(a,uint64_t(b),e);
    while((a>>64)&&(b>>64)){
        while((a>>64)>(b>>64))a=strip128(a-b,uint64_t(b),e);
        if((a>>64)==(b>>64)){
            if(a==b)return 0;
            if(uint64_t(a)<uint64_t(b)){e^=unsigned(a&b)>>1;auto t=a;a=b;b=t;}
            a=strip128(a-b,uint64_t(b),e);break;
        }
        e^=unsigned(a&b)>>1;auto t=a;a=b;b=t;
    }
    if(b>>64){e^=unsigned(a&b)>>1;auto t=a;a=b;b=t;}
    const uint64_t den=b;
    if(den==1)return 1-2*int(e&1);
    if(a>>64){uint64_t words[2]{uint64_t(a),uint64_t(a>>64)};
        a=odd_gcd_residue(words,2,den);e^=unsigned(den)>>1;}
    return (1-2*int(e&1))*jacobi_binary(uint64_t(a),den);
}
}
