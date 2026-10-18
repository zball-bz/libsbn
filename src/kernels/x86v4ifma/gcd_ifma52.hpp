// Original libsbn GCD kernels; LGPL-3.0-or-later.
#pragma once
#include <cstdint>
#include <immintrin.h>

namespace sbn::v3::gcd_small::ifma52 {
using U=uint64_t;
using V=__m512i;
inline constexpr U digit_mask=(U(1)<<52)-1;
struct Product { V lo,hi; };

// x,y,c,d are base-2^52 digits; neg=(-d) mod 2^52 and nonzero is
// all lanes iff d!=0. c*x-d*y = c*x+neg*y-[d!=0]*2^52*y.
// The low sum is deliberately redundant (up to 2*(2^52-1)).
static inline Product difference(V x,V y,V c,V neg,__mmask8 nonzero) {
    const V zero=_mm512_setzero_si512();
    const V lo=_mm512_madd52lo_epu64(_mm512_madd52lo_epu64(zero,x,c),y,neg);
    const V hi=_mm512_madd52hi_epu64(_mm512_madd52hi_epu64(zero,x,c),y,neg);
    return {lo,_mm512_mask_sub_epi64(hi,nonzero,hi,y)};
}

// Normalize eight signed redundant digits. Only carry's last lane is live
// across blocks. lo,hi and the incoming carry are small enough that their
// sums fit signed 64 bits (the row products above are bounded by 2^105).
static inline V carry(V lo,V hi,V& state) {
    const V zero=_mm512_setzero_si512(),mask=_mm512_set1_epi64(digit_mask);
    const V t=_mm512_add_epi64(lo,_mm512_alignr_epi64(hi,state,7));
    const V q=_mm512_srai_epi64(t,52),r=_mm512_and_si512(t,mask);
    const V candidate=_mm512_add_epi64(r,_mm512_alignr_epi64(q,zero,7));
    // If all corrected digits fit, none changes its provisional carry.
    // Unsigned comparison also catches a negative candidate. This is an
    // exact check, including zero/maximal digits, not a probabilistic bound.
    if(!_mm512_cmpgt_epu64_mask(candidate,mask)) {
        state=_mm512_add_epi64(hi,q);
        return candidate;
    }
    alignas(64) int64_t l[8],h[8];alignas(64) U out[8];
    _mm512_store_si512(l,lo);_mm512_store_si512(h,hi);
    __int128 c=int64_t(_mm_extract_epi64(_mm512_extracti64x2_epi64(state,3),1));
    for(unsigned i=0;i<8;++i) {
        const __int128 v=__int128(l[i])+c;
        out[i]=U(v)&digit_mask;c=__int128(h[i])+(v>>52);
    }
    state=_mm512_set1_epi64(U(c));
    return _mm512_load_si512(out);
}
}
