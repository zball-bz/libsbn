// SPDX-License-Identifier: LGPL-3.0-or-later
#include "common/m61.hpp"
#include "common/m64.hpp"
#include <immintrin.h>

namespace sbn::v3::m61 {
uint64_t words_bulk(const void *data,size_t n) noexcept {
    const auto *p=static_cast<const uint8_t *>(data);
    const size_t groups=n/8,tail=n%8;
    const auto mask=_mm512_set1_epi64(modulus);
    auto r=_mm512_setzero_si512();
    if(tail){
        const auto x=_mm512_maskz_loadu_epi64(__mmask8((1u<<tail)-1),p+groups*64);
        r=_mm512_add_epi64(_mm512_and_si512(x,mask),_mm512_srli_epi64(x,61));
    }
    // B=2^64 is 8 modulo 2^61-1. Each lane advances eight words, hence
    // multiplication by 2^24, a rotation in the 61-bit residue. Fold without
    // canonicalizing each step: r<=modulus+7 initially, <=modulus+1 afterward.
    // Every sum below is <2^63, so ordinary u64 lane additions cannot overflow.
    for(size_t g=groups;g-- >0;){
        const auto x=_mm512_loadu_si512(p+g*64);
        const auto rotated=_mm512_add_epi64(_mm512_and_si512(_mm512_slli_epi64(r,24),mask),_mm512_srli_epi64(r,37));
        const auto input=_mm512_add_epi64(_mm512_and_si512(x,mask),_mm512_srli_epi64(x,61));
        const auto sum=_mm512_add_epi64(rotated,input);
        r=_mm512_add_epi64(_mm512_and_si512(sum,mask),_mm512_srli_epi64(sum,61));
    }
    uint64_t lanes[8];_mm512_storeu_si512(lanes,r);
    __uint128_t sum=0;for(unsigned i=0;i<8;++i)sum+=__uint128_t(lanes[i])<<(3*i);
    return reduce(sum);
}
} // namespace sbn::v3::m61
namespace sbn::v3::m64 {
uint64_t words_bulk(const void*data,size_t n) noexcept {
    const auto*p=static_cast<const uint8_t*>(data);auto sum=_mm512_setzero_si512();
    const auto ones=_mm512_set1_epi64(-1);size_t j=0;
    for(;j+8<=n;j+=8){const auto before=sum;sum=_mm512_add_epi64(sum,_mm512_loadu_si512(p+8*j));
        sum=_mm512_mask_sub_epi64(sum,_mm512_cmplt_epu64_mask(sum,before),sum,ones);}
    uint64_t lanes[8];_mm512_storeu_si512(lanes,sum);__uint128_t total=0;
    for(auto v:lanes)total+=v;
    for(;j<n;++j){uint64_t v;std::memcpy(&v,p+8*j,8);total+=v;}
    return reduce(total);
}
}
