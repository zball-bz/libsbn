#pragma once
// Fixed decimal leaves. Eight-digit scalar reductions use independent packed
// byte/word operations; the only vector operation in that path is a bounded
// masked load. All arithmetic is exact, including the digit-validity check.
#include <immintrin.h>
#include <stdint.h>
namespace sbn::v3::radix::decimal_detail {
inline uint64_t parse8(const uint8_t *p,unsigned n,bool ascii,uint64_t &bad) noexcept {
    const auto mask=__mmask16((1u<<n)-1);
    auto bytes=_mm_maskz_loadu_epi8(mask,p);
    if(ascii)bytes=_mm_sub_epi8(bytes,_mm_maskz_set1_epi8(mask,'0'));
    uint64_t x=uint64_t(_mm_cvtsi128_si64(bytes))<<((8-n)*8);
    // A valid byte is 0..9, so adding 0x76 neither carries nor sets bit 7.
    // At the first invalid byte either x or x+0x76 has bit 7 set.
    bad|=(x|(x+0x7676767676767676ULL))&0x8080808080808080ULL;
    x=x*10+(x>>8);
    // Four two-digit values. The two products are independent; their low
    // halves sum below 2^32 and cannot carry into the desired result.
    const uint64_t a=(x&0x000000ff000000ffULL)*0x000f424000000064ULL;
    const uint64_t b=((x>>16)&0x000000ff000000ffULL)*0x0000271000000001ULL;
    return (a+b)>>32;
}
[[gnu::always_inline]] inline uint64_t parse19(const uint8_t *p,unsigned n,bool ascii,uint64_t &bad) noexcept {
    if(n<=8)return parse8(p,n,ascii,bad);
    if(n<=16)return parse8(p,n-8,ascii,bad)*100000000ULL+parse8(p+n-8,8,ascii,bad);
    const auto a=parse8(p,n-16,ascii,bad),b=parse8(p+n-16,8,ascii,bad),c=parse8(p+n-8,8,ascii,bad);
    return a*10000000000000000ULL+b*100000000ULL+c;
}
inline bool parse_small(uint64_t *out,size_t stride,const uint8_t *p,unsigned n,bool ascii) noexcept {
    uint64_t bad=0;
    __uint128_t value;
    if(n<=19)value=parse19(p,n,ascii,bad);
    else value=__uint128_t(parse19(p,n-19,ascii,bad))*10000000000000000000ULL+parse19(p+n-19,19,ascii,bad);
    out[0]=uint64_t(value);if(stride>1)out[1]=uint64_t(value>>64);
    for(size_t j=2;j<stride;++j)out[j]=0;
    return !bad;
}
// Range-specific decimal division uses 32-bit products, rather than general
// b^4/b^2 division in 52-bit lanes. Quotient and remainder occupy disjoint
// halves after x + q*(2^w-D). This packing identity is described in xjb/zmij;
// the implementation here operates on eight full integer chunks.
// For m=ceil(2^s/D), xmax*(m*D-2^s)<2^s guarantees floor(x*m/2^s)
// equals floor(x/D) throughout 0..xmax. The three domains are nested.
static_assert(99999999ULL*(109951163ULL*10000-(1ULL<<40))<(1ULL<<40));
static_assert(9999ULL*(5243ULL*100-(1ULL<<19))<(1ULL<<19));
static_assert(99ULL*(6554ULL*10-(1ULL<<16))<(1ULL<<16));
inline __m512i emit_words(__m512i x) noexcept {
    const auto q4=_mm512_srli_epi64(_mm512_mul_epu32(x,_mm512_set1_epi64(109951163)),40);
    const auto y=_mm512_add_epi64(x,_mm512_mul_epu32(q4,_mm512_set1_epi64(0x100000000ULL-10000)));
    const auto q2=_mm512_srli_epi16(_mm512_mulhi_epu16(y,_mm512_set1_epi16(5243)),3);
    const auto z=_mm512_add_epi32(y,_mm512_mullo_epi32(q2,_mm512_set1_epi32(65536-100)));
    const auto q1=_mm512_mulhi_epu16(z,_mm512_set1_epi16(6554));
    const auto d=_mm512_add_epi16(z,_mm512_mullo_epi16(q1,_mm512_set1_epi16(256-10)));
    return _mm512_shuffle_epi8(d,_mm512_broadcast_i64x2(_mm_set_epi64x(0x08090a0b0c0d0e0fULL,0x0001020304050607ULL)));
}
inline unsigned format_limb(uint8_t *out,uint64_t v,bool ascii) noexcept {
    const uint64_t q8=v/100000000ULL,q16=v/10000000000000000ULL;
    auto bytes=emit_words(_mm512_set_epi64(v-q8*100000000ULL,q8-q16*100000000ULL,q16,0,0,0,0,0));
    const uint64_t nonzero=_mm512_test_epi8_mask(bytes,bytes);
    if(ascii)bytes=_mm512_add_epi8(bytes,_mm512_set1_epi8('0'));
    _mm512_storeu_si512(out,bytes);
    return nonzero?unsigned(__builtin_ctzll(nonzero)):64;
}
}
