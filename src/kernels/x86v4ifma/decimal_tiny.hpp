#pragma once
#include "sbn3/decimal.h"
#include <array>
#include <immintrin.h>
#include <stdint.h>

namespace sbn::v3::tiny_decimal {
using Wide=__uint128_t;
inline constexpr uint64_t d8=100000000ULL,d16=10000000000000000ULL;
inline constexpr Wide d32=Wide(d16)*d16;
inline constexpr auto powers=[] {
    std::array<Wide,39> p{};p[0]=1;
    for(unsigned i=1;i<p.size();++i)p[i]=p[i-1]*10;
    return p;
}();
static_assert([] {
    // A binary bit-length bin crosses at most one decimal power. Exact
    // endpoint checks therefore certify the multiply/shift estimate for
    // every value in all 128 bins, without a runtime logarithm.
    auto length=[](Wide x){unsigned n=1;while(x>=10){x/=10;++n;}return n;};
    for(unsigned bits=1;bits<=128;++bits){
        const Wide lo=Wide(1)<<(bits-1),hi=bits==128?~Wide(0):(lo<<1)-1;
        const unsigned n=bits*1233>>12;
        if(n+unsigned(lo>=powers[n])!=length(lo)||n+unsigned(hi>=powers[n])!=length(hi))return false;
    }return true;
}());
inline unsigned length64(uint64_t x) noexcept {
    const unsigned bits=64-unsigned(__builtin_clzll(x|1));
    const unsigned n=bits*1233>>12;
    return n+unsigned((x|1)>=uint64_t(powers[n]));
}
inline unsigned length128(uint64_t lo,uint64_t hi) noexcept {
    const unsigned bits=128-unsigned(__builtin_clzll(hi)); // hi != 0
    const unsigned n=bits*1233>>12;
    const Wide x=(Wide(hi)<<64)|lo;
    return n+unsigned(x>=powers[n]);
}
// Parallel digit positions, rather than eight independent eight-digit words.
// Gareau/Lemire's biased-reciprocal identity, also credited by xjb:
// digit(k)=floor(10*((floor(2^52/10^k)*(n+1)) mod 2^52)/2^52), n<10^8.
// https://lemire.me/blog/2026/05/18/simd-accelerated-integer-to-string-conversion/
static_assert([] {
    // For y=n+1<=10^8, the reciprocal error is less than 1/D. The bias
    // then puts the scaled fractional part strictly inside the digit's
    // interval [digit/10,(digit+1)/10), including decimal boundaries.
    constexpr uint64_t b=uint64_t(1)<<52;
    for(uint64_t d=10;d<=d8;d*=10)if((b%d)*d8>=b)return false;
    return true;
}());
[[gnu::always_inline]] inline __m128i digits8(uint64_t n) noexcept {
    constexpr uint64_t b=uint64_t(1)<<52;
    const auto c=_mm512_setr_epi64(b/100000000,b/10000000,b/1000000,b/100000,
                                 b/10000,b/1000,b/100,b/10);
    const auto r=_mm512_madd52lo_epu64(c,_mm512_set1_epi64(n),c);
    const auto d=_mm512_madd52hi_epu64(_mm512_set1_epi64('0'),r,_mm512_set1_epi64(10));
    return _mm512_cvtepi64_epi8(d);
}
[[gnu::always_inline]] inline __m128i indices16() noexcept {
    return _mm_setr_epi8(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
}
[[gnu::always_inline]] inline __m256i indices32() noexcept {
    return _mm256_setr_epi64x(0x0706050403020100ULL,0x0f0e0d0c0b0a0908ULL,
                            0x1716151413121110ULL,0x1f1e1d1c1b1a1918ULL);
}
[[gnu::always_inline]] inline __m512i indices64() noexcept {
    return _mm512_setr_epi64(0x0706050403020100ULL,0x0f0e0d0c0b0a0908ULL,
        0x1716151413121110ULL,0x1f1e1d1c1b1a1918ULL,0x2726252423222120ULL,
        0x2f2e2d2c2b2a2928ULL,0x3736353433323130ULL,0x3f3e3d3c3b3a3938ULL);
}
[[gnu::always_inline]] inline char *write64(char *out,uint64_t x) noexcept {
    const unsigned n=length64(x);
    const uint64_t q8=x/d8,q16=x/d16;
    const auto a=digits8(q16),b=digits8(q8-q16*d8),c=digits8(x-q8*d8);
    auto text=_mm256_inserti128_si256(_mm256_castsi128_si256(_mm_unpacklo_epi64(a,b)),c,1);
    // Bias the constant vector, not an 8-bit GPR temporary: a partial-byte
    // update before vpbroadcastb can retain a previous iteration's register
    // dependency after LTO inlining on x86.
    text=_mm256_permutexvar_epi8(_mm256_sub_epi8(_mm256_add_epi8(indices32(),_mm256_set1_epi8(24)),_mm256_set1_epi8(n)),text);
    _mm256_mask_storeu_epi8(out,__mmask32((uint32_t(1)<<n)-1),text);
    return out+n;
}
// q0=floor(hi*floor(2^128/D)/2^64) is at most one below floor(x/D)
// whenever (2^128 mod D)+2^64<D. Thus D=10^32 needs just the high input
// word for its estimate, one exact residual, and one bounded correction.
inline constexpr uint64_t reciprocal32=uint64_t(~Wide(0)/d32); // 3402823
static_assert((Wide(-1)%d32+1)+(Wide(1)<<64)<d32);
[[gnu::always_inline]] inline uint64_t split32(Wide &x) noexcept {
    uint64_t q=uint64_t((Wide(uint64_t(x>>64))*reciprocal32)>>64);
    x-=Wide(q)*d32;
    const bool fix=x>=d32;x-=fix?d32:0;
    return q+fix;
}
/* Constant normalized 2/1 division, adapted from GMP 6.3.0's
 * udiv_qrnnd_preinv through the existing divrem_words.hpp core.
 * Copyright 1991-2016 Free Software Foundation, Inc.
 * Retained terms: src/core/x86_64/scalar-donor-notice.txt (LGPLv3+ / GPLv2+). */
[[gnu::always_inline]] inline uint64_t split16(Wide x,uint64_t &remainder) noexcept {
    constexpr uint64_t d=d16<<10;
    constexpr uint64_t inv=uint64_t((~Wide(0))/d);
    x<<=10;const uint64_t nh=uint64_t(x>>64),nl=uint64_t(x);
    Wide estimate=Wide(nh)*inv;
    estimate+=Wide(nh+1)<<64;estimate+=nl;
    uint64_t q=uint64_t(estimate>>64),r=nl-q*d;
    const uint64_t mask=0-uint64_t(r>uint64_t(estimate));
    q+=mask;r+=mask&d;
    const bool fix=r>=d;q+=fix;r-=fix?d:0;
    remainder=r>>10;return q;
}
[[gnu::always_inline]] inline char *write128(char *out,sbn3_uint128 v) noexcept {
    if(!v.hi)return write64(out,v.lo);
    const unsigned n=length128(v.lo,v.hi);
    Wide value=(Wide(v.hi)<<64)|v.lo;
    const uint64_t a=split32(value);uint64_t c;
    const uint64_t b=split16(value,c),bh=b/d8,ch=c/d8;
    const auto p0=digits8(a),p1=digits8(bh),p2=digits8(b-bh*d8),p3=digits8(ch),p4=digits8(c-ch*d8);
    auto text=_mm512_inserti64x2(_mm512_setzero_si512(),_mm_unpacklo_epi64(p0,p1),0);
    text=_mm512_inserti64x2(text,_mm_unpacklo_epi64(p2,p3),1);
    text=_mm512_inserti64x2(text,p4,2);
    text=_mm512_permutexvar_epi8(_mm512_sub_epi8(_mm512_add_epi8(indices64(),_mm512_set1_epi8(40)),_mm512_set1_epi8(n)),text);
    _mm512_mask_storeu_epi8(out,__mmask64((uint64_t(1)<<n)-1),text);
    return out+n;
}
// Sixteen decimal digits -> two base-10^8 words. Validation runs alongside
// the reduction, with a single decision at the public parse boundary.
[[gnu::always_inline]] inline uint64_t parse16(const char *in,unsigned n,uint64_t &bad) noexcept {
    const auto mask=__mmask16((uint32_t(1)<<n)-1);
    auto d=_mm_maskz_loadu_epi8(mask,in);
    d=_mm_sub_epi8(d,_mm_maskz_set1_epi8(mask,'0'));
    bad|=_mm_mask_cmpgt_epu8_mask(mask,d,_mm_set1_epi8(9));
    d=_mm_shuffle_epi8(d,_mm_add_epi8(_mm_sub_epi8(indices16(),_mm_set1_epi8(16)),_mm_set1_epi8(n)));
    const auto pairs=_mm_maddubs_epi16(d,_mm_set1_epi16(0x010a));
    const auto quads=_mm_madd_epi16(pairs,_mm_set1_epi32(0x00010064));
    const auto octs=_mm_madd_epi16(_mm_packus_epi32(quads,_mm_setzero_si128()),_mm_set1_epi32(0x00012710));
    const uint64_t two=uint64_t(_mm_cvtsi128_si64(octs));
    return uint64_t(uint32_t(two))*d8+(two>>32);
}
inline bool trim(const char *&in,size_t &n,unsigned limit) noexcept {
    if(n<=limit)return false;
    while(n>limit&&*in=='0'){++in;--n;}
    return n>limit;
}
inline sbn3_decimal_status too_long(const char *in,size_t n) noexcept {
    for(size_t j=0;j<n;++j)if(uint8_t(uint8_t(in[j])-'0')>9)return SBN3_DECIMAL_INVALID;
    return SBN3_DECIMAL_OVERFLOW;
}
[[gnu::always_inline]] inline Wide parse20(const char *in,unsigned count,uint64_t &bad) noexcept {
    const auto mask=__mmask32(~uint32_t(0)>>(32-count));
    auto d=_mm256_maskz_loadu_epi8(mask,in);
    d=_mm256_sub_epi8(d,_mm256_maskz_set1_epi8(mask,'0'));
    bad|=_mm256_mask_cmpgt_epu8_mask(mask,d,_mm256_set1_epi8(9));
    const unsigned skip=32-unsigned(count);
    d=_mm256_maskz_permutexvar_epi8(__mmask32(~uint32_t(0)<<skip),
        _mm256_add_epi8(_mm256_sub_epi8(indices32(),_mm256_set1_epi8(32)),_mm256_set1_epi8(count)),d);
    const auto pairs=_mm256_maddubs_epi16(d,_mm256_set1_epi16(0x010a));
    const auto quads=_mm256_madd_epi16(pairs,_mm256_set1_epi32(0x00010064));
    const auto octs=_mm256_madd_epi16(_mm256_packus_epi32(quads,_mm256_setzero_si256()),_mm256_set1_epi32(0x00012710));
    const auto first=uint64_t(_mm_cvtsi128_si64(_mm256_castsi256_si128(octs)));
    const uint64_t prefix=first>>32; // the leading eight-digit word is zero for <=20 digits
    const auto suffix=uint64_t(_mm_cvtsi128_si64(_mm256_extracti128_si256(octs,1)));
    return Wide(prefix)*d16+uint64_t(uint32_t(suffix))*d8+(suffix>>32);
}
[[gnu::always_inline]] inline sbn3_decimal_status read64(uint64_t *out,const char *in,size_t count) noexcept {
    if(!count)return SBN3_DECIMAL_INVALID;
    if(trim(in,count,20))return too_long(in,count);
    uint64_t bad=0;
    const Wide value=count<=16?Wide(parse16(in,unsigned(count),bad)):parse20(in,unsigned(count),bad);
    if(bad)return SBN3_DECIMAL_INVALID;
    if(value>>64)return SBN3_DECIMAL_OVERFLOW;
    *out=uint64_t(value);return SBN3_DECIMAL_OK;
}
[[gnu::always_inline]] inline sbn3_decimal_status read128(sbn3_uint128 *out,const char *in,size_t count) noexcept {
    if(!count)return SBN3_DECIMAL_INVALID;
    if(trim(in,count,39))return too_long(in,count);
    uint64_t bad=0;Wide value;
    if(count<=16)value=parse16(in,unsigned(count),bad);
    else if(count<=32)value=Wide(parse16(in,unsigned(count-16),bad))*d16+parse16(in+count-16,16,bad);
    else{
        const uint64_t a=parse16(in,unsigned(count-32),bad),b=parse16(in+count-32,16,bad),c=parse16(in+count-16,16,bad);
        if(bad)return SBN3_DECIMAL_INVALID;
        if(a>3402823||(a==3402823&&(b>6692093846346337ULL||(b==6692093846346337ULL&&c>4607431768211455ULL))))return SBN3_DECIMAL_OVERFLOW;
        value=Wide(a)*d32+Wide(b)*d16+c;
    }
    if(bad)return SBN3_DECIMAL_INVALID;
    *out={uint64_t(value),uint64_t(value>>64)};return SBN3_DECIMAL_OK;
}
}
