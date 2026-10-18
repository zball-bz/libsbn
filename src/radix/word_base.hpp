#pragma once
#include <array>
#include <stdint.h>
namespace sbn::v3::radix {
// Largest whole radix power in one u64; independent of the value and size.
// The reciprocal gives a quotient underestimate by at most one for any u64
// numerator, followed by one exact correction. It splits a word into b^8
// groups for the existing SIMD digit emitter without hardware division.
struct WordBase {uint64_t power=0,reciprocal8=0;unsigned digits=0;uint64_t inverse=0;unsigned shift=0;};
constexpr auto make_word_bases(unsigned bits=64,unsigned reciprocal_bits=64){
    std::array<WordBase,65> out{};
    for(unsigned b=2;b<=64;++b){
        uint64_t p=1,b8=1;unsigned k=0;
        for(unsigned j=0;j<8;++j)b8*=b;
        while((__uint128_t)p*b<(__uint128_t(1)<<bits)){p*=b;++k;}
        out[b]={p,uint64_t((__uint128_t(1)<<reciprocal_bits)/b8),k};
        out[b].shift=unsigned(__builtin_clzll(p));
        const uint64_t normalized=p<<out[b].shift;
        out[b].inverse=uint64_t(((__uint128_t(~normalized)<<64)|UINT64_MAX)/normalized);
    }
    return out;
}
inline constexpr auto word_bases=make_word_bases();
// A redundant radix-2^51 fraction needs D < 2^50 to keep every digit below
// 2^52. Its emitted word uses a 52-bit reciprocal rather than a 64-bit one.
inline constexpr auto fraction_bases=make_word_bases(50,52);
inline uint64_t split_radix_word(uint64_t &q,uint64_t b8,uint64_t reciprocal) noexcept {
    uint64_t hi=uint64_t((__uint128_t(q)*reciprocal)>>64);
    uint64_t r=q-hi*b8;
    if(r>=b8){++hi;r-=b8;}
    q=hi;return r;
}
}
