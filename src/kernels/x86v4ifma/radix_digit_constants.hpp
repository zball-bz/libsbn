#pragma once
// Per-base integer codec constants, evaluated at compile time. No runtime initializer.
#include <stdint.h>
#include <array>
namespace sbn::v3::radix::leaf_detail {
struct DigitConstants {
    uint64_t b2=0,b4=0,b8=0,m4=0,m2=0;unsigned s4=0,s2=0;uint16_t m1=0;unsigned s1=0;
};
constexpr bool constant_magic(uint64_t d,uint64_t limit,unsigned W,uint64_t cap,uint64_t&m,unsigned&s){
    using Wide=__uint128_t;
    for(unsigned k=W;k-->0;){
        if(k<63&&d<=(uint64_t(1)<<k))continue;
        const Wide p=Wide(1)<<(W+k);const uint64_t v=uint64_t((p+d-1)/d);
        if(v<cap&&(Wide(v)*d-p)*(limit-1)<p){m=v;s=k;return true;}
    }
    return false;
}
constexpr auto make_digit_constants(){
    std::array<DigitConstants,65> out{};
    for(unsigned b=3;b<64;++b){
        if(!(b&(b-1)))continue;
        auto&p=out[b];p.b2=b*b;p.b4=p.b2*p.b2;p.b8=p.b4*p.b4;
        uint64_t m=0;unsigned s=0;
        if(!constant_magic(p.b4,p.b8,52,uint64_t(1)<<52,p.m4,p.s4))return std::array<DigitConstants,65>{};
        if(!constant_magic(p.b2,p.b4,52,uint64_t(1)<<52,p.m2,p.s2))return std::array<DigitConstants,65>{};
        if(!constant_magic(b,p.b2,16,uint64_t(1)<<16,m,s))return std::array<DigitConstants,65>{};
        p.m1=uint16_t(m);p.s1=s;
    }
    return out;
}
inline constexpr auto digit_constants=make_digit_constants();
static_assert(digit_constants[3].m4!=0);
static_assert(digit_constants[10].m4!=0);
static_assert(digit_constants[63].m4!=0);
}
