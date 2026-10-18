#pragma once
#include <array>
#include <stdint.h>
namespace sbn::v3::radix::leaf_detail {
struct alignas(64) IntegerPackMap {
    std::array<uint64_t,8> i{},j{},k{},right{},left{},extra{};
    uint8_t m0=0,m1=0,m2=0;
};
// Coefficients of sum(w[j] * base^(8*j)), in radix 2^52. Each input
// word is below 2^48. Sixteen low products sum below 2^56; the corresponding
// high products sum below 2^52. Carry normalization is exact, including a
// carry propagating through an arbitrarily long run of maximal digits.
struct alignas(64) ParseRadix {
    uint64_t power[16][16]{};
    unsigned one_vector_words=0;
};
consteval auto make_parse_radices() {
    std::array<ParseRadix,65> a{};
    constexpr uint64_t mask=(uint64_t(1)<<52)-1;
    for(unsigned b=3;b<=63;++b){
        uint64_t step=1;for(unsigned j=0;j<8;++j)step*=b;
        uint64_t v[16]{1};
        for(unsigned j=0;j<16;++j){
            for(unsigned k=0;k<16;++k)a[b].power[j][k]=v[k];
            bool fits=true;for(unsigned k=8;k<16;++k)fits&=v[k]==0;
            if(fits)a[b].one_vector_words=j+1;
            uint64_t carry=0;
            for(unsigned k=0;k<16;++k){const __uint128_t x=__uint128_t(v[k])*step+carry;
                v[k]=uint64_t(x)&mask;carry=uint64_t(x>>52);}
        }
    }
    return a;
}
inline constexpr auto parse_radices=make_parse_radices();
}
