#pragma once
#include "backend/pq16/roots.hpp"
#include <array>
// Published, immutable engine constants. No runtime initializer or retained
// operation state: the small pow2 stages share these tables like the leaf
// constants. Larger stages keep the ordinary caller-owned table builder.
namespace sbn::v3::pq16::root_bank {
inline constexpr unsigned max_log2=12;
inline constexpr unsigned branch=1u<<max_log2;
struct Root {double re,im;};
constexpr Root root(uint64_t k,unsigned lg){
    const uint64_t n=uint64_t(1)<<lg,quarter=n/4;
    k&=n-1;const unsigned q=unsigned(k>>(lg-2));k&=quarter-1;
    const bool reflect=k>n/8;if(reflect)k=quarter-k;
    const uint64_t index=k<<(PQ16_ROOT_GRID_LOG2-lg);
    const auto *a=PQ16_ROOT_COARSE[index>>8],*b=PQ16_ROOT_FINE[index&255];
    double c=double(a[0]*b[0]-a[1]*b[1]),s=double(a[0]*b[1]+a[1]*b[0]);
    if(reflect){const double t=c;c=s;s=t;}
    switch(q){case 0:return {c,s};case 1:return {-s,c};case 2:return {-c,-s};default:return {s,-c};}
}
constexpr uint32_t reverse(uint32_t k,unsigned bits){
    uint32_t r=0;for(unsigned j=0;j<bits;++j){r=(r<<1)|(k&1);k>>=1;}return r;
}
template<unsigned Lg,bool Radix8>constexpr auto make_stage(){
    constexpr unsigned n=1u<<Lg;std::array<double,n> out{};
    for(unsigned group=0;group<n/64;++group){
        const unsigned j=group*(Radix8?8:16);
        for(unsigned block=0;block<4;++block)for(unsigned lane=0;lane<8;++lane){
            const unsigned scale=Radix8?(block<2?1:1u<<(block-1)):(block&1?2:1);
            const unsigned k=Radix8?(j+lane)*scale+(block==1?n/8:0):(j+lane+(block>=2?8:0))*scale;
            const Root w=root(k,Lg);const size_t at=64*group+16*block+lane;
            out[at]=w.re;out[at+8]=w.im;
        }
    }
    return out;
}
template<unsigned Lg,bool Radix8>alignas(128) inline constexpr auto stage=make_stage<Lg,Radix8>();
template<bool Radix8>inline const double *twiddle(unsigned lg){
    switch(lg){
    case 6:return stage<6,Radix8>.data();case 7:return stage<7,Radix8>.data();
    case 8:return stage<8,Radix8>.data();case 9:return stage<9,Radix8>.data();
    case 10:return stage<10,Radix8>.data();case 11:return stage<11,Radix8>.data();
    case 12:return stage<12,Radix8>.data();default:return nullptr;
    }
}
constexpr auto make_pq(){
    std::array<double,branch/2> out{};
    for(unsigned g=0;g<branch/64;++g)for(unsigned k=0;k<16;++k){
        const auto w=root(reverse(g*16+k,max_log2-2),max_log2);
        out[32*g+k]=w.re;out[32*g+16+k]=w.im;
    }
    return out;
}
alignas(128) inline constexpr auto pq=make_pq();
// Two stage kinds at 64..4096 plus the prefix-stable PQ table.
inline constexpr size_t bytes=2*(2*branch-64)*sizeof(double)+sizeof(pq);
}
