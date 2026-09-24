#pragma once
#include "backend/pq16/roots.hpp"
#include <array>
#include <stddef.h>
#include <stdint.h>
// Published, immutable engine constants. No runtime initializer or retained
// operation state: the small pow2 stages share these tables like the leaf
// constants. Larger stages keep the ordinary caller-owned table builder.
namespace sbn::v3::pq16::root_bank {
inline constexpr unsigned max_log2=17;
inline constexpr unsigned compact_min_log2=16;
inline constexpr unsigned branch=1u<<max_log2;
// The next power-of-two product consumes its top radix-4 stage, but its
// radix-8 children are already in the common bank. Publish that one stage
// and the longer prefix-stable PQ table, without an unused radix-8 peer.
inline constexpr unsigned pq_max_log2=18;
inline constexpr unsigned pq_branch=1u<<pq_max_log2;
struct Root {double re,im;};
constexpr Root rational_root(uint64_t k,uint64_t n){
    k%=n;const unsigned quadrant=unsigned(k/(n/4));k%=n/4;
    const bool reflect=k>n/8;if(reflect)k=n/4-k;
    const uint64_t num=k<<PQ16_ROOT_GRID_LOG2,index=num/n,rem=num%n;
    const auto *a=PQ16_ROOT_COARSE[index>>8],*b=PQ16_ROOT_FINE[index&255];
    long double c=a[0]*b[0]-a[1]*b[1],s=a[0]*b[1]+a[1]*b[0];
    if(rem){
        const long double d=0xc.90fdaa22168c235p-1L*static_cast<long double>(rem)/(n*static_cast<long double>(1u<<PQ16_ROOT_GRID_LOG2));
        const long double dc=1-d*d/2,ds=d-d*d*d/6,x=c;c=x*dc-s*ds;s=s*dc+x*ds;
    }
    if(reflect){const auto x=c;c=s;s=x;}
    switch(quadrant){case 0:return {double(c),double(s)};case 1:return {double(-s),double(c)};
        case 2:return {double(-c),double(-s)};default:return {double(s),double(-c)};}
}
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
template<unsigned Lg>using Stage=std::array<double,(1u<<Lg)/(Lg>=compact_min_log2?2:1)>;
constexpr bool has_stage(unsigned lg,bool compact,bool radix8=false){
    return lg>=6&&(lg<=max_log2||(!radix8&&lg==pq_max_log2))&&compact==(lg>=compact_min_log2);
}
template<unsigned Lg,bool Radix8>constexpr auto make_stage(){
    constexpr unsigned n=1u<<Lg;constexpr bool compact=Lg>=compact_min_log2;Stage<Lg> out{};
    for(unsigned group=0;group<n/64;++group){
        const unsigned j=group*(Radix8?8:16);
        for(unsigned block=0;block<(compact?2:4);++block)for(unsigned lane=0;lane<8;++lane){
            const unsigned scale=Radix8?(block<2?1:1u<<(block-1)):(block&1?2:1);
            const unsigned k=Radix8?(j+lane)*scale+(block==1?n/8:0):(j+lane+(block>=2?8:0))*scale;
            const unsigned compact_k=Radix8?(j+lane)*(block?2:1):j+lane+8*block;
            const Root w=root(compact?compact_k:k,Lg);const size_t at=(compact?32:64)*group+16*block+lane;
            out[at]=w.re;out[at+8]=w.im;
        }
    }
    return out;
}
// Instantiate data once in root_bank.cpp. Including an ISA kernel does not
// repeat a megabyte of constexpr evaluation in every translation unit.
template<unsigned Lg,bool Radix8>alignas(128) extern const Stage<Lg> stage;
#define SBN3_ROOT_STAGE_EXTERN(LG) \
    extern template const Stage<LG> stage<LG,false>; \
    extern template const Stage<LG> stage<LG,true>;
SBN3_ROOT_STAGE_EXTERN(6) SBN3_ROOT_STAGE_EXTERN(7) SBN3_ROOT_STAGE_EXTERN(8)
SBN3_ROOT_STAGE_EXTERN(9) SBN3_ROOT_STAGE_EXTERN(10) SBN3_ROOT_STAGE_EXTERN(11)
SBN3_ROOT_STAGE_EXTERN(12) SBN3_ROOT_STAGE_EXTERN(13) SBN3_ROOT_STAGE_EXTERN(14)
SBN3_ROOT_STAGE_EXTERN(15) SBN3_ROOT_STAGE_EXTERN(16) SBN3_ROOT_STAGE_EXTERN(17)
#undef SBN3_ROOT_STAGE_EXTERN
extern template const Stage<18> stage<18,false>;
template<bool Radix8>inline const double *twiddle(unsigned lg,bool compact){
    if(!has_stage(lg,compact,Radix8))return nullptr;
    switch(lg){
    case 6:return stage<6,Radix8>.data();case 7:return stage<7,Radix8>.data();
    case 8:return stage<8,Radix8>.data();case 9:return stage<9,Radix8>.data();
    case 10:return stage<10,Radix8>.data();case 11:return stage<11,Radix8>.data();
    case 12:return stage<12,Radix8>.data();case 13:return stage<13,Radix8>.data();
    case 14:return stage<14,Radix8>.data();case 15:return stage<15,Radix8>.data();
    case 16:return stage<16,Radix8>.data();case 17:return stage<17,Radix8>.data();
    case 18:if constexpr(!Radix8)return stage<18,false>.data();else return nullptr;
    default:return nullptr;
    }
}
constexpr auto make_pq(){
    std::array<double,pq_branch/2> out{};
    for(unsigned g=0;g<pq_branch/64;++g)for(unsigned k=0;k<16;++k){
        const auto w=root(reverse(g*16+k,pq_max_log2-2),pq_max_log2);
        out[32*g+k]=w.re;out[32*g+16+k]=w.im;
    }
    return out;
}
alignas(128) extern const std::array<double,pq_branch/2> pq;
// Small right-angle transforms need an odd-radix twist in addition to the
// common power-of-two roots. Publish exactly the W1 execution-shape band.
template<unsigned M,unsigned Lg>constexpr auto make_rac(){
    constexpr unsigned n=1u<<Lg,N=M*n;std::array<double,2*N> out{};
    for(unsigned b=0;b<M;++b)for(unsigned j=0;j<n;++j){
        const auto w=rational_root(uint64_t(4*b+(M%4)*M)*j,4*uint64_t(N));
        const size_t at=2*size_t(b)*n+2*(j&~7u)+(j&7u);out[at]=w.re;out[at+8]=w.im;
    }
    return out;
}
template<unsigned M,unsigned Lg>alignas(128) extern const std::array<double,2*M*(1u<<Lg)> rac;
#define SBN3_RAC_EXTERN(M,L) extern template const std::array<double,2*M*(1u<<L)> rac<M,L>;
#define SBN3_RAC_EXTERN_BAND(M) SBN3_RAC_EXTERN(M,6) SBN3_RAC_EXTERN(M,7) SBN3_RAC_EXTERN(M,8) SBN3_RAC_EXTERN(M,9) SBN3_RAC_EXTERN(M,10) SBN3_RAC_EXTERN(M,11)
SBN3_RAC_EXTERN_BAND(3) SBN3_RAC_EXTERN(3,12)
SBN3_RAC_EXTERN_BAND(5) SBN3_RAC_EXTERN(5,12)
SBN3_RAC_EXTERN_BAND(7)
#undef SBN3_RAC_EXTERN_BAND
#undef SBN3_RAC_EXTERN
template<unsigned M>inline const double *rac_twiddle(unsigned n){
    switch(n){case 64:return rac<M,6>.data();case 128:return rac<M,7>.data();
    case 256:return rac<M,8>.data();case 512:return rac<M,9>.data();
    case 1024:return rac<M,10>.data();case 2048:return rac<M,11>.data();
    case 4096:if constexpr(M!=7)return rac<M,12>.data();else return nullptr;
    default:return nullptr;}
}
inline const double *rac_twiddle(unsigned M,unsigned n){
    switch(M){case 3:return rac_twiddle<3>(n);case 5:return rac_twiddle<5>(n);case 7:return rac_twiddle<7>(n);default:return nullptr;}
}
// The same CT cross-roots serve every supported digit width. Publish the
// cache-sized band once; larger transforms use the ordinary arena builder.
template<unsigned M,unsigned Lg>constexpr auto make_ct(){
    constexpr unsigned n=1u<<Lg,N=M*n;std::array<double,2*(M-1)*n> out{};
    for(unsigned b=1;b<M;++b)for(unsigned j=0;j<n;++j){
        const auto w=rational_root(uint64_t(b)*j,N);
        const size_t at=2*size_t(b-1)*n+2*(j&~7u)+(j&7u);out[at]=w.re;out[at+8]=w.im;
    }
    return out;
}
template<unsigned M,unsigned Lg>alignas(128) extern const std::array<double,2*(M-1)*(1u<<Lg)> ct;
#define SBN3_CT_EXTERN(M,L) extern template const std::array<double,2*(M-1)*(1u<<L)> ct<M,L>;
#define SBN3_CT_EXTERN_BAND(M) SBN3_CT_EXTERN(M,7) SBN3_CT_EXTERN(M,8) SBN3_CT_EXTERN(M,9) SBN3_CT_EXTERN(M,10) SBN3_CT_EXTERN(M,11) SBN3_CT_EXTERN(M,12)
SBN3_CT_EXTERN_BAND(3) SBN3_CT_EXTERN_BAND(5) SBN3_CT_EXTERN_BAND(7)
#undef SBN3_CT_EXTERN_BAND
#undef SBN3_CT_EXTERN
template<unsigned M>inline const double *ct_twiddle(unsigned n){
    switch(n){case 128:return ct<M,7>.data();case 256:return ct<M,8>.data();
    case 512:return ct<M,9>.data();case 1024:return ct<M,10>.data();
    case 2048:return ct<M,11>.data();case 4096:return ct<M,12>.data();default:return nullptr;}
}
inline const double *ct_twiddle(unsigned M,unsigned n){
    switch(M){case 3:return ct_twiddle<3>(n);case 5:return ct_twiddle<5>(n);case 7:return ct_twiddle<7>(n);default:return nullptr;}
}
// The factored CT representation needs only these short rotations beyond
// the 4096-branch coarse prefix. Publishing all of them through max_log2
// costs 13.5 KiB, rather than multiple MiB of full cross-root planes.
inline constexpr unsigned ct_coarse_branch=4096;
template<unsigned M,unsigned Lg>constexpr auto make_ct_fine(){
    static_assert(Lg>12&&Lg<=max_log2);
    constexpr unsigned n=1u<<Lg,factor=n/ct_coarse_branch,stride=factor<8?8:factor;
    std::array<double,2*(M-1)*stride> out{};
    for(unsigned b=1;b<M;++b)for(unsigned j=0;j<stride;++j){
        const auto w=rational_root(uint64_t(b)*(j%factor),uint64_t(M)*n);
        const size_t at=2*size_t(b-1)*stride+2*(j&~7u)+(j&7u);out[at]=w.re;out[at+8]=w.im;}
    return out;
}
template<unsigned M,unsigned Lg>alignas(128) extern const decltype(make_ct_fine<M,Lg>()) ct_fine;
#define SBN3_CT_FINE_EXTERN(M,L) extern template const decltype(make_ct_fine<M,L>()) ct_fine<M,L>;
#define SBN3_CT_FINE_BAND(M) SBN3_CT_FINE_EXTERN(M,13) SBN3_CT_FINE_EXTERN(M,14) SBN3_CT_FINE_EXTERN(M,15) SBN3_CT_FINE_EXTERN(M,16) SBN3_CT_FINE_EXTERN(M,17)
SBN3_CT_FINE_BAND(3) SBN3_CT_FINE_BAND(5) SBN3_CT_FINE_BAND(7)
#undef SBN3_CT_FINE_BAND
#undef SBN3_CT_FINE_EXTERN
template<unsigned M>inline const double *ct_fine_twiddle(unsigned n){
    switch(n){case 8192:return ct_fine<M,13>.data();case 16384:return ct_fine<M,14>.data();
    case 32768:return ct_fine<M,15>.data();case 65536:return ct_fine<M,16>.data();
    case 131072:return ct_fine<M,17>.data();default:return nullptr;}
}
inline const double *ct_fine_twiddle(unsigned M,unsigned n){
    switch(M){case 3:return ct_fine_twiddle<3>(n);case 5:return ct_fine_twiddle<5>(n);case 7:return ct_fine_twiddle<7>(n);default:return nullptr;}
}
inline constexpr size_t ct_fine_bytes=16*(2+4+6)*(8+8+8+16+32);
// Two stage kinds at 64..131072 plus the prefix-stable PQ table.
inline constexpr size_t rac_bytes=16*((3+5)*(8192-64)+7*(4096-64));
inline constexpr size_t ct_bytes=16*(2+4+6)*(8192-128);
inline constexpr size_t stage_words=((1u<<compact_min_log2)-64)+((2*branch)-(1u<<compact_min_log2))/2;
inline constexpr size_t bytes=2*stage_words*sizeof(double)+sizeof(Stage<pq_max_log2>)+sizeof(pq)+rac_bytes+ct_bytes+ct_fine_bytes;
}
