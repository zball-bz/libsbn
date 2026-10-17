#pragma once
#include "sbn3/product.h"
#include "product/short_tuning.hpp"
#include <initializer_list>
namespace sbn::v3::query_bounds {
// Lower bound of the existing FFT ranking score, not a new timing model.
// A legal plan has radix in {1,3,5,7}, a power-of-two branch, and enough
// complex slots for both encoded operands. Wide digits stop at 20 bits.
// Relaxing the branch/radix and dropping nonnegative score terms lets the
// caller avoid an FFT search that cannot beat an already valid candidate.
inline double relaxed_score(size_t an,size_t bn,unsigned workers) noexcept {
    if(std::max(an,bn)>(size_t(1)<<20))return INFINITY;
    const size_t sum=an+bn;
    const bool wide=workers==1 && sum>=256 && sum<=16384;
    const unsigned bits=wide?20:16,minimum_branch=wide?64:128;
    const size_t need=((64*an+bits-1)/bits+(64*bn+bits-1)/bits)/2;
    size_t points=SIZE_MAX;
    for(unsigned radix:{1u,3u,5u,7u}){
        size_t branch=minimum_branch;
        while(radix*branch<need)branch*=2;
        points=std::min(points,radix*branch);
    }
    size_t branch=minimum_branch;
    while(2*branch<=points/7)branch*=2;
    double lower;
    if(workers==1 && sum>=256 && sum<=262144){
        // In the native model every radix/width term, including each
        // negative adjustment paired with its required positive term,
        // is nonnegative. Right-angle's .955 factor is the smallest.
        pq16::Shape relaxed{uint32_t(points),uint32_t(branch),1,false,pq16::Recipe::RightAngle,16,false};
        lower=pq16::native_cost(relaxed,points,0);
    }else{
        // Relax effective parallelism to `workers`, turn off radix/centered
        // and cache penalties, and remove the constant launch term.
        sbn3_mul_info relaxed{};relaxed.workers=workers;relaxed.M2=branch;relaxed.C=2*workers;
        relaxed.transform_trunks=2*points;
        lower=pq16_product_cost(relaxed,1,1);
        relaxed.transform_trunks=0;
        lower-=pq16_product_cost(relaxed,1,1);
    }
    return std::max(0.,lower)*(1.-1e-9); // protect the comparison from score rounding
}
// W1 with an+bn in [256, 16384], where the score is pq16::native_cost of the plan's shape: a
// geometry-consistent capacity relaxation. Per radix M and width class (wide codecs at most 20
// bits, otherwise 16) it keeps N = M*branch with the smallest power-of-two branch, counted
// from 1, whose complex points hold both operands' digits. That candidate need not be
// executable (no recipe's real minimum branch or support rule is enforced); it only sits
// below every legal shape of its class: for fixed radix, width and recipe the cost grows with
// the branch (the branch term outweighs any indicator switching on at a doubling;
// product_query_bound_test checks every legal shape of the domain), centering and balanced
// digits only add (factors >= 1, terms >= 0), the nonnegative fill term is off, and
// right-angle carries the smallest recipe factor of the CT form, so the cheaper of
// right-angle and PFA bounds the class. Values come from native_cost itself: a constant
// table evaluated at compile time for shapes up to 8192 points, larger ones at run time.
namespace floor_detail {
inline constexpr unsigned logs=14;                       // branch 2^0..2^13
inline constexpr uint32_t radices[4]={1,3,5,7};
constexpr double value(unsigned bits,uint32_t radix,uint32_t branch) noexcept {
    double m=INFINITY;                                    // an+bn = N turns the fill term off
    for(auto recipe:{pq16::Recipe::RightAngle,pq16::Recipe::PfaPQ}){
        const double v=pq16::native_cost({radix*branch,branch,radix,false,recipe,bits,false},radix*branch,0);
        m=v<m?v:m;
    }
    return m;
}
struct Table {double v[2][4][logs];};                     // width class (16, 20 bits) x radix x log2(branch)
constexpr Table make() noexcept {
    Table t{};
    for(unsigned c=0;c<2;++c)for(unsigned r=0;r<4;++r)for(unsigned k=0;k<logs;++k)
        t.v[c][r][k]=radices[r]<<k<=8192?value(c?20:16,radices[r],uint32_t(1)<<k):-1.;
    return t;
}
inline constexpr Table table=make();
} // namespace floor_detail
inline bool small_domain(size_t an,size_t bn,unsigned workers) noexcept {
    return workers==1 && an+bn>=256 && an+bn<=16384;
}
// the relaxed floor of width class c (0: 16 bits, 1: 20 bits) and radix index r
inline double class_floor(size_t an,size_t bn,unsigned c,unsigned r) noexcept {
    const unsigned bits=c?20:16;
    const size_t need=((64*an+bits-1)/bits+(64*bn+bits-1)/bits)/2;
    const uint32_t radix=floor_detail::radices[r];
    unsigned k=0;
    while((size_t(radix)<<k)<need)++k;
    if(k<floor_detail::logs && floor_detail::table.v[c][r][k]>=0)return floor_detail::table.v[c][r][k];
    return floor_detail::value(bits,radix,uint32_t(1)<<k);
}
inline double fft_score(size_t an,size_t bn,unsigned workers) noexcept {
    const double relaxed=relaxed_score(an,bn,workers);
    if(!small_domain(an,bn,workers))return relaxed;
    double lower=INFINITY;
    for(unsigned c=0;c<2;++c)for(unsigned r=0;r<4;++r)lower=std::min(lower,class_floor(an,bn,c,r));
    return std::max(relaxed,lower*(1.-1e-9));
}
// False when no legal FFT plan can score below `best`, so the caller may skip the FFT search.
// Staged: the relaxed bound first, then the eight class floors, stopping at the first that
// could beat `best`. Equivalent to best > fft_score(an,bn,workers).
inline bool fft_may_beat(size_t an,size_t bn,unsigned workers,double best) noexcept {
    if(best<=relaxed_score(an,bn,workers))return false;
    if(!small_domain(an,bn,workers))return true;
    for(unsigned c=0;c<2;++c)for(unsigned r=0;r<4;++r)
        if(!(best<=class_floor(an,bn,c,r)*(1.-1e-9)))return true;
    return false;
}
} // namespace sbn::v3::query_bounds
