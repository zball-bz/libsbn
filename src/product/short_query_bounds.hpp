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
inline double fft_score(size_t an,size_t bn,unsigned workers) noexcept {
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
} // namespace sbn::v3::query_bounds
