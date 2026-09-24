#pragma once
#include <stddef.h>
#include <stdint.h>
#include <algorithm>
#include "algorithms/local_divrem.hpp"
#include "backend/u52/division_geometry.hpp"
namespace sbn::v3 {
inline bool fused_division_shape(size_t nn,size_t dn,unsigned reuse_hint=0) noexcept {
    // Include ceil(3*dn/2) for odd divisors: integer rounding must not
    // alternate between fused and blocked recipes along a 1.5-ratio sweep.
    return dn>=8192&&nn>dn&&nn-dn>=512&&nn-dn<=dn/2+dn%2&&reuse_hint<=1;
}
inline LocalDivisionPlan local_division_auto_plan(size_t nn,size_t dn,size_t block,unsigned reuse_hint=0) noexcept {
    return fused_division_shape(nn,dn,reuse_hint)?local_fused_division_plan(nn,dn,reuse_hint):
        local_division_plan_for_block(nn,dn,block,reuse_hint);
}
// Native one-shot crossover. Two-vector quotient updates amortize the
// reciprocal and packing once the quotient is long enough. No plan search.
inline bool small_division_u52(size_t nn,size_t dn) noexcept {
    if(nn<dn || dn<3)return false;
    const size_t qn=nn-dn+1;
    // A quotient increment fitting one paired leaf (832 bits = 13 limbs)
    // needs no further quotient update. Its optimized prefix amortizes the
    // reciprocal earlier; retain the old crossover for longer quotients.
    // Zen5 evidence: scalar-threshold320, all 180 affected paired-leaf shapes.
    constexpr size_t paired_limbs=2*u52::division_block_bits/64;
    const size_t crossover=qn<=paired_limbs+1?320:384;
    return qn>5 && dn*(qn-5)>=crossover;
}
// For q/n=rho and t equal quotient blocks, the smooth transform model is
// (2/3)rho + 2rho/t + t/3 + 1/6 multiplications. One vs two blocks crosses
// at rho=1/3. At rho=1, two and three tie. The cyclic kernel also requires
// a block no larger than half the divisor, except a short one-block quotient.
inline size_t division_block_limbs(size_t q,size_t dn,size_t maximum=0) noexcept {
    if(q<=dn/3)return q;
    const size_t half=maximum?maximum:(dn+1)/2;
    const size_t blocks=std::max<size_t>(2,(q+half-1)/half);
    return (q+blocks-1)/blocks;
}
// Native W1 crossover: scalar work thresholds, followed by one block-size
// calculation. No cost tables, logarithms, candidate plans or grid search.
inline size_t local_division_block(size_t nn,size_t dn,unsigned reuse_hint=0) noexcept {
    if(nn<=dn||dn<64||dn>local_division_max_limbs)return 0;
    const size_t q=nn-dn;
    const uint64_t work=uint64_t(dn)*q;
    const uint64_t threshold=q<=dn/3?400000:4*q<=3*dn?350000:240000;
    if(work<threshold)return 0;
    size_t capacity=(dn+1)/2;
    if(dn>=256&&(q>=4*dn||reuse_hint>=2)){
        // Rounding the residual ring already determines its legal block
        // capacity. Fill that transform instead of adding quotient passes
        // while leaving the same FFT half empty. This is one geometry
        // calculation, with no alternative inverse/product plans.
        if(const size_t room=product::local_cyclic_operand_capacity(dn))capacity=room;
    }
    // Two or more applications amortize a half-divisor-sized reciprocal;
    // retain one quotient block when it still fits the short cyclic ring.
    if(reuse_hint>=2&&q<=capacity)return q;
    // Long numerators stream full-capacity blocks plus one tail. A larger
    // numerator changes only the pass count, not retained state or scratch.
    if(q>dn)return capacity;
    return division_block_limbs(q,dn,capacity);
}
}
