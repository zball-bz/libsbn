#include "algorithms/local_inverse.hpp"
#include "algorithms/inverse_seed.hpp"
#include "algorithms/newton_contract.hpp"
#include "sbn3/divrem.h"
#include "backend/u52/kernels.hpp"
#include "runtime/scratch.hpp"
#include "value/limbs.hpp"
#include <algorithm>
#include <string.h>
namespace sbn::v3 {
namespace {
size_t words(size_t n) noexcept { return (8*n+63)&~size_t(63); }
size_t approximate_bytes(size_t n) noexcept {
    if(n<=15)return 0;
    const size_t m=newton_contract::next_precision(n);
    const size_t multiply=std::max(u52::scratch_bytes(n,m+1),
                                   u52::scratch_bytes(m+1,newton_contract::residual_words(m,n)));
    return 64+words(m+1)+std::max(approximate_bytes(m),words(n+m+1)+words(n+4)+multiply+64);
}
// Same recurrence/guard limbs as inverse_rung, using full local products.
// High cancellation is exact; the omitted low residual limbs only affect
// the final reciprocal below its two retained guard limbs.
void approximate(uint64_t *v,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    if(n<=15){inverse_seed(v,d,n);return;}
    FrameMark mark(scratch);
    const size_t m=newton_contract::next_precision(n),length=n+m+1;
    auto *u=scratch.alloc<uint64_t>(m+1);
    approximate(u,d+n-m,m,scratch);
    auto *residual=scratch.alloc<uint64_t>(length),*correction=scratch.alloc<uint64_t>(n+4);
    const size_t shift=newton_contract::residual_shift(m),rn=newton_contract::residual_words(m,n);
    auto work=scratch.subframe(std::max(u52::scratch_bytes(n,m+1),u52::scratch_bytes(m+1,rn)),64);
    u52::multiply(residual,d,n,u,m+1,work);
    const bool negative=residual[n+m]==0;
    --residual[n+m]; // subtract B^(n+m), wrapping iff the residual is negative
    if(negative){
        for(size_t j=0;j<length;++j)residual[j]=~residual[j];
        const uint64_t one=1;
        require(!limbs::add_to(residual,length,&one,1),SBN3_FATAL_MATH,"local inverse absolute residual");
    }
    require(residual[n]<newton_contract::inverse_residual_limit &&
            limbs::zero(residual+n+1,length-n-1),SBN3_FATAL_MATH,"local inverse residual bound");
    u52::multiply(correction,u,m+1,residual+shift,rn,work);
    require(correction[n+2]<newton_contract::inverse_correction_limit && !correction[n+3],
            SBN3_FATAL_MATH,"local inverse correction bound");
    memset(v,0,(n-m)*8);
    memcpy(v+n-m,u,(m+1)*8);
    const size_t offset=newton_contract::correction_offset(m),count=n-m+1;
    const auto spill=negative?limbs::add_to(v,n+1,correction+offset,count)
                             :limbs::sub_from(v,n+1,correction+offset,count);
    require(!spill,SBN3_FATAL_MATH,"local inverse correction overflow");
    if(!v[n])memset(v,0,n*8);
    else if(v[n]>1)memset(v,0xff,n*8);
    v[n]=1;
}
bool at_least_divisor(const uint64_t *r,const uint64_t *d,size_t n) noexcept {
    if(r[n])return true;
    for(size_t j=n;j-->0;)if(r[j]!=d[j])return r[j]>d[j];
    return true;
}
}
size_t local_inverse_bytes(size_t n) noexcept {
    require(n && n<=8192,SBN3_FATAL_SIZE,"local inverse size");
    if(n<=32)return words(2*n)+words(n)+words(3*n+1)+64;
    return std::max(approximate_bytes(n),words(2*n+1)+u52::scratch_bytes(n+1,n)+128);
}
void local_inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    require(n && n<=8192 && (d[n-1]>>63),SBN3_FATAL_ARGUMENT,"local inverse normalized input");
    if(n<=32){
        FrameMark mark(scratch);
        auto *numerator=scratch.alloc<uint64_t>(2*n),*remainder=scratch.alloc<uint64_t>(n),
             *work=scratch.alloc<uint64_t>(3*n+1);
        memset(numerator,0xff,2*n*8);
        require(sbn3_divrem_basecase(out,remainder,numerator,2*n,d,n,work)==n+1 && out[n]==1,
                SBN3_FATAL_MATH,"local inverse small quotient");
        return;
    }
    approximate(out,d,n,scratch);
    FrameMark mark(scratch);
    auto *product=scratch.alloc<uint64_t>(2*n+1);
    auto work=scratch.subframe(u52::scratch_bytes(n+1,n),64);
    u52::multiply(product,out,n+1,d,n,work);
    const uint64_t one=1;
    unsigned corrections=0;
    // floor((B^(2n)-1)/D): first remove any overshoot, then form the
    // all-ones numerator's exact remainder by complementing the product.
    while(product[2*n]){
        require(++corrections<=3,SBN3_FATAL_MATH,"local inverse downward correction");
        require(!limbs::sub_from(out,n+1,&one,1) && !limbs::sub_from(product,2*n+1,d,n),
                SBN3_FATAL_MATH,"local inverse downward borrow");
    }
    for(size_t j=0;j<2*n;++j)product[j]=~product[j];
    require(product[n]<4 && limbs::zero(product+n+1,n-1),SBN3_FATAL_MATH,"local inverse exact residual");
    while(at_least_divisor(product,d,n)){
        require(++corrections<=3,SBN3_FATAL_MATH,"local inverse upward correction");
        require(!limbs::add_to(out,n+1,&one,1) && !limbs::sub_from(product,2*n,d,n),
                SBN3_FATAL_MATH,"local inverse upward carry");
    }
    require(out[n]==1,SBN3_FATAL_MATH,"local inverse framing");
}
}
