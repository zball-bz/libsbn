#include "algorithms/local_inverse.hpp"
#include "algorithms/inverse_seed.hpp"
#include "algorithms/newton_contract.hpp"
#include "sbn3/divrem.h"
#include "backend/u52/kernels.hpp"
#include "backend/pq16/kernels.hpp"
#include "product/short_tuning.hpp"
#include "runtime/scratch.hpp"
#include "value/limbs.hpp"
#include <algorithm>
#include <string.h>
namespace sbn::v3 {
namespace {
// The local recurrence needs exact integer products, not an operation-level
// plan for each rung. Small products retain u52; larger ones use one classic
// FFT geometry, with its complete temporary storage included in the query.
pq16::Shape local_product_shape(size_t a,size_t b) noexcept {
    if(std::min(a,b)<512)return {};
    auto s=pq16::query(a,b);s.recipe=pq16::Recipe::PfaPQ;
    return s.nfull && pq16::supported(s,a,b,1)?s:pq16::Shape{};
}
size_t local_product_bytes(size_t a,size_t b) noexcept {
    const auto s=local_product_shape(a,b);
    return s.nfull?pq16::table_bytes(s)+pq16::scratch_bytes(s,a,b)+512:u52::scratch_bytes(a,b);
}
double local_product_cost(size_t a,size_t b) noexcept {
    const auto s=local_product_shape(a,b);
    return s.nfull?.27*double(pq16::table_bytes(s))+pq16::native_cost(s,a,b):u52_product_cost(a,b,1);
}
void local_product(uint64_t *out,const uint64_t *a,size_t an,const uint64_t *b,size_t bn,Frame &space) noexcept {
    const auto s=local_product_shape(an,bn);
    if(!s.nfull){u52::multiply(out,a,an,b,bn,space);return;}
    FrameMark mark(space);
    auto table=space.subframe(pq16::table_bytes(s)+128,128);
    auto *prepared=pq16::prepare(table,s);
    auto work=space.subframe(pq16::scratch_bytes(s,an,bn)+128,128);
    pq16::multiply(out,a,an,b,bn,*prepared,work,nullptr);
}
pq16::Shape residual_shape(size_t n,size_t m) noexcept {
    const auto full=local_product_shape(n,m+1);
    if(!full.nfull)return {};
    const auto ring=pq16::cyclic_shape(n+2,16);
    const size_t r=pq16::cyclic_period(ring);
    return ring.nfull && r>=n+2 && r<n+m && ring.nfull<full.nfull &&
           pq16::cyclic_supported(ring,n,m+1)?ring:pq16::Shape{};
}
size_t residual_bytes(size_t n,size_t m) noexcept {
    const auto s=residual_shape(n,m);
    return s.nfull?pq16::table_bytes(s)+32*size_t(s.nfull)+1536:local_product_bytes(n,m+1);
}
double residual_cost(size_t n,size_t m) noexcept {
    const auto s=residual_shape(n,m);
    return s.nfull?.27*double(pq16::table_bytes(s))+pq16::native_cost(s,n,m+1):local_product_cost(n,m+1);
}
bool local_residual(uint64_t *out,const uint64_t *d,size_t n,const uint64_t *u,size_t m,Frame &space) noexcept {
    const auto s=residual_shape(n,m);
    const size_t length=n+m+1;
    if(s.nfull){
        FrameMark mark(space);
        auto table=space.subframe(pq16::table_bytes(s)+128,128);
        auto *prepared=pq16::prepare(table,s);
        auto work=space.subframe(32*size_t(s.nfull)+1152,128);
        pq16::cyclic_multiply(out,d,n,u,m+1,false,nullptr,*prepared,work,nullptr);
        const size_t r=pq16::cyclic_period(s);
        // |D*U-B^(n+m)|<16*B^n and r>=n+2 give a unique signed lift.
        limbs::cyclic_sub_power(out,r,n+m-r);
        const bool negative=limbs::cyclic_absolute(out,r);
        memset(out+r,0,(length-r)*sizeof(uint64_t));
        return negative;
    }
    local_product(out,d,n,u,m+1,space);
    const bool negative=out[n+m]==0;
    --out[n+m];
    if(negative){
        for(size_t j=0;j<length;++j)out[j]=~out[j];
        const uint64_t one=1;
        require(!limbs::add_to(out,length,&one,1),SBN3_FATAL_MATH,"local inverse absolute residual");
    }
    return negative;
}
size_t words(size_t n) noexcept { return (8*n+63)&~size_t(63); }
size_t approximate_bytes(size_t n) noexcept {
    if(n<=15)return 0;
    const size_t m=newton_contract::next_precision(n);
    const size_t multiply=std::max(residual_bytes(n,m),
                                   local_product_bytes(m+1,newton_contract::residual_words(m,n)));
    return 64+words(m+1)+std::max(approximate_bytes(m),words(n+m+1)+words(n+4)+multiply+64);
}
// Same recurrence/guard limbs as inverse_rung, using exact local products.
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
    auto work=scratch.subframe(std::max(residual_bytes(n,m),local_product_bytes(m+1,rn)),64);
    const bool negative=local_residual(residual,d,n,u,m,work);
    require(residual[n]<newton_contract::inverse_residual_limit &&
            limbs::zero(residual+n+1,length-n-1),SBN3_FATAL_MATH,"local inverse residual bound");
    local_product(correction,u,m+1,residual+shift,rn,work);
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
double local_inverse_approximate_cost(size_t n) noexcept {
    if(n<=32)return 100.+.33*double(n)*double(n);
    const size_t m=newton_contract::next_precision(n),rn=newton_contract::residual_words(m,n);
    // Exact selected product recipes, plus the linear residual/correction
    // scans. Do not keep the old n^1.5 envelope after entering the FFT band.
    return local_inverse_approximate_cost(m)+residual_cost(n,m)+
           local_product_cost(m+1,rn)+2.*double(n);
}
size_t local_inverse_bytes(size_t n) noexcept {
    require(n && n<=8192,SBN3_FATAL_SIZE,"local inverse size");
    if(n<=32)return words(2*n)+words(n)+words(3*n+1)+64;
    return std::max(approximate_bytes(n),words(2*n+1)+local_product_bytes(n+1,n)+128);
}
size_t local_inverse_approximate_bytes(size_t n) noexcept {
    require(n && n<=8192,SBN3_FATAL_SIZE,"local approximate inverse size");
    return n<=32?local_inverse_bytes(n):approximate_bytes(n);
}
void local_inverse_approximate(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    require(n && n<=8192 && (d[n-1]>>63),SBN3_FATAL_ARGUMENT,"local approximate inverse normalized input");
    if(n<=32){local_inverse(out,d,n,scratch);return;}
    // Prefix truncation takes the predecessor's <3-ulp error to <8 ulps.
    // The exact Newton update has error <64/B, since n<=2m-1. Dropping
    // residual words and rounding the correction add <1+2/B^2. Saturating
    // to [B^n,2B^n-1] leaves error <3, including D=B^n/2.
    approximate(out,d,n,scratch);
    require(out[n]==1,SBN3_FATAL_MATH,"local approximate inverse framing");
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
    auto work=scratch.subframe(local_product_bytes(n+1,n),64);
    local_product(product,out,n+1,d,n,work);
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
