#include "algorithms/local_inverse.hpp"
#include "algorithms/inverse_seed.hpp"
#include "algorithms/reciprocal.hpp"
#include "algorithms/newton_contract.hpp"
#include "sbn3/divrem.h"
#include "product/local_windows.hpp"
#include "product/compact_windows.hpp"
#include "product/local_program.hpp"
#include "runtime/scratch.hpp"
#include "value/limbs.hpp"
#include "value/divrem_words.hpp"
#include "algorithms/small_division.hpp"
#include "product/native_capabilities.hpp"
#include "backend/u52/kernels.hpp"
#include <algorithm>
#include <string.h>
namespace sbn::v3 {
bool local_dyadic_divide_supported(size_t n) noexcept {
    return n&&n<=local_division_max_limbs&&!local_division_block(2*n,n);
}
bool local_dyadic_divide_native(size_t n) noexcept {
    return small_division_u52(2*n,n)&&native_available();
}
size_t local_dyadic_divide_bytes(size_t n,bool native) noexcept {
    return native?u52::divide_scratch_bytes(2*n+1,n)+64:n==1?0:8*(2*n+1)+64;
}
void local_dyadic_divide(uint64_t *q,const uint64_t *a,const uint64_t *d,size_t n,
                         bool native,void *scratch,size_t bytes) noexcept {
    if(native){auto space=Frame::external(scratch,bytes);u52::divide_dyadic_quotient(q,a,d,n,space);return;}
    if(n==1){
        const auto inverse=divrem_words::invert_limb(d[0]);uint64_t rem=a[1];
        udiv_qrnnd_preinv(q[1],rem,rem,a[0],d[0],inverse);
        // B+inverse=floor((B^2-1)/D). This estimate is below R*B/D
        // by less than two integer units; no last remainder is needed.
        q[0]=rem+uint64_t((__uint128_t(rem)*inverse)>>64);
        return;
    }
    auto *numerator=static_cast<uint64_t *>(scratch);
    // Compute the high quotient exactly, then use the initial 3/2 estimate
    // for the last word. With V=B+inverse and leading pairs r,d, it is
    // floor((r_hi*V+r_lo)/B). Normalization bounds its error against the
    // full R*B/D by strictly less than 3 (docs/division.md). Saturate when
    // equal leading pairs would make the truncated quotient reach B.
    memset(numerator,0,8*(n-1));memcpy(numerator+n-1,a,8*(n+1));
    const auto inverse=divrem_words::invert_pi1(d[n-1],d[n-2]);
    const auto high=n==2?divrem_words::divrem_2_prepared(q+1,numerator,2*n,d,inverse):
                         divrem_words::sbpi1_div_qr(q+1,numerator,2*n,d,n,inverse);
    require(!high,SBN3_FATAL_MATH,"dyadic quotient capacity");
    if(numerator[n-1]==d[n-1]&&numerator[n-2]==d[n-2])q[0]=UINT64_MAX;
    else q[0]=numerator[n-1]+uint64_t((__uint128_t(numerator[n-1])*inverse+numerator[n-2])>>64);
}
namespace {
size_t words(size_t n) noexcept { return (8*n+63)&~size_t(63); }
size_t approximate_bytes(size_t n) noexcept {
    if(n>local_inverse_max_limbs)return reciprocal_bytes(n,product::CompactWindowFactory{});
    return reciprocal_bytes(n,product::LocalWindowFactory{});
}
void approximate(uint64_t *v,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    if(n>local_inverse_max_limbs){product::CompactWindowFactory factory;reciprocal(v,d,n,scratch,factory);return;}
    product::LocalWindowFactory factory;reciprocal(v,d,n,scratch,factory);
}

bool at_least_divisor(const uint64_t *r,const uint64_t *d,size_t n) noexcept {
    if(r[n])return true;
    for(size_t j=n;j-->0;)if(r[j]!=d[j])return r[j]>d[j];
    return true;
}
}
bool local_refinement_supported(size_t n,bool quotient) noexcept {
    if(n<=local_inverse_max_limbs)return false;
    const size_t m=newton_contract::next_precision(n);
    return product::compact_window_supported(quotient?refinement_products<RefinementKind::Quotient>(m,n):
                                                     refinement_products<RefinementKind::Inverse>(m,n));
}
size_t local_refinement_compile(size_t n,bool division,product::LocalWindowProgram &program) noexcept {
    require(n>15&&n<=local_inverse_max_limbs,SBN3_FATAL_ARGUMENT,"local program precision");
    program.count=0;product::LocalWindowCompiler compiler{program};
    return division?quotient_bytes(n,compiler):reciprocal_bytes(n,compiler);
}
void local_inverse_replay(uint64_t *out,const uint64_t *d,size_t n,Frame &space,
                          product::LocalWindowReplay &program) noexcept {
    reciprocal(out,d,n,space,program);
    require(out[n]==1,SBN3_FATAL_MATH,"local inverse replay framing");
}
void local_divide_replay(uint64_t *out,const uint64_t *a,const uint64_t *d,size_t n,Frame &space,
                         product::LocalWindowReplay &program) noexcept {
    quotient(out,a,d,n,space,program);
}
size_t local_inverse_bytes(size_t n) noexcept {
    require(n && n<=local_inverse_max_limbs,SBN3_FATAL_SIZE,"local inverse size");
    if(n<=32)return words(2*n)+words(n)+words(3*n+1)+64;
    return std::max(approximate_bytes(n),words(2*n+1)+product::local_window_detail::full_bytes(n+1,n)+128);
}
size_t local_inverse_approximate_bytes(size_t n) noexcept {
    require(n&&(n<=local_inverse_max_limbs||local_refinement_supported(n,false)),SBN3_FATAL_SIZE,"local approximate inverse size");
    return n<=4?local_inverse_bytes(n):approximate_bytes(n);
}
void local_inverse_approximate(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    require(n&&(n<=local_inverse_max_limbs||local_refinement_supported(n,false))&&(d[n-1]>>63),SBN3_FATAL_ARGUMENT,"local approximate inverse normalized input");
    if(n<=4){local_inverse(out,d,n,scratch);return;}
    // Prefix truncation takes the predecessor's <3-ulp error to <8 ulps.
    // The exact Newton update has error <64/B, since n<=2m-1. Dropping
    // residual words and rounding the correction add <1+2/B^2. Saturating
    // to [B^n,2B^n-1] leaves error <3, including D=B^n/2.
    approximate(out,d,n,scratch);
    require(out[n]==1,SBN3_FATAL_MATH,"local approximate inverse framing");
}
void local_inverse(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch) noexcept {
    require(n && n<=local_inverse_max_limbs && (d[n-1]>>63),SBN3_FATAL_ARGUMENT,"local inverse normalized input");
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
    auto work=scratch.subframe(product::local_window_detail::full_bytes(n+1,n),64);
    product::local_window_detail::full(product,{out,n+1},{d,n},work);
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
size_t local_divide_terminal_bytes(size_t n,LocalTerminal recipe) noexcept {
    if(recipe!=LocalTerminal::Ordinary){
        require(local_refinement_supported(n,true),SBN3_FATAL_ARGUMENT,"compact quotient support");
        return quotient_terminal_bytes(n,product::CompactWindowFactory{{},recipe==LocalTerminal::CompactRetained});
    }
    require(n>=4&&n<=local_divide_max_limbs,SBN3_FATAL_SIZE,"local quotient size");
    return quotient_terminal_bytes(n,product::LocalWindowFactory{});
}
void local_divide_terminal(uint64_t*q,const uint64_t*a,const uint64_t*d,const uint64_t*u,
                           size_t n,Frame&space,LocalTerminal recipe) noexcept {
    if(recipe!=LocalTerminal::Ordinary){
        require(local_refinement_supported(n,true),SBN3_FATAL_ARGUMENT,"compact quotient support");
        product::CompactWindowFactory factory{{},recipe==LocalTerminal::CompactRetained};quotient_terminal(q,a,d,u,n,space,factory);return;
    }
    require(n>=4&&n<=local_divide_max_limbs,SBN3_FATAL_ARGUMENT,"local quotient input");
    product::LocalWindowFactory factory;quotient_terminal(q,a,d,u,n,space,factory);
}

size_t local_divide_bytes(size_t n) noexcept {
    if(local_refinement_supported(n,true))return quotient_bytes(n,product::CompactWindowFactory{});
    require(n>=4&&n<=local_divide_max_limbs,SBN3_FATAL_SIZE,"local quotient size");
    const size_t m=newton_contract::next_precision(n);
    return words(m+1)+std::max(local_inverse_approximate_bytes(m),local_divide_terminal_bytes(n))+128;
}
void local_divide(uint64_t*q,const uint64_t*a,const uint64_t*d,size_t n,Frame&space) noexcept {
    if(local_refinement_supported(n,true)){product::CompactWindowFactory factory;quotient(q,a,d,n,space,factory);return;}
    require(n>=4&&n<=local_divide_max_limbs,SBN3_FATAL_SIZE,"local quotient size");
    FrameMark mark(space);const size_t m=newton_contract::next_precision(n);auto *u=space.alloc<uint64_t>(m+1);
    local_inverse_approximate(u,d+n-m,m,space);
    local_divide_terminal(q,a,d,u,n,space);
}
}
