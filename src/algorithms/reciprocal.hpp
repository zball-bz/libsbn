#pragma once
#include "algorithms/refinement.hpp"
#include "algorithms/inverse_seed.hpp"
#include <algorithm>

namespace sbn::v3 {
inline constexpr size_t refinement_words_bytes(size_t words) noexcept {return (8*words+63)&~size_t(63);}

// The complete precision ladder is parameterized by a product provider, not
// by codec/transform choices. A provider quotes its own transient storage and
// reports whether a residual window needs caller storage across its next use.
template<class Factory> size_t reciprocal_bytes(size_t n,const Factory &factory) noexcept {
    if(n<=15)return 0;
    const size_t m=newton_contract::next_precision(n);
    const auto shape=refinement_products<RefinementKind::Inverse>(m,n);
    const size_t work=factory.bytes(shape)+refinement_words_bytes(factory.retained_words(shape));
    return std::max(reciprocal_bytes(m,factory),work)+256;
}
template<class Factory> void reciprocal(uint64_t *out,const uint64_t *d,size_t n,Frame &scratch,Factory &factory) noexcept {
    if(n<=15){inverse_seed(out,d,n);return;}
    FrameMark mark(scratch);const size_t m=newton_contract::next_precision(n);
    reciprocal(out,d+n-m,m,scratch,factory);
    const auto shape=refinement_products<RefinementKind::Inverse>(m,n);
    const size_t retained=factory.retained_words(shape);
    auto *rho=retained?scratch.alloc<uint64_t>(retained):nullptr;
    factory.with_group(shape,scratch,nullptr,[&](auto &products) __attribute__((always_inline)) {
        bounded_refinement<RefinementKind::Inverse>({m,n,nullptr,rho},products,nullptr,d,out,out);
    });
}
template<class Factory> size_t quotient_terminal_bytes(size_t n,const Factory &factory) noexcept {
    const size_t m=newton_contract::next_precision(n);
    return refinement_words_bytes(m+1)+factory.bytes(refinement_products<RefinementKind::Quotient>(m,n))+256;
}
template<class Factory> void quotient_terminal(uint64_t *out,const uint64_t *a,const uint64_t *d,const uint64_t *u,
                                               size_t n,Frame &scratch,Factory &factory) noexcept {
    const size_t m=newton_contract::next_precision(n);FrameMark mark(scratch);
    auto *coarse=scratch.alloc<uint64_t>(m+1);
    const auto shape=refinement_products<RefinementKind::Quotient>(m,n);
    // The cancellation consumes all of A before returning rho. Caller output
    // can then retain rho, including the supported out==A case.
    factory.with_group(shape,scratch,out,[&](auto &products) __attribute__((always_inline)) {
        bounded_refinement<RefinementKind::Quotient>({m,n,coarse,out},products,a,d,u,out);
    });
}
template<class Factory> size_t quotient_bytes(size_t n,const Factory &factory) noexcept {
    const size_t m=newton_contract::next_precision(n);
    return refinement_words_bytes(m+1)+std::max(reciprocal_bytes(m,factory),quotient_terminal_bytes(n,factory))+128;
}
template<class Factory> void quotient(uint64_t *out,const uint64_t *a,const uint64_t *d,size_t n,
                                      Frame &scratch,Factory &factory) noexcept {
    FrameMark mark(scratch);const size_t m=newton_contract::next_precision(n);
    auto *u=scratch.alloc<uint64_t>(m+1);reciprocal(u,d+n-m,m,scratch,factory);
    quotient_terminal(out,a,d,u,n,scratch,factory);
}
} // namespace sbn::v3
