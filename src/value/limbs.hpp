#pragma once
#include "core/x86_64/word.hpp"
#include <string.h>
namespace sbn::v3::limbs {
// Internal fixed-span helpers: n>=sn, spans/alignment are proved by the
// caller's value plan. The add/sub kernels are the existing v2 asm ports.
inline uint64_t add_to(uint64_t *r,size_t n,const uint64_t *s,size_t sn) noexcept {
    uint64_t carry=sn?sbn3i_add_n(r,r,s,long(sn)):0;
    for(size_t k=sn;carry&&k<n;++k)carry=++r[k]==0;
    return carry;
}
inline uint64_t sub_from(uint64_t *r,size_t n,const uint64_t *s,size_t sn) noexcept {
    uint64_t borrow=sn?sbn3i_sub_n(r,r,s,long(sn)):0;
    for(size_t k=sn;borrow&&k<n;++k){borrow=r[k]==0;--r[k];}
    return borrow;
}
inline bool zero(const uint64_t *a,size_t n) noexcept {
    uint64_t any=0;for(size_t k=0;k<n;++k)any|=a[k];return any==0;
}
// Canonical R -> R-B^exponent mod (B^n-1), B=2^64, exponent<n.
inline void cyclic_sub_power(uint64_t *r,size_t n,size_t exponent) noexcept {
    uint64_t borrow=1;
    for(size_t k=exponent;borrow&&k<n;++k){borrow=r[k]==0;--r[k];}
    // Machine underflow added B^n; the modulus contributes B^n-1 instead.
    if(borrow)for(size_t k=0;k<n;++k)if(r[k]--)break;
}
// P is odd and floor(P/2)=2^(64n-1)-1. P-R is one's complement.
inline bool cyclic_absolute(uint64_t *r,size_t n) noexcept {
    const bool negative=r[n-1]>>63;
    if(negative)for(size_t k=0;k<n;++k)r[k]=~r[k];
    return negative;
}
inline void cyclic_sub_part(uint64_t *r,size_t n,size_t offset,const uint64_t *a,size_t an) noexcept {
    uint64_t borrow=an?sbn3i_sub_n(r+offset,r+offset,a,long(an)):0;
    for(size_t j=offset+an;borrow&&j<n;++j){borrow=r[j]==0;--r[j];}
    if(borrow)for(size_t j=0;j<n;++j)if(r[j]--)break;
}
// an<n, shift<n; the shifted value crosses the ring seam at most once.
inline void cyclic_sub_shifted(uint64_t *r,size_t n,const uint64_t *a,size_t an,size_t shift) noexcept {
    const size_t first=an<n-shift?an:n-shift;
    cyclic_sub_part(r,n,shift,a,first);
    if(an>first)cyclic_sub_part(r,n,0,a+first,an-first);
}
}
