// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace sbn::v3::m64 {
// Residue modulo 2^64-1. Unlike m61, this modulus divides every limb-aligned
// minus ring. All-ones is the redundant zero; published checks are canonical.
inline uint64_t reduce(__uint128_t x) noexcept {
    const uint64_t lo=uint64_t(x),hi=uint64_t(x>>64),sum=lo+hi;
    const uint64_t r=sum+(sum<lo);return r==UINT64_MAX?0:r;
}
inline uint64_t add(uint64_t a,uint64_t b) noexcept {return reduce(__uint128_t(a)+b);}
inline uint64_t multiply(uint64_t a,uint64_t b) noexcept {return reduce(__uint128_t(a)*b);}
inline uint64_t words_scalar(const void*data,size_t n) noexcept {
    const auto*p=static_cast<const uint8_t*>(data);__uint128_t sum=0;
    for(size_t j=0;j<n;++j){uint64_t x;std::memcpy(&x,p+8*j,8);sum+=x;}
    return reduce(sum);
}
uint64_t words_bulk(const void*,size_t) noexcept;
inline uint64_t words(const void*p,size_t n) noexcept {return n>=64?words_bulk(p,n):words_scalar(p,n);}
inline void add_at(uint64_t&sum,const void*p,size_t n,uint64_t) noexcept {sum=add(sum,words(p,n));}
}
