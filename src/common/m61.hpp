// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
namespace sbn::v3::m61 {
inline constexpr uint64_t modulus=(uint64_t(1)<<61)-1;
inline uint64_t reduce(__uint128_t x) noexcept {
    const uint64_t r=uint64_t(x&modulus)+uint64_t(x>>61);
    return r>=modulus?r-modulus:r;
}
inline uint64_t multiply(uint64_t a,uint64_t b) noexcept {return reduce(__uint128_t(a)*b);}
// Eight independent chains avoid a word-by-word dependency. Unaligned loads
// use memcpy; all callers supply little-endian complete limbs.
inline uint64_t words_scalar(const void *data,size_t n) noexcept {
    const auto *p=static_cast<const uint8_t *>(data);uint64_t r[8]{};
    const size_t groups=n/8,tail=n%8;
    for(size_t i=0;i<tail;++i){uint64_t x;std::memcpy(&x,p+8*(groups*8+i),8);r[i]=reduce(x);}
    for(size_t g=groups;g-- >0;)for(unsigned i=0;i<8;++i){uint64_t x;std::memcpy(&x,p+8*(g*8+i),8);r[i]=reduce((__uint128_t(r[i])<<24)+x);}
    __uint128_t sum=0;for(unsigned i=0;i<8;++i)sum+=__uint128_t(r[i])<<(3*i);
    return reduce(sum);
}
// Native bundle bulk kernel; other tiers can bind the same contract to the
// scalar reference. No vector type crosses the private kernel boundary.
uint64_t words_bulk(const void *data,size_t n) noexcept;
inline uint64_t words(const void *data,size_t n) noexcept {
    return n>=64?words_bulk(data,n):words_scalar(data,n);
}
inline void add_at(uint64_t &sum,const void *data,size_t limbs,uint64_t word_offset) noexcept {
    const uint64_t term=multiply(words(data,limbs),uint64_t(1)<<((3*(word_offset%61))%61));
    sum+=term;if(sum>=modulus)sum-=modulus;
}
} // namespace sbn::v3::m61
