// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/bitwise.h"
#include <algorithm>
namespace sbn::v3::bits {
using U=uint64_t;
size_t normalized_size(const U*,size_t) noexcept;
size_t first_nonzero(const U*,size_t) noexcept;
size_t first_not_ones(const U*,size_t) noexcept;
size_t population(const U*,size_t) noexcept;
size_t population_xor(const U*,size_t,size_t,const U*,size_t,size_t,size_t,size_t) noexcept;
struct Shape {size_t words;bool negative;};
inline Shape shape(sbn3_bit_operation op,size_t an,bool a,size_t bn,bool b) noexcept {
    if(op==SBN3_BIT_XOR)return {std::max(an,bn),a!=b};
    if(op==SBN3_BIT_AND)return {a&&b?std::max(an,bn):a?bn:b?an:std::min(an,bn),a&&b};
    return {!a&&!b?std::max(an,bn):!a?bn:!b?an:std::min(an,bn),a||b};
}
size_t binary(U*,const U*,size_t,bool,const U*,size_t,bool,sbn3_bit_operation) noexcept;
size_t adjust_copy(U*,const U*,size_t,bool increment) noexcept;
}
