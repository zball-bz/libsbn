#pragma once
#include "backend/u52/kernels.hpp"
#include "product/short_tuning.hpp"
namespace sbn::v3 {
// The arithmetic kernel converts a long rectangular input one bounded strip
// at a time. Charge those actual multiplications, rather than applying the
// full-buffer cache-pressure term to a buffer the kernel never creates.
inline double u52_execution_cost(size_t an,size_t bn,unsigned workers=1) noexcept {
    if(!u52::streams(an,bn))return u52_product_cost(an,bn,workers);
    const size_t n=std::max(an,bn),small=std::min(an,bn),chunk=u52::strip_limbs;
    return double(n/chunk)*u52_product_cost(chunk,small,workers)+
           (n%chunk?u52_product_cost(n%chunk,small,workers):0.);
}
}
