#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
// r approximates B^precision/sqrt(2), precision>=n+2. Writes n+1 limbs
// of floor(2*r/B^(precision-n)); the certified rsqrt bound puts it within
// one of floor(B^n*sqrt(2)). r has precision+1 limbs, its high limb zero.
void sqrt2_from_rsqrt(uint64_t *z, size_t n, const uint64_t *r, size_t precision) noexcept;
// z is within two integers of floor(B^n*sqrt(2)), n>=1. A prebound linear
// SQR of n+1 limbs computes one complete integer square in work[0..2n+2).
// At most two unit corrections establish z²<=2B^(2n)<(z+1)². The work
// buffer contains the final z²; no further multiply/query/allocation occurs.
unsigned sqrt2_certify(sbn3_mul_binding *square, uint64_t *z, size_t n, uint64_t *work) noexcept;
} // namespace sbn::v3
