#pragma once
#include "sbn3/product.h"
namespace sbn::v3 {
struct ProductStage;
// Small positive a (1..2^64-1). The seed writes n+1 limbs, 1<=n<=3:
// 0<=R<B^n, 0<=B^n/sqrt(a)-R<=1; the high output limb is zero.
void rsqrt_seed(uint64_t *R, uint64_t a, size_t n) noexcept;
// 2<=m<n<=2m-1; |R-B^m/sqrt(a)|<=4 and R has m limbs.
// Output has n+1 limbs, high limb zero, absolute error <3 ulps.
// Both products use ring>=2m+2. square is cached SQR(A=m), multiply is
// cached MUL(A=m,B=m+1), sharing one reserved R spectrum. All tables and
// buffers exist before execution; residual has m+2 limbs with low_square,
// otherwise ring limbs; correction has ring limbs. A stage controller may
// rebind the already resident workspace between products.
struct RsqrtRung {
    size_t m, n, ring;
    sbn3_mul_binding *square, *multiply;
    sbn3_spectrum *r;
    uint64_t *residual, *correction;
    bool low_square = false; // square plan emits exactly m+2 low words
    sbn3_team *team = nullptr;
    const ProductStage *stage = nullptr;
};
void rsqrt_rung(const RsqrtRung &, uint64_t a, const uint64_t *R, uint64_t *out) noexcept;
struct RsqrtProgram {
    size_t target, seed_limbs, rung_count;
    const RsqrtRung *rungs;
    uint64_t *values[2];
};
void rsqrt_program(const RsqrtProgram &, uint64_t a, uint64_t *R) noexcept;
} // namespace sbn::v3
