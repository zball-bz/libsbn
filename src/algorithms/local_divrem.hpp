#pragma once
#include "product/local_product.hpp"
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3 {
class Frame;
inline constexpr size_t local_division_max_limbs=32768;
using LocalDivisionProduct=product::LocalProductPlan;
struct LocalDivisionPlan {
    size_t numerator_limbs=0,denominator_limbs=0,block_limbs=0,storage_bytes=0;
    size_t persistent_bytes=0,work_bytes=0;
    size_t fused_precision=0; // nonzero: one fused short-quotient estimate
    LocalDivisionProduct quotient{},residual{};
};
using LocalDivisionPreparedProduct=product::PreparedLocalProduct;
struct LocalDivisionDivisor {
    const uint64_t *divisor=nullptr;
    unsigned shift=0;
    uint64_t head_inverse=0;
    LocalDivisionPreparedProduct quotient{},residual{};
};
inline bool local_division_shares_tables(const LocalDivisionPlan&p) noexcept {
    return product::local_product_same_tables(p.quotient,p.residual);
}
struct LocalDivisionMetrics {uint64_t head_limbs=0,head_corrections=0;unsigned products=0;};
inline size_t local_division_head(size_t quotient_limbs,size_t block_limbs) noexcept {
    const size_t rem=quotient_limbs%block_limbs;
    return rem==1 || (block_limbs>=128&&rem<=4)?rem:0;
}
// Fixed single-worker recipe, no competing plans or retained state.
// 3<=dn<=local_division_max_limbs, dn<=nn<=2^40, pieces in 1..3.
// The reciprocal is capped at local_inverse_max_limbs; longer quotients
// use additional blocks.
LocalDivisionPlan local_division_plan(size_t nn,size_t dn,unsigned pieces) noexcept;
// Production entry: the caller has already selected one block length.
// The pieces wrapper above is retained for fixed-recipe experiments/tests.
LocalDivisionPlan local_division_plan_for_block(size_t nn,size_t dn,size_t block_limbs,unsigned reuse_hint=0) noexcept;
LocalDivisionPlan local_fused_division_plan(size_t nn,size_t dn,unsigned reuse_hint=0) noexcept;
// prepare may use the full storage bound transiently. Its retained D/U,
// tables and spectra fit persistent_bytes; apply uses only work_bytes.
LocalDivisionDivisor local_division_prepare(const LocalDivisionPlan &,const uint64_t *d,Frame &) noexcept;
uint64_t local_division_apply(const LocalDivisionPlan &,const LocalDivisionDivisor &,
                             uint64_t *q,uint64_t *r,const uint64_t *n,size_t nn,Frame &,
                             LocalDivisionMetrics * =nullptr) noexcept;
// The plan also covers shorter numerators: the same inverse precision and
// product geometries are retained. All spans are disjoint and caller-owned.
// Writes max(nn-dn+1,0) quotient and dn remainder limbs; returns correction count.
uint64_t local_division_execute(const LocalDivisionPlan &,uint64_t *q,uint64_t *r,
                               const uint64_t *n,size_t nn,const uint64_t *d,Frame &,
                               LocalDivisionMetrics * =nullptr) noexcept;
}
