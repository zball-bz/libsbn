// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/window.hpp"
#include "external/row_cache.hpp"
#include "planning/product.hpp"
#include "sbn3/newton.h"
namespace sbn::v3::external {
struct InverseRungPlan {
    size_t m=0,n=0,storage_bytes=0,total_resident_bytes=0;
    SourceWindow denominator{};
    RowCachePlan cache{};
    SpilledProductPlan residual_product{},correction_product{};
    SignedLiftPlan lift{};
    StoredSumPlan finish{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
    double estimated_ns=0;
    unsigned product_quotes=0,basis_queries=0;
};
// Same normalized reciprocal contract as algorithms/inverse_rung.hpp.
// Whole LIN and certified cyclic residual recipes are compared before execute.
// A cyclic output supplies a complemented view to the shared-U correction;
// LIN retains the signed-lift fallback. No runtime plan search.
planning::ExternalPlanStatus query_inverse_rung(size_t m,size_t n,SourceWindow denominator,unsigned workers,
    const planning::ExternalProductBudget &,const planning::ExternalProductProfile &,InverseRungPlan &) noexcept;
IoError run_inverse_rung(const InverseRungPlan &,Transport &,ValueStore &,const StoredInteger &denominator,
    const StoredInteger &approximation,StoredInteger &out,std::span<uint8_t>,sbn3_team &,bool *clamped=nullptr) noexcept;
struct InverseOptions {
    unsigned workers=1;
    size_t maximum_native_seed=SIZE_MAX; // optional policy/diagnostic cap
};
struct InverseBudget {
    size_t resident_bytes=0,other_live_bytes=0; // excludes named engine banks and this plan
    uint64_t owned_disk_limit=UINT64_MAX,other_owned_disk_bytes=0,write_limit=UINT64_MAX;
};
struct InversePlan {
    size_t target=0,seed_limbs=0,rung_count=0,storage_bytes=0,alignment=4096,total_resident_bytes=0,shared_bytes=0;
    size_t source_capacity_words=0;
    SourceWindow denominator{},seed_source{};
    sbn3_newton_plan seed{};
    sbn3_newton_info seed_info{};
    size_t seed_io=0,seed_index=0,seed_input=0,seed_output=0,seed_binding=0;
    InverseRungPlan rungs[32]{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
};
// Largest complete native seed on the mathematical halving ladder, followed
// by precompiled stored rungs. Query reads metadata only, never operand data.
planning::ExternalPlanStatus query_stored_inverse(const StoredInteger &,size_t precision,const InverseOptions &,
    const InverseBudget &,const planning::ExternalProductProfile &,InversePlan &) noexcept;
// Shape-only compilation; normalized leading-bit truth is an execution
// precondition. Rebinding changes only bounded source views/read accounting,
// never product geometry, retained bytes, native seed or storage/write caps.
planning::ExternalPlanStatus query_inverse_source(SourceWindow,size_t precision,const InverseOptions &,
    const InverseBudget &,const planning::ExternalProductProfile &,InversePlan &) noexcept;
bool rebind_stored_inverse(InversePlan &,const StoredInteger &) noexcept;
// The caller prepares an aligned, unleased arena interval and assigns it
// exclusively through return, including transiently unleased native gaps.
IoError run_stored_inverse(const InversePlan &,Transport &,ValueStore &,const StoredInteger &,
    StoredInteger &,sbn3_arena &,size_t offset,sbn3_team &) noexcept;

struct QuotientRungPlan {
    size_t m=0,n=0,storage_bytes=0,total_resident_bytes=0;
    SourceWindow numerator{},denominator{};
    RowCachePlan cache{};
    SpilledProductPlan coarse_product{},residual_product{},correction_product{};
    SignedLiftPlan lift{};
    StoredSumPlan finish{};
    uint64_t write_bytes=0,peak_backing_bytes=0;
    unsigned product_quotes=0,basis_queries=0;
    double estimated_ns=0;
};
// A and D are normalized n-word magnitude prefixes. The existing quotient
// refinement treats A as n+1 words with a virtual zero high word. Result Q
// obeys |Q-B^n*A/D|<3. The supplied half reciprocal satisfies the native bound.
planning::ExternalPlanStatus query_quotient_rung(size_t m,size_t n,SourceWindow numerator,SourceWindow denominator,
    unsigned workers,const planning::ExternalProductBudget &,const planning::ExternalProductProfile &,QuotientRungPlan &) noexcept;
IoError run_quotient_rung(const QuotientRungPlan &,Transport &,ValueStore &,const StoredInteger &numerator,
    const StoredInteger &denominator,const StoredInteger &inverse,StoredInteger &,std::span<uint8_t>,sbn3_team &) noexcept;
struct QuotientPlan {
    size_t target=0,storage_bytes=0,alignment=4096,total_resident_bytes=0;
    size_t source_capacity_words[2]{};
    SourceWindow numerator{},denominator{};
    int64_t result_exponent2=0;
    InversePlan inverse{};
    QuotientRungPlan refinement{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
};
// Positive stored N/D, precision >=4 and both magnitude prefixes available.
// The result's binary exponent restores the original value scales. Error
// relative to the original ratio is strictly below 7 output ulps (3 from the
// refinement and <4 from normalized-prefix truncation). No exact-divrem claim.
planning::ExternalPlanStatus query_stored_quotient(const StoredInteger &,const StoredInteger &,size_t precision,
    const InverseOptions &,const InverseBudget &,const planning::ExternalProductProfile &,QuotientPlan &) noexcept;
planning::ExternalPlanStatus query_quotient_sources(SourceWindow,SourceWindow,size_t precision,
    const InverseOptions &,const InverseBudget &,const planning::ExternalProductProfile &,QuotientPlan &) noexcept;
bool rebind_stored_quotient(QuotientPlan &,const StoredInteger &,const StoredInteger &) noexcept;
// Same exclusive initially-unleased interval contract as stored reciprocal.
IoError run_stored_quotient(const QuotientPlan &,Transport &,ValueStore &,const StoredInteger &,const StoredInteger &,
    StoredInteger &,sbn3_arena &,size_t offset,sbn3_team &) noexcept;

struct RsqrtRungPlan {
    size_t m=0,n=0,storage_bytes=0,total_resident_bytes=0;
    uint64_t radicand=0;
    RowCachePlan cache{};
    SpilledProductPlan square{},correction{};
    SignedLiftPlan lift{};
    StoredSumPlan finish{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
    unsigned product_quotes=0,basis_queries=0;
};
// Existing rsqrt recurrence, |R-B^m/sqrt(a)|<=4, 2<=m<n<=2m-1.
// Cached A rows serve SQR and MUL. The radicand is folded into integer CRT
// emission, not the modular coefficient domain. Output has n+1 words/top zero.
planning::ExternalPlanStatus query_rsqrt_rung(uint64_t radicand,size_t m,size_t n,unsigned workers,
    const planning::ExternalProductBudget &,const planning::ExternalProductProfile &,RsqrtRungPlan &) noexcept;
IoError run_rsqrt_rung(const RsqrtRungPlan &,Transport &,ValueStore &,const StoredInteger &,StoredInteger &,
    std::span<uint8_t>,sbn3_team &,bool *clamped=nullptr) noexcept;
struct RsqrtPlan {
    size_t target=0,seed_limbs=0,rung_count=0,storage_bytes=0,alignment=4096,total_resident_bytes=0,shared_bytes=0;
    uint64_t radicand=0;
    sbn3_newton_plan seed{};
    sbn3_newton_info seed_info{};
    size_t seed_io=0,seed_index=0,seed_output=0,seed_binding=0;
    RsqrtRungPlan rungs[32]{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
};
planning::ExternalPlanStatus query_stored_rsqrt(uint64_t radicand,size_t precision,const InverseOptions &,
    const InverseBudget &,const planning::ExternalProductProfile &,RsqrtPlan &) noexcept;
// Integer R, exponent zero, |R-B^n/sqrt(a)|<3; same exclusive arena contract.
IoError run_stored_rsqrt(const RsqrtPlan &,Transport &,ValueStore &,StoredInteger &,
    sbn3_arena &,size_t offset,sbn3_team &) noexcept;

struct MulRsqrtPlan {
    size_t n=0,m=0,storage_bytes=0,alignment=4096,total_resident_bytes=0;
    uint64_t scalar=1,radicand=0;
    SourceWindow input{};
    RsqrtPlan half{};
    SpilledProductPlan square{},multiply{},correction{};
    SignedLiftPlan lift{};
    StoredSumPlan finish{};
    uint64_t write_bytes=0,worst_write_bytes=0,peak_backing_bytes=0;
};
// Fused final rsqrt refinement of Q=scalar*X. Positive X has n+1 physical
// words, scalar*X < B^(n+1), n>=4. Result has X's binary exponent and raw
// integer error <2^38 relative to scalar*X/sqrt(a), as in algorithms/mul_rsqrt.
// Only the half-precision rsqrt is materialized. Each of the final three
// products has its own bounded geometry selection; no enlarged common basis.
planning::ExternalPlanStatus query_stored_mul_rsqrt(const StoredInteger &,size_t n,uint64_t radicand,uint64_t scalar,
    const InverseOptions &,const InverseBudget &,const planning::ExternalProductProfile &,MulRsqrtPlan &) noexcept;
// high_exclusive bounds X[n]; zero denotes B. No operand reads or handles.
planning::ExternalPlanStatus query_mul_rsqrt_source(SourceWindow,size_t n,uint64_t high_exclusive,uint64_t radicand,uint64_t scalar,
    const InverseOptions &,const InverseBudget &,const planning::ExternalProductProfile &,MulRsqrtPlan &) noexcept;
IoError run_stored_mul_rsqrt(const MulRsqrtPlan &,Transport &,ValueStore &,const StoredInteger &,StoredInteger &,
    sbn3_arena &,size_t offset,sbn3_team &) noexcept;
} // namespace sbn::v3::external
