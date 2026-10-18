// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/integer.hpp"
#include "external/row_cache.hpp"
#include "external/resident_product.hpp"
#include "planning/product.hpp"
#include "sbn3/series.h"
#include <array>
namespace sbn::v3::external {
struct IntegerShape {size_t limbs=0;int64_t exponent2=0;SourceWindow source{};};
using StoredTriple=std::array<StoredInteger,3>;
enum class StoredRounding { finite, tail };
struct StoredMergeStage {
    enum class Kind { product, sum, zero } kind=Kind::zero;
    unsigned a=0,b=0,out=0;
    SpilledProductPlan product{};
    ResidentProductPlan resident{};
    bool use_resident=false,borrow_result=false;
    uint64_t resident_write_bytes=0;
    StoredSumPlan sum{};
};
struct StoredMergePlan {
    sbn3_series_recipe recipe=SBN3_SERIES_HYPERDESCENT;
    unsigned need=0,stage_count=0,summary_quotes=0,basis_queries=0,retained_inputs=0;
    unsigned uses[11]{};
    unsigned resident_inputs=0;
    size_t input_offsets[6]{},input_bytes[6]{},stage_work_limit[5]{};
    IntegerShape shape[11]{};
    StoredMergeStage stage[5]{};
    uint64_t peak_backing_bytes=0,normal_write_bytes=0;
    size_t storage_bytes=0,total_resident_bytes=0,shared_bytes=0,alignment=4096;
    size_t precision=0;
    RowCachePlan cache{};
    unsigned cache_first=0,cache_last=0,cache_common=4;
    bool cached_pair=false;
    ResidentPairPlan native_pair{};
    int pair_first=-1; // product/sum/product episode sharing the original B
    double estimated_ns=0;
    StoredRounding rounding=StoredRounding::tail;
    bool bounded_inputs=false;
};
// Exact finite merge; inputs are (T,D,U)_left then (T,D,U)_right. Supports
// numerator+denominator, and optional CommonP2B3 U. Formula cuts/precision are
// owned by the higher recipe. This adapter does not introduce a new split rule.
// retained_inputs names inputs with external aliases/read/checkpoint pins;
// their backing remains charged after this recipe's last local read.
// Nonzero precision selects finite-node rounding (each product/sum keeps P)
// or PSR tail rounding (need==3, numerator products P+1, final N/D P).
// Bounded input bindings accept shorter normalized views and instantiate only
// scalar exponents/lengths; every product replays the preselected capacity.
planning::ExternalPlanStatus query_stored_merge(sbn3_series_recipe,unsigned need,
    const std::array<IntegerShape,6> &,unsigned workers,
    const planning::ExternalProductBudget &,const planning::ExternalProductProfile &,
    StoredMergePlan &,unsigned retained_inputs=0,size_t precision=0,
    StoredRounding=StoredRounding::tail,bool bounded_inputs=false) noexcept;
// Replay an all-native merge with selected inputs loaned from its arena tail.
// Pack inputs in last-use order; every stage gets the dead prefix preceding
// its still-live inputs. Same arithmetic plans, no backend/geometry queries.
// Rejection is atomic. Caller must account for the producer's omitted writes.
bool lend_stored_merge_inputs(StoredMergePlan &,unsigned mask,const std::array<size_t,6>&bytes,
    size_t storage_limit,const planning::ExternalProductBudget &,const planning::ExternalProductProfile &) noexcept;
// Apply after native-child loans are fixed: sharing may not evict a child or
// grow the already granted workspace, stored writes, or peak backing.
void share_stored_merge_transform(StoredMergePlan &,size_t storage_limit,
    const planning::ExternalProductBudget &,const planning::ExternalProductProfile &) noexcept;
// Explicit take: after successful binding checks, children become disposable.
// Their last use releases backing. Caller-held checkpoint/read pins remain
// immutable. On error no output is published; all taken private state retires.
// An input_loans array has six slots at the queried arena-tail offsets. Those
// bytes become reusable WORKSPACE after their last reader joins; callers must
// not retain a preserved alias. Each live loan and its extent describe the same
// immutable words until then. Final results still own ordinary Store values.
IoError run_stored_merge(const StoredMergePlan &,Transport &,ValueStore &,
    StoredTriple &left,StoredTriple &right,StoredTriple &out,
    std::span<uint8_t>,sbn3_team &,std::span<const ResidentInteger> input_loans={}) noexcept;
} // namespace sbn::v3::external
