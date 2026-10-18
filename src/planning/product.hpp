// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/spilled_product.hpp"
namespace sbn::v3::external {struct RowCachePlan;}
namespace sbn::v3::planning {
struct ExternalProductRequest {
    size_t an=0,bn=0;
    unsigned workers=1,primes=0,trunk_bits=0;
    bool square=false;
    external::SourceWindow a_source{},b_source{};
    size_t stored_result_limbs=SIZE_MAX; // output placement envelope; default full
    external::CyclicResidualSpec cyclic{};
    external::MiddleWindowSpec middle{};
};
struct ExternalProductBudget {
    size_t resident_bytes=0,other_live_bytes=0;
    // other_live excludes the p48 shared bank, which is added exactly once;
    // it includes Team/stacks, arena/control, caller-held values and other banks.
    uint64_t owned_disk_limit=UINT64_MAX,other_owned_disk_bytes=0,write_limit=UINT64_MAX;
};
struct ExternalProductProfile {
    size_t leaf_cells=4096,target_transfer_bytes=1u<<20;
    // Analytic scenario defaults, not device measurements. Safety uses exact
    // quotes only. Estimates remain conservative service sums; streaming
    // placement uses explicit cache and phase-lifetime bounds.
    double io_bytes_per_ns=4.0,request_ns=15000.0,memory_bytes_per_ns=64.0;
    double vector_level_ns=1.0,funnel_cell_ns=3.0;
    double write_saving_slowdown=1.05;
    // Active expanded operand rows, distinct from whole-operation RAM. Native
    // profile reserves half of this machine's aggregate L3 for arithmetic.
    size_t active_row_bytes=32u<<20;
    unsigned io_workers=4; // independent prime files; execution caps this to the provider's concurrency
};
enum class ExternalPlanStatus { ready,invalid,unsupported,no_admitted_plan };
struct ExternalProductSelection {
    external::SpilledProductPlan plan{};
    size_t total_resident_bytes=0;
    uint64_t owned_disk_bytes=0;
    double estimated_ns=0;
    unsigned basis_queries=0,summary_quotes=0;
    size_t smallest_quoted_resident=SIZE_MAX;
    bool admitted=false;
};
// For an operation already lowered to external execution. Small/resident entry
// points retain their own existing selection. LIN uses one ordinary basis
// query; cyclic residuals use one certified period per available prime basis.
// At most two cache-derived shapes and one serial lifetime replay; constrained requests may fall back
// to four earlier shapes. Each has direct ordinary/raw-source lifetime replays.
// No T/layout grid or retention-subset search.
ExternalPlanStatus choose_external_product(const ExternalProductRequest &,const ExternalProductBudget &,
                                           const ExternalProductProfile &,ExternalProductSelection &) noexcept;
// Pure analytic estimates, shared by bounded episode comparisons. They do not
// authorize storage or measure performance.
double estimate_external_product(const external::SpilledProductPlan &,const ExternalProductProfile &) noexcept;
double estimate_forward_rows(const external::ForwardPlanesPlan &,const ExternalProductProfile &) noexcept;
double estimate_row_cache(const external::RowCachePlan &,const ExternalProductProfile &) noexcept;
// Promote one existing reusable cache and all of its consumers together, only
// within the already admitted build/apply workspace. No geometry search.
bool complete_cache_within(external::RowCachePlan &,std::span<external::SpilledProductPlan *>,size_t workspace_bytes,
                           const ExternalProductProfile &) noexcept;
// Two uses do not automatically justify a stored spectrum. Compare the whole
// cached pair (build included) with raw-source prime streaming, replaying only
// the two fixed arithmetic shapes. Atomic on rejection; no basis search.
// storage_limit includes the cache's former control/build/apply grant.
bool prefer_raw_pair(external::RowCachePlan &,external::SpilledProductPlan &,external::SpilledProductPlan &,
                     size_t storage_limit,const ExternalProductProfile &) noexcept;
external::IoError run_selected_product(const ExternalProductSelection &,external::Transport &,
    const external::Region &,const external::Region &,external::Region &,std::span<external::Region>,
    std::span<uint8_t>,sbn3_team &,external::CyclicResidualResult * = nullptr,const external::Region *subtract=nullptr) noexcept;
} // namespace sbn::v3::planning
