// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/series_merge.hpp"
#include "planning/precision.hpp"
namespace sbn::v3::external {
struct SeriesBudget {
    size_t resident_bytes=0,other_live_bytes=0;
    // Other live excludes supplied plan metadata, IO and controller allowance,
    // and the published p48/pq16 banks, which this compiler adds exactly once.
    uint64_t owned_disk_limit=UINT64_MAX,other_owned_disk_bytes=0,write_limit=UINT64_MAX;
};
struct SeriesOptions {
    bool store_result=false;
    // Diagnostic placement override for small integration gates: force the
    // first block out of the resident suffix and this many finite prefix levels.
    // Zero is the production policy; it does not change mathematical cuts.
    unsigned force_stored_depth=0;
};
struct SeriesNode {
    sbn3_series_range range{};
    unsigned need=0;
    size_t left=SIZE_MAX,right=SIZE_MAX;
    bool native=false,right_first=false;
    series::FinitePlan finite{};
    planning::SlotEnvelope slots{};
    planning::SlotLayout layout{};
    StoredMergePlan merge{};
    std::array<IntegerShape,3> result{};
    uint64_t return_bytes=0,peak_bytes=0,write_bytes=0;
    unsigned loan_mask=0;
    size_t loan_offsets[3]{}; // compact native returns into dead workspace for the parent
};
struct SeriesBlock {
    series::PrecisionBlock block{};
    planning::PrecisionEpisode resident{};
    size_t root=SIZE_MAX;
    StoredMergePlan merge{};
};
struct SeriesMetadata {
    size_t bytes=0,alignment=4096,block_capacity=0,node_capacity=0;
    size_t blocks=0,nodes=0,summaries=0,choices=0,memo=0;
};
bool query_series_metadata(size_t block_capacity,size_t node_capacity,SeriesMetadata &) noexcept;
struct SeriesPlan {
    const series::PsrSpec *spec=nullptr;
    SeriesOptions options{};
    std::span<SeriesBlock> blocks{};
    std::span<SeriesNode> nodes{};
    std::span<planning::Decomposition> choices{};
    planning::ResidentFrontier resident{};
    size_t work_bytes=0,work_alignment=4096,io_bytes=0,shared_bytes=0,fixed_bytes=0,total_resident_bytes=0;
    size_t metadata_bytes=0,minimum_free_regions=0;
    const uint8_t *metadata_base=nullptr;
    unsigned max_depth=0;
    uint64_t peak_backing_bytes=0,write_bytes=0;
    uint64_t additional_backing_bytes=0;
    uint64_t error_units=0; // same finite/rounding certificate as resident PSR
    std::array<IntegerShape,3> result_shape{}; // includes backing, not just live windows
    uint64_t native_quotes=0,product_quotes=0,memo_hits=0,memo_misses=0;
};
// Metadata lives at stable addresses through compile/run; the spec and formula
// contexts are immutable. Only coarse external nodes are materialized. All
// product choices and native frontiers are fixed before arithmetic starts.
// Compile owns the metadata exclusively. Failure invalidates out and leaves
// that metadata discardable; no previous plan may keep using the same span.
planning::ExternalPlanStatus compile_series(const series::PsrSpec &,SeriesBudget,const SeriesOptions &,
    const planning::ExternalProductProfile &,const SeriesMetadata &,std::span<uint8_t>,SeriesPlan &) noexcept;
struct SeriesResult {
    sbn3_series_values resident{};
    StoredTriple stored{};
    bool in_store=false;
};
// Resident results borrow the supplied retained region. Stored results own
// their values. No OS allocation or scratch growth occurs during execution.
// Cancellation is checked at native-operation/IO boundaries, never in kernels.
IoError run_series(const SeriesPlan &,Transport &,ValueStore &,sbn3_arena &,sbn3_team &,
                   const sbn3_lease &work,std::span<uint8_t>io,SeriesResult &) noexcept;
} // namespace sbn::v3::external
