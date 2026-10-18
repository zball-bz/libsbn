// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/forward_planes.hpp"
#include "external/middle_rows.hpp"
#include "external/integer_emit.hpp"
namespace sbn::v3::external {
struct RowCacheBinding;
struct SpilledProductSpec {
    size_t an=0,bn=0,outer_cells=0,inner_cells=0,stripe_cells=0;
    unsigned primes=0,trunk_bits=0,prime_batch=0,workers=1;
    bool square=false;
    size_t retained_bytes=0; // chunk-aligned B-family prefix, live through all passes
    SourceWindow a_source{},b_source{};
    CyclicResidualSpec cyclic{};
    // 0/1: retain this input as raw limbs; 2: materialize ordinary/cached rows.
    unsigned raw_operand=2;
    bool compact_middle=false; // two expanded banks double as blocked packed IO slots
    bool duplex_middle=false; // three roles/slots, subject to transport capability
    bool stream_final=false; // two or three inverse/output stripe banks
    size_t forward_stripe_cells=0; // zero follows inverse stripe; otherwise an independent IO staging width
    size_t raw_cache_bytes=SIZE_MAX; // finite: normalized cell-aligned raw prefix; suffix is streamed per prime
    size_t inverse_stripe_cells=0; // zero follows stripe_cells; final IO width independent of source staging
    MiddleWindowSpec middle{};
    size_t raw_stripe_cells=0; // zero follows stripe_cells; finite-prefix source read target
};
struct SpilledProductPlan {
    SpilledProductSpec spec{};
    ForwardPlanesPlan forward[2]{};
    PrefixForwardPlan raw_forward{};
    MiddleRowsPlan middle{};
    InversePlanesPlan inverse{};
    IntegerEmitPlan emit{};
    uint64_t output_bytes=0,plane_bytes=0,plane_allocation_bytes=0;
    uint64_t output_first_byte=0,stored_output_bytes=0,max_stored_output_bytes=0;
    uint64_t normal_read_bytes=0,normal_write_bytes=0;
    unsigned plane_count=0;
    // inverse_offset starts the final-phase loan: dead raw prefix/index plus
    // pool when present. Retained result planes are outside this loan.
    size_t storage_bytes=0,shared_bytes=0,control=0,plane_indexes=0,plane_index_stride=0,plane_bits=0,plane_bits_stride=0;
    size_t output_index=0,output_bits=0,io_data=0,io_page=0,pool=0,pool_bytes=0,inverse_offset=0,emit_offset=0;
    size_t output_index_bytes=0,output_bits_bytes=0;
    size_t retained=0;
    size_t inverse_second=0,inverse_third=0;
    unsigned final_banks=0;
    uint64_t product_scale[10]{}; // folded into each middle pass's normalization
    size_t raw_value=0,raw_bytes=0,raw_cached_bytes=0,raw_phase=0,raw_input_index=0;
    uint64_t raw_read_bytes=0;
    bool cached_a=false;
    bool cached_square=false;
};
// Explicit geometry, pure bounded query. Private storage includes all plane
// indexes, coverage, output fragments and shared phase pools. Add context/Team,
// caller-held state and shared_bytes ONCE before applying a total RAM limit.
bool query_spilled_product(const SpilledProductSpec &,SpilledProductPlan &) noexcept;
// Execution-only prefix placement. Phase geometry and tokens stay fixed; the
// later stage bases move with this prefix. No backend query or selection.
// Size and IO changes are exact.
bool retain_spilled_prefix(SpilledProductPlan &,size_t bytes) noexcept;
// Fixed arithmetic/total grant. Reserve compact middle banks first, then put
// the remaining grant into B retention. Any displaced IO is reflected exactly.
bool compact_middle_within(SpilledProductPlan &,size_t storage_limit) noexcept;
bool duplex_middle_within(SpilledProductPlan &,size_t storage_limit) noexcept;
// Admit a second inverse/output stripe bank only in already available phase
// space. Retention and submitted bytes remain unchanged.
bool stream_final_within(SpilledProductPlan &,size_t storage_limit) noexcept;
// Same arithmetic and plane layout. Widen forward transfers within existing
// phase storage, without displacing retained data or increasing IO volume.
bool widen_forward_within(SpilledProductPlan &,size_t stripe_cells,size_t storage_limit) noexcept;
// Enlarge final checked reads in phase-dead storage. Arithmetic tiles remain
// cache-sized; raw-prefix retention, source staging and IO volume stay fixed.
bool widen_inverse_within(SpilledProductPlan &,size_t stripe_cells,size_t storage_limit) noexcept;
bool cache_raw_prefix(SpilledProductPlan &,size_t cached_bytes) noexcept;
// Fit a normalized raw prefix and its missing-row staging in a total private
// grant. A zero-prefix bound and two direct fills; no prefix-size search.
bool fit_raw_prefix(SpilledProductPlan &,size_t storage_limit,uint64_t write_limit=UINT64_MAX) noexcept;
// Fixed arithmetic, one raw input plus one prime generated immediately before
// consumption. Complete overlapping phase storage is included. No SQR/cache.
bool use_raw_operand(SpilledProductPlan &,unsigned operand) noexcept;
bool rebind_spilled_sources(SpilledProductPlan &,SourceWindow a,SourceWindow b) noexcept;
bool rebind_spilled_subtract(SpilledProductPlan &,SourceWindow) noexcept;
// Same arithmetic, but A's outer rows come from a separately admitted cache.
// Removes A construction/index grants and source read/plane-write traffic.
bool use_cached_a(SpilledProductPlan &) noexcept;
bool use_completed_cached_a(SpilledProductPlan &) noexcept;
// Square immutable cached A into fresh product rows; no B forward/materialization.
// Transform capacity and output support remain the already selected envelope.
bool use_cached_square(SpilledProductPlan &) noexcept;
// Complete arithmetic/carry/m61 stay unchanged. Placement exposes only this
// chunk-aligned-start interval; all other finalized chunks incur no Store IO.
bool project_spilled_output(SpilledProductPlan &,size_t first_limb,size_t end_limb) noexcept;
// Query envelope: reserve at most this many stored result limbs. The initial
// placement is a prefix; the owning recipe rebinds its chosen interval later.
bool bound_spilled_output(SpilledProductPlan &,size_t max_stored_limbs) noexcept;
// Source windows must have nonzero most-significant limbs, unless empty.
// Their product has at most one fewer limb than the sum of source lengths.
bool keep_high_spilled_output(SpilledProductPlan &,size_t keep_limbs) noexcept;
// All Region extents are preassigned/admitted by the caller on this transport.
// Inputs are sealed values, output and temporary scratch planes are building.
// Product runs under the complete supplied resident grant; no OS allocation.
IoError run_spilled_product(const SpilledProductPlan &,Transport &,const Region &a,const Region &b,
                            Region &output,std::span<Region>temporary,std::span<uint8_t>storage,sbn3_team &,WordExtent * = nullptr,CyclicResidualResult * = nullptr,
                            const Region *subtract=nullptr,MiddleWindowResult *middle=nullptr) noexcept;
// Stable descriptors may belong to separate preallocated Store slots.
IoError run_spilled_product_indirect(const SpilledProductPlan &,Transport &,const Region &a,const Region &b,
                            Region &output,std::span<Region *>temporary,std::span<uint8_t>storage,sbn3_team &,WordExtent * = nullptr,CyclicResidualResult * = nullptr,
                            const Region *subtract=nullptr,MiddleWindowResult *middle=nullptr) noexcept;
IoError run_spilled_cached(const SpilledProductPlan &,Transport &,RowCacheBinding &,const Region &fresh,
                            Region &output,std::span<Region *>temporary,std::span<uint8_t>storage,sbn3_team &,WordExtent * = nullptr,CyclicResidualResult * = nullptr,
                            const Region *subtract=nullptr,MiddleWindowResult *middle=nullptr) noexcept;
} // namespace sbn::v3::external
