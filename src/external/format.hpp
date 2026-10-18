// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/integer.hpp"
#include "external/resident_product.hpp"
#include "planning/product.hpp"
#include "radix/geometry.hpp"
#include "sbn3/radix.h"
namespace sbn::v3::external {
inline constexpr unsigned format_classes=96,format_powers=40;
struct FormatOptions {
    sbn3_radix_options radix{};
    uint64_t maximum_resident_fragments=UINT64_MAX; // diagnostic placement cap
};
struct FormatBudget {
    size_t resident_bytes=0,other_live_bytes=0;
    uint64_t owned_disk_limit=UINT64_MAX,other_owned_disk_bytes=0,write_limit=UINT64_MAX;
};
struct FormatClass {
    uint64_t fragments=0,left_fragments=0;
    size_t limbs=0,right_limbs=0,split_limbs=0,window=0;
    unsigned left=0,right=0,power=0;
    bool resident=false;
    bool native_frontier=false;
    sbn3_format_plan leaf{};
    sbn3_format_info leaf_info{};
    size_t leaf_bytes=0,leaf_alignment=4096,input=0,digits=0,binding=0;
    SpilledProductPlan split{};
    SpilledProductPlan carry_witness{}; // exact original LIN, only on an all-zero wrap gap
    uint64_t descendants_writes=0,descendants_normal_writes=0,backing=0,frontiers=0;
};
struct FormatPower {
    size_t limbs=0;
    bool resident=false;
    ResidentProductPlan native{};
    SpilledProductPlan external{};
    uint64_t allocation=0,writes=0;
};
struct FormatPlan {
    radix::BaseInfo base{};
    FormatOptions options{};
    SourceWindow source{};
    size_t fractional_words=0;
    uint64_t digits=0,fragments=0,seed=1;
    unsigned classes=0,powers=0;
    FormatClass node[format_classes]{};
    FormatPower power[format_powers]{};
    size_t storage_bytes=0,alignment=size_t(2)<<20,fixed_bytes=0,shared_bytes=0,total_resident_bytes=0;
    size_t io=0,index=0,output_index=0,output_bits=0,seam=0,phase=0,phase_bytes=0;
    size_t output_index_bytes=0,output_bits_bytes=0;
    uint64_t write_bytes=0,normal_write_bytes=0,peak_backing_bytes=0,output_allocation=0;
};
// Fractional output for a nonnegative limb-aligned dyadic interval
// [X,X+2^exponent2), exponent2=-64*P. The integer part must fit one word.
// Root precision must cover node_limbs(base,ceil(digits/64)); extra input words
// are views, never copied to a normalization file. Result is a certified prefix
// (not necessarily the longest); no exact-format claim on a radix boundary.
// Repeated chooses the regular grid; fresh uses the request-scaled dyadic rail.
planning::ExternalPlanStatus query_stored_format(const StoredInteger &,unsigned base,uint64_t digits,
    const FormatOptions &,const FormatBudget &,const planning::ExternalProductProfile &,FormatPlan &) noexcept;
planning::ExternalPlanStatus query_format_source(SourceWindow,size_t fractional_words,unsigned base,uint64_t digits,
    const FormatOptions &,const FormatBudget &,const planning::ExternalProductProfile &,FormatPlan &) noexcept;
struct FormatResult {
    StoredValue digits{}; // physical requested-width output; expose only certified prefix
    uint64_t certified_digits=0,integer_part=0;
    uint64_t middle_products=0,middle_fallbacks=0;
};
// Caller gives one prepared, aligned, exclusive initially-unleased interval.
// Input is preserved. Output chunks are finalized in reverse order and written
// once; unpublished output/rails/child values are released on IO failure.
IoError run_stored_format(const FormatPlan &,Transport &,ValueStore &,const StoredInteger &,FormatResult &,
    sbn3_arena &,size_t offset,sbn3_team &) noexcept;
} // namespace sbn::v3::external
