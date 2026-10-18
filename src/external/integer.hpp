// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/values.hpp"
#include "external/source_window.hpp"
#include "product/window.hpp"
#include "sbn3/team.h"
namespace sbn::v3::external {
// Exact signed magnitude times 2^exponent2. Stored magnitude may have high
// zero padding; logical_bytes, rather than a presumed normalized word, bounds
// every read. Suffix windows retain the original immutable backing/index.
struct StoredInteger {
    StoredValue value{};
    int64_t exponent2=0;
    bool negative=false;
    size_t first_limb=0,word_count=SIZE_MAX;
    WordExtent extent{};
    bool extent_known=false;
    size_t limbs() const noexcept {return value?(word_count==SIZE_MAX?size_t(value.region().layout.logical_bytes/8)-first_limb:word_count):0;}
    SourceWindow source() const noexcept {return {value?size_t(value.region().layout.logical_bytes/8):0,first_limb,limbs(),first_limb,first_limb+limbs()};}
};
// Ephemeral immutable magnitude in a caller-owned arena. The enclosing
// operation owns its lifetime; this is never a durable/Store handle.
struct ResidentInteger {
    std::span<const uint64_t> words{};
    WordExtent extent{};
    int64_t exponent2=0;
    bool negative=false;
};
// No IO/copy: normalize the known high/low zero prefix, then discard low whole
// words toward zero. A zero limit means exact. Interior zero words exposed by
// truncation may stay in the window; this changes no value or rounding error.
bool limit_stored_integer(StoredInteger &,size_t words) noexcept;
// High prefix of the magnitude, virtually shifted so its highest bit is set.
// Caller keeps the original value; short inputs receive virtual low zero words,
// never a normalization file or padding pass.
bool normalized_source(const StoredInteger &,size_t prefix_words,SourceWindow &) noexcept;
// Borrowed stored magnitude or a virtual scalar. The owning value, when used,
// must stay live through the call. Constants require no Store allocation/IO.
struct LinearInput {
    const StoredValue *value=nullptr;
    SourceWindow source{};
    uint64_t scalar=0;
    int64_t exponent2=0;
    bool negative=false;
    std::span<const uint64_t> resident{}; // exclusive owner keeps this immutable through the call
    size_t limbs() const noexcept {return value||resident.data()?source.limbs:size_t(scalar!=0);}
};
// Normalize/truncate a plain resident magnitude with the same rule as a stored
// integer. The backing span stays borrowed; no value/shift file is created.
bool limit_resident_input(LinearInput &,const WordExtent &,size_t words) noexcept;
inline LinearInput linear_input(const StoredInteger&v) noexcept {return {&v.value,v.source(),0,v.exponent2,v.negative};}
inline LinearInput linear_constant(uint64_t v,int64_t e=0,bool negative=false) noexcept {return {nullptr,{},v,e,negative};}
struct StoredSumPlan {
    size_t an=0,bn=0,result_limbs=0,storage_bytes=0;
    size_t transfer_bytes=0;
    unsigned workers=1;
    size_t a2=0,b2=0;
    int64_t ae=0,be=0,exponent=0;
    size_t io[3]{},index[3]{},a=0,b=0,raw=0;
    uint64_t output_bytes=0,output_allocation=0,write_bytes=0;
    size_t precision=0;
    size_t first_output_limb=0,end_output_limb=0;
    product::MagnitudeBound bound{};
};
bool query_stored_sum(size_t an,int64_t ae,size_t bn,int64_t be,StoredSumPlan &,size_t precision=0,unsigned workers=1) noexcept;
bool project_stored_sum(StoredSumPlan &,product::Window,product::MagnitudeBound={}) noexcept;
// Exact streaming add/subtract. Positive sums need one input scan; opposing
// signs first compare most-significant windows (one extra scan in the worst
// cancellation case). No result reread, in-place rewrite or hidden allocation.
IoError run_stored_sum(const StoredSumPlan &,Transport &,ValueStore &,
                      const StoredInteger &,const StoredInteger &,StoredInteger &,
                      std::span<uint8_t>) noexcept;
IoError run_stored_linear(const StoredSumPlan &,Transport &,ValueStore &,
                      const LinearInput &,const LinearInput &,StoredInteger &,std::span<uint8_t>,sbn3_team * = nullptr,
                      sbn3_team_scope * = nullptr) noexcept;
} // namespace sbn::v3::external
