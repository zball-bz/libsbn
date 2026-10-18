// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "external/store.hpp"
#include "product/program.hpp"
#include "external/source_window.hpp"

namespace sbn::v3::external {
// Stored-boundary promotion: the COMPLETE arithmetic episode must fit its
// supplied grant. This is not a spilled transform or a low-RAM fallback.
struct ResidentProductPlan {
    ProductProgramPlan product{};
    size_t storage_bytes = 0, alignment = 4096;
    size_t a = 0, b = 0, result = 0, tables = 0, work = 0, transfer = 0, index = 0;
    size_t transfer_bytes = 0, output_limbs = 0;
    SourceWindow source[2]{};
    size_t source_capacity[2]{},output_first=0,maximum_output_limbs=0,result_prefix=0;
    unsigned borrowed_inputs=0; // immutable resident spans; omit their copied input arrays
    bool require_zero_high=true;
    bool square=false;
};
sbn3_query_result query_resident_product(size_t an, size_t bn, const sbn3_mul_options &,
                                        size_t transfer_bytes, ResidentProductPlan &) noexcept;
sbn3_query_result query_resident_square(size_t words,const sbn3_mul_options &,size_t transfer_bytes,
                                      ResidentProductPlan &) noexcept;
// Same fixed arithmetic; source rebinding changes only the loaded view. The
// first call establishes the backing-size envelope before execution.
bool set_resident_sources(ResidentProductPlan &,SourceWindow,SourceWindow) noexcept;
bool rebind_resident_sources(ResidentProductPlan &,SourceWindow,SourceWindow) noexcept;
// Complete output stays in the planned result array. The stored view may be a
// high window, with full-product m61 checked before it is exposed.
bool keep_high_resident_output(ResidentProductPlan &,size_t keep) noexcept;
// Reserve a result array beyond all intervening/consumer workspace. The
// result can follow this operation's dead transform work, avoiding a duplicated
// prefix hole. Arithmetic identity is unchanged; the full layout is requoted.
bool reserve_resident_result_after(ResidentProductPlan &,size_t prefix) noexcept;
bool compact_resident_product(ResidentProductPlan &) noexcept;
// Same arithmetic, resident source windows with no bit shift/complement or
// virtual words. Sources stay immutable and outside this operation's private
// storage through return. Bounded-input support is required; no padding loan.
bool borrow_resident_inputs(ResidentProductPlan &,unsigned mask) noexcept;
struct ResidentProductInput {
    const Region *stored=nullptr;
    std::span<const uint64_t> words{};
};
struct ResidentProductValue {std::span<const uint64_t> words{};WordExtent extent{};};
// Two independent products with one immutable common operand. Fresh stored
// inputs are staged in their own eventual output arrays; the native pair
// consumes them only after building the common spectrum. Each full product is checked before its projected view is exposed. A joined
// consumer retires the first view before that buffer stages the second input.
struct ResidentPairPlan {
    ProductProgramPlan product{};
    SourceWindow source[3]{}; // common, first fresh, second fresh
    size_t capacity[3]{},source_capacity[3]{};
    size_t storage_bytes=0,alignment=4096,common=0,result=0,tables=0,work=0,transfer=0,index=0;
    size_t transfer_bytes=0,consume_bytes=0;
    unsigned borrowed_inputs=0;
};
bool query_resident_pair(const ResidentProductPlan &first,const ResidentProductPlan &second,
                         size_t consume_bytes,ResidentPairPlan &) noexcept; // common is each original B
bool borrow_resident_pair_inputs(ResidentPairPlan &,unsigned mask) noexcept;
bool rebind_resident_pair(ResidentPairPlan &,SourceWindow common,SourceWindow first,SourceWindow second) noexcept;
IoError run_resident_pair(const ResidentPairPlan &,Transport &,const ResidentProductInput (&)[3],
                         const ResidentProductPlan (&outputs)[2],std::span<uint8_t>,sbn3_team &,
                         void*context,IoError (*consume_first)(void*,const ResidentProductValue&,sbn3_team_scope*),
                         ResidentProductValue &second) noexcept;
IoError run_resident_product_borrowed(const ResidentProductPlan &,Transport &,const Region &,const Region &,
                                    std::span<uint8_t>,sbn3_team &,ResidentProductValue &) noexcept;
IoError run_resident_product_borrowed(const ResidentProductPlan &,Transport &,ResidentProductInput,ResidentProductInput,
                                    std::span<uint8_t>,sbn3_team &,ResidentProductValue &) noexcept;
// Source/output allocations are disjoint; input regions remain sealed and
// unchanged. Output starts building and seals only on success. Math invariant
// failures are fatal; IO/cancellation failures invalidate the unpublished output.
// The caller owns and keeps the complete storage range resident/exclusive.
IoError run_resident_product(const ResidentProductPlan &, Transport &, const Region &, const Region &,
                            Region &, std::span<uint8_t> storage, sbn3_team &,WordExtent * = nullptr) noexcept;
IoError run_resident_product(const ResidentProductPlan &,Transport &,ResidentProductInput,ResidentProductInput,
                            Region &,std::span<uint8_t>,sbn3_team &,WordExtent * = nullptr) noexcept;
// Retain an exact low prefix only when every omitted high word is zero.
// Full arithmetic, residue checks and workspace stay unchanged.
bool project_resident_output(ResidentProductPlan &,size_t limbs) noexcept;
} // namespace sbn::v3::external
