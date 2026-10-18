// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "product/program.hpp"

namespace sbn::v3::floating {
// Immutable private recipe. Bit bounds describe retained prefixes, not a
// requirement to read or materialize every low bit of an exact input.
struct FloatProductPlan {
    ProductProgramPlan product{};
    std::size_t precision_bits=0, a_prefix_bits=0, b_prefix_bits=0;
    std::size_t a_limbs=0, b_limbs=0, product_limbs=0, product_capacity=0;
    std::size_t a_offset=0, b_offset=0, product_offset=0, tables_offset=0, work_offset=0;
    std::size_t tables_bytes=0, payload_bytes=0, scratch_bytes=0, alignment=64;
    unsigned workers=1;
    bool square=false, small=false;
};

// Query and prepare precede all numerical execution. A full p+128-bit prefix
// bound accepts longer exact inputs without an input-length-dependent plan.
sbn3_query_result float_product_query(std::size_t precision_bits, std::size_t a_bits,
    std::size_t b_bits, bool square, unsigned workers, FloatProductPlan&) noexcept;

// Caller keeps both the plan and this exclusive scratch span alive. Preparing
// tables performs no allocation or plan query. One prepared recipe may be
// reused serially; output/input values must not overlap its entire span.
struct FloatProductPrepared {
    const FloatProductPlan* plan=nullptr;
    ProductProgram product{};
    std::span<std::byte> storage{};
    std::uint64_t* a=nullptr;
    std::uint64_t* b=nullptr;
    std::uint64_t* result=nullptr;
    void* work=nullptr;
};
FloatProductPrepared float_product_prepare(const FloatProductPlan&,
    std::span<std::byte> scratch) noexcept;

// ex supplies an existing idle Team or an admitted scope. Scratch is already
// bound by prepare; ex.scratch is not used. No query occurs in this function.
// A square ignores b. Exact-base output/input aliases are supported because
// both input prefixes are staged before the output is written.
float_report float_product_execute(const FloatProductPrepared&, float_ref,
    float_view a, float_view b, execution ex={}, rounding=rounding::nearest_even) noexcept;
}
