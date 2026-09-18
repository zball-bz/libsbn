#pragma once
#include "sbn3/newton.h"
#include "sbn3/product.h"
namespace sbn::v3 {
// Integer Q in [0,B^(n+1)), B=2^64, positive u64 a. Produces X with
// |X-Q/sqrt(a)| < 2^38; n>=4. Uses a half-precision rsqrt and three
// products. All lengths below are words unless explicitly named bytes.
inline constexpr uint64_t mul_rsqrt_error = uint64_t(1) << 38;
struct MulRsqrtChoice {
    sbn3_mul_options options{};
    uint64_t arithmetic_id = 0, execution_id = 0;
    size_t table_bytes = 0, workspace_bytes = 0, alignment = 128;
};
struct MulRsqrtPlan {
    size_t n = 0, m = 0, root_at = 0, input_at = 0, result_at = 0, value_words = 0;
    sbn3_newton_plan half_root{};
    size_t half_bytes = 0, square_ring = 0;
    MulRsqrtChoice choices[3]{}; // low square, Q*r, high(X)*|E|
    size_t program_at = 0, local_at[2]{}, shared_at = 0, tables_bytes = 0;
    size_t workspace_at = 0, workspace_bytes = 0, storage_bytes = 0, alignment = 128;
    uint64_t plan_id = 0;
    double estimated_tail_ns = 0, standalone_rung_ns = 0;
};
sbn3_query_result mul_rsqrt_query(size_t n, unsigned workers, MulRsqrtPlan &,
                                 size_t preferred_input_at = 0) noexcept;
// Engine-owned values[0,value_words) initially contains Q at input_at;
// every other word is scratch. Consumes Q. Returns an interior view at
// result_at, alive while that value lease lives. No implicit input copy.
// The separate arena range [offset,offset+storage_bytes) is resident and
// unleased; it is shared between the half-root and product phases.
sbn3_const_limbs mul_rsqrt_execute(const MulRsqrtPlan &, uint64_t a, uint64_t *values,
                                   size_t capacity, sbn3_arena &, size_t offset, sbn3_team &) noexcept;
} // namespace sbn::v3
