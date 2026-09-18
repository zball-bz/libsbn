#include "algorithms/divide_terminal.hpp"
#include "algorithms/product_stage.hpp"
#include "product/build_apply.hpp"
#include "common/checked.hpp"
#include "algorithms/newton_contract.hpp"
#include "value/limbs.hpp"
#include "value/parallel_limbs.hpp"
#include <string.h>
namespace sbn::v3 {
void divide_terminal(const DivideTerminal &p, const uint64_t *A, const uint64_t *D, const uint64_t *U,
                     uint64_t *Q) noexcept {
    const size_t m = p.m, n = p.n, k = p.ring;
    const size_t output_capacity = p.output_capacity ? p.output_capacity : k;
    require(m >= 3 && m < n && n - m <= m - 1 && n <= (size_t(1) << 31) &&
                k >= newton_contract::division_ring_min(m) && k <= (size_t(1) << 31) && output_capacity >= k,
            SBN3_FATAL_ARGUMENT, "division terminal shape");
    require(A && D && U && Q && A[n] <= 1 && (D[n - 1] >> 63) && U[m] == 1, SBN3_FATAL_ARGUMENT,
            "division terminal inputs");
    const size_t bn = p.input_words ? p.input_words : n + 1;
    require(bn >= m + 1 && bn >= newton_contract::residual_words(m, n) && bn <= n + 1, SBN3_FATAL_ARGUMENT,
            "division reduced operand span");
    auto *x = p.work[0], *y = p.work[1];
    ProductStageRun stage(p.stage);
    auto *cached_product = stage.bind(p.cached_inverse, 0);
    auto *cache = stage.cache(p.u);
    // q0=floor(A_hi*U/B^m), only half-precision quotient information.
    parallel_limbs::copy(p.team, x, A + n - m, m + 1);
    parallel_limbs::fill(p.team, x + m + 1, bn - m - 1);
    sbn3_product_inputs in{};
    if (!cache)
        in.a = {U, m + 1};
    in.b = {x, bn};
    if(cache)spectrum_compute_multiply(cached_product,cache,{U,m+1},{x,bn},{y,output_capacity});
    else sbn3_product_execute(cached_product, &in, {y, output_capacity});
    // Retain q0 in the dead A_hi buffer. Delay writes to the caller output
    // until the residual has consumed all of A and D; Q may then equal A.
    parallel_limbs::copy(p.team, x, y + m, m + 1);

    in.a = {x, m + 1};
    in.b = {D, n};
    sbn3_product_execute(stage.bind(p.residual, 1), &in, {y, output_capacity});
    // |D*q0-A*B^m|<32 B^n. The shifted A crosses at most one ring seam.
    parallel_limbs::cyclic_sub_shifted(p.team, y, k, A, n + 1, m);
    const bool negative = parallel_limbs::cyclic_absolute(p.team, y, k);
    require(y[n] < newton_contract::division_residual_limit &&
                parallel_limbs::zero(p.team, y + n + 1, k - n - 1),
            SBN3_FATAL_MATH, "division residual certificate");
    // The certificate makes words above n zero. rho therefore occupies only
    // n-m+3 words. Inputs A/D are now dead. Borrow the output's low words
    // for rho, preserving q0 in x until the correction product completes.
    const size_t shift = newton_contract::residual_shift(m), copy = n + 1 - shift;
    parallel_limbs::copy(p.team, Q, y + shift, copy);
    parallel_limbs::fill(p.team, Q + copy, bn - copy);
    cached_product = stage.bind(p.cached_inverse, 0);
    cache = stage.cache(p.u);
    // The compact stage retires the cache during the large residual product.
    // Rebuild it once from the still-live half inverse for the correction.
    in.a = cache ? sbn3_const_limbs{} : sbn3_const_limbs{U, m + 1};
    in.b = {Q, bn};
    if(p.stage && cache)spectrum_compute_multiply(cached_product,cache,{U,m+1},{Q,bn},{y,output_capacity});
    else sbn3_product_execute(cached_product, &in, {y, output_capacity});
    require(y[n + 2] < newton_contract::division_correction_limit &&
                parallel_limbs::zero(p.team, y + n + 3, k - n - 3),
            SBN3_FATAL_MATH, "division correction certificate");
    const uint64_t *correction = y + newton_contract::correction_offset(m);
    const size_t count = n - m + 1;
    parallel_limbs::fill(p.team, Q, n - m);
    parallel_limbs::copy(p.team, Q + n - m, x, m + 1);
    const uint64_t spill = negative ? parallel_limbs::add_to(p.team, Q, n + 1, correction, count)
                                    : parallel_limbs::sub_from(p.team, Q, n + 1, correction, count);
    require(!spill, SBN3_FATAL_MATH, "division quotient correction overflow");
}
} // namespace sbn::v3
