#include "algorithms/inverse_rung.hpp"
#include "algorithms/product_stage.hpp"
#include "product/build_apply.hpp"
#include "common/checked.hpp"
#include "algorithms/newton_contract.hpp"
#include "value/limbs.hpp"
#include "value/parallel_limbs.hpp"
#include <string.h>
namespace sbn::v3 {
void inverse_rung(const InverseRung &p, const uint64_t *D, const uint64_t *U, uint64_t *V,
                  sbn3_product_metrics *residual_counts, sbn3_product_metrics *correction_counts) noexcept {
    const size_t m = p.m, n = p.n, k = p.ring;
    require(m >= 3 && m < n && n - m <= m - 1 && n <= (size_t(1) << 31) &&
                k >= newton_contract::inverse_ring_min(n) && k <= (size_t(1) << 31),
            SBN3_FATAL_ARGUMENT, "inverse rung shape");
    require(D && U && V && (D[n - 1] >> 63) && U[m] == 1, SBN3_FATAL_ARGUMENT, "inverse normalized input");
    ProductStageRun stage(p.stage);
    auto *product = stage.bind(p.product, 0);
    auto *cache = stage.cache(p.u);
    sbn3_product_inputs input{};
    if (!cache)
        input.a = {U, m + 1};
    input.b = {D, n};
    if(cache)spectrum_compute_multiply(product,cache,{U,m+1},{D,n},{p.residual,k});
    else sbn3_product_execute(product, &input, {p.residual, k});
    if (residual_counts)
        sbn3_product_get_metrics(product, residual_counts);

    // R=D*U-B^(n+m), |R|<10 B^n from the input error bound.
    // P=B^k-1>2|R|, so this lift reconstructs R exactly, without the
    // unwanted upper half of the full D*U integer product.
    limbs::cyclic_sub_power(p.residual, k, (n + m) % k);
    const bool negative = parallel_limbs::cyclic_absolute(p.team, p.residual, k);
    require(p.residual[n] < newton_contract::inverse_residual_limit &&
                parallel_limbs::zero(p.team, p.residual + n + 1, k - n - 1),
            SBN3_FATAL_MATH, "inverse residual certificate");

    // rho=floor(|R|/B^(m-2)). Its two retained guard limbs make the
    // discarded contribution to the final correction smaller than 2/B^2.
    const size_t shift = newton_contract::residual_shift(m), copy = n + 1 - shift;
    parallel_limbs::copy(p.team, p.correction, p.residual + shift, copy);
    if (copy < n)
        parallel_limbs::fill(p.team, p.correction + copy, n - copy);
    input.b = {p.correction, n};
    sbn3_product_execute(product, &input, {p.residual, k});
    if (correction_counts)
        sbn3_product_get_metrics(product, correction_counts);
    // U*rho<32 B^(n+2)<P: the second cyclic result is the exact integer.
    require(p.residual[n + 2] < newton_contract::inverse_correction_limit &&
                parallel_limbs::zero(p.team, p.residual + n + 3, k - n - 3),
            SBN3_FATAL_MATH, "inverse correction certificate");

    parallel_limbs::fill(p.team, V, n - m);
    parallel_limbs::copy(p.team, V + n - m, U, m + 1);
    const auto *correction = p.residual + newton_contract::correction_offset(m);
    const size_t count = n - m + 1;
    const uint64_t spill = negative ? parallel_limbs::add_to(p.team, V, n + 1, correction, count)
                                    : parallel_limbs::sub_from(p.team, V, n + 1, correction, count);
    require(!spill, SBN3_FATAL_MATH, "inverse correction overflow");
    // The implicit-high-one reciprocal convention excludes the exact upper
    // endpoint 2B^n. Clamping contributes at most one additional ulp there.
    if (V[n] == 0)
        memset(V, 0, n * 8);
    else if (V[n] > 1)
        memset(V, 0xff, n * 8);
    V[n] = 1;
}
} // namespace sbn::v3
