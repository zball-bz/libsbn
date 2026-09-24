#include "algorithms/rsqrt.hpp"
#include "product/stage.hpp"
#include "common/checked.hpp"
#include "algorithms/newton_contract.hpp"
#include "value/limbs.hpp"
#include "value/parallel_limbs.hpp"
#include <string.h>
#include <cmath>
namespace sbn::v3 {
namespace {
uint64_t scaled_square(uint64_t *s, const uint64_t *r, size_t n, uint64_t a) {
    mul_basecase_assumed(s, r, n, r, n);
    return sbn3i_mul_1(s, s, long(2 * n), a);
}
} // namespace
void rsqrt_seed(uint64_t *R, uint64_t a, size_t n) noexcept {
    require(a && R && n >= 1 && n <= 3, SBN3_FATAL_ARGUMENT, "rsqrt seed input");
    if (a == 1) {
        memset(R, 0xff, n * 8);
        R[n] = 0;
        return;
    }
    // Only the initial estimate is floating point. With w=log2(sqrt(a))
    // in [0.5,32], its relative error is <2^-50+2^(w-64).
    memset(R, 0, (n + 1) * 8);
    R[n - 1] = uint64_t(0x1p64 / std::sqrt(double(a)));
    uint64_t s[6]{}, product[9]{}, correction[3]{}, next[3]{};
    for (size_t step = 0; step < n; ++step) {
        const uint64_t high = scaled_square(s, R, n, a);
        require(high <= 1, SBN3_FATAL_MATH, "rsqrt seed estimate range");
        const bool negative = !high; // a*R^2-B^(2n), uniquely signed near zero
        if (negative) {
            uint64_t carry = 1;
            for (size_t j = 0; j < 2 * n; ++j) {
                const __uint128_t v = __uint128_t(~s[j]) + carry;
                s[j] = uint64_t(v);
                carry = uint64_t(v >> 64);
            }
        }
        mul_basecase_assumed(product, R, n, s, 2 * n);
        for (size_t j = 0; j < n; ++j)
            correction[j] = (product[2 * n + j] >> 1) | (j + 1 < n ? product[2 * n + j + 1] << 63 : 0);
        const auto spill =
            negative ? limbs::add_to(R, n, correction, n) : limbs::sub_from(R, n, correction, n);
        require(!spill, SBN3_FATAL_MATH, "rsqrt seed correction range");
    }
    // n=1/2/3 fixed-precision Newton steps suffice for 64n absolute bits:
    // the unrounded final error is <2^-28 in the worst (n=2,w=32) case;
    // integer correction rounding leaves <2 ulp. Certify the exact strict
    // seed floor(sqrt((B^(2n)-1)/a)), including perfect-square endpoints.
    const uint64_t one = 1;
    for (unsigned tries = 0; tries < 4; ++tries) {
        if (scaled_square(s, R, n, a)) {
            limbs::sub_from(R, n, &one, 1);
            continue;
        }
        memcpy(next, R, n * 8);
        const auto carry = limbs::add_to(next, n, &one, 1);
        if (carry || scaled_square(s, next, n, a))
            return;
        memcpy(R, next, n * 8);
    }
    fatal(SBN3_FATAL_MATH, "rsqrt seed integer certificate");
}
void rsqrt_rung(const RsqrtRung &p, uint64_t a, const uint64_t *R, uint64_t *out) noexcept {
    const size_t m = p.m, n = p.n, k = p.ring;
    require(a && m >= 2 && m < n && n - m <= m - 1 && n <= (size_t(1) << 31) &&
                k >= newton_contract::rsqrt_ring_min(m) && k <= (size_t(1) << 31),
            SBN3_FATAL_ARGUMENT, "rsqrt rung shape");
    ProductStageRun stage(p.stage);
    auto *square = stage.bind(p.square, 0);
    auto *cache = stage.cache(p.r);
    sbn3_product_inputs in{};
    if (!cache)
        in.a = {R, m};
    const size_t sn = p.low_square ? m + newton_contract::guard_words : k;
    if (cache)
        sbn3_spectrum_compute_square(square, cache, {R, m}, {p.residual, sn});
    else
        sbn3_product_execute(square, &in, {p.residual, sn});
    const auto carry = parallel_limbs::mul_1(p.team, p.residual, p.residual, sn, a);
    bool negative;
    if (p.low_square) {
        // E=a*R^2-B^(2m), |E|<2^36*B^m. Mod B^(m+2), the
        // B^(2m) term vanishes (m>=2) and E has a unique signed lift.
        // The scalar multiplication's carry is outside this exact low window.
        (void)carry;
        negative = p.residual[sn - 1] >> 63;
        if (negative) {
            parallel_limbs::complement(p.team, p.residual, sn);
            parallel_limbs::add_word(p.residual, sn, 1);
        }
    } else {
        require(!carry, SBN3_FATAL_MATH, "rsqrt scalar square overflow");
        limbs::cyclic_sub_power(p.residual, k, 2 * m);
        negative = parallel_limbs::cyclic_absolute(p.team, p.residual, k);
    }
    require(p.residual[m] < newton_contract::rsqrt_residual_limit &&
                parallel_limbs::zero(p.team, p.residual + m + 1, sn - m - 1),
            SBN3_FATAL_MATH, "rsqrt residual certificate");
    in.b = {p.residual, m + 1};
    sbn3_product_execute(stage.bind(p.multiply, 1), &in, {p.correction, k});
    require(parallel_limbs::zero(p.team, p.correction + 2 * m + 1, k - 2 * m - 1), SBN3_FATAL_MATH,
            "rsqrt correction certificate");
    // Correction = floor(R*|E| / (2*B^(3m-n))). Both cached products use
    // the same full basis. We square only the m-limb approximation, and
    // never redo its forward transform for the correction.
    const size_t first = 3 * m - n, count = n - m + 2;
    parallel_limbs::each(p.team, count, parallel_limbs::parts(p.team, count),
                         [&](size_t begin, size_t end, unsigned) {
                             for (size_t j = begin; j < end; ++j) {
                                 const size_t pos = first + j;
                                 const uint64_t lo = pos < k ? p.correction[pos] : 0,
                                                hi = pos + 1 < k ? p.correction[pos + 1] : 0;
                                 p.residual[j] = (lo >> 1) | (hi << 63);
                             }
                         });
    parallel_limbs::fill(p.team, out, n - m);
    parallel_limbs::copy(p.team, out + n - m, R, m);
    out[n] = 0;
    const auto spill = negative ? parallel_limbs::add_to(p.team, out, n + 1, p.residual, count)
                                : parallel_limbs::sub_from(p.team, out, n + 1, p.residual, count);
    require(!spill, SBN3_FATAL_MATH, "rsqrt correction overflow");
    if (out[n]) {
        memset(out, 0xff, n * 8);
        out[n] = 0;
    }
}
void rsqrt_program(const RsqrtProgram &p, uint64_t a, uint64_t *R) noexcept {
    require(p.target && p.seed_limbs <= p.target && p.seed_limbs >= 1 && p.seed_limbs <= 3 &&
                p.rung_count <= 32,
            SBN3_FATAL_ARGUMENT, "rsqrt program shape");
    if (!p.rung_count) {
        require(p.target == p.seed_limbs, SBN3_FATAL_ARGUMENT, "rsqrt seed target");
        rsqrt_seed(R, a, p.target);
        return;
    }
    rsqrt_seed(p.values[0], a, p.seed_limbs);
    size_t m = p.seed_limbs;
    unsigned live = 0;
    for (size_t j = 0; j < p.rung_count; ++j) {
        const auto &r = p.rungs[j];
        require(r.m == m && r.n <= p.target, SBN3_FATAL_ARGUMENT, "rsqrt precision ladder");
        auto *out = j + 1 == p.rung_count ? R : p.values[live ^ 1];
        rsqrt_rung(r, a, p.values[live], out);
        m = r.n;
        live ^= 1;
    }
    require(m == p.target, SBN3_FATAL_ARGUMENT, "rsqrt incomplete ladder");
}
} // namespace sbn::v3
