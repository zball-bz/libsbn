#include "algorithms/sqrt2_rational.hpp"
#include "common/checked.hpp"
#include "value/limbs.hpp"
namespace sbn::v3 {
namespace {
// For p_1=3, p_{j+1}=2*p_j^2-1, bit_length(p_20)=1333320.
// 2^L < 2*p_20-1 < (1+sqrt(2))^(2^20) < 2*p_20 < 2^(L+1).
// Reproducible integer witness: results/sqrt2_2026-09-09/rational-size-certificate.json.
constexpr uint64_t log_lower = 1333320, log_upper = 1333321;
constexpr unsigned log_denominator = 20;
void doubled(uint64_t *a, size_t n, bool minus_one) {
    uint64_t carry = 0;
    for (size_t j = 0; j < n; ++j) {
        const uint64_t v = a[j];
        a[j] = (v << 1) | carry;
        carry = v >> 63;
    }
    a[n] = carry;
    // p is always odd, hence 2*p^2 has a nonzero low word (bit 1 set).
    if (minus_one)
        --a[0];
}
} // namespace
size_t rational_sqrt2_capacity(unsigned iteration) noexcept {
    require(iteration >= 1 && iteration <= 40, SBN3_FATAL_ARGUMENT, "rational sqrt2 iteration");
    const __uint128_t bits = __uint128_t(log_upper) << iteration, den = __uint128_t(64) << log_denominator;
    return size_t((bits + den - 1) / den);
}
unsigned rational_sqrt2_iterations(size_t fractional_limbs) noexcept {
    require(fractional_limbs >= 1 && fractional_limbs <= (size_t(1) << 31), SBN3_FATAL_ARGUMENT,
            "rational sqrt2 precision");
    const __uint128_t need = (__uint128_t(fractional_limbs) * 64 + 3) << log_denominator;
    unsigned k = 1;
    while ((__uint128_t(log_lower) << (k + 1)) < need)
        ++k;
    return k; // q_k > (1+sqrt(2))^(2^k)/4 gives 2*q_k^2 > B^fractional_limbs.
}
RationalSqrt2Value rational_sqrt2(const RationalSqrt2Program &p) noexcept {
    require(p.last_iteration >= 1 && p.last_iteration <= 40, SBN3_FATAL_ARGUMENT, "rational sqrt2 program");
    unsigned pi = 0, qi = 1, temp = 2;
    p.values[pi][0] = 3;
    p.values[qi][0] = 2;
    for (unsigned k = 1; k < p.last_iteration; ++k) {
        const auto &step = p.steps[k - 1];
        const size_t n = step.input_limbs, next = step.output_limbs;
        require(n == rational_sqrt2_capacity(k) && next == rational_sqrt2_capacity(k + 1) &&
                    next <= 2 * n + 1,
                SBN3_FATAL_ARGUMENT, "rational sqrt2 stage sizes");
        if (step.p)
            sbn3_spectrum_compute(step.multiply, step.p, {p.values[pi], n});
        sbn3_product_inputs input{};
        if (!step.p)
            input.a = {p.values[pi], n};
        input.b = {p.values[qi], n};
        sbn3_product_execute(step.multiply, &input, {p.values[temp], 2 * n});
        doubled(p.values[temp], 2 * n, false);
        input.b = {};
        sbn3_product_execute(step.square, &input, {p.values[qi], 2 * n});
        doubled(p.values[qi], 2 * n, true);
        require(limbs::zero(p.values[temp] + next, 2 * n + 1 - next) &&
                    limbs::zero(p.values[qi] + next, 2 * n + 1 - next),
                SBN3_FATAL_MATH, "rational sqrt2 capacity certificate");
        const unsigned dead = pi;
        pi = qi;
        qi = temp;
        temp = dead;
    }
    return {p.values[pi], p.values[qi], rational_sqrt2_capacity(p.last_iteration)};
}
void normalize_rational_sqrt2(RationalSqrt2Value v, size_t n, uint64_t *A, uint64_t *D) noexcept {
    require(v.limbs && v.limbs <= n, SBN3_FATAL_ARGUMENT, "rational normalization capacity");
    size_t qn = v.limbs;
    while (qn && !v.q[qn - 1])
        --qn;
    require(qn, SBN3_FATAL_MATH, "zero rational denominator");
    const size_t words = n - qn;
    const unsigned shift = unsigned(__builtin_clzll(v.q[qn - 1]));
    auto copy = [&](uint64_t *out, const uint64_t *in) {
        for (size_t j = 0; j <= n; ++j) {
            if (j < words) {
                out[j] = 0;
                continue;
            }
            const size_t index = j - words;
            const uint64_t lo = index < v.limbs ? in[index] : 0,
                           hi = index && index - 1 < v.limbs ? in[index - 1] : 0;
            out[j] = shift ? (lo << shift) | (hi >> (64 - shift)) : lo;
        }
    };
    copy(A, v.p);
    copy(D, v.q);
    require(!D[n] && (D[n - 1] >> 63) && A[n] <= 1, SBN3_FATAL_MATH, "rational normalization range");
}
} // namespace sbn::v3
