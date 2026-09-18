// Logarithmic product estimator: every interval must contain the exact value
// obtained from an independent big-integer product, and stay tight.
#include "series/log_product.hpp"
#include "../oracle/oracle.h"
#include <assert.h>
#include <initializer_list>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
using namespace sbn::v3::series;
namespace {
uint64_t state = 0x9e3779b97f4a7c15ULL;
uint64_t rnd() { state ^= state << 13; state ^= state >> 7; state ^= state << 17; return state; }
// log2 of a positive big integer to about 1e-18: bit length plus log2 of its top 64 bits.
long double log2_exact(const ref_number *x, long double &slack) {
    const size_t bits = ref_sizeinbase(x, 2);
    ref_int top;
    ref_init(top);
    if (bits > 64)
        ref_fdiv_q_2exp(top, x, bits - 64);
    else
        ref_set(top, x);
    const uint64_t t = ref_get_ui(top);
    ref_clear(top);
    // Truncating the low bits lowers the value by less than one unit of the 64-bit top: relative 2^-63.
    slack = bits > 64 ? 2.0e-19L : 0;
    return (bits > 64 ? (long double)(bits - 64) : 0) + log2l((long double)t);
}
void set128(ref_number *out, __uint128_t x) {
    const uint64_t w[]{uint64_t(x), uint64_t(x >> 64)};
    ref_import(out, 2, -1, 8, 0, 0, w);
}
size_t checked = 0;
long double worst_width = 0, worst_margin = 1;
void check_linear(uint64_t alpha, int64_t beta, uint64_t a, uint64_t b) {
    ref_int product, factor;
    ref_inits(product, factor, nullptr);
    ref_set_ui(product, 1);
    for (uint64_t k = a; k < b; ++k) {
        const __uint128_t v = beta < 0 ? __uint128_t(alpha) * k - (__uint128_t(uint64_t(-(beta + 1))) + 1)
                                       : __uint128_t(alpha) * k + uint64_t(beta);
        set128(factor, v);
        ref_mul(product, product, factor);
    }
    long double slack = 0;
    const long double exact = log2_exact(product, slack);
    const auto got = linear_log2_sum(alpha, beta, a, b);
    // The reference itself carries long double rounding of one log2 and the top-word truncation.
    const long double tolerance = slack + 8 * 1.1e-19L * (fabsl(exact) + 1);
    if (!(got.lower <= exact + tolerance && exact - tolerance <= got.upper)) {
        fprintf(stderr, "alpha=%llu beta=%lld [%llu,%llu): exact %.18Lf not in [%.18Lf, %.18Lf]\n",
                (unsigned long long)alpha, (long long)beta, (unsigned long long)a, (unsigned long long)b, exact,
                (long double)got.lower, (long double)got.upper);
        assert(false);
    }
    // Tightness: analytic radius below 2e-5 bit plus the binary64 evaluation allowance, a few 1e-14 of the value.
    const long double width = (long double)got.upper - (long double)got.lower;
    assert(width <= 1.0e-4L + 6.0e-14L * fabsl(exact));
    worst_width = fmaxl(worst_width, width);
    worst_margin = fminl(worst_margin, fminl(exact - got.lower, got.upper - exact));
    ++checked;
    ref_clears(product, factor, nullptr);
}
} // namespace
int main() {
    // The cases of the design check (bench/probes/series_precision_math_2026-09-17/check.py) ...
    check_linear(1, 0, 1, 128);
    check_linear(6, -5, 1, 99);
    check_linear(2, -1, 1, 65);
    check_linear(UINT64_MAX, -2, 1, 32);
    check_linear(6, -5, (uint64_t(1) << 48) - 50, uint64_t(1) << 48);
    // ... ranges around the direct/Stirling switch, short ranges at large starts, large negative and
    // positive offsets that nearly cancel the start, long ranges.
    for (uint64_t a : {uint64_t(1), uint64_t(2), uint64_t(7), uint64_t(8), uint64_t(9), uint64_t(1000),
                       (uint64_t(1) << 32) + 5, (uint64_t(1) << 47) + 12345})
        for (uint64_t h : {uint64_t(1), uint64_t(2), uint64_t(15), uint64_t(16), uint64_t(17), uint64_t(18),
                           uint64_t(100), uint64_t(3000)})
            for (uint64_t alpha : {uint64_t(1), uint64_t(2), uint64_t(6), uint64_t(1000003), uint64_t(1) << 40}) {
                check_linear(alpha, 0 + (a ? 0 : 1), a, a + h);
                check_linear(alpha, 12345, a, a + h);
                const __uint128_t first = __uint128_t(alpha) * a;
                if (first > 1 && first - 1 <= __uint128_t(INT64_MAX))
                    check_linear(alpha, -int64_t(uint64_t(first - 1)), a, a + h); // first factor is exactly 1
                if (first > 7 && first - 7 <= __uint128_t(INT64_MAX))
                    check_linear(alpha, -int64_t(uint64_t(first - 7)), a, a + h);
            }
    check_linear(1, 0, 1, 40001);      // 40000!
    check_linear(2, 1, 0, 30000);      // (2k+1)!! style
    check_linear(6, -1, 1, 20001);
    check_linear(1, INT64_MAX, 5, 2005);
    for (unsigned j = 0; j < 300; ++j) {
        const uint64_t alpha = 1 + rnd() % (j % 3 ? 1000 : (uint64_t(1) << 50));
        const uint64_t a = rnd() % (j % 2 ? 100000 : (uint64_t(1) << 47));
        const uint64_t h = 1 + rnd() % 2500;
        const __uint128_t first = __uint128_t(alpha) * a;
        int64_t beta = int64_t(rnd() % 1000003);
        if (j % 4 == 0 && first > 3) {
            const __uint128_t room = first - 1 < __uint128_t(INT64_MAX) ? first - 1 : __uint128_t(INT64_MAX);
            beta = -int64_t(uint64_t(rnd() % uint64_t(room)) );
        }
        if (__uint128_t(alpha) * a + (beta > 0 ? beta : 0) < 1 && beta <= 0)
            beta = 1;
        check_linear(alpha, beta, a, a + h);
    }
    // Factor products: constants, powers, several factors; rejection of non-positive factors and polynomials.
    {
        FactorProduct f{};
        f.constant_low = 32;
        f.count = 2;
        f.factor[0] = {2, 1, 5};
        f.factor[1] = {6, -5, 2};
        LogBounds got{};
        assert(product_log2_sum(f, 1, 2001, got));
        ref_int product, factor;
        ref_inits(product, factor, nullptr);
        ref_set_ui(product, 1);
        for (uint64_t k = 1; k < 2001; ++k) {
            ref_mul_ui(product, product, 32);
            for (unsigned r = 0; r < 5; ++r)
                ref_mul_ui(product, product, 2 * k + 1);
            for (unsigned r = 0; r < 2; ++r)
                ref_mul_ui(product, product, 6 * k - 5);
        }
        long double slack = 0;
        const long double exact = log2_exact(product, slack);
        assert(got.lower <= exact + 1e-12L && exact - 1e-12L <= got.upper && got.upper - got.lower < 1e-3L);
        ref_clears(product, factor, nullptr);
        LogBounds at{};
        assert(product_log2_at(f, 10, at) && at.lower <= log2l(32.0L * powl(21, 5) * powl(55, 2)) + 1e-15L &&
               log2l(32.0L * powl(21, 5) * powl(55, 2)) - 1e-15L <= at.upper);
        f.factor[1] = {6, -7, 2}; // 6k-7 < 0 at k = 1
        assert(!product_log2_sum(f, 1, 100, got) && product_log2_sum(f, 2, 100, got));
        f.degree = 1;
        assert(!product_log2_sum(f, 2, 100, got));
        f.degree = 0;
        assert(product_log2_sum(f, 5, 5, got) && got.lower == 0 && got.upper == 0);
    }
    printf("log product estimator: %zu ranges against exact integer products, widest interval %.3Le bit, "
           "smallest margin %.3Le bit PASS\n",
           checked, worst_width, worst_margin);
}
