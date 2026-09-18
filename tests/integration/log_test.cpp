#include "product_support.hpp"
#include "sbn3/log.h"
#include "series/scaled_ratio.hpp"
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace {
uint64_t absolute(int64_t x) { return x < 0 ? uint64_t(-(x + 1)) + 1 : uint64_t(x); }
// Independent fixed-point summation. Each power is rounded down after
// multiplication by a^2/b^2<=1/4. Power error<4/3 units; each term error<3
// units and the uncomputed tail after power==0 is <2 units.
uint64_t atanh_reference(ref_number *out, uint64_t a, uint64_t b, size_t words) {
    ref_set_ui(out, 0);
    if (!a)
        return 0;
    assert(__uint128_t(a) * 2 <= b);
    ref_int power, aa, bb, den, term;
    ref_inits(power, aa, bb, den, term, nullptr);
    ref_set_ui(aa, a);
    ref_mul(aa, aa, aa);
    ref_set_ui(bb, b);
    ref_mul(bb, bb, bb);
    ref_set_ui(power, a);
    ref_mul_2exp(power, power, 64 * words);
    ref_set_ui(den, b);
    ref_fdiv_q(power, power, den);
    uint64_t count = 0;
    while (ref_sgn(power)) {
        ref_set_ui(den, 2 * count + 1);
        ref_fdiv_q(term, power, den);
        ref_add(out, out, term);
        ref_mul(power, power, aa);
        ref_fdiv_q(power, power, bb);
        ++count;
    }
    ref_clears(power, aa, bb, den, term, nullptr);
    return 3 * count + 2;
}
void set128(ref_number *out, __uint128_t x) {
    const uint64_t w[]{uint64_t(x), uint64_t(x >> 64)};
    ref_import(out, 2, -1, 8, 0, 0, w);
}
void round_reference(ref_number *out, ref_number *value, __uint128_t error) {
    ref_int lo, hi, e;
    ref_inits(lo, hi, e, nullptr);
    set128(e, error);
    ref_sub(lo, value, e);
    ref_add(hi, value, e);
    ref_fdiv_q_2exp(lo, lo, 256);
    ref_fdiv_q_2exp(hi, hi, 256);
    assert(ref_cmp(lo, hi) == 0);
    ref_set(out, lo);
    ref_clears(lo, hi, e, nullptr);
}
void expected_log(ref_number *out, uint32_t n, size_t words) {
    if (n == 1) {
        ref_set_ui(out, 0);
        return;
    }
    const unsigned power = 31 - __builtin_clz(n);
    const uint64_t base = uint64_t(1) << power;
    ref_int a, b;
    ref_inits(a, b, nullptr);
    const auto ea = atanh_reference(a, 1, 3, words + 4),
               eb = atanh_reference(b, n - base, n + base, words + 4);
    ref_mul_ui(a, a, power);
    ref_add(a, a, b);
    ref_mul_ui(a, a, 2);
    round_reference(out, a, 2 * (__uint128_t(power) * ea + eb) + 1);
    ref_clears(a, b, nullptr);
}
void expected_atanh(ref_number *out, uint64_t numerator, uint64_t denominator, size_t words) {
    ref_int a;
    ref_init(a);
    const auto error = atanh_reference(a, numerator, denominator, words + 4);
    round_reference(out, a, error);
    ref_clear(a);
}
void expected_arccoth(ref_number *out, uint64_t n, size_t words) { expected_atanh(out, 1, n, words); }
void power_word(ref_number *out, uint64_t n, uint64_t exponent) {
    ref_int b;
    ref_init(b);
    ref_set_ui(b, n);
    ref_set_ui(out, 1);
    while (exponent) {
        if (exponent & 1)
            ref_mul(out, out, b);
        exponent >>= 1;
        if (exponent)
            ref_mul(b, b, b);
    }
    ref_clear(b);
}
void identity(uint32_t n) {
    sbn3_log_formula f{}, again{};
    allocation_watch_start();
    assert(sbn3_log_formula_query(n, &f) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    assert(sbn3_log_formula_query(n, &again) == SBN3_SUPPORTED && again.identity == f.identity);
    assert(f.count <= SBN3_LOG_MAX_TERMS && f.divisor);
    ref_int num, den, a, b, target;
    ref_inits(num, den, a, b, target, nullptr);
    ref_set_ui(num, 1);
    ref_set_ui(den, 1);
    for (unsigned j = 0; j < f.count; ++j) {
        const auto &t = f.term[j];
        assert(t.argument >= 2 && t.coefficient && t.numerator >= 1 && 2 * t.numerator <= t.argument &&
               std::gcd(t.numerator, t.argument) == 1 &&
               (!j || f.term[j - 1].argument < t.argument ||
                (f.term[j - 1].argument == t.argument && f.term[j - 1].numerator < t.numerator)));
        // exp(2 atanh(a/b)) = (b+a)/(b-a)
        power_word(a, t.argument + t.numerator, absolute(t.coefficient));
        power_word(b, t.argument - t.numerator, absolute(t.coefficient));
        if (f.term[j].coefficient > 0) {
            ref_mul(num, num, a);
            ref_mul(den, den, b);
        } else {
            ref_mul(num, num, b);
            ref_mul(den, den, a);
        }
    }
    power_word(target, n, f.divisor);
    ref_mul(target, target, den);
    assert(ref_cmp(num, target) == 0);
    ref_clears(num, den, a, b, target, nullptr);
}
void compare(const uint64_t *data, size_t count, const ref_number *want) {
    ref_int got;
    ref_init(got);
    ref_import(got, count, -1, 8, 0, 0, data);
    assert(ref_cmp(got, want) == 0);
    ref_clear(got);
}
void arccoth(uint64_t m, size_t n, unsigned workers, unsigned leaves = 8) {
    Fixture fixture(workers);
    auto object = fixture.allocate(sbn3_formula_object_bytes(), 64);
    sbn3_formula_plan plan{};
    sbn3_formula_info info{};
    const sbn3_formula_options options{{workers, leaves, 0, 0, 0}, 0, 0, 0};
    allocation_watch_start();
    const auto rc = sbn3_arccoth_query(m, n, &options, object.data, &plan, &info);
    assert(!allocation_watch_stop());
    if (rc != SBN3_SUPPORTED)
        fprintf(stderr, "ArcCoth(%llu) query %u: %s\n", (unsigned long long)m, rc,
                info.rejection ? info.rejection : "");
    assert(rc == SBN3_SUPPORTED);
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    ref_int want;
    ref_init(want);
    expected_arccoth(want, m, n);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_formula_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_formula_bind(&plan, object.data, fixture.arena, offset, fixture.team, &binding);
        const auto result = sbn3_formula_execute_inplace(binding);
        assert(!allocation_watch_stop());
        compare(result.data, result.count, want);
        allocation_watch_start();
        sbn3_formula_unbind(binding);
        assert(!allocation_watch_stop());
    }
    ref_clear(want);
}
// One explicit series through the generic formula service. With a reference the
// full output must equal the independent fixed-point Taylor summation; the
// returned limbs let two different series be compared at sizes beyond the oracle.
unsigned atanh_value(uint64_t numerator, uint64_t m, sbn3_arccoth_series series, size_t n, unsigned workers,
                     bool reference, std::vector<uint64_t> *out = nullptr, unsigned leaves = 8) {
    Fixture fixture(workers);
    auto object = fixture.allocate(sbn3_formula_object_bytes(), 64);
    sbn3_formula_def def{};
    assert(sbn3_atanh_series_definition(numerator, m, series, &def) == SBN3_SUPPORTED);
    sbn3_formula_plan plan{};
    sbn3_formula_info info{};
    const sbn3_formula_options options{{workers, leaves, 0, 0, 0}, 0, 0, 0}; // leaves 0: service policy
    const auto rc = sbn3_formula_query(&def, n, &options, object.data, &plan, &info);
    if (rc != SBN3_SUPPORTED)
        fprintf(stderr, "atanh(%llu/%llu) series %u query %u: %s\n", (unsigned long long)numerator,
                (unsigned long long)m, unsigned(series), rc, info.rejection ? info.rejection : "");
    assert(rc == SBN3_SUPPORTED);
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    sbn3_formula_binding *binding = nullptr;
    allocation_watch_start();
    sbn3_formula_bind(&plan, object.data, fixture.arena, offset, fixture.team, &binding);
    const auto result = sbn3_formula_execute_inplace(binding);
    assert(!allocation_watch_stop());
    assert(result.count == n + 1);
    if (reference) {
        ref_int want;
        ref_init(want);
        expected_atanh(want, numerator, m, n);
        compare(result.data, result.count, want);
        ref_clear(want);
    }
    if (out)
        out->assign(result.data, result.data + result.count);
    sbn3_formula_unbind(binding);
    return info.blocks;
}
unsigned series_value(uint64_t m, sbn3_arccoth_series series, size_t n, unsigned workers, bool reference,
                      std::vector<uint64_t> *out = nullptr, unsigned leaves = 8) {
    return atanh_value(1, m, series, n, workers, reference, out, leaves);
}
// Rational arguments: the generalized series data, its domain, full values of
// both series against the independent reference, the two series against each
// other beyond the oracle's reach, and the relations that use them.
void rational_series() {
    using u128 = __uint128_t;
    auto gcd128 = [](u128 a, u128 b) {
        while (b) {
            const u128 t = a % b;
            a = b;
            b = t;
        }
        return a;
    };
    const uint64_t pairs[][2]{{3, 253}, {11, 6261}, {13, 499}, {34, 16841}, {13, 1037}, {73, 31177},
                              {11, 501}, {7, 493},  {17, 10223}, {19, 3106}, {2, 23},  {1, 2},
                              {2, 5},   {3, 7},    {5, 12},     {1, 3},     {32767, 65536}};
    for (const auto &pair : pairs) {
        const uint64_t a = pair[0], b = pair[1];
        sbn3_formula_def fast{}, plain{}, automatic{};
        assert(sbn3_atanh_series_definition(a, b, SBN3_ARCCOTH_TAYLOR, &plain) == SBN3_SUPPORTED);
        const u128 A = u128(a) * a, B = u128(b) * b;
        if (a != 1 || (b & (b - 1))) {
            assert(plain.recipe == SBN3_SERIES_COMMON_P2B3 &&
                   ((u128(plain.P.constant_high) << 64) | plain.P.constant_low) == u128(a) * b &&
                   ((u128(plain.Q.constant_high) << 64) | plain.Q.constant_low) == B &&
                   ((u128(plain.R.constant_high) << 64) | plain.R.constant_low) == A);
        }
        // Independent statement of the accelerated data and of its representability.
        const u128 q = 3 * B * (B - A) * (B - A), r = 8 * A * A * A, lin = 8 * (9 * B * B - 15 * A * B + 4 * A * A),
                   con = 4 * B * (3 * B - 5 * A), g = gcd128(gcd128(q, r), gcd128(lin, con));
        const bool representable = lin / g <= u128(INT64_MAX) && con / g <= u128(INT64_MAX);
        const bool ok = sbn3_atanh_series_definition(a, b, SBN3_ARCCOTH_ACCELERATED, &fast) == SBN3_SUPPORTED;
        assert(ok == representable);
        assert(sbn3_atanh_series_definition(a, b, SBN3_ARCCOTH_AUTO, &automatic) == SBN3_SUPPORTED &&
               !memcmp(&automatic, ok ? &fast : &plain, sizeof automatic));
        if (ok)
            assert(fast.begin == 1 && fast.denominator_exponent == 2 && fast.P.degree == 1 &&
                   ((u128(fast.P.constant_high) << 64) | fast.P.constant_low) == u128(a) * b &&
                   u128(fast.P.coefficient[1]) * g == lin && u128(-fast.P.coefficient[0]) * g == con &&
                   ((u128(fast.Q.constant_high) << 64) | fast.Q.constant_low) * g == q &&
                   ((u128(fast.R.constant_high) << 64) | fast.R.constant_low) * g == r);
        for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(17), size_t(64)}) {
            if (ok)
                atanh_value(a, b, SBN3_ARCCOTH_ACCELERATED, n, n == 17 ? 3 : 1, true);
            if (n == 17 || !ok)
                atanh_value(a, b, SBN3_ARCCOTH_TAYLOR, n, n == 17 ? 16 : 1, true);
        }
    }
    sbn3_formula_def untouched{};
    for (const auto &bad : {std::pair<uint64_t, uint64_t>{0, 5}, {3, 5}, {2, 4}, {6, 15}, {1, 1}, {5, 0}})
        for (auto series : {SBN3_ARCCOTH_AUTO, SBN3_ARCCOTH_TAYLOR, SBN3_ARCCOTH_ACCELERATED})
            assert(sbn3_atanh_series_definition(bad.first, bad.second, series, &untouched) == SBN3_UNSUPPORTED);
    assert(sbn3_atanh_series_definition(3, 65537, SBN3_ARCCOTH_ACCELERATED, &untouched) == SBN3_UNSUPPORTED &&
           sbn3_atanh_series_definition(3, 65537, SBN3_ARCCOTH_AUTO, &untouched) == SBN3_SUPPORTED);
    atanh_value(3, 65537, SBN3_ARCCOTH_AUTO, 3, 3, true);
    atanh_value(3, 253, SBN3_ARCCOTH_ACCELERATED, 257, 16, true);
    for (const auto &pair : {std::pair<uint64_t, uint64_t>{3, 253}, {73, 31177}}) {
        std::vector<uint64_t> x, y;
        const unsigned blocks = atanh_value(pair.first, pair.second, SBN3_ARCCOTH_ACCELERATED, 300000, 16, false, &x, 0);
        atanh_value(pair.first, pair.second, SBN3_ARCCOTH_TAYLOR, 300000, 16, false, &y);
        assert(blocks > 1 && x == y);
    }
    // The catalog's effect, pinned: log 10 = 3 L(3/253) + 10 L(1/9) and
    // log 3 = L(73/31177) + 6 L(1/11), L(x) = 2 atanh(x); log 2 and log 7 keep
    // their integer relations.
    sbn3_log_formula f{};
    assert(sbn3_log_formula_query(10, &f) == SBN3_SUPPORTED && f.count == 2 && f.divisor == 1 &&
           f.term[0].argument == 9 && f.term[0].numerator == 1 && f.term[0].coefficient == 10 &&
           f.term[1].argument == 253 && f.term[1].numerator == 3 && f.term[1].coefficient == 3);
    assert(sbn3_log_formula_query(3, &f) == SBN3_SUPPORTED && f.count == 2 && f.divisor == 1 &&
           f.term[0].argument == 11 && f.term[0].numerator == 1 && f.term[0].coefficient == 6 &&
           f.term[1].argument == 31177 && f.term[1].numerator == 73 && f.term[1].coefficient == 1);
    assert(sbn3_log_formula_query(2, &f) == SBN3_SUPPORTED && f.count == 1 && f.term[0].argument == 3 &&
           f.term[0].numerator == 1);
    assert(sbn3_log_formula_query(7, &f) == SBN3_SUPPORTED && f.count == 3);
    for (unsigned j = 0; j < f.count; ++j)
        assert(f.term[j].numerator == 1);
}
void accelerated_series() {
    // Independent statement of the representability rule: the linear coefficient
    // of P, 8(9x^2-15x+4)/gcd(8,3x(x-1)^2), must fit int64.
    auto representable = [](uint64_t m) {
        if (m < 2 || m > 65536)
            return false;
        const __uint128_t x = __uint128_t(m) * m, q = 3 * x * (x - 1) * (x - 1);
        const unsigned g = q % 8 == 0 ? 8 : q % 4 == 0 ? 4 : q % 2 == 0 ? 2 : 1;
        return 8 * (9 * x * x - 15 * x + 4) / g <= __uint128_t(INT64_MAX);
    };
    uint64_t last = 0, last_even2 = 0;
    for (uint64_t m = 2; m <= 70000; ++m) {
        sbn3_formula_def fast{}, automatic{}, plain{};
        const bool ok = sbn3_arccoth_series_definition(m, SBN3_ARCCOTH_ACCELERATED, &fast) == SBN3_SUPPORTED;
        assert(ok == representable(m));
        assert(sbn3_arccoth_series_definition(m, SBN3_ARCCOTH_TAYLOR, &plain) == SBN3_SUPPORTED &&
               sbn3_arccoth_definition(m, &automatic) == SBN3_SUPPORTED);
        assert(!memcmp(&automatic, ok ? &fast : &plain, sizeof automatic));
        if (ok) {
            // The data really is the documented P, Q, R divided by their common power of two.
            const __uint128_t x = __uint128_t(m) * m, q = 3 * x * (x - 1) * (x - 1);
            const unsigned g = unsigned(8 / fast.R.constant_low);
            assert(q % 4 == 0 && g == (q % 8 == 0 ? 8u : 4u) && fast.R.constant_low * g == 8);
            assert(fast.begin == 1 && fast.denominator_exponent == 2 && fast.P.constant_low == m &&
                   fast.P.degree == 1 && __uint128_t(fast.P.coefficient[1]) * g == 8 * (9 * x * x - 15 * x + 4) &&
                   __uint128_t(-fast.P.coefficient[0]) * g == 4 * x * (3 * x - 5) &&
                   ((__uint128_t(fast.Q.constant_high) << 64) | fast.Q.constant_low) * g == q);
            last = m;
            if (m % 4 == 2)
                last_even2 = m;
        }
    }
    assert(last == 31817 && last_even2 == 26754);
    sbn3_formula_def untouched{};
    assert(sbn3_arccoth_series_definition(3, sbn3_arccoth_series(3), &untouched) == SBN3_UNSUPPORTED);
    // Full outputs against the independent reference: both series, the two
    // gcd classes, powers of two, and both representability boundaries.
    for (uint64_t m : {uint64_t(2), uint64_t(3), uint64_t(4), uint64_t(5), uint64_t(6), uint64_t(7), uint64_t(8),
                       uint64_t(10), uint64_t(26), uint64_t(97), uint64_t(1024), uint64_t(4801), uint64_t(8749),
                       uint64_t(26754), uint64_t(31817)})
        for (size_t n : {size_t(1), size_t(2), size_t(3), size_t(17), size_t(64)}) {
            series_value(m, SBN3_ARCCOTH_ACCELERATED, n, n == 17 ? 3 : 1, true);
            if (n == 17)
                series_value(m, SBN3_ARCCOTH_TAYLOR, n, 16, true);
        }
    for (uint64_t m : {uint64_t(26758), uint64_t(31818), uint64_t(65537)}) { // fall back to Taylor
        sbn3_formula_def fast{};
        assert(sbn3_arccoth_series_definition(m, SBN3_ARCCOTH_ACCELERATED, &fast) == SBN3_UNSUPPORTED);
        arccoth(m, 3, 3);
    }
    series_value(26, SBN3_ARCCOTH_ACCELERATED, 257, 16, true);
    // Wide batch leaves (two and three words per term) at the policy batch and
    // at the largest one; one-word batches for comparison (m = 3).
    for (uint64_t m : {uint64_t(3), uint64_t(26), uint64_t(8749), uint64_t(31817)})
        for (unsigned leaves : {0u, 64u, 1u}) {
            series_value(m, SBN3_ARCCOTH_ACCELERATED, 64, 3, true, nullptr, leaves);
            series_value(m, SBN3_ARCCOTH_ACCELERATED, 5, 1, true, nullptr, leaves);
        }
    // Beyond the oracle's reach the two series check each other: complete
    // outputs must agree bit for bit, through several limited blocks.
    for (uint64_t m : {uint64_t(3), uint64_t(26), uint64_t(8749)}) {
        std::vector<uint64_t> a, b;
        const unsigned blocks = series_value(m, SBN3_ARCCOTH_ACCELERATED, 300000, 16, false, &a);
        series_value(m, SBN3_ARCCOTH_TAYLOR, 300000, 16, false, &b);
        assert(blocks > 1 && a == b);
        series_value(m, SBN3_ARCCOTH_ACCELERATED, 300000, 16, false, &b, 0); // policy leaf batch, policy blocks
        assert(a == b);
    }
}
void logarithm(uint32_t value, size_t n, unsigned workers) {
    Fixture fixture(workers);
    auto object = fixture.allocate(sbn3_log_object_bytes(value), 64);
    sbn3_formula_sum_plan plan{};
    sbn3_formula_sum_info info{}, again{};
    sbn3_formula_options options{{workers, 8, 0, 0, 0}, 0, 0, 0};
    allocation_watch_start();
    auto rc = sbn3_log_query(value, n, &options, object.data, object.bytes, &plan, &info);
    assert(!allocation_watch_stop());
    if (rc != SBN3_SUPPORTED)
        fprintf(stderr, "Log(%u) query %u: %s\n", value, rc, info.rejection ? info.rejection : "");
    assert(rc == SBN3_SUPPORTED);
    const auto saved = plan;
    options.memory_budget = info.object_bytes + info.storage_bytes - 1;
    assert(sbn3_log_query(value, n, &options, object.data, object.bytes, &plan, &again) ==
               SBN3_QUERY_CAPACITY &&
           again.storage_bytes == info.storage_bytes && !memcmp(&plan, &saved, sizeof plan));
    options.memory_budget = 0;
    assert(sbn3_log_query(value, n, &options, object.data, object.bytes, &plan, &again) == SBN3_SUPPORTED &&
           again.plan_id == info.plan_id);
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    fixture.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    auto output = fixture.allocate(info.output_limbs * 8, 64);
    ref_int want;
    ref_init(want);
    expected_log(want, value, n);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_formula_sum_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_formula_sum_bind(&plan, object.data, fixture.arena, offset, fixture.team, &binding);
        const uint64_t *data = nullptr;
        if (repeat)
            data = sbn3_formula_sum_execute_inplace(binding).data;
        else {
            sbn3_formula_sum_execute(binding, {static_cast<uint64_t *>(output.data), info.output_limbs});
            data = static_cast<uint64_t *>(output.data);
        }
        assert(!allocation_watch_stop());
        compare(data, info.output_limbs, want);
        allocation_watch_start();
        sbn3_formula_sum_unbind(binding);
        assert(!allocation_watch_stop());
    }
    ref_clear(want);
}
void combinations() {
    for (unsigned mode = 0; mode < 6; ++mode) {
        Fixture fixture(1);
        sbn3_formula_sum_term terms[2]{};
        uint64_t divisor = 1;
        assert(sbn3_arccoth_definition(3, &terms[0].formula) == SBN3_SUPPORTED);
        terms[1].formula = terms[0].formula;
        terms[0].coefficient = 7;
        terms[1].coefficient = -5;
        if (mode == 1) {
            terms[1].coefficient = -7;
        }
        if (mode == 2) {
            terms[0].coefficient = 11;
            divisor = 3;
        }
        if (mode == 3) {
            assert(sbn3_arccoth_definition(uint64_t(1) << 62, &terms[0].formula) == SBN3_SUPPORTED);
            assert(sbn3_arccoth_definition(uint64_t(1) << 63, &terms[1].formula) == SBN3_SUPPORTED);
            terms[0].coefficient = INT64_MAX;
            terms[1].coefficient = INT64_MIN;
        }
        if (mode == 4) {
            terms[0] = {};
            terms[0].formula.recipe = SBN3_SERIES_HYPERDESCENT;
            terms[0].formula.begin = 1;
            terms[0].formula.P.constant_low = terms[0].formula.R.constant_low = 1;
            terms[0].formula.Q.constant_low = 2; // sum 2^-k = exactly 1
            terms[0].coefficient = 1;
            terms[1].coefficient = 2;
        }
        if (mode == 5) {
            terms[0].coefficient = 1;
            terms[1].coefficient = 0;
            divisor = UINT64_MAX;
        }
        auto object = fixture.allocate(sbn3_formula_sum_object_bytes(2), 64);
        sbn3_formula_sum_plan plan{};
        sbn3_formula_sum_info info{};
        const sbn3_formula_options options{{1, 8, 0, 0, 0}, 0, 0, 0};
        assert(sbn3_formula_sum_query(terms, 2, divisor, 2, &options, object.data, object.bytes, &plan,
                                      &info) == SBN3_SUPPORTED);
        assert(info.components == (mode == 1 ? 0u : (mode == 3 || mode == 4) ? 2u : 1u));
        const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
        sbn3_error error{};
        assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
        sbn3_formula_sum_binding *binding = nullptr;
        allocation_watch_start();
        sbn3_formula_sum_bind(&plan, object.data, fixture.arena, offset, fixture.team, &binding);
        const auto result = sbn3_formula_sum_execute_inplace(binding);
        assert(!allocation_watch_stop());
        if (mode == 1)
            assert(result.data[0] == 0 && result.data[1] == 0 && result.data[2] == 0);
        else if (mode == 3)
            assert(result.data[0] == 9 && result.data[1] == UINT64_MAX - 3 && result.data[2] == 0);
        else {
            ref_int want;
            ref_init(want);
            if (mode == 5) {
                ref_int value, den;
                ref_inits(value, den, nullptr);
                const auto error = atanh_reference(value, 1, 3, 6);
                ref_set_ui(den, UINT64_MAX);
                ref_fdiv_q(value, value, den);
                round_reference(want, value, error + 1);
                ref_clears(value, den, nullptr);
            } else {
                expected_log(want, 2, 2);
                if (mode == 4) {
                    ref_int integer;
                    ref_init(integer);
                    ref_set_ui(integer, 1);
                    ref_mul_2exp(integer, integer, 128);
                    ref_add(want, want, integer);
                    ref_clear(integer);
                }
            }
            compare(result.data, result.count, want);
            ref_clear(want);
        }
        sbn3_formula_sum_unbind(binding);
    }
}
void zero_guard() {
    uint64_t safe[]{0, 0};
    sbn::v3::series::guard_separated(safe, 2, 16, true);
    const pid_t pid = fork();
    assert(pid >= 0);
    if (!pid) {
        rlimit no_core{0, 0};
        setrlimit(RLIMIT_CORE, &no_core);
        uint64_t near_next[]{UINT64_MAX - 7, UINT64_MAX};
        sbn::v3::series::guard_separated(near_next, 2, 16, true);
        _exit(0);
    }
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid && WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}
} // namespace
int main() {
    sbn3_log_formula invalid{};
    assert(sbn3_log_formula_query(0, &invalid) == SBN3_UNSUPPORTED);
    sbn3_formula_def invalid_arc{};
    assert(sbn3_arccoth_definition(0, &invalid_arc) == SBN3_UNSUPPORTED);
    assert(sbn3_arccoth_definition(1, &invalid_arc) == SBN3_UNSUPPORTED);
    for (uint32_t n = 1; n <= 128; ++n)
        identity(n);
    for (uint32_t n : {65537u, 1000000007u, UINT32_MAX})
        identity(n);
    zero_guard();
    for (uint64_t m : {uint64_t(2), uint64_t(3), uint64_t(26), uint64_t(4801), UINT64_MAX, uint64_t(1) << 63})
        arccoth(m, 1, 1);
    arccoth(3, 3, 3);
    arccoth(26, 17, 16);
    arccoth(2, 64, 1);
    arccoth(3, 17, 3, 64); // exact short batch followed by a limited output
    accelerated_series();
    rational_series();
    for (uint32_t n : {1u, 2u, 3u, 5u, 6u, 7u, 10u, 11u, 35u, 64u, 210u, UINT32_MAX})
        logarithm(n, 1, 1);
    logarithm(2, 3, 3);
    logarithm(3, 17, 16);
    logarithm(65537, 4, 3);
    logarithm(5, 64, 1);
    logarithm(35, 9, 3);
    logarithm(2, 257, 16);
    logarithm(10, 257, 16);
    combinations();
    puts("ArcCoth/atanh/Log: exact identities, independent full-value references, Taylor/accelerated series "
         "cross-checks and representability bounds, rational arguments and their relations, wide "
         "arguments/coefficient guards, zero, capacities, threads, rebind and no allocation PASS");
}
