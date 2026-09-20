// Format service gate: dyadic numbers -> digits against exact integer arithmetic.
// EXACT: integer digits and floor(frac * b^D) must match exactly (including digit-boundary ties and
// the exact fallback). ENCLOSED: every certified digit is shared by the whole interval [X, X + 2^e).
#include "product_support.hpp"
#include "sbn3/radix.h"
#include <initializer_list>
#include <string>
#include <math.h>
namespace {
struct Pool {
    Fixture &f;
    size_t offset = 0, bytes = 0;
    Pool(Fixture &fixture, size_t size) : f(fixture), bytes(size) {
        auto block = f.allocate(size, size_t(1) << 21);
        offset = size_t(reinterpret_cast<uintptr_t>(block.data) - reinterpret_cast<uintptr_t>(f.arena->base));
        sbn3_arena_release(f.arena, &f.leases.back());
        f.leases.pop_back();
    }
};
struct Big {
    ref_int v;
    Big() { ref_init(v); }
    ~Big() { ref_clear(v); }
    Big(const Big &) = delete;
};
void power(ref_number *out, unsigned base, uint64_t e) {
    Big acc;
    ref_set_ui(out, 1);
    ref_set_ui(acc.v, base);
    for (; e; e >>= 1) {
        if (e & 1)
            ref_mul(out, out, acc.v);
        if (e > 1)
            ref_mul(acc.v, acc.v, acc.v);
    }
}
// digits of a nonnegative integer, MSD first, exactly `count` digits (value < base^count)
std::string digits_of(const ref_number *value, unsigned base, uint64_t count) {
    std::string out(count, '\0');
    Big x, q, d;
    ref_set(x.v, value);
    uint64_t b8 = 1;
    for (unsigned j = 0; j < 8; ++j)
        b8 *= base;
    ref_set_ui(d.v, b8);
    for (uint64_t pos = count; pos;) {
        uint64_t w = ref_fdiv_ui(x.v, b8);
        ref_fdiv_q(q.v, x.v, d.v);
        ref_set(x.v, q.v);
        for (unsigned j = 0; j < 8 && pos; ++j) {
            out[--pos] = char(w % base);
            w /= base;
        }
    }
    assert(ref_sgn(x.v) == 0);
    return out;
}
struct Expect {
    std::string integer, fraction; // exact digits of floor(X) (no leading zeros) and floor(frac * b^D)
    std::string upper;             // ENCLOSED: digits of ceil((frac + 2^-p) b^D) - 1
};
Expect reference(const std::vector<uint64_t> &m, int64_t exponent2, unsigned base, uint64_t digits, bool enclosed) {
    Expect e;
    Big M, I, F, P, t;
    ref_import(M.v, m.size(), -1, 8, 0, 0, m.data());
    const uint64_t p = exponent2 < 0 ? uint64_t(-exponent2) : 0;
    if (exponent2 >= 0)
        ref_mul_2exp(I.v, M.v, size_t(exponent2));
    else
        ref_fdiv_q_2exp(I.v, M.v, p);
    if (p)
        ref_fdiv_r_2exp(F.v, M.v, p);
    const size_t width = ref_sgn(I.v) ? size_t(double(ref_sizeinbase(I.v, 2)) / log2(double(base))) + 3 : 0;
    e.integer = digits_of(I.v, base, width);
    const size_t first = e.integer.find_first_not_of('\0');
    e.integer = first == std::string::npos ? std::string() : e.integer.substr(first);
    power(P.v, base, digits + 8);
    ref_mul(t.v, F.v, P.v);
    ref_fdiv_q_2exp(t.v, t.v, p);
    e.fraction = digits_of(t.v, base, digits + 8); // the last eight digits are the guard of the ENCLOSED rule
    power(P.v, base, digits);
    if (enclosed) {
        // ceil((F + 1) P / 2^p) - 1
        ref_add_ui(t.v, F.v, 1);
        ref_mul(t.v, t.v, P.v);
        Big one;
        ref_set_ui(one.v, 1);
        ref_mul_2exp(one.v, one.v, p);
        ref_add(t.v, t.v, one.v);
        ref_sub_ui(t.v, t.v, 1);
        ref_fdiv_q_2exp(t.v, t.v, p);
        ref_sub_ui(t.v, t.v, 1);
        // when the interval reaches 1 the fraction digits are all b-1 at best: cap
        ref_sub_ui(one.v, P.v, 1);
        if (ref_cmp(t.v, one.v) > 0)
            ref_set(t.v, one.v);
        e.upper = digits_of(t.v, base, digits);
    }
    return e;
}
struct Stats {
    uint64_t cases = 0, fallbacks = 0, ties = 0, enclosed = 0, shortened = 0;
} stats;
void check(Pool &pool, unsigned workers, unsigned base, const std::vector<uint64_t> &m, int64_t exponent2,
           uint64_t digits, sbn3_radix_mode mode, bool alphabet, int expect_fallback = -1, size_t budget = 0) {
    Fixture &f = pool.f;
    sbn3_format_spec spec{base, m.size(), exponent2, digits, mode};
    sbn3_radix_options options{};
    options.workers = workers;
    options.memory_budget = budget;
    options.repeated = unsigned(random_word() & 1); // the integer tree from 32 limbs, or from 256
    // Explicit fallback assertions exercise the tree recipe. The direct
    // one-use fraction recipe is exact and has no boundary fallback.
    if(expect_fallback>=0)options.repeated=1;
    unsigned char table[64];
    sbn3_radix_alphabet(base, table);
    if (alphabet) {
        options.use_alphabet = 1;
        memcpy(options.alphabet, table, 64);
    }
    sbn3_format_plan plan{};
    sbn3_format_info info{};
    allocation_watch_start();
    const auto rc = sbn3_format_query(&spec, &options, &plan, &info);
    assert(allocation_watch_stop() == 0);
    assert(rc == SBN3_SUPPORTED);
    assert(info.storage_bytes <= pool.bytes);
    assert(mode == SBN3_RADIX_ENCLOSED || info.fraction_digits == digits);
    sbn3_format_binding *binding = nullptr;
    allocation_watch_start();
    sbn3_format_bind(&plan, f.arena, pool.offset, f.team, &binding);
    assert(allocation_watch_stop() == 0);
    std::vector<unsigned char> out(info.digit_bytes + 64, 0xee);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_format_result result{};
        std::fill(out.begin(), out.end(), 0xee);
        allocation_watch_start();
        sbn3_format_execute(binding, {m.data(), m.size()}, out.data(), &result);
        assert(allocation_watch_stop() == 0);
        for (size_t j = info.digit_bytes; j < out.size(); ++j)
            assert(out[j] == 0xee);
        auto value = [&](size_t at) { // digit value of an output byte
            if (!alphabet)
                return unsigned(out[at]);
            for (unsigned v = 0; v < base; ++v)
                if (table[v] == out[at])
                    return v;
            assert(false);
            return 0u;
        };
        const auto e = reference(m, exponent2, base, info.fraction_digits, mode == SBN3_RADIX_ENCLOSED);
        // integer area
        assert(result.integer_first <= info.integer_digits);
        assert(info.integer_digits - result.integer_first == e.integer.size());
        for (size_t j = 0; j < e.integer.size(); ++j)
            assert(value(size_t(result.integer_first) + j) == unsigned(uint8_t(e.integer[j])));
        for (size_t j = 0; j < result.integer_first; ++j)
            assert(value(j) == 0);
        // fraction area
        assert(result.fraction_digits <= info.fraction_digits);
        if (mode == SBN3_RADIX_EXACT)
            assert(result.fraction_digits == digits);
        size_t common = 0;
        if (mode == SBN3_RADIX_ENCLOSED) {
            while (common < e.upper.size() && e.fraction[common] == e.upper[common])
                ++common;
            assert(result.fraction_digits <= common);
            ++stats.enclosed;
            if (result.fraction_digits < info.fraction_digits) {
                // only an exact guard of all b-1 may shorten the result
                for (size_t j = info.fraction_digits; j < info.fraction_digits + 8; ++j)
                    assert(unsigned(uint8_t(e.fraction[j])) == base - 1);
                ++stats.shortened;
            }
        }
        for (size_t j = 0; j < result.fraction_digits; ++j)
            assert(value(info.fraction_offset + j) == unsigned(uint8_t(e.fraction[j])));
        if (expect_fallback >= 0 && int(result.exact_fallback) != expect_fallback) {
            fprintf(stderr, "fallback %u, expected %d: base %u limbs %zu digits %llu workers %u\n", result.exact_fallback,
                    expect_fallback, base, m.size(), (unsigned long long)digits, workers);
            assert(false);
        }
        stats.fallbacks += result.exact_fallback;
        ++stats.cases;
    }
    allocation_watch_start();
    sbn3_format_unbind(binding);
    assert(allocation_watch_stop() == 0);
}
std::vector<uint64_t> random_limbs(size_t n) {
    std::vector<uint64_t> m(n);
    for (auto &w : m)
        w = random_word();
    return m;
}
// floor(T 2^p / b^t) + adjust as a p-bit fraction (limbs = ceil(p / 64)), T a random t digit string
std::vector<uint64_t> near_boundary(unsigned base, uint64_t t, size_t limbs, int adjust) {
    Big T, P, x;
    ref_set_ui(T.v, 1 + random_word() % (base - 1));
    for (uint64_t j = 1; j < t; ++j) {
        ref_mul_ui(T.v, T.v, base);
        ref_add_ui(T.v, T.v, random_word() % base);
    }
    if (ref_fdiv_ui(T.v, base) == 0)
        ref_add_ui(T.v, T.v, 1);
    power(P.v, base, t);
    ref_mul_2exp(x.v, T.v, 64 * limbs);
    ref_fdiv_q(x.v, x.v, P.v);
    if (adjust > 0)
        ref_add_ui(x.v, x.v, uint64_t(adjust));
    if (adjust < 0)
        ref_sub_ui(x.v, x.v, uint64_t(-adjust));
    std::vector<uint64_t> m(limbs, 0);
    size_t count = 0;
    ref_export(m.data(), &count, -1, 8, 0, 0, x.v);
    assert(count <= limbs);
    return m;
}
} // namespace
int main() {
    for (unsigned workers : {1u, 4u}) {
        Fixture f(workers, false);
        Pool pool(f, size_t(192) << 20);
        const unsigned bases[] = {2, 8, 16, 32, 64, 3, 7, 10, 12, 36, 62, 63};
        for (unsigned base : bases) {
            for (size_t limbs : {size_t(1), size_t(2), size_t(5), size_t(31), size_t(33), size_t(40), size_t(100), size_t(300)}) {
                const int64_t width = int64_t(64) * int64_t(limbs);
                const int64_t exponents[] = {0, 5, 70, -width, -width / 2, -(width / 2) - 13, -width - 100, -1, -width + 3};
                for (int64_t e : exponents) {
                    if (limbs > 40 && (e == 5 || e == -1))
                        continue;
                    const uint64_t wanted[] = {0, 1, 7, 64, 65, 100, 1000};
                    const uint64_t digits = wanted[random_word() % 7];
                    auto m = random_limbs(limbs);
                    check(pool, workers, base, m, e, digits, SBN3_RADIX_EXACT, base == 36 || base == 62);
                    if (e <= 0)
                        check(pool, workers, base, m, e, digits ? digits : 50, SBN3_RADIX_ENCLOSED, false);
                }
            }
            // structured values: zero, all ones, exact short fractions, one bit
            for (size_t limbs : {size_t(1), size_t(8), size_t(64)}) {
                const int64_t e = -int64_t(64) * int64_t(limbs);
                std::vector<uint64_t> zero(limbs, 0), ones(limbs, ~uint64_t(0)), half(limbs, 0), bit(limbs, 0);
                half[limbs - 1] = uint64_t(3) << 61; // 0.375
                bit[0] = 1;
                for (const auto &m : {zero, ones, half, bit}) {
                    check(pool, workers, base, m, e, 200, SBN3_RADIX_EXACT, false);
                    check(pool, workers, base, m, e, 200, SBN3_RADIX_ENCLOSED, false);
                    check(pool, workers, base, m, 0, 3, SBN3_RADIX_EXACT, false);
                }
            }
        }
        // digit-boundary ties of the fraction: x just below / at / above T / b^t, many more bits than digits
        for (unsigned base : {3u, 10u, 12u, 63u}) {
            for (uint64_t t : {uint64_t(1), uint64_t(5), uint64_t(64), uint64_t(100), uint64_t(200)}) {
                for (uint64_t digits : {t, t + 1, t + 30, uint64_t(64) * ((t + 63) / 64), uint64_t(300)}) {
                    if (digits < t)
                        continue;
                    const size_t limbs = size_t((digits + 80) * 6 / 64 + 8);
                    const int64_t e = -int64_t(64) * int64_t(limbs);
                    // floor(T 2^p / b^t): just below the boundary unless the quotient is exact (even bases, small t)
                    check(pool, workers, base, near_boundary(base, t, limbs, 0), e, digits, SBN3_RADIX_EXACT, false);
                    check(pool, workers, base, near_boundary(base, t, limbs, -1), e, digits, SBN3_RADIX_EXACT, false, 1);
                    check(pool, workers, base, near_boundary(base, t, limbs, 1), e, digits, SBN3_RADIX_EXACT, false);
                    check(pool, workers, base, near_boundary(base, t, limbs, 0), e, digits, SBN3_RADIX_ENCLOSED, false);
                    check(pool, workers, base, near_boundary(base, t, limbs, 1), e, digits, SBN3_RADIX_ENCLOSED, false);
                    ++stats.ties;
                }
            }
        }
        // larger trees, integer and fraction, with the team
        // Small trees need no huge-page-aligned empty pool. A sub-2-MiB
        // caller budget must admit and execute these ordinary conversions.
        for (size_t n : {129u,181u,255u})
            check(pool, workers, 10, random_limbs(n), 0, 0, SBN3_RADIX_EXACT, false, -1, size_t(1)<<20);
        for (unsigned base : {10u, 7u}) {
            check(pool, workers, base, random_limbs(3000), 0, 0, SBN3_RADIX_EXACT, false);
            check(pool, workers, base, random_limbs(3000), -int64_t(64) * 3000, 57000, SBN3_RADIX_EXACT, false);
            check(pool, workers, base, random_limbs(3000), -int64_t(64) * 2000 - 17, 38000, SBN3_RADIX_ENCLOSED, base == 10);
        }
    }
    printf("radix format: %llu conversions (%llu exact fallbacks, %llu tie groups, %llu enclosed of which %llu shortened) "
           "against exact arithmetic PASS\n",
           (unsigned long long)stats.cases, (unsigned long long)stats.fallbacks, (unsigned long long)stats.ties,
           (unsigned long long)stats.enclosed, (unsigned long long)stats.shortened);
}
