#include "product_support.hpp"
#include "sbn3/bitwindow.h"
#include "verify/bbp_terms.hpp"
#include "verify/word_montgomery.hpp"
#include <algorithm>

using namespace sbn::v3;
namespace {
template <unsigned B> struct ScalarLane {
    using V = uint64_t;
    using M = unsigned;
    static constexpr unsigned bits = B;
    static V splat(uint64_t x) { return x; }
    static V zero() { return 0; }
    static V one() { return 1; }
    static V add(V a, V b) { return a + b; }
    static V sub(V a, V b) { return a - b; }
    static V bit_xor(V a, V b) { return a ^ b; }
    static V digit(V x) { return x & ((uint64_t(1) << B) - 1); }
    static V carry(V x) { return x >> B; }
    static V lo(V a, V b) { return digit(a * b); }
    static V hi(V a, V b) { return uint64_t((__uint128_t(a) * b) >> B); }
    static M lt(V a, V b) { return a < b; }
    static M nonzero(V x) { return x != 0; }
    static M mask_not(M x) { return !x; }
    static M bit(V x, unsigned j) { return (x >> j) & 1; }
    static V select(M m, V yes, V no) { return m ? yes : no; }
};
template <unsigned B, unsigned L> void to_ref(ref_number *r, const uint64_t (&v)[L]) {
    ref_set_ui(r, 0);
    for (unsigned j = L; j--;) {
        ref_mul_2exp(r, r, B);
        ref_add_ui(r, r, v[j]);
    }
}
template <unsigned B, unsigned L> void montgomery() {
    using O = ScalarLane<B>;
    ref_int m, a, b, r, want;
    ref_inits(m, a, b, r, want, nullptr);
    for (unsigned trial = 0; trial < 100; ++trial) {
        uint64_t mod[L], x[L], y[L], got[L];
        for (unsigned j = 0; j < L; ++j)
            mod[j] = trial == 0 ? O::digit(UINT64_MAX) : O::digit(random_word());
        mod[0] |= 3;
        mod[L - 1] |= uint64_t(1) << (B - 1);
        to_ref<B>(m, mod);
        for (unsigned j = 0; j < L; ++j)
            x[j] = trial < 2 ? mod[j] : O::digit(random_word());
        if (trial < 2)
            --x[0];
        to_ref<B>(a, x);
        ref_mod(a, a, m);
        ref_set(r, a);
        for (unsigned j = 0; j < L; ++j) {
            x[j] = O::digit(ref_get_ui(r));
            ref_fdiv_q_2exp(r, r, B);
        }
        for (unsigned j = 0; j < L; ++j)
            y[j] = trial < 2 ? x[j] : O::digit(random_word());
        to_ref<B>(b, y);
        ref_mod(b, b, m);
        ref_set(r, b);
        for (unsigned j = 0; j < L; ++j) {
            y[j] = O::digit(ref_get_ui(r));
            ref_fdiv_q_2exp(r, r, B);
        }
        const bbp::WordMontgomery<O, L> mont(mod);
        assert(O::lo(mod[0], mont.inverse) == O::digit(UINT64_MAX));
        mont.multiply(got, x, y);
        to_ref<B>(r, got);
        assert(ref_cmp(r, m) < 0);
        ref_mul_2exp(r, r, B * L);
        ref_mod(r, r, m);
        ref_mul(want, a, b);
        ref_mod(want, want, m);
        assert(!ref_cmp(r, want));
        mont.multiply(x, x, x); // Exact alias and dedicated square call shape.
        to_ref<B>(r, x);
        assert(ref_cmp(r, m) < 0);
        ref_mul_2exp(r, r, B * L);
        ref_mod(r, r, m);
        ref_mul(want, a, a);
        ref_mod(want, want, m);
        assert(!ref_cmp(r, want));
    }
    ref_clears(m, a, b, r, want, nullptr);
}

bbp::Fixed reference(const bbp::RationalTerm &term, unsigned digits) {
    ref_int m, r, a, n;
    ref_inits(m, r, a, n, nullptr);
    ref_import(m, bbp::modulus_words, -1, 8, 0, 0, term.modulus);
    if (term.exponent >= 0) {
        ref_set_ui(r, 1);
        ref_mod(r, r, m);
        ref_set_ui(a, 2);
        ref_mod(a, a, m);
        for (uint64_t e = uint64_t(term.exponent); e; e >>= 1) {
            if (e & 1) {
                ref_mul(r, r, a);
                ref_mod(r, r, m);
            }
            ref_mul(a, a, a);
            ref_mod(a, a, m);
        }
        ref_mul_ui(r, r, term.numerator);
        ref_mod(r, r, m);
        ref_mul_2exp(n, r, digits * bbp::radix_bits);
        ref_fdiv_q(r, n, m);
    } else {
        ref_set_ui(n, term.numerator);
        const int64_t shift = int64_t(digits * bbp::radix_bits) + term.exponent;
        if (shift >= 0)
            ref_mul_2exp(n, n, size_t(shift));
        else
            ref_fdiv_q_2exp(n, n, size_t(-shift));
        ref_fdiv_q(r, n, m);
    }
    bbp::Fixed out{};
    for (unsigned j = 0; j < digits; ++j) {
        out.d[j] = ref_get_ui(r) & bbp::mask;
        ref_fdiv_q_2exp(r, r, bbp::radix_bits);
    }
    ref_clears(m, r, a, n, nullptr);
    return out;
}

void compare(const std::vector<bbp::RationalTerm> &terms, unsigned digits) {
    bbp::Fixed got{}, want{};
    allocation_watch_start();
    bbp::native_rational_sum(got, terms.data(), terms.size(), digits);
    assert(!allocation_watch_stop());
    for (const auto &t : terms)
        want.add(reference(t, digits), digits, t.negative);
    if (memcmp(&got, &want, sizeof got)) {
        for (size_t i = 0; i < terms.size(); ++i) {
            bbp::Fixed g{};
            bbp::native_rational_sum(g, &terms[i], 1, digits);
            bbp::Fixed w{};
            w.add(reference(terms[i], digits), digits, terms[i].negative);
            if (memcmp(&g, &w, sizeof g)) {
                fprintf(stderr, "term %zu, W%u, m=%016llx:%016llx:%016llx:%016llx, E=%lld, P=%llu\n", i,
                        bbp::radix_bits * digits, (unsigned long long)terms[i].modulus[3],
                        (unsigned long long)terms[i].modulus[2], (unsigned long long)terms[i].modulus[1],
                        (unsigned long long)terms[i].modulus[0], (long long)terms[i].exponent,
                        (unsigned long long)terms[i].numerator);
                break;
            }
        }
    }
    assert(!memcmp(&got, &want, sizeof got));
}
void kernels() {
    const unsigned widths[] = {1,   2,   31,  32,  33,  48,  49,  50,  51,  52,  53,  63,
                               64,  65,  101, 102, 103, 104, 105, 127, 128, 129, 153, 154,
                               155, 156, 157, 205, 206, 207, 208, 209, 255, 256};
    std::vector<bbp::RationalTerm> terms;
    for (unsigned bits : widths)
        for (unsigned i = 0; i < 24; ++i) {
            bbp::RationalTerm t{};
            for (unsigned j = 0; j < 4; ++j) {
                const unsigned left = bits > 64 * j ? std::min(64u, bits - 64 * j) : 0;
                t.modulus[j] = left == 64 ? random_word()
                               : left     ? random_word() & ((uint64_t(1) << left) - 1)
                                          : 0;
            }
            t.modulus[(bits - 1) / 64] |= uint64_t(1) << ((bits - 1) % 64);
            if (i % 3 == 0)
                t.modulus[0] |= 1;
            if (i == 0) {
                for (auto &x : t.modulus)
                    x = 0;
                t.modulus[(bits - 1) / 64] = uint64_t(1) << ((bits - 1) % 64);
            }
            if (i == 1) {
                for (unsigned j = 0; j < 4; ++j)
                    t.modulus[j] =
                        64 * j >= bits
                            ? 0
                            : (bits - 64 * j >= 64 ? UINT64_MAX : (uint64_t(1) << (bits - 64 * j)) - 1);
            }
            const int64_t es[] = {0,
                                  1,
                                  2,
                                  31,
                                  32,
                                  63,
                                  64,
                                  255,
                                  256,
                                  -1,
                                  -52,
                                  -223,
                                  -384,
                                  -449,
                                  -512,
                                  int64_t(1) << 32,
                                  (int64_t(1) << 48) - 7,
                                  (int64_t(1) << 62) + 3};
            t.exponent = es[i % (sizeof es / sizeof *es)];
            t.numerator = i % 4 ? random_word() : 1;
            t.negative = i & 1;
            terms.push_back(t);
        }
    // q=(a*k+b)^2 including the actual multi-digit Huvent regime, without
    // attempting to sum trillions of terms merely to reach wide denominators.
    for (uint64_t k : {(uint64_t(1) << 26) - 1, (uint64_t(1) << 40) + 3, (uint64_t(1) << 50) - 1}) {
        for (unsigned b : {1u, 5u, 7u, 11u}) {
            const __uint128_t q = __uint128_t(12 * k + b) * (12 * k + b);
            terms.push_back({{uint64_t(q), uint64_t(q >> 64), 0, 0}, 3, int64_t((k << 1) + b), bool(b & 2)});
        }
    }
    // Quotient estimates immediately next to an integer: r=m-1 and
    // r=(m-1)/2 at wide moduli round up in binary64. Also exercise the other
    // side and every 52-bit leading-digit transition, including top digit 1.
    for (unsigned bit : {31u, 51u, 52u, 53u, 63u, 64u, 103u, 104u, 155u, 156u, 207u, 208u, 254u, 255u}) {
        for (int delta : {-3, -1, 1, 3})
            for (unsigned shift : {0u, 1u}) {
                bbp::RationalTerm t{};
                t.modulus[bit / 64] = uint64_t(1) << (bit % 64);
                if (delta > 0)
                    t.modulus[0] += unsigned(delta);
                else {
                    uint64_t borrow = unsigned(-delta);
                    for (auto &v : t.modulus) {
                        const uint64_t old = v;
                        v -= borrow;
                        borrow = old < borrow;
                    }
                }
                t.numerator = 1;
                t.exponent = bit - shift;
                t.negative = false;
                terms.push_back(t);
            }
    }
    for (unsigned digits : {4u, 8u}) {
        compare(terms, digits);
        for (const auto &term : terms)
            compare(std::vector<bbp::RationalTerm>{term}, digits);
        for (size_t count : {1u, 7u, 8u, 9u, 15u, 16u, 17u, 33u})
            compare(std::vector<bbp::RationalTerm>(terms.begin() + 400, terms.begin() + 400 + count), digits);
    }
}

void word_streams() {
    constexpr uint64_t top = (uint64_t(1) << 52) - 1;
    const bbp::WordStream shapes[] = {
        {4, 1, 1, 1, 0, 10, 1, true, false},    {12, 1, 1, 3, 0, 6, 2, true, true},
        {0, 3, 1, top, 0, 3, 1, false, true},   {2, 3, 3, top - 2, 0, 5, 3, true, false},
        {0, top, 1, top, 0, 1, 1, true, false}, {0, 1, 1, top, 0, 1, 1, false, false}};
    for (auto s : shapes)
        for (unsigned digits : {4u, 8u})
            for (uint64_t begin : {0u, 3u, 4096u})
                for (unsigned count : {1u, 7u, 31u, 32u, 33u, 97u})
                    for (bool small : {false, true}) {
                        s.exponent =
                            small ? int64_t(s.stride * (begin + count - 1)) : ((int64_t(1) << 40) - 7);
                        bbp::Fixed got{}, want{};
                        allocation_watch_start();
                        bbp::native_word_sum(got, s, begin, begin + count, digits);
                        assert(!allocation_watch_stop());
                        for (uint64_t k = begin; k < begin + count; ++k) {
                            bbp::RationalTerm t{};
                            uint64_t m = s.scale;
                            for (unsigned j = 0; j < s.power; ++j)
                                m *= s.a * k + s.b;
                            assert(m <= top);
                            t.modulus[0] = m;
                            t.numerator = s.numerator;
                            t.exponent = s.exponent - int64_t(uint64_t(s.stride) * k);
                            t.negative = s.negative != bool(s.alternating && (k & 1));
                            want.add(reference(t, digits), digits, t.negative);
                        }
                        assert(!memcmp(&got, &want, sizeof got));
                    }
}

void wide_streams() {
    constexpr uint64_t top = (uint64_t(1) << 52) - 1;
    struct Shape {
        uint64_t a, b, scale, begin;
        unsigned power;
    };
    const Shape shapes[] = {
        {2, 1, 1, (uint64_t(1) << 25) - 40, 2}, // cross the one/two-digit boundary
        {2, 1, 3, (uint64_t(1) << 25) - 40, 2},
        {2, 1, 1, (uint64_t(1) << 51) - 40, 1},
        {2, 1, 1, (uint64_t(1) << 17) - 40, 3},
        {2, 1, 3, (uint64_t(1) << 9) - 40, 5},
        {0, top, top, 0, 1}, // almost 2^104: canonical fallback, not lazy
    };
    for (auto shape : shapes)
        for (unsigned digits : {4u, 8u})
            for (uint64_t numerator : {uint64_t(1), uint64_t(3), top - 2})
                for (unsigned count : {1u, 15u, 16u, 17u, 97u, 1103u}) {
                    bbp::WordStream s{shape.a, shape.b,     shape.scale, numerator,      int64_t(1) << 62,
                                      6,       shape.power, true,        bool(count & 1)};
                    bbp::Fixed got{}, want{};
                    allocation_watch_start();
                    bbp::native_word_sum(got, s, shape.begin, shape.begin + count, digits);
                    assert(!allocation_watch_stop());
                    for (uint64_t k = shape.begin; k < shape.begin + count; ++k) {
                        bbp::WideModulus m{};
                        m.d[0] = shape.scale;
                        for (unsigned j = 0; j < shape.power; ++j)
                            assert(m.multiply(shape.a * k + shape.b));
                        assert(m.bits() <= 104);
                        bbp::RationalTerm t{};
                        for (unsigned j = 0; j < 4; ++j)
                            t.modulus[j] = m.d[j];
                        t.numerator = numerator;
                        t.exponent = s.exponent - int64_t(6 * k);
                        t.negative = s.negative != bool(k & 1);
                        want.add(reference(t, digits), digits, t.negative);
                    }
                    assert(!memcmp(&got, &want, sizeof got));
                }
}

void accumulation_limits() {
    // All residues equal m-1: 2^52 == 1 (mod m). Exercise both signs,
    // many radix-52 flushes and the external radix-48 carry chain.
    constexpr uint64_t m = (uint64_t(1) << 52) - 1;
    for (unsigned digits : {4u, 8u})
        for (bool negative : {false, true}) {
            bbp::WordStream s{0, m, 1, m - 1, 52 * 16383, 52, 1, false, negative};
            bbp::Fixed got{};
            allocation_watch_start();
            bbp::native_word_sum(got, s, 0, 16384, digits);
            assert(!allocation_watch_stop());
            const bbp::RationalTerm term{{m, 0, 0, 0}, m - 1, 0, negative};
            const auto one = reference(term, digits);
            ref_int exact;
            ref_init(exact);
            ref_set_ui(exact, 0);
            for (unsigned j = digits; j--;) {
                ref_mul_2exp(exact, exact, 48);
                ref_add_ui(exact, exact, one.d[j]);
            }
            ref_mul_2exp(exact, exact, 14);
            if (negative)
                ref_neg(exact, exact);
            ref_fdiv_r_2exp(exact, exact, 48 * digits);
            bbp::Fixed want{};
            for (unsigned j = 0; j < digits; ++j) {
                want.d[j] = ref_get_ui(exact) & bbp::mask;
                ref_fdiv_q_2exp(exact, exact, 48);
            }
            ref_clear(exact);
            assert(!memcmp(&got, &want, sizeof got));
            // The generic entry accepts larger job lists and must flush columns.
            std::vector<bbp::RationalTerm> jobs(16385, term);
            got = {};
            allocation_watch_start();
            bbp::native_rational_sum(got, jobs.data(), jobs.size(), digits);
            assert(!allocation_watch_stop());
            want.add(one, digits, negative);
            assert(!memcmp(&got, &want, sizeof got));
        }
}

// Independent Catalan verifier: Euler transform of sum (-1)^k/(2k+1)^2.
// N transformed terms leave <2^-N; integer rounding contributes <2N ulps
// at precision P=N+40. Neither Huvent coefficients nor Montgomery are used.
void catalan_reference(uint64_t offset, uint64_t (&bits)[2]) {
    const unsigned n = unsigned(offset) + 128 + 40, p = n + 40;
    std::vector<ref_number> d(n);
    ref_int scale, sum, t;
    ref_inits(scale, sum, t, nullptr);
    ref_set_ui(scale, 1);
    ref_mul_2exp(scale, scale, p);
    for (unsigned k = 0; k < n; ++k) {
        ref_init(&d[k]);
        ref_set_ui(t, uint64_t(2 * k + 1) * (2 * k + 1));
        ref_fdiv_q(&d[k], scale, t);
    }
    ref_set_ui(sum, 0);
    for (unsigned k = 0; k < n; ++k) {
        ref_fdiv_q_2exp(t, &d[0], k + 1);
        ref_add(sum, sum, t);
        for (unsigned j = 0; j + 1 < n - k; ++j)
            ref_sub(&d[j], &d[j], &d[j + 1]);
    }
    ref_fdiv_q_2exp(t, sum, p - offset - 128);
    bits[1] = ref_get_ui(t);
    ref_fdiv_q_2exp(t, t, 64);
    bits[0] = ref_get_ui(t);
    for (auto &x : d)
        ref_clear(&x);
    ref_clears(scale, sum, t, nullptr);
}

void formulas() {
    sbn3_bbp_stream log2{1, 1, 1, 1, -1, 1, 1, 0}; // sum 2^(-k-1)/(k+1)
    sbn3_bbp_stream bellard[7];
    for (unsigned i = 0; i < 7; ++i) {
        const auto t = bbp::formula[i];
        bellard[i] = {t.negative ? -1 : 1, t.a, t.b, 1, t.shift, 10, 1, 1};
    }
    const sbn3_bbp_stream huvent_original[] = {{3, 4, 1, 1, -1, 2, 2, 1},  {-3, 4, 2, 1, -1, 2, 2, 1},
                                               {3, 4, 3, 1, -2, 2, 2, 1},  {-1, 4, 1, 1, -2, 6, 2, 1},
                                               {-1, 4, 2, 1, -3, 6, 2, 1}, {-1, 4, 3, 1, -5, 6, 2, 1}};
    uint64_t expected[4][2];
    const uint64_t offsets[] = {0, 1, 64, 256};
    for (unsigned i = 0; i < 4; ++i)
        catalan_reference(offsets[i], expected[i]);
    sbn3_bbp_result parallel{};
    for (unsigned workers : {1u, 3u, 16u}) {
        Fixture f(workers, false);
        for (unsigned bits : {192u, 384u}) {
            for (unsigned i = 0; i < 4; ++i) {
                sbn3_bbp_result a{}, b{};
                allocation_watch_start();
                sbn3_catalan_bbp(f.team, offsets[i], bits, &a);
                assert(!allocation_watch_stop());
                assert(a.stable && a.bits[0] == expected[i][0] && a.bits[1] == expected[i][1]);
                sbn3_bbp_window(f.team, huvent_original, 6, offsets[i], bits, &b);
                assert(b.stable && a.bits[0] == b.bits[0] && a.bits[1] == b.bits[1]);
            }
            sbn3_bbp_result a{}, b{};
            sbn3_bbp_window(f.team, &log2, 1, 0, bits, &a);
            assert(a.stable && a.bits[0] == 0xb17217f7d1cf79abULL && a.bits[1] == 0xc9e3b39803f2f6afULL);
            for (uint64_t offset : {0u, 1027u}) {
                sbn3_bbp_window(f.team, bellard, 7, offset, bits, &a);
                sbn3_pi_bbp(f.team, offset, bits, &b);
                assert(a.stable && b.stable && a.bits[0] == b.bits[0] && a.bits[1] == b.bits[1]);
            }
            // A constant wide denominator makes an independent exact geometric
            // sum possible. Exercise full public lowering with 100/200-bit q,
            // large signed numerators and both alternating/fixed signs.
            for (unsigned root_bits : {20u, 40u})
                for (unsigned alternating : {0u, 1u}) {
                    sbn3_bbp_stream s{INT64_MIN, 0,          (uint64_t(1) << root_bits) + 3, 3, -4, 17,
                                      5,         alternating};
                    bbp::WideModulus m{};
                    m.d[0] = s.denominator_scale;
                    for (unsigned i = 0; i < s.power; ++i)
                        assert(m.multiply(s.b));
                    assert(m.multiply((uint64_t(1) << 17) + (alternating ? 1 : -1)));
                    bbp::RationalTerm exact{};
                    for (unsigned i = 0; i < 4; ++i)
                        exact.modulus[i] = m.d[i];
                    exact.numerator = uint64_t(1) << 63;
                    exact.exponent = 1027 + 13;
                    exact.negative = true;
                    bbp::Fixed want{};
                    want.add(reference(exact, bits / bbp::radix_bits), bits / bbp::radix_bits, true);
                    sbn3_bbp_window(f.team, &s, 1, 1027, bits, &a);
                    assert(a.stable && a.bits[0] == want.extract(bits - 64) &&
                           a.bits[1] == want.extract(bits - 128));
                }
        }
        sbn3_bbp_result a{};
        allocation_watch_start();
        sbn3_catalan_bbp(f.team, 200003, 192, &a);
        assert(!allocation_watch_stop());
        if (workers == 1)
            parallel = a;
        assert(a.stable && !memcmp(&parallel, &a, sizeof a));
        // Exact dyadic value at a window boundary must report inconclusive.
        sbn3_bbp_stream dyadic{1, 0, 1, 1, -1, 1, 1, 0};
        sbn3_bbp_window(f.team, &dyadic, 1, 0, 192, &a);
        assert(!a.stable);
    }
}
void queries() {
    sbn3_bbp_stream s{1, 4, 1, 1, 0, 6, 2, 1};
    sbn3_bbp_window_info info{}, before{};
    assert(sbn3_bbp_window_supported(&s, 1, uint64_t(1) << 40, 192, &info));
    assert(info.max_modulus_bits > 64);
    before = info;
    s.power = 9;
    assert(!sbn3_bbp_window_supported(&s, 1, 0, 192, &info));
    assert(!memcmp(&info, &before, sizeof info));
    s.power = 8;
    assert(!sbn3_bbp_window_supported(&s, 1, uint64_t(1) << 40, 192, nullptr));
    s.power = 5;
    assert(sbn3_bbp_window_supported(&s, 1, uint64_t(1) << 40, 384, &info));
    assert(info.max_modulus_bits > 156);
    assert(!sbn3_bbp_window_supported(&s, 1, 0, 128, nullptr));
    assert(!sbn3_bbp_window_supported(&s, 1, 0, 224, nullptr));
    assert(!sbn3_bbp_window_supported(&s, 1, 0, 448, nullptr));
    assert(!sbn3_bbp_window_supported(&s, 1, UINT64_MAX, 192, nullptr));
    s.numerator = INT64_MIN;
    s.power = 2;
    assert(sbn3_bbp_window_supported(&s, 1, 0, 192, nullptr));
    s.shift = INT32_MIN;
    assert(sbn3_bbp_window_supported(&s, 1, 0, 192, &info));
    assert(!info.terms);
    s.shift = 0;
    s.a = UINT64_MAX;
    assert(!sbn3_bbp_window_supported(&s, 1, 100, 192, nullptr));
    s.stride = 0;
    assert(!sbn3_bbp_window_supported(&s, 1, 0, 192, nullptr));
    assert(!sbn3_bbp_window_supported(nullptr, 1, 0, 192, nullptr));
}
} // namespace
int main() {
    montgomery<32, 1>();
    montgomery<32, 2>();
    montgomery<32, 3>();
    montgomery<32, 4>();
    montgomery<32, 5>();
    montgomery<32, 6>();
    montgomery<32, 7>();
    montgomery<32, 8>();
    montgomery<52, 1>();
    montgomery<52, 2>();
    montgomery<52, 3>();
    montgomery<52, 4>();
    montgomery<52, 5>();
    kernels();
    word_streams();
    wide_streams();
    accumulation_limits();
    queries();
    formulas();
    puts("BitWindow CIOS u32/u52, 1..256-bit moduli, mixed exponents/tails, Huvent/Euler, Bellard, threads, "
         "no allocation PASS");
}
