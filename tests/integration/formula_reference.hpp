#pragma once
// Independent integer reference for FormulaDef evaluation: exact sequential
// (T,D,U) with the leaf conventions of formula_def.hpp, plus fresh 61-bit
// prime certificates. Test-only; shares nothing with the production leaf.
#include "product_support.hpp"
#include "series/formula_def.hpp"
#include <algorithm>
#include <sys/random.h>
namespace formula_reference {
using namespace sbn::v3;
using namespace sbn::v3::series;
using u128 = __uint128_t;
using i128 = __int128_t;
// ---------------------------------------------------------------- reference
struct Ref {
    ref_int t, d, u, tmp, f;
    int64_t exponent = 0;
    Ref() { ref_inits(t, d, u, tmp, f, nullptr); }
    ~Ref() { ref_clears(t, d, u, tmp, f, nullptr); }
    Ref(const Ref &) = delete;
};
// Exact value of one factor product at k (sign in the return value).
[[maybe_unused]] static bool product(ref_number *out, ref_number *tmp, const FactorProduct &p, uint64_t k) {
    ref_set_ui(out, p.constant_high);
    ref_mul_2exp(out, out, 64);
    ref_add_ui(out, out, p.constant_low);
    bool negative = p.negative != (p.alternating && (k & 1));
    for (unsigned i = 0; i < p.count; ++i) {
        const auto &x = p.factor[i];
        ref_set_ui(tmp, x.a);
        ref_mul_ui(tmp, tmp, k);
        if (x.b >= 0)
            ref_add_ui(tmp, tmp, uint64_t(x.b));
        else
            ref_sub_ui(tmp, tmp, uint64_t(-(x.b + 1)) + 1);
        if (ref_sgn(tmp) < 0) {
            ref_neg(tmp, tmp);
            if (x.power & 1)
                negative = !negative;
        }
        for (unsigned e = 0; e < x.power; ++e)
            ref_mul(out, out, tmp);
    }
    if (p.degree) {
        ref_int poly, term;
        ref_inits(poly, term, nullptr);
        ref_set_ui(poly, 0);
        for (unsigned i = p.degree + 1; i-- > 0;) {
            ref_mul_ui(poly, poly, k);
            const int64_t c = p.coefficient[i];
            if (c >= 0)
                ref_add_ui(poly, poly, uint64_t(c));
            else
                ref_sub_ui(poly, poly, uint64_t(-(c + 1)) + 1);
        }
        if (ref_sgn(poly) < 0) {
            ref_neg(poly, poly);
            negative = !negative;
        }
        ref_mul(out, out, poly);
        ref_clears(poly, term, nullptr);
    }
    return negative;
}
// Sequential exact evaluation of [a,b) with the same conventions as the leaf.
[[maybe_unused]] static void reference(const FormulaDef &def, uint64_t a, uint64_t b, Ref &r) {
    ref_set_ui(r.t, 0);
    ref_set_ui(r.d, 1);
    ref_set_ui(r.u, 1);
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    r.exponent = 0;
    ref_int p, q, w;
    ref_inits(p, q, w, nullptr);
    for (uint64_t k = a; k < b; ++k) {
        bool pn, qn, wn;
        const int64_t e = bbp ? def.shift - int64_t(uint64_t(def.stride) * k) : 0;
        if (def.explicit_first && k == def.begin) {
            ref_set_ui(p, def.first.t);
            pn = def.first.t_negative;
            ref_set_ui(q, def.first.d);
            qn = false;
            ref_set_ui(w, def.first.u);
            wn = false;
        } else {
            pn = product(p, r.tmp, def.P, k);
            qn = product(q, r.tmp, def.Q, k);
            wn = common ? product(w, r.tmp, def.R, k) : (ref_set_ui(w, 1), false);
            if (qn) { // leaf normalization to a positive denominator
                pn = !pn;
                wn = !wn;
                qn = false;
            }
        }
        if (pn)
            ref_neg(p, p);
        if (wn)
            ref_neg(w, w);
        if (bbp) {
            // T at the running exponent; the new term is scaled down by 2^(e_prev-e).
            if (k > a) {
                ref_mul(r.t, r.t, q);
                ref_mul_2exp(r.t, r.t, size_t(r.exponent - e));
                ref_mul(r.tmp, r.d, p);
                ref_add(r.t, r.t, r.tmp);
            } else
                ref_set(r.t, p);
            r.exponent = e;
        } else {
            ref_mul(r.t, r.t, q);
            if (common)
                ref_mul(p, p, r.u);
            ref_add(r.t, r.t, p);
        }
        ref_mul(r.d, r.d, q);
        if (common)
            ref_mul(r.u, r.u, w);
    }
    ref_clears(p, q, w, nullptr);
}
[[maybe_unused]] static void compare(const Ref &r, unsigned need, const sbn3_series_values &v, sbn3_series_recipe recipe) {
    ref_int got;
    ref_init(got);
    const ref_number *want[3]{r.t, r.d, r.u};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j)) {
            const auto &m = v.value[j].mantissa;
            assert(m.size <= m.capacity);
            ref_import(got, m.size, -1, 8, 0, 0, m.data);
            if (m.negative)
                ref_neg(got, got);
            assert(!ref_cmp(got, want[j]));
            assert(v.value[j].exponent2 == (j == 0 && recipe == SBN3_SERIES_BINARY_BBP ? r.exponent : 0));
            if (m.size)
                assert(m.data[m.size - 1]); // normalized
            else
                assert(!m.negative);
        }
    ref_clear(got);
}
// Fresh 61-bit prime certificate for large ranges (both paths of a pair).
[[maybe_unused]] static uint64_t challenge() {
    uint64_t x;
    do {
        assert(getrandom(&x, sizeof x, 0) == sizeof x);
        x = (x & ((uint64_t(1) << 60) - 1)) | (uint64_t(1) << 60) | 1;
    } while (!ref_prime61(x));
    return x;
}
[[maybe_unused]] static uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return u128(a) * b % q; }
[[maybe_unused]] static uint64_t powmod(uint64_t a, uint64_t n, uint64_t q) {
    uint64_t r = 1;
    for (; n; n >>= 1, a = mm(a, a, q))
        if (n & 1)
            r = mm(r, a, q);
    return r;
}
[[maybe_unused]] static uint64_t product_mod(const FactorProduct &p, uint64_t k, uint64_t q) {
    uint64_t v = (mm(p.constant_high % q, powmod(2, 64, q), q) + p.constant_low % q) % q;
    bool negative = p.negative != (p.alternating && (k & 1));
    for (unsigned i = 0; i < p.count; ++i) {
        const auto &x = p.factor[i];
        const i128 value = i128(u128(x.a) * k) + x.b;
        const u128 magnitude = value < 0 ? u128(-(value + 1)) + 1 : u128(value);
        if (value < 0 && (x.power & 1))
            negative = !negative;
        const uint64_t f = uint64_t(magnitude % q);
        for (unsigned e = 0; e < x.power; ++e)
            v = mm(v, f, q);
    }
    if (p.degree) {
        uint64_t poly = 0;
        for (unsigned i = p.degree + 1; i-- > 0;) {
            poly = mm(poly, k % q, q);
            const int64_t c = p.coefficient[i];
            const uint64_t m = uint64_t(c < 0 ? -(c + 1) : c) + (c < 0 ? 1 : 0);
            poly = (poly + (c < 0 ? q - m % q : m % q)) % q;
        }
        v = mm(v, poly, q);
    }
    return negative && v ? q - v : v;
}
[[maybe_unused]] static void certificate(const FormulaDef &def, uint64_t a, uint64_t b, unsigned need, const sbn3_series_values &out) {
    const bool common = def.recipe == SBN3_SERIES_COMMON_P2B3, bbp = def.recipe == SBN3_SERIES_BINARY_BBP;
    for (unsigned trial = 0; trial < 2; ++trial) {
        const uint64_t q = challenge();
        uint64_t t = 0, d = 1, u = 1;
        for (uint64_t k = a; k < b; ++k) {
            uint64_t p, dd, w;
            if (def.explicit_first && k == def.begin) {
                p = def.first.t % q;
                if (def.first.t_negative && p)
                    p = q - p;
                dd = def.first.d % q;
                w = def.first.u % q;
            } else {
                p = product_mod(def.P, k, q);
                dd = product_mod(def.Q, k, q);
                w = common ? product_mod(def.R, k, q) : 1;
                // Positive denominator normalization: detect Q's sign from its constant/(-1)^k.
                if (def.Q.negative != (def.Q.alternating && (k & 1))) {
                    p = p ? q - p : 0;
                    dd = dd ? q - dd : 0;
                    w = w ? q - w : 0;
                }
            }
            if (bbp) {
                t = mm(t, dd, q);
                if (k > a)
                    t = mm(t, powmod(2, def.stride, q), q);
                t = (t + mm(d, p, q)) % q;
            } else {
                t = mm(t, dd, q);
                t = (t + mm(common ? u : 1, p, q)) % q;
            }
            d = mm(d, dd, q);
            u = mm(u, w, q);
        }
        const uint64_t expected[]{t, d, u};
        for (unsigned j = 0; j < 3; ++j)
            if (need & (1u << j)) {
                const auto &v = out.value[j];
                uint64_t got = ref_mod_words(v.mantissa.data, v.mantissa.size, q);
                if (v.mantissa.negative)
                    got = got ? q - got : 0;
                assert(got == expected[j]);
            }
    }
    if (bbp && (need & 1))
        assert(out.value[0].exponent2 == def.shift - int64_t(uint64_t(def.stride) * (b - 1)));
}
} // namespace formula_reference
