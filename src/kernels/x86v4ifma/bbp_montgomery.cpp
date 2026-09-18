#include "verify/bbp_terms.hpp"
#include "verify/word_montgomery.hpp"
#include "common/checked.hpp"
#include <immintrin.h>
#include <array>
#include <utility>

namespace sbn::v3::bbp {
namespace {
struct U52 {
    using V = __m512i;
    using M = __mmask8;
    static constexpr unsigned bits = 52;
    static V splat(uint64_t x) { return _mm512_set1_epi64(x); }
    static V zero() { return _mm512_setzero_si512(); }
    static V one() { return splat(1); }
    static V bit_xor(V a, V b) { return _mm512_xor_si512(a, b); }
    static V add(V a, V b) { return _mm512_add_epi64(a, b); }
    static V sub(V a, V b) { return _mm512_sub_epi64(a, b); }
    static V digit(V x) { return _mm512_and_si512(x, splat((uint64_t(1) << bits) - 1)); }
    static V carry(V x) { return _mm512_srli_epi64(x, bits); }
    static V lo(V a, V b) { return _mm512_madd52lo_epu64(zero(), a, b); }
    static V hi(V a, V b) { return _mm512_madd52hi_epu64(zero(), a, b); }
    static M lt(V a, V b) { return _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT); }
    static M nonzero(V x) { return _mm512_test_epi64_mask(x, x); }
    static M mask_not(M x) { return M(~x); }
    static M bit(V x, unsigned b) { return _mm512_test_epi64_mask(x, splat(uint64_t(1) << b)); }
    static V select(M m, V yes, V no) { return _mm512_mask_blend_epi64(m, no, yes); }
};
using V = U52::V;
using D = __m512d;
constexpr uint64_t digit_mask = (uint64_t(1) << 52) - 1;
constexpr unsigned max_l = (max_modulus_bits + 51) / 52;

struct Prepared {
    WideModulus m;
    uint64_t exponent, numerator;
    unsigned bits;
    bool negative;
};

// Included inside the native island's anonymous namespace. The caller admits
// m<R/8, R=B^L. x<2m => 2^bit*x*x<mR, so raw REDC stays below 2m<R.
template <unsigned L>
inline __attribute__((always_inline)) void square_double_lazy(V (&x)[L], const WordMontgomery<U52, L> &mont,
                                                              __mmask8 bit) {
    if constexpr (L == 1) {
        const V a = _mm512_mask_add_epi64(x[0], bit, x[0], x[0]);
        const V lo = U52::lo(a, x[0]), hi = U52::hi(a, x[0]);
        const V u = U52::lo(lo, U52::sub(U52::zero(), mont.inverse));
        x[0] = U52::add(U52::sub(hi, U52::hi(u, mont.m[0])), mont.m[0]);
    } else {
        V t[2 * L + 1];
        for (auto &v : t)
            v = U52::zero();
#pragma unroll
        for (unsigned i = 0; i < L; ++i) {
#pragma unroll
            for (unsigned j = i; j < L; ++j) {
                V lo = U52::lo(x[i], x[j]), hi = U52::hi(x[i], x[j]);
                if (i != j) {
                    lo = U52::add(lo, lo);
                    hi = U52::add(hi, hi);
                }
                t[i + j] = U52::add(t[i + j], lo);
                t[i + j + 1] = U52::add(t[i + j + 1], hi);
            }
        }
        V carry = U52::zero();
#pragma unroll
        for (unsigned i = 0; i < 2 * L; ++i) {
            const V v = U52::add(_mm512_mask_add_epi64(t[i], bit, t[i], t[i]), carry);
            t[i] = U52::digit(v);
            carry = U52::carry(v);
        }
        t[2 * L] = carry;
#pragma unroll
        for (unsigned i = 0; i < L; ++i) {
            const V u = U52::lo(t[i], mont.inverse);
            carry = U52::zero();
#pragma unroll
            for (unsigned j = 0; j < L; ++j) {
                const V v = U52::add(_mm512_madd52lo_epu64(t[i + j], u, mont.m[j]), carry);
                if (j)
                    t[i + j] = U52::digit(v);
                carry = U52::add(U52::hi(u, mont.m[j]), U52::carry(v));
            }
            const V v = U52::add(t[i + L], carry);
            t[i + L] = U52::digit(v);
            t[i + L + 1] = U52::add(t[i + L + 1], U52::carry(v));
        }
        for (unsigned j = 0; j < L; ++j)
            x[j] = t[L + j];
    }
}

template <unsigned L, unsigned G>
inline __attribute__((always_inline)) void power_batch(V (&r)[G][L], const V (&m)[G][L],
                                                       const std::array<WordMontgomery<U52, L>, G> &mont,
                                                       const V (&exponent)[G], unsigned remaining) {
    __mmask8 eligible = 0xff;
    for (unsigned g = 0; g < G; ++g)
        eligible &= U52::lt(m[g][L - 1], U52::splat(uint64_t(1) << 49));

    if (eligible == 0xff) {
        for (unsigned bit = remaining; bit--;) {
#pragma unroll
            for (unsigned g = 0; g < G; ++g)
                square_double_lazy<L>(r[g], mont[g], U52::bit(exponent[g], bit));
        }
        for (unsigned g = 0; g < G; ++g)
            mont[g].reduce(r[g], U52::zero());
    } else {
        for (unsigned bit = remaining; bit--;) {
#pragma unroll
            for (unsigned g = 0; g < G; ++g) {
                mont[g].multiply(r[g], r[g], r[g]);
                mont[g].masked_double(r[g], U52::bit(exponent[g], bit));
            }
        }
    }
}

// Exact radix-52 emission, only low 48*N bits are retained. Input r is the
// canonical shifted residue P*2^(E+W) mod m. Signed initial state is deliberate.
template <unsigned N> void flush_output(Fixed &out, V (&sums)[N]) {
    int64_t carry = 0;
    __uint128_t bits = 0;
    unsigned available = 0;
    Fixed value{};
    for (unsigned j = 0; j < N; ++j) {
        carry += _mm512_reduce_add_epi64(sums[j]);
        bits |= __uint128_t(uint64_t(carry) & digit_mask) << available;
        carry >>= 52;
        available += 52;
        value.d[j] = uint64_t(bits) & mask;
        bits >>= 48;
        available -= 48;
        sums[j] = U52::zero();
    }
    out.add(value, N);
}
template <unsigned L>
inline __attribute__((always_inline)) void output_redc_step(V (&x)[L], V u,
                                                            const WordMontgomery<U52, L> &mont) {
    const auto carry_bit =
        L == 1 ? _mm512_cmp_epi64_mask(x[0], U52::zero(), _MM_CMPINT_GT) : U52::nonzero(x[0]);
    V carry = _mm512_mask_add_epi64(U52::hi(u, mont.m[0]), carry_bit, U52::hi(u, mont.m[0]), U52::one());
#pragma unroll
    for (unsigned j = 1; j < L; ++j) {
        const V t = U52::add(_mm512_madd52lo_epu64(x[j], u, mont.m[j]), carry);
        x[j - 1] = U52::digit(t);
        carry = U52::add(U52::hi(u, mont.m[j]), _mm512_srai_epi64(t, 52));
    }
    x[L - 1] = carry;
}
template <unsigned L, unsigned N, unsigned G>
inline __attribute__((always_inline)) void
accumulate_output(V (&sums)[N], V (&r)[G][L], const std::array<WordMontgomery<U52, L>, G> &mont,
                  const __mmask8 (&negative)[G], const __mmask8 (&active)[G]) {
    for (unsigned g = 0; g < G; ++g) {
        V borrow = U52::zero();
#pragma unroll
        for (unsigned j = 0; j < L; ++j) {
            const V sub = U52::add(r[g][j], borrow);
            V neg = U52::sub(U52::zero(), sub);
            borrow = U52::select(U52::nonzero(sub), U52::one(), U52::zero());
            if (j + 1 < L)
                neg = U52::digit(neg);
            r[g][j] = U52::select(negative[g], neg, r[g][j]);
        }
    }
#pragma unroll
    for (unsigned j = 0; j < N; ++j) {
#pragma unroll
        for (unsigned g = 0; g < G; ++g) {
            const V u = U52::lo(r[g][0], mont[g].inverse);
            sums[j] = _mm512_mask_add_epi64(sums[j], active[g], sums[j], u);
            if (j + 1 < N)
                output_redc_step(r[g], u, mont[g]);
        }
    }
}

template <unsigned L> D leading(const V (&v)[L]) {
    D x = _mm512_cvtepu64_pd(v[L - 1]);
    if constexpr (L > 1)
        x = _mm512_fmadd_pd(_mm512_cvtepu64_pd(v[L - 2]), _mm512_set1_pd(0x1p-52), x);
    return x;
}

// Exact div-step for a per-lane shift. Multi-digit: r<m, shift<=48; leading
// two digits give absolute quotient error <1 for shifts up to 48. Single-digit additionally
// permits r=1,shift=52 (R seed), shift<=31 for canonical r (prefix), or
// r<B,shift=0 (ordinary numerator). These quotients have error <1 for m>=3.
// Integer low products/carries recover the exact remainder before correction.
template <unsigned L> V shift_digit(const WordMontgomery<U52, L> &mont, V (&r)[L], V shifts, D scale) {
    V q = _mm512_cvttpd_epu64(_mm512_mul_pd(leading(r), scale));
    if constexpr (L == 1) {
        // True difference is in (-m,2m), inside signed i64. The low 64
        // product/shift therefore recover it exactly even when they wrap.
        V next = U52::sub(_mm512_sllv_epi64(r[0], shifts), _mm512_mullo_epi64(q, mont.m[0]));
        const auto below = _mm512_cmp_epi64_mask(next, U52::zero(), _MM_CMPINT_LT);
        next = _mm512_mask_add_epi64(next, below, next, mont.m[0]);
        q = _mm512_mask_sub_epi64(q, below, q, U52::one());
        const auto above = U52::mask_not(U52::lt(next, mont.m[0]));
        r[0] = _mm512_mask_sub_epi64(next, above, next, mont.m[0]);
        return _mm512_mask_add_epi64(q, above, q, U52::one());
    }
    V product_carry = U52::zero(), shift_carry = U52::zero(), borrow = U52::zero();
    for (unsigned j = 0; j < L; ++j) {
        const V p = U52::add(U52::lo(q, mont.m[j]), product_carry);
        product_carry = U52::add(U52::hi(q, mont.m[j]), U52::carry(p));
        const V shifted = U52::add(U52::digit(_mm512_sllv_epi64(r[j], shifts)), shift_carry);
        shift_carry = _mm512_srlv_epi64(r[j], U52::sub(U52::splat(52), shifts));
        const V subtrahend = U52::add(U52::digit(p), borrow);
        r[j] = U52::digit(U52::sub(shifted, subtrahend));
        borrow = U52::select(U52::lt(shifted, subtrahend), U52::one(), U52::zero());
    }
    V top = U52::sub(U52::sub(shift_carry, product_carry), borrow);
    const auto below = _mm512_cmp_epi64_mask(top, U52::zero(), _MM_CMPINT_LT);
    V carry = U52::zero();
    for (unsigned j = 0; j < L; ++j) {
        const V v = U52::add(U52::add(r[j], U52::select(below, mont.m[j], U52::zero())), carry);
        r[j] = U52::digit(v);
        carry = U52::carry(v);
    }
    top = U52::add(top, carry);
    q = _mm512_mask_sub_epi64(q, below, q, U52::one());
    const auto above = mont.reduce(r, top);
    return _mm512_mask_add_epi64(q, above, q, U52::one());
}

template <unsigned L, unsigned N, unsigned G = 1>
void batch(V (&sums)[N], const Prepared *jobs, unsigned count) {
    alignas(64) uint64_t md[G][L][8], seed[G][L][8]{}, numerators[G][L][8]{}, exponents[G][8], steps[G][8];
    uint64_t max_e = 0;
    for (unsigned lane = 0; lane < count; ++lane)
        if ((jobs[lane].exponent + 48 * N) > max_e)
            max_e = (jobs[lane].exponent + 48 * N);
    const unsigned length = max_e ? 64 - __builtin_clzll(max_e) : 0;
    const unsigned remaining = length > 5 ? length - 5 : 0;
    unsigned max_steps = 0;
    __mmask8 negative[G]{};
    for (unsigned g = 0; g < G; ++g)
        for (unsigned lane = 0; lane < 8; ++lane) {
            const unsigned index = 8 * g + lane;
            const auto &p = jobs[index < count ? index : 0];
            for (unsigned j = 0; j < L; ++j)
                md[g][j][lane] = p.m.extract(52 * j) & digit_mask;
            exponents[g][lane] = p.exponent + 48 * N;
            if constexpr (L > 1) {
                const unsigned prefix = unsigned((p.exponent + 48 * N) >> remaining);
                seed[g][(p.bits - 1) / 52][lane] = uint64_t(1) << ((p.bits - 1) % 52);
                steps[g][lane] = 52 * L + prefix - (p.bits - 1);
                if (steps[g][lane] > max_steps)
                    max_steps = unsigned(steps[g][lane]);
            }
            numerators[g][0][lane] = p.numerator & digit_mask;
            if constexpr (L > 1)
                numerators[g][1][lane] = p.numerator >> 52;
            if (p.negative)
                negative[g] |= __mmask8(1u << lane);
        }
    V m[G][L], r[G][L], p[G][L], exponent[G];
    D inverse[G];
    for (unsigned g = 0; g < G; ++g)
        for (unsigned j = 0; j < L; ++j) {
            m[g][j] = _mm512_load_si512(md[g][j]);
            r[g][j] = _mm512_load_si512(seed[g][j]);
            p[g][j] = _mm512_load_si512(numerators[g][j]);
        }
    const auto mont = [&]<size_t... I>(std::index_sequence<I...>) {
        return std::array<WordMontgomery<U52, L>, G>{WordMontgomery<U52, L>(m[I])...};
    }(std::make_index_sequence<G>{});
    for (unsigned g = 0; g < G; ++g) {
        inverse[g] = _mm512_div_pd(_mm512_set1_pd(1), leading(m[g]));
        exponent[g] = _mm512_load_si512(exponents[g]);
        if constexpr (L == 1) {
            // Two exact modular shifts construct R*2^prefix; no R^2 table.
            r[g][0] = U52::one();
            shift_digit(mont[g], r[g], U52::splat(52), _mm512_mul_pd(inverse[g], _mm512_set1_pd(0x1p52)));
            const V prefix = _mm512_srlv_epi64(exponent[g], U52::splat(remaining));
            const D power = _mm512_cvtepu64_pd(_mm512_sllv_epi64(U52::one(), prefix));
            shift_digit(mont[g], r[g], prefix, _mm512_mul_pd(inverse[g], power));
        } else {
            V left = _mm512_load_si512(steps[g]);
            // At most two 48-bit steps construct the Montgomery prefix seed.
            constexpr unsigned seed_step_bits = 48;
            for (unsigned i = 0; i < max_steps; i += seed_step_bits) {
                const V shift = _mm512_min_epu64(left, U52::splat(seed_step_bits));
                const D power = _mm512_cvtepu64_pd(_mm512_sllv_epi64(U52::one(), shift));
                shift_digit(mont[g], r[g], shift, _mm512_mul_pd(inverse[g], power));
                left = U52::sub(left, shift);
            }
        }
    }
    power_batch<L, G>(r, m, mont, exponent, remaining);
    __mmask8 active[G];
    for (unsigned g = 0; g < G; ++g) {
        mont[g].multiply(r[g], r[g], p[g]); // Ordinary P: numerator + domain exit.
        const unsigned valid = count > 8 * g ? (count - 8 * g < 8 ? count - 8 * g : 8) : 0;
        active[g] = __mmask8((1u << valid) - 1);
    }
    accumulate_output<L, N, G>(sums, r, mont, negative, active);
}

template <unsigned N> void dispatch(V (&out)[N], const Prepared *p, unsigned count, unsigned l) {
    switch (l) {
    case 1:
        if (count <= 8)
            batch<1, N>(out, p, count);
        else
            batch<1, N, 4>(out, p, count);
        break;
    case 2:
        if (count <= 8)
            batch<2, N>(out, p, count);
        else
            batch<2, N, 2>(out, p, count);
        break;
    case 3:
        if (count <= 8)
            batch<3, N>(out, p, count);
        else
            batch<3, N, 2>(out, p, count);
        break;
    case 4:
        batch<4, N>(out, p, count);
        break;
    case 5:
        batch<5, N>(out, p, count);
        break;
    }
}

template <unsigned N> void sum(Fixed &out, const RationalTerm *terms, size_t count) {
    V sums[N];
    for (auto &v : sums)
        v = U52::zero();
    Prepared buckets[max_l][32];
    unsigned fill[max_l]{};
    for (size_t i = 0; i < count; ++i) {
        const auto &term = terms[i];
        if (!term.numerator)
            continue;
        WideModulus m{};
        unsigned twos, bits;
        if (!(term.modulus[1] | term.modulus[2] | term.modulus[3])) {
            require(term.modulus[0] != 0, SBN3_FATAL_ARGUMENT, "BBP rational modulus");
            twos = __builtin_ctzll(term.modulus[0]);
            m.d[0] = term.modulus[0] >> twos;
            bits = 64 - __builtin_clzll(m.d[0]);
        } else {
            for (unsigned j = 0; j < modulus_words; ++j)
                m.d[j] = term.modulus[j];
            twos = m.strip_twos();
            bits = m.bits();
        }
        require(term.exponent >= -(int64_t(1) << 62), SBN3_FATAL_ARGUMENT, "BBP rational exponent");
        const int64_t e = term.exponent - twos;
        if (e < 0) {
            out.add(scaled_tail(m, term.numerator, e, N), N, term.negative);
            continue;
        }
        if (bits == 1)
            continue; // Integer contribution from a power-of-two denominator.
        const unsigned l = (bits + 51) / 52;
        const uint64_t numerator =
            bits <= 64 && term.numerator >= m.d[0] ? term.numerator % m.d[0] : term.numerator;
        auto &p = buckets[l - 1][fill[l - 1]++];
        p = {m, uint64_t(e), numerator, bits, term.negative};
        if (fill[l - 1] == (l == 1 ? 32u : l <= 3 ? 16u : 8u)) {
            dispatch<N>(sums, buckets[l - 1], fill[l - 1], l);
            fill[l - 1] = 0;
        }
    }
    for (unsigned l = 1; l <= max_l; ++l)
        if (fill[l - 1])
            dispatch<N>(sums, buckets[l - 1], fill[l - 1], l);
    flush_output(out, sums);
}
// A regular factored source can construct all eight lane moduli directly.
// Four independent groups expose ILP; sums stay vectorized until the chunk ends.
template <unsigned N> void word_sum(Fixed &out, const WordStream &s, uint64_t begin, uint64_t end) {
    V sums[N];
    for (auto &v : sums)
        v = U52::zero();
    unsigned blocks = 0;
    const uint64_t exponent_shift = 48 * N - (s.numerator == 1 ? 52 : 0);
    auto modulus = [&](uint64_t k) {
        const uint64_t root = s.a * k + s.b;
        uint64_t m = s.scale;
        for (unsigned j = 0; j < s.power; ++j)
            m *= root;
        return m;
    };
    uint64_t k = begin;
    if (k < end && modulus(k) == 1) {
        if (!s.a)
            return; // All main terms are integers.
        ++k;
    }
    if (k == end)
        return; // Empty range, or the sole term was an exact integer.
    const bool ordinary_numerator = k == end || s.numerator < modulus(k);
    for (; end - k >= 32; k += 32) {
        const uint64_t max_e = (uint64_t(s.exponent) + exponent_shift) - uint64_t(s.stride) * k;
        const unsigned length = max_e ? 64 - __builtin_clzll(max_e) : 0;
        const unsigned remaining = length > 5 ? length - 5 : 0;
        V m[4][1], r[4][1], e[4], p[4][1];
        D inverse[4];
        for (unsigned g = 0; g < 4; ++g) {
            const V index = U52::add(U52::splat(k + 8 * g), _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            const V root = U52::add(_mm512_mullo_epi64(index, U52::splat(s.a)), U52::splat(s.b));
            m[g][0] = root;
            for (unsigned j = 1; j < s.power; ++j)
                m[g][0] = _mm512_mullo_epi64(m[g][0], root);
            if (s.scale != 1)
                m[g][0] = _mm512_mullo_epi64(m[g][0], U52::splat(s.scale));
            e[g] = U52::sub(U52::splat((uint64_t(s.exponent) + exponent_shift)),
                            _mm512_mullo_epi64(index, U52::splat(s.stride)));
            inverse[g] = _mm512_div_pd(_mm512_set1_pd(1), leading(m[g]));
        }
        const auto mont = [&]<size_t... I>(std::index_sequence<I...>) {
            return std::array<WordMontgomery<U52, 1>, 4>{WordMontgomery<U52, 1>(m[I])...};
        }(std::make_index_sequence<4>{});
        for (unsigned g = 0; g < 4; ++g) {
            r[g][0] = U52::one();
            shift_digit(mont[g], r[g], U52::splat(52), _mm512_mul_pd(inverse[g], _mm512_set1_pd(0x1p52)));
            const V prefix = _mm512_srlv_epi64(e[g], U52::splat(remaining));
            const D power = _mm512_cvtepu64_pd(_mm512_sllv_epi64(U52::one(), prefix));
            shift_digit(mont[g], r[g], prefix, _mm512_mul_pd(inverse[g], power));
            p[g][0] = U52::splat(s.numerator);
            if (!ordinary_numerator)
                shift_digit(mont[g], p[g], U52::zero(), inverse[g]);
        }
        power_batch<1, 4>(r, m, mont, e, remaining);
        const __mmask8 negative =
            __mmask8((s.negative ? 0xff : 0) ^ (s.alternating ? (k & 1 ? 0x55 : 0xaa) : 0));
        for (unsigned g = 0; g < 4; ++g)
            if (s.numerator != 1)
                mont[g].multiply(r[g], r[g], p[g]);
        const __mmask8 signs[4] = {negative, negative, negative, negative};
        const __mmask8 active[4] = {0xff, 0xff, 0xff, 0xff};
        accumulate_output<1, N, 4>(sums, r, mont, signs, active);
        if (++blocks == 32) {
            flush_output(out, sums);
            blocks = 0;
        }
    }
    flush_output(out, sums);
    RationalTerm tail[31];
    unsigned count = 0;
    for (; k < end; ++k)
        tail[count++] = {{modulus(k), 0, 0, 0},
                         s.numerator,
                         s.exponent - int64_t(uint64_t(s.stride) * k),
                         s.negative != bool(s.alternating && (k & 1))};
    if (count)
        sum<N>(out, tail, count);
}
// Factored monotone source, odd denominator <2^104, numerator<2^52.
// Split once at 2^52, then create both modulus digits in SIMD. No job packing.
template <unsigned N> void factored_sum(Fixed &out, const WordStream &s, uint64_t begin, uint64_t end) {
    if (begin == end)
        return;
    auto modulus = [&](uint64_t k) {
        const uint64_t root = s.a * k + s.b;
        __uint128_t m = s.scale;
        for (unsigned j = 0; j < s.power; ++j)
            m *= root;
        return m;
    };
    constexpr uint64_t B = uint64_t(1) << 52;
    if (modulus(end - 1) < B) {
        word_sum<N>(out, s, begin, end);
        return;
    }
    if (modulus(begin) < B) {
        uint64_t lo = begin, hi = end;
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo) / 2;
            if (modulus(mid) < B)
                lo = mid + 1;
            else
                hi = mid;
        }
        word_sum<N>(out, s, begin, lo);
        begin = lo;
    }
    V sums[N];
    for (auto &v : sums)
        v = U52::zero();
    unsigned blocks = 0;
    uint64_t k = begin;
    const uint64_t exponent_shift = 48 * N - (s.numerator == 1 ? 104 : 0);
    for (; end - k >= 16; k += 16) {
        const uint64_t max_e = uint64_t(s.exponent) + exponent_shift - uint64_t(s.stride) * k;
        const unsigned length = max_e ? 64 - __builtin_clzll(max_e) : 0,
                       remaining = length > 5 ? length - 5 : 0;
        V m[2][2], r[2][2], e[2], p[2][2], left[2];
        D inverse[2];
        for (unsigned g = 0; g < 2; ++g) {
            const V index = U52::add(U52::splat(k + 8 * g), _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
            const V root = U52::add(_mm512_mullo_epi64(index, U52::splat(s.a)), U52::splat(s.b));
            const V root0 = U52::digit(root), root1 = _mm512_srli_epi64(root, 52);
            m[g][0] = U52::splat(s.scale & digit_mask);
            m[g][1] = U52::splat(s.scale >> 52);
            for (unsigned j = 0; j < s.power; ++j) {
                V hi = U52::hi(m[g][0], root0);
                hi = _mm512_madd52lo_epu64(hi, m[g][0], root1);
                hi = _mm512_madd52lo_epu64(hi, m[g][1], root0);
                m[g][0] = U52::lo(m[g][0], root0);
                m[g][1] = U52::digit(hi);
            }
            e[g] = U52::sub(U52::splat(uint64_t(s.exponent) + exponent_shift),
                            _mm512_mullo_epi64(index, U52::splat(s.stride)));
            inverse[g] = _mm512_div_pd(_mm512_set1_pd(1), leading(m[g]));
            const V zeros = _mm512_lzcnt_epi64(m[g][1]);
            r[g][0] = U52::zero();
            r[g][1] = _mm512_sllv_epi64(U52::one(), U52::sub(U52::splat(63), zeros));
            left[g] =
                U52::add(_mm512_srlv_epi64(e[g], U52::splat(remaining)), U52::sub(zeros, U52::splat(11)));
            p[g][0] = U52::splat(s.numerator);
            p[g][1] = U52::zero();
        }
        const unsigned max_steps =
            unsigned(max_e >> remaining) +
            unsigned(__builtin_clzll(uint64_t(_mm_cvtsi128_si64(_mm512_castsi512_si128(m[0][1]))))) - 11;
        const std::array<WordMontgomery<U52, 2>, 2> mont{WordMontgomery<U52, 2>(m[0]),
                                                         WordMontgomery<U52, 2>(m[1])};
        for (unsigned i = 0; i < max_steps; i += 48)
            for (unsigned g = 0; g < 2; ++g) {
                const V shift = _mm512_min_epu64(left[g], U52::splat(48));
                const D power = _mm512_cvtepu64_pd(_mm512_sllv_epi64(U52::one(), shift));
                shift_digit(mont[g], r[g], shift, _mm512_mul_pd(inverse[g], power));
                left[g] = U52::sub(left[g], shift);
            }
        power_batch<2, 2>(r, m, mont, e, remaining);
        if (s.numerator != 1)
            for (unsigned g = 0; g < 2; ++g)
                mont[g].multiply(r[g], r[g], p[g]);
        const __mmask8 negative =
            __mmask8((s.negative ? 0xff : 0) ^ (s.alternating ? (k & 1 ? 0x55 : 0xaa) : 0));
        const __mmask8 signs[2] = {negative, negative}, active[2] = {0xff, 0xff};
        accumulate_output<2, N, 2>(sums, r, mont, signs, active);
        if (++blocks == 64) {
            flush_output(out, sums);
            blocks = 0;
        }
    }
    flush_output(out, sums);
    RationalTerm tail[15];
    unsigned count = 0;
    for (; k < end; ++k) {
        const auto m = modulus(k);
        tail[count++] = {{uint64_t(m), uint64_t(m >> 64), 0, 0},
                         s.numerator,
                         s.exponent - int64_t(uint64_t(s.stride) * k),
                         s.negative != bool(s.alternating && (k & 1))};
    }
    if (count)
        sum<N>(out, tail, count);
}

} // namespace

void native_rational_sum(Fixed &out, const RationalTerm *terms, size_t count, unsigned digits) {
    require(digits == 4 || digits == 8, SBN3_FATAL_ARGUMENT, "BBP fraction width");
    // Each radix-52 column sums at most 1024 terms: sum<2^62, plus a
    // small carry. Bound each flush before horizontal signed-i64 reduction.
    while (count) {
        const size_t n = count < 1024 ? count : 1024;
        if (digits == 4)
            sum<4>(out, terms, n);
        else
            sum<8>(out, terms, n);
        terms += n;
        count -= n;
    }
}
} // namespace sbn::v3::bbp

namespace sbn::v3::bbp {
void native_word_sum(Fixed &out, const WordStream &s, uint64_t begin, uint64_t end, unsigned digits) {
    if (digits == 4)
        factored_sum<4>(out, s, begin, end);
    else
        factored_sum<8>(out, s, begin, end);
}
} // namespace sbn::v3::bbp
