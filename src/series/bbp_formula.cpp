#include "series/bbp_formula.hpp"
#include "common/checked.hpp"
#include "common/identity.hpp"
#include <algorithm>
namespace sbn::v3::series {
namespace {
using u128 = __uint128_t;
unsigned bits(uint64_t x) {
    return x ? 64 - __builtin_clzll(x) : 0;
}
unsigned bits(u128 x) {
    return x >> 64 ? 64 + bits(uint64_t(x >> 64)) : bits(uint64_t(x));
}
unsigned ceil_log(uint64_t n) {
    return n > 1 ? bits(n - 1) : 0;
}
size_t words(u128 b) {
    return size_t((b + 63) / 64);
}
uint64_t magnitude(int64_t x) {
    return x < 0 ? uint64_t(-(x + 1)) + 1 : uint64_t(x);
}
// Denominator as a factored product on k: scale * (a k + b)^power.
FactorProduct denominator(const sbn3_bbp_stream &s) {
    FactorProduct q{};
    q.constant_low = s.denominator_scale;
    q.count = 1;
    q.factor[0] = {s.a, int64_t(s.b), s.power};
    return q;
}
} // namespace
uint64_t bbp_terms_for(const BbpFormula &f, uint64_t bits_needed) noexcept {
    uint32_t stride_min = 0;
    for (unsigned j = 0; j < f.count; ++j)
        stride_min = j ? std::min(stride_min, f.stream[j].stride) : f.stream[j].stride;
    if (!stride_min)
        return 0;
    return bits_needed / stride_min + 1;
}
bool bbp_prepare(const BbpFormula &input, uint64_t terms, DataBbp &out) noexcept {
    out = {};
    out.def = input;
    out.terms = terms;
    auto reject = [&](const char *why) {
        out.rejection = why;
        return false;
    };
    const auto &d = out.def;
    if (!d.count || d.count > max_bbp_streams)
        return reject("1..32 streams");
    if (!terms || u128(terms) * d.count > (u128(1) << 48))
        return reject("virtual index must stay below 2^48");
    u128 amplitude = 0;
    u128 h = identity::word(identity::fnv_seed, 0x4242505f44454631ULL);
    for (unsigned j = 0; j < d.count; ++j) {
        const auto &s = d.stream[j];
        if (!s.numerator || !s.b || !s.denominator_scale || !s.stride || !s.power || s.power > 8 || s.alternating > 1)
            return reject("stream fields outside the window contract");
        if (u128(s.a) * (terms - 1) + s.b > UINT64_MAX)
            return reject("a*k+b exceeds 64 bits on this domain");
        if (u128(s.stride) * (terms - 1) >= (u128(1) << 62))
            return reject("exponent range");
        out.shift_max = j ? std::max<int64_t>(out.shift_max, s.shift) : s.shift;
        out.shift_min = j ? std::min<int64_t>(out.shift_min, s.shift) : s.shift;
        out.stride_min = j ? std::min(out.stride_min, s.stride) : s.stride;
        const char *why = nullptr;
        // Reuse the factored-product analysis: this is a denominator, so it
        // must be sign-definite and needs the lower envelope.
        struct Access {
            static bool analyze(const FactorProduct &q, uint64_t end, ProductAnalysis &a, const char *&why) {
                FormulaDef probe{};
                probe.recipe = SBN3_SERIES_BINARY_BBP;
                probe.begin = 0;
                probe.Q = q;
                probe.stride = 1;
                DataFormula df{};
                if (!data_formula_prepare(probe, end, df)) {
                    why = df.rejection;
                    return false;
                }
                a = df.Q;
                return true;
            }
        };
        if (!Access::analyze(denominator(s), terms, out.Q[j], why))
            return reject(why);
        for (uint64_t x : {uint64_t(s.numerator), s.a, s.b, s.denominator_scale, uint64_t(int64_t(s.shift)),
                           uint64_t(s.stride), uint64_t(s.power), uint64_t(s.alternating)})
            h = identity::word(uint64_t(h), x);
    }
    // Amplitude of one complete k row relative to 2^(shift_min - stride*k):
    // sum_j |num_j| 2^(shift_j - shift_min) / Q_j(k) <= sum_j |num_j| 2^(shift_j-shift_min) (Q >= 1).
    for (unsigned j = 0; j < d.count; ++j) {
        const auto &s = d.stream[j];
        const unsigned spread = unsigned(s.shift - out.shift_min);
        if (spread > 60 || bits(u128(magnitude(s.numerator))) + spread > 100)
            return reject("stream shift spread too large");
        amplitude += u128(magnitude(s.numerator)) << spread;
    }
    out.amplitude_bits = bits(amplitude);
    unsigned leaf_bits = 1;
    for (unsigned j = 0; j < d.count; ++j)
        leaf_bits = std::max({leaf_bits, bits(magnitude(d.stream[j].numerator)), out.Q[j].up_at(terms - 1)});
    out.leaf_words = unsigned(words(leaf_bits));
    out.formula_id = uint64_t(h);
    out.parameter_id = identity::word(identity::fnv_seed, terms);
    return true;
}
sbn3_query_result DataBbp::bounds(sbn3_series_range r, uint64_t n, unsigned need, sbn3_series_shape &shape) const noexcept {
    shape = {};
    const uint64_t domain = terms * def.count;
    if (rejection || r.end > domain || r.begin >= r.end || !n || n > r.end - r.begin || !need || (need & ~3u))
        return SBN3_UNSUPPORTED;
    // Any contiguous window of n virtual indices inside r, at any stream
    // phase: it covers every stream at most ceil(n/count) times and the
    // n % count leftover terms belong to distinct streams. Per-stream widths
    // are taken at the largest k of the envelope (they increase with k), so
    // this is a phase-independent upper bound; it over-reports by at most the
    // per-row growth times the rows spanned.
    const uint64_t w0 = r.end - n, w1 = r.end;
    const uint64_t k_top = (r.end - 1) / def.count;
    unsigned widths[max_bbp_streams]{};
    for (unsigned j = 0; j < def.count; ++j)
        widths[j] = Q[j].up_at(std::min(k_top, terms - 1));
    std::sort(widths, widths + def.count, [](unsigned a, unsigned b) { return a > b; });
    u128 d = 0;
    const uint64_t full_rows = n / def.count, remainder = n % def.count;
    for (unsigned j = 0; j < def.count; ++j)
        d += u128(widths[j]) * (full_rows + (j < remainder ? 1 : 0));
    // Mantissa at the minimal exponent of the range: a term's exponent exceeds
    // the range minimum by at most (shift_max - shift_min) + stride_max*k_last
    // - stride_min*k_first; with unequal strides this grows with k itself.
    const uint64_t k_first = w0 / def.count, k_last = (w1 - 1) / def.count;
    uint32_t stride_max = 0;
    unsigned numerator_bits = 0;
    for (unsigned j = 0; j < def.count; ++j) {
        stride_max = std::max(stride_max, def.stream[j].stride);
        numerator_bits = std::max(numerator_bits, bits(magnitude(def.stream[j].numerator)));
    }
    const u128 spread = u128(shift_max - shift_min) + u128(stride_max) * k_last - u128(stride_min) * k_first;
    const u128 t = d + ceil_log(n) + numerator_bits + spread + 1;
    const u128 limit = u128(SIZE_MAX / 8) * 64;
    if (std::max(t, d + 1) > limit)
        return SBN3_QUERY_CAPACITY;
    if (need & 1)
        shape.limbs[0] = std::max(size_t(1), words(t));
    if (need & 2)
        shape.limbs[1] = std::max(size_t(1), words(d + 1));
    return SBN3_SUPPORTED;
}
uint64_t DataBbp::attenuation_bits(uint64_t i) const noexcept {
    require(i <= terms * def.count, SBN3_FATAL_ARGUMENT, "bbp attenuation index");
    return uint64_t(stride_min) * (i / def.count);
}
double DataBbp::work(sbn3_series_range r) const noexcept {
    double w = double(r.end - r.begin);
    for (unsigned j = 0; j < def.count; ++j) {
        const uint64_t k0 = r.begin > j ? (r.begin - j + def.count - 1) / def.count : 0;
        const uint64_t k1 = r.end > j ? (r.end - j + def.count - 1) / def.count : 0;
        if (k1 > k0)
            w += double(Q[j].up_sum(k0, k1));
    }
    return w;
}
void DataBbp::leaf(uint64_t i, unsigned need, sbn3_series_values &out) const noexcept {
    require(!rejection && i < terms * def.count, SBN3_FATAL_ARGUMENT, "bbp leaf index");
    const unsigned j = unsigned(i % def.count);
    const uint64_t k = i / def.count;
    const auto &s = def.stream[j];
    auto set = [](sbn3_series_value &v, const uint64_t *a, size_t n, bool negative, int64_t exponent) {
        while (n && !a[n - 1])
            --n;
        require(v.mantissa.data && v.mantissa.capacity >= n, SBN3_FATAL_WORKSPACE, "bbp leaf capacity");
        for (size_t x = 0; x < n; ++x)
            v.mantissa.data[x] = a[x];
        v.mantissa.size = n;
        v.mantissa.negative = n && negative;
        v.exponent2 = exponent;
    };
    if (need & 1) {
        const uint64_t m = magnitude(s.numerator);
        set(out.value[0], &m, 1, (s.numerator < 0) != bool(s.alternating && (k & 1)),
            s.shift - int64_t(uint64_t(s.stride) * k));
    }
    if (need & 2) {
        // scale * (a k + b)^power, at most 8 factors of 64 bits and a 64-bit scale.
        uint64_t w[10]{s.denominator_scale};
        size_t n = 1;
        const uint64_t root = s.a * k + s.b;
        for (unsigned e = 0; e < s.power; ++e) {
            u128 carry = 0;
            for (size_t x = 0; x < n; ++x) {
                carry += u128(w[x]) * root;
                w[x] = uint64_t(carry);
                carry >>= 64;
            }
            if (carry)
                w[n++] = uint64_t(carry);
        }
        set(out.value[1], w, n, false, 0);
    }
}
FiniteFormula bbp_finite_formula(const DataBbp &d) noexcept {
    require(!d.rejection && d.terms, SBN3_FATAL_ARGUMENT, "bbp formula not prepared");
    return FiniteFormula{&d,
                         SBN3_SERIES_BINARY_BBP,
                         d.formula_id,
                         d.parameter_id,
                         d.leaf_words,
                         [](const void *p, sbn3_series_range r, uint64_t n, unsigned need, sbn3_series_shape *s) {
                             return static_cast<const DataBbp *>(p)->bounds(r, n, need, *s);
                         },
                         [](const void *p, sbn3_series_range r) { return static_cast<const DataBbp *>(p)->work(r); },
                         [](const void *p, uint64_t k, unsigned need, sbn3_series_values *v) {
                             static_cast<const DataBbp *>(p)->leaf(k, need, *v);
                         },
                         nullptr};
}
BbpFormula bbp_log2() noexcept {
    BbpFormula f{};
    f.count = 1;
    f.stream[0] = {1, 1, 1, 1, -1, 1, 1, 0}; // 2^(-1-k)/(k+1)
    return f;
}
BbpFormula bbp_pi_standard() noexcept {
    BbpFormula f{};
    f.count = 4;
    f.stream[0] = {4, 8, 1, 1, 0, 4, 1, 0};
    f.stream[1] = {-2, 8, 4, 1, 0, 4, 1, 0};
    f.stream[2] = {-1, 8, 5, 1, 0, 4, 1, 0};
    f.stream[3] = {-1, 8, 6, 1, 0, 4, 1, 0};
    return f;
}
BbpFormula bbp_pi_bellard() noexcept {
    // pi = sum_j sum_k (-1)^(k+negative_j) 2^(shift_j-10k)/(a_j*k+b_j) (verify/bbp.hpp rows).
    BbpFormula f{};
    f.count = 7;
    const struct { unsigned a, b; int shift; bool negative; } rows[7] = {{4, 1, -1, true},  {4, 3, -6, true}, {10, 1, 2, false}, {10, 3, 0, true},
                                                                         {10, 5, -4, true}, {10, 7, -4, true}, {10, 9, -6, false}};
    for (unsigned j = 0; j < 7; ++j)
        f.stream[j] = {rows[j].negative ? -1 : 1, rows[j].a, rows[j].b, 1, rows[j].shift, 10, 1, 1};
    return f;
}
BbpFormula bbp_catalan_huvent() noexcept {
    BbpFormula f{};
    f.count = 9;
    const sbn3_bbp_stream rows[9] = {{3, 12, 1, 1, -1, 6, 2, 1},  {-3, 12, 5, 1, -3, 6, 2, 1}, {-3, 12, 7, 1, -4, 6, 2, 1},
                                     {3, 12, 11, 1, -6, 6, 2, 1}, {-3, 6, 1, 1, -3, 6, 2, 1},  {-3, 6, 5, 1, -7, 6, 2, 1},
                                     {-1, 4, 1, 3, -1, 6, 2, 1},  {-1, 4, 3, 3, -4, 6, 2, 1},  {-1, 2, 1, 3, -4, 6, 2, 1}};
    for (unsigned j = 0; j < 9; ++j)
        f.stream[j] = rows[j];
    return f;
}
} // namespace sbn::v3::series
