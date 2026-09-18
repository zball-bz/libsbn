// One BBP definition, three evaluations: exact finite (BinaryBBP recipe over
// the interleaved virtual index), the existing bit window, and the bounded
// prefix (PSR with the absolute-contribution certificate + ratio terminal).
// log(2), the standard four-stream pi formula, Bellard and Huvent cover
// coefficients, even denominators, powers, different shifts and signs.
#include "product_support.hpp"
#include "series/bbp_formula.hpp"
#include "series/psr.hpp"
#include "series/terminal.hpp"
#include "sbn3/verify.h"
#include <sys/random.h>
using namespace sbn::v3;
using namespace sbn::v3::series;
namespace {
using u128 = __uint128_t;
struct Named {
    const char *name;
    BbpFormula formula;
};
uint64_t magnitude(int64_t x) { return x < 0 ? uint64_t(-(x + 1)) + 1 : uint64_t(x); }
// Exact rational S over virtual indices [a,b): T (at exponent e) / D.
struct Exact {
    ref_int t, d, tmp, q;
    int64_t exponent = 0;
    Exact() { ref_inits(t, d, tmp, q, nullptr); }
    ~Exact() { ref_clears(t, d, tmp, q, nullptr); }
};
void reference(const BbpFormula &f, uint64_t a, uint64_t b, Exact &r) {
    ref_set_ui(r.t, 0);
    ref_set_ui(r.d, 1);
    for (uint64_t i = a; i < b; ++i) {
        const auto &s = f.stream[i % f.count];
        const uint64_t k = i / f.count;
        const int64_t e = s.shift - int64_t(uint64_t(s.stride) * k);
        ref_set_ui(r.q, s.denominator_scale);
        for (unsigned p = 0; p < s.power; ++p)
            ref_mul_ui(r.q, r.q, s.a * k + s.b);
        ref_set_ui(r.tmp, magnitude(s.numerator));
        if ((s.numerator < 0) != bool(s.alternating && (k & 1)))
            ref_neg(r.tmp, r.tmp);
        if (i > a) {
            ref_mul(r.t, r.t, r.q);
            if (r.exponent > e)
                ref_mul_2exp(r.t, r.t, size_t(r.exponent - e));
            else if (r.exponent < e)
                ref_mul_2exp(r.tmp, r.tmp, size_t(e - r.exponent));
            ref_mul(r.tmp, r.tmp, r.d);
            ref_add(r.t, r.t, r.tmp);
            r.exponent = std::min(r.exponent, e);
        } else {
            ref_set(r.t, r.tmp);
            r.exponent = e;
        }
        ref_mul(r.d, r.d, r.q);
    }
}
uint64_t challenge() {
    uint64_t x;
    do {
        assert(getrandom(&x, sizeof x, 0) == sizeof x);
        x = (x & ((uint64_t(1) << 60) - 1)) | (uint64_t(1) << 60) | 1;
    } while (!ref_prime61(x));
    return x;
}
uint64_t mm(uint64_t a, uint64_t b, uint64_t q) { return u128(a) * b % q; }
uint64_t powmod(uint64_t a, uint64_t n, uint64_t q) {
    uint64_t r = 1;
    for (; n; n >>= 1, a = mm(a, a, q))
        if (n & 1)
            r = mm(r, a, q);
    return r;
}
// Exact finite value through the BinaryBBP recipe; small ranges against the
// integer reference, larger ones against fresh prime certificates.
void exact_finite(const Named &c, uint64_t a, uint64_t b, unsigned workers, unsigned batch, bool small) {
    Fixture fixture(workers, false);
    DataBbp data{};
    const uint64_t terms = (b + c.formula.count - 1) / c.formula.count;
    assert(bbp_prepare(c.formula, terms, data));
    const auto f = bbp_finite_formula(data);
    FinitePlan plan{};
    const sbn3_series_options options{workers, batch, 0, 0, 0};
    allocation_watch_start();
    assert(finite_query(f, {a, b}, 3, options, plan) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    auto meta = fixture.allocate(plan.info.plan_bytes, plan.info.plan_alignment),
         tables = fixture.allocate(plan.info.prepared_bytes, plan.info.prepared_alignment),
         scratch = fixture.allocate(plan.info.workspace_bytes, plan.info.workspace_alignment);
    sbn3_series_values out{};
    for (unsigned j = 0; j < 2; ++j) {
        auto l = fixture.allocate(plan.info.output.limbs[j] * 8, 64);
        out.value[j].mantissa = {static_cast<uint64_t *>(l.data), plan.info.output.limbs[j], 0, 0};
    }
    FiniteBinding bound{};
    allocation_watch_start();
    finite_prepare(plan, *fixture.arena, *fixture.team, meta, tables, scratch, bound);
    finite_execute(bound, out);
    assert(!allocation_watch_stop());
    const auto &t = out.value[0], &d = out.value[1];
    assert(!d.mantissa.negative && d.mantissa.size);
    if (small) {
        Exact r;
        reference(c.formula, a, b, r);
        ref_int got;
        ref_init(got);
        ref_import(got, t.mantissa.size, -1, 8, 0, 0, t.mantissa.data);
        if (t.mantissa.negative)
            ref_neg(got, got);
        assert(t.exponent2 == r.exponent && !ref_cmp(got, r.t));
        ref_import(got, d.mantissa.size, -1, 8, 0, 0, d.mantissa.data);
        assert(!ref_cmp(got, r.d));
        ref_clear(got);
    }
    for (unsigned trial = 0; trial < 2; ++trial) {
        const uint64_t q = challenge();
        uint64_t tm = 0, dm = 1;
        int64_t exponent = 0;
        for (uint64_t i = a; i < b; ++i) {
            const auto &s = c.formula.stream[i % c.formula.count];
            const uint64_t k = i / c.formula.count;
            const int64_t e = s.shift - int64_t(uint64_t(s.stride) * k);
            uint64_t qv = s.denominator_scale % q;
            for (unsigned p = 0; p < s.power; ++p)
                qv = mm(qv, (s.a * k + s.b) % q, q);
            uint64_t pv = magnitude(s.numerator) % q;
            if ((s.numerator < 0) != bool(s.alternating && (k & 1)))
                pv = pv ? q - pv : 0;
            if (i > a) {
                tm = mm(tm, qv, q);
                if (exponent > e)
                    tm = mm(tm, powmod(2, uint64_t(exponent - e), q), q);
                else if (exponent < e)
                    pv = mm(pv, powmod(2, uint64_t(e - exponent), q), q);
                tm = (tm + mm(pv, dm, q)) % q;
                exponent = std::min(exponent, e);
            } else {
                tm = pv;
                exponent = e;
            }
            dm = mm(dm, qv, q);
        }
        uint64_t got = ref_mod_words(t.mantissa.data, t.mantissa.size, q);
        if (t.mantissa.negative)
            got = got ? q - got : 0;
        assert(got == tm && ref_mod_words(d.mantissa.data, d.mantissa.size, q) == dm && t.exponent2 == exponent);
    }
    printf("exact finite %s [%llu,%llu) W%u batch=%u T=%zu D=%zu words exponent=%lld PASS\n", c.name,
           (unsigned long long)a, (unsigned long long)b, workers, batch, t.mantissa.size, d.mantissa.size,
           (long long)t.exponent2);
    fflush(stdout);
}
// floor(2^128 * frac(2^offset * S)) from the exact finite rational.
void exact_window(const BbpFormula &f, uint64_t rows, uint64_t offset, uint64_t bits[2]) {
    Exact r;
    reference(f, 0, rows * f.count, r);
    ref_int a, dd;
    ref_inits(a, dd, nullptr);
    ref_set(a, r.t);
    ref_set(dd, r.d);
    const __int128 scale = __int128(offset) + 128 + r.exponent;
    if (scale >= 0)
        ref_mul_2exp(a, a, size_t(scale));
    else
        ref_mul_2exp(dd, dd, size_t(-scale));
    ref_fdiv_q(a, a, dd);      // floor(2^(offset+128) S)
    ref_fdiv_r_2exp(a, a, 128); // keep 128 fractional-window bits
    uint64_t words[2]{};
    size_t count = 0;
    ref_export(words, &count, -1, 8, 0, 0, a);
    bits[0] = words[1];
    bits[1] = words[0];
    ref_clears(a, dd, nullptr);
}
// Window backend fed with the same stream definition, compared with the exact
// finite value over at least the window's own term rows.
void window_consistency(const Named &c, uint64_t offset, unsigned workers) {
    Fixture fixture(workers, false);
    sbn3_bbp_window_info info{};
    assert(sbn3_bbp_window_supported(c.formula.stream, c.formula.count, offset, 192, &info));
    sbn3_bbp_result result{};
    allocation_watch_start();
    sbn3_bbp_window(fixture.team, c.formula.stream, c.formula.count, offset, 192, &result);
    assert(!allocation_watch_stop());
    uint64_t rows = 0;
    for (unsigned j = 0; j < c.formula.count; ++j) {
        const auto &s = c.formula.stream[j];
        const int64_t limit = int64_t(offset) + s.shift + 192 + 64;
        rows = std::max<uint64_t>(rows, limit < 0 ? 1 : uint64_t(limit) / s.stride + 2);
    }
    uint64_t bits[2]{};
    exact_window(c.formula, rows, offset, bits);
    if (result.stable)
        assert(result.bits[0] == bits[0] && result.bits[1] == bits[1]);
    printf("window %s offset=%llu W%u stable=%u bits=%016llx%016llx exact=%016llx%016llx terms=%llu err=%llu %s\n",
           c.name, (unsigned long long)offset, workers, result.stable, (unsigned long long)result.bits[0],
           (unsigned long long)result.bits[1], (unsigned long long)bits[0], (unsigned long long)bits[1],
           (unsigned long long)result.terms, (unsigned long long)result.error_ulps,
           result.stable ? "PASS" : "inconclusive (allowed)");
    fflush(stdout);
}
// Built-in verification entries must equal the generic window of the shared rows.
void builtin_agreement(unsigned workers) {
    Fixture fixture(workers, false);
    const auto bellard = bbp_pi_bellard(), huvent = bbp_catalan_huvent();
    for (uint64_t offset : {uint64_t(0), uint64_t(1), uint64_t(63), uint64_t(4096), uint64_t(1) << 20}) {
        sbn3_bbp_result a{}, b{};
        sbn3_pi_bbp(fixture.team, offset, 192, &a);
        sbn3_bbp_window(fixture.team, bellard.stream, bellard.count, offset, 192, &b);
        assert(a.stable && b.stable && a.bits[0] == b.bits[0] && a.bits[1] == b.bits[1]);
        sbn3_catalan_bbp(fixture.team, offset, 192, &a);
        sbn3_bbp_window(fixture.team, huvent.stream, huvent.count, offset, 192, &b);
        assert(a.stable && b.stable && a.bits[0] == b.bits[0] && a.bits[1] == b.bits[1]);
    }
    printf("built-in pi/catalan windows equal the shared definitions W%u PASS\n", workers);
}
struct Decay {
    const DataBbp *data;
};
uint64_t decay(const void *p, uint64_t a) { return static_cast<const Decay *>(p)->data->attenuation_bits(a); }
// Bounded prefix: PSR (absolute-contribution certificate) + ratio terminal.
// Returns floor(S * 2^(64*fractional)) with fractional+1 limbs.
std::vector<uint64_t> prefix(Fixture &fixture, const Named &c, size_t fractional, unsigned workers, uint64_t max_block) {
    const size_t guard = 2, working = fractional + guard, psr_fractional = working + 1;
    // Tail: omitted rows attenuate by stride_min per row beyond the amplitude.
    DataBbp probe{};
    assert(bbp_prepare(c.formula, 2, probe));
    const uint64_t rows = bbp_terms_for(c.formula, 64 * psr_fractional + 8 + probe.amplitude_bits + magnitude(probe.shift_max) + 2);
    assert(rows);
    static DataBbp data{}; // prepared object outlives the PSR binding (static: single-threaded test)
    assert(bbp_prepare(c.formula, rows, data));
    const Decay policy{&data};
    // Like the pi service, amortize each block over at least the root
    // precision's worth of bit growth; otherwise slowly decaying streams cut
    // the low-precision tail into more than the PSR's 128 blocks.
    PsrSpec spec{bbp_finite_formula(data), {0, rows * c.formula.count}, {workers, 8, 0, 0, 0},
                 {&policy, decay, max_block ? 4u : 128u, max_block, max_block ? 0.0 : 64.0 * double(psr_fractional)},
                 psr_fractional};
    PsrInfo info{};
    allocation_watch_start();
    assert(psr_query(spec, info) == SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    fixture.cursor = up(offset + info.storage_bytes, 4096) + 4096;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    sbn3_series_values out{};
    auto result = fixture.allocate(2 * info.output_limbs * 8, 128);
    for (unsigned j = 0; j < 2; ++j)
        out.value[j].mantissa = {static_cast<uint64_t *>(result.data) + j * info.output_limbs, info.output_limbs, 0, 0};
    std::vector<uint64_t> value;
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        PsrBinding binding{};
        allocation_watch_start();
        psr_prepare(spec, info, *fixture.arena, *fixture.team, offset, binding);
        psr_execute(binding, out);
        assert(!allocation_watch_stop());
        psr_release(binding);
        RatioTerminalPlan terminal{};
        assert(ratio_terminal_query(fractional, guard, workers, info.error_units + 2, terminal) == SBN3_SUPPORTED);
        const size_t at = up(fixture.base + fixture.cursor, terminal.storage_alignment) - fixture.base;
        fixture.cursor = up(at + terminal.storage_bytes, 4096) + 4096;
        assert(sbn3_arena_prepare(fixture.arena, at, terminal.storage_bytes, &error) == SBN3_OK);
        auto output = fixture.allocate(terminal.output_limbs * 8, 64);
        allocation_watch_start();
        ratio_terminal_execute(terminal, out.value[0], out.value[1], fixture.arena, at, fixture.team,
                               {static_cast<uint64_t *>(output.data), terminal.output_limbs});
        assert(!allocation_watch_stop());
        std::vector<uint64_t> got(static_cast<uint64_t *>(output.data), static_cast<uint64_t *>(output.data) + terminal.output_limbs);
        if (repeat)
            assert(got == value);
        value = got;
    }
    printf("prefix %s fractional=%zu W%u rows=%llu blocks=%u error<%llu ulps bytes=%zu value=%llx.%016llx%016llx\n", c.name,
           fractional, workers, (unsigned long long)rows, info.blocks, (unsigned long long)info.error_units,
           info.storage_bytes, (unsigned long long)value[fractional], (unsigned long long)value[fractional - 1],
           (unsigned long long)value[fractional - 2]);
    fflush(stdout);
    return value;
}
// F3: bounds() must cover every contiguous sub-window of at most max_terms
// virtual indices at every stream phase, for unequal denominator widths,
// powers, scales and strides, including max_terms not a multiple of count.
void capacity_windows(const Named &c, uint64_t rows, uint64_t max_terms) {
    DataBbp data{};
    assert(bbp_prepare(c.formula, rows, data));
    const uint64_t domain = rows * c.formula.count;
    for (uint64_t begin : {uint64_t(0), uint64_t(1), uint64_t(3)})
        for (uint64_t end = std::min<uint64_t>(domain, begin + 2); end <= domain; end += (domain > 40 ? 7 : 1)) {
            if (end <= begin)
                continue;
            sbn3_series_shape shape{};
            assert(data.bounds({begin, end}, std::min(max_terms, end - begin), 3, shape) == SBN3_SUPPORTED);
            for (uint64_t a = begin; a < end; ++a)
                for (uint64_t m = 1; m <= std::min(max_terms, end - a); ++m) {
                    Exact r;
                    reference(c.formula, a, a + m, r);
                    assert(ref_sizeinbase(r.d, 2) <= 64 * shape.limbs[1]);
                    assert(!ref_sgn(r.t) || ref_sizeinbase(r.t, 2) <= 64 * shape.limbs[0]);
                }
        }
    sbn3_series_shape whole{};
    assert(data.bounds({0, domain}, std::min(max_terms, domain), 3, whole) == SBN3_SUPPORTED);
    printf("capacity windows %s rows=%llu max_terms=%llu envelope T/D=%zu/%zu limbs PASS\n", c.name,
           (unsigned long long)rows, (unsigned long long)max_terms, whole.limbs[0], whole.limbs[1]);
}
} // namespace
int main() {
    {
        BbpFormula review{}; // the review counterexample: Q0 = (k + INT64_MAX)^4, Q1 = 1
        review.count = 2;
        review.stream[0] = {1, 1, uint64_t(INT64_MAX), 1, 0, 1, 4, 0};
        review.stream[1] = {1, 0, 1, 1, 0, 1, 1, 0};
        DataBbp data{};
        assert(bbp_prepare(review, 2, data));
        sbn3_series_shape shape{};
        assert(data.bounds({0, 4}, 1, 3, shape) == SBN3_SUPPORTED && shape.limbs[1] >= 4);
        BbpFormula mixed{};
        mixed.count = 3;
        mixed.stream[0] = {3, 12, 1, 1, -1, 6, 2, 1};              // Huvent-like, power 2
        mixed.stream[1] = {-1, 2, 1, uint64_t(1) << 40, -4, 1, 1, 0}; // scale 2^40, stride 1
        mixed.stream[2] = {5, 8, 7, 1, 2, 10, 8, 1};               // power 8, stride 10
        const Named cases[]{{"review", review}, {"mixed widths", mixed}, {"pi-bbp", bbp_pi_standard()}, {"huvent", bbp_catalan_huvent()}};
        for (const auto &c : cases)
            for (uint64_t max_terms : {uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(5), uint64_t(8)})
                capacity_windows(c, c.formula.count == 2 ? 2 : 6, max_terms);
        capacity_windows(cases[1], 40, 7);
        puts("F3 BBP capacity covers all sub-windows and phases PASS");
    }
    const Named cases[]{{"log2", bbp_log2()}, {"pi-bbp", bbp_pi_standard()}, {"bellard", bbp_pi_bellard()}, {"huvent", bbp_catalan_huvent()}};
    for (unsigned workers : {1u, 3u, 16u}) {
        for (const auto &c : cases) {
            exact_finite(c, 0, 5 * c.formula.count, workers, 3, true);
            exact_finite(c, 0, 60 * c.formula.count, workers, 8, true);
            exact_finite(c, 7, 41 * c.formula.count + 2, workers, 4, true); // nonzero start, partial rows
            exact_finite(c, 0, 900 * c.formula.count, workers, 8, false);
        }
        builtin_agreement(workers);
    }
    exact_finite(cases[2], 3, 20000 * 7, 16, 8, false);
    for (const auto &c : cases)
        for (uint64_t offset : {uint64_t(0), uint64_t(1), uint64_t(63), uint64_t(64), uint64_t(1000), uint64_t(4096), uint64_t(20000)})
            window_consistency(c, offset, offset > 2000 ? 16 : 1);
    // Bounded prefixes: known leading limbs, then Prefix/Window agreement on the last 128 bits.
    const uint64_t known_log2[2]{0xb17217f7d1cf79abULL, 0xc9e3b39803f2f6afULL};
    const uint64_t known_pi[2]{0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL};
    for (unsigned workers : {1u, 3u, 16u})
        for (size_t fractional : {size_t(4), size_t(37), size_t(300)}) {
            Fixture fixture(workers, false);
            for (const auto &c : cases) {
                const uint64_t max_block = fractional == 37 && c.formula.count <= 4 ? 64 : 0; // small cuts; 128-block PSR cap
                auto v = prefix(fixture, c, fractional, workers, max_block);
                if (!strcmp(c.name, "log2"))
                    assert(v[fractional] == 0 && v[fractional - 1] == known_log2[0] && (fractional < 2 || v[fractional - 2] == known_log2[1]));
                else if (!strcmp(c.name, "huvent"))
                    assert(v[fractional] == 0);
                else
                    assert(v[fractional] == 3 && v[fractional - 1] == known_pi[0] && (fractional < 2 || v[fractional - 2] == known_pi[1]));
                sbn3_bbp_result window{};
                sbn3_bbp_window(fixture.team, c.formula.stream, c.formula.count, 64 * (fractional - 2), 192, &window);
                assert(window.stable && window.bits[0] == v[1] && window.bits[1] == v[0]);
            }
        }
    {
        Fixture fixture(16, false);
        auto a = prefix(fixture, cases[1], 2048, 16, 0), b = prefix(fixture, cases[2], 2048, 16, 0);
        assert(a == b); // two pi definitions, one exact prefix
    }
    // Streams with different strides in one definition: the prefix of
    // log2 + pi-BBP (strides 1 and 4) equals the sum of the separate prefixes.
    {
        BbpFormula mixed = bbp_log2();
        const auto pi = bbp_pi_standard();
        for (unsigned j = 0; j < pi.count; ++j)
            mixed.stream[mixed.count++] = pi.stream[j];
        const Named both{"log2+pi mixed stride", mixed};
        window_consistency(both, 5000, 16); // its own team; no other team may be live
        exact_finite(both, 0, 40 * mixed.count, 16, 8, true);
        exact_finite(both, 3, 700 * mixed.count + 1, 16, 8, false);
        Fixture fixture(16, false);
        for (size_t fractional : {size_t(9), size_t(150)}) {
            auto sum = prefix(fixture, both, fractional, 16, 0);
            auto x = prefix(fixture, cases[0], fractional, 16, 0), y = prefix(fixture, cases[1], fractional, 16, 0);
            ref_int a, b, c;
            ref_inits(a, b, c, nullptr);
            ref_import(a, x.size(), -1, 8, 0, 0, x.data());
            ref_import(b, y.size(), -1, 8, 0, 0, y.data());
            ref_import(c, sum.size(), -1, 8, 0, 0, sum.data());
            ref_add(a, a, b);
            ref_sub(a, a, c); // floor(x)+floor(y)-floor(x+y) is 0 or -1
            ref_add_ui(a, a, 1);
            assert(ref_sgn(a) >= 0 && ref_cmp_ui(a, 1) <= 0);
            ref_clears(a, b, c, nullptr);
        }
        puts("mixed-stride BBP exact/window/prefix: prefix equals the sum of its parts PASS");
    }
    puts("BBP unified definition: exact finite, window consistency, built-in agreement, bounded prefix, "
         "prefix/window agreement, threads, rebind, no allocation PASS");
}
