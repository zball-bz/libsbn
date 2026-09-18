// Structured formula adapter gates: typed/data exact identity, independent
// integer references for data-only formulas, signs/zero/cancellation,
// nonzero start, wide leaves, capacity and validation rejection, threads,
// rebinding and zero execution-time allocation.
#include "formula_reference.hpp"
#include "series/schedule.hpp"
#include <math.h>
#include <memory>
using namespace sbn::v3;
using namespace sbn::v3::series;
namespace {
using namespace formula_reference;
// ---------------------------------------------------------------- execution
struct Result {
    sbn3_series_info info{};
    std::vector<uint64_t> words[3];
    bool negative[3]{};
    int64_t exponent[3]{};
    size_t size[3]{};
    size_t table_bytes = 0;
};
static Result run(Fixture &fixture, const FiniteFormula &formula, uint64_t begin, uint64_t end, unsigned need,
                  unsigned batch, bool rebind) {
    Result r{};
    FinitePlan plan{};
    const sbn3_series_options options{sbn3_team_workers(fixture.team), batch, 0, 0, 0};
    allocation_watch_start();
    const auto rc = finite_query(formula, {begin, end}, need, options, plan);
    assert(!allocation_watch_stop());
    assert(rc == SBN3_SUPPORTED);
    r.info = plan.info;
    auto meta = fixture.allocate(plan.info.plan_bytes, plan.info.plan_alignment),
         tables = fixture.allocate(plan.info.prepared_bytes, plan.info.prepared_alignment),
         scratch = fixture.allocate(plan.info.workspace_bytes, plan.info.workspace_alignment);
    sbn3_series_values out{};
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j)) {
            auto l = fixture.allocate(plan.info.output.limbs[j] * 8, 64);
            out.value[j].mantissa = {static_cast<uint64_t *>(l.data), plan.info.output.limbs[j], 0, 0};
        }
    for (unsigned pass = 0; pass < (rebind ? 2u : 1u); ++pass) {
        FiniteBinding bound{};
        allocation_watch_start();
        finite_prepare(plan, *fixture.arena, *fixture.team, meta, tables, scratch, bound);
        assert(!allocation_watch_stop());
        for (unsigned repeat = 0; repeat < 2; ++repeat) {
            allocation_watch_start();
            finite_execute(bound, out);
            assert(!allocation_watch_stop());
            for (unsigned j = 0; j < 3; ++j)
                if (need & (1u << j)) {
                    const auto &v = out.value[j];
                    std::vector<uint64_t> words(v.mantissa.data, v.mantissa.data + v.mantissa.size);
                    if (pass || repeat) {
                        assert(words == r.words[j] && v.mantissa.negative == r.negative[j] &&
                               v.exponent2 == r.exponent[j]);
                    }
                    r.words[j] = words;
                    r.negative[j] = v.mantissa.negative;
                    r.exponent[j] = v.exponent2;
                    r.size[j] = v.mantissa.size;
                }
        }
    }
    return r;
}
static void fill(sbn3_series_values &v, Result &r, unsigned need) {
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            v.value[j] = {{r.words[j].data(), r.words[j].size(), r.words[j].size(), r.negative[j]}, r.exponent[j]};
}
static const char *name(sbn3_series_recipe r) {
    return r == SBN3_SERIES_HYPERDESCENT ? "hyper" : r == SBN3_SERIES_COMMON_P2B3 ? "common" : "bbp";
}
// Typed built-in versus its data definition: bit-identical exact T/D/U and
// separately certified.
static void pair(const Formula &typed, const FormulaDef &def, uint64_t begin, uint64_t end, unsigned workers,
                 unsigned need, unsigned batch, bool exact_reference) {
    Fixture fixture(workers, false);
    DataFormula data{};
    assert(data_formula_prepare(def, end, data));
    const auto a = finite_formula(typed), b = data_finite_formula(data);
    assert(a.recipe == b.recipe);
    auto ra = run(fixture, a, begin, end, need, batch, false);
    auto rb = run(fixture, b, begin, end, need, batch, true);
    for (unsigned j = 0; j < 3; ++j)
        if (need & (1u << j))
            assert(ra.words[j] == rb.words[j] && ra.negative[j] == rb.negative[j] && ra.exponent[j] == rb.exponent[j]);
    sbn3_series_values v{};
    fill(v, rb, need);
    if (exact_reference) {
        Ref r;
        reference(def, begin, end, r);
        compare(r, need, v, def.recipe);
    }
    certificate(def, begin, end, need, v);
    printf("pair %s [%llu,%llu) W%u need=%u batch=%u typed/data T=%zu/%zu D=%zu/%zu U=%zu/%zu words; plan bytes prepared %zu/%zu "
           "workspace %zu/%zu stages %zu/%zu work %.3g/%.3g leaf_words %u/%u %s PASS\n",
           name(def.recipe), (unsigned long long)begin, (unsigned long long)end, workers, need, batch,
           ra.size[0], rb.size[0], ra.size[1], rb.size[1], ra.size[2], rb.size[2], ra.info.prepared_bytes,
           rb.info.prepared_bytes, ra.info.workspace_bytes, rb.info.workspace_bytes, ra.info.stages, rb.info.stages,
           ra.info.estimated_work, rb.info.estimated_work, a.max_leaf_limbs, b.max_leaf_limbs,
           data.leaf_kind == LeafKind::HyperWordBatch ? "batch" : "factored");
    fflush(stdout);
}
// Data-only formula against the independent integer reference.
static void solo(const char *label, const FormulaDef &def, uint64_t domain_end, uint64_t begin, uint64_t end,
                 unsigned workers, unsigned need, unsigned batch, bool exact_reference = true) {
    Fixture fixture(workers, false);
    DataFormula data{};
    if (!data_formula_prepare(def, domain_end, data)) {
        fprintf(stderr, "%s rejected: %s\n", label, data.rejection);
        assert(false);
    }
    const auto f = data_finite_formula(data);
    auto r = run(fixture, f, begin, end, need, batch, true);
    sbn3_series_values v{};
    fill(v, r, need);
    if (exact_reference) {
        Ref ref;
        reference(def, begin, end, ref);
        compare(ref, need, v, def.recipe);
    }
    certificate(def, begin, end, need, v);
    printf("%s %s [%llu,%llu) W%u need=%u batch=%u T/D/U=%zu/%zu/%zu of %zu/%zu/%zu words leaf_words=%u %s PASS\n", label,
           name(def.recipe), (unsigned long long)begin, (unsigned long long)end, workers, need, batch, r.size[0],
           r.size[1], r.size[2], r.info.output.limbs[0], r.info.output.limbs[1], r.info.output.limbs[2],
           f.max_leaf_limbs, data.leaf_kind == LeafKind::HyperWordBatch ? "batch" : "factored");
    fflush(stdout);
}
// ---------------------------------------------------------------- analysis gates
static long double bit_length(const ref_number *x) {
    return ref_sgn(x) ? (long double)ref_sizeinbase(x, 2) : 0.L;
}
// The data envelope must contain every actual subtree produced by the data
// split policy: term counts and exact T/D/U bit lengths at each depth.
static void envelope(const DataFormula &d, sbn3_series_range root, sbn3_series_range current, unsigned depth) {
    if (current.end - current.begin <= 8)
        return;
    uint64_t terms = 0;
    sbn3_series_shape shape{}, normalized{};
    uint64_t normalized_terms = 0;
    assert(d.serial_envelope(root, depth, 7, terms, shape) == SBN3_SUPPORTED);
    assert(current.end - current.begin <= terms);
    Ref r;
    reference(d.def, current.begin, current.end, r);
    assert(bit_length(r.t) <= 64.L * shape.limbs[0]);
    assert(bit_length(r.d) <= 64.L * shape.limbs[1]);
    assert(bit_length(r.u) <= 64.L * shape.limbs[2]);
    if (d.Q.twos) {
        assert(d.serial_envelope(root, depth, 7, normalized_terms, normalized, true) == SBN3_SUPPORTED);
        assert(normalized_terms == terms && normalized.limbs[1] <= shape.limbs[1]);
        size_t low_zero_words = 0;
        for (; low_zero_words < 4096; ++low_zero_words) {
            ref_int q;
            ref_init(q);
            ref_fdiv_r_2exp(q, r.d, 64 * (low_zero_words + 1));
            const bool zero = !ref_sgn(q);
            ref_clear(q);
            if (!zero)
                break;
        }
        assert(bit_length(r.d) - 64.L * low_zero_words <= 64.L * normalized.limbs[1]);
    }
    const auto m = d.split_point(current, .5);
    assert(current.begin < m && m < current.end);
    envelope(d, root, {current.begin, m}, depth + 1);
    envelope(d, root, {m, current.end}, depth + 1);
}
// bounds() must cover every subrange with at most max_terms terms.
static void capacity(const DataFormula &d, uint64_t a, uint64_t b, uint64_t maximum) {
    sbn3_series_shape shape{}, normalized{};
    const unsigned need = d.def.recipe == SBN3_SERIES_COMMON_P2B3 ? 7 : 3;
    assert(d.bounds({a, b}, maximum, need, shape) == SBN3_SUPPORTED);
    const bool twos = d.Q.twos && d.def.recipe != SBN3_SERIES_BINARY_BBP;
    if (twos)
        assert(d.bounds({a, b}, maximum, need, normalized, true) == SBN3_SUPPORTED);
    for (uint64_t count = 1; count <= maximum; ++count)
        for (uint64_t start : {a, a + 1, b - count, (a + b - count) / 2})
            if (start >= a && start + count <= b) {
                Ref r;
                reference(d.def, start, start + count, r);
                assert(bit_length(r.t) <= 64.L * shape.limbs[0]);
                assert(bit_length(r.d) <= 64.L * shape.limbs[1]);
                if (need & 4)
                    assert(bit_length(r.u) <= 64.L * shape.limbs[2]);
            }
}
// Root-anchored attenuation certificate: |U(begin,b)/D(begin,b)| <= 2^-att(b), monotone. Between two
// interior indices the difference of two certificates may exceed the true attenuation by less than a bit.
static void attenuation(const DataFormula &d, uint64_t a, uint64_t b) {
    Ref r;
    reference(d.def, a, b, r);
    uint64_t bits = d.attenuation_bits(b) - d.attenuation_bits(a);
    assert(d.attenuation_bits(b) >= d.attenuation_bits(a));
    if (a != d.def.begin && bits)
        --bits;
    ref_int scaled;
    ref_init(scaled);
    ref_abs(scaled, r.u);
    ref_mul_2exp(scaled, scaled, size_t(bits));
    assert(ref_cmp(scaled, r.d) <= 0);
    ref_clear(scaled);
}
static void rejections() {
    DataFormula d{};
    auto expect = [&](const FormulaDef &def, uint64_t end, const char *fragment) {
        assert(!data_formula_prepare(def, end, d));
        assert(d.rejection && strstr(d.rejection, fragment));
        printf("rejected as expected: %s\n", d.rejection);
    };
    { // Hyperdescent needs R == 1
        auto def = euler_definition();
        def.R.count = 1;
        def.R.factor[0] = {1, 1, 1};
        expect(def, 100, "R == 1");
    }
    { // Hyperdescent cannot absorb a negative denominator
        auto def = euler_definition();
        def.Q.negative = true;
        expect(def, 100, "Q > 0");
    }
    { // denominator zero at the first index
        auto def = euler_definition();
        def.begin = 0;
        expect(def, 100, "zero or changes sign");
    }
    { // BinaryBBP needs a stride
        auto def = binary_log_definition(1);
        def.stride = 0;
        expect(def, 100, "stride");
    }
    { // zero constant
        auto def = euler_definition();
        def.P.constant_low = 0;
        expect(def, 100, "zero constant");
    }
    { // leaf wider than 64 words on this domain
        auto def = euler_definition();
        def.Q.count = 6;
        for (unsigned i = 0; i < 6; ++i)
            def.Q.factor[i] = {1, 1, 16};
        expect(def, uint64_t(1) << 48, "64 words");
        assert(data_formula_prepare(def, 1u << 12, d)); // narrow domain: 6*16*13 bits = 20 words
        assert(d.leaf_words > 8 && d.leaf_words <= 64);
    }
    // query/compile/prepare share one precondition set.
    {
        Fixture fixture(1, false);
        DataFormula data{};
        assert(data_formula_prepare(euler_definition(), 100, data));
        auto f = data_finite_formula(data);
        FinitePlan plan{};
        const sbn3_series_options options{1, 8, 0, 0, 0};
        assert(finite_query(f, {1, 50}, 3, options, plan) == SBN3_SUPPORTED);
        auto storage = fixture.allocate(plan.info.plan_bytes + 4096, 64);
        sbn3_series_info info{};
        sbn3_series_plan *compiled = nullptr;
        assert(finite_compile(f, {1, 50}, 3, options, storage.data, storage.bytes, info, compiled) == SBN3_SUPPORTED);
        for (unsigned variant = 0; variant < 4; ++variant) {
            auto broken = f;
            if (variant == 0)
                broken.max_leaf_limbs = 65;
            else if (variant == 1)
                broken.leaf = nullptr;
            else if (variant == 2)
                broken.serial_envelope = f.serial_envelope ? nullptr : broken.serial_envelope, broken.split_point = nullptr,
                broken.serial_envelope = [](const void *, sbn3_series_range, unsigned, unsigned, uint64_t *, sbn3_series_shape *) { return SBN3_SUPPORTED; };
            else
                broken.limit_words = [](const void *, uint64_t) { return size_t(4); }, broken.limit_context = nullptr;
            assert(finite_query(broken, {1, 50}, 3, options, plan) == SBN3_UNSUPPORTED);
            assert(finite_compile(broken, {1, 50}, 3, options, storage.data, storage.bytes, info, compiled) == SBN3_UNSUPPORTED);
        }
        sbn3_series_options budget = options;
        budget.workspace_budget = 1;
        assert(finite_query(f, {1, 50}, 3, budget, plan) == SBN3_QUERY_CAPACITY);
        assert(finite_compile(f, {1, 50}, 3, budget, storage.data, storage.bytes, info, compiled) == SBN3_QUERY_CAPACITY);
        puts("query/compile validation unified PASS");
    }
}
} // namespace
int main() {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    rejections();
    const Formula typed_e{FormulaKind::Euler}, typed_pi{FormulaKind::Chudnovsky}, typed_log{FormulaKind::BinaryLog, 3};
    const FormulaDef data_e = euler_definition(), data_pi = chudnovsky_definition(), data_log = binary_log_definition(3);
    // Typed versus data: same convention, bit-identical exact values.
    for (unsigned w : {1u, 3u, 16u}) {
        for (unsigned mask = 1; mask <= 3; ++mask) {
            pair(typed_e, data_e, 1, 38, w, mask, 3, true);
            pair(typed_log, data_log, 1, 38, w, mask, 3, true);
        }
        for (unsigned mask = 1; mask <= 7; ++mask)
            pair(typed_pi, data_pi, 0, 37, w, mask, 3, true);
        pair(typed_e, data_e, 13, 1000, w, 3, 8, true);
        pair(typed_pi, data_pi, 13, 1000, w, 7, 8, true);
        pair(typed_log, data_log, 13, 1000, w, 3, 1, true);
    }
    pair(typed_e, data_e, 1, 262145, 16, 3, 8, false);
    pair(typed_pi, data_pi, 0, 65536, 16, 7, 8, false);
    pair(typed_pi, data_pi, 0, 65536, 16, 3, 8, false);
    pair(typed_log, data_log, 1, 8193, 16, 3, 8, false);
    // New consumers defined only by data.
    const auto exp_half = exp_reciprocal_definition(2), exp_third = exp_reciprocal_definition(3),
               exp_neg = exp_minus_one_definition();
    for (unsigned w : {1u, 3u, 16u}) {
        solo("exp(1/2)", exp_half, 400, 0, 40, w, 3, 8);
        solo("exp(1/2)", exp_half, 400, 0, 333, w, 3, 8);
        solo("exp(1/2) nonzero start", exp_half, 400, 7, 100, w, 3, 3);
        solo("exp(1/3)", exp_third, 400, 0, 300, w, 3, 8);
        solo("exp(-1) cancellation", exp_neg, 400, 0, 50, w, 3, 8);
        solo("exp(-1) cancellation", exp_neg, 400, 0, 397, w, 3, 1);
    }
    {
        // Zero numerator value inside the range and a sign change of P.
        FormulaDef zero = euler_definition();
        zero.P.count = 1;
        zero.P.factor[0] = {1, -5, 1}; // P(k) = k-5: negative, zero at k=5, then positive
        solo("zero/sign P hyper", zero, 100, 1, 12, 3, 3, 4);
        solo("zero/sign P hyper single", zero, 100, 5, 6, 1, 3, 4);
        solo("zero/sign P hyper", zero, 100, 1, 90, 16, 3, 8);
        // Negative denominator constant and alternating Q for Common (leaf normalization).
        FormulaDef negative{};
        negative.recipe = SBN3_SERIES_COMMON_P2B3;
        negative.begin = 1;
        negative.P.count = 1;
        negative.P.factor[0] = {2, 1, 1};
        negative.Q.constant_low = 3;
        negative.Q.negative = true;
        negative.Q.alternating = true;
        negative.Q.count = 1;
        negative.Q.factor[0] = {1, 0, 2};
        negative.R.count = 1;
        negative.R.factor[0] = {1, 1, 1};
        for (unsigned w : {1u, 16u})
            for (unsigned mask : {7u, 3u, 5u, 1u})
                solo("negative alternating Q common", negative, 500, 1, 120, w, mask, 5);
        // Known powers of two folded at preparation: Q = (4k+8)*(2k) -> 8*(k+2)*k.
        FormulaDef twos{};
        twos.recipe = SBN3_SERIES_COMMON_P2B3;
        twos.begin = 1;
        twos.P.count = 1;
        twos.P.factor[0] = {1, 3, 1};
        twos.Q.count = 2;
        twos.Q.factor[0] = {4, 8, 1};
        twos.Q.factor[1] = {2, 0, 1};
        twos.R.count = 1;
        twos.R.factor[0] = {1, 0, 1};
        DataFormula prepared{};
        assert(data_formula_prepare(twos, 600, prepared));
        assert(prepared.def.Q.factor[0].a == 1 && prepared.def.Q.factor[0].b == 2 && prepared.Q.twos == 3);
        assert(prepared.contraction);
        solo("folded powers of two common", twos, 600, 1, 500, 16, 7, 8);
        capacity(prepared, 1, 90, 17);
        // BinaryBBP with a general numerator, alternating sign, even denominator power.
        FormulaDef bbp{};
        bbp.recipe = SBN3_SERIES_BINARY_BBP;
        bbp.begin = 0;
        bbp.P.constant_low = 3;
        bbp.P.alternating = true;
        bbp.P.count = 1;
        bbp.P.factor[0] = {2, 1, 1};
        bbp.Q.count = 1;
        bbp.Q.factor[0] = {4, 1, 2};
        bbp.shift = -1;
        bbp.stride = 4;
        for (unsigned w : {1u, 3u, 16u})
            solo("general bbp", bbp, 5000, 0, 300, w, 3, 8);
        solo("general bbp nonzero start", bbp, 5000, 11, 4000, 16, 3, 8);
        // Factored Hyper leaf when the batch word path does not apply (wide constant P).
        FormulaDef wide_p = euler_definition();
        wide_p.P.constant_low = 0;
        wide_p.P.constant_high = 5; // P = 5*2^64
        solo("hyper factored leaf", wide_p, 300, 1, 200, 3, 3, 8);
        // Wide single-term leaves beyond the 8-word stack path use reserved scratch.
        FormulaDef wide = euler_definition();
        wide.Q.count = 1;
        wide.Q.factor[0] = {1, 1, 16}; // (k+1)^16
        const uint64_t top = uint64_t(1) << 40;
        DataFormula wide_data{};
        assert(data_formula_prepare(wide, top, wide_data) && wide_data.leaf_words > 8);
        solo("wide leaf hyper", wide, top, top - 40, top, 1, 3, 8);
        solo("wide leaf hyper", wide, top, top - 200, top, 16, 3, 8);
        FormulaDef wide_common = chudnovsky_definition();
        wide_common.P.factor[3].power = 8; // (B k + A)^8 in P: 11-word leaves near 2^40
        assert(data_formula_prepare(wide_common, top, wide_data) && wide_data.leaf_words > 8);
        solo("wide leaf common", wide_common, top, top - 100, top, 16, 7, 8);
        solo("wide leaf common", wide_common, top, top - 33, top, 3, 7, 4);
    }
    // Analysis facts: capacity, smooth envelope, attenuation certificates.
    {
        DataFormula pi{}, e{}, half{};
        assert(data_formula_prepare(data_pi, 70000, pi) && pi.smooth_split && pi.contraction);
        assert(data_formula_prepare(data_e, 300000, e) && e.contraction);
        assert(data_formula_prepare(exp_half, 3000, half) && half.contraction);
        for (uint64_t a : {uint64_t(0), uint64_t(1), uint64_t(65520)})
            for (unsigned maximum : {1u, 3u, 17u})
                capacity(pi, a, a + 2 * maximum + 1, maximum);
        capacity(e, 1, 60, 13);
        capacity(half, 0, 60, 13);
        for (auto root : {sbn3_series_range{0, 700}, {1, 1500}, {65520, 66900}})
            envelope(pi, root, root, 0);
        for (auto [a, b] : {std::pair<uint64_t, uint64_t>{0, 1}, {0, 40}, {1, 40}, {17, 300}, {65000, 65999}}) {
            attenuation(pi, a, b);
            attenuation(e, std::max<uint64_t>(1, a), std::max<uint64_t>(2, b));
            if (a < 2999)
                attenuation(half, a, std::min<uint64_t>(b, 2999));
        }
        printf("attenuation data/typed Chudnovsky at k=1000: %llu / %llu bits; e at k=1000: %llu / %llu\n",
               (unsigned long long)pi.attenuation_bits(1000), 47ull * 999,
               (unsigned long long)e.attenuation_bits(1000), (unsigned long long)(integer_log_sum(1, 1000) - 999));
        // Real-valued total attenuation: sound against the exact log2|D/U|, at
        // least as strong as the integer certificate, and used by tail_terms.
        auto exact_log2 = [](const ref_number *x) {
            const size_t bits = ref_sizeinbase(x, 2);
            ref_int top;
            ref_init(top);
            if (bits > 64)
                ref_fdiv_q_2exp(top, x, bits - 64);
            else
                ref_set(top, x);
            const long double v = (long double)ref_get_ui(top);
            ref_clear(top);
            return log2l(v) + (bits > 64 ? (long double)(bits - 64) : 0.L);
        };
        for (const auto *d : {&pi, &e, &half})
            for (uint64_t n : {uint64_t(2), uint64_t(3), uint64_t(40), uint64_t(1000), uint64_t(2999)}) {
                if (n > d->index_end)
                    continue;
                Ref r;
                reference(d->def, d->def.begin, n, r);
                const long double total = d->attenuation_total(n), exact = exact_log2(r.d) - exact_log2(r.u);
                assert(total <= exact + 1e-6L && total + 1e-6L >= (long double)d->attenuation_bits(n));
            }
        // Tightness of the root-anchored certificate and the proven direction of log2|Q/R|: Chudnovsky's
        // ratio falls to its limit, e's and the accelerated atanh's rise, zeta(3)'s (2+1/k)^5 falls. A
        // product whose offsets interleave has no established direction and keeps the two-ended floor.
        {
            FormulaDef zeta{};
            zeta.recipe = SBN3_SERIES_COMMON_P2B3;
            zeta.begin = 0;
            zeta.P.alternating = true;
            zeta.P.count = 1;
            zeta.P.factor[0] = {1, 0, 5};
            zeta.P.degree = 2;
            zeta.P.coefficient[0] = 77;
            zeta.P.coefficient[1] = 250;
            zeta.P.coefficient[2] = 205;
            zeta.Q.constant_low = 32;
            zeta.Q.count = 1;
            zeta.Q.factor[0] = {2, 1, 5};
            zeta.R.count = 1;
            zeta.R.factor[0] = {1, 0, 5};
            zeta.explicit_first = true;
            zeta.first = {77, 1, 1, false};
            FormulaDef mixed{};
            mixed.recipe = SBN3_SERIES_COMMON_P2B3;
            mixed.begin = 1;
            mixed.Q.constant_low = 64;
            mixed.Q.count = 2;
            mixed.Q.factor[0] = {1, 0, 1};
            mixed.Q.factor[1] = {1, 9, 1};
            mixed.R.count = 2;
            mixed.R.factor[0] = {1, 4, 1};
            mixed.R.factor[1] = {1, 5, 1};
            auto z = std::make_unique<DataFormula>(), m = std::make_unique<DataFormula>();
            assert(data_formula_prepare(zeta, 4000, *z) && z->contraction && z->ratio_direction == -1);
            assert(data_formula_prepare(mixed, 4000, *m) && m->contraction && m->ratio_direction == 0);
            assert(pi.ratio_direction == -1 && e.ratio_direction == 1 && half.ratio_direction == 1);
            unsigned checked = 0;
            long double worst = 0;
            for (const auto *d : {&pi, &e, &half, z.get(), m.get()})
                for (uint64_t n : {uint64_t(2), uint64_t(3), uint64_t(33), uint64_t(40), uint64_t(257), uint64_t(1000),
                                   uint64_t(2999)}) {
                    if (n > d->index_end || n <= d->def.begin)
                        continue;
                    attenuation(*d, d->def.begin, n);
                    for (uint64_t a : {d->def.begin + 1, n / 2, n - 1})
                        if (a > d->def.begin && a < n)
                            attenuation(*d, a, n);
                    Ref r;
                    reference(d->def, d->def.begin, n, r);
                    const long double exact = exact_log2(r.d) - exact_log2(r.u);
                    const long double loss = exact - (long double)d->attenuation_bits(n);
                    // One bit for the floor, two for an explicit first term, 0.3% for the bucket slope
                    // (3% when no direction is established and Q, R are taken at opposite bucket ends).
                    if (!(loss >= 0 && loss <= 3 + (d->ratio_direction ? 0.003L : 0.03L) * exact)) {
                        fprintf(stderr, "attenuation tightness: direction %d n=%llu exact %.3Lf certificate %llu\n",
                                d->ratio_direction, (unsigned long long)n, exact,
                                (unsigned long long)d->attenuation_bits(n));
                        assert(false);
                    }
                    worst = fmaxl(worst, loss / fmaxl(exact, 1));
                    ++checked;
                }
            printf("root-anchored attenuation: %u prefixes within 3 bits + 0.3%% of the exact log2|D/U| "
                   "(largest relative loss %.4Lf)\n", checked, worst);
        }
        printf("real attenuation totals: Chudnovsky k=1000 %.2Lf, e k=1000 %.2Lf (log2(999!)=%.2Lf)\n",
               pi.attenuation_total(1000), e.attenuation_total(1000), lgammal(1000.L) / logl(2.0L));
        const uint64_t n = e.tail_terms(64 * 100 + 8);
        assert(n && e.attenuation_total(n) >= 6408 && e.attenuation_total(n - 1) < 6408);
        printf("tail terms for 100 limbs of e: %llu (typed factorial bound: %llu)\n", (unsigned long long)n,
               (unsigned long long)[&] { uint64_t m = 1; while (factorial_log_bounds(0, m).lower < 6408) ++m; return m; }());
    }
    // F1: Hyperdescent with a polynomial numerator and no factors selects the
    // batch leaf; the batch must evaluate the polynomial. Compare leaf batches of
    // 1/3/8 terms with the exact reference, including sign changes, a zero term
    // and cancellation; also force the factored leaf for the same definition.
    {
        struct Case { const char *label; int64_t c0, c1, c2; bool alternating; uint64_t q_a; int64_t q_b; };
        const Case cases[]{{"P=k", 0, 1, 0, false, 1, 0}, {"P=k-5 (sign change, zero at 5)", -5, 1, 0, false, 1, 0},
                           {"P=(-1)^k (3k^2-40k+1)", 1, -40, 3, true, 1, 0}, {"P=7-2k over Q=2k+1", 7, -2, 0, false, 2, 1}};
        for (const auto &c : cases) {
            FormulaDef def{};
            def.recipe = SBN3_SERIES_HYPERDESCENT;
            def.begin = 1;
            def.P.degree = c.c2 ? 2 : 1;
            def.P.coefficient[0] = c.c0;
            def.P.coefficient[1] = c.c1;
            def.P.coefficient[2] = c.c2;
            def.P.alternating = c.alternating;
            def.Q.count = 1;
            def.Q.factor[0] = {c.q_a, c.q_b, 1};
            DataFormula batch{};
            assert(data_formula_prepare(def, 4000, batch) && batch.leaf_kind == LeafKind::HyperWordBatch);
            // Two-term check straight from the leaf programs: T = P(1) Q(2) + P(2), D = Q(1) Q(2).
            {
                uint64_t words[3][3][8]{};
                sbn3_series_values a{}, b{}, both{};
                for (unsigned j = 0; j < 3; ++j) {
                    a.value[j].mantissa = {words[0][j], 8, 0, 0};
                    b.value[j].mantissa = {words[1][j], 8, 0, 0};
                    both.value[j].mantissa = {words[2][j], 8, 0, 0};
                }
                batch.leaf(1, 3, a);
                batch.leaf(2, 3, b);
                batch.batch({1, 3}, 3, both);
                Ref r;
                reference(def, 1, 3, r);
                compare(r, 3, both, def.recipe);
            }
            auto wide = def;
            wide.P.constant_high = 1; // 2^64 * polynomial: forces the factored leaf
            DataFormula factored{};
            assert(data_formula_prepare(wide, 4000, factored) && factored.leaf_kind == LeafKind::Factored);
            for (unsigned leaf_terms : {1u, 3u, 8u})
                for (auto range : {sbn3_series_range{1, 2}, {1, 6}, {3, 41}, {1, 700}}) {
                    solo(c.label, def, 4000, range.begin, range.end, 3, 3, leaf_terms);
                    solo(c.label, wide, 4000, range.begin, range.end, 3, 3, leaf_terms);
                }
        }
        puts("F1 polynomial numerators in the batch leaf PASS");
    }
    // Wide Common batch: products above one word are applied as (constant, index
    // part) 128-bit passes. Cover two and three words per term, a 2^70-scale
    // constant, polynomial numerators with sign change and zero, a negative
    // denominator, alternation and an explicit first term; every batch length
    // against the independent exact reference and the per-term leaf program.
    {
        auto accelerated = [](uint64_t m) {
            const unsigned __int128 x = (unsigned __int128)m * m;
            unsigned __int128 q = 3 * x * (x - 1) * (x - 1);
            uint64_t g = 8;
            while (q % g)
                g /= 2;
            q /= g;
            FormulaDef d{};
            d.recipe = SBN3_SERIES_COMMON_P2B3;
            d.begin = 1;
            d.P.constant_low = m;
            d.P.degree = 1;
            d.P.coefficient[0] = -int64_t(4 * x * (3 * x - 5) / g);
            d.P.coefficient[1] = int64_t(8 * (9 * x * x - 15 * x + 4) / g);
            d.Q.constant_low = uint64_t(q);
            d.Q.constant_high = uint64_t(q >> 64);
            d.Q.count = 2;
            d.Q.factor[0] = {6, -1, 1};
            d.Q.factor[1] = {6, -5, 1};
            d.R.constant_low = 8 / g;
            d.R.count = 2;
            d.R.factor[0] = {1, 0, 1};
            d.R.factor[1] = {2, -1, 1};
            return d;
        };
        FormulaDef signs = accelerated(26);
        signs.P.degree = 2; // (-1)^k (3k^2 - 40k + 1) * 26: sign changes, large cancellation
        signs.P.coefficient[0] = 1;
        signs.P.coefficient[1] = -40;
        signs.P.coefficient[2] = 3;
        signs.P.alternating = true;
        signs.Q.negative = true;
        signs.Q.constant_high = 64; // about 2^70
        signs.R.alternating = true; // ratio factors must keep their sign; the alternation carries it
        signs.R.negative = true;
        FormulaDef zero = accelerated(97);
        zero.P.degree = 1; // P = 97 (k - 5): an exact zero term
        zero.P.coefficient[0] = -5;
        zero.P.coefficient[1] = 1;
        FormulaDef first = accelerated(4801);
        first.begin = 0;
        first.explicit_first = true;
        first.first = {7, 3, 5, true};
        first.Q.factor[1] = {6, 1, 1}; // keep factors positive from k = 1
        struct Case { const char *label; FormulaDef def; unsigned words; };
        const Case cases[]{{"wide batch accelerated(26)", accelerated(26), 2},
                           {"wide batch accelerated(31817)", accelerated(31817), 3},
                           {"wide batch signs/cancellation", signs, 3},
                           {"wide batch zero term", zero, 2},
                           {"wide batch explicit first", first, 3}};
        // The kind depends on the prepared domain: on a short one accelerated(26)
        // still fits single words; the production domains are this large.
        constexpr uint64_t domain = uint64_t(1) << 27;
        {
            DataFormula narrow{};
            assert(data_formula_prepare(accelerated(26), 5000, narrow) &&
                   narrow.leaf_kind == LeafKind::CommonWordBatch && narrow.batch_words == 1);
        }
        for (const auto &c : cases) {
            DataFormula d{};
            if (!data_formula_prepare(c.def, domain, d)) {
                fprintf(stderr, "%s rejected: %s\n", c.label, d.rejection);
                assert(false);
            }
            if (d.leaf_kind != LeafKind::CommonWideBatch || d.batch_words != c.words)
                fprintf(stderr, "%s kind %u words %u\n", c.label, unsigned(d.leaf_kind), d.batch_words);
            assert(d.leaf_kind == LeafKind::CommonWideBatch && d.batch_words == c.words);
            // The batch program against the exact reference, straight from the leaf programs.
            for (auto range : {sbn3_series_range{c.def.begin, c.def.begin + 1}, {c.def.begin, c.def.begin + 2},
                               {c.def.begin + 3, c.def.begin + 9}, {c.def.begin, c.def.begin + 64},
                               {4900, 4964}})
                for (unsigned need : {7u, 3u, 5u, 2u}) {
                    uint64_t words[3][208]{};
                    sbn3_series_values both{};
                    for (unsigned j = 0; j < 3; ++j)
                        both.value[j].mantissa = {words[j], 208, 0, 0};
                    d.batch(range, need, both);
                    Ref r;
                    reference(c.def, range.begin, range.end, r);
                    compare(r, need, both, c.def.recipe);
                }
            for (unsigned leaf_terms : {1u, 3u, 8u, 32u, 64u})
                for (auto range : {sbn3_series_range{c.def.begin, c.def.begin + 1}, {c.def.begin, c.def.begin + 6},
                                   {c.def.begin + 2, c.def.begin + 41}, {c.def.begin, c.def.begin + 700}})
                    solo(c.label, c.def, domain, range.begin, range.end, 3, 7, leaf_terms);
            solo(c.label, c.def, domain, c.def.begin, 5000, 16, 3, 32);
        }
        puts("wide Common batch leaves: two/three words, signs, zero, cancellation, explicit first PASS");
    }
    // F6: wide linear coefficients with a negative offset prepare and evaluate correctly.
    {
        FormulaDef def = euler_definition();
        def.Q.count = 1;
        def.Q.factor[0] = {UINT64_MAX, -2, 1}; // (2^64-1) k - 2, positive from k = 1
        DataFormula d{};
        assert(data_formula_prepare(def, 100, d));
        assert(d.Q.lo_at(1) >= 64 && d.Q.up_at(1) >= 64);
        solo("wide coefficient negative offset", def, 100, 1, 20, 1, 3, 4);
        solo("wide coefficient negative offset", def, 100, 7, 99, 16, 3, 8);
        FormulaDef common = def;
        common.recipe = SBN3_SERIES_COMMON_P2B3;
        common.R.count = 1;
        common.R.factor[0] = {UINT64_MAX / 3, -1, 1};
        solo("wide coefficient common", common, 100, 1, 60, 3, 7, 8);
        // Offset larger than the coefficient: origin k0 > 1.
        FormulaDef late = euler_definition();
        late.begin = 5;
        late.Q.count = 1;
        late.Q.factor[0] = {3, -13, 1}; // positive from k = 5
        solo("negative offset origin", late, 200, 5, 90, 3, 3, 8);
        puts("F6 wide coefficients and negative offsets PASS");
    }
    // F4: structural infinite-tail certificate, separate from the finite contraction.
    {
        struct Case { const char *label; FormulaDef def; bool certifiable; };
        FormulaDef divergent{};
        divergent.recipe = SBN3_SERIES_COMMON_P2B3;
        divergent.begin = 1;
        divergent.Q.constant_high = uint64_t(1) << 36; // Q = 2^100
        divergent.R.count = 1;
        divergent.R.factor[0] = {1, 0, 1}; // sum (k-1)!/2^(100k): finite contraction, divergent series
        FormulaDef harmonic{};
        harmonic.recipe = SBN3_SERIES_COMMON_P2B3;
        harmonic.begin = 1;
        harmonic.Q.count = 1;
        harmonic.Q.factor[0] = {1, 1, 1}; // Q = k+1
        harmonic.R.count = 1;
        harmonic.R.factor[0] = {1, 0, 1}; // R = k: ratio -> 1, no geometric tail
        FormulaDef geometric{};
        geometric.recipe = SBN3_SERIES_COMMON_P2B3;
        geometric.begin = 1;
        geometric.Q.constant_low = 3;
        geometric.Q.count = 1;
        geometric.Q.factor[0] = {1, 0, 1}; // Q = 3k
        geometric.R.count = 1;
        geometric.R.factor[0] = {1, 0, 1}; // R = k: ratio 1/3 from the first term
        FormulaDef late_contraction = geometric;
        late_contraction.R.factor[0] = {1, 5, 1}; // R = k+5: |R/Q| > 1 for k < 3, no certificate from begin = 1
        FormulaDef growing_numerator = euler_definition();
        growing_numerator.P.degree = 4;
        growing_numerator.P.coefficient[4] = 1000; // P = 1000 k^4 over k!: still convergent
        const Case cases[]{{"e", euler_definition(), true}, {"Chudnovsky", chudnovsky_definition(), true},
                           {"exp(1/3)", exp_reciprocal_definition(3), true}, {"binary log", binary_log_definition(2), true},
                           {"geometric ratio 1/3", geometric, true}, {"1000 k^4 / k!", growing_numerator, true},
                           {"divergent (k-1)!/2^(100k)", divergent, false}, {"harmonic-like ratio -> 1", harmonic, false}};
        {
            // Support boundary: contraction must hold from the first index; a
            // series whose early terms grow is rejected by the finite certificate.
            DataFormula d{};
            assert(data_formula_prepare(late_contraction, uint64_t(1) << 40, d));
            assert(!d.contraction && !d.tail_rejection && d.infinite_tail_terms(64) == 0);
            puts("tail certificate R=k+5 over 3k: rejected by the finite contraction requirement (support boundary)");
        }
        for (const auto &c : cases) {
            DataFormula d{};
            const uint64_t domain = std::min<uint64_t>(uint64_t(1) << 40, formula_domain_limit(c.def));
            assert(data_formula_prepare(c.def, domain, d));
            printf("tail certificate %s: domain=2^%u contraction=%d rejection=%s\n", c.label,
                   unsigned(64 - __builtin_clzll(domain) - 1), d.contraction, d.tail_rejection ? d.tail_rejection : "-");
            assert((d.tail_rejection == nullptr) == c.certifiable);
            if (!c.certifiable)
                continue;
            assert(d.contraction);
            for (uint64_t bits : {uint64_t(64), uint64_t(6408), uint64_t(64 * 5000)}) {
                const uint64_t n = d.infinite_tail_terms(bits);
                assert(n && d.infinite_tail_bits(n) >= (long double)bits);
                if (n > std::max<uint64_t>(d.first_positive, 1))
                    assert(d.infinite_tail_bits(n - 1) < (long double)bits);
            }
            // Soundness: the exact tail sum_{k in [N, M)} term_k (M far beyond
            // convergence at these sizes) must stay below 2^-infinite_tail_bits(N).
            for (uint64_t N : {uint64_t(2), uint64_t(20), uint64_t(60), uint64_t(200)}) {
                if (N < d.first_positive)
                    continue;
                const long double certified = d.infinite_tail_bits(N);
                if (certified < -1.0e29L)
                    continue; // no geometric bound from this N (allowed for small N)
                Ref head, tail;
                reference(c.def, c.def.begin, N, head); // prefix product U/D scales the tail
                reference(c.def, N, N + 400, tail);
                ref_int num, den;
                ref_inits(num, den, nullptr);
                ref_abs(num, tail.t);
                ref_mul(num, num, head.u);
                if (c.def.recipe == SBN3_SERIES_BINARY_BBP) {
                    // tail.t carries exponent2; the head has none (weights are in T).
                    ref_set(den, tail.d);
                    ref_mul(den, den, head.d);
                    if (tail.exponent < 0)
                        ref_mul_2exp(den, den, size_t(-tail.exponent));
                    else
                        ref_mul_2exp(num, num, size_t(tail.exponent));
                } else {
                    ref_set(den, tail.d);
                    ref_mul(den, den, head.d);
                }
                // |tail| <= 2^-certified  <=>  num * 2^certified <= den (certified may be fractional: use floor)
                const long double floor_bits = floorl(certified);
                if (floor_bits >= 0)
                    ref_mul_2exp(num, num, size_t(floor_bits));
                else
                    ref_mul_2exp(den, den, size_t(-floor_bits));
                assert(ref_cmp(num, den) <= 0);
                ref_clears(num, den, nullptr);
            }
            printf("  terms for 6408 bits: attenuation-only %llu, infinite-certified %llu\n",
                   (unsigned long long)d.tail_terms(6408), (unsigned long long)d.infinite_tail_terms(6408));
        }
        // The divergent example is still a legal finite sum.
        solo("divergent as a finite sum", divergent, 40, 1, 12, 1, 7, 4);
        puts("F4 structural tail certificates PASS");
    }
    // Known power of two moved into the exponents: same T/D and U/D, smaller D.
    {
        auto def = chudnovsky_definition();
        Fixture fixture(16, false);
        DataFormula plain{}, extracted{};
        assert(data_formula_prepare(def, 3000, plain));
        def.extract_twos = true;
        assert(data_formula_prepare(def, 3000, extracted) && extracted.twos_shift == 15);
        auto e = euler_definition();
        e.extract_twos = true;
        DataFormula rejected{};
        assert(!data_formula_prepare(e, 100, rejected) && strstr(rejected.rejection, "implicit U"));
        for (auto range : {sbn3_series_range{0, 37}, {13, 700}, {0, 2900}}) {
            auto a = run(fixture, data_finite_formula(plain), range.begin, range.end, 7, 8, false);
            auto b = run(fixture, data_finite_formula(extracted), range.begin, range.end, 7, 8, true);
            const uint64_t factored = range.end - std::max<uint64_t>(range.begin, 1);
            assert(b.exponent[0] == -int64_t(15 * factored) && b.exponent[2] == b.exponent[0] && b.exponent[1] == 0);
            assert(b.size[1] < a.size[1] && b.words[0] == a.words[0] && b.words[2] == a.words[2]);
            ref_int da, db;
            ref_inits(da, db, nullptr);
            ref_import(da, a.size[1], -1, 8, 0, 0, a.words[1].data());
            ref_import(db, b.size[1], -1, 8, 0, 0, b.words[1].data());
            ref_mul_2exp(db, db, size_t(15 * factored)); // D_plain == D_extracted * 2^(15 m)
            assert(!ref_cmp(da, db));
            ref_clears(da, db, nullptr);
            sbn3_series_shape sa{}, sb{};
            assert(plain.bounds(range, range.end - range.begin, 7, sa) == SBN3_SUPPORTED &&
                   extracted.bounds(range, range.end - range.begin, 7, sb) == SBN3_SUPPORTED);
            assert(sb.limbs[1] <= sa.limbs[1] && sb.limbs[0] == sa.limbs[0] && sb.limbs[2] == sa.limbs[2]);
            printf("twos extraction [%llu,%llu): D %zu -> %zu words, T/U words identical, exponents %lld PASS\n",
                   (unsigned long long)range.begin, (unsigned long long)range.end, a.size[1], b.size[1],
                   (long long)b.exponent[0]);
        }
    }
    puts("formula definition adapter: typed/data identity, references, signs, zero, cancellation, start, wide leaves, "
         "capacity, validation, threads, rebind, no allocation PASS");
}
