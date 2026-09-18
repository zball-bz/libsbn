// Unified precision planning of the partial-sum reduction (docs/series-precision-planner-math-2026-09-17.md):
// one rule decides, per remaining range, between an exact tree and limited blocks. Gates:
//   * the exact/bounded crossing moves with the requested precision, one limb before and after;
//   * an exact plan returns the exact finite T and D (no nonzero bit dropped), including canonical
//     low zero words in exponents, alternating signs, a Common and a BinaryBBP recipe;
//   * limited plans stay inside the certified error with the root-anchored attenuation certificate,
//     also with mixed limited and exact blocks and with the rule disabled;
//   * worker counts, rebind, no allocation on any face.
#include "formula_reference.hpp"
#include "series/psr.hpp"
#include "sbn3/log.h"
#include <memory>
using namespace formula_reference;
namespace {
uint64_t decay(const void *p, uint64_t a) {
    return static_cast<const DataFormula *>(p)->attenuation_bits(a);
}
FormulaDef from_public(const sbn3_formula_def &in) {
    FormulaDef d{};
    d.recipe = in.recipe;
    d.begin = in.begin;
    auto product = [](const sbn3_formula_product &p) {
        FactorProduct f{};
        f.constant_low = p.constant_low;
        f.constant_high = p.constant_high;
        f.negative = p.negative != 0;
        f.alternating = p.alternating != 0;
        f.count = p.factor_count;
        for (unsigned i = 0; i < f.count; ++i)
            f.factor[i] = {p.factor[i].a, p.factor[i].b, p.factor[i].power};
        f.degree = p.degree;
        for (unsigned i = 0; i <= f.degree; ++i)
            f.coefficient[i] = p.coefficient[i];
        return f;
    };
    d.P = product(in.P);
    d.Q = product(in.Q);
    d.R = product(in.R);
    d.shift = in.shift;
    d.stride = in.stride;
    d.explicit_first = in.explicit_first != 0;
    d.first = {in.first_t, in.first_d, in.first_u, in.first_t_negative != 0};
    return d;
}
FormulaDef zeta3() {
    FormulaDef d{};
    d.recipe = SBN3_SERIES_COMMON_P2B3;
    d.begin = 0;
    d.P.alternating = true;
    d.P.count = 1;
    d.P.factor[0] = {1, 0, 5};
    d.P.degree = 2;
    d.P.coefficient[0] = 77;
    d.P.coefficient[1] = 250;
    d.P.coefficient[2] = 205;
    d.Q.constant_low = 32;
    d.Q.count = 1;
    d.Q.factor[0] = {2, 1, 5};
    d.R.count = 1;
    d.R.factor[0] = {1, 0, 5};
    d.explicit_first = true;
    d.first = {77, 1, 1, false};
    return d;
}
FormulaDef sine() { // sum (-1)^k/(2k+1)!
    FormulaDef d{};
    d.recipe = SBN3_SERIES_HYPERDESCENT;
    d.begin = 0;
    d.P.alternating = true;
    d.Q.constant_low = 2;
    d.Q.count = 2;
    d.Q.factor[0] = {1, 0, 1};
    d.Q.factor[1] = {2, 1, 1};
    d.explicit_first = true;
    d.first = {1, 1, 1, false};
    return d;
}
struct Request {
    size_t fractional = 0;
    unsigned workers = 16;
    bool exact_suffix = true;
    uint64_t maximum_block_terms = 0;
    double minimum_block_bits = 0;
};
struct Outcome {
    unsigned blocks = 0, exact_blocks = 0;
    bool exact_value = false;
};
// value = numerator 2^en / (denominator 2^ed); reference = t 2^er / d.
Outcome run(Fixture &fixture, const DataFormula &data, uint64_t end, const Request &q, const Ref &ref) {
    PsrSpec spec{data_finite_formula(data),
                 {data.def.begin, end},
                 {q.workers, 8, 0, 0, 0},
                 {&data, decay, q.maximum_block_terms ? 4u : 128u, q.maximum_block_terms, q.minimum_block_bits},
                 q.fractional};
    spec.exact_suffix = q.exact_suffix;
    PsrInfo info{};
    allocation_watch_start();
    const auto status = psr_query(spec, info);
    assert(!allocation_watch_stop());
    assert(status == SBN3_SUPPORTED && info.blocks && info.exact_blocks <= info.blocks);
    const size_t offset = up(fixture.base + fixture.cursor, info.storage_alignment) - fixture.base;
    sbn3_error error{};
    assert(sbn3_arena_prepare(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    const size_t pair_offset = up(offset + info.storage_bytes, 4096) + 4096;
    assert(sbn3_arena_prepare(fixture.arena, pair_offset, 2 * info.output_limbs * 8, &error) == SBN3_OK);
    sbn3_lease pair{};
    sbn3_arena_acquire(fixture.arena, pair_offset, 2 * info.output_limbs * 8, &pair);
    Outcome outcome{info.blocks, info.exact_blocks, false};
    ref_int n, d, left, right, limit;
    ref_inits(n, d, left, right, limit, nullptr);
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        sbn3_series_values out{};
        for (unsigned j = 0; j < 2; ++j)
            out.value[j].mantissa = {static_cast<uint64_t *>(pair.data) + j * info.output_limbs, info.output_limbs, 0, 0};
        PsrBinding binding{};
        allocation_watch_start();
        psr_prepare(spec, info, *fixture.arena, *fixture.team, offset, binding);
        psr_execute(binding, out);
        psr_release(binding);
        assert(!allocation_watch_stop());
        const auto &N = out.value[0], &D = out.value[1];
        assert(D.mantissa.size && !D.mantissa.negative);
        ref_import(n, N.mantissa.size, -1, 8, 0, 0, N.mantissa.data);
        if (N.mantissa.negative)
            ref_neg(n, n);
        ref_import(d, D.mantissa.size, -1, 8, 0, 0, D.mantissa.data);
        const int64_t en = N.mantissa.size ? N.exponent2 : 0, ed = D.exponent2, er = ref.exponent;
        // |n 2^en ref.d - ref.t 2^(er+ed) d| 2^(64 f) < error_units d 2^ed ref.d, scaled to integers.
        const int64_t low = std::min({en, er + ed, ed});
        ref_mul(left, n, ref.d);
        ref_mul_2exp(left, left, size_t(en - low));
        ref_mul(right, ref.t, d);
        ref_mul_2exp(right, right, size_t(er + ed - low));
        ref_sub(left, left, right);
        ref_abs(left, left);
        ref_mul_2exp(left, left, 64 * q.fractional);
        ref_mul(limit, d, ref.d);
        ref_mul_2exp(limit, limit, size_t(ed - low));
        ref_mul_ui(limit, limit, info.error_units);
        assert(ref_cmp(left, limit) < 0);
        // Exact: the pair is the finite T and D themselves, up to the power of two held in the exponents.
        bool exact = true;
        for (unsigned j = 0; j < 2 && exact; ++j) {
            const ref_number *mine = j ? d : n, *theirs = j ? ref.d : ref.t;
            const int64_t e0 = j ? ed : en, e1 = j ? 0 : er, m = std::min(e0, e1);
            ref_mul_2exp(left, mine, size_t(e0 - m));
            ref_mul_2exp(right, theirs, size_t(e1 - m));
            exact = ref_cmp(left, right) == 0;
        }
        if (!repeat)
            outcome.exact_value = exact;
        assert(outcome.exact_value == exact);
        if (info.exact_blocks == info.blocks)
            assert(exact); // an all-exact plan must not drop a bit
    }
    ref_clears(n, d, left, right, limit, nullptr);
    sbn3_arena_release(fixture.arena, &pair);
    assert(sbn3_arena_trim(fixture.arena, pair_offset, 2 * info.output_limbs * 8, &error) == SBN3_OK);
    assert(sbn3_arena_trim(fixture.arena, offset, info.storage_bytes, &error) == SBN3_OK);
    return outcome;
}
// The crossing of one finite range: the smallest precision whose demand holds the exact span.
void crossing(Fixture &fixture, const char *label, const FormulaDef &def, uint64_t end, unsigned workers) {
    auto data = std::make_unique<DataFormula>();
    assert(data_formula_prepare(def, end, *data) && data->contraction);
    const auto formula = data_finite_formula(*data);
    sbn3_series_shape span{};
    assert(formula.exact_span && formula.exact_span(formula.context, {def.begin, end}, 3, &span) == SBN3_SUPPORTED);
    const size_t words = std::max(span.limbs[0], span.limbs[1]);
    assert(words > 8);
    const size_t first_exact = words + 1 - 4; // span + 1 <= fractional + guard
    Ref ref;
    reference(data->def, def.begin, end, ref);
    // The span fact is a true bound and close to the values it bounds.
    const size_t actual = (std::max(ref_sizeinbase(ref.t, 2), ref_sizeinbase(ref.d, 2)) + 63) / 64;
    assert(actual <= words);
    for (size_t fractional : {first_exact - 1, first_exact, first_exact + 1, first_exact + 40}) {
        const auto got = run(fixture, *data, end, {fractional, workers}, ref);
        const bool expected = fractional >= first_exact;
        assert((got.blocks == 1 && got.exact_blocks == 1) == expected);
        assert(!expected || got.exact_value);
    }
    // Far below the crossing the chain is limited; its last block may be an exact suffix (mixed plan).
    const size_t small = std::max<size_t>(4, words / 5);
    const auto limited = run(fixture, *data, end, {small, workers}, ref);
    assert(limited.exact_blocks < limited.blocks || limited.blocks == 1);
    // Capped block lengths: the rule only ever takes a suffix that respects the cap.
    const auto capped = run(fixture, *data, end, {first_exact + 1, workers, true, 64}, ref);
    assert(capped.blocks > 1);
    // The rule disabled: limited blocks everywhere, same certified error.
    const auto disabled = run(fixture, *data, end, {first_exact + 1, workers, false}, ref);
    assert(disabled.exact_blocks == 0);
    printf("%-18s [%llu,%llu) W%-2u span %zu words (actual %zu): exact from %zu limbs; at %zu limbs %u blocks/%u exact; "
           "capped %u/%u%s PASS\n",
           label, (unsigned long long)def.begin, (unsigned long long)end, workers, words, actual, first_exact, small,
           limited.blocks, limited.exact_blocks, capped.blocks, capped.exact_blocks, capped.exact_value ? " (exact value)" : "");
    fflush(stdout);
}
// Limited chains with many blocks: the tight root-anchored attenuation lowers every kept precision,
// so the certified error is checked against the exact finite sum across block sizes.
void chain(Fixture &fixture, const char *label, const FormulaDef &def, uint64_t end, size_t fractional, unsigned workers,
           double block_bits) {
    auto data = std::make_unique<DataFormula>();
    assert(data_formula_prepare(def, end, *data) && data->contraction);
    Ref ref;
    reference(data->def, def.begin, end, ref);
    const auto got = run(fixture, *data, end, {fractional, workers, true, 0, block_bits}, ref);
    assert(got.blocks > 2 && got.exact_blocks <= 1);
    printf("%-18s [%llu,%llu) W%-2u %zu limbs: %u limited blocks (%u exact), attenuation %llu bits, direction %d PASS\n", label,
           (unsigned long long)def.begin, (unsigned long long)end, workers, fractional, got.blocks, got.exact_blocks,
           (unsigned long long)data->attenuation_bits(end), data->ratio_direction);
    fflush(stdout);
}
} // namespace
int main() {
    sbn3_formula_def atanh{};
    assert(sbn3_atanh_series_definition(1, 3, SBN3_ARCCOTH_ACCELERATED, &atanh) == SBN3_SUPPORTED);
    const FormulaDef accelerated = from_public(atanh);
    for (unsigned workers : {1u, 3u, 16u}) {
        Fixture fixture(workers, false);
        crossing(fixture, "e", euler_definition(), 2500, workers);
        crossing(fixture, "exp(1/2)", exp_reciprocal_definition(2), 2000, workers);
        crossing(fixture, "sin(1)", sine(), 1200, workers);
        crossing(fixture, "exp(-1)", exp_minus_one_definition(), 1500, workers);
        crossing(fixture, "zeta(3) AZ", zeta3(), 300, workers);
        crossing(fixture, "binary log r=3", binary_log_definition(3), 900, workers);
        chain(fixture, "zeta(3) AZ", zeta3(), 4000, 600, workers, 64.0 * 64);
        chain(fixture, "atanh(1/3) accel", accelerated, 3000, 140, workers, 64.0 * 48);
        chain(fixture, "chudnovsky", chudnovsky_definition(), 1500, 1090, workers, 64.0 * 96);
        chain(fixture, "binary log r=3", binary_log_definition(3), 6000, 270, workers, 64.0 * 40);
    }
    puts("precision planner: exact/bounded crossing, exact values, mixed plans, three recipes, certified error, "
         "threads, rebind, no allocation PASS");
}
