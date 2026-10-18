// SPDX-License-Identifier: LGPL-3.0-or-later
#include "formula_reference.hpp"
#include "planning/precision.hpp"
#include <memory>
using namespace formula_reference;
using namespace sbn::v3::planning;

static uint64_t decay(const void *p, uint64_t a) {
    return static_cast<const DataFormula *>(p)->attenuation_bits(a);
}
static void check_ratio(const Ref &exact, const sbn3_series_values &v, size_t fractional, uint64_t error) {
    ref_int n, d, lhs, rhs, bound;
    ref_inits(n, d, lhs, rhs, bound, nullptr);
    const auto &N = v.value[0]; const auto &D = v.value[1];
    ref_import(n, N.mantissa.size, -1, 8, 0, 0, N.mantissa.data);
    ref_import(d, D.mantissa.size, -1, 8, 0, 0, D.mantissa.data);
    if (N.mantissa.negative) ref_neg(n, n);
    assert(!D.mantissa.negative && D.mantissa.size);
    const int64_t en = N.mantissa.size ? N.exponent2 : 0, ed = D.exponent2, er = exact.exponent;
    const int64_t low = std::min({en, er+ed, ed});
    ref_mul(lhs, n, exact.d); ref_mul_2exp(lhs, lhs, size_t(en-low));
    ref_mul(rhs, exact.t, d); ref_mul_2exp(rhs, rhs, size_t(er+ed-low));
    ref_sub(lhs, lhs, rhs); ref_abs(lhs, lhs); ref_mul_2exp(lhs, lhs, 64*fractional);
    ref_mul(bound, d, exact.d); ref_mul_ui(bound, bound, error); ref_mul_2exp(bound, bound, size_t(ed-low));
    assert(ref_cmp(lhs, bound) < 0);
    ref_clears(n, d, lhs, rhs, bound, nullptr);
}
static void one(const char *label, const FormulaDef &def, uint64_t end, size_t fractional,
                unsigned workers, uint64_t max_terms = 0) {
    auto data = std::make_unique<DataFormula>();
    assert(data_formula_prepare(def, end, *data) && data->contraction);
    const PsrSpec spec{data_finite_formula(*data), {def.begin, end}, {workers, 8, 0, 0, 0},
                      {data.get(), decay, 4, max_terms, 0}, fractional};
    Ref reference_value; reference(data->def, def.begin, end, reference_value);
    constexpr size_t limit = 96; // Test fixture capacity, not a production controller limit.
    auto episodes = std::make_unique<PrecisionEpisode[]>(limit);
    std::array<BlockSummary, limit> summaries{};
    std::array<Decomposition, limit> choices{};
    auto memo = std::make_unique<ProductPlanMemo>();
    SlotEnvelope complete{};
    size_t count = 0;
    uint64_t begin = def.begin;
    while (begin < end) {
        assert(count < limit);
        auto &e = episodes[count];
        allocation_watch_start();
        const auto rc = summarize_precision_episode(spec, begin, e, memo.get());
        assert(!allocation_watch_stop());
        assert(rc == SummaryStatus::ok && e.block.range.begin == begin && e.block.range.end > begin);
        summaries[count] = e.summary;
        for (unsigned d = 0; d < 2; ++d) {
            assert(e.summary.candidate[d].supported);
            assert(join(complete, e.summary.candidate[d].slots, complete) == SummaryStatus::ok);
        }
        begin = e.block.range.end; ++count;
    }
    // Both clients use precisely the shared policy, including exact suffixes.
    PsrInfo prior{};
    allocation_watch_start(); assert(psr_query(spec, prior) == SBN3_SUPPORTED); assert(!allocation_watch_stop());
    assert(prior.blocks == count);
    unsigned exact = 0;
    for (size_t i = 0; i < count; ++i) exact += episodes[i].block.exact_words != 0;
    assert(prior.exact_blocks == exact);
    if (count == 1) {
        assert(episodes[0].direct_root && complete[Slot::result0].bytes == 0 && complete[Slot::result1].bytes == 0);
    }
    SlotLayout shape{};
    assert(layout(complete, shape) == SummaryStatus::ok && shape.bytes < (size_t(16)<<20));
    constexpr size_t fixed = size_t(32)<<20; // Test context, descriptors, tables, stacks.
    ResidentFrontier frontier{};
    allocation_watch_start();
    assert(select_resident_suffix(std::span(summaries.data(), count), {fixed+shape.bytes, fixed},
                                  std::span(choices.data(), count), frontier) == SummaryStatus::ok);
    assert(!allocation_watch_stop() && frontier.first == 0);
    Fixture f(workers, false, size_t(96)<<20);
    auto region = f.allocate(shape.bytes, shape.alignment);
    // Execute both the ordinary and compact finite-tree decompositions. The
    // accumulator pair stays inside the same grant and survives each release.
    for (unsigned mode = 0; mode < 2; ++mode) {
        PrecisionState tail{};
        for (size_t i = count; i-- > 0;) {
            PrecisionBinding binding{};
            allocation_watch_start();
            prepare_precision_episode(episodes[i], mode ? Decomposition::compact : choices[i], complete,
                                      *f.arena, *f.team, region, binding);
            execute_precision_episode(binding, tail);
            release_precision_episode(binding);
            assert(!allocation_watch_stop());
            assert(tail.ready && tail.begin == episodes[i].block.range.begin);
        }
        check_ratio(reference_value, tail.values, fractional, 2*count+1);
    }
    // Admission includes the tail pair, not just the subtree. Check a boundary
    // using exact bytes rather than a power-of-two or guessed scratch budget.
    size_t minimum = SIZE_MAX;
    for (unsigned d = 0; d < 2; ++d) {
        SlotLayout l{}; assert(layout(summaries[count-1].candidate[d].slots, l) == SummaryStatus::ok);
        minimum = std::min(minimum, l.bytes);
    }
    assert(minimum > 0);
    assert(select_resident_suffix(std::span(summaries.data(), count), {fixed+minimum-1, fixed},
                                  std::span(choices.data(), count), frontier) == SummaryStatus::ok && frontier.first == count);
    printf("OOC precision %-12s W%u terms=%llu precision=%zu blocks=%zu exact=%u region=%zu PASS\n",
           label, workers, (unsigned long long)(end-def.begin), fractional, count, exact, shape.bytes);
}
int main() {
    for (unsigned w : {1u, 4u}) {
        one("e exact", euler_definition(), 518, 80, w);
        one("e limited", euler_definition(), 1201, 24, w);
        one("exp(-1)", exp_minus_one_definition(), 618, 28, w, 73);
        one("pi", chudnovsky_definition(), 257, 180, w);
        one("pi larger", chudnovsky_definition(), 901, 600, w);
        one("binary log", binary_log_definition(3), 1301, 55, w);
    }
    // The new generator is not restricted by the resident controller's 128
    // stored blocks. This is a pure policy check, not a large arithmetic run.
    auto data = std::make_unique<DataFormula>();
    const auto def = binary_log_definition(3);
    assert(data_formula_prepare(def, 1501, *data));
    PsrSpec s{data_finite_formula(*data), {def.begin, 1501}, {1, 8, 0, 0, 0},
              {data.get(), decay, 4, 4, 0}, 80};
    size_t blocks = 0;
    for (uint64_t a = s.range.begin; a < s.range.end; ++blocks) {
        PrecisionBlock b{}; assert(precision_block_query(s, a, b) == SBN3_SUPPORTED); a = b.range.end;
    }
    assert(blocks > 128);
    puts("OOC precision: shared cuts, complete merge/tail grants, consuming replay, certified ratios and no allocations PASS");
}
