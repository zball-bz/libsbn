// SPDX-License-Identifier: LGPL-3.0-or-later
#include "planning/subtree.hpp"
#include "common/checked.hpp"

namespace sbn::v3::planning {
SummaryStatus select_resident_suffix(std::span<const BlockSummary> blocks, ResidentBudget budget,
                                    std::span<Decomposition> choices, ResidentFrontier &out) noexcept {
    if (choices.size() < blocks.size()) return SummaryStatus::capacity;
    if (budget.fixed_bytes > budget.total_bytes) return SummaryStatus::capacity;
    // Validate first, so malformed input cannot be mistaken for a budget boundary.
    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto &b = blocks[i];
        if (b.range.begin >= b.range.end || (i && blocks[i-1].range.end != b.range.begin))
            return SummaryStatus::invalid;
        for (const auto &c : b.candidate) if (c.supported) {
            SlotLayout ignored{};
            const auto status = layout(c.slots, ignored);
            if (status != SummaryStatus::ok) return status;
        }
    }
    ResidentFrontier result{};
    result.first = blocks.size();
    result.total_bytes = budget.fixed_bytes;
    for (size_t i = 0; i < blocks.size(); ++i) choices[i] = Decomposition::external;
    // Sizing is independent of the budget: choose the smaller envelope extension,
    // not "ordinary if it fits". Otherwise MORE RAM can admit FEWER blocks when
    // the two decompositions need different slot proportions. Ties prefer ordinary.
    for (size_t i = blocks.size(); i-- > 0;) {
        bool have = false;
        SlotEnvelope best{};
        SlotLayout best_layout{};
        unsigned selected = 0;
        for (unsigned d = 0; d < 2; ++d) if (blocks[i].candidate[d].supported) {
            SlotEnvelope combined{};
            auto status = join(result.envelope, blocks[i].candidate[d].slots, combined);
            if (status != SummaryStatus::ok) return status;
            SlotLayout placed{};
            status = layout(combined, placed);
            if (status != SummaryStatus::ok) return status;
            if (!have || placed.bytes < best_layout.bytes) {
                have = true; best = combined; best_layout = placed; selected = d;
            }
        }
        if (!have || best_layout.bytes > budget.total_bytes - budget.fixed_bytes) break;
        result.first = i;
        result.envelope = best;
        result.layout = best_layout;
        result.total_bytes = budget.fixed_bytes + best_layout.bytes;
        choices[i] = Decomposition(selected);
    }
    // Realization spends available headroom on ordinary decompositions, head to
    // tail. These expansions never remove a block from the admitted suffix.
    for (size_t i = result.first; i < blocks.size(); ++i) {
        const auto &ordinary = blocks[i].candidate[0];
        if (!ordinary.supported) continue;
        SlotEnvelope expanded{};
        auto status = join(result.envelope, ordinary.slots, expanded);
        if (status != SummaryStatus::ok) return status;
        SlotLayout placed{};
        status = layout(expanded, placed);
        if (status != SummaryStatus::ok) return status;
        if (placed.bytes <= budget.total_bytes - budget.fixed_bytes) {
            result.envelope = expanded; result.layout = placed;
            result.total_bytes = budget.fixed_bytes + placed.bytes;
            choices[i] = Decomposition::ordinary;
        }
    }
    out = result;
    return SummaryStatus::ok;
}

namespace {
SummaryStatus describe(const sbn3_series_info &info, ResidentCandidate &out) noexcept {
    SlotEnvelope slots{};
    for (unsigned i = 0; i < 3; ++i) {
        size_t bytes;
        if (!mul_size(info.output.limbs[i], sizeof(uint64_t), bytes)) return SummaryStatus::overflow;
        slots.slots[i] = {bytes, 64};
    }
    slots[Slot::metadata] = {info.plan_bytes, info.plan_alignment};
    slots[Slot::prepared] = {info.prepared_bytes, info.prepared_alignment};
    slots[Slot::workspace] = {info.workspace_bytes, info.workspace_alignment};
    SlotLayout checked{};
    const auto status = layout(slots, checked);
    if (status != SummaryStatus::ok) return status;
    out = {true, slots, info.schedule_id};
    return SummaryStatus::ok;
}
}
SummaryStatus summarize_native_subtree(const series::FiniteFormula &formula, sbn3_series_range range,
                                      unsigned need, const sbn3_series_options &options, NativeSubtree &out,
                                      series::ProductPlanMemo *memo) noexcept {
    NativeSubtree next{};
    next.summary.range = range;
    auto native_options = options;
    native_options.workspace_budget = 0;
    native_options.max_serial_prefix = 0;
    bool have = false;
    for (unsigned d = 0; d < 2; ++d) {
        const auto rc = series::finite_query_prefix(formula, range, need, native_options, d, next.plans[d], memo);
        if (rc == SBN3_UNSUPPORTED) continue;
        if (rc == SBN3_QUERY_CAPACITY) return SummaryStatus::capacity;
        const auto status = describe(next.plans[d].info, next.summary.candidate[d]);
        if (status != SummaryStatus::ok) return status;
        have = true;
    }
    if (!have) return SummaryStatus::unsupported;
    out = next;
    return SummaryStatus::ok;
}
} // namespace sbn::v3::planning
