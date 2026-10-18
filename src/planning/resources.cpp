// SPDX-License-Identifier: LGPL-3.0-or-later
#include "planning/resources.hpp"
#include "common/checked.hpp"
#include <algorithm>

namespace sbn::v3::planning {
namespace {
bool valid(const Storage &s) noexcept {
    return s.alignment && !(s.alignment & (s.alignment - 1));
}
}
SummaryStatus layout(const SlotEnvelope &e, SlotLayout &out) noexcept {
    SlotLayout next{};
    for (unsigned i = 0; i < slot_count; ++i) {
        const auto &s = e.slots[i];
        if (!valid(s)) return SummaryStatus::invalid;
        if (!s.bytes) continue;
        if (!align_size(next.bytes, s.alignment, next.offsets[i]) ||
            !add_size(next.offsets[i], s.bytes, next.bytes)) return SummaryStatus::overflow;
        next.alignment = std::max(next.alignment, s.alignment);
    }
    if (!align_size(next.bytes, next.alignment, next.bytes)) return SummaryStatus::overflow;
    out = next;
    return SummaryStatus::ok;
}
SummaryStatus join(const SlotEnvelope &a, const SlotEnvelope &b, SlotEnvelope &out) noexcept {
    SlotEnvelope next{};
    for (unsigned i = 0; i < slot_count; ++i) {
        const auto &x = a.slots[i];
        const auto &y = b.slots[i];
        if (!valid(x) || !valid(y)) return SummaryStatus::invalid;
        next.slots[i] = {std::max(x.bytes, y.bytes),
                        std::max(x.bytes ? x.alignment : size_t(1), y.bytes ? y.alignment : size_t(1))};
    }
    SlotLayout checked{};
    const auto status = layout(next, checked);
    if (status == SummaryStatus::ok) out = next;
    return status;
}
bool contains(const SlotEnvelope &capacity, const SlotEnvelope &required) noexcept {
    for (unsigned i = 0; i < slot_count; ++i) {
        const auto &a = capacity.slots[i];
        const auto &b = required.slots[i];
        if (!valid(a) || !valid(b) || a.bytes < b.bytes || (b.bytes && a.alignment < b.alignment))
            return false;
    }
    return true;
}
SummaryStatus admit(const SlotEnvelope &e, ResidentBudget budget, SlotLayout &out, size_t &total) noexcept {
    SlotLayout next{};
    auto status = layout(e, next);
    if (status != SummaryStatus::ok) return status;
    size_t need;
    if (!add_size(budget.fixed_bytes, next.bytes, need)) return SummaryStatus::overflow;
    out = next;
    total = need;
    return need <= budget.total_bytes ? SummaryStatus::ok : SummaryStatus::capacity;
}
} // namespace sbn::v3::planning
