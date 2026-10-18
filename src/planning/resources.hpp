// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include <cstddef>
#include <cstdint>

namespace sbn::v3::planning {
// A reusable resident-region layout. Aliases within a kernel are already
// resolved by its quote; different roles here are physically disjoint.
enum class Slot : unsigned { result0, result1, result2, metadata, prepared, workspace, count };
inline constexpr unsigned slot_count = unsigned(Slot::count);
struct Storage {
    size_t bytes = 0, alignment = 1;
};
struct SlotEnvelope {
    Storage slots[slot_count]{};
    Storage &operator[](Slot s) noexcept { return slots[unsigned(s)]; }
    const Storage &operator[](Slot s) const noexcept { return slots[unsigned(s)]; }
};
struct SlotLayout {
    size_t offsets[slot_count]{};
    size_t bytes = 0, alignment = 1;
};
struct ResidentBudget {
    size_t total_bytes = 0;
    // Already counted once: context tables, Team/stacks, IO, metadata and any
    // other live grants OUTSIDE the described region. Zero total is not infinity.
    size_t fixed_bytes = 0;
};
enum class SummaryStatus { ok, invalid, overflow, capacity, unsupported };

// On invalid/overflow, destination is unchanged. All arithmetic is checked.
SummaryStatus layout(const SlotEnvelope &, SlotLayout &) noexcept;
SummaryStatus join(const SlotEnvelope &, const SlotEnvelope &, SlotEnvelope &) noexcept;
bool contains(const SlotEnvelope &capacity, const SlotEnvelope &required) noexcept;
SummaryStatus admit(const SlotEnvelope &, ResidentBudget, SlotLayout &, size_t &total_bytes) noexcept;
} // namespace sbn::v3::planning
