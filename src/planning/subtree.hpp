// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "planning/resources.hpp"
#include "series/arithmetic.hpp"
#include <span>

namespace sbn::v3::planning {
enum class Decomposition : unsigned { ordinary, compact, external };
struct ResidentCandidate {
    bool supported = false;
    SlotEnvelope slots{};
    uint64_t schedule_id = 0;
};
struct BlockSummary {
    sbn3_series_range range{};
    ResidentCandidate candidate[2]{};
};
struct ResidentFrontier {
    size_t first = 0; // [first, count) admitted; prefix is external
    SlotEnvelope envelope{};
    SlotLayout layout{};
    size_t total_bytes = 0; // region + fixed budget, including alignment
};
// Inputs are an ordered contiguous partition. Candidate slots must cover the
// COMPLETE sequential episode at each block: add its tail/merge needs before
// using this as a precision-block frontier. No overlap between episodes is assumed.
// O(count) scalar work; no allocation or backend calls. Decisions before first
// are external. A supported empty suffix means external execution is required,
// not that the whole operation is infeasible. All remaining capacity is explicit.
SummaryStatus select_resident_suffix(std::span<const BlockSummary>, ResidentBudget,
                                    std::span<Decomposition>, ResidentFrontier &) noexcept;

struct NativeSubtree {
    BlockSummary summary{};
    series::FinitePlan plans[2]{};
};
// Quotes one exact/limited finite subtree through the existing native planner.
// Does NOT include an enclosing precision-block accumulation/merge, runtime
// context or the caller-owned NativeSubtree records. These belong to the caller's
// envelope/fixed grant. An engine-owned table pool is likewise charged there.
// The whole-region selector, not workspace_budget/max_serial_prefix, decides
// admission: this queries precisely prefixes 0 and 1 with no workspace filter.
SummaryStatus summarize_native_subtree(const series::FiniteFormula &, sbn3_series_range,
                                      unsigned need, const sbn3_series_options &, NativeSubtree &,
                                      series::ProductPlanMemo * = nullptr) noexcept;
} // namespace sbn::v3::planning
