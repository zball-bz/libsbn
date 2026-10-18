// SPDX-License-Identifier: LGPL-3.0-or-later
#pragma once
#include "planning/subtree.hpp"
#include "series/precision_policy.hpp"

namespace sbn::v3::planning {
struct PrecisionStorage {
    size_t lane_words = 0, x_words = 0, y_words = 0, tail_words = 0;
    size_t finite_work_offset = 0;
};
struct PrecisionMergePlan {
    ProductProgramPlan multiply{},contraction{};
    size_t work_offset=0,work_bytes=0,alignment=128;
    size_t fast_work_bytes=0,compact_work_bytes=0,finite_pool_bytes=0;
    unsigned original_batch[2]{};
};
// Stable-address query output: finite callbacks reference block.exact_words and
// the caller's immutable PsrSpec. Do not copy/move a prepared episode.
struct PrecisionEpisode {
    const series::PsrSpec *spec = nullptr;
    series::PrecisionBlock block{};
    NativeSubtree finite{};
    PrecisionMergePlan merge[2]{};
    bool direct_root = false;
    PrecisionStorage storage[2]{};
    BlockSummary summary{}; // COMPLETE finite + tail + consuming merge
};
// result0 = shared T/D and merge X lane, result1 = U and merge Y,
// result2 = adjacent tail N/D pair; workspace = max(finite phase, merge phase).
// Runtime/engine state and these caller-owned descriptors are in fixed_bytes.
SummaryStatus summarize_precision_episode(const series::PsrSpec &, uint64_t begin,
                                          PrecisionEpisode &, series::ProductPlanMemo * = nullptr) noexcept;
// Query-time realization inside the already selected suffix envelope. Enlarges
// only execution batches that fit; arithmetic identity/value lanes stay fixed.
void realize_precision_episode(PrecisionEpisode &,Decomposition,const SlotEnvelope &) noexcept;

struct PrecisionState {
    const series::PsrSpec *owner = nullptr;
    sbn3_series_values values{};
    uint64_t begin = 0;
    bool ready = false;
};
struct PrecisionBinding {
    const PrecisionEpisode *episode = nullptr;
    unsigned decomposition = 0;
    sbn3_arena *arena = nullptr;
    sbn3_team *team = nullptr;
    sbn3_lease region{}, metadata{}, finite_prepared{}, finite_work{};
    SlotEnvelope envelope{};
    SlotLayout layout{};
    series::FiniteBinding finite{};
    bool used = false;
};
// Explicit allocation-free preparation into one already prepared, retained
// region. The shared envelope must contain the selected candidate. The caller
// keeps region and spec/episode alive through release; State headers survive
// release, but their words are owned by that region, not this binding.
void prepare_precision_episode(const PrecisionEpisode &, Decomposition, const SlotEnvelope &,
                               sbn3_arena &, sbn3_team &, const sbn3_lease &, PrecisionBinding &) noexcept;
void execute_precision_episode(PrecisionBinding &, PrecisionState &) noexcept;
void release_precision_episode(PrecisionBinding &) noexcept;
} // namespace sbn::v3::planning
