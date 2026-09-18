#pragma once
#include "sbn3/series.h"
#include "common/shared_preparation.hpp"
namespace sbn::v3::series {
// Consumer-owned, canonical words describing already chosen implementations.
// The scheduler stores/hashes them, without interpreting product backends.
struct StagePreparation {
    SharedPreparations shared{};
    uint64_t decisions[32]{};
};
// A consumer may materialize additional serial nodes when precision differs.
// Exact subtrees retain compact per-depth preparation. Coarse cuts are stored;
// a registered serial policy can recompute cuts inside its prepared envelope,
// without selecting kernels, querying resources, or allocating during execute.
struct ScheduleRefinement {
    const void *context = nullptr;
    bool (*split_serial)(const void *, sbn3_series_range, unsigned need) = nullptr;
    bool right_inline = false;
    // Private executor contract: parent output may cover consumed child
    // values. Caller-provided output capacities include the value-lane envelope.
    bool value_lanes = false;
    uint64_t split_policy_id = 0;
    uint64_t (*split_point)(const void *, sbn3_series_range, double fraction) = nullptr;
    // Bounds every descendant at this relative serial depth, including a
    // child with more than half the parent's term count. Called at prepare.
    sbn3_query_result (*serial_envelope)(const void *, sbn3_series_range, unsigned depth,
                                        unsigned need, uint64_t *max_terms, sbn3_series_shape *) = nullptr;
    // Optional immutable prepared data, accounted separately from per-stage
    // bytes. The normal resources callback must exclude this shared payload.
    sbn3_query_result (*prepared_resources)(const void *, const sbn3_series_stage *,
                                            sbn3_series_resources *, StagePreparation *) = nullptr;
    // Query/compile-only context; never retained for runtime serial splitting.
    const void *prepared_context = nullptr;
};
sbn3_query_result query_schedule(const sbn3_series_spec &, const sbn3_series_options &,
                                 const sbn3_series_oracle &, sbn3_series_info &,
                                 const ScheduleRefinement & = {});
// Replay only the already selected schedule. Resource identities/bytes are
// checked against the pure query; do not repeat the candidate search at bind.
sbn3_series_plan *prepare_schedule(const sbn3_series_spec &, const sbn3_series_options &,
                                   const sbn3_series_oracle &, const sbn3_series_info &, void *, size_t,
                                   const ScheduleRefinement & = {});
// Compile directly into an admitted metadata span. With one prefix candidate
// this performs the query and node emission together, retaining its decisions.
sbn3_query_result compile_schedule(const sbn3_series_spec &, const sbn3_series_options &,
                                   const sbn3_series_oracle &, void *, size_t, sbn3_series_info &,
                                   sbn3_series_plan *&, const ScheduleRefinement & = {});
// Disjoint preparation regions, independent of logical execution order.
void prepare_stages(const sbn3_series_plan *, sbn3_team *,
                    void (*)(void *, const sbn3_series_stage *, const sbn3_series_resources *), void *);
SharedPlacement shared_placement(const sbn3_series_plan *, size_t stage_index, unsigned slot = 0);
const uint64_t *stage_decisions(const sbn3_series_plan *, size_t stage_index);
} // namespace sbn::v3::series
