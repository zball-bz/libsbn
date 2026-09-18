#pragma once
// Wrap-around splits in the NTT band: C = rail * fraction mod (2^(64 ring) - 1) through the public product
// service, with the transform of the rail power cached (the transform-only interface of the y-cruncher
// notes). Product bindings own their workspace and must be bound by the team's owner while the team is
// idle, so the tree binds one stage at a time inside a pool of the radix binding's storage: the spectrum
// of the stage's rail power and one cached consumer per worker group.
#include "sbn3/product.h"
#include <stddef.h>
#include <stdint.h>
namespace sbn::v3::radix {
inline constexpr unsigned ring_max_groups = 8;
struct RingPlan {
    bool enabled = false;
    unsigned np = 0, algorithm = 0, workers = 1;
    int trunk_bits = 0;
    size_t ring = 0, common_limbs = 0, fresh_limbs = 0;
    size_t table_bytes = 0, work_bytes = 0, work_alignment = 64, output_limbs = 0;
    size_t spectrum_bytes = 0, spectrum_alignment = 64;
    double predicted_ns = 0;
    // Pool bytes of `groups` consumers and the spectrum, every part aligned to its own requirement.
    size_t pool_bytes(unsigned groups) const noexcept;
};
// The cheapest supported (prime count, trunk width, algorithm) ring of at least minimum_ring limbs by the
// product cost model. False when nothing is supported.
bool ring_plan(size_t fresh_limbs, size_t common_limbs, size_t minimum_ring, unsigned workers, RingPlan &) noexcept;
// The plans of a ring stage, reproduced from the pinned parameters (bind time; fatal when they moved).
struct RingStage {
    sbn3_mul_plan producer{}, consumer{};
    sbn3_spectrum_desc future{};
};
void ring_replay(const RingPlan &, uint64_t generation, RingStage &) noexcept;
// One stage bound inside the pool (owner thread, idle team).
struct RingBound {
    sbn3_spectrum *spectrum = nullptr;
    sbn3_lease spectrum_lease{};
    sbn3_mul_binding *consumers[ring_max_groups]{};
    sbn3_lease tables[ring_max_groups]{}, work[ring_max_groups]{};
    unsigned groups = 0;
};
void ring_bind(RingBound &, const RingPlan &, const RingStage &, unsigned groups, sbn3_arena *, size_t pool_offset,
               size_t pool_bytes, sbn3_team *, const uint64_t *common) noexcept;
void ring_unbind(RingBound &, sbn3_arena *) noexcept;
} // namespace sbn::v3::radix
