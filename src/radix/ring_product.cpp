#include "radix/ring_product.hpp"
#include "product/cost_model.hpp"
#include "runtime/arena.hpp"
#include "common/checked.hpp"
#include <algorithm>
#include <math.h>
namespace sbn::v3::radix {
namespace {
constexpr size_t align_to(size_t x, size_t a) noexcept { return (x + a - 1) & ~(a - 1); }
struct Queried {
    sbn3_mul_plan producer{}, consumer{};
    sbn3_product_info producer_info{}, consumer_info{};
    sbn3_spectrum_desc future{};
};
bool query(size_t fresh, size_t common, size_t ring, unsigned np, int T, unsigned algorithm, unsigned workers,
           uint64_t generation, Queried &out) noexcept {
    sbn3_mul_options o{};
    o.workers = workers;
    o.prime_count = np;
    o.trunk_bits = T;
    o.algorithm = algorithm;
    o.borrow_output = 1;
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_MUL;
    r.a_limbs = common;
    r.b_limbs = fresh;
    r.cyclic_limbs = ring;
    if (sbn3_product_query(&r, &o, &out.producer, &out.producer_info) != SBN3_SUPPORTED)
        return false;
    if (sbn3_spectrum_query(&out.producer, SBN3_SPECTRUM_COLUMNS, generation, &out.future) != SBN3_SUPPORTED)
        return false;
    r.cached_a[0] = &out.future;
    return sbn3_product_query(&r, &o, &out.consumer, &out.consumer_info) == SBN3_SUPPORTED;
}
void fill(RingPlan &p, const Queried &q) noexcept {
    const auto &i = q.consumer_info.mul;
    p.table_bytes = std::max<size_t>(i.table_bytes, 128);
    p.work_bytes = std::max<size_t>(i.workspace_bytes, 128);
    p.work_alignment = std::max<size_t>(i.workspace_alignment, 64);
    p.output_limbs = sbn3_mul_output_capacity(&i);
    p.spectrum_bytes = q.future.storage_bytes;
    p.spectrum_alignment = std::max<size_t>(q.consumer_info.spectrum_alignment, 128);
}
} // namespace
size_t RingPlan::pool_bytes(unsigned groups) const noexcept {
    size_t at = align_to(spectrum_bytes, 4096);
    for (unsigned g = 0; g < groups; ++g) {
        at = align_to(at, 4096) + align_to(table_bytes, 4096);
        at = align_to(at, std::max<size_t>(work_alignment, 4096)) + align_to(work_bytes, 4096);
    }
    return align_to(at, 4096);
}
bool ring_plan(size_t fresh, size_t common, size_t minimum_ring, unsigned workers, RingPlan &out) noexcept {
    out = {};
    const bool deep = fresh > native_policy::small_model_max_words;
    Queried q{};
    double best = INFINITY;
    for (unsigned np : {5u, 6u, 8u, 10u})
        for (int T = 24 * int(np) - 8; T >= 24 * int(np) - 32; T -= 8)
            for (unsigned algorithm : {unsigned(SBN3_MUL_FLAT), unsigned(SBN3_MUL_BAILEY)}) {
                size_t ring = 2 * size_t(T);
                while (ring < minimum_ring)
                    ring *= 2;
                if ((algorithm == SBN3_MUL_FLAT && ring > (size_t(1) << 19)) || (algorithm == SBN3_MUL_BAILEY && ring < 2048))
                    continue;
                if (!query(fresh, common, ring, np, T, algorithm, workers, 1, q))
                    continue;
                const double cost = cost_model::cached_share(cost_model::cyclic_product(q.consumer_info.mul, deep).nanoseconds);
                if (!(cost < best))
                    continue;
                best = cost;
                out = {};
                out.enabled = true;
                out.np = np;
                out.algorithm = algorithm;
                out.workers = workers;
                out.trunk_bits = T;
                out.ring = ring;
                out.common_limbs = common;
                out.fresh_limbs = fresh;
                out.predicted_ns = cost;
                fill(out, q);
            }
    return out.enabled;
}
void ring_replay(const RingPlan &p, uint64_t generation, RingStage &stage) noexcept {
    Queried q{};
    RingPlan check = p;
    require(p.enabled && query(p.fresh_limbs, p.common_limbs, p.ring, p.np, p.trunk_bits, p.algorithm, p.workers, generation, q),
            SBN3_FATAL_MATH, "radix ring replay");
    fill(check, q);
    require(check.table_bytes == p.table_bytes && check.work_bytes == p.work_bytes && check.output_limbs == p.output_limbs &&
                check.spectrum_bytes == p.spectrum_bytes,
            SBN3_FATAL_MATH, "radix ring resources");
    stage.producer = q.producer;
    stage.consumer = q.consumer;
    stage.future = q.future;
}
void ring_bind(RingBound &b, const RingPlan &p, const RingStage &stage, unsigned groups, sbn3_arena *arena, size_t pool_offset,
               size_t pool_bytes, sbn3_team *team, const uint64_t *common) noexcept {
    require(groups && groups <= ring_max_groups && p.pool_bytes(groups) <= pool_bytes, SBN3_FATAL_WORKSPACE, "radix ring pool",
            p.pool_bytes(groups), pool_bytes);
    b = {};
    b.groups = groups;
    const uintptr_t base = reinterpret_cast<uintptr_t>(arena->base) + pool_offset;
    size_t at = 0;
    b.spectrum_lease = arena->acquire(pool_offset, align_to(p.spectrum_bytes, 4096));
    sbn3_spectrum_reserve_plan(&stage.producer, SBN3_SPECTRUM_COLUMNS, stage.future.generation, arena, &b.spectrum_lease,
                               &b.spectrum);
    at = align_to(p.spectrum_bytes, 4096);
    for (unsigned g = 0; g < groups; ++g) {
        at = align_to(at, 4096);
        b.tables[g] = arena->acquire(pool_offset + at, align_to(p.table_bytes, 4096));
        at += align_to(p.table_bytes, 4096);
        at = align_to(base + at, std::max<size_t>(p.work_alignment, 4096)) - base;
        b.work[g] = arena->acquire(pool_offset + at, align_to(p.work_bytes, 4096));
        at += align_to(p.work_bytes, 4096);
        sbn3_product_bind(&stage.consumer, arena, &b.tables[g], &b.work[g], team, b.spectrum, nullptr, &b.consumers[g]);
    }
    sbn3_spectrum_compute(b.consumers[0], b.spectrum, {common, p.common_limbs});
}
void ring_unbind(RingBound &b, sbn3_arena *arena) noexcept {
    for (unsigned g = 0; g < b.groups; ++g) {
        sbn3_mul_unbind(b.consumers[g]);
        arena->release(b.work[g]);
        arena->release(b.tables[g]);
    }
    if (b.spectrum) {
        sbn3_spectrum_release(b.spectrum);
        arena->release(b.spectrum_lease);
    }
    b = {};
}
} // namespace sbn::v3::radix
