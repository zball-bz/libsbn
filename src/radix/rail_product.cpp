#include "radix/rail_product.hpp"
#include "common/checked.hpp"
#include <algorithm>
namespace sbn::v3::radix {
bool rail_product_plan(size_t fresh_limbs, size_t common_limbs, size_t minimum_ring, RailProductPlan &out) noexcept {
    out = {};
    if (fresh_limbs < rail_product_min_limbs || minimum_ring > rail_product_max_ring || !common_limbs)
        return false;
    // The shortest transform among the digit widths; equal lengths keep the narrower (better conditioned) digits.
    pq16::Shape best{};
    for (unsigned bits = 16; bits <= 20; ++bits) {
        const auto s = pq16::cyclic_shape(minimum_ring, bits);
        if (!s.nfull || pq16::cyclic_period(s) < minimum_ring || !pq16::cyclic_supported(s, common_limbs, fresh_limbs))
            continue;
        if (!best.nfull || s.nfull < best.nfull)
            best = s;
    }
    if (!best.nfull)
        return false;
    out.enabled = true;
    out.shape = best;
    out.ring = pq16::cyclic_period(best);
    out.fresh_limbs = fresh_limbs;
    out.common_limbs = common_limbs;
    out.table_bytes = (pq16::table_bytes(best) + 255) & ~size_t(127);
    out.cache_bytes = ((size_t(2) * best.nfull + 32) * sizeof(double) + 127) & ~size_t(127);
    out.scratch_bytes = size_t(32) * best.nfull + 4096;
    return true;
}
bool rail_linear_plan(size_t fresh_limbs, size_t common_limbs, RailProductPlan &out) noexcept {
    out = {};
    if (std::min(fresh_limbs, common_limbs) < rail_product_min_limbs / 2 || fresh_limbs + common_limbs > 131072)
        return false;
    const auto s = pq16::select(common_limbs, fresh_limbs, 1);
    if (!s.nfull || !pq16::supported(s, common_limbs, fresh_limbs, 1))
        return false;
    out.enabled = true;
    out.shape = s;
    out.ring = 0;
    out.fresh_limbs = fresh_limbs;
    out.common_limbs = common_limbs;
    out.table_bytes = (pq16::table_bytes(s) + 255) & ~size_t(127);
    out.cache_bytes = ((size_t(2) * s.nfull + 32) * sizeof(double) + 127) & ~size_t(127);
    out.scratch_bytes = std::max<size_t>(pq16::cached_scratch_bytes(s, common_limbs, fresh_limbs), size_t(32) * s.nfull) + 4096;
    return true;
}
void rail_product_prepare(RailProduct &p, const RailProductPlan &plan, Frame &storage, const uint64_t *common) noexcept {
    require(plan.enabled, SBN3_FATAL_ARGUMENT, "rail product plan");
    p = {};
    p.plan = plan;
    p.common = common;
    const size_t before = storage.used();
    p.tables = pq16::prepare(storage, plan.shape);
    require(storage.used() - before <= plan.table_bytes, SBN3_FATAL_WORKSPACE, "rail product tables",
            storage.used() - before, plan.table_bytes);
    auto *cache = static_cast<double *>(storage.allocate(plan.cache_bytes, 64));
    pq16::forward_spectrum(cache, common, plan.common_limbs, *p.tables, nullptr);
    p.cache = cache;
}
void rail_product_execute(const RailProduct &p, Frame &work, sbn3_team_scope *self, const uint64_t *fresh,
                          uint64_t *out) noexcept {
    if (p.plan.ring)
        pq16::cyclic_multiply(out, p.common, p.plan.common_limbs, fresh, p.plan.fresh_limbs, false, p.cache, *p.tables,
                              work, self);
    else
        pq16::apply_spectrum(out, p.cache, p.common, p.plan.common_limbs, fresh, p.plan.fresh_limbs, false, *p.tables, work,
                             self);
}
} // namespace sbn::v3::radix
