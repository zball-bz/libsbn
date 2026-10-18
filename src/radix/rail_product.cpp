#include "radix/rail_product.hpp"
#include "common/checked.hpp"
#include "product/cost_model.hpp"
#include <algorithm>
namespace sbn::v3::radix {
bool rail_product_plan(size_t fresh_limbs, size_t common_limbs, size_t minimum_ring, RailProductPlan &out,
                       uint64_t applications,size_t minimum_fresh) noexcept {
    out = {};
    if (fresh_limbs < minimum_fresh || minimum_ring > rail_product_max_ring || !common_limbs)
        return false;
    // A shorter wide-codec transform may need much more preparation than
    // a classic 16-bit shape. Rank complete finite use, not point count.
    pq16::Shape best{};double cost=INFINITY;
    auto consider=[&](pq16::Shape s){
        if(!s.nfull||pq16::cyclic_period(s)<minimum_ring||
           !pq16::cyclic_supported(s,common_limbs,fresh_limbs))return;
        const double ordinary=pq16::native_cost(s,common_limbs,fresh_limbs);
        const double price=cost_model::cached_share(ordinary)+(applications?
            (.27*double(pq16::table_bytes(s))+cost_model::prepare_share(ordinary))/double(applications):0.);
        if(price<cost){best=s;cost=price;}
    };
    for (unsigned bits = 16; bits <= 20; ++bits) {
        const auto s = pq16::cyclic_shape(minimum_ring, bits);
        consider(s);
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
    // Cached cyclic input owns its spectrum separately. CT/PFA execution
    // uses one 2*N-double work plane, at most 48 doubles of padding.
    out.scratch_bytes = size_t(16) * best.nfull + 512;
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
    out.scratch_bytes = pq16::cached_scratch_bytes(s, common_limbs, fresh_limbs) + 128;
    return true;
}
void rail_product_reserve(RailProduct &p, const RailProductPlan &plan, Frame &storage) noexcept {
    require(plan.enabled, SBN3_FATAL_ARGUMENT, "rail product plan");
    p = {};
    p.plan = plan;
    const size_t before = storage.used();
    p.tables = pq16::prepare(storage, plan.shape);
    require(storage.used() - before <= plan.table_bytes, SBN3_FATAL_WORKSPACE, "rail product tables",
            storage.used() - before, plan.table_bytes);
    auto *cache = static_cast<double *>(storage.allocate(plan.cache_bytes, 64));
    p.cache = cache;
}
void rail_product_forward(RailProduct &p,const uint64_t *common) noexcept {
    require(p.tables&&p.cache&&!p.common,SBN3_FATAL_ARGUMENT,"rail cache completion");
    p.common=common;
    pq16::forward_spectrum(const_cast<double *>(p.cache),common,p.plan.common_limbs,*p.tables,nullptr);
}
void rail_product_prepare(RailProduct &p,const RailProductPlan &plan,Frame &storage,const uint64_t *common) noexcept {
    rail_product_reserve(p,plan,storage);rail_product_forward(p,common);
}
bool rail_product_square_supported(const RailProductPlan &p,size_t n) noexcept {
    return p.enabled&&p.ring&&p.common_limbs==n&&2*n<=p.ring&&
        pq16::supported(p.shape,n,n,1)&&pq16::cyclic_supported(p.shape,n,n);
}
bool rail_product_prefix_supported(const RailProductPlan &p,size_t n) noexcept {
    return p.enabled&&p.ring&&n&&n<=p.fresh_limbs&&p.common_limbs+n<=p.ring&&
        pq16::cyclic_supported(p.shape,p.common_limbs,n);
}
void rail_product_prefix(const RailProduct &p,Frame &work,const uint64_t *fresh,size_t n,uint64_t *out) noexcept {
    require(p.common&&rail_product_prefix_supported(p.plan,n),SBN3_FATAL_MATH,"rail exact prefix");
    pq16::cyclic_multiply(out,p.common,p.plan.common_limbs,fresh,n,false,p.cache,*p.tables,work,nullptr,p.plan.common_limbs+n);
}
void rail_product_square(const RailProduct &p,Frame &work,uint64_t *out) noexcept {
    require(p.common&&rail_product_square_supported(p.plan,p.plan.common_limbs),SBN3_FATAL_MATH,"rail cached square");
    // The complete square fits below the period. Prefix closing is exact,
    // needs no signed linear-tail storage, and fuses the cached copy with
    // pointwise/inverse leaves in the CT backend.
    pq16::cyclic_multiply(out,p.common,p.plan.common_limbs,p.common,p.plan.common_limbs,true,p.cache,*p.tables,
                          work,nullptr,2*p.plan.common_limbs);
}
bool rail_product_try_window(const RailProduct &p,Frame &work,const uint64_t *fresh,size_t live,
                             size_t origin,size_t words,uint64_t *residue,uint64_t *out) noexcept {
    const auto &q=p.plan;
    if(!q.ring||q.shape.recipe!=pq16::Recipe::CooleyTukeyPQ||q.shape.bits<=16||origin<2)return false;
    require(live&&live<=q.fresh_limbs&&origin+words<=q.ring,SBN3_FATAL_ARGUMENT,"rail output window");
    const size_t high=live+q.common_limbs>q.ring?live+q.common_limbs-q.ring:0;
    if(high>=origin)return false;
    pq16::cyclic_tail_multiply(residue,p.common,q.common_limbs,fresh,live,p.cache,*p.tables,work,origin-1);
    // The tail's signed modular error at origin-1 is <2^40. An interior
    // guard word proves that error cannot change the next word. Its exact
    // residue is also nonzero: the wrapped high part, ending below it,
    // therefore did not carry through the gap into the requested window.
    constexpr uint64_t error=uint64_t(1)<<40;
    const uint64_t guard=residue[origin-1];
    if(guard<error||guard>=UINT64_MAX-error)return false;
    memcpy(out,residue+origin,words*sizeof(uint64_t));
    return true;
}
void rail_product_execute(const RailProduct &p, Frame &work, sbn3_team_scope *self, const uint64_t *fresh,
                          uint64_t *out,size_t live_limbs) noexcept {
    const size_t live=live_limbs?live_limbs:p.plan.fresh_limbs;
    require(live<=p.plan.fresh_limbs,SBN3_FATAL_ARGUMENT,"radix bounded FFT input");
    if (p.plan.ring)
        pq16::cyclic_multiply(out, p.common, p.plan.common_limbs, fresh, live, false, p.cache, *p.tables,
                              work, self);
    else
        pq16::apply_spectrum(out, p.cache, p.common, p.plan.common_limbs, fresh, live, false, *p.tables, work,
                             self);
}
} // namespace sbn::v3::radix
