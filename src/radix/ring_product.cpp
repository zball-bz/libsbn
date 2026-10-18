#include "radix/ring_product.hpp"
#include "radix/programs.hpp"
#include "product/cost_model.hpp"
#include "product/backend.hpp"
#include "product/root_prepare_cost.hpp"
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
           uint64_t generation, Queried &out, bool cached = true,unsigned prime_batch=0) noexcept {
    sbn3_mul_options o{};
    o.workers = workers;
    o.prime_count = np;
    o.trunk_bits = T;
    o.algorithm = algorithm;
    o.borrow_output = 2;
    o.prime_batch=algorithm==SBN3_MUL_BAILEY?prime_batch:0;
    sbn3_product_request r{};
    r.kind = SBN3_PRODUCT_MUL;
    r.a_limbs = common;
    r.b_limbs = fresh;
    r.cyclic_limbs = ring;
    if (sbn3_product_query(&r, &o, &out.producer, &out.producer_info) != SBN3_SUPPORTED)
        return false;
    if(!cached){out.consumer=out.producer;out.consumer_info=out.producer_info;out.future={};return true;}
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
bool ring_plan(size_t fresh, size_t common, size_t minimum_ring, unsigned workers, RingPlan &out,
               PlanTranscript *transcript, uint64_t applications) noexcept {
    out = {};
    const bool deep = fresh > native_policy::small_model_max_words;
    Queried q{};
    const bool cached=applications!=1;
    struct Candidate {unsigned np,algorithm,batch;int T;size_t ring;double cost;};
    Candidate candidates[7*4*2];unsigned count=0;
    auto period=[&](int T){size_t n=2*size_t(T);while(n<minimum_ring)n*=2;return n;};
    auto eligible=[](size_t ring,unsigned algorithm){return
        !(algorithm==SBN3_MUL_FLAT&&ring>(size_t(1)<<19))&&
        !(algorithm==SBN3_MUL_BAILEY&&ring<2048);};
    auto price=[&](const sbn3_mul_info &i){
        const double ordinary=cost_model::cyclic_product(i,deep).nanoseconds;
        const double apply=cached?cost_model::cached_share(ordinary):ordinary;
        const double prepare=root_prepare_cost(i)+(cached?cost_model::prepare_share(ordinary):0.);
        return apply+(applications?prepare/double(applications):0.);
    };
    auto materialize=[&](const Candidate &c){
        if(!query(fresh,common,c.ring,c.np,c.T,c.algorithm,workers,1,q,cached,c.batch))return false;
        const double ordinary=cost_model::cyclic_product(q.consumer_info.mul,deep).nanoseconds;
        out={};out.enabled=true;out.cached=cached;out.np=c.np;out.algorithm=c.algorithm;
        out.workers=workers;out.prime_batch=c.batch;out.trunk_bits=c.T;out.ring=c.ring;
        out.common_limbs=common;out.fresh_limbs=fresh;
        out.predicted_ns=cached?cost_model::cached_share(ordinary):ordinary;
        out.prepare_ns=root_prepare_cost(q.producer_info.mul)+(cached?cost_model::prepare_share(ordinary):0.);
        fill(out,q);return true;
    };
    // [1][np:7][algorithm:4][batch:6][reserved:6][T:12], plus the resource check.
    auto check=[&]{return PlanTranscript::check(fresh*3+common,minimum_ring,workers,
        out.enabled?out.ring+31*(out.table_bytes+31*(out.work_bytes+31*out.spectrum_bytes))+uint64_t(out.cached):0);};
    if(transcript&&transcript->replay){
        const uint64_t e=transcript->next();const unsigned np=unsigned(e>>1&127),algorithm=unsigned(e>>8&15),batch=unsigned(e>>12&63);
        const int T=int(e>>24&4095);
        if((e&1)&&np>=4&&np<=10&&batch<=np&&T>=24*int(np)-32&&T<=24*int(np)-8&&
           (algorithm==SBN3_MUL_FLAT||algorithm==SBN3_MUL_BAILEY)&&eligible(period(T),algorithm)&&
           materialize({np,algorithm,batch,T,period(T),0.})&&check()==e>>product_choice_bits)return true;
        out={};++transcript->searched;
    }
    for(unsigned np:{4u,5u,6u,7u,8u,9u,10u})for(int T=24*int(np)-8;T>=24*int(np)-32;T-=8)
        for(unsigned algorithm:{unsigned(SBN3_MUL_FLAT),unsigned(SBN3_MUL_BAILEY)}){
            const size_t ring=period(T);if(!eligible(ring,algorithm))continue;
            sbn3_mul_options options{};options.workers=workers;options.prime_count=np;options.trunk_bits=T;
            options.algorithm=algorithm;options.borrow_output=2;
            options.prime_batch=algorithm==SBN3_MUL_BAILEY?(tree_policy().ring_prime_batch?tree_policy().ring_prime_batch:deep?1:0):0;
            sbn3_product_request request{};request.kind=SBN3_PRODUCT_MUL;request.a_limbs=common;request.b_limbs=fresh;request.cyclic_limbs=ring;
            const auto *backend=backend_lookup(np);
            const auto quote=cached?backend->geometry_query_cached_a:backend->geometry_query;
            sbn3_mul_info info{};
            if(quote){if(quote(request,options,info)!=SBN3_SUPPORTED)continue;}
            else{if(!query(fresh,common,ring,np,T,algorithm,workers,1,q,cached,options.prime_batch))continue;info=q.consumer_info.mul;}
            candidates[count++]={np,algorithm,options.prime_batch,T,ring,price(info)};
        }
    // Only the winner receives identities, scales and a future spectrum
    // descriptor. A failed full admission tries the next unchanged score.
    for(unsigned attempt=0;attempt<count;++attempt){unsigned best=count;double cost=INFINITY;
        for(unsigned j=0;j<count;++j)if(candidates[j].cost<cost){best=j;cost=candidates[j].cost;}
        if(best==count)break;
        if(materialize(candidates[best]))break;
        candidates[best].cost=INFINITY;
    }
    if (transcript && !transcript->replay)
        transcript->record(out.enabled ? 1 | uint64_t(out.np) << 1 | uint64_t(out.algorithm) << 8 | uint64_t(out.prime_batch)<<12 | uint64_t(out.trunk_bits) << 24 : 0,
                           check());
    return out.enabled;
}
void ring_replay(const RingPlan &p, uint64_t generation, RingStage &stage) noexcept {
    Queried q{};
    RingPlan check = p;
    require(p.enabled && query(p.fresh_limbs, p.common_limbs, p.ring, p.np, p.trunk_bits, p.algorithm, p.workers, generation, q,p.cached,p.prime_batch),
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
    if(p.cached){
        b.spectrum_lease = arena->acquire(pool_offset, align_to(p.spectrum_bytes, 4096));
        sbn3_spectrum_reserve_plan(&stage.producer, SBN3_SPECTRUM_COLUMNS, stage.future.generation, arena, &b.spectrum_lease,
                                   &b.spectrum);
    }
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
    if(p.cached)sbn3_spectrum_compute(b.consumers[0], b.spectrum, {common, p.common_limbs});
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
