// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/newton.hpp"
#include "external/trace.hpp"
#include "external/owned_product.hpp"
#include "external/shared.hpp"
#include "algorithms/newton_contract.hpp"
#include "common/checked.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <utility>
namespace sbn::v3::external {
namespace {
bool add(uint64_t a,uint64_t b,uint64_t&out){return !__builtin_add_overflow(a,b,&out);}
bool region(size_t words,RegionLayout&out){size_t bytes;return mul_size(words,8,bytes)&&!region_layout(bytes,0,out);}
}
planning::ExternalPlanStatus query_rsqrt_rung(uint64_t a,size_t m,size_t n,unsigned workers,
    const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,RsqrtRungPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    if(!a||m<2||m>=n||n-m>m-1||n>(size_t(1)<<31))return ExternalPlanStatus::invalid;
    RsqrtRungPlan p{};p.m=m;p.n=n;p.radicand=a;
    auto grant=budget;grant.owned_disk_limit=grant.write_limit=UINT64_MAX;grant.other_owned_disk_bytes=0;
    planning::ExternalProductSelection selected{};
    const auto status=planning::choose_external_product({m,m+1,workers,0,0,false,{m+1,0,m,0,m},{m+1,0,m+1}},grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;
    p.product_quotes=selected.summary_quotes;p.basis_queries=selected.basis_queries;
    auto common=selected.plan;
    if(!retain_spilled_prefix(common,0)||!query_row_cache(common.forward[0],p.cache)||!use_cached_a(common)||
       !query_signed_lift(m+2,{0,m+1,0},{m,newton_contract::rsqrt_residual_limit,true},p.lift)||
       !query_stored_sum(m,int64_t(64*(n-m)),n-m+1,0,p.finish,0,workers)||
       !project_stored_sum(p.finish,{0,n+1,0},{n,2,true}))return ExternalPlanStatus::unsupported;
    size_t fixed;
    if(!add_size(budget.other_live_bytes,common.shared_bytes,fixed)||fixed>budget.resident_bytes||
       p.cache.storage_bytes>budget.resident_bytes-fixed)return ExternalPlanStatus::no_admitted_plan;
    const size_t room=budget.resident_bytes-fixed-p.cache.storage_bytes;
    if(std::max({common.storage_bytes,p.cache.build_bytes,p.lift.storage_bytes})>room)return ExternalPlanStatus::no_admitted_plan;
    const size_t retained=size_t(std::min(uint64_t((room-common.storage_bytes)/chunk_bytes*chunk_bytes),
                                        uint64_t(common.spec.primes)*common.plane_bytes));
    if(!retain_spilled_prefix(common,retained))return ExternalPlanStatus::unsupported;
    p.square=common;
    if(!use_cached_square(p.square)||!project_spilled_output(p.square,0,m+2))return ExternalPlanStatus::unsupported;
    p.square.emit.scalar=a;p.square.emit.bound={2*m,2,true};
    p.correction=common;p.correction.emit.bound={2*m,newton_contract::rsqrt_residual_limit,true};
    const size_t first=(3*m-n)&~(chunk_bytes/8-1);
    if(!project_spilled_output(p.correction,first,2*m+1))return ExternalPlanStatus::unsupported;
    size_t grouped;
    if(!add_size(p.cache.storage_bytes,std::max({common.storage_bytes,p.square.storage_bytes,p.cache.build_bytes,p.lift.storage_bytes}),grouped))return ExternalPlanStatus::unsupported;
    p.storage_bytes=std::max(grouped,p.finish.storage_bytes);
    SpilledProductPlan*cache_consumers[]{&p.square,&p.correction};
    planning::complete_cache_within(p.cache,cache_consumers,p.storage_bytes-p.cache.storage_bytes,profile);
    if(!add_size(fixed,p.storage_bytes,p.total_resident_bytes)||p.total_resident_bytes>budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    RegionLayout r{},s{},e{},c{},result{};
    if(!region(m+1,r)||!region(m+2,s)||!region(m+1,e)||region_layout(p.correction.stored_output_bytes,0,c)||!region(n+1,result))return ExternalPlanStatus::unsupported;
    uint64_t base,planes;
    if(!add(budget.other_owned_disk_bytes,r.allocation_bytes,base)||
       __builtin_mul_overflow(common.plane_allocation_bytes,uint64_t(common.plane_count),&planes))return ExternalPlanStatus::unsupported;
    auto peak=[&](std::initializer_list<uint64_t> values){uint64_t live=base;
        for(uint64_t bytes:values)if(!add(live,bytes,live))return false;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,live);return true;};
    if(!peak({p.cache.owned_disk_bytes,planes,s.allocation_bytes})||
       !peak({p.cache.owned_disk_bytes,s.allocation_bytes,e.allocation_bytes})||
       !peak({p.cache.owned_disk_bytes,e.allocation_bytes,planes,c.allocation_bytes})||
       !peak({c.allocation_bytes,result.allocation_bytes})||!peak({result.allocation_bytes,result.allocation_bytes}))return ExternalPlanStatus::unsupported;
    for(uint64_t bytes:{p.cache.normal_write_bytes,p.square.normal_write_bytes,p.lift.write_bytes,p.correction.normal_write_bytes,p.finish.write_bytes})
        if(!add(p.write_bytes,bytes,p.write_bytes))return ExternalPlanStatus::unsupported;
    if(!add(p.write_bytes,result.data_bytes+result.index_bytes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;
    if(p.worst_write_bytes>budget.write_limit||p.peak_backing_bytes>budget.owned_disk_limit)return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
IoError run_rsqrt_rung(const RsqrtRungPlan&p,Transport&t,ValueStore&store,const StoredInteger&r,StoredInteger&out,
                      std::span<uint8_t>w,sbn3_team&team,bool*clamped) noexcept {
    if(clamped)*clamped=false;
    if(!r.value||r.value.store()!=&store||out.value||r.negative||r.exponent2||r.first_limb||r.limbs()!=p.m+1||
       !r.extent_known||r.extent.end>p.m||!w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes)
        return {IoCode::invalid,"stored rsqrt rung binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"rsqrt rung write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto source=r.source();source.limbs=source.upper_limb=p.m;auto work=w.subspan(p.cache.storage_bytes);
    RowCacheBinding cache;struct End {RowCacheBinding&cache;~End(){release_row_cache(cache);}} cleanup{cache};
    if(auto e=build_row_cache(p.cache,t,store,r.value,source,w.first(p.cache.storage_bytes),work,team,cache))return e;
    StoredInteger square,residual,correction,result;
    if(auto e=run_owned_cached_product(p.square,t,store,cache,r.value,square.value,work,team,&square.extent))return e;
    square.extent_known=true;
    if(auto e=run_signed_lift(p.lift,t,store,square,residual,work))return e;
    square.value.reset();const bool negative=residual.negative;
    if(auto e=run_owned_cached_product(p.correction,t,store,cache,residual.value,correction.value,work,team,&correction.extent))return e;
    correction.extent_known=true;residual.value.reset();release_row_cache(cache);
    const size_t first=3*p.m-p.n-size_t(p.correction.output_first_byte/8);
    auto cy=linear_input(correction);cy.source.first_limb=first;cy.source.limbs=p.n-p.m+1;cy.source.bit_shift=-1;cy.negative=!negative;
    auto coarse=linear_input(r);coarse.source=source;coarse.exponent2=int64_t(64*(p.n-p.m));
    if(auto e=run_stored_linear(p.finish,t,store,coarse,cy,result,w,&team))return e;
    correction.value.reset();require(!result.negative,SBN3_FATAL_MATH,"stored rsqrt negative result");
    if(result.extent.end>p.n){StoredInteger fixed;
        if(auto e=write_word_pattern(t,store,p.n+1,UINT64_MAX,0,fixed,w))return e;
        result=std::move(fixed);if(clamped)*clamped=true;
    }
    out=std::move(result);return {};
}
planning::ExternalPlanStatus query_stored_rsqrt(uint64_t a,size_t target,const InverseOptions&options,
    const InverseBudget&budget,const planning::ExternalProductProfile&profile,RsqrtPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    if(!a||!target||target>(size_t(1)<<31)||!options.workers||options.workers>32||options.maximum_native_seed<2)return ExternalPlanStatus::invalid;
    RsqrtPlan p{};p.radicand=a;p.target=target;p.shared_bytes=engine_shared_bytes();size_t fixed;
    if(!add_size(budget.other_live_bytes,sizeof(RsqrtPlan),fixed)||!add_size(fixed,p.shared_bytes,fixed)||fixed>=budget.resident_bytes)
        return ExternalPlanStatus::no_admitted_plan;
    size_t descending[32]{},count=0,n=target;bool found=false;
    while(true){
        if(n<=options.maximum_native_seed){sbn3_newton_options native{};native.workers=options.workers;
            if(sbn3_newton_query(SBN3_NEWTON_RSQRT,n,&native,&p.seed,&p.seed_info)==SBN3_SUPPORTED){
                p.alignment=std::max(size_t(4096),p.seed_info.storage_alignment);
                size_t at=0,output;auto put=[&](size_t bytes,size_t alignment,size_t&offset){return align_size(at,alignment,offset)&&add_size(offset,bytes,at);};
                if(!mul_size(n+1,8,output)||!put(1u<<20,4096,p.seed_io)||!put(index_page_bytes,4096,p.seed_index)||
                   !put(output,64,p.seed_output)||!put(p.seed_info.storage_bytes,p.seed_info.storage_alignment,p.seed_binding)||
                   !align_size(at,p.alignment,p.storage_bytes))return ExternalPlanStatus::unsupported;
                if(p.storage_bytes<=budget.resident_bytes-fixed){p.seed_limbs=n;found=true;break;}
            }
        }
        if(n<=2||count==32)break;descending[count++]=n;n=newton_contract::next_precision(n);
    }
    if(!found)return ExternalPlanStatus::no_admitted_plan;
    RegionLayout seed{};if(!region(p.seed_limbs+1,seed)||!add(budget.other_owned_disk_bytes,seed.allocation_bytes,p.peak_backing_bytes))return ExternalPlanStatus::unsupported;
    p.write_bytes=p.worst_write_bytes=seed.data_bytes+seed.index_bytes;
    size_t m=p.seed_limbs;
    for(size_t i=count;i-- >0;){n=descending[i];auto&step=p.rungs[p.rung_count++];
        planning::ExternalProductBudget grant{aligned_phase_budget(budget.resident_bytes,fixed,p.alignment),fixed-p48_root_bank::bytes(),UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
        const auto status=query_rsqrt_rung(a,m,n,options.workers,grant,profile,step);if(status!=ExternalPlanStatus::ready)return status;
        p.storage_bytes=std::max(p.storage_bytes,step.storage_bytes);p.peak_backing_bytes=std::max(p.peak_backing_bytes,step.peak_backing_bytes);
        if(!add(p.write_bytes,step.write_bytes,p.write_bytes)||!add(p.worst_write_bytes,step.worst_write_bytes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;m=n;
    }
    if(!align_size(p.storage_bytes,p.alignment,p.storage_bytes)||!add_size(fixed,p.storage_bytes,p.total_resident_bytes))return ExternalPlanStatus::unsupported;
    if(p.total_resident_bytes>budget.resident_bytes||p.worst_write_bytes>budget.write_limit||p.peak_backing_bytes>budget.owned_disk_limit)
        return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
IoError run_stored_rsqrt(const RsqrtPlan&p,Transport&t,ValueStore&store,StoredInteger&out,sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    if(out.value||!p.radicand||team.arena!=&arena||team.busy||p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||
       !arena.unleased(offset,p.storage_bytes)||(uintptr_t(arena.base+offset)&(p.alignment-1)))return {IoCode::invalid,"stored rsqrt binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored rsqrt write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto workspace=std::span(arena.base+offset,p.storage_bytes);StoredInteger current;
    {
        sbn3_lease loans[3]{};struct Release {sbn3_arena&a;sbn3_lease*p;~Release(){for(unsigned i=0;i<3;++i)if(p[i].token)sbn3_arena_release(&a,p+i);}} release{arena,loans};
        OperationSpan trace(t,"native_rsqrt_seed",p.seed_limbs,p.target);
        loans[0]=arena.acquire(offset+p.seed_io,1u<<20);loans[1]=arena.acquire(offset+p.seed_index,index_page_bytes);
        loans[2]=arena.acquire(offset+p.seed_output,(p.seed_limbs+1)*8);
        if(t.cancel.test())return {IoCode::cancelled,"rsqrt native seed boundary"};
        sbn3_newton_binding*binding=nullptr;sbn3_newton_bind(&p.seed,&arena,offset+p.seed_binding,&team,&binding);
        const sbn3_newton_inputs input{{},{},p.radicand};
        sbn3_newton_execute(binding,&input,{reinterpret_cast<uint64_t *>(workspace.data()+p.seed_output),p.seed_limbs+1});sbn3_newton_unbind(binding);
        StoreBuffers io{workspace.subspan(p.seed_io,1u<<20),workspace.subspan(p.seed_index,index_page_bytes)};
        RegionLease value;if(auto e=store.reserve((p.seed_limbs+1)*8,RegionKind::value,value))return e;
        ValueWriter writer;if(auto e=writer.open(t,value.region(),io,&current.extent))return e;
        if(auto e=writer.append(workspace.subspan(p.seed_output,(p.seed_limbs+1)*8)))return e;
        if(auto e=writer.seal())return e;if(auto e=value.publish(current.value))return e;current.extent_known=true;
    }
    auto lease=arena.acquire(offset,p.storage_bytes);
    struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){sbn3_arena_release(&a,&l);}} release{arena,lease};
    for(size_t i=0;i<p.rung_count;++i){StoredInteger next;
        if(auto e=run_rsqrt_rung(p.rungs[i],t,store,current,next,workspace,team))return e;current=std::move(next);
    }
    out=std::move(current);return {};
}
} // namespace sbn::v3::external
