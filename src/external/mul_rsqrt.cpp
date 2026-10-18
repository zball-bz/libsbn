// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/newton.hpp"
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
bool admissible(const StoredInteger&x,size_t n,uint64_t scalar){
    if(!x.value||x.negative||!scalar||!x.extent_known||x.first_limb||x.limbs()!=n+1||x.extent.end>n+1)return false;
    const uint64_t high=x.extent.end==n+1?x.extent.top_word:0;
    return (__uint128_t(high)+1)*scalar<=(__uint128_t(1)<<64);
}
}
planning::ExternalPlanStatus query_mul_rsqrt_source(SourceWindow input,size_t n,uint64_t high_exclusive,uint64_t a,uint64_t scalar,
    const InverseOptions&options,const InverseBudget&budget,const planning::ExternalProductProfile&profile,MulRsqrtPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    if(n<4||n>(size_t(1)<<31)||!a||!scalar||!resolve_source_window(n+1,input)||input.limbs!=n+1||input.first_limb||input.low_zero_words||input.bit_shift||
       (high_exclusive?__uint128_t(high_exclusive)*scalar>(__uint128_t(1)<<64):scalar>1))return ExternalPlanStatus::invalid;
    MulRsqrtPlan p{};p.n=n;p.m=newton_contract::next_precision(n);p.input=input;p.scalar=scalar;p.radicand=a;
    RegionLayout source{},root{};if(!region(p.input.storage_limbs,source)||!region(p.m+1,root))return ExternalPlanStatus::unsupported;
    auto seed_budget=budget;
    if(!add_size(budget.other_live_bytes,sizeof(MulRsqrtPlan)-sizeof(RsqrtPlan),seed_budget.other_live_bytes)||
       !add(budget.other_owned_disk_bytes,source.allocation_bytes,seed_budget.other_owned_disk_bytes))return ExternalPlanStatus::unsupported;
    auto status=query_stored_rsqrt(a,p.m,options,seed_budget,profile,p.half);if(status!=ExternalPlanStatus::ready)return status;
    size_t fixed;if(!add_size(budget.other_live_bytes,sizeof(MulRsqrtPlan),fixed)||!add_size(fixed,engine_shared_bytes(),fixed))return ExternalPlanStatus::unsupported;
    planning::ExternalProductBudget grant{aligned_phase_budget(budget.resident_bytes,fixed,p.half.alignment),fixed-p48_root_bank::bytes(),UINT64_MAX,0,UINT64_MAX};
    planning::ExternalProductSelection selected;
    SourceWindow root_view{p.m+1,0,p.m,0,p.m};
    status=planning::choose_external_product({p.m+1,p.m+1,options.workers,0,0,true,root_view,root_view},grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;p.square=selected.plan;
    p.square.emit.scalar=a;p.square.emit.bound={2*p.m,2,true};
    if(!project_spilled_output(p.square,0,p.m+2)||
       !query_signed_lift(p.m+2,{0,p.m+1,0},{p.m,newton_contract::rsqrt_residual_limit,true},p.lift))return ExternalPlanStatus::unsupported;
    // The final square has one consumer and already owns its geometry. Recover
    // a*R^2-B^(2m) directly in a smaller ring, without a residual-lift file.
    planning::ExternalProductRequest cyclic_request{p.m+1,p.m+1,options.workers,0,0,true,root_view,root_view};
    cyclic_request.cyclic={p.m+2,2*p.m,{p.m,newton_contract::rsqrt_residual_limit,true}};
    planning::ExternalProductSelection cyclic{};
    if(planning::choose_external_product(cyclic_request,grant,profile,cyclic)==ExternalPlanStatus::ready&&
       project_spilled_output(cyclic.plan,0,p.m+1)){
        cyclic.plan.emit.scalar=a;
        const double old_cost=planning::estimate_external_product(p.square,profile)+
            (double(p.m+2)*8+p.lift.write_bytes)/profile.io_bytes_per_ns;
        if(cyclic.plan.normal_write_bytes<=p.square.normal_write_bytes+p.lift.write_bytes&&
           planning::estimate_external_product(cyclic.plan,profile)<=old_cost*profile.write_saving_slowdown){
            p.square=cyclic.plan;p.lift={};
        }
    }
    status=planning::choose_external_product({p.m,n+1,options.workers,0,0,false,root_view,p.input},grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;p.multiply=selected.plan;p.multiply.emit.scalar=scalar;
    const size_t main_first=p.m&~(chunk_bytes/8-1);
    if(!project_spilled_output(p.multiply,main_first,n+p.m+1))return ExternalPlanStatus::unsupported;
    const size_t product_words=n+p.m+1-main_first,at=p.m-main_first;
    SourceWindow hi{product_words,at+n-p.m,p.m+1,at,product_words};
    status=planning::choose_external_product({p.m+1,p.m+1,options.workers,0,0,false,hi,{p.m+1,0,p.m+1}},grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;p.correction=selected.plan;
    p.correction.emit.bound={2*p.m+1,newton_contract::rsqrt_residual_limit,true};
    const size_t first=(3*p.m-n)&~(chunk_bytes/8-1);
    if(!project_spilled_output(p.correction,first,2*p.m+2)||!query_stored_sum(n+1,0,n-p.m+2,0,p.finish,0,options.workers)||
       !project_stored_sum(p.finish,{0,n+1,0},{n+1,1,true}))return ExternalPlanStatus::unsupported;
    p.alignment=p.half.alignment;
    if(!align_size(std::max({p.half.storage_bytes,p.square.storage_bytes,p.multiply.storage_bytes,p.correction.storage_bytes,p.lift.storage_bytes,p.finish.storage_bytes}),p.alignment,p.storage_bytes)||
       !add_size(fixed,p.storage_bytes,p.total_resident_bytes))return ExternalPlanStatus::unsupported;
    RegionLayout square{},residual{},coarse{},correction{},result{};
    const bool ring=bool(p.square.spec.cyclic);
    if(!region(ring?0:p.m+2,square)||!region(p.m+1,residual)||region_layout(p.multiply.stored_output_bytes,0,coarse)||
       region_layout(p.correction.stored_output_bytes,0,correction)||!region(n+1,result))return ExternalPlanStatus::unsupported;
    uint64_t base,square_planes,main_planes,correction_planes;
    if(!add(budget.other_owned_disk_bytes,source.allocation_bytes,base)||
       __builtin_mul_overflow(p.square.plane_allocation_bytes,uint64_t(p.square.plane_count),&square_planes)||
       __builtin_mul_overflow(p.multiply.plane_allocation_bytes,uint64_t(p.multiply.plane_count),&main_planes)||
       __builtin_mul_overflow(p.correction.plane_allocation_bytes,uint64_t(p.correction.plane_count),&correction_planes))return ExternalPlanStatus::unsupported;
    p.peak_backing_bytes=p.half.peak_backing_bytes;
    auto peak=[&](std::initializer_list<uint64_t>values){uint64_t live=base;for(uint64_t b:values)if(!add(live,b,live))return false;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,live);return true;};
    if(!peak({root.allocation_bytes,square_planes,ring?residual.allocation_bytes:square.allocation_bytes})||
       !peak({root.allocation_bytes,square.allocation_bytes,residual.allocation_bytes})||
       !peak({root.allocation_bytes,residual.allocation_bytes,main_planes,coarse.allocation_bytes})||
       !peak({residual.allocation_bytes,coarse.allocation_bytes,correction_planes,correction.allocation_bytes})||
       !peak({coarse.allocation_bytes,correction.allocation_bytes,result.allocation_bytes}))return ExternalPlanStatus::unsupported;
    uint64_t writes=0;
    for(uint64_t bytes:{p.square.normal_write_bytes,p.multiply.normal_write_bytes,p.correction.normal_write_bytes,p.lift.write_bytes,p.finish.write_bytes})
        if(!add(writes,bytes,writes))return ExternalPlanStatus::unsupported;
    if(!add(p.half.write_bytes,writes,p.write_bytes)||!add(p.half.worst_write_bytes,writes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.worst_write_bytes>budget.write_limit)
        return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
planning::ExternalPlanStatus query_stored_mul_rsqrt(const StoredInteger&x,size_t n,uint64_t a,uint64_t scalar,
    const InverseOptions&options,const InverseBudget&budget,const planning::ExternalProductProfile&profile,MulRsqrtPlan&out) noexcept {
    if(!admissible(x,n,scalar))return planning::ExternalPlanStatus::invalid;
    const uint64_t high=x.extent.end==n+1?x.extent.top_word:0;
    return query_mul_rsqrt_source(x.source(),n,high+1,a,scalar,options,budget,profile,out);
}
IoError run_stored_mul_rsqrt(const MulRsqrtPlan&p,Transport&t,ValueStore&store,const StoredInteger&x,StoredInteger&out,
    sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    if(!admissible(x,p.n,p.scalar)||out.value||x.value.store()!=&store||x.source()!=p.input||team.arena!=&arena||team.busy||
       p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||!arena.unleased(offset,p.storage_bytes)||
       (uintptr_t(arena.base+offset)&(p.alignment-1)))return {IoCode::invalid,"stored mul-rsqrt binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored mul-rsqrt write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    StoredInteger root,square,residual,coarse,correction,result;
    if(auto e=run_stored_rsqrt(p.half,t,store,root,arena,offset,team))return e;
    auto lease=arena.acquire(offset,p.storage_bytes);
    struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){sbn3_arena_release(&a,&l);}} release{arena,lease};
    const auto w=std::span(arena.base+offset,p.storage_bytes);
    CyclicResidualResult cyclic{};const bool ring=bool(p.square.spec.cyclic);
    if(auto e=run_owned_product(p.square,t,store,root.value,root.value,square.value,w,team,&square.extent,ring?&cyclic:nullptr))return e;
    square.extent_known=true;
    if(!ring){
        if(auto e=run_signed_lift(p.lift,t,store,square,residual,w))return e;
        square.value.reset();
    }
    const bool negative=ring?cyclic.complemented:residual.negative;
    if(auto e=run_owned_product(p.multiply,t,store,root.value,x.value,coarse.value,w,team,&coarse.extent))return e;
    root.value.reset();coarse.extent_known=true;coarse.first_limb=p.m-size_t(p.multiply.output_first_byte/8);coarse.word_count=p.n+1;
    // Keep the encoded ring value positive; its magnitude interpretation lives
    // in the source view, alongside the separately resolved sign.
    const auto&error_value=ring?square:residual;
    auto correction_plan=p.correction;auto residual_source=error_value.source();residual_source.ones_complement=ring&&cyclic.complemented;
    require(rebind_spilled_sources(correction_plan,correction_plan.spec.a_source,residual_source),SBN3_FATAL_WORKSPACE,"mul-rsqrt residual source");
    if(auto e=run_owned_product(correction_plan,t,store,coarse.value,error_value.value,correction.value,w,team,&correction.extent))return e;
    square.value.reset();residual.value.reset();correction.extent_known=true;
    auto cy=linear_input(correction);cy.source.first_limb=3*p.m-p.n-size_t(p.correction.output_first_byte/8);
    cy.source.limbs=p.n-p.m+2;cy.source.bit_shift=-1;cy.negative=!negative;
    if(auto e=run_stored_linear(p.finish,t,store,linear_input(coarse),cy,result,w,&team))return e;
    require(!result.negative,SBN3_FATAL_MATH,"stored mul-rsqrt negative result");
    result.exponent2=x.exponent2;out=std::move(result);return {};
}
} // namespace sbn::v3::external
