// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/newton.hpp"
#include "external/trace.hpp"
#include "external/owned_product.hpp"
#include "external/shared.hpp"
#include "algorithms/refinement.hpp"
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
static planning::ExternalPlanStatus query_inverse_rung_kind(size_t m,size_t n,SourceWindow denominator,unsigned workers,
    const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,InverseRungPlan&out,bool cyclic,
    unsigned&quotes,unsigned&basis_queries) noexcept {
    using planning::ExternalPlanStatus;
    if(m<3||m>=n||n-m>m-1||n>(size_t(1)<<31)||!resolve_source_window(n,denominator)||denominator.limbs!=n)return ExternalPlanStatus::invalid;
    InverseRungPlan p{};p.m=m;p.n=n;p.denominator=denominator;
    const auto shape=refinement_products<RefinementKind::Inverse>(m,n);
    const size_t rn=shape.products[0].window.words;
    auto source=denominator;source.storage_limbs=std::max(source.storage_limbs,rn);
    auto budget0=budget;budget0.owned_disk_limit=budget0.write_limit=UINT64_MAX;budget0.other_owned_disk_bytes=0;
    planning::ExternalProductSelection base{};
    planning::ExternalProductRequest request{m+1,n,workers,0,0,false,{m+1,0,m+1},source};
    if(cyclic)request.cyclic={n+4,n+m,shape.products[0].bound};
    auto status=planning::choose_external_product(request,budget0,profile,base);
    quotes+=base.summary_quotes;basis_queries+=base.basis_queries;
    if(status!=ExternalPlanStatus::ready)return status;
    p.product_quotes=base.summary_quotes;p.basis_queries=base.basis_queries;
    auto product=base.plan;
    if(!retain_spilled_prefix(product,0)||!query_row_cache(product.forward[0],p.cache)||!use_cached_a(product))return ExternalPlanStatus::unsupported;
    if(!cyclic&&!query_signed_lift(n+2,{shape.products[0].window.origin,rn,0},shape.products[0].bound,p.lift))return ExternalPlanStatus::unsupported;
    auto correction=product;
    const size_t residual_first=cyclic?shape.products[0].window.origin&~(chunk_bytes/8-1):0;
    const size_t residual_end=cyclic?shape.products[0].window.origin+rn:n+2;
    const size_t residual_words=residual_end-residual_first;
    if(cyclic){
        auto spec=product.spec;spec.cyclic={};spec.bn=rn;spec.retained_bytes=0;
        spec.b_source={residual_words,shape.products[0].window.origin-residual_first,rn};
        ++p.product_quotes;++quotes;
        if(!query_spilled_product(spec,correction)||!use_cached_a(correction))return ExternalPlanStatus::unsupported;
    }
    if(!query_stored_sum(m+1,int64_t(64*(n-m)),shape.products[1].window.words,0,p.finish,0,workers)||
       !project_stored_sum(p.finish,{0,n+1,0},{n,3,true}))return ExternalPlanStatus::unsupported;
    size_t fixed;
    if(!add_size(budget.other_live_bytes,product.shared_bytes,fixed)||fixed>budget.resident_bytes||
       p.cache.storage_bytes>budget.resident_bytes-fixed)return ExternalPlanStatus::no_admitted_plan;
    const size_t room=budget.resident_bytes-fixed-p.cache.storage_bytes;
    if(std::max({product.storage_bytes,correction.storage_bytes,p.cache.build_bytes,p.lift.storage_bytes})>room)return ExternalPlanStatus::no_admitted_plan;
    for(auto*q:{&product,&correction}){
        const size_t retained=size_t(std::min(uint64_t((room-q->storage_bytes)/chunk_bytes*chunk_bytes),uint64_t(q->spec.primes)*q->plane_bytes));
        if(!retain_spilled_prefix(*q,retained))return ExternalPlanStatus::unsupported;
    }
    p.residual_product=product;if(!cyclic)p.residual_product.emit.bound={n+m,2,true};
    if(!project_spilled_output(p.residual_product,residual_first,residual_end))return ExternalPlanStatus::unsupported;
    p.correction_product=correction;p.correction_product.emit.bound=shape.products[1].bound;
    const size_t first=shape.products[1].window.origin&~(chunk_bytes/8-1),end=shape.products[1].window.origin+shape.products[1].window.words;
    if((!cyclic&&!rebind_spilled_sources(p.correction_product,product.spec.a_source,{rn,0,rn}))||
       !project_spilled_output(p.correction_product,first,end))return ExternalPlanStatus::unsupported;
    size_t grouped;
    if(!add_size(p.cache.storage_bytes,std::max({product.storage_bytes,correction.storage_bytes,p.cache.build_bytes,p.lift.storage_bytes}),grouped))return ExternalPlanStatus::unsupported;
    p.storage_bytes=std::max(grouped,p.finish.storage_bytes);
    SpilledProductPlan*cache_consumers[]{&p.residual_product,&p.correction_product};
    planning::complete_cache_within(p.cache,cache_consumers,p.storage_bytes-p.cache.storage_bytes,profile);
    p.product_quotes+=2;quotes+=2;
    if(planning::prefer_raw_pair(p.cache,p.residual_product,p.correction_product,budget.resident_bytes-fixed,profile))
        p.storage_bytes=std::max({p.residual_product.storage_bytes,p.correction_product.storage_bytes,p.lift.storage_bytes,p.finish.storage_bytes});
    if(!add_size(fixed,p.storage_bytes,p.total_resident_bytes)||p.total_resident_bytes>budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    RegionLayout d{},u{},res{},rho{},corr{},output{};
    if(!region(denominator.storage_limbs,d)||!region(m+1,u)||!region(residual_words,res)||!region(cyclic?0:rn,rho)||
       region_layout(p.correction_product.stored_output_bytes,0,corr)||!region(n+1,output))return ExternalPlanStatus::unsupported;
    uint64_t input,scratch,correction_scratch,build,first_peak,lift_peak,second_peak,finish_peak,clamp_peak;
    if(!add(d.allocation_bytes,u.allocation_bytes,input)||!add(input,budget.other_owned_disk_bytes,input)||
       __builtin_mul_overflow(p.residual_product.plane_allocation_bytes,uint64_t(p.residual_product.plane_count),&scratch)||
       __builtin_mul_overflow(p.correction_product.plane_allocation_bytes,uint64_t(p.correction_product.plane_count),&correction_scratch)||
       !add(input,p.cache.owned_disk_bytes,build)||!add(build,scratch,first_peak)||!add(first_peak,res.allocation_bytes,first_peak)||
       !add(build,res.allocation_bytes,lift_peak)||!add(lift_peak,rho.allocation_bytes,lift_peak)||
       !add(build,cyclic?res.allocation_bytes:rho.allocation_bytes,second_peak)||!add(second_peak,correction_scratch,second_peak)||!add(second_peak,corr.allocation_bytes,second_peak)||
       !add(input,corr.allocation_bytes,finish_peak)||!add(finish_peak,output.allocation_bytes,finish_peak)||
       !add(input,2*output.allocation_bytes,clamp_peak))return ExternalPlanStatus::unsupported;
    p.peak_backing_bytes=std::max({build,first_peak,lift_peak,second_peak,finish_peak,clamp_peak});
    if(!add(p.cache.normal_write_bytes,p.residual_product.normal_write_bytes,p.write_bytes)||
       !add(p.write_bytes,p.lift.write_bytes,p.write_bytes)||!add(p.write_bytes,p.correction_product.normal_write_bytes,p.write_bytes)||
       !add(p.write_bytes,p.finish.write_bytes,p.write_bytes)||!add(p.write_bytes,output.data_bytes+output.index_bytes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;
    if(p.peak_backing_bytes>budget.owned_disk_limit||p.worst_write_bytes>budget.write_limit)return ExternalPlanStatus::no_admitted_plan;
    p.estimated_ns=(p.cache.storage_bytes?planning::estimate_row_cache(p.cache,profile):0.)+
        planning::estimate_external_product(p.residual_product,profile)+planning::estimate_external_product(p.correction_product,profile)+
        (cyclic?0.0:(double(n+2)*8+p.lift.write_bytes)/profile.io_bytes_per_ns);
    out=p;return ExternalPlanStatus::ready;
}
planning::ExternalPlanStatus query_inverse_rung(size_t m,size_t n,SourceWindow denominator,unsigned workers,
    const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,InverseRungPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    InverseRungPlan linear{},cyclic{};unsigned quotes=0,bases=0;
    const auto a=query_inverse_rung_kind(m,n,denominator,workers,budget,profile,linear,false,quotes,bases);
    if(a==ExternalPlanStatus::invalid)return a;
    const auto b=query_inverse_rung_kind(m,n,denominator,workers,budget,profile,cyclic,true,quotes,bases);
    const bool use=b==ExternalPlanStatus::ready&&(a!=ExternalPlanStatus::ready||
        (cyclic.write_bytes<=linear.write_bytes&&cyclic.estimated_ns<=linear.estimated_ns*profile.write_saving_slowdown));
    if(!use&&a!=ExternalPlanStatus::ready)return a;
    out=use?cyclic:linear;out.product_quotes=quotes;
    out.basis_queries=bases;return ExternalPlanStatus::ready;
}
IoError run_inverse_rung(const InverseRungPlan&p,Transport&t,ValueStore&store,const StoredInteger&d,
    const StoredInteger&u,StoredInteger&out,std::span<uint8_t>w,sbn3_team&team,bool*clamped) noexcept {
    if(clamped)*clamped=false;
    SourceWindow normalized;
    if(!d.value||!u.value||out.value||d.value.store()!=&store||u.value.store()!=&store||
       !normalized_source(d,p.n,normalized)||normalized!=p.denominator||u.negative||u.exponent2||u.first_limb||
       u.limbs()!=p.m+1||!u.extent_known||u.extent.end!=p.m+1||u.extent.top_word!=1||
       !w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes)return {IoCode::invalid,"stored inverse rung binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"inverse rung write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto work=w.subspan(p.cache.storage_bytes);RowCacheBinding cache;
    struct End {RowCacheBinding&cache;~End(){release_row_cache(cache);}} cleanup{cache};
    if(p.cache.storage_bytes)if(auto e=build_row_cache(p.cache,t,store,u.value,u.source(),w.first(p.cache.storage_bytes),work,team,cache))return e;
    auto residual_plan=p.residual_product;
    const auto common=p.cache.storage_bytes?row_cache_source(cache):u.source();
    require(rebind_spilled_sources(residual_plan,common,normalized),SBN3_FATAL_WORKSPACE,"inverse residual source");
    StoredInteger product,rho,correction,result;
    CyclicResidualResult cyclic{};const bool ring=bool(residual_plan.spec.cyclic);
    const auto residual_error=residual_plan.cached_a?
        run_owned_cached_product(residual_plan,t,store,cache,d.value,product.value,work,team,&product.extent,ring?&cyclic:nullptr):
        run_owned_product(residual_plan,t,store,u.value,d.value,product.value,work,team,&product.extent,ring?&cyclic:nullptr);
    if(residual_error)return residual_error;
    product.extent_known=true;
    SourceWindow correction_source;const StoredValue*residual_value=nullptr;
    if(ring){
        const auto shape=refinement_products<RefinementKind::Inverse>(p.m,p.n);
        // product remains the positive encoded R window. Only this explicit
        // magnitude view is complemented; no StoredInteger falsely denotes -R.
        correction_source=product.source();correction_source.first_limb=shape.products[0].window.origin-size_t(residual_plan.output_first_byte/8);
        correction_source.limbs=shape.products[0].window.words;correction_source.lower_limb=correction_source.first_limb;
        correction_source.upper_limb=correction_source.first_limb+correction_source.limbs;correction_source.ones_complement=cyclic.complemented;
        residual_value=&product.value;
    }else{
        if(auto e=run_signed_lift(p.lift,t,store,product,rho,work))return e;
        product.value.reset();correction_source=rho.source();residual_value=&rho.value;
    }
    const bool negative=ring?cyclic.complemented:rho.negative;
    auto correction_plan=p.correction_product;
    require(rebind_spilled_sources(correction_plan,common,correction_source),SBN3_FATAL_WORKSPACE,"inverse correction source");
    const auto correction_error=correction_plan.cached_a?
        run_owned_cached_product(correction_plan,t,store,cache,*residual_value,correction.value,work,team,&correction.extent):
        run_owned_product(correction_plan,t,store,u.value,*residual_value,correction.value,work,team,&correction.extent);
    if(correction_error)return correction_error;
    correction.extent_known=true;product.value.reset();rho.value.reset();release_row_cache(cache);
    const size_t origin=newton_contract::correction_offset(p.m);
    correction.first_limb=origin-size_t(correction_plan.output_first_byte/8);correction.word_count=p.n-p.m+1;
    correction.negative=!negative;
    auto coarse=linear_input(u);coarse.exponent2=int64_t(64*(p.n-p.m));
    if(auto e=run_stored_linear(p.finish,t,store,coarse,linear_input(correction),result,w,&team))return e;
    correction.value.reset();require(!result.negative,SBN3_FATAL_MATH,"stored inverse negative result");
    const uint64_t high=result.extent.end==p.n+1?result.extent.top_word:0;
    if(high!=1){
        StoredInteger fixed;if(auto e=write_word_pattern(t,store,p.n+1,high?UINT64_MAX:0,1,fixed,w))return e;
        result=std::move(fixed);if(clamped)*clamped=true;
    }
    require(result.extent.end==p.n+1&&result.extent.top_word==1,SBN3_FATAL_MATH,"stored inverse normalization");
    out=std::move(result);return {};
}
planning::ExternalPlanStatus query_inverse_source(SourceWindow denominator,size_t target,const InverseOptions&options,
    const InverseBudget&budget,const planning::ExternalProductProfile&profile,InversePlan&out) noexcept {
    using planning::ExternalPlanStatus;
    InversePlan p{};
    if(!target||target>(size_t(1)<<31)||!options.workers||options.workers>32||options.maximum_native_seed<3||
       !resolve_source_window(target,denominator)||denominator.limbs!=target)return ExternalPlanStatus::invalid;
    p.denominator=denominator;p.source_capacity_words=denominator.storage_limbs;
    p.target=target;p.shared_bytes=engine_shared_bytes();size_t fixed;
    if(!add_size(budget.other_live_bytes,sizeof(InversePlan),fixed)||!add_size(fixed,p.shared_bytes,fixed)||fixed>=budget.resident_bytes)
        return ExternalPlanStatus::no_admitted_plan;
    size_t descending[32]{},count=0,n=target;bool found=false;
    while(true){
        if(n<=options.maximum_native_seed){
            sbn3_newton_options native{};native.workers=options.workers;
            if(sbn3_newton_query(SBN3_NEWTON_INVERSE,n,&native,&p.seed,&p.seed_info)==SBN3_SUPPORTED){
                size_t at=0,input_bytes,output_bytes;
                auto put=[&](size_t bytes,size_t alignment,size_t&offset){return align_size(at,alignment,offset)&&add_size(offset,bytes,at);};
                p.alignment=std::max(size_t(4096),p.seed_info.storage_alignment);
                if(!mul_size(n+2,8,input_bytes)||!mul_size(n+1,8,output_bytes)||
                   !put(1u<<20,4096,p.seed_io)||!put(index_page_bytes,4096,p.seed_index)||
                   !put(input_bytes,64,p.seed_input)||!put(output_bytes,64,p.seed_output)||
                   !put(p.seed_info.storage_bytes,p.seed_info.storage_alignment,p.seed_binding)||!align_size(at,p.alignment,p.storage_bytes))return ExternalPlanStatus::unsupported;
                if(p.storage_bytes<=budget.resident_bytes-fixed){p.seed_limbs=n;found=true;break;}
            }
        }
        if(n<=3||count==32)break;
        descending[count++]=n;n=newton_contract::next_precision(n);
    }
    if(!found)return ExternalPlanStatus::no_admitted_plan;
    p.seed_source=denominator;p.seed_source.first_limb+=target-p.seed_limbs;p.seed_source.limbs=p.seed_limbs;
    RegionLayout input{},seed{};if(!region(p.denominator.storage_limbs,input)||!region(p.seed_limbs+1,seed))return ExternalPlanStatus::unsupported;
    p.write_bytes=p.worst_write_bytes=seed.data_bytes+seed.index_bytes;
    if(!add(budget.other_owned_disk_bytes,input.allocation_bytes,p.peak_backing_bytes)||!add(p.peak_backing_bytes,seed.allocation_bytes,p.peak_backing_bytes))return ExternalPlanStatus::unsupported;
    size_t m=p.seed_limbs;
    for(size_t i=count;i-- >0;){n=descending[i];auto source=denominator;source.first_limb+=target-n;source.limbs=n;
        auto&step=p.rungs[p.rung_count++];
        planning::ExternalProductBudget grant{aligned_phase_budget(budget.resident_bytes,fixed,p.alignment),fixed-p48_root_bank::bytes(),UINT64_MAX,budget.other_owned_disk_bytes,UINT64_MAX};
        const auto status=query_inverse_rung(m,n,source,options.workers,grant,profile,step);
        if(status!=ExternalPlanStatus::ready)return status;
        p.storage_bytes=std::max(p.storage_bytes,step.storage_bytes);p.peak_backing_bytes=std::max(p.peak_backing_bytes,step.peak_backing_bytes);
        if(!add(p.write_bytes,step.write_bytes,p.write_bytes)||!add(p.worst_write_bytes,step.worst_write_bytes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;
        m=n;
    }
    if(!align_size(p.storage_bytes,p.alignment,p.storage_bytes)||!add_size(fixed,p.storage_bytes,p.total_resident_bytes))return ExternalPlanStatus::unsupported;
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.worst_write_bytes>budget.write_limit)return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
planning::ExternalPlanStatus query_stored_inverse(const StoredInteger&d,size_t target,const InverseOptions&options,
    const InverseBudget&budget,const planning::ExternalProductProfile&profile,InversePlan&out) noexcept {
    SourceWindow source;if(!normalized_source(d,target,source))return planning::ExternalPlanStatus::invalid;
    return query_inverse_source(source,target,options,budget,profile,out);
}
bool rebind_stored_inverse(InversePlan&p,const StoredInteger&d) noexcept {
    SourceWindow source;if(!normalized_source(d,p.target,source)||source.storage_limbs>p.source_capacity_words)return false;
    p.denominator=source;p.seed_source=source;p.seed_source.first_limb+=p.target-p.seed_limbs;p.seed_source.limbs=p.seed_limbs;
    for(size_t i=0;i<p.rung_count;++i){auto&s=p.rungs[i];s.denominator=source;s.denominator.first_limb+=p.target-s.n;s.denominator.limbs=s.n;
        if(!resolve_source_window(s.n,s.denominator))return false;
        if(!rebind_spilled_sources(s.residual_product,s.residual_product.spec.a_source,s.denominator))return false;}
    return true;
}
IoError run_stored_inverse(const InversePlan&p,Transport&t,ValueStore&store,const StoredInteger&d,StoredInteger&out,
    sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    SourceWindow source;
    if(!d.value||d.value.store()!=&store||out.value||!normalized_source(d,p.target,source)||source!=p.denominator||
       team.arena!=&arena||team.busy||p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||
       !arena.unleased(offset,p.storage_bytes)||(uintptr_t(arena.base+offset)&(p.alignment-1)))return {IoCode::invalid,"stored inverse binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored inverse write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto workspace=std::span(arena.base+offset,p.storage_bytes);StoredInteger current;
    {
        sbn3_lease loans[4]{};
        OperationSpan trace(t,"native_inverse_seed",p.seed_limbs,p.target);
        struct Release {sbn3_arena&a;sbn3_lease*p;~Release(){for(unsigned i=0;i<4;++i)if(p[i].token)sbn3_arena_release(&a,p+i);}} release{arena,loans};
        loans[0]=arena.acquire(offset+p.seed_io,1u<<20);loans[1]=arena.acquire(offset+p.seed_index,index_page_bytes);
        loans[2]=arena.acquire(offset+p.seed_input,(p.seed_limbs+2)*8);loans[3]=arena.acquire(offset+p.seed_output,(p.seed_limbs+1)*8);
        StoreBuffers io{workspace.subspan(p.seed_io,1u<<20),workspace.subspan(p.seed_index,index_page_bytes)};
        ValueReader reader;if(auto e=reader.open(t,d.value.region(),io))return e;
        if(auto e=read_source_words(reader,p.seed_source,0,p.seed_limbs,workspace.subspan(p.seed_input,(p.seed_limbs+2)*8)))return e;
        if(t.cancel.test())return {IoCode::cancelled,"inverse native seed boundary"};
        sbn3_newton_binding*binding=nullptr;sbn3_newton_bind(&p.seed,&arena,offset+p.seed_binding,&team,&binding);
        const sbn3_newton_inputs inputs{{},{reinterpret_cast<uint64_t *>(workspace.data()+p.seed_input),p.seed_limbs},0};
        sbn3_newton_execute(binding,&inputs,{reinterpret_cast<uint64_t *>(workspace.data()+p.seed_output),p.seed_limbs+1});sbn3_newton_unbind(binding);
        RegionLease seed;if(auto e=store.reserve((p.seed_limbs+1)*8,RegionKind::value,seed))return e;
        ValueWriter writer;if(auto e=writer.open(t,seed.region(),io,&current.extent))return e;
        if(auto e=writer.append(workspace.subspan(p.seed_output,(p.seed_limbs+1)*8)))return e;
        if(auto e=writer.seal())return e;if(auto e=seed.publish(current.value))return e;current.extent_known=true;
    }
    auto retained=arena.acquire(offset,p.storage_bytes);
    struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){sbn3_arena_release(&a,&l);}} release{arena,retained};
    for(size_t i=0;i<p.rung_count;++i){StoredInteger next;
        if(auto e=run_inverse_rung(p.rungs[i],t,store,d,current,next,workspace,team))return e;
        current=std::move(next);
    }
    out=std::move(current);return {};
}
static planning::ExternalPlanStatus query_quotient_rung_kind(size_t m,size_t n,SourceWindow numerator,SourceWindow denominator,
    unsigned workers,const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,QuotientRungPlan&out,bool cyclic) noexcept {
    using planning::ExternalPlanStatus;
    if(m<3||m>=n||n-m>m-1||n>(size_t(1)<<31)||!resolve_source_window(n,numerator)||numerator.limbs!=n||
       !resolve_source_window(n,denominator)||denominator.limbs!=n)return ExternalPlanStatus::invalid;
    QuotientRungPlan p{};p.m=m;p.n=n;p.numerator=numerator;p.denominator=denominator;
    const auto shape=refinement_products<RefinementKind::Quotient>(m,n);
    const size_t rn=shape.products[1].window.words,small=std::max(m+1,rn);
    auto high=numerator;high.first_limb+=n-m;high.limbs=m;
    auto grant=budget;grant.owned_disk_limit=grant.write_limit=UINT64_MAX;grant.other_owned_disk_bytes=0;
    planning::ExternalProductSelection selected{};
    auto status=planning::choose_external_product({m+1,small,workers,0,0,false,{m+1,0,m+1},high},grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;
    p.product_quotes=selected.summary_quotes;p.basis_queries=selected.basis_queries;
    auto common=selected.plan;
    if(!retain_spilled_prefix(common,0)||!query_row_cache(common.forward[0],p.cache)||!use_cached_a(common))return ExternalPlanStatus::unsupported;
    if((!cyclic&&!query_signed_lift(n+2,{shape.products[1].window.origin,rn,0},shape.products[1].bound,p.lift,n,m))||
       !query_stored_sum(m+1,int64_t(64*(n-m)),shape.products[2].window.words,0,p.finish,0,workers)||
       !project_stored_sum(p.finish,{0,n+1,0},{n,4,true}))return ExternalPlanStatus::unsupported;
    size_t fixed;
    if(!add_size(budget.other_live_bytes,common.shared_bytes,fixed)||fixed>budget.resident_bytes||
       p.cache.storage_bytes>budget.resident_bytes-fixed)return ExternalPlanStatus::no_admitted_plan;
    const size_t room=budget.resident_bytes-fixed-p.cache.storage_bytes;
    if(std::max({common.storage_bytes,p.cache.build_bytes,p.lift.storage_bytes})>room)return ExternalPlanStatus::no_admitted_plan;
    const size_t retained=size_t(std::min(uint64_t((room-common.storage_bytes)/chunk_bytes*chunk_bytes),
                                       uint64_t(common.spec.primes)*common.plane_bytes));
    if(!retain_spilled_prefix(common,retained))return ExternalPlanStatus::unsupported;
    p.coarse_product=common;p.coarse_product.emit.bound=shape.products[0].bound;p.coarse_product.emit.bound.verify=true;
    const size_t coarse_first=m&~(chunk_bytes/8-1);
    if(!project_spilled_output(p.coarse_product,coarse_first,2*m+1))return ExternalPlanStatus::unsupported;
    p.correction_product=common;p.correction_product.emit.bound=shape.products[2].bound;
    const size_t correction_first=shape.products[2].window.origin&~(chunk_bytes/8-1);
    if(!rebind_spilled_sources(p.correction_product,common.spec.a_source,{rn,0,rn})||
       !project_spilled_output(p.correction_product,correction_first,n+3))return ExternalPlanStatus::unsupported;
    if(!add_size(grant.other_live_bytes,p.cache.storage_bytes,grant.other_live_bytes))return ExternalPlanStatus::unsupported;
    SourceWindow coarse{2*m+1-coarse_first,m-coarse_first,m+1,m-coarse_first,2*m+1-coarse_first};
    planning::ExternalProductRequest residual{m+1,n,workers,0,0,false,coarse,denominator};
    if(cyclic)residual.cyclic={shape.products[1].bound.words+2,m,shape.products[1].bound,numerator,true};
    status=planning::choose_external_product(residual,grant,profile,selected);
    if(status!=ExternalPlanStatus::ready)return status;
    p.product_quotes+=selected.summary_quotes;p.basis_queries+=selected.basis_queries;
    p.residual_product=selected.plan;
    if(cyclic){const size_t first=shape.products[1].window.origin&~(chunk_bytes/8-1);
        if(!project_spilled_output(p.residual_product,first,shape.products[1].window.origin+rn))return ExternalPlanStatus::unsupported;
    }else{p.residual_product.emit.bound={n+m,4,true};
        if(!project_spilled_output(p.residual_product,0,n+2))return ExternalPlanStatus::unsupported;}
    size_t grouped;
    if(!add_size(p.cache.storage_bytes,std::max({common.storage_bytes,p.cache.build_bytes,p.residual_product.storage_bytes,p.lift.storage_bytes}),grouped))return ExternalPlanStatus::unsupported;
    p.storage_bytes=std::max(grouped,p.finish.storage_bytes);
    SpilledProductPlan*cache_consumers[]{&p.coarse_product,&p.correction_product};
    planning::complete_cache_within(p.cache,cache_consumers,p.storage_bytes-p.cache.storage_bytes,profile);
    p.product_quotes+=2;
    if(planning::prefer_raw_pair(p.cache,p.coarse_product,p.correction_product,budget.resident_bytes-fixed,profile))
        p.storage_bytes=std::max({p.coarse_product.storage_bytes,p.correction_product.storage_bytes,p.residual_product.storage_bytes,p.lift.storage_bytes,p.finish.storage_bytes});
    if(!add_size(fixed,p.storage_bytes,p.total_resident_bytes)||p.total_resident_bytes>budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    RegionLayout a{},d{},u{},c{},res{},rho{},corr{},output{};
    if(!region(numerator.storage_limbs,a)||!region(denominator.storage_limbs,d)||!region(m+1,u)||
       region_layout(p.coarse_product.stored_output_bytes,0,c)||region_layout(p.residual_product.stored_output_bytes,0,res)||!region(cyclic?0:rn,rho)||
       region_layout(p.correction_product.stored_output_bytes,0,corr)||!region(n+1,output))return ExternalPlanStatus::unsupported;
    uint64_t inputs,small_planes,large_planes;
    if(!add(a.allocation_bytes,d.allocation_bytes,inputs)||!add(inputs,u.allocation_bytes,inputs)||!add(inputs,budget.other_owned_disk_bytes,inputs)||
       __builtin_mul_overflow(p.coarse_product.plane_allocation_bytes,uint64_t(p.coarse_product.plane_count),&small_planes)||
       __builtin_mul_overflow(p.residual_product.plane_allocation_bytes,uint64_t(p.residual_product.plane_count),&large_planes))return ExternalPlanStatus::unsupported;
    auto peak=[&](std::initializer_list<uint64_t> values){uint64_t live=inputs;
        for(uint64_t v:values)if(!add(live,v,live))return false;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,live);return true;};
    if(!peak({p.cache.owned_disk_bytes,small_planes,c.allocation_bytes})||
       !peak({p.cache.owned_disk_bytes,c.allocation_bytes,large_planes,res.allocation_bytes})||
       !peak({p.cache.owned_disk_bytes,c.allocation_bytes,res.allocation_bytes,rho.allocation_bytes})||
       !peak({p.cache.owned_disk_bytes,c.allocation_bytes,cyclic?res.allocation_bytes:rho.allocation_bytes,small_planes,corr.allocation_bytes})||
       !peak({c.allocation_bytes,corr.allocation_bytes,output.allocation_bytes}))return ExternalPlanStatus::unsupported;
    for(uint64_t bytes:{p.cache.normal_write_bytes,p.coarse_product.normal_write_bytes,p.residual_product.normal_write_bytes,
                       p.lift.write_bytes,p.correction_product.normal_write_bytes,p.finish.write_bytes})
        if(!add(p.write_bytes,bytes,p.write_bytes))return ExternalPlanStatus::unsupported;
    if(p.peak_backing_bytes>budget.owned_disk_limit||p.write_bytes>budget.write_limit)return ExternalPlanStatus::no_admitted_plan;
    p.estimated_ns=(p.cache.storage_bytes?planning::estimate_row_cache(p.cache,profile):0.)+
        planning::estimate_external_product(p.coarse_product,profile)+planning::estimate_external_product(p.residual_product,profile)+
        planning::estimate_external_product(p.correction_product,profile)+
        (cyclic?0.:double(p.residual_product.stored_output_bytes+p.lift.write_bytes+8*(n-m+2))/profile.io_bytes_per_ns);
    out=p;return ExternalPlanStatus::ready;
}
planning::ExternalPlanStatus query_quotient_rung(size_t m,size_t n,SourceWindow numerator,SourceWindow denominator,
    unsigned workers,const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,QuotientRungPlan&out) noexcept {
    using planning::ExternalPlanStatus;QuotientRungPlan linear{},cyclic{};
    const auto a=query_quotient_rung_kind(m,n,numerator,denominator,workers,budget,profile,linear,false);
    if(a==ExternalPlanStatus::invalid)return a;
    const auto b=query_quotient_rung_kind(m,n,numerator,denominator,workers,budget,profile,cyclic,true);
    const bool use=b==ExternalPlanStatus::ready&&(a!=ExternalPlanStatus::ready||
        (cyclic.write_bytes<=linear.write_bytes&&cyclic.estimated_ns<=linear.estimated_ns*profile.write_saving_slowdown));
    if(!use&&a!=ExternalPlanStatus::ready)return a;
    out=use?cyclic:linear;out.product_quotes=linear.product_quotes+cyclic.product_quotes;
    out.basis_queries=linear.basis_queries+cyclic.basis_queries;return ExternalPlanStatus::ready;
}
IoError run_quotient_rung(const QuotientRungPlan&p,Transport&t,ValueStore&store,const StoredInteger&a,
    const StoredInteger&d,const StoredInteger&u,StoredInteger&out,std::span<uint8_t>w,sbn3_team&team) noexcept {
    SourceWindow as,ds;
    if(!a.value||!d.value||!u.value||out.value||a.value.store()!=&store||d.value.store()!=&store||u.value.store()!=&store||
       !normalized_source(a,p.n,as)||as!=p.numerator||!normalized_source(d,p.n,ds)||ds!=p.denominator||
       u.negative||u.exponent2||u.first_limb||u.limbs()!=p.m+1||!u.extent_known||u.extent.end!=p.m+1||u.extent.top_word!=1||
       !w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes)return {IoCode::invalid,"stored quotient rung binding"};
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"quotient rung write admission",0,p.write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto work=w.subspan(p.cache.storage_bytes);RowCacheBinding cache;
    struct End {RowCacheBinding&cache;~End(){release_row_cache(cache);}} cleanup{cache};
    if(p.cache.storage_bytes)if(auto e=build_row_cache(p.cache,t,store,u.value,u.source(),w.first(p.cache.storage_bytes),work,team,cache))return e;
    StoredInteger coarse,product,rho,correction,result;
    auto high=as;high.first_limb+=p.n-p.m;high.limbs=p.m;
    auto coarse_plan=p.coarse_product;
    const auto common=p.cache.storage_bytes?row_cache_source(cache):u.source();
    require(rebind_spilled_sources(coarse_plan,common,high),SBN3_FATAL_WORKSPACE,"quotient coarse source");
    const auto coarse_error=coarse_plan.cached_a?
        run_owned_cached_product(coarse_plan,t,store,cache,a.value,coarse.value,work,team,&coarse.extent):
        run_owned_product(coarse_plan,t,store,u.value,a.value,coarse.value,work,team,&coarse.extent);
    if(coarse_error)return coarse_error;
    coarse.extent_known=true;coarse.first_limb=p.m-size_t(coarse_plan.output_first_byte/8);coarse.word_count=p.m+1;
    auto residual_plan=p.residual_product;
    require(rebind_spilled_sources(residual_plan,coarse.source(),ds),SBN3_FATAL_WORKSPACE,"quotient residual source");
    const bool ring=bool(residual_plan.spec.cyclic);CyclicResidualResult cyclic{};
    if(ring)require(rebind_spilled_subtract(residual_plan,as),SBN3_FATAL_WORKSPACE,"quotient subtractand source");
    if(auto e=run_owned_product(residual_plan,t,store,coarse.value,d.value,product.value,work,team,&product.extent,
        ring?&cyclic:nullptr,ring?&a.value:nullptr))return e;
    product.extent_known=true;SourceWindow correction_source;const StoredValue*residual_value=nullptr;
    if(ring){const auto shape=refinement_products<RefinementKind::Quotient>(p.m,p.n);
        correction_source=product.source();correction_source.first_limb=shape.products[1].window.origin-size_t(residual_plan.output_first_byte/8);
        correction_source.limbs=shape.products[1].window.words;correction_source.lower_limb=correction_source.first_limb;
        correction_source.upper_limb=correction_source.first_limb+correction_source.limbs;correction_source.ones_complement=cyclic.complemented;
        residual_value=&product.value;
    }else{LinearInput subtract{&a.value,as,0,int64_t(64*p.m),false};
        if(auto e=run_signed_lift(p.lift,t,store,product,rho,work,&subtract))return e;
        product.value.reset();correction_source=rho.source();residual_value=&rho.value;
    }
    const bool negative=ring?cyclic.complemented:rho.negative;
    auto correction_plan=p.correction_product;
    require(rebind_spilled_sources(correction_plan,common,correction_source),SBN3_FATAL_WORKSPACE,"quotient correction source");
    const auto correction_error=correction_plan.cached_a?
        run_owned_cached_product(correction_plan,t,store,cache,*residual_value,correction.value,work,team,&correction.extent):
        run_owned_product(correction_plan,t,store,u.value,*residual_value,correction.value,work,team,&correction.extent);
    if(correction_error)return correction_error;
    correction.extent_known=true;rho.value.reset();product.value.reset();release_row_cache(cache);
    correction.first_limb=newton_contract::correction_offset(p.m)-size_t(correction_plan.output_first_byte/8);
    correction.word_count=p.n-p.m+1;correction.negative=!negative;
    auto first=linear_input(coarse);first.exponent2=int64_t(64*(p.n-p.m));
    if(auto e=run_stored_linear(p.finish,t,store,first,linear_input(correction),result,w,&team))return e;
    require(!result.negative,SBN3_FATAL_MATH,"stored quotient negative result");out=std::move(result);return {};
}
namespace {
bool quotient_scale(const StoredInteger&a,const StoredInteger&d,const SourceWindow&as,const SourceWindow&ds,size_t n,int64_t&out){
    const __int128 ae=__int128(a.exponent2)+64*(__int128(as.first_limb)-a.first_limb-as.low_zero_words)-as.bit_shift;
    const __int128 de=__int128(d.exponent2)+64*(__int128(ds.first_limb)-d.first_limb-ds.low_zero_words)-ds.bit_shift;
    const __int128 e=ae-de-64*__int128(n);if(e<INT64_MIN||e>INT64_MAX)return false;out=int64_t(e);return true;
}
}
planning::ExternalPlanStatus query_quotient_sources(SourceWindow numerator,SourceWindow denominator,size_t n,
    const InverseOptions&options,const InverseBudget&budget,const planning::ExternalProductProfile&profile,QuotientPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    QuotientPlan p{};
    if(n<4||n>(size_t(1)<<31)||!resolve_source_window(n,numerator)||numerator.limbs!=n||
       !resolve_source_window(n,denominator)||denominator.limbs!=n)return ExternalPlanStatus::invalid;
    p.numerator=numerator;p.denominator=denominator;p.source_capacity_words[0]=numerator.storage_limbs;p.source_capacity_words[1]=denominator.storage_limbs;
    p.target=n;const size_t m=newton_contract::next_precision(n);
    RegionLayout input{};if(!region(p.numerator.storage_limbs,input))return ExternalPlanStatus::unsupported;
    auto inverse_budget=budget;
    if(!add_size(budget.other_live_bytes,sizeof(QuotientPlan)-sizeof(InversePlan),inverse_budget.other_live_bytes)||
       !add(budget.other_owned_disk_bytes,input.allocation_bytes,inverse_budget.other_owned_disk_bytes))return ExternalPlanStatus::unsupported;
    auto prefix=denominator;prefix.first_limb+=n-m;prefix.limbs=m;
    auto status=query_inverse_source(prefix,m,options,inverse_budget,profile,p.inverse);
    if(status!=ExternalPlanStatus::ready)return status;
    size_t fixed;
    if(!add_size(budget.other_live_bytes,sizeof(QuotientPlan),fixed)||!add_size(fixed,engine_shared_bytes(),fixed))return ExternalPlanStatus::unsupported;
    planning::ExternalProductBudget grant{aligned_phase_budget(budget.resident_bytes,fixed,p.inverse.alignment),fixed-p48_root_bank::bytes(),budget.owned_disk_limit,
                                         budget.other_owned_disk_bytes,UINT64_MAX};
    status=query_quotient_rung(m,n,p.numerator,p.denominator,options.workers,grant,profile,p.refinement);
    if(status!=ExternalPlanStatus::ready)return status;
    p.alignment=p.inverse.alignment;
    if(!align_size(std::max(p.inverse.storage_bytes,p.refinement.storage_bytes),p.alignment,p.storage_bytes)||
       !add_size(fixed,p.storage_bytes,p.total_resident_bytes)||!add(p.inverse.write_bytes,p.refinement.write_bytes,p.write_bytes)||
       !add(p.inverse.worst_write_bytes,p.refinement.write_bytes,p.worst_write_bytes))return ExternalPlanStatus::unsupported;
    p.peak_backing_bytes=std::max(p.inverse.peak_backing_bytes,p.refinement.peak_backing_bytes);
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.worst_write_bytes>budget.write_limit)
        return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
planning::ExternalPlanStatus query_stored_quotient(const StoredInteger&a,const StoredInteger&d,size_t n,
    const InverseOptions&options,const InverseBudget&budget,const planning::ExternalProductProfile&profile,QuotientPlan&out) noexcept {
    SourceWindow as,ds;int64_t exponent;
    if(!normalized_source(a,n,as)||!normalized_source(d,n,ds)||!quotient_scale(a,d,as,ds,n,exponent))return planning::ExternalPlanStatus::invalid;
    const auto status=query_quotient_sources(as,ds,n,options,budget,profile,out);
    if(status==planning::ExternalPlanStatus::ready)out.result_exponent2=exponent;return status;
}
bool rebind_stored_quotient(QuotientPlan&p,const StoredInteger&a,const StoredInteger&d) noexcept {
    SourceWindow as,ds;int64_t exponent;
    if(!normalized_source(a,p.target,as)||!normalized_source(d,p.target,ds)||
       as.storage_limbs>p.source_capacity_words[0]||ds.storage_limbs>p.source_capacity_words[1]||
       !quotient_scale(a,d,as,ds,p.target,exponent)||!rebind_stored_inverse(p.inverse,d))return false;
    p.numerator=as;p.denominator=ds;p.result_exponent2=exponent;auto&r=p.refinement;r.numerator=as;r.denominator=ds;
    auto high=as;high.first_limb+=p.target-r.m;high.limbs=r.m;
    return rebind_spilled_sources(r.coarse_product,r.coarse_product.spec.a_source,high)&&
        rebind_spilled_sources(r.residual_product,r.residual_product.spec.a_source,ds)&&
        (!r.residual_product.spec.cyclic.stored_subtract||rebind_spilled_subtract(r.residual_product,as));
}
IoError run_stored_quotient(const QuotientPlan&p,Transport&t,ValueStore&store,const StoredInteger&a,const StoredInteger&d,
    StoredInteger&out,sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    SourceWindow as,ds;int64_t exponent;
    if(out.value||!a.value||!d.value||a.value.store()!=&store||d.value.store()!=&store||
       !normalized_source(a,p.target,as)||as!=p.numerator||!normalized_source(d,p.target,ds)||ds!=p.denominator||
       !quotient_scale(a,d,as,ds,p.target,exponent)||exponent!=p.result_exponent2||team.arena!=&arena||team.busy||
       p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||!arena.unleased(offset,p.storage_bytes)||
       (uintptr_t(arena.base+offset)&(p.alignment-1)))return {IoCode::invalid,"stored quotient binding"};
    if(t.counters.write_submitted>t.write_limit||p.worst_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored quotient write admission",0,p.worst_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    StoredInteger inverse,result;
    if(auto e=run_stored_inverse(p.inverse,t,store,d,inverse,arena,offset,team))return e;
    auto lease=arena.acquire(offset,p.storage_bytes);
    struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){sbn3_arena_release(&a,&l);}} release{arena,lease};
    if(auto e=run_quotient_rung(p.refinement,t,store,a,d,inverse,result,{arena.base+offset,p.storage_bytes},team))return e;
    result.exponent2=p.result_exponent2;out=std::move(result);return {};
}
} // namespace sbn::v3::external
