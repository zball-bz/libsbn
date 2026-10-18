// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/series_merge.hpp"
#include "external/owned_product.hpp"
#include "common/checked.hpp"
#include "product/cost_model.hpp"
#include <algorithm>
#include <cmath>
#include <utility>
namespace sbn::v3::external {
namespace {
using planning::ExternalPlanStatus;
size_t product_keep(const StoredMergePlan&p,unsigned out){return p.precision+(p.rounding==StoredRounding::tail&&out<9);}
bool bytes(const IntegerShape&s,uint64_t &n){auto source=s.source;return resolve_source_window(s.limbs,source)&&!__builtin_mul_overflow(uint64_t(source.storage_limbs),uint64_t(8),&n);}
bool allocation(const IntegerShape&s,uint64_t &n){uint64_t logical;RegionLayout l{};
    if(!bytes(s,logical)||region_layout(logical,0,l))return false;n=l.allocation_bytes;return true;}
bool compose(StoredMergePlan&p,const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile){
    using Kind=StoredMergeStage::Kind;
    unsigned uses[11];std::copy(std::begin(p.uses),std::end(p.uses),uses);
    uint64_t owned[11]{},held=budget.other_owned_disk_bytes,cache_bytes=0;bool alive[11]{};
    for(unsigned i=0;i<6;++i)if(uses[i]||(p.retained_inputs&(1u<<i))){
        if((!(p.resident_inputs&(1u<<i))&&!allocation(p.shape[i],owned[i]))||__builtin_add_overflow(held,owned[i],&held))return false;alive[i]=true;
    }
    p.peak_backing_bytes=held;p.normal_write_bytes=0;p.storage_bytes=0;p.estimated_ns=0;p.alignment=4096;
    for(unsigned i=0;i<p.stage_count;++i){auto&s=p.stage[i];auto&r=p.shape[s.out];
        if(p.cached_pair&&i==p.cache_first){
            uint64_t peak;if(__builtin_add_overflow(held,p.cache.owned_disk_bytes,&peak))return false;
            p.peak_backing_bytes=std::max(p.peak_backing_bytes,peak);cache_bytes=p.cache.owned_disk_bytes;
            size_t build;if(!add_size(p.cache.storage_bytes,p.cache.build_bytes,build))return false;
            p.storage_bytes=std::max(p.storage_bytes,build);
            if(__builtin_add_overflow(p.normal_write_bytes,p.cache.normal_write_bytes,&p.normal_write_bytes))return false;
            p.estimated_ns+=planning::estimate_row_cache(p.cache,profile);
            if(!(p.retained_inputs&(1u<<p.cache_common))){held-=owned[p.cache_common];alive[p.cache_common]=false;}
        }
        size_t work=0;uint64_t scratch=0,write=0;
        if(s.kind==Kind::product){
            if(__builtin_add_overflow(p.shape[s.a].exponent2,p.shape[s.b].exponent2,&r.exponent2))return false;
            const auto&q=s.product;r.limbs=size_t(q.output_bytes/8);
            if(p.precision)r.limbs=std::min(r.limbs,product_keep(p,s.out));
            r.source={size_t(q.max_stored_output_bytes/8),0,r.limbs};
            if(s.use_resident){
                const bool paired=p.pair_first>=0&&(i==unsigned(p.pair_first)||i==unsigned(p.pair_first+2));
                work=paired?p.native_pair.storage_bytes:s.resident.storage_bytes;write=s.borrow_result?0:s.resident_write_bytes;
                p.alignment=std::max(p.alignment,s.resident.alignment);
                // Promotion removes the external plane traffic. The arithmetic
                // volume estimate remains conservative until qualified natively.
                auto compute=q;compute.normal_read_bytes=compute.normal_write_bytes=0;compute.spec.retained_bytes=0;
                const double traffic=double((s.resident.borrowed_inputs&1?0:s.resident.source[0].limbs)+
                                            (s.resident.borrowed_inputs&2?0:s.resident.source[1].limbs))*8+write;
                p.estimated_ns+=planning::estimate_external_product(compute,profile)+traffic/profile.io_bytes_per_ns;
                if(paired&&i==unsigned(p.pair_first))p.estimated_ns-=planning::estimate_forward_rows(q.forward[1],profile);
            }else{
                work=q.storage_bytes;write=q.normal_write_bytes;
                if(__builtin_mul_overflow(q.plane_allocation_bytes,uint64_t(q.plane_count),&scratch))return false;
                p.estimated_ns+=planning::estimate_external_product(q,profile);
            }
        }else if(s.kind==Kind::sum){
            if(!p.precision&&!query_stored_sum(p.shape[s.a].limbs,p.shape[s.a].exponent2,p.shape[s.b].limbs,p.shape[s.b].exponent2,s.sum,0,s.sum.workers))return false;
            r={s.sum.result_limbs,s.sum.exponent};work=s.sum.storage_bytes;write=s.sum.write_bytes;
            // Conservative sign-comparison scan; the same scenario prices both
            // proposals. It is not a measured sum latency or an IO upper proof.
            const double traffic=16.*(double(s.sum.an)+s.sum.bn)+write;
            p.estimated_ns+=traffic/profile.io_bytes_per_ns+traffic/profile.memory_bytes_per_ns+
                traffic/(4*chunk_bytes)*profile.request_ns;
        }else r.limbs=0;
        size_t phase=work;
        if(cache_bytes&&!add_size(p.cache.storage_bytes,work,phase))return false;
        p.storage_bytes=std::max(p.storage_bytes,phase);
        if(s.borrow_result)owned[s.out]=0;
        else if(!allocation(r,owned[s.out]))return false;
        uint64_t peak;
        if(__builtin_add_overflow(held,cache_bytes,&peak)||__builtin_add_overflow(peak,scratch,&peak)||
           __builtin_add_overflow(peak,owned[s.out],&peak)||__builtin_add_overflow(p.normal_write_bytes,write,&p.normal_write_bytes)||
           __builtin_add_overflow(held,owned[s.out],&held))return false;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,peak);alive[s.out]=true;
        for(unsigned source:{s.a,s.b})if(!--uses[source]&&alive[source]&&!(source<6&&(p.retained_inputs&(1u<<source)))){
            held-=owned[source];alive[source]=false;
        }
        if(p.cached_pair&&i==p.cache_last)cache_bytes=0;
    }
    for(unsigned i=0;i<6;++i)if(p.resident_inputs&(1u<<i)){
        size_t end;if(!add_size(p.input_offsets[i],p.input_bytes[i],end))return false;p.storage_bytes=std::max(p.storage_bytes,end);
    }
    return align_size(p.storage_bytes,p.alignment,p.storage_bytes)&&add_size(budget.other_live_bytes,p.shared_bytes,p.total_resident_bytes)&&
        add_size(p.total_resident_bytes,p.storage_bytes,p.total_resident_bytes)&&std::isfinite(p.estimated_ns);
}
bool admitted(const StoredMergePlan&p,const planning::ExternalProductBudget&b){
    return p.total_resident_bytes<=b.resident_bytes&&p.peak_backing_bytes<=b.owned_disk_limit&&p.normal_write_bytes<=b.write_limit;
}
void try_cached_pair(StoredMergePlan&p,const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile){
    using Kind=StoredMergeStage::Kind;
    unsigned first=p.stage_count,last=p.stage_count;
    for(unsigned i=0;i<p.stage_count;++i)if(p.stage[i].kind==Kind::product){
        if(p.stage[i].out==7)first=i;if(p.stage[i].out==9)last=i;
    }
    if(first>=last||last==p.stage_count||p.stage[first].use_resident||p.stage[last].use_resident)return;
    const auto&a=p.stage[first].product,&b=p.stage[last].product;
    if(a.spec.bn!=b.spec.bn||a.spec.b_source!=b.spec.b_source||a.spec.primes!=b.spec.primes||a.spec.trunk_bits!=b.spec.trunk_bits||
       a.spec.outer_cells!=b.spec.outer_cells||a.spec.inner_cells!=b.spec.inner_cells||a.spec.cyclic||b.spec.cyclic)return;
    // Reuse a compatible Fourier prefix; never enlarge the smaller consumer's
    // transform support just to share a cache. Each fixed geometry is replayed
    // once with the common operand in A's role, without another basis choice.
    const auto&anchor=a.forward[1].spec.live_rows>=b.forward[1].spec.live_rows?a:b;
    StoredMergePlan candidate=p;candidate.cached_pair=true;candidate.cache_first=first;candidate.cache_last=last;
    if(!query_row_cache(anchor.forward[1],candidate.cache))return;
    size_t fixed;
    if(!add_size(budget.other_live_bytes,p.shared_bytes,fixed)||!add_size(fixed,candidate.cache.storage_bytes,fixed)||fixed>budget.resident_bytes)return;
    const size_t room=budget.resident_bytes-fixed;
    if(candidate.cache.build_bytes>room)return;
    __uint128_t original_io=0;
    for(unsigned index:{first,last}){
        const auto&original=p.stage[index].product;auto spec=original.spec;
        std::swap(spec.an,spec.bn);std::swap(spec.a_source,spec.b_source);spec.retained_bytes=0;spec.raw_operand=2;
        SpilledProductPlan apply{};++p.summary_quotes;++candidate.summary_quotes;
        if(!query_spilled_product(spec,apply)||!use_cached_a(apply)||
           !bound_spilled_output(apply,size_t(original.max_stored_output_bytes/8))||apply.storage_bytes>room)return;
        apply.emit.bound=original.emit.bound;apply.emit.scalar=original.emit.scalar;
        const size_t held=size_t(std::min(uint64_t((room-apply.storage_bytes)/chunk_bytes*chunk_bytes),uint64_t(spec.primes)*apply.plane_bytes));
        if(!retain_spilled_prefix(apply,held))return;
        original_io+=__uint128_t(original.normal_read_bytes)+original.normal_write_bytes;
        auto&s=candidate.stage[index];const unsigned fresh=s.a;s.a=candidate.cache_common;s.b=fresh;s.product=apply;
    }
    if(!compose(candidate,budget,profile)||!admitted(candidate,budget))return;
    SpilledProductPlan*consumers[]{&candidate.stage[first].product,&candidate.stage[last].product};
    if(planning::complete_cache_within(candidate.cache,consumers,candidate.storage_bytes-candidate.cache.storage_bytes,profile))
        if(!compose(candidate,budget,profile)||!admitted(candidate,budget))return;
    __uint128_t cached_io=__uint128_t(candidate.cache.normal_read_bytes)+candidate.cache.normal_write_bytes;
    for(unsigned index:{first,last})cached_io+=__uint128_t(candidate.stage[index].product.normal_read_bytes)+candidate.stage[index].product.normal_write_bytes;
    if(cached_io>original_io)return;
    if(!admitted(p,budget)||(candidate.normal_write_bytes<=p.normal_write_bytes&&
       (candidate.estimated_ns<p.estimated_ns||
        (candidate.estimated_ns<=p.estimated_ns*profile.write_saving_slowdown&&candidate.normal_write_bytes<p.normal_write_bytes))))p=candidate;
}
void borrow_products(StoredMergePlan&p,size_t room){
    // A single-use resident product may survive until its linear consumer,
    // including across another product/cache build. Visit producers backward:
    // later loans are already included in each intervening stage's complete
    // footprint, so earlier loans are disjoint without an interval search.
    for(unsigned i=p.stage_count;i-- >0;){auto&s=p.stage[i];
        if(s.kind!=StoredMergeStage::Kind::product||!s.use_resident||p.uses[s.out]!=1)continue;
        if(p.pair_first>=0&&(i==unsigned(p.pair_first)||i==unsigned(p.pair_first+2)))continue;
        unsigned consumer=p.stage_count;
        for(unsigned j=i+1;j<p.stage_count;++j)if(p.stage[j].a==s.out||p.stage[j].b==s.out){consumer=j;break;}
        if(consumer==p.stage_count||p.stage[consumer].kind!=StoredMergeStage::Kind::sum)continue;
        size_t prefix=0;bool valid=true;
        for(unsigned j=i+1;j<=consumer;++j){const auto&next=p.stage[j];
            size_t phase=next.kind==StoredMergeStage::Kind::product?(next.use_resident?next.resident.storage_bytes:next.product.storage_bytes):next.sum.storage_bytes;
            if(p.pair_first>=0&&(j==unsigned(p.pair_first)||j==unsigned(p.pair_first+2)))phase=p.native_pair.storage_bytes;
            if(p.cached_pair&&j>=p.cache_first&&j<=p.cache_last){
                if(j==p.cache_first)phase=std::max(phase,p.cache.build_bytes);
                if(!add_size(phase,p.cache.storage_bytes,phase)){valid=false;break;}
            }
            prefix=std::max(prefix,phase);
        }
        auto held=s.resident;
        const size_t limit=p.resident_inputs?std::min(room,p.stage_work_limit[i]):room;
        if(valid&&reserve_resident_result_after(held,prefix)&&held.storage_bytes<=limit){s.resident=held;s.borrow_result=true;}
    }
}
bool promote_products(StoredMergePlan&p,const planning::ExternalProductBudget&budget,unsigned workers,
                      const planning::ExternalProductProfile&profile){
    size_t fixed;if(!add_size(budget.other_live_bytes,p.shared_bytes,fixed)||fixed>=budget.resident_bytes)return false;
    const size_t room=budget.resident_bytes-fixed;
    for(unsigned i=0;i<p.stage_count;++i){auto&s=p.stage[i];
        if(s.kind!=StoredMergeStage::Kind::product||(p.cached_pair&&i>=p.cache_first&&i<=p.cache_last))continue;
        sbn3_mul_options options{};options.workers=workers;options.algorithm=SBN3_MUL_BAILEY;options.borrow_output=1;
        ResidentProductPlan native{};++p.summary_quotes;++p.basis_queries;
        if(query_resident_product(s.product.spec.an,s.product.spec.bn,options,1u<<20,native)!=SBN3_SUPPORTED)continue;
        if(native.storage_bytes>room){++p.summary_quotes;if(!compact_resident_product(native))continue;}
        if(native.storage_bytes>room||!set_resident_sources(native,s.product.spec.a_source,s.product.spec.b_source))continue;
        native.maximum_output_limbs=size_t(s.product.max_stored_output_bytes/8);
        RegionLayout output;if(region_layout(native.maximum_output_limbs*8,0,output))continue;
        s.resident=native;s.use_resident=true;s.borrow_result=false;s.resident_write_bytes=output.data_bytes+output.index_bytes;
    }
    borrow_products(p,room);
    return compose(p,budget,profile);
}
void try_native_pair(StoredMergePlan&p,size_t limit,const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile){
    if(p.cached_pair||p.pair_first>=0)return;
    unsigned first=p.stage_count,second=p.stage_count;
    for(unsigned i=0;i<p.stage_count;++i){if(p.stage[i].out==7)first=i;if(p.stage[i].out==9)second=i;}
    if(first>=second||second!=first+2||!p.stage[first].use_resident||!p.stage[second].use_resident)return;
    auto next=p;
    // N product, rounded numerator sum, D product: the first result is
    // consumed at the joined boundary while its common spectrum stays live.
    next.pair_first=int(first);
    auto&a=next.stage[first];auto&b=next.stage[second];
    ++p.summary_quotes;
    if(!query_resident_pair(a.resident,b.resident,next.stage[first+1].sum.storage_bytes,next.native_pair))return;
    const unsigned mask=p.resident_inputs;
    if(mask&&!borrow_resident_pair_inputs(next.native_pair,((mask>>a.b)&1u)|(((mask>>a.a)&1u)<<1)|(((mask>>b.a)&1u)<<2)))return;
    const auto cost=[](const ProductProgramPlan&q){return cost_model::linear_product(q.info,q.an,q.bn).nanoseconds;};
    const double pair_cost=(5./3)*cost(next.native_pair.product);
    if(pair_cost>=cost(a.resident.product)+cost(b.resident.product))return;
    size_t fixed;if(!add_size(budget.other_live_bytes,p.shared_bytes,fixed)||fixed>=budget.resident_bytes)return;
    const size_t room=std::min(limit,budget.resident_bytes-fixed);if(next.native_pair.storage_bytes>room)return;
    if(mask){
        unsigned last[6]{};
        for(unsigned j=0;j<next.stage_count;++j)for(unsigned i:{next.stage[j].a,next.stage[j].b})if(i<6&&(mask&(1u<<i)))last[i]=j;
        // Both fresh operands are read by the first paired stage. The old
        // slot placement is preserved so the producer's handoff stays valid.
        for(unsigned j=0;j<next.stage_count;++j){size_t space=room;
            for(unsigned i=0;i<6;++i)if((mask&(1u<<i))&&last[i]>=j)space=std::min(space,next.input_offsets[i]);
            next.stage_work_limit[j]=space;
        }
        if(next.native_pair.storage_bytes>next.stage_work_limit[first])return;
    }
    for(auto&s:next.stage)if(s.use_resident){s.borrow_result=false;if(!reserve_resident_result_after(s.resident,0))return;}
    a.borrow_result=true;
    borrow_products(next,room);
    if(!compose(next,budget,profile)||next.storage_bytes>room||!admitted(next,budget)||next.normal_write_bytes>p.normal_write_bytes||
       next.peak_backing_bytes>p.peak_backing_bytes)return;
    next.summary_quotes=p.summary_quotes;p=next;
}
}
void share_stored_merge_transform(StoredMergePlan&p,size_t limit,const planning::ExternalProductBudget&budget,
    const planning::ExternalProductProfile&profile) noexcept {try_native_pair(p,limit,budget,profile);}
bool lend_stored_merge_inputs(StoredMergePlan&plan,unsigned mask,const std::array<size_t,6>&bytes,size_t limit,
    const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile) noexcept {
    if(!mask||(mask&~63u)||plan.cached_pair||plan.resident_inputs||(mask&plan.retained_inputs)||
       !limit||(limit&(plan.alignment-1)))return false;
    auto p=plan;p.resident_inputs=mask;unsigned order[6],last[6]{},count=0;
    for(unsigned i=0;i<6;++i)if(mask&(1u<<i)){
        auto source=p.shape[i].source;
        if(!p.uses[i]||!bytes[i]||bytes[i]%8||!resolve_source_window(p.shape[i].limbs,source)||bytes[i]/8<source.storage_limbs)return false;
        order[count++]=i;p.input_bytes[i]=bytes[i];
    }
    for(unsigned j=0;j<p.stage_count;++j){const auto&s=p.stage[j];
        if(s.kind==StoredMergeStage::Kind::product&&!s.use_resident)return false;
        for(unsigned i:{s.a,s.b})if(i<6&&(mask&(1u<<i)))last[i]=j;
    }
    std::sort(order,order+count,[&](unsigned a,unsigned b){return last[a]<last[b]||(last[a]==last[b]&&a<b);});
    size_t at=limit;
    for(unsigned k=count;k-- >0;){const unsigned i=order[k];
        if(bytes[i]>at)return false;at=(at-bytes[i])&~size_t(63);p.input_offsets[i]=at;
    }
    for(unsigned j=0;j<p.stage_count;++j){auto&s=p.stage[j];size_t room=limit;
        for(unsigned k=0;k<count;++k)if(last[order[k]]>=j)room=std::min(room,p.input_offsets[order[k]]);
        p.stage_work_limit[j]=room;
        if(s.kind==StoredMergeStage::Kind::product){
            s.borrow_result=false;
            if(!reserve_resident_result_after(s.resident,0)||!borrow_resident_inputs(s.resident,
                ((mask>>s.a)&1u)|(((mask>>s.b)&1u)<<1))||s.resident.storage_bytes>room)return false;
        }else if(s.kind==StoredMergeStage::Kind::sum&&s.sum.storage_bytes>room)return false;
    }
    if(p.pair_first>=0){const unsigned i=unsigned(p.pair_first),common=p.stage[i].b;
        if(!borrow_resident_pair_inputs(p.native_pair,((mask>>common)&1u)|(((mask>>p.stage[i].a)&1u)<<1)|
            (((mask>>p.stage[i+2].a)&1u)<<2))||p.native_pair.storage_bytes>p.stage_work_limit[i])return false;
        p.stage[i].borrow_result=true;
    }
    borrow_products(p,limit);
    if(!compose(p,budget,profile)||!admitted(p,budget)||p.storage_bytes>limit)return false;
    plan=p;return true;
}
ExternalPlanStatus query_stored_merge(sbn3_series_recipe recipe,unsigned need,
    const std::array<IntegerShape,6>&inputs,unsigned workers,
    const planning::ExternalProductBudget&budget,const planning::ExternalProductProfile&profile,
    StoredMergePlan&out,unsigned retained_inputs,size_t precision,StoredRounding rounding,bool bounded_inputs) noexcept {
    using Kind=StoredMergeStage::Kind;
    if((need!=3 && need!=7)||recipe<SBN3_SERIES_HYPERDESCENT||recipe>SBN3_SERIES_BINARY_BBP||
       (need==7 && recipe!=SBN3_SERIES_COMMON_P2B3)||(precision&&rounding==StoredRounding::tail&&need!=3)||
       (bounded_inputs&&!precision)||!workers||(retained_inputs&~63u)||precision>SIZE_MAX-2)return ExternalPlanStatus::invalid;
    StoredMergePlan p{};p.recipe=recipe;p.need=need;p.retained_inputs=retained_inputs;p.precision=precision;
    p.rounding=rounding;p.bounded_inputs=bounded_inputs;
    for(unsigned i=0;i<6;++i)p.shape[i]=inputs[i];
    auto product=[&](unsigned a,unsigned b,unsigned dest){
        auto&s=p.stage[p.stage_count++];s.kind=Kind::product;s.a=a;s.b=b;s.out=dest;
        ++p.uses[a];++p.uses[b];
    };
    unsigned secondary=3;
    if(recipe!=SBN3_SERIES_HYPERDESCENT){product(recipe==SBN3_SERIES_COMMON_P2B3?2:1,3,6);secondary=6;}
    product(0,4,7);
    {auto&s=p.stage[p.stage_count++];s.kind=Kind::sum;s.a=7;s.b=secondary;s.out=8;++p.uses[7];++p.uses[secondary];}
    product(1,4,9);
    if(need&4)product(2,5,10);
    ++p.uses[8];++p.uses[9];if(need&4)++p.uses[10];
    unsigned uses[11];std::copy(std::begin(p.uses),std::end(p.uses),uses);
    uint64_t held=0,owned[11]{};
    for(unsigned i=0;i<6;++i)if(uses[i]||(retained_inputs&(1u<<i))){
        if(!allocation(p.shape[i],owned[i])||__builtin_add_overflow(held,owned[i],&held))return ExternalPlanStatus::unsupported;
    }
    uint64_t initial;if(__builtin_add_overflow(held,budget.other_owned_disk_bytes,&initial))return ExternalPlanStatus::unsupported;
    p.peak_backing_bytes=initial;
    if(initial>budget.owned_disk_limit)return ExternalPlanStatus::no_admitted_plan;
    for(unsigned i=0;i<p.stage_count;++i){auto&s=p.stage[i];const auto a=p.shape[s.a],b=p.shape[s.b];
        auto&r=p.shape[s.out];
        uint64_t peak=0,write=0;size_t space=0,total=0;
        if(s.kind==Kind::sum){
            // A limited sum has at most P+1 aligned words and one carry word.
            // Actual low exponent is instantiated cheaply from result headers.
            if(precision){if(!query_stored_sum(precision+1,0,precision+1,0,s.sum,precision,workers))return ExternalPlanStatus::unsupported;}
            else if(!query_stored_sum(a.limbs,a.exponent2,b.limbs,b.exponent2,s.sum,0,workers))return ExternalPlanStatus::unsupported;
            r={s.sum.result_limbs,s.sum.exponent};space=s.sum.storage_bytes;write=s.sum.write_bytes;
            if(!add_size(space,budget.other_live_bytes,total)||
               __builtin_add_overflow(initial,s.sum.output_allocation,&peak))return ExternalPlanStatus::unsupported;
        }else{
            if(__builtin_add_overflow(a.exponent2,b.exponent2,&r.exponent2))return ExternalPlanStatus::unsupported;
            if(!a.limbs||!b.limbs){s.kind=Kind::zero;r.limbs=0;peak=initial;total=budget.other_live_bytes;}
            else{
                const size_t an=precision&&rounding==StoredRounding::tail&&s.a==0?std::min(a.limbs,precision):a.limbs,bn=b.limbs;
                if(!add_size(an,bn,r.limbs))return ExternalPlanStatus::unsupported;
                auto grant=budget;
                grant.other_owned_disk_bytes=initial-owned[s.a]-owned[s.b];
                // Compose shared producers before enforcing episode disk/write
                // caps. A separate-product quote must not veto a legal cache.
                grant.owned_disk_limit=UINT64_MAX;grant.write_limit=UINT64_MAX;
                planning::ExternalProductSelection choice{};
                auto source_a=a.source,source_b=b.source;
                if(!resolve_source_window(a.limbs,source_a)||!resolve_source_window(b.limbs,source_b))return ExternalPlanStatus::invalid;
                source_a.limbs=std::min(source_a.limbs,an);
                const size_t keep=product_keep(p,s.out);
                const size_t stored=precision&&r.limbs>keep&&r.limbs-keep>chunk_bytes/8?keep+chunk_bytes/8:r.limbs;
                const auto status=planning::choose_external_product({an,bn,workers,0,0,false,source_a,source_b,stored},grant,profile,choice);
                p.summary_quotes+=choice.summary_quotes;p.basis_queries+=choice.basis_queries;
                if(status!=ExternalPlanStatus::ready)return status;
                s.product=choice.plan;space=s.product.storage_bytes;total=choice.total_resident_bytes;
                p.shared_bytes=std::max(p.shared_bytes,s.product.shared_bytes);peak=choice.owned_disk_bytes;write=s.product.normal_write_bytes;
                if(precision){r.limbs=std::min(r.limbs,keep);r.source={size_t(s.product.stored_output_bytes/8),0,r.limbs};}
            }
        }
        if(total>budget.resident_bytes||__builtin_add_overflow(p.normal_write_bytes,write,&p.normal_write_bytes))
            return ExternalPlanStatus::no_admitted_plan;
        p.storage_bytes=std::max(p.storage_bytes,space);p.total_resident_bytes=std::max(p.total_resident_bytes,total);
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,peak);
        if(!allocation(r,owned[s.out])||__builtin_add_overflow(held,owned[s.out],&held))return ExternalPlanStatus::unsupported;
        for(unsigned source:{s.a,s.b})if(!--uses[source] && !(source<6&&(retained_inputs&(1u<<source))))held-=owned[source];
        if(__builtin_add_overflow(held,budget.other_owned_disk_bytes,&initial))return ExternalPlanStatus::unsupported;
    }
    // The shared bank stays resident during linear/zero stages too.
    size_t complete;if(!add_size(budget.other_live_bytes,p.shared_bytes,complete)||
        !add_size(complete,p.storage_bytes,complete))return ExternalPlanStatus::unsupported;
    if(complete>budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    p.total_resident_bytes=complete;
    if(!compose(p,budget,profile))return ExternalPlanStatus::unsupported;
    const unsigned initial_quotes=p.summary_quotes,initial_bases=p.basis_queries;
    auto promoted=p;const bool native_ready=promote_products(promoted,budget,workers,profile);
    unsigned quotes=promoted.summary_quotes,bases=promoted.basis_queries;
    try_cached_pair(p,budget,profile);
    quotes+=p.summary_quotes-initial_quotes;bases+=p.basis_queries-initial_bases;
    if(p.cached_pair){auto mixed=p;
        const auto before_quotes=mixed.summary_quotes,before_bases=mixed.basis_queries;
        const bool mixed_ready=promote_products(mixed,budget,workers,profile);
        quotes+=mixed.summary_quotes-before_quotes;bases+=mixed.basis_queries-before_bases;
        if(mixed_ready&&admitted(mixed,budget)&&
           mixed.normal_write_bytes<=p.normal_write_bytes&&mixed.estimated_ns<=p.estimated_ns*profile.write_saving_slowdown)p=mixed;
    }
    if(native_ready&&admitted(promoted,budget)&&(!admitted(p,budget)||
       (promoted.normal_write_bytes<=p.normal_write_bytes&&promoted.estimated_ns<=p.estimated_ns*profile.write_saving_slowdown)))p=promoted;
    p.summary_quotes=quotes;p.basis_queries=bases;
    if(!bounded_inputs)try_native_pair(p,SIZE_MAX,budget,profile);
    if(!admitted(p,budget))return ExternalPlanStatus::no_admitted_plan;
    out=p;return ExternalPlanStatus::ready;
}
IoError run_stored_merge(const StoredMergePlan&p,Transport&t,ValueStore&store,
    StoredTriple&left,StoredTriple&right,StoredTriple&out,std::span<uint8_t>w,sbn3_team&team,std::span<const ResidentInteger>loans) noexcept {
    if(!p.stage_count||p.stage_count>5||&left==&right||&out==&left||&out==&right||
       !w.data()||(uintptr_t(w.data())&(p.alignment-1))||w.size()<p.storage_bytes||
       (p.resident_inputs?loans.size()!=6:!loans.empty()))return {IoCode::invalid,"stored merge binding"};
    for(const auto&v:out)if(v.value)return {IoCode::invalid,"stored merge output initialized"};
    for(unsigned i=0;i<6;++i){const auto&v=i<3?left[i]:right[i-3];
        if(p.resident_inputs&(1u<<i)){
            const auto&x=loans[i];const auto base=uintptr_t(w.data()+p.input_offsets[i]),at=uintptr_t(x.words.data());
            auto expected=p.shape[i].source;
            if(v.value||!x.words.data()||at<base||at-base>p.input_bytes[i]||
               x.words.size()>(p.input_bytes[i]-size_t(at-base))/8||!resolve_source_window(p.shape[i].limbs,expected)||
               x.words.size()>expected.storage_limbs||(p.bounded_inputs?x.words.size()>p.shape[i].limbs:
                 x.words.size()!=p.shape[i].limbs||x.exponent2!=p.shape[i].exponent2)||
               x.extent.end>x.words.size())return {IoCode::invalid,"resident merge input loan"};
            continue;
        }
        if((p.uses[i]||(p.retained_inputs&(1u<<i)))&&(!v.value||v.value.store()!=&store||v.value.region().layout.logical_bytes%8||
           (p.bounded_inputs?v.limbs()>p.shape[i].limbs:(v.limbs()!=p.shape[i].limbs||v.exponent2!=p.shape[i].exponent2))))return {IoCode::invalid,"stored merge input shape"};
        if(v.value&&!v.value.unique()&&!(p.retained_inputs&(1u<<i)))return {IoCode::invalid,"stored merge undeclared input pin"};
        if(p.precision&&p.uses[i]&&!v.extent_known)return {IoCode::invalid,"stored merge missing extent"};
        if(p.uses[i]){
            auto expected=p.shape[i].source,actual=v.source();
            if(!resolve_source_window(p.shape[i].limbs,expected)||!resolve_source_window(p.shape[i].limbs,actual)||
               actual.storage_limbs>expected.storage_limbs)return {IoCode::invalid,"stored merge source backing envelope"};
        }
    }
    if(t.counters.write_submitted>t.write_limit||p.normal_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored merge write admission",0,p.normal_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    StoredInteger value[11];unsigned uses[11];std::copy(std::begin(p.uses),std::end(p.uses),uses);
    for(unsigned i=0;i<6;++i){value[i]=std::move(i<3?left[i]:right[i-3]);if(!uses[i])value[i].value.reset();}
    LinearInput borrowed[11]{};
    for(unsigned i=0;i<6;++i)if(p.resident_inputs&(1u<<i)){
        const auto&v=loans[i];borrowed[i]={nullptr,{v.words.size(),0,v.words.size()},0,v.exponent2,v.negative,v.words};
    }
    if(p.precision){
        auto limit=[&](unsigned i,size_t n){return p.resident_inputs&(1u<<i)?limit_resident_input(borrowed[i],loans[i].extent,n):limit_stored_integer(value[i],n);};
        for(unsigned i=0;i<6;++i)if(uses[i])require(limit(i,i==0&&p.rounding==StoredRounding::tail?p.precision:0),SBN3_FATAL_MATH,"stored merge normalize input");
        if(p.recipe==SBN3_SERIES_HYPERDESCENT&&p.rounding==StoredRounding::tail)require(limit(3,p.precision+1),SBN3_FATAL_MATH,"stored hyperdescent tail limit");
    }
    RowCacheBinding cache;
    unsigned completed=0;
    auto release_stage=[&](unsigned j){const auto&s=p.stage[j];
        for(unsigned source:{s.a,s.b})if(!--uses[source]){value[source].value.reset();borrowed[source]={};}
        completed|=1u<<j;
    };
    auto input=[&](unsigned i){return borrowed[i].resident.data()?borrowed[i]:linear_input(value[i]);};
    struct CacheEnd {RowCacheBinding&cache;~CacheEnd(){release_row_cache(cache);}} cache_end{cache};
    for(unsigned i=0;i<p.stage_count;++i){if(completed&(1u<<i))continue;const auto&s=p.stage[i];auto&r=value[s.out];
        if(p.cached_pair&&i==p.cache_first){
            auto &common=value[p.cache_common];
            if(auto e=build_row_cache(p.cache,t,store,common.value,common.source(),w.first(p.cache.storage_bytes),
                w.subspan(p.cache.storage_bytes),team,cache))return e;
            common.value.reset(); // saved rows/residue now own every local read
        }
        auto work=cache.control?w.subspan(p.cache.storage_bytes):p.resident_inputs?w.first(p.stage_work_limit[i]):w;
        if(s.kind==StoredMergeStage::Kind::sum){
            const auto lhs=borrowed[s.a].resident.data()?borrowed[s.a]:linear_input(value[s.a]);
            const auto rhs=borrowed[s.b].resident.data()?borrowed[s.b]:linear_input(value[s.b]);
            auto sum=s.sum;
            if(p.precision){
                require(query_stored_sum(lhs.limbs(),lhs.exponent2,rhs.limbs(),rhs.exponent2,sum,p.precision,s.sum.workers)&&
                    sum.storage_bytes<=work.size()&&sum.output_bytes<=s.sum.output_bytes,SBN3_FATAL_MATH,"stored sum bounded instantiation");
            }
            if(auto e=run_stored_linear(sum,t,store,lhs,rhs,r,work,&team))return e;
        }else if(s.kind==StoredMergeStage::Kind::product){
            if(s.use_resident){
                auto native=s.resident;const auto lhs=input(s.a),rhs=input(s.b);
                const ResidentProductInput a{lhs.value?&lhs.value->region():nullptr,lhs.resident},b{rhs.value?&rhs.value->region():nullptr,rhs.resident};
                require(rebind_resident_sources(native,lhs.source,rhs.source),SBN3_FATAL_WORKSPACE,"resident merge source envelope");
                if(p.precision)require(keep_high_resident_output(native,product_keep(p,s.out)),SBN3_FATAL_WORKSPACE,"resident merge output envelope");
                const __int128 exponent=__int128(lhs.exponent2)+rhs.exponent2+__int128(native.output_first)*64;
                require(exponent>=INT64_MIN&&exponent<=INT64_MAX,SBN3_FATAL_SIZE,"resident merge exponent");
                const bool negative=lhs.negative!=rhs.negative;
                if(p.pair_first>=0&&i==unsigned(p.pair_first)){
                    const auto&second=p.stage[i+2];const auto fresh1=input(second.a);auto pair=p.native_pair;
                    require(rebind_resident_pair(pair,rhs.source,lhs.source,fresh1.source),SBN3_FATAL_WORKSPACE,"merge pair sources");
                    ResidentProductPlan views[]{native,second.resident};
                    require(rebind_resident_sources(views[1],fresh1.source,rhs.source)&&
                        (!p.precision||keep_high_resident_output(views[1],product_keep(p,second.out))),SBN3_FATAL_WORKSPACE,"merge pair output");
                    const __int128 second_exp=__int128(fresh1.exponent2)+rhs.exponent2+__int128(views[1].output_first)*64;
                    require(second_exp>=INT64_MIN&&second_exp<=INT64_MAX,SBN3_FATAL_SIZE,"merge pair exponent");
                    const bool second_negative=fresh1.negative!=rhs.negative;
                    const ResidentProductInput inputs[]{b,a,{fresh1.value?&fresh1.value->region():nullptr,fresh1.resident}};
                    auto consume=[&](const ResidentProductValue&result,sbn3_team_scope*scope)->IoError{
                        borrowed[s.out]={nullptr,{result.words.size(),0,result.words.size()},0,int64_t(exponent),negative,result.words};
                        if(p.precision)require(limit_resident_input(borrowed[s.out],result.extent,product_keep(p,s.out)),SBN3_FATAL_MATH,"paired numerator limit");
                        release_stage(i);
                        const auto&linear=p.stage[i+1];auto sum=linear.sum;const auto x=input(linear.a),y=input(linear.b);
                        if(p.precision)require(query_stored_sum(x.limbs(),x.exponent2,y.limbs(),y.exponent2,sum,p.precision,linear.sum.workers)&&
                            sum.storage_bytes<=pair.consume_bytes&&sum.output_bytes<=linear.sum.output_bytes,SBN3_FATAL_WORKSPACE,"pair linear consumer");
                        if(auto e=run_stored_linear(sum,t,store,x,y,value[linear.out],work.subspan(pair.transfer,pair.consume_bytes),scope?nullptr:&team,scope))return e;
                        release_stage(i+1);return {};
                    };
                    ResidentProductValue result;
                    if(auto e=run_resident_pair(pair,t,inputs,views,work,team,&consume,
                        [](void*ptr,const ResidentProductValue&v,sbn3_team_scope*s){return (*static_cast<decltype(consume)*>(ptr))(v,s);},result))return e;
                    auto&dest=value[second.out];RegionLease lease;if(auto e=store.reserve(result.words.size()*8,RegionKind::value,lease))return e;
                    StoreBuffers io{work.subspan(pair.transfer,pair.transfer_bytes),work.subspan(pair.index,index_page_bytes)};
                    ValueWriter writer;if(auto e=writer.open(t,lease.region(),io,&dest.extent))return e;
                    if(auto e=writer.append({reinterpret_cast<const uint8_t*>(result.words.data()),result.words.size()*8}))return e;
                    if(auto e=writer.seal())return e;if(auto e=lease.publish(dest.value))return e;
                    dest.extent_known=true;dest.exponent2=int64_t(second_exp);dest.negative=second_negative;
                    if(p.precision)require(limit_stored_integer(dest,product_keep(p,second.out)),SBN3_FATAL_MATH,"paired denominator limit");
                    release_stage(i+2);continue;
                }
                if(s.borrow_result){
                    ResidentProductValue result;
                    if(auto e=run_resident_product_borrowed(native,t,a,b,work,team,result))return e;
                    borrowed[s.out]={nullptr,{result.words.size(),0,result.words.size()},0,int64_t(exponent),negative,result.words};
                    if(p.precision)require(limit_resident_input(borrowed[s.out],result.extent,product_keep(p,s.out)),SBN3_FATAL_MATH,"resident merge limit");
                }else{
                    RegionLease result;if(auto e=store.reserve(native.output_limbs*8,RegionKind::value,result))return e;
                    if(auto e=run_resident_product(native,t,a,b,result.region(),work,team,&r.extent))return e;
                    if(auto e=result.publish(r.value))return e;r.extent_known=true;r.exponent2=int64_t(exponent);r.negative=negative;
                    if(p.precision)require(limit_stored_integer(r,product_keep(p,s.out)),SBN3_FATAL_MATH,"resident merge limit");
                }
            }else{
            auto product=s.product;
            const auto source=product.cached_a?row_cache_source(cache):value[s.a].source();
            require(rebind_spilled_sources(product,source,value[s.b].source()),SBN3_FATAL_WORKSPACE,"stored product source envelope");
            if(p.precision)require(keep_high_spilled_output(product,product_keep(p,s.out)),SBN3_FATAL_WORKSPACE,"stored product output envelope");
            const auto error=product.cached_a?run_owned_cached_product(product,t,store,cache,value[s.b].value,r.value,work,team,&r.extent):
                run_owned_product(product,t,store,value[s.a].value,value[s.b].value,r.value,work,team,&r.extent);
            if(error)return error;
            r.extent_known=true;
            const __int128 exponent=__int128(value[s.a].exponent2)+value[s.b].exponent2+__int128(product.output_first_byte)*8;
            require(exponent>=INT64_MIN&&exponent<=INT64_MAX,SBN3_FATAL_SIZE,"stored product exponent");r.exponent2=int64_t(exponent);
            r.negative=value[s.a].negative!=value[s.b].negative;
            if(p.precision)require(limit_stored_integer(r,product_keep(p,s.out)),SBN3_FATAL_MATH,"stored product limit");
            }
        }else{
            RegionLease zero;if(auto e=store.reserve(0,RegionKind::value,zero))return e;
            // Empty values need no payload/index IO, but use the same writer state machine.
            ValueWriter writer;StoreBuffers io{work.first(chunk_bytes),work.subspan(chunk_bytes,index_page_bytes)};
            if(auto e=writer.open(t,zero.region(),io))return e;
            if(auto e=writer.seal())return e;
            if(auto e=zero.publish(r.value))return e;
            r.exponent2=p.shape[s.out].exponent2;
            r.extent_known=true;
        }
        release_stage(i);
        if(p.cached_pair&&i==p.cache_last)release_row_cache(cache);
    }
    out[0]=std::move(value[8]);out[1]=std::move(value[9]);if(p.need&4)out[2]=std::move(value[10]);return {};
}
} // namespace sbn::v3::external
