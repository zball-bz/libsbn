// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/series.hpp"
#include "external/trace.hpp"
#include "external/shared.hpp"
#include "common/checked.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <new>
#include <utility>
namespace sbn::v3::external {
namespace {
constexpr size_t transfer_bytes=1u<<20,controller_allowance=512u<<10;
constexpr unsigned depth_limit=96;
using planning::ExternalPlanStatus;
using planning::SummaryStatus;
bool plus(uint64_t a,uint64_t b,uint64_t&out){return !__builtin_add_overflow(a,b,&out);}
bool value_layout(const IntegerShape&s,RegionLayout&out){
    auto source=s.source;if(!resolve_source_window(s.limbs,source))return false;
    uint64_t bytes;if(__builtin_mul_overflow(uint64_t(source.storage_limbs),uint64_t(8),&bytes))return false;
    return !region_layout(bytes,0,out);
}
bool returned(const std::array<IntegerShape,3>&v,unsigned need,uint64_t&disk,uint64_t&writes){
    disk=writes=0;
    for(unsigned j=0;j<3;++j)if(need&(1u<<j)){RegionLayout l{};
        if(!value_layout(v[j],l)||!plus(disk,l.allocation_bytes,disk)||!plus(writes,l.data_bytes,writes)||!plus(writes,l.index_bytes,writes))return false;
    }
    return true;
}
std::array<IntegerShape,6> inputs(const std::array<IntegerShape,3>&a,const std::array<IntegerShape,3>&b){
    std::array<IntegerShape,6> out{};std::copy(a.begin(),a.end(),out.begin());std::copy(b.begin(),b.end(),out.begin()+3);return out;
}
std::array<IntegerShape,3> outputs(const StoredMergePlan&p){
    std::array<IntegerShape,3> out{};
    for(unsigned j=0;j<3;++j)if(p.need&(1u<<j)){
        out[j]=p.shape[8+j];out[j].exponent2=0;
        if(p.precision)out[j].limbs=std::min(out[j].limbs,p.precision);
        auto src=out[j].source;require(resolve_source_window(p.shape[8+j].limbs,src),SBN3_FATAL_SIZE,"series return envelope");
        out[j].source={src.storage_limbs,0,out[j].limbs};
    }
    return out;
}
struct Compiler {
    SeriesPlan &p;
    SeriesBudget budget;
    const planning::ExternalProductProfile &profile;
    std::span<SeriesNode> capacity;
    series::ProductPlanMemo *memo;
    size_t count=0;
    ExternalPlanStatus error=ExternalPlanStatus::ready;
    planning::ExternalProductBudget arithmetic_budget() const {
        return {budget.resident_bytes,p.fixed_bytes-p48_root_bank::bytes(),UINT64_MAX,0,UINT64_MAX};
    }
    bool add_work(size_t bytes,size_t alignment){
        p.work_bytes=std::max(p.work_bytes,bytes);p.work_alignment=std::max(p.work_alignment,alignment);
        return p.fixed_bytes<=budget.resident_bytes&&p.work_bytes<=budget.resident_bytes-p.fixed_bytes;
    }
    void lend(size_t index,StoredMergePlan&merge,unsigned side){
        auto&n=capacity[index];if(!n.native||n.loan_mask||merge.cached_pair)return;
        std::array<size_t,6> sizes{};unsigned mask=0,common=0;__uint128_t best=0;
        for(unsigned i=0;i<3;++i)if((n.need&(1u<<i))&&merge.uses[side+i]&&n.result[i].limbs){
            if(!mul_size(n.result[i].limbs,8,sizes[side+i]))return;
            mask|=1u<<(side+i);const __uint128_t saving=__uint128_t(sizes[side+i])*(merge.uses[side+i]+1);
            if(saving>best){best=saving;common=1u<<(side+i);}
        }
        if(!mask)return;
        // One complete child, or its most-used/largest value. These are two
        // lifetime forms, not a size/geometry grid. All arithmetic is replayed.
        const unsigned choices[]{mask,common};
        for(unsigned attempt=0;attempt<2;++attempt){const auto take=choices[attempt];if(attempt&&take==mask)break;
            auto candidate=merge;
            const size_t limit=p.work_bytes&~(merge.alignment-1);
            if(!lend_stored_merge_inputs(candidate,take,sizes,limit,arithmetic_budget(),profile))continue;
            // Compact after the native producer joins. Destination may borrow
            // its dead work, but may not overwrite another un-copied return.
            bool alias=false;
            for(unsigned i=0;i<3;++i)if(take&(1u<<(side+i)))for(unsigned j=0;j<3;++j)if(i!=j&&(take&(1u<<(side+j)))){
                const auto start=candidate.input_offsets[side+i],length=sizes[side+i],from=n.layout.offsets[j],bytes=n.slots.slots[j].bytes;
                alias|=start<from+bytes&&from<start+length;
            }
            if(alias)continue;
            uint64_t saved=0,ignored;
            if(!returned(n.result,take>>side,ignored,saved))continue;
            for(unsigned i=0;i<3;++i)if(take&(1u<<(side+i)))
                candidate.estimated_ns+=2.*double(sizes[side+i])/profile.memory_bytes_per_ns;
            if(__uint128_t(candidate.normal_write_bytes)>__uint128_t(merge.normal_write_bytes)+saved||
               candidate.peak_backing_bytes>merge.peak_backing_bytes||
               candidate.estimated_ns>merge.estimated_ns+double(saved)/profile.io_bytes_per_ns)continue;
            n.loan_mask=take>>side;
            for(unsigned i=0;i<3;++i)if(n.loan_mask&(1u<<i))n.loan_offsets[i]=candidate.input_offsets[side+i];
            merge=candidate;return;
        }
    }
    bool refresh(size_t index){
        auto&n=capacity[index];
        if(!returned(n.result,n.need&~n.loan_mask,n.return_bytes,n.write_bytes))return false;
        if(n.native){n.peak_bytes=n.return_bytes;return true;}
        if(!refresh(n.left)||!refresh(n.right))return false;
        const auto&a=capacity[n.right_first?n.right:n.left],&b=capacity[n.right_first?n.left:n.right];uint64_t peak;
        if(!plus(a.return_bytes,b.peak_bytes,peak)||!plus(a.write_bytes,b.write_bytes,n.write_bytes)||
           !plus(n.write_bytes,n.merge.normal_write_bytes,n.write_bytes))return false;
        n.peak_bytes=std::max({a.peak_bytes,peak,n.merge.peak_backing_bytes});return true;
    }
    size_t node(const series::FiniteFormula&formula,sbn3_series_range range,unsigned need,unsigned depth,unsigned forced){
        if(error!=ExternalPlanStatus::ready)return SIZE_MAX;
        if(count==capacity.size()||depth>depth_limit){error=ExternalPlanStatus::no_admitted_plan;return SIZE_MAX;}
        const size_t id=count++;auto&n=capacity[id];n={};n.range=range;n.need=need;p.max_depth=std::max(p.max_depth,depth);
        sbn3_series_shape logical{};
        if(series::finite_value_shape(formula,range,need,logical)!=SBN3_SUPPORTED){error=ExternalPlanStatus::unsupported;return SIZE_MAX;}
        const size_t precision=formula.limit_words?formula.limit_words(formula.limit_context,range.begin):0;
        if(!precision){error=ExternalPlanStatus::unsupported;return SIZE_MAX;}
        for(unsigned j=0;j<3;++j)if(need&(1u<<j))n.result[j]={std::min(logical.limbs[j],precision),0,{}};
        planning::NativeSubtree native{};
        ++p.native_quotes;
        const auto native_status=planning::summarize_native_subtree(formula,range,need,p.spec->series,native,memo);
        if(!forced&&native_status==SummaryStatus::ok){
            for(unsigned which=0;which<2;++which)if(native.summary.candidate[which].supported){
                planning::SlotLayout layout{};const auto&slots=native.summary.candidate[which].slots;
                if(planning::layout(slots,layout)==SummaryStatus::ok&&p.fixed_bytes<=budget.resident_bytes&&layout.bytes<=budget.resident_bytes-p.fixed_bytes){
                    n.native=true;n.finite=native.plans[which];n.slots=slots;n.layout=layout;
                    if(!returned(n.result,need,n.return_bytes,n.write_bytes)||!add_work(layout.bytes,layout.alignment)){error=ExternalPlanStatus::no_admitted_plan;return SIZE_MAX;}
                    n.peak_bytes=n.return_bytes;return id;
                }
            }
        }
        uint64_t middle;
        if(!series::finite_split(formula,range,.5,middle)){error=ExternalPlanStatus::unsupported;return SIZE_MAX;}
        unsigned ln,rn;sbn3_series_child_needs(formula.recipe,need,&ln,&rn);
        n.left=node(formula,{range.begin,middle},ln,depth+1,forced?forced-1:0);
        n.right=node(formula,{middle,range.end},rn,depth+1,forced?forced-1:0);
        if(error!=ExternalPlanStatus::ready)return SIZE_MAX;
        const auto&left=capacity[n.left],&right=capacity[n.right];
        error=query_stored_merge(formula.recipe,need,inputs(left.result,right.result),p.spec->series.workers,
            arithmetic_budget(),profile,n.merge,0,precision,StoredRounding::finite,true);
        if(error!=ExternalPlanStatus::ready)return SIZE_MAX;
        p.product_quotes+=n.merge.summary_quotes;
        n.result=outputs(n.merge);
        for(unsigned j=0;j<3;++j)if(need&(1u<<j)){
            n.result[j].limbs=std::min(n.result[j].limbs,std::min(logical.limbs[j],precision));
            n.result[j].source.limbs=n.result[j].limbs;
        }
        uint64_t ignored,lr,rl;
        if(!returned(n.result,need,n.return_bytes,ignored)||!plus(left.return_bytes,right.peak_bytes,lr)||
           !plus(right.return_bytes,left.peak_bytes,rl)||!plus(left.write_bytes,right.write_bytes,n.write_bytes)||
           !plus(n.write_bytes,n.merge.normal_write_bytes,n.write_bytes)||!add_work(n.merge.storage_bytes,n.merge.alignment)){
            error=ExternalPlanStatus::no_admitted_plan;return SIZE_MAX;
        }
        lr=std::max({left.peak_bytes,lr,n.merge.peak_backing_bytes});rl=std::max({right.peak_bytes,rl,n.merge.peak_backing_bytes});
        n.right_first=rl<lr;n.peak_bytes=std::min(lr,rl);return id;
    }
};
}
bool query_series_metadata(size_t blocks,size_t nodes,SeriesMetadata&out) noexcept {
    if(!blocks||!nodes)return false;
    SeriesMetadata p{};p.block_capacity=blocks;p.node_capacity=nodes;size_t at=0,n;
    auto put=[&](size_t count,size_t width,size_t alignment,size_t&offset){return mul_size(count,width,n)&&align_size(at,alignment,offset)&&add_size(offset,n,at);};
    if(!put(blocks,sizeof(SeriesBlock),alignof(SeriesBlock),p.blocks)||!put(nodes,sizeof(SeriesNode),alignof(SeriesNode),p.nodes)||
       !put(blocks,sizeof(planning::BlockSummary),alignof(planning::BlockSummary),p.summaries)||
       !put(blocks,sizeof(planning::Decomposition),alignof(planning::Decomposition),p.choices)||
       !put(1,sizeof(series::ProductPlanMemo),alignof(series::ProductPlanMemo),p.memo)||!align_size(at,4096,p.bytes))return false;
    out=p;return true;
}
ExternalPlanStatus compile_series(const series::PsrSpec&spec,SeriesBudget budget,const SeriesOptions&options,
    const planning::ExternalProductProfile&profile,const SeriesMetadata&layout,std::span<uint8_t>metadata,SeriesPlan&out) noexcept {
    out={};
    SeriesMetadata check{};
    if(!series::precision_spec_supported(spec)||options.force_stored_depth>depth_limit||
       !query_series_metadata(layout.block_capacity,layout.node_capacity,check)||layout.bytes!=check.bytes||
       !metadata.data()||(uintptr_t(metadata.data())&4095)||metadata.size()<check.bytes)return ExternalPlanStatus::invalid;
    SeriesPlan p{};p.spec=&spec;p.options=options;p.metadata_bytes=check.bytes;p.metadata_base=metadata.data();p.io_bytes=transfer_bytes+index_page_bytes;
    p.shared_bytes=engine_shared_bytes();
    if(!add_size(budget.other_live_bytes,p.shared_bytes,p.fixed_bytes)||!add_size(p.fixed_bytes,p.metadata_bytes,p.fixed_bytes)||
       !add_size(p.fixed_bytes,p.io_bytes+controller_allowance,p.fixed_bytes)||p.fixed_bytes>=budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    auto*blocks=reinterpret_cast<SeriesBlock *>(metadata.data()+check.blocks);
    auto*nodes=reinterpret_cast<SeriesNode *>(metadata.data()+check.nodes);
    auto*summaries=reinterpret_cast<planning::BlockSummary *>(metadata.data()+check.summaries);
    auto*choices=reinterpret_cast<planning::Decomposition *>(metadata.data()+check.choices);
    for(size_t i=0;i<check.block_capacity;++i){::new(blocks+i)SeriesBlock{};::new(summaries+i)planning::BlockSummary{};::new(choices+i)planning::Decomposition{};}
    for(size_t i=0;i<check.node_capacity;++i)::new(nodes+i)SeriesNode{};
    auto*memo=::new(metadata.data()+check.memo)series::ProductPlanMemo{};
    size_t count=0;
    for(uint64_t begin=spec.range.begin;begin<spec.range.end;){
        if(count==check.block_capacity)return ExternalPlanStatus::no_admitted_plan;
        auto&b=blocks[count];
        if(series::precision_block_query(spec,begin,b.block)!=SBN3_SUPPORTED)return ExternalPlanStatus::unsupported;
        ++p.native_quotes;
        const auto rc=planning::summarize_precision_episode(spec,begin,b.resident,memo);
        summaries[count]=rc==SummaryStatus::ok?b.resident.summary:planning::BlockSummary{};
        summaries[count].range=b.block.range;
        if(count==0&&options.force_stored_depth)for(auto&c:summaries[count].candidate)c.supported=false;
        begin=b.block.range.end;++count;
    }
    p.blocks={blocks,count};p.choices={choices,count};p.error_units=2*uint64_t(count)+1;
    if(planning::select_resident_suffix({summaries,count},{budget.resident_bytes,p.fixed_bytes},p.choices,p.resident)!=SummaryStatus::ok)return ExternalPlanStatus::no_admitted_plan;
    for(size_t i=p.resident.first;i<count;++i)planning::realize_precision_episode(blocks[i].resident,choices[i],p.resident.envelope);
    p.work_bytes=p.resident.layout.bytes;p.work_alignment=std::max(p.work_alignment,p.resident.layout.alignment);
    Compiler compiler{p,budget,profile,{nodes,check.node_capacity},memo};
    std::array<IntegerShape,3> tail{};uint64_t tail_disk=0,ignored=0;
    if(p.resident.first<count){
        const size_t words=blocks[p.resident.first].block.precision+series::precision_guard_words;
        tail[0]=tail[1]={words,0,{}};
        if(p.resident.first||options.store_result){
            if(!returned(tail,3,tail_disk,p.write_bytes))return ExternalPlanStatus::unsupported;
            p.peak_backing_bytes=tail_disk;
        }
    }
    for(size_t i=p.resident.first;i-- >0;){auto&b=blocks[i];
        const auto formula=series::precision_formula(spec,b.block.exact_words?&b.block.exact_words:nullptr);
        b.root=compiler.node(formula,b.block.range,b.block.need,0,i==0?options.force_stored_depth:0);
        if(compiler.error!=ExternalPlanStatus::ready)return compiler.error;
        const auto&n=nodes[b.root];uint64_t peak;
        if(!plus(n.peak_bytes,tail_disk,peak)||!plus(p.write_bytes,n.write_bytes,p.write_bytes))return ExternalPlanStatus::unsupported;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,peak);
        if(b.block.next_precision){
            const auto rc=query_stored_merge(spec.formula.recipe,3,inputs(n.result,tail),spec.series.workers,
                compiler.arithmetic_budget(),profile,b.merge,0,b.block.precision+series::precision_guard_words,StoredRounding::tail,true);
            if(rc!=ExternalPlanStatus::ready)return rc;
            p.product_quotes+=b.merge.summary_quotes;
            if(!compiler.add_work(b.merge.storage_bytes,b.merge.alignment)||!plus(p.write_bytes,b.merge.normal_write_bytes,p.write_bytes))return ExternalPlanStatus::no_admitted_plan;
            p.peak_backing_bytes=std::max(p.peak_backing_bytes,b.merge.peak_backing_bytes);tail=outputs(b.merge);
        }else tail=n.result;
        if(!returned(tail,3,tail_disk,ignored))return ExternalPlanStatus::unsupported;
    }
    p.result_shape=tail;
    p.nodes={nodes,compiler.count};p.minimum_free_regions=p.nodes.empty()?2:3*size_t(p.max_depth+1)+40;
    // Lower producer/consumer lifetimes only after the complete work grant is
    // known. Keep the existing split, visit order and every arithmetic plan.
    for(size_t i=p.nodes.size();i-- >0;)if(!nodes[i].native)
        compiler.lend(nodes[i].right_first?nodes[i].left:nodes[i].right,nodes[i].merge,nodes[i].right_first?0:3);
    for(size_t i=0;i<p.resident.first;++i)if(blocks[i].block.next_precision)compiler.lend(blocks[i].root,blocks[i].merge,0);
    for(auto&n:p.nodes)if(!n.native)share_stored_merge_transform(n.merge,p.work_bytes,compiler.arithmetic_budget(),profile);
    for(size_t i=0;i<p.resident.first;++i)if(blocks[i].block.next_precision)
        share_stored_merge_transform(blocks[i].merge,p.work_bytes,compiler.arithmetic_budget(),profile);
    // Reprice the actual stored frontier: loaned native returns never acquire
    // a Store Region. No hypothetical byte reduction is deducted at runtime.
    p.write_bytes=p.peak_backing_bytes=tail_disk=0;tail={};
    if(p.resident.first<count){
        tail[0]=tail[1]={blocks[p.resident.first].block.precision+series::precision_guard_words,0,{}};
        if(p.resident.first||options.store_result){
            if(!returned(tail,3,tail_disk,p.write_bytes))return ExternalPlanStatus::unsupported;
            p.peak_backing_bytes=tail_disk;
        }
    }
    for(size_t i=p.resident.first;i-- >0;){auto&b=blocks[i];
        if(!compiler.refresh(b.root))return ExternalPlanStatus::unsupported;
        const auto&n=nodes[b.root];uint64_t peak;
        if(!plus(n.peak_bytes,tail_disk,peak)||!plus(p.write_bytes,n.write_bytes,p.write_bytes))return ExternalPlanStatus::unsupported;
        p.peak_backing_bytes=std::max(p.peak_backing_bytes,peak);
        if(b.block.next_precision){
            if(!plus(p.write_bytes,b.merge.normal_write_bytes,p.write_bytes))return ExternalPlanStatus::unsupported;
            p.peak_backing_bytes=std::max(p.peak_backing_bytes,b.merge.peak_backing_bytes);tail=outputs(b.merge);
        }else tail=n.result;
        if(!returned(tail,3,tail_disk,ignored))return ExternalPlanStatus::unsupported;
    }
    p.additional_backing_bytes=p.peak_backing_bytes;
    if(!align_size(p.work_bytes,4096,p.work_bytes)||!add_size(p.fixed_bytes,p.work_bytes,p.total_resident_bytes)||
       !plus(p.peak_backing_bytes,budget.other_owned_disk_bytes,p.peak_backing_bytes))return ExternalPlanStatus::unsupported;
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.write_bytes>budget.write_limit)return ExternalPlanStatus::no_admitted_plan;
    p.memo_hits=memo->hits;p.memo_misses=memo->misses;out=p;return ExternalPlanStatus::ready;
}
namespace {
struct Execution {
    const SeriesPlan&p;Transport&t;ValueStore&store;sbn3_arena&arena;sbn3_team&team;const sbn3_lease&work;StoreBuffers io;
    std::span<uint8_t> bytes()const{return {static_cast<uint8_t *>(work.data),work.bytes};}
    IoError publish(const sbn3_series_values&v,unsigned need,StoredTriple&out){
        RegionLease regions[3];
        for(unsigned j=0;j<3;++j)if(need&(1u<<j))if(auto e=store.reserve(v.value[j].mantissa.size*8,RegionKind::value,regions[j]))return e;
        for(unsigned j=0;j<3;++j)if(need&(1u<<j)){
            const auto&x=v.value[j];ValueWriter writer;
            if(auto e=writer.open(t,regions[j].region(),io,&out[j].extent))return e;
            if(auto e=writer.append({reinterpret_cast<const uint8_t *>(x.mantissa.data),x.mantissa.size*8}))return e;
            if(auto e=writer.seal())return e;
            out[j].exponent2=x.exponent2;out[j].negative=x.mantissa.negative;out[j].extent_known=true;
        }
        for(unsigned j=0;j<3;++j)if(need&(1u<<j))if(auto e=regions[j].publish(out[j].value))return e;
        return {};
    }
    IoError node(size_t index,StoredTriple&out,std::span<ResidentInteger,3>loans){
        if(t.cancel.test())return {IoCode::cancelled,"series node boundary"};
        const auto&n=p.nodes[index];
        if(n.native){
            OperationSpan trace(t,"native_subtree",n.range.begin,n.range.end);
            sbn3_lease roles[planning::slot_count]{};
            struct End {sbn3_arena&a;sbn3_lease*roles;~End(){for(unsigned j=0;j<planning::slot_count;++j)if(roles[j].token)sbn3_arena_release(&a,roles+j);}} end{arena,roles};
            const size_t base=static_cast<uint8_t *>(work.data)-arena.base;
            for(unsigned j=0;j<planning::slot_count;++j)if(n.slots.slots[j].bytes)roles[j]=arena.acquire(base+n.layout.offsets[j],n.slots.slots[j].bytes);
            sbn3_series_values values{};
            for(unsigned j=0;j<3;++j)if(n.need&(1u<<j))values.value[j].mantissa={static_cast<uint64_t *>(roles[j].data),roles[j].bytes/8,0,0};
            series::FiniteBinding binding{};
            series::finite_prepare(n.finite,arena,team,roles[unsigned(planning::Slot::metadata)],roles[unsigned(planning::Slot::prepared)],roles[unsigned(planning::Slot::workspace)],binding);
            series::finite_execute(binding,values);
            if(auto e=publish(values,n.need&~n.loan_mask,out))return e;
            for(unsigned j=0;j<3;++j)if(n.loan_mask&(1u<<j)){
                const auto&v=values.value[j];require(v.mantissa.size<=n.result[j].limbs,SBN3_FATAL_MATH,"series resident return bound");
                auto*destination=reinterpret_cast<uint64_t *>(bytes().data()+n.loan_offsets[j]);
                if(v.mantissa.size)std::memmove(destination,v.mantissa.data,v.mantissa.size*8);
                auto&r=loans[j];r.words={destination,v.mantissa.size};r.exponent2=v.exponent2;r.negative=v.mantissa.negative;
                r.extent.observe(0,{reinterpret_cast<const uint8_t *>(destination),v.mantissa.size*8});
            }
        }else{
            StoredTriple left,right;std::array<ResidentInteger,6>children{};
            const std::span<ResidentInteger,3>l(children.data(),3),r(children.data()+3,3);
            if(n.right_first){if(auto e=node(n.right,right,r))return e;if(auto e=node(n.left,left,l))return e;}
            else{if(auto e=node(n.left,left,l))return e;if(auto e=node(n.right,right,r))return e;}
            if(auto e=run_stored_merge(n.merge,t,store,left,right,out,bytes(),team,n.merge.resident_inputs?std::span<const ResidentInteger>(children):std::span<const ResidentInteger>{}))return e;
        }
        for(unsigned j=0;j<3;++j)if(n.need&(1u<<j))require((n.loan_mask&(1u<<j)?loans[j].words.size():out[j].limbs())<=n.result[j].limbs,SBN3_FATAL_MATH,"series output shape");
        return {};
    }
};
}
IoError run_series(const SeriesPlan&p,Transport&t,ValueStore&store,sbn3_arena&arena,sbn3_team&team,
    const sbn3_lease&work,std::span<uint8_t>io,SeriesResult&out) noexcept {
    if(!p.spec||p.blocks.empty()||!work.data||work.bytes<p.work_bytes||(uintptr_t(work.data)&(p.work_alignment-1))||
       !io.data()||io.size()<p.io_bytes||(uintptr_t(io.data())&4095)||overlaps(io.data(),io.size(),work.data,work.bytes)||
       overlaps(p.metadata_base,p.metadata_bytes,work.data,work.bytes)||overlaps(p.metadata_base,p.metadata_bytes,io.data(),io.size())||
       team.arena!=&arena||team.width!=p.spec->series.workers||team.busy||out.in_store||out.resident.value[0].mantissa.data)
        return {IoCode::invalid,"series bindings",0,p.work_bytes,work.bytes};
    for(const auto&v:out.stored)if(v.value)return {IoCode::invalid,"series output already initialized"};
    if((p.resident.first||p.options.store_result)&&store.available_regions()<p.minimum_free_regions)
        return {IoCode::capacity,"series metadata regions",0,p.minimum_free_regions,store.available_regions()};
    if(store.usage().free_bytes<p.additional_backing_bytes)
        return {IoCode::capacity,"series backing admission",0,p.additional_backing_bytes,store.usage().free_bytes};
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"series write admission",0,p.write_bytes,t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    Execution e{p,t,store,arena,team,work,{io.first(transfer_bytes),io.subspan(transfer_bytes,index_page_bytes)}};
    planning::PrecisionState state{};StoredTriple tail;
    for(size_t i=p.blocks.size();i-- >p.resident.first;){
        if(t.cancel.test())return {IoCode::cancelled,"series resident episode boundary"};
        planning::PrecisionBinding binding{};
        OperationSpan trace(t,"resident_series_block",p.blocks[i].block.precision,i);
        planning::prepare_precision_episode(p.blocks[i].resident,p.choices[i],p.resident.envelope,arena,team,work,binding);
        planning::execute_precision_episode(binding,state);planning::release_precision_episode(binding);
    }
    if(state.ready){
        if(!p.resident.first&&!p.options.store_result){out.resident=state.values;return {};}
        if(auto error=e.publish(state.values,3,tail))return error;
    }
    for(size_t i=p.resident.first;i-- >0;){StoredTriple finite;std::array<ResidentInteger,6>loans{};
        if(auto error=e.node(p.blocks[i].root,finite,std::span<ResidentInteger,3>(loans.data(),3)))return error;
        if(p.blocks[i].block.next_precision){StoredTriple result;
            const auto&merge=p.blocks[i].merge;
            if(auto error=run_stored_merge(merge,t,store,finite,tail,result,e.bytes(),team,
                merge.resident_inputs?std::span<const ResidentInteger>(loans):std::span<const ResidentInteger>{}))return error;
            tail=std::move(result);
        }else tail=std::move(finite);
    }
    require(tail[1].value&&tail[1].limbs()&&!tail[1].negative,SBN3_FATAL_MATH,"series positive denominator");
    out.stored=std::move(tail);out.in_store=true;return {};
}
} // namespace sbn::v3::external
