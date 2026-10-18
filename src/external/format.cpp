// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/format.hpp"
#include "external/owned_product.hpp"
#include "external/shared.hpp"
#include "radix/power_bank.hpp"
#include "radix/format_frontier.hpp"
#include "common/checked.hpp"
#include "runtime/arena.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <utility>
namespace sbn::v3::external {
namespace {
constexpr size_t transfer=1u<<20;
bool add(uint64_t a,uint64_t b,uint64_t&out){return !__builtin_add_overflow(a,b,&out);}
bool region(size_t words,RegionLayout&out){size_t bytes;return mul_size(words,8,bytes)&&!region_layout(bytes,0,out);}
struct Compiler {
    FormatPlan&p;const FormatBudget&budget;const planning::ExternalProductProfile&profile;
    size_t available=0,source_capacity=0;
    planning::ExternalPlanStatus status=planning::ExternalPlanStatus::ready;
    unsigned classify(uint64_t fragments){
        for(unsigned i=0;i<p.classes;++i)if(p.node[i].fragments==fragments)return i;
        if(p.classes==format_classes){status=planning::ExternalPlanStatus::unsupported;return 0;}
        const unsigned id=p.classes++;auto&node=p.node[id];node.fragments=fragments;node.limbs=radix::node_limbs(p.base,fragments);
        const uint64_t digits=64*fragments+8;
        if(fragments<=p.options.maximum_resident_fragments&&node.limbs<=available/8&&digits<=available-node.limbs*8){
            const sbn3_format_spec spec{p.base.base,node.limbs,-int64_t(node.limbs*64),digits,SBN3_RADIX_EXACT};
            auto options=p.options.radix;options.use_alphabet=0;options.memory_budget=0;
            node.native_frontier=radix::format_frontier_query(p.base.base,node.limbs,fragments,options,node.leaf,node.leaf_info)==SBN3_SUPPORTED;
            if(node.native_frontier||sbn3_format_query(&spec,&options,&node.leaf,&node.leaf_info)==SBN3_SUPPORTED){
                size_t at=0;node.leaf_alignment=std::max(size_t(4096),node.leaf_info.storage_alignment);
                auto put=[&](size_t bytes,size_t alignment,size_t&offset){return align_size(at,alignment,offset)&&add_size(offset,bytes,at);};
                if(!put((node.limbs+2)*8,64,node.input)||!put(std::max(node.leaf_info.digit_bytes,size_t(digits)),64,node.digits)||
                   !put(node.leaf_info.storage_bytes,node.leaf_info.storage_alignment,node.binding)||
                   !align_size(at,p.alignment,node.leaf_bytes)){status=planning::ExternalPlanStatus::unsupported;return id;}
                if(node.leaf_alignment<=p.alignment&&node.leaf_bytes<=available){node.resident=true;node.frontiers=1;p.phase_bytes=std::max(p.phase_bytes,node.leaf_bytes);return id;}
            }
        }
        if(fragments<=16){status=planning::ExternalPlanStatus::no_admitted_plan;return id;}
        uint64_t left=p.seed;while(2*left<fragments)left*=2;
        node.left_fragments=left;node.power=unsigned(__builtin_ctzll(left/p.seed));p.powers=std::max(p.powers,node.power+1);
        node.left=classify(left);if(status!=planning::ExternalPlanStatus::ready)return id;
        node.right=classify(fragments-left);if(status!=planning::ExternalPlanStatus::ready)return id;
        const auto&l=p.node[node.left];const auto&r=p.node[node.right];node.right_limbs=r.limbs;
        const size_t shift=p.base.twos*left;
        if(r.limbs+shift>node.limbs){status=planning::ExternalPlanStatus::unsupported;return id;}
        node.split_limbs=node.limbs-shift;node.window=node.split_limbs-r.limbs;
        uint64_t writes=0,normal_writes=0,backing=0;
        if(p.base.odd!=1){
            const size_t power=radix::limbs_for_bits(radix::power_bits(p.base.odd_bound,64*left));
            planning::ExternalProductBudget grant{p.fixed_bytes+available,p.fixed_bytes-p48_root_bank::bytes()};
            planning::ExternalProductSelection choice;
            status=planning::choose_external_product({node.split_limbs,power,p.options.radix.workers,0,0,false,
                {source_capacity,0,node.split_limbs},{power,0,power}},grant,profile,choice);
            if(status!=planning::ExternalPlanStatus::ready)return id;node.split=choice.plan;
            const size_t first=node.window&~(chunk_bytes/8-1);
            if(!project_spilled_output(node.split,first,node.window+r.limbs)){status=planning::ExternalPlanStatus::unsupported;return id;}
            p.phase_bytes=std::max(p.phase_bytes,node.split.storage_bytes);writes=node.split.normal_write_bytes;
            // One mathematically derived ring, ranked on complete traffic and
            // transform work. The original LIN is the deterministic carry
            // witness, with its workspace/backing/writes still admitted.
            planning::ExternalProductRequest request{node.split_limbs,power,p.options.radix.workers,0,0,false,
                {source_capacity,0,node.split_limbs},{power,0,power}};
            request.middle={node.window,node.window+r.limbs};
            planning::ExternalProductSelection ring;
            if(planning::choose_external_product(request,grant,profile,ring)==planning::ExternalPlanStatus::ready&&
               ring.plan.output_bytes<node.split.output_bytes&&
               project_spilled_output(ring.plan,first,request.middle.end)&&
               ring.plan.normal_write_bytes<=node.split.normal_write_bytes&&
               planning::estimate_external_product(ring.plan,profile)<planning::estimate_external_product(node.split,profile)){
                node.carry_witness=node.split;node.split=ring.plan;
                if(!add(writes,node.split.normal_write_bytes,writes)){status=planning::ExternalPlanStatus::unsupported;return id;}
                p.phase_bytes=std::max(p.phase_bytes,node.split.storage_bytes);
            }
            normal_writes=node.split.normal_write_bytes;
            RegionLayout output{};uint64_t planes;
            if(region_layout(node.split.stored_output_bytes,0,output)||__builtin_mul_overflow(node.split.plane_allocation_bytes,uint64_t(node.split.plane_count),&planes)||
               !add(output.allocation_bytes,std::max(planes,r.backing),backing)){status=planning::ExternalPlanStatus::unsupported;return id;}
            if(node.split.spec.middle){uint64_t witness_planes;
                if(__builtin_mul_overflow(node.carry_witness.plane_allocation_bytes,uint64_t(node.carry_witness.plane_count),&witness_planes)||
                   !add(output.allocation_bytes,std::max({planes,witness_planes,r.backing}),backing)){
                    status=planning::ExternalPlanStatus::unsupported;return id;}
            }
        }else backing=r.backing;
        node.backing=std::max(backing,l.backing);
        if(!add(writes,l.descendants_writes,writes)||!add(writes,r.descendants_writes,node.descendants_writes)||
           !add(normal_writes,l.descendants_normal_writes,normal_writes)||
           !add(normal_writes,r.descendants_normal_writes,node.descendants_normal_writes)||
           !add(l.frontiers,r.frontiers,node.frontiers))status=planning::ExternalPlanStatus::unsupported;
        return id;
    }
};
}
planning::ExternalPlanStatus query_stored_format(const StoredInteger&input,unsigned base,uint64_t digits,
    const FormatOptions&options,const FormatBudget&budget,const planning::ExternalProductProfile&profile,FormatPlan&out) noexcept {
    if(!input.value||input.negative||input.exponent2>=0||input.exponent2%64||!input.extent_known||input.value.region().layout.logical_bytes%8)
        return planning::ExternalPlanStatus::invalid;
    const uint64_t fraction=uint64_t(-(__int128(input.exponent2)/64));
    if(fraction>input.limbs()||input.extent.end>input.first_limb+fraction+1)return planning::ExternalPlanStatus::unsupported;
    return query_format_source(input.source(),size_t(fraction),base,digits,options,budget,profile,out);
}
planning::ExternalPlanStatus query_format_source(SourceWindow input,size_t fraction,unsigned base,uint64_t digits,
    const FormatOptions&options,const FormatBudget&budget,const planning::ExternalProductProfile&profile,FormatPlan&out) noexcept {
    using planning::ExternalPlanStatus;
    out={};auto&p=out;p.options=options;
    if(!radix::base_info(base,p.base)||!digits||digits>(uint64_t(1)<<40)||!options.radix.workers||options.radix.workers>32||
       options.maximum_resident_fragments<16||input.limbs==SIZE_MAX||!resolve_source_window(input.limbs,input)||
       fraction>input.limbs||input.low_zero_words||input.bit_shift)return ExternalPlanStatus::invalid;
    if(options.radix.use_alphabet){bool used[256]{};for(unsigned i=0;i<base;++i){const unsigned c=options.radix.alphabet[i];if(used[c])return ExternalPlanStatus::invalid;used[c]=true;}}
    p.fractional_words=size_t(fraction);p.digits=digits;p.fragments=(digits+63)/64;
    const size_t limbs=radix::node_limbs(p.base,p.fragments);
    if(limbs>fraction)return ExternalPlanStatus::unsupported;
    p.source=input;p.source.first_limb+=fraction-limbs;p.source.limbs=limbs;p.source.lower_limb=p.source.first_limb;p.source.upper_limb=p.source.first_limb+limbs;
    p.seed=1;if(!options.radix.repeated){p.seed=p.fragments;while(p.seed>16)p.seed=(p.seed+1)/2;}
    RegionLayout output{};if(region_layout(digits,0,output))return ExternalPlanStatus::unsupported;p.output_allocation=output.allocation_bytes;
    ChunkWriterStorage index;if(chunk_writer_storage(output,index))return ExternalPlanStatus::unsupported;
    p.output_index_bytes=index.index_bytes;p.output_bits_bytes=index.coverage_bytes;
    size_t at=0;auto put=[&](size_t bytes,size_t alignment,size_t&offset){return align_size(at,alignment,offset)&&add_size(offset,bytes,at);};
    if(!put(transfer,4096,p.io)||!put(index_page_bytes,4096,p.index)||!put(index.index_bytes,4096,p.output_index)||
       !put(index.coverage_bytes,64,p.output_bits)||!put(chunk_bytes,4096,p.seam)||!align_size(at,p.alignment,p.phase))return ExternalPlanStatus::unsupported;
    p.shared_bytes=engine_shared_bytes()+radix::fixed_power_bytes()+radix::digit_constant_bytes();
    size_t fixed;
    if(!add_size(budget.other_live_bytes,sizeof(FormatPlan),fixed)||!add_size(fixed,65536,fixed)||!add_size(fixed,p.shared_bytes,fixed)||
       !add_size(fixed,p.phase,p.fixed_bytes)||p.fixed_bytes>=budget.resident_bytes)return ExternalPlanStatus::no_admitted_plan;
    const size_t available=(budget.resident_bytes-p.fixed_bytes)&~(p.alignment-1);
    Compiler compiler{p,budget,profile,available,std::max(p.source.storage_limbs,limbs+chunk_bytes/8)};
    compiler.classify(p.fragments);if(compiler.status!=ExternalPlanStatus::ready)return compiler.status;
    if(p.base.odd==1)p.powers=0;
    if(p.powers>format_powers)return ExternalPlanStatus::unsupported;
    uint64_t rails=0,power_temporary=0;p.write_bytes=output.data_bytes+output.index_bytes;
    for(unsigned i=0;i<p.powers;++i){auto&power=p.power[i];
        power.limbs=radix::limbs_for_bits(radix::power_bits(p.base.odd_bound,(64*p.seed)<<i));RegionLayout layout;
        if(!region(power.limbs,layout)||!add(rails,layout.allocation_bytes,rails))return ExternalPlanStatus::unsupported;
        power.allocation=layout.allocation_bytes;power.writes=layout.data_bytes+layout.index_bytes;
        if(i){const size_t previous=p.power[i-1].limbs;sbn3_mul_options native{};native.workers=options.radix.workers;native.borrow_output=1;
            if(query_resident_square(previous,native,transfer,power.native)==SBN3_SUPPORTED&&power.native.alignment<=p.alignment&&power.native.storage_bytes<=available){
                power.resident=true;if(!project_resident_output(power.native,power.limbs))return ExternalPlanStatus::unsupported;
                p.phase_bytes=std::max(p.phase_bytes,power.native.storage_bytes);
            }else{
                planning::ExternalProductBudget grant{p.fixed_bytes+available,p.fixed_bytes-p48_root_bank::bytes()};planning::ExternalProductSelection choice;
                auto status=planning::choose_external_product({previous,previous,options.radix.workers,0,0,true},grant,profile,choice);
                if(status!=ExternalPlanStatus::ready)return status;power.external=choice.plan;
                if(!project_spilled_output(power.external,0,power.limbs))return ExternalPlanStatus::unsupported;
                power.external.emit.bound={power.limbs,1,true};power.writes=power.external.normal_write_bytes;
                p.phase_bytes=std::max(p.phase_bytes,power.external.storage_bytes);uint64_t temp;
                if(__builtin_mul_overflow(power.external.plane_allocation_bytes,uint64_t(power.external.plane_count),&temp))return ExternalPlanStatus::unsupported;
                power_temporary=std::max(power_temporary,temp);
            }
        }else p.phase_bytes=std::max(p.phase_bytes,4096+8*power.limbs);
        if(!add(p.write_bytes,power.writes,p.write_bytes))return ExternalPlanStatus::unsupported;
    }
    if(!add(p.write_bytes,p.node[0].descendants_normal_writes,p.normal_write_bytes)||
       !add(p.write_bytes,p.node[0].descendants_writes,p.write_bytes))return ExternalPlanStatus::unsupported;
    RegionLayout source;if(!region(p.source.storage_limbs,source)||!add(budget.other_owned_disk_bytes,source.allocation_bytes,p.peak_backing_bytes)||
       !add(p.peak_backing_bytes,p.output_allocation,p.peak_backing_bytes)||!add(p.peak_backing_bytes,rails,p.peak_backing_bytes)||
       !add(p.peak_backing_bytes,std::max(power_temporary,p.node[0].backing),p.peak_backing_bytes)||
       !align_size(p.phase_bytes,p.alignment,p.phase_bytes)||!add_size(p.phase,p.phase_bytes,p.storage_bytes)||
       !add_size(p.fixed_bytes,p.phase_bytes,p.total_resident_bytes))return ExternalPlanStatus::unsupported;
    if(p.total_resident_bytes>budget.resident_bytes||p.peak_backing_bytes>budget.owned_disk_limit||p.write_bytes>budget.write_limit)
        return ExternalPlanStatus::no_admitted_plan;
    return ExternalPlanStatus::ready;
}
namespace {
struct Run {
    const FormatPlan&p;Transport&t;ValueStore&store;sbn3_arena&arena;sbn3_team&team;size_t offset;
    std::span<uint8_t>w;StoreBuffers io;ChunkWriter output{};
    StoredInteger power[format_powers]{};
    uint64_t cursor=0,following=0,guard=0,common_prefix=UINT64_MAX;
    uint64_t middle_products=0,middle_fallbacks=0;
    unsigned guard_error=2;
    bool have_following=false;
    std::span<uint8_t>phase()const{return w.subspan(p.phase,p.phase_bytes);}
    IoError prepend(uint64_t at,std::span<uint8_t>digits){
        const uint64_t raw_end=at+digits.size();
        for(uint64_t j=std::max(at,p.digits);j<std::min(raw_end,p.digits+8);++j){uint64_t factor=1;
            for(uint64_t k=j+1;k<p.digits+8;++k)factor*=p.base.base;guard+=uint64_t(digits[size_t(j-at)])*factor;}
        if(at>=p.digits)return {};
        digits=digits.first(size_t(std::min(raw_end,p.digits)-at));
        require(at+digits.size()==cursor,SBN3_FATAL_MATH,"radix reverse output order");
        if(common_prefix==UINT64_MAX)for(size_t j=digits.size();j-- >0;)if(digits[j]!=p.base.base-1){common_prefix=at+j;break;}
        if(p.options.radix.use_alphabet)for(auto&c:digits)c=p.options.radix.alphabet[c];
        auto seam=w.subspan(p.seam,chunk_bytes);
        while(!digits.empty()){
            // This entire leaf is finalized after the right-hand carry seam.
            // Interior chunks can be handed to the ordinary bounded writer in
            // one range. Only a chunk crossing leaves needs the seam buffer.
            // ChunkWriter handles alignment, transfer limits and checksums;
            // submission is joined before the leaf's scratch is recycled.
            if(cursor%chunk_bytes==0&&digits.size()>=chunk_bytes){
                const size_t n=digits.size()/chunk_bytes*chunk_bytes;
                if(auto e=output.write((cursor-n)/chunk_bytes,digits.last(n)))return e;
                cursor-=n;digits=digits.first(digits.size()-n);continue;
            }
            const uint64_t last=cursor-1,chunk=last/chunk_bytes,begin=chunk*chunk_bytes;
            const size_t n=size_t(std::min(uint64_t(digits.size()),cursor-begin));const size_t start=size_t(cursor-begin)-n;
            std::memcpy(seam.data()+start,digits.data()+digits.size()-n,n);cursor-=n;digits=digits.first(digits.size()-n);
            if(cursor==begin){const size_t count=size_t(std::min(uint64_t(chunk_bytes),p.digits-begin));
                if(auto e=output.write(chunk,seam.first(count)))return e;}
        }
        return {};
    }
    IoError build_powers(){
        for(unsigned i=0;i<p.powers;++i){const auto&q=p.power[i];RegionLease lease;
            if(auto e=store.reserve(q.limbs*8,RegionKind::value,lease))return e;
            if(!i){auto*words=reinterpret_cast<uint64_t *>(phase().data());std::fill_n(words,q.limbs,uint64_t(0));words[0]=1;
                for(uint64_t k=0;k<64*p.seed;++k){uint64_t carry=0;for(size_t j=0;j<q.limbs;++j){const __uint128_t v=__uint128_t(words[j])*p.base.odd+carry;words[j]=uint64_t(v);carry=uint64_t(v>>64);}
                    require(!carry,SBN3_FATAL_MATH,"radix seed power capacity");}
                ValueWriter writer;if(auto e=writer.open(t,lease.region(),io,&power[i].extent))return e;
                if(auto e=writer.append(phase().first(q.limbs*8)))return e;if(auto e=writer.seal())return e;
            }else if(q.resident){if(auto e=run_resident_product(q.native,t,power[i-1].value.region(),power[i-1].value.region(),lease.region(),phase(),team,&power[i].extent))return e;}
            else {lease.reset();if(auto e=run_owned_product(q.external,t,store,power[i-1].value,power[i-1].value,power[i].value,phase(),team,&power[i].extent))return e;
                power[i].extent_known=true;continue;}
            if(auto e=lease.publish(power[i].value))return e;power[i].extent_known=true;
        }
        return {};
    }
    IoError visit(unsigned id,const StoredInteger&value,uint64_t first){
        const auto&node=p.node[id];
        if(node.resident){
            sbn3_lease inputs=arena.acquire(offset+p.phase+node.input,(node.limbs+2)*8);
            sbn3_lease digits=arena.acquire(offset+p.phase+node.digits,std::max(node.leaf_info.digit_bytes,size_t(64*node.fragments+8)));
            struct Release {sbn3_arena&a;sbn3_lease&x,&y;~Release(){a.release(y);a.release(x);}} release{arena,inputs,digits};
            ValueReader reader;if(auto e=reader.open(t,value.value.region(),io))return e;
            if(auto e=read_source_words(reader,value.source(),0,node.limbs,{static_cast<uint8_t *>(inputs.data),inputs.bytes}))return e;
            if(t.cancel.test())return {IoCode::cancelled,"radix native frontier boundary"};
            sbn3_format_binding*binding=nullptr;sbn3_format_bind(&node.leaf,&arena,offset+p.phase+node.binding,&team,&binding);
            const size_t count=size_t(64*node.fragments);auto text=std::span(static_cast<uint8_t *>(digits.data),count);
            uint64_t overlap=0;
            if(node.native_frontier){
                overlap=radix::format_frontier_execute(binding,{static_cast<uint64_t*>(inputs.data),node.limbs},text.data());
                auto word=overlap;for(unsigned i=8;i-- >0;){static_cast<uint8_t*>(digits.data)[count+i]=uint8_t(word%p.base.base);word/=p.base.base;}
            }else{sbn3_format_result result{};sbn3_format_execute(binding,{static_cast<uint64_t*>(inputs.data),node.limbs},text.data(),&result);
                for(unsigned i=0;i<8;++i)overlap=overlap*p.base.base+static_cast<uint8_t*>(digits.data)[count+i];}
            sbn3_format_unbind(binding);
            if(have_following&&overlap!=following){uint64_t b8=1;for(unsigned i=0;i<8;++i)b8*=p.base.base;
                // Frontier digits are exact; only its overlap can be one low.
                // The original binary-cut seam contributes at most one more.
                // Their sum is 0..2, and crossing b^8 increments the prefix once.
                const uint64_t delta=following>=overlap?following-overlap:b8-overlap+following;
                require(delta<=1+unsigned(node.native_frontier),SBN3_FATAL_MATH,"external radix seam",overlap,following);
                if(following<overlap)for(size_t j=count;j-- >0;){if(++text[j]<p.base.base)break;text[j]=0;}
            }
            if(!have_following){auto extra=std::span(static_cast<uint8_t *>(digits.data)+count,size_t(8));
                if(node.native_frontier)guard_error=3; // one extra overlap ulp at the global right edge
                if(auto e=prepend(64*first+count,extra))return e;}
            following=0;for(unsigned i=0;i<8;++i)following=following*p.base.base+text[i];have_following=true;
            return prepend(first*64,text);
        }
        StoredInteger right;
        if(p.base.odd==1){value.value.share(right.value);right.first_limb=value.first_limb+node.window;right.word_count=node.right_limbs;}
        else {auto product=node.split;auto source=value.source();source.limbs=node.split_limbs;source.upper_limb=source.first_limb+source.limbs;
            require(rebind_spilled_sources(product,source,power[node.power].source()),SBN3_FATAL_WORKSPACE,"radix split source replay");
            MiddleWindowResult middle{};
            if(auto e=run_owned_product(product,t,store,value.value,power[node.power].value,right.value,phase(),team,&right.extent,
                                       nullptr,nullptr,product.spec.middle?&middle:nullptr))return e;
            if(product.spec.middle){++middle_products;
                if(!middle.exact){
                    // A carry can only cross an all-zero finalized gap. Do
                    // not guess its direction or modify sealed chunks. The
                    // preplanned exact witness replaces this private child.
                    ++middle_fallbacks;right.value.reset();product=node.carry_witness;
                    require(rebind_spilled_sources(product,source,power[node.power].source()),SBN3_FATAL_WORKSPACE,"radix carry witness source");
                    if(auto e=run_owned_product(product,t,store,value.value,power[node.power].value,right.value,phase(),team,&right.extent))return e;
                }
            }
            right.first_limb=node.window-size_t(product.output_first_byte/8);right.word_count=node.right_limbs;right.extent_known=true;
        }
        if(auto e=visit(node.right,right,first+node.left_fragments))return e;right.value.reset();
        StoredInteger left;value.value.share(left.value);left.first_limb=value.first_limb+node.limbs-p.node[node.left].limbs;left.word_count=p.node[node.left].limbs;
        return visit(node.left,left,first);
    }
};
}
IoError run_stored_format(const FormatPlan&p,Transport&t,ValueStore&store,const StoredInteger&input,FormatResult&out,
    sbn3_arena&arena,size_t offset,sbn3_team&team) noexcept {
    if(!p.classes||!input.value||input.value.store()!=&store||input.negative||!input.extent_known||input.exponent2!=-int64_t(64*p.fractional_words)||
       input.source().storage_limbs!=p.source.storage_limbs||p.fractional_words>input.limbs()||input.extent.end>input.first_limb+p.fractional_words+1||out.digits||
       team.arena!=&arena||team.busy||team.width!=p.options.radix.workers||p.storage_bytes>SIZE_MAX-offset||!arena.contains(offset,offset+p.storage_bytes)||
       !arena.unleased(offset,p.storage_bytes)||(uintptr_t(arena.base+offset)&(p.alignment-1)))return {IoCode::invalid,"stored radix binding"};
    if(t.counters.write_submitted>t.write_limit||p.write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"stored radix write admission",0,p.write_bytes,t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    const auto w=std::span(arena.base+offset,p.storage_bytes);auto fixed=arena.acquire(offset,p.phase);
    struct Release {sbn3_arena&a;sbn3_lease&l;~Release(){a.release(l);}} release{arena,fixed};
    Run run{p,t,store,arena,team,offset,w,{w.subspan(p.io,transfer),w.subspan(p.index,index_page_bytes)}};run.cursor=p.digits;
    RegionLease result;if(auto e=store.reserve(p.digits,RegionKind::value,result))return e;
    if(auto e=run.output.open(t,result.region(),run.io,w.subspan(p.output_index,p.output_index_bytes),w.subspan(p.output_bits,p.output_bits_bytes)))return e;
    uint64_t integer=0;
    if(input.limbs()>p.fractional_words){ValueReader reader;if(auto e=reader.open(t,input.value.region(),run.io))return e;
        if(auto e=reader.read((input.first_limb+p.fractional_words)*8,{reinterpret_cast<uint8_t *>(&integer),8}))return e;}
    if(auto e=run.build_powers())return e;
    StoredInteger root;input.value.share(root.value);root.first_limb=input.first_limb+p.fractional_words-p.node[0].limbs;root.word_count=p.node[0].limbs;
    if(auto e=run.visit(0,root,0))return e;
    out.middle_products=run.middle_products;out.middle_fallbacks=run.middle_fallbacks;
    require(!run.cursor,SBN3_FATAL_MATH,"radix incomplete reverse output");
    if(auto e=run.output.seal())return e;
    uint64_t b8=1;for(unsigned i=0;i<8;++i)b8*=p.base.base;
    if(auto e=result.publish(out.digits))return e;
    out.certified_digits=run.guard<=b8-1-run.guard_error?p.digits:(run.common_prefix==UINT64_MAX?0:run.common_prefix);out.integer_part=integer;
    return {};
}
} // namespace sbn::v3::external
