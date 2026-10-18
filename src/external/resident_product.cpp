// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/resident_product.hpp"
#include "external/trace.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include <cstring>
#include <algorithm>
#include <bit>

namespace sbn::v3::external {
namespace {
static_assert(std::endian::native == std::endian::little);
bool disjoint(const RegionLayout &a, const RegionLayout &b) {
    return a.data_offset <= b.data_offset ? a.allocation_bytes <= b.data_offset-a.data_offset
                                         : b.allocation_bytes <= a.data_offset-b.data_offset;
}
struct Call {
    const ProductProgram *program;
    Frame *work;
    sbn3_const_limbs a, b;
    sbn3_limbs out;
    bool bounded=false;
};
void multiply(void *p, sbn3_team_scope *scope) {
    auto &c = *static_cast<Call *>(p);
    if(c.bounded)product_program_bounded(*c.program,*c.work,scope,c.a,c.b,c.out);
    else product_program_execute(*c.program, *c.work, scope, c.a, c.b, c.out);
}
}
namespace {
bool layout(ResidentProductPlan&p){
    auto place=[&](ResidentProductPlan&v,bool last){
        v.alignment=std::max(size_t(4096),v.product.info.workspace_alignment);size_t cursor=0;
        auto reserve=[&](size_t size,size_t alignment,size_t&offset){return align_size(cursor,alignment,offset)&&add_size(offset,size,cursor);};
        auto words=[&](size_t count,size_t&offset){size_t bytes;return mul_size(count,8,bytes)&&reserve(bytes,64,offset);};
        if(v.product.an>SIZE_MAX-8||v.product.bn>SIZE_MAX-8||
           (!(v.borrowed_inputs&1)&&!words(v.product.an+8,v.a))||(!v.square&&!(v.borrowed_inputs&2)&&!words(v.product.bn+8,v.b)))return false;
        auto result=[&](){cursor=std::max(cursor,v.result_prefix);return words(sbn3_mul_output_capacity(&v.product.info),v.result);};
        if(!last&&!result())return false;
        if(!reserve(v.product.prepared_bytes,128,v.tables)||!reserve(v.product.info.workspace_bytes,v.product.info.workspace_alignment,v.work))return false;
        if(last&&!result())return false;
        return reserve(v.transfer_bytes,4096,v.transfer)&&reserve(index_page_bytes,4096,v.index)&&align_size(cursor,v.alignment,v.storage_bytes);
    };
    auto early=p,late=p;const bool early_ok=place(early,false);
    // A borrowed output outlives this operation's input/tables/work. Putting it
    // after that dead work often satisfies the consumer's prefix without a
    // large hole before both output AND transform scratch. This is a two-phase
    // lifetime layout, not another arithmetic/backend selection.
    const bool late_ok=p.result_prefix&&place(late,true);
    if(!early_ok&&!late_ok)return false;
    p=late_ok&&(!early_ok||late.storage_bytes<early.storage_bytes)?late:early;return true;
}
}
sbn3_query_result query_resident_product(size_t an,size_t bn,const sbn3_mul_options&options,size_t io_bytes,ResidentProductPlan&out) noexcept {
    if(!an||!bn||io_bytes<chunk_bytes||io_bytes%chunk_bytes)return SBN3_UNSUPPORTED;
    ResidentProductPlan p{};if(!add_size(an,bn,p.output_limbs))return SBN3_UNSUPPORTED;
    const auto rc=product_program_query(an,bn,options,p.product);if(rc!=SBN3_SUPPORTED)return rc;
    p.transfer_bytes=io_bytes;p.maximum_output_limbs=p.output_limbs;
    if(!set_resident_sources(p,{an,0,an},{bn,0,bn})||!layout(p))return SBN3_UNSUPPORTED;out=p;return SBN3_SUPPORTED;
}
sbn3_query_result query_resident_square(size_t n,const sbn3_mul_options&options,size_t io_bytes,ResidentProductPlan&out) noexcept {
    if(!n||n>SIZE_MAX/2||io_bytes<chunk_bytes||io_bytes%chunk_bytes)return SBN3_UNSUPPORTED;
    ResidentProductPlan p{};sbn3_product_request request{};request.kind=SBN3_PRODUCT_SQR;request.a_limbs=n;
    sbn3_product_info info{};const auto rc=sbn3_product_query(&request,&options,&p.product.plan,&info);
    if(rc!=SBN3_SUPPORTED)return rc;p.product.info=info.mul;
    if(product_program_describe(n,n,p.product)!=SBN3_SUPPORTED)return SBN3_UNSUPPORTED;
    p.square=true;p.transfer_bytes=io_bytes;p.output_limbs=p.maximum_output_limbs=2*n;
    if(!set_resident_sources(p,{n,0,n},{n,0,n})||!layout(p))return SBN3_UNSUPPORTED;out=p;return SBN3_SUPPORTED;
}
bool set_resident_sources(ResidentProductPlan&p,SourceWindow a,SourceWindow b) noexcept {
    if(!resolve_source_window(p.product.an,a)||!resolve_source_window(p.product.bn,b))return false;
    p.source[0]=a;p.source[1]=b;p.source_capacity[0]=a.storage_limbs;p.source_capacity[1]=b.storage_limbs;return true;
}
bool rebind_resident_sources(ResidentProductPlan&p,SourceWindow a,SourceWindow b) noexcept {
    if(!resolve_source_window(p.product.an,a)||!resolve_source_window(p.product.bn,b)||
       a.storage_limbs>p.source_capacity[0]||b.storage_limbs>p.source_capacity[1])return false;
    p.source[0]=a;p.source[1]=b;return true;
}
bool keep_high_resident_output(ResidentProductPlan&p,size_t keep) noexcept {
    if(!keep)return false;size_t end=0;
    if(p.source[0].limbs&&p.source[1].limbs&&!add_size(p.source[0].limbs,p.source[1].limbs,end))return false;
    const size_t low=end>keep&&end-keep>1?end-keep-1:0;
    const size_t first=low&~(chunk_bytes/8-1);if(end-first>p.maximum_output_limbs)return false;
    p.output_first=first;p.output_limbs=end-first;p.require_zero_high=false;return true;
}
bool reserve_resident_result_after(ResidentProductPlan&p,size_t prefix) noexcept {p.result_prefix=prefix;return layout(p);}
bool compact_resident_product(ResidentProductPlan&p) noexcept {
    if(p.square)return false;
    ProductProgramPlan compact{};if(product_program_rebatch(p.product,1,compact)!=SBN3_SUPPORTED)return false;
    p.product=compact;return layout(p);
}
bool borrow_resident_inputs(ResidentProductPlan&p,unsigned mask) noexcept {
    if(mask&~3u||(mask&&!(p.product.contract&program_bounded_inputs)))return false;
    for(unsigned i=0;i<2;++i)if(mask&(1u<<i)){
        const auto&s=p.source[i];if(s.bit_shift||s.low_zero_words||s.ones_complement)return false;
    }
    auto next=p;next.borrowed_inputs=mask;if(!layout(next))return false;p=next;return true;
}
static IoError execute(const ResidentProductPlan&p,Transport&transport,ResidentProductInput a,ResidentProductInput b,
                       Region*out,std::span<uint8_t>storage,sbn3_team&team,WordExtent*extent,ResidentProductValue*borrowed) noexcept {
    OperationSpan trace(transport,p.square?"native_square":"native_product",p.product.an,p.product.bn,p.product.info.np,p.product.info.trunk_bits,p.product.info.C,p.product.info.M2);
    const ResidentProductInput inputs[]{a,b};
    auto valid=[&](unsigned i){const auto&in=inputs[i];const auto&s=p.source[i];
        if(p.borrowed_inputs&(1u<<i))return !in.stored&&in.words.data()&&in.words.size()<=SIZE_MAX/8&&
            in.words.size()==s.storage_limbs&&!s.bit_shift&&!s.low_zero_words&&!s.ones_complement&&
            !overlaps(in.words.data(),in.words.size()*8,storage.data(),p.storage_bytes);
        return in.stored&&in.words.empty()&&in.stored->state==RegionState::sealed&&
            in.stored->layout.logical_bytes%8==0&&in.stored->layout.logical_bytes/8==s.storage_limbs&&
            (!out||(disjoint(out->layout,in.stored->layout)&&!(out->id==in.stored->id)));
    };
    if(!valid(0)||!valid(1)||(!out&&!borrowed)||
       (p.square&&(p.source[0]!=p.source[1]||a.stored!=b.stored||a.words.data()!=b.words.data()))||
       (out&&(out->state!=RegionState::building||out->layout.logical_bytes!=p.output_limbs*8))||
       team.width<p.product.info.workers||team.busy||!storage.data()||storage.size()<p.storage_bytes||
       !p.alignment||(uintptr_t(storage.data())&(p.alignment-1)))return {IoCode::invalid,"stored product binding",0,p.storage_bytes,storage.size()};
    auto failure = [&](IoError error) { if(out)out->state = RegionState::invalid; return error; };
    StoreBuffers io{storage.subspan(p.transfer, p.transfer_bytes), storage.subspan(p.index, index_page_bytes)};
    ValueReader reader;
    const size_t offsets[]{p.a,p.b},capacity[]{p.product.an,p.product.bn};
    sbn3_const_limbs values[2];
    for(unsigned i=0;i<2;++i){
        if(i&&p.square){values[i]=values[0];continue;}
        if(p.borrowed_inputs&(1u<<i)){
            const auto&s=p.source[i];values[i]={inputs[i].words.data()+s.first_limb,s.limbs};continue;
        }
        if(auto e=reader.open(transport,*inputs[i].stored,io))return failure(e);
        const auto&s=p.source[i];auto data=storage.subspan(offsets[i],(capacity[i]+8)*8);
        std::memset(data.data()+s.limbs*8,0,(capacity[i]+8-s.limbs)*8);
        if(auto e=read_source_words(reader,s,0,s.limbs,data))return failure(e);
        values[i]={reinterpret_cast<const uint64_t *>(data.data()),capacity[i]};
    }
    if (transport.cancel.test()) return failure({IoCode::cancelled, "stored product before compute"});
    auto *result = reinterpret_cast<uint64_t *>(storage.data()+p.result);
    const uint64_t left=m61::words(values[0].data,values[0].count);
    const uint64_t expected=m61::multiply(left,p.square?left:m61::words(values[1].data,values[1].count));
    if(values[0].count&&values[1].count){
        auto tables = Frame::external(storage.data()+p.tables, p.product.prepared_bytes);
        auto work = Frame::external(storage.data()+p.work, p.product.info.workspace_bytes);
        const auto program = product_program_prepare(p.product, tables);
        Call call{&program, &work, values[0],values[1],
                  {result, sbn3_mul_output_capacity(&p.product.info)},bool(p.borrowed_inputs)};
        sbn3_team_run(&team, multiply, &call);
    }
    const size_t used=values[0].count&&values[1].count?values[0].count+values[1].count:0;
    std::fill(result+used,result+p.product.an+p.product.bn,uint64_t(0));
    require(m61::words(result, p.product.an+p.product.bn) == expected, SBN3_FATAL_MATH, "stored product m61");
    if(p.require_zero_high)for(size_t i=p.output_limbs;i<p.product.an+p.product.bn;++i)require(!result[i],SBN3_FATAL_MATH,"stored resident high support");
    const auto view=storage.subspan(p.result+p.output_first*8,p.output_limbs*8);
    if(borrowed){borrowed->words={reinterpret_cast<const uint64_t *>(view.data()),p.output_limbs};borrowed->extent={};borrowed->extent.observe(0,view);return {};}
    ValueWriter writer;
    if (auto e = writer.open(transport, *out, io,extent)) return failure(e);
    if (auto e = writer.append(view)) return failure(e);
    if (auto e = writer.seal()) return failure(e);
    return {};
}
IoError run_resident_product(const ResidentProductPlan&p,Transport&t,const Region&a,const Region&b,Region&out,
    std::span<uint8_t>w,sbn3_team&team,WordExtent*extent) noexcept {return execute(p,t,{&a,{}},{&b,{}},&out,w,team,extent,nullptr);}
IoError run_resident_product_borrowed(const ResidentProductPlan&p,Transport&t,const Region&a,const Region&b,
    std::span<uint8_t>w,sbn3_team&team,ResidentProductValue&out) noexcept {
    if(out.words.data())return {IoCode::invalid,"resident product result already bound"};
    return execute(p,t,{&a,{}},{&b,{}},nullptr,w,team,nullptr,&out);
}
IoError run_resident_product(const ResidentProductPlan&p,Transport&t,ResidentProductInput a,ResidentProductInput b,Region&out,
    std::span<uint8_t>w,sbn3_team&team,WordExtent*extent) noexcept {return execute(p,t,a,b,&out,w,team,extent,nullptr);}
IoError run_resident_product_borrowed(const ResidentProductPlan&p,Transport&t,ResidentProductInput a,ResidentProductInput b,
    std::span<uint8_t>w,sbn3_team&team,ResidentProductValue&out) noexcept {
    if(out.words.data())return {IoCode::invalid,"resident product result already bound"};
    return execute(p,t,a,b,nullptr,w,team,nullptr,&out);
}
bool project_resident_output(ResidentProductPlan&p,size_t limbs) noexcept {
    if(limbs>p.product.an+p.product.bn)return false;p.output_limbs=limbs;return true;
}
namespace {
bool pair_layout(ResidentPairPlan&p){
    p.alignment=std::max(size_t(4096),p.product.info.workspace_alignment);size_t cursor=0;
    auto put=[&](size_t n,size_t align,size_t&at){return align_size(cursor,align,at)&&add_size(at,n,cursor);};
    size_t common,output,io;
    if(!mul_size(p.product.an+8,8,common)||!mul_size(std::max(p.product.an+p.product.bn+8,
        sbn3_mul_output_capacity(&p.product.info)),8,output)||!add_size(p.transfer_bytes,index_page_bytes,io))return false;
    if(!((p.borrowed_inputs&1)||put(common,64,p.common))||
       !put(p.product.prepared_bytes,128,p.tables)||
       !put(p.product.pair_workspace_bytes,p.product.info.workspace_alignment,p.work)||
       !put(std::max(io,p.consume_bytes),4096,p.transfer)||!add_size(p.transfer,p.transfer_bytes,p.index)||
       !put(output,64,p.result)||!align_size(cursor,p.alignment,p.storage_bytes))return false;
    return true;
}
struct PairCall {const ProductProgram*program;Frame*work;sbn3_const_limbs common,fresh;sbn3_limbs out;ProductPairContinuation next;sbn3_team_scope**continuation_scope;};
void pair_multiply(void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<PairCall*>(ptr);
    *c.continuation_scope=scope;
    product_program_pair_continue(*c.program,*c.work,scope,c.common,c.fresh,c.out,c.next);}
}
bool query_resident_pair(const ResidentProductPlan&a,const ResidentProductPlan&b,size_t consume,ResidentPairPlan&out) noexcept {
    if(a.product.bn!=b.product.bn||a.source[1]!=b.source[1]||a.product.info.workers!=b.product.info.workers)return false;
    const auto&anchor=a.product.an>=b.product.an?a:b;
    auto options=product_program_batch_options(anchor.product,anchor.product.info.prime_batch);options.borrow_output=0;
    ResidentPairPlan p{};
    if(product_program_replay(a.product.bn,std::max(a.product.an,b.product.an),options,anchor.product.plan.opaque[1],p.product)!=SBN3_SUPPORTED||
       !p.product.pair_workspace_bytes||!backend_lookup(p.product.plan.opaque[1])->program_pair_continue||
       (p.product.contract&(program_bounded_inputs|program_consume_inputs))!=(program_bounded_inputs|program_consume_inputs))return false;
    p.capacity[0]=a.product.bn;p.capacity[1]=a.product.an;p.capacity[2]=b.product.an;
    p.source[0]=a.source[1];p.source[1]=a.source[0];p.source[2]=b.source[0];
    for(unsigned i=0;i<3;++i)p.source_capacity[i]=p.source[i].storage_limbs;
    p.transfer_bytes=std::max(a.transfer_bytes,b.transfer_bytes);p.consume_bytes=consume;
    if(!pair_layout(p))return false;out=p;return true;
}
bool borrow_resident_pair_inputs(ResidentPairPlan&p,unsigned mask) noexcept {
    if(mask&~7u)return false;auto next=p;next.borrowed_inputs=mask;
    for(unsigned i=0;i<3;++i)if(mask&(1u<<i)){
        const auto&s=p.source[i];if(s.bit_shift||s.low_zero_words||s.ones_complement)return false;
    }
    if(!pair_layout(next))return false;p=next;return true;
}
bool rebind_resident_pair(ResidentPairPlan&p,SourceWindow a,SourceWindow b,SourceWindow c) noexcept {
    SourceWindow sources[]{a,b,c};
    for(unsigned i=0;i<3;++i)if(!resolve_source_window(p.capacity[i],sources[i])||sources[i].storage_limbs>p.source_capacity[i])return false;
    std::copy(std::begin(sources),std::end(sources),p.source);return true;
}
IoError run_resident_pair(const ResidentPairPlan&p,Transport&t,const ResidentProductInput(&inputs)[3],
    const ResidentProductPlan(&views)[2],std::span<uint8_t>w,sbn3_team&team,void*context,
    IoError(*consume)(void*,const ResidentProductValue&,sbn3_team_scope*),ResidentProductValue&out) noexcept {
    OperationSpan trace(t,"native_shared_pair",p.product.an,p.product.bn,p.product.info.np,p.product.info.trunk_bits,p.product.info.C,p.product.info.M2);
    if(w.size()<p.storage_bytes||(uintptr_t(w.data())&(p.alignment-1))||team.busy||team.width<p.product.info.workers||
       out.words.data()||!consume)return {IoCode::invalid,"resident pair binding"};
    StoreBuffers io{w.subspan(p.transfer,p.transfer_bytes),w.subspan(p.index,index_page_bytes)};
    sbn3_const_limbs values[3]{};uint64_t residue[3]{};
    const auto load=[&](unsigned i)->IoError{const auto&in=inputs[i];const auto&s=p.source[i];
        if(p.borrowed_inputs&(1u<<i)){
            if(in.stored||!in.words.data()||in.words.size()!=s.storage_limbs||s.bit_shift||s.low_zero_words||s.ones_complement||
               overlaps(in.words.data(),in.words.size()*8,w.data(),p.storage_bytes))return {IoCode::invalid,"resident pair loan"};
            values[i]={in.words.data()+s.first_limb,s.limbs};
        }else{
            if(!in.stored||!in.words.empty()||in.stored->state!=RegionState::sealed||in.stored->layout.logical_bytes/8!=s.storage_limbs)
                return {IoCode::invalid,"resident pair stored source"};
            ValueReader reader;if(auto e=reader.open(t,*in.stored,io))return e;
            auto data=w.subspan(i?p.result:p.common,(p.capacity[i]+8)*8);
            std::memset(data.data()+s.limbs*8,0,(p.capacity[i]+8-s.limbs)*8);
            if(auto e=read_source_words(reader,s,0,s.limbs,data))return e;
            values[i]={reinterpret_cast<const uint64_t*>(data.data()),s.limbs};
        }
        residue[i]=m61::words(values[i].data,values[i].count);return {};
    };
    if(auto e=load(0))return e;if(auto e=load(1))return e;
    if(t.cancel.test())return {IoCode::cancelled,"resident pair before compute"};
    TeamWaitScope io_waits(team,10000);
    const size_t n=p.product.an+p.product.bn,capacity=sbn3_mul_output_capacity(&p.product.info);
    auto*z=reinterpret_cast<uint64_t*>(w.data()+p.result);
    const auto finish=[&](unsigned i){const size_t used=values[0].count&&values[i+1].count?values[0].count+values[i+1].count:0;
        std::fill(z+used,z+n,uint64_t(0));
        require(m61::words(z,n)==m61::multiply(residue[0],residue[i+1]),SBN3_FATAL_MATH,"resident pair m61");
        const auto&v=views[i];require(v.output_first<=n&&v.output_limbs<=n-v.output_first,SBN3_FATAL_WORKSPACE,"resident pair view");
        ResidentProductValue result{{z+v.output_first,v.output_limbs},{}};
        result.extent.observe(0,{reinterpret_cast<const uint8_t*>(result.words.data()),result.words.size()*8});return result;
    };
    auto tables=Frame::external(w.data()+p.tables,p.product.prepared_bytes);
    auto work=Frame::external(w.data()+p.work,p.product.pair_workspace_bytes);
    const auto program=product_program_prepare(p.product,tables);
    IoError error{};sbn3_team_scope*continuation_scope=nullptr;
    const auto transition=[&](sbn3_const_limbs&fresh,sbn3_limbs&result){
        error=consume(context,finish(0),continuation_scope);if(error)return false;
        if(t.cancel.test()){error={IoCode::cancelled,"resident pair product boundary"};return false;}
        error=load(2);if(error)return false;fresh=values[2];result={z,capacity};return true;
    };
    if(values[0].count&&values[1].count){
        ProductPairContinuation next{const_cast<void*>(static_cast<const void*>(&transition)),[](void*ptr,sbn3_const_limbs&fresh,sbn3_limbs&result){
            return (*static_cast<const decltype(transition)*>(ptr))(fresh,result);}};
        PairCall call{&program,&work,values[0],values[1],{z,capacity},next,&continuation_scope};sbn3_team_run(&team,pair_multiply,&call);
    }else{
        // Exact zero first product. Still run its linear consumer before the
        // next source load; the same planned storage and arithmetic suffice.
        sbn3_const_limbs fresh;sbn3_limbs result;
        if(transition(fresh,result)&&values[0].count&&fresh.count){
            Call single{&program,&work,values[0],fresh,result,true};
            auto consume_mul=[](void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<Call*>(ptr);
                product_program_bounded(*c.program,*c.work,scope,c.a,c.b,c.out,true);};
            sbn3_team_run(&team,consume_mul,&single);
        }
    }
    if(error)return error;out=finish(1);return {};
}
} // namespace sbn::v3::external
