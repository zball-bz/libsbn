// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/integer_emit.hpp"
#include "external/two_slot.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <new>
namespace sbn::v3::external {
namespace {
// A rotated source meets the regular segment boundaries at an arithmetic
// progression. Count physical integrity chunks, including duplicated boundary
// chunks, without walking the segments or inspecting operands.
bool subtraction_reads(const IntegerEmitPlan&p,SourceWindow source,uint64_t&reads){
    const size_t K=size_t(p.carry.output_bytes/8),S=size_t(p.carry.segment_bytes/8),Q=chunk_bytes/8;
    if(!K||!S||K%S||source.limbs==SIZE_MAX||!resolve_source_window(source.limbs,source)||source.limbs>K)return false;
    size_t bytes;if(!mul_size(source.storage_limbs,8,bytes))return false;RegionLayout l{};
    if(region_layout(bytes,0,l)||l.index_bytes>p.subtract_index_bytes)return false;
    reads=l.index_bytes;if(!source.limbs)return true;size_t begin,end;source_word_range(source,0,source.limbs,begin,end);
    if(begin==end)return true;
    uint64_t chunks=(end-begin+begin%Q+Q-1)/Q,extra=0;
    const __int128 physical=__int128(source.first_limb)-source.low_zero_words;
    const __int128 low=std::max(__int128(1),__int128(begin)+(source.bit_shift>=0)-physical);
    const __int128 high=std::min(__int128(source.limbs)-1,__int128(end)-(source.bit_shift<=0)-physical);
    __int128 first=S-(p.cyclic.subtract_word%K)%S;
    if(first<low)first+=(low-first+S-1)/S*S;
    if(first<=high){
        const uint64_t count=uint64_t(1+(high-first)/S);extra=count;
        if(!source.bit_shift){
            const uint64_t g=std::gcd(S,Q),at=uint64_t(physical+first);
            if(at%g==0){const uint64_t modulus=Q/g,step=S/g;
                uint64_t inverse=1;for(unsigned j=0;j<6;++j)inverse*=2-step*inverse;
                const uint64_t start=((0-at/g)*inverse)&(modulus-1);
                if(start<count)extra-=1+(count-1-start)/modulus;
            }
        }
    }
    uint64_t data;if(__builtin_add_overflow(chunks,extra,&chunks)||__builtin_mul_overflow(chunks,uint64_t(chunk_bytes),&data)||
       __builtin_add_overflow(reads,data,&reads))return false;
    return true;
}
}
bool query_integer_emit(unsigned np,unsigned T,size_t rows,size_t inner,size_t stripe,uint64_t output,unsigned workers,IntegerEmitPlan&out,CyclicResidualSpec cyclic,size_t slots,bool ordered,MiddleWindowSpec middle,size_t guard_first) noexcept {
    const auto *crt=product::p48_crt_backend(np);
    if(!crt||!crt->prepared_bytes(T)||!rows||!inner||!stripe||inner%stripe||!workers||workers>32)return false;
    IntegerEmitPlan p{};p.crt=crt;p.trunk_bits=T;p.rows=rows;p.inner_cells=inner;p.stripe_cells=stripe;p.workers=unsigned(std::min(rows,size_t(workers)));p.team_workers=workers;p.cyclic=cyclic;
    p.middle=middle;p.middle_guard_first=guard_first;
    if(middle&&(cyclic||middle.first>=middle.end||middle.end>output/8||guard_first>middle.first))return false;
    p.row_slots=slots?slots:p.workers;if(p.row_slots<p.workers||p.row_slots>UINT32_MAX)return false;
    uint64_t segments,length;if(__builtin_mul_overflow(uint64_t(rows),uint64_t(inner/stripe),&segments)||
       __builtin_mul_overflow(uint64_t(stripe),uint64_t(T),&length)||
       !query_carry_output(segments,length,output,p.carry,bool(cyclic)||bool(middle),ordered&&length%chunk_bytes==0?inner/stripe:0))return false;
    if((cyclic||middle)&&(output%8||length%8))return false;
    size_t cursor=0,row_bytes,buffers,tails;
    auto put=[&](size_t n,size_t align,size_t&at){return align_size(cursor,align,at)&&add_size(at,n,cursor);};
    if(!add_size(size_t(length),256,row_bytes)||!align_size(row_bytes,4096,p.row_buffer_stride)||!mul_size(p.row_buffer_stride,p.row_slots,buffers)||
       !mul_size(p.row_slots,sizeof(product::CrtCarry),tails)||
       !put(crt->prepared_bytes(T),64,p.prepared)||!put(p.carry.descriptor_bytes,alignof(CarrySegment),p.descriptors)||
       !put(p.carry.fragment_bytes,4096,p.fragments)||!put(chunk_bytes,4096,p.assembly)||
       !put(tails,64,p.tails)||!put(buffers,4096,p.rows_buffer))return false;
    if(cyclic.stored_subtract){
        auto source=cyclic.subtract_source;size_t bytes;RegionLayout input;
        if(!cyclic||source.limbs==SIZE_MAX||!resolve_source_window(source.limbs,source)||source.limbs>output/8||
           !mul_size(source.storage_limbs,8,bytes)||region_layout(bytes,0,input))return false;
        p.cyclic.subtract_source=source;p.subtract_index_bytes=size_t(input.index_bytes);p.subtract_capacity_words=source.storage_limbs;
        if(!put((1u<<20)+index_page_bytes,4096,p.subtract_io)||!put(p.subtract_index_bytes,4096,p.subtract_index)||
           !put(buffers,4096,p.subtract_rows)||!subtraction_reads(p,source,p.subtract_read_bytes))return false;
    }
    if(!align_size(cursor,4096,p.storage_bytes))return false;
    p.pipeline=p.workers>1;
    out=p;return true;
}
bool rebind_emit_subtract(IntegerEmitPlan&p,SourceWindow source) noexcept {
    if(!p.cyclic.stored_subtract||source.limbs!=p.cyclic.subtract_source.limbs||!resolve_source_window(source.limbs,source)||
       source.storage_limbs>p.subtract_capacity_words)return false;
    uint64_t reads;if(!subtraction_reads(p,source,reads))return false;
    p.cyclic.subtract_source=source;p.subtract_read_bytes=reads;return true;
}
IoError prepare_integer_emit(const IntegerEmitPlan&p,std::span<uint8_t>w,ChunkWriter&writer,sbn3_team&team,IntegerEmitBinding&b,const Region*subtract) noexcept {
    if(!w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes||team.busy||team.width<p.workers)return {IoCode::invalid,"integer emission grants",0,p.storage_bytes,w.size()};
    if(p.cyclic.stored_subtract!=bool(subtract))return {IoCode::invalid,"cyclic subtractand presence"};
    if(subtract){
        if(subtract->state!=RegionState::sealed||subtract->layout.logical_bytes!=p.cyclic.subtract_source.storage_limbs*8||!writer.transport())
            return {IoCode::invalid,"cyclic subtractand binding"};
        StoreBuffers io{w.subspan(p.subtract_io,1u<<20),w.subspan(p.subtract_io+(1u<<20),index_page_bytes)};
        if(auto e=b.subtract_reader.open_pinned(*writer.transport(),*subtract,io,w.subspan(p.subtract_index,p.subtract_index_bytes)))return e;
    }
    b.subtract_residue=0;
    auto *segments=reinterpret_cast<CarrySegment *>(w.data()+p.descriptors);
    for(size_t i=0;i<p.carry.segments;++i)::new(segments+i)CarrySegment{};
    auto *tails=reinterpret_cast<product::CrtCarry *>(w.data()+p.tails);
    for(size_t i=0;i<p.row_slots;++i)::new(tails+i)product::CrtCarry{};
    b.plan=&p;b.workspace=w;b.writer=&writer;b.team=&team;
    p.crt->prepare_scaled(p.trunk_bits,p.preconditioned,w.subspan(p.prepared,p.crt->prepared_bytes(p.trunk_bits)));
    if(auto e=b.carry.open(p.carry,writer,{segments,size_t(p.carry.segments)},w.subspan(p.fragments,p.carry.fragment_bytes),w.subspan(p.assembly,chunk_bytes)))return e;
    if(p.middle)b.carry.inspect_middle(p.middle_guard_first*8,p.middle.first*8);
    return {};
}
namespace {
struct Batch {IntegerEmitBinding*b;const CoefficientStripe*s;size_t first,count;};
void finish_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot){const auto&p=*b.plan;auto w=b.workspace;
    auto *tails=reinterpret_cast<product::CrtCarry *>(w.data()+p.tails);
    auto output=w.subspan(p.rows_buffer+slot*p.row_buffer_stride,p.row_buffer_stride);
    if(p.scalar!=1){
        uint64_t carry=0;auto*words=reinterpret_cast<uint64_t *>(output.data());
        for(size_t j=0;j<p.carry.segment_bytes/8;++j){const __uint128_t x=__uint128_t(words[j])*p.scalar+carry;
            words[j]=uint64_t(x);carry=uint64_t(x>>64);}
        for(auto&word:tails[slot].limb){const __uint128_t x=__uint128_t(word)*p.scalar+carry;word=uint64_t(x);carry=uint64_t(x>>64);}
        require(!carry,SBN3_FATAL_MATH,"scaled CRT tail width");
    }
    if(p.cyclic){
        const size_t words=size_t(p.carry.segment_bytes/8),K=size_t(p.carry.output_bytes/8);
        const size_t sid=row*(p.inner_cells/p.stripe_cells)+s.first_column/p.stripe_cells;
        const size_t unit=p.cyclic.subtract_word%K;
        auto*data=reinterpret_cast<uint64_t *>(output.data());uint64_t carry=0;
        const auto*subtract=p.cyclic.stored_subtract?reinterpret_cast<const uint64_t *>(w.data()+p.subtract_rows+slot*p.row_buffer_stride):nullptr;
        for(size_t j=0;j<words;++j){const uint64_t addend=UINT64_MAX-(subtract?subtract[j]:uint64_t(sid*words+j==unit));
            const __uint128_t x=__uint128_t(data[j])+addend+carry;data[j]=uint64_t(x);carry=uint64_t(x>>64);}
        for(auto&word:tails[slot].limb){const __uint128_t x=__uint128_t(word)+carry;word=uint64_t(x);carry=uint64_t(x>>64);}
        require(!carry,SBN3_FATAL_MATH,"cyclic subtraction carry width");
    }
}
void emit_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot){const auto&p=*b.plan;auto w=b.workspace;
    auto*tails=reinterpret_cast<product::CrtCarry*>(w.data()+p.tails);
    p.crt->emit(w.subspan(p.prepared,p.crt->prepared_bytes(p.trunk_bits)),
        s.data.data()+row*p.stripe_cells*8,p.rows*p.stripe_cells*8,p.stripe_cells,
        w.subspan(p.rows_buffer+slot*p.row_buffer_stride,p.row_buffer_stride),tails[slot]);
    finish_row(b,s,row,slot);
}
void rows(void*ptr,uint64_t lo,uint64_t hi,unsigned){auto&a=*static_cast<Batch *>(ptr);
    for(size_t i=lo;i<hi;++i)emit_row(*a.b,*a.s,a.first+i,unsigned(i));
}
void run(void*ptr,sbn3_team_scope*s){auto &a=*static_cast<Batch *>(ptr);const auto&p=*a.b->plan;
    sbn3_team_scope part{s->team,s->first,p.workers,false,s->epoch};sbn3_team_for(&part,0,a.count,1,SBN3_STATIC,rows,&a);}
IoError load_subtract(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot){
    const auto&p=*b.plan;if(!p.cyclic.stored_subtract)return {};
    const size_t K=size_t(p.carry.output_bytes/8),words=size_t(p.carry.segment_bytes/8);
    const auto&source=p.cyclic.subtract_source;const size_t shift=p.cyclic.subtract_word%K;
    const size_t sid=row*(p.inner_cells/p.stripe_cells)+s.first_column/p.stripe_cells;
    auto data=b.workspace.subspan(p.subtract_rows+slot*p.row_buffer_stride,p.row_buffer_stride);std::memset(data.data(),0,words*8+16);
    size_t from=(sid*words+K-shift)%K,at=0;
    while(at<words){const size_t end=from<source.limbs?source.limbs:K,count=std::min(words-at,end-from);
        if(from<source.limbs){
            if(auto e=read_source_words(b.subtract_reader,source,from,count,data.subspan(at*8)))return e;
            m61::add_at(b.subtract_residue,data.data()+at*8,count,(from%61+p.cyclic.subtract_word%61)%61);
        }
        at+=count;from=(from+count)%K;
    }
    return {};
}
IoError submit_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot){
    const auto&p=*b.plan;const auto*tails=reinterpret_cast<const product::CrtCarry *>(b.workspace.data()+p.tails);
    const uint64_t sid=row*(p.inner_cells/p.stripe_cells)+s.first_column/p.stripe_cells;
    return b.carry.submit(sid,b.workspace.subspan(p.rows_buffer+slot*p.row_buffer_stride,size_t(p.carry.segment_bytes)),tails[slot]);
}
struct EmitPipeline {
    IntegerEmitBinding*b;
    const CoefficientStripe*s;
    unsigned workers,slots;
    using Exchange=SlotExchange<32>;
    Exchange exchange{};
    IoError error{};
    void fail(IoError e){error=e;exchange.stop();}
};
void emit_io(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<EmitPipeline *>(ptr);const auto&p=*c.b->plan;
    auto load=[&](size_t row,unsigned slot){
        if(auto e=load_subtract(*c.b,*c.s,row,slot)){c.fail(e);return false;}
        c.exchange.publish(slot,EmitPipeline::Exchange::ready);return true;
    };
    for(unsigned slot=0;slot<c.slots;++slot)if(!load(slot,slot))return;
    for(size_t row=0;row<p.rows;++row){const unsigned slot=unsigned(row%c.slots);
        if(!c.exchange.wait(slot,EmitPipeline::Exchange::computed))return;
        if(auto e=submit_row(*c.b,*c.s,row,slot)){c.fail(e);return;}
        if(row+c.slots<p.rows&&!load(row+c.slots,slot))return;
    }
}
void emit_worker(void*ptr,sbn3_team_region*region){
    auto&c=*static_cast<EmitPipeline *>(ptr);const unsigned rank=sbn3_region_rank(region);
    for(size_t row=rank;row<c.b->plan->rows;row+=c.workers){const unsigned slot=unsigned(row%c.slots);
        if(!c.exchange.wait(slot,EmitPipeline::Exchange::ready))return;
        emit_row(*c.b,*c.s,row,slot);
        os::publish_stores();
        if(c.exchange.stopped())return;
        c.exchange.publish(slot,EmitPipeline::Exchange::computed);
    }
}
void emit_compute(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<EmitPipeline *>(ptr);sbn3_team_scope part{scope->team,scope->first,c.workers,false,scope->epoch};
    sbn3_team_region_run(&part,emit_worker,ptr);
}
void emit_pipeline(void*ptr,sbn3_team_scope*scope){
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,1,emit_io,ptr,emit_compute,ptr);
}
}
IoError load_integer_subtract_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot) noexcept {
    require(row<b.plan->rows&&slot<b.plan->row_slots,SBN3_FATAL_WORKSPACE,"integer input row slot");return load_subtract(b,s,row,slot);
}
void compute_integer_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot) noexcept {
    require(row<b.plan->rows&&slot<b.plan->row_slots,SBN3_FATAL_WORKSPACE,"integer output row slot");emit_row(b,s,row,slot);
}
void finish_integer_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot) noexcept {
    require(row<b.plan->rows&&slot<b.plan->row_slots,SBN3_FATAL_WORKSPACE,"integer fused output row slot");finish_row(b,s,row,slot);
}
IoError submit_integer_row(IntegerEmitBinding&b,const CoefficientStripe&s,size_t row,unsigned slot) noexcept {
    require(row<b.plan->rows&&slot<b.plan->row_slots,SBN3_FATAL_WORKSPACE,"integer submit row slot");return submit_row(b,s,row,slot);
}
IoError emit_integer_stripe(IntegerEmitBinding&b,const CoefficientStripe&s) noexcept {
    if(!b.plan||!b.team||b.team->busy)return {IoCode::invalid,"integer stripe binding"};const auto&p=*b.plan;
    if(s.primes!=p.crt->primes||s.rows!=p.rows||s.columns!=p.stripe_cells||s.first_column%p.stripe_cells||
       s.first_column>p.inner_cells-p.stripe_cells||s.data.size()<s.primes*s.rows*s.columns*8)return {IoCode::invalid,"integer stripe shape"};
    if(p.pipeline){
        const unsigned workers=std::min(p.workers,std::min(p.team_workers,b.team->width)-1);
        // Many compute lanes own one slot each. A single lane alternates two
        // slots, allowing its next row to run while the previous row is written.
        // No slot can be awaited by two distinct compute lanes.
        EmitPipeline call{&b,&s,workers,workers==1?2:workers};
        sbn3_team_run(b.team,emit_pipeline,&call);return call.error;
    }
    for(size_t first=0;first<p.rows;first+=p.workers){
        Batch batch{&b,&s,first,std::min(size_t(p.workers),p.rows-first)};
        for(size_t i=0;i<batch.count;++i)if(auto e=load_subtract(b,s,first+i,unsigned(i)))return e;
        sbn3_team_run(b.team,run,&batch);
        for(size_t i=0;i<batch.count;++i)if(auto e=submit_row(b,s,first+i,unsigned(i)))return e;
    }
    return {};
}
IoError finish_integer_emit(IntegerEmitBinding&b,uint64_t expected,CyclicResidualResult *result,MiddleWindowResult*middle) noexcept {
    if(auto e=b.carry.finish())return e;
    if(b.plan->middle){
        require(middle&&!result&&b.plan->carry.cyclic,SBN3_FATAL_ARGUMENT,"middle result descriptor");
        require(b.carry.residue()==expected,SBN3_FATAL_MATH,"external cyclic middle m64",expected,b.carry.residue());
        middle->exact=b.carry.middle_exact();return b.writer->seal();
    }
    require(!middle,SBN3_FATAL_ARGUMENT,"ordinary product given middle descriptor");
    if(b.plan->cyclic){
        require(result&&b.plan->carry.cyclic,SBN3_FATAL_ARGUMENT,"cyclic residual result descriptor");
        const size_t K=size_t(b.plan->carry.output_bytes/8);const auto raw=b.carry.full_extent();
        const bool complemented=raw.end==K&&(raw.top_word>>63);
        auto magnitude=complemented?b.carry.complement_extent():raw;
        const auto&bound=b.plan->cyclic.bound;
        require(bound.verify&&bound.high_exclusive&&K>=2&&bound.words<=K-2&&
            (magnitude.end<=bound.words||(magnitude.end==bound.words+1&&magnitude.top_word<bound.high_exclusive)),
            SBN3_FATAL_MATH,"cyclic residual signed magnitude");
        const uint64_t ring=(uint64_t(1)<<((3*(K%61))%61))-1;
        uint64_t residue=b.carry.residue();
        if(complemented)residue=residue>=ring?residue-ring:residue+m61::modulus-ring;
        require(residue==expected,SBN3_FATAL_MATH,"cyclic residual signed m61",expected,residue);
        magnitude.residue_m61=complemented&&residue?m61::modulus-residue:residue;
        *result={complemented,magnitude};return b.writer->seal();
    }
    require(!result,SBN3_FATAL_ARGUMENT,"linear product given cyclic descriptor");
    require(b.carry.residue()==expected,SBN3_FATAL_MATH,"external integer output m61",expected,b.carry.residue());
    const auto&limit=b.plan->bound;
    if(limit.verify&&limit.high_exclusive){const auto extent=b.carry.full_extent();
        require(extent.end<=limit.words||(extent.end==limit.words+1&&extent.top_word<limit.high_exclusive),
                SBN3_FATAL_MATH,"external integer magnitude bound");
    }
    return b.writer->seal();
}
} // namespace sbn::v3::external
