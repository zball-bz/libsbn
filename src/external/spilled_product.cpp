// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/spilled_product.hpp"
#include "external/row_cache.hpp"
#include "external/two_slot.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "common/m64.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <new>
namespace sbn::v3::external {
namespace {
constexpr size_t io_bytes=1u<<20;
struct Control {
    ChunkWriter writers[20],output;
    ScratchPlane planes[20];
    InversePlanesBinding inverse;
    IntegerEmitBinding emit;
    ~Control(){for(auto&p:planes)p.discard();for(auto&w:writers)w.abandon();output.abandon();}
};
struct FinalStream {
    const SpilledProductPlan*p;std::span<ScratchPlane>planes;IntegerEmitBinding*emit;
    InversePlanesBinding inverse[3]{};CoefficientStripe stripes[3]{};
    unsigned io_workers=1;
    bool duplex=false,error_claimed=false;
    TwoSlotExchange exchange{};ThreeSlotExchange separate{};IoError error{};
    void fail(IoError e){bool expected=false;
        if(__atomic_compare_exchange_n(&error_claimed,&expected,true,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))error=e;
        exchange.stop();separate.stop();}
};
bool final_write_slot(FinalStream&c,unsigned slot){
    for(size_t row=0;row<c.p->emit.rows;++row)
        if(auto e=submit_integer_row(*c.emit,c.stripes[slot],row,unsigned(slot*c.p->emit.rows+row))){c.fail(e);return false;}
    return true;
}
bool final_load_slot(FinalStream&c,unsigned slot,size_t column,sbn3_team_scope*scope){
    const auto&p=*c.p;
    if(auto e=load_inverse_stripe(c.inverse[slot],column,c.planes,scope)){c.fail(e);return false;}
    c.stripes[slot].first_column=column;c.stripes[slot].rows=p.emit.rows;c.stripes[slot].columns=p.inverse.spec.stripe_cells;
    for(size_t row=0;row<p.emit.rows;++row)
        if(auto e=load_integer_subtract_row(*c.emit,c.stripes[slot],row,unsigned(slot*p.emit.rows+row))){c.fail(e);return false;}
    return true;
}
void final_io(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<FinalStream*>(ptr);const auto&p=*c.p;const size_t jobs=p.spec.inner_cells/p.inverse.spec.stripe_cells;
    for(size_t job=0;job<jobs;++job){const unsigned slot=unsigned(job%2);
        if(job>=2&&(!c.exchange.wait(slot,TwoSlotExchange::computed)||!final_write_slot(c,slot)))return;
        const size_t column=job*p.inverse.spec.stripe_cells;
        if(!final_load_slot(c,slot,column,scope))return;
        c.exchange.publish(slot,TwoSlotExchange::ready);
    }
    for(size_t job=jobs-std::min(size_t(2),jobs);job<jobs;++job){const unsigned slot=unsigned(job%2);
        if(!c.exchange.wait(slot,TwoSlotExchange::computed)||!final_write_slot(c,slot))return;}
}
void final_read(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<FinalStream*>(ptr);const auto&p=*c.p;
    for(size_t column=0,job=0;column<p.spec.inner_cells;column+=p.inverse.spec.stripe_cells,++job){const unsigned slot=unsigned(job%3);
        if(!c.separate.wait(slot,ThreeSlotExchange::empty,ThreeSlotExchange::reader)||!final_load_slot(c,slot,column,scope))return;
        c.separate.publish(slot,ThreeSlotExchange::ready);
    }
}
void final_write(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<FinalStream*>(ptr);const auto&p=*c.p;
    for(size_t job=0;job<p.spec.inner_cells/p.inverse.spec.stripe_cells;++job){const unsigned slot=unsigned(job%3);
        if(!c.separate.wait(slot,ThreeSlotExchange::computed,ThreeSlotExchange::writer)||!final_write_slot(c,slot))return;
        c.separate.publish(slot,ThreeSlotExchange::empty);
    }
}
struct FinalRows {IntegerEmitBinding*emit;CoefficientStripe*stripe;size_t base;bool reconstructed=false;};
void final_rows(void*ptr,uint64_t lo,uint64_t hi,unsigned){auto&c=*static_cast<FinalRows*>(ptr);
    for(size_t row=lo;row<hi;++row){if(c.reconstructed)finish_integer_row(*c.emit,*c.stripe,row,unsigned(c.base+row));
        else compute_integer_row(*c.emit,*c.stripe,row,unsigned(c.base+row));}}
void final_compute(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<FinalStream*>(ptr);const auto&p=*c.p;
    for(size_t column=0,job=0;column<p.spec.inner_cells;column+=p.inverse.spec.stripe_cells,++job){const unsigned slot=unsigned(job%(c.duplex?3:2));
        const bool ready=c.duplex?c.separate.wait(slot,ThreeSlotExchange::ready,ThreeSlotExchange::computer):c.exchange.wait(slot,TwoSlotExchange::ready);
        if(!ready)return;
        if(p.inverse.fused.workspace_bytes){
            const auto&b=c.inverse[slot];const auto&i=*b.plan;const auto&e=p.emit;auto w=c.emit->workspace;
            product::p48_outer_inverse_emit(b.workspace.subspan(i.prepared,i.info.prepared_bytes),
                b.workspace.subspan(i.coefficients,i.coefficient_bytes),*e.crt,w.subspan(e.prepared,e.crt->prepared_bytes(e.trunk_bits)),
                w.subspan(e.rows_buffer+slot*e.rows*e.row_buffer_stride,e.rows*e.row_buffer_stride),e.row_buffer_stride,
                {reinterpret_cast<product::CrtCarry*>(w.data()+e.tails)+slot*e.rows,e.rows},b.workspace.subspan(i.work,i.fused.workspace_bytes),scope);
        }else transform_inverse_stripe(c.inverse[slot],column,scope,c.stripes[slot]);
        FinalRows rows{c.emit,&c.stripes[slot],slot*p.emit.rows,bool(p.inverse.fused.workspace_bytes)};
        sbn3_team_scope part{scope->team,scope->first,std::min(scope->width,p.emit.workers),false,scope->epoch};
        sbn3_team_for(&part,0,p.emit.rows,1,SBN3_STATIC,final_rows,&rows);
        if(c.exchange.stopped())return;
        if(c.duplex)c.separate.publish(slot,ThreeSlotExchange::computed);else c.exchange.publish(slot,TwoSlotExchange::computed);
    }
}
void final_consumers(void*ptr,sbn3_team_scope*s){sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,1,final_write,ptr,final_compute,ptr);}
void final_action(void*ptr,sbn3_team_scope*s){auto&c=*static_cast<FinalStream*>(ptr);
    sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,c.io_workers,c.duplex?final_read:final_io,ptr,c.duplex?final_consumers:final_compute,ptr);}
IoError run_final_stream(const SpilledProductPlan&p,std::span<ScratchPlane>planes,IntegerEmitBinding&emit,
                         std::span<uint8_t>pool,sbn3_team&team){
    FinalStream c{&p,planes,&emit};
    c.duplex=p.final_banks==3&&planes[0].transport()->duplex;
    if(p.inverse.spec.stripe_cells*48%chunk_bytes==0&&!planes[0].transport()->read_batch){
        const unsigned cap=std::max(1u,std::min({planes[0].transport()->parallel_reads,p.spec.primes,p.spec.workers-1-unsigned(c.duplex)}));
        const unsigned waves=(p.spec.primes+cap-1)/cap;c.io_workers=(p.spec.primes+waves-1)/waves;
    }
    auto spec=p.inverse.spec;spec.workers=p.spec.workers-c.io_workers-unsigned(c.duplex);InversePlanesPlan inverse;
    require(query_inverse_planes(spec,inverse,bool(p.inverse.fused.workspace_bytes))&&inverse.storage_bytes<=p.inverse.storage_bytes,SBN3_FATAL_WORKSPACE,"final inverse partition");inverse.pipeline=false;
    prepare_inverse_planes(inverse,pool.first(p.inverse.storage_bytes),team,c.inverse[0]);
    prepare_inverse_planes(inverse,pool.subspan(p.inverse_second,p.inverse.storage_bytes),team,c.inverse[1]);
    if(c.duplex)prepare_inverse_planes(inverse,pool.subspan(p.inverse_third,p.inverse.storage_bytes),team,c.inverse[2]);
    sbn3_team_run(&team,final_action,&c);return c.error;
}
bool disjoint(const RegionLayout&a,const RegionLayout&b){return a.data_offset<=b.data_offset?a.allocation_bytes<=b.data_offset-a.data_offset:b.allocation_bytes<=a.data_offset-b.data_offset;}
bool payload(const RegionLayout&l,uint64_t&out){return !__builtin_add_overflow(l.data_bytes,l.index_bytes,&out);}
bool final_storage(SpilledProductPlan&p){
    if(!p.spec.stream_final)return true;
    if(p.spec.workers<2||p.inverse.spec.stripe_cells>=p.spec.inner_cells)return false;
    p.final_banks=p.spec.workers>=4&&p.spec.inner_cells/p.inverse.spec.stripe_cells>=3?3:2;
    auto spec=p.inverse.spec;spec.workers=p.spec.workers-1;InversePlanesPlan inverse;
    if(!query_inverse_planes(spec,inverse,true))return false;inverse.pipeline=false;
    size_t slots;if(!mul_size(p.emit.rows,p.final_banks,slots))return false;IntegerEmitPlan emit;
    if(!query_integer_emit(p.spec.primes,p.spec.trunk_bits,p.emit.rows,p.spec.inner_cells,p.inverse.spec.stripe_cells,p.output_bytes,
                           p.spec.workers,emit,p.emit.cyclic,slots,true,p.emit.middle,p.emit.middle_guard_first))return false;
    emit.scalar=p.emit.scalar;emit.bound=p.emit.bound;emit.preconditioned=p.emit.preconditioned;emit.pipeline=false;
    p.inverse=inverse;p.emit=emit;return true;
}
bool layout_storage(SpilledProductPlan&p){
    p.middle.pipeline=p.middle.recycle_a=p.middle.compact=p.middle.duplex=false;p.middle.storage_bytes=p.middle.serial_storage_bytes;
    for(auto&f:p.forward){f.pipeline=f.pipeline_double_raw=false;f.storage_bytes=f.serial_storage_bytes;}
    if(p.spec.compact_middle&&(p.spec.raw_operand<2||!enable_middle_compact_pipeline(p.middle,SIZE_MAX)))return false;
    if(p.spec.duplex_middle&&p.spec.raw_operand>=2&&!enable_middle_duplex(p.middle,SIZE_MAX))return false;
    if(!final_storage(p))return false;
    size_t inverse_end=p.inverse.storage_bytes;p.inverse_second=p.inverse_third=0;
    if(p.spec.stream_final&&(!align_size(inverse_end,4096,p.inverse_second)||!add_size(p.inverse_second,p.inverse.storage_bytes,inverse_end)))return false;
    if(p.spec.stream_final&&p.final_banks==3&&(!align_size(inverse_end,4096,p.inverse_third)||!add_size(p.inverse_third,p.inverse.storage_bytes,inverse_end)))return false;
    size_t emit_end;if(!align_size(inverse_end,4096,p.emit_offset)||!add_size(p.emit_offset,p.emit.storage_bytes,emit_end))return false;
    p.pool_bytes=std::max({p.cached_a||p.spec.raw_operand==0?size_t(0):p.forward[0].storage_bytes,
        p.cached_square||p.spec.raw_operand==1?size_t(0):p.forward[1].storage_bytes,p.middle.storage_bytes});
    p.raw_bytes=p.raw_cached_bytes=p.raw_phase=0;p.raw_forward={};
    if(p.spec.raw_operand<2){
        size_t bytes,phase;const unsigned raw=p.spec.raw_operand;
        if(!mul_size(raw?p.spec.bn:p.spec.an,8,bytes))return false;p.raw_cached_bytes=bytes;
        size_t forward=forward_memory_workspace(p.forward[raw]);
        if(p.spec.raw_cache_bytes!=SIZE_MAX){
            if(!query_prefix_forward(p.forward[raw],p.spec.raw_cache_bytes,p.raw_forward,
                                    p.spec.raw_stripe_cells?p.spec.raw_stripe_cells:p.spec.stripe_cells))return false;
            p.raw_cached_bytes=bytes=p.spec.raw_cache_bytes;forward=p.raw_forward.storage_bytes;
        }
        if(!add_size(bytes,p.spec.trunk_bits-1,bytes)||
           !mul_size(bytes/p.spec.trunk_bits,p.spec.trunk_bits,p.raw_bytes)||!add_size(p.raw_bytes,64,p.raw_bytes)||
           !align_size(size_t(p.plane_bytes),4096,p.raw_phase)||
           !add_size(p.raw_phase,std::max(forward,p.middle.storage_bytes),phase))return false;
        p.pool_bytes=std::max(p.pool_bytes,phase);
    }
    // After every middle pass joins, the normalized raw prefix and its input
    // index are dead. Final inverse/emit may reuse their contiguous range with
    // the phase pool. Keep retained B outside that loan: it is still an input.
    p.pool_bytes=std::max(p.pool_bytes,emit_end>p.raw_bytes?emit_end-p.raw_bytes:size_t(0));
    size_t cursor=0,n;auto put=[&](size_t bytes,size_t align,size_t&at){return align_size(cursor,align,at)&&add_size(at,bytes,cursor);};
    const bool valid=put(sizeof(Control),64,p.control)&&mul_size(p.plane_index_stride,p.plane_count,n)&&put(n,4096,p.plane_indexes)&&
       mul_size(p.plane_bits_stride,p.plane_count,n)&&put(n,64,p.plane_bits)&&
       put(p.output_index_bytes,4096,p.output_index)&&put(p.output_bits_bytes,64,p.output_bits)&&
       put(io_bytes,4096,p.io_data)&&put(index_page_bytes,4096,p.io_page)&&put(p.spec.retained_bytes,4096,p.retained)&&put(p.raw_bytes,4096,p.raw_value)&&
       put(p.spec.raw_operand<2?p.forward[p.spec.raw_operand].input_index_bytes:0,4096,p.raw_input_index)&&
       put(p.pool_bytes,4096,p.pool)&&
       align_size(cursor,4096,p.storage_bytes);
    p.inverse_offset=p.spec.raw_operand<2?p.raw_value:p.pool;
    // Borrow only phase-dead workspace already required by another pass. No
    // retained plane is displaced and the product's IO/total RAM do not grow.
    if(valid){
        if(!p.spec.compact_middle&&(p.spec.raw_operand>=2||!enable_middle_recycled_pipeline(p.middle)))
            enable_middle_pipeline(p.middle,p.spec.raw_operand<2?p.pool_bytes-p.raw_phase:p.pool_bytes);
        if(p.spec.duplex_middle&&!enable_middle_duplex(p.middle,p.spec.raw_operand<2?p.pool_bytes-p.raw_phase:p.pool_bytes))return false;
        for(unsigned i=0;i<2;++i)if(p.spec.raw_operand!=i&&!(i==0&&p.cached_a)&&!(i==1&&p.cached_square))
            enable_forward_pipeline(p.forward[i],p.pool_bytes);
    }
    return valid;
}
bool raw_reads(const ForwardPlanesPlan&f,size_t cached,unsigned primes,size_t stripe,uint64_t&bytes){
    const size_t words=cached==SIZE_MAX?f.spec.source.limbs:std::min(cached/8,f.spec.source.limbs);
    uint64_t prefix,tail;const size_t step=stripe==f.spec.inner_cells?0:stripe*f.spec.trunk_bits/8;
    RegionLayout input;if(region_layout(f.source_bytes,0,input)||!source_slice_reads(f.spec.source,0,words,0,prefix)||
        !source_slice_reads(f.spec.source,words,f.spec.source.limbs-words,step,tail)||
        __builtin_mul_overflow(tail,uint64_t(primes),&tail)||__builtin_add_overflow(prefix,tail,&bytes)||
        __builtin_add_overflow(bytes,input.index_bytes,&bytes))return false;
    return true;
}
bool clear_raw(SpilledProductPlan&p){
    if(p.spec.raw_operand>=2)return true;
    uint64_t family,reads;
    if(__builtin_mul_overflow(uint64_t(p.spec.primes),p.plane_bytes,&family)||
       p.normal_read_bytes<p.raw_read_bytes||
       __builtin_add_overflow(p.normal_read_bytes-p.raw_read_bytes,p.forward[p.spec.raw_operand].source_read_bytes,&reads)||
       __builtin_add_overflow(reads,family,&p.normal_read_bytes)||
       __builtin_add_overflow(p.normal_write_bytes,family,&p.normal_write_bytes))return false;
    p.spec.raw_operand=2;p.spec.raw_cache_bytes=SIZE_MAX;p.spec.duplex_middle=false;p.raw_read_bytes=0;p.plane_count=2*p.spec.primes;
    return layout_storage(p);
}
}
bool query_spilled_product(const SpilledProductSpec&s,SpilledProductPlan&out) noexcept {
    SpilledProductPlan p{};p.spec=s;p.spec.raw_operand=2;p.spec.duplex_middle=false;
    const auto *crt=product::p48_crt_backend(s.primes);
    if(!crt||!crt->prepared_bytes(s.trunk_bits)||(s.square&&s.an!=s.bn)||!s.inner_cells||!s.outer_cells||
       s.raw_operand>2||(s.square&&s.raw_operand<2)||(s.raw_operand==2&&s.raw_cache_bytes!=SIZE_MAX))return false;
    size_t abits,bbits,words;if(!mul_size(s.an,64,abits)||!mul_size(s.bn,64,bbits)||!add_size(s.an,s.bn,words)||!mul_size(words,8,p.output_bytes))return false;
    const size_t nat=abits/s.trunk_bits+(abits%s.trunk_bits!=0),nbt=bbits/s.trunk_bits+(bbits%s.trunk_bits!=0);
    if(!crt->capacity(s.trunk_bits,std::min(nat,nbt)))return false;
    size_t need=1;if(nat&&nbt){if(!add_size(nat,nbt,need))return false;--need;}
    const size_t cells=need/8+(need%8!=0);size_t rows=cells/s.inner_cells+(cells%s.inner_cells!=0);
    if(s.cyclic||s.middle){
        size_t ring_cells,bytes;
        if(!mul_size(s.outer_cells,s.inner_cells,ring_cells)||!mul_size(ring_cells,s.trunk_bits,bytes)||bytes%8||
           s.an>bytes/8||s.bn>bytes/8)return false;
        if(s.middle){
            if(s.cyclic||!s.middle.first||s.middle.first>=s.middle.end||s.middle.end>words||s.middle.end>bytes/8||
               bytes/8<words-s.middle.first+1)return false;
        }else if(bytes/8<s.cyclic.minimum_words||!s.cyclic.bound.verify||!s.cyclic.bound.high_exclusive||
                 bytes/8<2||s.cyclic.bound.words>bytes/8-2)return false;
        p.output_bytes=bytes;rows=s.outer_cells;
    }else if(!rows||rows>s.outer_cells)return false;
    p.stored_output_bytes=p.max_stored_output_bytes=p.output_bytes;
    for(unsigned i=0;i<2;++i){ForwardPlanesSpec f{i?s.bn:s.an,s.outer_cells,s.inner_cells,rows,
            s.forward_stripe_cells?s.forward_stripe_cells:s.stripe_cells,s.trunk_bits,s.primes,s.prime_batch,s.workers};
        f.source=i?s.b_source:s.a_source;f.ring_checksum=bool(s.middle);
        if(!query_forward_planes(f,p.forward[i]))return false;}
    p.spec.a_source=p.forward[0].spec.source;p.spec.b_source=p.forward[1].spec.source;
    if(s.square&&p.spec.a_source!=p.spec.b_source)return false;
    p.spec.stripe_cells=s.stripe_cells?s.stripe_cells:p.forward[0].spec.stripe_cells;
    p.spec.forward_stripe_cells=p.forward[0].spec.stripe_cells;
    size_t raw_stripe,packed_stripe;
    if(!mul_size(p.spec.stripe_cells,s.trunk_bits,raw_stripe)||!mul_size(p.spec.stripe_cells,48,packed_stripe)||
       (p.spec.stripe_cells<s.inner_cells&&(raw_stripe%chunk_bytes||packed_stripe%chunk_bytes)))return false;
    auto inverse_spec=p.forward[0].kernel;inverse_spec.stripe_cells=s.inverse_stripe_cells?s.inverse_stripe_cells:p.spec.stripe_cells;
    if(inverse_spec.stripe_cells<p.spec.stripe_cells||inverse_spec.stripe_cells%p.spec.stripe_cells)return false;
    const unsigned log=unsigned(__builtin_ctzll(s.outer_cells))+unsigned(__builtin_ctzll(s.inner_cells));
    if(!query_middle_rows({0,log,s.workers,s.inner_cells,0,rows,0,p.forward[0].row_stride,1,s.square},p.middle)||
       !query_inverse_planes(inverse_spec,p.inverse)||
       !query_integer_emit(s.primes,s.trunk_bits,rows,s.inner_cells,inverse_spec.stripe_cells,p.output_bytes,s.workers,p.emit,s.cyclic,0,true,s.middle,words>p.output_bytes/8?words-size_t(p.output_bytes/8):s.middle.first))return false;
    p.spec.cyclic=p.emit.cyclic;
    p.emit.preconditioned=true;
    for(unsigned q=0;q<s.primes;++q)p.product_scale[q]=crt->input_scale(q);
    p.shared_bytes=p.forward[0].info.shared_bytes;
    p.plane_count=s.primes*(s.square?1:2);p.plane_bytes=p.forward[0].plane_bytes;
    uint64_t family;if(__builtin_mul_overflow(uint64_t(s.primes),p.plane_bytes,&family)||s.retained_bytes%chunk_bytes||s.retained_bytes>family)return false;
    RegionLayout plane{},output{};
    if(region_layout(p.plane_bytes,0,plane)||region_layout(p.output_bytes,0,output))return false;
    p.plane_allocation_bytes=plane.allocation_bytes;
    ChunkWriterStorage pi{},oi{};if(chunk_writer_storage(plane,pi)||chunk_writer_storage(output,oi))return false;
    p.plane_index_stride=pi.index_bytes;if(!align_size(pi.coverage_bytes,64,p.plane_bits_stride))return false;
    p.output_index_bytes=oi.index_bytes;p.output_bits_bytes=oi.coverage_bytes;
    if(!layout_storage(p))return false;
    uint64_t input_read=p.forward[0].source_read_bytes,output_write,planes;
    if(!payload(output,output_write)||
       (!s.square&&__builtin_add_overflow(input_read,p.forward[1].source_read_bytes,&input_read))||
       __builtin_mul_overflow(uint64_t(s.primes),p.plane_bytes,&planes))return false;
    const uint64_t passes=s.square?2:3;
    uint64_t traffic;if(__builtin_mul_overflow(planes,passes,&traffic)||__builtin_add_overflow(input_read,traffic,&p.normal_read_bytes)||
       __builtin_add_overflow(traffic,output_write,&p.normal_write_bytes))return false;
    if((s.inner_cells*48)%chunk_bytes){uint64_t guards;
        if(__builtin_mul_overflow(uint64_t(s.primes),uint64_t(rows),&guards)||__builtin_mul_overflow(guards,uint64_t(chunk_bytes),&guards)||
           __builtin_add_overflow(p.normal_read_bytes,guards,&p.normal_read_bytes))return false;}
    p.normal_read_bytes-=2*uint64_t(s.retained_bytes);p.normal_write_bytes-=2*uint64_t(s.retained_bytes);
    if(__builtin_add_overflow(p.normal_read_bytes,p.emit.subtract_read_bytes,&p.normal_read_bytes))return false;
    if((s.inner_cells*48)%chunk_bytes)for(unsigned q=0;q<s.primes;++q){
        const uint64_t start=q*p.plane_bytes;
        const uint64_t held=start<s.retained_bytes?std::min(p.plane_bytes,uint64_t(s.retained_bytes)-start):0;
        p.normal_read_bytes-=(held/p.forward[0].row_stride)*chunk_bytes;
    }
    if(s.raw_operand<2&&!use_raw_operand(p,s.raw_operand))return false;
    if(s.duplex_middle){p.spec.duplex_middle=true;if(!layout_storage(p))return false;}
    out=p;return true;
}
bool use_raw_operand(SpilledProductPlan&p,unsigned operand) noexcept {
    if(operand>1||p.spec.square||p.cached_a||p.cached_square)return false;
    if(p.spec.raw_operand<2)return p.spec.raw_operand==operand;
    auto next=p;uint64_t family,old,reads;
    next.spec.raw_operand=operand;next.plane_count=p.spec.primes;
    next.spec.compact_middle=next.spec.duplex_middle=false;
    if(!layout_storage(next))return false;
    if(__builtin_mul_overflow(uint64_t(p.spec.primes),p.plane_bytes,&family)||
       __builtin_add_overflow(family,p.forward[operand].source_read_bytes,&old)||
       p.normal_read_bytes<old||p.normal_write_bytes<family||
       !raw_reads(next.forward[operand],next.spec.raw_cache_bytes,next.spec.primes,
            next.raw_forward.storage_bytes?next.raw_forward.kernel.stripe_cells:next.spec.inner_cells,reads)||
       __builtin_add_overflow(p.normal_read_bytes-old,reads,&next.normal_read_bytes))return false;
    next.normal_write_bytes-=family;next.raw_read_bytes=reads;p=next;return true;
}
bool cache_raw_prefix(SpilledProductPlan&p,size_t bytes) noexcept {
    if(p.spec.raw_operand>=2)return false;auto next=p;next.spec.raw_cache_bytes=bytes;
    if(!layout_storage(next))return false;uint64_t reads;
    if(!raw_reads(next.forward[next.spec.raw_operand],bytes,next.spec.primes,
        next.raw_forward.storage_bytes?next.raw_forward.kernel.stripe_cells:next.spec.inner_cells,reads)||
        next.normal_read_bytes<next.raw_read_bytes||
        __builtin_add_overflow(next.normal_read_bytes-next.raw_read_bytes,reads,&next.normal_read_bytes))return false;
    next.raw_read_bytes=reads;p=next;return true;
}
bool fit_raw_prefix(SpilledProductPlan&p,size_t limit,uint64_t write_limit) noexcept {
    if(p.spec.raw_operand>=2)return false;auto next=p;
    if(!retain_spilled_prefix(next,0)||!cache_raw_prefix(next,0)||next.storage_bytes>limit)return false;
    const size_t capacity=next.forward[next.spec.raw_operand].input_bytes,T=next.spec.trunk_bits;
    auto grow=[&](){
        const size_t free=limit-next.storage_bytes;
        const uint64_t family=uint64_t(next.spec.primes)*next.plane_bytes;
        const size_t held=size_t(std::min(uint64_t(free/chunk_bytes*chunk_bytes),family));
        const uint64_t writes=next.normal_write_bytes-2*held;
        if(writes>write_limit)return false;
        const size_t unused=free-held;
        const uint64_t paid=std::min(uint64_t(free),((write_limit-writes)/2/chunk_bytes)*chunk_bytes);
        const size_t extra=(paid>=free-unused?free:unused+size_t(paid))/4096*4096;
        size_t allocation;if(!align_size(next.raw_bytes,4096,allocation)||!add_size(allocation,extra,allocation))return false;
        const size_t cached=std::min(capacity,(allocation-64)/T*T)/T*T;
        if(cached<=next.spec.raw_cache_bytes)return true;
        auto grown=next;if(!cache_raw_prefix(grown,cached)||grown.storage_bytes>limit)return false;next=grown;return true;
    };
    // The zero-prefix quote bounds every suffix buffer. The first fill may
    // remove entire staged rows; refill that released capacity once.
    if(!grow()||!grow())return false;
    p=next;return true;
}
bool retain_spilled_prefix(SpilledProductPlan&p,size_t bytes) noexcept {
    uint64_t family;if(__builtin_mul_overflow(uint64_t(p.spec.primes),p.plane_bytes,&family)||bytes%chunk_bytes||bytes>family)return false;
    const size_t old=p.spec.retained_bytes;
    size_t storage,pool,raw,input,inverse;
    if(p.storage_bytes<old||p.pool<old||p.raw_value<old||p.raw_input_index<old||p.inverse_offset<old||
       !add_size(p.storage_bytes-old,bytes,storage)||!add_size(p.pool-old,bytes,pool)||
       !add_size(p.raw_value-old,bytes,raw)||!add_size(p.raw_input_index-old,bytes,input)||
       !add_size(p.inverse_offset-old,bytes,inverse))return false;
    auto guard_saved=[&](size_t retained){
        uint64_t sum=0;
        if(!p.cached_square&&(p.spec.inner_cells*48)%chunk_bytes){
            for(unsigned q=0;q<p.spec.primes;++q){const uint64_t begin=q*p.plane_bytes;
                const uint64_t held=begin<retained?std::min(p.plane_bytes,uint64_t(retained)-begin):0;
                sum+=(held/p.forward[0].row_stride)*chunk_bytes;
            }
        }
        return sum;
    };
    const unsigned passes=p.cached_square?1:2;
    p.normal_read_bytes+=passes*uint64_t(old)+guard_saved(old);p.normal_write_bytes+=passes*uint64_t(old);
    p.normal_read_bytes-=passes*uint64_t(bytes)+guard_saved(bytes);p.normal_write_bytes-=passes*uint64_t(bytes);
    p.storage_bytes=storage;p.pool=pool;p.raw_value=raw;p.raw_input_index=input;p.inverse_offset=inverse;p.spec.retained_bytes=bytes;return true;
}
bool compact_middle_within(SpilledProductPlan&p,size_t available) noexcept {
    if(p.spec.raw_operand<2||p.cached_square)return false;
    auto next=p;if(!retain_spilled_prefix(next,0))return false;next.spec.compact_middle=true;next.spec.duplex_middle=false;
    if(!layout_storage(next)||next.storage_bytes>available)return false;
    const size_t held=size_t(std::min(uint64_t((available-next.storage_bytes)/chunk_bytes*chunk_bytes),uint64_t(next.spec.primes)*next.plane_bytes));
    if(!retain_spilled_prefix(next,held)||next.storage_bytes>available)return false;p=next;return true;
}
bool duplex_middle_within(SpilledProductPlan&p,size_t available) noexcept {
    if(p.cached_square)return false;auto next=p;
    if(next.spec.raw_operand>=2){if(!retain_spilled_prefix(next,0))return false;next.spec.compact_middle=true;}
    next.spec.duplex_middle=true;if(!layout_storage(next)||next.storage_bytes>available)return false;
    if(next.spec.raw_operand>=2){const size_t held=size_t(std::min(uint64_t((available-next.storage_bytes)/chunk_bytes*chunk_bytes),uint64_t(next.spec.primes)*next.plane_bytes));
        if(!retain_spilled_prefix(next,held)||next.storage_bytes>available)return false;}
    p=next;return true;
}
bool stream_final_within(SpilledProductPlan&p,size_t available) noexcept {
    auto next=p;next.spec.stream_final=true;
    if(!layout_storage(next)||next.storage_bytes>available)return false;
    p=next;return true;
}
bool widen_inverse_within(SpilledProductPlan&p,size_t stripe,size_t available) noexcept {
    if(!p.spec.stream_final||stripe<p.inverse.spec.stripe_cells||stripe>=p.spec.inner_cells||
       stripe>SIZE_MAX/48||(stripe*48)%chunk_bytes)return false;
    auto next=p;auto spec=p.inverse.spec;spec.stripe_cells=stripe;
    if(!query_inverse_planes(spec,next.inverse,true))return false;
    next.spec.inverse_stripe_cells=stripe;
    if(!layout_storage(next)||next.storage_bytes>available||next.raw_cached_bytes!=p.raw_cached_bytes||
       next.spec.retained_bytes!=p.spec.retained_bytes)return false;
    if(next.normal_read_bytes<p.emit.subtract_read_bytes||
       __builtin_add_overflow(next.normal_read_bytes-p.emit.subtract_read_bytes,next.emit.subtract_read_bytes,&next.normal_read_bytes)||
       next.normal_read_bytes>p.normal_read_bytes)return false;
    p=next;return true;
}
bool widen_forward_within(SpilledProductPlan&p,size_t stripe,size_t available) noexcept {
    if(stripe<p.forward[0].spec.stripe_cells||stripe<p.forward[1].spec.stripe_cells)return false;
    auto next=p;
    for(unsigned k=0;k<2;++k){auto spec=p.forward[k].spec;spec.stripe_cells=stripe;ForwardPlanesPlan f;
        if(!query_forward_planes(spec,f))return false;f.parallel_writes=p.forward[k].parallel_writes;
        if(!(k==0&&p.cached_a)&&!(k==1&&(p.spec.square||p.cached_square))&&p.spec.raw_operand!=k){
            if(next.normal_read_bytes<p.forward[k].source_read_bytes||
               __builtin_add_overflow(next.normal_read_bytes-p.forward[k].source_read_bytes,f.source_read_bytes,&next.normal_read_bytes))return false;
        }
        next.forward[k]=f;
    }
    next.spec.forward_stripe_cells=stripe;
    if(!layout_storage(next)||next.storage_bytes>available)return false;
    if(next.spec.raw_operand<2){uint64_t reads;
        if(!raw_reads(next.forward[next.spec.raw_operand],next.spec.raw_cache_bytes,next.spec.primes,
            next.raw_forward.storage_bytes?next.raw_forward.kernel.stripe_cells:next.spec.inner_cells,reads)||
           next.normal_read_bytes<next.raw_read_bytes||
           __builtin_add_overflow(next.normal_read_bytes-next.raw_read_bytes,reads,&next.normal_read_bytes))return false;
        next.raw_read_bytes=reads;}
    if(next.normal_read_bytes>p.normal_read_bytes||
       next.normal_write_bytes>p.normal_write_bytes)return false;
    p=next;return true;
}
bool rebind_spilled_sources(SpilledProductPlan&p,SourceWindow a,SourceWindow b) noexcept {
    auto fa=p.forward[0],fb=p.forward[1];
    if(!rebind_forward_source(fa,a)||!rebind_forward_source(fb,b)||(p.spec.square&&fa.spec.source!=fb.spec.source))return false;
    uint64_t raw=p.raw_read_bytes;
    if(p.spec.raw_operand<2&&!raw_reads(p.spec.raw_operand?fb:fa,p.spec.raw_cache_bytes,p.spec.primes,
        p.raw_forward.storage_bytes?p.raw_forward.kernel.stripe_cells:p.spec.inner_cells,raw))return false;
    uint64_t old=p.cached_a?0:p.spec.raw_operand==0?p.raw_read_bytes:p.forward[0].source_read_bytes;
    uint64_t now=p.cached_a?0:p.spec.raw_operand==0?raw:fa.source_read_bytes,total;
    if(!p.spec.square&&!p.cached_square&&(__builtin_add_overflow(old,p.spec.raw_operand==1?p.raw_read_bytes:p.forward[1].source_read_bytes,&old)||
                       __builtin_add_overflow(now,p.spec.raw_operand==1?raw:fb.source_read_bytes,&now)))return false;
    if(p.normal_read_bytes<old||__builtin_add_overflow(p.normal_read_bytes-old,now,&total))return false;
    p.forward[0]=fa;p.forward[1]=fb;p.spec.a_source=fa.spec.source;p.spec.b_source=fb.spec.source;p.normal_read_bytes=total;p.raw_read_bytes=raw;return true;
}
bool rebind_spilled_subtract(SpilledProductPlan&p,SourceWindow source) noexcept {
    auto emit=p.emit;if(!rebind_emit_subtract(emit,source)||p.normal_read_bytes<p.emit.subtract_read_bytes)return false;
    uint64_t reads;if(__builtin_add_overflow(p.normal_read_bytes-p.emit.subtract_read_bytes,emit.subtract_read_bytes,&reads))return false;
    p.emit=emit;p.spec.cyclic=emit.cyclic;p.normal_read_bytes=reads;return true;
}
bool use_cached_a(SpilledProductPlan&p) noexcept {
    if(p.cached_a||p.spec.middle||p.spec.square)return false;
    if(p.spec.raw_operand<2){auto next=p;if(!clear_raw(next)||!use_cached_a(next))return false;p=next;return true;}
    auto next=p;uint64_t family;
    if(__builtin_mul_overflow(uint64_t(p.spec.primes),p.plane_bytes,&family)||p.normal_read_bytes<p.forward[0].source_read_bytes||p.normal_write_bytes<family)return false;
    next.cached_a=true;next.plane_count=p.spec.primes;
    next.normal_read_bytes-=p.forward[0].source_read_bytes;next.normal_write_bytes-=family;
    if(!layout_storage(next))return false;p=next;return true;
}
bool use_completed_cached_a(SpilledProductPlan&p) noexcept {
    if(!p.cached_a||p.spec.square)return false;
    auto spec=p.middle.spec;spec.a_transformed=true;MiddleRowsPlan middle;
    if(!query_middle_rows(spec,middle))return false;
    if(p.middle.pipeline&&!(p.middle.compact?enable_middle_compact_pipeline(middle,p.middle.storage_bytes):
                           enable_middle_pipeline(middle,p.middle.storage_bytes)))return false;
    if(p.middle.duplex&&!enable_middle_duplex(middle,p.middle.storage_bytes))return false;
    if(middle.storage_bytes!=p.middle.storage_bytes)return false;
    p.middle=middle;return true;
}
bool use_cached_square(SpilledProductPlan&p) noexcept {
    if(p.cached_square||p.spec.square||p.spec.an>p.spec.bn||p.spec.cyclic)return false;
    auto next=p;if(!next.cached_a&&!use_cached_a(next))return false;
    auto middle=next.middle.spec;middle.square=middle.cached_square=true;
    if(!query_middle_rows(middle,next.middle))return false;
    uint64_t family,output;RegionLayout layout;
    if(__builtin_mul_overflow(uint64_t(p.spec.primes),p.plane_bytes,&family)||region_layout(p.stored_output_bytes,0,layout)||
       !payload(layout,output)||__builtin_mul_overflow(family,uint64_t(2),&next.normal_read_bytes)||
       __builtin_add_overflow(family,output,&next.normal_write_bytes))return false;
    next.normal_read_bytes-=p.spec.retained_bytes;next.normal_write_bytes-=p.spec.retained_bytes;
    next.cached_square=true;next.spec.compact_middle=next.spec.duplex_middle=false;
    if(!layout_storage(next))return false;p=next;return true;
}
bool project_spilled_output(SpilledProductPlan&p,size_t first,size_t end) noexcept {
    uint64_t begin,stop;
    if(__builtin_mul_overflow(uint64_t(first),uint64_t(8),&begin)||__builtin_mul_overflow(uint64_t(end),uint64_t(8),&stop)||
       begin%chunk_bytes||begin>stop||stop>p.output_bytes||stop-begin>p.max_stored_output_bytes)return false;
    RegionLayout old{},next{};uint64_t a,b;
    if(region_layout(p.stored_output_bytes,0,old)||region_layout(stop-begin,0,next)||!payload(old,a)||!payload(next,b)||
       p.normal_write_bytes<a)return false;
    uint64_t total;if(__builtin_add_overflow(p.normal_write_bytes-a,b,&total))return false;
    p.output_first_byte=begin;p.stored_output_bytes=stop-begin;p.normal_write_bytes=total;return true;
}
bool bound_spilled_output(SpilledProductPlan&p,size_t limbs) noexcept {
    const size_t count=std::min(limbs,size_t(p.output_bytes/8));
    if(!project_spilled_output(p,0,count))return false;
    p.max_stored_output_bytes=p.stored_output_bytes;return true;
}
bool keep_high_spilled_output(SpilledProductPlan&p,size_t keep) noexcept {
    if(!keep||p.spec.cyclic||p.spec.middle)return false;
    const size_t an=p.spec.a_source.limbs,bn=p.spec.b_source.limbs;
    size_t end=0;if(an&&bn&&!add_size(an,bn,end))return false;
    const size_t low=end>keep&&end-keep>1?end-keep-1:0;
    const size_t first=low&~(chunk_bytes/8-1);
    return project_spilled_output(p,first,end);
}
IoError run_spilled_product(const SpilledProductPlan&p,Transport&t,const Region&a,const Region&b,Region&result,
                            std::span<Region>temp,std::span<uint8_t>w,sbn3_team&team,WordExtent *extent,CyclicResidualResult*cyclic,const Region*subtract,MiddleWindowResult*middle) noexcept {
    Region *pointers[20];if(temp.size()>20)return {IoCode::invalid,"spilled region count"};
    for(size_t i=0;i<temp.size();++i)pointers[i]=&temp[i];
    return run_spilled_product_indirect(p,t,a,b,result,{pointers,temp.size()},w,team,extent,cyclic,subtract,middle);
}
static IoError run_impl(const SpilledProductPlan&p,Transport&t,const Region*a,const Region&b,Region&result,
                            std::span<Region *>temp,std::span<uint8_t>w,sbn3_team&team,WordExtent *extent,RowCacheBinding*cache,CyclicResidualResult*cyclic,const Region*subtract,MiddleWindowResult*middle) noexcept {
    if(bool(p.spec.cyclic)!=(cyclic!=nullptr)||bool(p.spec.middle)!=(middle!=nullptr))return {IoCode::invalid,"product/residual result kind"};
    if(p.spec.cyclic.stored_subtract!=bool(subtract)||(subtract&&(subtract->state!=RegionState::sealed||
       subtract->layout.logical_bytes!=p.spec.cyclic.subtract_source.storage_limbs*8||!disjoint(result.layout,subtract->layout))))
        return {IoCode::invalid,"product subtractand binding"};
    if(p.cached_a?(!cache||!row_cache_compatible(*cache,p,t,w)):(cache||!a||a->state!=RegionState::sealed||a->layout.logical_bytes!=p.forward[0].source_bytes))
        return {IoCode::invalid,"spilled A representation"};
    if(!w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes||team.busy||team.width<p.spec.workers||
       b.state!=RegionState::sealed||(!p.cached_square&&b.layout.logical_bytes!=p.forward[1].source_bytes)||
       result.state!=RegionState::building||result.kind!=RegionKind::value||result.layout.logical_bytes!=p.stored_output_bytes||
       temp.size()!=p.plane_count||(p.spec.square && a!=&b)||(!p.cached_a&&!disjoint(result.layout,a->layout))||!disjoint(result.layout,b.layout)||
       (p.cached_a&&(!row_cache_disjoint(*cache,result)||!row_cache_disjoint(*cache,b))))
        return {IoCode::invalid,"spilled product binding",0,p.storage_bytes,w.size()};
    for(size_t i=0;i<temp.size();++i){if(!temp[i])return {IoCode::invalid,"null spilled temporary"};const auto &r=*temp[i];
        if(r.state!=RegionState::building||r.kind!=RegionKind::scratch||r.layout.logical_bytes!=p.plane_bytes||
           (!p.cached_a&&!disjoint(r.layout,a->layout))||!disjoint(r.layout,b.layout)||!disjoint(r.layout,result.layout)||
           (p.cached_a&&!row_cache_disjoint(*cache,r))||(subtract&&!disjoint(r.layout,subtract->layout)))return {IoCode::invalid,"spilled temporary extent"};
        for(size_t j=0;j<i;++j)if(!disjoint(r.layout,temp[j]->layout))return {IoCode::invalid,"spilled temporary alias"};
    }
    TeamWaitScope io_waits(team,10000);
    auto *c=::new(w.data()+p.control)Control{};
    struct End {Control*p;~End(){p->~Control();}} end{c};
    StoreBuffers io{w.subspan(p.io_data,io_bytes),w.subspan(p.io_page,index_page_bytes)};
    auto fail=[&](IoError e){if(result.state!=RegionState::sealed)result.state=RegionState::invalid;return e;};
    const bool raw_mode=p.spec.raw_operand<2;
    const unsigned rhs=p.spec.square||p.cached_a||raw_mode?0:p.spec.primes;
    for(unsigned q=0;q<p.plane_count;++q){size_t offset=0,held=0;
        if(q>=rhs){const uint64_t start=(q-rhs)*p.plane_bytes;
            if(start<p.spec.retained_bytes){offset=size_t(start);held=size_t(std::min(p.plane_bytes,uint64_t(p.spec.retained_bytes)-start));}}
        if(auto e=c->writers[q].open(t,*temp[q],io,w.subspan(p.plane_indexes+q*p.plane_index_stride,p.plane_index_stride),
            w.subspan(p.plane_bits+q*p.plane_bits_stride,p.plane_bits_stride),w.subspan(p.retained+offset,held)))return fail(e);
    }
    if(auto e=c->output.open(t,result,io,w.subspan(p.output_index,p.output_index_bytes),w.subspan(p.output_bits,p.output_bits_bytes),{},extent))return fail(e);
    if(auto e=c->output.project(p.output_bytes,p.output_first_byte))return fail(e);
    auto pool=w.subspan(p.pool,p.pool_bytes);uint64_t residues[2]{};
    const unsigned operands=p.cached_square?1:(p.spec.square?1:2),first_operand=p.cached_a?1:0;
    if(p.cached_a)residues[0]=row_cache_residue(*cache);
    for(unsigned operand=first_operand;operand<operands;++operand){const unsigned slot=(operand-first_operand)*p.spec.primes;
        if(raw_mode&&operand==p.spec.raw_operand)continue;
        const unsigned target=raw_mode?0:slot;
        if(auto e=run_forward_planes(p.forward[operand],t,operand?b:*a,{c->writers+target,p.spec.primes},pool,team,&residues[operand]))return fail(e);
        for(unsigned q=0;q<p.spec.primes;++q){const unsigned index=slot+q;
            const unsigned at=raw_mode?q:index;
            if(auto e=c->writers[at].finish_plane(c->planes[at],w.subspan(p.plane_bits+at*p.plane_bits_stride,p.plane_bits_stride)))return fail(e);}
    }
    std::span<uint8_t>raw_value;ValueReader raw_reader;
    if(raw_mode){
        const unsigned operand=p.spec.raw_operand;const auto&f=p.forward[operand];raw_value=w.subspan(p.raw_value,p.raw_bytes);
        if(auto e=raw_reader.open_pinned(t,operand?b:*a,io,w.subspan(p.raw_input_index,f.input_index_bytes)))return fail(e);
        const size_t words=std::min(p.raw_cached_bytes/8,f.spec.source.limbs);
        if(auto e=read_source_words(raw_reader,f.spec.source,0,words,raw_value))return fail(e);
        std::memset(raw_value.data()+words*8,0,raw_value.size()-words*8);
        residues[operand]=p.spec.middle?m64::words(raw_value.data(),words):m61::words(raw_value.data(),words);
    }
    if(p.spec.square||p.cached_square)residues[1]=residues[0];
    auto cached=p.cached_a?row_cache_planes(*cache):std::span<ScratchPlane>{};
    for(unsigned q=0;q<p.spec.primes;++q){auto plan=p.middle;plan.spec.prime=q;plan.spec.product_scale=p.product_scale[q];
        if(raw_mode){
            if(t.cancel.test())return fail({IoCode::cancelled,"resident operand prime boundary"});
            auto prime=pool.first(size_t(p.plane_bytes)),phase=pool.subspan(p.raw_phase);
            const auto&f=p.forward[p.spec.raw_operand];
            if(p.raw_forward.storage_bytes){uint64_t suffix=0;
                if(auto e=run_forward_prefix_prime(f,p.raw_forward,q,raw_reader,raw_value,prime,phase,team,q?nullptr:&suffix))return fail(e);
                if(!q){auto&value=residues[p.spec.raw_operand];if(p.spec.middle)value=m64::add(value,suffix);else{value+=suffix;if(value>=m61::modulus)value-=m61::modulus;}}
            }else if(auto e=run_forward_memory_prime(f,q,raw_value,prime,phase,team))return fail(e);
            const auto e=plan.recycle_a?run_middle_consuming_a(plan,prime,c->planes[q],phase,team):
                                       run_middle_memory_a(plan,prime,c->planes[q],phase,team);
            if(e)return fail(e);
        }else if(p.cached_square){
            if(auto e=run_middle_cached_square(plan,cached[q],c->writers[q],pool,team))return fail(e);
            if(auto e=c->writers[q].finish_plane(c->planes[q],w.subspan(p.plane_bits+q*p.plane_bits_stride,p.plane_bits_stride)))return fail(e);
        }else if(auto e=run_middle_rows(plan,p.cached_a?cached[q]:c->planes[q],c->planes[rhs+q],pool,team))return fail(e);
        if(!p.spec.square&&!p.cached_a&&!raw_mode)c->planes[q].discard();}
    auto final=w.subspan(p.inverse_offset,p.storage_bytes-p.inverse_offset);
    if(auto e=prepare_integer_emit(p.emit,final.subspan(p.emit_offset,p.emit.storage_bytes),c->output,team,c->emit,subtract))return fail(e);
    if(p.spec.stream_final){if(auto e=run_final_stream(p,{c->planes+rhs,p.spec.primes},c->emit,final,team))return fail(e);}
    else{
        prepare_inverse_planes(p.inverse,final.first(p.inverse.storage_bytes),team,c->inverse);
        for(size_t column=0;column<p.spec.inner_cells;column+=p.inverse.spec.stripe_cells){CoefficientStripe stripe;
            if(auto e=read_inverse_stripe(c->inverse,column,{c->planes+rhs,p.spec.primes},stripe))return fail(e);
            if(auto e=emit_integer_stripe(c->emit,stripe))return fail(e);}
    }
    uint64_t expected=p.spec.middle?m64::multiply(m64::multiply(residues[0],residues[1]),p.emit.scalar):
        m61::multiply(m61::multiply(residues[0],residues[1]),m61::reduce(p.emit.scalar));
    if(p.spec.cyclic){const uint64_t unit=p.spec.cyclic.stored_subtract?c->emit.subtract_residue:uint64_t(1)<<((3*(p.spec.cyclic.subtract_word%61))%61);
        expected=expected>=unit?expected-unit:expected+m61::modulus-unit;}
    return finish_integer_emit(c->emit,expected,cyclic,middle);
}
IoError run_spilled_product_indirect(const SpilledProductPlan&p,Transport&t,const Region&a,const Region&b,Region&r,
    std::span<Region *>temporary,std::span<uint8_t>w,sbn3_team&team,WordExtent*extent,CyclicResidualResult*cyclic,const Region*subtract,MiddleWindowResult*middle) noexcept {
    return run_impl(p,t,&a,b,r,temporary,w,team,extent,nullptr,cyclic,subtract,middle);
}
IoError run_spilled_cached(const SpilledProductPlan&p,Transport&t,RowCacheBinding&cache,const Region&b,Region&r,
    std::span<Region *>temporary,std::span<uint8_t>w,sbn3_team&team,WordExtent*extent,CyclicResidualResult*cyclic,const Region*subtract,MiddleWindowResult*middle) noexcept {
    return run_impl(p,t,nullptr,b,r,temporary,w,team,extent,&cache,cyclic,subtract,middle);
}
} // namespace sbn::v3::external
