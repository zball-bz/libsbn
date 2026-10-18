// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/middle_rows.hpp"
#include "common/checked.hpp"
#include "runtime/team.hpp"
#include "external/two_slot.hpp"
#include <algorithm>
#include <cstring>

namespace sbn::v3::external {
bool query_middle_rows(const MiddleRowsSpec &s, MiddleRowsPlan &out) noexcept {
    if(!s.rows || !s.cells || (s.cells&(s.cells-1)) || s.global_log>32||
       (s.cached_square&&(!s.square||s.byte_offset%chunk_bytes||s.row_stride%chunk_bytes)))return false;
    const unsigned log=unsigned(__builtin_ctzll(s.cells));
    if(log>s.global_log)return false;
    const uint64_t outer=uint64_t(1)<<(s.global_log-log);
    if(s.first_row>=outer || s.rows>outer-s.first_row)return false;
    MiddleRowsPlan p{};p.spec=s;
    product::ModularRowSpec row{product::RowOperation::product,s.prime,s.global_log,s.workers,s.cells,0,1,s.product_scale,
        s.a_transformed,s.cached_square&&s.a_transformed};
    if(!product::p48_row_query(row,p.row))return false;
    uint64_t payload,last;
    if(__builtin_mul_overflow(uint64_t(s.cells),uint64_t(48),&payload) || s.row_stride<payload ||
       __builtin_mul_overflow(s.rows-1,s.row_stride,&last) || __builtin_add_overflow(last,s.byte_offset,&last) ||
       __builtin_add_overflow(last,s.cached_square?s.row_stride:payload,&p.byte_end) || p.byte_end>INT64_MAX)return false;
    size_t cursor=0;
    auto reserve=[&](size_t n,size_t&at,size_t alignment=64){return align_size(cursor,alignment,at)&&add_size(at,n,cursor);};
    if(!reserve(p.row.prepared_bytes,p.prepared)||!reserve(p.row.scratch_bytes,p.scratch)||
       !reserve(s.cached_square?std::max(p.row.array_bytes,size_t(s.row_stride)):p.row.array_bytes,p.b,4096))return false;
    if(s.square)p.a=p.b;else if(!reserve(p.row.array_bytes,p.a,4096))return false;
    if(!align_size(cursor,64,p.storage_bytes))return false;
    p.serial_storage_bytes=p.storage_bytes;
    if(s.workers>1&&s.rows>1){
        row.workers=s.workers-1;
        size_t both;
        if(!product::p48_row_query(row,p.pipeline_row)||!align_size(s.cached_square?size_t(s.row_stride):size_t(payload),4096,p.slot_bytes)||
           !mul_size(p.slot_bytes,s.square?2:4,both)||!align_size(cursor,4096,p.slot_offset)||
           !add_size(p.slot_offset,both,p.pipeline_storage_bytes))return false;
        if(!s.cached_square&&s.byte_offset%chunk_bytes==0&&s.row_stride%chunk_bytes==0){
            if(!reserve(p.row.array_bytes,p.second_b,4096))return false;
            if(s.square)p.second_a=p.second_b;else if(!reserve(p.row.array_bytes,p.second_a,4096))return false;
            if(!align_size(cursor,4096,p.compact_storage_bytes))return false;
            if(s.workers>2&&s.rows>=3&&payload%chunk_bytes==0){
                row.workers=s.workers-2;if(!product::p48_row_query(row,p.duplex_row))return false;
                if(!reserve(p.row.array_bytes,p.third_b,4096))return false;
                if(s.square)p.third_a=p.third_b;else if(!reserve(p.row.array_bytes,p.third_a,4096))return false;
                if(!align_size(cursor,4096,p.duplex_storage_bytes))return false;
            }
        }
    }
    out=p;return true;
}
bool enable_middle_pipeline(MiddleRowsPlan&p,size_t available) noexcept {
    if(!p.pipeline_storage_bytes||p.pipeline_storage_bytes>available)return false;
    p.pipeline=true;p.recycle_a=p.compact=p.duplex=false;p.storage_bytes=p.pipeline_storage_bytes;return true;
}
bool enable_middle_compact_pipeline(MiddleRowsPlan&p,size_t available) noexcept {
    if(!p.compact_storage_bytes||p.compact_storage_bytes>available)return false;
    p.pipeline=p.compact=true;p.recycle_a=p.duplex=false;p.storage_bytes=p.compact_storage_bytes;return true;
}
bool enable_middle_recycled_pipeline(MiddleRowsPlan&p) noexcept {
    if(!p.pipeline_storage_bytes||p.spec.square||p.spec.rows<4||p.spec.byte_offset%4096||p.spec.row_stride%4096)return false;
    p.pipeline=true;p.recycle_a=true;p.compact=p.duplex=false;p.storage_bytes=p.serial_storage_bytes;return true;
}
bool enable_middle_duplex(MiddleRowsPlan&p,size_t available) noexcept {
    if(!p.pipeline||!p.duplex_storage_bytes||(!p.compact&&!p.recycle_a)||(p.recycle_a&&p.spec.rows<6))return false;
    const size_t bytes=p.recycle_a?p.serial_storage_bytes:p.duplex_storage_bytes;
    if(bytes>available)return false;p.duplex=true;p.storage_bytes=bytes;return true;
}
namespace {
uint64_t stamp(const MiddleRowsStats*s){return s?os::now_ns():0;}
struct Call {std::span<const uint8_t>prepared;std::span<uint8_t>scratch;std::span<uint64_t>a,b;};
void execute(void *ptr,sbn3_team_scope *scope){auto &c=*static_cast<Call *>(ptr);product::p48_row_execute(c.prepared,c.scratch,c.a,c.b,scope);}
struct ResidentRows {const MiddleRowsPlan*p;std::span<const uint8_t>a;std::span<uint8_t>workspace;uint64_t rows;};
void resident_rows(void*ptr,std::span<uint8_t>b,sbn3_team_scope*scope){
    auto&c=*static_cast<ResidentRows *>(ptr);const auto&p=*c.p;auto w=c.workspace;
    product::PackedRowBatchSpec s{p.spec.prime,p.spec.global_log,p.spec.workers,p.spec.cells,
        p.spec.first_row,c.rows,p.spec.row_stride,p.spec.product_scale,p.spec.square};
    s.a_transformed=p.spec.a_transformed;s.b_transformed=p.spec.cached_square&&p.spec.a_transformed;
    product::p48_rows_packed(s,p.row,c.a,b,w.subspan(p.prepared,p.row.prepared_bytes),w.subspan(p.scratch,p.row.scratch_bytes),
        {reinterpret_cast<uint64_t *>(w.data()+p.a),p.spec.cells*8},{reinterpret_cast<uint64_t *>(w.data()+p.b),p.spec.cells*8},scope);
}
struct Pipeline {
    const MiddleRowsPlan *p;
    ScratchPlane *a,*b;
    PlaneView source;
    PlaneUpdate *output;
    std::span<const uint8_t> memory_a;
    std::span<uint8_t> work;
    uint64_t first;
    MiddleRowsStats *stats;
    ChunkWriter *writer=nullptr;
    std::span<uint8_t> recycled{};
    TwoSlotExchange exchange{};
    IoError error{};
    std::span<uint8_t> bank(unsigned i,bool a_bank=false){
        const size_t offset=i==2?(a_bank?p->third_a:p->third_b):i?(a_bank?p->second_a:p->second_b):(a_bank?p->a:p->b);
        return work.subspan(offset,p->row.array_bytes);
    }
    std::span<uint8_t> slot(unsigned i,bool a_slot=false){
        if(!recycled.empty())return recycled.subspan(i*p->spec.row_stride,p->spec.cells*48);
        const size_t per=p->spec.square?1:2;
        return work.subspan(p->slot_offset+(i*per+unsigned(a_slot))*p->slot_bytes,
            p->spec.cached_square?size_t(p->spec.row_stride):p->spec.cells*48);
    }
    bool wait(unsigned i,uint32_t desired){
        return exchange.wait(i,TwoSlotExchange::State(desired));
    }
    void publish(unsigned i,uint32_t state){
        exchange.publish(i,TwoSlotExchange::State(state));
    }
    void fail(IoError e){error=e;exchange.stop();}
};
void compute_row(Pipeline&c,uint64_t row,unsigned slot,sbn3_team_scope*scope,const product::ModularRowInfo&info,unsigned workers){
    const auto&p=*c.p;auto w=c.work;
    product::PackedRowBatchSpec spec{p.spec.prime,p.spec.global_log,workers,p.spec.cells,
        p.spec.first_row+row,1,p.spec.cells*48,p.spec.product_scale,p.spec.square};
    spec.a_transformed=p.spec.a_transformed;spec.b_transformed=p.spec.square&&p.spec.a_transformed;
    if(p.compact){
        const unsigned log=p.spec.global_log-unsigned(__builtin_ctzll(p.spec.cells));
        const uint64_t frequency=log?__builtin_bitreverse64(spec.first_row)>>(64-log):0;
        product::ModularRowSpec row_spec{product::RowOperation::product,p.spec.prime,p.spec.global_log,workers,
            p.spec.cells,frequency,product::p48_row_coset(p.spec.prime,p.spec.global_log,frequency),p.spec.product_scale,
            spec.a_transformed,spec.b_transformed};
        auto prep=w.subspan(p.prepared,info.prepared_bytes);product::p48_row_prepare(row_spec,info,prep);
        product::p48_row_blocked_product(prep,w.subspan(p.scratch,info.scratch_bytes),
            {reinterpret_cast<uint64_t *>(c.bank(slot,true).data()),p.spec.cells*8},
            {reinterpret_cast<uint64_t *>(c.bank(slot).data()),p.spec.cells*8},scope);
    }else{
        const auto a=p.spec.square?std::span<const uint8_t>{}:c.a?std::span<const uint8_t>(c.slot(slot,true)):
            c.memory_a.subspan(size_t(p.spec.byte_offset+row*p.spec.row_stride),p.spec.cells*48);
        product::p48_rows_packed(spec,info,a,c.slot(slot),w.subspan(p.prepared,info.prepared_bytes),
            w.subspan(p.scratch,info.scratch_bytes),{reinterpret_cast<uint64_t *>(w.data()+p.a),p.spec.cells*8},
            {reinterpret_cast<uint64_t *>(w.data()+p.b),p.spec.cells*8},scope);
    }
}
IoError write_row(Pipeline&c,uint64_t row,unsigned slot){
    const auto&p=*c.p;const uint64_t at=p.spec.byte_offset+row*p.spec.row_stride;
    if(p.compact){const size_t block=std::min(p.spec.cells,product::p48_row_io_cells);
        for(size_t j=0;j<p.spec.cells;j+=block)if(auto e=c.b->patch(*c.output,at+j*48,c.bank(slot).subspan(j*64,block*48)))return e;
        return {};
    }
    if(c.writer)std::memset(c.slot(slot).data()+p.spec.cells*48,0,size_t(p.spec.row_stride)-p.spec.cells*48);
    return c.writer?c.writer->write(at/chunk_bytes,c.slot(slot)):c.b->patch(*c.output,at,c.slot(slot));
}
void pipeline_io(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<Pipeline *>(ptr);const auto&p=*c.p;
    auto write=[&](uint64_t row,unsigned slot){
        const auto start=stamp(c.stats);const auto e=write_row(c,row,slot);
        if(c.stats)c.stats->write_ns+=stamp(c.stats)-start;
        if(e)c.fail(e);return !e;
    };
    for(uint64_t row=c.first;row<p.spec.rows;++row){const unsigned slot=unsigned((row-c.first)%2);
        if(row>=c.first+2){
            const auto start=stamp(c.stats);if(!c.wait(slot,2))return;
            if(c.stats)c.stats->io_wait_ns+=stamp(c.stats)-start;
            if(!write(row-2,slot))return;
        }
        const auto start=stamp(c.stats);const uint64_t at=p.spec.byte_offset+row*p.spec.row_stride;
        IoError e;
        if(p.compact){const size_t block=std::min(p.spec.cells,product::p48_row_io_cells);
            for(size_t j=0;j<p.spec.cells&&!e;j+=block){
                if(!p.spec.square){auto target=c.bank(slot,true).subspan(j*64,block*48);
                    if(c.a)e=c.a->read(c.source,at+j*48,target);
                    else std::memcpy(target.data(),c.memory_a.data()+at+j*48,target.size());}
                if(!e)e=c.b->read(*c.output,at+j*48,c.bank(slot).subspan(j*64,block*48));
            }
        }else{
            if(!p.spec.square&&c.a)if(auto err=c.a->read(c.source,at,c.slot(slot,true))){c.fail(err);return;}
            e=c.writer?c.b->read(c.source,at,c.slot(slot)):c.b->read(*c.output,at,c.slot(slot));
        }
        if(e){c.fail(e);return;}
        if(c.stats)c.stats->read_ns+=stamp(c.stats)-start;
        c.publish(slot,1);
    }
    for(uint64_t row=std::max(c.first,p.spec.rows-std::min(uint64_t(2),p.spec.rows-c.first));row<p.spec.rows;++row){
        const unsigned slot=unsigned((row-c.first)%2);const auto start=stamp(c.stats);
        if(!c.wait(slot,2))return;
        if(c.stats)c.stats->io_wait_ns+=stamp(c.stats)-start;
        if(!write(row,slot))return;
    }
}
void pipeline_compute(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<Pipeline *>(ptr);const auto&p=*c.p;
    for(uint64_t row=c.first;row<p.spec.rows;++row){const unsigned slot=unsigned((row-c.first)%2);
        auto start=stamp(c.stats);if(!c.wait(slot,1))return;
        if(c.stats)c.stats->compute_wait_ns+=stamp(c.stats)-start;
        start=stamp(c.stats);
        compute_row(c,row,slot,scope,p.pipeline_row,p.spec.workers-1);
        if(c.stats)c.stats->compute_ns+=stamp(c.stats)-start;
        if(c.exchange.stopped())return;
        c.publish(slot,2);
    }
}
void pipeline_action(void*ptr,sbn3_team_scope*scope){
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,1,pipeline_io,ptr,pipeline_compute,ptr);
}
struct Duplex {
    Pipeline*call;PlaneChunkReader a,b;ThreeSlotExchange exchange{};bool error_claimed=false;
    unsigned read_workers=1,compute_workers=0;
    product::ModularRowInfo compute_info{};
    void fail(IoError e){bool expected=false;
        if(__atomic_compare_exchange_n(&error_claimed,&expected,true,false,__ATOMIC_ACQ_REL,__ATOMIC_ACQUIRE))call->error=e;
        exchange.stop();
    }
};
IoError read_input(Duplex&d,uint64_t row,unsigned slot,bool first){
    auto&c=*d.call;const auto&p=*c.p;const uint64_t at=p.spec.byte_offset+row*p.spec.row_stride;
    if(first&&p.spec.square)return {};
    if(p.compact){const size_t block=std::min(p.spec.cells,product::p48_row_io_cells);
        for(size_t j=0;j<p.spec.cells;j+=block){auto target=c.bank(slot,first).subspan(j*64,block*48);
            if(first&&!c.a)std::memcpy(target.data(),c.memory_a.data()+at+j*48,target.size());
            else if(auto e=(first?d.a:d.b).read(at+j*48,target))return e;
        }
        return {};
    }
    return first?IoError{}:d.b.read(at,c.slot(slot));
}
struct ReadInput {Duplex*d;uint64_t row;unsigned slot;bool first;IoError error{};};
void read_input_action(void*ptr,sbn3_team_scope*){auto&r=*static_cast<ReadInput*>(ptr);r.error=read_input(*r.d,r.row,r.slot,r.first);}
void duplex_read(void*ptr,sbn3_team_scope*scope){
    auto&d=*static_cast<Duplex*>(ptr);auto&c=*d.call;const auto&p=*c.p;
    for(uint64_t row=c.first;row<p.spec.rows;++row){const unsigned slot=unsigned((row-c.first)%3);auto start=stamp(c.stats);
        if(!d.exchange.wait(slot,ThreeSlotExchange::empty,ThreeSlotExchange::reader))return;
        if(c.stats)c.stats->read_wait_ns+=stamp(c.stats)-start;start=stamp(c.stats);
        ReadInput a{&d,row,slot,true,{}},b{&d,row,slot,false,{}};
        if(d.read_workers==2)sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,1,read_input_action,&a,read_input_action,&b);
        else{a.error=read_input(d,row,slot,true);if(!a.error)b.error=read_input(d,row,slot,false);}
        const auto e=a.error?a.error:b.error;
        if(c.stats)c.stats->read_ns+=stamp(c.stats)-start;
        if(e){d.fail(e);return;}if(d.exchange.stopped())return;
        d.exchange.publish(slot,ThreeSlotExchange::ready);
    }
}
void duplex_compute(void*ptr,sbn3_team_scope*scope){
    auto&d=*static_cast<Duplex*>(ptr);auto&c=*d.call;const auto&p=*c.p;
    for(uint64_t row=c.first;row<p.spec.rows;++row){const unsigned slot=unsigned((row-c.first)%3);auto start=stamp(c.stats);
        if(!d.exchange.wait(slot,ThreeSlotExchange::ready,ThreeSlotExchange::computer))return;
        if(c.stats)c.stats->compute_wait_ns+=stamp(c.stats)-start;start=stamp(c.stats);
        compute_row(c,row,slot,scope,d.compute_info,d.compute_workers);
        if(c.stats)c.stats->compute_ns+=stamp(c.stats)-start;if(d.exchange.stopped())return;
        d.exchange.publish(slot,ThreeSlotExchange::computed);
    }
}
void duplex_write(void*ptr,sbn3_team_scope*){
    auto&d=*static_cast<Duplex*>(ptr);auto&c=*d.call;const auto&p=*c.p;
    for(uint64_t row=c.first;row<p.spec.rows;++row){const unsigned slot=unsigned((row-c.first)%3);auto start=stamp(c.stats);
        if(!d.exchange.wait(slot,ThreeSlotExchange::computed,ThreeSlotExchange::writer))return;
        if(c.stats)c.stats->write_wait_ns+=stamp(c.stats)-start;start=stamp(c.stats);const auto e=write_row(c,row,slot);
        if(c.stats)c.stats->write_ns+=stamp(c.stats)-start;if(e){d.fail(e);return;}if(d.exchange.stopped())return;
        d.exchange.publish(slot,ThreeSlotExchange::empty);
    }
}
void duplex_consumers(void*ptr,sbn3_team_scope*s){sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,1,duplex_write,ptr,duplex_compute,ptr);}
void duplex_action(void*ptr,sbn3_team_scope*s){auto&d=*static_cast<Duplex*>(ptr);sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,d.read_workers,duplex_read,ptr,duplex_consumers,ptr);}
}
static IoError run_rows(const MiddleRowsPlan &p,ScratchPlane *a,std::span<const uint8_t>memory_a,ScratchPlane &b,
                        std::span<uint8_t>workspace,sbn3_team &team,MiddleRowsStats *stats,std::span<uint8_t>disposable={}) noexcept {
    if(p.spec.cached_square||!workspace.data() || (uintptr_t(workspace.data())&63) || workspace.size()<p.storage_bytes ||
       (p.compact&&(!p.pipeline||p.recycle_a||!p.compact_storage_bytes))||
       (p.duplex&&(!p.pipeline||!p.duplex_storage_bytes||(!p.compact&&!p.recycle_a)))||
       p.recycle_a!=!disposable.empty()||(p.recycle_a&&(!p.pipeline||a||p.spec.square||p.spec.rows<4||
          disposable.data()!=memory_a.data()||disposable.size()!=memory_a.size()||
          (uintptr_t(disposable.data())&4095)||p.spec.byte_offset%4096||p.spec.row_stride%4096))||
       team.busy || team.width<(p.pipeline?p.spec.workers:p.row.workers) || b.logical_bytes()<p.byte_end ||
       (a?(a->logical_bytes()<p.byte_end||(p.spec.square?a!=&b:(a==&b||a->overlaps_backing(b)))):
          (p.spec.square||!memory_a.data()||memory_a.size()<p.byte_end||overlaps(memory_a.data(),memory_a.size(),workspace.data(),workspace.size()))))
        return {IoCode::invalid,"middle row grants",0,p.storage_bytes,workspace.size()};
    PlaneView source{};if(a&&!p.spec.square)if(auto e=a->view(source))return e;
    PlaneUpdate output;if(auto e=b.begin_update(output))return e;
    auto fail=[&](IoError e){b.discard();return e;};
    auto prep=workspace.subspan(p.prepared,p.row.prepared_bytes),scratch=workspace.subspan(p.scratch,p.row.scratch_bytes);
    auto ab=workspace.subspan(p.a,p.row.array_bytes),bb=workspace.subspan(p.b,p.row.array_bytes);
    auto aw=std::span(reinterpret_cast<uint64_t *>(ab.data()),p.spec.cells*8);
    auto bw=std::span(reinterpret_cast<uint64_t *>(bb.data()),p.spec.cells*8);
    const size_t packed=p.spec.cells*48;
    const unsigned outer_log=p.spec.global_log-unsigned(__builtin_ctzll(p.spec.cells));
    uint64_t first=0;
    if((!a||p.spec.square)&&p.spec.byte_offset<=b.resident_bytes()&&packed<=b.resident_bytes()-p.spec.byte_offset){
        first=std::min(p.spec.rows,1+(b.resident_bytes()-p.spec.byte_offset-packed)/p.spec.row_stride);
        const size_t bytes=size_t((first-1)*p.spec.row_stride+packed);
        ResidentRows call{&p,p.spec.square?std::span<const uint8_t>{}:memory_a.subspan(size_t(p.spec.byte_offset),bytes),workspace,first};
        if(auto e=b.edit_resident(output,p.spec.byte_offset,bytes,team,resident_rows,&call))return fail(e);
    }
    const bool pipelined=p.pipeline&&p.spec.rows-first>1;
    const bool duplex=p.duplex&&b.transport()->duplex&&(!a||a->transport()==b.transport());
    if(stats){*stats={};stats->rows=p.spec.rows-first;stats->compute_workers=pipelined?p.pipeline_row.workers:p.row.workers;}
    const auto start=stamp(stats);
    for(uint64_t i=first;i<p.spec.rows;++i){
        const unsigned consumed=duplex?3:2;
        if(p.pipeline&&p.spec.rows-i>1&&(!p.recycle_a||i>=consumed)){
            Pipeline call{&p,a,&b,source,&output,memory_a,workspace,i,stats};
            if(p.recycle_a)call.recycled=disposable.subspan(size_t(p.spec.byte_offset),size_t(consumed-1)*size_t(p.spec.row_stride)+packed);
            if(duplex){
                auto*original=b.transport();Transport reads=*original;reads.write=nullptr;
                Duplex split{&call,{},{}};
                split.read_workers=p.parallel_reads&&p.spec.workers>3&&a&&!p.spec.square&&reads.parallel_reads>=2?2:1;
                split.compute_workers=p.spec.workers-1-split.read_workers;
                product::ModularRowSpec spec{product::RowOperation::product,p.spec.prime,p.spec.global_log,split.compute_workers,
                    p.spec.cells,0,1,p.spec.product_scale,p.spec.a_transformed,p.spec.square&&p.spec.a_transformed};
                require(product::p48_row_query(spec,split.compute_info)&&split.compute_info.scratch_bytes<=p.row.scratch_bytes,
                    SBN3_FATAL_WORKSPACE,"duplex compute partition");
                if(a&&!p.spec.square)if(auto e=a->borrow_chunks(source,reads,split.a))return fail(e);
                if(auto e=b.borrow_chunks(output,reads,split.b))return fail(e);
                if(stats){stats->duplex=true;stats->compute_workers=split.compute_info.workers;}
                sbn3_team_run(&team,duplex_action,&split);
                // The writer issues full replacements only: all stage reads
                // belong to this one read lane, including failed/retried IO.
                original->counters.read_submitted=reads.counters.read_submitted;
                original->counters.read_completed=reads.counters.read_completed;
                original->counters.reads=reads.counters.reads;
                if(stats)stats->io_wait_ns+=stats->read_wait_ns+stats->write_wait_ns;
            }else sbn3_team_run(&team,pipeline_action,&call);
            if(call.error)return fail(call.error);
            break;
        }
        const uint64_t index=p.spec.first_row+i;
        const uint64_t frequency=outer_log?__builtin_bitreverse64(index)>>(64-outer_log):0;
        product::ModularRowSpec s{product::RowOperation::product,p.spec.prime,p.spec.global_log,p.spec.workers,
            p.spec.cells,frequency,product::p48_row_coset(p.spec.prime,p.spec.global_log,frequency),p.spec.product_scale,
            p.spec.a_transformed,p.spec.cached_square&&p.spec.a_transformed};
        auto tick=stamp(stats);product::p48_row_prepare(s,p.row,prep);
        if(stats)stats->compute_ns+=stamp(stats)-tick;
        const uint64_t at=p.spec.byte_offset+i*p.spec.row_stride;
        tick=stamp(stats);
        if(!p.spec.square){
            if(a){if(auto e=a->read(source,at,ab.first(packed)))return fail(e);}
            else std::memcpy(ab.data(),memory_a.data()+at,packed);
        }
        if(auto e=b.read(output,at,bb.first(packed)))return fail(e);
        if(stats)stats->read_ns+=stamp(stats)-tick;
        tick=stamp(stats);
        if(!p.spec.square)product::p48_row_unpack(prep,ab,aw);
        product::p48_row_unpack(prep,bb,bw);
        Call call{prep,scratch,aw,bw};sbn3_team_run(&team,execute,&call);
        product::p48_row_pack(prep,bw,bb);
        if(stats)stats->compute_ns+=stamp(stats)-tick;
        tick=stamp(stats);
        if(auto e=b.patch(output,at,bb.first(packed)))return fail(e);
        if(stats)stats->write_ns+=stamp(stats)-tick;
    }
    if(stats)stats->elapsed_ns=stamp(stats)-start;
    if(auto e=b.finish(output))return fail(e);
    return {};
}
IoError run_middle_rows(const MiddleRowsPlan&p,ScratchPlane&a,ScratchPlane&b,std::span<uint8_t>w,sbn3_team&team,MiddleRowsStats*stats) noexcept {
    return run_rows(p,&a,{},b,w,team,stats);
}
IoError run_middle_memory_a(const MiddleRowsPlan&p,std::span<const uint8_t>a,ScratchPlane&b,std::span<uint8_t>w,sbn3_team&team,MiddleRowsStats*stats) noexcept {
    return run_rows(p,nullptr,a,b,w,team,stats);
}
IoError run_middle_consuming_a(const MiddleRowsPlan&p,std::span<uint8_t>a,ScratchPlane&b,std::span<uint8_t>w,sbn3_team&team,MiddleRowsStats*stats) noexcept {
    return run_rows(p,nullptr,a,b,w,team,stats,a);
}
IoError run_middle_cached_square(const MiddleRowsPlan&p,ScratchPlane&source,ChunkWriter&out,
                                std::span<uint8_t>w,sbn3_team&team,MiddleRowsStats*stats) noexcept {
    if(!p.spec.cached_square||!p.spec.square||p.recycle_a||p.compact||p.duplex||!w.data()||(uintptr_t(w.data())&63)||w.size()<p.storage_bytes||
       team.busy||team.width<(p.pipeline?p.spec.workers:p.row.workers)||source.logical_bytes()<p.byte_end||!out.region()||
       out.logical_bytes()<p.byte_end||out.written_chunks())return {IoCode::invalid,"cached square row grants"};
    PlaneView view;if(auto e=source.view(view))return e;
    if(stats){*stats={};stats->rows=p.spec.rows;stats->compute_workers=p.pipeline?p.pipeline_row.workers:p.row.workers;}
    if(p.pipeline){
        Pipeline call{&p,nullptr,&source,view,nullptr,{},w,0,stats,&out};const auto start=stamp(stats);
        sbn3_team_run(&team,pipeline_action,&call);if(stats)stats->elapsed_ns=stamp(stats)-start;
        return call.error;
    }
    auto prep=w.subspan(p.prepared,p.row.prepared_bytes),scratch=w.subspan(p.scratch,p.row.scratch_bytes);
    auto row=w.subspan(p.b,std::max(p.row.array_bytes,size_t(p.spec.row_stride)));
    auto words=std::span(reinterpret_cast<uint64_t *>(row.data()),p.spec.cells*8);
    const unsigned outer_log=p.spec.global_log-unsigned(__builtin_ctzll(p.spec.cells));
    for(uint64_t i=0;i<p.spec.rows;++i){const uint64_t index=p.spec.first_row+i;
        const uint64_t frequency=outer_log?__builtin_bitreverse64(index)>>(64-outer_log):0;
        product::ModularRowSpec s{product::RowOperation::product,p.spec.prime,p.spec.global_log,p.spec.workers,
            p.spec.cells,frequency,product::p48_row_coset(p.spec.prime,p.spec.global_log,frequency),p.spec.product_scale,
            p.spec.a_transformed,p.spec.a_transformed};
        product::p48_row_prepare(s,p.row,prep);const uint64_t at=p.spec.byte_offset+i*p.spec.row_stride;
        if(auto e=source.read(view,at,row.first(size_t(p.spec.row_stride))))return e;
        product::p48_row_unpack(prep,row,words);Call call{prep,scratch,words,words};sbn3_team_run(&team,execute,&call);
        product::p48_row_pack(prep,words,row);
        std::memset(row.data()+p.spec.cells*48,0,size_t(p.spec.row_stride)-p.spec.cells*48);
        if(auto e=out.write(at/chunk_bytes,row.first(size_t(p.spec.row_stride))))return e;
    }
    return {};
}
} // namespace sbn::v3::external
