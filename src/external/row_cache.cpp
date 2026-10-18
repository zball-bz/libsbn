// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/row_cache.hpp"
#include "external/spilled_product.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
#include <new>
namespace sbn::v3::external {
namespace {
constexpr size_t io_bytes=1u<<20;
struct Control {
    ForwardPlanesPlan forward{};
    Transport *transport=nullptr;
    std::span<uint8_t> storage{};
    RegionId source_id{};
    uint64_t source_revision=0,residue=0;
    RowCacheKind kind=RowCacheKind::outer_rows;
    RegionLease regions[10];
    ChunkWriter writers[10];
    ScratchPlane planes[10];
    bool ready=false;
    ~Control(){for(auto&p:planes)p.discard();for(auto&w:writers)w.abandon();}
};
const Control *control(const RowCacheBinding&b){return static_cast<const Control *>(b.control);}
struct ColumnCall {const RowCachePlan*p;unsigned prime;std::span<uint8_t>data,work;};
void columns(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<ColumnCall *>(ptr);const auto&p=*c.p;const auto&s=p.forward.spec;auto w=c.work;
    product::PackedRowBatchSpec batch{c.prime,unsigned(__builtin_ctzll(s.outer_cells))+unsigned(__builtin_ctzll(s.inner_cells)),
        s.workers,s.inner_cells,0,s.live_rows,p.forward.row_stride,1,false};batch.operation=product::RowOperation::forward;
    product::p48_rows_packed(batch,p.column,{},c.data,w.subspan(p.row_prepared,p.column.prepared_bytes),
        w.subspan(p.row_scratch,p.column.scratch_bytes),{},
        {reinterpret_cast<uint64_t *>(w.data()+p.row_array),s.inner_cells*8},scope);
}
}
bool query_row_cache(const ForwardPlanesPlan &forward,RowCachePlan &out,RowCacheKind kind) noexcept {
    if(!forward.spec.prime_count||forward.spec.prime_count>10||!forward.storage_bytes)return false;
    if(kind!=RowCacheKind::outer_rows&&kind!=RowCacheKind::completed_columns)return false;
    RowCachePlan p{};p.forward=forward;p.kind=kind;p.build_bytes=forward.storage_bytes;RegionLayout l{};
    if(region_layout(forward.plane_bytes,0,l))return false;
    ChunkWriterStorage meta{};if(chunk_writer_storage(l,meta))return false;
    p.index_stride=meta.index_bytes;if(!align_size(meta.coverage_bytes,64,p.bits_stride))return false;
    const size_t np=forward.spec.prime_count;size_t at=0,n;
    auto put=[&](size_t bytes,size_t alignment,size_t &offset){return align_size(at,alignment,offset)&&add_size(offset,bytes,at);};
    if(!put(sizeof(Control),64,p.control)||!mul_size(np,p.index_stride,n)||!put(n,4096,p.indexes)||
       !mul_size(np,p.bits_stride,n)||!put(n,64,p.bits)||!put(io_bytes,4096,p.io_data)||!put(index_page_bytes,4096,p.io_page)||
       !align_size(at,4096,p.storage_bytes)||__builtin_mul_overflow(l.allocation_bytes,uint64_t(np),&p.owned_disk_bytes)||
       __builtin_mul_overflow(forward.plane_bytes,uint64_t(np),&p.normal_write_bytes))return false;
    p.normal_read_bytes=forward.source_read_bytes;
    if(kind==RowCacheKind::completed_columns){
        const auto&s=forward.spec;product::ModularRowSpec row{product::RowOperation::forward,0,
            unsigned(__builtin_ctzll(s.outer_cells))+unsigned(__builtin_ctzll(s.inner_cells)),s.workers,s.inner_cells,0,1,1};
        if(!product::p48_row_query(row,p.column))return false;
        at=0;
        if(!put(p.column.prepared_bytes,64,p.row_prepared)||!put(p.column.scratch_bytes,64,p.row_scratch)||
           !put(p.column.array_bytes,4096,p.row_array))return false;
        p.phase_bytes=std::max(at,forward_memory_workspace(forward));
        size_t bytes;
        if(!mul_size(s.input_limbs,8,bytes)||!add_size(bytes,s.trunk_bits-1,bytes)||
           !mul_size(bytes/s.trunk_bits,s.trunk_bits,p.raw_bytes)||!add_size(p.raw_bytes,64,p.raw_bytes))return false;
        at=0;
        if(!put(p.raw_bytes,4096,p.raw)||!put(forward.input_index_bytes,4096,p.source_index)||
           !put(size_t(forward.plane_bytes),4096,p.prime)||!put(p.phase_bytes,4096,p.phase)||!align_size(at,4096,p.build_bytes))return false;
        auto once=forward;once.spec.stripe_cells=s.inner_cells;
        if(!rebind_forward_source(once,s.source))return false;p.normal_read_bytes=once.source_read_bytes;
    }
    out=p;return true;
}
IoError build_row_cache(const RowCachePlan&p,Transport&t,ValueStore&store,const StoredValue&source,SourceWindow view,
                       std::span<uint8_t>memory,std::span<uint8_t>work,sbn3_team&team,RowCacheBinding&out) noexcept {
    if(out.control||!source||source.store()!=&store||!memory.data()||(uintptr_t(memory.data())&4095)||memory.size()<p.storage_bytes||
       overlaps(memory.data(),memory.size(),work.data(),work.size())||team.busy||team.width<p.forward.info.workers)
        return {IoCode::invalid,"row cache grants",0,p.storage_bytes,memory.size()};
    auto forward=p.forward;
    if(!rebind_forward_source(forward,view)||source.region().layout.logical_bytes!=forward.source_bytes||
       !work.data()||(uintptr_t(work.data())&4095)||work.size()<p.build_bytes)return {IoCode::invalid,"row cache source/work"};
    if(t.counters.write_submitted>t.write_limit||p.normal_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"row cache write allowance",0,p.normal_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    auto *c=::new(memory.data()+p.control)Control{};out.control=c;
    auto fail=[&](IoError e){release_row_cache(out);return e;};
    c->forward=forward;c->transport=&t;c->storage=memory.first(p.storage_bytes);c->source_id=source.region().id;c->source_revision=source.region().revision;c->kind=p.kind;
    StoreBuffers io{memory.subspan(p.io_data,io_bytes),memory.subspan(p.io_page,index_page_bytes)};
    const unsigned np=forward.spec.prime_count;
    for(unsigned q=0;q<np;++q)if(auto e=store.reserve(forward.plane_bytes,RegionKind::scratch,c->regions[q]))return fail(e);
    for(unsigned q=0;q<np;++q)if(auto e=c->writers[q].open(t,c->regions[q].region(),io,
        memory.subspan(p.indexes+q*p.index_stride,p.index_stride),memory.subspan(p.bits+q*p.bits_stride,p.bits_stride)))return fail(e);
    if(p.kind==RowCacheKind::outer_rows){
        if(auto e=run_forward_planes(forward,t,source.region(),{c->writers,np},work,team,&c->residue))return fail(e);
        for(unsigned q=0;q<np;++q)if(auto e=c->writers[q].finish_plane(c->planes[q],memory.subspan(p.bits+q*p.bits_stride,p.bits_stride)))return fail(e);
    }else{
        auto raw=work.subspan(p.raw,p.raw_bytes),prime=work.subspan(p.prime,size_t(forward.plane_bytes)),phase=work.subspan(p.phase,p.phase_bytes);
        ValueReader reader;if(auto e=reader.open_pinned(t,source.region(),io,work.subspan(p.source_index,forward.input_index_bytes)))return fail(e);
        if(auto e=read_source_words(reader,forward.spec.source,0,forward.spec.source.limbs,raw))return fail(e);
        std::memset(raw.data()+forward.valid_input_bytes,0,raw.size()-size_t(forward.valid_input_bytes));
        c->residue=m61::words(raw.data(),forward.spec.source.limbs);
        for(unsigned q=0;q<np;++q){
            if(t.cancel.test())return fail({IoCode::cancelled,"column cache prime boundary"});
            if(forward.row_stride>forward.spec.inner_cells*48)
                for(size_t row=0;row<forward.spec.live_rows;++row)std::memset(prime.data()+row*forward.row_stride+forward.spec.inner_cells*48,0,size_t(forward.row_stride)-forward.spec.inner_cells*48);
            if(auto e=run_forward_memory_prime(forward,q,raw,prime,phase,team))return fail(e);
            ColumnCall call{&p,q,prime,phase};sbn3_team_run(&team,columns,&call);
            if(auto e=c->writers[q].write(0,prime))return fail(e);
            if(auto e=c->writers[q].finish_plane(c->planes[q],memory.subspan(p.bits+q*p.bits_stride,p.bits_stride)))return fail(e);
        }
    }
    c->ready=true;return {};
}
void release_row_cache(RowCacheBinding&b) noexcept {
    if(b.control){static_cast<Control *>(b.control)->~Control();b.control=nullptr;}
}
bool row_cache_compatible(const RowCacheBinding&b,const SpilledProductPlan&p,const Transport&t,std::span<uint8_t>w) noexcept {
    const auto*c=control(b);if(!c||!c->ready||c->transport!=&t||p.spec.square||!p.cached_a)return false;
    if((c->kind==RowCacheKind::completed_columns)!=p.middle.spec.a_transformed)return false;
    if(overlaps(c->storage.data(),c->storage.size(),w.data(),w.size()))return false;
    const auto&a=c->forward.spec,&x=p.forward[0].spec;
    return a.input_limbs==x.input_limbs&&a.outer_cells==x.outer_cells&&a.inner_cells==x.inner_cells&&
        a.live_rows>=x.live_rows&&a.trunk_bits==x.trunk_bits&&a.prime_count==x.prime_count&&a.source==x.source&&
        c->forward.row_stride==p.forward[0].row_stride;
}
bool row_cache_disjoint(const RowCacheBinding&b,const Region&r) noexcept {
    const auto*c=control(b);if(!c||!c->ready)return false;
    for(unsigned q=0;q<c->forward.spec.prime_count;++q){const auto&a=c->regions[q].region().layout;const auto&x=r.layout;
        if(a.data_offset<=x.data_offset?a.allocation_bytes>x.data_offset-a.data_offset:x.allocation_bytes>a.data_offset-x.data_offset)return false;
    }
    return true;
}
SourceWindow row_cache_source(const RowCacheBinding&b) noexcept {
    const auto*c=control(b);require(c&&c->ready,SBN3_FATAL_LIFETIME,"row cache source");return c->forward.spec.source;
}
uint64_t row_cache_residue(const RowCacheBinding&b) noexcept {
    const auto*c=control(b);require(c&&c->ready,SBN3_FATAL_LIFETIME,"row cache residue");return c->residue;
}
std::span<ScratchPlane> row_cache_planes(RowCacheBinding&b) noexcept {
    auto*c=static_cast<Control *>(b.control);require(c&&c->ready,SBN3_FATAL_LIFETIME,"row cache planes");return {c->planes,c->forward.spec.prime_count};
}
} // namespace sbn::v3::external
