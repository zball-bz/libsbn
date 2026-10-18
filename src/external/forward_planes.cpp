// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/forward_planes.hpp"
#include "external/two_slot.hpp"
#include "common/checked.hpp"
#include "common/m61.hpp"
#include "common/m64.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <cstring>
namespace sbn::v3::external {
namespace {
void add_residue(const ForwardPlanesPlan&p,uint64_t&sum,const void*data,size_t n,uint64_t at){
    if(p.spec.ring_checksum)m64::add_at(sum,data,n,at);else m61::add_at(sum,data,n,at);
}
bool disjoint(const RegionLayout&a,const RegionLayout&b){return a.data_offset<=b.data_offset?a.allocation_bytes<=b.data_offset-a.data_offset:b.allocation_bytes<=a.data_offset-b.data_offset;}
struct Call {const ForwardPlanesPlan*p;std::span<uint8_t>ws;uint64_t column;unsigned first,count;};
void transform(void *ptr,sbn3_team_scope*scope){auto &a=*static_cast<Call *>(ptr);const auto &p=*a.p;
    product::p48_outer_forward(a.ws.subspan(p.prepared,p.info.prepared_bytes),a.column,a.first,a.count,
        a.ws.subspan(p.raw,p.info.raw_bytes),a.ws.subspan(p.packed,p.info.packed_bytes),a.ws.subspan(p.work,p.info.workspace_bytes),scope);}
struct MemoryCall {const ForwardPlanesPlan*p;std::span<const uint8_t>source;std::span<uint8_t>plane,work;uint64_t column;unsigned prime;};
void memory_transform(void*ptr,sbn3_team_scope*scope){auto&a=*static_cast<MemoryCall *>(ptr);const auto&p=*a.p;
    product::p48_outer_forward_resident(a.work.subspan(p.prepared,p.info.prepared_bytes),a.column,a.prime,a.source,a.plane,
        size_t(p.row_stride),a.work.subspan(p.work,p.info.workspace_bytes),scope);}
IoError read_stripe(const ForwardPlanesPlan&p,ValueReader&reader,uint64_t column,std::span<uint8_t>raw,uint64_t*residue){
    std::memset(raw.data(),0,raw.size());
    if(p.spec.stripe_cells==p.spec.inner_cells){
        if(auto e=read_source_words(reader,p.spec.source,0,size_t(p.valid_input_bytes/8),raw))return e;
        if(residue)add_residue(p,*residue,raw.data(),size_t(p.valid_input_bytes/8),0);
    }else for(uint64_t r=0;r<p.spec.outer_cells;++r){
        const uint64_t at=(r*p.spec.inner_cells+column)*p.spec.trunk_bits;
        if(at>=p.valid_input_bytes)continue;
        const size_t bytes=size_t(std::min(p.valid_input_bytes-at,uint64_t(p.spec.stripe_cells)*p.spec.trunk_bits));
        if(auto e=read_source_words(reader,p.spec.source,size_t(at/8),bytes/8,raw.subspan(size_t(r)*p.info.raw_stride,bytes+16)))return e;
        if(residue)add_residue(p,*residue,raw.data()+size_t(r)*p.info.raw_stride,bytes/8,at/8);
    }
    return {};
}
struct WriteBatch {const ForwardPlanesPlan*p;std::span<ChunkWriter>outputs;uint64_t column;unsigned first;std::span<const uint8_t>bank;IoError errors[10]{};};
void write_primes(void*ptr,uint64_t lo,uint64_t hi,unsigned){
    auto&c=*static_cast<WriteBatch*>(ptr);const auto&p=*c.p;
    for(unsigned q=unsigned(lo);q<hi;++q)for(uint64_t r=0;r<p.spec.live_rows;++r){
        const uint64_t at=r*p.row_stride+c.column*48;
        const auto payload=c.bank.subspan((size_t(q)*p.spec.live_rows+size_t(r))*p.info.fragment_bytes,p.info.fragment_bytes);
        if(auto e=c.outputs[c.first+q].write(at/chunk_bytes,payload)){c.errors[q]=e;break;}
    }
}
IoError write_batch(const ForwardPlanesPlan&p,std::span<ChunkWriter>outputs,uint64_t column,unsigned first,unsigned count,std::span<const uint8_t>bank,sbn3_team_scope*scope=nullptr){
    // Independent prime writers include checksum/copy work. A batch provider
    // does not make that CPU work parallel; retain an available writer scope.
    if(outputs[first].transport()->write_batch&&(!scope||sbn3_team_width(scope)==1||count==1)){
        ChunkWrite ranges[io_batch_max];size_t used=0;
        // Alternate independent prime files within a bounded submission. A
        // prime-major group can contain only one inode and serialize fallback
        // filesystem work even when several independent outputs are ready.
        for(size_t r=0;r<p.spec.live_rows;++r)for(unsigned q=0;q<count;++q){
            ranges[used++]={&outputs[first+q],(r*p.row_stride+column*48)/chunk_bytes,
                bank.subspan((size_t(q)*p.spec.live_rows+r)*p.info.fragment_bytes,p.info.fragment_bytes)};
            if(used==io_batch_max){if(auto e=ChunkWriter::write_many({ranges,used}))return e;used=0;}
        }
        return used?ChunkWriter::write_many({ranges,used}):IoError{};
    }
    WriteBatch c{&p,outputs,column,first,bank,{}};
    if(scope&&sbn3_team_width(scope)>1&&count>1)sbn3_team_for(scope,0,count,1,SBN3_STATIC,write_primes,&c);
    else write_primes(&c,0,count,0);
    for(unsigned q=0;q<count;++q)if(c.errors[q])return c.errors[q];
    return {};
}
struct ForwardPipeline {
    const ForwardPlanesPlan*p;
    Transport*t;
    ValueReader*reader;
    std::span<ChunkWriter>outputs;
    std::span<uint8_t>work;
    uint64_t*residue;
    uint64_t waves,jobs;
    unsigned io_workers=1;
    TwoSlotExchange exchange{};
    IoError error{};
    std::span<uint8_t>raw(uint64_t stripe){return work.subspan(p->pipeline_double_raw&&stripe%2?p->second_raw:p->raw,p->info.raw_bytes);}
    std::span<uint8_t>packed(unsigned i){return work.subspan(i?p->second_packed:p->packed,p->info.packed_bytes);}
    void fail(IoError e){error=e;exchange.stop();}
};
void forward_io(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<ForwardPipeline *>(ptr);const auto&p=*c.p;
    auto write=[&](uint64_t job){
        const unsigned first=unsigned(job%c.waves)*p.spec.prime_batch,count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
        const auto e=write_batch(p,c.outputs,(job/c.waves)*p.spec.stripe_cells,first,count,c.packed(unsigned(job%2)),scope);
        if(e)c.fail(e);return !e;
    };
    for(uint64_t job=0;job<c.jobs;++job){const unsigned slot=unsigned(job%2);
        if(job>=2){if(!c.exchange.wait(slot,TwoSlotExchange::computed)||!write(job-2))return;}
        if(c.t->cancel.test()){c.fail({IoCode::cancelled,"forward pipeline cancellation"});return;}
        const uint64_t stripe=job/c.waves;
        if(job%c.waves==0){
            if(job&&!p.pipeline_double_raw&&!c.exchange.wait(unsigned((job-1)%2),TwoSlotExchange::computed))return;
            // Reusing raw stripe s-2 is safe: completing job j-2 implies its
            // final prime batch is done, even when there is only one wave.
            if(auto e=read_stripe(p,*c.reader,stripe*p.spec.stripe_cells,c.raw(stripe),c.residue)){c.fail(e);return;}
        }
        if(p.info.fragment_bytes>p.spec.stripe_cells*48)std::memset(c.packed(slot).data(),0,p.info.packed_bytes);
        c.exchange.publish(slot,TwoSlotExchange::ready);
    }
    for(uint64_t job=c.jobs-std::min(uint64_t(2),c.jobs);job<c.jobs;++job)
        if(!c.exchange.wait(unsigned(job%2),TwoSlotExchange::computed)||!write(job))return;
}
void forward_compute(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<ForwardPipeline *>(ptr);const auto&p=*c.p;
    for(uint64_t job=0;job<c.jobs;++job){const unsigned slot=unsigned(job%2);
        if(!c.exchange.wait(slot,TwoSlotExchange::ready))return;
        const unsigned first=unsigned(job%c.waves)*p.spec.prime_batch,count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
        const uint64_t stripe=job/c.waves;
        product::p48_outer_forward(c.work.subspan(p.prepared,p.info.prepared_bytes),stripe*p.spec.stripe_cells,first,count,
            c.raw(stripe),c.packed(slot),c.work.subspan(p.work,p.info.workspace_bytes),scope);
        if(c.exchange.stopped())return;
        c.exchange.publish(slot,TwoSlotExchange::computed);
    }
}
void forward_pipeline(void*ptr,sbn3_team_scope*scope){
    auto&c=*static_cast<ForwardPipeline*>(ptr);
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,c.io_workers,forward_io,ptr,forward_compute,ptr);
}
}
bool rebind_forward_source(ForwardPlanesPlan&p,SourceWindow source) noexcept {
    if(!resolve_source_window(p.spec.input_limbs,source))return false;
    uint64_t storage,offset,valid;
    if(__builtin_mul_overflow(uint64_t(source.storage_limbs),uint64_t(8),&storage)||
       __builtin_mul_overflow(uint64_t(source.first_limb),uint64_t(8),&offset)||
       __builtin_mul_overflow(uint64_t(source.limbs),uint64_t(8),&valid))return false;
    RegionLayout backing{};if(storage>p.source_capacity_bytes||region_layout(storage,0,backing)||backing.index_bytes>p.input_index_bytes)return false;
    uint64_t reads=backing.index_bytes;
    if(valid){
        size_t begin,end;source_word_range(source,0,source.limbs,begin,end);
        const uint64_t skip=(uint64_t(begin)*8)%chunk_bytes,cover=skip+(end-begin)*8;
        const uint64_t chunks=begin<end?cover/chunk_bytes+(cover%chunk_bytes!=0):0;
        if(__builtin_mul_overflow(chunks,uint64_t(chunk_bytes),&reads)||
           __builtin_add_overflow(reads,backing.index_bytes,&reads))return false;
        const __int128 physical=__int128(source.first_limb)-source.low_zero_words;
        if((source.bit_shift||physical%(chunk_bytes/8)) && p.spec.stripe_cells<p.spec.inner_cells){
            uint64_t stripe,extra;
            if(__builtin_mul_overflow(uint64_t(p.spec.stripe_cells),uint64_t(p.spec.trunk_bits),&stripe))return false;
            // Only a cut touched by BOTH physical reads duplicates a chunk.
            // Leading virtual zero stripes do not read or duplicate anything.
            const __int128 lo=__int128(source.lower_limb)+(source.bit_shift>=0),
                           hi=__int128(source.upper_limb)-(source.bit_shift<=0);
            uint64_t cuts=0;
            if(hi>=physical&&hi>=lo){const __int128 step=stripe/8;
                const __int128 first=std::max(__int128(1),lo>physical?(lo-physical+step-1)/step:__int128(0));
                const __int128 last=std::min(__int128((source.limbs-1)/(stripe/8)),(hi-physical)/step);
                if(first<=last)cuts=uint64_t(last-first+1);
            }
            if(__builtin_mul_overflow(cuts,uint64_t(chunk_bytes),&extra)||__builtin_add_overflow(reads,extra,&reads))return false;
        }
    }
    p.spec.source=source;p.source_bytes=storage;p.source_offset=offset;p.valid_input_bytes=valid;p.source_read_bytes=reads;
    return true;
}
bool query_forward_planes(const ForwardPlanesSpec&s,ForwardPlanesPlan&out) noexcept {
    ForwardPlanesPlan p{};p.spec=s;
    if(!s.inner_cells || (s.inner_cells&(s.inner_cells-1)) || s.trunk_bits<64 || s.trunk_bits>232 || s.trunk_bits%8)return false;
    if(!mul_size(s.input_limbs,8,p.input_bytes))return false;
    // Simultaneously align input raw stripes and packed output stripes to Q.
    const size_t need=std::max(size_t(1024),chunk_bytes/(size_t(1)<<__builtin_ctz(s.trunk_bits)));
    if(!p.spec.stripe_cells)p.spec.stripe_cells=std::min(s.inner_cells,need);
    size_t input_chunk,output_chunk;
    if(!mul_size(p.spec.stripe_cells,s.trunk_bits,input_chunk)||!mul_size(p.spec.stripe_cells,48,output_chunk))return false;
    if(p.spec.stripe_cells<s.inner_cells && (input_chunk%chunk_bytes || output_chunk%chunk_bytes))return false;
    const uint64_t cells=p.input_bytes/s.trunk_bits+(p.input_bytes%s.trunk_bits!=0);
    p.kernel={s.outer_cells,s.inner_cells,s.live_rows,p.spec.stripe_cells,cells,s.trunk_bits,s.prime_count,s.prime_batch,s.workers};
    if(!product::p48_outer_query(p.kernel,p.info))return false;
    uint64_t row;
    if(__builtin_mul_overflow(uint64_t(s.inner_cells),uint64_t(48),&row)||!align_size(row,chunk_bytes,p.row_stride)||
       __builtin_mul_overflow(p.row_stride,uint64_t(s.live_rows),&p.plane_bytes)||p.plane_bytes>INT64_MAX)return false;
    auto source=s.source;if(!resolve_source_window(s.input_limbs,source))return false;
    uint64_t source_bytes;if(__builtin_mul_overflow(uint64_t(source.storage_limbs),uint64_t(8),&source_bytes))return false;
    RegionLayout input{};if(region_layout(source_bytes,0,input))return false;p.input_index_bytes=size_t(input.index_bytes);
    p.source_capacity_bytes=source_bytes;
    if(!rebind_forward_source(p,source))return false;
    p.transfer_bytes=1u<<20;
    size_t cursor=0;auto put=[&](size_t bytes,size_t alignment,size_t&offset){return align_size(cursor,alignment,offset)&&add_size(offset,bytes,cursor);};
    if(!put(p.info.prepared_bytes,64,p.prepared)||!put(p.info.workspace_bytes,64,p.work)||!put(p.info.raw_bytes,4096,p.raw)||
       !put(p.info.packed_bytes,4096,p.packed)||!put(p.transfer_bytes,4096,p.transfer)||!put(index_page_bytes,4096,p.index_page)||
       !put(p.input_index_bytes,4096,p.input_index)||!align_size(cursor,4096,p.storage_bytes))return false;
    p.serial_storage_bytes=p.storage_bytes;
    const unsigned waves=(s.prime_count+s.prime_batch-1)/s.prime_batch;
    if(s.workers>1&&(waves>1||p.spec.stripe_cells<s.inner_cells)){
        if(!put(p.info.packed_bytes,4096,p.second_packed)||!align_size(cursor,4096,p.single_raw_pipeline_bytes))return false;
        if(p.spec.stripe_cells<s.inner_cells){if(!put(p.info.raw_bytes,4096,p.second_raw))return false;}
        else p.second_raw=p.raw;
        if(!align_size(cursor,4096,p.pipeline_storage_bytes))return false;
    }
    out=p;return true;
}
bool enable_forward_pipeline(ForwardPlanesPlan&p,size_t available) noexcept {
    if(!p.single_raw_pipeline_bytes||p.single_raw_pipeline_bytes>available)return false;
    p.pipeline_double_raw=p.second_raw!=p.raw&&p.pipeline_storage_bytes<=available;
    p.pipeline=true;p.storage_bytes=p.pipeline_double_raw?p.pipeline_storage_bytes:p.single_raw_pipeline_bytes;return true;
}
IoError run_forward_planes(const ForwardPlanesPlan&p,Transport&t,const Region&input,std::span<ChunkWriter>outputs,
                           std::span<uint8_t>ws,sbn3_team&team,uint64_t *input_m61) noexcept {
    if(input_m61)*input_m61=0;
    if(!ws.data()||(uintptr_t(ws.data())&4095)||ws.size()<p.storage_bytes||input.state!=RegionState::sealed||
       input.layout.logical_bytes!=p.source_bytes||outputs.size()!=p.spec.prime_count||team.busy||team.width<(p.pipeline?p.spec.workers:p.info.workers))
        return {IoCode::invalid,"forward plane binding",0,p.storage_bytes,ws.size()};
    for(size_t i=0;i<outputs.size();++i){const Region*r=outputs[i].region();
        if(!r||r->kind!=RegionKind::scratch||r->state!=RegionState::writing||r->layout.logical_bytes!=p.plane_bytes||
           outputs[i].transport()!=&t||outputs[i].written_chunks()||!disjoint(input.layout,r->layout))return {IoCode::invalid,"forward plane output"};
        for(size_t j=0;j<i;++j)if(!disjoint(r->layout,outputs[j].region()->layout))return {IoCode::invalid,"forward plane output alias"};
    }
    auto fail=[&](IoError e){for(auto&w:outputs)w.abandon();return e;};
    StoreBuffers io{ws.subspan(p.transfer,p.transfer_bytes),ws.subspan(p.index_page,index_page_bytes)};
    ValueReader reader;if(auto e=reader.open_pinned(t,input,io,ws.subspan(p.input_index,p.input_index_bytes)))return fail(e);
    uint64_t residue=0;
    if(p.pipeline){
        const unsigned io_workers=p.parallel_writes&&p.spec.workers>2&&
            p.info.fragment_bytes==p.spec.stripe_cells*48&&p.info.fragment_bytes%chunk_bytes==0?
            std::max(1u,std::min({t.parallel_writes,p.spec.prime_batch,p.spec.workers/2})):1;
        auto kernel=p.kernel;kernel.workers-=io_workers;auto info=p.info;
        info.workers=std::min(info.workers,kernel.workers);info.workspace_bytes=info.worker_bytes*info.workers;
        product::p48_outer_prepare(kernel,info,ws.subspan(p.prepared,p.info.prepared_bytes));
        const uint64_t waves=(p.spec.prime_count+p.spec.prime_batch-1)/p.spec.prime_batch;
        ForwardPipeline call{&p,&t,&reader,outputs,ws,input_m61?&residue:nullptr,waves,(p.spec.inner_cells/p.spec.stripe_cells)*waves,io_workers};
        sbn3_team_run(&team,forward_pipeline,&call);if(call.error)return fail(call.error);
        if(input_m61)*input_m61=residue;return {};
    }
    product::p48_outer_prepare(p.kernel,p.info,ws.subspan(p.prepared,p.info.prepared_bytes));
    auto raw=ws.subspan(p.raw,p.info.raw_bytes),bank=ws.subspan(p.packed,p.info.packed_bytes);
    for(uint64_t column=0;column<p.spec.inner_cells;column+=p.spec.stripe_cells){
        if(auto e=read_stripe(p,reader,column,raw,input_m61?&residue:nullptr))return fail(e);
        for(unsigned first=0;first<p.spec.prime_count;first+=p.spec.prime_batch){
            const unsigned count=std::min(p.spec.prime_batch,p.spec.prime_count-first);
            if(t.cancel.test())return fail({IoCode::cancelled,"forward plane cancellation"});
            if(p.info.fragment_bytes>p.spec.stripe_cells*48)std::memset(bank.data(),0,bank.size());
            Call call{&p,ws,column,first,count};sbn3_team_run(&team,transform,&call);
            if(auto e=write_batch(p,outputs,column,first,count,bank))return fail(e);
        }
    }
    if(input_m61)*input_m61=residue;return {};
}
IoError run_forward_memory_prime(const ForwardPlanesPlan&p,unsigned prime,std::span<const uint8_t>source,
    std::span<uint8_t>plane,std::span<uint8_t>ws,sbn3_team&team) noexcept {
    if(prime>=p.spec.prime_count||source.size()<p.kernel.input_cells*p.spec.trunk_bits+64||!source.data()||
       !plane.data()||plane.size()<p.plane_bytes||!ws.data()||(uintptr_t(ws.data())&4095)||
       ws.size()<forward_memory_workspace(p)||team.busy||team.width<p.info.workers||
       overlaps(source.data(),source.size(),plane.data(),plane.size())||
       overlaps(source.data(),source.size(),ws.data(),ws.size())||
       overlaps(plane.data(),plane.size(),ws.data(),ws.size()))return {IoCode::invalid,"resident forward grants"};
    product::p48_outer_prepare(p.kernel,p.info,ws.subspan(p.prepared,p.info.prepared_bytes));
    for(uint64_t column=0;column<p.spec.inner_cells;column+=p.spec.stripe_cells){
        MemoryCall call{&p,source,plane,ws,column,prime};sbn3_team_run(&team,memory_transform,&call);
    }
    return {};
}
bool source_slice_reads(const SourceWindow&source,size_t offset,size_t count,size_t step,uint64_t&bytes) noexcept {
    auto s=source;if(s.limbs==SIZE_MAX||!resolve_source_window(s.limbs,s)||offset>s.limbs||count>s.limbs-offset)return false;
    bytes=0;if(!count)return true;size_t begin,end;source_word_range(s,offset,count,begin,end);if(begin==end)return true;
    constexpr size_t Q=chunk_bytes/8;
    if(step&&step%Q)return false;
    uint64_t chunks=(end-begin+begin%Q+Q-1)/Q;
    const __int128 physical=__int128(s.first_limb)-s.low_zero_words;
    if(step&&(s.bit_shift||physical%Q)){
        const __int128 lo=__int128(s.lower_limb)+(s.bit_shift>=0),hi=__int128(s.upper_limb)-(s.bit_shift<=0);
        if(hi>=physical&&hi>=lo){
            const __int128 first=std::max(__int128(offset/step)+1,lo>physical?(lo-physical+step-1)/step:__int128(0));
            const __int128 last=std::min(__int128((offset+count-1)/step),(hi-physical)/step);
            if(first<=last&&__builtin_add_overflow(chunks,uint64_t(last-first+1),&chunks))return false;
        }
    }
    return !__builtin_mul_overflow(chunks,uint64_t(chunk_bytes),&bytes);
}
bool query_prefix_forward(const ForwardPlanesPlan&f,size_t cached,PrefixForwardPlan&out,size_t max_stripe) noexcept {
    if(cached%f.spec.trunk_bits||cached>f.input_bytes)return false;
    PrefixForwardPlan p;p.kernel=f.kernel;p.cached_bytes=cached;p.kernel.prime_batch=1;
    p.kernel.stripe_cells=std::min(f.spec.stripe_cells,std::max(size_t(1),f.spec.inner_cells/2));
    if(max_stripe)p.kernel.stripe_cells=std::min(p.kernel.stripe_cells,max_stripe);
    // A full-width small transform needs no per-row IO cuts or overlap.
    if(p.kernel.stripe_cells*f.spec.trunk_bits%chunk_bytes||p.kernel.stripe_cells*48%chunk_bytes)p.kernel.stripe_cells=f.spec.inner_cells;
    p.pipeline=f.spec.workers>1&&p.kernel.stripe_cells<p.kernel.inner_cells;
    p.kernel.workers-=unsigned(p.pipeline);
    if(!product::p48_outer_query(p.kernel,p.info))return false;
    p.first_row=(cached/f.spec.trunk_bits)/f.spec.inner_cells;
    size_t skipped;if(!mul_size(p.first_row,p.info.raw_stride,skipped))return false;
    p.raw_bytes=p.info.raw_bytes>skipped?p.info.raw_bytes-skipped:64;
    size_t cursor=0;auto put=[&](size_t n,size_t align,size_t&at){return align_size(cursor,align,at)&&add_size(at,n,cursor);};
    if(!put(p.info.prepared_bytes,64,p.prepared)||!put(p.info.workspace_bytes,64,p.work)||
       !put(p.raw_bytes,4096,p.raw[0])||(p.pipeline&&!put(p.raw_bytes,4096,p.raw[1]))||!align_size(cursor,4096,p.storage_bytes))return false;
    out=p;return true;
}
namespace {
struct PrefixCall {
    const ForwardPlanesPlan*f;const PrefixForwardPlan*p;unsigned prime;ValueReader*reader;
    std::span<const uint8_t>prefix;std::span<uint8_t>plane,work;
    bool checksum;uint64_t residue=0;TwoSlotExchange exchange{};IoError error{};
};
IoError prefix_load(PrefixCall&c,size_t column,unsigned slot){
    const auto&f=*c.f;const auto&p=*c.p;auto raw=c.work.subspan(p.raw[slot],p.raw_bytes);std::memset(raw.data(),0,raw.size());
    if(p.kernel.stripe_cells==p.kernel.inner_cells){
        if(p.cached_bytes<f.valid_input_bytes){const size_t into=p.cached_bytes-p.first_row*p.info.raw_stride,bytes=size_t(f.valid_input_bytes)-p.cached_bytes;
            if(auto e=read_source_words(*c.reader,f.spec.source,p.cached_bytes/8,bytes/8,raw.subspan(into,bytes+16)))return e;
            if(c.checksum)add_residue(f,c.residue,raw.data()+into,bytes/8,p.cached_bytes/8);}
        return {};
    }
    for(size_t row=p.first_row;row<f.spec.outer_cells;++row){
        const size_t at=(row*f.spec.inner_cells+column)*f.spec.trunk_bits;if(at>=f.valid_input_bytes)break;
        const size_t stop=size_t(std::min(f.valid_input_bytes,uint64_t(at+p.kernel.stripe_cells*f.spec.trunk_bits)));
        const size_t begin=std::max(at,p.cached_bytes);if(begin>=stop)continue;
        const size_t into=(row-p.first_row)*p.info.raw_stride+(begin-at),bytes=stop-begin;
        if(auto e=read_source_words(*c.reader,f.spec.source,begin/8,bytes/8,raw.subspan(into,bytes+16)))return e;
        if(c.checksum)add_residue(f,c.residue,raw.data()+into,bytes/8,begin/8);
    }
    return {};
}
void prefix_transform(PrefixCall&c,size_t column,unsigned slot,sbn3_team_scope*scope){
    const auto&p=*c.p;
    product::p48_outer_forward_cached(c.work.subspan(p.prepared,p.info.prepared_bytes),column,c.prime,c.prefix,
        p.cached_bytes/c.f->spec.trunk_bits,c.work.subspan(p.raw[slot],p.raw_bytes),c.plane,size_t(c.f->row_stride),
        c.work.subspan(p.work,p.info.workspace_bytes),scope);
}
void prefix_io(void*ptr,sbn3_team_scope*){
    auto&c=*static_cast<PrefixCall*>(ptr);const auto&p=*c.p;
    for(size_t column=0,job=0;column<p.kernel.inner_cells;column+=p.kernel.stripe_cells,++job){const unsigned slot=unsigned(job%2);
        if(job>=2&&!c.exchange.wait(slot,TwoSlotExchange::computed))return;
        if(auto e=prefix_load(c,column,slot)){c.error=e;c.exchange.stop();return;}
        c.exchange.publish(slot,TwoSlotExchange::ready);
    }
}
void prefix_compute(void*ptr,sbn3_team_scope*s){
    auto&c=*static_cast<PrefixCall*>(ptr);const auto&p=*c.p;
    for(size_t column=0,job=0;column<p.kernel.inner_cells;column+=p.kernel.stripe_cells,++job){const unsigned slot=unsigned(job%2);
        if(!c.exchange.wait(slot,TwoSlotExchange::ready))return;prefix_transform(c,column,slot,s);
        if(c.exchange.stopped())return;c.exchange.publish(slot,TwoSlotExchange::computed);
    }
}
void prefix_action(void*ptr,sbn3_team_scope*s){
    auto&c=*static_cast<PrefixCall*>(ptr);
    if(c.p->pipeline)sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,1,prefix_io,ptr,prefix_compute,ptr);
    else for(size_t column=0;column<c.p->kernel.inner_cells;column+=c.p->kernel.stripe_cells){
        if(auto e=prefix_load(c,column,0)){c.error=e;return;}prefix_transform(c,column,0,s);}
}
}
IoError run_forward_prefix_prime(const ForwardPlanesPlan&f,const PrefixForwardPlan&p,unsigned prime,ValueReader&reader,
    std::span<const uint8_t>prefix,std::span<uint8_t>plane,std::span<uint8_t>w,sbn3_team&team,uint64_t*residue) noexcept {
    if(residue)*residue=0;
    if(prime>=f.spec.prime_count||prefix.size()<p.cached_bytes+64||!prefix.data()||!plane.data()||plane.size()<f.plane_bytes||
       !w.data()||(uintptr_t(w.data())&4095)||w.size()<p.storage_bytes||team.busy||team.width<f.spec.workers||
       overlaps(prefix.data(),prefix.size(),plane.data(),plane.size())||overlaps(prefix.data(),prefix.size(),w.data(),w.size())||
       overlaps(plane.data(),plane.size(),w.data(),w.size()))return {IoCode::invalid,"prefix forward grants"};
    product::p48_outer_prepare(p.kernel,p.info,w.subspan(p.prepared,p.info.prepared_bytes));
    PrefixCall c{&f,&p,prime,&reader,prefix,plane,w,residue!=nullptr};sbn3_team_run(&team,prefix_action,&c);
    if(!c.error&&residue)*residue=c.residue;return c.error;
}
} // namespace sbn::v3::external
