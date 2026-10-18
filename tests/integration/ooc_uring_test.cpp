// SPDX-License-Identifier: LGPL-3.0-or-later
#include "file_store_support.hpp"
#include "external/owned_product.hpp"
#include "runtime/linux/uring.hpp"
#include <linux/memfd.h>
#include <sys/syscall.h>
static StoredValue value(FileDevice&d,Transport&t,StoreBuffers io,std::span<uint8_t>bytes){
    RegionLease r;StoredValue out;assert(!d.store.reserve(bytes.size(),RegionKind::value,r));
    ValueWriter w;assert(!w.open(t,r.region(),io));assert(!w.append(bytes));assert(!w.seal());assert(!r.publish(out));return out;
}
static void raw_ring(){
    sbn::v3::os::Uring ring;assert(ring.open(3)==-EINVAL);assert(ring.open(4,0)==-EINVAL);assert(!ring.open(4));assert(ring.mapped_bytes()<=ring.memory_bound(4));
    const int fd=int(syscall(SYS_memfd_create,"sbn-uring-gate",MFD_CLOEXEC));assert(fd>=0);assert(!ftruncate(fd,4*chunk_bytes));
    Aligned input(4*chunk_bytes),output(input.bytes);fill(input.span());std::memset(output.data,0,output.bytes);
    sbn::v3::os::UringRequest requests[4];
    for(unsigned i=0;i<4;++i)requests[i]={fd,input.data+i*chunk_bytes,i*chunk_bytes,unsigned(chunk_bytes),true,0};
    allocation_watch_start();assert(!ring.transfer(requests));assert(!allocation_watch_stop());for(auto&r:requests)assert(r.result==int(chunk_bytes));
    for(unsigned i=0;i<4;++i)requests[i]={fd,output.data+i*chunk_bytes,i*chunk_bytes,unsigned(chunk_bytes),false,0};
    requests[1].fd=-1;requests[2].offset=8*chunk_bytes;
    allocation_watch_start();assert(!ring.transfer(requests));assert(!allocation_watch_stop());
    assert(requests[0].result==int(chunk_bytes)&&requests[1].result==-EBADF&&requests[2].result==0&&requests[3].result==int(chunk_bytes));
    assert(!memcmp(input.data,output.data,chunk_bytes)&&!memcmp(input.data+3*chunk_bytes,output.data+3*chunk_bytes,chunk_bytes));
    for(unsigned i=0;i<4;++i)requests[i]={fd,output.data+i*chunk_bytes,i*chunk_bytes,unsigned(chunk_bytes),false,0};
    assert(!ring.transfer(requests)&&!memcmp(input.data,output.data,input.bytes));
    assert(ring.usage().batches==3&&ring.usage().requests==12);ring.close();assert(!close(fd));
    puts("Raw ring: mapped bound, batch data, EBADF/EOF joined with successful peers, reuse and no hot allocation PASS");
}
static void transport(){
    FileDevice disk(8,8u<<20,true);assert(!disk.files.enable_uring(4));disk.inner=disk.files.transport();auto t=disk.transport();
    assert(t.read_batch&&t.write_batch&&t.batch_limit==4);assert(disk.files.uring_mapped_bytes()<=FileStore::uring_memory_bound(4));
    RegionLease region;assert(!disk.store.reserve(8*chunk_bytes,RegionKind::scratch,region));
    Aligned input(4*chunk_bytes),output(input.bytes),io(4*chunk_bytes),page(index_page_bytes);fill(input.span());
    ReadRequest rr[4];WriteRequest wr[4];
    for(unsigned i=0;i<4;++i){const auto at=region.region().layout.data_offset+i*chunk_bytes;
        wr[i]={at,input.span().subspan(i*chunk_bytes,chunk_bytes),{}};rr[i]={at,output.span().subspan(i*chunk_bytes,chunk_bytes),{}};}
    t.write_limit=input.bytes-1;assert(write_batch_exact(t,wr).code==IoCode::capacity&&!t.counters.write_submitted&&!disk.files.uring_write_usage().requests);
    t.write_limit=input.bytes;allocation_watch_start();assert(!write_batch_exact(t,wr));assert(!read_batch_exact(t,rr));assert(!allocation_watch_stop());
    assert(!memcmp(input.data,output.data,input.bytes)&&t.counters.write_submitted==input.bytes&&t.counters.read_completed==input.bytes);
    rr[1].data=rr[0].data;const auto old=t.counters.read_submitted;assert(read_batch_exact(t,rr).code==IoCode::invalid&&t.counters.read_submitted==old);
    rr[1].data=output.span().subspan(chunk_bytes,chunk_bytes);
    disk.fail_read=disk.reads+2;const auto completed=t.counters.read_completed;
    assert(read_batch_exact(t,rr).code==IoCode::io&&t.counters.read_completed-completed==3*chunk_bytes);disk.fail_read=0;
    disk.cancelled=true;assert(read_batch_exact(t,rr).code==IoCode::cancelled);disk.cancelled=false;
    t.write_limit=UINT64_MAX;disk.cancel_after_write=disk.writes+2;
    assert(write_batch_exact(t,wr).code==IoCode::cancelled);assert(t.counters.write_completed==2*input.bytes);disk.cancelled=false;disk.cancel_after_write=0;
    auto saved=value(disk,t,{io.span(),page.span()},input.span());wr[0].offset=saved.region().layout.data_offset;
    assert(write_batch_exact(t,wr).code==IoCode::invalid&&wr[0].result.system_error==EROFS);
    ValueReader reader;assert(!reader.open(t,saved.region(),{io.span(),page.span()}));assert(!reader.read(0,output.span()));assert(!memcmp(input.data,output.data,input.bytes));
    saved.reset();region.reset();assert(!disk.files.collect());
    Aligned large(3u<<20),large_out(large.bytes);fill(large.span());RegionLease large_region;
    assert(!disk.store.reserve(large.bytes,RegionKind::value,large_region));ChunkWriterStorage spec;assert(!chunk_writer_storage(large_region.region().layout,spec));
    Aligned large_index(spec.index_bytes),large_bits(spec.coverage_bytes);ChunkWriter many;StoredValue published;
    assert(!many.open(t,large_region.region(),{io.span(),page.span()},large_index.span(),large_bits.span()));
    const auto submitted=disk.files.uring_write_usage().requests;
    allocation_watch_start();assert(!many.write(0,large.span()));assert(!many.seal());assert(!allocation_watch_stop());
    assert(disk.files.uring_write_usage().requests-submitted==3);assert(!large_region.publish(published));
    assert(!reader.open(t,published.region(),{io.span(),page.span()}));assert(!reader.read(0,large_out.span()));assert(!memcmp(large.data,large_out.data,large.bytes));
    published.reset();disk.close();puts("Batch transport: cap/alias/error/cancel/readonly gates and multi-MiB value publication PASS");
}
static void integrity(){
    FileDevice disk(8,8u<<20,true);assert(!disk.files.enable_uring(4));disk.inner=disk.files.transport();auto t=disk.transport();
    RegionLease region;assert(!disk.store.reserve(8*chunk_bytes,RegionKind::scratch,region));
    ChunkWriterStorage cs;assert(!chunk_writer_storage(region.region().layout,cs));PlaneIndexStorage ps;assert(!plane_index_storage(region.region().layout,ps));
    Aligned image(cs.index_bytes),coverage(cs.coverage_bytes),dirty(ps.dirty_bytes),held(2*chunk_bytes),io(4*chunk_bytes),page(index_page_bytes),input(8*chunk_bytes),output(input.bytes);fill(input.span());
    ChunkWriter writer;assert(!writer.open(t,region.region(),{io.span(),page.span()},image.span(),coverage.span(),held.span()));
    assert(!writer.write(0,input.span()));ScratchPlane plane;assert(!writer.finish_plane(plane,dirty.span()));PlaneView view;assert(!plane.view(view));
    ReadRange ranges[4];for(unsigned i=0;i<4;++i)ranges[i]={i*2*chunk_bytes,output.span().subspan(i*2*chunk_bytes,2*chunk_bytes)};
    const auto before=t.counters;allocation_watch_start();assert(!plane.read_many(view,ranges));assert(!allocation_watch_stop());
    assert(!memcmp(input.data,output.data,input.bytes)&&t.counters.read_submitted-before.read_submitted==6*chunk_bytes);
    input.data[3*chunk_bytes+17]^=0x80;const int fd=disk.files.descriptor(region.region());
    assert(pwrite(fd,input.data+3*chunk_bytes,chunk_bytes,3*chunk_bytes)==ssize_t(chunk_bytes));
    const auto bytes=t.counters.read_submitted;assert(plane.read_many(view,ranges).code==IoCode::corrupt);
    assert(t.counters.read_submitted-bytes==7*chunk_bytes&&region.region().state==RegionState::invalid);
    plane.discard();region.reset();disk.close();puts("Batch integrity: retained chunks issue no IO; corrupt data gets one reread and private state is invalidated PASS");
}
static void complete_product(bool large){
    SpilledProductSpec spec{large?1600003u:150003u,large?1500007u:130007u,8,large?32768u:8192u,1024,6,128,2,8,false};spec.stream_final=true;spec.raw_operand=1;
    spec.raw_cache_bytes=3*128;spec.retained_bytes=15*chunk_bytes;spec.inverse_stripe_cells=2048;spec.duplex_middle=large;
    SpilledProductPlan p;assert(query_spilled_product(spec,p));if(large){assert(p.middle.duplex);for(auto&f:p.forward)f.parallel_writes=true;}
    FileDevice disk(24,256u<<20,true);auto t=disk.transport();
    Aligned av(spec.an*8),bv(spec.bn*8),out((spec.an+spec.bn)*8),baseline(out.bytes),io(1u<<20),page(index_page_bytes);fill(av.span());fill(bv.span());
    auto a=value(disk,t,{io.span(),page.span()},av.span()),b=value(disk,t,{io.span(),page.span()},bv.span());
    Fixture team(8,false,128u<<20);const auto storage=team.allocate(p.storage_bytes,4096);
    for(unsigned pass=0;pass<2;++pass){
        if(pass){assert(!disk.files.enable_uring(16));disk.inner=disk.files.transport();t=disk.transport();}
        const auto before=t.counters;t.write_limit=before.write_submitted+p.normal_write_bytes;StoredValue result;WordExtent extent;
        allocation_watch_start();assert(!run_owned_product(p,t,disk.store,a,b,result,{static_cast<uint8_t*>(storage.data),storage.bytes},*team.team,&extent));assert(!allocation_watch_stop());
        assert(t.counters.read_submitted-before.read_submitted==p.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==p.normal_write_bytes);
        ValueReader reader;assert(!reader.open(t,result.region(),{io.span(),page.span()}));assert(!reader.read(0,out.span()));
        if(!pass)std::memcpy(baseline.data,out.data,out.bytes);else assert(!memcmp(baseline.data,out.data,out.bytes)&&disk.files.uring_read_usage().requests&&disk.files.uring_write_usage().requests);
        assert(ref_product_equal(reinterpret_cast<uint64_t*>(av.data),spec.an,reinterpret_cast<uint64_t*>(bv.data),spec.bn,reinterpret_cast<uint64_t*>(out.data),spec.an+spec.bn));
        result.reset();assert(!disk.files.collect());
    }
    a.reset();b.reset();disk.close();puts("Complete product: same plan, exact sync/uring result, partial raw and retained plane, exact IO and allocation gate PASS");
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);raw_ring();transport();integrity();complete_product(false);complete_product(true);}
