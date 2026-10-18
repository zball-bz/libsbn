// SPDX-License-Identifier: LGPL-3.0-or-later
#include "file_store_support.hpp"
#include "external/middle_rows.hpp"
#include "runtime/team.hpp"

static void one(Fixture&f,unsigned mode,unsigned fault=0,bool compact=false,size_t n=8192,bool duplex=false,bool capable=true){
    printf("Begin middle mode%u fault%u compact%u duplex%u cells%zu\n",mode,fault,unsigned(compact),unsigned(duplex),n);
    const size_t rows=5,prefix=compact?chunk_bytes:19,stride=n*48+(compact?chunk_bytes:64),bytes=prefix+rows*stride+31;
    const bool square=mode==1,raw=mode==2;
    MiddleRowsSpec spec{5,unsigned(__builtin_ctzll(n))+3,4,n,1,rows,prefix,stride,17,square};
    spec.a_transformed=mode==3;
    MiddleRowsPlan serial;assert(query_middle_rows(spec,serial));auto pipe=serial;
    if(compact){
        assert(!enable_middle_compact_pipeline(pipe,serial.compact_storage_bytes-1));
        assert(enable_middle_compact_pipeline(pipe,serial.compact_storage_bytes));
        assert(pipe.storage_bytes<pipe.pipeline_storage_bytes);
    }else{
        assert(!enable_middle_pipeline(pipe,serial.pipeline_storage_bytes-1));
        assert(enable_middle_pipeline(pipe,serial.pipeline_storage_bytes));
    }
    assert(pipe.storage_bytes>serial.storage_bytes&&pipe.pipeline_row.workers<=3);
    if(duplex){assert(!enable_middle_duplex(pipe,pipe.duplex_storage_bytes-1));assert(enable_middle_duplex(pipe,pipe.duplex_storage_bytes));pipe.parallel_reads=true;}
    Aligned wa(bytes),wb(bytes);fill(wa.span());fill(wb.span());
    // Canonical packed residues, with arbitrary untouched row/chunk guards.
    const auto prime=product::p48_row_prime(5);
    for(size_t r=0;r<rows;++r)for(size_t i=0;i<n*8;++i)for(auto *p:{wa.data,wb.data}){
        const uint64_t v=random_word()%prime;std::memcpy(p+prefix+r*stride+6*i,&v,6);
    }
    std::vector<uint8_t>expected;IoCounters counts{};
    Aligned work(pipe.storage_bytes);
    for(unsigned variant=0;variant<2;++variant){
        auto ar=fresh(bytes,0,11,RegionKind::scratch),br=fresh(bytes,ar.layout.allocation_bytes,12,RegionKind::scratch);
        Device d(size_t(br.layout.data_offset+br.layout.allocation_bytes));auto t=d.transport();t.duplex=capable;t.parallel_reads=capable?2:1;
        Aligned io(4*chunk_bytes),page(index_page_bytes);StoreBuffers buffers{io.span(),page.span()};
        write_value(t,ar,buffers,wa.span());write_value(t,br,buffers,wb.span());
        PlaneIndexStorage pi{};assert(!plane_index_storage(ar.layout,pi));
        Aligned ia(pi.image_bytes),ib(pi.image_bytes),da(pi.dirty_bytes),db(pi.dirty_bytes);
        ScratchPlane a,b;assert(!a.take(t,ar,buffers,ia.span(),da.span()));assert(!b.take(t,br,buffers,ib.span(),db.span()));
        const auto before=t.counters;
        if(variant&&fault){
            if(fault==1)d.fail_read=d.reads+3;
            if(fault==2)d.fail_write=d.writes+2;
            if(fault==3)d.cancel_after_write=d.writes+1;
            if(fault==4)d.short_read=d.reads+3;
            if(fault==5){d.corrupt_reads=-1;d.corrupt_at=ar.layout.data_offset+prefix+1;}
        }
        MiddleRowsStats stats;allocation_watch_start();
        const auto e=raw?run_middle_memory_a(variant?pipe:serial,wa.span(),b,work.span(),*f.team,&stats):
            run_middle_rows(variant?pipe:serial,a,square?a:b,work.span(),*f.team,&stats);
        assert(!allocation_watch_stop()&&!f.team->busy);
        if(variant&&fault){assert(e);assert((square?ar:br).state==RegionState::invalid);
            a.discard();b.discard();continue;}
        assert(!e&&stats.rows==rows&&stats.compute_ns&&stats.read_ns&&stats.write_ns);
        assert(stats.duplex==(variant&&duplex&&capable));
        if(!variant){counts=t.counters;counts.read_submitted-=before.read_submitted;counts.write_submitted-=before.write_submitted;}
        else{assert(counts.read_submitted==t.counters.read_submitted-before.read_submitted);
             assert(counts.write_submitted==t.counters.write_submitted-before.write_submitted);}
        auto&out=square?a:b;PlaneView v;assert(!out.view(v));Aligned got(bytes);assert(!out.read(v,0,got.span()));
        if(!variant)expected.assign(got.data,got.data+bytes);
        else assert(!std::memcmp(expected.data(),got.data,bytes));
        for(size_t i=0;i<bytes;++i){const bool in=i>=prefix&&(i-prefix)/stride<rows&&(i-prefix)%stride<n*48;
            if(!in)assert(got.data[i]==(square?wa.data[i]:wb.data[i]));}
        assert(!out.release());a.discard();b.discard();
    }
    printf("Middle pipeline mode%u fault%u compact%u: bit-exact, guarded patches, identical IO, joined cleanup, allocation-free PASS\n",mode,fault,unsigned(compact));
}
static void cached_square(Fixture&f){
    puts("Begin immutable cached square");
    constexpr size_t n=8192,rows=5,stride=n*48,bytes=rows*stride;
    MiddleRowsSpec spec{5,16,4,n,1,rows,0,stride,17,true,true,true};
    MiddleRowsPlan serial;assert(query_middle_rows(spec,serial));auto pipe=serial;
    assert(enable_middle_pipeline(pipe,serial.pipeline_storage_bytes));Aligned work(pipe.storage_bytes),input(bytes);
    for(size_t i=0;i<bytes/6;++i){const uint64_t v=random_word()%product::p48_row_prime(5);std::memcpy(input.data+6*i,&v,6);}
    std::vector<uint8_t> expected;uint64_t reads=0,writes=0;
    for(unsigned variant=0;variant<3;++variant){
        auto ar=fresh(bytes,0,51,RegionKind::scratch),br=fresh(bytes,ar.layout.allocation_bytes,52,RegionKind::scratch);
        Device d(size_t(br.layout.data_offset+br.layout.allocation_bytes));auto t=d.transport();
        Aligned io(4*chunk_bytes),page(index_page_bytes);StoreBuffers buffers{io.span(),page.span()};write_value(t,ar,buffers,input.span());
        PlaneIndexStorage pi;assert(!plane_index_storage(ar.layout,pi));Aligned index(pi.image_bytes),dirty(pi.dirty_bytes);
        ScratchPlane a,b;assert(!a.take(t,ar,buffers,index.span(),dirty.span()));
        ChunkWriterStorage cw;assert(!chunk_writer_storage(br.layout,cw));Aligned out_index(cw.index_bytes),coverage(cw.coverage_bytes),out_dirty(pi.dirty_bytes);
        ChunkWriter out;assert(!out.open(t,br,buffers,out_index.span(),coverage.span()));
        const auto before=t.counters;if(variant==2)d.fail_write=d.writes+2;
        allocation_watch_start();const auto e=run_middle_cached_square(variant?pipe:serial,a,out,work.span(),*f.team);assert(!allocation_watch_stop());
        if(variant==2){assert(e&&br.state==RegionState::invalid);PlaneView v;assert(!a.view(v));
            Aligned got(bytes);assert(!a.read(v,0,got.span()));assert(!memcmp(got.data,input.data,bytes));out.abandon();a.discard();continue;}
        assert(!e);if(!variant){reads=t.counters.read_submitted-before.read_submitted;writes=t.counters.write_submitted-before.write_submitted;}
        else assert(reads==t.counters.read_submitted-before.read_submitted&&writes==t.counters.write_submitted-before.write_submitted);
        assert(!out.finish_plane(b,out_dirty.span()));PlaneView v;assert(!b.view(v));Aligned got(bytes);assert(!b.read(v,0,got.span()));
        if(!variant)expected.assign(got.data,got.data+bytes);else assert(!memcmp(got.data,expected.data(),bytes));
        a.discard();b.discard();
    }
    puts("Immutable completed-spectrum square: bit-exact, identical IO, source survives in-flight output failure PASS");
}
static void recycled(Fixture&f,unsigned rows,unsigned fault=0,bool duplex=false,bool capable=true){
    const size_t n=8192,prefix=duplex?chunk_bytes:4096,stride=n*48+(duplex?chunk_bytes:4096);
    const size_t bytes=prefix+rows*stride+31;
    MiddleRowsSpec spec{5,17,4,n,1,rows,prefix,stride,17,false};
    MiddleRowsPlan serial;assert(query_middle_rows(spec,serial));auto pipe=serial;
    assert(enable_middle_recycled_pipeline(pipe)&&pipe.pipeline&&pipe.recycle_a&&pipe.storage_bytes==serial.storage_bytes);
    if(duplex)assert(enable_middle_duplex(pipe,pipe.storage_bytes));
    auto bad=serial;bad.spec.rows=3;assert(!enable_middle_recycled_pipeline(bad));
    bad=serial;bad.spec.byte_offset=19;assert(!enable_middle_recycled_pipeline(bad));
    Aligned a(bytes),b(bytes),work(serial.storage_bytes);fill(a.span());fill(b.span());
    for(size_t r=0;r<rows;++r)for(size_t i=0;i<n*8;++i)for(auto*p:{a.data,b.data}){
        const uint64_t v=random_word()%product::p48_row_prime(5);memcpy(p+prefix+r*stride+6*i,&v,6);}
    const std::vector<uint8_t> original(a.data,a.data+bytes);std::vector<uint8_t> expected;uint64_t reads=0,writes=0;
    for(unsigned variant=0;variant<2;++variant){
        auto br=fresh(bytes,0,72,RegionKind::scratch);Device d(size_t(br.layout.allocation_bytes));auto t=d.transport();t.duplex=capable;
        Aligned io(4*chunk_bytes),page(index_page_bytes);StoreBuffers buffers{io.span(),page.span()};write_value(t,br,buffers,b.span());
        PlaneIndexStorage pi;assert(!plane_index_storage(br.layout,pi));Aligned index(pi.image_bytes),dirty(pi.dirty_bytes);
        ScratchPlane output;assert(!output.take(t,br,buffers,index.span(),dirty.span()));const auto before=t.counters;
        // Immutable adapters must reject a destructive plan before touching B.
        assert(run_middle_memory_a(pipe,a.span(),output,work.span(),*f.team).code==IoCode::invalid);
        assert(br.state==RegionState::scratch_owned&&t.counters.read_submitted==before.read_submitted);
        if(variant&&fault){
            if(fault==1)d.fail_read=d.reads+20;
            if(fault==2)d.fail_write=d.writes+20;
            if(fault==3)d.cancel_after_write=d.writes+20;
            if(fault==4)d.short_read=d.reads+20;
            if(fault==5){d.corrupt_reads=-1;d.corrupt_at=prefix+3*stride+1;}
        }
        MiddleRowsStats stats;allocation_watch_start();
        const auto e=variant?run_middle_consuming_a(pipe,a.span(),output,work.span(),*f.team,&stats):
            run_middle_memory_a(serial,a.span(),output,work.span(),*f.team,&stats);
        assert(!allocation_watch_stop()&&!f.team->busy);
        const unsigned borrowed=duplex&&capable?3:2;
        for(size_t i=0;i<bytes;++i)if(i<prefix||i>=prefix+(borrowed-1)*stride+n*48||(i-prefix)%stride>=n*48)assert(a.data[i]==original[i]);
        if(variant&&fault){assert(e&&br.state==RegionState::invalid);output.discard();continue;}
        assert(!e&&stats.rows==rows);
        assert(stats.duplex==(variant&&duplex&&capable));
        if(!variant){reads=t.counters.read_submitted-before.read_submitted;writes=t.counters.write_submitted-before.write_submitted;}
        else assert(reads==t.counters.read_submitted-before.read_submitted&&writes==t.counters.write_submitted-before.write_submitted);
        PlaneView view;assert(!output.view(view));Aligned got(bytes);assert(!output.read(view,0,got.span()));
        if(!variant)expected.assign(got.data,got.data+bytes);else assert(!memcmp(expected.data(),got.data,bytes));
        output.discard();
    }
    printf("Recycled middle rows=%u fault=%u: no extra storage/IO, exact result, only dead A rows overwritten, cleanup PASS\n",rows,fault);
}
static void files(Fixture&f,unsigned fault=0){
    constexpr size_t cells=8192,rows=7,bytes=cells*48*rows;
    MiddleRowsPlan serial;assert(query_middle_rows({5,16,4,cells,0,rows,0,cells*48,17,false},serial));
    auto p=serial;assert(enable_middle_compact_pipeline(p,p.compact_storage_bytes));assert(enable_middle_duplex(p,p.duplex_storage_bytes));
    p.parallel_reads=true;
    Aligned a(bytes),b(bytes),work(p.storage_bytes);for(size_t i=0;i<bytes/6;++i)for(auto*x:{a.data,b.data}){
        const uint64_t v=random_word()%product::p48_row_prime(5);memcpy(x+6*i,&v,6);}
    std::vector<uint8_t>want;
    for(unsigned variant=0;variant<2;++variant){
        FileDevice disk(8,2*bytes+(1u<<20));auto t=disk.transport();assert(t.duplex);
        RegionLease ar,br;assert(!disk.store.reserve(bytes,RegionKind::scratch,ar));assert(!disk.store.reserve(bytes,RegionKind::scratch,br));
        Aligned io(1u<<20),page(index_page_bytes);StoreBuffers buffers{io.span(),page.span()};
        write_value(t,ar.region(),buffers,a.span());write_value(t,br.region(),buffers,b.span());
        PlaneIndexStorage q;assert(!plane_index_storage(ar.region().layout,q));Aligned ai(q.image_bytes),bi(q.image_bytes),ad(q.dirty_bytes),bd(q.dirty_bytes);
        ScratchPlane A,B;assert(!A.take(t,ar.region(),buffers,ai.span(),ad.span()));assert(!B.take(t,br.region(),buffers,bi.span(),bd.span()));
        if(variant&&fault==1)disk.fail_write=disk.writes+2;
        if(variant&&fault==2)disk.cancel_after_write=disk.writes+2;
        MiddleRowsStats stats;allocation_watch_start();const auto e=run_middle_rows(variant?p:serial,A,B,work.span(),*f.team,&stats);
        assert(!allocation_watch_stop()&&!f.team->busy);
        if(variant&&fault)assert(e&&br.region().state==RegionState::invalid);
        else{assert(!e&&stats.duplex==bool(variant));PlaneView v;assert(!B.view(v));Aligned out(bytes);assert(!B.read(v,0,out.span()));
            if(!variant)want.assign(out.data,out.data+bytes);else assert(!memcmp(out.data,want.data(),bytes));}
        assert(disk.files.usage().read_bytes==t.counters.read_completed&&disk.files.usage().write_bytes==t.counters.write_completed);
        A.discard();B.discard();ar.reset();br.reset();disk.close();
    }
    printf("FileStore duplex fault%u: exact result, joined cleanup and byte counters PASS\n",fault);
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);Fixture f(4,false,64u<<20);for(unsigned mode=0;mode<4;++mode)one(f,mode);
    for(unsigned fault=1;fault<=5;++fault)one(f,0,fault);cached_square(f);
    for(unsigned rows:{4u,5u,7u})recycled(f,rows);for(unsigned fault=1;fault<=5;++fault)recycled(f,7,fault);
    for(unsigned mode=0;mode<4;++mode)one(f,mode,0,true,65536);
    for(unsigned fault=1;fault<=5;++fault)one(f,0,fault,true,65536);
    for(unsigned mode=0;mode<4;++mode)one(f,mode,0,true,65536,true);
    for(unsigned fault=1;fault<=5;++fault)one(f,0,fault,true,65536,true);
    one(f,0,0,true,65536,true,false);recycled(f,7,0,true,false);
    recycled(f,7,0,true);for(unsigned fault=1;fault<=5;++fault)recycled(f,7,fault,true);
    for(unsigned fault=0;fault<3;++fault)files(f,fault);}
