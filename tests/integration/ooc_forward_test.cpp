// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/forward_planes.hpp"
#include "external/middle_rows.hpp"
#include "external/inverse_planes.hpp"
#include "runtime/team.hpp"
using namespace sbn::v3::product;

static void chunks(){
    constexpr size_t n=3*chunk_bytes+17;
    auto r=fresh(n,0,7,RegionKind::scratch);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
    Aligned data(n),io_data(2*chunk_bytes),page(index_page_bytes),readback(n);fill(data.span());StoreBuffers io{io_data.span(),page.span()};
    ChunkWriterStorage q{};assert(!chunk_writer_storage(r.layout,q));Aligned image(q.index_bytes),bits(q.coverage_bytes);
    ChunkWriter writer;assert(!writer.open(t,r,io,image.span(),bits.span()));
    const size_t order[]{2,0,3,1};allocation_watch_start();
    for(size_t k:order)assert(!writer.write(k,data.span().subspan(k*chunk_bytes,std::min(chunk_bytes,n-k*chunk_bytes))));
    assert(!allocation_watch_stop() && t.counters.read_submitted==0 && t.counters.write_submitted==r.layout.data_bytes);
    ScratchPlane plane;PlaneIndexStorage need{};assert(!plane_index_storage(r.layout,need));
    const auto before=t.counters;
    assert(!writer.finish_plane(plane,bits.span().first(need.dirty_bytes)));
    assert(t.counters.read_submitted==before.read_submitted && t.counters.write_submitted==before.write_submitted);
    PlaneView v;assert(!plane.view(v));assert(!plane.read(v,0,readback.span()));assert(!memcmp(data.data,readback.data,n));
    assert(!plane.release());assert(t.counters.write_submitted==r.layout.data_bytes+r.layout.index_bytes);
    ValueReader pinned;Aligned value_image(q.index_bytes);auto counts=t.counters;
    assert(!pinned.open_pinned(t,r,io,value_image.span()));assert(t.counters.read_submitted-counts.read_submitted==r.layout.index_bytes);
    counts=t.counters;disk.corrupt_at=23;disk.corrupt_reads=1;
    assert(!pinned.read(0,readback.span()));assert(!memcmp(data.data,readback.data,n));
    assert(t.counters.read_submitted-counts.read_submitted==r.layout.data_bytes+chunk_bytes);
    for(unsigned mode=0;mode<4;++mode){
        auto f=fresh(n,0,100+mode,RegionKind::scratch);Device d(size_t(f.layout.allocation_bytes));auto io_t=d.transport();
        ChunkWriter w;assert(!w.open(io_t,f,io,image.span(),bits.span()));
        if(mode==2)d.fail_write=1;
        auto error=w.write(1,data.span().subspan(chunk_bytes,chunk_bytes));
        if(mode==0)error=w.write(1,data.span().subspan(chunk_bytes,chunk_bytes));
        if(mode==1)error=w.seal();
        if(mode==3){d.cancelled=true;error=w.seal();}
        assert(error && f.state==RegionState::invalid);
    }
    puts("Chunk construction: permuted writes, coverage, direct handoff, zero initialization IO and failures PASS");
}
static void retained_chunks(){
    constexpr size_t n=3*chunk_bytes+17,held=2*chunk_bytes;
    for(unsigned mode=0;mode<4;++mode){
        auto r=fresh(n,0,200+mode,RegionKind::scratch);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
        Aligned data(n),readback(n),retained(held),io_data(2*chunk_bytes),page(index_page_bytes);fill(data.span());
        StoreBuffers io{io_data.span(),page.span()};ChunkWriterStorage q{};PlaneIndexStorage pi{};
        assert(!chunk_writer_storage(r.layout,q)&&!plane_index_storage(r.layout,pi));Aligned image(q.index_bytes),bits(q.coverage_bytes);
        ChunkWriter writer;ScratchPlane plane;allocation_watch_start();
        assert(!writer.open(t,r,io,image.span(),bits.span(),retained.span()));assert(!writer.write(0,data.span()));
        assert(t.counters.write_submitted==r.layout.data_bytes-held);
        assert(std::all_of(disk.bytes.begin(),disk.bytes.begin()+held,[](uint8_t c){return c==0xa5;}));
        assert(!writer.finish_plane(plane,bits.span().first(pi.dirty_bytes)));
        PlaneView v;assert(!plane.view(v));assert(!plane.read(v,0,readback.span().first(held)));
        assert(t.counters.read_submitted==0&&!memcmp(data.data,readback.data,held));
        if(mode==1){
            retained.data[5]^=1;
            assert(plane.read(v,0,readback.span().first(17)).code==IoCode::corrupt);
            assert(t.counters.read_submitted==0&&r.state==RegionState::invalid);plane.discard();
        }else{
            PlaneUpdate edit;assert(!plane.begin_update(edit));
            for(size_t k=held-9;k<held+9;++k)data.data[k]^=0x71;
            assert(!plane.patch(edit,held-9,data.span().subspan(held-9,18)));assert(!plane.finish(edit));
            assert(!plane.view(v));assert(!plane.read(v,0,readback.span()));assert(!memcmp(data.data,readback.data,n));
            const auto before=t.counters;
            if(mode==2)disk.fail_write=disk.writes+1;
            if(mode==3)disk.cancelled=true;
            const auto error=plane.release();
            if(mode){
                assert(error.code==(mode==2?IoCode::io:IoCode::cancelled)&&r.state==RegionState::invalid);
                assert(t.counters.write_submitted-before.write_submitted==(mode==2?held:0));plane.discard();
            }else{
                assert(!error&&r.state==RegionState::sealed);
                assert(t.counters.write_submitted-before.write_submitted==held+r.layout.index_bytes);
                ValueReader reader;assert(!reader.open(t,r,io));assert(!reader.read(0,readback.span()));assert(!memcmp(data.data,readback.data,n));
            }
        }
        assert(!allocation_watch_stop());
    }
    puts("Retained scratch: cross-boundary patch, materialize-before-index, RAM corruption and release failures PASS");
}
static uint64_t mm(uint64_t a,uint64_t b,uint64_t p){return __uint128_t(a)*b%p;}
static std::vector<uint64_t> digits(std::span<const uint8_t>input,unsigned T,uint64_t prime){
    const size_t bytes=T/8,n=input.size()/bytes+(input.size()%bytes!=0);std::vector<uint64_t>d(n);
    for(size_t j=0;j<n;++j)for(size_t k=bytes;k-- >0;){const size_t at=j*bytes+k;d[j]=(d[j]*256+(at<input.size()?input[at]:0))%prime;}
    return d;
}
static uint64_t eval(const std::vector<uint64_t>&a,uint64_t x,uint64_t p){uint64_t y=0;for(size_t j=a.size();j-- >0;)y=(mm(y,x,p)+a[j])%p;return y;}
static void pipeline(unsigned T,unsigned np,unsigned batch,size_t C,size_t M,size_t an,size_t bn,unsigned workers,unsigned parallel=0){
    const size_t nat=(an*64+T-1)/T,nbt=(bn*64+T-1)/T;
    const size_t live=nat&&nbt?(nat+nbt-1+7)/8:1,rows=(live+M-1)/M;
    assert(rows<=C);
    ForwardPlanesSpec spec{an,C,M,rows,0,T,np,batch,workers};ForwardPlanesPlan fa{},fb{};
    assert(query_forward_planes(spec,fa));spec.input_limbs=bn;assert(query_forward_planes(spec,fb));
    for(const auto*fp:{&fa,&fb}){
        size_t last_read=64;
        for(size_t cell=0;cell<fp->kernel.input_cells;++cell)
            last_read=std::max(last_read,(cell/M)*fp->info.raw_stride+(cell%fp->spec.stripe_cells)*T+T+64);
        assert(fp->info.raw_bytes==last_read);
    }
    if(workers>1){const bool compact=np==4||batch==np;
        assert(enable_forward_pipeline(fa,compact?fa.single_raw_pipeline_bytes:fa.pipeline_storage_bytes));
        assert(enable_forward_pipeline(fb,compact?fb.single_raw_pipeline_bytes:fb.pipeline_storage_bytes));
        if(compact)assert(!fa.pipeline_double_raw&&!fb.pipeline_double_raw);}
    auto a=fresh(an*8,0,1),b=fresh(bn*8,a.layout.allocation_bytes,2);
    std::array<Region,20> regions{};uint64_t end=b.layout.data_offset+b.layout.allocation_bytes;
    for(unsigned q=0;q<2*np;++q){regions[q]=fresh(size_t(fa.plane_bytes),end,100+q,RegionKind::scratch);end+=regions[q].layout.allocation_bytes;}
    Device disk{size_t(end)};auto transport=disk.transport();
    if(parallel){fa.parallel_writes=fb.parallel_writes=true;transport.parallel_writes=std::max(2u,parallel);}
    Aligned av(an*8),bv(bn*8),io_data(8*chunk_bytes),page(index_page_bytes);fill(av.span());fill(bv.span());
    for(size_t j=0;j<bv.bytes;++j)bv.data[j]^=0x91;
    StoreBuffers io{io_data.span(),page.span()};write_value(transport,a,io,av.span());write_value(transport,b,io,bv.span());
    const std::vector<uint8_t> original(disk.bytes.begin(),disk.bytes.begin()+b.layout.data_offset+b.layout.allocation_bytes);
    ChunkWriterStorage cw{};assert(!chunk_writer_storage(regions[0].layout,cw));PlaneIndexStorage pi{};assert(!plane_index_storage(regions[0].layout,pi));
    std::array<std::unique_ptr<Aligned>,20> images,bits;
    std::array<ChunkWriter,20> writers;std::array<ScratchPlane,20> planes;
    for(unsigned q=0;q<2*np;++q){images[q]=std::make_unique<Aligned>(cw.index_bytes);bits[q]=std::make_unique<Aligned>(cw.coverage_bytes);
        assert(!writers[q].open(transport,regions[q],io,images[q]->span(),bits[q]->span()));}
    Fixture f(workers,false,size_t(128)<<20);const auto workspace=f.allocate(std::max(fa.storage_bytes,fb.storage_bytes),4096);
    auto space=std::span(static_cast<uint8_t *>(workspace.data),workspace.bytes);
    for(unsigned operand=0;operand<2;++operand){const auto &p=operand?fb:fa;const auto &source=operand?b:a;
        std::memset(space.data(),0xa5,space.size()); // omitted TFT input blocks must not depend on arena zeros
        const auto before=transport.counters;allocation_watch_start();
        assert(!run_forward_planes(p,transport,source,std::span(writers.data()+operand*np,np),space,*f.team));assert(!allocation_watch_stop());
        assert(transport.counters.read_submitted-before.read_submitted==source.layout.data_bytes+source.layout.index_bytes);
        assert(transport.counters.write_submitted-before.write_submitted==np*p.plane_bytes);
        const auto written=transport.counters.write_submitted;
        for(unsigned q=0;q<np;++q){const unsigned at=operand*np+q;
            assert(!writers[at].finish_plane(planes[at],bits[at]->span().first(pi.dirty_bytes)));}
        assert(transport.counters.write_submitted==written);
    }
    const unsigned log=unsigned(__builtin_ctzll(C))+unsigned(__builtin_ctzll(M));
    MiddleRowsSpec ms{0,log,workers,M,0,rows,0,fa.row_stride,1,false};MiddleRowsPlan mid{};assert(query_middle_rows(ms,mid));
    const auto midspace=f.allocate(mid.storage_bytes,64);
    for(unsigned q=0;q<np;++q){ms.prime=q;assert(query_middle_rows(ms,mid));allocation_watch_start();
        assert(!run_middle_rows(mid,planes[q],planes[np+q],{static_cast<uint8_t *>(midspace.data),midspace.bytes},*f.team));assert(!allocation_watch_stop());}
    InversePlanesPlan ip{};assert(query_inverse_planes(fa.kernel,ip));
    const auto invspace=f.allocate(ip.storage_bytes,64);InversePlanesBinding inverse{};
    prepare_inverse_planes(ip,{static_cast<uint8_t *>(invspace.data),invspace.bytes},*f.team,inverse);
    const size_t width=ip.spec.stripe_cells;
    std::vector<std::vector<uint64_t>> coefficients(np,std::vector<uint64_t>(rows*M*8));
    const auto before_inverse=transport.counters;
    for(size_t column=0;column<M;column+=width){
        CoefficientStripe stripe{};allocation_watch_start();
        assert(!read_inverse_stripe(inverse,column,std::span(planes.data()+np,np),stripe));assert(!allocation_watch_stop());
        for(unsigned q=0;q<np;++q)for(size_t r=0;r<rows;++r)for(size_t j=0;j<width;++j)for(unsigned lane=0;lane<8;++lane){
            const uint64_t value=stripe.data[((q*rows+r)*width+j)*8+lane],p=p48_row_prime(q);
            assert(value<4*p);coefficients[q][(r*M+column+j)*8+lane]=value%p;
        }
    }
    assert(transport.counters.read_submitted-before_inverse.read_submitted==np*fa.plane_bytes);
    assert(transport.counters.write_submitted==before_inverse.write_submitted);
    for(unsigned q=0;q<np;++q){const uint64_t p=p48_row_prime(q);const auto da=digits(av.span(),T,p),db=digits(bv.span(),T,p);
        if(da.size()*db.size()<1000000){
            std::vector<uint64_t> expected(rows*M*8);
            for(size_t i=0;i<da.size();++i)for(size_t j=0;j<db.size();++j){expected[i+j]+=mm(da[i],db[j],p);if(expected[i+j]>=p)expected[i+j]-=p;}
            assert(expected==coefficients[q]);
        }else for(uint64_t x:{7,1009,65537})assert(eval(coefficients[q],x,p)==mm(eval(da,x,p),eval(db,x,p),p));
    }
    assert(!memcmp(original.data(),disk.bytes.data(),original.size()));
    for(unsigned q=0;q<2*np;++q)planes[q].discard();
    printf("Forward/middle/inverse modular pipeline T%u NP%u batch%u %zux%zu rows%zu J%zu W%u: input read once, planes write once PASS\n",T,np,batch,C,M,rows,width,workers);
}
static void forward_failures(bool parallel=false){
    ForwardPlanesPlan p;assert(query_forward_planes({19001,4,4096,3,1024,128,6,2,4},p));
    assert(!enable_forward_pipeline(p,p.single_raw_pipeline_bytes-1));assert(enable_forward_pipeline(p,p.single_raw_pipeline_bytes));
    p.parallel_writes=parallel;
    Fixture f(4,false,64u<<20);auto lease=f.allocate(p.storage_bytes,4096);auto workspace=std::span(static_cast<uint8_t *>(lease.data),lease.bytes);
    for(unsigned fault=1;fault<=6;++fault){
        auto source=fresh(19001*8,0,71);std::array<Region,6>regions;uint64_t end=source.layout.allocation_bytes;
        for(unsigned q=0;q<6;++q){regions[q]=fresh(size_t(p.plane_bytes),end,90+q,RegionKind::scratch);end+=regions[q].layout.allocation_bytes;}
        Device device{size_t(end)};auto t=device.transport();Aligned input(19001*8),data(4*chunk_bytes),page(index_page_bytes);fill(input.span());
        if(parallel)t.parallel_writes=2;
        StoreBuffers io{data.span(),page.span()};write_value(t,source,io,input.span());
        const auto original=std::vector<uint8_t>(device.bytes.begin(),device.bytes.begin()+source.layout.allocation_bytes);
        ChunkWriterStorage cw;assert(!chunk_writer_storage(regions[0].layout,cw));std::array<ChunkWriter,6>writers;
        std::array<std::unique_ptr<Aligned>,6>images,bits;
        for(unsigned q=0;q<6;++q){images[q]=std::make_unique<Aligned>(cw.index_bytes);bits[q]=std::make_unique<Aligned>(cw.coverage_bytes);
            assert(!writers[q].open(t,regions[q],io,images[q]->span(),bits[q]->span()));}
        if(fault==1)device.fail_read=device.reads+3;
        if(fault==2)device.fail_write=device.writes+2;
        if(fault==3)device.cancel_after_write=device.writes+1;
        if(fault==4)device.short_read=device.reads+3;
        if(fault==5){device.corrupt_at=source.layout.data_offset+17;device.corrupt_reads=-1;}
        if(fault==6)t.write_limit=t.counters.write_submitted+5*chunk_bytes;
        uint64_t residue=123;allocation_watch_start();const auto error=run_forward_planes(p,t,source,writers,workspace,*f.team,&residue);
        assert(!allocation_watch_stop()&&error&&!residue&&!f.team->busy);
        assert(t.counters.write_submitted<=t.write_limit);
        for(auto&r:regions)assert(r.state==RegionState::invalid);
        assert(!memcmp(original.data(),device.bytes.data(),original.size()));
    }
    puts("Forward two-slot failures: read/write/short/corrupt/cancel, immutable source, abandoned output and joined cleanup PASS");
}
int main(){
    chunks();
    retained_chunks();
    pipeline(64,1,1,1,1,0,1,1);
    pipeline(80,4,2,8,16,257,173,1);
    pipeline(96,5,3,16,32,619,321,4);
    pipeline(128,6,3,16,32,619,321,4);
    pipeline(144,7,3,16,32,619,321,4);
    pipeline(176,8,4,16,32,619,321,4);
    pipeline(128,6,3,256,4096,900003,700007,4); // partial input block; later TFT blocks are not physically initialized
    pipeline(224,10,4,4,2048,65009,52003,4);
    pipeline(72,4,1,4,4096,42009,31003,4);
    pipeline(128,6,6,4,4096,55009,32003,4); // one prime wave, several raw-stripe reuses
    forward_failures();
    pipeline(224,10,4,4,2048,65009,52003,4,true);
    pipeline(128,6,6,4,4096,55009,32003,4,true);
    pipeline(128,6,4,4,4096,55009,32003,8,4);
    forward_failures(true);
    ForwardPlanesSpec invalid{1,4,4096,1,1024,72,4,1,1};ForwardPlanesPlan p{};assert(!query_forward_planes(invalid,p));
    puts("OOC forward producer: coverage, pinned source index, shared decode, outer TFT and normalized inverse PASS");
}
