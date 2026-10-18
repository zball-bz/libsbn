// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/scratch_plane.hpp"
#include "external/chunk_writer.hpp"
#include "external/middle_rows.hpp"
#include "product/modular_row.hpp"
#include <sys/mman.h>
using namespace sbn::v3::product;

static uint64_t mm(uint64_t a,uint64_t b,uint64_t p){return __uint128_t(a)*b%p;}
static uint64_t power(uint64_t a,uint64_t n,uint64_t p){uint64_t r=1;for(;n;n>>=1,a=mm(a,a,p))if(n&1)r=mm(r,a,p);return r;}
static uint64_t primitive(uint64_t p){
    uint64_t n=p-1;std::vector<uint64_t> factors;
    for(uint64_t d=2;d*d<=n;++d)if(n%d==0){factors.push_back(d);while(n%d==0)n/=d;}
    if(n>1)factors.push_back(n);
    for(uint64_t g=2;;++g){bool good=true;for(uint64_t d:factors)if(power(g,(p-1)/d,p)==1)good=false;if(good)return g;}
}
static uint64_t gamma_of(const ModularRowSpec &s){const auto p=p48_row_prime(s.prime);return power(primitive(p),((p-1)>>s.global_log)*s.global_prefix,p);}
static std::vector<uint64_t> reference(const ModularRowSpec&s,const uint64_t*a,const uint64_t*b,bool sparse=false){
    const uint64_t p=p48_row_prime(s.prime),theta=power(s.local_tau,s.cells,p);
    const uint64_t shift=mm(gamma_of(s),power(s.local_tau,p-2,p),p);
    std::vector<uint64_t> result(s.cells*8);std::vector<size_t> nz;
    for(size_t j=0;j<s.cells*8;++j)if(b[j] || !sparse)nz.push_back(j);
    for(size_t i=0;i<s.cells*8;++i)for(size_t j:nz){
        const unsigned wrap=(i%8+j%8)>=8;
        const size_t raw=i/8+j/8+wrap,at=(raw%s.cells)*8+(i%8+j%8)%8;
        uint64_t factor=wrap?shift:1;if(raw>=s.cells)factor=mm(factor,theta,p);
        const uint64_t value=mm(mm(a[i],b[j],p),mm(factor,s.product_scale,p),p);
        result[at]+=value;if(result[at]>=p)result[at]-=p;
    }
    return result;
}
struct RowCall {std::span<const uint8_t> prepared;std::span<uint8_t>scratch;std::span<uint64_t>a,b;};
static void row_action(void *p,sbn3_team_scope *scope){auto &c=*static_cast<RowCall *>(p);p48_row_execute(c.prepared,c.scratch,c.a,c.b,scope);}
static void row_test(Fixture&f,ModularRowSpec s,bool square,bool sparse=false){
    ModularRowInfo info{};assert(p48_row_query(s,info));
    Aligned prepared(info.prepared_bytes),scratch(info.scratch_bytes),a(info.array_bytes),b(info.array_bytes);
    const uint64_t p=p48_row_prime(s.prime);auto *av=reinterpret_cast<uint64_t *>(a.data),*bv=reinterpret_cast<uint64_t *>(b.data);
    for(size_t j=0;j<s.cells*8;++j){av[j]=(uint64_t(731)*j+19)%p;bv[j]=sparse?0:(p-1-(uint64_t(13)*j)%p);}
    if(sparse){bv[0]=3;bv[1]=7;bv[7]=11;bv[8]=23;bv[s.cells*8-1]=p-1;}
    const auto want=reference(s,av,square?av:bv,sparse);
    allocation_watch_start();p48_row_prepare(s,info,prepared.span());
    RowCall call{prepared.span(),scratch.span(),{av,s.cells*8},{square?av:bv,s.cells*8}};
    sbn3_team_run(f.team,row_action,&call);assert(!allocation_watch_stop());
    auto *got=square?av:bv;
    for(size_t j=0;j<want.size();++j)assert(got[j]<4*p && got[j]%p==want[j]);
    // Exact-base expand/compact uses no second operand-sized image.
    std::span<uint64_t> values(got,s.cells*8);std::span<uint8_t> image(reinterpret_cast<uint8_t *>(got),info.array_bytes);
    p48_row_pack(prepared.span(),values,image);p48_row_unpack(prepared.span(),image,values);
    assert(std::equal(want.begin(),want.end(),got));
}
static void unary(Fixture&f,size_t n){
    ModularRowSpec s{RowOperation::forward,9,32,4,n,37,3,1};
    ModularRowInfo info{};assert(p48_row_query(s,info));
    Aligned prepared(info.prepared_bytes),scratch(info.scratch_bytes),data(info.array_bytes);
    auto *v=reinterpret_cast<uint64_t *>(data.data);const uint64_t p=p48_row_prime(s.prime);
    for(size_t i=0;i<n*8;++i)v[i]=(i*57+19)%p;
    p48_row_prepare(s,info,prepared.span());RowCall call{prepared.span(),scratch.span(),{},{v,n*8}};
    allocation_watch_start();sbn3_team_run(f.team,row_action,&call);assert(!allocation_watch_stop());
    s.operation=RowOperation::inverse;p48_row_prepare(s,info,prepared.span());
    allocation_watch_start();sbn3_team_run(f.team,row_action,&call);assert(!allocation_watch_stop());
    for(size_t i=0;i<n*8;++i)assert(v[i]<4*p && v[i]%p==(i*57+19)%p);
}
struct PackedCall {
    PackedRowBatchSpec spec;ModularRowInfo info;
    std::span<const uint8_t>a;std::span<uint8_t>b,prepared,scratch;
    std::span<uint64_t>aw,bw;
};
static void packed_action(void*ptr,sbn3_team_scope*scope){auto&c=*static_cast<PackedCall *>(ptr);
    p48_rows_packed(c.spec,c.info,c.a,c.b,c.prepared,c.scratch,c.aw,c.bw,scope);
}
static void packed_rows(Fixture&f,size_t n,unsigned workers,bool square,unsigned frontiers=0){
    assert(!square||frontiers==0||frontiers==3);
    constexpr size_t rows=2;const size_t stride=n*48+64;
    ModularRowSpec s{RowOperation::product,9,unsigned(__builtin_ctzll(n))+2,workers,n,0,1,17};
    ModularRowInfo info{};assert(p48_row_query(s,info));const auto p=p48_row_prime(s.prime);
    Aligned a(rows*stride),b(rows*stride),prepared(info.prepared_bytes),scratch(info.scratch_bytes),av(info.array_bytes),bv(info.array_bytes);
    std::memset(a.data,0x73,a.bytes);std::memset(b.data,0x53,b.bytes);
    auto*ap=reinterpret_cast<uint64_t *>(av.data);auto*bp=reinterpret_cast<uint64_t *>(bv.data);
    std::array<std::vector<uint64_t>,rows>want;
    for(size_t i=0;i<rows;++i){s.global_prefix=2-i;s.local_tau=gamma_of(s);p48_row_prepare(s,info,prepared.span());
        for(size_t j=0;j<n*8;++j){ap[j]=square?0:(j*719+i+7)%p;bp[j]=0;}
        if(square){ap[0]=7;ap[7]=11;ap[n*8-1]=p-1;}
        bp[0]=5;bp[7]=3;bp[n*8-1]=13;
        want[i]=reference(s,ap,square?ap:bp,true);
        auto complete=[&](uint64_t*words){auto forward=s;forward.operation=RowOperation::forward;
            p48_row_prepare(forward,info,prepared.span());RowCall fc{prepared.span(),scratch.span(),{},{words,n*8}};
            sbn3_team_run(f.team,row_action,&fc);};
        if(frontiers&1)complete(ap);
        if(!square&&(frontiers&2))complete(bp);
        p48_row_pack(prepared.span(),{ap,n*8},a.span().subspan(i*stride,n*48));
        p48_row_pack(prepared.span(),{square?ap:bp,n*8},b.span().subspan(i*stride,n*48));
    }
    const auto saved=std::vector<uint8_t>(a.data,a.data+a.bytes);
    PackedCall call{{s.prime,s.global_log,workers,n,1,rows,stride,s.product_scale,square},info,
        a.span(),b.span(),prepared.span(),scratch.span(),{ap,n*8},{bp,n*8}};
    call.spec.a_transformed=bool(frontiers&1);call.spec.b_transformed=bool(frontiers&2);
    allocation_watch_start();sbn3_team_run(f.team,packed_action,&call);assert(!allocation_watch_stop());
    assert(!memcmp(saved.data(),a.data,a.bytes));
    for(size_t i=0;i<rows;++i){std::memcpy(bv.data,b.data+i*stride,n*48);p48_row_unpack(prepared.span(),bv.span(),{bp,n*8});
        assert(std::equal(want[i].begin(),want[i].end(),bp));
        for(size_t j=n*48;j<stride;++j)assert(b.data[i*stride+j]==0x53);
    }
    printf("Packed row batch n%zu W%u SQR%u frontier%u: independent polynomial result, exact row guards and allocation PASS\n",n,workers,square,frontiers);
}
static void packed_unary_guard(Fixture&f,size_t n,unsigned workers){
    constexpr size_t rows=2;const size_t stride=n*48,bytes=rows*stride;
    auto*map=static_cast<uint8_t*>(mmap(nullptr,bytes+8192,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));assert(map!=MAP_FAILED);
    assert(!mprotect(map,4096,PROT_NONE)&&!mprotect(map+4096+bytes,4096,PROT_NONE));auto*packed=map+4096;
    ModularRowSpec s{RowOperation::forward,9,unsigned(__builtin_ctzll(n))+2,workers,n,0,1,1};ModularRowInfo info{};assert(p48_row_query(s,info));
    Aligned prepared(info.prepared_bytes),scratch(info.scratch_bytes),a(info.array_bytes),b(info.array_bytes);
    auto*words=reinterpret_cast<uint64_t*>(b.data);p48_row_prepare(s,info,prepared.span());
    for(size_t row=0;row<rows;++row){for(size_t j=0;j<n*8;++j)words[j]=37*j+row+1;p48_row_pack(prepared.span(),{words,n*8},{packed+row*stride,stride});}
    const std::vector<uint8_t> saved(packed,packed+bytes);
    PackedCall call{{s.prime,s.global_log,workers,n,1,rows,stride,1,false},info,{},
        {packed,bytes},prepared.span(),scratch.span(),{reinterpret_cast<uint64_t*>(a.data),n*8},{words,n*8}};
    for(auto op:{RowOperation::forward,RowOperation::inverse}){call.spec.operation=op;
        allocation_watch_start();sbn3_team_run(f.team,packed_action,&call);assert(!allocation_watch_stop());}
    assert(!memcmp(packed,saved.data(),bytes));assert(!munmap(map,bytes+8192));
    printf("Packed unary n%zu W%u: exact forward/inverse, no tail padding, guard pages and allocation PASS\n",n,workers);
}
struct EditCall {ScratchPlane*plane;PlaneUpdate*update;unsigned called=0;};
static void edit_action(void*ptr,std::span<uint8_t>bytes,sbn3_team_scope*){
    auto&c=*static_cast<EditCall *>(ptr);++c.called;uint8_t x=0;
    assert(c.plane->read(*c.update,0,{&x,1}).code==IoCode::stale);
    assert(c.plane->finish(*c.update).code==IoCode::stale);
    for(auto&v:bytes)v^=0x5a;
}
static void resident_edit(){
    Fixture f(4,false,size_t(64)<<20);
    for(unsigned mode=0;mode<3;++mode){
        const size_t n=3*chunk_bytes+101;auto r=fresh(n,0,901+mode,RegionKind::scratch);
        Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
        ChunkWriterStorage q{};assert(!chunk_writer_storage(r.layout,q));
        Aligned input(n),retained(size_t(r.layout.data_bytes)),image(q.index_bytes),bits(q.coverage_bytes),data(4*chunk_bytes),page(index_page_bytes),out(n);
        fill(input.span());ChunkWriter writer;StoreBuffers io{data.span(),page.span()};
        assert(!writer.open(t,r,io,image.span(),bits.span(),retained.span()));assert(!writer.write(0,input.span()));
        ScratchPlane plane;assert(!writer.finish_plane(plane,bits.span()));PlaneUpdate update;assert(!plane.begin_update(update));
        const size_t first=chunk_bytes-19,count=2*chunk_bytes+37;
        if(mode==1)retained.data[chunk_bytes+7]^=1;if(mode==2)disk.cancelled=true;
        EditCall call{&plane,&update};const auto before=t.counters;
        allocation_watch_start();const auto e=plane.edit_resident(update,first,count,*f.team,edit_action,&call);assert(!allocation_watch_stop());
        assert(t.counters.read_submitted==before.read_submitted&&t.counters.write_submitted==before.write_submitted);
        if(mode){assert(e.code==(mode==1?IoCode::corrupt:IoCode::cancelled)&&!call.called&&r.state==RegionState::invalid);plane.discard();continue;}
        assert(!e&&call.called==1);assert(!plane.finish(update));PlaneView v;assert(!plane.view(v));assert(!plane.read(v,0,out.span()));
        for(size_t i=0;i<n;++i)assert(out.data[i]==uint8_t(input.data[i]^((i>=first&&i<first+count)?0x5a:0)));
        assert(!plane.release());ValueReader reader;assert(!reader.open(t,r,io));assert(!reader.read(0,out.span()));
    }
    puts("Resident plane callback: guarded range, re-entry denial, refreshed persisted checksums, corruption/cancel before mutation PASS");
}
static void plane_cases(){
    const size_t n=records_per_page*chunk_bytes+101;
    auto r=fresh(n,0,7,RegionKind::scratch);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
    Aligned value(n),data(4*chunk_bytes),page(index_page_bytes),output(n);fill(value.span());
    StoreBuffers io{data.span(),page.span()};write_value(t,r,io,value.span());
    const std::vector<uint8_t> old_index(disk.bytes.begin()+r.layout.index_offset,disk.bytes.begin()+r.layout.index_offset+r.layout.index_bytes);
    Aligned stale_data(chunk_bytes),stale_page(index_page_bytes);ValueReader old_reader;
    assert(!old_reader.open(t,r,{stale_data.span(),stale_page.span()}));
    PlaneIndexStorage need{};assert(!plane_index_storage(r.layout,need));Aligned image(need.image_bytes),bits(need.dirty_bytes);
    ScratchPlane plane;allocation_watch_start();assert(!plane.take(t,r,io,image.span(),bits.span()));assert(!allocation_watch_stop());
    assert(r.state==RegionState::scratch_owned);
    ScratchPlane duplicate;assert(duplicate.take(t,r,io,image.span(),bits.span()).code==IoCode::invalid);
    ValueReader stale_disk;assert(stale_disk.open(t,r,io).code==IoCode::incomplete);
    PlaneView old{};assert(!plane.view(old));assert(!plane.read(old,0,output.span()));assert(!memcmp(value.data,output.data,n));
    PlaneUpdate update;assert(!plane.begin_update(update));assert(plane.read(old,0,output.span().first(1)).code==IoCode::stale);
    std::array<uint8_t,71> patch{};fill(patch);const uint64_t before_read=t.counters.read_submitted,before_write=t.counters.write_submitted;
    allocation_watch_start();assert(!plane.patch(update,chunk_bytes-19,patch));assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-before_read==2*chunk_bytes && t.counters.write_submitted-before_write==2*chunk_bytes);
    std::memcpy(value.data+chunk_bytes-19,patch.data(),patch.size());
    // A complete replacement skips its old read; final short chunk is padded.
    fill(output.span().first(2*chunk_bytes));auto before=t.counters;
    assert(!plane.patch(update,4*chunk_bytes,output.span().first(2*chunk_bytes)));
    assert(t.counters.read_submitted==before.read_submitted);
    std::memcpy(value.data+4*chunk_bytes,output.data,2*chunk_bytes);
    before=t.counters;assert(!plane.patch(update,n-101,output.span().first(101)));
    assert(t.counters.read_submitted==before.read_submitted && t.counters.write_submitted-before.write_submitted==chunk_bytes);
    std::memcpy(value.data+n-101,output.data,101);
    assert(!plane.finish(update,false));
    assert(!memcmp(old_index.data(),disk.bytes.data()+r.layout.index_offset,old_index.size()));
    PlaneView current{};assert(!plane.view(current));before=t.counters;
    assert(!plane.read(current,0,output.span()));assert(!memcmp(value.data,output.data,n));
    assert(t.counters.read_submitted-before.read_submitted==r.layout.data_bytes); // authoritative RAM index
    disk.corrupt_at=chunk_bytes+20;disk.corrupt_reads=1;before=t.counters;
    assert(!plane.read(current,0,output.span()));assert(!memcmp(value.data,output.data,n));
    assert(t.counters.read_submitted-before.read_submitted==r.layout.data_bytes+chunk_bytes);
    disk.corrupt_reads=0; // even retry used current RAM expectations, not the stale disk index
    before=t.counters;assert(!plane.release());assert(r.state==RegionState::sealed);
    assert(t.counters.write_submitted-before.write_submitted==r.layout.index_bytes);
    assert(plane.read(current,0,output.span().first(1)).code==IoCode::stale);
    assert(old_reader.read(0,output.span().first(1)).code==IoCode::stale);
    ValueReader reader;assert(!reader.open(t,r,io));assert(!reader.read(0,output.span()));assert(!memcmp(value.data,output.data,n));
    // Re-take, modify, and consume without paying an index writeback.
    assert(!plane.take(t,r,io,image.span(),bits.span()));PlaneUpdate next;assert(!plane.begin_update(next));
    assert(!plane.patch(next,0,output.span().first(chunk_bytes)));assert(!plane.finish(next));before=t.counters;plane.discard();
    assert(r.state==RegionState::invalid && t.counters.write_submitted==before.write_submitted);
    auto immutable=fresh(1,r.layout.allocation_bytes,9);Device vdisk(size_t(immutable.layout.data_offset+immutable.layout.allocation_bytes));auto vt=vdisk.transport();
    write_value(vt,immutable,io,value.span().first(1));
    ScratchPlane rejected;assert(rejected.take(vt,immutable,io,image.span(),bits.span()).code==IoCode::invalid);
    puts("Scratch plane: guarded patches, stale views, current RAM indexes, release and consumed discard PASS");
}
static void plane_failure(){
    auto r=fresh(2*chunk_bytes,0,1,RegionKind::scratch);Device disk(size_t(r.layout.allocation_bytes));auto t=disk.transport();
    Aligned value(2*chunk_bytes),buf(2*chunk_bytes),page(index_page_bytes);fill(value.span());StoreBuffers io{buf.span(),page.span()};
    write_value(t,r,io,value.span());PlaneIndexStorage q{};assert(!plane_index_storage(r.layout,q));Aligned image(q.image_bytes),bits(q.dirty_bytes);
    ScratchPlane plane;assert(!plane.take(t,r,io,image.span(),bits.span()));PlaneUpdate u;assert(!plane.begin_update(u));
    disk.fail_write=disk.writes+1;assert(plane.patch(u,0,value.span()).code==IoCode::io);assert(r.state==RegionState::invalid);
    assert(plane.finish(u).code==IoCode::io);plane.discard();
    disk.fail_write=0;auto next_id=r.id;++next_id.incarnation;
    assert(!begin_region(r,next_id,r.index_instance+1,r.layout,RegionKind::scratch));write_value(t,r,io,value.span());
    assert(!plane.take(t,r,io,image.span(),bits.span()));PlaneUpdate next;assert(!plane.begin_update(next));
    assert(!plane.patch(next,0,value.span().first(chunk_bytes)));assert(!plane.finish(next));
    disk.fail_write=disk.writes+1;assert(plane.release().code==IoCode::io && r.state==RegionState::invalid);plane.discard();
}
static void stored_rows(bool resident_prefix=false){
    Fixture f(4,false,size_t(64)<<20);constexpr size_t n=8192,rows=2,prefix=19;
    ModularRowSpec spec{RowOperation::product,5,15,4,n,1,1,1};spec.local_tau=gamma_of(spec);
    ModularRowInfo info{};assert(p48_row_query(spec,info));
    const size_t bytes=prefix+rows*n*48+17;
    auto a=fresh(bytes,0,1,RegionKind::scratch),b=fresh(bytes,a.layout.allocation_bytes,2,RegionKind::scratch);
    Device disk(size_t(b.layout.data_offset+b.layout.allocation_bytes));auto t=disk.transport();
    Aligned av(info.array_bytes),bv(info.array_bytes),prepared(info.prepared_bytes),scratch(info.scratch_bytes);
    Aligned wire_a(bytes),wire_b(bytes),io_a(4*chunk_bytes),io_b(4*chunk_bytes),page_a(index_page_bytes),page_b(index_page_bytes);
    fill(wire_a.span());fill(wire_b.span());
    std::array<std::vector<uint64_t>,rows> expected;
    auto *ap=reinterpret_cast<uint64_t *>(av.data),*bp=reinterpret_cast<uint64_t *>(bv.data);
    const uint64_t p=p48_row_prime(spec.prime);
    for(size_t row=0;row<rows;++row){
        spec.global_prefix=2-row;spec.local_tau=gamma_of(spec);p48_row_prepare(spec,info,prepared.span());
        for(size_t i=0;i<n*8;++i){ap[i]=(i*157+row+3)%p;bp[i]=0;}bp[0]=3;bp[7]=5;bp[n*8-1]=11;
        expected[row]=reference(spec,ap,bp,true);
        p48_row_pack(prepared.span(),{ap,n*8},wire_a.span().subspan(prefix+row*n*48,n*48));
        p48_row_pack(prepared.span(),{bp,n*8},wire_b.span().subspan(prefix+row*n*48,n*48));
    }
    StoreBuffers ai{io_a.span(),page_a.span()},bi{io_b.span(),page_b.span()};write_value(t,a,ai,wire_a.span());
    if(!resident_prefix)write_value(t,b,bi,wire_b.span());
    const auto preserved_a=std::vector<uint8_t>(disk.bytes.begin(),disk.bytes.begin()+a.layout.allocation_bytes);
    PlaneIndexStorage need{};assert(!plane_index_storage(a.layout,need));Aligned axi(need.image_bytes),bxi(need.image_bytes),ad(need.dirty_bytes),bd(need.dirty_bytes);
    ScratchPlane A,B;assert(!A.take(t,a,ai,axi.span(),ad.span()));
    ChunkWriterStorage cq{};assert(!chunk_writer_storage(b.layout,cq));Aligned coverage(cq.coverage_bytes),retained(up(prefix+n*48,chunk_bytes));
    ChunkWriter writer;
    if(resident_prefix){assert(!writer.open(t,b,bi,bxi.span(),coverage.span(),retained.span()));
        assert(!writer.write(0,wire_b.span()));assert(!writer.finish_plane(B,bd.span()));}
    else assert(!B.take(t,b,bi,bxi.span(),bd.span()));
    MiddleRowsSpec ms{spec.prime,spec.global_log,4,n,1,rows,prefix,n*48,1,false};
    MiddleRowsPlan plan{};assert(query_middle_rows(ms,plan));
    const auto region=f.allocate(plan.storage_bytes,64);
    std::span<uint8_t> workspace(static_cast<uint8_t *>(region.data),region.bytes);
    allocation_watch_start();
    assert(!(resident_prefix?run_middle_memory_a(plan,wire_a.span(),B,workspace,*f.team):run_middle_rows(plan,A,B,workspace,*f.team)));
    assert(!allocation_watch_stop());
    PlaneView product{};assert(!B.view(product));
    for(size_t row=0;row<rows;++row){
        assert(!B.read(product,prefix+row*n*48,bv.span().first(n*48)));
        p48_row_unpack(prepared.span(),bv.span(),{bp,n*8});assert(std::equal(expected[row].begin(),expected[row].end(),bp));
    }
    std::array<uint8_t,prefix> guard{};assert(!B.read(product,0,guard));assert(!memcmp(guard.data(),wire_b.data,prefix));
    assert(!memcmp(preserved_a.data(),disk.bytes.data(),preserved_a.size()));
    ms.square=true;MiddleRowsPlan sq{};assert(query_middle_rows(ms,sq));
    assert(sq.storage_bytes+sq.row.array_bytes==plan.storage_bytes);
    allocation_watch_start();assert(!run_middle_rows(sq,A,A,workspace,*f.team));assert(!allocation_watch_stop());
    PlaneView square{};assert(!A.view(square));
    const uint64_t gen=primitive(p),root=power(gen,(p-1)/(8*n),p);
    for(size_t row=0;row<rows;++row){
        assert(!A.read(square,prefix+row*n*48,av.span().first(n*48)));p48_row_unpack(prepared.span(),av.span(),{ap,n*8});
        const uint64_t x0=power(gen,((p-1)>>(spec.global_log+3))*(2-row),p);
        for(unsigned k:{7u,123u,5017u}){
            const uint64_t x=mm(x0,power(root,k,p),p);uint64_t input=0,output=0;
            for(size_t i=n*8;i-- >0;){input=(mm(input,x,p)+(i*157+row+3)%p)%p;output=(mm(output,x,p)+ap[i])%p;}
            assert(output==mm(input,input,p));
        }
    }
    assert(!B.release());A.discard();
    puts("Stored p48 middle rows: verified packed reads, shared-array native transform, B overwrite and preserved guards PASS");
}
int main(){
    plane_cases();plane_failure();resident_edit();
    {
    Fixture f(4,false,size_t(64)<<20);
    for(unsigned q:{0u,5u,9u})for(size_t n:{size_t(1),size_t(2),size_t(4),size_t(8),size_t(16),size_t(32)}){
        ModularRowSpec s{RowOperation::product,q,unsigned(__builtin_ctzll(n))+2,4,n,3,1,17};
        row_test(f,s,false);s.local_tau=gamma_of(s);row_test(f,s,false);row_test(f,s,true);
    }
    for(size_t n:{size_t(8192),size_t(32768),size_t(131072)}){
        ModularRowSpec s{RowOperation::product,9,32,4,n,17,3,1};row_test(f,s,false,true);unary(f,n);
    }
    packed_rows(f,8192,4,false);packed_rows(f,32768,3,false);packed_rows(f,32768,4,true);
    for(unsigned frontiers:{1u,2u,3u})packed_rows(f,32768,3,false,frontiers);
    packed_rows(f,32768,4,true,3);
    packed_unary_guard(f,8192,2);packed_unary_guard(f,32768,3);
    ModularRowSpec top{RowOperation::product,9,32,1,1,UINT32_MAX,3,0};row_test(f,top,false);
    ModularRowInfo bound{};top.cells=size_t(1)<<32;top.global_prefix=0;top.workers=16;top.product_scale=1;
    assert(p48_row_query(top,bound) && bound.scratch_bytes<1024*1024 && bound.prepared_bytes<16384);
    }
    stored_rows();stored_rows(true);
    puts("OOC planes/rows: independent polynomial products, scaled/SQR/coset roots, parallel inverse and allocation gates PASS");
}
