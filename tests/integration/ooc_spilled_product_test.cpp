// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/spilled_product.hpp"
using namespace sbn::v3::external;
static void one(SpilledProductSpec spec,bool ones=false,unsigned fail=0){
    SpilledProductPlan plan{};allocation_watch_start();assert(query_spilled_product(spec,plan));assert(!allocation_watch_stop());
    auto a=fresh(spec.an*8,0,1),b=fresh(spec.bn*8,a.layout.allocation_bytes,2);
    uint64_t end=b.layout.data_offset+b.layout.allocation_bytes;
    std::array<Region,20> temp{};
    for(unsigned i=0;i<plan.plane_count;++i){temp[i]=fresh(size_t(plan.plane_bytes),end,100+i,RegionKind::scratch);end+=temp[i].layout.allocation_bytes;}
    auto result=fresh(size_t(plan.output_bytes),end,1000);end+=result.layout.allocation_bytes;
    Device disk{size_t(end)};auto t=disk.transport();Aligned av(spec.an*8),bv(spec.bn*8),io_data(4*chunk_bytes),index(index_page_bytes),output(size_t(plan.output_bytes));
    if(spec.stream_final){t.parallel_reads=spec.workers>=8?4:2;t.duplex=true;}
    fill(av.span());fill(bv.span());if(ones){std::memset(av.data,255,av.bytes);std::memset(bv.data,255,bv.bytes);}
    else for(size_t j=0;j<bv.bytes;++j)bv.data[j]^=0xb1;
    StoreBuffers io{io_data.span(),index.span()};write_value(t,a,io,av.span());write_value(t,b,io,bv.span());
    const std::vector<uint8_t> originals(disk.bytes.begin(),disk.bytes.begin()+b.layout.data_offset+b.layout.allocation_bytes);
    Fixture fixture(spec.workers,false,size_t(256)<<20);assert(plan.storage_bytes<(size_t(64)<<20));
    const auto workspace=fixture.allocate(plan.storage_bytes,4096);
    const auto before=t.counters;if(fail==1)disk.fail_write=disk.writes+2;
    if(fail==2)disk.fail_write_at=result.layout.data_offset+2*chunk_bytes;
    if(fail==3)disk.cancel_write_at=result.layout.data_offset;
    if(fail==4)disk.arm_read_failure_at=result.layout.data_offset;
    if(fail==5)disk.fail_read_at=(spec.raw_operand?b:a).layout.data_offset+3*plan.raw_forward.kernel.stripe_cells*spec.trunk_bits+chunk_bytes;
    allocation_watch_start();const auto rc=run_spilled_product(plan,t,a,spec.square?a:b,result,{temp.data(),plan.plane_count},
        {static_cast<uint8_t *>(workspace.data),workspace.bytes},*fixture.team);assert(!allocation_watch_stop());
    assert(!memcmp(originals.data(),disk.bytes.data(),originals.size()));
    if(spec.retained_bytes && !fail){
        auto full=spec;full.retained_bytes=0;SpilledProductPlan baseline{};assert(query_spilled_product(full,baseline));
        assert(baseline.normal_write_bytes-plan.normal_write_bytes==2*spec.retained_bytes);
        assert(baseline.normal_read_bytes-plan.normal_read_bytes>=2*spec.retained_bytes);
        const unsigned rhs=spec.square||spec.raw_operand<2?0:spec.primes;
        for(unsigned q=0;q<spec.primes;++q){const uint64_t start=q*plan.plane_bytes;
            const size_t n=start<spec.retained_bytes?size_t(std::min(plan.plane_bytes,uint64_t(spec.retained_bytes)-start)):0;
            const auto at=temp[rhs+q].layout.data_offset;
            assert(std::all_of(disk.bytes.begin()+at,disk.bytes.begin()+at+n,[](uint8_t b){return b==0xa5;}));
        }
    }
    if(fail){assert(rc.code==(fail==3?IoCode::cancelled:IoCode::io)&&result.state==RegionState::invalid);for(unsigned i=0;i<plan.plane_count;++i)assert(temp[i].state==RegionState::invalid);return;}
    assert(!rc&&result.state==RegionState::sealed);
    const auto reads=t.counters.read_submitted-before.read_submitted,writes=t.counters.write_submitted-before.write_submitted;
    assert(reads==plan.normal_read_bytes&&writes==plan.normal_write_bytes);
    for(unsigned i=0;i<plan.plane_count;++i)assert(temp[i].state==RegionState::invalid);
    ValueReader reader;assert(!reader.open(t,result,io));assert(!reader.read(0,output.span()));
    assert(ref_product_equal(reinterpret_cast<uint64_t *>(av.data),spec.an,
        reinterpret_cast<uint64_t *>(spec.square?av.data:bv.data),spec.bn,reinterpret_cast<uint64_t *>(output.data),spec.an+spec.bn));
    printf("Spilled %s NP%u T%u %zux%zu W%u: private=%zu shared=%zu R/W=%llu/%llu exact ledger and result PASS\n",
        spec.square?"SQR":"MUL",spec.primes,spec.trunk_bits,spec.an,spec.bn,spec.workers,plan.storage_bytes,plan.shared_bytes,
        (unsigned long long)reads,(unsigned long long)writes);
}
static void middle_one(SpilledProductSpec spec,unsigned pattern){
    SpilledProductPlan plan;assert(query_spilled_product(spec,plan));
    assert(project_spilled_output(plan,spec.middle.first&~(chunk_bytes/8-1),spec.middle.end));
    auto a=fresh(spec.an*8,0,1),b=fresh(spec.bn*8,a.layout.allocation_bytes,2);
    uint64_t end=b.layout.data_offset+b.layout.allocation_bytes;std::array<Region,20> temp{};
    for(unsigned i=0;i<plan.plane_count;++i){temp[i]=fresh(size_t(plan.plane_bytes),end,100+i,RegionKind::scratch);end+=temp[i].layout.allocation_bytes;}
    auto result=fresh(size_t(plan.stored_output_bytes),end,1000);end+=result.layout.allocation_bytes;
    Device disk{size_t(end)};auto t=disk.transport();t.parallel_reads=2;t.duplex=true;
    std::vector<uint64_t> av(spec.an),bv(spec.bn),exact(spec.an+spec.bn);
    for(auto&v:av)v=pattern?UINT64_MAX:random_word();for(auto&v:bv)v=pattern?UINT64_MAX:random_word();
    if(pattern==2){std::fill(bv.begin(),bv.end(),0);bv.front()=bv.back()=1;}
    if(pattern==3)std::fill(av.begin(),av.end(),0);
    ref_int x,y;ref_inits(x,y,nullptr);ref_import(x,av.size(),-1,8,0,0,av.data());ref_import(y,bv.size(),-1,8,0,0,bv.data());
    ref_mul(x,x,y);size_t count;ref_export(exact.data(),&count,-1,8,0,0,x);ref_clears(x,y,nullptr);
    Aligned transfer(4*chunk_bytes),index(index_page_bytes);StoreBuffers io{transfer.span(),index.span()};
    write_value(t,a,io,{reinterpret_cast<uint8_t*>(av.data()),av.size()*8});
    write_value(t,b,io,{reinterpret_cast<uint8_t*>(bv.data()),bv.size()*8});
    Fixture f(spec.workers,false,64u<<20);auto work=f.allocate(plan.storage_bytes,4096);MiddleWindowResult middle;
    const auto before=t.counters;allocation_watch_start();
    assert(!run_spilled_product(plan,t,a,b,result,{temp.data(),plan.plane_count},{static_cast<uint8_t*>(work.data),work.bytes},*f.team,nullptr,nullptr,nullptr,&middle));
    assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-before.read_submitted==plan.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==plan.normal_write_bytes);
    if(pattern==2||pattern==3)assert(!middle.exact);else assert(middle.exact);
    std::vector<uint64_t> got(size_t(plan.stored_output_bytes/8));ValueReader reader;assert(!reader.open(t,result,io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t*>(got.data()),got.size()*8}));
    const size_t start=spec.middle.first-size_t(plan.output_first_byte/8);
    if(middle.exact)assert(!std::memcmp(got.data()+start,exact.data()+spec.middle.first,(spec.middle.end-spec.middle.first)*8));
    if(pattern==2){assert(got[start]==exact[spec.middle.first]+1);}
    printf("Cyclic middle NP%u T%u %zux%zu K%llu [%zu,%zu) pattern%u exact%u: independent product and exact IO ledger PASS\n",
        spec.primes,spec.trunk_bits,spec.an,spec.bn,(unsigned long long)(plan.output_bytes/8),spec.middle.first,spec.middle.end,pattern,unsigned(middle.exact));
}
int main(){
    for(unsigned raw:{0u,1u,2u})for(unsigned pattern:{0u,1u,2u,3u}){
        SpilledProductSpec s{2000,1001,16,16,0,4,80,2,4,false};s.middle={800,2000};s.raw_operand=raw;middle_one(s,pattern);
    }
    for(unsigned raw:{0u,1u,2u}){
        SpilledProductSpec s{48001,26003,1,4096,1024,6,128,2,4,false};
        s.middle={30001,48001};s.raw_operand=raw;s.stream_final=true;
        if(raw<2)s.raw_cache_bytes=3*128;middle_one(s,0);
    }
    one({5,6,1,1,0,4,80,2,1,false},true); // output extends past coefficient cells: final carry
    one({257,173,8,16,0,4,80,2,4,false});
    one({501,601,8,32,0,5,104,3,4,false});
    one({20001,13007,4,2048,0,6,128,3,4,false});
    one({30013,17011,4,2048,0,8,176,4,4,false},true);
    one({65009,52003,8,2048,0,10,224,4,4,false});
    one({20003,20003,4,2048,0,6,128,3,4,true},true);
    one({257,173,8,16,0,4,80,2,4,false},false,true);
    one({257,173,8,16,0,4,80,2,4,false,5*chunk_bytes});
    one({20001,13007,4,2048,0,6,128,3,4,false,15*chunk_bytes});
    one({20003,20003,4,2048,0,6,128,3,4,true,24*chunk_bytes},true);
    for(unsigned operand:{0u,1u}){
        SpilledProductSpec tiny{257,173,8,16,0,4,80,2,4,false};tiny.raw_operand=operand;
        one(tiny,true);tiny.retained_bytes=5*chunk_bytes;one(tiny);one(tiny,false,true);
        SpilledProductSpec wide{30013,17011,4,2048,0,8,176,4,4,false};wide.raw_operand=operand;
        one(wide,true);wide.retained_bytes=15*chunk_bytes;one(wide);
        SpilledProductPlan p{};assert(query_spilled_product(wide,p));
        auto q=p;assert(use_cached_a(q)&&q.spec.raw_operand==2&&q.cached_a);
    }
    for(bool square:{false,true})for(unsigned workers:{2u,4u}){
        SpilledProductSpec streamed{20003,square?20003u:13007u,4,2048,1024,6,128,2,workers,square};
        streamed.stream_final=true;one(streamed,true);one(streamed,false,2);
        if(!square){streamed.raw_operand=0;one(streamed);}
    }
    for(unsigned np:{4u,8u,10u}){
        const unsigned T=np==4?80:24*np-16;
        SpilledProductSpec streamed{65009,52003,16,2048,1024,np,T,2,4,false};
        streamed.stream_final=true;one(streamed,true);one(streamed);
    }
    SpilledProductSpec recycled{150003,130007,8,8192,1024,6,128,2,4,false};recycled.stream_final=true;
    one(recycled,true);one(recycled,false,2);one(recycled,false,3);one(recycled,false,4);
    auto wide=recycled;wide.forward_stripe_cells=4096;wide.workers=8;
    one(wide,true);one(wide,false,2);wide.raw_operand=0;one(wide);wide.raw_operand=1;one(wide);
    for(unsigned operand:{0u,1u})for(size_t cached:{size_t(0),size_t(3*128),size_t(900*128)}){
        auto prefix=wide;prefix.raw_operand=operand;prefix.raw_cache_bytes=cached;
        one(prefix,true);one(prefix,false,1);
    }
    auto broken=wide;broken.raw_operand=0;broken.raw_cache_bytes=3*128;one(broken,false,5);
    // Final IO width is independent of the raw-source staging width. Exercise
    // both two-bank and three-bank lifetimes, partial raw prefixes and late IO
    // failure/cancel after output has started. Exact bytes remain unchanged.
    for(size_t M:{size_t(8192),size_t(16384)}){
        auto gathered=wide;gathered.inner_cells=M;gathered.inverse_stripe_cells=4096;
        one(gathered,true);one(gathered,false,2);one(gathered,false,3);
        if(M/4096>3)one(gathered,false,4); // another input read must follow the first emitted slot
        gathered.raw_cache_bytes=3*128;one(gathered,true);
        gathered.retained_bytes=15*chunk_bytes;one(gathered,true);
        gathered.raw_stripe_cells=2048;one(gathered,true);one(gathered,false,1);
        auto narrow=gathered;narrow.inverse_stripe_cells=0;SpilledProductPlan a{},b{};
        assert(query_spilled_product(narrow,a));b=a;
        assert(widen_inverse_within(b,4096,SIZE_MAX));
        assert(b.normal_read_bytes==a.normal_read_bytes&&b.normal_write_bytes==a.normal_write_bytes&&
               b.raw_cached_bytes==a.raw_cached_bytes&&b.raw_forward.kernel.stripe_cells==a.raw_forward.kernel.stripe_cells);
        const auto bytes=b.storage_bytes;assert(!widen_inverse_within(b,M,SIZE_MAX)&&b.storage_bytes==bytes);
        assert(!widen_inverse_within(b,4096,bytes-1)&&b.storage_bytes==bytes);
    }
    auto tiny_prefix=SpilledProductSpec{257,173,8,16,0,4,80,2,4,false};
    tiny_prefix.raw_operand=0;tiny_prefix.raw_cache_bytes=3*80;one(tiny_prefix,true);
    puts("Complete spilled integer products: exact CRT bound, three passes, carry-safe write-once output, cleanup and allocation gates PASS");
}
