// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/newton.hpp"
#include "external/owned_product.hpp"
#include "common/m61.hpp"
#include <bit>
using namespace sbn::v3;
static unsigned cyclic_rungs=0;
static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v){
    RegionLease r;StoredInteger out;assert(!s.reserve(v.size()*8,RegionKind::value,r));ValueWriter w;
    assert(!w.open(t,r.region(),io,&out.extent));assert(!w.append({reinterpret_cast<const uint8_t *>(v.data()),v.size()*8}));assert(!w.seal());
    assert(!r.publish(out.value));out.extent_known=true;return out;
}
static void source_views(){
    constexpr size_t words=6007;std::vector<uint64_t> input(words);for(auto&v:input)v=random_word();
    auto region=fresh(words*8);Device disk(size_t(region.layout.allocation_bytes));auto t=disk.transport();
    Aligned io_data(4*chunk_bytes),page(index_page_bytes),output(4101*8+16);StoreBuffers io{io_data.span(),page.span()};
    write_value(t,region,io,{reinterpret_cast<const uint8_t *>(input.data()),input.size()*8});ValueReader reader;assert(!reader.open(t,region,io));
    for(int shift:{-63,-17,-1,0,1,17,63})for(size_t first:{size_t(3),size_t(2048),size_t(2049)})for(size_t zeros:{size_t(0),size_t(17),size_t(2048),size_t(18000)})for(bool complement:{false,true}){
        SourceWindow s{words,first,3001,3,words-5,shift,zeros};assert(resolve_source_window(3001,s));
        s.ones_complement=complement;
        allocation_watch_start();assert(!read_source_words(reader,s,0,3001,output.span()));assert(!allocation_watch_stop());
        const auto*p=reinterpret_cast<const uint64_t *>(output.data);
        auto at=[&](__int128 j){return j>=s.lower_limb&&j<s.upper_limb?(complement?~input[size_t(j)]:input[size_t(j)]):uint64_t(0);};
        for(size_t j=0;j<3001;++j){const __int128 k=__int128(s.first_limb)+j-s.low_zero_words;const uint64_t v=shift>0?(at(k)<<shift)|(at(k-1)>>(64-shift)):
            shift<0?(at(k)>>-shift)|(at(k+1)<<(64+shift)):at(k);assert(p[j]==v);}
        if(shift||s.low_zero_words)assert(!p[3001]&&!p[3002]);
    }
    puts("Stored source shifts: both directions, bounded neighbours and padding guards PASS");
}
static void shifted_product(size_t zeros,bool partial=false,bool complement=false){
    std::vector<uint64_t>a(60001),b(52003),x(20007),y(17001);for(auto*v:{&a,&b})for(auto&word:*v)word=random_word();
    SourceWindow as{a.size(),2048,x.size(),100,40000,17,zeros},bs{b.size(),1000,y.size(),100,40000,-3};
    as.ones_complement=complement;
    auto at=[&](__int128 index){return index>=as.lower_limb&&index<as.upper_limb?(complement?~a[size_t(index)]:a[size_t(index)]):uint64_t(0);};
    for(size_t i=0;i<x.size();++i){const __int128 k=__int128(as.first_limb)+i-zeros;x[i]=(at(k)<<17)|(at(k-1)>>47);}
    for(size_t i=0;i<y.size();++i)y[i]=(b[bs.first_limb+i]>>3)|(b[bs.first_limb+i+1]<<61);
    ValueStoreStorage q{};assert(query_value_store(32,q));Aligned metadata(q.bytes),data(4*chunk_bytes),page(index_page_bytes);
    Device disk(16u<<20);auto t=disk.transport();StoreBuffers io{data.span(),page.span()};ValueStore store;
    assert(!store.open(303,403,0,disk.bytes.size(),metadata.span(),32));auto av=upload(store,t,io,a),bv=upload(store,t,io,b);StoredInteger out;
    SpilledProductSpec spec{x.size(),y.size(),4,2048,1024,6,128,3,4,false,8*chunk_bytes,as,bs};SpilledProductPlan p{};
    if(partial){spec.raw_operand=0;spec.raw_cache_bytes=13*128;}
    assert(query_spilled_product(spec,p));Fixture f(4,false,32u<<20);auto memory=f.allocate(p.storage_bytes,4096);const auto before=t.counters;
    allocation_watch_start();assert(!run_owned_product(p,t,store,av.value,bv.value,out.value,{static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,&out.extent));assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-before.read_submitted==p.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==p.normal_write_bytes);
    std::vector<uint64_t>result(x.size()+y.size());ValueReader reader;assert(!reader.open(t,out.value.region(),io));assert(!reader.read(0,{reinterpret_cast<uint8_t *>(result.data()),result.size()*8}));
    assert(ref_product_equal(x.data(),x.size(),y.data(),y.size(),result.data(),result.size()));
    printf("Shifted multi-stripe product zeros%zu: bounded bit views, full m61 and exact physical guard IO PASS\n",zeros);
}
static void cyclic_residual(int perturb,unsigned np=6,unsigned T=128,size_t m=8193,size_t n=16001,bool streamed=false){
    const size_t origin=m-2,rn=n-m+3;
    std::vector<uint64_t>d(n),u(m+1);for(auto&x:d)x=random_word();d.back()|=uint64_t(1)<<63;
    ref_int D,H,U,power,error,expected,got;ref_inits(D,H,U,power,error,expected,got,nullptr);
    ref_import(D,n,-1,8,0,0,d.data());ref_fdiv_q_2exp(H,D,64*(n-m));
    ref_set_ui(power,1);ref_mul_2exp(power,power,128*m);ref_sub_ui(power,power,1);ref_fdiv_q(U,power,H);
    if(perturb>0)ref_add_ui(U,U,unsigned(perturb));else ref_sub_ui(U,U,unsigned(-perturb));
    size_t used=0;ref_export(u.data(),&used,-1,8,0,0,U);assert(used==m+1);
    ref_mul(error,U,D);ref_set_ui(power,1);ref_mul_2exp(power,power,64*(n+m));ref_sub(error,error,power);
    const bool negative=ref_sgn(error)<0;ref_abs(error,error);
    ref_fdiv_q_2exp(expected,error,64*origin);ref_fdiv_r_2exp(expected,expected,64*rn);
    ValueStoreStorage q{};assert(query_value_store(32,q));Aligned metadata(q.bytes),io_data(4*chunk_bytes),page(index_page_bytes),raw(rn*8+16);
    Device disk(16u<<20);auto t=disk.transport();StoreBuffers io{io_data.span(),page.span()};ValueStore store;
    assert(!store.open(505,605,0,disk.bytes.size(),metadata.span(),32));auto av=upload(store,t,io,u),bv=upload(store,t,io,d);StoredValue result;
    const size_t cells=std::bit_ceil(((n+2)*8+T-1)/T),inner=cells/4;
    SpilledProductSpec spec{m+1,n,4,inner,streamed?1024:inner,np,T,3,4};spec.cyclic={n+2,n+m,{n,16,true}};
    spec.stream_final=streamed;if(streamed){t.duplex=true;t.parallel_reads=2;}
    SpilledProductPlan plan;assert(query_spilled_product(spec,plan));assert(plan.output_bytes/8<n+m);
    const size_t first=origin&~(chunk_bytes/8-1);assert(project_spilled_output(plan,first,origin+rn));
    Fixture f(4,false,32u<<20);auto memory=f.allocate(plan.storage_bytes,4096);const auto before=t.counters;CyclicResidualResult state;
    allocation_watch_start();assert(!run_owned_product(plan,t,store,av.value,bv.value,result,
        {static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,nullptr,&state));assert(!allocation_watch_stop());
    assert(state.complemented==negative&&state.magnitude.residue_m61==ref_fdiv_ui(error,m61::modulus));
    assert(t.counters.read_submitted-before.read_submitted==plan.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==plan.normal_write_bytes);
    ValueReader reader;assert(!reader.open(t,result.region(),io));SourceWindow view{size_t(plan.stored_output_bytes/8),origin-first,rn};view.ones_complement=state.complemented;
    assert(!read_source_words(reader,view,0,rn,raw.span()));ref_import(got,rn,-1,8,0,0,raw.data);assert(ref_cmp(got,expected)==0);
    ref_clears(D,H,U,power,error,expected,got,nullptr);
    printf("Cyclic cancellation NP%u T%u perturb%d: wrapped product, signed exact residual window, projected output and IO ledger PASS\n",np,T,perturb);
}
static void inverse(size_t m,size_t n,unsigned shift,int perturb){
    const size_t original=n+17;std::vector<uint64_t>d(original),u(m+1);
    for(auto&v:d)v=random_word();d.back()=(random_word()>>shift)|(uint64_t(1)<<(63-shift));
    ref_int D,head,U,power,got,error,bound;ref_inits(D,head,U,power,got,error,bound,nullptr);
    ref_import(D,d.size(),-1,8,0,0,d.data());ref_mul_2exp(D,D,shift);ref_fdiv_q_2exp(D,D,64*(original-n));
    ref_fdiv_q_2exp(head,D,64*(n-m));ref_set_ui(power,1);ref_mul_2exp(power,power,128*m);ref_sub_ui(power,power,1);
    ref_fdiv_q(U,power,head);if(perturb>0)ref_add_ui(U,U,unsigned(perturb));else if(perturb<0)ref_sub_ui(U,U,unsigned(-perturb));
    size_t used=0;ref_export(u.data(),&used,-1,8,0,0,U);assert(used==m+1&&u[m]==1);
    ValueStoreStorage metadata_plan{};assert(query_value_store(64,metadata_plan));Aligned metadata(metadata_plan.bytes),io_data(4*chunk_bytes),page(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{io_data.span(),page.span()};ValueStore store;
    assert(!store.open(301,401,0,disk.bytes.size(),metadata.span(),64));auto divisor=upload(store,t,io,d),old=upload(store,t,io,u);
    SourceWindow source;assert(normalized_source(divisor,n,source)&&source.bit_shift==int(shift));
    InverseRungPlan p{};allocation_watch_start();const auto status=query_inverse_rung(m,n,source,4,{16u<<20,4u<<20,64u<<20},{},p);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"inverse query %zu/%zu status%u\n",m,n,unsigned(status));
    assert(status==planning::ExternalPlanStatus::ready);
    if(p.residual_product.spec.cyclic)++cyclic_rungs;
    Fixture f(4,false,32u<<20);const auto memory=f.allocate(p.storage_bytes,4096);StoredInteger out;bool clamped=false;
    const auto before=t.counters;allocation_watch_start();const auto rc=run_inverse_rung(p,t,store,divisor,old,out,{static_cast<uint8_t *>(memory.data),memory.bytes},*f.team,&clamped);assert(!allocation_watch_stop());
    if(rc)fprintf(stderr,"inverse %zu/%zu error %u %s\n",m,n,unsigned(rc.code),rc.where);assert(!rc);
    assert(t.counters.write_submitted-before.write_submitted==(clamped?p.worst_write_bytes:p.write_bytes));assert(store.usage().peak_bytes<=p.peak_backing_bytes);
    std::vector<uint64_t> result(n+1);ValueReader reader;assert(!reader.open(t,out.value.region(),io));assert(!reader.read(0,{reinterpret_cast<uint8_t *>(result.data()),result.size()*8}));
    assert(result[n]==1);ref_import(got,result.size(),-1,8,0,0,result.data());ref_mul(error,got,D);
    ref_set_ui(power,1);ref_mul_2exp(power,power,128*n);ref_sub(error,error,power);ref_abs(error,error);ref_mul_ui(bound,D,3);
    assert(ref_cmp(error,bound)<0);
    const auto held=store.usage().live_bytes;out.value.reset();const auto after=store.usage().live_bytes;
    t.write_limit=t.counters.write_submitted+p.worst_write_bytes-1;
    assert(run_inverse_rung(p,t,store,divisor,old,out,{static_cast<uint8_t *>(memory.data),memory.bytes},*f.team).code==IoCode::capacity);
    assert(!out.value&&store.usage().live_bytes==after&&held>after);
    ref_clears(D,head,U,power,got,error,bound,nullptr);
    printf("Stored inverse m=%zu n=%zu shift=%u perturb=%d private=%zu cyclic=%u K=%llu clamped=%u: strict <3 ulps and write/backing bounds PASS\n",m,n,shift,perturb,p.storage_bytes,unsigned(bool(p.residual_product.spec.cyclic)),(unsigned long long)(p.residual_product.output_bytes/8),unsigned(clamped));
}
static void program(size_t n,unsigned shift,size_t max_seed,unsigned pattern=0){
    std::vector<uint64_t>d(n);for(auto&v:d)v=pattern==2?UINT64_MAX:random_word();
    d.back()=(d.back()>>shift)|(uint64_t(1)<<(63-shift));
    if(pattern==1){std::fill(d.begin(),d.end(),0);d.back()=uint64_t(1)<<(63-shift);}
    ValueStoreStorage q{};assert(query_value_store(96,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(302,402,0,disk.bytes.size(),metadata.span(),96));auto value=upload(store,t,io,d);
    auto plan=std::make_unique<InversePlan>();allocation_watch_start();
    const auto status=query_stored_inverse(value,n,{4,max_seed},{32u<<20,4u<<20,64u<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"inverse ladder query n%zu status%u\n",n,unsigned(status));
    assert(status==planning::ExternalPlanStatus::ready);
    if(max_seed==SIZE_MAX)assert(!plan->rung_count&&plan->seed_limbs==n);else assert(plan->rung_count&&plan->seed_limbs<=max_seed);
    Fixture f(4,false,32u<<20);auto memory=f.allocate(plan->storage_bytes,plan->alignment);
    const size_t at=static_cast<uint8_t *>(memory.data)-f.arena->base;
    f.leases.pop_back();sbn3_arena_release(f.arena,&memory);
    const auto before=t.counters;StoredInteger result;allocation_watch_start();
    auto error=run_stored_inverse(*plan,t,store,value,result,*f.arena,at,*f.team);assert(!allocation_watch_stop());
    if(error)fprintf(stderr,"inverse ladder n%zu error %u %s\n",n,unsigned(error.code),error.where);assert(!error);
    assert(f.arena->unleased(at,plan->storage_bytes));
    assert(t.counters.write_submitted-before.write_submitted<=plan->worst_write_bytes&&store.usage().peak_bytes<=plan->peak_backing_bytes);
    std::vector<uint64_t>u(n+1);ValueReader reader;assert(!reader.open(t,result.value.region(),io));assert(!reader.read(0,{reinterpret_cast<uint8_t *>(u.data()),u.size()*8}));
    ref_int D,U,power,diff,bound;ref_inits(D,U,power,diff,bound,nullptr);
    ref_import(D,n,-1,8,0,0,d.data());ref_mul_2exp(D,D,shift);ref_import(U,n+1,-1,8,0,0,u.data());assert(u[n]==1);
    ref_set_ui(power,1);ref_mul_2exp(power,power,128*n);ref_mul(diff,U,D);ref_sub(diff,diff,power);ref_abs(diff,diff);ref_mul_ui(bound,D,3);assert(ref_cmp(diff,bound)<0);
    ref_clears(D,U,power,diff,bound,nullptr);
    result.value.reset();const auto live=store.usage().live_bytes;disk.cancel_after_write=disk.writes+1;
    assert(run_stored_inverse(*plan,t,store,value,result,*f.arena,at,*f.team).code==IoCode::cancelled);
    assert(!result.value&&store.usage().live_bytes==live&&f.arena->unleased(at,plan->storage_bytes));disk.cancelled=false;
    printf("Stored reciprocal ladder n=%zu seed=%zu rungs=%zu shift=%u pattern=%u: <3 ulps, cap/lease/cancel checks PASS\n",n,plan->seed_limbs,plan->rung_count,shift,pattern);
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);source_views();shifted_product(0);shifted_product(8192);shifted_product(16384);
    shifted_product(0,true);shifted_product(8192,true);shifted_product(16384,true,true);
    cyclic_residual(5);cyclic_residual(-6);
    cyclic_residual(5,9,200,6145,12001);cyclic_residual(-6,10,224,7169,14001);
    cyclic_residual(5,6,128,131073,240001,true);cyclic_residual(-6,6,128,131073,240001,true);
    inverse(3,5,0,0);inverse(8,15,17,5);inverse(33,65,63,-6);inverse(129,255,1,0);inverse(4097,8191,17,5);
    program(127,17,SIZE_MAX);program(127,63,15);program(1023,0,15,1);program(1023,0,15,2);
    assert(cyclic_rungs);puts("Stored Newton: cyclic residuals, complemented correction, linear fallback and cached refinement PASS");
}
