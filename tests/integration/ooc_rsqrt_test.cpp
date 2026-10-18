// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/newton.hpp"
#include "external/owned_product.hpp"
using namespace sbn::v3;
static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v){
    RegionLease region;StoredInteger result;assert(!s.reserve(v.size()*8,RegionKind::value,region));ValueWriter writer;
    assert(!writer.open(t,region.region(),io,&result.extent));assert(!writer.append({reinterpret_cast<const uint8_t *>(v.data()),v.size()*8}));
    assert(!writer.seal());assert(!region.publish(result.value));result.extent_known=true;return result;
}
static void read(Transport&t,StoreBuffers io,const StoredInteger&v,ref_number*out){
    std::vector<uint64_t>words(v.limbs());ValueReader reader;assert(!reader.open(t,v.value.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(words.data()),words.size()*8}));ref_import(out,words.size(),-1,8,0,0,words.data());
}
static void certificate(const ref_number*r,uint64_t a,size_t n){
    ref_int v,power,lo,hi;ref_inits(v,power,lo,hi,nullptr);ref_set_ui(power,1);ref_mul_2exp(power,power,128*n);
    ref_sub_ui(lo,r,3);ref_add_ui(hi,r,3);assert(ref_sgn(lo)>=0);
    ref_mul(v,lo,lo);ref_mul_ui(v,v,a);assert(ref_cmp(v,power)<0);
    ref_mul(v,hi,hi);ref_mul_ui(v,v,a);assert(ref_cmp(v,power)>0);
    ref_set_ui(v,1);ref_mul_2exp(v,v,64*n);assert(ref_cmp(r,v)<0);
    ref_clears(v,power,lo,hi,nullptr);
}
static void cached_square(size_t m,size_t inner,size_t retained,uint64_t scalar){
    std::vector<uint64_t>r(m+1);for(size_t i=0;i<m;++i)r[i]=random_word();
    ValueStoreStorage q{};assert(query_value_store(64,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(32u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(801,901,0,disk.bytes.size(),metadata.span(),64));auto value=upload(store,t,io,r);
    SpilledProductSpec spec{m,m+1,4,inner,std::min(inner,size_t(1024)),6,128,3,4,false,retained,
                            {m+1,0,m,0,m},{m+1,0,m+1}};
    SpilledProductPlan product;assert(query_spilled_product(spec,product));RowCachePlan cp;assert(query_row_cache(product.forward[0],cp));
    assert(use_cached_a(product));auto square=product;assert(use_cached_square(square));square.emit.scalar=scalar;
    assert(retain_spilled_prefix(square,0)&&retain_spilled_prefix(square,retained));
    Fixture f(4,false,32u<<20);auto cache_storage=f.allocate(cp.storage_bytes,4096);
    auto work=f.allocate(std::max({cp.forward.storage_bytes,product.storage_bytes,square.storage_bytes}),4096);
    RowCacheBinding cache;allocation_watch_start();assert(!build_row_cache(cp,t,store,value.value,spec.a_source,
        {static_cast<uint8_t *>(cache_storage.data),cache_storage.bytes},{static_cast<uint8_t *>(work.data),work.bytes},*f.team,cache));assert(!allocation_watch_stop());
    auto before=t.counters;StoredInteger result;allocation_watch_start();assert(!run_owned_cached_product(square,t,store,cache,value.value,result.value,
        {static_cast<uint8_t *>(work.data),work.bytes},*f.team,&result.extent));assert(!allocation_watch_stop());
    assert(t.counters.read_submitted-before.read_submitted==square.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==square.normal_write_bytes);
    ref_int R,want,got;ref_inits(R,want,got,nullptr);ref_import(R,m,-1,8,0,0,r.data());ref_mul(want,R,R);ref_mul_ui(want,want,scalar);read(t,io,result,got);assert(!ref_cmp(want,got));
    result.value.reset();before=t.counters;assert(!run_owned_cached_product(product,t,store,cache,value.value,result.value,
        {static_cast<uint8_t *>(work.data),work.bytes},*f.team,&result.extent));
    assert(t.counters.read_submitted-before.read_submitted==product.normal_read_bytes&&t.counters.write_submitted-before.write_submitted==product.normal_write_bytes);
    ref_mul(want,R,R);read(t,io,result,got);assert(!ref_cmp(want,got));result.value.reset();
    const auto held=store.usage().live_bytes;disk.fail_write=disk.writes+1;
    assert(run_owned_cached_product(square,t,store,cache,value.value,result.value,{static_cast<uint8_t *>(work.data),work.bytes},*f.team).code==IoCode::io);
    assert(!result.value&&store.usage().live_bytes==held);disk.fail_write=0;release_row_cache(cache);
    ref_clears(R,want,got,nullptr);
    printf("Cached square m=%zu inner=%zu retained=%zu scale=%llu: source reuse, exact IO, scalar carries and failure cleanup PASS\n",m,inner,retained,(unsigned long long)scalar);
}
static void rung(uint64_t a,size_t m,size_t n,int perturb){
    // Start from an existing native approximation, then establish the exact
    // strict predecessor floor using independent integer square comparisons.
    // The lightweight test oracle's sqrt only supports its special sqrt2 case.
    std::vector<uint64_t>r(m+1);
    {sbn3_newton_options options{4,0,0,0};sbn3_newton_plan plan{};sbn3_newton_info info{};
        assert(sbn3_newton_query(SBN3_NEWTON_RSQRT,m,&options,&plan,&info)==SBN3_SUPPORTED);Fixture seed(4,false,32u<<20);
        auto output=seed.allocate((m+1)*8,64),space=seed.allocate(info.storage_bytes,info.storage_alignment);
        const size_t offset=static_cast<uint8_t *>(space.data)-seed.arena->base;seed.leases.pop_back();sbn3_arena_release(seed.arena,&space);
        sbn3_newton_binding*b=nullptr;sbn3_newton_bind(&plan,seed.arena,offset,seed.team,&b);sbn3_newton_inputs input{{},{},a};
        sbn3_newton_execute(b,&input,{static_cast<uint64_t *>(output.data),m+1});sbn3_newton_unbind(b);std::memcpy(r.data(),output.data,(m+1)*8);
    }
    ref_int R,P,trial,square;ref_inits(R,P,trial,square,nullptr);ref_import(R,r.size(),-1,8,0,0,r.data());
    ref_set_ui(P,1);ref_mul_2exp(P,P,128*m);bool floor=false;
    for(unsigned tries=0;tries<6;++tries){ref_mul(square,R,R);ref_mul_ui(square,square,a);
        if(ref_cmp(square,P)>=0){ref_sub_ui(R,R,1);continue;}
        ref_add_ui(trial,R,1);ref_mul(square,trial,trial);ref_mul_ui(square,square,a);
        if(ref_cmp(square,P)<0){ref_set(R,trial);continue;}floor=true;break;
    }
    assert(floor);ref_clears(trial,square,nullptr);
    if(perturb>0)ref_add_ui(R,R,unsigned(perturb));else if(perturb<0)ref_sub_ui(R,R,unsigned(-perturb));
    std::fill(r.begin(),r.end(),0);size_t used=0;ref_export(r.data(),&used,-1,8,0,0,R);assert(used<=m);
    ValueStoreStorage q{};assert(query_value_store(64,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(802,902,0,disk.bytes.size(),metadata.span(),64));auto value=upload(store,t,io,r);RsqrtRungPlan plan;
    allocation_watch_start();auto status=query_rsqrt_rung(a,m,n,4,{16u<<20,4u<<20,64u<<20},{},plan);assert(!allocation_watch_stop());assert(status==planning::ExternalPlanStatus::ready);
    Fixture f(4,false,32u<<20);auto work=f.allocate(plan.storage_bytes,4096);const auto before=t.counters;StoredInteger result;bool clamped=false;
    allocation_watch_start();auto error=run_rsqrt_rung(plan,t,store,value,result,{static_cast<uint8_t *>(work.data),work.bytes},*f.team,&clamped);assert(!allocation_watch_stop());
    if(error)fprintf(stderr,"rsqrt rung %zu/%zu error%u %s\n",m,n,unsigned(error.code),error.where);assert(!error);
    assert(t.counters.write_submitted-before.write_submitted==(clamped?plan.worst_write_bytes:plan.write_bytes)&&store.usage().peak_bytes<=plan.peak_backing_bytes);
    read(t,io,result,R);certificate(R,a,n);result.value.reset();const auto held=store.usage().live_bytes;
    t.write_limit=t.counters.write_submitted+plan.worst_write_bytes-1;
    assert(run_rsqrt_rung(plan,t,store,value,result,{static_cast<uint8_t *>(work.data),work.bytes},*f.team).code==IoCode::capacity);
    assert(!result.value&&store.usage().live_bytes==held);ref_clears(R,P,nullptr);
    printf("Stored rsqrt a=%llu m=%zu n=%zu perturb=%d private=%zu clamped=%u: <3 ulps and write/backing bounds PASS\n",
        (unsigned long long)a,m,n,perturb,plan.storage_bytes,unsigned(clamped));
}
static void complete(uint64_t a,size_t n,size_t cap){
    ValueStoreStorage q{};assert(query_value_store(96,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(803,903,0,disk.bytes.size(),metadata.span(),96));auto plan=std::make_unique<RsqrtPlan>();allocation_watch_start();
    const auto status=query_stored_rsqrt(a,n,{4,cap},{32u<<20,4u<<20,64u<<20},{},*plan);assert(!allocation_watch_stop());assert(status==planning::ExternalPlanStatus::ready);
    Fixture f(4,false,32u<<20);auto work=f.allocate(plan->storage_bytes,plan->alignment);const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;
    f.leases.pop_back();sbn3_arena_release(f.arena,&work);const auto before=t.counters;StoredInteger result;allocation_watch_start();
    auto e=run_stored_rsqrt(*plan,t,store,result,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"rsqrt ladder %zu error%u %s\n",n,unsigned(e.code),e.where);assert(!e);
    assert(t.counters.write_submitted-before.write_submitted<=plan->worst_write_bytes&&store.usage().peak_bytes<=plan->peak_backing_bytes&&f.arena->unleased(offset,plan->storage_bytes));
    ref_int R;ref_init(R);read(t,io,result,R);certificate(R,a,n);ref_clear(R);result.value.reset();assert(!store.usage().live_bytes);
    disk.cancel_after_write=disk.writes+(plan->rung_count?3:1);assert(run_stored_rsqrt(*plan,t,store,result,*f.arena,offset,*f.team).code==IoCode::cancelled);
    assert(!result.value&&!store.usage().live_bytes&&f.arena->unleased(offset,plan->storage_bytes));disk.cancelled=false;
    printf("Stored rsqrt ladder a=%llu n=%zu seed=%zu rungs=%zu: <3 ulps, caps, cancelled publication and leases PASS\n",
        (unsigned long long)a,n,plan->seed_limbs,plan->rung_count);
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);
    cached_square(301,128,0,UINT64_MAX);cached_square(7001,2048,3*chunk_bytes,10005);
    rung(2,2,3,0);rung(10005,129,255,3);rung(UINT64_MAX,513,1023,-3);rung(1,4097,8191,0);
    complete(2,127,SIZE_MAX);complete(10005,1023,15);complete(1,8191,127);complete(UINT64_MAX,127,7);
    puts("Stored rsqrt: cached square plus correction, CRT scalar fusion and complete ladder PASS");
}
