// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/format.hpp"
using namespace sbn::v3;
static uint64_t middle_count=0,middle_fallback_count=0,frontier_plans=0;
static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v,size_t fraction){
    RegionLease lease;StoredInteger value;assert(!s.reserve(v.size()*8,RegionKind::value,lease));ValueWriter writer;
    assert(!writer.open(t,lease.region(),io,&value.extent));assert(!writer.append({reinterpret_cast<const uint8_t *>(v.data()),v.size()*8}));
    assert(!writer.seal());assert(!lease.publish(value.value));value.exponent2=-int64_t(64*fraction);value.extent_known=true;return value;
}
static void one(unsigned base,uint64_t digits,unsigned repeated,unsigned pattern,bool forced,uint64_t maximum_fragments=16){
    radix::BaseInfo b;assert(radix::base_info(base,b));const size_t n=radix::node_limbs(b,(digits+63)/64)+3;
    std::vector<uint64_t>input(n+1);for(auto&v:input)v=pattern==1?UINT64_MAX:pattern==2?0:random_word();input[n]=3;
    if(pattern==2)input[n-1]=uint64_t(1)<<63;
    if(pattern==3){uint64_t carry=12345;for(size_t j=n;j-- >0;){const __uint128_t x=__uint128_t(carry)<<64;
        input[j]=uint64_t(x/100000);carry=uint64_t(x%100000);}}
    ValueStoreStorage q;assert(query_value_store(128,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();ValueStore store;StoreBuffers io{data.span(),index.span()};
    assert(!store.open(1001,1101,0,disk.bytes.size(),metadata.span(),128));auto value=upload(store,t,io,input,n);
    auto plan=std::make_unique<FormatPlan>();FormatOptions options{};options.radix.workers=4;options.radix.repeated=repeated;
    options.radix.use_alphabet=1;sbn3_radix_alphabet(base,options.radix.alphabet);options.maximum_resident_fragments=forced?maximum_fragments:UINT64_MAX;
    allocation_watch_start();const auto status=query_stored_format(value,base,digits,options,{32u<<20,4u<<20,64u<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"format query b%u d%llu status%u fixed%zu phase%zu classes%u\n",base,(unsigned long long)digits,unsigned(status),plan->fixed_bytes,plan->phase_bytes,plan->classes);assert(status==planning::ExternalPlanStatus::ready);
    assert(forced?!plan->node[0].resident:plan->node[0].resident);
    uint64_t native_frontiers=0;for(unsigned i=0;i<plan->classes;++i)native_frontiers+=plan->node[i].resident&&plan->node[i].native_frontier;
    frontier_plans+=native_frontiers;if(forced&&maximum_fragments>=128)assert(native_frontiers);
    Fixture f(4,false,32u<<20);auto work=f.allocate(plan->storage_bytes,plan->alignment);const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;
    f.leases.pop_back();sbn3_arena_release(f.arena,&work);FormatResult result;const auto before=t.counters;allocation_watch_start();
    const auto e=run_stored_format(*plan,t,store,value,result,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"format b%u d%llu error%u %s need%llu have%llu\n",base,(unsigned long long)digits,unsigned(e.code),e.where,(unsigned long long)e.need,(unsigned long long)e.have);assert(!e);
    assert(result.integer_part==3&&result.certified_digits<=digits&&f.arena->unleased(offset,plan->storage_bytes));
    middle_count+=result.middle_products;middle_fallback_count+=result.middle_fallbacks;
    assert(t.counters.write_submitted-before.write_submitted<=plan->write_bytes&&store.usage().peak_bytes<=plan->peak_backing_bytes);
    if(!result.middle_fallbacks)assert(t.counters.write_submitted-before.write_submitted==plan->normal_write_bytes);
    if(!forced&&digits>2*chunk_bytes&&digits<t.max_transfer)
        assert(t.counters.writes-before.writes<=3); // short tail, contiguous interior, index page
    std::vector<uint8_t>text(digits);ValueReader reader;assert(!reader.open(t,result.digits.region(),io));assert(!reader.read(0,text));
    // Independent exact word-by-word fraction conversion; no SRT or shared leaf.
    input.resize(n);
    std::vector<uint8_t>reference(digits),actual(digits);
    for(size_t j=0;j<digits;++j){uint64_t carry=0;
        for(auto&v:input){const __uint128_t x=__uint128_t(v)*base+carry;v=uint64_t(x);carry=uint64_t(x>>64);}
        reference[j]=uint8_t(carry);unsigned digit=0;while(digit<base&&options.radix.alphabet[digit]!=text[j])++digit;assert(digit<base);actual[j]=uint8_t(digit);
        if(j<result.certified_digits)assert(digit==carry);
    }
    if(reference!=actual){bool changed=false;for(size_t j=reference.size();j-- >0;){if(reference[j]){--reference[j];changed=true;break;}reference[j]=uint8_t(base-1);}
        assert(changed&&reference==actual);}
    if(!pattern)assert(result.certified_digits==digits);
    result.digits.reset();const auto held=store.usage().live_bytes;
    t.write_limit=t.counters.write_submitted+plan->write_bytes-1;FormatResult rejected;
    assert(run_stored_format(*plan,t,store,value,rejected,*f.arena,offset,*f.team).code==IoCode::capacity);assert(!rejected.digits&&store.usage().live_bytes==held);t.write_limit=UINT64_MAX;
    disk.cancel_after_write=disk.writes+1;
    assert(run_stored_format(*plan,t,store,value,rejected,*f.arena,offset,*f.team).code==IoCode::cancelled);
    assert(!rejected.digits&&store.usage().live_bytes==held&&f.arena->unleased(offset,plan->storage_bytes));disk.cancelled=false;
    printf("Stored radix b%u D%llu repeated%u pattern%u forced%u: classes%u rails%u frontiers%llu cert%llu private%zu writes%llu exact reference/ledger/cleanup PASS\n",
        base,(unsigned long long)digits,repeated,pattern,unsigned(forced),plan->classes,plan->powers,(unsigned long long)plan->node[0].frontiers,
        (unsigned long long)result.certified_digits,plan->storage_bytes,(unsigned long long)plan->write_bytes);
}
static void large_query(){
    constexpr uint64_t digits=5000000000ULL;radix::BaseInfo b;assert(radix::base_info(10,b));const size_t n=radix::node_limbs(b,(digits+63)/64)+2;
    ValueStoreStorage q;assert(query_value_store(16,q));Aligned metadata(q.bytes);ValueStore store;assert(!store.open(1002,1102,0,uint64_t(1)<<40,metadata.span(),16));
    RegionLease lease;assert(!store.reserve((n+1)*8,RegionKind::value,lease));lease.region().state=RegionState::sealed;StoredInteger input;
    assert(!lease.publish(input.value));input.extent_known=true;input.extent.end=n+1;input.extent.top_word=3;input.exponent2=-int64_t(64*n);
    auto plan=std::make_unique<FormatPlan>();FormatOptions options{};options.radix.workers=16;
    allocation_watch_start();const auto status=query_stored_format(input,10,digits,options,{size_t(4)<<30,size_t(64)<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"5B radix query status%u classes%u fixed%zu phase%zu total%zu\n",unsigned(status),plan->classes,plan->fixed_bytes,plan->phase_bytes,plan->total_resident_bytes);
    assert(status==planning::ExternalPlanStatus::ready&&plan->total_resident_bytes<=(size_t(4)<<30));
    printf("5B/4GiB radix QUERY ONLY: classes%u rails%u frontiers%llu RAM%zu backing%llu writes%llu seed%llu PASS (no operands/arithmetic/IO)\n",plan->classes,plan->powers,
        (unsigned long long)plan->node[0].frontiers,plan->total_resident_bytes,(unsigned long long)plan->peak_backing_bytes,(unsigned long long)plan->write_bytes,(unsigned long long)plan->seed);
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);
    one(10,4111,0,0,false);one(10,4111,0,0,true);one(10,4111,1,0,true);
    one(10,2*chunk_bytes+123,0,0,false); // one multi-chunk interior plus a short final seam
    one(3,8197,0,0,true);one(7,4111,1,0,true);one(15,4111,0,0,true);one(63,4111,0,0,true);
    one(16,8197,0,0,true);one(64,8197,1,0,true);one(10,4111,0,1,true);one(10,4111,0,2,true);
    one(10,250001,0,0,true);one(10,250001,0,2,true);one(10,35011,0,0,true);one(10,4111,0,3,true);large_query();
    // Native overlap words must also survive external seams, including the
    // all-nine uncertainty and the exact terminating-dyadic carry witness.
    one(10,35011,0,0,true,128);one(10,35011,0,1,true,128);one(10,35011,0,2,true,128);
    one(3,35011,0,0,true,128);
    printf("Stored middle products %llu, deterministic carry witnesses %llu\n",(unsigned long long)middle_count,(unsigned long long)middle_fallback_count);
    assert(middle_count);
    printf("Native overlap-word frontier plans exercised: %llu\n",(unsigned long long)frontier_plans);assert(frontier_plans);
    puts("Stored fractional SRT: existing split precision, odd powers, right-first carry seams, write-once reverse chunks and bounded native frontiers PASS");
}
