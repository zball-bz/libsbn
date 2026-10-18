// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/newton.hpp"
#include "external/series.hpp"
#include "external/format.hpp"
#include "formula_reference.hpp"
using namespace sbn::v3;
static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v){
    RegionLease r;StoredInteger out;assert(!s.reserve(v.size()*8,RegionKind::value,r));ValueWriter writer;
    assert(!writer.open(t,r.region(),io,&out.extent));assert(!writer.append({reinterpret_cast<const uint8_t *>(v.data()),v.size()*8}));
    assert(!writer.seal());assert(!r.publish(out.value));out.extent_known=true;return out;
}
static void read(Transport&t,StoreBuffers io,const StoredInteger&v,ref_number*out){
    std::vector<uint64_t>words(v.limbs());ValueReader reader;assert(!reader.open(t,v.value.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(words.data()),words.size()*8}));ref_import(out,words.size(),-1,8,0,0,words.data());
}
static void fixed_windows(){
    ValueStoreStorage q;assert(query_value_store(16,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(4u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(810,910,0,disk.bytes.size(),metadata.span(),16));ref_int raw,expected,guard,got;ref_inits(raw,expected,guard,got,nullptr);
    for(size_t bits:{size_t(7),size_t(64),size_t(65),size_t(93),size_t(192),size_t(2000)})for(unsigned pattern=0;pattern<5;++pattern){
        ref_set_ui(expected,pattern==4?0:12345);ref_mul_2exp(raw,expected,bits);
        if(pattern==1||pattern==3){ref_set_ui(guard,1);ref_mul_2exp(guard,guard,bits);ref_sub_ui(guard,guard,pattern==1?3:1);}
        else ref_set_ui(guard,pattern==0?3:0);
        ref_add(raw,raw,guard);std::vector<uint64_t>words((bits+128)/64+1);size_t used=0;ref_export(words.data(),&used,-1,8,0,0,raw);
        auto input=upload(store,t,io,words);input.exponent2=-int64_t(bits+64*4);FixedOutputPlan plan;assert(query_fixed_output(input,4,3,plan));
        Aligned work(plan.storage_bytes);StoredInteger result;const auto before=t.counters;
        if(pattern==2||pattern==3){const pid_t child=fork();assert(child>=0);
            if(!child){sbn3_set_fatal_hook([](const sbn3_fatal_info*i,void*){
                    _exit(i->kind==SBN3_FATAL_MATH&&!std::strcmp(i->where,"stored constant guard separation")?77:78);
                },nullptr);(void)run_fixed_output(plan,t,store,input,result,work.span());_exit(79);}
            int status=0;assert(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==77);
        }else{allocation_watch_start();assert(!run_fixed_output(plan,t,store,input,result,work.span()));assert(!allocation_watch_stop());
            assert(t.counters.write_submitted-before.write_submitted==plan.write_bytes);read(t,io,result,got);assert(!ref_cmp(got,expected));}
    }
    ref_clears(raw,expected,guard,got,nullptr);puts("Stored fixed output: bit/word guards, both exact safety margins, certified zero and ambiguous-floor rejection PASS");
}
static void fused(uint64_t a,size_t n,uint64_t scalar,size_t cap){
    std::vector<uint64_t>x(n+1);for(auto&word:x)word=random_word();x[n]=scalar==1?UINT64_MAX-1:3;
    ValueStoreStorage q{};assert(query_value_store(128,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(96u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(811,911,0,disk.bytes.size(),metadata.span(),128));auto input=upload(store,t,io,x);input.exponent2=-int64_t(64*n+17);
    auto plan=std::make_unique<external::MulRsqrtPlan>();allocation_watch_start();
    const auto status=query_stored_mul_rsqrt(input,n,a,scalar,{4,cap},{32u<<20,4u<<20,96u<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"mul rsqrt query n%zu status%u\n",n,unsigned(status));assert(status==planning::ExternalPlanStatus::ready);
    Fixture f(4,false,32u<<20);auto work=f.allocate(plan->storage_bytes,plan->alignment);const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;
    f.leases.pop_back();sbn3_arena_release(f.arena,&work);const auto before=t.counters;StoredInteger output;allocation_watch_start();
    const auto e=run_stored_mul_rsqrt(*plan,t,store,input,output,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"mul rsqrt n%zu error%u %s\n",n,unsigned(e.code),e.where);assert(!e);
    assert(output.exponent2==input.exponent2&&t.counters.write_submitted-before.write_submitted<=plan->worst_write_bytes&&store.usage().peak_bytes<=plan->peak_backing_bytes&&f.arena->unleased(offset,plan->storage_bytes));
    ref_int X,Y,target,check;ref_inits(X,Y,target,check,nullptr);ref_import(X,x.size(),-1,8,0,0,x.data());ref_mul_ui(X,X,scalar);ref_mul(target,X,X);read(t,io,output,Y);
    ref_sub_ui(check,Y,uint64_t(1)<<38);assert(ref_sgn(check)>0);ref_mul(check,check,check);ref_mul_ui(check,check,a);assert(ref_cmp(check,target)<0);
    ref_add_ui(check,Y,uint64_t(1)<<38);ref_mul(check,check,check);ref_mul_ui(check,check,a);assert(ref_cmp(check,target)>0);ref_clears(X,Y,target,check,nullptr);
    output.value.reset();const auto held=store.usage().live_bytes;t.write_limit=t.counters.write_submitted+plan->worst_write_bytes-1;
    assert(run_stored_mul_rsqrt(*plan,t,store,input,output,*f.arena,offset,*f.team).code==IoCode::capacity);assert(!output.value&&store.usage().live_bytes==held);
    t.write_limit=UINT64_MAX;disk.cancel_after_write=disk.writes+3;
    assert(run_stored_mul_rsqrt(*plan,t,store,input,output,*f.arena,offset,*f.team).code==IoCode::cancelled);
    assert(!output.value&&store.usage().live_bytes==held&&f.arena->unleased(offset,plan->storage_bytes));disk.cancelled=false;
    printf("Fused stored X/sqrt(a) a=%llu n=%zu scalar=%llu half_seed=%zu: <2^38 ulps, scales, writes/backing, cancellation PASS\n",
        (unsigned long long)a,n,(unsigned long long)scalar,plan->half.seed_limbs);
}
static void arctan(ref_number*result,unsigned q,size_t words){
    ref_int power,term,divisor;ref_inits(power,term,divisor,nullptr);ref_set_ui(power,1);ref_mul_2exp(power,power,64*words);
    ref_set_ui(divisor,q);ref_fdiv_q(power,power,divisor);ref_set_ui(result,0);
    for(uint64_t k=0;ref_sgn(power);++k){ref_set_ui(divisor,2*k+1);ref_fdiv_q(term,power,divisor);
        if(k&1)ref_sub(result,result,term);else ref_add(result,result,term);
        ref_set_ui(divisor,q*q);ref_fdiv_q(power,power,divisor);
    }ref_clears(power,term,divisor,nullptr);
}
static void constant(bool pi,size_t fractional){
    using namespace series;const size_t working=fractional+6,precision=working+2;
    const auto def=pi?chudnovsky_definition():euler_definition();const uint64_t terms=pi?uint64_t(2*precision):uint64_t(16*precision);
    auto formula=std::make_unique<DataFormula>();assert(data_formula_prepare(def,def.begin+terms,*formula));
    const PsrSpec spec{data_finite_formula(*formula),{def.begin,def.begin+terms},{4,8,0,0,0},
        {formula.get(),[](const void*p,uint64_t i){return static_cast<const DataFormula *>(p)->attenuation_bits(i);},4,83,0},precision};
    SeriesMetadata layout;assert(query_series_metadata(128,64,layout));Aligned metadata(layout.bytes);SeriesPlan plan;
    allocation_watch_start();assert(compile_series(spec,{32u<<20,4u<<20,96u<<20},{true,2},{},layout,metadata.span(),plan)==planning::ExternalPlanStatus::ready);assert(!allocation_watch_stop());
    ValueStoreStorage slots;assert(query_value_store(160,slots));Aligned state(slots.bytes);Device disk(96u<<20);auto t=disk.transport();ValueStore store;
    assert(!store.open(812,912,0,disk.bytes.size(),state.span(),160));Fixture f(4,false,32u<<20);const size_t io_index=f.leases.size();auto io=f.allocate(plan.io_bytes,4096);
    const size_t work_index=f.leases.size();auto work=f.allocate(plan.work_bytes,std::max(plan.work_alignment,size_t(2)<<20));const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;
    SeriesResult series;allocation_watch_start();assert(!run_series(plan,t,store,*f.arena,*f.team,work,{static_cast<uint8_t *>(io.data),io.bytes},series));assert(!allocation_watch_stop());assert(series.in_store);
    assert(t.counters.write_submitted<=plan.write_bytes);uint64_t bound=plan.write_bytes;f.retire(work_index);
    auto prepare=[&](size_t bytes,size_t alignment){assert(!(uintptr_t(work.data)&(alignment-1)));sbn3_error e{};assert(sbn3_arena_prepare(f.arena,offset,bytes,&e)==SBN3_OK);};
    InverseBudget budget{32u<<20,(4u<<20)+layout.bytes+io.bytes,96u<<20};auto quotient=std::make_unique<QuotientPlan>();
    allocation_watch_start();assert(query_stored_quotient(series.stored[pi?1:0],series.stored[pi?0:1],working,{4,7},budget,{},*quotient)==planning::ExternalPlanStatus::ready);assert(!allocation_watch_stop());
    prepare(quotient->storage_bytes,quotient->alignment);StoredInteger ratio;allocation_watch_start();
    assert(!run_stored_quotient(*quotient,t,store,series.stored[pi?1:0],series.stored[pi?0:1],ratio,*f.arena,offset,*f.team));assert(!allocation_watch_stop());
    bound+=quotient->worst_write_bytes;series.stored={};quotient.reset();StoredInteger approximate;
    if(pi){auto terminal=std::make_unique<external::MulRsqrtPlan>();allocation_watch_start();
        assert(query_stored_mul_rsqrt(ratio,working,10005,4270934400ULL,{4,7},budget,{},*terminal)==planning::ExternalPlanStatus::ready);assert(!allocation_watch_stop());
        prepare(terminal->storage_bytes,terminal->alignment);allocation_watch_start();assert(!run_stored_mul_rsqrt(*terminal,t,store,ratio,approximate,*f.arena,offset,*f.team));assert(!allocation_watch_stop());
        bound+=terminal->worst_write_bytes;ratio.value.reset();
    }else approximate=std::move(ratio);
    FixedOutputPlan finish;allocation_watch_start();assert(query_fixed_output(approximate,fractional,uint64_t(1)<<40,finish,pi?0:1));assert(!allocation_watch_stop());
    prepare(finish.storage_bytes,4096);auto lease=f.arena->acquire(offset,finish.storage_bytes);StoredInteger result;allocation_watch_start();
    assert(!run_fixed_output(finish,t,store,approximate,result,{static_cast<uint8_t *>(lease.data),lease.bytes}));assert(!allocation_watch_stop());f.arena->release(lease);
    bound+=finish.write_bytes;assert(t.counters.write_submitted<=bound&&f.arena->unleased(offset,finish.storage_bytes));
    StoreBuffers buffers{{static_cast<uint8_t *>(io.data),1u<<20},{static_cast<uint8_t *>(io.data)+(1u<<20),index_page_bytes}};
    ref_int got,want,a,b;ref_inits(got,want,a,b,nullptr);read(t,buffers,result,got);
    if(pi){arctan(a,5,fractional+3);arctan(b,239,fractional+3);ref_mul_ui(a,a,16);ref_mul_ui(b,b,4);ref_sub(want,a,b);ref_fdiv_q_2exp(want,want,192);}
    else {formula_reference::Ref exact;formula_reference::reference(def,def.begin,def.begin+terms,exact);
        ref_mul_2exp(a,exact.t,64*fractional);ref_fdiv_q(want,a,exact.d);
        // Hyperdescent starts at 1 and returns e-1; add the k=0 term.
        ref_set_ui(a,1);ref_mul_2exp(a,a,64*fractional);ref_add(want,want,a);}
    assert(!ref_cmp(got,want));
    std::vector<uint64_t>fraction(fractional+1);size_t used=0;ref_export(fraction.data(),&used,-1,8,0,0,want);fraction.resize(fractional);
    ref_clears(got,want,a,b,nullptr);
    printf("Stored %-2s %zu fractional words: series -> quotient -> %sfixed output, independent %s match, same arena and aggregate writes PASS\n",
        pi?"pi":"e",fractional,pi?"fused mul-rsqrt -> ":"",pi?"Machin":"exact series");
    approximate.value.reset();
    // Reuse the same aligned phase region. The old series-only IO loan is
    // retired before the formatter's own bounded IO window becomes live.
    const size_t old_io=static_cast<uint8_t *>(io.data)-f.arena->base;f.retire(io_index);sbn3_error trim{};
    assert(sbn3_arena_trim(f.arena,old_io,io.bytes,&trim)==SBN3_OK);
    const uint64_t decimal=fractional>32?1151:200;auto format=std::make_unique<FormatPlan>();FormatOptions options{};
    options.radix.workers=4;options.radix.use_alphabet=1;sbn3_radix_alphabet(10,options.radix.alphabet);options.maximum_resident_fragments=16;
    // This fixture retains the old series metadata while forcing an external
    // radix split. Include that metadata and the carry-witness plans in its
    // deliberately small phase grant; production examples start at 4 GiB.
    allocation_watch_start();const auto format_status=query_stored_format(result,10,decimal,options,{40u<<20,(4u<<20)+layout.bytes,96u<<20},{},*format);assert(!allocation_watch_stop());
    if(format_status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"terminal format n%zu status%u fixed%zu phase%zu classes%u metadata%zu\n",
        fractional,unsigned(format_status),format->fixed_bytes,format->phase_bytes,format->classes,layout.bytes);
    assert(format_status==planning::ExternalPlanStatus::ready);
    prepare(format->storage_bytes,format->alignment);FormatResult text;allocation_watch_start();
    assert(!run_stored_format(*format,t,store,result,text,*f.arena,offset,*f.team));assert(!allocation_watch_stop());
    assert(text.certified_digits==decimal&&text.integer_part==(pi?3:2)&&f.arena->unleased(offset,format->storage_bytes));
    bound+=format->write_bytes;assert(t.counters.write_submitted<=bound);
    std::vector<uint8_t>digits(decimal);ValueReader reader;StoreBuffers final_io{
        {f.arena->base+offset+format->io,1u<<20},{f.arena->base+offset+format->index,index_page_bytes}};
    assert(!reader.open(t,text.digits.region(),final_io));assert(!reader.read(0,digits));
    for(size_t i=0;i<decimal;++i){uint64_t carry=0;for(auto&word:fraction){const __uint128_t x=__uint128_t(word)*10+carry;word=uint64_t(x);carry=uint64_t(x>>64);}
        assert(digits[i]==uint8_t('0'+carry));}
    printf("Stored %-2s full decimal chain: %llu certified digits, %llu radix frontiers, independent %s -> exact decimal reference PASS\n",
        pi?"pi":"e",(unsigned long long)decimal,(unsigned long long)format->node[0].frontiers,pi?"Machin":"series");
}
static void large_queries(){
    // Synthetic metadata only: deliberately no transport/data backing. These
    // handles are never read or executed; this gate admits the real-scale plan.
    constexpr size_t n=259525640;ValueStoreStorage q;assert(query_value_store(16,q));Aligned metadata(q.bytes);ValueStore store;
    assert(!store.open(813,913,0,uint64_t(1)<<40,metadata.span(),16));
    auto stub=[&](size_t words,uint64_t top){RegionLease lease;StoredInteger v;assert(!store.reserve(words*8,RegionKind::value,lease));
        lease.region().state=RegionState::sealed;assert(!lease.publish(v.value));v.extent.observe(0,{reinterpret_cast<const uint8_t *>(&top),8});
        v.extent.end=words;v.extent.top_word=top;v.extent_known=true;return v;};
    auto numerator=stub(n+3,uint64_t(1)<<63),denominator=stub(n+5,uint64_t(1)<<62),x=stub(n+1,1);
    x.exponent2=-int64_t(n*64+24);const InverseBudget budget{size_t(4)<<30,size_t(64)<<20};
    auto division=std::make_unique<QuotientPlan>();auto terminal=std::make_unique<external::MulRsqrtPlan>();
    allocation_watch_start();const auto dq=query_stored_quotient(numerator,denominator,n,{16,SIZE_MAX},budget,{},*division);assert(!allocation_watch_stop());
    if(dq!=planning::ExternalPlanStatus::ready)fprintf(stderr,"5B quotient status%u\n",unsigned(dq));assert(dq==planning::ExternalPlanStatus::ready);
    allocation_watch_start();const auto rq=query_stored_mul_rsqrt(x,n,10005,4270934400ULL,{16,SIZE_MAX},budget,{},*terminal);assert(!allocation_watch_stop());
    if(rq!=planning::ExternalPlanStatus::ready)fprintf(stderr,"5B fused root status%u\n",unsigned(rq));assert(rq==planning::ExternalPlanStatus::ready);
    assert(division->total_resident_bytes<=budget.resident_bytes&&terminal->total_resident_bytes<=budget.resident_bytes);
    printf("5B/4GiB finishing QUERIES ONLY: quotient seed%zu rungs%zu RAM%zu backing%llu writes%llu; fused root seed%zu rungs%zu RAM%zu backing%llu writes%llu PASS\n",
        division->inverse.seed_limbs,division->inverse.rung_count,division->total_resident_bytes,(unsigned long long)division->peak_backing_bytes,(unsigned long long)division->worst_write_bytes,
        terminal->half.seed_limbs,terminal->half.rung_count,terminal->total_resident_bytes,(unsigned long long)terminal->peak_backing_bytes,(unsigned long long)terminal->worst_write_bytes);
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);fixed_windows();fused(10005,127,4270934400ULL,15);fused(UINT64_MAX,1023,1,15);fused(1,8191,1,127);
    constant(false,17);constant(true,17);constant(true,65);large_queries();
    puts("Stored numerical constant finishing: half rsqrt fusion, guard-separated fixed output and independent pi reference PASS");
}
