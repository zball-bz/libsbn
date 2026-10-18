// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/newton.hpp"
#include "external/owned_product.hpp"
#include "external/series.hpp"
#include "formula_reference.hpp"
using namespace sbn::v3;
static StoredInteger upload(ValueStore&s,Transport&t,StoreBuffers io,const std::vector<uint64_t>&v){
    RegionLease r;StoredInteger out;assert(!s.reserve(v.size()*8,RegionKind::value,r));ValueWriter writer;
    assert(!writer.open(t,r.region(),io,&out.extent));assert(!writer.append({reinterpret_cast<const uint8_t *>(v.data()),v.size()*8}));
    assert(!writer.seal());assert(!r.publish(out.value));out.extent_known=true;return out;
}
static void load(Transport&t,StoreBuffers io,const StoredInteger&v,ref_number*out){
    std::vector<uint64_t>words(v.limbs());ValueReader r;assert(!r.open(t,v.value.region(),io));
    assert(!r.read(v.first_limb*8,{reinterpret_cast<uint8_t *>(words.data()),words.size()*8}));ref_import(out,words.size(),-1,8,0,0,words.data());
}
static void subtraction(){
    ValueStoreStorage q{};assert(query_value_store(24,q));Aligned metadata(q.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(8u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(601,701,0,disk.bytes.size(),metadata.span(),24));
    ref_int A,E,P,want,got;ref_inits(A,E,P,want,got,nullptr);
    for(unsigned pattern=0;pattern<4;++pattern)for(int sign:{-1,0,1}){
        constexpr size_t words=11;const size_t shift=pattern==3?4:0;
        std::vector<uint64_t>a(words),p(words);for(auto&v:a)v=pattern==0?0:pattern==1?UINT64_MAX:random_word();
        if(pattern==3){std::fill(a.begin(),a.end(),0);a[0]=1;}
        ref_import(A,words,-1,8,0,0,a.data());ref_mul_2exp(A,A,64*shift);ref_fdiv_r_2exp(A,A,64*words);
        ref_set_ui(E,sign?1:0);if(sign){ref_mul_2exp(E,E,193);ref_add_ui(E,E,17);if(sign<0)ref_neg(E,E);}
        ref_add(P,A,E);ref_fdiv_r_2exp(P,P,64*words);size_t used=0;ref_export(p.data(),&used,-1,8,0,0,P);
        auto subvalue=upload(store,t,io,a),input=upload(store,t,io,p);StoredInteger output;
        LinearInput sub=pattern==3?linear_constant(1,int64_t(64*shift)):linear_input(subvalue);
        SignedLiftPlan plan{};assert(query_signed_lift(words,{1,4,0},{4,2,true},plan,sub.limbs(),shift));Aligned workspace(plan.storage_bytes);
        const auto before=t.counters;allocation_watch_start();assert(!run_signed_lift(plan,t,store,input,output,workspace.span(),&sub));assert(!allocation_watch_stop());
        assert(t.counters.write_submitted-before.write_submitted==plan.write_bytes&&output.negative==(sign<0));
        load(t,io,output,got);ref_abs(want,E);ref_fdiv_q_2exp(want,want,64);assert(!ref_cmp(got,want));
    }
    ref_clears(A,E,P,want,got,nullptr);
    puts("Modular cancellation: both wrap directions, exact equality, signed output and virtual scalar PASS");
}
static void cyclic_subtraction(size_t m,size_t n,size_t inner,unsigned np,unsigned T,int shift,int sign,unsigned raw,bool stream=false){
    std::vector<uint64_t>u(m+1),d(n);u[m]=1;u[0]=sign?123:0;
    for(auto&v:d)v=random_word();d.back()=(d.back()>>4)|(uint64_t(1)<<59);
    if(!sign&&shift>0)d[0]&=~((uint64_t(1)<<shift)-1);
    ref_int U,D,P,C,S,E,got;ref_inits(U,D,P,C,S,E,got,nullptr);
    ref_import(U,u.size(),-1,8,0,0,u.data());ref_import(D,d.size(),-1,8,0,0,d.data());ref_mul(P,U,D);
    ref_fdiv_q_2exp(C,P,64*m);
    if(shift>0){ref_fdiv_q_2exp(C,C,unsigned(shift));if(sign<0)ref_add_ui(C,C,1);ref_mul_2exp(C,C,unsigned(shift));}
    else if(sign<0)ref_add_ui(C,C,1);
    ref_mul_2exp(S,C,64*m);ref_sub(E,P,S);
    if(shift>0)ref_fdiv_q_2exp(S,C,unsigned(shift));else ref_mul_2exp(S,C,unsigned(-shift));
    const size_t physical=n+size_t(shift<0),prefix=3;std::vector<uint64_t>sub(prefix+physical+2,0);
    sub[0]=sub[1]=sub[2]=sub.back()=UINT64_MAX;size_t used=0;
    ref_export(sub.data()+prefix,&used,-1,8,0,0,S);assert(used<=physical);
    ValueStoreStorage meta{};assert(query_value_store(48,meta));Aligned metadata(meta.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(812,813,0,disk.bytes.size(),metadata.span(),48));
    auto a=upload(store,t,io,u),b=upload(store,t,io,d),c=upload(store,t,io,sub);
    SourceWindow source{sub.size(),prefix,n,prefix,prefix+physical,shift};
    SpilledProductSpec spec{u.size(),d.size(),4,inner,0,np,T,2,4,false};
    spec.cyclic={n+2,m,{n,4,true},source,true};spec.raw_operand=raw;
    if(stream){spec.stripe_cells=1024;spec.stream_final=true;}
    SpilledProductPlan plan{};assert(query_spilled_product(spec,plan));
    if(stream){const auto before=plan.normal_read_bytes;
        assert(plan.final_banks==3&&widen_inverse_within(plan,2048,SIZE_MAX));
        assert(plan.normal_read_bytes<=before&&plan.emit.row_slots==3*plan.emit.rows);
    }
    assert(retain_spilled_prefix(plan,size_t(np)*size_t(plan.plane_bytes)));
    Fixture f(4,false,64u<<20);auto work=f.allocate(plan.storage_bytes,4096);StoredValue output;
    CyclicResidualResult sign_result{};WordExtent extent{};const auto before=t.counters;
    const auto saved=std::vector<uint8_t>(disk.bytes.begin()+c.value.region().layout.data_offset,
        disk.bytes.begin()+c.value.region().layout.data_offset+c.value.region().layout.allocation_bytes);
    allocation_watch_start();const auto error=run_owned_product(plan,t,store,a.value,b.value,output,
        {static_cast<uint8_t *>(work.data),work.bytes},*f.team,&extent,&sign_result,&c.value);assert(!allocation_watch_stop());
    if(error)fprintf(stderr,"cyclic subtract error%u %s\n",unsigned(error.code),error.where);assert(!error);
    const auto reads=t.counters.read_submitted-before.read_submitted,writes=t.counters.write_submitted-before.write_submitted;
    if(reads!=plan.normal_read_bytes)fprintf(stderr,"cyclic subtract IO %llu expected%llu (source%llu)\n",
        (unsigned long long)reads,(unsigned long long)plan.normal_read_bytes,(unsigned long long)plan.emit.subtract_read_bytes);
    assert(reads==plan.normal_read_bytes&&writes==plan.normal_write_bytes);
    assert(!memcmp(saved.data(),disk.bytes.data()+c.value.region().layout.data_offset,saved.size()));
    std::vector<uint64_t>words(size_t(plan.output_bytes/8));ValueReader reader;assert(!reader.open(t,output.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(words.data()),words.size()*8}));
    if(sign_result.complemented)for(auto&v:words)v=~v;
    ref_import(got,words.size(),-1,8,0,0,words.data());ref_abs(E,E);assert(!ref_cmp(got,E));
    if(sign)assert(sign_result.complemented==(sign<0));
    ref_clears(U,D,P,C,S,E,got,nullptr);
    printf("Cyclic stored subtract m%zu n%zu shift%d sign%d raw%u: exact signed result, immutable third source, exact reads/writes PASS\n",m,n,shift,sign,raw);
}
static unsigned cyclic_rungs=0;
static void rung(size_t m,size_t n,unsigned shift,int perturb){
    std::vector<uint64_t>a(n+19),d(n+37),u(m+1);for(auto*v:{&a,&d})for(auto&word:*v)word=random_word();
    a.back()=(a.back()>>17)|(uint64_t(1)<<46);d.back()=(d.back()>>shift)|(uint64_t(1)<<(63-shift));
    ref_int A,D,H,U,P,Q,error,bound;ref_inits(A,D,H,U,P,Q,error,bound,nullptr);
    ref_import(A,a.size(),-1,8,0,0,a.data());ref_mul_2exp(A,A,17);ref_fdiv_q_2exp(A,A,64*(a.size()-n));
    ref_import(D,d.size(),-1,8,0,0,d.data());ref_mul_2exp(D,D,shift);ref_fdiv_q_2exp(D,D,64*(d.size()-n));
    ref_fdiv_q_2exp(H,D,64*(n-m));ref_set_ui(P,1);ref_mul_2exp(P,P,128*m);ref_sub_ui(P,P,1);ref_fdiv_q(U,P,H);
    if(perturb>0)ref_add_ui(U,U,unsigned(perturb));else if(perturb<0)ref_sub_ui(U,U,unsigned(-perturb));
    size_t used=0;ref_export(u.data(),&used,-1,8,0,0,U);assert(used==m+1&&u[m]==1);
    ValueStoreStorage meta{};assert(query_value_store(96,meta));Aligned metadata(meta.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(64u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(602,702,0,disk.bytes.size(),metadata.span(),96));auto av=upload(store,t,io,a),dv=upload(store,t,io,d),uv=upload(store,t,io,u);
    SourceWindow as,ds;assert(normalized_source(av,n,as)&&normalized_source(dv,n,ds));QuotientRungPlan plan{};
    allocation_watch_start();const auto status=query_quotient_rung(m,n,as,ds,4,{24u<<20,4u<<20,64u<<20},{},plan);assert(!allocation_watch_stop());
    assert(status==planning::ExternalPlanStatus::ready);cyclic_rungs+=bool(plan.residual_product.spec.cyclic);
    Fixture f(4,false,32u<<20);auto work=f.allocate(plan.storage_bytes,4096);
    const auto before=t.counters;StoredInteger result;allocation_watch_start();
    const auto e=run_quotient_rung(plan,t,store,av,dv,uv,result,{static_cast<uint8_t *>(work.data),work.bytes},*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"quotient rung %zu/%zu error %u %s\n",m,n,unsigned(e.code),e.where);assert(!e);
    assert(t.counters.write_submitted-before.write_submitted==plan.write_bytes&&store.usage().peak_bytes<=plan.peak_backing_bytes);
    load(t,io,result,Q);ref_mul(error,Q,D);ref_mul_2exp(P,A,64*n);ref_sub(error,error,P);ref_abs(error,error);ref_mul_ui(bound,D,3);assert(ref_cmp(error,bound)<0);
    result.value.reset();const auto live=store.usage().live_bytes;t.write_limit=t.counters.write_submitted+plan.write_bytes-1;
    assert(run_quotient_rung(plan,t,store,av,dv,uv,result,{static_cast<uint8_t *>(work.data),work.bytes},*f.team).code==IoCode::capacity);
    assert(!result.value&&store.usage().live_bytes==live);t.write_limit=UINT64_MAX;
    disk.fail_write=disk.writes+3;
    assert(run_quotient_rung(plan,t,store,av,dv,uv,result,{static_cast<uint8_t *>(work.data),work.bytes},*f.team).code==IoCode::io);
    assert(!result.value&&store.usage().live_bytes==live);disk.fail_write=0;
    ref_clears(A,D,H,U,P,Q,error,bound,nullptr);
    printf("Stored quotient m=%zu n=%zu shift=%u perturb=%d private=%zu: <3 ulps, exact writes, cap/failure cleanup PASS\n",m,n,shift,perturb,plan.storage_bytes);
}
static void complete(size_t n,size_t seed_cap,unsigned pattern,size_t short_words=0){
    std::vector<uint64_t>a(n+23),d(n+41);for(auto*v:{&a,&d})for(auto&word:*v)word=random_word();
    if(short_words){a.resize(short_words+2);d.resize(short_words);}
    a.back()=(a.back()>>13)|(uint64_t(1)<<50);d.back()=(d.back()>>47)|(uint64_t(1)<<16);
    if(pattern==1){std::fill(a.begin(),a.end(),0);std::fill(d.begin(),d.end(),0);a.back()=uint64_t(1)<<50;d.back()=uint64_t(1)<<16;}
    ValueStoreStorage meta{};assert(query_value_store(128,meta));Aligned metadata(meta.bytes),data(4*chunk_bytes),index(index_page_bytes);
    Device disk(128u<<20);auto t=disk.transport();StoreBuffers io{data.span(),index.span()};ValueStore store;
    assert(!store.open(603,703,0,disk.bytes.size(),metadata.span(),128));auto av=upload(store,t,io,a),dv=upload(store,t,io,d);
    av.exponent2=123;dv.exponent2=-45;auto plan=std::make_unique<QuotientPlan>();allocation_watch_start();
    const auto status=short_words?query_quotient_sources({n+64,64,n},{n+64,64,n},n,{4,seed_cap},{32u<<20,4u<<20,128u<<20},{},*plan):
        query_stored_quotient(av,dv,n,{4,seed_cap},{32u<<20,4u<<20,128u<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"quotient program query %zu status%u\n",n,unsigned(status));assert(status==planning::ExternalPlanStatus::ready);
    if(short_words){const auto geometry=plan->refinement.residual_product.spec;const auto bytes=plan->storage_bytes,writes=plan->worst_write_bytes;
        allocation_watch_start();assert(rebind_stored_quotient(*plan,av,dv));assert(!allocation_watch_stop());
        const auto&actual=plan->refinement.residual_product.spec;
        assert(bytes==plan->storage_bytes&&writes==plan->worst_write_bytes&&geometry.primes==actual.primes&&geometry.trunk_bits==actual.trunk_bits&&
               geometry.outer_cells==actual.outer_cells&&geometry.inner_cells==actual.inner_cells&&geometry.prime_batch==actual.prime_batch);}
    Fixture f(4,false,32u<<20);auto memory=f.allocate(plan->storage_bytes,plan->alignment);const size_t offset=static_cast<uint8_t *>(memory.data)-f.arena->base;
    f.leases.pop_back();sbn3_arena_release(f.arena,&memory);const auto before=t.counters;StoredInteger result;
    allocation_watch_start();const auto e=run_stored_quotient(*plan,t,store,av,dv,result,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"quotient program %zu error %u %s\n",n,unsigned(e.code),e.where);assert(!e);
    assert(f.arena->unleased(offset,plan->storage_bytes)&&store.usage().peak_bytes<=plan->peak_backing_bytes);
    assert(t.counters.write_submitted-before.write_submitted<=plan->worst_write_bytes&&result.exponent2==plan->result_exponent2);
    ref_int A,D,Q,error,bound;ref_inits(A,D,Q,error,bound,nullptr);ref_import(A,a.size(),-1,8,0,0,a.data());ref_import(D,d.size(),-1,8,0,0,d.data());
    load(t,io,result,Q);ref_mul(error,Q,D);ref_mul_ui(bound,D,7);const int64_t scale=result.exponent2+dv.exponent2-av.exponent2;
    if(scale<0)ref_mul_2exp(A,A,size_t(-scale));else {ref_mul_2exp(error,error,size_t(scale));ref_mul_2exp(bound,bound,size_t(scale));}
    ref_sub(error,error,A);ref_abs(error,error);assert(ref_cmp(error,bound)<0);ref_clears(A,D,Q,error,bound,nullptr);
    result.value.reset();const auto live=store.usage().live_bytes;disk.cancel_after_write=disk.writes+3;
    assert(run_stored_quotient(*plan,t,store,av,dv,result,*f.arena,offset,*f.team).code==IoCode::cancelled);
    assert(!result.value&&store.usage().live_bytes==live&&f.arena->unleased(offset,plan->storage_bytes));disk.cancelled=false;
    printf("Stored N/D n=%zu native_seed=%zu inverse_rungs=%zu pattern=%u short%zu: <7 original-ratio ulps, scale/caps/cancel/leases PASS\n",n,plan->inverse.seed_limbs,plan->inverse.rung_count,pattern,short_words);
}
static void series_finish(const char*label,const series::FormulaDef&definition,uint64_t terms){
    using namespace series;constexpr size_t precision=16,target=14;
    auto formula=std::make_unique<DataFormula>();assert(data_formula_prepare(definition,definition.begin+terms,*formula));
    const PsrSpec spec{data_finite_formula(*formula),{definition.begin,definition.begin+terms},{4,8,0,0,0},
        {formula.get(),[](const void*p,uint64_t i){return static_cast<const DataFormula *>(p)->attenuation_bits(i);},4,83,0},precision};
    SeriesMetadata layout{};assert(query_series_metadata(64,32,layout));Aligned metadata(layout.bytes);SeriesPlan series_plan;
    allocation_watch_start();assert(compile_series(spec,{32u<<20,4u<<20,64u<<20},{true,2},{},layout,metadata.span(),series_plan)==planning::ExternalPlanStatus::ready);assert(!allocation_watch_stop());
    ValueStoreStorage storage{};assert(query_value_store(128,storage));Aligned state(storage.bytes);Device disk(64u<<20);auto t=disk.transport();ValueStore store;
    assert(!store.open(604,704,0,disk.bytes.size(),state.span(),128));Fixture f(4,false,32u<<20);
    auto io=f.allocate(series_plan.io_bytes,4096);const size_t work_index=f.leases.size();auto work=f.allocate(series_plan.work_bytes,series_plan.work_alignment);
    const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;SeriesResult values;allocation_watch_start();
    assert(!run_series(series_plan,t,store,*f.arena,*f.team,work,{static_cast<uint8_t *>(io.data),io.bytes},values));assert(!allocation_watch_stop());assert(values.in_store);
    assert(t.counters.write_submitted<=series_plan.write_bytes);const uint64_t series_writes=t.counters.write_submitted;f.retire(work_index);
    auto plan=std::make_unique<QuotientPlan>();allocation_watch_start();
    const auto status=query_stored_quotient(values.stored[0],values.stored[1],target,{4,7},
        {32u<<20,(4u<<20)+layout.bytes+io.bytes,64u<<20},{},*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"series quotient %s status%u N%zu D%zu\n",label,unsigned(status),values.stored[0].limbs(),values.stored[1].limbs());
    assert(status==planning::ExternalPlanStatus::ready&&!(uintptr_t(work.data)&(plan->alignment-1)));
    sbn3_error preparation{};assert(sbn3_arena_prepare(f.arena,offset,plan->storage_bytes,&preparation)==SBN3_OK);
    StoredInteger output;allocation_watch_start();assert(!run_stored_quotient(*plan,t,store,values.stored[0],values.stored[1],output,*f.arena,offset,*f.team));assert(!allocation_watch_stop());
    assert(f.arena->unleased(offset,plan->storage_bytes)&&t.counters.write_submitted-series_writes<=plan->worst_write_bytes);
    formula_reference::Ref expected;formula_reference::reference(definition,definition.begin,definition.begin+terms,expected);
    StoreBuffers buffers{{static_cast<uint8_t *>(io.data),1u<<20},{static_cast<uint8_t *>(io.data)+(1u<<20),index_page_bytes}};
    ref_int Q,error,bound,N;ref_inits(Q,error,bound,N,nullptr);load(t,buffers,output,Q);ref_mul(error,Q,expected.d);ref_mul_ui(bound,expected.d,8);ref_set(N,expected.t);
    const int64_t shift=output.exponent2-expected.exponent;
    if(shift<0)ref_mul_2exp(N,N,size_t(-shift));else {ref_mul_2exp(error,error,size_t(shift));ref_mul_2exp(bound,bound,size_t(shift));}
    ref_sub(error,error,N);ref_abs(error,error);assert(ref_cmp(error,bound)<0);ref_clears(Q,error,bound,N,nullptr);
    printf("Stored series->quotient %-10s blocks%zu nodes%zu: same arena interval, preserved N/D, exact finite-series error <8 output ulps PASS\n",label,series_plan.blocks.size(),series_plan.nodes.size());
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);subtraction();
    for(int sign:{-1,0,1}){
        cyclic_subtraction(64,127,4,4,80,13,sign,0);
        cyclic_subtraction(15001,30001,512,6,128,-17,sign,1);
        cyclic_subtraction(15001,30001,8192,6,128,13,sign,0,true);
    }
    rung(3,4,0,0);rung(3,5,17,5);rung(129,255,63,-6);rung(4097,8191,1,5);
    complete(127,SIZE_MAX,0);complete(127,15,0);complete(1023,15,1);complete(8191,127,0);
    complete(257,15,0,1);complete(257,SIZE_MAX,0,5);
    series_finish("e",series::euler_definition(),413);series_finish("Chudnovsky",series::chudnovsky_definition(),221);
    series_finish("BinaryBBP",series::binary_log_definition(3),511);
    assert(cyclic_rungs);
    puts("Stored quotient: shared half reciprocal, projected modular cancellation, complete scale and storage contracts PASS");
}
