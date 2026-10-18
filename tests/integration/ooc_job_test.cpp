// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/job.hpp"
#include "formula_reference.hpp"
#ifdef SBN3_TEST_FILE_STORE
#include "file_store_support.hpp"
#endif
using namespace sbn::v3;
static void arctan(ref_number*out,unsigned q,size_t words){
    ref_int p,t,d;ref_inits(p,t,d,nullptr);ref_set_ui(p,1);ref_mul_2exp(p,p,64*words);ref_set_ui(d,q);ref_fdiv_q(p,p,d);ref_set_ui(out,0);
    for(uint64_t k=0;ref_sgn(p);++k){ref_set_ui(d,2*k+1);ref_fdiv_q(t,p,d);if(k&1)ref_sub(out,out,t);else ref_add(out,out,t);ref_set_ui(d,q*q);ref_fdiv_q(p,p,d);}
    ref_clears(p,t,d,nullptr);
}
static void reference(bool pi,const series::FormulaDef&def,uint64_t terms,size_t words,ref_number*out){
    ref_int a,b;ref_inits(a,b,nullptr);
    if(pi){arctan(a,5,words+3);arctan(b,239,words+3);ref_mul_ui(a,a,16);ref_mul_ui(b,b,4);ref_sub(out,a,b);ref_fdiv_q_2exp(out,out,192);}
    else {formula_reference::Ref finite;formula_reference::reference(def,def.begin,def.begin+terms,finite);ref_set(a,finite.t);ref_set(b,finite.d);
        const int64_t shift=int64_t(64*words)+finite.exponent;if(shift>=0)ref_mul_2exp(a,a,size_t(shift));else ref_mul_2exp(b,b,size_t(-shift));
        ref_fdiv_q(out,a,b);if(def.recipe==SBN3_SERIES_HYPERDESCENT){ref_set_ui(a,1);ref_mul_2exp(a,a,64*words);ref_add(out,out,a);}}
    ref_clears(a,b,nullptr);
}
static void one(unsigned kind,uint64_t digits){
    using namespace series;
    const auto def=kind==0?euler_definition():kind==1?chudnovsky_definition():binary_log_definition(3);
    TerminalSpec terminal=kind==0?TerminalSpec{0,1,1,1,1,false}:kind==1?TerminalSpec{23,24,4270934400ULL,10005,0,true}:TerminalSpec{-3,-2,1,1,0,false};
    JobPrecision precision;assert(query_job_precision(10,digits,terminal,128,1,precision));
    const uint64_t terms=kind==1?uint64_t(2*precision.series):uint64_t(32*precision.series);
    auto formula=std::make_unique<DataFormula>();assert(data_formula_prepare(def,def.begin+terms,*formula));
    JobSpec spec{};spec.series={data_finite_formula(*formula),{def.begin,def.begin+terms},{4,8,0,0,0},
        {formula.get(),[](const void*p,uint64_t i){return static_cast<const DataFormula *>(p)->attenuation_bits(i);},4,83,0},precision.series};
    spec.terminal=terminal;spec.digits=digits;spec.newton.maximum_native_seed=7;spec.force_stored_depth=2;
    spec.format.maximum_resident_fragments=16;spec.format.radix.use_alphabet=1;sbn3_radix_alphabet(10,spec.format.radix.alphabet);
    SeriesMetadata meta;assert(query_series_metadata(128,64,meta));Aligned metadata(meta.bytes);auto plan=std::make_unique<JobPlan>();
    allocation_watch_start();const auto status=compile_job(spec,{64u<<20,4u<<20,128u<<20},{},meta,metadata.span(),*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"job compile k%u status%u fixed%zu bytes%zu RAM%zu\n",kind,unsigned(status),plan->common_fixed_bytes,plan->storage_bytes,plan->total_resident_bytes);
    assert(status==planning::ExternalPlanStatus::ready&&plan->total_resident_bytes<=(64u<<20));
#ifdef SBN3_TEST_FILE_STORE
    FileDevice disk(192,plan->peak_backing_bytes,kind==1);auto t=disk.transport();auto&store=disk.store;
#else
    ValueStoreStorage q;assert(query_value_store(192,q));Aligned state(q.bytes);Device disk(128u<<20);auto t=disk.transport();ValueStore store;
    assert(!store.open(1201,1301,0,disk.bytes.size(),state.span(),192));
#endif
    Fixture f(4,false,64u<<20);auto work=f.allocate(plan->storage_bytes,plan->alignment);
    const size_t offset=static_cast<uint8_t *>(work.data)-f.arena->base;f.leases.pop_back();sbn3_arena_release(f.arena,&work);JobState job;
    plan->write_limit=plan->write_bytes-1;assert(advance_job(*plan,job,t,store,*f.arena,offset,*f.team).code==IoCode::capacity);assert(!t.counters.write_submitted&&!store.usage().live_bytes);
    plan->write_limit=UINT64_MAX;
    allocation_watch_start();assert(!advance_job(*plan,job,t,store,*f.arena,offset,*f.team));assert(!allocation_watch_stop());assert(job.next==JobPhase::quotient);
    assert(f.arena->unleased(offset,plan->storage_bytes));const auto held=store.usage().live_bytes;const auto first_id=job.pair[0].value.region().id;
    StoredValue checkpoint[2];for(unsigned i=0;i<2;++i)job.pair[i].value.share(checkpoint[i]);
    disk.fail_write=disk.writes+1;const auto prior=t.counters;
    allocation_watch_start();const auto failed=advance_job(*plan,job,t,store,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    assert(failed.code==IoCode::io&&job.next==JobPhase::quotient&&job.pair[0].value.region().id==first_id&&store.usage().live_bytes==held);
    assert(t.counters.write_submitted>prior.write_submitted&&job.charged_write_bytes==t.counters.write_submitted&&f.arena->unleased(offset,plan->storage_bytes));disk.fail_write=0;
    disk.cancelled=true;assert(advance_job(*plan,job,t,store,*f.arena,offset,*f.team).code==IoCode::cancelled);assert(job.next==JobPhase::quotient);disk.cancelled=false;
    allocation_watch_start();assert(!advance_job(*plan,job,t,store,*f.arena,offset,*f.team));assert(!allocation_watch_stop());
    assert(job.working.value&&!job.pair[0].value&&checkpoint[0].region().id==first_id);for(auto&pin:checkpoint)pin.reset();
    allocation_watch_start();const auto e=run_job(*plan,job,t,store,*f.arena,offset,*f.team);assert(!allocation_watch_stop());
    if(e)fprintf(stderr,"job k%u phase%u error%u %s\n",kind,unsigned(job.next),unsigned(e.code),e.where);assert(!e&&job.next==JobPhase::complete);
    assert(job.text.certified_digits==digits&&job.text.integer_part==(kind==0?2:kind==1?3:0)&&job.charged_write_bytes==t.counters.write_submitted);
    assert(store.usage().peak_bytes<=plan->peak_backing_bytes&&f.arena->unleased(offset,plan->storage_bytes));
    Aligned io_data(4*chunk_bytes),index(index_page_bytes);StoreBuffers io{io_data.span(),index.span()};ValueReader reader;
    std::vector<uint64_t>binary(precision.fractional+1);assert(!reader.open(t,job.binary.value.region(),io));
    assert(!reader.read(0,{reinterpret_cast<uint8_t *>(binary.data()),binary.size()*8}));ref_int want,got;ref_inits(want,got,nullptr);
    reference(kind==1,def,terms,precision.fractional,want);ref_import(got,binary.size(),-1,8,0,0,binary.data());assert(!ref_cmp(got,want));ref_clears(want,got,nullptr);
    std::vector<uint8_t>text(digits);assert(!reader.open(t,job.text.digits.region(),io));assert(!reader.read(0,text));binary.resize(precision.fractional);
    for(size_t i=0;i<digits;++i){uint64_t carry=0;for(auto&word:binary){const __uint128_t x=__uint128_t(word)*10+carry;word=uint64_t(x);carry=uint64_t(x>>64);}assert(text[i]=='0'+carry);}
    const auto complete=t.counters;assert(!advance_job(*plan,job,t,store,*f.arena,offset,*f.team));assert(t.counters.write_submitted==complete.write_submitted);
    printf("Generic stored job family%u D%llu: precompiled choices, retry-preserved boundary, write charge, exact binary/decimal and cleanup PASS (RAM%zu backing%llu)\n",
        kind,(unsigned long long)digits,plan->total_resident_bytes,(unsigned long long)plan->peak_backing_bytes);
#ifdef SBN3_TEST_FILE_STORE
    assert(disk.files.usage().peak_claimed_bytes<=plan->peak_backing_bytes);
    assert(disk.files.usage().write_bytes==t.counters.write_completed);
    job={};assert(!store.usage().live_regions);disk.close();
    printf("File job family%u %s: peak file claims within job bound, IO bytes reconcile, all %llu objects explicitly reclaimed PASS\n",kind,
        kind==1?"tmpfs DIRECT":"memfd",(unsigned long long)disk.files.usage().removed);
#endif
}
[[maybe_unused]] static void large(){
    using namespace series;constexpr size_t GiB=size_t(1)<<30;TerminalSpec terminal{23,24,4270934400ULL,10005,0,true};
    JobPrecision precision;assert(query_job_precision(10,5000000000ULL,terminal,128,1,precision));Formula formula{FormulaKind::Chudnovsky};
    const uint64_t terms=(64*uint64_t(precision.series)+160)/47;JobSpec spec;
    spec.series={finite_formula(formula),{0,terms},{16,8,0,0,0},{nullptr,[](const void*,uint64_t at){return at?47*(at-1):0;},128,0,64.*16384*16},precision.series};
    spec.terminal=terminal;spec.digits=5000000000ULL;
    SeriesMetadata layout;assert(query_series_metadata(128,256,layout));Aligned metadata(layout.bytes);auto plan=std::make_unique<JobPlan>();
    allocation_watch_start();const auto status=compile_job(spec,{4*GiB,64u<<20,256*uint64_t(GiB),0,1024*uint64_t(GiB)},{},layout,metadata.span(),*plan);assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"large job status%u fixed%zu bytes%zu total%zu series_prefix%zu\n",unsigned(status),plan->common_fixed_bytes,plan->storage_bytes,plan->total_resident_bytes,plan->series.resident.first);
    assert(status==planning::ExternalPlanStatus::ready&&plan->total_resident_bytes<=4*GiB);
    printf("5B/4GiB WHOLE JOB QUERY ONLY: F%zu N%zu P%zu series_prefix%zu radix_frontiers%llu RAM%zu backing%llu writes%llu regions%zu PASS (no operands/arithmetic/IO)\n",
        precision.fractional,precision.working,precision.series,plan->series.resident.first,(unsigned long long)plan->format.node[0].frontiers,
        plan->total_resident_bytes,(unsigned long long)plan->peak_backing_bytes,(unsigned long long)plan->write_bytes,plan->minimum_regions);
}
int main(){setvbuf(stdout,nullptr,_IOLBF,0);one(0,200);one(1,1151);one(2,200);
#ifndef SBN3_TEST_FILE_STORE
    large();
#endif
    puts("Generic constant job: one precompiled phase region, propagated precision certificate, owned safe points and cumulative write admission PASS");}
