// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "formula_reference.hpp"
#include "external/series.hpp"
using namespace formula_reference;
static unsigned native_return_loans=0;
static uint64_t decay(const void*p,uint64_t at){return static_cast<const DataFormula *>(p)->attenuation_bits(at);}
static void expected_state(const PsrSpec&s,Ref&expected,int64_t (&exponent)[2]){
    PsrInfo info{};assert(psr_query(s,info)==SBN3_SUPPORTED);Fixture f(s.series.workers,false,32u<<20);
    const size_t at=up(f.base+f.cursor,info.storage_alignment)-f.base;f.cursor=up(at+info.storage_bytes,4096)+4096;
    sbn3_error error{};assert(sbn3_arena_prepare(f.arena,at,info.storage_bytes,&error)==SBN3_OK);
    auto pair=f.allocate(info.output_limbs*16,128);sbn3_series_values result{};
    for(unsigned j=0;j<2;++j)result.value[j].mantissa={static_cast<uint64_t *>(pair.data)+j*info.output_limbs,info.output_limbs,0,0};
    PsrBinding binding;psr_prepare(s,info,*f.arena,*f.team,at,binding);psr_execute(binding,result);psr_release(binding);
    ref_number*dest[]{expected.t,expected.d};
    for(unsigned j=0;j<2;++j){const auto&v=result.value[j];ref_import(dest[j],v.mantissa.size,-1,8,0,0,v.mantissa.data);if(v.mantissa.negative)ref_neg(dest[j],dest[j]);exponent[j]=v.exponent2;}
}
static void one(const char*label,const FormulaDef&def,uint64_t terms,size_t precision,unsigned depth,unsigned workers=1,bool store_output=false,uint64_t max_terms=83){
    auto formula=std::make_unique<DataFormula>();const auto end=def.begin+terms;assert(data_formula_prepare(def,end,*formula));
    const PsrSpec spec{data_finite_formula(*formula),{def.begin,end},{workers,8,0,0,0},{formula.get(),decay,4,max_terms,0},precision};
    Ref expected;int64_t expected_exponent[2];expected_state(spec,expected,expected_exponent);
    SeriesMetadata meta{};assert(query_series_metadata(64,32,meta));Aligned metadata(meta.bytes);
    SeriesPlan plan;allocation_watch_start();
    const auto status=compile_series(spec,{32u<<20,4u<<20,32u<<20},{store_output,depth},{},meta,metadata.span(),plan);
    assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready)fprintf(stderr,"series compile %s depth%u status%u\n",label,depth,unsigned(status));
    assert(status==planning::ExternalPlanStatus::ready&&plan.total_resident_bytes<=(32u<<20));
    assert(depth?(plan.resident.first==1&&plan.nodes.size()>=3):plan.resident.first==0);
    unsigned promoted=0,borrowed=0;
    auto inspect=[&](const StoredMergePlan&m){for(unsigned i=0;i<m.stage_count;++i){promoted+=m.stage[i].use_resident;borrowed+=m.stage[i].borrow_result;}};
    for(const auto&node:plan.nodes)if(!node.native)inspect(node.merge);
    unsigned returns=0;for(const auto&node:plan.nodes)returns+=unsigned(__builtin_popcount(node.loan_mask));native_return_loans+=returns;
    for(size_t i=0;i<plan.resident.first;++i)if(plan.blocks[i].block.next_precision)inspect(plan.blocks[i].merge);
    if(depth)assert(promoted&&borrowed);
    ValueStoreStorage storage{};assert(query_value_store(128,storage));Aligned values(storage.bytes);
    Device disk(32u<<20);auto transport=disk.transport();ValueStore store;assert(!store.open(101,201,0,disk.bytes.size(),values.span(),128));
    Fixture fixture(workers,false,32u<<20);const auto work=fixture.allocate(plan.work_bytes,plan.work_alignment),io=fixture.allocate(plan.io_bytes,4096);
    SeriesResult result;allocation_watch_start();
    const auto error=run_series(plan,transport,store,*fixture.arena,*fixture.team,work,{static_cast<uint8_t *>(io.data),io.bytes},result);
    assert(!allocation_watch_stop());
    if(error)fprintf(stderr,"series execute %s code%u %s need%llu have%llu\n",label,unsigned(error.code),error.where,(unsigned long long)error.need,(unsigned long long)error.have);
    assert(!error&&result.in_store==(bool(depth)||store_output));
    const auto writes=transport.counters.write_submitted;assert(writes<=plan.write_bytes&&store.usage().peak_bytes<=plan.peak_backing_bytes);
    if(!depth&&!store_output)assert(!writes&&!transport.counters.read_submitted&&plan.nodes.empty());
    ref_int got,want;ref_inits(got,want,nullptr);const ref_number*reference[]{expected.t,expected.d};
    StoreBuffers buffers{{static_cast<uint8_t *>(io.data),1u<<20},{static_cast<uint8_t *>(io.data)+(1u<<20),index_page_bytes}};
    for(unsigned j=0;j<2;++j){int64_t exponent;
        if(result.in_store){const auto&v=result.stored[j];std::vector<uint64_t> words(v.limbs());ValueReader reader;
            assert(!reader.open(transport,v.value.region(),buffers));assert(!reader.read(v.first_limb*8,{reinterpret_cast<uint8_t *>(words.data()),words.size()*8}));
            ref_import(got,words.size(),-1,8,0,0,words.data());if(v.negative)ref_neg(got,got);exponent=v.exponent2;
        }else{const auto&v=result.resident.value[j];ref_import(got,v.mantissa.size,-1,8,0,0,v.mantissa.data);if(v.mantissa.negative)ref_neg(got,got);exponent=v.exponent2;}
        ref_set(want,reference[j]);const auto low=std::min(exponent,expected_exponent[j]);
        ref_mul_2exp(got,got,size_t(exponent-low));ref_mul_2exp(want,want,size_t(expected_exponent[j]-low));
        if(ref_cmp(got,want))fprintf(stderr,"series mismatch %s depth%u component%u\n",label,depth,j);
        assert(!ref_cmp(got,want));
    }
    ref_clears(got,want,nullptr);
    printf("Series %-10s W%u depth%u blocks%zu nodes%zu resident_suffix%zu writes=%llu/%llu RAM=%zu: resident N/D match PASS\n",label,workers,depth,plan.blocks.size(),plan.nodes.size(),plan.resident.first,(unsigned long long)writes,(unsigned long long)plan.write_bytes,plan.total_resident_bytes);
    if(depth)printf("  native products=%u, borrowed numerator outputs=%u; exact result and consuming lifetimes PASS\n",promoted,borrowed);
    if(returns)printf("  native return loans=%u; exact resident N/D and IO/cancellation bounds PASS\n",returns);
    if(depth){
        result.stored={};const auto held=store.usage().live_bytes;const auto prior=transport.counters;transport.write_limit=prior.write_submitted+plan.write_bytes-1;
        SeriesResult rejected;assert(run_series(plan,transport,store,*fixture.arena,*fixture.team,work,{static_cast<uint8_t *>(io.data),io.bytes},rejected).code==IoCode::capacity);
        assert(!rejected.in_store&&transport.counters.write_submitted==prior.write_submitted&&store.usage().live_bytes==held);
        transport.write_limit=UINT64_MAX;disk.cancel_after_write=disk.writes+1;
        assert(run_series(plan,transport,store,*fixture.arena,*fixture.team,work,{static_cast<uint8_t *>(io.data),io.bytes},rejected).code==IoCode::cancelled);
        assert(!rejected.in_store&&store.usage().live_bytes==held);disk.cancelled=false;disk.cancel_after_write=0;
        SeriesPlan rejected_plan=plan;
        assert(compile_series(spec,{32u<<20,4u<<20,32u<<20,0,plan.write_bytes-1},{store_output,depth},{},meta,metadata.span(),rejected_plan)==planning::ExternalPlanStatus::no_admitted_plan);
        assert(!rejected_plan.spec);
    }
}
int main(){
    setvbuf(stdout,nullptr,_IOLBF,0);
    one("e",euler_definition(),361,12,0);one("pi",chudnovsky_definition(),221,60,0,4);
    one("e stored",euler_definition(),361,12,0,1,true);
    one("e prefix",euler_definition(),361,12,2);one("signed",exp_minus_one_definition(),413,16,2);
    one("pi prefix",chudnovsky_definition(),221,60,2);one("BBP prefix",binary_log_definition(3),511,20,2);
    one("e wide",euler_definition(),6001,1100,2,4,false,0);
    assert(native_return_loans);
    puts("Complete bounded series: compiled prefix/suffix, native return loans, finite/tail rounding, whole write/backing gates and no allocations PASS");
}
