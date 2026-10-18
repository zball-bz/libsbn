// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "planning/product.hpp"
#include "runtime/team.hpp"
using namespace sbn::v3::planning;
static void query_cases(){
    constexpr size_t GiB=size_t(1)<<30,MiB=size_t(1)<<20;
    ExternalProductProfile profile{};
    for(size_t n:{size_t(1)<<20,size_t(103810254),size_t(259525633),(size_t(1)<<28)+12345})
        for(size_t ram:{4*GiB,6*GiB,8*GiB,12*GiB}){
            ExternalProductSelection s{};ExternalProductRequest r{n,n-17,16,0,0,false};
            ExternalProductBudget b{ram,64*MiB};allocation_watch_start();
            const auto rc=choose_external_product(r,b,profile,s);assert(!allocation_watch_stop());
            assert(rc==ExternalPlanStatus::ready&&s.summary_quotes<=7&&s.basis_queries<=1&&s.total_resident_bytes<=ram);
            const auto&p=s.plan;assert(p.spec.retained_bytes%chunk_bytes==0&&p.storage_bytes+p.shared_bytes+b.other_live_bytes==s.total_resident_bytes);
            assert(p.forward[0].spec.stripe_cells==p.spec.forward_stripe_cells&&p.spec.forward_stripe_cells>=p.spec.stripe_cells);
            assert(p.inverse.spec.stripe_cells>=p.spec.stripe_cells&&p.emit.stripe_cells==p.inverse.spec.stripe_cells);
            auto baseline=p;assert(retain_spilled_prefix(baseline,0));
            assert(baseline.normal_write_bytes-p.normal_write_bytes==2*p.spec.retained_bytes);
            assert(baseline.normal_read_bytes-p.normal_read_bytes>=2*p.spec.retained_bytes);
            if(p.spec.compact_middle){
                assert(p.middle.pipeline&&p.middle.compact&&p.spec.raw_operand==2);
                assert(p.middle.storage_bytes==p.middle.compact_storage_bytes&&p.middle.storage_bytes<=p.pool_bytes);
            }
            printf("External plan n=%zu RAM=%zuGiB NP%u T%u C%zu M%zu J%zu g%u retained=%.3fGiB total=%.3fGiB R/W=%.3f/%.3fGiB quotes=%u PASS\n",
                n,ram/GiB,p.spec.primes,p.spec.trunk_bits,p.spec.outer_cells,p.spec.inner_cells,p.spec.stripe_cells,p.spec.prime_batch,
                double(p.spec.retained_bytes)/GiB,double(s.total_resident_bytes)/GiB,double(p.normal_read_bytes)/GiB,double(p.normal_write_bytes)/GiB,s.summary_quotes);
            fflush(stdout);
        }
    ExternalProductSelection rejected{};
    assert(choose_external_product({100000,90000,4,6,128,false},{4096,4096},profile,rejected)==ExternalPlanStatus::no_admitted_plan);
    ExternalProductBudget disk{64*MiB,4*MiB,1};
    assert(choose_external_product({100000,90000,4,6,128,false},disk,profile,rejected)==ExternalPlanStatus::no_admitted_plan);
    ExternalProductRequest large{379625185,379625168,16};ExternalProductBudget grant{4*GiB,64*MiB};
    ExternalProductSelection streamed;assert(choose_external_product(large,grant,profile,streamed)==ExternalPlanStatus::ready);
    assert(streamed.plan.spec.raw_operand<2||streamed.plan.middle.compact);
    auto spec=streamed.plan.spec;spec.compact_middle=spec.duplex_middle=spec.stream_final=false;spec.retained_bytes=0;SpilledProductPlan serial;
    assert(query_spilled_product(spec,serial));
    const size_t room=grant.resident_bytes-grant.other_live_bytes-serial.shared_bytes-serial.storage_bytes;
    assert(retain_spilled_prefix(serial,std::min(size_t(serial.spec.primes*serial.plane_bytes),room/chunk_bytes*chunk_bytes)));
    assert(serial.normal_write_bytes<=streamed.plan.normal_write_bytes||streamed.plan.spec.raw_operand<2);
    grant.write_limit=streamed.plan.normal_write_bytes-2*chunk_bytes;ExternalProductSelection capped;
    assert(choose_external_product(large,grant,profile,capped)==ExternalPlanStatus::ready);
    assert(capped.plan.normal_write_bytes<=grant.write_limit);
    auto cached=streamed.plan;assert(use_cached_a(cached)&&use_completed_cached_a(cached));
    assert(cached.middle.spec.a_transformed);
    auto stored_spec=streamed.plan.spec;stored_spec.raw_operand=2;stored_spec.raw_cache_bytes=SIZE_MAX;stored_spec.retained_bytes=0;stored_spec.compact_middle=true;
    SpilledProductPlan raw;assert(query_spilled_product(stored_spec,raw));
    assert(use_raw_operand(raw,0)&&!raw.spec.compact_middle&&raw.middle.recycle_a);
    grant.write_limit=UINT64_MAX;large.bn=large.an;ExternalProductSelection equal;
    assert(choose_external_product(large,grant,profile,equal)==ExternalPlanStatus::ready);
    assert(use_cached_square(equal.plan)&&!equal.plan.spec.compact_middle);
    assert(use_completed_cached_a(equal.plan));
    for(size_t n:{size_t(337379336),size_t(337379337),size_t(337379338),size_t(348117717),size_t(379625062)}){
        ExternalProductSelection p;assert(choose_external_product({n,n-17,16},{4*GiB,64*MiB},profile,p)==ExternalPlanStatus::ready);
        assert(p.total_resident_bytes<=4*GiB&&p.plan.spec.raw_operand<2);
        if(n>=337379337)assert(p.plan.spec.raw_cache_bytes!=SIZE_MAX&&p.plan.raw_cached_bytes<p.plan.forward[p.plan.spec.raw_operand].input_bytes);
        printf("Raw boundary n=%zu: cached=%zu source=%llu R/W=%llu/%llu total=%zu PASS\n",n,p.plan.raw_cached_bytes,
            (unsigned long long)p.plan.forward[p.plan.spec.raw_operand].input_bytes,(unsigned long long)p.plan.normal_read_bytes,
            (unsigned long long)p.plan.normal_write_bytes,p.total_resident_bytes);
    }
    puts("Compact middle admission: unchanged total grant, explicit retention trade, hard write cap preserved PASS");
}
static void execute(){
    ExternalProductRequest request{65009,52003,4,10,224,false};
    ExternalProductBudget budget{10u<<20,4u<<20};ExternalProductSelection chosen{};
    assert(choose_external_product(request,budget,{},chosen)==ExternalPlanStatus::ready);
    const auto&p=chosen.plan;assert(p.spec.retained_bytes);
    auto a=fresh(request.an*8,0,1),b=fresh(request.bn*8,a.layout.allocation_bytes,2);
    uint64_t at=b.layout.data_offset+b.layout.allocation_bytes;std::array<Region,20> temp{};
    for(unsigned q=0;q<p.plane_count;++q){temp[q]=fresh(size_t(p.plane_bytes),at,100+q,RegionKind::scratch);at+=temp[q].layout.allocation_bytes;}
    auto result=fresh(size_t(p.output_bytes),at,1000);at+=result.layout.allocation_bytes;
    Device disk{size_t(at)};auto transport=disk.transport();Aligned av(request.an*8),bv(request.bn*8),data(4*chunk_bytes),index(index_page_bytes),output(size_t(p.output_bytes));
    fill(av.span());fill(bv.span());StoreBuffers io{data.span(),index.span()};write_value(transport,a,io,av.span());write_value(transport,b,io,bv.span());
    Fixture fixture(4,false,64u<<20);const auto storage=fixture.allocate(p.storage_bytes,4096);
    __atomic_store_n(&fixture.team->idle_spin_ns,uint64_t(123456),__ATOMIC_RELEASE);
    const auto before=transport.counters;allocation_watch_start();
    const auto error=run_selected_product(chosen,transport,a,b,result,{temp.data(),p.plane_count},
                                 {static_cast<uint8_t *>(storage.data),storage.bytes},*fixture.team);
    if(error)fprintf(stderr,"Selected product error %u %s offset=%llu need=%llu have=%llu raw=%u retained=%zu pool=%zu final=%zu\n",unsigned(error.code),error.where,
        (unsigned long long)error.offset,(unsigned long long)error.need,(unsigned long long)error.have,p.spec.raw_operand,p.spec.retained_bytes,p.pool,p.inverse_offset);
    assert(!error);assert(!allocation_watch_stop());
    assert(__atomic_load_n(&fixture.team->idle_spin_ns,__ATOMIC_ACQUIRE)==123456);
    assert(transport.counters.read_submitted-before.read_submitted==p.normal_read_bytes);
    assert(transport.counters.write_submitted-before.write_submitted==p.normal_write_bytes);
    ValueReader reader;assert(!reader.open(transport,result,io));assert(!reader.read(0,output.span()));
    assert(ref_product_equal(reinterpret_cast<uint64_t *>(av.data),request.an,reinterpret_cast<uint64_t *>(bv.data),request.bn,
                             reinterpret_cast<uint64_t *>(output.data),request.an+request.bn));
    puts("Selected retained product: total-budget plan executed, exact IO totals and independent result PASS");
}
int main(){query_cases();execute();puts("External selection: bounded quotes, total RAM/disk/write admission and executable retention PASS");}
