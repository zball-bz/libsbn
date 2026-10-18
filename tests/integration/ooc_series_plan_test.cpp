// SPDX-License-Identifier: LGPL-3.0-or-later
#include "ooc_store_support.hpp"
#include "external/series.hpp"
using namespace sbn::v3::series;
static uint64_t decay(const void *,uint64_t at){return at?47*(at-1):0;}
int main(){
    constexpr size_t GiB=size_t(1)<<30;
    struct Case {unsigned digits_b,ram_gib;size_t words;};
    const Case cases[]{{2,4,103810254+4},{2,6,103810254+4},{2,8,103810254+4},{2,12,103810254+4},{5,4,259525633+4}};
    Formula formula{FormulaKind::Chudnovsky};SeriesMetadata layout{};
    assert(query_series_metadata(128,256,layout));Aligned metadata(layout.bytes);
    for(const auto&c:cases){
    const uint64_t terms=(uint64_t(c.words)*64+110)/47;
    const PsrSpec spec{finite_formula(formula),{0,terms},{16,8,0,0,0},{nullptr,decay,128,0,64.*16384*16},c.words};SeriesPlan plan;
    const uint64_t start=tick();allocation_watch_start();
    const auto status=compile_series(spec,{c.ram_gib*GiB,64u<<20,256*uint64_t(GiB)},{},{},layout,metadata.span(),plan);
    assert(!allocation_watch_stop());
    if(status!=planning::ExternalPlanStatus::ready){
        auto*nodes=reinterpret_cast<SeriesNode *>(metadata.data+layout.nodes);size_t used=0;while(used<layout.node_capacity&&nodes[used].range.end)++used;
        fprintf(stderr,"series query %uB/%uGiB status=%u initialized_nodes=%zu metadata=%zu\n",c.digits_b,c.ram_gib,unsigned(status),used,layout.bytes);
    }
    assert(status==planning::ExternalPlanStatus::ready&&plan.total_resident_bytes<=c.ram_gib*GiB);
    if(c.digits_b==5)assert(plan.resident.first>0&&!plan.nodes.empty());
    unsigned rebatched=0;
    for(size_t i=plan.resident.first;i<plan.blocks.size();++i){
        const auto&e=plan.blocks[i].resident;const auto&m=e.merge[unsigned(plan.choices[i])];
        assert(m.work_offset+m.work_bytes<=plan.resident.envelope[planning::Slot::workspace].bytes);
        if(e.block.next_precision){
            assert(m.work_bytes<=m.fast_work_bytes);
            rebatched+=m.multiply.info.prime_batch<m.original_batch[0];
            rebatched+=m.contraction.info.prime_batch<m.original_batch[1];
        }
    }
    printf("%uB-scale pi / %uGiB query only: terms=%llu blocks=%zu resident_first=%zu nodes=%zu depth=%u RAM=%zu backing=%llu writes=%llu metadata=%zu queries=%llu/%llu rebatched=%u elapsed_ms=%.3f PASS\n",
        c.digits_b,c.ram_gib,(unsigned long long)terms,plan.blocks.size(),plan.resident.first,plan.nodes.size(),plan.max_depth,plan.total_resident_bytes,
        (unsigned long long)plan.peak_backing_bytes,(unsigned long long)plan.write_bytes,layout.bytes,
        (unsigned long long)plan.native_quotes,(unsigned long long)plan.product_quotes,rebatched,double(tick()-start)/1e6);
    }
    puts("No operands allocated, no arithmetic executed, no Store IO performed");
}
