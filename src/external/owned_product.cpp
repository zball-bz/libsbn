// SPDX-License-Identifier: LGPL-3.0-or-later
#include "external/owned_product.hpp"
#include "external/trace.hpp"
namespace sbn::v3::external {
static IoError run_owned(const SpilledProductPlan&p,Transport&t,ValueStore&store,
                         const StoredValue*a,const StoredValue&b,StoredValue&out,RowCacheBinding*cache,
                         std::span<uint8_t>workspace,sbn3_team&team,WordExtent *extent,CyclicResidualResult*cyclic,const StoredValue*subtract,MiddleWindowResult*middle) noexcept {
    OperationSpan trace(t,p.spec.middle?"middle_product":p.spec.cyclic?"cyclic_residual":p.cached_a?"cached_product":"stored_product",
        p.spec.an,p.spec.bn,p.spec.primes,p.spec.trunk_bits,p.spec.outer_cells,p.spec.inner_cells);
    if(!b||out||b.store()!=&store||p.plane_count>20||
       (p.cached_a?(!cache||a):(!a||!*a||cache||a->store()!=&store))||bool(p.spec.cyclic)!=(cyclic!=nullptr)||bool(p.spec.middle)!=(middle!=nullptr)||
       p.spec.cyclic.stored_subtract!=bool(subtract)||(subtract&&(!*subtract||subtract->store()!=&store)))
        return {IoCode::invalid,"owned product inputs/output"};
    if(t.counters.write_submitted>t.write_limit || p.normal_write_bytes>t.write_limit-t.counters.write_submitted)
        return {IoCode::capacity,"owned product writes",0,p.normal_write_bytes,
                t.counters.write_submitted<=t.write_limit?t.write_limit-t.counters.write_submitted:0};
    RegionLease result,planes[20];Region *pointers[20];
    if(auto e=store.reserve(p.stored_output_bytes,RegionKind::value,result))return e;
    for(unsigned q=0;q<p.plane_count;++q){
        if(auto e=store.reserve(p.plane_bytes,RegionKind::scratch,planes[q]))return e;
        pointers[q]=&planes[q].region();
    }
    const auto*sub=subtract?&subtract->region():nullptr;
    const auto error=cache?run_spilled_cached(p,t,*cache,b.region(),result.region(),{pointers,p.plane_count},workspace,team,extent,cyclic,sub,middle):
        run_spilled_product_indirect(p,t,a->region(),b.region(),result.region(),{pointers,p.plane_count},workspace,team,extent,cyclic,sub,middle);
    if(error)return error;
    return result.publish(out);
}
IoError run_owned_product(const SpilledProductPlan&p,Transport&t,ValueStore&s,const StoredValue&a,const StoredValue&b,
                         StoredValue&r,std::span<uint8_t>w,sbn3_team&team,WordExtent*extent,CyclicResidualResult*cyclic,const StoredValue*subtract,MiddleWindowResult*middle) noexcept {
    return run_owned(p,t,s,&a,b,r,nullptr,w,team,extent,cyclic,subtract,middle);
}
IoError run_owned_cached_product(const SpilledProductPlan&p,Transport&t,ValueStore&s,RowCacheBinding&cache,const StoredValue&b,
                         StoredValue&r,std::span<uint8_t>w,sbn3_team&team,WordExtent*extent,CyclicResidualResult*cyclic,const StoredValue*subtract,MiddleWindowResult*middle) noexcept {
    return run_owned(p,t,s,nullptr,b,r,&cache,w,team,extent,cyclic,subtract,middle);
}
} // namespace sbn::v3::external
