#include "algorithms/inverse_rung.hpp"
#include "algorithms/refinement.hpp"
#include "product/cyclic_windows.hpp"
namespace sbn::v3 {
void inverse_rung(const InverseRung &p,const uint64_t *D,const uint64_t *U,uint64_t *V,
                  sbn3_product_metrics *residual_counts,sbn3_product_metrics *correction_counts) noexcept {
    const size_t k=p.ring,n=p.n;
    require(n<=(size_t(1)<<31)&&(k>=n||p.repair_bytes)&&k>=p.m+1&&k<=(size_t(1)<<31),SBN3_FATAL_ARGUMENT,"inverse rung shape");
    const size_t input=p.correction_input_words?p.correction_input_words:std::min(n,k);
    product::CyclicWindowGroup group{};
    group.slots[0]={p.product,0,std::min(n,k),nullptr,true,true,residual_counts};
    group.slots[1]={p.correction_input_words?p.correction_product:p.product,
                    p.correction_input_words?1u:0u,input,p.correction,true,false,correction_counts};
    group.cache=p.u;group.stage=p.stage;group.value=p.residual;group.period=k;
    group.capacity=k<n+3?std::max(k,n)+3:k;group.execute_capacity=k;group.team=p.team;
    group.consume_value=p.consume_residual;
    group.repair_bytes=p.repair_bytes;
    // The old approximation occupies only [0,m+1). Its unused output tail
    // can retain the low witness while the product consumes rho in place.
    if(p.repair_bytes){group.witness=V+p.m+1;group.witness_words=n-p.m;}
    product::CyclicWindowProducts products(group);
    bounded_refinement<RefinementKind::Inverse>({p.m,n,nullptr,p.correction,p.team},products,nullptr,D,U,V);
}
} // namespace sbn::v3
