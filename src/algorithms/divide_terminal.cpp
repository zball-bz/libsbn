#include "algorithms/divide_terminal.hpp"
#include "algorithms/refinement.hpp"
#include "product/cyclic_windows.hpp"
namespace sbn::v3 {
void divide_terminal(const DivideTerminal &p,const uint64_t *A,const uint64_t *D,const uint64_t *U,uint64_t *Q) noexcept {
    const size_t n=p.n,k=p.ring,bn=p.input_words?p.input_words:n+1;
    const size_t output=p.output_capacity?p.output_capacity:k,capacity=k<n+3?std::max(k,n)+3:k;
    require(n<=(size_t(1)<<31)&&(k>=n||p.repair_bytes)&&k<=(size_t(1)<<31)&&output>=capacity&&
                bn>=p.m+1&&bn>=newton_contract::residual_words(p.m,n)&&bn<=n+1&&bn<=k,
            SBN3_FATAL_ARGUMENT,"division terminal shape");
    product::CyclicWindowGroup group{};
    group.slots[0]={p.cached_inverse,0,bn,p.work[0],true,true,nullptr};
    group.slots[1]={p.fresh_residual_in_cached?p.cached_inverse:p.residual,
                    p.fresh_residual_in_cached?0u:1u,std::min(n,k),nullptr,false,false,nullptr,p.fresh_residual_in_cached};
    group.slots[2]={p.cached_inverse,0,bn,Q,true,p.stage&&!p.retain_u_across_residual,nullptr};
    group.cache=p.u;group.stage=p.stage;group.value=p.work[1];group.period=k;
    group.capacity=capacity;group.execute_capacity=output;group.team=p.team;
    group.repair_bytes=p.repair_bytes;
    product::CyclicWindowProducts products(group);
    bounded_refinement<RefinementKind::Quotient>({p.m,n,p.work[0],Q,p.team},products,A,D,U,Q);
}
} // namespace sbn::v3
