#pragma once
#include "product/window.hpp"
#include "backend/u52/kernels.hpp"
#include "backend/pq16/kernels.hpp"
#include "backend/pq16/cost_model.hpp"
namespace sbn::v3::product {
struct LocalProductPlan {
    size_t an=0,bn=0,ring=0,persistent_bytes=0,work_bytes=0;
    pq16::Shape shape{};
    bool cached=false;
};
struct PreparedLocalProduct {
    const uint64_t *original=nullptr;
    const pq16::Tables *tables=nullptr;
    double *cache=nullptr;
};
// Predecessor in the {1,3,5,7}*2^k transform grid. Numerical admission below
// rejects branches smaller than the kernel's minimum; this is geometry
// arithmetic, not construction of competing product plans.
inline pq16::Shape previous_local_fft(pq16::Shape s) noexcept {
    unsigned m=0,n=0;
    switch(s.radix){case 1:m=7;n=s.branch/8;break;case 3:m=5;n=s.branch/2;break;
        case 5:m=1;n=s.branch*4;break;case 7:m=3;n=s.branch*2;break;default:return {};}
    return {m*n,n,m,false,pq16::Recipe::CooleyTukeyPQ,16,false};
}
inline LocalProductPlan local_product_query(size_t an,size_t bn,bool residual,bool keep) noexcept {
    LocalProductPlan p;p.an=an;p.bn=bn;
    if(std::min(an,bn)>=128){
        // The engine publishes the small right-angle twist tables, so the
        // ordinary W1 execution recipe needs no extra per-call root build.
        p.shape=pq16::execution_shape(pq16::query(an,bn),1);
        if(!pq16::supported(p.shape,an,bn,1))p.shape={};
        if(residual&&p.shape.nfull){
            const auto cycle=pq16::cyclic_shape(an,16);const size_t period=pq16::cyclic_period(cycle);
            if(cycle.nfull<p.shape.nfull&&period<an+bn+1&&pq16::cyclic_supported(cycle,an,bn)){p.shape=cycle;p.ring=period;}
        }
        if(p.shape.nfull){
            // Visit adjacent smaller geometries. The required width follows
            // from capacity; its unsigned/balanced admission is the engine's
            // contract. Stop when even the widest codec cannot fit.
            auto footprint=[&](pq16::Shape s){return pq16::table_bytes(s)+
                (keep?16*size_t(s.nfull)+pq16::cached_scratch_bytes(s,an,bn):pq16::scratch_bytes(s,an,bn));};
            size_t budget=0;const size_t minimum=residual?pq16::wide_cyclic_minimum_words(an,bn):an+bn;
            double cost=pq16::native_cost(p.shape,an,bn);
            auto geometry=p.shape;
            for(;;){
                geometry=previous_local_fft(geometry);if(!geometry.nfull)break;
                const size_t bits=std::max<size_t>(17,(32*minimum+geometry.nfull-1)/geometry.nfull);
                if(bits>20)break;
                auto candidate=geometry;candidate.bits=unsigned(bits);
                const size_t ring=pq16::cyclic_period(candidate);
                if(residual){
                    if(ring>=an+bn+1)continue;
                    if(!pq16::cyclic_supported(candidate,an,bn))candidate.balanced=true;
                    if(!pq16::cyclic_supported(candidate,an,bn))continue;
                }else{
                    if(!pq16::supported(candidate,an,bn,1))candidate.balanced=true;
                    if(!pq16::supported(candidate,an,bn,1))continue;
                }
                if(!pq16::tables_published(candidate))continue;
                const double price=pq16::native_cost(candidate,an,bn);
                if(price>=cost)continue;
                if(!budget)budget=footprint(p.shape);
                if(footprint(candidate)>budget)continue;
                cost=price;p.shape=candidate;p.ring=residual?ring:0;
            }
        }
    }
    p.cached=p.shape.nfull&&keep;
    p.persistent_bytes=p.shape.nfull?pq16::table_bytes(p.shape)+(p.cached?16*size_t(p.shape.nfull):0)+512:0;
    p.work_bytes=p.shape.nfull?(p.cached?pq16::cached_scratch_bytes(p.shape,an,bn):pq16::scratch_bytes(p.shape,an,bn))+128:u52::scratch_bytes(an,bn);
    if(p.shape.nfull&&!residual){
        // The short-tail high-prefix entry below may execute inside this
        // reservation. Quote it explicitly across FFT codec changes.
        const size_t tail=std::min({size_t(192),bn,(an+4)/2});
        if(tail)p.work_bytes=std::max(p.work_bytes,u52::high_prefix_scratch_bytes(tail)+128);
    }
    return p;
}
inline PreparedLocalProduct local_product_prepare(const LocalProductPlan&p,const uint64_t*a,Frame&space,
                                             const pq16::Tables*shared=nullptr) noexcept {
    PreparedLocalProduct v;v.original=a;
    if(p.shape.nfull){
        v.tables=shared?shared:pq16::prepare(space,p.shape);
        if(p.cached){
            v.cache=static_cast<double*>(space.allocate(16*size_t(p.shape.nfull)+128,128));
            pq16::forward_spectrum(v.cache,a,p.an,*v.tables,nullptr);
        }
    }
    return v;
}
inline void local_product_apply(const LocalProductPlan &plan,const PreparedLocalProduct &p,
                   uint64_t*out,const uint64_t*b,size_t count,Frame&space,Window window={}) noexcept {
        // A short tail does not need the full cached transform. Its guarded
        // high prefix can underestimate by one; Barrett's correction budget
        // covers that extra unit. This fits inside the full FFT workspace.
        if(window.error_bits&&window.origin==plan.an&&window.words==count&&plan.shape.nfull&&count<=192&&2*count<=plan.an+4){
            u52::high_prefix(out+plan.an,p.original,plan.an,b,count,space);return;
        }
        if(plan.ring)pq16::cyclic_multiply(out,p.original,plan.an,b,count,false,p.cache,*p.tables,space,nullptr);
        else if(plan.shape.nfull){
            if(p.cache)pq16::apply_spectrum(out,p.cache,p.original,plan.an,b,count,false,*p.tables,space,nullptr);
            else pq16::multiply(out,p.original,plan.an,b,count,*p.tables,space,nullptr);
        }
        else u52::multiply(out,p.original,plan.an,b,count,space);
}

inline bool local_product_same_tables(const LocalProductPlan &a,const LocalProductPlan &b) noexcept {
    return a.shape.nfull&&a.shape.nfull==b.shape.nfull&&a.shape.branch==b.shape.branch&&a.shape.radix==b.shape.radix&&
           a.shape.recipe==b.shape.recipe&&a.shape.bits==b.shape.bits&&a.shape.centered==b.shape.centered&&a.shape.balanced==b.shape.balanced;
}
inline size_t local_product_table_bytes(const LocalProductPlan &p) noexcept {return p.shape.nfull?pq16::table_bytes(p.shape):0;}
inline size_t local_cyclic_operand_capacity(size_t minimum) noexcept {
    const auto ring=pq16::cyclic_shape(minimum,16);return ring.nfull?pq16::cyclic_period(ring)/2:0;
}
} // namespace sbn::v3::product
