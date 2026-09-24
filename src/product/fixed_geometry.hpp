#pragma once
#include "product/backend.hpp"
#include "product/native_capabilities.hpp"
#include "product/tuning_native.hpp"
#include <algorithm>

namespace sbn::v3 {
struct FixedNttGeometry {
    unsigned np=0,algorithm=0,workers=1;
    int T=0;
    size_t ring=0;
};
inline sbn3_mul_options fixed_ntt_options(const FixedNttGeometry &g) noexcept {
    sbn3_mul_options o{};o.prime_count=g.np;o.algorithm=g.algorithm;
    o.workers=g.workers;o.trunk_bits=g.T;o.borrow_output=1;return o;
}
// One certified geometry per arithmetic basis. The widest legal codec fixes
// its transform order; a cyclic period then tightens T arithmetically without
// changing that order. No T/layout/column grid and no bindable trial plans.
inline bool fixed_ntt_geometry(size_t a,size_t b,size_t minimum_ring,unsigned workers,
                               unsigned prime_count,FixedNttGeometry &out,size_t flat_span=size_t(1)<<19,
                               size_t flat_trunks=0,bool repeated=false) {
    if(!native_available())return false;
    const unsigned first=prime_count?prime_count:4,last=prime_count?prime_count:(workers==32||(workers>=16&&repeated)?10:8);
    bool found=false;double least=0;
    for(unsigned np=first;np<=last;++np){
        const auto *backend=backend_lookup(np);if(!backend||!backend->geometry_query)continue;
        FixedNttGeometry c{np,std::max(a,b)<=(size_t(1)<<19)?SBN3_MUL_FLAT:SBN3_MUL_BAILEY,workers,0,0};
        sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.a_limbs=a;r.b_limbs=b;
        sbn3_mul_info i{};
        if(backend->geometry_query(r,fixed_ntt_options(c),i)!=SBN3_SUPPORTED)continue;
        c.T=int(i.trunk_bits);
        if(minimum_ring){
            const size_t need=std::max({minimum_ring,a,b});
            size_t factor=2;while(factor*size_t(c.T)<need)factor*=2;
            const int step=np==4?4:8,minimum=np==4?80:24*int(np)-32;
            const int needed=int((need+factor-1)/factor);
            c.T=std::max(minimum,(needed+step-1)/step*step);
            c.ring=factor*size_t(c.T);r.cyclic_limbs=c.ring;
            // Integer reconstruction guards live outside the transform and
            // do not change its layout class. Serial window groups bound
            // actual coefficients per prime, independently of codec width.
            const bool use_flat=flat_trunks?factor<=flat_trunks/64:c.ring<=flat_span;
            c.algorithm=use_flat?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
            if(backend->geometry_query(r,fixed_ntt_options(c),i)!=SBN3_SUPPORTED)continue;
        }
        // Measured wider-base gains are limited to repeated Flat episodes
        // with all primes in one worker wave. Bailey is not calibrated;
        // retain the existing W32 policy and explicit-prime capabilities.
        if(!prime_count&&np>8&&workers!=32&&
           (i.algorithm!=SBN3_MUL_FLAT||np>workers))continue;
        // Flat also uses a TFT: charge live slots, not its padded root order.
        // Its M2 denotes N/8, not a Bailey column tower, so tower penalties
        // do not apply to that representation.
        const double slots=8.*double(i.C)*double(i.lbv);
        const double fill=minimum_ring?1.:double(i.nat+i.nyt-1)/double(i.transform_trunks);
        const double tower=i.algorithm==SBN3_MUL_FLAT?1.:mul_m2_cost(i.M2);
        const double volume=double(np)*slots*(np<=8?mul_np_cost[np]:1.)*tower*(1.+.12*(1.-fill));
        if(!found||volume<least){out=c;least=volume;found=true;}
    }
    return found;
}
}
