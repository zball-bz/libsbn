// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "product/program.hpp"
#include "product/fixed_geometry.hpp"
namespace sbn::v3::product {
// Frame-owned exact minus-ring product. The period is internal and returned
// explicitly; public callers' ring semantics are never rounded here.
struct CyclicProgramPlan {ProductProgramPlan product{};size_t period=0;bool sum2=false;};
inline bool cyclic_program_query(size_t minimum,size_t short_words,unsigned workers,CyclicProgramPlan &out,bool sum2=false,bool legacy_capacity=false) {
    FixedNttGeometry g{};
    if(!fixed_ntt_geometry(minimum,short_words,minimum,workers,0,g,size_t(1)<<19,0,false,
                           sum2&&!legacy_capacity?FixedNttProduct::sum2:FixedNttProduct::multiply))return false;
    auto options=fixed_ntt_options(g);options.borrow_output=0;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.a_limbs=g.ring;r.b_limbs=short_words;r.cyclic_limbs=g.ring;
    if(sum2){r.kind=SBN3_PRODUCT_MAC2;r.a1_limbs=r.a_limbs;r.b1_limbs=r.b_limbs;}
    sbn3_product_info i{};out={};
    const auto *backend=backend_lookup(g.np);
    if(!backend||backend->product_query(r,options,out.product.plan,i)!=SBN3_SUPPORTED)return false;
    out.product.info=i.mul;out.product.a1=sum2?g.ring:0;out.product.b1=sum2?short_words:0;
    out.period=g.ring;out.sum2=sum2;
    return product_program_describe(g.ring,short_words,out.product)==SBN3_SUPPORTED;
}
}
