#pragma once
#include "sbn3/newton.h"
#include "runtime/arena.hpp"
#include "radix/programs.hpp"
#include "radix/rail_product.hpp"
#include "common/identity.hpp"
namespace sbn::v3::radix {
// Integer formatting and fractional parsing both multiply fresh values by a
// fixed reciprocal. Reuse the tree's cached linear-FFT kernel in its serial
// band; other sizes keep the ordinary product program. The cache belongs to
// this binding, is included in its storage, and is completed only after the
// reciprocal has been computed.
struct ScaleProductPlan : ProductShape {
    RailProductPlan cached{};
    size_t episode_bytes() const noexcept {
        return cached.enabled?cached.episode_bytes():ProductShape::episode_bytes();
    }
};
inline sbn3_query_result scale_product_plan(size_t a,size_t b,unsigned workers,bool repeated,
                                           ScaleProductPlan &p,PlanTranscript *transcript) noexcept {
    p={};
    if(repeated&&workers==1&&rail_linear_plan(a,b,p.cached)){
        p.an=a;p.bn=b;p.output_limbs=a+b;p.work_bytes=p.cached.scratch_bytes;p.work_alignment=64;
        const auto &s=p.cached.shape;
        uint64_t h=identity::fnv_seed;
        const uint64_t fields[]={uint64_t(a),uint64_t(b),uint64_t(s.nfull),uint64_t(s.branch),uint64_t(s.radix),
                                 uint64_t(s.bits),uint64_t(s.recipe),uint64_t(s.balanced),uint64_t(s.centered)};
        for(uint64_t v:fields)h=identity::word(h,v);
        p.arithmetic_id=h;
        return SBN3_SUPPORTED;
    }
    return product_shape(a,b,workers,p,nullptr,transcript,repeated?0:1);
}
struct ScaleProduct {ProductProgram plain{};RailProduct cached{};};
inline void scale_product_prepare(ScaleProduct &p,const ScaleProductPlan &q,ProgramSetBuilder &builder,
                                   Frame &spectra) noexcept {
    p={};
    if(q.cached.enabled)rail_product_reserve(p.cached,q.cached,spectra);
    else p.plain=builder.prepare(q);
}
inline void scale_product_complete(ScaleProduct &p,const uint64_t *reciprocal) noexcept {
    if(p.cached.plan.enabled)rail_product_forward(p.cached,reciprocal);
}
inline uint64_t *scale_product_execute(const ScaleProduct &p,const ScaleProductPlan &q,Frame &scratch,
                                       sbn3_team *team,const uint64_t *fresh,const uint64_t *reciprocal) noexcept {
    auto *z=scratch.alloc<uint64_t>(q.output_limbs);
    auto work=scratch.subframe(q.work_bytes,q.work_alignment);
    if(q.cached.enabled)rail_product_execute(p.cached,work,nullptr,fresh,z);
    else{TeamProduct job{&p.plain,&work,{fresh,q.an},{reciprocal,q.bn},{z,q.output_limbs}};run_product(team,job);}
    return z;
}
// The scaling episode runs before/after the tree with no live work Frames.
// Values and retained tree tables live outside this overlapping arena range.
inline void divide_scale(const sbn3_newton_plan &plan,sbn3_arena &arena,size_t offset,sbn3_team *team,
                          sbn3_lease &work,uint64_t *numerator,const uint64_t *denominator,size_t n) noexcept {
    const size_t bytes=work.bytes;
    arena.release(work);
    sbn3_newton_binding *binding=nullptr;
    sbn3_newton_bind(&plan,&arena,offset,team,&binding);
    const sbn3_newton_inputs in{{numerator,n+1},{denominator,n},0};
    sbn3_newton_execute(binding,&in,{numerator,n+1});
    sbn3_newton_unbind(binding);
    work=arena.acquire(offset,bytes);
}
}
