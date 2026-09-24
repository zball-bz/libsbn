#pragma once
#include "product/window.hpp"
#include "backend/pq16/kernels.hpp"
#include "backend/pq16/cost_model.hpp"

namespace sbn::v3::product {
inline unsigned window_shared_mask(const WindowGroupShape &s) noexcept {
    unsigned mask=0;
    for(unsigned j=0;j<s.count;++j)if(s.products[j].a.value_id==s.products[0].a.value_id)mask|=1u<<j;
    return mask;
}
inline double shared_fft_work(const WindowGroupShape &s,pq16::Shape shape) noexcept {
    const unsigned mask=window_shared_mask(s);
    // F+F+I is the backend's ordinary-product unit. Reused inputs remove a
    // forward, while the shared forward is paid once for the whole group.
    double work=pq16::native_cost(shape,s.products[0].a.words,s.products[0].b.words)/3;
    for(unsigned j=0;j<s.count;++j){const auto &r=s.products[j];
        work+=pq16::native_cost(shape,r.a.words,r.b.words)*((mask>>j&1)?2./3.:1.);}
    return work;
}
inline bool shared_fft_supported(const WindowGroupShape &s,pq16::Shape shape) noexcept {
    if(!shape.nfull)return false;
    for(unsigned j=0;j<s.count;++j){const auto &r=s.products[j];
        if(r.cancellation){if(!pq16::cyclic_supported(shape,r.a.words,r.b.words))return false;}
        else if(shape.bits>16||shape.balanced){
            if(r.a.words+r.b.words>=pq16::cyclic_period(shape)||
               !pq16::cyclic_supported(shape,r.a.words,r.b.words))return false;
        }else if(!pq16::supported(shape,r.a.words,r.b.words,1))return false;
    }
    return true;
}
// At most one capacity-derived shape per codec. This constructs no public
// plans and runs no T/radix/layout/worker Cartesian search. Numeric domains
// belong to the FFT backend; the arithmetic consumer supplies only windows.
inline pq16::Shape wide_window_fft(const WindowGroupShape &s,pq16::Shape current,double current_work) noexcept {
    size_t minimum=0;
    for(unsigned j=0;j<s.count;++j){const auto &r=s.products[j];
        minimum=std::max(minimum,r.bound.words+2);
        if(r.cancellation)minimum=std::max({minimum,std::max(r.a.words,r.b.words)+1,2*std::min(r.a.words,r.b.words)+1});
        else minimum=std::max(minimum,r.a.words+r.b.words+2);
    }
    for(unsigned bits=17;bits<=20;++bits){
        const auto c=pq16::cyclic_shape(minimum,bits);
        // These groups are fresh: a smaller transform is not a win when
        // it introduces an operation-sized root builder. Numerical support
        // remains available to explicit callers outside this default policy.
        if(!shared_fft_supported(s,c)||!pq16::tables_published(c))continue;
        const double cost=shared_fft_work(s,c);
        if(cost<current_work){current=c;current_work=cost;}
    }
    return current;
}
// A precompiled cyclic provider can recover the three arithmetic guard
// words independently. Unlike the local linear-prefix emitter, it need not
// enlarge the transform for those guards. Cancellation still fits the ring.
inline pq16::Shape cyclic_window_fft(const WindowGroupShape &s) noexcept {
    size_t minimum=0;
    for(unsigned j=0;j<s.count;++j){const auto&r=s.products[j];
        minimum=std::max({minimum,r.a.words,r.b.words,r.cancellation?r.bound.words:r.bound.words-std::min<size_t>(2,r.bound.words)});}
    pq16::Shape best{};double cost=INFINITY;
    for(unsigned bits=16;bits<=20;++bits){auto c=pq16::cyclic_shape(minimum,bits,1,true);
        if(!c.nfull)continue;
        auto supports=[&](pq16::Shape candidate){const size_t ring=pq16::cyclic_period(candidate);
            if(!pq16::compact_table_setup(candidate))return false;
            for(unsigned j=0;j<s.count;++j){const auto&r=s.products[j];
                if(!pq16::cyclic_supported(candidate,r.a.words,r.b.words)||r.bound.high_exclusive>=(uint64_t(1)<<63)||
                   r.bound.words>ring+(r.cancellation?0:2)||r.window.origin+r.window.words>ring+3)return false;}
            return true;};
        // Same period, wider signed-input support. A capacity-only query
        // prefers unsigned at equal point counts; this group knows its inputs.
        if(!supports(c)&&bits==16&&!c.balanced){c.recipe=pq16::Recipe::CooleyTukeyPQ;c.balanced=true;}
        if(supports(c)){const double candidate=shared_fft_work(s,c);if(candidate<cost){cost=candidate;best=c;}}
    }
    return best;
}
// Substitute the plus representation only at an already-selected unsigned
// 16-bit odd-radix geometry. This adds no codec/order candidate search.
inline pq16::Shape plus_window_fft(const WindowGroupShape&s,pq16::Shape current) noexcept {
    if(!current.nfull||current.bits!=16||current.balanced||current.centered||current.radix==1)return current;
    auto candidate=current;candidate.recipe=pq16::Recipe::RightAngle;
    if(!pq16::tables_published(candidate))return current;
    const size_t ring=pq16::cyclic_period(candidate);
    for(unsigned j=0;j<s.count;++j){const auto&r=s.products[j];
        if(!pq16::bounded_plus_supported(candidate,r.a.words,r.b.words)||r.bound.words>=ring||
           r.bound.high_exclusive>=(uint64_t(1)<<63))return current;
        if(r.cancellation&&r.subtract.words>1&&(r.subtract.shift>=ring||r.subtract.words>2*ring-r.subtract.shift))return current;
    }
    return shared_fft_work(s,candidate)<shared_fft_work(s,current)?candidate:current;
}

} // namespace sbn::v3::product
