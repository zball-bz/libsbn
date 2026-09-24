#pragma once
#include "sbn3/mul.h"
#include "product/local_windows.hpp"
#include "product/difference.hpp"
#include "product/cyclic_reconstruct.hpp"
#include "backend/pq16/numeric_domain.hpp"
#include "product/u52_execution_cost.hpp"
#include "product/fixed_geometry.hpp"
#include "product/cost_model.hpp"
#include "product/root_prepare_cost.hpp"
// A smaller cyclic transform plus exact low products. This provider changes
// product geometry only; the arithmetic refinement equations stay unchanged.
// Its signed lift requires |E| < B^(ring+g)/2. The extra word in each bound
// supplies that margin and uniquely identifies the multiple of B^ring-1.
namespace sbn::v3::product::compact_window_detail {
struct Plan {pq16::Shape fft{};size_t ring=0,capacity=0,low_words=0,bytes=0;unsigned count=0,shared=0;bool keep_cache=false;};
inline Plan query(const WindowGroupShape &s,bool keep_cache=false) noexcept {
    if((s.count!=2&&s.count!=3)||s.products[s.count-2].bound.words<=native_policy::window_inline_words)return {};
    size_t extent=0;for(unsigned j=0;j<s.count;++j)extent=std::max(extent,s.products[j].bound.words+1);
    if(extent>pq16::cyclic_max_words+pq16::cyclic_max_words/4)return {};
    const auto upper=cyclic_window_fft(s);
    auto plan=[&](pq16::Shape fft){const size_t ring=pq16::cyclic_period(fft);
        const size_t low=extent>ring?extent-ring:1;
        return Plan{fft,ring,std::max(extent,ring+low),low,0,s.count,window_shared_mask(s),keep_cache};};
    auto cost=[&](const Plan &p){double v=shared_fft_work(s,p.fft);
        if(s.count==3&&!keep_cache)v+=pq16::native_cost(p.fft,s.products[0].a.words,s.products[0].b.words)/3;
        for(unsigned j=0;j<s.count;++j){const auto&r=s.products[j];const size_t g=r.bound.words>=p.ring?r.bound.words-p.ring+1:1;
            if(g<=8)continue;
            const auto low=local_window_detail::linear_shape(g,g);
            v+=low.nfull?pq16::native_cost(low,g,g):u52_execution_cost(g,g);}
        return v;};
    Plan p{};double best=INFINITY;
    if(upper.nfull){p=plan(upper);best=cost(p);}
    // Consider just the immediately smaller legal 16-bit ring, regardless
    // of its radix. An unsupported enclosing order must not hide that ring.
    // The quarter-period witness limit covers the largest radix gap.
    size_t previous=0;
    for(unsigned radix:{1u,3u,5u,7u}){size_t branch=128;
        while(size_t(radix)*branch<extent)branch*=2;
        const size_t ring=size_t(radix)*branch/2;
        if(ring<=pq16::cyclic_max_words)previous=std::max(previous,ring);}
    if(previous<extent&&extent-previous<=previous/4){
        auto fft=pq16::cyclic_shape(previous,16,1,true);
        if(pq16::cyclic_period(fft)==previous){
            auto supported=[&](pq16::Shape f){
                for(unsigned j=0;j<s.count;++j){const auto&r=s.products[j];
                    if(r.a.words>=previous||r.b.words>2*previous||r.bound.high_exclusive>=(uint64_t(1)<<63)||
                       !pq16::cyclic_supported(f,r.a.words,std::min(r.b.words,previous)))return false;
                    if(r.cancellation&&r.subtract.words>1&&(r.subtract.shift>=previous||r.subtract.words>2*previous-r.subtract.shift))return false;}
                return pq16::compact_table_setup(f);};
            if(!supported(fft)){fft.recipe=pq16::Recipe::CooleyTukeyPQ;fft.balanced=true;}
            if(supported(fft)){auto lower=plan(fft);const double work=cost(lower);if(work<best){p=lower;best=work;}}
        }
    }
    if(!p.ring)return {};
    if(!upper.nfull){
        // Price against the existing NTT recipe only when the enclosing FFT
        // is unavailable. One legal smaller ring is considered, no grid.
        const auto&r=s.products[s.count-2];FixedNttGeometry g;
        if(!fixed_ntt_geometry(r.a.words,r.b.words,r.bound.words,1,0,g,
                              native_policy::window_parallel_flat_words,native_policy::window_serial_flat_trunks))return {};
        sbn3_product_request request{};request.kind=SBN3_PRODUCT_MUL;
        request.a_limbs=r.a.words;request.b_limbs=r.b.words;request.cyclic_limbs=g.ring;
        sbn3_mul_info info{};const auto *backend=backend_lookup(g.np);
        if(!backend||backend->geometry_query(request,fixed_ntt_options(g),info)!=SBN3_SUPPORTED)return {};
        const double ordinary=cost_model::cyclic_product(info,false).nanoseconds;
        const double baseline=ordinary*(s.count==2?5.:8.)/3.+root_prepare_cost(info);
        if(best>=baseline)return {};
    }
    size_t low_work=0;
    for(unsigned j=0;j<s.count;++j){const size_t words=s.products[j].bound.words,g=words>=p.ring?words-p.ring+1:1;
        if(g>8)low_work=std::max(low_work,local_window_detail::full_bytes(g,g));}
    const size_t transient=std::max((keep_cache&&s.count==3?48:32)*size_t(p.fft.nfull)+2048,16*size_t(p.fft.nfull)+low_work+512);
    p.bytes=8*(p.capacity+2*p.low_words)+pq16::table_bytes(p.fft)+transient+2048;
    return p;
}
class Products {
    Plan p_;Frame &space_;uint64_t *value_,*low_;const pq16::Tables *tables_;double *cache_=nullptr;Frame::Mark cache_mark_{};
    size_t low_product(Span a,Span b,ShiftedSpan sub,MagnitudeBound bound){
        const size_t g=bound.words>=p_.ring?bound.words-p_.ring+1:1;
        require(g<=p_.low_words,SBN3_FATAL_WORKSPACE,"repair low capacity");
        memset(low_,0,16*g);FrameMark temporary(space_);
        if(g<=8){
            for(size_t j=0;j<std::min(b.words,g);++j){uint64_t carry=0;
                for(size_t i=0;i+j<g&&i<a.words;++i){const auto x=(unsigned __int128)a.data[i]*b.data[j]+low_[i+j]+carry;low_[i+j]=uint64_t(x);carry=uint64_t(x>>64);}}
        }else if(a.words&&b.words)local_window_detail::full(low_,{a.data,std::min(a.words,g)},{b.data,std::min(b.words,g)},space_);
        if(sub.shift<g&&sub.value.words)limbs::sub_from(low_+sub.shift,g-sub.shift,sub.value.data,std::min(sub.value.words,g-sub.shift));
        return g;
    }
    void calculate(unsigned slot,Span a,Span b){
        const size_t r=p_.ring;require(a.words<r&&b.words<=2*r,SBN3_FATAL_ARGUMENT,"repair support");
        require(!overlaps(a.data,8*a.words,value_,8*p_.capacity),SBN3_FATAL_ARGUMENT,"repair A lifetime");
        const bool shared=p_.shared>>slot&1;
        if(!shared&&cache_&&!p_.keep_cache){space_.rewind(cache_mark_);cache_=nullptr;}
        if(shared&&!cache_){cache_mark_=space_.mark();cache_=static_cast<double*>(space_.allocate(16*size_t(p_.fft.nfull)+128,128));
            pq16::forward_spectrum(cache_,a.data,a.words,*tables_,nullptr);}
        if(b.words>r){
            require(!overlaps(b.data,8*b.words,value_,8*p_.capacity),SBN3_FATAL_ARGUMENT,"repair folded input lifetime");
            memcpy(value_,b.data,8*r);
            const auto carry=limbs::add_to(value_,r,b.data+r,b.words-r);add_cyclic_word(value_,r,carry);
            b={value_,r};
        }
        pq16::cyclic_multiply(value_,a.data,a.words,b.data,b.words,false,shared?cache_:nullptr,*tables_,space_,nullptr);
        memset(value_+r,0,8*(p_.capacity-r));
    }
    WindowResult finish(Window w,MagnitudeBound bound,size_t g){
        const bool negative=reconstruct_cyclic(value_,p_.ring,low_,g);
        memset(value_+p_.ring+g,0,8*(p_.capacity-p_.ring-g));
        if(bound.verify)require(bound.words<p_.capacity&&value_[bound.words]<bound.high_exclusive&&
            limbs::zero(value_+bound.words+1,p_.capacity-bound.words-1),SBN3_FATAL_MATH,"repair magnitude");
        return {value_+w.origin,w.words,0,negative};
    }
  public:
    Products(Plan p,Frame &space):p_(p),space_(space),value_(space.alloc<uint64_t>(p.capacity)),low_(space.alloc<uint64_t>(2*p.low_words)),
        tables_(pq16::prepare(space,p.fft)){}
    WindowResult multiply(unsigned slot,Span a,Span b,Window w,MagnitudeBound bound){
        const auto g=low_product(a,b,{},bound);calculate(slot,a,b);auto result=finish(w,bound,g);
        require(!result.negative,SBN3_FATAL_MATH,"positive repaired product");return result;
    }
    WindowResult cancel(unsigned slot,Span a,Span b,ShiftedSpan sub,Window w,MagnitudeBound bound){
        const auto g=low_product(a,b,sub,bound);calculate(slot,a,b);
        if(sub.value.words==1&&sub.value.data[0]==1)limbs::cyclic_sub_power(value_,p_.ring,sub.shift%p_.ring);
        else limbs::cyclic_sub_shifted(value_,p_.ring,sub.value.data,sub.value.words,sub.shift);
        return finish(w,bound,g);
    }
    WindowResult retain(WindowResult result,uint64_t *destination){
        if(p_.count==3){memcpy(destination,result.data,8*result.words);result.data=destination;}
        return result;
    }
};
struct Factory {
    LocalWindowFactory base;
    bool keep_cache=false;
    size_t bytes(const WindowGroupShape &s)const{const auto p=query(s,keep_cache);return p.ring?p.bytes:base.bytes(s);}
    size_t retained_words(const WindowGroupShape&s)const{return base.retained_words(s);}
    template<class F>void with_group(const WindowGroupShape&s,Frame&space,uint64_t*out,F&&f)const{
        const auto p=query(s,keep_cache);if(!p.ring){base.with_group(s,space,out,f);return;}
        FrameMark mark(space);Products products(p,space);f(products);
    }
};
}

namespace sbn::v3::product {
using CompactWindowFactory=compact_window_detail::Factory;
inline bool compact_window_supported(const WindowGroupShape &s) noexcept {return bool(compact_window_detail::query(s).ring);}
}
