#pragma once
#include "product/window.hpp"
#include "product/difference.hpp"
#include "backend/pq16/kernels.hpp"
#include <bit>
namespace sbn::v3::product {
struct PeeledWindowPlan {
    pq16::Shape fft{};size_t ring=0,capacity=0,bytes=0,residual_words=0;
    unsigned shared_mask=0,count=0;
};
class PeeledWindows {
    PeeledWindowPlan p_;Frame &space_;uint64_t *value_,*rho_;const pq16::Tables *tables_;double *cache_;bool ready_=false;
    void calculate(unsigned slot,Span a,Span b){
        const size_t ac=std::min(a.words,p_.ring/2),bc=std::min(b.words,p_.ring);
        const bool shared=bool(p_.shared_mask>>slot&1);
        if(shared&&!ready_){pq16::forward_spectrum(cache_,a.data,ac,*tables_,nullptr);ready_=true;}
        pq16::cyclic_multiply(value_,a.data,ac,b.data,bc,false,shared?cache_:nullptr,*tables_,space_,nullptr);
        for(size_t j=ac;j<a.words;++j)add_shifted_word_product(value_,p_.capacity,p_.ring,b,j%p_.ring,a.data[j]);
        for(size_t j=bc;j<b.words;++j)add_shifted_word_product(value_,p_.capacity,p_.ring,{a.data,ac},j%p_.ring,b.data[j]);
        memset(value_+p_.ring,0,(p_.capacity-p_.ring)*8);
    }
    static void low_product(uint64_t *low,unsigned g,Span a,Span b,ShiftedSpan sub){
        memset(low,0,g*8);
        for(unsigned j=0;j<g;++j){uint64_t carry=0;for(unsigned i=0;i+j<g;++i){
            const unsigned __int128 v=(unsigned __int128)(i<a.words?a.data[i]:0)*(j<b.words?b.data[j]:0)+low[i+j]+carry;
            low[i+j]=uint64_t(v);carry=uint64_t(v>>64);}}
        uint64_t borrow=0;
        for(size_t j=sub.shift;j<g;++j){const size_t at=j-sub.shift;const unsigned __int128 x=(unsigned __int128)(at<sub.value.words?sub.value.data[at]:0)+borrow;
            const uint64_t old=low[j];low[j]=old-uint64_t(x);borrow=(unsigned __int128)old<x;}
    }
    template<size_t G>bool lift(const uint64_t *low){uint64_t x[G];memcpy(x,low,G*8);return reconstruct_cyclic(value_,p_.ring,x);}
    bool lift(unsigned g,const uint64_t *low){switch(g){
#define LIFT(G) case G:return lift<G>(low);
        LIFT(1) LIFT(2) LIFT(3) LIFT(4) LIFT(5) LIFT(6) LIFT(7) LIFT(8)
#undef LIFT
        }fatal(SBN3_FATAL_MATH,"peel lift width");}
    WindowResult finish(Window w,MagnitudeBound bound,bool negative){
        if(bound.verify){
            require(bound.words<p_.capacity&&value_[bound.words]<bound.high_exclusive&&
                    limbs::zero(value_+bound.words+1,p_.capacity-bound.words-1),SBN3_FATAL_MATH,"peeled product window bound");
        }
        return {value_+w.origin,w.words,0,negative};
    }
  public:
    PeeledWindows(PeeledWindowPlan p,Frame &space):p_(p),space_(space),value_(space.alloc<uint64_t>(p.capacity)),rho_(p.count==2?space.alloc<uint64_t>(p.residual_words):nullptr),tables_(pq16::prepare(space,p.fft)),cache_(static_cast<double*>(space.allocate(16*size_t(p.fft.nfull)+128,128))){}
    WindowResult multiply(unsigned slot,Span a,Span b,Window w,MagnitudeBound bound){
        const unsigned g=unsigned(bound.words>=p_.ring?bound.words-p_.ring+1:1);uint64_t low[8];low_product(low,g,a,b,{});
        calculate(slot,a,b);require(!lift(g,low),SBN3_FATAL_MATH,"positive peeled product");return finish(w,bound,false);
    }
    WindowResult cancel(unsigned slot,Span a,Span b,ShiftedSpan sub,Window w,MagnitudeBound bound){
        const unsigned g=unsigned(bound.words>=p_.ring?bound.words-p_.ring+1:1);uint64_t low[8];low_product(low,g,a,b,sub);calculate(slot,a,b);
        if(sub.value.words==1&&sub.value.data[0]==1)limbs::cyclic_sub_power(value_,p_.ring,sub.shift%p_.ring);
        else limbs::cyclic_sub_shifted(value_,p_.ring,sub.value.data,sub.value.words,sub.shift);
        return finish(w,bound,lift(g,low));
    }
    WindowResult retain(WindowResult x,uint64_t *destination){if(p_.count==2)destination=rho_;memcpy(destination,x.data,x.words*8);x.data=destination;return x;}
};
} // namespace sbn::v3::product
