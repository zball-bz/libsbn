#pragma once
#include "product/window.hpp"
#include "product/window_fft.hpp"
#include "product/window_policy.hpp"
#include "product/peeled_windows.hpp"
#include "product/leaf_windows.hpp"
#include "backend/u52/kernels.hpp"
#include "backend/pq16/kernels.hpp"
#include "value/limbs.hpp"
#include "value/plus_ring.hpp"

namespace sbn::v3::product {
namespace local_window_detail {
inline constexpr size_t fft_min_words=native_policy::window_fft_min_words;
inline size_t bytes(size_t words) noexcept {return (8*words+63)&~size_t(63);}
inline pq16::Shape linear_shape(size_t a,size_t b) noexcept {
    if(std::min(a,b)<fft_min_words)return {};
    auto s=pq16::execution_shape(pq16::query(a,b),1);
    return s.nfull&&pq16::supported(s,a,b,1)?s:pq16::Shape{};
}
inline size_t full_bytes(size_t a,size_t b) noexcept {
    const auto s=linear_shape(a,b);
    return s.nfull?pq16::table_bytes(s)+pq16::scratch_bytes(s,a,b)+512:u52::scratch_bytes(a,b);
}
inline void full(uint64_t *out,Span a,Span b,Frame &space) noexcept {
    const auto s=linear_shape(a.words,b.words);
    if(!s.nfull){u52::multiply(out,a.data,a.words,b.data,b.words,space);return;}
    FrameMark mark(space);auto table=space.subframe(pq16::table_bytes(s)+128,128);
    auto *prepared=pq16::prepare(table,s);
    auto work=space.subframe(pq16::scratch_bytes(s,a.words,b.words)+128,128);
    pq16::multiply(out,a.data,a.words,b.data,b.words,*prepared,work,nullptr);
}
inline bool middle_supported(const WindowProductShape &r) noexcept {
    if(!r.cancellation||r.b.words<128||r.a.words<8||r.a.words>=fft_min_words||
       r.window.error_bits<64||r.window.origin+3!=r.a.words||
       r.window.origin+r.window.words!=r.b.words+1||r.bound.words!=r.b.words)return false;
    const size_t na=(r.a.words*64+51)/52,origin=52*(na-5);
    return r.subtract.shift*64>=origin;
}
inline size_t middle_bytes(const WindowProductShape &r) noexcept {
    const size_t na=(r.a.words*64+51)/52,nb=((r.b.words+2)*64+51)/52;
    const size_t count=(52*(nb-na+7)+63)/64;
    return bytes(count)+u52::middle_guard_scratch_bytes(r.a.words,r.b.words+2)+256;
}
// The four retained guard diagonals bound the discarded contribution. At
// window.origin=a.words-3, the resulting signed word-window error is <2^64.
// Cancellation's declared high bound makes the signed modular lift unique.
inline WindowResult middle(uint64_t *out,Span a,Span b,ShiftedSpan sub,Window w,Frame &space) noexcept {
    FrameMark mark(space);
    const size_t na=(a.words*64+51)/52,nb=((b.words+2)*64+51)/52;
    const size_t origin=52*(na-5),top=52*nb,span=top-origin,count=(52*(nb-na+7)+63)/64;
    auto *band=space.alloc<uint64_t>(count);
    u52::middle_guard_zero2(band,a.data,a.words,b.data,b.words,space);
    const size_t used=(span+63)/64;
    if(sub.shift*64<top){
        require(sub.shift*64>=origin,SBN3_FATAL_ARGUMENT,"middle addend origin");
        const size_t start=sub.shift*64-origin,word=start/64;const unsigned bits=unsigned(start%64);uint64_t borrow=0;
        for(size_t j=0;j<used;++j){
            uint64_t value=0;if(j>=word){const size_t at=j-word;value=at<sub.value.words?sub.value.data[at]<<bits:0;
                if(bits&&at&&at-1<sub.value.words)value|=sub.value.data[at-1]>>(64-bits);}
            const unsigned __int128 amount=(unsigned __int128)value+borrow;
            const uint64_t old=band[j];band[j]=old-uint64_t(amount);borrow=(unsigned __int128)old<amount;
        }
    }
    if(span%64)band[used-1]&=(uint64_t(1)<<(span%64))-1;
    const bool negative=bool(band[(span-1)/64]>>((span-1)%64)&1);
    if(negative){for(size_t j=0;j<used;++j)band[j]=~band[j];
        if(span%64)band[used-1]&=(uint64_t(1)<<(span%64))-1;
        const uint64_t one=1;limbs::add_to(band,used,&one,1);}
    const size_t shift=64*w.origin;require(shift>=origin,SBN3_FATAL_ARGUMENT,"middle window origin");
    const size_t word=(shift-origin)/64;const unsigned bits=unsigned((shift-origin)%64);
    for(size_t j=0;j<w.words;++j){const size_t at=word+j;uint64_t value=at<used?band[at]>>bits:0;
        if(bits&&at+1<used)value|=band[at+1]<<(64-bits);out[j]=value;}
    return {out,w.words,64,negative};
}
inline pq16::Shape cancellation_shape(const WindowProductShape &r) noexcept {
    const auto full=linear_shape(r.a.words,r.b.words);if(!full.nfull)return {};
    const auto shape=pq16::cyclic_shape(r.bound.words+2,16,1,false);const size_t ring=pq16::cyclic_period(shape);
    return shape.nfull&&ring<r.a.words+r.b.words-1&&shape.nfull<full.nfull&&
        pq16::cyclic_supported(shape,r.a.words,r.b.words)?shape:pq16::Shape{};
}
inline void check(const uint64_t *value,size_t length,MagnitudeBound bound) noexcept {
    if(!bound.verify)return;
    require(bound.words<length&&value[bound.words]<bound.high_exclusive&&
                limbs::zero(value+bound.words+1,length-bound.words-1),SBN3_FATAL_MATH,"local window bound");
}
} // namespace local_window_detail

// A selected physical group. Arithmetic clients only use bytes and the
// window interface; codec/radix decisions remain in this product header.
struct LocalWindowPlan {
    pq16::Shape shared{},cancel_shape{};
    size_t value_words=0,correction_words=0,work_bytes=0,storage_bytes=0;
    unsigned count=0,cancellation=0,shared_mask=0;
    bool middle=false;
};
[[gnu::always_inline]] inline LocalWindowPlan local_windows_query(const WindowGroupShape &shape) noexcept {
    using namespace local_window_detail;
    require(shape.count==2||shape.count==3,SBN3_FATAL_ARGUMENT,"local window group count");
    LocalWindowPlan p{};p.count=shape.count;p.cancellation=shape.count-2;
    const auto &c=shape.products[p.cancellation],&last=shape.products[shape.count-1],&first=shape.products[0];
    require(c.cancellation&&!last.cancellation&&first.a.value_id==last.a.value_id,
            SBN3_FATAL_ARGUMENT,"local window reuse group");
    p.middle=middle_supported(c);
    if(shape.count==2){
        p.cancel_shape=p.middle?pq16::Shape{}:cancellation_shape(c);
        const auto tail=linear_shape(last.a.words,last.b.words);
        if(p.cancel_shape.nfull&&p.cancel_shape.nfull==tail.nfull&&
           p.cancel_shape.branch==tail.branch&&p.cancel_shape.radix==tail.radix){p.shared=p.cancel_shape;p.shared_mask=3;}
    }else if(first.a.words>=fft_min_words){
        const size_t minimum=std::max({2*first.a.words+2,last.a.words+last.b.words,c.bound.words+2});
        const auto ring=pq16::cyclic_shape(minimum,16,1,false),a=linear_shape(first.a.words,first.b.words),
                   b=linear_shape(last.a.words,last.b.words),res=linear_shape(c.a.words,c.b.words);
        if(ring.nfull&&a.nfull&&b.nfull&&res.nfull&&ring.nfull<=a.nfull&&ring.nfull<=b.nfull&&
           ring.nfull<res.nfull&&pq16::cyclic_supported(ring,c.a.words,c.b.words)){
            p.shared=p.cancel_shape=ring;p.shared_mask=5;
        }
    }
    if(first.a.words>=fft_min_words){
        double cost=p.shared.nfull?shared_fft_work(shape,p.shared):0;
        if(!p.shared.nfull)for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
            const auto s=r.cancellation&&p.cancel_shape.nfull?p.cancel_shape:linear_shape(r.a.words,r.b.words);
            if(!s.nfull){cost=0;break;}cost+=pq16::native_cost(s,r.a.words,r.b.words);}
        if(cost>0){auto wide=wide_window_fft(shape,p.shared,cost);
            wide=plus_window_fft(shape,wide);
            if(wide.nfull){p.shared=p.cancel_shape=wide;p.shared_mask=window_shared_mask(shape);}}
    }
    size_t transient=0;
    for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
        size_t words=r.a.words+r.b.words;
        if(r.cancellation){
            if(p.middle)words=r.window.words;
            else if(p.cancel_shape.nfull)words=pq16::cyclic_period(p.cancel_shape)+(p.cancel_shape.recipe==pq16::Recipe::RightAngle);
            else words=std::max(words,r.subtract.shift+r.subtract.words)+1;
        }
        if(shape.count==2&&j==1)p.correction_words=words;else p.value_words=std::max(p.value_words,words);
        size_t work;
        if(p.middle&&r.cancellation)work=middle_bytes(r);
        else if(p.shared.nfull)work=((p.shared_mask>>j)&1)?pq16::cached_scratch_bytes(p.shared,r.a.words,r.b.words):
                                                                   pq16::scratch_bytes(p.shared,r.a.words,r.b.words);
        else if(r.cancellation&&p.cancel_shape.nfull)work=pq16::table_bytes(p.cancel_shape)+32*size_t(p.cancel_shape.nfull)+1536;
        else work=full_bytes(r.a.words,r.b.words);
        transient=std::max(transient,work);
    }
    // A shared FFT loads the complete fresh operand before it emits any
    // integer words. The reciprocal correction may therefore overwrite the
    // dead residual buffer. IFMA leaves retain their disjoint-output contract.
    if(shape.count==2&&p.shared.nfull){
        p.value_words=std::max(p.value_words,p.correction_words);p.correction_words=0;
    }
    p.work_bytes=transient+(p.shared.nfull?pq16::table_bytes(p.shared)+16*size_t(p.shared.nfull)+2048:256);
    p.storage_bytes=bytes(p.value_words)+bytes(p.correction_words)+p.work_bytes+256;

    return p;
}
[[gnu::always_inline]] inline bool peel_window_possible(const WindowGroupShape &shape) noexcept {
    if(shape.products[0].a.words<local_window_detail::fft_min_words)return false;
    size_t support=0;for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
        support=std::max({support,r.a.words,r.b.words,r.bound.words+1});}
    const size_t ring=size_t(1)<<(std::bit_width(support)-1);
    for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
        if(r.a.words-std::min(r.a.words,ring/2)+r.b.words-std::min(r.b.words,ring)>8||r.bound.words>=ring+8)return false;}
    // If the unpeeled 17-bit power-of-two recipe already fits at this point
    // count, peeling cannot satisfy our strict transform-reduction rule.
    const pq16::Shape wide{unsigned(2*ring),unsigned(2*ring),1,false,pq16::Recipe::CooleyTukeyPQ,17,false};
    return !shared_fft_supported(shape,wide);
}
[[gnu::noinline]] inline PeeledWindowPlan peeled_windows_query(const WindowGroupShape &shape,const LocalWindowPlan &p) noexcept {
    using namespace local_window_detail;
    if(!peel_window_possible(shape))return {};
    const auto &c=shape.products[shape.count-2];
    // Only consider a nearby smaller power-of-two FFT. Arithmetic bounds
    // determine the low-word lift; input lengths bound exact tail work.
    size_t support=0;for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
        support=std::max({support,r.a.words,r.b.words,r.bound.words+1});}
    if(support>=512){
        const size_t ring=size_t(1)<<(std::bit_width(support)-1);
        bool possible=true;size_t capacity=ring+1;
        for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
            const size_t ac=std::min(r.a.words,ring/2),bc=std::min(r.b.words,ring);
            possible&=r.a.words-ac+r.b.words-bc<=8&&r.bound.words<ring+8;
            capacity=std::max({capacity,r.bound.words+1,r.window.origin+r.window.words});}
        if(possible){
            const pq16::Shape fft{unsigned(2*ring),unsigned(2*ring),1,false,pq16::Recipe::PfaPQ,16,false};
            uint32_t ceiling=p.shared.nfull;
            if(!ceiling){ceiling=UINT32_MAX;for(unsigned j=0;j<shape.count;++j){const auto &r=shape.products[j];
                const auto s=linear_shape(r.a.words,r.b.words);if(!s.nfull){ceiling=0;break;}ceiling=std::min(ceiling,s.nfull);}}
            possible=fft.nfull<ceiling;
            for(unsigned j=0;j<shape.count&&possible;++j){const auto &r=shape.products[j];
                possible=pq16::cyclic_supported(fft,std::min(r.a.words,ring/2),std::min(r.b.words,ring));}
            if(possible){
                const size_t retained=shape.count==2?c.window.words:0;
                const size_t storage=bytes(capacity)+bytes(retained)+(shape.count==2?32:48)*size_t(fft.nfull)+pq16::table_bytes(fft)+4096;
                if(storage<=p.storage_bytes)return {fft,ring,capacity,storage,c.window.words,window_shared_mask(shape),shape.count};
            }
        }
    }

    return {};
}

class LocalWindowProducts {
    const LocalWindowPlan &p_;
    Frame &space_;FrameMark lifetime_;
    uint64_t *value_,*correction_,*window_output_;
    const pq16::Tables *tables_=nullptr;double *cache_=nullptr;
    void prepare(Span a) noexcept {
        if(tables_)return;
        tables_=pq16::prepare(space_,p_.shared);
        cache_=static_cast<double*>(space_.allocate(16*size_t(p_.shared.nfull)+128,128));
        pq16::forward_spectrum(cache_,a.data,a.words,*tables_,nullptr);
    }
  public:
    [[gnu::always_inline]] LocalWindowProducts(const LocalWindowPlan &p,Frame &space,uint64_t *window_output=nullptr) noexcept
        :p_(p),space_(space),lifetime_(space),value_(space.alloc<uint64_t>(p.value_words)),
         correction_(p.correction_words?space.alloc<uint64_t>(p.correction_words):value_),window_output_(window_output) {}
    [[gnu::always_inline]] WindowResult multiply(unsigned slot,Span a,Span b,Window w,MagnitudeBound limit) noexcept {
        auto *out=slot==p_.count-1?correction_:value_;
        if(p_.shared.nfull){
            prepare(a);
            if(p_.shared.recipe==pq16::Recipe::RightAngle){
                pq16::bounded_plus_multiply(out,a.data,a.words,b.data,b.words,cache_,*tables_,space_);
            }else if(p_.shared.bits>16||p_.shared.balanced){
                // The selected ring covers the complete positive product.
                // Cyclic prefix emission avoids the balanced linear-tail
                // correction and writes only the requested integer support.
                require(a.words+b.words<pq16::cyclic_period(p_.shared),SBN3_FATAL_MATH,"shared cyclic prefix support");
                pq16::cyclic_multiply(out,a.data,a.words,b.data,b.words,false,cache_,*tables_,space_,nullptr,a.words+b.words);
            }else pq16::apply_spectrum(out,cache_,a.data,a.words,b.data,b.words,false,*tables_,space_,nullptr);
        }
        else local_window_detail::full(out,a,b,space_);
        local_window_detail::check(out,a.words+b.words,limit);return {out+w.origin,w.words,0,false};
    }
    [[gnu::always_inline]] WindowResult cancel(unsigned slot,Span a,Span b,ShiftedSpan sub,Window w,MagnitudeBound limit) noexcept {
        if(p_.middle){auto *out=window_output_?window_output_:value_;
            const auto result=local_window_detail::middle(out,a,b,sub,w,space_);
            require(out[w.words-1]<limit.high_exclusive,SBN3_FATAL_MATH,"middle window magnitude");return result;}
        bool negative;
        if(p_.cancel_shape.nfull){
            const pq16::Tables *table=nullptr;double *cache=nullptr;
            if(p_.shared.nfull){if((p_.shared_mask>>slot)&1){prepare(a);cache=cache_;}table=tables_;}
            FrameMark temporary(space_);
            if(!table)table=pq16::prepare(space_,p_.cancel_shape);
            const size_t ring=pq16::cyclic_period(p_.cancel_shape);
            if(p_.cancel_shape.recipe==pq16::Recipe::CooleyTukeyPQ&&p_.cancel_shape.bits>16&&
               w.error_bits>=64&&w.origin&&w.origin+w.words<=ring&&limit.words+1<ring){
                pq16::cyclic_tail_multiply(value_,a.data,a.words,b.data,b.words,cache,*table,space_,w.origin);
                // Dropping each addend's low part loses at most one borrow
                // at origin. Together with the emitter's <2^40 error, this
                // stays inside the existing 64-bit residual error allowance.
                auto subtract=[&](size_t start,const uint64_t *v,size_t count){
                    if(!count)return;
                    if(start<w.origin){const size_t skip=std::min(count,w.origin-start);start+=skip;v+=skip;count-=skip;}
                    if(count)limbs::sub_from(value_+start,ring-start,v,count);
                };
                if(sub.value.words==1&&sub.value.data[0]==1)subtract(sub.shift%ring,sub.value.data,1);
                else{const size_t first=std::min(sub.value.words,ring-sub.shift);
                    subtract(sub.shift,sub.value.data,first);
                    if(sub.value.words>first)subtract(0,sub.value.data+first,sub.value.words-first);}
                negative=value_[ring-1]>>63;
                if(negative){for(size_t j=w.origin;j<ring;++j)value_[j]=~value_[j];
                    const uint64_t one=1;limbs::add_to(value_+w.origin,ring-w.origin,&one,1);}
                memset(value_+ring,0,(p_.value_words-ring)*8);
                local_window_detail::check(value_,p_.value_words,limit);return {value_+w.origin,w.words,64,negative};
            }
            if(p_.cancel_shape.recipe==pq16::Recipe::RightAngle){
                pq16::bounded_plus_multiply(value_,a.data,a.words,b.data,b.words,cache,*table,space_);
                if(sub.value.words==1&&sub.value.data[0]==1)limbs::plus_sub_power(value_,ring,sub.shift);
                else limbs::plus_sub_shifted(value_,ring,sub.value.data,sub.value.words,sub.shift);
                negative=limbs::plus_absolute(value_,ring);memset(value_+ring+1,0,(p_.value_words-ring-1)*8);
            }else{
                pq16::cyclic_multiply(value_,a.data,a.words,b.data,b.words,false,cache,*table,space_,nullptr);
                if(sub.value.words==1&&sub.value.data[0]==1)limbs::cyclic_sub_power(value_,ring,sub.shift%ring);
                else limbs::cyclic_sub_shifted(value_,ring,sub.value.data,sub.value.words,sub.shift);
                negative=limbs::cyclic_absolute(value_,ring);memset(value_+ring,0,(p_.value_words-ring)*8);
            }
        }else{
            local_window_detail::full(value_,a,b,space_);const size_t words=a.words+b.words;
            memset(value_+words,0,(p_.value_words-words)*8);
            negative=limbs::sub_from(value_+sub.shift,p_.value_words-sub.shift,sub.value.data,sub.value.words);
            if(negative){for(size_t j=0;j<p_.value_words;++j)value_[j]=~value_[j];
                const uint64_t one=1;limbs::add_to(value_,p_.value_words,&one,1);}
        }
        local_window_detail::check(value_,p_.value_words,limit);return {value_+w.origin,w.words,0,negative};
    }
    [[gnu::always_inline]] WindowResult retain(WindowResult r,uint64_t *destination) noexcept {
        if(p_.count==3){if(r.data!=destination)memcpy(destination,r.data,r.words*8);r.data=destination;}
        return r;
    }
};
struct LocalWindowFactory {
    [[gnu::always_inline]] size_t bytes(const WindowGroupShape &shape) const noexcept {
        if(leaf_windows_supported(shape))return leaf_windows_query(shape).bytes;
        const auto base=local_windows_query(shape);
        if(peel_window_possible(shape)){const auto p=peeled_windows_query(shape,base);if(p.ring)return p.bytes;}
        return base.storage_bytes;
    }
    constexpr size_t retained_words(const WindowGroupShape &shape) const noexcept {
        return shape.count==2?0:shape.products[shape.count-2].window.words;
    }
    template<class F> [[gnu::noinline]] bool with_peeled_group(const WindowGroupShape &shape,Frame &space,F &f) const noexcept {
        const auto base=local_windows_query(shape);const auto plan=peeled_windows_query(shape,base);
        if(!plan.ring)return false;
        FrameMark mark(space);PeeledWindows products(plan,space);f(products);return true;
    }
    template<class F> [[gnu::always_inline]] void with_group(const WindowGroupShape &shape,Frame &space,
                                                            uint64_t *window_output,F &&f) const noexcept {
        if(leaf_windows_supported(shape)){LeafWindowProducts products(leaf_windows_query(shape),space);f(products);return;}
        if(peel_window_possible(shape)&&with_peeled_group(shape,space,f))return;
        const auto plan=local_windows_query(shape);LocalWindowProducts products(plan,space,window_output);f(products);
    }
};
} // namespace sbn::v3::product
