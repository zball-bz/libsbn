#pragma once
#include "product/window.hpp"
#include "backend/u52/kernels.hpp"
#include "value/limbs.hpp"
namespace sbn::v3::product {
// Below the guarded-middle domain every stage is an ordinary small product.
// Keep this provider free of transform descriptors and capacity selection.
struct LeafWindowPlan {size_t value_words=0,correction_words=0,bytes=0;unsigned count=0;};
[[gnu::always_inline]] inline bool leaf_windows_supported(const WindowGroupShape &s) noexcept {
    for(unsigned j=0;j<s.count;++j)if(std::max(s.products[j].a.words,s.products[j].b.words)>=128)return false;
    return true;
}
[[gnu::always_inline]] inline LeafWindowPlan leaf_windows_query(const WindowGroupShape &s) noexcept {
    LeafWindowPlan p{};p.count=s.count;size_t work=0;
    for(unsigned j=0;j<s.count;++j){const auto &r=s.products[j];size_t words=r.a.words+r.b.words;
        if(r.cancellation)words=std::max(words,r.subtract.shift+r.subtract.words)+1;
        if(s.count==2&&j==1)p.correction_words=words;else p.value_words=std::max(p.value_words,words);
        work=std::max(work,u52::scratch_bytes(r.a.words,r.b.words));}
    p.bytes=8*(p.value_words+p.correction_words)+work+512;return p;
}
class LeafWindowProducts {
    Frame &space_;FrameMark lifetime_;uint64_t *value_,*correction_;size_t capacity_;unsigned count_;
    [[gnu::always_inline]] static void bound(const uint64_t *a,size_t n,MagnitudeBound b) noexcept {
        if(b.verify)require(b.words<n&&a[b.words]<b.high_exclusive&&limbs::zero(a+b.words+1,n-b.words-1),
                            SBN3_FATAL_MATH,"leaf window bound");
    }
  public:
    [[gnu::always_inline]] LeafWindowProducts(LeafWindowPlan p,Frame &space) noexcept
        :space_(space),lifetime_(space),value_(space.alloc<uint64_t>(p.value_words)),
         correction_(p.correction_words?space.alloc<uint64_t>(p.correction_words):value_),capacity_(p.value_words),count_(p.count){}
    [[gnu::always_inline]] WindowResult multiply(unsigned slot,Span a,Span b,Window w,MagnitudeBound limit) noexcept {
        auto *out=slot==count_-1?correction_:value_;u52::multiply(out,a.data,a.words,b.data,b.words,space_);
        bound(out,a.words+b.words,limit);return {out+w.origin,w.words,0,false};
    }
    [[gnu::always_inline]] WindowResult cancel(unsigned,Span a,Span b,ShiftedSpan sub,Window w,MagnitudeBound limit) noexcept {
        u52::multiply(value_,a.data,a.words,b.data,b.words,space_);
        memset(value_+a.words+b.words,0,(capacity_-a.words-b.words)*8);
        const bool negative=limbs::sub_from(value_+sub.shift,capacity_-sub.shift,sub.value.data,sub.value.words);
        if(negative){for(size_t j=0;j<capacity_;++j)value_[j]=~value_[j];const uint64_t one=1;limbs::add_to(value_,capacity_,&one,1);}
        bound(value_,capacity_,limit);return {value_+w.origin,w.words,0,negative};
    }
    [[gnu::always_inline]] WindowResult retain(WindowResult r,uint64_t *destination) noexcept {
        if(count_==3){memcpy(destination,r.data,r.words*8);r.data=destination;}return r;
    }
};
} // namespace sbn::v3::product
