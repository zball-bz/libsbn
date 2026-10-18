#pragma once
#include "product/local_windows.hpp"
#include "product/cyclic_reconstruct.hpp"

namespace sbn::v3::product {
struct LowLiftStep {size_t words=0,an=0,bn=0;pq16::Shape full{};};
struct LowLiftProgram {LowLiftStep steps[3]{};unsigned count=0;};
inline LowLiftProgram compile_low_lift_program(const WindowGroupShape& shape,size_t ring) noexcept {
    LowLiftProgram out{};out.count=shape.count;
    require(shape.count<=3&&ring,SBN3_FATAL_ARGUMENT,"low lift compile shape");
    for(unsigned j=0;j<shape.count;++j)if(shape.products[j].bound.words>=ring){
        const auto& p=shape.products[j];auto& step=out.steps[j];
        step.words=p.bound.words-ring+1;step.an=std::min(p.a.words,step.words);step.bn=std::min(p.b.words,step.words);
        if(step.words>8)step.full=local_window_detail::linear_shape(step.an,step.bn);
    }
    return out;
}
inline size_t low_lift_bytes(size_t words) noexcept {
    return local_window_detail::bytes(2*words)+
        (words>8?local_window_detail::full_bytes(words,words):0)+512;
}
// Independent exact low words of a*b-sub, computed before the cyclic
// provider consumes any operand. The remaining half of low is temporary
// full-product output. Both it and the kernel scratch belong to this phase.
inline uint64_t *low_lift_prepare(Frame &space,size_t words,Span a,Span b,ShiftedSpan sub,const LowLiftStep* selected=nullptr) noexcept {
    if(selected)require(selected->words==words&&selected->an==std::min(a.words,words)&&
        selected->bn==std::min(b.words,words),SBN3_FATAL_MATH,"compiled low lift shape");
    auto *low=space.alloc<uint64_t>(2*words);memset(low,0,16*words);
    if(words<=8){
        for(size_t j=0;j<std::min(b.words,words);++j){uint64_t carry=0;
            for(size_t i=0;i+j<words&&i<a.words;++i){const auto x=(unsigned __int128)a.data[i]*b.data[j]+low[i+j]+carry;
                low[i+j]=uint64_t(x);carry=uint64_t(x>>64);}}
    }else{
        FrameMark mark(space);
        if(selected)local_window_detail::full_selected(low,{a.data,selected->an},{b.data,selected->bn},space,selected->full);
        else local_window_detail::full(low,{a.data,std::min(a.words,words)},{b.data,std::min(b.words,words)},space);
    }
    if(sub.shift<words&&sub.value.words)
        limbs::sub_from(low+sub.shift,words-sub.shift,sub.value.data,std::min(sub.value.words,words-sub.shift));
    return low;
}
inline size_t window_group_low_lift_bytes(const WindowGroupShape&s,size_t ring) noexcept {
    if(ring>=s.products[s.count-2].bound.words)return 0;
    size_t bytes=0;
    for(unsigned j=0;j<s.count;++j)if(s.products[j].bound.words>=ring)
        bytes=std::max(bytes,low_lift_bytes(s.products[j].bound.words-ring+1));
    // A larger low product can switch to a smaller physical recipe. Quote
    // every member instead of assuming its scratch is monotone in length.
    return bytes;
}
}
