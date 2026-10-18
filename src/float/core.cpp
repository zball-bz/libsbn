// Original libsbn code; LGPL-3.0-or-later.
#include "float/core.hpp"
#include <algorithm>
#include <cstring>

namespace sbn::v3::floating {
float_report special(float_ref out,float_class c,bool negative,float_flag flags)noexcept{
    auto*r=detail::float_access::begin(out);auto&m=detail::float_access::metadata(out);
    r->size=0;r->negative=0;m.classification=c;m.exponent2=0;m.negative=negative;
    detail::float_access::finish(out);
    return {c==float_class::finite?error_bound{}:error_bound::unbounded(),flags};
}
namespace {
bool lower_nonzero(integer_view x,std::size_t n)noexcept{
    const auto whole=n/64;for(std::size_t i=0;i<whole;++i)if(x.data()[i])return true;
    return n%64&&(x.data()[whole]&((std::uint64_t(1)<<(n%64))-1));
}
std::uint64_t shifted_word(integer_view x,std::size_t bit)noexcept{
    const auto i=bit/64;const unsigned s=bit%64;
    if(i>=x.size())return 0;
    return (x.data()[i]>>s)|(s&&i+1<x.size()?x.data()[i+1]<<(64-s):0);
}
struct Span {
    integer_view x;std::int64_t first,last;unsigned shift;
    explicit Span(float_view v)noexcept:x(v.magnitude()){
        const auto e=v.exponent2();first=e/64-std::int64_t(e%64<0);
        shift=unsigned(e-first*64);last=first+std::int64_t(x.size())+unsigned(shift!=0);
    }
    bool active(std::int64_t i)const noexcept{return i>=first&&i<last;}
    std::uint64_t word(std::int64_t i)const noexcept{
        if(!active(i))return 0;
        const auto j=std::size_t(i-first);
        return (j<x.size()?x.data()[j]<<shift:0)|(shift&&j?x.data()[j-1]>>(64-shift):0);
    }
};
int magnitude_compare(float_view a,float_view b)noexcept{
    if(a.zero()||b.zero())return int(!a.zero())-int(!b.zero());
    const auto at=a.exponent2()+std::int64_t(magnitude_bits(a.magnitude()));
    const auto bt=b.exponent2()+std::int64_t(magnitude_bits(b.magnitude()));
    if(at!=bt)return at>bt?1:-1;
    Span x(a),y(b);auto i=std::max(x.last,y.last)-1;
    const auto low=std::min(x.first,y.first);
    while(i>=low){
        if(!x.active(i)&&!y.active(i)){
            const auto xn=i>=x.first?std::min(i,x.last-1):low-1;
            const auto yn=i>=y.first?std::min(i,y.last-1):low-1;
            i=std::max(xn,yn);if(i<low)break;
        }
        const auto u=x.word(i),v=y.word(i);if(u!=v)return u>v?1:-1;--i;
    }
    return 0;
}
// Keep the highest nonzero-result window. Skipped zero runs and borrow runs
// are represented by their endpoints, never by exponent-gap-sized buffers.
struct Window {
    std::uint64_t*words;std::size_t capacity,head=0;std::int64_t last=0;
    bool any=false,sticky=false;
    Window(std::uint64_t*w,std::size_t n):words(w),capacity(n){std::fill_n(w,n,0);}
    void push(std::int64_t i,std::uint64_t w)noexcept{
        if(!w)return;
        if(!any){any=true;last=i;words[capacity-1]=w;return;}
        const auto distance=std::uint64_t(i-last);require(distance,SBN3_FATAL_MATH,"float window order");
        if(distance>=capacity){
            for(std::size_t j=0;j<capacity;++j)sticky|=words[j]!=0;
            std::fill_n(words,capacity,0);head=0;
        }else for(std::size_t j=0;j<distance;++j){sticky|=words[head]!=0;words[head]=0;if(++head==capacity)head=0;}
        last=i;words[(head+capacity-1)%capacity]=w;
    }
    void ones(std::int64_t i,std::uint64_t count)noexcept{
        if(count>=capacity){
            for(std::size_t j=0;j<capacity;++j)sticky|=words[j]!=0;
            sticky|=count>capacity;std::fill_n(words,capacity,UINT64_MAX);
            head=0;any=true;last=i+std::int64_t(count)-1;
        }else for(std::uint64_t j=0;j<count;++j)push(i+std::int64_t(j),UINT64_MAX);
    }
    float_report emit(float_ref out,bool negative,rounding mode)noexcept{
        if(!any)return special(out,float_class::finite,mode==rounding::downward);
        std::rotate(words,words+head,words+capacity);
        return round_magnitude(out,integer_view{{words,capacity}},(last-std::int64_t(capacity)+1)*64,negative,mode,sticky);
    }
};
float_report add_finite(float_ref out,float_view a,float_view b,rounding mode,std::uint64_t*buffer,std::size_t n)noexcept{
    bool subtract=a.negative()!=b.negative();bool negative=a.negative();
    if(subtract){const auto c=magnitude_compare(a,b);if(!c)return special(out,float_class::finite,mode==rounding::downward);
        if(c<0){std::swap(a,b);negative=a.negative();}}
    Span x(a),y(b);auto i=std::min(x.first,y.first);const auto end=std::max(x.last,y.last);
    unsigned carry=0;Window window(buffer,n);
    while(i<end){
        if(!x.active(i)&&!y.active(i)){
            const auto next=std::min(i<x.first?x.first:end,i<y.first?y.first:end);
            if(carry){if(subtract)window.ones(i,std::uint64_t(next-i));else{window.push(i,1);carry=0;}}
            i=next;continue;
        }
        const auto u=x.word(i),v=y.word(i);std::uint64_t w;
        if(subtract){const __uint128_t sub=__uint128_t(v)+carry;w=u-std::uint64_t(sub);carry=__uint128_t(u)<sub;}
        else{const __uint128_t sum=__uint128_t(u)+v+carry;w=std::uint64_t(sum);carry=unsigned(sum>>64);}
        window.push(i,w);++i;
    }
    if(subtract)require(!carry,SBN3_FATAL_MATH,"float nonnegative subtraction");
    else if(carry)window.push(end,carry);
    return window.emit(out,negative,mode);
}
}

float_report round_magnitude(float_ref out,integer_view input,std::int64_t exponent,bool negative,rounding mode,bool sticky)noexcept{
    if(input.zero())return special(out,float_class::finite,negative);
    const auto p=out.precision().count,len=magnitude_bits(input),cut=len>p?len-p:0;
    const auto original_top=exponent+std::int64_t(len);
    const bool guard=cut&&((input.data()[(cut-1)/64]>>((cut-1)%64))&1);
    const bool tail=sticky||(cut&&lower_nonzero(input,cut-1));
    const bool discarded=guard||tail;
    require(!sticky||cut,SBN3_FATAL_MATH,"float sticky without discarded bits");
    const auto kept=len-cut,words=(kept+63)/64;
    auto*r=detail::float_access::begin(out);auto&m=detail::float_access::metadata(out);
    for(std::size_t i=0;i<words;++i)r->data[i]=shifted_word(input,cut+64*i);
    bool increment=false;
    switch(mode){
        case rounding::nearest_even:increment=guard&&(tail||(r->data[0]&1));break;
        case rounding::toward_zero:break;
        case rounding::upward:increment=discarded&&!negative;break;
        case rounding::downward:increment=discarded&&negative;break;
        case rounding::away_zero:increment=discarded;break;
    }
    unsigned carry=increment;
    for(std::size_t i=0;i<words&&carry;++i){++r->data[i];carry=!r->data[i];}
    auto e=exponent+std::int64_t(cut);
    const auto bound_exp=e-(mode==rounding::nearest_even?1:0);
    if(carry||(kept%64&&r->data[words-1]>>(kept%64))){
        for(std::size_t i=0;i<words;++i)r->data[i]=(r->data[i]>>1)|(i+1<words?r->data[i+1]<<63:std::uint64_t(carry)<<63);
        ++e;
    }
    const auto top=e+std::int64_t(64*(words-1)+64-unsigned(__builtin_clzll(r->data[words-1])));
    if(top>float_exponent_limit)return special(out,float_class::infinity,negative,float_flag::overflow|float_flag::rounded);
    if(top<-float_exponent_limit){auto report=special(out,float_class::finite,negative,float_flag::underflow|float_flag::rounded);report.error=error_bound::power_of_two(original_top);return report;}
    r->size=words;r->negative=0;m.exponent2=e;m.negative=negative;m.classification=float_class::finite;
    detail::float_access::finish(out);
    return {discarded?error_bound::power_of_two(bound_exp):error_bound{},discarded?float_flag::rounded:float_flag::none};
}
float_report sum_finite(float_ref out,float_view a,float_view b,execution ex,rounding mode)noexcept{
    require(a.finite()&&b.finite(),SBN3_FATAL_ARGUMENT,"finite sum intermediates");
    if(a.zero()&&b.zero())return special(out,float_class::finite,a.negative()==b.negative()?a.negative():mode==rounding::downward);
    if(a.zero())return round_magnitude(out,b.magnitude(),b.exponent2(),b.negative(),mode);
    if(b.zero())return round_magnitude(out,a.magnitude(),a.exponent2(),a.negative(),mode);
    const auto p=out.precision().count,n=(p+63)/64+2;
    if(p<=256){alignas(64)std::uint64_t words[6];return add_finite(out,a,b,mode,words,n);}
    const auto need=n*8+63;
    require(ex.scratch.size()>=need,SBN3_FATAL_WORKSPACE,"float add scratch",need,ex.scratch.size());
    for(auto v:{a,b})require(!overlaps(ex.scratch.data(),ex.scratch.size(),v.magnitude().data(),v.magnitude().size()*8),SBN3_FATAL_ARGUMENT,"float add input scratch alias");
    const auto*r=detail::float_access::begin(out);
    require(!overlaps(ex.scratch.data(),ex.scratch.size(),r->data,((p+63)/64)*8),SBN3_FATAL_ARGUMENT,"float add output scratch alias");
    auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
    return add_finite(out,a,b,mode,frame.alloc<std::uint64_t>(n),n);
}
}

namespace sbn {
float_report round_into(float_ref out,float_view x,rounding mode)noexcept{
    using namespace v3::floating;validate(x);check_output(out,x);
    if(!x.finite())return special(out,x.classification(),x.negative());
    return round_magnitude(out,x.magnitude(),x.exponent2(),x.negative(),mode);
}
float_report scale2_into(float_ref out,float_view x,std::int64_t shift,rounding mode)noexcept{
    using namespace v3::floating;validate(x);check_output(out,x);
    if(!x.finite()||x.zero())return special(out,x.classification(),x.negative());
    const __int128 exponent=__int128(x.exponent2())+shift;
    const auto top=exponent+magnitude_bits(x.magnitude());
    if(top>float_exponent_limit)return special(out,float_class::infinity,x.negative(),float_flag::overflow|float_flag::rounded);
    // Quantization can raise the top exponent by one. Keep that boundary
    // case in the common rounder, just as add/sub/FMA do after cancellation.
    if(top<-__int128(float_exponent_limit)-1){
        auto r=special(out,float_class::finite,x.negative(),float_flag::underflow|float_flag::rounded);
        r.error=error_bound::power_of_two(top<INT64_MIN?INT64_MIN:std::int64_t(top));return r;
    }
    return round_magnitude(out,x.magnitude(),std::int64_t(exponent),x.negative(),mode);
}
requirements add_requirements(float_ref out,float_view a,float_view b)noexcept{
    const auto p=out.precision().count;
    if(!p||p>float_precision_limit)return {support::unsupported,0,64,0,"float precision"};
    const auto n=(p+63)/64;
    const bool trivial=!a.finite()||!b.finite()||a.zero()||b.zero();
    return {out.capacity()<n?support::capacity:support::yes,trivial||p<=256?0:(n+2)*8+63,64,n,""};
}
float_report add_into(float_ref out,float_view a,float_view b,execution ex,rounding mode)noexcept{
    using namespace v3::floating;validate(a);validate(b);check_output(out,a,b);
    if(a.is_nan()||b.is_nan())return special(out,float_class::quiet_nan,false);
    if(!a.finite()||!b.finite()){
        if(!a.finite()&&!b.finite()&&a.negative()!=b.negative())return special(out,float_class::quiet_nan,false,float_flag::invalid);
        const auto x=a.finite()?b:a;return special(out,float_class::infinity,x.negative());
    }
    return sum_finite(out,a,b,ex,mode);
}
float_report subtract_into(float_ref out,float_view a,float_view b,execution ex,rounding mode)noexcept{return add_into(out,a,b.negated(),ex,mode);}
std::partial_ordering compare(float_view a,float_view b)noexcept{
    using namespace v3::floating;validate(a);validate(b);
    if(a.is_nan()||b.is_nan())return std::partial_ordering::unordered;
    if(a.zero()&&b.zero())return std::partial_ordering::equivalent;
    if(a.negative()!=b.negative())return a.negative()?std::partial_ordering::less:std::partial_ordering::greater;
    int c=0;
    if(!a.finite()||!b.finite())c=int(!a.finite())-int(!b.finite());
    else c=magnitude_compare(a,b);
    if(a.negative())c=-c;
    return c<0?std::partial_ordering::less:c>0?std::partial_ordering::greater:std::partial_ordering::equivalent;
}
}
