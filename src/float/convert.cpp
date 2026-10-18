// Original libsbn code; LGPL-3.0-or-later.
#include "float/convert.hpp"
#include "value/divrem_words.hpp" // Retains the word-kernel donor notice.
#include <algorithm>
#include <cstring>

namespace sbn::v3::floating {
namespace {
struct Conversion {
    requirements need{};
    std::size_t shift=0,retained_bits=0;
    bool left=false,discarded=false,increment=false;
};
bool valid_mode(rounding mode) noexcept {
    return mode==rounding::nearest_even||mode==rounding::toward_zero||mode==rounding::upward||
           mode==rounding::downward||mode==rounding::away_zero;
}
bool bit(integer_view m,std::size_t at) noexcept {
    return at/64<m.size()&&((m.data()[at/64]>>(at%64))&1);
}
bool lower_nonzero(integer_view m,std::size_t count) noexcept {
    const auto whole=std::min(count/64,m.size());
    for(std::size_t i=0;i<whole;++i)if(m.data()[i])return true;
    return count%64&&whole<m.size()&&(m.data()[whole]&((UINT64_C(1)<<(count%64))-1));
}
Conversion describe(integer_ref out,float_view input,rounding mode) noexcept {
    Conversion p;p.need.scratch_alignment=8;
    if(!input.finite()||!valid_mode(mode)){
        p.need.status=support::unsupported;p.need.reason="integer conversion finite input/rounding";return p;
    }
    validate(input);
    if(input.zero())return p;
    const auto m=input.magnitude();const auto width=magnitude_bits(m);
    if(input.exponent2()>=0){
        const __uint128_t bits=__uint128_t(width)+std::uint64_t(input.exponent2());
        if(bits>SIZE_MAX-63){p.need.status=support::unsupported;p.need.reason="integer conversion bit length";return p;}
        p.left=true;p.shift=std::size_t(input.exponent2());p.retained_bits=std::size_t(bits);
        p.need.output_limbs=(p.retained_bits+63)/64;
    }else{
        const __uint128_t distance=-__int128_t(input.exponent2());
        if(distance>SIZE_MAX){p.need.status=support::unsupported;p.need.reason="integer conversion shift";return p;}
        p.shift=std::size_t(distance);p.retained_bits=width>p.shift?width-p.shift:0;
        const bool guard=p.shift<=width&&bit(m,p.shift-1);
        const bool sticky=p.shift>width||lower_nonzero(m,p.shift-1);
        p.discarded=guard||sticky;
        switch(mode){
            case rounding::nearest_even:p.increment=guard&&(sticky||bit(m,p.shift));break;
            case rounding::toward_zero:break;
            case rounding::upward:p.increment=p.discarded&&!input.negative();break;
            case rounding::downward:p.increment=p.discarded&&input.negative();break;
            case rounding::away_zero:p.increment=p.discarded;break;
        }
        // Include a possible carry into one further bit. This needs an
        // extra limb only when the retained magnitude fills whole limbs.
        p.need.output_limbs=(p.retained_bits+unsigned(p.increment)+63)/64;
    }
    std::size_t bytes=0;
    if(!mul_size(p.need.output_limbs,8,bytes)){
        p.need.status=support::unsupported;p.need.reason="integer conversion output bytes";return p;
    }
    if(out.capacity()<p.need.output_limbs){
        p.need.status=support::capacity;p.need.reason="integer conversion output capacity";return p;
    }
    const auto destination=out.view().data();
    if(destination!=m.data()&&overlaps(destination,bytes,m.data(),m.size()*8)){
        p.need.status=support::unsupported;p.need.reason="integer conversion partial alias";return p;
    }
    return p;
}
}
}

namespace sbn {
requirements to_integer_requirements(integer_ref out,float_view input,rounding mode) noexcept {
    return v3::floating::describe(out,input,mode).need;
}
float_report to_integer_into(integer_ref out,float_view input,rounding mode) noexcept {
    using namespace v3::floating;
    const auto p=describe(out,input,mode);
    v3::require(p.need.supported(),p.need.status==support::capacity?SBN3_FATAL_WORKSPACE:SBN3_FATAL_ARGUMENT,
            p.need.reason,p.need.output_limbs,out.capacity());
    const auto m=input.magnitude();const bool negative=input.negative();auto* result=out.begin_write();
    if(m.zero()){result->size=result->negative=0;out.finish_write();return {};}
    std::size_t size=0;
    if(p.left){
        const auto whole=p.shift/64;const unsigned part=unsigned(p.shift%64);
        size=m.size()+whole;
        if(part){
            // The kernel walks high to low, also safe for a same-base
            // destination shifted upward by whole limbs.
            const auto carry=v3::divrem_words::shift_left(result->data+whole,m.data(),m.size(),part);
            if(carry)result->data[size++]=carry;
        }else std::memmove(result->data+whole,m.data(),m.size()*8);
        if(whole)std::fill_n(result->data,whole,0);
    }else{
        size=(p.retained_bits+63)/64;
        const auto whole=p.shift/64;const unsigned part=unsigned(p.shift%64);
        // Low-to-high traversal preserves a same-base right-shift source.
        // Read the extra high source limb even if its own shifted result
        // is zero, without demanding an unnecessary output limb for it.
        if(part)for(std::size_t i=0;i<size;++i){
            const auto j=i+whole;
            result->data[i]=(m.data()[j]>>part)|(j+1<m.size()?m.data()[j+1]<<(64-part):0);
        }else if(size)std::memmove(result->data,m.data()+whole,size*8);
        bool carry=p.increment;
        for(std::size_t i=0;i<size&&carry;++i){++result->data[i];carry=result->data[i]==0;}
        if(carry)result->data[size++]=1;
    }
    while(size&&!result->data[size-1])--size;
    result->size=size;result->negative=size&&negative;out.finish_write();
    return {p.discarded?error_bound::power_of_two(mode==rounding::nearest_even?-1:0):error_bound{},
            p.discarded?float_flag::rounded:float_flag::none};
}
}
