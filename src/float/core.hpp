// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/float.hpp"
#include "common/checked.hpp"
#include "runtime/scratch.hpp"

namespace sbn::detail {
struct float_access {
    static sbn3_int*begin(float_ref r)noexcept{return r.magnitude_.begin_write();}
    static float_metadata&metadata(float_ref r)noexcept{return *r.metadata_;}
    static void finish(float_ref r)noexcept{r.magnitude_.finish_write();}
};
}
namespace sbn::v3::floating {
inline std::size_t magnitude_bits(integer_view x)noexcept{return x.size()?64*(x.size()-1)+64-unsigned(__builtin_clzll(x.data()[x.size()-1])):0;}
inline void validate(float_view x)noexcept{
    require(x.magnitude().size()<=float_precision_limit/64,SBN3_FATAL_SIZE,"float input length");
    if(x.finite()&&!x.zero()){
        const __int128 top=__int128(x.exponent2())+magnitude_bits(x.magnitude());
        require(top>=-float_exponent_limit&&top<=float_exponent_limit,SBN3_FATAL_ARGUMENT,"float input exponent");
    }
}
inline bool overlaps_words(const void*a,std::size_t an,const void*b,std::size_t bn)noexcept{return overlaps(a,an*8,b,bn*8);}
inline void check_output(float_ref out,float_view a,float_view b={})noexcept{
    const auto p=out.precision().count;detail::valid_precision({p});
    require(out.capacity()>=(p+63)/64,SBN3_FATAL_WORKSPACE,"float output capacity",(p+63)/64,out.capacity());
    const auto*r=detail::float_access::begin(out);
    for(auto v:{a,b})require(r->data==v.magnitude().data()||!overlaps_words(r->data,(p+63)/64,v.magnitude().data(),v.magnitude().size()),SBN3_FATAL_ARGUMENT,"float partial alias");
}
float_report special(float_ref,float_class,bool,float_flag={})noexcept;
// Sticky describes a nonzero positive magnitude below the source's low bit.
// It is only used when at least one source bit is actually discarded.
float_report round_magnitude(float_ref,integer_view,std::int64_t,bool,rounding,bool sticky=false)noexcept;
// Exact finite intermediates (e.g. the unrounded product in FMA) may exceed
// the public exponent domain before cancellation. Caller validates source
// extents/aliases and output capacity; all intermediate word indices must fit
// the wider internal range. Final output still uses normal range semantics.
float_report sum_finite(float_ref,float_view,float_view,execution,rounding)noexcept;
}
