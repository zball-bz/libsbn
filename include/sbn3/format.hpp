// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/radix.hpp"
#include <format>
#include <limits>
namespace sbn::detail {
[[noreturn]]inline void format_error(const char*why){
#if defined(__cpp_exceptions)
    throw std::format_error(why);
#else
    fail(SBN3_FATAL_ARGUMENT,why);
#endif
}
}
namespace std {
// ASCII integer presentation: d/x/X/o/b/B, fill/alignment, sign, alternate
// prefix, zero padding and fixed/dynamic width. No locale grouping ('L').
template<>struct formatter<sbn::integer,char> {
    char fill=' ',alignment=0,sign=0,type='d';bool alternate=false,zero=false,dynamic=false;
    size_t width=0,arg_id=0;
    static constexpr bool align_char(char c)noexcept{return c=='<'||c=='>'||c=='^';}
    constexpr auto parse(format_parse_context&ctx){
        auto it=ctx.begin(),end=ctx.end();
        if(it!=end&&it+1!=end&&align_char(it[1])){if(*it=='{'||*it=='}')sbn::detail::format_error("invalid integer fill");fill=*it;alignment=it[1];it+=2;}
        else if(it!=end&&align_char(*it))alignment=*it++;
        if(it!=end&&(*it=='+'||*it=='-'||*it==' '))sign=*it++;
        if(it!=end&&*it=='#'){alternate=true;++it;}
        if(it!=end&&*it=='0'){zero=true;++it;}
        if(it!=end&&*it=='{'){
            dynamic=true;++it;
            if(it!=end&&*it=='}')arg_id=ctx.next_arg_id();
            else{if(it==end||*it<'0'||*it>'9')sbn::detail::format_error("invalid integer width argument");
                for(;it!=end&&*it>='0'&&*it<='9';++it){const unsigned digit=*it-'0';if(arg_id>(numeric_limits<size_t>::max()-digit)/10)sbn::detail::format_error("integer width argument overflow");arg_id=10*arg_id+digit;}ctx.check_arg_id(arg_id);}
            if(it==end||*it!='}')sbn::detail::format_error("unterminated integer width argument");++it;
        }else for(;it!=end&&*it>='0'&&*it<='9';++it){const unsigned digit=*it-'0';if(width>(numeric_limits<size_t>::max()-digit)/10)sbn::detail::format_error("integer width overflow");width=10*width+digit;}
        if(it!=end&&*it!='}')type=*it++;
        if((type!='d'&&type!='x'&&type!='X'&&type!='o'&&type!='b'&&type!='B')||(it!=end&&*it!='}'))sbn::detail::format_error("unsupported integer presentation");
        return it;
    }
    template<class Ctx>auto format(const sbn::integer&value,Ctx&ctx)const{
        size_t field=width;
        if(dynamic)field=visit_format_arg([](auto v)->size_t{
            using T=decltype(v);
            if constexpr(is_integral_v<T>&&!is_same_v<T,bool>&&!is_same_v<T,char>){
                if constexpr(is_signed_v<T>)if(v<0)sbn::detail::format_error("negative integer format width");
                if(__uint128_t(v)>numeric_limits<size_t>::max())sbn::detail::format_error("integer format width overflow");return size_t(v);
            }else sbn::detail::format_error("integer format width must be integral");
        },ctx.arg(arg_id));
        const unsigned base=type=='x'||type=='X'?16:type=='o'?8:type=='b'||type=='B'?2:10;
        auto text=sbn::to_string(value,type=='X'?sbn::radix{"0123456789ABCDEF"}:sbn::radix{base});auto digits=text.view();
        char prefix_sign=0;if(value.negative()){prefix_sign='-';digits.remove_prefix(1);}else if(sign=='+'||sign==' ')prefix_sign=sign;
        string_view prefix;if(alternate){if(base==16)prefix=type=='X'?"0X":"0x";else if(base==2)prefix=type=='B'?"0B":"0b";else if(base==8&&digits!="0")prefix="0";}
        const auto count=digits.size()+prefix.size()+unsigned(prefix_sign!=0),padding=field>count?field-count:0;
        const bool zeros=zero&&!alignment;const auto before=zeros||alignment=='<'?0:alignment=='^'?padding/2:padding;
        auto out=ctx.out();auto pad=[&](size_t n,char c){while(n--)*out++=c;};
        pad(before,fill);if(prefix_sign)*out++=prefix_sign;for(char c:prefix)*out++=c;if(zeros)pad(padding,'0');
        out=std::format_to(out,"{}",digits);if(!zeros)pad(padding-before,fill);return out;
    }
};
template<>struct formatter<sbn::integer_text,char>:formatter<string_view,char>{
    template<class Ctx>auto format(const sbn::integer_text&text,Ctx&ctx)const{return formatter<string_view,char>::format(text.view(),ctx);}
};
}
