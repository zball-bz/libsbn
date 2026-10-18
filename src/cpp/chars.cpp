// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/chars.hpp"
#include "sbn3/decimal.h"
#include "radix/leaf.hpp"
#include "common/checked.hpp"
namespace sbn {
namespace {
using v3::overlaps;
size_t aligned(size_t n,size_t a)noexcept{size_t r;detail::require(v3::align_size(n,a,r),SBN3_FATAL_SIZE,"character workspace");return r;}
size_t length(const char*first,const char*last)noexcept{return first==last?0:size_t(last-first);}
bool decimal(radix r)noexcept{return r.base==10&&(r.alphabet.empty()||r.alphabet=="0123456789");}
struct Format {requirements need;size_t digits=0,work_at=0;bool small=false,stack=false;};
Format quote_format(std::span<char>out,integer_view a,radix r,unsigned workers)noexcept{
    (void)out;Format p;
    if(decimal(r)&&a.size()<=8&&workers&&workers<=32){p.small=true;p.digits=(a.zero()?1:a.size()==2?39:20*a.size())+unsigned(a.negative());p.need.scratch_alignment=8;return p;}
    const auto f=format_requirements(a,r,workers);p.need=f;p.digits=f.characters;
    if(!f.supported())return p;
    // Decimal leaves use fixed stack storage when the caller's tight buffer
    // cannot hold the conservative digit bound. No heap or runtime object.
    p.stack=p.digits<=1024;
    if(!p.stack){p.work_at=aligned(p.digits,f.scratch_alignment);p.need.scratch_bytes=p.work_at+f.scratch_bytes;}
    return p;
}
std::to_chars_result write(const Format&p,std::span<char>out,integer_view a,radix r,execution ex)noexcept{
    if(!p.need.supported())return {out.data(),std::errc::invalid_argument};
    detail::require(ex.scratch.size()>=p.need.scratch_bytes&&(!p.need.scratch_bytes||!(uintptr_t(ex.scratch.data())&(p.need.scratch_alignment-1))),SBN3_FATAL_WORKSPACE,"to_chars workspace",p.need.scratch_bytes,ex.scratch.size());
    detail::require(!overlaps(out.data(),out.size(),a.data(),a.size()*8)&&!overlaps(out.data(),out.size(),ex.scratch.data(),p.need.scratch_bytes),SBN3_FATAL_ARGUMENT,"to_chars overlap");
    char small[1024];
    if(p.small){char*begin=out.size()>=p.digits?out.data():small,*end=begin;if(a.negative())*end++='-';
        if(a.zero())*end++='0';else if(a.size()==1)end=sbn3_decimal_write_u64(end,a.data()[0]);else if(a.size()==2)end=sbn3_decimal_write_u128(end,{a.data()[0],a.data()[1]});else end=sbn3_decimal_write_small(end,a.data(),a.size());
        const auto n=size_t(end-begin);if(n>out.size())return {out.empty()?out.data():out.data()+out.size(),std::errc::value_too_large};
        if(begin!=out.data())std::memcpy(out.data(),begin,n);return {out.data()+n,{}};
    }
    std::span<char>target=p.stack?std::span<char>{small,sizeof small}:std::span<char>{reinterpret_cast<char*>(ex.scratch.data()),p.digits};
    auto work=ex;if(p.work_at)work.scratch=work.scratch.subspan(p.work_at);
    const auto result=format_into(target,a,r,work);detail::require(bool(result),SBN3_FATAL_MATH,"to_chars admitted format");
    if(result.text.size()>out.size())return {out.empty()?out.data():out.data()+out.size(),std::errc::value_too_large};
    if(result.text.size()&&result.text.data()!=out.data())std::memmove(out.data(),result.text.data(),result.text.size());
    return {out.data()+result.text.size(),{}};
}
struct Prefix {std::string_view text{};size_t consumed=0;bool fold=false,valid=false;const char*why="invalid integer prefix";};
Prefix prefix(std::string_view s,radix r)noexcept{
    Prefix p;if(r.base<2||r.base>64||(!r.alphabet.empty()&&r.alphabet.size()!=r.base)){p.why="invalid radix";return p;}
    if(s.empty())return p;
    unsigned char decode[256];const bool custom=!r.alphabet.empty();
    if(custom){std::memset(decode,255,sizeof decode);for(unsigned j=0;j<r.base;++j){const auto c=static_cast<unsigned char>(r.alphabet[j]);if(c=='-'||decode[c]!=255){p.why="invalid alphabet";return p;}decode[c]=j;}}
    const size_t start=!s.empty()&&s.front()=='-'?1:0;size_t end=start;
    if(decimal(r))end+=v3::radix::validate_digits(reinterpret_cast<const uint8_t*>(s.data()+start),s.size()-start,10,v3::radix::decimal_ascii_plan().decode);
    else for(;end<s.size();++end){const auto c=static_cast<unsigned char>(s[end]);unsigned digit=255;
        if(custom)digit=decode[c];
        else if(c>='0'&&c<='9')digit=c-'0';
        else if(r.base<=36){if(c>='a'&&c<='z')digit=10+c-'a';else if(c>='A'&&c<='Z'){digit=10+c-'A';if(digit<r.base)p.fold=true;}}
        else{if(c>='A'&&c<='Z')digit=10+c-'A';else if(c>='a'&&c<='z')digit=36+c-'a';else if(c=='+')digit=62;else if(c=='/')digit=63;}
        if(digit>=r.base)break;
    }
    if(end==start)return p;p.valid=true;p.consumed=end;p.text=s.substr(0,end);return p;
}
struct Parse {requirements need;Prefix prefix;size_t work_at=0;};
Parse quote_parse(integer_ref out,std::string_view text,radix r,unsigned workers)noexcept{
    Parse p;p.prefix=prefix(text,r);if(!p.prefix.valid){p.need.status=support::unsupported;p.need.reason=p.prefix.why;return p;}
    p.need=parse_requirements(out,p.prefix.text,r,workers);
    if(p.prefix.fold){p.work_at=aligned(p.prefix.text.size(),p.need.scratch_alignment);p.need.scratch_bytes+=p.work_at;}
    return p;
}
std::from_chars_result read(const Parse&p,const char*first,integer_ref out,radix r,execution ex)noexcept{
    if(!p.prefix.valid)return {first,std::errc::invalid_argument};
    if(!p.need.supported())return {first+p.prefix.consumed,std::errc::result_out_of_range};
    detail::require(ex.scratch.size()>=p.need.scratch_bytes&&(!p.need.scratch_bytes||!(uintptr_t(ex.scratch.data())&(p.need.scratch_alignment-1))),SBN3_FATAL_WORKSPACE,"from_chars workspace",p.need.scratch_bytes,ex.scratch.size());
    auto text=p.prefix.text;auto work=ex;
    detail::require(!overlaps(text.data(),text.size(),out.view().data(),out.capacity()*8)&&!overlaps(text.data(),text.size(),ex.scratch.data(),p.need.scratch_bytes)&&!overlaps(out.view().data(),out.capacity()*8,ex.scratch.data(),p.need.scratch_bytes),SBN3_FATAL_ARGUMENT,"from_chars overlap");
    if(p.prefix.fold){auto*folded=reinterpret_cast<char*>(ex.scratch.data());for(size_t i=0;i<text.size();++i){const auto c=text[i];folded[i]=c>='A'&&c<='Z'?char(c-'A'+'a'):c;}text={folded,text.size()};work.scratch=work.scratch.subspan(p.work_at);}
    const auto error=parse_integer_into(out,text,r,work);detail::require(!error,SBN3_FATAL_MATH,"validated character prefix changed");return {first+p.prefix.consumed,{}};
}
}
requirements to_chars_requirements(std::span<char>out,integer_view a,radix r,unsigned workers)noexcept{return quote_format(out,a,r,workers).need;}
std::to_chars_result to_chars(char*first,char*last,integer_view a,radix r,execution ex)noexcept{
    const std::span<char>out{first,length(first,last)};return write(quote_format(out,a,r,ex.workers()),out,a,r,ex);
}
std::to_chars_result to_chars(char*first,char*last,const integer&a,radix r)noexcept{
    const std::span<char>out{first,length(first,last)};const auto p=quote_format(out,a.view(),r,1);
    if(!p.need.supported())return {first,std::errc::invalid_argument};workspace w(a.resource());const auto err=w.try_reserve(p.need.scratch_bytes,p.need.scratch_alignment);
    if(err)return {first,std::errc::not_enough_memory};return write(p,out,a.view(),r,w.on());
}
requirements from_chars_requirements(integer_ref out,std::string_view text,radix r,unsigned workers)noexcept{return quote_parse(out,text,r,workers).need;}
std::from_chars_result from_chars(const char*first,const char*last,integer_ref out,radix r,execution ex)noexcept{
    const auto p=quote_parse(out,{first,length(first,last)},r,ex.workers());return read(p,first,out,r,ex);
}
std::from_chars_result from_chars(const char*first,const char*last,integer&out,radix r)noexcept{
    auto p=quote_parse(out.ref(),{first,length(first,last)},r,1);if(!p.prefix.valid)return {first,std::errc::invalid_argument};
    detail::require(!overlaps(first,length(first,last),out.data(),out.capacity()*8),SBN3_FATAL_ARGUMENT,"from_chars input/output overlap");
    if(p.need.status==support::capacity){const auto error=out.try_reserve(p.need.output_limbs);if(error)return {first+p.prefix.consumed,std::errc::not_enough_memory};p=quote_parse(out.ref(),{first,length(first,last)},r,1);}
    if(!p.need.supported())return {first+p.prefix.consumed,std::errc::result_out_of_range};
    workspace w(out.resource());const auto error=w.try_reserve(p.need.scratch_bytes,p.need.scratch_alignment);if(error)return {first+p.prefix.consumed,std::errc::not_enough_memory};return read(p,first,out.ref(),r,w.on());
}
}
