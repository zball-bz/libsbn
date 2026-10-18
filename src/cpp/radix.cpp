// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/radix.hpp"
#include "sbn3/radix.h"
#include "sbn3/decimal.h"
#include "radix/leaf.hpp"
#include "radix/once.hpp"
#include "runtime/resident.hpp"
#include "value/bitwise.hpp"
#include <new>
namespace sbn {
namespace {
using v3::ResidentSession;using v3::overlaps;
std::size_t align(std::size_t n,std::size_t a)noexcept{std::size_t v;detail::require(v3::align_size(n,a,v),SBN3_FATAL_SIZE,"radix C++ layout");return v;}
bool decimal(radix r)noexcept{return r.base==10&&(r.alphabet.empty()||r.alphabet=="0123456789");}
bool options(radix r,unsigned workers,sbn3_radix_options&o)noexcept{
    if(r.base<2||r.base>64||!workers||workers>32||(!r.alphabet.empty()&&r.alphabet.size()!=r.base))return false;
    o={};o.workers=workers;o.use_alphabet=1;
    if(r.alphabet.empty())sbn3_radix_alphabet(r.base,o.alphabet);
    else{bool seen[256]{};for(unsigned j=0;j<r.base;++j){const auto c=static_cast<unsigned char>(r.alphabet[j]);if(c=='-'||seen[c])return false;seen[c]=true;o.alphabet[j]=c;}}
    return true;
}
struct Work {std::size_t bytes=0,at=0,alignment=64;};
Work service_work(std::size_t bytes,std::size_t alignment,std::size_t prefix=0)noexcept{
    Work w;w.alignment=std::max(std::size_t(64),alignment);
    w.at=align(prefix+(bytes?sizeof(ResidentSession):0),w.alignment);w.bytes=w.at+bytes;return w;
}
void check(const requirements&r,execution ex)noexcept{
    detail::require(r.supported(),SBN3_FATAL_ARGUMENT,r.reason);detail::require(!ex.scope,SBN3_FATAL_TEAM,"radix needs idle root Team");
    detail::require(ex.scratch.size()>=r.scratch_bytes&&(!r.scratch_bytes||!(uintptr_t(ex.scratch.data())&(r.scratch_alignment-1))),SBN3_FATAL_WORKSPACE,"radix scratch",r.scratch_bytes,ex.scratch.size());
}
struct Format {
    text_requirements need;sbn3_format_spec spec{};sbn3_radix_options opts{};sbn3_format_info info{};Work work{};bool small=false,local=false;
};
Format format_plan(integer_view a,radix r,unsigned workers)noexcept{
    Format p;p.need.scratch_alignment=64;
    // Standard decimal leaves need neither alphabet preparation nor a plan.
    if(decimal(r)&&workers&&workers<=32&&a.size()<=8){p.small=true;p.need.characters=std::max(std::size_t(1),20*a.size())+unsigned(a.negative())+1;return p;}
    if(!options(r,workers,p.opts)){p.need.status=support::unsupported;p.need.reason="radix alphabet/workers";return p;}
    if(a.zero()){p.small=true;p.need.characters=2;return p;}
    p.spec={r.base,a.size(),0,0,SBN3_RADIX_EXACT};sbn3_query_result status;
    p.local=v3::radix::format_once_local_query(p.spec,&p.opts,p.info,status);
    if(!p.local)status=sbn3_format_once_query(&p.spec,&p.opts,&p.info);
    if(status!=SBN3_SUPPORTED){p.need.status=status==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported;p.need.reason="format size";return p;}
    if(p.local){p.work.bytes=p.info.storage_bytes;p.work.alignment=p.info.storage_alignment;}
    else p.work=service_work(p.info.storage_bytes,p.info.storage_alignment);
    p.need.scratch_bytes=p.work.bytes;p.need.scratch_alignment=p.work.alignment;p.need.characters=p.info.digit_bytes+2;return p;
}
format_result format_run(const Format&p,std::span<char>out,integer_view a,radix r,execution ex)noexcept{
    if(!p.need.supported())return {{},{conversion_errc::unsupported,0}};
    if(out.size()<p.need.characters)return {{},{conversion_errc::capacity,0}};
    check(p.need,ex);
    detail::require(!overlaps(out.data(),out.size(),a.data(),a.size()*8)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,a.data(),a.size()*8)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,out.data(),out.size()),SBN3_FATAL_ARGUMENT,"format spans overlap");
    if(p.small){char*cur=out.data();if(a.negative())*cur++='-';
        if(a.zero())*cur++=r.alphabet.empty()?'0':r.alphabet[0];
        else if(a.size()==1)cur=sbn3_decimal_write_u64(cur,a.data()[0]);
        else if(a.size()==2)cur=sbn3_decimal_write_u128(cur,{a.data()[0],a.data()[1]});
        else cur=sbn3_decimal_write_small(cur,a.data(),a.size());
        *cur=0;return {{out.data(),std::size_t(cur-out.data())},{}};
    }
    sbn3_format_once_result result{};auto*digits=reinterpret_cast<unsigned char*>(out.data()+1);
    if(p.local)detail::require(v3::radix::format_once_local_raw(p.spec,&p.opts,a.data(),digits,ex.scratch.data(),p.info.storage_bytes,&result),SBN3_FATAL_MATH,"format local replay");
    else if(!p.info.storage_bytes)sbn3_format_once(&p.spec,&p.opts,a.data(),digits,nullptr,&result);
    else{
        auto*s=::new(ex.scratch.data())ResidentSession(ex.scratch.subspan(p.work.at,p.info.storage_bytes),ex.team);
        const sbn3_radix_workspace w{s->arena(),s->team(),0,p.info.storage_bytes};
        sbn3_format_once(&p.spec,&p.opts,a.data(),digits,&w,&result);s->~ResidentSession();
    }
    const auto end=std::size_t(result.integer_digits);auto first=std::size_t(result.integer_first);
    if(first==end){detail::require(end,SBN3_FATAL_MATH,"integer format zero width");--first;digits[first]=p.opts.alphabet[0];}
    char*begin=out.data()+1+first;if(a.negative())*--begin='-';out[1+end]=0;
    return {{begin,std::size_t(out.data()+1+end-begin)},{}};
}
struct Parse {
    requirements need;conversion_error error{};std::string_view digits{};bool negative=false,small=false,stage=false,local=false;
    std::size_t prefix=0,sessionat=0;Work work{};sbn3_parse_spec spec{};sbn3_radix_options opts{};sbn3_parse_info info{};
};
void parse_place(Parse&p,integer_ref*dest)noexcept{
    if(p.error)return;
    if(dest&&dest->capacity()<p.info.limbs){p.need.status=support::capacity;p.need.reason="parse destination capacity";p.error={conversion_errc::capacity,0};return;}
    if(p.small){p.need.scratch_alignment=8;return;}
    p.stage=dest&&(uintptr_t(dest->view().data())&63);p.sessionat=align(p.stage?8*p.info.limbs:0,alignof(ResidentSession));
    if(p.local){p.work.alignment=64;p.work.at=align(p.sessionat,64);p.work.bytes=p.work.at+p.info.storage_bytes;}
    else p.work=service_work(p.info.storage_bytes,p.info.storage_alignment,p.sessionat);
    p.need.scratch_bytes=p.work.bytes;p.need.scratch_alignment=p.work.alignment;
}
Parse parse_plan(integer_ref*dest,std::string_view text,radix r,unsigned workers)noexcept{
    Parse p;
    if(!options(r,workers,p.opts)){p.error={conversion_errc::invalid_radix,0};p.need.status=support::unsupported;p.need.reason="radix alphabet/workers";return p;}
    if(text.empty()){p.error={conversion_errc::empty,0};p.need.status=support::unsupported;p.need.reason="empty integer text";return p;}
    const bool plus_digit=std::memchr(p.opts.alphabet,'+',r.base);
    if(text[0]=='-'||(text[0]=='+'&&!plus_digit)){p.negative=text[0]=='-';p.prefix=1;text.remove_prefix(1);}
    if(text.empty()){p.error={conversion_errc::empty,p.prefix};p.need.status=support::unsupported;p.need.reason="sign without digits";return p;}
    // Leading zero digits do not increase the required binary capacity.
    std::size_t zeros=0;while(zeros+1<text.size()&&static_cast<unsigned char>(text[zeros])==p.opts.alphabet[0])++zeros;
    p.prefix+=zeros;text.remove_prefix(zeros);p.digits=text;p.spec={r.base,text.size(),0,0};
    sbn3_query_result status;p.local=v3::radix::parse_once_local_query(p.spec,&p.opts,p.info,status);
    if(!p.local)status=sbn3_parse_once_query(&p.spec,&p.opts,&p.info);
    if(status!=SBN3_SUPPORTED){p.need.status=status==SBN3_QUERY_CAPACITY?support::capacity:support::unsupported;p.need.reason="parse size";p.error={conversion_errc::unsupported,0};return p;}
    p.need.output_limbs=p.info.limbs;
    p.small=decimal(r)&&p.info.limbs<=8;
    // Only unaligned destinations need an output-sized temporary. Syntax is
    // validated before writing, so transactional parsing needs no full copy.
    parse_place(p,dest);return p;
}
conversion_error parse_run(const Parse&p,integer_ref out,execution ex)noexcept{
    if(p.error)return p.error;check(p.need,ex);
    const auto inputbytes=p.digits.size();
    detail::require(!overlaps(p.digits.data(),inputbytes,out.view().data(),out.capacity()*8)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,p.digits.data(),inputbytes)&&!overlaps(ex.scratch.data(),p.need.scratch_bytes,out.view().data(),out.capacity()*8),SBN3_FATAL_ARGUMENT,"parse spans overlap");
    if(p.small){
        auto*dst=out.begin_write();sbn3_decimal_status rc;
        if(p.info.limbs==1)rc=sbn3_decimal_read_u64(dst->data,p.digits.data(),inputbytes);
        else if(p.info.limbs==2){sbn3_uint128 v;rc=sbn3_decimal_read_u128(&v,p.digits.data(),inputbytes);if(rc==SBN3_DECIMAL_OK){dst->data[0]=v.lo;dst->data[1]=v.hi;}}
        else rc=sbn3_decimal_read_small(dst->data,p.info.limbs,p.digits.data(),inputbytes);
        if(rc==SBN3_DECIMAL_INVALID){std::size_t i=0;while(i<inputbytes&&p.digits[i]>='0'&&p.digits[i]<='9')++i;return {conversion_errc::invalid_digit,p.prefix+i};}
        detail::require(rc==SBN3_DECIMAL_OK,SBN3_FATAL_MATH,"decimal parse size quote");
        dst->size=v3::bits::normalized_size(dst->data,p.info.limbs);dst->negative=dst->size&&p.negative;out.finish_write();return {};
    }
    // A SIMD validation pass before mutation preserves the caller's value on
    // malformed input without allocating a second large integer.
    uint8_t decode[256];std::memset(decode,0xff,sizeof decode);for(unsigned j=0;j<p.spec.base;++j)decode[p.opts.alphabet[j]]=j;
    const auto*digits=reinterpret_cast<const uint8_t*>(p.digits.data());const auto invalid=v3::radix::validate_digits(digits,inputbytes,p.spec.base,decode);
    if(invalid!=inputbytes)return {conversion_errc::invalid_digit,p.prefix+invalid};
    auto*dst=out.begin_write();auto*value=p.stage?reinterpret_cast<uint64_t*>(ex.scratch.data()):dst->data;
    sbn3_parse_once_result result{};
    if(p.local)detail::require(v3::radix::parse_once_local_raw(p.spec,&p.opts,digits,value,p.info.storage_bytes?ex.scratch.data()+p.work.at:nullptr,p.info.storage_bytes,&result),SBN3_FATAL_MATH,"parse local replay");
    else if(!p.info.storage_bytes)sbn3_parse_once(&p.spec,&p.opts,digits,value,nullptr,&result);
    else{
        auto*s=::new(ex.scratch.data()+p.sessionat)ResidentSession(ex.scratch.subspan(p.work.at,p.info.storage_bytes),ex.team);
        const sbn3_radix_workspace w{s->arena(),s->team(),0,p.info.storage_bytes};
        sbn3_parse_once(&p.spec,&p.opts,digits,value,&w,&result);s->~ResidentSession();
    }
    detail::require(result.valid,SBN3_FATAL_MATH,"validated radix digits changed during parse");
    const auto n=v3::bits::normalized_size(value,result.limbs);if(p.stage&&n)std::memcpy(dst->data,value,8*n);dst->size=n;dst->negative=n&&p.negative;out.finish_write();return {};
}
}
void integer_text::reserve(std::size_t n)noexcept{
    if(n<=(block_.data?block_.bytes:sizeof small_))return;
    auto fresh=resource_->allocate(n,64);if(size_)std::memcpy(fresh.data,data(),size_);static_cast<char*>(fresh.data)[size_]=0;
    resource_->deallocate(block_);block_=fresh;offset_=0;
}
integer_text::integer_text(const integer_text&x)noexcept:integer_text(*x.resource_){reserve(x.size_+1);std::memcpy(storage(),x.data(),x.size_+1);size_=x.size_;}
integer_text::integer_text(integer_text&&x)noexcept:resource_(x.resource_),block_(x.block_),offset_(x.offset_),size_(x.size_){
    if(!block_.data)std::memcpy(small_,x.small_,sizeof small_);x.block_={};x.offset_=x.size_=0;x.small_[0]=0;
}
integer_text&integer_text::operator=(const integer_text&x)noexcept{if(this!=&x){reserve(x.size_+1);std::memmove(storage(),x.data(),x.size_+1);offset_=0;size_=x.size_;}return *this;}
integer_text&integer_text::operator=(integer_text&&x)noexcept{
    if(this==&x)return *this;
    if(resource_!=x.resource_){*this=x;x.size_=x.offset_=0;x.storage()[0]=0;return *this;}
    resource_->deallocate(block_);block_=x.block_;offset_=x.offset_;size_=x.size_;if(!block_.data)std::memcpy(small_,x.small_,sizeof small_);
    x.block_={};x.size_=x.offset_=0;x.small_[0]=0;return *this;
}
text_requirements format_requirements(integer_view a,radix r,unsigned workers)noexcept{return format_plan(a,r,workers).need;}
format_result format_into(std::span<char>out,integer_view a,radix r,execution ex)noexcept{return format_run(format_plan(a,r,ex.workers()),out,a,r,ex);}
integer_text to_string(const integer&a,radix r)noexcept{
    const auto p=format_plan(a.view(),r,1);detail::require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    integer_text out(a.resource());out.reserve(p.need.characters);workspace w(a.resource());w.reserve(p.need.scratch_bytes,p.need.scratch_alignment);
    const auto result=format_run(p,{out.storage(),p.need.characters},a.view(),r,w.on());detail::require(bool(result),SBN3_FATAL_MATH,"format admitted output");
    out.offset_=result.text.data()-out.storage();out.size_=result.text.size();return out;
}
requirements parse_requirements(integer_ref out,std::string_view text,radix r,unsigned workers)noexcept{return parse_plan(&out,text,r,workers).need;}
conversion_error parse_integer_into(integer_ref out,std::string_view text,radix r,execution ex)noexcept{return parse_run(parse_plan(&out,text,r,ex.workers()),out,ex);}
integer_parse_result parse_integer(std::string_view text,radix r,memory_resource&resource)noexcept{
    auto p=parse_plan(nullptr,text,r,1);integer_parse_result result{integer(resource),p.error};if(result.error)return result;
    result.value.reserve(p.need.output_limbs);auto out=result.value.ref();parse_place(p,&out);
    workspace w(resource);w.reserve(p.need.scratch_bytes,p.need.scratch_alignment);result.error=parse_run(p,out,w.on());return result;
}
}
