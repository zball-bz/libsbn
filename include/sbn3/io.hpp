// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "sbn3/chars.hpp"
#include <istream>
#include <ostream>
#include <exception>
namespace sbn {
namespace detail {
class stream_token {
    char small_[128];memory_resource*resource_;allocation block_{};std::size_t size_=0;
public:
    explicit stream_token(memory_resource&r)noexcept:resource_(&r){}
    ~stream_token(){resource_->deallocate(block_);}
    char*data()noexcept{return block_.data?static_cast<char*>(block_.data):small_;}
    std::size_t size()const noexcept{return size_;}
    bool append(char c)noexcept{
        const auto cap=block_.data?block_.bytes:sizeof small_;
        if(size_==cap){if(cap>SIZE_MAX/2)return false;allocation next;const auto error=resource_->try_allocate(2*cap,64,next);if(error)return false;
            std::memcpy(next.data,data(),size_);resource_->deallocate(block_);block_=next;}
        data()[size_++]=c;return true;
    }
};
inline unsigned stream_digit(int c)noexcept{
    if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='a'&&c<='f')return unsigned(c-'a'+10);if(c>='A'&&c<='F')return unsigned(c-'A'+10);return 255;
}
}
// Resident convenience IO. Streams retain their normal error/exception
// policy. Arithmetic still uses planned storage; token buffering is an IO
// resource boundary. Large formatting materializes one resident text value.
inline std::ostream&operator<<(std::ostream&out,const integer&value){
    std::ostream::sentry sentry(out);if(!sentry)return out;
    const auto flags=out.flags(),basefield=flags&std::ios_base::basefield;
    const unsigned base=basefield==std::ios_base::hex?16:basefield==std::ios_base::oct?8:10;
    const bool upper=flags&std::ios_base::uppercase;
    auto text=to_string(value,base==16&&upper?radix{"0123456789ABCDEF"}:radix{base});auto digits=text.view();
    char sign=0;if(value.negative()){sign='-';digits.remove_prefix(1);}else if(flags&std::ios_base::showpos)sign='+';
    std::string_view prefix;if((flags&std::ios_base::showbase)&&!value.zero()){if(base==16)prefix=upper?"0X":"0x";else if(base==8)prefix="0";}
    const auto width=out.width(0);const auto count=digits.size()+prefix.size()+unsigned(sign!=0);
    const auto padding=width>0&&std::uintmax_t(width)>count?std::uintmax_t(width)-count:0;
    auto pad=[&]{char block[128];std::memset(block,out.fill(),sizeof block);for(std::uintmax_t n=padding;n;){const auto take=std::min(n,std::uintmax_t(sizeof block));out.write(block,std::streamsize(take));n-=take;}};
    const auto adjustment=flags&std::ios_base::adjustfield;
    if(adjustment!=std::ios_base::left&&adjustment!=std::ios_base::internal)pad();
    if(sign)out.put(sign);if(!prefix.empty())out.write(prefix.data(),std::streamsize(prefix.size()));
    if(adjustment==std::ios_base::internal)pad();out.write(digits.data(),std::streamsize(digits.size()));
    if(adjustment==std::ios_base::left)pad();return out;
}
inline std::ostream&operator<<(std::ostream&out,const integer_text&text){return out<<text.view();}
inline std::istream&operator>>(std::istream&in,integer&value){
    std::istream::sentry sentry(in);if(!sentry)return in;
    using traits=std::char_traits<char>;auto*buffer=in.rdbuf();detail::stream_token token(value.resource());
    auto read=[&](bool advance){
        if(in.bad())return traits::eof();
#if defined(__cpp_exceptions)
        try{return advance?buffer->snextc():buffer->sgetc();}
        catch(...){const auto original=std::current_exception();try{in.setstate(std::ios_base::badbit);}catch(...){std::rethrow_exception(original);}return traits::eof();}
#else
        return advance?buffer->snextc():buffer->sgetc();
#endif
    };
    auto at=[&]{return read(false);};auto consume=[&]{(void)read(true);};auto put=[&](char c){if(token.append(c))return true;in.setstate(std::ios_base::badbit);return false;};
    auto c=at();if(c=='-'||c=='+'){if(!put(char(c)))return in;consume();}
    const auto flags=in.flags()&std::ios_base::basefield;unsigned base=flags==std::ios_base::hex?16:flags==std::ios_base::oct?8:flags==std::ios_base::dec?10:0;
    bool digits=false;c=at();
    if((base==0||base==16)&&c=='0'){
        consume();c=at();if(c=='x'||c=='X'){base=16;consume();}
        else{if(!base)base=8;if(!put('0'))return in;digits=true;}
    }
    if(!base)base=10;
    while((c=at())!=traits::eof()&&detail::stream_digit(c)<base){const char normalized=c>='A'&&c<='F'?char(c-'A'+'a'):char(c);if(!put(normalized))return in;consume();digits=true;}
    if(in.bad())return in;
    std::ios_base::iostate state=std::ios_base::goodbit;if(c==traits::eof())state|=std::ios_base::eofbit;
    if(!digits)state|=std::ios_base::failbit;
    else{
        const auto*first=token.data()+(token.data()[0]=='+'?1:0);const auto parsed=from_chars(first,token.data()+token.size(),value,radix{base});
        if(parsed.ec==std::errc::not_enough_memory)state|=std::ios_base::badbit;
        else if(parsed.ec!=std::errc{})state|=std::ios_base::failbit;
    }
    if(state!=std::ios_base::goodbit)in.setstate(state);return in;
}
}
