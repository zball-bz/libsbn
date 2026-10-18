// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/sbn.hpp"
#include "sbn3/format.hpp"
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;
static void roundtrip(const integer&a,unsigned base){
    const auto expected=sbn::to_string(a,base);std::vector<char>buffer(expected.size()+16,'!');
    const auto need=sbn::to_chars_requirements(buffer,a.view(),base);assert(need.supported());sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    allocation_watch_start();auto r=sbn::to_chars(buffer.data(),buffer.data()+expected.size(),a.view(),base,work.on());assert(!allocation_watch_stop());
    assert(r.ec==std::errc{}&&r.ptr==buffer.data()+expected.size());assert(std::string_view(buffer.data(),expected.size())==expected.view());for(size_t j=expected.size();j<buffer.size();++j)assert(buffer[j]=='!');
    auto short_need=sbn::to_chars_requirements({buffer.data(),expected.size()-1},a.view(),base);work.reserve(short_need.scratch_bytes,short_need.scratch_alignment);std::fill(buffer.begin(),buffer.end(),'!');
    allocation_watch_start();r=sbn::to_chars(buffer.data(),buffer.data()+expected.size()-1,a.view(),base,work.on());assert(!allocation_watch_stop());assert(r.ec==std::errc::value_too_large);for(auto c:buffer)assert(c=='!');
    std::string input(expected.view());input+='!';integer got=731;got.reserve(a.size()+1);auto pn=sbn::from_chars_requirements(got.ref(),input,base);assert(pn.supported());work.reserve(pn.scratch_bytes,pn.scratch_alignment);
    allocation_watch_start();auto parsed=sbn::from_chars(input.data(),input.data()+input.size(),got.ref(),base,work.on());assert(!allocation_watch_stop());assert(parsed.ec==std::errc{}&&parsed.ptr==input.data()+expected.size()&&got==a);
}
int main(){
    for(unsigned base=2;base<=36;++base)for(int n:{0,1,-1,13,-13,31415,-31415,123456789}){
        char expected[256],actual[256];std::fill_n(actual,256,'!');auto ref=std::to_chars(expected,expected+256,n,base);auto r=sbn::to_chars(actual,actual+256,integer(n),base);
        assert(r.ec==std::errc{}&&std::string_view(actual,r.ptr-actual)==std::string_view(expected,ref.ptr-expected)&&*r.ptr=='!');
        std::string text(actual,r.ptr);for(auto&c:text)if(c>='a'&&c<='z')c=char(c-'a'+'A');text+='!';integer value=57;
        auto parsed=sbn::from_chars(text.data(),text.data()+text.size(),value,base);assert(parsed.ec==std::errc{}&&parsed.ptr==text.data()+text.size()-1&&value==n);
    }
    for(size_t n:{1u,2u,7u,9u,33u,257u,1025u})for(unsigned base:{3u,7u,10u,15u,16u,36u,37u,64u})roundtrip(-(integer(1)<<(64*n-1))+17,base);
    for(const std::string text:{"","-","+1"," 2","x"}){integer value=57;auto r=sbn::from_chars(text.data(),text.data()+text.size(),value);assert(r.ec==std::errc::invalid_argument&&r.ptr==text.data()&&value==57);}
    {integer value=0;std::string text="-aBcDeFz";auto r=sbn::from_chars(text.data(),text.data()+text.size(),value,16);assert(r.ec==std::errc{}&&r.ptr==text.data()+7&&value==-0xabcdef);}
    {integer value=0;std::string text="+!";auto r=sbn::from_chars(text.data(),text.data()+text.size(),value,64);assert(r.ec==std::errc{}&&r.ptr==text.data()+1&&value==62);}
    {char out[32];integer a=-1234567;allocation_watch_start();auto r=sbn::to_chars(out,out+32,a);integer b=0;auto p=sbn::from_chars(out,r.ptr,b);assert(!allocation_watch_stop());assert(r.ec==std::errc{}&&p.ec==std::errc{}&&b==a);}
    {std::ostringstream s;s<<std::hex<<std::showbase<<std::internal<<std::setfill('_')<<std::setw(12)<<integer(-255);assert(s.str()=="-0x_______ff");}
    {std::ostringstream s;s<<std::showpos<<integer(0)<<' '<<std::hex<<std::uppercase<<std::showbase<<integer(255);assert(s.str()=="+0 +0XFF");}
    {std::istringstream s("  -0xAbCd +077 123x");s.unsetf(std::ios::basefield);integer a,b,c;s>>a>>b>>c;assert(a==-43981&&b==63&&c==123&&s.peek()=='x');}
    {std::istringstream s("-123456789012345678901234567890");integer a;s>>a;assert(!s.fail()&&s.eof());auto text=sbn::to_string(a);assert(text.view()=="-123456789012345678901234567890");}
    {std::istringstream s("hello");integer a=57;s>>a;assert(s.fail()&&a==57);}
    assert(std::format("{}",integer(-1234))=="-1234");
    assert(std::format("{:+#012x}",integer(255))==std::format("{:+#012x}",255));
    assert(std::format("{:*>15X}",integer(-255))==std::format("{:*>15X}",-255));
    assert(std::format("{:^13}",integer(1234))==std::format("{:^13}",1234));
    assert(std::format("{:#b}",integer(0))=="0b0");
    assert(std::format("{:0{}}",integer(-1234),17)==std::format("{:0{}}",-1234,17));
    assert(std::format("{0:>{1}o}",integer(1234),17)==std::format("{0:>{1}o}",1234,17));
    auto text=sbn::to_string(integer(1)<<257);assert(std::format("{}",text)==text.view());
    puts("C++ chars prefix/tight buffers/case folding/no allocation, stream and std::format adapters PASS");
}
