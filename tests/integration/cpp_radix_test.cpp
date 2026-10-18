// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/radix.hpp"
#include "sbn3/division.hpp"
#include <cassert>
#include <charconv>
#include <cstdio>
#include <string>
#include <vector>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;
static uint64_t rng=7852134;static uint64_t next(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static void roundtrip(const integer&a,sbn::radix radix){
    auto text=sbn::to_string(a,radix);auto value=sbn::parse_integer(text.view(),radix);assert(value&&value.value==a);
    auto copied=text;auto moved=std::move(copied);assert(copied.empty()&&moved.view()==text.view());copied=text;assert(copied.view()==text.view());
    auto format=sbn::format_requirements(a.view(),radix);assert(format.supported());
    std::vector<char> bytes(format.characters);sbn::workspace w;w.reserve(format.scratch_bytes,format.scratch_alignment);
    allocation_watch_start();const auto written=sbn::format_into(bytes,a.view(),radix,w.on());assert(!allocation_watch_stop());assert(written&&written.text==text.view());
    integer dst=731;dst.reserve(a.size()+1);auto parse=sbn::parse_requirements(dst.ref(),written.text,radix);assert(parse.supported());w.reserve(parse.scratch_bytes,parse.scratch_alignment);
    allocation_watch_start();const auto error=sbn::parse_integer_into(dst.ref(),written.text,radix,w.on());assert(!allocation_watch_stop());assert(!error&&dst==a);
    // Validation failure preserves both the number and all its logical limbs.
    std::string bad(text.view());bad.back()='!';dst=731;parse=sbn::parse_requirements(dst.ref(),bad,radix);assert(parse.supported());w.reserve(parse.scratch_bytes,parse.scratch_alignment);
    allocation_watch_start();const auto invalid=sbn::parse_integer_into(dst.ref(),bad,radix,w.on());assert(!allocation_watch_stop());assert(invalid&&invalid.code==sbn::conversion_errc::invalid_digit&&invalid.index==bad.size()-1&&dst==731);
}
int main(){
    for(unsigned base=2;base<=64;++base)for(int value:{0,1,-1,13,-13,31415,-31415}){
        auto str=sbn::to_string(value,base);if(base<=36){char independent[64];auto r=std::to_chars(independent,independent+64,value,base);assert(str.view()==std::string_view(independent,r.ptr-independent));}
        auto parsed=sbn::parse_integer(str.view(),base);assert(parsed&&parsed.value==value);
    }
    for(std::size_t n:{1u,2u,3u,7u,8u,9u,33u,161u,513u,1025u,8193u}){
        std::vector<uint64_t> words(n);for(auto&v:words)v=next();words.back()|=1;
        for(unsigned base:{2u,3u,7u,10u,15u,16u,36u,37u,64u})roundtrip(integer(sbn::integer_view(words,n%2)),base);
    }
    for(auto r:{sbn::radix{"abcdefghij"},sbn::radix{"/0123+"}})roundtrip(-(integer(1)<<513)+37,r);
    auto plus=sbn::parse_integer("+",64);assert(plus&&plus.value==62);
    auto zero=sbn::parse_integer("-000000000000000000000000000000000000000000000");assert(zero&&zero.value.zero()&&!zero.value.negative());
    assert(sbn::parse_integer("+123").value==123);assert(!sbn::parse_integer("")&&!sbn::parse_integer("-")&&!sbn::parse_integer("+")&&!sbn::parse_integer(" 2"));
    assert(!sbn::parse_integer("1",1)&&!sbn::parse_integer("1",65));assert(!sbn::parse_integer("1",sbn::radix{"01-"}));
    assert(!sbn::parse_integer("1",sbn::radix{"001"}));
    // Independent base-7 Horner value for a nontrivial input.
    integer expected=0;const std::string digits="165036261005216502450125";for(char c:digits)expected=expected*7+(c-'0');
    auto got=sbn::parse_integer(digits,7);assert(got&&got.value==expected);
    auto small=integer(-1234567);allocation_watch_start();auto text=sbn::to_string(small);auto back=sbn::parse_integer(text.view());assert(!allocation_watch_stop());assert(back&&back.value==small);
    puts("C++ radix2..64/custom alphabets/signs/transactional parsing/resource text/leaf and tree PASS");
}
