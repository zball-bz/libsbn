// Original libsbn tests; LGPL-3.0-or-later. A normal exception-enabled caller.
#include "sbn3/io.hpp"
#include "sbn3/format.hpp"
#include <cassert>
#include <cstdio>
#include <sstream>
#include <stdexcept>
struct throwing_buffer:std::streambuf {
    char digit='1';unsigned count=0;
    int_type underflow()override{
        if(count++==400)throw std::runtime_error("read failure");
        setg(&digit,&digit,&digit+1);return digit;
    }
};
int main(){
    sbn::integer value=123;bool rejected=false;
    try{(void)std::vformat("{:Q}",std::make_format_args(value));}catch(const std::format_error&){rejected=true;}assert(rejected);
    rejected=false;int width=-3;try{(void)std::vformat("{:>{}}",std::make_format_args(value,width));}catch(const std::format_error&){rejected=true;}assert(rejected);
    rejected=false;const char*wrong="width";try{(void)std::vformat("{:>{}}",std::make_format_args(value,wrong));}catch(const std::format_error&){rejected=true;}assert(rejected);
    {std::istringstream input("bad");input.exceptions(std::ios::failbit);rejected=false;try{input>>value;}catch(const std::ios_base::failure&){rejected=true;}assert(rejected&&value==123);}
    const auto before=sbn::default_resource().statistics().live_allocations;
    {throwing_buffer buffer;std::istream input(&buffer);input>>value;assert(input.bad()&&value==123);}
    assert(sbn::default_resource().statistics().live_allocations==before);
    {throwing_buffer buffer;std::istream input(&buffer);input.exceptions(std::ios::badbit);rejected=false;
        try{input>>value;}catch(const std::runtime_error&e){rejected=std::string_view(e.what())=="read failure";}
        assert(rejected&&input.bad()&&value==123);}
    assert(sbn::default_resource().statistics().live_allocations==before);
    puts("C++ exception-enabled format/stream errors preserve values and release token resources PASS");
}
