// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/number_theory.hpp"
#include <cassert>
#include <cstdio>
#include <numeric>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
using sbn::integer;
int main(){
    for(int a=-31;a<=31;++a)for(int b=-31;b<=31;++b){
        integer x=a,y=b;auto g=sbn::gcd(x,y);assert(g==std::gcd(a,b));
        auto e=sbn::extended_gcd(x,y);assert(e.gcd==g&&e.x*x+e.y*y==g);
        auto l=sbn::lcm(x,y);assert(l==(a&&b?std::abs(a*b)/std::gcd(a,b):0));
        if(b){const auto inv=sbn::inverse_mod(x,y);assert(bool(inv)==(std::gcd(a,b)==1));if(inv)assert(sbn::mod(x*(*inv),y)==sbn::mod(1,y));}
    }
    for(unsigned n:{3u,17u,257u,1025u,2049u}){
        const auto A=integer(1)<<(64*n),B=(integer(1)<<(64*n-1))-1,common=(integer(1)<<137)+3;
        const auto a=-(A*common),b=B*common;
        assert(sbn::gcd(a,b)==common);auto e=sbn::extended_gcd(a,b);assert(e.gcd==common&&e.x*a+e.y*b==common);
        integer g;g.reserve(std::max(a.size(),b.size()));const auto need=sbn::gcd_requirements(g.ref(),a.view(),b.view());assert(need.supported());
        sbn::workspace w;w.reserve(need.scratch_bytes,need.scratch_alignment);
        allocation_watch_start();sbn::gcd_into(g.ref(),a.view(),b.view(),w.on());assert(!allocation_watch_stop());assert(g==common);
        auto alias=a;const auto aneed=sbn::gcd_requirements(alias.ref(),alias.view(),b.view());assert(aneed.supported());w.reserve(aneed.scratch_bytes,aneed.scratch_alignment);
        allocation_watch_start();sbn::gcd_into(alias.ref(),alias.view(),b.view(),w.on());assert(!allocation_watch_stop());assert(alias==common);
    }
    // Independent Legendre table for a prime, obtained by enumerating squares.
    bool square[97]{};for(int j=1;j<97;++j)square[j*j%97]=true;
    for(int a=-291;a<=291;++a){const int r=(a%97+97)%97,expected=r==0?0:square[r]?1:-1;assert(sbn::jacobi(a,97)==expected);}
    assert(sbn::jacobi(0,1)==1&&sbn::jacobi(-1,3)==-1);
    puts("C++ GCD/Bezout/LCM/modular inverse/Jacobi and allocation-free alias execution PASS");
}
