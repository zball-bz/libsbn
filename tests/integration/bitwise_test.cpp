// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/bitwise.h"
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>
using U=uint64_t;using Big=std::vector<U>;
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
static U rng=0xa2345f616acfa24bULL;static U random_word(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static void trim(Big&a){while(!a.empty()&&!a.back())a.pop_back();}
static Big twos(Big a,bool negative,size_t n){a.resize(n);if(negative){for(auto&v:a)v=~v;for(auto&v:a)if(++v)break;}return a;}
struct Value {Big a;bool negative=false;};
static Value magnitude(Big a){bool neg=a.back()>>63;if(neg){for(auto&v:a)v=~v;for(auto&v:a)if(++v)break;}trim(a);return {a,!a.empty()&&neg};}
static Value expected(Big a,bool sa,Big b,bool sb,unsigned op){size_t n=std::max(a.size(),b.size())+2;a=twos(a,sa,n);b=twos(b,sb,n);for(size_t i=0;i<n;++i)a[i]=op==0?a[i]&b[i]:op==1?a[i]|b[i]:a[i]^b[i];return magnitude(a);}
static void equal(const sbn3_int&r,const Value&w){assert(r.size==w.a.size()&&bool(r.negative)==w.negative);assert(std::equal(w.a.begin(),w.a.end(),r.data));}
static size_t checks=0;
static void logic(const Big&a,const Big&b,unsigned sa,unsigned sb){
    const size_t cap=std::max(a.size(),b.size())+2;
    for(unsigned op=0;op<3;++op){const auto want=expected(a,sa,b,sb,op);
        for(unsigned alias=0;alias<3;++alias){Big aa=a,bb=b,rr(cap+1,0x137acebdeULL);aa.resize(cap+1,0x137acebdeULL);bb.resize(cap+1,0x137acebdeULL);
            U*p=alias==1?aa.data():alias==2?bb.data():rr.data();sbn3_int r{p,cap,0,0};
            allocation_watch_start();sbn3_int_bitwise(&r,sbn3_bit_operation(op),{aa.data(),a.size(),sa},{bb.data(),b.size(),sb});assert(!allocation_watch_stop());
            equal(r,want);assert(p[cap]==0x137acebdeULL);++checks;}
    }
}
static void unary(Big a,unsigned sign){
    Big fixed=twos(a,sign,a.size()+4);Big inverse=fixed;for(auto&v:inverse)v=~v;
    const Value inv=magnitude(inverse);
    const size_t physical=fixed.size()*64;
    for(unsigned alias=0;alias<2;++alias){Big aa=a,rr(a.size()+3);aa.resize(a.size()+3);sbn3_int r{alias?aa.data():rr.data(),a.size()+2,0,0};
        allocation_watch_start();sbn3_int_complement(&r,{aa.data(),a.size(),sign});assert(!allocation_watch_stop());equal(r,inv);++checks;
    }
    auto valuebit=[&](size_t j){return j<physical?bool((fixed[j/64]>>(j%64))&1):bool(sign&&std::any_of(a.begin(),a.end(),[](U x){return x!=0;}));};
    for(size_t bit: {size_t(0),size_t(1),size_t(31),size_t(63),size_t(64),size_t(65),size_t(127),a.size()*32,a.size()*64,a.size()*64+63}){
        allocation_watch_start();int got=sbn3_int_test_bit({a.data(),a.size(),sign},bit);assert(!allocation_watch_stop());assert(got==valuebit(bit));
        for(unsigned which=0;which<2;++which){size_t want=bit;while(want<physical&&valuebit(want)!=bool(which))++want;if(want>=physical&&valuebit(want)!=bool(which))want=SIZE_MAX;
            assert(sbn3_int_scan_bit({a.data(),a.size(),sign},bit,which)==want);}
        for(size_t count:{size_t(0),size_t(1),size_t(63),size_t(129)}){size_t want=0;for(size_t j=0;j<count;++j)want+=valuebit(bit+j);assert(sbn3_int_popcount_range({a.data(),a.size(),sign},bit,count)==want);}
        for(unsigned op=0;op<3;++op){auto t=fixed;const U mask=U(1)<<(bit%64);if(op==0)t[bit/64]&=~mask;else if(op==1)t[bit/64]|=mask;else t[bit/64]^=mask;const auto want=magnitude(t);
            for(unsigned alias=0;alias<2;++alias){auto aa=a;aa.resize(fixed.size()+1);Big out(fixed.size()+1);sbn3_int r{alias?aa.data():out.data(),fixed.size(),0,0};
                allocation_watch_start();sbn3_int_edit_bit(&r,{aa.data(),a.size(),sign},bit,sbn3_bit_edit(op));assert(!allocation_watch_stop());equal(r,want);++checks;}}
        // Independent finite two's-complement arithmetic shift with explicit sign fill.
        auto t=fixed;for(size_t j=0;j<physical;++j){const bool v=valuebit(j+bit);U &w=t[j/64];const U m=U(1)<<(j%64);w=v?w|m:w&~m;}const auto want=magnitude(t);
        for(unsigned alias=0;alias<2;++alias){auto aa=a;aa.resize(a.size()+2);Big out(a.size()+2);sbn3_int r{alias?aa.data():out.data(),a.size()+1,0,0};
            allocation_watch_start();sbn3_int_arshift(&r,{aa.data(),a.size(),sign},bit);assert(!allocation_watch_stop());equal(r,want);++checks;}
    }
    Big mag=a;trim(mag);size_t population=0;for(U v:mag)for(unsigned j=0;j<64;++j)population+=(v>>j)&1;
    assert(sbn3_int_popcount_magnitude({a.data(),a.size(),sign})==population);
    assert(sbn3_int_bit_length({a.data(),a.size(),sign})==(mag.empty()?0:64*mag.size()-__builtin_clzll(mag.back())));
    const bool neg=sign&&!mag.empty();assert(sbn3_int_test_bit({a.data(),a.size(),sign},SIZE_MAX)==neg);
    assert(sbn3_int_popcount_range({a.data(),a.size(),sign},SIZE_MAX-17,100)==(neg?100u:0u));
}
struct Guard {
    void *base;size_t bytes;U*p;
    explicit Guard(size_t n){size_t page=size_t(sysconf(_SC_PAGESIZE));bytes=((n*8+page-1)/page+1)*page;base=mmap(nullptr,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);assert(base!=MAP_FAILED);assert(!mprotect(static_cast<char*>(base)+bytes-page,page,PROT_NONE));p=reinterpret_cast<U*>(static_cast<char*>(base)+bytes-page-n*8);}
    ~Guard(){assert(!munmap(base,bytes));}
};
int main(){
    for(int a=-80;a<=80;++a)for(int b=-80;b<=80;++b)for(unsigned op=0;op<3;++op){U am=a<0?-a:a,bm=b<0?-b:b,out[2]{};sbn3_int r{out,2,0,0};sbn3_int_bitwise(&r,sbn3_bit_operation(op),{&am,1,unsigned(a<0)},{&bm,1,unsigned(b<0)});int v=op==0?a&b:op==1?a|b:a^b;assert(r.size==size_t(v!=0)&&bool(r.negative)==(v<0)&&(!v||r.data[0]==U(v<0?-v:v)));++checks;}
    for(size_t n:{0u,1u,2u,3u,4u,7u,8u,9u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u,128u,129u,255u,256u,257u,1025u}){
        for(unsigned pattern=0;pattern<6;++pattern){Big a(n),b(n?pattern%2?n/3+1:n:0);
            for(auto&v:a)v=random_word();for(auto&v:b)v=random_word();
            if(pattern==1){std::fill(a.begin(),a.end(),0);if(n)a.back()=1;}
            if(pattern==2){std::fill(a.begin(),a.end(),UINT64_MAX);std::fill(b.begin(),b.end(),UINT64_MAX-1);}
            if(pattern==3)std::fill(b.begin(),b.end(),0);
            if(pattern==4){if(n)a.back()=0;if(b.size())b.back()=0;}
            if(pattern==5){b=a;}
            for(unsigned sa=0;sa<2;++sa)for(unsigned sb=0;sb<2;++sb){logic(a,b,sa,sb);
                auto aa=twos(a,sa,n+4),bb=twos(b,sb,n+4);const size_t end=(n+4)*64;
                for(size_t first:{size_t(0),size_t(3),size_t(63),n*32,n*64}){size_t count=std::min(size_t(137),end-first),want=0;for(size_t j=0;j<count;++j)want+=((aa[(first+j)/64]^bb[(first+j)/64])>>((first+j)%64))&1;
                    allocation_watch_start();size_t got=sbn3_int_hamdist_range({a.data(),a.size(),sa},{b.data(),b.size(),sb},first,count);assert(!allocation_watch_stop());assert(got==want);}
            }
            if(n<=129)for(unsigned sa=0;sa<2;++sa)unary(a,sa);
        }
    }
    for(size_t n:{1u,2u,7u,8u,9u,31u,32u,33u,511u,512u,513u})for(unsigned sign=0;sign<4;++sign){
        Guard a(n),b(n),r(n+1);Big aa(n),bb(n);for(size_t j=0;j<n;++j){a.p[j]=aa[j]=random_word();b.p[j]=bb[j]=random_word();}
        for(unsigned op=0;op<3;++op){sbn3_int out{r.p,n+1,0,0};sbn3_int_bitwise(&out,sbn3_bit_operation(op),{a.p,n,sign&1},{b.p,n,sign>>1});equal(out,expected(aa,sign&1,bb,sign>>1,op));}
    }
    U a[3]{~U(0),~U(0),0};sbn3_int out{a,3,2,1};sbn3_int_arshift(&out,{a,2,1},64);assert(out.size==2&&out.negative&&a[0]==0&&a[1]==1);
    sbn3_int zero{nullptr,0,0,0};sbn3_int_and(&zero,{nullptr,0,1},{nullptr,0,1});assert(!zero.size&&!zero.negative);
    assert(sbn3_int_scan_bit({nullptr,0,0},SIZE_MAX,0)==SIZE_MAX);
    puts("bitwise signs/aliases/prefix carries/ranges/scans/edits/arshift and guard pages PASS");printf("%zu operation checks; allocation-free gates PASS\n",checks);
}
