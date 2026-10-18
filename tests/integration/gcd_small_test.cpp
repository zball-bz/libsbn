#include "sbn3/gcd.h"
#include "sbn3/divrem.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <numeric>
#include <vector>
#include <sys/mman.h>
#include <unistd.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
extern "C" void gcd_ifma52_test();
using U=uint64_t;using W=__uint128_t;using Big=std::vector<U>;
static U seed=0xac570314ed620926ULL;
static U random_word(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void trim(Big&a){while(!a.empty()&&!a.back())a.pop_back();}
static int cmp(const Big&a,const Big&b){if(a.size()!=b.size())return a.size()>b.size()?1:-1;
    for(size_t n=a.size();n-->0;)if(a[n]!=b[n])return a[n]>b[n]?1:-1;return 0;}
static void sub(Big&a,const Big&b){U borrow=0;for(size_t i=0;i<a.size();++i){const W v=W(i<b.size()?b[i]:0)+borrow;const U x=a[i];a[i]=x-U(v);borrow=W(x)<v;}assert(!borrow);trim(a);}
static size_t zeros(const Big&a){size_t i=0;while(i<a.size()&&!a[i])++i;return 64*i+unsigned(__builtin_ctzll(a[i]));}
static void shift(Big&a,size_t k){const size_t words=k/64;const unsigned bits=k%64;
    a.erase(a.begin(),a.begin()+words);if(bits)for(size_t i=0;i<a.size();++i)a[i]=(a[i]>>bits)|(i+1<a.size()?a[i+1]<<(64-bits):0);trim(a);}
static Big left(Big a,size_t k){const unsigned bits=k%64;U carry=0;
    if(bits){for(auto &v:a){const U x=v;v=(v<<bits)|carry;carry=x>>(64-bits);}if(carry)a.push_back(carry);}
    if(!a.empty())a.insert(a.begin(),k/64,0);return a;}
// Deliberately different full-width binary algorithm; no production head or
// matrix routine, no GMP dependency. Used for the modest-size random corpus.
static Big reference(Big a,Big b){trim(a);trim(b);if(a.empty())return b;if(b.empty())return a;
    const auto common=std::min(zeros(a),zeros(b));shift(a,zeros(a));shift(b,zeros(b));
    while(cmp(a,b)){if(cmp(a,b)>0)std::swap(a,b);sub(b,a);shift(b,zeros(b));}
    return left(a,common);}
static Big mul_word(const Big&a,U b){Big r(a.size()+1);W c=0;for(size_t i=0;i<a.size();++i){c+=W(a[i])*b;r[i]=U(c);c>>=64;}r[a.size()]=U(c);trim(r);return r;}
static size_t checks=0;
static void check(const Big&a,const Big&b,const Big&want){
    constexpr U guard=0x9c436789aabb52feULL;const auto before_a=a,before_b=b;
    const size_t bytes=sbn3_gcd_small_scratch_bytes(a.size(),b.size());assert(bytes!=SIZE_MAX&&bytes%8==0);
    const bool az=std::all_of(a.begin(),a.end(),[](U v){return !v;}),bz=std::all_of(b.begin(),b.end(),[](U v){return !v;});
    const size_t bound=az&&bz?0:az?b.size():bz?a.size():std::min(a.size(),b.size());
    Big scratch(bytes/8+3,guard),out(bound+2,guard);void*w=bytes?scratch.data()+1:nullptr;
    allocation_watch_start();const size_t used=sbn3_gcd_small(out.data()+1,a.data(),a.size(),b.data(),b.size(),w);assert(!allocation_watch_stop());
    assert(used==want.size()&&(!used||!memcmp(out.data()+1,want.data(),used*8)));assert(a==before_a&&b==before_b);
    assert(std::all_of(out.begin()+1+used,out.end(),[&](U v){return v==guard;}));
    assert(out.front()==guard&&out.back()==guard&&scratch.front()==guard&&scratch[bytes/8+1]==guard&&scratch.back()==guard);++checks;
}
int main(){
    gcd_ifma52_test();
    assert(sbn3_gcd_small_scratch_bytes(1025,2)==SIZE_MAX);
    assert(sbn3_gcd_small_scratch_bytes(1024,1)==0&&sbn3_gcd_small_scratch_bytes(1024,2)==0);
    for(size_t n=1;n<=1024;++n)assert(sbn3_gcd_small_scratch_bytes(n,n)<=20*n+256);
    assert(sbn3_gcd_small(nullptr,nullptr,0,nullptr,0,nullptr)==0);
    for(U a=0;a<256;++a)for(U b=0;b<256;++b)assert(sbn3_gcd_u64(a,b)==std::gcd(a,b));
    // Sweep the actual two-word precision, including the leading-head gate.
    for(unsigned bits=65;bits<=128;++bits)for(unsigned sample=0;sample<8;++sample){
        const unsigned high_bits=bits-64;const U mask=high_bits==64?UINT64_MAX:(U(1)<<high_bits)-1;
        Big a{random_word(),(random_word()&mask)|(U(1)<<(high_bits-1))};
        Big b{random_word(),(random_word()&mask)|(U(1)<<(high_bits-1))};
        check(a,b,reference(a,b));
    }
    for(size_t n=1;n<=64;++n)for(unsigned sample=0;sample<24;++sample){
        Big a(n),b(n);for(U &v:a)v=random_word();for(U&v:b)v=random_word();
        if(sample==0){std::fill(a.begin(),a.end(),0);std::fill(b.begin(),b.end(),0);}
        if(sample==1)std::fill(a.begin(),a.end(),UINT64_MAX);
        if(sample==2)b=a;
        if(sample==3){b=a;U c=1;for(auto &v:b){const U old=v;v+=c;c=c&&v<old;if(!c)break;}}
        if(sample==4){std::fill(a.begin(),a.end(),0);a.back()=U(1)<<(n%64);}
        if(sample==5){a.back()=0;b.back()=0;}
        check(a,b,reference(a,b));
    }
    // Unequal lengths force the remainder-only fallback, including a long
    // quotient with an output buffer holding only the actual small GCD.
    for(size_t n=4;n<=64;++n)for(size_t m:{size_t(3),(n+1)/2}){
        Big a(n),b(m);for(auto&v:a)v=random_word();for(auto&v:b)v=random_word();
        a.back()|=1;b.back()|=1;const auto g=reference(a,b);check(a,b,g);check(b,a,g);
    }
    for(size_t n:{65u,127u,128u,159u,160u,161u,255u,256u,511u,512u,769u,1023u,1024u}){
        for(unsigned sample=0;sample<24;++sample){
            Big g(n-1);for(auto &v:g)v=sample%4==0?UINT64_MAX:sample%4==1?0:random_word();g.back()|=U(1)<<63;
            U x=random_word()>>2,y=random_word()>>2;x|=1;y|=1;while(std::gcd(x,y)!=1)y+=2;
            check(mul_word(g,x),mul_word(g,y),g);
        }
        Big a(n);for(auto&v:a)v=random_word();a.back()|=U(1)<<63;check(a,a,a);check(a,Big(1,0),a);
        for(U d:{U(1),U(2),U(3),U(5),U(17),U(1)<<63,UINT64_MAX,random_word()|1}){
            U r=0;for(size_t j=n;j-->0;)r=U(((W(r)<<64)|a[j])%d);check(a,Big{d},Big{std::gcd(r,d)});}
        for(Big b:{Big{0,1},Big{UINT64_MAX,1},Big{UINT64_MAX,UINT64_MAX},Big{random_word(),random_word()|1}}){
            Big q(n),rem(2),tmp(n+3);sbn3_divrem_basecase(q.data(),rem.data(),a.data(),n,b.data(),2,tmp.data());check(a,b,reference(rem,b));}
    }
    // Large near-equal values must reduce before choosing a packed format.
    // Both orders, odd word differences and two-word powers of two exercise
    // the short-divisor exit without using the production GCD as its oracle.
    for(size_t n:{159u,160u,161u,255u,512u,1024u}){
        Big b(n);for(auto&v:b)v=random_word();b[0]|=1;b.back()=U(1)<<63;
        for(W gap:{W(1),W(3),W(17),W(UINT64_MAX-2),W(1)<<64,W(1)<<95}){
            U expected=1;
            if(!(gap>>64)){
                const U d=U(gap);U r=0;
                for(size_t j=n;j-->0;)r=U(((W(r)<<64)|b[j])%d);
                expected=std::gcd(r,d);
            }
            Big a=b;W carry=gap;
            for(auto&v:a){const W t=W(v)+U(carry);v=U(t);carry=(carry>>64)+(t>>64);}
            assert(!carry);check(a,b,Big{expected});check(b,a,Big{expected});
            a=b;Big delta{U(gap),U(gap>>64)};trim(delta);sub(a,delta);
            check(a,b,Big{expected});check(b,a,Big{expected});
        }
    }
    // Consecutive Fibonacci values drive long runs of quotient one.
    Big a{1},b{1};for(unsigned i=0;i<4096;++i){Big c(std::max(a.size(),b.size())+1);W carry=0;
        for(size_t j=0;j+1<c.size();++j){carry+=(j<a.size()?a[j]:0);carry+=(j<b.size()?b[j]:0);c[j]=U(carry);carry>>=64;}c.back()=U(carry);trim(c);a=b;b=c;if(i%127==0)check(a,b,Big{1});}
    for(size_t n=0;n<=129;++n){Big a(n),b(n),out(n),expected(n);for(size_t i=0;i<n;++i){a[i]=i%9?random_word():0;b[i]=i%7?random_word():0;expected[i]=std::gcd(a[i],b[i]);}
        allocation_watch_start();sbn3_gcd_u64_batch(out.data(),a.data(),b.data(),n);assert(!allocation_watch_stop());assert(out==expected);
        const auto before=a;sbn3_gcd_u64_batch(a.data(),a.data(),b.data(),n);assert(a==expected);a=before;sbn3_gcd_u64_batch(b.data(),a.data(),b.data(),n);assert(b==expected);}
    const size_t page=size_t(sysconf(_SC_PAGESIZE));auto *p=static_cast<U*>(mmap(nullptr,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    assert(p!=MAP_FAILED&&mprotect(reinterpret_cast<char*>(p)+page,page,PROT_NONE)==0);
    for(size_t n=1;n<=17;++n){U*at=p+page/8-n;for(size_t i=0;i<n;++i)at[i]=random_word();Big b(n,63),out(n);sbn3_gcd_u64_batch(out.data(),at,b.data(),n);for(size_t i=0;i<n;++i)assert(out[i]==std::gcd(at[i],b[i]));}
    assert(!munmap(p,2*page));printf("small GCD: %zu array checks, exhaustive byte pairs, batch tails, guards, no allocation PASS\n",checks);
}
