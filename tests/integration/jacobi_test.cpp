// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/jacobi.h"
#include "sbn3/gcd.h"
#include "gcd/jacobi_state.hpp"
#include <vector>
#include <cassert>
#include <cstdio>
#include <algorithm>
using U=uint64_t;using Big=std::vector<U>;
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
static U seed=0x9187317a;static U rnd(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
static void trim(Big&a){while(a.size()&&!a.back())a.pop_back();}
static int cmp(const Big&a,const Big&b){if(a.size()!=b.size())return a.size()>b.size()?1:-1;for(size_t i=a.size();i-->0;)if(a[i]!=b[i])return a[i]>b[i]?1:-1;return 0;}
static unsigned shift(Big&a){size_t j=0;while(!a[j])++j;a.erase(a.begin(),a.begin()+j);const unsigned s=__builtin_ctzll(a[0]);
    if(s)for(size_t k=0;k<a.size();++k)a[k]=(a[k]>>s)|(k+1<a.size()?a[k+1]<<(64-s):0);trim(a);return s&1;}
static void sub(Big&a,const Big&b){U c=0;for(size_t j=0;j<a.size();++j){const __uint128_t v=__uint128_t(j<b.size()?b[j]:0)+c;const U old=a[j];a[j]-=U(v);c=__uint128_t(old)<v;}assert(!c);trim(a);}
// Independent binary Jacobi; never calls production divrem/head/state tables.
static int reference(Big a,Big b){trim(a);trim(b);int s=1;assert(b.size()&&(b[0]&1));
    while(a.size()){
        unsigned t=shift(a);if(t&&((b[0]&7)==3||(b[0]&7)==5))s=-s;
        if(cmp(a,b)<0){if((a[0]&3)==3&&(b[0]&3)==3)s=-s;std::swap(a,b);}
        sub(a,b);
    }return b.size()==1&&b[0]==1?s:0;
}
static size_t count=0;
static void check(Big a,Big b,int expected=2,sbn3_team*team=nullptr){
    if(expected==2)expected=reference(a,b);const auto aa=a,bb=b;
    const unsigned w=team?sbn3_team_workers(team):1;const sbn3_jacobi_options o{w};sbn3_jacobi_info i{};
    assert(sbn3_jacobi_query(a.size(),b.size(),&o,&i)==SBN3_SUPPORTED);
    assert(i.scratch_bytes==sbn3_gcd_scratch_bytes(a.size(),b.size(),w));
    Big memory(i.scratch_bytes/8+24,0xa167390eca81390eULL);
    void*p=reinterpret_cast<void*>((uintptr_t(memory.data()+8)+63)&~uintptr_t(63));
    for(unsigned sign=0;sign<2;++sign){allocation_watch_start();
        int got=sbn3_int_jacobi({a.data(),a.size(),sign},{b.data(),b.size(),0},p,i.scratch_bytes,team);
        assert(!allocation_watch_stop());const int want=sign&&(b[0]&2)?-expected:expected;
        if(got!=want){fprintf(stderr,"Jacobi mismatch n=%zu/%zu sign=%u got=%d expected=%d case=%zu\n",a.size(),b.size(),sign,got,want,count);abort();}
        assert(a==aa&&b==bb);++count;
    }
    for(auto*q=memory.data();q<static_cast<U*>(p);++q)assert(*q==0xa167390eca81390eULL);
    for(auto*q=static_cast<U*>(p)+i.scratch_bytes/8;q<memory.data()+memory.size();++q)assert(*q==0xa167390eca81390eULL);
}
int main(){
    for(U a=0;a<256;++a)for(U b=1;b<256;b+=2)assert(sbn3_jacobi_u64(a,b)==reference({a},{b}));
    for(unsigned j=0;j<2000;++j){U a=rnd(),b=rnd()|1;assert(sbn3_jacobi_u64(a,b)==reference({a},{b}));}
    // All valid low-bit states/directions/quotient residues, including
    // partial rather than maximal quotients and both even orientations.
    for(U a=1;a<48;++a)for(U b=1;b<48;b+=2){
        auto s=sbn::v3::gcd::JacobiState::initial(a,b);U x=a,y=b;
        const int want=reference({a},{b});
        while(x!=y){unsigned d=x>y;U q=((d?x:y)-1)/(d?y:x);if(q>1&&rnd()&1)q=1;s(d,q);if(d)x-=q*y;else y-=q*x;
            assert(((s.bits>>1)&3)==(x&3)&&((s.bits>>3)&3)==(y&3));
            assert(s.sign()*(s.numerator_is_a()?reference({x},{y}):reference({y},{x}))==want);}
    }
    for(size_t n:{1,2,3,4,8,15,16,31,32,47,48,49,95,96,97,159,160,161,383,768,1024,1025,1537}){
        for(unsigned k=0;k<6;++k){Big a(n),b(k==1?1:k==2?2:k==3?std::max<size_t>(1,n/3):n);
            for(auto&v:a)v=rnd();for(auto&v:b)v=rnd();a.back()|=U(1)<<63;b.back()|=U(1)<<63;b[0]|=1;
            if(k==4){a=b;a[0]&=~U(1);}if(k==5){std::fill(a.begin(),a.end(),0);a.back()=1;}
            check(a,b);if(k<4){a[0]|=1;check(b,a);}
        }
    }
    check({}, {1});check({}, {13});check({0,0},{1,0});check({1,0,0},{17,0,0});check({41},{193});
    // Non-coprime endpoints, leading-zero declared lengths, and equality.
    for(size_t n:{3,161,1025,4097}){Big b(n,UINT64_MAX),a=b;check(a,b,0);a[0]-=1;check(a,b,-1);}
    // Existing caller team; initial reductions and streamed HGCD leaves.
    sbn3_arena*arena=nullptr;sbn3_arena_config ac{size_t(1)<<30,8u<<20};sbn3_error err{};
    assert(sbn3_arena_create(&ac,&arena,&err)==SBN3_OK);
    assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&err)==SBN3_OK);
    sbn3_lease control{};sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
    sbn3_team_config cfg{};cfg.workers=4;cfg.stack_offset=65536;for(int&c:cfg.cpu_ids)c=-1;
    sbn3_team*team=nullptr;assert(sbn3_team_create(arena,&control,&cfg,&team,&err)==SBN3_OK);
    Big a(4097),b(4097);for(auto&v:a)v=rnd();for(auto&v:b)v=rnd();b[0]|=1;
    check(a,b,reference(a,b),team);
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);
    printf("Jacobi: %zu signed calls plus word/state exhaustive gates passed\n",count);
}
