#pragma once
#include "value/limbs.hpp"
#include "sbn3/team.h"
#include "tuning/native_policy.hpp"
#include <algorithm>
namespace sbn::v3::parallel_limbs {
// Idle, pre-existing team; stack-only summaries. Small spans retain the
// original word kernels. No allocation, plan query, or nested team episode.
// Zen5: 2 MiB spans still fit the shared cache and pay more for coordination
// than they save. Enable value passes only from 8 MiB (2^20 words).
inline constexpr size_t minimum_parallel_words=native_policy::value_parallel_words;
inline unsigned parts(sbn3_team *team,size_t n) noexcept {
    return team && n>=minimum_parallel_words?sbn3_team_workers(team):1;
}
inline size_t cut(size_t n,unsigned p,unsigned j) noexcept {return n/p*j+std::min(size_t(j),n%p);}
template<class F> inline void each(sbn3_team *team,size_t n,unsigned p,F &&fn) noexcept {
    if(p==1){fn(0,n,0);return;}
    struct Job {size_t n;unsigned parts;F &fn;};Job job{n,p,fn};
    sbn3_team_run(team,[](void *v,sbn3_team_scope *scope){auto &j=*static_cast<Job *>(v);
        sbn3_team_for(scope,0,j.parts,1,SBN3_STATIC,[](void *v,uint64_t begin,uint64_t end,unsigned){auto &j=*static_cast<Job *>(v);
            for(unsigned k=unsigned(begin);k<end;++k)j.fn(cut(j.n,j.parts,k),cut(j.n,j.parts,k+1),k);
        },&j);
    },&job);
}
inline void fill(sbn3_team *t,uint64_t *r,size_t n,uint64_t word=0) noexcept {
    each(t,n,parts(t,n),[&](size_t b,size_t e,unsigned){for(size_t j=b;j<e;++j)r[j]=word;});
}
// Source and destination are disjoint (or identical); not a parallel memmove.
inline void copy(sbn3_team *t,uint64_t *r,const uint64_t *a,size_t n) noexcept {
    if(r==a)return;each(t,n,parts(t,n),[&](size_t b,size_t e,unsigned){if(e>b)memcpy(r+b,a+b,(e-b)*8);});
}
// In-place sub-word right shift. Snapshot each partition's one-word lookahead
// before dispatch, so no worker reads a neighbor's concurrently written word.
inline void right_shift(sbn3_team *t,uint64_t *r,size_t n,size_t input_words,unsigned bits) noexcept {
    require(bits<64,SBN3_FATAL_ARGUMENT,"parallel right shift");
    if(!n)return;
    if(!bits){if(n>input_words)fill(t,r+input_words,n-input_words);return;}
    const unsigned p=parts(t,n);uint64_t high[32]{};
    for(unsigned j=0;j<p;++j){const size_t e=cut(n,p,j+1);high[j]=e<input_words?r[e]:0;}
    each(t,n,p,[&](size_t b,size_t e,unsigned k){
        if(b==e)return;
        const size_t limit=std::min(e-1,input_words?input_words-1:0);
        size_t j=b;
        for(;j<limit;++j)r[j]=(r[j]>>bits)|(r[j+1]<<(64-bits));
        for(;j+1<e;++j)r[j]=j<input_words?r[j]>>bits:0;
        r[e-1]=e-1<input_words?(r[e-1]>>bits)|(high[k]<<(64-bits)):0;
    });
}
template<class F> inline void each_scope(sbn3_team_scope *scope,size_t n,unsigned p,F &&fn) noexcept {
    if(p==1){fn(0,n,0);return;}
    struct Job {size_t n;unsigned parts;F &fn;} job{n,p,fn};
    sbn3_team_for(scope,0,p,1,SBN3_STATIC,[](void *ptr,uint64_t begin,uint64_t end,unsigned){
        auto &j=*static_cast<Job *>(ptr);
        for(unsigned k=unsigned(begin);k<end;++k)j.fn(cut(j.n,j.parts,k),cut(j.n,j.parts,k+1),k);
    },&job);
}
inline unsigned scope_parts(sbn3_team_scope *s,size_t n) noexcept {
    return n>=minimum_parallel_words?sbn3_team_width(s):1;
}
inline void copy_scope(sbn3_team_scope *s,uint64_t *r,const uint64_t *a,size_t n) noexcept {
    if(r==a)return;
    each_scope(s,n,scope_parts(s,n),[&](size_t b,size_t e,unsigned){if(e>b)memcpy(r+b,a+b,(e-b)*8);});
}
inline bool zero(sbn3_team *t,const uint64_t *a,size_t n) noexcept {
    const unsigned p=parts(t,n);if(p==1)return limbs::zero(a,n);uint64_t any[32]{};
    each(t,n,p,[&](size_t b,size_t e,unsigned k){uint64_t v=0;for(size_t j=b;j<e;++j)v|=a[j];any[k]=v;});
    uint64_t v=0;for(unsigned k=0;k<p;++k)v|=any[k];return !v;
}
inline void complement(sbn3_team *t,uint64_t *r,size_t n) noexcept {
    each(t,n,parts(t,n),[&](size_t b,size_t e,unsigned){for(size_t j=b;j<e;++j)r[j]=~r[j];});
}
inline uint64_t add_word(uint64_t *r,size_t n,uint64_t value) noexcept {
    if(!n)return value;const __uint128_t x=__uint128_t(r[0])+value;r[0]=uint64_t(x);uint64_t c=uint64_t(x>>64);
    for(size_t j=1;c&&j<n;++j)c=++r[j]==0;return c;
}
inline uint64_t sub_word(uint64_t *r,size_t n,uint64_t value) noexcept {
    if(!n)return value;uint64_t c=r[0]<value;r[0]-=value;for(size_t j=1;c&&j<n;++j){c=r[j]==0;--r[j];}return c;
}
// Independent block products followed by exact carry stitching. Each seam
// modifies only its block, after the parallel phase has joined. A pathological
// carry chain still visits at most n words in total; work remains O(n).
inline uint64_t mul_1(sbn3_team *t,uint64_t *r,const uint64_t *a,size_t n,uint64_t word) noexcept {
    if(!n)return 0;const unsigned p=parts(t,n);if(p==1)return sbn3i_mul_1(r,a,long(n),word);uint64_t high[32]{};
    each(t,n,p,[&](size_t b,size_t e,unsigned k){high[k]=sbn3i_mul_1(r+b,a+b,long(e-b),word);});
    uint64_t c=0;for(unsigned k=0;k<p;++k){const size_t b=cut(n,p,k),e=cut(n,p,k+1);c=high[k]+(c?add_word(r+b,e-b,c):0);}return c;
}
template<bool subtract> inline uint64_t addsub(sbn3_team *t,uint64_t *r,size_t n,const uint64_t *a,size_t an) noexcept {
    if(!an)return 0;const unsigned p=parts(t,an);if(p==1)return subtract?limbs::sub_from(r,n,a,an):limbs::add_to(r,n,a,an);uint64_t high[32]{};
    each(t,an,p,[&](size_t b,size_t e,unsigned k){high[k]=subtract?sbn3i_sub_n(r+b,r+b,a+b,long(e-b)):sbn3i_add_n(r+b,r+b,a+b,long(e-b));});
    uint64_t c=0;for(unsigned k=0;k<p;++k){const size_t b=cut(an,p,k),e=cut(an,p,k+1);
        c=high[k]+(c?(subtract?sub_word(r+b,e-b,1):add_word(r+b,e-b,1)):0);}
    return c?(subtract?sub_word(r+an,n-an,1):add_word(r+an,n-an,1)):0;
}
inline uint64_t add_to(sbn3_team *t,uint64_t *r,size_t n,const uint64_t *a,size_t an) noexcept {return addsub<false>(t,r,n,a,an);}
inline uint64_t sub_from(sbn3_team *t,uint64_t *r,size_t n,const uint64_t *a,size_t an) noexcept {return addsub<true>(t,r,n,a,an);}
inline uint64_t add_to_scope(sbn3_team_scope *s,uint64_t *r,size_t n,const uint64_t *a,size_t an) noexcept {
    if(!an)return 0;
    const unsigned p=scope_parts(s,an);if(p==1)return limbs::add_to(r,n,a,an);
    uint64_t high[32]{};
    each_scope(s,an,p,[&](size_t b,size_t e,unsigned k){high[k]=sbn3i_add_n(r+b,r+b,a+b,long(e-b));});
    uint64_t carry=0;
    for(unsigned k=0;k<p;++k){const size_t b=cut(an,p,k),e=cut(an,p,k+1);carry=high[k]+(carry?add_word(r+b,e-b,1):0);}
    return carry?add_word(r+an,n-an,1):0;
}
inline bool cyclic_absolute(sbn3_team *t,uint64_t *r,size_t n) noexcept {const bool negative=r[n-1]>>63;if(negative)complement(t,r,n);return negative;}
inline void cyclic_sub_shifted(sbn3_team *t,uint64_t *r,size_t n,const uint64_t *a,size_t an,size_t shift) noexcept {
    const size_t first=std::min(an,n-shift);
    if(sub_from(t,r+shift,n-shift,a,first))sub_word(r,n,1);
    if(an>first&&sub_from(t,r,n,a+first,an-first))sub_word(r,n,1);
}
}
