#pragma once
// Original libsbn code; LGPL-3.0-or-later.
#include "value/bitwise.hpp"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include <cstring>
using namespace sbn::v3;
namespace {
using U=uint64_t;
sbn3_int_view normalized(sbn3_int_view a){
    if constexpr(SBN3_CHECK_SMALL){require(a.negative<=1&&a.size<=SIZE_MAX/64,SBN3_FATAL_ARGUMENT,"bit value domain");valid_span(a.data,a.size*8,"bit value input");}
    a.size=bits::normalized_size(a.data,a.size);a.negative=a.size&&a.negative;return a;
}
void destination(sbn3_int*r,size_t n,sbn3_int_view a,sbn3_int_view b={}){
    if constexpr(SBN3_CHECK_SMALL){require(r&&n<=r->capacity&&r->capacity<=SIZE_MAX/8,SBN3_FATAL_WORKSPACE,"bit output capacity",n,r?r->capacity:0);valid_span(r->data,n*8,"bit output");
        require(!(uintptr_t(r->data)&7),SBN3_FATAL_ARGUMENT,"bit output alignment");
        for(auto x:{a,b})require((r->data==x.data||!overlaps(r->data,n*8,x.data,x.size*8))&&!overlaps(r,sizeof*r,x.data,x.size*8),SBN3_FATAL_ARGUMENT,"bit partial alias");
        require(!overlaps(r,sizeof*r,r->data,n*8),SBN3_FATAL_ARGUMENT,"bit descriptor alias");}
}
template<sbn3_bit_operation Op>[[gnu::always_inline]]inline void binary(sbn3_int*r,sbn3_int_view a,sbn3_int_view b){
    constexpr auto op=Op;
    a=normalized(a);b=normalized(b);const auto s=bits::shape(op,a.size,a.negative,b.size,b.negative);
    destination(r,s.words+unsigned(s.negative),a,b);
    if(a.data==b.data&&a.size==b.size&&a.negative==b.negative){
        if(op==SBN3_BIT_XOR){r->size=r->negative=0;return;}
        if(a.size&&r->data!=a.data)memmove(r->data,a.data,a.size*8);r->size=a.size;r->negative=a.negative;return;
    }
    if(a.size<=2&&b.size<=2){
        auto wide=[](sbn3_int_view v){return v.size?__uint128_t(v.data[0])|(v.size>1?__uint128_t(v.data[1])<<64:0):0;};
        auto x=wide(a),y=wide(b);if(a.negative)x=__uint128_t(0)-x;if(b.negative)y=__uint128_t(0)-y;
        __uint128_t z=Op==SBN3_BIT_AND?x&y:Op==SBN3_BIT_OR?x|y:x^y;if(s.negative)z=__uint128_t(0)-z;
        const U lo=U(z),hi=U(z>>64);const bool extra=s.negative&&!z;
        r->size=extra?3:hi?2:bool(lo);if(r->size)r->data[0]=lo;if(r->size>1)r->data[1]=hi;if(extra)r->data[2]=1;
    }else r->size=bits::dispatch_signs<8,4,Op>(r->data,a.data,a.size,a.negative,b.data,b.size,b.negative);
    r->negative=r->size&&s.negative;
}
U mask(unsigned bits){return bits==64?UINT64_MAX:bits?(U(1)<<bits)-1:0;}
size_t finite_population(sbn3_int_view a,size_t first,size_t count){
    const size_t width=a.size*64;if(!count||first>=width)return 0;count=std::min(count,width-first);
    size_t at=first/64,total=0;const unsigned low=first%64;
    if(low){const unsigned take=unsigned(std::min(count,size_t(64-low)));total=__builtin_popcountll((a.data[at]>>low)&mask(take));count-=take;++at;}
    const size_t words=count/64;total+=bits::population(a.data+at,words);at+=words;
    if(count%64)total+=__builtin_popcountll(a.data[at]&mask(count%64));return total;
}
size_t trailing(sbn3_int_view a){const size_t p=bits::first_nonzero(a.data,a.size);return p<a.size?p*64+__builtin_ctzll(a.data[p]):SIZE_MAX;}
size_t scan_magnitude(sbn3_int_view a,size_t first,bool which){
    if(first==SIZE_MAX)return SIZE_MAX;
    const size_t i=first/64;if(i>=a.size)return which?SIZE_MAX:first;
    U x=(which?a.data[i]:~a.data[i])&~mask(first%64);if(x)return i*64+__builtin_ctzll(x);
    const size_t p=which?bits::first_nonzero(a.data+i+1,a.size-i-1):bits::first_not_ones(a.data+i+1,a.size-i-1);
    if(p==a.size-i-1)return which?SIZE_MAX:a.size*64;
    x=which?a.data[i+1+p]:~a.data[i+1+p];return (i+1+p)*64+__builtin_ctzll(x);
}
bool test(sbn3_int_view a,size_t bit){
    const size_t word=bit/64;if(word>=a.size)return a.negative;
    U x=a.data[word];if(a.negative)x=bits::first_nonzero(a.data,word)<word?~x:U(0)-x;
    return (x>>(bit%64))&1;
}
}
extern "C" void sbn3_int_bitwise(sbn3_int*r,sbn3_bit_operation op,sbn3_int_view a,sbn3_int_view b){
    require(unsigned(op)<=SBN3_BIT_XOR,SBN3_FATAL_ARGUMENT,"bit operation");
    switch(op){case SBN3_BIT_AND:binary<SBN3_BIT_AND>(r,a,b);break;case SBN3_BIT_OR:binary<SBN3_BIT_OR>(r,a,b);break;default:binary<SBN3_BIT_XOR>(r,a,b);}
}
extern "C" void sbn3_int_and(sbn3_int*r,sbn3_int_view a,sbn3_int_view b){binary<SBN3_BIT_AND>(r,a,b);}
extern "C" void sbn3_int_or(sbn3_int*r,sbn3_int_view a,sbn3_int_view b){binary<SBN3_BIT_OR>(r,a,b);}
extern "C" void sbn3_int_xor(sbn3_int*r,sbn3_int_view a,sbn3_int_view b){binary<SBN3_BIT_XOR>(r,a,b);}
extern "C" void sbn3_int_complement(sbn3_int*r,sbn3_int_view a){
    a=normalized(a);destination(r,a.size+unsigned(!a.negative),a);
    r->size=bits::adjust_copy(r->data,a.data,a.size,!a.negative);r->negative=r->size&&!a.negative;
}
extern "C" void sbn3_int_arshift(sbn3_int*r,sbn3_int_view a,size_t amount){
    a=normalized(a);const size_t words=amount/64;const unsigned shift=amount%64;
    bool discarded=false;
    if(a.negative){const size_t lo=std::min(words,a.size);discarded=bits::first_nonzero(a.data,lo)<lo;
        if(!discarded&&words<a.size&&shift)discarded=bool(a.data[words]&mask(shift));}
    const size_t need=words<a.size?a.size-words+unsigned(a.negative&&discarded&&!shift&&words):unsigned(bool(a.size)&&a.negative);
    destination(r,need,a);sbn3_int_rshift(r,a,amount);
    if(a.negative&&discarded){r->size=bits::adjust_copy(r->data,r->data,r->size,true);r->negative=1;}
}
extern "C" size_t sbn3_int_bit_length(sbn3_int_view a){a=normalized(a);return a.size?a.size*64-__builtin_clzll(a.data[a.size-1]):0;}
extern "C" size_t sbn3_int_popcount_magnitude(sbn3_int_view a){
    if constexpr(SBN3_CHECK_SMALL){require(a.negative<=1&&a.size<=SIZE_MAX/64,SBN3_FATAL_ARGUMENT,"popcount span");valid_span(a.data,a.size*8,"popcount input");}
    if(a.size<=2)return (a.size?__builtin_popcountll(a.data[0]):0)+(a.size>1?__builtin_popcountll(a.data[1]):0);
    return bits::population(a.data,a.size);
}
extern "C" int sbn3_int_test_bit(sbn3_int_view a,size_t bit){return test(normalized(a),bit);}
extern "C" size_t sbn3_int_scan_bit(sbn3_int_view a,size_t first,unsigned which){
    require(which<=1,SBN3_FATAL_ARGUMENT,"scan bit value");a=normalized(a);
    if(!a.negative)return scan_magnitude(a,first,which);
    if(first==SIZE_MAX)return SIZE_MAX;
    if(first/64>=a.size)return which?first:SIZE_MAX;
    const size_t pivot=trailing(a);
    if(which&&first<=pivot)return pivot;
    if(!which&&first<pivot)return first;
    return scan_magnitude(a,std::max(first,pivot+1),!which);
}
extern "C" size_t sbn3_int_popcount_range(sbn3_int_view a,size_t first,size_t count){
    a=normalized(a);size_t p=finite_population(a,first,count);if(!a.negative||!count)return p;
    const size_t width=a.size*64;if(first>=width)return count;
    const size_t pivot=trailing(a);
    if(first<pivot)p+=std::min(count,pivot-first);
    if(first<=pivot&&pivot-first<count)--p;
    return count-p;
}
extern "C" size_t sbn3_int_hamdist_range(sbn3_int_view a,sbn3_int_view b,size_t first,size_t count){
    a=normalized(a);b=normalized(b);const size_t width=std::max(a.size,b.size)*64;
    const bool negate=a.negative!=b.negative;if(!count||first>=width)return negate?count:0;
    const size_t ap=a.negative?bits::first_nonzero(a.data,a.size):SIZE_MAX,bp=b.negative?bits::first_nonzero(b.data,b.size):SIZE_MAX;
    const size_t used=std::min(count,width-first);size_t remaining=used,at=first/64,sum=0;
    auto word=[&](size_t i){U x=i<a.size?a.data[i]:0,y=i<b.size?b.data[i]:0;if(ap!=SIZE_MAX&&i<=ap)--x;if(bp!=SIZE_MAX&&i<=bp)--y;return x^y;};
    const unsigned low=first%64;
    if(low){const unsigned take=unsigned(std::min(remaining,size_t(64-low)));sum=__builtin_popcountll((word(at)>>low)&mask(take));remaining-=take;++at;}
    const size_t words=remaining/64;sum+=bits::population_xor(a.data,a.size,ap,b.data,b.size,bp,at,words);at+=words;
    if(remaining%64)sum+=__builtin_popcountll(word(at)&mask(remaining%64));return negate?count-sum:sum;
}
extern "C" void sbn3_int_edit_bit(sbn3_int*r,sbn3_int_view a,size_t bit,sbn3_bit_edit op){
    require(unsigned(op)<=SBN3_BIT_FLIP,SBN3_FATAL_ARGUMENT,"bit edit");a=normalized(a);
    const bool before=test(a,bit),after=op==SBN3_BIT_FLIP?!before:op==SBN3_BIT_SET;
    if(before==after){destination(r,a.size,a);if(a.size&&r->data!=a.data)memmove(r->data,a.data,a.size*8);r->size=a.size;r->negative=a.negative;return;}
    const size_t at=bit/64;require(at<SIZE_MAX/64-1,SBN3_FATAL_SIZE,"bit edit index",bit,SIZE_MAX-128);
    const bool increase=after!=bool(a.negative);const size_t n=std::max(a.size,at+1),need=n+unsigned(increase);
    destination(r,need,a);if(a.size&&r->data!=a.data)memmove(r->data,a.data,a.size*8);
    if(n>a.size)memset(r->data+a.size,0,(n-a.size)*8);
    const U delta=U(1)<<(bit%64),old=r->data[at];r->data[at]=increase?old+delta:old-delta;
    bool carry=increase?r->data[at]<old:old<delta;
    for(size_t i=at+1;carry&&i<n;++i){if(increase)carry=++r->data[i]==0;else{carry=r->data[i]==0;--r->data[i];}}
    size_t out=n;if(increase&&carry)r->data[out++]=1;
    require(increase||!carry,SBN3_FATAL_MATH,"bit edit magnitude borrow");r->size=bits::normalized_size(r->data,out);r->negative=r->size&&a.negative;
}
