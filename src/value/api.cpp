#include "sbn3/value.h"
#include "common/checked.hpp"
#include "common/small_checks.h"
#include "core/x86_64/word.hpp"
#include <string.h>
#include <algorithm>
using namespace sbn::v3;
namespace {
void source(sbn3_int_view a){
    if constexpr(SBN3_CHECK_SMALL){require(a.negative<=1 && a.size<=size_t(LONG_MAX),SBN3_FATAL_ARGUMENT,"value sign/length");valid_span(a.data,bytes_for(a.size,8),"value source");}
}
sbn3_int_view normalized(sbn3_int_view a){source(a);while(a.size&&!a.data[a.size-1])--a.size;if(!a.size)a.negative=0;return a;}
int magnitude(sbn3_int_view a,sbn3_int_view b){
    if(a.size!=b.size)return a.size>b.size?1:-1;
    for(size_t k=a.size;k-- >0;)if(a.data[k]!=b.data[k])return a.data[k]>b.data[k]?1:-1;return 0;
}
void output(sbn3_int *r,size_t need,sbn3_int_view a,sbn3_int_view b={}){
    if constexpr(SBN3_CHECK_SMALL){require(r&&need<=r->capacity,SBN3_FATAL_SIZE,"value capacity",need,r?r->capacity:0);const size_t bytes=bytes_for(need,8);
        valid_span(r->data,bytes,"value output");require(!(uintptr_t(r->data)&7),SBN3_FATAL_ARGUMENT,"value output alignment");
        for(auto v:{a,b})require((r->data==v.data||!overlaps(r->data,bytes,v.data,v.size*8))&&!overlaps(r,sizeof *r,v.data,v.size*8),SBN3_FATAL_ARGUMENT,"value partial alias");
        require(!overlaps(r,sizeof *r,r->data,bytes),SBN3_FATAL_ARGUMENT,"value descriptor alias");}
}
void finish(sbn3_int *r,size_t n,unsigned sign){while(n&&!r->data[n-1])--n;r->size=n;r->negative=n?sign:0;}
void sum(sbn3_int *r,sbn3_int_view a,sbn3_int_view b,bool subtract){
    a=normalized(a);b=normalized(b);if(subtract&&b.size)b.negative^=1;
    const size_t largest=std::max(a.size,b.size);size_t need=0;
    require(!largest||add_size(largest,1,need),SBN3_FATAL_SIZE,"value addition size");output(r,need,a,b);
    if(!largest){r->size=r->negative=0;return;}
    if(a.negative==b.negative){
        if(a.size<b.size)std::swap(a,b);
        uint64_t carry=b.size?sbn3i_add_n(r->data,a.data,b.data,long(b.size)):0;
        if(a.size>b.size)memmove(r->data+b.size,a.data+b.size,(a.size-b.size)*8);
        for(size_t k=b.size;carry&&k<a.size;++k)carry=++r->data[k]==0;
        r->data[a.size]=carry;finish(r,a.size+1,a.negative);
    }else{
        const int cmp=magnitude(a,b);if(!cmp){r->size=r->negative=0;return;}
        if(cmp<0)std::swap(a,b);
        uint64_t borrow=b.size?sbn3i_sub_n(r->data,a.data,b.data,long(b.size)):0;
        if(a.size>b.size)memmove(r->data+b.size,a.data+b.size,(a.size-b.size)*8);
        for(size_t k=b.size;borrow&&k<a.size;++k){borrow=r->data[k]==0;--r->data[k];}
        require(!borrow,SBN3_FATAL_MATH,"value subtraction borrow");finish(r,a.size,a.negative);
    }
}
}
extern "C" void sbn3_int_normalize(sbn3_int *r){
    if constexpr(SBN3_CHECK_SMALL)require(r&&r->size<=r->capacity,SBN3_FATAL_ARGUMENT,"normalize value");
    auto a=normalized({r->data,r->size,r->negative});r->size=a.size;r->negative=a.negative;
}
extern "C" int sbn3_int_compare(sbn3_int_view a,sbn3_int_view b){a=normalized(a);b=normalized(b);if(a.negative!=b.negative)return a.negative?-1:1;const auto c=magnitude(a,b);return a.negative?-c:c;}
extern "C" void sbn3_int_add(sbn3_int *r,sbn3_int_view a,sbn3_int_view b){sum(r,a,b,false);}
extern "C" void sbn3_int_sub(sbn3_int *r,sbn3_int_view a,sbn3_int_view b){sum(r,a,b,true);}
extern "C" void sbn3_int_lshift(sbn3_int *r,sbn3_int_view a,size_t bits){
    a=normalized(a);if(!a.size){output(r,0,a);r->size=r->negative=0;return;}
    const size_t words=bits/64;const unsigned part=bits%64;size_t n=0,need=0;
    require(add_size(a.size,words,n)&&add_size(n,unsigned(part!=0),need),SBN3_FATAL_SIZE,"left shift size");output(r,need,a);
    if(part){r->data[n]=a.data[a.size-1]>>(64-part);for(size_t k=a.size;k-- >0;)r->data[k+words]=(a.data[k]<<part)|(k?a.data[k-1]>>(64-part):0);}
    else memmove(r->data+words,a.data,a.size*8);
    if(words)memset(r->data,0,words*8);finish(r,need,a.negative);
}
extern "C" void sbn3_int_rshift(sbn3_int *r,sbn3_int_view a,size_t bits){
    a=normalized(a);const size_t words=bits/64;const unsigned part=bits%64;
    if(words>=a.size){output(r,0,a);r->size=r->negative=0;return;}
    const size_t n=a.size-words;output(r,n,a);
    if(part)for(size_t k=0;k<n;++k)r->data[k]=(a.data[k+words]>>part)|(k+words+1<a.size?a.data[k+words+1]<<(64-part):0);
    else memmove(r->data,a.data+words,n*8);
    finish(r,n,a.negative);
}
