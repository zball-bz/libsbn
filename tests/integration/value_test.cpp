#include "sbn3/value.h"
#include <assert.h>
#include "../oracle/oracle.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <algorithm>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();
static uint64_t state=9181726381;
static uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
static void import(ref_int x,sbn3_int_view a){ref_import(x,a.size,-1,8,0,0,a.data);if(a.negative)ref_neg(x,x);}
static void check(const sbn3_int &x,const ref_int want){ref_int got;ref_init(got);import(got,{x.data,x.size,x.negative});assert(ref_cmp(got,want)==0);assert(!x.size?!x.negative:x.data[x.size-1]!=0);ref_clear(got);}
int main(){
    ref_int A,B,R;ref_inits(A,B,R,nullptr);
    for(size_t n:{0u,1u,2u,3u,8u,17u,64u,1024u})for(unsigned pattern=0;pattern<8;++pattern){
        const size_t m=pattern%3?n:n/2;std::vector<uint64_t> a(n+16),b(n+16),r(n+16);
        for(size_t k=0;k<n;++k){a[k]=pattern==0?UINT64_MAX:pattern==1?0:random_word();b[k]=pattern==2?UINT64_MAX:pattern==3?0:random_word();}
        if(n&&pattern==4)a[n-1]=b[n-1]=0;
        for(unsigned sa=0;sa<2;++sa)for(unsigned sb=0;sb<2;++sb){
            const sbn3_int_view av{a.data(),n,sa},bv{b.data(),m,sb};import(A,av);import(B,bv);const int cmp=ref_cmp(A,B);
            assert(sbn3_int_compare(av,bv)==(cmp>0?1:cmp<0?-1:0));
            for(unsigned sub=0;sub<2;++sub)for(unsigned alias=0;alias<3;++alias){
                auto aa=a,bb=b;auto *dst=alias==1?aa.data():alias==2?bb.data():r.data();sbn3_int result{dst,r.size(),0,0};
                if(sub)ref_sub(R,A,B);else ref_add(R,A,B);
                allocation_watch_start();if(sub)sbn3_int_sub(&result,{aa.data(),n,sa},{bb.data(),m,sb});else sbn3_int_add(&result,{aa.data(),n,sa},{bb.data(),m,sb});assert(!allocation_watch_stop());check(result,R);
            }
        }
        for(size_t bits:{0u,1u,31u,63u,64u,65u,127u,128u,191u,640u})for(unsigned sign=0;sign<2;++sign)for(unsigned alias=0;alias<2;++alias){
            auto aa=a;sbn3_int result{alias?aa.data():r.data(),r.size(),0,0};import(A,{aa.data(),n,sign});ref_mul_2exp(R,A,bits);
            allocation_watch_start();sbn3_int_lshift(&result,{aa.data(),n,sign},bits);assert(!allocation_watch_stop());check(result,R);
            aa=a;result={alias?aa.data():r.data(),r.size(),0,0};ref_tdiv_q_2exp(R,A,bits);
            allocation_watch_start();sbn3_int_rshift(&result,{aa.data(),n,sign},bits);assert(!allocation_watch_stop());check(result,R);
        }
    }
    sbn3_int z{nullptr,0,0,1};sbn3_int_normalize(&z);assert(!z.size&&!z.negative);sbn3_int_rshift(&z,{nullptr,0,1},SIZE_MAX);assert(!z.size&&!z.negative);
    ref_clears(A,B,R,nullptr);puts("signed value compare/add/sub/shift/normalize: reference/modular oracle, aliases, long carry and no allocation PASS");
}
