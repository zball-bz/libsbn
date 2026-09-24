#include "product_support.hpp"
#include "product/compact_windows.hpp"
#include "sbn3/newton.h"
#include "algorithms/local_inverse.hpp"
#include "algorithms/reciprocal.hpp"
using namespace sbn::v3;
namespace repair=sbn::v3::product::compact_window_detail;
static void lifts(){
    unsigned cases=0;
    for(size_t g:{1u,3u,8u,9u,64u,453u,695u,1024u}){
        const size_t n=g+19;std::vector<uint64_t> original(n+g),value(n+g),low(g);
        ref_int e,z,m,l,w;ref_inits(e,z,m,l,w,nullptr);ref_set_ui(m,1);ref_mul_2exp(m,m,64*n);ref_sub_ui(m,m,1);
        ref_set_ui(l,1);ref_mul_2exp(l,l,64*g);
        for(unsigned pattern=0;pattern<6;++pattern)for(unsigned negative=0;negative<2;++negative){
            for(auto &x:original)x=pattern<2?0:pattern==2?UINT64_MAX:random_word();
            original.back()&=(uint64_t(1)<<62)-1;if(pattern==1)original[0]=1;
            ref_import(e,original.size(),-1,8,0,0,original.data());if(negative)ref_neg(e,e);ref_mod(z,e,m);
            for(unsigned redundant=0;redundant<1u+unsigned(ref_sgn(z)==0);++redundant){
                std::fill(value.begin(),value.end(),0);std::fill(low.begin(),low.end(),0);
                if(redundant)std::fill(value.begin(),value.begin()+n,UINT64_MAX);else ref_export(value.data(),nullptr,-1,8,0,0,z);
                ref_mod(w,e,l);ref_export(low.data(),nullptr,-1,8,0,0,w);
                auto tailed=value;memcpy(tailed.data()+n,low.data(),8*g);
                allocation_watch_start();const bool sign=reconstruct_cyclic(value.data(),n,low.data(),g);
                const bool tail_sign=reconstruct_cyclic_tail(tailed.data(),n,g);assert(!allocation_watch_stop());
                assert(sign==(ref_sgn(e)<0)&&tail_sign==sign&&value==original&&tailed==original);++cases;
            }
        }
        ref_clears(e,z,m,l,w,nullptr);
    }
    printf("dynamic signed low-product lift: %u independent cases PASS\n",cases);
}
static void arithmetic(bool keep=false){
    unsigned cases=0,selected=0;
    for(size_t n:{512u,5793u,6597u,6889u,7194u,10624u,15689u,25268u,26386u,32800u,35000u,60097u,71468u,84990u,98304u,114688u,115098u,120194u,131072u,139000u,143355u,143356u,143360u})for(bool inverse:{true,false}){
        if(keep&&(inverse||(n!=25268&&n!=35000&&n!=60097&&n!=115098&&n!=139000&&n!=143355)))continue;
        Fixture f(1);repair::Factory factory{{},keep};const size_t bytes=inverse?reciprocal_bytes(n,factory):quotient_bytes(n,factory);
        auto work=f.allocate(bytes,128);auto *d=f.guarded(n),*a=f.guarded(n+1),*out=f.guarded(up(n+1,8));
        const auto shape=inverse?refinement_products<RefinementKind::Inverse>(newton_contract::next_precision(n),n):refinement_products<RefinementKind::Quotient>(newton_contract::next_precision(n),n);
        const auto candidate=repair::query(shape,keep);selected+=bool(candidate.ring);
        ref_int D,A,Q,E,L;ref_inits(D,A,Q,E,L,nullptr);
        for(unsigned pattern=0;pattern<6;++pattern){
            if(n>=131072){printf("repair boundary n=%zu inverse=%d pattern=%u ring=%zu bits=%u balanced=%d\n",n,inverse,pattern,candidate.ring,candidate.fft.bits,candidate.fft.balanced);fflush(stdout);}
            for(size_t j=0;j<n;++j){d[j]=pattern==1?UINT64_MAX:pattern==2?0:pattern==3?(j%2?UINT64_MAX:0):random_word();a[j]=pattern==4?0:pattern==1?UINT64_MAX:random_word();}
            d[n-1]|=uint64_t(1)<<63;a[n]=pattern==4?0:1;
            ref_import(D,n,-1,8,0,0,d);
            if(inverse){ref_set_ui(A,1);ref_mul_2exp(A,A,128*n);}
            else{ref_import(A,n+1,-1,8,0,0,a);ref_mul_2exp(A,A,64*n);}
            for(unsigned alias=0;alias<(inverse?1u:2u);++alias){
                std::vector<uint64_t> saved_a(a,a+n+1),saved_d(d,d+n);auto *result=alias?a:out;
                allocation_watch_start();{Frame scratch(*f.arena,work);
                    if(inverse)reciprocal(result,d,n,scratch,factory);else quotient(result,a,d,n,scratch,factory);
                }assert(!allocation_watch_stop());
                ref_import(Q,n+1,-1,8,0,0,result);ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);ref_mul_ui(L,D,3);
                if(ref_cmp(E,L)>=0){fprintf(stderr,"bad repair n=%zu inverse=%d pattern=%u alias=%u ring=%zu\n",n,inverse,pattern,alias,candidate.ring);abort();}
                assert(!memcmp(d,saved_d.data(),8*n));if(!alias)assert(!memcmp(a,saved_a.data(),8*(n+1)));memcpy(a,saved_a.data(),8*(n+1));++cases;
            }
        }
        ref_clears(D,A,Q,E,L,nullptr);printf("repair arithmetic n=%zu inverse=%d ring=%zu low=%zu bytes=%zu PASS\n",n,inverse,candidate.ring,candidate.low_words,bytes);fflush(stdout);
    }
    assert(selected);printf("repair drivers: %u strict-error/alias/allocation cases, %u repaired top groups PASS\n",cases,selected);
}
static void services(){
    unsigned cases=0,extended=0;
    for(size_t n:{17109u,32768u,34219u,65536u,71468u,98304u,114688u,115098u,120194u,131072u,139000u,143355u,143356u,143360u,169979u,230195u,440872u,881744u})for(bool inverse:{true,false}){
        const bool compact=local_refinement_supported(n,!inverse);
        if(n<=114688)assert(compact);else extended+=compact;
        const sbn3_newton_options options{1,0,0,0};sbn3_newton_plan plan{};sbn3_newton_info info{};
        const auto kind=inverse?SBN3_NEWTON_INVERSE:SBN3_NEWTON_DIVIDE;
        assert(sbn3_newton_query(kind,n,&options,&plan,&info)==SBN3_SUPPORTED);
        assert(info.workers==1);if(compact)assert(info.lease_peak==1);
        Fixture f(1);const size_t at=up(f.cursor,info.storage_alignment);f.cursor=at+info.storage_bytes;
        sbn3_error error{};assert(sbn3_arena_prepare(f.arena,at,info.storage_bytes,&error)==SBN3_OK);
        auto *d=f.guarded(n),*a=f.guarded(up(n+1,8)),*out=f.guarded(up(n+1,8));
        ref_int D,A,Q,E,L;ref_inits(D,A,Q,E,L,nullptr);
        for(unsigned pattern=0;pattern<3;++pattern){
            for(size_t j=0;j<n;++j){d[j]=pattern==1?UINT64_MAX:pattern==2?0:random_word();a[j]=random_word();}
            d[n-1]|=uint64_t(1)<<63;a[n]=1;
            ref_import(D,n,-1,8,0,0,d);
            if(inverse){ref_set_ui(A,1);ref_mul_2exp(A,A,128*n);}
            else{ref_import(A,n+1,-1,8,0,0,a);ref_mul_2exp(A,A,64*n);}
            sbn3_newton_binding *binding=nullptr;sbn3_newton_bind(&plan,f.arena,at,f.team,&binding);
            auto *result=(!inverse&&pattern==2)?a:out;const sbn3_newton_inputs inputs{{a,n+1},{d,n},0};
            allocation_watch_start();sbn3_newton_execute(binding,&inputs,{result,n+1});assert(!allocation_watch_stop());
            sbn3_newton_unbind(binding);
            ref_import(Q,n+1,-1,8,0,0,result);ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);ref_mul_ui(L,D,3);assert(ref_cmp(E,L)<0);++cases;
        }
        ref_clears(D,A,Q,E,L,nullptr);
    }
    assert(extended);
    printf("repaired public fresh/alias/strict-error cases: %u, %u extended groups PASS\n",cases,extended);
}
int main(){lifts();arithmetic();arithmetic(true);services();}
