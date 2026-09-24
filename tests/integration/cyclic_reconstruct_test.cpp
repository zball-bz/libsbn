#include "product_support.hpp"
#include "product/cyclic_reconstruct.hpp"
#include "algorithms/newton_planner.hpp"
#include "sbn3/newton.h"
using namespace sbn::v3;
template<size_t G>static unsigned lift_cases(){
    unsigned count=0;ref_int e,z,M,L,got;ref_inits(e,z,M,L,got,nullptr);
    ref_set_ui(L,1);ref_mul_2exp(L,L,64*G);
    for(size_t n:{3ul,4ul,7ul,16ul,33ul})for(unsigned pattern=0;pattern<6;++pattern)for(unsigned negative=0;negative<2;++negative){
        if(n<G)continue;
        std::vector<uint64_t> original(n+G),r(n+G);uint64_t low[G];
        for(auto &v:original)v=pattern<2?0:pattern==2?UINT64_MAX:random_word();
        original.back()&=(uint64_t(1)<<62)-1;if(pattern==1)original[0]=1;
        ref_import(e,original.size(),-1,8,0,0,original.data());if(negative)ref_neg(e,e);
        ref_set_ui(M,1);ref_mul_2exp(M,M,64*n);ref_sub_ui(M,M,1);ref_mod(z,e,M);
        const bool zero=ref_sgn(z)==0;
        for(unsigned redundant=0;redundant<1u+unsigned(zero);++redundant){
            std::fill(r.begin(),r.end(),0);memset(low,0,sizeof low);
            if(redundant)std::fill(r.begin(),r.begin()+n,UINT64_MAX);else ref_export(r.data(),nullptr,-1,8,0,0,z);
            ref_mod(got,e,L);ref_export(low,nullptr,-1,8,0,0,got);
            const bool sign=reconstruct_cyclic(r.data(),n,low);
            assert(sign==(ref_sgn(e)<0)&&r==original);++count;
        }
    }
    if constexpr(G==3)for(unsigned trial=0;trial<32;++trial){
        uint64_t a[3],b[3],p[3];for(unsigned i=0;i<3;++i){a[i]=random_word();b[i]=random_word();}
        product_low_words(p,a,b);ref_import(e,3,-1,8,0,0,a);ref_import(z,3,-1,8,0,0,b);
        ref_mul(got,e,z);ref_mod(got,got,L);ref_import(e,3,-1,8,0,0,p);assert(ref_cmp(e,got)==0);++count;
    }
    ref_clears(e,z,M,L,got,nullptr);return count;
}
static void tight_service(sbn3_newton_kind kind,unsigned np,size_t start,unsigned &tight){
    const sbn3_newton_options options{1,np,0,0};sbn3_newton_plan p{};sbn3_newton_info i{};
    size_t n=start;newton_detail::Plan detail{};
    for(unsigned attempt=0;attempt<4;++attempt){
        assert(sbn3_newton_query(kind,n,&options,&p,&i)==SBN3_SUPPORTED);memcpy(&detail,p.opaque,sizeof detail);
        const size_t ring=detail.choices[detail.choice_count-1].ring;
        if(ring==n)break;n=ring;assert(n<262144);
    }
    assert(detail.choices[detail.choice_count-1].ring==n);++tight;
    Fixture f(1,false);const size_t at=up(f.cursor,128);f.cursor=at+i.storage_bytes;
    sbn3_error error{};assert(sbn3_arena_prepare(f.arena,at,i.storage_bytes,&error)==SBN3_OK);
    auto *d=f.guarded(n),*a=f.guarded(n+1),*out=f.guarded(n+1);
    for(unsigned pattern=0;pattern<3;++pattern){
        for(size_t j=0;j<n;++j){d[j]=pattern?UINT64_MAX:random_word();a[j]=pattern?UINT64_MAX:random_word();}
        d[n-1]|=uint64_t(1)<<63;a[n]=1;
        if(pattern==2){memset(d,0,n*8);d[n-1]=uint64_t(1)<<63;}
        sbn3_newton_binding *b=nullptr;sbn3_newton_bind(&p,f.arena,at,f.team,&b);
        const sbn3_newton_inputs inputs{{a,n+1},{d,n},0};
        allocation_watch_start();sbn3_newton_execute(b,&inputs,{out,n+1});assert(!allocation_watch_stop());sbn3_newton_unbind(b);
        ref_int D,A,Q,E,T;ref_inits(D,A,Q,E,T,nullptr);ref_import(D,n,-1,8,0,0,d);ref_import(Q,n+1,-1,8,0,0,out);
        if(kind==SBN3_NEWTON_INVERSE){ref_set_ui(A,1);ref_mul_2exp(A,A,128*n);}
        else{ref_import(A,n+1,-1,8,0,0,a);ref_mul_2exp(A,A,64*n);}
        ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);ref_mul_ui(T,D,3);assert(ref_cmp(E,T)<0);ref_clears(D,A,Q,E,T,nullptr);
    }
    printf("tight cyclic Newton kind=%u NP%u n=ring=%zu PASS\n",kind,np,n);
}
int main(){
    printf("cyclic low-word lifting: %u cases PASS\n",lift_cases<1>()+lift_cases<2>()+lift_cases<3>()+
           lift_cases<4>()+lift_cases<5>()+lift_cases<6>()+lift_cases<7>()+lift_cases<8>());
    unsigned tight=0;
    for(auto kind:{SBN3_NEWTON_INVERSE,SBN3_NEWTON_DIVIDE})for(unsigned np:{4u,6u})
        for(size_t n:{1024ul,8192ul,32768ul})tight_service(kind,np,n,tight);
    assert(tight==12);puts("tight cyclic integration/allocation/strict error gates PASS");
}
