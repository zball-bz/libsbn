#include "product_support.hpp"
#include "sbn3/newton.h"
static void scenario(size_t n,unsigned workers){
    Fixture f(workers,false);sbn3_newton_options options{workers,0,0,unsigned(n%17==0)};
    sbn3_newton_plan plan{};sbn3_newton_info info{};
    allocation_watch_start();const auto rc=sbn3_newton_query(SBN3_NEWTON_DIVIDE,n,&options,&plan,&info);
    assert(!allocation_watch_stop()&&rc==SBN3_SUPPORTED);
    const auto saved=plan;options.memory_budget=info.storage_bytes-1;sbn3_newton_info need{};
    assert(sbn3_newton_query(SBN3_NEWTON_DIVIDE,n,&options,&plan,&need)==SBN3_QUERY_CAPACITY&&
           need.storage_bytes==info.storage_bytes&&!memcmp(&plan,&saved,sizeof plan));
    const size_t at=up(f.base+f.cursor,info.storage_alignment)-f.base;f.cursor=at+info.storage_bytes;
    sbn3_error error{};assert(sbn3_arena_prepare(f.arena,at,info.storage_bytes,&error)==SBN3_OK);
    const size_t cap=up(n+1,8);auto *d=f.guarded(n),*a=f.guarded(cap),*q=f.guarded(cap);
    ref_int D,A,Q,E,L;ref_inits(D,A,Q,E,L,nullptr);
    for(unsigned pattern=0;pattern<(n<=24?69u:5u);++pattern){
        for(size_t j=0;j<n;++j){d[j]=random_word();a[j]=random_word();}
        d[n-1]|=uint64_t(1)<<63;a[n]=1;
        if(pattern==1){memset(d,0,8*n);d[n-1]=uint64_t(1)<<63;a[n]=0;}
        if(pattern==2){memset(d,0xff,8*n);memset(a,0xff,8*n);}
        if(pattern==3)memset(a,0,8*(n+1));
        if(pattern==4){memset(d,0xff,8*n);memcpy(a,d,8*n);a[n]=0;if(n>1)--a[1];else --a[0];}
        if(pattern>=5)a[n]=pattern&1;
        std::vector<uint64_t> original(a,a+n+1),denominator(d,d+n),expected;
        ref_import(D,n,-1,8,0,0,d);ref_import(A,n+1,-1,8,0,0,a);ref_mul_2exp(A,A,64*n);ref_mul_ui(L,D,3);
        for(bool alias:{false,true}){
            memcpy(a,original.data(),8*(n+1));auto *out=alias?a:q;
            for(size_t j=n+1;j<cap;++j)out[j]=0xa5a5a5a5a5a5a5a5ULL;
            sbn3_newton_binding *b=nullptr;sbn3_newton_inputs input{{a,n+1},{d,n},0};
            allocation_watch_start();sbn3_newton_bind(&plan,f.arena,at,f.team,&b);
            sbn3_newton_execute(b,&input,{out,n+1});assert(!allocation_watch_stop());
            ref_import(Q,n+1,-1,8,0,0,out);ref_mul(E,Q,D);ref_sub(E,E,A);ref_abs(E,E);assert(ref_cmp(E,L)<0);
            for(size_t j=n+1;j<cap;++j)assert(out[j]==0xa5a5a5a5a5a5a5a5ULL);
            assert(!memcmp(d,denominator.data(),8*n));
            if(!alias){assert(!memcmp(a,original.data(),8*(n+1)));expected.assign(q,q+n+1);}
            else assert(!memcmp(out,expected.data(),8*(n+1)));
            sbn3_newton_metrics metrics{};sbn3_newton_get_metrics(b,&metrics);
            if(options.timing)assert(metrics.compute_ns);
            allocation_watch_start();sbn3_newton_unbind(b);assert(!allocation_watch_stop());
        }
    }
    ref_clears(D,A,Q,E,L,nullptr);
}
int main(){
    for(size_t n=1;n<=512;++n)scenario(n,1);
    for(size_t n:{1u,3u,26u,79u,489u,490u,507u,508u,512u,1024u})scenario(n,16);
    puts("small approximate divide: every limb 1..512, normalization/extremes/zero/saturation/in-place/rebind/budget/allocation gates PASS");
}
