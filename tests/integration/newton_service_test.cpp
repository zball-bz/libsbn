#include "product_support.hpp"
#include "sbn3/newton.h"
struct Prepared {
    Fixture &f;sbn3_newton_plan plan{};sbn3_newton_info info{};sbn3_newton_binding *bound=nullptr;size_t offset=0;
    Prepared(Fixture &fixture,sbn3_newton_kind kind,size_t n):f(fixture){
        sbn3_newton_options options{sbn3_team_workers(f.team),0,0,1};
        allocation_watch_start();const auto rc=sbn3_newton_query(kind,n,&options,&plan,&info);assert(!allocation_watch_stop());
        if(rc!=SBN3_SUPPORTED)fprintf(stderr,"Newton query kind=%u n=%zu: %u\n",kind,n,rc);assert(rc==SBN3_SUPPORTED);
        const auto saved=plan;options.memory_budget=info.storage_bytes-1;sbn3_newton_info need{};
        assert(sbn3_newton_query(kind,n,&options,&plan,&need)==SBN3_QUERY_CAPACITY && need.storage_bytes==info.storage_bytes && !memcmp(&saved,&plan,sizeof plan));
        offset=up(f.base+f.cursor,info.storage_alignment)-f.base;f.cursor=up(offset+info.storage_bytes,4096)+4096;
        sbn3_error e{};assert(sbn3_arena_prepare(f.arena,offset,info.storage_bytes,&e)==SBN3_OK);
        bind();
    }
    void bind(){allocation_watch_start();sbn3_newton_bind(&plan,f.arena,offset,f.team,&bound);assert(!allocation_watch_stop());}
    void close(){if(bound){allocation_watch_start();sbn3_newton_unbind(bound);assert(!allocation_watch_stop());bound=nullptr;}}
    ~Prepared(){close();}
};
static void sqrt2(size_t n,unsigned workers){
    Fixture f(workers);Prepared a(f,SBN3_SQRT2_RSQRT,n),b(f,SBN3_SQRT2_RATIONAL,n);
    auto *x=f.guarded(up(n+1,8)),*y=f.guarded(up(n+1,8));
    allocation_watch_start();sbn3_newton_execute(a.bound,nullptr,{x,n+1});sbn3_newton_execute(b.bound,nullptr,{y,n+1});assert(!allocation_watch_stop());
    assert(!memcmp(x,y,(n+1)*8));ref_int expected,got;ref_inits(expected,got,nullptr);ref_set_ui(expected,1);ref_mul_2exp(expected,expected,128*n+1);ref_sqrt(expected,expected);
    ref_import(got,n+1,-1,8,0,0,x);assert(ref_cmp(expected,got)==0);ref_clears(expected,got,nullptr);
    sbn3_newton_metrics am{},bm{};sbn3_newton_get_metrics(a.bound,&am);sbn3_newton_get_metrics(b.bound,&bm);
    assert(am.verify_ns && bm.verify_ns && am.products_executed && bm.products_executed);
    // Reuse the same plan and already-resident address range for a new binding.
    a.close();a.bind();allocation_watch_start();sbn3_newton_execute(a.bound,nullptr,{x,n+1});assert(!allocation_watch_stop());assert(!memcmp(x,y,(n+1)*8));
    printf("newton C API sqrt2 n=%zu W%u: identical routes/exact floor/no allocation/rebind PASS; bytes=%zu/%zu, products=%u/%u\n",n,workers,a.info.storage_bytes,b.info.storage_bytes,am.products_executed,bm.products_executed);fflush(stdout);
}
static void approximate(sbn3_newton_kind kind,size_t n,unsigned workers){
    Fixture f(workers);Prepared p(f,kind,n);auto *d=f.guarded(n),*a=f.guarded(up(n+1,8)),*out=f.guarded(up(n+1,8));
    for(size_t j=0;j<n;++j){d[j]=random_word();a[j]=random_word();}d[n-1]|=uint64_t(1)<<63;a[n]=1;
    const sbn3_newton_inputs input{{a,n+1},{d,n},UINT64_MAX};
    allocation_watch_start();sbn3_newton_execute(p.bound,&input,{out,n+1});assert(!allocation_watch_stop());
    ref_int D,A,O,P,E,T;ref_inits(D,A,O,P,E,T,nullptr);ref_import(D,n,-1,8,0,0,d);ref_import(A,n+1,-1,8,0,0,a);ref_import(O,n+1,-1,8,0,0,out);
    ref_set_ui(P,1);ref_mul_2exp(P,P,128*n);
    if(kind==SBN3_NEWTON_RSQRT){ref_sub_ui(E,O,3);ref_mul(E,E,E);ref_mul_ui(E,E,UINT64_MAX);ref_add_ui(T,O,3);ref_mul(T,T,T);ref_mul_ui(T,T,UINT64_MAX);assert(ref_cmp(E,P)<0&&ref_cmp(P,T)<0);}
    else{if(kind==SBN3_NEWTON_DIVIDE)ref_mul_2exp(P,A,64*n);ref_mul(E,O,D);ref_sub(E,E,P);ref_abs(E,E);ref_mul_ui(T,D,3);assert(ref_cmp(E,T)<0);}
    ref_clears(D,A,O,P,E,T,nullptr);printf("newton C API kind=%u n=%zu W%u: <3 ulps/no allocation PASS\n",kind,n,workers);
    // Expired stage bindings/cache objects must release every arena reference;
    // replay in the same resident range must produce exactly the same value.
    const std::vector<uint64_t> previous(out,out+n+1);
    p.close();p.bind();
    auto *again=kind==SBN3_NEWTON_DIVIDE?a:out;
    allocation_watch_start();sbn3_newton_execute(p.bound,&input,{again,n+1});assert(!allocation_watch_stop());
    assert(!memcmp(again,previous.data(),(n+1)*8));
}
int main(){
    for(size_t n:{1u,7u,64u,1024u,8192u})sqrt2(n,1);sqrt2(65536,16);
    for(auto kind:{SBN3_NEWTON_INVERSE,SBN3_NEWTON_RSQRT,SBN3_NEWTON_DIVIDE}){approximate(kind,4,1);approximate(kind,17,1);approximate(kind,8192,16);approximate(kind,524288,16);approximate(kind,524289,16);}
    puts("Newton library service gates PASS");
}
