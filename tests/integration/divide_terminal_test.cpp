#include "newton_setup.hpp"
#include "algorithms/divide_terminal.hpp"
using namespace sbn::v3;
static void verify(const uint64_t *A,const uint64_t *D,const uint64_t *Q,size_t n){
    ref_int a,d,q,r,bound;ref_inits(a,d,q,r,bound,nullptr);
    ref_import(a,n+1,-1,8,0,0,A);ref_import(d,n,-1,8,0,0,D);ref_import(q,n+1,-1,8,0,0,Q);
    ref_mul(r,q,d);ref_mul_2exp(a,a,64*n);ref_sub(r,r,a);ref_abs(r,r);ref_mul_ui(bound,d,3);
    if(ref_cmp(r,bound)>=0){fprintf(stderr,"division terminal >=3 ulps at n=%zu\n",n);assert(false);}
    ref_clears(a,d,q,r,bound,nullptr);
}
static DivideTerminal prepare(Fixture &f,size_t m,size_t n,unsigned np=6,unsigned alg=0){
    sbn3_mul_options o{};o.workers=sbn3_team_workers(f.team);o.prime_count=np;o.borrow_output=1;o.algorithm=alg?alg:n<=8192?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
    sbn3_product_request req{};req.a_limbs=m+1;req.b_limbs=n+1;sbn3_mul_plan plan{};sbn3_product_info info{};bool found=false;
    for(int T=np==4?88:24*int(np)-8;T>=(np==4?80:24*int(np)-32);T-=np==4?4:8){
        size_t ring=2*size_t(T);while(ring<2*m+4)ring*=2;o.trunk_bits=T;req.cyclic_limbs=ring;
        if(sbn3_product_query(&req,&o,&plan,&info)==SBN3_SUPPORTED){found=true;break;}}
    assert(found);sbn3_spectrum_desc future{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,n,&future)==SBN3_SUPPORTED);
    const auto first=f.leases.size();auto *producer=f.product(req,o,info,plan);auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *cache=nullptr;
    sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,n,f.arena,&storage,&cache);req.cached_a[0]=&future;
    auto *cached=f.product(req,o,info,plan,cache);req.cached_a[0]=nullptr;req.b_limbs=n;auto *residual=f.product(req,o,info,plan);
    f.unbind(producer);f.retire(first+1);f.retire(first);
    return {m,n,req.cyclic_limbs,cached,residual,cache,{f.guarded(req.cyclic_limbs),f.guarded(req.cyclic_limbs)}};
}
static void test(size_t n,unsigned workers,unsigned pattern,int delta,bool integrated,unsigned np=6){
    const size_t m=n/2+1;Fixture f(workers);auto terminal=prepare(f,m,n,np);
    auto *A=f.guarded(n+1),*D=f.guarded(n),*U=f.guarded(m+1),*Q=f.guarded(n+1);
    for(size_t j=0;j<n;++j){D[j]=pattern==1?UINT64_MAX:pattern==2?0:random_word();A[j]=pattern==0?0:pattern==3?UINT64_MAX:random_word();}
    D[n-1]|=uint64_t(1)<<63;A[n]=pattern>=3?1:0;
    if(pattern==4){memcpy(A,D,n*8);A[n]=0;}
    if(pattern==5){ // A=2D-1: quotient at the upper side of an integer boundary.
        uint64_t carry=0;for(size_t j=0;j<n;++j){const uint64_t w=D[j];A[j]=(w<<1)|carry;carry=w>>63;}A[n]=carry;
        for(size_t j=0;j<=n;++j)if(A[j]--)break;
    }
    if(integrated){
        PreparedInverse inverse(f,m);
        assert(inverse.steps.empty() || inverse.steps.back().n==m); // no final n-limb inverse rung exists
        allocation_watch_start();inverse.execute(D+n-m,U);divide_terminal(terminal,A,D,U,Q);assert(!allocation_watch_stop());
    }else{
        ref_int h,u,p,lo,hi;ref_inits(h,u,p,lo,hi,nullptr);ref_import(h,m,-1,8,0,0,D+n-m);
        ref_set_ui(p,1);ref_mul_2exp(p,p,128*m);ref_fdiv_q(u,p,h);
        if(delta<0)ref_sub_ui(u,u,unsigned(-delta));else ref_add_ui(u,u,unsigned(delta));
        ref_set_ui(lo,1);ref_mul_2exp(lo,lo,64*m);ref_mul_2exp(hi,lo,1);ref_sub_ui(hi,hi,1);
        if(ref_cmp(u,lo)<0)ref_set(u,lo);if(ref_cmp(u,hi)>0)ref_set(u,hi);memset(U,0,(m+1)*8);size_t count;ref_export(U,&count,-1,8,0,0,u);
        ref_clears(h,u,p,lo,hi,nullptr);allocation_watch_start();divide_terminal(terminal,A,D,U,Q);assert(!allocation_watch_stop());
    }
    verify(A,D,Q,n);sbn3_product_metrics cached{},residual{};sbn3_product_get_metrics(terminal.cached_inverse,&cached);sbn3_product_get_metrics(terminal.residual,&residual);
    assert(cached.mul.executions==2 && residual.mul.executions==1);
    sbn3_spectrum_release(terminal.u);
    printf("fused divide n=%zu m=%zu W%u pattern%u delta%d integrated=%u: <3 ulps/half inverse/3 products/reuse PASS\n",n,m,workers,pattern,delta,integrated);fflush(stdout);
}
int main(){
    for(size_t n:{4u,5u,31u,32u,129u,1024u})for(unsigned p=0;p<6;++p)for(int delta:{-7,7})test(n,1,p,delta,false);
    for(unsigned np:{4u,8u,9u,10u})test(129,3,5,-7,false,np);
    for(size_t n:{4u,31u,129u,8192u,65536u})test(n,1,3,0,true);
    test(8192,16,5,0,true);test(65536,16,3,0,true);
    puts("fused inverse/division terminal gates PASS");
}
