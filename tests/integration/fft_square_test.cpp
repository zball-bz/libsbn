#include "product_support.hpp"
#include "runtime/scratch.hpp"
#include "backend/pq16/kernels.hpp"
#include <sys/mman.h>
using namespace sbn::v3;
struct Task {const pq16::Tables *tables;Frame *space;const uint64_t *a,*b;uint64_t *out;size_t n;};
static void execute(void *arg,sbn3_team_scope *scope){auto &x=*static_cast<Task *>(arg);pq16::multiply(x.out,x.a,x.n,x.b,x.n,*x.tables,*x.space,scope);}
static void one(pq16::Shape shape,size_t n,unsigned workers){
    assert(shape.nfull);Fixture f(workers);
    const size_t square_bytes=pq16::scratch_bytes(shape,n,n,workers,true),mul_bytes=pq16::scratch_bytes(shape,n,n,workers);
    assert(square_bytes<=mul_bytes);
    if(!shape.centered || shape.radix==1)assert(square_bytes+16ul*shape.nfull<=mul_bytes);
    auto tl=f.allocate(pq16::table_bytes(shape),128),sw=f.allocate(square_bytes,128),mw=f.allocate(mul_bytes,128);
    Frame tf(*f.arena,tl),sf(*f.arena,sw),mf(*f.arena,mw);
    allocation_watch_start();auto *tables=pq16::prepare(tf,shape);assert(!allocation_watch_stop());
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ));
    auto *a=f.guarded(n),*copy=f.guarded(n),*out=f.guarded(2*n),*other=f.guarded(2*n);
    Task square{tables,&sf,a,a,out,n},mul{tables,&mf,a,copy,other,n};
    for(unsigned pattern=0;pattern<6;++pattern){
        for(size_t j=0;j<n;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?0:pattern==2?uint64_t(1)<<63:pattern==3?(j&1?UINT64_MAX:1):random_word();
        memcpy(copy,a,n*8);
        SBN3_FRAME_UNPOISON(sf.data(),sf.capacity());memset(sf.data(),0xff,sf.capacity());SBN3_FRAME_POISON(sf.data(),sf.capacity());
        allocation_watch_start();sbn3_team_run(f.team,execute,&square);sbn3_team_run(f.team,execute,&mul);assert(!allocation_watch_stop());
        assert(!memcmp(out,other,2*n*8) && !memcmp(a,copy,n*8));verify_product(a,n,a,n,out);
        assert(!sf.used() && !mf.used() && sf.peak()<=square_bytes && mf.peak()<=mul_bytes);
    }
    assert(!mprotect(tl.data,up(tl.bytes,4096),PROT_READ|PROT_WRITE));
    printf("FFT SQR n=%zu N=%u M=%u B=%u balanced=%u centered=%u recipe=%u W=%u scratch=%zu/%zu: GMP/alias/guard/alloc PASS\n",n,shape.nfull,shape.radix,shape.bits,shape.balanced,shape.centered,unsigned(shape.recipe),workers,square_bytes,mul_bytes);fflush(stdout);
}
static void public_gate(){
    for(unsigned workers:{1u,3u,16u})for(size_t n:{129u,512u,1000u,8192u,35734u}){
        Fixture f(workers);sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.workers=workers;
        sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=n;sbn3_mul_plan plan{};sbn3_product_info info{};
        auto *binding=f.product(req,o,info,plan);auto *a=f.guarded(n),*out=f.guarded(2*n);
        for(size_t j=0;j<n;++j)a[j]=random_word();sbn3_product_inputs in{};in.a={a,n};
        allocation_watch_start();sbn3_product_execute(binding,&in,{out,2*n});assert(!allocation_watch_stop());verify_product(a,n,a,n,out);
        o.workspace_budget=info.mul.workspace_bytes;assert(sbn3_product_query(&req,&o,&plan,&info)==SBN3_SUPPORTED);
        sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);assert(m.mul.worker_peak_bytes<=info.mul.per_worker_bytes);
    }
}
int main(){
    for(size_t n:{1u,25u,64u,128u,129u,256u,448u,511u,768u,1024u,4096u,8192u,32768u,35734u,49152u,65536u,90000u})
        one(pq16::execution_shape(pq16::query(n,n),1),n,1);
    for(unsigned workers:{3u,16u})for(size_t n:{129u,1024u,8192u,35734u,90000u})one(pq16::execution_shape(pq16::query(n,n),workers),n,workers);
    for(unsigned bits=16;bits<=20;++bits)for(size_t n:{128u,160u,211u,300u,444u,512u,609u,790u,1000u,1800u,3000u,4000u,6000u,8192u}){
        const auto shape=pq16::select(n,n,1,bits,0,true);if(shape.nfull)one(shape,n,1);
    }
    for(size_t n:{32769u,35734u,49152u,57344u})one(pq16::select(n,n,1,16,0,true),n,1);
    public_gate();puts("FFT native square gates PASS");
}
