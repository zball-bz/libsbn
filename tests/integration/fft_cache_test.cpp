#include "product_support.hpp"
#include <sys/mman.h>

static uint64_t payload_hash(const sbn3_spectrum *s,const sbn3_spectrum_desc &d){
    const auto *p=reinterpret_cast<const unsigned char *>(s);uint64_t h=1;
    for(size_t j=1024;j<d.storage_bytes;++j)h=(h^p[j])*1099511628211ULL;return h;
}
static void scenario(size_t an,size_t bn,unsigned workers,unsigned bits,bool via_producer,unsigned pattern){
    Fixture f(workers);sbn3_product_request initial{};initial.a_limbs=initial.b_limbs=an;
    sbn3_mul_options o{};o.workers=workers;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=int(bits);
    sbn3_mul_plan producer_plan{};sbn3_product_info producer_info{};
    if(sbn3_product_query(&initial,&o,&producer_plan,&producer_info)!=SBN3_SUPPORTED)return;
    sbn3_spectrum_desc future{};allocation_watch_start();assert(sbn3_spectrum_query(&producer_plan,SBN3_SPECTRUM_COLUMNS,73,&future)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
    assert(future.backend_id==101 && future.format_version==3 && !future.np && !future.instance_id);
    sbn3_product_request req{};req.a_limbs=an;req.b_limbs=bn;req.cached_a[0]=&future;
    sbn3_mul_plan plan{};sbn3_product_info info{};const auto status=sbn3_product_query(&req,&o,&plan,&info);
    if(status!=SBN3_SUPPORTED)return; // e.g. a wide codec's admitted minimum size
    auto *a=f.guarded(an),*b=f.guarded(bn);for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?(uint64_t(1)<<63):random_word();
    std::vector<uint64_t> original(a,a+an);
    sbn3_spectrum *s=nullptr;sbn3_spectrum_desc actual{};
    if(via_producer){auto *p=f.product(initial,o,producer_info,producer_plan);s=f.cache(p,{a,an},1,producer_info,actual); // fixture generation 7
        f.unbind(p);future=actual;req.cached_a[0]=&future;
    }else{const auto storage=f.allocate(future.storage_bytes,128);allocation_watch_start();sbn3_spectrum_reserve_plan(&producer_plan,SBN3_SPECTRUM_COLUMNS,73,f.arena,&storage,&s);assert(!allocation_watch_stop());}
    auto *mul=f.product(req,o,info,plan,s);assert(plan.opaque[1]==101 && info.cached_mask==1);
    if(!via_producer){assert(!sbn3_spectrum_can_apply(&plan,s,0));allocation_watch_start();sbn3_spectrum_compute(mul,s,{a,an});assert(!allocation_watch_stop());}
    assert(sbn3_spectrum_can_apply(&plan,s,0));sbn3_spectrum_describe(s,&actual);
    sbn3_product_request square_req=req;square_req.kind=SBN3_PRODUCT_SQR;square_req.b_limbs=0;
    sbn3_mul_plan square_plan{};sbn3_product_info square_info{};auto *square=f.product(square_req,o,square_info,square_plan,s);
    auto *out=f.guarded(up(std::max(an+bn,2*an),8));
    const auto h=payload_hash(s,actual);
    const uintptr_t cache_first=up(reinterpret_cast<uintptr_t>(s)+4096,4096),cache_end=(reinterpret_cast<uintptr_t>(s)+actual.storage_bytes)&~uintptr_t(4095);
    if(cache_end>cache_first)assert(!mprotect(reinterpret_cast<void *>(cache_first),cache_end-cache_first,PROT_READ));
    const uintptr_t source_start=reinterpret_cast<uintptr_t>(a)&~uintptr_t(4095),source_end=up(reinterpret_cast<uintptr_t>(a)+an*8,4096);
    assert(!mprotect(reinterpret_cast<void *>(source_start),source_end-source_start,PROT_NONE));
    for(unsigned round=0;round<3;++round){
        for(size_t j=0;j<bn;++j)b[j]=round==0?UINT64_MAX:round==1?0:random_word();
        sbn3_product_inputs input{};input.b={b,bn};allocation_watch_start();sbn3_product_execute(mul,&input,{out,an+bn});assert(!allocation_watch_stop());
        verify_product(original.data(),an,b,bn,out);
        input={};allocation_watch_start();sbn3_product_execute(square,&input,{out,2*an});assert(!allocation_watch_stop());verify_product(original.data(),an,original.data(),an,out);
    }
    assert(payload_hash(s,actual)==h);sbn3_product_metrics mm{},sm{};sbn3_product_get_metrics(mul,&mm);sbn3_product_get_metrics(square,&sm);
    assert(mm.row_forward==1 && sm.row_forward==0 && mm.row_inverse==1 && sm.row_inverse==1);
    assert(mm.mul.worker_peak_bytes<=info.mul.per_worker_bytes && sm.mul.worker_peak_bytes<=square_info.mul.per_worker_bytes);
    assert(!mprotect(reinterpret_cast<void *>(source_start),source_end-source_start,PROT_READ|PROT_WRITE));
    if(cache_end>cache_first)assert(!mprotect(reinterpret_cast<void *>(cache_first),cache_end-cache_first,PROT_READ|PROT_WRITE));
    auto small=o;small.workspace_budget=info.mul.workspace_bytes-1;sbn3_product_info required{};assert(sbn3_product_query(&req,&small,&plan,&required)==SBN3_QUERY_CAPACITY);
    sbn3_spectrum_release(s);
    printf("FFT cache %zu x %zu W%u B%u codec%u producer=%u: GMP/SQR/read-only/source-retired/alloc/counts PASS\n",an,bn,workers,info.mul.trunk_bits,actual.codec_mode,via_producer);fflush(stdout);
}
static void concurrent(){
    Fixture f(4);const size_t n=1024;sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=16;
    sbn3_product_request req{};req.a_limbs=req.b_limbs=n;sbn3_mul_plan p{};sbn3_product_info i{};assert(sbn3_product_query(&req,&o,&p,&i)==SBN3_SUPPORTED);
    sbn3_spectrum_desc d{};assert(sbn3_spectrum_query(&p,SBN3_SPECTRUM_COLUMNS,9,&d)==SBN3_SUPPORTED);auto storage=f.allocate(d.storage_bytes,128);sbn3_spectrum *s=nullptr;
    sbn3_spectrum_reserve_plan(&p,SBN3_SPECTRUM_COLUMNS,9,f.arena,&storage,&s);req.cached_a[0]=&d;o.workers=2;
    auto *mul=f.product(req,o,i,p,s);req.kind=SBN3_PRODUCT_SQR;req.b_limbs=0;auto *sq=f.product(req,o,i,p,s);
    auto *a=f.guarded(n),*b=f.guarded(n),*x=f.guarded(2*n),*y=f.guarded(2*n);for(size_t j=0;j<n;++j){a[j]=random_word();b[j]=random_word();}
    sbn3_spectrum_compute(mul,s,{a,n});sbn3_spectrum_release(s);
    struct Job{sbn3_mul_binding *binding;sbn3_product_inputs in;sbn3_limbs out;};Job jobs[2]{{mul,{{},{b,n},{},{}},{x,2*n}},{sq,{},{y,2*n}}};
    auto root=[](void *v,sbn3_team_scope *scope){auto *j=static_cast<Job *>(v);auto child=[](void *v,sbn3_team_scope *s){auto &j=*static_cast<Job *>(v);sbn3_product_execute_on_scope(j.binding,s,&j.in,j.out);};
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,child,j,child,j+1);};
    allocation_watch_start();sbn3_team_run(f.team,root,jobs);assert(!allocation_watch_stop());verify_product(a,n,b,n,x);verify_product(a,n,a,n,y);
    puts("FFT cache concurrent MUL/SQR, released user reference, child scopes PASS");
}
int main(){
    concurrent();
    for(unsigned bits=16;bits<=20;++bits)for(size_t n:{128u,211u,512u,790u,1024u,3000u,8192u})scenario(n,n,1,bits,false,2);
    for(size_t n:{64u,129u,448u,1024u,8192u,32769u,49152u,65536u,90000u})for(unsigned p=0;p<2;++p)scenario(n,n,1,0,p!=0,p);
    for(unsigned workers:{3u,16u})for(size_t n:{129u,1024u,8192u,35734u,65536u,90000u})scenario(n,n/2,workers,16,false,0);
    scenario(49152,129,1,16,false,2);scenario(90000,37,3,16,false,2);
    puts("FFT spectrum cache gates PASS");
}
