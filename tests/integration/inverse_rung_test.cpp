#include "product_support.hpp"
#include "algorithms/inverse_rung.hpp"
#include "algorithms/inverse_program.hpp"
#include "algorithms/inverse_seed.hpp"
using namespace sbn::v3;

static void seed(uint64_t *D,uint64_t *U,size_t n,size_t m,unsigned pattern,int delta){
    ref_int d,u,h,power,lo,hi,error;ref_inits(d,u,h,power,lo,hi,error,nullptr);
    for(size_t j=0;j<n;++j)D[j]=pattern==1?UINT64_MAX:pattern==2||pattern==3?0:pattern==4?(j&1?UINT64_MAX:0):random_word();
    D[n-1]|=uint64_t(1)<<63;if(pattern==3)D[0]=1;
    ref_import(d,n,-1,8,0,0,D);ref_set_ui(lo,1);ref_mul_2exp(lo,lo,64*m);ref_mul_2exp(hi,lo,1);ref_sub_ui(hi,hi,1);
    if(pattern==5){ref_add_ui(u,lo,37);ref_set_ui(power,1);ref_mul_2exp(power,power,64*(n+m));ref_fdiv_q(d,power,u);
        memset(D,0,n*8);size_t written;ref_export(D,&written,-1,8,0,0,d);assert(written<=n);}
    ref_fdiv_q_2exp(h,d,64*(n-m));ref_set_ui(power,1);ref_mul_2exp(power,power,128*m);
    if(pattern!=5)ref_fdiv_q(u,power,h);
    if(delta<0)ref_sub_ui(u,u,unsigned(-delta));else ref_add_ui(u,u,unsigned(delta));
    if(ref_cmp(u,lo)<0)ref_set(u,lo);if(ref_cmp(u,hi)>0)ref_set(u,hi);
    ref_mul(error,u,h);ref_sub(error,error,power);ref_abs(error,error);ref_mul_ui(h,h,8);assert(ref_cmp(error,h)<=0);
    memset(U,0,(m+1)*8);size_t written;ref_export(U,&written,-1,8,0,0,u);assert(written<=m+1 && U[m]==1);
    ref_clears(d,u,h,power,lo,hi,error,nullptr);
}
static void check(const uint64_t *D,const uint64_t *V,size_t n){
    ref_int d,v,e,bound;ref_inits(d,v,e,bound,nullptr);
    ref_import(d,n,-1,8,0,0,D);ref_import(v,n+1,-1,8,0,0,V);ref_mul(e,d,v);
    ref_set_ui(bound,1);ref_mul_2exp(bound,bound,128*n);ref_sub(e,e,bound);ref_abs(e,e);ref_mul_ui(bound,d,3);
    if(ref_cmp(e,bound)>=0){fprintf(stderr,"inverse rung error >= 3 ulps at n=%zu\n",n);assert(false);}
    assert(V[n]==1);ref_clears(d,v,e,bound,nullptr);
}
static void run(unsigned np,int T,unsigned algorithm,unsigned workers,unsigned pattern,int delta,unsigned frontier=1){
    const size_t ring=(algorithm==SBN3_MUL_FLAT?size_t(256):size_t(1024))*T/8;
    const size_t n=ring-32,m=n/2+1;Fixture f(workers);
    sbn3_mul_options o{};o.workers=workers;o.prime_count=np;o.trunk_bits=T;o.algorithm=algorithm;o.borrow_output=1;
    if(algorithm==SBN3_MUL_BAILEY){o.column_log2=6;o.row_log2=4;}
    sbn3_product_request req{};req.a_limbs=m+1;req.b_limbs=n;req.cyclic_limbs=ring;
    sbn3_mul_plan fresh_plan{},cached_plan{};sbn3_product_info fresh_info{},cached_info{};sbn3_spectrum_desc future{};
    allocation_watch_start();
    assert(sbn3_product_query(&req,&o,&fresh_plan,&fresh_info)==SBN3_SUPPORTED);
    assert(sbn3_spectrum_query(&fresh_plan,static_cast<sbn3_spectrum_frontier>(frontier),19,&future)==SBN3_SUPPORTED);
    assert(!future.instance_id);auto cached_req=req;cached_req.cached_a[0]=&future;
    assert(sbn3_product_query(&cached_req,&o,&cached_plan,&cached_info)==SBN3_SUPPORTED);
    assert(!allocation_watch_stop());
    // All consumer planning precedes resource allocation and source values.
    auto *producer=f.product(req,o,fresh_info,fresh_plan);
    const auto cache_storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *spectrum=nullptr;
    allocation_watch_start();sbn3_spectrum_reserve(producer,static_cast<sbn3_spectrum_frontier>(frontier),19,f.arena,&cache_storage,&spectrum);assert(!allocation_watch_stop());
    auto *product=f.product(cached_req,o,cached_info,cached_plan,spectrum);
    assert(!sbn3_spectrum_can_apply(&cached_plan,spectrum,0));f.unbind(producer);
    auto *D=f.guarded(n),*U=f.guarded(m+1),*V=f.guarded(n+1),*R=f.guarded(ring),*C=f.guarded(ring);
    seed(D,U,n,m,pattern,delta);std::vector<uint64_t> oldD(D,D+n),oldU(U,U+m+1);
    InverseRung step{m,n,ring,product,spectrum,R,C};sbn3_product_metrics a{},b{};
    allocation_watch_start();inverse_rung(step,D,U,V,&a,&b);assert(!allocation_watch_stop());
    check(D,V,n);assert(!memcmp(D,oldD.data(),n*8) && !memcmp(U,oldU.data(),(m+1)*8));
    assert(sbn3_spectrum_can_apply(&cached_plan,spectrum,0));
    const auto &g=cached_info.mul;const uint64_t rows=np*(algorithm==SBN3_MUL_FLAT?1:g.C);
    assert(a.row_forward==2*rows && b.row_forward==rows); // A build + B, then only B
    if(algorithm==SBN3_MUL_BAILEY)assert(a.column_forward==2*np*g.lbw && b.column_forward==np*g.lbw*(frontier?1:2));
    assert(a.mul.worker_peak_bytes<=g.per_worker_bytes && b.mul.workspace_used_bytes<=g.workspace_bytes);
    sbn3_spectrum_release(spectrum);
    printf("inverse rung n=%zu m=%zu ring=%zu NP%u T%d W%u h%u pattern%u delta%d: <3 ulps/native CYC/reuse/preallocation PASS\n",n,m,ring,np,T,workers,frontier,pattern,delta);fflush(stdout);
}
static void incomplete_gate(){
    const auto child=fork();assert(child>=0);
    if(child==0){
        Fixture f(1);sbn3_mul_options o{};o.workers=1;o.prime_count=6;o.trunk_bits=128;o.algorithm=SBN3_MUL_FLAT;
        sbn3_product_request req{};req.a_limbs=64;req.b_limbs=128;req.cyclic_limbs=256;sbn3_mul_plan plan{};sbn3_product_info info{};
        auto *producer=f.product(req,o,info,plan);auto storage=f.allocate(info.spectrum_bytes,64);sbn3_spectrum *s=nullptr;
        sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,7,f.arena,&storage,&s);
        sbn3_spectrum_desc d{};sbn3_spectrum_describe(s,&d);req.cached_a[0]=&d;
        auto *consumer=f.product(req,o,info,plan,s);auto *b=f.guarded(128),*r=f.guarded(256);memset(b,0,128*8);
        sbn3_product_inputs in{};in.b={b,128};sbn3_product_execute(consumer,&in,{r,256});_exit(11);
    }
    int status=0;assert(waitpid(child,&status,0)==child);assert(WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
}
static void seed_gate(){
    for(size_t n=1;n<=15;++n)for(unsigned trial=0;trial<80;++trial){
        uint64_t d[15]{},u[16]{};
        for(size_t j=0;j<n;++j)d[j]=trial==0?0:trial==1?UINT64_MAX:random_word();d[n-1]|=uint64_t(1)<<63;
        allocation_watch_start();inverse_seed(u,d,n);assert(!allocation_watch_stop());check(d,u,n);
    }
    puts("v2 BMI2/ADX inverse seed: n=1..15, 1200 cases, <3 ulps PASS");
}
static void program_gate(size_t target,unsigned workers,unsigned pattern){
    Fixture f(workers);std::vector<size_t> sizes;size_t m=target;
    while(m>15){sizes.push_back(m);m=m/2+1;}
    const size_t seed_size=m;std::vector<InverseRung> steps;steps.reserve(sizes.size());std::vector<sbn3_spectrum *> handles;
    for(size_t j=sizes.size();j-- >0;){const size_t n=sizes[j];
        sbn3_mul_options o{};o.workers=workers;o.prime_count=6;o.borrow_output=1;o.algorithm=n<=8192?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
        sbn3_product_request req{};req.a_limbs=m+1;req.b_limbs=n;
        sbn3_mul_plan producer_plan{},consumer_plan{};sbn3_product_info pi{},ci{};bool found=false;
        // This test supplies explicit legal periods. The library executor
        // consumes preplanned stages; the joint cost chooser is separate.
        for(int T:{136,128,120,112}){size_t ring=2*size_t(T);while(ring<n+4)ring*=2;
            o.trunk_bits=T;req.cyclic_limbs=ring;
            if(sbn3_product_query(&req,&o,&producer_plan,&pi)==SBN3_SUPPORTED){found=true;break;}}
        assert(found);sbn3_spectrum_desc future{};
        assert(sbn3_spectrum_query(&producer_plan,SBN3_SPECTRUM_COLUMNS,n,&future)==SBN3_SUPPORTED);
        auto *producer=f.product(req,o,pi,producer_plan);auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *cache=nullptr;
        sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,n,f.arena,&storage,&cache);
        req.cached_a[0]=&future;auto *product=f.product(req,o,ci,consumer_plan,cache);f.unbind(producer);
        handles.push_back(cache);steps.push_back({m,n,req.cyclic_limbs,product,cache,f.guarded(req.cyclic_limbs),f.guarded(req.cyclic_limbs)});
        m=n;
    }
    auto *d=f.guarded(target),*out=f.guarded(target+1),*v0=f.guarded(target+1),*v1=f.guarded(target+1);
    for(size_t j=0;j<target;++j)d[j]=pattern==0?UINT64_MAX:pattern==1?0:random_word();d[target-1]|=uint64_t(1)<<63;
    std::vector<uint64_t> before(d,d+target);const InverseProgram program{target,seed_size,steps.size(),steps.data(),{v0,v1}};
    allocation_watch_start();inverse_program(program,d,out);assert(!allocation_watch_stop());check(d,out,target);
    assert(!memcmp(d,before.data(),target*8));for(auto *h:handles)sbn3_spectrum_release(h);
    printf("inverse program n=%zu W%u steps=%zu pattern%u: allocation-free execution/<3 ulps/all tables prebuilt PASS\n",target,workers,steps.size(),pattern);fflush(stdout);
}
int main(){
    incomplete_gate();seed_gate();
    for(unsigned pattern=0;pattern<6;++pattern)for(int delta:{-7,0,7}){
        run(6,128,SBN3_MUL_FLAT,1,pattern,delta);
        run(8,176,SBN3_MUL_BAILEY,3,pattern,delta);
    }
    run(4,80,SBN3_MUL_BAILEY,3,4,7,0);run(6,136,SBN3_MUL_FLAT,3,5,0,0);
    run(9,208,SBN3_MUL_BAILEY,16,0,-7);run(10,224,SBN3_MUL_BAILEY,16,4,7);
    for(size_t n:{1u,2u,7u,15u,16u,31u,64u,129u,1024u,8192u,65536u})for(unsigned p=0;p<3;++p)program_gate(n,1,p);
    program_gate(8192,16,2);program_gate(65536,16,2);
    puts("preplanned native inverse rung gates PASS");
}
