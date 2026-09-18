#include "product_support.hpp"
#include "algorithms/rsqrt.hpp"
#include "algorithms/sqrt2_finish.hpp"
#include "value/limbs.hpp"
using namespace sbn::v3;
static void check(uint64_t a,const uint64_t *R,size_t n,bool seed=false){
    ref_int r,p,lo,hi;ref_inits(r,p,lo,hi,nullptr);ref_import(r,n+1,-1,8,0,0,R);assert(R[n]==0);
    ref_set_ui(p,1);ref_mul_2exp(p,p,128*n);
    if(seed){ref_mul(lo,r,r);ref_mul_ui(lo,lo,a);ref_add_ui(hi,r,1);ref_mul(hi,hi,hi);ref_mul_ui(hi,hi,a);
        assert(ref_cmp(lo,p)<0 && ref_cmp(hi,p)>=0);
    }else{ref_sub_ui(lo,r,3);ref_mul(lo,lo,lo);ref_mul_ui(lo,lo,a);ref_add_ui(hi,r,3);ref_mul(hi,hi,hi);ref_mul_ui(hi,hi,a);
        if(!(ref_cmp(lo,p)<0 && ref_cmp(hi,p)>0)){fprintf(stderr,"rsqrt >=3 ulps a=%llu n=%zu\n",(unsigned long long)a,n);assert(false);}}
    ref_clears(r,p,lo,hi,nullptr);
}
static RsqrtRung prepare(Fixture &f,size_t m,size_t n,unsigned np=6,unsigned algorithm=0){
    sbn3_mul_options o{};o.workers=sbn3_team_workers(f.team);o.prime_count=np;o.borrow_output=1;
    o.algorithm=algorithm?algorithm:n<=8192?SBN3_MUL_FLAT:SBN3_MUL_BAILEY;
    sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=m;
    sbn3_mul_plan plan{};sbn3_product_info info{};bool found=false;
    for(int T=np==4?88:24*int(np)-8;T>=(np==4?80:24*int(np)-32);T-=np==4?4:8){
        size_t ring=2*size_t(T);while(ring<2*m+2)ring*=2;o.trunk_bits=T;req.cyclic_limbs=ring;
        if(sbn3_product_query(&req,&o,&plan,&info)==SBN3_SUPPORTED){found=true;break;}}
    assert(found);sbn3_spectrum_desc future{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,n,&future)==SBN3_SUPPORTED);
    const size_t producer_leases=f.leases.size();
    auto *producer=f.product(req,o,info,plan);auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *spectrum=nullptr;
    sbn3_spectrum_reserve(producer,SBN3_SPECTRUM_COLUMNS,n,f.arena,&storage,&spectrum);req.cached_a[0]=&future;
    auto *square=f.product(req,o,info,plan,spectrum);req.kind=SBN3_PRODUCT_MUL;req.b_limbs=m+1;
    auto *multiply=f.product(req,o,info,plan,spectrum);f.unbind(producer);
    f.retire(producer_leases+1);f.retire(producer_leases);
    assert(!sbn3_spectrum_can_apply(&plan,spectrum,0));
    return {m,n,req.cyclic_limbs,square,multiply,spectrum,f.guarded(req.cyclic_limbs),f.guarded(req.cyclic_limbs)};
}
static void seed_gate(){
    for(size_t n=1;n<=3;++n)for(unsigned j=0;j<768;++j){
        uint64_t a;
        if(j<64)a=uint64_t(1)<<j;
        else if(j<128)a=(uint64_t(1)<<(j-64))+1;
        else if(j<192)a=(uint64_t(1)<<(j-128))-1;
        else if(j<384){const uint64_t v=uint32_t(random_word());a=v*v;if(j%3==0)--a;else if(j%3==1)++a;}
        else a=random_word();if(!a)a=1;uint64_t r[4]{};
        allocation_watch_start();rsqrt_seed(r,a,n);assert(!allocation_watch_stop());check(a,r,n,true);}
    for(uint64_t a:{uint64_t(1)<<32,UINT64_MAX})for(size_t n=1;n<=3;++n){uint64_t r[4]{};rsqrt_seed(r,a,n);check(a,r,n,true);}
    puts("rsqrt integer seed n=1..3, full u64 radicand range PASS");
}
static void rung_gate(uint64_t a,int delta,unsigned np,unsigned algorithm,unsigned workers){
    const size_t m=3,n=5;Fixture f(workers);auto step=prepare(f,m,n,np,algorithm);
    auto *r=f.guarded(m+1),*out=f.guarded(n+1);rsqrt_seed(r,a,m);
    const uint64_t change=unsigned(delta<0?-delta:delta);
    if(delta<0)limbs::sub_from(r,m+1,&change,1);else limbs::add_to(r,m+1,&change,1);
    if(r[m]){memset(r,0xff,m*8);r[m]=0;}
    allocation_watch_start();rsqrt_rung(step,a,r,out);assert(!allocation_watch_stop());check(a,out,n);
    sbn3_product_metrics s{},p{};sbn3_product_get_metrics(step.square,&s);sbn3_product_get_metrics(step.multiply,&p);
    assert(s.row_forward>0 && p.row_forward>0);
    sbn3_spectrum_release(step.r);
}
static void program_gate(size_t target,uint64_t a,unsigned workers){
    Fixture f(workers);sbn3_mul_binding *checker=nullptr;uint64_t *z=nullptr,*square=nullptr;const size_t digits=target>=3?target-2:0;
    if(a==2 && digits){sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=digits+1;
        sbn3_mul_options o{};o.workers=workers;sbn3_product_info info{};sbn3_mul_plan plan{};checker=f.product(req,o,info,plan);
        z=f.guarded(digits+1);square=f.guarded(up(info.mul.output_limbs,8));}
    std::vector<size_t> sizes;size_t m=target;
    while(m>3){sizes.push_back(m);m=m/2+1;}
    const size_t seed=m;std::vector<RsqrtRung> rungs;rungs.reserve(sizes.size());
    for(size_t j=sizes.size();j-- >0;){rungs.push_back(prepare(f,m,sizes[j]));m=sizes[j];}
    auto *out=f.guarded(target+1),*v0=f.guarded(target+1),*v1=f.guarded(target+1);
    RsqrtProgram p{target,seed,rungs.size(),rungs.data(),{v0,v1}};
    allocation_watch_start();rsqrt_program(p,a,out);
    if(checker){sqrt2_from_rsqrt(z,digits,out,target);sqrt2_certify(checker,z,digits,square);}
    assert(!allocation_watch_stop());check(a,out,target);
    if(checker){
        ref_int want,got,sq;ref_inits(want,got,sq,nullptr);ref_set_ui(want,1);ref_mul_2exp(want,want,128*digits+1);ref_sqrt(want,want);
        ref_import(got,digits+1,-1,8,0,0,z);assert(ref_cmp(got,want)==0);ref_mul(want,got,got);
        ref_import(sq,2*digits+2,-1,8,0,0,square);assert(ref_cmp(sq,want)==0);ref_clears(want,got,sq,nullptr);
        printf("sqrt2 via 2*rsqrt(2): %zu fractional bits, prefix %llx.%016llx, integer square bound PASS\n",64*digits,(unsigned long long)z[digits],(unsigned long long)z[digits-1]);
    }
    for(const auto &r:rungs)sbn3_spectrum_release(r.r);
    printf("rsqrt program a=%llu n=%zu W%u steps=%zu: <3 ulps/CYC SQR/MUL reuse/preallocation PASS\n",(unsigned long long)a,target,workers,rungs.size());fflush(stdout);
}
int main(){
    seed_gate();
    for(uint64_t a:{uint64_t(1),uint64_t(2),uint64_t(3),uint64_t(4),uint64_t(5),uint64_t(17),uint64_t(1)<<32,UINT64_MAX})
        for(int delta:{-2,0,2})rung_gate(a,delta,6,SBN3_MUL_FLAT,1);
    for(unsigned np:{4u,8u,9u,10u})rung_gate(2,2,np,SBN3_MUL_FLAT,3);
    for(size_t n:{1u,2u,3u,4u,7u,16u,63u,128u,1024u,8192u,65536u})program_gate(n,2,1);
    for(uint64_t a:{uint64_t(1),uint64_t(3),uint64_t(4),uint64_t(17),UINT64_MAX})program_gate(129,a,1);
    program_gate(8192,2,16);program_gate(65536,2,16);
    puts("small-radicand reciprocal square root gates PASS");
}
