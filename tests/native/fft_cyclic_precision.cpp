#include <cstddef>
static void coefficient(size_t,const double *,const double *);
#define PQ16_COEFFICIENT_OBSERVER ::coefficient
#include "backend/pq16/island.hpp"
#include "../integration/product_support.hpp"
#include <cmath>
using namespace sbn::v3;
using namespace sbn::v3::pq16;
static std::vector<uint64_t> exact;
static std::vector<unsigned char> seen;
static double maximum;
static void coefficient(size_t limb,const double *a,const double *b){
    for(unsigned t=0;t<2;++t)for(unsigned lane=0;lane<8;++lane)for(unsigned part=0;part<2;++part){
        const size_t at=4*limb+16*t+2*lane+part;const double x=(t?b:a)[lane+8*part];
        assert(at<exact.size()&&!seen[at]++&&std::isfinite(x));maximum=std::max(maximum,std::abs(x-double(exact[at])));
    }
}
static void check(unsigned M,unsigned branch,unsigned pattern){
    const unsigned N=M*branch;const size_t ring=N/2,an=ring/2,bn=ring;Fixture f(1);
    const Shape shape{N,branch,M,false,Recipe::PfaPQ,16,false};auto tl=f.allocate(table_bytes(shape),128);Frame tables(*f.arena,tl);
    pq16_plan plan{};plan.builder=&tables;pq16_plan_ensure(&plan,branch,M!=1);
    auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring);auto al=f.allocate(16*size_t(N),128),bl=f.allocate(16*size_t(N),128);
    for(unsigned side=0;side<2;++side){auto *p=side?b:a;const size_t size=side?bn:an;
        for(size_t j=0;j<size;++j)p[j]=pattern==0?UINT64_MAX:pattern==1?random_word():pattern==2?(j%2?0:UINT64_MAX):pattern==3?(j==0||j+1==size?UINT64_MAX:0):
            pattern==4?UINT64_C(0x8000800080008000):pattern==5?UINT64_C(0x0000ffff0000ffff):pattern==6?(uint64_t(1)<<63):random_word();}
    std::vector<uint64_t> da(4*an),db(4*bn),full(da.size()+db.size());
    for(size_t j=0;j<da.size();++j)da[j]=(a[j/4]>>(16*(j%4)))&65535;
    for(size_t j=0;j<db.size();++j)db[j]=(b[j/4]>>(16*(j%4)))&65535;
    ref_int A,B,P;ref_inits(A,B,P,nullptr);ref_import(A,da.size(),-1,8,0,0,da.data());ref_import(B,db.size(),-1,8,0,0,db.data());ref_mul(P,A,B);
    size_t count=0;ref_export(full.data(),&count,-1,8,0,0,P);ref_clears(A,B,P,nullptr);
    exact.assign(2*N,0);for(size_t j=0;j<full.size();++j)exact[j%(2*N)]+=full[j];seen.assign(2*N,0);maximum=0;
    auto *x=static_cast<double *>(al.data),*y=static_cast<double *>(bl.data);q_cctx unused{};
    allocation_watch_start();
    if(M==1){pq16_input_stage_w(x,a,an,branch,&plan,0,nullptr,1);pq16_fwd_core_w(x,branch,&plan,nullptr,1);
        pq16_input_stage_w(y,b,bn,branch,&plan,0,nullptr,1);pq16_fwd_core_w(y,branch,&plan,nullptr,1);}
    else{pq16_pfa_fwd_w(x,a,an,branch,M,&plan,0,nullptr,1);pq16_pfa_fwd_w(y,b,bn,branch,M,&plan,0,nullptr,1);}
    const bool ok=pq16_conv_emit_w(out,ring,x,y,branch,M,N,0,1,&unused,nullptr,&plan,nullptr,1);
    assert(!allocation_watch_stop()&&ok);for(auto v:seen)assert(v==1);
    printf("FFT cyclic coefficient gate N=%u M=%u pattern=%u max_error=%.9g\n",N,M,pattern,maximum);fflush(stdout);assert(maximum<.25);
}
int main(){for(unsigned pattern=0;pattern<8;++pattern){check(1,65536,pattern);check(3,16384,pattern);check(5,8192,pattern);check(7,4096,pattern);}puts("FFT cyclic precision margin PASS");}
