// Explicit scalar squares against the oracle: sqr_basecase_assumed for n = 0..20
// (straight-line kernels 2..14, the product kernels beyond), and SQR product
// bindings of 1..16 limbs, automatic (scalar up to 14) and forced scalar, which
// select the square kernel at bind time. Data: all ones, random, sparse, high
// zero limbs; guarded output end, untouched words past 2n, no allocation.
#include "product_support.hpp"
#include "core/x86_64/word.hpp"
using namespace sbn::v3;
static void fill(uint64_t *a,size_t n,unsigned pattern){
    for(size_t j=0;j<n;++j){
        const uint64_t w=random_word();
        a[j]=pattern==0?UINT64_MAX:pattern==2?((w&7)?0:random_word()):w;
    }
    if(pattern==3)for(size_t j=n/2;j<n;++j)a[j]=0;
}
static void binding(Fixture &f,size_t n,unsigned alg){
    sbn3_mul_options o{};o.workers=1;o.algorithm=alg;o.borrow_output=1;
    sbn3_product_request req{};req.kind=SBN3_PRODUCT_SQR;req.a_limbs=n;
    sbn3_product_info i{};sbn3_mul_plan p{};auto *b=f.product(req,o,i,p);
    if(alg || n<=14)assert(i.mul.algorithm==SBN3_MUL_SCALAR);
    const size_t rn=up(2*n,i.mul.output_alignment/8);
    auto *a=f.guarded(n),*r=f.guarded(rn);
    for(unsigned pattern=0;pattern<4;++pattern){
        fill(a,n,pattern);memset(r,0x91,rn*8);
        sbn3_product_inputs in{{a,n},{},{},{}};
        allocation_watch_start();sbn3_product_execute(b,&in,{r,2*n});assert(!allocation_watch_stop());
        verify_product(a,n,a,n,r);
        for(size_t j=2*n;j<rn;++j)assert(r[j]==0x9191919191919191ULL);
    }
}
int main(){
    size_t direct=0,bound=0;
    {
        std::vector<uint64_t> a(20),r(44);
        for(size_t n=0;n<=20;++n)for(unsigned pattern=0;pattern<4;++pattern)for(int rep=0;rep<200;++rep){
            fill(a.data(),n,pattern);
            for(auto &w:r)w=0x5a5a5a5a5a5a5a5aull;
            allocation_watch_start();sqr_basecase_assumed(r.data(),a.data(),n);assert(!allocation_watch_stop());
            if(n)verify_product(a.data(),n,a.data(),n,r.data());
            for(size_t j=2*n;j<r.size();++j)assert(r[j]==0x5a5a5a5a5a5a5a5aull);
            ++direct;
        }
    }
    Fixture f(1);
    for(size_t n=1;n<=16;++n){binding(f,n,SBN3_MUL_AUTO);++bound;if(n<=14){binding(f,n,SBN3_MUL_SCALAR);++bound;}}
    printf("{\"status\":\"passed\",\"direct_checks\":%zu,\"bindings\":%zu}\n",direct,bound);
}
