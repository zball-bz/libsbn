#include "product_support.hpp"
#include <sys/mman.h>
static void scenario(size_t ring,size_t an,size_t bn,bool square){
    Fixture f(1);sbn3_product_request r{};r.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;r.a_limbs=an;r.b_limbs=square?0:bn;r.cyclic_limbs=ring;
    sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_SCALAR;sbn3_product_info i{};sbn3_mul_plan p{};auto *binding=f.product(r,o,i,p);assert(i.cyclic_limbs==ring&&i.mul.output_limbs==ring&&i.mul.algorithm==SBN3_MUL_SCALAR);
    auto *a=f.guarded(an?an:1),*b=f.guarded(bn?bn:1),*out=f.guarded(ring);
    ref_int A,B,P,R,M;ref_inits(A,B,P,R,M,nullptr);ref_set_ui(M,1);ref_mul_2exp(M,M,64*ring);ref_sub_ui(M,M,1);
    for(unsigned pattern=0;pattern<8;++pattern){
        for(size_t j=0;j<an;++j)a[j]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(j+1==an?UINT64_MAX:0):pattern==3?(j%2?0:UINT64_MAX):random_word();
        for(size_t j=0;j<bn;++j)b[j]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(j==0?UINT64_MAX:0):pattern==3?(j%2?UINT64_MAX:0):random_word();
        sbn3_product_inputs in{};in.a={a,an};if(!square)in.b={b,bn};
        assert(!mprotect((void *)(uintptr_t(a)&~uintptr_t(4095)),4096,PROT_READ)&&!mprotect((void *)(uintptr_t(b)&~uintptr_t(4095)),4096,PROT_READ));allocation_watch_start();sbn3_product_execute(binding,&in,{out,ring});assert(!allocation_watch_stop());
        assert(!mprotect((void *)(uintptr_t(a)&~uintptr_t(4095)),4096,PROT_READ|PROT_WRITE)&&!mprotect((void *)(uintptr_t(b)&~uintptr_t(4095)),4096,PROT_READ|PROT_WRITE));
        ref_import(A,an,-1,8,0,0,a);ref_import(B,square?an:bn,-1,8,0,0,square?a:b);ref_mul(P,A,B);ref_mod(P,P,M);ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0&&ref_cmp(R,M)<0);
        sbn3_product_metrics metrics{};sbn3_product_get_metrics(binding,&metrics);assert(!metrics.row_forward&&!metrics.row_inverse&&!metrics.mul.worker_peak_bytes);
    }
    ref_clears(A,B,P,R,M,nullptr);
}
int main(){
    for(size_t r:{1u,2u,3u,5u,16u,31u,64u,129u,256u,512u}){
        scenario(r,r,r,false);scenario(r,r,r,true);scenario(r,(r+1)/2,r,false);scenario(r,r,1,false);scenario(r,0,r,false);
    }
    puts("scalar native cyclic MUL/SQR: arbitrary small rings, seam/long carry, canonical zero, reference/modular oracle, guards, zero scratch/alloc PASS");
}
