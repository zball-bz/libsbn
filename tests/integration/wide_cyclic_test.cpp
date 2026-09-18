#include "product_support.hpp"
#include "backend/pq16/kernels.hpp"
using namespace sbn::v3;
static void scenario(unsigned bits,size_t minimum,bool square,bool cached){
    const auto shape=pq16::cyclic_shape(minimum,bits);if(!shape.nfull)return;const size_t ring=pq16::cyclic_period(shape),an=ring/2-1,bn=ring-1;
    Fixture f(1);sbn3_product_request req{};req.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;req.a_limbs=an;req.b_limbs=square?0:bn;req.cyclic_limbs=ring;
    sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=int(bits);sbn3_product_info i{};sbn3_mul_plan p{};auto *binding=f.product(req,o,i,p);
    auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring);for(size_t j=0;j<an;++j)a[j]=UINT64_MAX;
    sbn3_spectrum *cache=nullptr;sbn3_spectrum_desc d{};
    if(cached){cache=f.cache(binding,{a,an},1,i,d);req.cached_a[0]=&d;f.unbind(binding);binding=f.product(req,o,i,p,cache);}
    ref_int A,B,P,R,M;ref_inits(A,B,P,R,M,nullptr);ref_set_ui(M,1);ref_mul_2exp(M,M,64*ring);ref_sub_ui(M,M,1);
    for(unsigned pattern=0;pattern<8;++pattern){
        if(!cached)for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?0:pattern==2?(j%2?0:UINT64_MAX):random_word();
        for(size_t j=0;j<bn;++j)b[j]=pattern==0?UINT64_MAX:pattern==1?0:pattern==2?(j%2?UINT64_MAX:0):random_word();
        sbn3_product_inputs in{};if(!cached)in.a={a,an};if(!square)in.b={b,bn};allocation_watch_start();sbn3_product_execute(binding,&in,{out,ring});assert(!allocation_watch_stop());
        ref_import(A,an,-1,8,0,0,a);ref_import(B,square?an:bn,-1,8,0,0,square?a:b);ref_mul(P,A,B);ref_mod(P,P,M);ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0 && ref_cmp(R,M)<0);
        sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);assert(m.mul.worker_peak_bytes<=i.mul.per_worker_bytes&&m.row_forward==(cached?(square?0:1u):(square?1:2u))&&m.row_inverse==1);
    }
    if(cache)sbn3_spectrum_release(cache);ref_clears(A,B,P,R,M,nullptr);
    printf("wide cyclic B%u ring=%zu N%u M%u balanced=%u square=%u cache=%u PASS\n",bits,ring,shape.nfull,shape.radix,shape.balanced,square,cached);fflush(stdout);
}
int main(){for(unsigned bits=17;bits<=20;++bits)for(size_t min:{64u,128u,256u,512u,1024u,2048u,4096u,8192u})for(bool square:{false,true})for(bool cached:{false,true})scenario(bits,min,square,cached);puts("wide cyclic integer/carry/cache gates PASS");}
