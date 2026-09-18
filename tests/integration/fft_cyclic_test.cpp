#include "product_support.hpp"
static void verify(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *out,const sbn3_product_info &info){
    ref_int x,y,want,got,mod,diff;ref_inits(x,y,want,got,mod,diff,nullptr);
    ref_import(x,an,-1,8,0,0,a);ref_import(y,bn,-1,8,0,0,b);ref_mul(want,x,y);ref_import(got,info.mul.output_limbs,-1,8,0,0,out);
    if(info.kind==SBN3_PRODUCT_TMP){ref_fdiv_q_2exp(want,want,info.window.offset_bits);ref_sub(diff,got,want);ref_fdiv_r_2exp(diff,diff,info.window.width_bits);assert(ref_cmp_ui(diff,1)<=0);}
    else{ref_set_ui(mod,1);ref_mul_2exp(mod,mod,64*info.cyclic_limbs);ref_sub_ui(mod,mod,1);ref_mod(want,want,mod);assert(ref_cmp(want,got)==0 && ref_cmp(got,mod)<0);}
    ref_clears(x,y,want,got,mod,diff,nullptr);
}
static void scenario(size_t ring,size_t an,size_t bn,unsigned workers,bool square,bool mid,bool cached,bool automatic=false){
    Fixture f(workers);sbn3_mul_options o{};o.workers=workers;o.algorithm=automatic?SBN3_MUL_AUTO:SBN3_MUL_PQ16;
    sbn3_product_request req{};req.kind=mid?SBN3_PRODUCT_TMP:square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;req.a_limbs=an;req.b_limbs=square?0:bn;req.cyclic_limbs=mid?0:ring;
    auto *a=f.guarded(an),*b=f.guarded(bn);for(size_t j=0;j<an;++j)a[j]=UINT64_MAX;
    sbn3_mul_plan plan{};sbn3_product_info info{};auto *binding=f.product(req,o,info,plan);
    if(automatic)assert(plan.opaque[1]==101);
    sbn3_spectrum *cache=nullptr;sbn3_spectrum_desc desc{};
    if(cached){cache=f.cache(binding,{a,an},1,info,desc);req.cached_a[0]=&desc;f.unbind(binding);binding=f.product(req,o,info,plan,cache);}
    auto *out=f.guarded(up(info.mul.output_limbs,8));
    for(unsigned pattern=0;pattern<8;++pattern){
        if(!cached)for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?0:pattern==2?(j+1==an?uint64_t(1)<<63:0):pattern==3?(j%2?0:UINT64_MAX):random_word();
        for(size_t j=0;j<bn;++j)b[j]=pattern==0?UINT64_MAX:pattern==1?0:pattern==2?(j+1==bn?uint64_t(1)<<63:0):pattern==3?(j%2?UINT64_MAX:0):random_word();
        sbn3_product_inputs input{};if(!cached)input.a={a,an};if(!square)input.b={b,bn};
        allocation_watch_start();sbn3_product_execute(binding,&input,{out,info.mul.output_limbs});assert(!allocation_watch_stop());
        verify(a,an,square?a:b,square?an:bn,out,info);
        sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);assert(m.row_forward==(cached?(square?0u:1u):(square?1u:2u)) && m.row_inverse==1);
        assert(m.mul.worker_peak_bytes<=info.mul.per_worker_bytes);
    }
    if(cache)sbn3_spectrum_release(cache);
    printf("FFT %s ring=%zu %zu x %zu W%u cache=%u: reference/modular oracle/range/carry/alloc/counts PASS\n",mid?"TMP":square?"CYC SQR":"CYC MUL",info.mul.transform_trunks/4,an,bn,workers,cached);fflush(stdout);
}
int main(){
    for(size_t ring:{64u,128u,320u,384u,448u,512u,1024u,1280u,1536u,2560u,8192u,16384u,32768u}){
        for(bool cache:{false,true}){scenario(ring,ring/2,ring,1,false,false,cache);scenario(ring,ring/2,ring/2,1,true,false,cache);}
    }
    for(unsigned w:{3u,16u})for(size_t ring:{512u,1536u,8192u,32768u})scenario(ring,ring/2,ring,w,false,false,true);
    for(size_t an:{1u,16u,65u,128u,513u,2048u,8192u})for(bool cache:{false,true})scenario(0,an,2*an+7,1,false,true,cache);
    scenario(0,4000,16000,16,false,true,true);
    scenario(0,400,801,1,false,true,false,true);
    puts("FFT cyclic and middle-window gates PASS");
}
