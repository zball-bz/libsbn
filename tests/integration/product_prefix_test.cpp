#include "product_support.hpp"
#include "backend/pq16/kernels.hpp"
using namespace sbn::v3;
static void scenario(size_t ring,sbn3_mul_options o,bool square,bool cached){
    Fixture f(o.workers);const size_t an=ring/3,bn=ring/3-1;
    auto *a=f.guarded(an),*b=f.guarded(bn);for(size_t j=0;j<an;++j)a[j]=UINT64_MAX;
    sbn3_product_request req{};req.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;req.a_limbs=an;req.b_limbs=square?0:bn;req.cyclic_limbs=ring;
    sbn3_product_info fi{};sbn3_mul_plan fp{};auto *producer=f.product(req,o,fi,fp);
    sbn3_spectrum *cache=nullptr;sbn3_spectrum_desc desc{};
    if(cached){cache=f.cache(producer,{a,an},1,fi,desc);req.cached_a[0]=&desc;}
    ref_int A,B,P,Q,R;ref_inits(A,B,P,Q,R,nullptr);
    for(size_t prefix:{size_t(1),size_t(7),size_t(31),an+2,2*an-1}){
        req.window_limbs=prefix;sbn3_product_info i{};sbn3_mul_plan p{};auto *binding=f.product(req,o,i,p,cache);
        assert(i.mul.output_limbs==prefix && i.window.width_bits==64*prefix && !i.window.error_bits && i.cyclic_limbs==ring);
        auto *out=f.guarded(up(prefix,8));
        for(unsigned pattern=0;pattern<4;++pattern){
            if(!cached)for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:pattern==1?0:random_word();
            for(size_t j=0;j<bn;++j)b[j]=pattern==0?UINT64_MAX:pattern==1?0:random_word();
            for(size_t j=0;j<up(prefix,8);++j)out[j]=UINT64_C(0xabcd4567890ef123);
            sbn3_product_inputs in{};if(!cached)in.a={a,an};if(!square)in.b={b,bn};
            allocation_watch_start();sbn3_product_execute(binding,&in,{out,prefix});assert(!allocation_watch_stop());
            for(size_t j=prefix;j<up(prefix,8);++j)assert(out[j]==UINT64_C(0xabcd4567890ef123));
            ref_import(A,an,-1,8,0,0,a);ref_import(B,square?an:bn,-1,8,0,0,square?a:b);ref_mul(P,A,B);ref_fdiv_r_2exp(Q,P,64*prefix);ref_import(R,prefix,-1,8,0,0,out);assert(ref_cmp(Q,R)==0);
        }
        f.unbind(binding);
    }
    if(cache)sbn3_spectrum_release(cache);ref_clears(A,B,P,Q,R,nullptr);
    printf("exact prefix ring=%zu algorithm=%u NP%u bits=%d W%u square=%u cached=%u PASS\n",ring,o.algorithm,o.prime_count,o.trunk_bits,o.workers,square,cached);fflush(stdout);
}
int main(){
    for(unsigned np=4;np<=10;++np)for(unsigned algorithm:{unsigned(SBN3_MUL_FLAT),unsigned(SBN3_MUL_BAILEY)}){
        sbn3_mul_options o{};o.workers=3;o.prime_count=np;o.algorithm=algorithm;o.trunk_bits=np==4?80:24*int(np)-16;
        if(algorithm==SBN3_MUL_BAILEY){o.column_log2=5;o.row_log2=5;}
        for(bool square:{false,true})for(bool cached:{false,true})scenario(128*size_t(o.trunk_bits),o,square,cached);
    }
    for(unsigned bits=16;bits<=20;++bits){auto s=pq16::cyclic_shape(256,bits);if(!s.nfull)continue;sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=bits;
        for(bool square:{false,true})for(bool cached:{false,true})scenario(pq16::cyclic_period(s),o,square,cached);
    }
    {sbn3_mul_options u{};u.workers=3;u.algorithm=SBN3_MUL_BAILEY;u.prime_count=4;u.trunk_bits=84;u.column_log2=3;u.row_log2=7;
        scenario(128*84,u,false,true);scenario(128*84,u,true,true);}
    sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_SCALAR;scenario(128,o,false,false);scenario(128,o,true,false);
    sbn3_product_request bad{};bad.kind=SBN3_PRODUCT_SQR;bad.a_limbs=100;bad.cyclic_limbs=128;bad.window_limbs=32;sbn3_mul_plan p{};sbn3_product_info i{};
    assert(sbn3_product_query(&bad,&o,&p,&i)==SBN3_UNSUPPORTED);
    puts("product exact low-prefix integer/guard/cache gates PASS");
}
