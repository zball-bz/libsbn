#include "product_support.hpp"
static void verify(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *out,const sbn3_product_info &i){
    ref_int x,y,want,got,diff,mod,term,digit,exact;ref_inits(x,y,want,got,diff,mod,term,digit,exact,nullptr);
    ref_import(x,an,-1,8,0,0,a);ref_import(y,bn,-1,8,0,0,b);ref_mul(want,x,y);ref_import(got,i.mul.output_limbs,-1,8,0,0,out);
    if(i.kind==SBN3_PRODUCT_TMP){
        const size_t na=(64*an+51)/52,nb=(64*bn+51)/52,rn=nb-na+1;
        for(size_t j=0;j<na;++j){
            ref_fdiv_q_2exp(digit,x,52*j);ref_fdiv_r_2exp(digit,digit,52);
            ref_fdiv_q_2exp(term,y,52*(na-1-j));ref_fdiv_r_2exp(term,term,52*rn);
            ref_mul(term,term,digit);ref_add(exact,exact,term);
        }
        assert(ref_cmp(got,exact)==0); // stronger than the window certificate
    }
    ref_fdiv_q_2exp(want,want,i.window.offset_bits);ref_fdiv_r_2exp(want,want,i.window.width_bits);
    ref_fdiv_r_2exp(got,got,i.window.width_bits);ref_sub(diff,got,want);ref_fdiv_r_2exp(diff,diff,i.window.width_bits);
    if(i.kind!=SBN3_PRODUCT_TMP)assert(ref_sgn(diff)==0);
    else{ref_set_ui(mod,1);ref_mul_2exp(mod,mod,i.window.width_bits);ref_sub(term,mod,diff);if(ref_cmp(term,diff)<0)ref_set(diff,term);assert(!ref_sgn(diff)||ref_sizeinbase(diff,2)<=i.window.error_bits);}
    ref_clears(x,y,want,got,diff,mod,term,digit,exact,nullptr);
}
static void scenario(size_t an,size_t bn,sbn3_product_kind kind,size_t window){
    Fixture f(1);sbn3_product_request r{};r.kind=kind;r.a_limbs=an;r.b_limbs=bn;r.window_limbs=window;
    sbn3_mul_options o{};o.workers=1;sbn3_product_info i{};sbn3_mul_plan p{};
    auto *binding=f.product(r,o,i,p);assert(p.opaque[1]==100);
    auto *a=f.guarded(an?an:1),*b=f.guarded(bn?bn:1),*out=f.guarded(i.mul.output_limbs);
    for(unsigned pattern=0;pattern<8;++pattern){
        for(size_t j=0;j<an;++j)a[j]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(j==0?1:0):pattern==3?(j%2?UINT64_MAX:0):random_word();
        for(size_t j=0;j<bn;++j)b[j]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(j==0?1:0):pattern==3?(j%2?0:UINT64_MAX):random_word();
        sbn3_product_inputs in{{a,an},{b,bn},{},{}};allocation_watch_start();sbn3_product_execute(binding,&in,{out,i.mul.output_limbs});assert(!allocation_watch_stop());
        verify(a,an,b,bn,out,i);sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);assert(m.mul.worker_peak_bytes<=i.mul.per_worker_bytes);
    }
}
int main(){
    for(size_t a:{1u,2u,6u,7u,16u,103u,104u,105u,129u,208u,255u,256u,300u})
        for(size_t b:{a,a+1,2*a-1,2*a+1,3*a+7})scenario(a,b,SBN3_PRODUCT_TMP,0);
    scenario(300,8192,SBN3_PRODUCT_TMP,0);
    for(size_t a:{0u,1u,2u,6u,7u,16u,65u,128u,256u})for(size_t b:{1u,7u,17u,256u})
        for(size_t w:{size_t(1),(a+b+1)/2,a+b})for(auto kind:{SBN3_PRODUCT_LOW,SBN3_PRODUCT_HIGH})scenario(a,b,kind,w);
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_HIGH;r.a_limbs=r.b_limbs=257;r.window_limbs=1;sbn3_mul_plan p{};sbn3_product_info i{};assert(sbn3_product_query(&r,nullptr,&p,&i)==SBN3_UNSUPPORTED);
    puts("u52 middle donor/exact algebraic MP; exact short LOW/HIGH: reference/modular oracle, zero/long carry, guards, no allocation, budget PASS");
}
