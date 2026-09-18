#include "product_support.hpp"
#include "backend/pq16/kernels.hpp"
using namespace sbn::v3;
static void scenario(size_t ring,sbn3_mul_options o,unsigned frontier){
    Fixture f(o.workers);size_t an=ring/3,bn=an-1,window=an+2;
    auto *a=f.guarded(an),*b=f.guarded(bn);for(size_t j=0;j<an;++j)a[j]=j%3?random_word():UINT64_MAX;for(size_t j=0;j<bn;++j)b[j]=random_word();
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_SQR;r.a_limbs=an;r.cyclic_limbs=ring;r.window_limbs=window;
    sbn3_mul_plan producer{},plan{};sbn3_product_info i{},ci{};assert(sbn3_product_query(&r,&o,&producer,&i)==SBN3_SUPPORTED);
    sbn3_spectrum_desc d{};assert(sbn3_spectrum_query(&producer,(sbn3_spectrum_frontier)frontier,17,&d)==SBN3_SUPPORTED);
    auto storage=f.allocate(d.storage_bytes,i.spectrum_alignment);sbn3_spectrum *cache=nullptr;sbn3_spectrum_reserve_plan(&producer,(sbn3_spectrum_frontier)frontier,17,f.arena,&storage,&cache);
    r.cached_a[0]=&d;auto *square=f.product(r,o,ci,plan,cache);assert(!sbn3_spectrum_can_apply(&plan,cache,0));auto *out=f.guarded(up(window,8));
    for(size_t j=window;j<up(window,8);++j)out[j]=UINT64_MAX;
    allocation_watch_start();sbn3_spectrum_compute_square(square,cache,{a,an},{out,window});assert(!allocation_watch_stop());assert(sbn3_spectrum_can_apply(&plan,cache,0));
    for(size_t j=window;j<up(window,8);++j)assert(out[j]==UINT64_MAX);
    ref_int A,B,P,R;ref_inits(A,B,P,R,nullptr);ref_import(A,an,-1,8,0,0,a);ref_mul(P,A,A);ref_fdiv_r_2exp(P,P,64*window);ref_import(R,window,-1,8,0,0,out);assert(ref_cmp(P,R)==0);
    sbn3_product_metrics metrics{};sbn3_product_get_metrics(square,&metrics);assert(metrics.row_forward && metrics.row_inverse);
    r.kind=SBN3_PRODUCT_MUL;r.b_limbs=bn;r.window_limbs=0;auto *mul=f.product(r,o,i,plan,cache);auto *product=f.guarded(up(ring,8));sbn3_product_inputs in{};in.b={b,bn};
    allocation_watch_start();sbn3_product_execute(mul,&in,{product,ring});assert(!allocation_watch_stop());ref_import(B,bn,-1,8,0,0,b);ref_mul(P,A,B);ref_import(R,ring,-1,8,0,0,product);assert(ref_cmp(P,R)==0);
    ref_clears(A,B,P,R,nullptr);sbn3_spectrum_release(cache);
    printf("fused forward + low SQR + cached MUL reuse: np=%u alg=%u bits=%d W=%u frontier=%u PASS\n",o.prime_count,o.algorithm,o.trunk_bits,o.workers,frontier);fflush(stdout);
}
int main(){
    for(unsigned np:{4u,5u,6u,7u,8u,9u,10u})for(unsigned alg:{unsigned(SBN3_MUL_FLAT),unsigned(SBN3_MUL_BAILEY)})for(unsigned frontier:{0u,1u}){
        sbn3_mul_options o{};o.workers=3;o.algorithm=alg;o.prime_count=np;o.trunk_bits=np==4?80:24*int(np)-16;if(alg==SBN3_MUL_BAILEY){o.column_log2=5;o.row_log2=5;}
        scenario(128*size_t(o.trunk_bits),o,frontier);
    }
    for(unsigned bits=16;bits<=20;++bits){const auto s=pq16::cyclic_shape(512,bits);if(!s.nfull)continue;sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=bits;scenario(pq16::cyclic_period(s),o,1);}
    sbn3_mul_options o{};o.workers=16;o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=16;scenario(8192,o,1);
    puts("fused spectrum build/square contract gates PASS");
}
