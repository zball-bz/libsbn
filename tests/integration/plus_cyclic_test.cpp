#include "product_support.hpp"
static void scenario(size_t ring,bool square,bool cached,bool endpoint){
    Fixture f(1);sbn3_product_request req{};req.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;req.a_limbs=ring+1;req.b_limbs=square?0:ring+1;req.cyclic_limbs=ring;req.negacyclic=1;
    sbn3_mul_options o{};o.workers=1;sbn3_product_info i{};sbn3_mul_plan p{};auto *binding=f.product(req,o,i,p);
    assert(p.opaque[1]==101 && i.negacyclic && i.mul.output_limbs==ring+1);
    auto *a=f.guarded(ring+1),*b=f.guarded(ring+1),*out=f.guarded(ring+1);
    for(size_t j=0;j<ring;++j)a[j]=endpoint?0:UINT64_MAX;a[ring]=endpoint;
    sbn3_spectrum *cache=nullptr;sbn3_spectrum_desc d{};
    if(cached){cache=f.cache(binding,{a,ring+1},1,i,d);req.cached_a[0]=&d;f.unbind(binding);binding=f.product(req,o,i,p,cache);assert(sbn3_spectrum_can_apply(&p,cache,0));}
    ref_int x,y,want,got,mod;ref_inits(x,y,want,got,mod,nullptr);ref_set_ui(mod,1);ref_mul_2exp(mod,mod,64*ring);ref_add_ui(mod,mod,1);
    for(unsigned pattern=0;pattern<8;++pattern){
        if(!cached){for(size_t j=0;j<ring;++j)a[j]=endpoint?0:pattern==1?0:pattern==0?UINT64_MAX:random_word();a[ring]=endpoint;}
        for(size_t j=0;j<ring;++j)b[j]=pattern==0?UINT64_MAX:pattern==1||pattern==2?0:pattern==3?(j==0?1:0):random_word();b[ring]=pattern==2;
        sbn3_product_inputs input{};if(!cached)input.a={a,ring+1};if(!square)input.b={b,ring+1};
        allocation_watch_start();sbn3_product_execute(binding,&input,{out,ring+1});assert(!allocation_watch_stop());
        ref_import(x,ring+1,-1,8,0,0,a);ref_import(y,ring+1,-1,8,0,0,square?a:b);ref_mul(want,x,y);ref_mod(want,want,mod);ref_import(got,ring+1,-1,8,0,0,out);
        assert(ref_cmp(got,want)==0 && ref_cmp(got,mod)<0);
        sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);assert(m.mul.worker_peak_bytes<=i.mul.per_worker_bytes && m.row_inverse==1 && m.row_forward==(cached?(square?0:1u):(square?1:2u)));
    }
    if(cache)sbn3_spectrum_release(cache);ref_clears(x,y,want,got,mod,nullptr);
}
int main(){
    for(size_t r:{192u,320u,448u,768u,1280u,1792u,3072u,3584u,5120u})for(bool square:{false,true})for(bool cached:{false,true})for(bool endpoint:{false,true})scenario(r,square,cached,endpoint);
    sbn3_product_request r{};r.a_limbs=r.b_limbs=100;r.cyclic_limbs=100;r.negacyclic=1;sbn3_mul_plan p{};sbn3_product_info i{};assert(sbn3_product_query(&r,nullptr,&p,&i)==SBN3_UNSUPPORTED);
    puts("native RAC plus ring: fresh/cache MUL/SQR, endpoint 2^k, zero, signed carry, reference/modular oracle, guards, no allocation, transforms PASS");
}
