#include "product_support.hpp"
#include "product/backend.hpp"
#include "product/build_apply.hpp"
#include "runtime/scratch.hpp"
using namespace sbn::v3;
static uint64_t cache_hash(const sbn3_spectrum *s,size_t bytes){
    const auto *p=reinterpret_cast<const unsigned char *>(s);uint64_t h=1;
    for(size_t j=1024;j<bytes;++j)h=(h^p[j])*1099511628211ULL;return h;
}
static void scenario(unsigned np,int bits,unsigned lg,bool rescale){
    const size_t ring=(size_t(1)<<lg)*bits/64,an=ring/3-3,planned_b=an+7;
    Fixture f(1,false,size_t(512)<<20);auto *a=f.guarded(an),*fresh=f.guarded(an),*b=f.guarded(ring);
    for(size_t j=0;j<an;++j)a[j]=random_word();for(size_t j=0;j<ring;++j)b[j]=random_word();
    sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_FLAT;o.prime_count=np;o.trunk_bits=bits;o.crt_mode=rescale?1:2;
    sbn3_product_request req{};req.a_limbs=an;req.b_limbs=planned_b;req.cyclic_limbs=ring;
    sbn3_product_info info{};sbn3_mul_plan producer{};auto *builder=f.product(req,o,info,producer);
    sbn3_spectrum_desc d{};auto *cache=f.cache(builder,{a,an},SBN3_SPECTRUM_COLUMNS,info,d);f.unbind(builder);
    const uint64_t saved=cache_hash(cache,d.storage_bytes);
    req.cached_a[0]=&d;o.crt_mode=2;sbn3_mul_plan plan{};auto *binding=f.product(req,o,info,plan,cache);
    const auto *backend=binding->backend;assert(backend->product_fresh_supported&&backend->product_execute_fresh);
    assert(backend->product_live_contract&&(backend->product_live_contract(plan)&program_consume_inputs));
    assert(!backend->product_live_contract(producer));
    assert(backend->product_scratch_bytes&&backend->with_product_scratch);
    const size_t temporary=backend->product_scratch_bytes(plan);assert(temporary);
    allocation_watch_start();backend->with_product_scratch(binding,nullptr,[](void *,Frame &frame){
        auto *data=frame.allocate(frame.capacity(),1);memset(data,0xa5,frame.capacity());
    });assert(!allocation_watch_stop());
    assert(cache_hash(cache,d.storage_bytes)==saved);
    assert(backend->product_fresh_supported(plan,an,ring));
    assert(!backend->product_fresh_supported(plan,an+1,ring)&&!backend->product_fresh_supported(plan,an,ring+1));
    assert(!backend->product_fresh_supported(producer,an,ring));
    if(rescale&&np>4){sbn3_spectrum_desc owned{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,7,&owned)==SBN3_SUPPORTED);
        assert(memcmp(d.scale,owned.scale,sizeof d.scale));}
    const size_t capacity=sbn3_mul_output_capacity(&info.mul);auto *out=f.guarded(capacity);
    ref_int A,B,P,M,R;ref_inits(A,B,P,M,R,nullptr);ref_set_ui(M,1);ref_mul_2exp(M,M,64*ring);ref_sub_ui(M,M,1);
    for(unsigned pattern=0;pattern<3;++pattern){
        for(size_t j=0;j<an;++j)fresh[j]=pattern==0?0:pattern==1?UINT64_MAX:random_word();
        const std::vector<uint64_t> saved_fresh(fresh,fresh+an),saved_b(b,b+ring);
        ref_import(A,an,-1,8,0,0,fresh);ref_import(B,ring,-1,8,0,0,b);ref_mul(P,A,B);ref_mod(P,P,M);
        const sbn3_product_inputs in{{fresh,an},{b,ring},{},{}};
        allocation_watch_start();backend->product_execute_fresh(binding,nullptr,in,{out,capacity});assert(!allocation_watch_stop());
        ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0);
        sbn3_product_metrics m{};sbn3_product_get_metrics(binding,&m);
        assert(m.row_forward==2*np&&m.row_inverse==np&&m.mul.worker_peak_bytes<=info.mul.per_worker_bytes);
        assert(cache_hash(cache,d.storage_bytes)==saved);
        assert(!memcmp(fresh,saved_fresh.data(),8*an)&&!memcmp(b,saved_b.data(),8*ring));
        memcpy(out,b,ring*8);
        const sbn3_product_inputs folded{{fresh,an},{out,ring},{},{}};
        allocation_watch_start();backend->product_execute_fresh(binding,nullptr,folded,{out,capacity});assert(!allocation_watch_stop());
        ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0&&cache_hash(cache,d.storage_bytes)==saved);
        // The binding must still perform its original cached operation.
        ref_import(A,an,-1,8,0,0,a);ref_import(B,planned_b,-1,8,0,0,b);ref_mul(P,A,B);ref_mod(P,P,M);
        const sbn3_product_inputs cached{{},{b,planned_b},{},{}};
        allocation_watch_start();sbn3_product_execute(binding,&cached,{out,capacity});assert(!allocation_watch_stop());
        ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0&&cache_hash(cache,d.storage_bytes)==saved);
        // All B reads finish before integer emission, including when the
        // consumed span is an interior window of the destination.
        for(size_t offset:{size_t(0),ring/2}){
            const size_t live=planned_b-pattern;
            memcpy(out+offset,b,live*8);
            ref_import(B,live,-1,8,0,0,b);ref_mul(P,A,B);ref_mod(P,P,M);
            const sbn3_product_inputs aliased{{},{out+offset,live},{},{}};
            allocation_watch_start();backend->product_execute_live(binding,nullptr,aliased,{out,capacity});assert(!allocation_watch_stop());
            ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0&&cache_hash(cache,d.storage_bytes)==saved);
        }
    }
    f.unbind(binding);sbn3_spectrum_release(cache);ref_clears(A,B,P,M,R,nullptr);
    printf("NP%u lg%u rescale=%u cached/fresh binding reuse and fixed workspace PASS\n",np,lg,rescale);
}
int main(){for(bool scaled:{false,true}){scenario(4,80,14,scaled);scenario(6,128,14,scaled);
    scenario(6,128,18,scaled);scenario(8,176,18,scaled);scenario(10,224,18,scaled);}}
