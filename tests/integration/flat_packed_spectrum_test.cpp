#include "product_support.hpp"
#include "product/build_apply.hpp"
#include "product/spectrum_contract.hpp"
using namespace sbn::v3;
static uint64_t cache_hash(const sbn3_spectrum *cache,const sbn3_spectrum_desc &d){
    const auto *bytes=reinterpret_cast<const unsigned char *>(cache);uint64_t h=1;
    // Mutable lifecycle/refcount header precedes the immutable tables/planes.
    for(size_t j=1024;j<d.storage_bytes;++j)h=(h^bytes[j])*1099511628211ULL;
    return h;
}
struct Cache {sbn3_spectrum *handle=nullptr;sbn3_spectrum_desc future{},actual{};};
static Cache cached(Fixture &f,sbn3_mul_options o,sbn3_product_request r,const uint64_t *a){
    sbn3_product_info info{};sbn3_mul_plan plan{};auto *producer=f.product(r,o,info,plan);
    Cache c;assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,7,&c.future)==SBN3_SUPPORTED);
    c.handle=f.cache(producer,{a,r.a_limbs},SBN3_SPECTRUM_COLUMNS,info,c.actual);
    f.unbind(producer);return c;
}
static void family(unsigned np,unsigned bits){
    const size_t ring=(size_t(1)<<18)*bits/64,an=ring/3-7,bn=ring-5;
    Fixture f(2,false,size_t(512)<<20);auto *a=f.guarded(an),*b=f.guarded(bn);
    for(size_t j=0;j<an;++j)a[j]=j%3?random_word():UINT64_MAX;
    for(size_t j=0;j<bn;++j)b[j]=random_word();
    const std::vector<uint64_t> saved_a(a,a+an),saved_b(b,b+bn);
    sbn3_mul_options o{};o.algorithm=SBN3_MUL_FLAT;o.prime_count=np;o.trunk_bits=int(bits);o.workers=1;
    sbn3_product_request r{};r.a_limbs=an;r.b_limbs=bn;r.cyclic_limbs=ring;
    auto packed=cached(f,o,r,a);assert(packed.actual.format_version==spectrum_contract::flat_packed48_format&&packed.actual.block_stride==48);
    o.workers=2;auto lazy=cached(f,o,r,a);assert(lazy.actual.format_version==spectrum_contract::flat_lazy64_format&&lazy.actual.block_stride==64);
    assert(packed.actual.basis_id==lazy.actual.basis_id&&packed.actual.plane_bytes<lazy.actual.plane_bytes);
    o.workers=1;const auto packed_hash=cache_hash(packed.handle,packed.actual),lazy_hash=cache_hash(lazy.handle,lazy.actual);
    assert(!memcmp(a,saved_a.data(),8*an));
    ref_int A,B,P,Q,M,R;ref_inits(A,B,P,Q,M,R,nullptr);ref_import(A,an,-1,8,0,0,a);ref_import(B,bn,-1,8,0,0,b);
    ref_set_ui(M,1);ref_mul_2exp(M,M,64*ring);ref_sub_ui(M,M,1);
    for(auto kind:{SBN3_PRODUCT_MUL,SBN3_PRODUCT_SQR,SBN3_PRODUCT_MAC2,SBN3_PRODUCT_TMP}){
        const bool linear=kind==SBN3_PRODUCT_MAC2||kind==SBN3_PRODUCT_TMP;const size_t length=linear?ring/2-3:bn;
        r.kind=kind;r.cyclic_limbs=linear?0:ring;r.b_limbs=kind==SBN3_PRODUCT_SQR?0:length;r.cached_a[0]=&packed.future;
        r.a1_limbs=kind==SBN3_PRODUCT_MAC2?an:0;r.b1_limbs=kind==SBN3_PRODUCT_MAC2?length:0;
        r.cached_a[1]=kind==SBN3_PRODUCT_MAC2?&lazy.future:nullptr;
        sbn3_product_info info{};sbn3_mul_plan plan{};auto *binding=f.product(r,o,info,plan,packed.handle,kind==SBN3_PRODUCT_MAC2?lazy.handle:nullptr);
        assert(sbn3_spectrum_can_apply(&plan,packed.handle,0)&&!sbn3_spectrum_can_apply(&plan,lazy.handle,0));
        const size_t cap=sbn3_mul_output_capacity(&info.mul);auto *out=f.guarded(cap);
        sbn3_product_inputs in{};if(kind!=SBN3_PRODUCT_SQR)in.b={b,length};if(kind==SBN3_PRODUCT_MAC2)in.b1={b,length};
        allocation_watch_start();sbn3_product_execute(binding,&in,{out,cap});assert(!allocation_watch_stop());
        ref_import(B,length,-1,8,0,0,b);ref_mul(P,A,kind==SBN3_PRODUCT_SQR?A:B);if(kind==SBN3_PRODUCT_MAC2)ref_mul_ui(P,P,2);
        if(kind==SBN3_PRODUCT_TMP){
            ref_fdiv_q_2exp(P,P,info.window.offset_bits);ref_import(R,info.mul.output_limbs,-1,8,0,0,out);
            ref_sub(Q,R,P);ref_fdiv_r_2exp(Q,Q,info.window.width_bits);
            ref_int limit,high;ref_inits(limit,high,nullptr);ref_set_ui(limit,1);ref_mul_2exp(limit,limit,info.window.error_bits);
            ref_set_ui(high,1);ref_mul_2exp(high,high,info.window.width_bits);ref_sub(high,high,limit);
            assert(ref_cmp(Q,limit)<0||ref_cmp(Q,high)>0);ref_clears(limit,high,nullptr);
        }else{if(!linear)ref_mod(P,P,M);ref_import(R,info.mul.output_limbs,-1,8,0,0,out);assert(ref_cmp(P,R)==0);}
        sbn3_product_metrics metrics{};sbn3_product_get_metrics(binding,&metrics);assert(metrics.mul.worker_peak_bytes<=info.mul.per_worker_bytes);
        f.unbind(binding);
    }
    // Preserve a materialized packed cache under a wider-team consumer.
    r.kind=SBN3_PRODUCT_MUL;r.cyclic_limbs=ring;r.b_limbs=bn;r.a1_limbs=r.b1_limbs=0;r.cached_a[1]=nullptr;o.workers=2;
    sbn3_product_info info{};sbn3_mul_plan plan{};auto *binding=f.product(r,o,info,plan,packed.handle);
    sbn3_spectrum_desc rebuild{};assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,11,&rebuild)==SBN3_SUPPORTED);
    assert(rebuild.format_version==spectrum_contract::flat_packed48_format&&rebuild.block_stride==48);
    const auto rebuild_storage=f.allocate(rebuild.storage_bytes,64);sbn3_spectrum *rebuilt=nullptr;
    sbn3_spectrum_reserve_plan(&plan,SBN3_SPECTRUM_COLUMNS,11,f.arena,&rebuild_storage,&rebuilt);
    allocation_watch_start();sbn3_spectrum_compute(binding,rebuilt,{a,an});assert(!allocation_watch_stop());
    sbn3_spectrum_desc ready{};sbn3_spectrum_describe(rebuilt,&ready);assert(ready.format_version==rebuild.format_version);
    sbn3_spectrum_release(rebuilt);
    const size_t cap=sbn3_mul_output_capacity(&info.mul);auto *out=f.guarded(cap);const sbn3_product_inputs in{{},{b,bn},{},{}};
    allocation_watch_start();sbn3_product_execute(binding,&in,{out,cap});assert(!allocation_watch_stop());
    ref_import(B,bn,-1,8,0,0,b);ref_mul(P,A,B);ref_mod(P,P,M);ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0);f.unbind(binding);
    // Fused build/apply must emit the new representation and mark it ready.
    o.workers=1;r.cached_a[0]=nullptr;
    assert(sbn3_product_query(&r,&o,&plan,&info)==SBN3_SUPPORTED);sbn3_spectrum_desc future{};
    assert(sbn3_spectrum_query(&plan,SBN3_SPECTRUM_COLUMNS,9,&future)==SBN3_SUPPORTED);
    auto storage=f.allocate(future.storage_bytes,64);sbn3_spectrum *fresh=nullptr;
    sbn3_spectrum_reserve_plan(&plan,SBN3_SPECTRUM_COLUMNS,9,f.arena,&storage,&fresh);
    r.cached_a[0]=&future;binding=f.product(r,o,info,plan,fresh);
    allocation_watch_start();spectrum_compute_multiply(binding,fresh,{a,an},{b,bn},{out,cap});assert(!allocation_watch_stop());
    ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0&&sbn3_spectrum_can_apply(&plan,fresh,0));
    f.unbind(binding);sbn3_spectrum_release(fresh);
    assert(cache_hash(packed.handle,packed.actual)==packed_hash&&cache_hash(lazy.handle,lazy.actual)==lazy_hash);
    assert(!memcmp(a,saved_a.data(),8*an)&&!memcmp(b,saved_b.data(),8*bn));
    sbn3_spectrum_release(packed.handle);sbn3_spectrum_release(lazy.handle);
    ref_clears(A,B,P,Q,M,R,nullptr);
    printf("NP%u packed Flat: MUL/SQR/TMP/mixed-format MAC, wildcard identity, wider consumer and build/apply PASS\n",np);
}
int main(){family(4,80);family(5,104);family(6,128);family(7,152);family(8,176);family(9,200);family(10,224);}
