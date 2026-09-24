#include "product_support.hpp"
#include "product/build_apply.hpp"
#include "product/fft_choice.hpp"
#include "backend/pq16/kernels.hpp"
#include "runtime/team.hpp"
using namespace sbn::v3;
static void fresh_root_choice(){
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.a_limbs=r.b_limbs=45000;
    sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.workers=1;
    const auto prepared=pq16::select(r.a_limbs,r.b_limbs,1);
    assert(prepared.recipe==pq16::Recipe::RightAngle&&!pq16::tables_published(prepared));
    sbn3_mul_plan pinned{},fresh{},limited{},cached{};sbn3_product_info pi{},fi{},li{},ci{};
    assert(short_fft_query(r,o,prepared,pinned,pi)==SBN3_SUPPORTED);
    assert(sbn3_product_query(&r,&o,&fresh,&fi)==SBN3_SUPPORTED);
    assert((fi.mul.codec_mode&3)==unsigned(pq16::Recipe::CooleyTukeyPQ));
    assert(fi.mul.C==pi.mul.C&&fi.mul.M2==pi.mul.M2&&fi.mul.trunk_bits==pi.mul.trunk_bits&&fi.mul.table_bytes<pi.mul.table_bytes);
    // CT's tail workspace is larger even though its total storage is smaller.
    // A caller who budgets workspace alone must retain the fitting recipe.
    assert(fi.mul.workspace_bytes>pi.mul.workspace_bytes);o.workspace_budget=pi.mul.workspace_bytes;
    assert(sbn3_product_query(&r,&o,&limited,&li)==SBN3_SUPPORTED&&li.mul.workspace_bytes<=o.workspace_budget);
    assert((li.mul.codec_mode&3)==unsigned(pq16::Recipe::RightAngle));
    o.workspace_budget=0;sbn3_spectrum_desc desc{};
    assert(sbn3_spectrum_query(&pinned,SBN3_SPECTRUM_COLUMNS,1,&desc)==SBN3_SUPPORTED);
    r.cached_a[0]=&desc;
    assert(sbn3_product_query(&r,&o,&cached,&ci)==SBN3_SUPPORTED);
    assert((ci.mul.codec_mode&3)==unsigned(pq16::Recipe::RightAngle));
    puts("FFT fresh roots: ready alternative, workspace budget and cached basis preserved PASS");
}
static void extended_root_products(){
    unsigned cases=0;
    for(bool extended:{false,true})for(bool square:{false,true}){
        // The largest pow2 FFT has an existing short-operand guard. Use
        // supported operands and pin that shape rather than asking an
        // unsupported balanced 131072 x 131072 product to select it.
        const size_t an=extended?(square?98304:131072):57549,bn=extended?98304:57549;
        Fixture f(1);sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.workers=1;
        sbn3_product_request request{};request.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;
        request.a_limbs=an;request.b_limbs=square?0:bn;
        sbn3_mul_plan plan{};sbn3_product_info info{};sbn3_mul_binding *binding=nullptr;
        if(extended){
            const pq16::Shape shape{524288,524288,1,true,pq16::Recipe::PfaPQ,16,false};
            assert(short_fft_query(request,o,shape,plan,info)==SBN3_SUPPORTED);
            auto tables=f.allocate(info.mul.table_bytes,128),work=f.allocate(info.mul.workspace_bytes,info.mul.workspace_alignment);
            allocation_watch_start();sbn3_product_bind(&plan,f.arena,&tables,&work,f.team,nullptr,nullptr,&binding);
            assert(!allocation_watch_stop());f.bindings.push_back(binding);
        }else binding=f.product(request,o,info,plan);
        if(!extended)assert(info.mul.M2==262144&&info.mul.table_bytes<16384);
        else assert(info.mul.M2==524288&&info.mul.table_bytes>size_t(2)<<20);
        auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(sbn3_mul_output_capacity(&info.mul));
        for(unsigned pattern=0;pattern<3;++pattern){
            for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:random_word();
            for(size_t j=0;j<bn;++j)b[j]=pattern==1?(j&1?0:UINT64_MAX):random_word();
            const sbn3_product_inputs inputs{{a,an},{square?nullptr:b,square?0:bn},{},{}};
            allocation_watch_start();sbn3_product_execute(binding,&inputs,{out,sbn3_mul_output_capacity(&info.mul)});
            assert(!allocation_watch_stop());verify_product(a,an,square?a:b,square?an:bn,out);++cases;
        }
    }
    puts("FFT published and next dynamic PQ-root level: 12 exact/allocation-free products PASS");
    assert(cases==12);
}
int main(){
    fresh_root_choice();
    extended_root_products();
    unsigned checked=0;
    for(unsigned bits:{16u,17u,18u,19u,20u})for(unsigned workers:{1u,4u}){
        if(bits>16&&workers>1)continue;
        const auto shape=pq16::cyclic_shape(512,bits);if(!shape.nfull)continue;
        const size_t ring=pq16::cyclic_period(shape),an=ring/3,bn=ring-1;
        Fixture f(workers,false);sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;o.trunk_bits=int(bits);o.workers=workers;
        sbn3_product_request r{};r.kind=SBN3_PRODUCT_MUL;r.a_limbs=an;r.b_limbs=bn;r.cyclic_limbs=ring;
        sbn3_mul_plan producer{},plan{};sbn3_product_info info{},ci{};
        assert(sbn3_product_query(&r,&o,&producer,&info)==SBN3_SUPPORTED);
        sbn3_spectrum_desc d{};assert(sbn3_spectrum_query(&producer,SBN3_SPECTRUM_COLUMNS,1,&d)==SBN3_SUPPORTED);
        auto lease=f.allocate(d.storage_bytes,info.spectrum_alignment);sbn3_spectrum *cache=nullptr;
        sbn3_spectrum_reserve_plan(&producer,SBN3_SPECTRUM_COLUMNS,1,f.arena,&lease,&cache);
        r.cached_a[0]=&d;auto *binding=f.product(r,o,ci,plan,cache);
        auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(ring);
        for(size_t j=0;j<an;++j)a[j]=j%3?random_word():UINT64_MAX;
        for(size_t j=0;j<bn;++j)b[j]=random_word();
        assert(!sbn3_spectrum_can_apply(&plan,cache,0));const auto epoch=f.team->epoch;
        allocation_watch_start();spectrum_compute_multiply(binding,cache,{a,an},{b,bn},{out,ring});
        assert(!allocation_watch_stop()&&f.team->epoch==epoch+1&&sbn3_spectrum_can_apply(&plan,cache,0));
        ref_int A,B,P,M,R;ref_inits(A,B,P,M,R,nullptr);ref_import(A,an,-1,8,0,0,a);
        ref_set_ui(M,1);ref_mul_2exp(M,M,64*ring);ref_sub_ui(M,M,1);
        for(unsigned repeat=0;repeat<2;++repeat){
            if(repeat){for(size_t j=0;j<bn;++j)b[j]=~b[j];const sbn3_product_inputs in{{},{b,bn},{},{}};
                allocation_watch_start();sbn3_product_execute(binding,&in,{out,ring});assert(!allocation_watch_stop());}
            ref_import(B,bn,-1,8,0,0,b);ref_mul(P,A,B);ref_mod(P,P,M);ref_import(R,ring,-1,8,0,0,out);assert(ref_cmp(P,R)==0);
            sbn3_product_metrics metrics{};sbn3_product_get_metrics(binding,&metrics);assert(metrics.row_forward==(repeat?1u:2u)&&metrics.row_inverse==1);++checked;
        }
        ref_clears(A,B,P,M,R,nullptr);sbn3_spectrum_release(cache);
    }
    printf("FFT build/apply: %u cyclic/cached cases; one team episode and allocation-free PASS\n",checked);
}
