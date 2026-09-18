#include "product_support.hpp"
#include "product/build_apply.hpp"
using namespace sbn::v3;
static void verify(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *out,size_t rn,bool ring){
    ref_int x,y,want,got,mod;ref_inits(x,y,want,got,mod,nullptr);
    ref_import(x,an,-1,8,0,0,a);ref_import(y,bn,-1,8,0,0,b);ref_mul(want,x,y);
    if(ring){ref_set_ui(mod,1);ref_mul_2exp(mod,mod,64*rn);ref_sub_ui(mod,mod,1);ref_mod(want,want,mod);}
    ref_import(got,rn,-1,8,0,0,out);assert(ref_cmp(want,got)==0);ref_clears(x,y,want,got,mod,nullptr);
}
static void one(unsigned np,int T,unsigned algorithm,unsigned workers,unsigned frontier,bool cyclic){
    Fixture f(workers,false);sbn3_mul_options o{};o.workers=workers;o.prime_count=np;o.trunk_bits=T;
    o.algorithm=algorithm;o.borrow_output=2;o.prime_batch=algorithm==SBN3_MUL_FLAT?0:1;
    const size_t ring=(algorithm==SBN3_MUL_FLAT?256:1024)*size_t(T)/8;
    const size_t an=cyclic?ring-13:ring/4-13,bn=cyclic?ring-17:ring/3-17;
    if(algorithm==SBN3_MUL_BAILEY){o.column_log2=6;o.row_log2=4;}
    sbn3_product_request request{};request.a_limbs=an;request.b_limbs=bn;request.cyclic_limbs=cyclic?ring:0;
    sbn3_mul_plan producer{};sbn3_product_info info{};
    const auto status=sbn3_product_query(&request,&o,&producer,&info);
    if(status!=SBN3_SUPPORTED)fprintf(stderr,"build/apply query rejected: NP%u T%d alg%u W%u frontier%u ring%u an%zu bn%zu status%u\n",np,T,algorithm,workers,frontier,unsigned(cyclic),an,bn,status);
    assert(status==SBN3_SUPPORTED);
    sbn3_spectrum_desc desc{};assert(sbn3_spectrum_query(&producer,static_cast<sbn3_spectrum_frontier>(frontier),17,&desc)==SBN3_SUPPORTED);
    auto storage=f.allocate(desc.storage_bytes,64);sbn3_spectrum *cache=nullptr;
    sbn3_spectrum_reserve_plan(&producer,static_cast<sbn3_spectrum_frontier>(frontier),17,f.arena,&storage,&cache);
    request.cached_a[0]=&desc;sbn3_mul_plan plan{};auto *product=f.product(request,o,info,plan,cache);
    assert(!sbn3_spectrum_can_apply(&plan,cache,0));
    const size_t capacity=sbn3_mul_output_capacity(&info.mul),rn=info.mul.output_limbs;
    auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(capacity);
    for(size_t j=0;j<an;++j)a[j]=workers==32?UINT64_MAX:random_word();
    for(size_t j=0;j<bn;++j)b[j]=random_word();
    const std::vector<uint64_t> original(a,a+an),old_b(b,b+bn);
    allocation_watch_start();spectrum_compute_multiply(product,cache,{a,an},{b,bn},{out,capacity});assert(!allocation_watch_stop());
    assert(!memcmp(a,original.data(),an*8) && !memcmp(b,old_b.data(),bn*8));
    assert(sbn3_spectrum_can_apply(&plan,cache,0));verify(a,an,b,bn,out,rn,cyclic);
    sbn3_product_metrics metrics{};sbn3_product_get_metrics(product,&metrics);
    const auto &g=info.mul;const size_t rows=np*(algorithm==SBN3_MUL_FLAT?1:g.C);
    assert(metrics.row_forward==2*rows);
    if(algorithm==SBN3_MUL_BAILEY)assert(metrics.column_forward==2*np*g.lbw);
    const auto *data=static_cast<const unsigned char *>(storage.data);
    const std::vector<unsigned char> saved(data,data+storage.bytes);
    memset(a,0xcc,an*8); // Cache must survive reuse of the original operand.
    for(unsigned trial=0;trial<3;++trial){
        for(size_t j=0;j<bn;++j)b[j]=trial==0?0:trial==1?UINT64_MAX:random_word();
        const sbn3_product_inputs in{{},{b,bn},{},{}};
        allocation_watch_start();sbn3_product_execute(product,&in,{out,capacity});assert(!allocation_watch_stop());
        verify(original.data(),an,b,bn,out,rn,cyclic);
        assert(!memcmp(saved.data(),data,storage.bytes));
    }
    // A cached binding's local table budget may be tiny, but a NEW spectrum
    // queried from that plan must still reserve all of its own root tables.
    sbn3_spectrum_desc again{};
    assert(sbn3_spectrum_query(&plan,static_cast<sbn3_spectrum_frontier>(frontier),29,&again)==SBN3_SUPPORTED);
    assert(again.table_bytes==desc.table_bytes && again.storage_bytes>=desc.storage_bytes);
    auto second_storage=f.allocate(again.storage_bytes,64);sbn3_spectrum *second=nullptr;
    sbn3_spectrum_reserve_plan(&plan,static_cast<sbn3_spectrum_frontier>(frontier),29,f.arena,&second_storage,&second);
    request.cached_a[0]=&again;sbn3_mul_plan second_plan{};sbn3_product_info second_info{};
    auto *other=f.product(request,o,second_info,second_plan,second);
    assert(!sbn3_spectrum_can_apply(&second_plan,second,0));
    allocation_watch_start();spectrum_compute_multiply(other,second,{original.data(),an},{b,bn},{out,capacity});assert(!allocation_watch_stop());
    verify(original.data(),an,b,bn,out,rn,cyclic);
    sbn3_spectrum_release(second);
    sbn3_spectrum_release(cache);
    printf("build/apply NP%u T%d alg%u W%u frontier%u cyclic%u: full products, ready publication, raw-input reuse, readonly cache and no allocation PASS\n",np,T,algorithm,workers,frontier,unsigned(cyclic));
}
int main(){
    for(unsigned np:{4u,6u,8u,9u,10u}){
        const int T=np==4?80:24*int(np)-16;
        for(unsigned frontier:{0u,1u})for(bool ring:{false,true}){
            one(np,T,SBN3_MUL_BAILEY,3,frontier,ring);
            one(np,T,SBN3_MUL_FLAT,3,frontier,ring);
        }
    }
    one(9,200,SBN3_MUL_BAILEY,32,1,true);
    puts("build/apply native gate PASS");
}
