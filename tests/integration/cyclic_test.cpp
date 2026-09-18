#include "product_support.hpp"

static void verify(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *out,size_t rn){
    ref_int x,y,z,m,got;ref_inits(x,y,z,m,got,nullptr);
    ref_import(x,an,-1,8,0,0,a);ref_import(y,bn,-1,8,0,0,b);
    ref_mul(z,x,y);ref_set_ui(m,1);ref_mul_2exp(m,m,64*rn);ref_sub_ui(m,m,1);ref_mod(z,z,m);
    ref_import(got,rn,-1,8,0,0,out);
    if(ref_cmp(z,got)){fprintf(stderr,"cyclic mismatch an=%zu bn=%zu rn=%zu\n",an,bn,rn);assert(false);}
    assert(ref_cmp(got,m)<0);ref_clears(x,y,z,m,got,nullptr);
}
static void pattern(uint64_t *a,size_t n,unsigned p){
    for(size_t j=0;j<n;++j)a[j]=p==0?UINT64_MAX:p==1?0:p==2?(j?UINT64_MAX:UINT64_MAX-1):p==3?0:p==4?random_word():j%2?UINT64_MAX:1;
    if(p==3)a[n-1]=uint64_t(1)<<63;
}
static uint64_t cache_hash(const sbn3_spectrum *s,const sbn3_spectrum_desc &d){
    const auto *bytes=reinterpret_cast<const unsigned char *>(s);uint64_t h=1;
    for(size_t i=1024;i<d.storage_bytes;++i)h=(h^bytes[i])*1099511628211ULL;return h;
}
static void scenario(unsigned np,int T,unsigned alg,unsigned kind,int frontier,bool unequal=false,unsigned rowlog=4,unsigned workers=3){
    Fixture f(workers);sbn3_mul_options o{};o.workers=workers;o.prime_count=np;o.trunk_bits=T;o.algorithm=alg;o.borrow_output=1;
    if(alg==SBN3_MUL_BAILEY){o.column_log2=rowlog==14?3:6;o.row_log2=rowlog;}
    const size_t rn=(size_t(1)<<(alg==SBN3_MUL_FLAT?4:o.column_log2+o.row_log2))*T/8;
    const size_t an=unequal?13:rn,bn=kind==SBN3_PRODUCT_SQR?an:unequal?rn-1:rn;
    auto *a=f.guarded(an),*b=f.guarded(bn),*out=f.guarded(rn);
    sbn3_product_request req{};req.kind=static_cast<sbn3_product_kind>(kind);req.a_limbs=an;req.b_limbs=kind==SBN3_PRODUCT_SQR?0:bn;req.cyclic_limbs=rn;
    sbn3_mul_plan plan{};sbn3_product_info info{};auto *bound=f.product(req,o,info,plan);
    const auto &i=info.mul;
    assert(info.cyclic_limbs==rn && i.output_limbs==rn && i.full && i.lbv==i.M2 && i.transform_trunks*T==rn*64);
    assert(i.emit_limbs-rn<=(16*size_t(T)+63)/64+i.emit_quantum_limbs); // Carry fringe plus codec regroup padding.
    sbn3_spectrum *cache=nullptr;sbn3_spectrum_desc desc{};uint64_t saved_hash=0;
    if(frontier>=0){pattern(a,an,4);cache=f.cache(bound,{a,an},unsigned(frontier),info,desc);saved_hash=cache_hash(cache,desc);
        f.unbind(bound);req.cached_a[0]=&desc;bound=f.product(req,o,info,plan,cache);assert(sbn3_spectrum_can_apply(&plan,cache,0));}
    for(unsigned p=0;p<6;++p){
        if(!cache)pattern(a,an,p);pattern(b,bn,p);memset(out,0x9d,rn*8);
        sbn3_product_inputs in{};if(!cache)in.a={a,an};if(kind!=SBN3_PRODUCT_SQR)in.b={b,bn};
        allocation_watch_start();sbn3_product_execute(bound,&in,{out,rn});assert(!allocation_watch_stop());
        verify(a,an,kind==SBN3_PRODUCT_SQR?a:b,kind==SBN3_PRODUCT_SQR?an:bn,out,rn);
        sbn3_product_metrics metrics{};sbn3_product_get_metrics(bound,&metrics);
        assert(metrics.mul.worker_peak_bytes<=i.per_worker_bytes && metrics.mul.workspace_used_bytes<=i.workspace_bytes);
        const uint64_t rows=kind==SBN3_PRODUCT_SQR?unsigned(!cache):1+unsigned(!cache);
        assert(metrics.row_forward==rows*np*(alg==SBN3_MUL_FLAT?1:i.C));
        if(alg==SBN3_MUL_FLAT)assert(metrics.row_inverse==np && !metrics.column_forward);
        else assert(metrics.column_inverse==np*i.lbw && metrics.leaf_products==np*i.lbw*(i.C/8));
    }
    if(cache && kind==SBN3_PRODUCT_MUL){
        // One immutable MUL cache is legal for SQR after leaf-scale compensation.
        f.unbind(bound);req.kind=SBN3_PRODUCT_SQR;req.b_limbs=0;
        bound=f.product(req,o,info,plan,cache);sbn3_product_inputs in{};
        allocation_watch_start();sbn3_product_execute(bound,&in,{out,rn});assert(!allocation_watch_stop());verify(a,an,a,an,out,rn);
    }
    if(cache){assert(cache_hash(cache,desc)==saved_hash);sbn3_spectrum_release(cache);}
    printf("cyclic np=%u T=%d alg=%u kind=%u cached=%d unequal=%u rn=%zu W=%u fused=%u: reference/modular oracle/guard/reuse/counts/alloc PASS\n",np,T,alg,kind,frontier,unequal,rn,workers,info.mul.fused);fflush(stdout);
}
static void query_gate(){
    sbn3_product_request r{};r.a_limbs=128;r.b_limbs=64;r.cyclic_limbs=256;
    sbn3_mul_options o{};o.workers=1;o.prime_count=6;o.algorithm=SBN3_MUL_FLAT;
    sbn3_mul_plan p{};sbn3_product_info i{};
    allocation_watch_start();assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
    assert(i.mul.trunk_bits==128 && i.cyclic_limbs==256); // T136 cannot represent this period.
    const auto good=p;const size_t required=i.mul.workspace_bytes;
    o.workspace_budget=required-1;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_QUERY_CAPACITY);
    assert(!memcmp(&p,&good,sizeof p) && i.mul.workspace_bytes==required);o.workspace_budget=0;
    ++r.cyclic_limbs;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_UNSUPPORTED);assert(!memcmp(&p,&good,sizeof p));--r.cyclic_limbs;
    r.a_limbs=r.cyclic_limbs+1;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_UNSUPPORTED);r.a_limbs=128;
    r.kind=SBN3_PRODUCT_TMP;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_UNSUPPORTED);r.kind=SBN3_PRODUCT_MUL;
    o.algorithm=SBN3_MUL_AUTO;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_SUPPORTED && i.mul.algorithm==SBN3_MUL_FLAT);
    o.algorithm=SBN3_MUL_PQ16;assert(sbn3_product_query(&r,&o,&p,&i)==SBN3_UNSUPPORTED);
}
int main(){
    query_gate();
    for(unsigned np=4;np<=10;++np)for(unsigned alg:{unsigned(SBN3_MUL_FLAT),unsigned(SBN3_MUL_BAILEY)}){
        const int T=24*np-8;scenario(np,T,alg,SBN3_PRODUCT_MUL,-1);
        scenario(np,T,alg,SBN3_PRODUCT_SQR,-1);
        scenario(np,T,alg,SBN3_PRODUCT_MUL,0,true);
        scenario(np,T,alg,SBN3_PRODUCT_MUL,1);
    }
    for(int T:{80,84}){scenario(4,T,SBN3_MUL_FLAT,SBN3_PRODUCT_MUL,-1);scenario(4,T,SBN3_MUL_BAILEY,SBN3_PRODUCT_SQR,1);}
    scenario(6,128,SBN3_MUL_BAILEY,SBN3_PRODUCT_MUL,-1,true,14,1);
    scenario(8,176,SBN3_MUL_BAILEY,SBN3_PRODUCT_MUL,1,false,4,16);
    puts("native cyclic product gates PASS");
}
