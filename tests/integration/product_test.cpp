#include "product_support.hpp"
static void check(const sbn3_product_request &req,const sbn3_product_inputs &full,const sbn3_product_info &info,const uint64_t *out){
    ref_int a,b,r,t,got,limit,mod;ref_inits(a,b,r,t,got,limit,mod,nullptr);
    ref_import(a,full.a.count,-1,8,0,0,full.a.data);
    if(req.kind==SBN3_PRODUCT_SQR)ref_set(b,a);else ref_import(b,full.b.count,-1,8,0,0,full.b.data);
    ref_mul(r,a,b);
    if(req.kind==SBN3_PRODUCT_MAC2){ref_import(a,full.a1.count,-1,8,0,0,full.a1.data);ref_import(b,full.b1.count,-1,8,0,0,full.b1.data);ref_addmul(r,a,b);}
    ref_import(got,info.mul.output_limbs,-1,8,0,0,out);
    if(req.kind==SBN3_PRODUCT_TMP){const auto &c=info.window;ref_fdiv_q_2exp(r,r,c.offset_bits);ref_sub(t,got,r);ref_fdiv_r_2exp(t,t,c.width_bits);
        ref_set_ui(limit,1);ref_mul_2exp(limit,limit,c.error_bits);ref_set_ui(mod,1);ref_mul_2exp(mod,mod,c.width_bits);ref_sub(mod,mod,limit);
        if(!(ref_cmp(t,limit)<0 || ref_cmp(t,mod)>0)){fprintf(stderr,"TMP delta outside certificate\n");assert(false);}
    }else if(ref_cmp(r,got)){fprintf(stderr,"kind=%u product mismatch\n",req.kind);assert(false);}
    ref_clears(a,b,r,t,got,limit,mod,nullptr);
}
static uint64_t hash_cache(const sbn3_spectrum *s,const sbn3_spectrum_desc &d){
    // Refcount/header may change. Hash table + plane region after the immutable descriptor.
    const auto *p=reinterpret_cast<const unsigned char *>(s);uint64_t h=1;
    // Whole storage is initialized below by the fixture before prepare; skip the mutable header.
    for(size_t k=1024;k<d.storage_bytes;++k)h=(h^p[k])*1099511628211ULL;return h;
}
static void scenario(unsigned np,int T,unsigned kind,unsigned mask,unsigned frontier,size_t an=129,size_t bn=513,bool full=false,unsigned crt=0,unsigned rowlog=4,unsigned algorithm=0){
    Fixture f(3);sbn3_mul_options opt{};opt.workers=3;opt.prime_count=np;opt.trunk_bits=T;opt.crt_mode=crt;opt.column_log2=6;opt.row_log2=rowlog;opt.borrow_output=1;opt.algorithm=algorithm;if(algorithm==SBN3_MUL_FLAT)opt.column_log2=opt.row_log2=0;
    if(full){an=size_t(4096)*T/64;bn=an;if(kind==SBN3_PRODUCT_TMP){an=256*T/64;bn=8184*T/64;}}
    if(kind==SBN3_PRODUCT_SQR)bn=an;
    auto *a=f.guarded(an),*b=f.guarded(bn),*a1=f.guarded(an),*b1=f.guarded(bn);
    for(size_t j=0;j<an;++j)a[j]=a1[j]=UINT64_MAX;
    for(size_t j=0;j<bn;++j)b[j]=b1[j]=UINT64_MAX;
    sbn3_product_request req{};req.kind=static_cast<sbn3_product_kind>(kind);req.a_limbs=an;req.b_limbs=kind==SBN3_PRODUCT_SQR?0:bn;
    if(kind==SBN3_PRODUCT_MAC2){req.a1_limbs=an;req.b1_limbs=bn;}
    sbn3_product_inputs fullin{{a,an},{b,bn},{a1,an},{b1,bn}};
    sbn3_product_info info{};sbn3_mul_plan plan{};auto *seed=f.product(req,opt,info,plan);
    sbn3_spectrum *handles[2]{};sbn3_spectrum_desc desc[2]{};
    for(unsigned t=0;t<2;++t)if(mask&(1u<<t)){handles[t]=f.cache(seed,{t?a1:a,an},frontier^(t&1),info,desc[t]);req.cached_a[t]=&desc[t];}
    sbn3_mul_binding *bound=seed;
    if(mask){f.unbind(seed);bound=f.product(req,opt,info,plan,handles[0],handles[1]);
        for(unsigned t=0;t<2;++t)if(handles[t]){assert(sbn3_spectrum_can_apply(&plan,handles[t],t));sbn3_spectrum_retain(handles[t]);sbn3_spectrum_release(handles[t]);}}
    uint64_t hashes[2]{};for(unsigned t=0;t<2;++t)if(handles[t])hashes[t]=hash_cache(handles[t],desc[t]);
    const size_t rn=up(info.mul.output_limbs,8);auto *out=f.guarded(rn);
    sbn3_product_inputs input=fullin;if(mask&1)input.a={};if(mask&2)input.a1={};if(kind==SBN3_PRODUCT_SQR)input.b={};if(kind!=SBN3_PRODUCT_MAC2)input.a1=input.b1={};
    for(unsigned pattern=0;pattern<4;++pattern){
        if(pattern){if(!(mask&1))for(size_t j=0;j<an;++j)a[j]=pattern==1?0:random_word();
            if(!(mask&2))for(size_t j=0;j<an;++j)a1[j]=random_word();
            for(size_t j=0;j<bn;++j){b[j]=pattern==1?0:random_word();b1[j]=random_word();}}
        memset(out,0x9d,rn*8);allocation_watch_start();sbn3_product_execute(bound,&input,{out,info.mul.output_limbs});assert(!allocation_watch_stop());
        check(req,fullin,info,out);for(size_t j=info.mul.output_limbs;j<rn;++j)assert(out[j]==0x9d9d9d9d9d9d9d9dULL);
        sbn3_product_metrics m{};sbn3_product_get_metrics(bound,&m);const auto &g=info.mul;const uint64_t cols=np*g.lbw;
        assert(m.mul.worker_peak_bytes<=g.per_worker_bytes && m.mul.workspace_used_bytes<=g.workspace_bytes);
        if(g.algorithm==SBN3_MUL_FLAT){
            assert(!m.column_inverse && !m.column_forward && m.leaf_products==np*(g.lbw/8)*(kind==SBN3_PRODUCT_MAC2?2:1));
            if(kind==SBN3_PRODUCT_TMP)assert(m.row_mid==np && m.row_transpose_forward==np);
            else assert(m.row_inverse==np);
            continue;
        }
        assert(m.column_inverse==cols && m.leaf_products==cols*(g.C/8)*(kind==SBN3_PRODUCT_MAC2?2:1));
        unsigned fwd=kind==SBN3_PRODUCT_SQR?unsigned(!(mask&1)||!frontier):1+unsigned(!(mask&1)||!frontier);
        if(kind==SBN3_PRODUCT_MAC2)fwd+=1+unsigned(!(mask&2)||!(frontier^1));
        assert(m.column_forward==cols*fwd);
        if(kind==SBN3_PRODUCT_TMP)assert(m.row_mid==np*g.C && m.row_transpose_forward==np*g.C);
        else assert(m.row_inverse==np*(g.C+(g.fused?g.fused_items:0)));
    }
    for(unsigned t=0;t<2;++t)if(handles[t]){assert(hash_cache(handles[t],desc[t])==hashes[t]);sbn3_spectrum_release(handles[t]);}
    printf("product np=%u T=%d kind=%u mask=%u h=%u full=%u: reference/modular oracle/cert/counts/read-only/alloc/guard OK\n",np,T,kind,mask,frontier,info.mul.full);fflush(stdout);
}
#include "product_reuse_tests.hpp"
#include "small_policy_tests.hpp"
int main(){
    small_policy_query_gate();for(unsigned w:{1u,2u,4u,8u,16u})small_batch_gate(w);
    distinct_mac_reuse();distinct_mac_reuse(SBN3_MUL_FLAT);reuse_gate(0,SBN3_MUL_FLAT);reuse_gate(1,SBN3_MUL_FLAT);capacity_gate();reuse_gate(0);reuse_gate(1);unequal_mac();
    for(unsigned np:{6u,8u})for(unsigned kind=1;kind<=3;++kind)scenario(np,24*np-8,kind,1,1,129,513,false,0,14);
    scenario(6,128,0,0,0,8192,8194);scenario(6,128,3,3,1,8192,8194);
    scenario(6,128,1,0,0);scenario(6,128,0,1,0);scenario(6,128,0,1,1);
    for(unsigned np=4;np<=10;++np){const int top=24*np-8;
        for(unsigned kind=1;kind<=3;++kind){scenario(np,top,kind,0,0);scenario(np,top,kind,1,0);scenario(np,top,kind,1,1);}
        scenario(np,top,3,3,0);scenario(np,top,3,3,1);
        scenario(np,top,0,0,0,size_t(4096)*top/64,size_t(4096)*top/64+top/64);
        scenario(np,top,3,0,0,size_t(4096)*top/64,size_t(4096)*top/64+top/64);
        scenario(np,top,3,0,0,129,513,true);scenario(np,top,3,3,1,129,513,true);
        scenario(np,top,1,0,0,129,513,true);scenario(np,top,1,1,1,129,513,true);
        scenario(np,top,2,1,1,129,513,true);
    }
    for(int T:{80,84})scenario(4,T,3,3,1,129,513,true);
    for(int T:{80,84})for(unsigned kind=1;kind<=3;++kind)scenario(4,T,kind,1,1);
    for(unsigned np:{6u,8u,10u})for(unsigned kind=1;kind<=3;++kind)scenario(np,24*np-8,kind,1,1,129,513,false,1);
    for(unsigned np=4;np<=10;++np){const int T=24*np-8;
        for(unsigned kind=0;kind<=3;++kind){scenario(np,T,kind,0,0,129,513,false,0,4,SBN3_MUL_FLAT);
            scenario(np,T,kind,1,0,129,513,false,0,4,SBN3_MUL_FLAT);scenario(np,T,kind,1,1,129,513,false,0,4,SBN3_MUL_FLAT);}
        scenario(np,T,3,3,0,129,513,false,0,4,SBN3_MUL_FLAT);
        scenario(np,T,0,0,0,129,513,true,0,4,SBN3_MUL_FLAT);scenario(np,T,2,1,1,129,513,true,0,4,SBN3_MUL_FLAT);
        scenario(np,T,1,0,0,129,513,true,0,4,SBN3_MUL_FLAT);scenario(np,T,3,3,1,129,513,true,0,4,SBN3_MUL_FLAT);
    }
    for(int T:{80,84})for(unsigned kind=0;kind<=3;++kind)scenario(4,T,kind,1,1,129,513,false,1,4,SBN3_MUL_FLAT);
    for(int T:{80,84,88})for(unsigned kind=0;kind<=3;++kind)scenario(4,T,kind,1,1,129,513,false,2,4,SBN3_MUL_FLAT);
    scenario(6,128,0,1,0,24576,24576,false,2,4,SBN3_MUL_FLAT);
    scenario(6,128,0,1,1,32768,32768,false,2,4,SBN3_MUL_FLAT);
    puts("product recipes gates PASS");
}
