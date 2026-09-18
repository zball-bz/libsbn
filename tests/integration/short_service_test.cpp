#include "product_support.hpp"
#include "common/small_checks.h"
#include <cmath>
static void scenario(size_t an,size_t bn,unsigned alg,unsigned w,unsigned kind=0,unsigned bits=0){
    Fixture f(w);sbn3_mul_options o{};o.workers=w;o.algorithm=alg;o.borrow_output=1;
    o.trunk_bits=int(bits);
    sbn3_product_request req{};req.kind=static_cast<sbn3_product_kind>(kind);req.a_limbs=an;req.b_limbs=kind==SBN3_PRODUCT_SQR?0:bn;
    sbn3_product_info i{};sbn3_mul_plan p{};auto *b=f.product(req,o,i,p);if(alg)assert(i.mul.algorithm==alg);
    if(bits)assert(i.mul.trunk_bits==bits);
    const size_t rn=up(an+bn?an+bn:1,i.mul.output_alignment/8);
    auto *a=an?f.guarded(an):nullptr,*y=bn?f.guarded(bn):nullptr,*r=f.guarded(rn);
    for(unsigned pattern=0;pattern<4;++pattern){for(size_t j=0;j<an;++j)a[j]=pattern==0?UINT64_MAX:random_word();for(size_t j=0;j<bn;++j)y[j]=pattern==0?UINT64_MAX:random_word();if(pattern==1&&an)memset(a,0,an*8);if(pattern==2&&an)memset(a+an/2,0,(an-an/2)*8);
        memset(r,0x91,rn*8);sbn3_product_inputs in{{a,an},kind==SBN3_PRODUCT_SQR?sbn3_const_limbs{}:sbn3_const_limbs{y,bn},{},{}};
        allocation_watch_start();sbn3_product_execute(b,&in,{r,an+bn});assert(!allocation_watch_stop());verify_product(a,an,kind==SBN3_PRODUCT_SQR?a:y,bn,r);
        if(kind==SBN3_PRODUCT_MUL){sbn3_mul_execute_ptrs(b,a,y,r);verify_product(a,an,y,bn,r);sbn3_int out{r,an+bn,0,0};allocation_watch_start();sbn3_int_mul_execute(b,&out,{a,an,1},{y,bn,pattern&1});assert(!allocation_watch_stop());verify_product(a,an,y,bn,r);assert(out.negative==(out.size?(1^(pattern&1)):0));}
        sbn3_product_metrics m{};sbn3_product_get_metrics(b,&m);assert(m.mul.workspace_used_bytes<=i.mul.workspace_bytes && m.mul.worker_peak_bytes<=i.mul.per_worker_bytes);
        for(size_t j=an+bn;j<rn;++j)assert(r[j]==0x9191919191919191ULL);
    }
    o.workspace_budget=1;sbn3_mul_plan untouched;memset(&untouched,0x63,sizeof untouched);sbn3_product_info rejected{};assert(sbn3_product_query(&req,&o,&untouched,&rejected)==SBN3_QUERY_CAPACITY&&untouched.opaque[0]==0x6363636363636363ULL);
    printf("short service alg%u %zu x %zu W%u kind%u: GMP/sign/budget/Frame/alloc/guard OK\n",i.mul.algorithm,an,bn,w,kind);
}
static void variable_plans(){
    sbn3_product_spec s{512,512};sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;
    sbn3_mul_plan p{};sbn3_mul_info a{},b{};o.trunk_bits=16;assert(sbn3_mul_query(&s,&o,&p,&a)==SBN3_SUPPORTED);
    o.trunk_bits=17;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_SUPPORTED);
    assert(a.transform_trunks==b.transform_trunks&&a.basis_id!=b.basis_id&&a.arithmetic_id!=b.arithmetic_id&&a.execution_id!=b.execution_id);
    for(unsigned bits:{17u,18u,19u})for(size_t n:{512u,550u,608u})scenario(n,n,SBN3_MUL_PQ16,1,SBN3_PRODUCT_MUL,bits);
    for(unsigned j=0;j<=64;++j){size_t n=size_t(std::llround(std::exp2(9.+j/16.)));scenario(n,n,SBN3_MUL_PQ16,1);}
    for(unsigned j=0;j<32;j+=2){size_t n=size_t(std::llround(std::exp2(7.+j/16.)));scenario(n,n,SBN3_MUL_PQ16,1);}   // small band (2026-09-09): the codec opens at 128 limbs
    for(size_t n:{550u,583u,1069u,1117u,1722u,2233u,4277u})scenario(n,n,SBN3_MUL_PQ16,1,SBN3_PRODUCT_SQR);
    scenario(1023,1,SBN3_MUL_PQ16,1,SBN3_PRODUCT_MUL,19);scenario(12001,411,SBN3_MUL_PQ16,1,SBN3_PRODUCT_MUL,17);
    // Odd digit-count sums can use the last real coefficient slot.
    // The 64-limb-bit boundary must still close without output padding.
    for(unsigned bits:{17u,18u,19u,20u}){const size_t N=bits==17?24576:bits==18?8192:bits==19?2048:512,total=N*bits/32;
        scenario(total-1,1,SBN3_MUL_PQ16,1,SBN3_PRODUCT_MUL,bits);
        scenario(total/2-1,total/2+1,SBN3_MUL_PQ16,1,SBN3_PRODUCT_MUL,bits);}
    o.trunk_bits=19;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_SUPPORTED&&b.workspace_bytes<a.workspace_bytes);
    o.trunk_bits=0;o.workspace_budget=b.workspace_bytes;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_SUPPORTED&&b.workspace_bytes<=o.workspace_budget&&b.trunk_bits>16);
    o.workspace_budget=0;o.trunk_bits=21;memset(&p,0x63,sizeof p);assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_UNSUPPORTED&&p.opaque[0]==0x6363636363636363ULL);   // 21-bit digits: out of every envelope (2026-09-09)
    o.trunk_bits=20;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_SUPPORTED&&b.trunk_bits==20&&(b.codec_mode&8u));   // 20-bit balanced (M7 1792 CT) at 512 x 512
    o.trunk_bits=18;o.workers=2;assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_UNSUPPORTED);o.workers=1;s={8192,8192};assert(sbn3_mul_query(&s,&o,&p,&b)==SBN3_UNSUPPORTED);
    puts("variable FFT fine grid, explicit bits, basis identity and budget alternatives PASS");
}
static void budget_fallback(){
    constexpr size_t n=362;Fixture f(2);sbn3_mul_options o{};o.workers=2;
    sbn3_product_spec s{n,n};sbn3_mul_plan plan{};sbn3_mul_info ordinary{},small{};
    assert(sbn3_mul_query(&s,&o,&plan,&ordinary)==SBN3_SUPPORTED && ordinary.algorithm!=SBN3_MUL_SCALAR);
    o.algorithm=SBN3_MUL_SCALAR;assert(sbn3_mul_query(&s,&o,&plan,&small)==SBN3_SUPPORTED && small.workspace_bytes<ordinary.workspace_bytes);
    o.algorithm=SBN3_MUL_AUTO;o.workspace_budget=small.workspace_bytes;
    sbn3_product_request req{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};sbn3_product_info info{};auto *binding=f.product(req,o,info,plan);
    assert(info.mul.algorithm==SBN3_MUL_SCALAR && info.mul.workspace_bytes<=o.workspace_budget);
    auto *a=f.guarded(n),*b=f.guarded(n),*r=f.guarded(2*n);
    for(size_t j=0;j<n;++j){a[j]=UINT64_MAX;b[j]=random_word();}
    allocation_watch_start();sbn3_mul_execute(binding,{a,n},{b,n},{r,2*n});assert(!allocation_watch_stop());verify_product(a,n,b,n,r);
    o.algorithm=ordinary.algorithm;memset(&plan,0x83,sizeof plan);
    assert(sbn3_mul_query(&s,&o,&plan,&small)==SBN3_QUERY_CAPACITY && plan.opaque[0]==0x8383838383838383ULL);
    o={};o.workers=1;s={size_t(1)<<21,1};assert(sbn3_mul_query(&s,&o,&plan,&small)==SBN3_SUPPORTED && small.algorithm==SBN3_MUL_SCALAR);
    puts("AUTO budget alternatives / explicit rejection / long mul_1 query PASS");
}
struct Job{sbn3_mul_binding *b;sbn3_product_inputs in;sbn3_limbs out;};
static void job(void *a,sbn3_team_scope *s){auto &j=*static_cast<Job *>(a);sbn3_product_execute_on_scope(j.b,s,&j.in,j.out);}
static void pair(void *a,sbn3_team_scope *s){auto *j=static_cast<Job *>(a);sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,2,job,j,job,j+1);}
static void concurrent(){
    Fixture f(4);Job jobs[2]{};size_t ns[]={128,8192};for(unsigned k=0;k<2;++k){auto n=ns[k];sbn3_mul_options o{};o.workers=2;o.algorithm=k?SBN3_MUL_PQ16:SBN3_MUL_U52;sbn3_product_request req{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};sbn3_product_info i{};sbn3_mul_plan p{};jobs[k].b=f.product(req,o,i,p);auto *a=f.guarded(n),*b=f.guarded(n),*r=f.guarded(2*n);for(size_t j=0;j<n;++j){a[j]=random_word();b[j]=random_word();}jobs[k].in={{a,n},{b,n},{},{}};jobs[k].out={r,2*n};}
    for(unsigned rep=0;rep<4;++rep){allocation_watch_start();sbn3_team_run(f.team,pair,jobs);assert(!allocation_watch_stop());for(unsigned k=0;k<2;++k)verify_product(jobs[k].in.a.data,ns[k],jobs[k].in.b.data,ns[k],jobs[k].out.data);}
    puts("mixed u52/PQ16 concurrent child scopes OK");
}
static void pair_ct(void *a,sbn3_team_scope *s){auto *j=static_cast<Job *>(a);sbn3_team_invoke2(s,SBN3_PARALLEL_CHILDREN,1,job,j,job,j+1);}
static void concurrent_ct(){
    Fixture f(2);Job jobs[2]{};const size_t lengths[2]={609,790};
    for(unsigned k=0;k<2;++k){size_t n=lengths[k];sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;
        sbn3_product_request req{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};sbn3_product_info i{};sbn3_mul_plan p{};jobs[k].b=f.product(req,o,i,p);assert(k?(i.mul.codec_mode==1||i.mul.codec_mode==2):(i.mul.codec_mode&8u)); // 609: 20-bit balanced pow2 (2026-09-09); 790: 17-bit M3 -> CT/PQ or right-angle by cost
        auto *a=f.guarded(n),*b=f.guarded(n),*r=f.guarded(2*n);for(size_t j=0;j<n;++j){a[j]=random_word();b[j]=random_word();}jobs[k].in={{a,n},{b,n},{},{}};jobs[k].out={r,2*n};
    }
    for(unsigned rep=0;rep<4;++rep){allocation_watch_start();sbn3_team_run(f.team,pair_ct,jobs);assert(!allocation_watch_stop());for(unsigned k=0;k<2;++k)verify_product(jobs[k].in.a.data,lengths[k],jobs[k].in.b.data,lengths[k],jobs[k].out.data);}
    sbn3_product_spec s{1218,1218};sbn3_mul_options o{};o.algorithm=SBN3_MUL_PQ16;sbn3_mul_plan p{};sbn3_mul_info ct{},pfa{};   // 1218: 16-bit M5 5120 right-angle at W1 (609 moved to 20-bit balanced pow2, 2026-09-09)
    o.workers=1;assert(sbn3_mul_query(&s,&o,&p,&ct)==SBN3_SUPPORTED);o.workers=2;assert(sbn3_mul_query(&s,&o,&p,&pfa)==SBN3_SUPPORTED);
    assert(ct.codec_mode==2&&pfa.codec_mode==0&&ct.basis_id==pfa.basis_id&&ct.arithmetic_id==pfa.arithmetic_id&&ct.execution_id!=pfa.execution_id&&ct.table_bytes>pfa.table_bytes);
    puts("CT/RAC child scopes and recipe identity PASS");
}
static void resource_envelope(){
    for(unsigned width:{1u,4u}){
        Fixture f(width);constexpr size_t n=16;
        // Values allocated before binding lie inside its resource envelope,
        // but outside every actual resource. They must remain accepted.
        auto *a=f.guarded(n),*y=f.guarded(n),*r=f.guarded(2*n);
        for(size_t j=0;j<n;++j){a[j]=random_word();y[j]=random_word();}
        sbn3_mul_options o{};o.workers=width;o.algorithm=SBN3_MUL_U52;
        sbn3_product_request req{SBN3_PRODUCT_MUL,n,n,0,0,{},0,0,0};sbn3_product_info i{};sbn3_mul_plan p{};auto *b=f.product(req,o,i,p);
        allocation_watch_start();sbn3_mul_execute(b,{a,n},{y,n},{r,2*n});assert(!allocation_watch_stop());verify_product(a,n,y,n,r);
        if(width==1 && SBN3_CHECK_SMALL){
            const auto child=fork();assert(child>=0);
            if(!child){rlimit no_core{0,0};setrlimit(RLIMIT_CORE,&no_core);close(STDERR_FILENO);
                sbn3_mul_execute(b,{a,n},{y,n},{reinterpret_cast<uint64_t *>(b),2*n});_exit(0);}
            int status=0;assert(waitpid(child,&status,0)==child && WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);
        }
    }
    printf("resource gaps accepted; misuse validation=%d PASS\n",SBN3_CHECK_SMALL);
}
int main(){
    for(unsigned alg:{unsigned(SBN3_MUL_SCALAR),unsigned(SBN3_MUL_U52),unsigned(SBN3_MUL_PQ16)}){
        for(size_t n:{1u,2u,3u,4u,5u,6u,8u,16u,64u,128u,256u,512u}){scenario(n,n,alg,1);scenario(n,n,alg,2,SBN3_PRODUCT_SQR);}
        scenario(513,117,alg,3);
        scenario(64,64,alg,1,SBN3_PRODUCT_SQR);
    }
    for(size_t n:{1u,8u,9u,64u,255u,256u,512u,8192u})scenario(n,n,SBN3_MUL_AUTO,1);
    for(size_t n:{609u,664u,790u})scenario(n,n,SBN3_MUL_PQ16,1,SBN3_PRODUCT_SQR);
    for(size_t n:{279u,362u,470u,1025u,2049u,4097u,8193u,16385u,32769u})for(unsigned w:{1u,8u,16u})scenario(n,n,SBN3_MUL_AUTO,w);
    for(size_t n:{2049u,8193u}){scenario(n,n,SBN3_MUL_AUTO,4,SBN3_PRODUCT_SQR);scenario(n,n/7,SBN3_MUL_AUTO,3);}
    scenario(32769,2,SBN3_MUL_AUTO,1);scenario(2,32769,SBN3_MUL_AUTO,3);
    budget_fallback();
    variable_plans();
    scenario(0,8,SBN3_MUL_AUTO,1);scenario(0,0,SBN3_MUL_AUTO,1);concurrent();concurrent_ct();resource_envelope();
    sbn3_product_spec s{131072,131072};sbn3_mul_options o{};o.workers=1;o.algorithm=SBN3_MUL_PQ16;sbn3_mul_plan p{};sbn3_mul_info i{};assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_UNSUPPORTED);
    o.prime_count=6;s={64,64};assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_UNSUPPORTED);
    puts("short public service PASS");
}
