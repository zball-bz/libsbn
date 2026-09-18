#include "sbn3/mul.h"
#include "runtime/arena.hpp"
#include <assert.h>
#include "../oracle/oracle.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <atomic>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <unistd.h>
#include <sched.h>
extern "C" void allocation_watch_start();
extern "C" uint64_t allocation_watch_stop();

static size_t up(size_t n,size_t a){return (n+a-1)&~(a-1);}
static uint64_t rng=0x382a79e1be224411ULL;
static uint64_t random_word(){rng^=rng<<13;rng^=rng>>7;rng^=rng<<17;return rng;}
static uint64_t tick(){timespec t{};clock_gettime(CLOCK_MONOTONIC,&t);return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;}
struct Fixture {
    sbn3_arena *arena=nullptr;
    sbn3_team *team=nullptr;
    std::vector<sbn3_lease> leases;
    std::vector<sbn3_mul_binding *> bindings;
    uintptr_t base=0;size_t cursor=0;
    explicit Fixture(unsigned workers) {
        sbn3_error e{};sbn3_arena_config ac{size_t(1)<<34,size_t(256)<<20};
        assert(sbn3_arena_create(&ac,&arena,&e)==SBN3_OK);
        auto control=allocate(sbn3_team_storage_bytes(),64);
        base=reinterpret_cast<uintptr_t>(control.data);
        sbn3_team_config tc{};tc.workers=workers;tc.pin_threads=1;tc.stack_offset=64u<<10;
        for(int &c:tc.cpu_ids)c=-1;
        const auto status=sbn3_team_create(arena,&control,&tc,&team,&e);
        if(status!=SBN3_OK)fprintf(stderr,"extended team: %d %s errno=%d\n",status,e.where,e.system_error);
        assert(status==SBN3_OK);
        cursor=tc.stack_offset+sbn3_team_stack_virtual_bytes(workers)+4096;
    }
    sbn3_lease allocate(size_t bytes,size_t alignment) {
        cursor=up(base+cursor,alignment)-base;
        sbn3_error e{};auto status=sbn3_arena_prepare(arena,cursor,bytes,&e);
        if(status!=SBN3_OK)fprintf(stderr,"extended prepare: %d %s errno=%d bytes=%zu\n",status,e.where,e.system_error,bytes);
        assert(status==SBN3_OK);
        sbn3_lease l{};sbn3_arena_acquire(arena,cursor,bytes,&l);leases.push_back(l);
        cursor=up(cursor+bytes,4096)+4096;return l;
    }
    uint64_t *guarded(size_t limbs) {
        const size_t begin=up(cursor,4096),end=up(begin+limbs*8,4096),start=end-limbs*8;
        sbn3_error e{};assert(sbn3_arena_prepare(arena,begin,end-begin,&e)==SBN3_OK);
        sbn3_lease value{},guard{};sbn3_arena_acquire(arena,start,limbs*8,&value);
        assert(arena->reserve_guard(end,4096,guard)==SBN3_OK);
        leases.push_back(value);leases.push_back(guard);cursor=end+4096;
        return static_cast<uint64_t *>(value.data);
    }
    sbn3_mul_binding *bind(size_t an,size_t bn,sbn3_mul_options options,sbn3_mul_info &info) {
        sbn3_product_spec spec{an,bn};sbn3_mul_plan plan{};
        allocation_watch_start();
        assert(sbn3_mul_query(&spec,&options,&plan,&info)==SBN3_SUPPORTED);
        assert(allocation_watch_stop()==0);
        auto tables=allocate(info.table_bytes,64),work=allocate(info.workspace_bytes,info.workspace_alignment);
        sbn3_mul_binding *b=nullptr;allocation_watch_start();sbn3_mul_bind(&plan,arena,&tables,&work,team,&b);
        assert(allocation_watch_stop()==0);bindings.push_back(b);return b;
    }
    ~Fixture(){
        for(auto *b:bindings)sbn3_mul_unbind(b);
        sbn3_team_destroy(team);
        for(auto &l:leases)sbn3_arena_release(arena,&l);
        sbn3_arena_stats stats{};sbn3_arena_get_stats(arena,&stats);
        assert(!stats.lease_references && !stats.active_computations);sbn3_arena_destroy(arena);
    }
};
static void verify_product(const uint64_t *a,size_t an,const uint64_t *b,size_t bn,const uint64_t *r){
    assert(ref_product_equal(a,an,b,bn,r,an+bn));
}
static void shape(size_t an,size_t bn,unsigned w,unsigned group,int T,unsigned cl,unsigned rl,unsigned np=6,unsigned crt=1,unsigned codec=1){
    Fixture f(w);sbn3_mul_options options{};options.workers=w;options.prime_count=np;options.crt_mode=crt;options.codec_mode=codec;options.prime_batch=group;
    options.trunk_bits=T;options.borrow_output=1;options.column_log2=cl;options.row_log2=rl;
    sbn3_mul_info info{};auto *bound=f.bind(an,bn,options,info);
    auto *a=f.guarded(an),*b=f.guarded(bn);const size_t rn=up(an+bn,8);
    auto *r=f.guarded(rn);assert(!(reinterpret_cast<uintptr_t>(r)&63));
    for(unsigned pattern=0;pattern<6;++pattern){
        for(size_t i=0;i<an;++i)a[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(i&1?0:UINT64_MAX):random_word();
        for(size_t i=0;i<bn;++i)b[i]=pattern==0?0:pattern==1?UINT64_MAX:pattern==2?(i&1?UINT64_MAX:0):random_word();
        if(pattern==3){a[an-1]=0;b[bn-1]=0;}
        memset(r,0x9d,rn*8);
        sbn3_arena_stats before{},after{};sbn3_arena_get_stats(f.arena,&before);
        allocation_watch_start();sbn3_mul_execute(bound,{a,an},{b,bn},{r,an+bn});assert(allocation_watch_stop()==0);
        sbn3_arena_get_stats(f.arena,&after);assert(before.prepare_calls==after.prepare_calls && before.trim_calls==after.trim_calls);
        verify_product(a,an,b,bn,r);
        for(size_t j=an+bn;j<rn;++j)assert(r[j]==0x9d9d9d9d9d9d9d9dULL);
    }
    sbn3_mul_metrics m{};sbn3_mul_get_metrics(bound,&m);
    assert(m.worker_peak_bytes<=info.per_worker_bytes && m.table_used_bytes<=info.table_bytes && m.workspace_used_bytes<=info.workspace_bytes);
    printf("np=%u crt=%u codec=%u extended %zu x %zu W=%u g=%u T=%u C=%zu M2=%zu full=%u fused=%u: guard/reference/modular oracle/footprint OK\n",np,crt,codec,an,bn,w,group,info.trunk_bits,info.C,info.M2,info.full,info.fused);
}
struct Product {sbn3_mul_binding *binding;uint64_t *a,*b,*r;size_t an,bn;};
struct Tree {Product *products;unsigned count;std::atomic<unsigned> *arrivals;uint64_t deadline;};
static void tree(void *arg,sbn3_team_scope *scope){
    auto &t=*static_cast<Tree *>(arg);
    if(t.count==1){
        t.arrivals->fetch_add(1,std::memory_order_release);
        while(t.arrivals->load(std::memory_order_acquire)<4){assert(tick()<t.deadline);sched_yield();}
        auto &p=*t.products;sbn3_mul_execute_on_scope(p.binding,scope,{p.a,p.an},{p.b,p.bn},{p.r,p.an+p.bn});return;
    }
    unsigned left=t.count/2;Tree a{t.products,left,t.arrivals,t.deadline},b{t.products+left,t.count-left,t.arrivals,t.deadline};
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,sbn3_team_width(scope)/2,tree,&a,tree,&b);
}
static void nested_products(bool wide=false){
    Fixture f(16);Product products[4]{};
    for(unsigned k=0;k<4;++k){auto &p=products[k];p.an=127+8*k;p.bn=257-8*k;
        sbn3_mul_options o{};o.prime_count=6;o.crt_mode=1;o.workers=4;o.prime_count=wide?(k==0?6:7+k):4+k;o.prime_batch=2;o.borrow_output=1;o.column_log2=6;o.row_log2=4;
        sbn3_mul_info info{};p.binding=f.bind(p.an,p.bn,o,info);
        p.a=f.guarded(p.an);p.b=f.guarded(p.bn);p.r=f.guarded(p.an+p.bn);
        for(size_t j=0;j<p.an;++j)p.a[j]=random_word();for(size_t j=0;j<p.bn;++j)p.b[j]=random_word();}
    std::atomic<unsigned> arrivals{0};Tree root{products,4,&arrivals,tick()+10000000000ULL};
    allocation_watch_start();sbn3_team_run(f.team,tree,&root);assert(allocation_watch_stop()==0);
    for(const auto &p:products)verify_product(p.a,p.an,p.b,p.bn,p.r);
    puts("actual NP4/5/6/7 products in 16->8+8->4+4 subteams: concurrent/reference/modular oracle OK");
}
static void invalid_binding(unsigned kind) {
    Fixture f(1);sbn3_product_spec s{8,8};sbn3_mul_options o{};o.prime_count=6;o.crt_mode=1;o.workers=1;o.column_log2=6;o.row_log2=4;
    sbn3_mul_plan p{};sbn3_mul_info i{};assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_SUPPORTED);
    auto t=f.allocate(i.table_bytes,64),w=f.allocate(i.workspace_bytes,i.workspace_alignment);
    sbn3_mul_binding *binding=nullptr;
    if(kind==0){p.opaque[1]^=1;sbn3_mul_bind(&p,f.arena,&t,&w,f.team,&binding);}
    if(kind==1){p.opaque[9]^=1;sbn3_mul_bind(&p,f.arena,&t,&w,f.team,&binding);}
    if(kind==2){--w.bytes;sbn3_mul_bind(&p,f.arena,&t,&w,f.team,&binding);}
    if(kind==3){t=w;sbn3_mul_bind(&p,f.arena,&t,&w,f.team,&binding);}
    sbn3_mul_bind(&p,f.arena,&t,&w,f.team,&binding);
    auto *a=f.guarded(8),*b=f.guarded(8),*r=f.guarded(16);
    if(kind==4)sbn3_mul_execute(binding,{a,8},{b,8},{a,16});
    if(kind==5)sbn3_mul_execute(binding,{a,7},{b,8},{r,16});
    if(kind==6)sbn3_mul_execute(binding,{a,8},{b,8},{r+1,16});
    _exit(99);
}
static void negative_gates(){
    sbn3_product_spec s{127,257};sbn3_mul_options o{};o.prime_count=6;o.crt_mode=1;o.workers=16;sbn3_mul_plan p{},one{};sbn3_mul_info i{},small{};
    assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_SUPPORTED && i.prime_batch==6);
    o.prime_batch=1;assert(sbn3_mul_query(&s,&o,&one,&small)==SBN3_SUPPORTED);
    o.prime_batch=0;o.workspace_budget=small.workspace_bytes;
    assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_SUPPORTED && i.prime_batch==1 && i.arithmetic_id==small.arithmetic_id);
    o.workspace_budget=0;o.column_log2=6;o.row_log2=3;
    assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_UNSUPPORTED);
    o.column_log2=0;o.row_log2=0;o.workers=33;assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_UNSUPPORTED);
    o.workers=1;s.a_limbs=SIZE_MAX;assert(sbn3_mul_query(&s,&o,&p,&i)==SBN3_QUERY_CAPACITY);
    for(unsigned k=0;k<7;++k){pid_t pid=fork();assert(pid>=0);if(!pid){rlimit z{0,0};setrlimit(RLIMIT_CORE,&z);invalid_binding(k);}
        int status=0;assert(waitpid(pid,&status,0)==pid && WIFSIGNALED(status) && WTERMSIG(status)==SIGABRT);}
    puts("query budget fallback / invalid geometry / overflow / 7 binding fatal gates OK");
}
static void chooser_gates(){
    for(size_t n:{size_t(4097),size_t(1)<<20,size_t(1)<<24}){
        sbn3_product_spec spec{n,n/3};sbn3_mul_options o{};o.workers=5;o.algorithm=SBN3_MUL_BAILEY;sbn3_mul_plan plan{};sbn3_mul_info info{};
        size_t smallest=SIZE_MAX;
        for(unsigned np=4;np<=8;++np){o.prime_count=np;assert(sbn3_mul_query(&spec,&o,&plan,&info)==SBN3_SUPPORTED);
            if(info.workspace_bytes<smallest)smallest=info.workspace_bytes;}
        o.prime_count=0;o.workspace_budget=smallest;
        allocation_watch_start();assert(sbn3_mul_query(&spec,&o,&plan,&info)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
        assert(info.workspace_bytes<=smallest && info.np>=4 && info.np<=8);
        memset(&plan,0xb7,sizeof plan);o.workspace_budget=1;
        assert(sbn3_mul_query(&spec,&o,&plan,&info)==SBN3_QUERY_CAPACITY && plan.opaque[0]==0xb7b7b7b7b7b7b7b7ULL);
    }
    sbn3_product_spec spec{4097,1021};sbn3_mul_options a{};a.workers=3;a.prime_count=7;a.trunk_bits=152;a.crt_mode=2;a.codec_mode=1;
    sbn3_mul_plan p{},q{};sbn3_mul_info x{},y{};
    assert(sbn3_mul_query(&spec,&a,&p,&x)==SBN3_SUPPORTED);a.codec_mode=3;
    assert(sbn3_mul_query(&spec,&a,&q,&y)==SBN3_SUPPORTED);
    assert(x.arithmetic_id==y.arithmetic_id && x.execution_id!=y.execution_id && x.basis_id==y.basis_id);
    a.prime_count=0;assert(sbn3_mul_query(&spec,&a,&p,&x)==SBN3_SUPPORTED && x.np==7);
    a.codec_mode=2;assert(sbn3_mul_query(&spec,&a,&p,&x)==SBN3_UNSUPPORTED);a.codec_mode=0;
    a.prime_count=11;assert(sbn3_mul_query(&spec,&a,&p,&x)==SBN3_UNSUPPORTED);
    puts("NP4-8 chooser: budget/forced T/identity/no-allocation OK");
}
int main(){
    negative_gates();chooser_gates();
    shape(32768,32768,1,1,128,3,10);
    shape(32768,32768,4,4,128,4,9);
    shape(32768,32768,8,6,128,5,8);
    shape(8192,8192,2,4,84,3,8,4); // C8 cannot form a T84 fused row group; use the legal unfused tail
    shape(127,257,3,4,136,6,4);
    shape(8192,8192,5,5,128,6,4);
    shape(8192,4096,1,1,136,6,14);
    shape(8192,4096,3,5,128,6,14,6,1); // OT=4, ntp ends halfway through a vector
    shape(8191,4096,3,5,128,6,14,6,2);
    shape(16383,8193,16,6,120,0,0);
    shape(32768,65536,5,2,128,0,0);
    nested_products();nested_products(true);
    shape(8192,4096,3,8,192,6,14,9,2);
    shape(8191,4096,3,8,184,6,14,9,1);
    for(unsigned np=4;np<=10;++np){
        const int top=np==4?88:24*int(np)-8, step=np==4?4:8, count=np==4?3:4;
        for(int k=0;k<count;++k){const int T=top-k*step;
            for(unsigned crt=1;crt<=2;++crt){
                shape(257,511,3,np-1,T,6,4,np,crt);
                shape(T*64,T*64,5,np,T,6,4,np,crt,1);
            }
        }
        shape(8192,4096,3,np-1,top-step,6,14,np,2);
        if(np>4){for(int k=0;k<count;++k){const int T=top-k*step;if(T<=88)continue;
            shape(257,511,5,np,T,6,4,np,2,3);shape(T*64,T*64,3,np-1,T,6,4,np,1,3);}
            shape(8192,4096,3,np-1,top-step,6,14,np,2,3);
        }
    }
}
