#include "product_support.hpp"
#include "product/program.hpp"
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>
using namespace sbn::v3;

static sbn3_mul_options options(unsigned workers) {
    sbn3_mul_options o{};
    o.workers=workers; o.prime_count=9; o.trunk_bits=200;
    o.algorithm=SBN3_MUL_BAILEY; o.column_log2=5; o.row_log2=5;
    return o;
}
static void scenario(bool ring,bool prefix,bool square,bool cached,unsigned workers) {
    Fixture f(workers);
    auto o=options(workers);
    sbn3_product_request r{};
    r.kind=square?SBN3_PRODUCT_SQR:SBN3_PRODUCT_MUL;
    r.a_limbs=ring?(square?25600:12801):129;
    r.b_limbs=square?0:ring?25600:513;
    r.cyclic_limbs=ring?25600:0;
    r.window_limbs=prefix?31:0;
    if(prefix){r.a_limbs=8533;r.b_limbs=8532;}
    auto *a=f.guarded(r.a_limbs),*b=f.guarded(r.b_limbs?r.b_limbs:8);
    for(size_t j=0;j<r.a_limbs;++j)a[j]=random_word();
    for(size_t j=0;j<r.b_limbs;++j)b[j]=random_word();
    sbn3_mul_plan p{}; sbn3_product_info old{},next{};
    auto *producer=f.product(r,o,old,p);
    sbn3_spectrum_desc desc{}; sbn3_spectrum *cache=nullptr;
    if(cached){cache=f.cache(producer,{a,r.a_limbs},1,old,desc);r.cached_a[0]=&desc;}
    auto *reference=cached?f.product(r,o,old,p,cache):producer;
    o.borrow_output=2;
    auto *binding=f.product(r,o,next,p,cache);
    const size_t capacity=sbn3_mul_output_capacity(&next.mul);
    assert(next.mul.arithmetic_id==old.mul.arithmetic_id && next.mul.basis_id==old.mul.basis_id);
    assert(next.mul.output_limbs==old.mul.output_limbs && capacity>=next.mul.output_limbs);
    assert(next.mul.borrow_output==2 && capacity*8>=next.mul.transpose_bytes);
    // If the reference uses a separate E1, borrowing removes that storage.
    // A backend may independently stream E1 in worker-local scratch instead.
    if(old.mul.transpose_bytes)assert(next.mul.workspace_bytes<old.mul.workspace_bytes);
    auto *expected=f.guarded(up(old.mul.output_limbs,8));
    const size_t guarded=up(capacity,8)+8;
    auto *out=f.guarded(guarded);
    sbn3_product_inputs in{};
    if(!cached)in.a={a,r.a_limbs};
    if(!square)in.b={b,r.b_limbs};
    for(unsigned repeat=0;repeat<2;++repeat){
        for(size_t j=0;j<guarded;++j)out[j]=UINT64_C(0x91ac7a663377abcd);
        allocation_watch_start();
        sbn3_product_execute(reference,&in,{expected,old.mul.output_limbs});
        sbn3_product_execute(binding,&in,{out,capacity});
        assert(!allocation_watch_stop());
        assert(!memcmp(out,expected,next.mul.output_limbs*8));
        for(size_t j=capacity;j<guarded;++j)assert(out[j]==UINT64_C(0x91ac7a663377abcd));
        if(!ring&&!square){
            sbn3_mul_execute_ptrs(binding,a,b,out);
            verify_product(a,r.a_limbs,b,r.b_limbs,out);
        }
    }
    if(cache)sbn3_spectrum_release(cache);
}
static void invalid(unsigned which) {
    Fixture f(1);
    auto o=options(1);o.borrow_output=2;
    sbn3_product_request r{};r.a_limbs=129;r.b_limbs=513;
    sbn3_mul_plan p{};sbn3_product_info i{};
    auto *binding=f.product(r,o,i,p);
    const size_t capacity=sbn3_mul_output_capacity(&i.mul);
    assert(capacity>i.mul.output_limbs+r.a_limbs+8);
    auto *out=f.guarded(up(capacity,8)),*a=f.guarded(r.a_limbs),*b=f.guarded(r.b_limbs);
    // The second input is disjoint from the logical result, but overlaps E1's tail.
    if(which)a=out+up(i.mul.output_limbs,8);
    sbn3_product_inputs in{{a,r.a_limbs},{b,r.b_limbs},{},{}};
    sbn3_product_execute(binding,&in,{out,which?capacity:i.mul.output_limbs});
    _exit(99);
}
static void program() {
    Fixture f(3);auto o=options(3);o.borrow_output=2;
    ProductProgramPlan p{};assert(product_program_query(129,513,o,p)==SBN3_SUPPORTED);
    assert(!(p.contract&program_consume_inputs));
    auto tables=f.allocate(p.prepared_bytes,128),work=f.allocate(p.info.workspace_bytes,p.info.workspace_alignment);
    Frame frame(*f.arena,tables);auto prepared=product_program_prepare(p,frame);
    auto *a=f.guarded(129),*b=f.guarded(513),*out=f.guarded(up(prepared.output_capacity,8));
    for(size_t j=0;j<129;++j)a[j]=random_word();
    for(size_t j=0;j<513;++j)b[j]=random_word();
    struct Task {ProductProgram *p;Fixture *f;sbn3_lease work;uint64_t *a,*b,*out;} task{&prepared,&f,work,a,b,out};
    allocation_watch_start();
    sbn3_team_run(f.team,[](void *ptr,sbn3_team_scope *scope){auto &t=*static_cast<Task*>(ptr);Frame scratch(*t.f->arena,t.work);
        product_program_execute(*t.p,scratch,scope,{t.a,129},{t.b,513},{t.out,t.p->output_capacity});},&task);
    assert(!allocation_watch_stop());verify_product(a,129,b,513,out);
}
static void square_build() {
    Fixture f(3);auto o=options(3);o.borrow_output=2;
    sbn3_product_request r{};r.kind=SBN3_PRODUCT_SQR;r.a_limbs=r.cyclic_limbs=25600;
    sbn3_mul_plan p{};sbn3_product_info i{};
    auto *plain=f.product(r,o,i,p);sbn3_spectrum_desc desc{};
    assert(sbn3_spectrum_query(&p,SBN3_SPECTRUM_COLUMNS,1,&desc)==SBN3_SUPPORTED);
    auto storage=f.allocate(desc.storage_bytes,64);sbn3_spectrum *cache=nullptr;
    sbn3_spectrum_reserve_plan(&p,SBN3_SPECTRUM_COLUMNS,1,f.arena,&storage,&cache);
    r.cached_a[0]=&desc;auto *bound=f.product(r,o,i,p,cache);
    const size_t cap=sbn3_mul_output_capacity(&i.mul);
    auto *a=f.guarded(r.a_limbs),*out=f.guarded(up(cap,8)),*expected=f.guarded(up(cap,8));
    for(size_t j=0;j<r.a_limbs;++j)a[j]=random_word();
    sbn3_product_inputs in{{a,r.a_limbs},{},{},{}};
    allocation_watch_start();
    sbn3_product_execute(plain,&in,{expected,cap});
    sbn3_spectrum_compute_square(bound,cache,{a,r.a_limbs},{out,cap});
    assert(!allocation_watch_stop());assert(!memcmp(out,expected,i.mul.output_limbs*8));
    sbn3_spectrum_release(cache);
}
static void geometry_replay() {
    // Large shapes are pure queries: test the implicit -> pinned transition
    // without allocating large buffers or timing a multiplication.
    for(size_t n:{size_t(190880570),size_t(105850760),size_t(94906266)})
        for(unsigned np:{6u,7u,8u,9u,10u}) {
            if((n==190880570||n==94906266)&&np<9)continue;
            sbn3_mul_options o{};o.workers=32;o.prime_count=np;o.algorithm=SBN3_MUL_BAILEY;
            ProductProgramPlan p{},replayed{};
            assert(product_program_query(n,n,o,p)==SBN3_SUPPORTED && p.info.C==4096);
            assert(product_program_rebatch(p,1,replayed)==SBN3_SUPPORTED);
            assert(replayed.info.arithmetic_id==p.info.arithmetic_id && replayed.info.basis_id==p.info.basis_id);
            sbn3_spectrum_desc spectrum{};
            assert(sbn3_spectrum_query(&p.plan,SBN3_SPECTRUM_COLUMNS,17,&spectrum)==SBN3_SUPPORTED);
            sbn3_product_request r{};r.a_limbs=r.b_limbs=n;r.cached_a[0]=&spectrum;
            sbn3_mul_plan cached{};sbn3_product_info ci{};
            assert(sbn3_product_query(&r,&o,&cached,&ci)==SBN3_SUPPORTED);
            assert(ci.mul.basis_id==p.info.basis_id && ci.mul.C==p.info.C && ci.mul.M2==p.info.M2);
            o.workers=16;ProductProgramPlan old{};
            assert(product_program_query(n,n,o,old)==SBN3_SUPPORTED && old.info.C!=4096);
            o.workers=32;o.column_log2=unsigned(__builtin_ctzll(old.info.C));
            o.row_log2=unsigned(__builtin_ctzll(old.info.M2));
            ProductProgramPlan explicit_plan{};
            assert(product_program_query(n,n,o,explicit_plan)==SBN3_SUPPORTED);
            assert(explicit_plan.info.basis_id==old.info.basis_id && explicit_plan.info.C==old.info.C);
        }
}
int main(){
    geometry_replay();
    for(unsigned which=0;which<2;++which){
        const pid_t pid=fork();assert(pid>=0);
        if(!pid){rlimit z{0,0};setrlimit(RLIMIT_CORE,&z);invalid(which);}
        int status=0;assert(waitpid(pid,&status,0)==pid&&WIFSIGNALED(status)&&WTERMSIG(status)==SIGABRT);
    }
    for(unsigned workers:{1u,3u}){
        scenario(false,false,false,false,workers);
        for(bool cached:{false,true}){
            scenario(true,false,false,cached,workers);
            scenario(true,true,false,cached,workers);
            scenario(true,false,true,cached,workers);
        }
    }
    program();square_build();
    puts("extended output capacity: arithmetic identity/guards/CYC/prefix/cache/SQR/program/alias/no allocation PASS");
}
