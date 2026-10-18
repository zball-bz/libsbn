// Original libsbn tests; LGPL-3.0-or-later.
#include "algorithms/normalized_newton.hpp"
#include "runtime/scratch.hpp"
#include "runtime/team.hpp"
#include "sbn3/execution.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
using namespace sbn::v3;
constexpr std::uint64_t guard=UINT64_C(0xf329071cad8b654e);
std::uint64_t seed=UINT64_C(0x170caa287469c0fe);
std::size_t certificates=0,native_calls=0;
std::uint64_t random_word(){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;return seed;}
struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
// Independent scalar accumulation, never the production-backed large
// product witness used by the general tests/oracle ref_mul entry.
void product(ref_number* out,const ref_number* a,const std::vector<std::uint64_t>& b){
    Ref sum,term;ref_set_ui(sum.value,0);
    for(std::size_t j=0;j<b.size();++j)if(b[j]){
        ref_mul_ui(term.value,a,b[j]);if(j)ref_mul_2exp(term.value,term.value,64*j);
        ref_add(sum.value,sum.value,term.value);
    }
    ref_set(out,sum.value);
}
struct Program {
    NormalizedNewtonInfo info{};
    sbn::workspace metadata;
    NormalizedNewtonPrepared prepared{};
    Program(sbn3_newton_kind kind,std::size_t n,unsigned workers){
        allocation_watch_start();const auto rc=normalized_newton_query(kind,n,workers,info);assert(!allocation_watch_stop());
        assert(rc==SBN3_SUPPORTED&&info.n==n&&info.kind==kind&&info.workers<=workers);
        assert(info.workspace_bytes<(std::size_t(128)<<20));
        metadata.reserve(info.metadata_bytes+64,info.alignment);
        auto bytes=metadata.on().scratch;
        std::fill(bytes.begin()+info.metadata_bytes,bytes.begin()+info.metadata_bytes+64,std::byte{0x31});
        auto frame=Frame::external(bytes.data(),info.metadata_bytes);
        allocation_watch_start();prepared=normalized_newton_prepare(kind,n,workers,frame);assert(!allocation_watch_stop());
        assert(prepared.info.workspace_bytes==info.workspace_bytes&&prepared.info.workers==info.workers&&
               prepared.info.local==info.local);
    }
    void check(){auto bytes=metadata.on().scratch;for(std::size_t j=0;j<64;++j)assert(bytes[info.metadata_bytes+j]==std::byte{0x31});}
};
struct Job {
    const NormalizedNewtonPrepared* prepared;
    sbn::workspace values,work;
    std::uint64_t *a,*d,*q;
    std::vector<std::uint64_t> original_a,original_d;
    bool alias;
    Job(const Program& p,unsigned pattern,bool inplace=false):prepared(&p.prepared),alias(inplace){
        const auto n=p.info.n;
        values.reserve(8*(3*n+6)+384,128);auto bytes=values.on().scratch;
        auto frame=Frame::external(bytes.data(),bytes.size());
        a=frame.alloc<std::uint64_t>(n+2);d=frame.alloc<std::uint64_t>(n+1);q=frame.alloc<std::uint64_t>(n+2);
        for(std::size_t j=0;j<=n;++j)a[j]=random_word();a[n]&=1;
        std::fill_n(d,n,0);
        // Three nonzero words keep the large reference product linear while
        // exercising a genuine long radicand and a nonzero discarded tail.
        d[n-1]=UINT64_C(1)<<63;
        if(pattern){d[0]|=UINT64_C(0xfed5);d[n/2]|=UINT64_C(0x3179);}
        if(pattern==2&&n<=1025)for(std::size_t j=0;j<n;++j)d[j]=random_word();
        d[n-1]|=UINT64_C(1)<<63;
        if(!pattern)std::fill_n(a,n+1,0);
        a[n+1]=d[n]=q[n+1]=guard;
        original_a.assign(a,a+n+1);original_d.assign(d,d+n);
        work.reserve(p.info.workspace_bytes+64,p.info.alignment);
        auto space=work.on().scratch;
        std::fill(space.begin()+p.info.workspace_bytes,space.begin()+p.info.workspace_bytes+64,std::byte{0x62});
    }
    std::uint64_t* output(){return alias?a:q;}
    void execute(sbn3_team_scope* scope){
        const auto n=prepared->info.n;auto bytes=work.on().scratch;
        auto frame=Frame::external(bytes.data(),prepared->info.workspace_bytes);
        const sbn3_const_limbs A=prepared->info.kind==SBN3_NEWTON_DIVIDE?sbn3_const_limbs{a,n+1}:sbn3_const_limbs{};
        normalized_newton_execute(*prepared,A,{d,n},{output(),n+1},frame,scope);
        assert(frame.used()==0);
    }
    void check(){
        const auto n=prepared->info.n;assert(a[n+1]==guard&&d[n]==guard&&q[n+1]==guard);
        assert(std::equal(original_d.begin(),original_d.end(),d));
        if(!alias)assert(std::equal(original_a.begin(),original_a.end(),a));
        auto bytes=work.on().scratch;for(std::size_t j=0;j<64;++j)assert(bytes[prepared->info.workspace_bytes+j]==std::byte{0x62});
        Ref Q,lo,hi,lower,upper,target;ref_import(Q.value,n+1,-1,8,0,0,output());
        ref_sub_ui(lo.value,Q.value,3);ref_add_ui(hi.value,Q.value,3);
        product(lower.value,lo.value,original_d);product(upper.value,hi.value,original_d);
        if(prepared->info.kind==SBN3_NEWTON_INVERSE){ref_set_ui(target.value,1);ref_mul_2exp(target.value,target.value,128*n);assert(output()[n]==1);}
        else{ref_import(target.value,n+1,-1,8,0,0,original_a.data());ref_mul_2exp(target.value,target.value,64*n);}
        assert(ref_cmp(lower.value,target.value)<0&&ref_cmp(upper.value,target.value)>0);
        ++certificates;if(!prepared->info.local)++native_calls;
    }
};
void execute(void* raw,sbn3_team_scope* scope){static_cast<Job*>(raw)->execute(scope);}
struct Pair {Job* left;Job* right;};
void execute_pair(void* raw,sbn3_team_scope* scope){
    auto& pair=*static_cast<Pair*>(raw);
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,4,execute,pair.left,execute,pair.right);
}
class TestTeam {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    TestTeam(){
        sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config tc{};tc.workers=8;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);
    }
    ~TestTeam(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_stats stats{};
        sbn3_arena_get_stats(arena,&stats);assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena);}
};
}
int main(){
    NormalizedNewtonInfo bad;assert(normalized_newton_query(SBN3_NEWTON_RSQRT,17,1,bad)==SBN3_UNSUPPORTED);
    for(auto kind:{SBN3_NEWTON_INVERSE,SBN3_NEWTON_DIVIDE})for(std::size_t n:{1u,3u,17u,129u,257u,1025u}){
        Program program(kind,n,1);
        for(unsigned pattern=0;pattern<3;++pattern){Job job(program,pattern,kind==SBN3_NEWTON_DIVIDE&&pattern==2);
            allocation_watch_start();serial_team_run(execute,&job);assert(!allocation_watch_stop());job.check();program.check();}
    }
    TestTeam team;
    for(auto kind:{SBN3_NEWTON_INVERSE,SBN3_NEWTON_DIVIDE})for(std::size_t n:{8193u,16385u}){
        Program program(kind,n,4);assert(!program.info.local);
        Job left(program,1),right(program,2,kind==SBN3_NEWTON_DIVIDE);
        Pair pair{&left,&right};
        allocation_watch_start();sbn3_team_run(team.value,execute_pair,&pair);assert(!allocation_watch_stop());
        left.check();right.check();program.check();
        // The same immutable native program opens and closes a fresh binding
        // in another epoch, after both nonzero-worker child calls have joined.
        Job again(program,1,kind==SBN3_NEWTON_DIVIDE);
        allocation_watch_start();sbn3_team_run(team.value,execute,&again);assert(!allocation_watch_stop());again.check();program.check();
    }
    assert(native_calls);
    std::printf("normalized Newton: %zu independent exact certificates (%zu native), frozen replay, scoped reuse and allocation gates PASS\n",certificates,native_calls);
}
