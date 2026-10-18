// Original libsbn tests; LGPL-3.0-or-later.
#include "algorithms/normalized_local.hpp"
#include "algorithms/local_inverse.hpp"
#include "runtime/scratch.hpp"
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
constexpr std::uint64_t guard=UINT64_C(0x9bf73200e4531ac6);
std::uint64_t state=UINT64_C(0xf724b812acf50031);
std::size_t certificates=0;
std::uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
struct Ref {
    ref_int value;
    Ref(){ref_init(value);}
    ~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
// General ref_mul can use a production witness. These tests use only exact
// independent scalar products, additions and shifts, also for signed bounds.
void product(ref_number* out,const ref_number* a,const std::vector<std::uint64_t>& b){
    Ref sum,term;ref_set_ui(sum.value,0);
    for(std::size_t i=0;i<b.size();++i)if(b[i]){
        ref_mul_ui(term.value,a,b[i]);if(i)ref_mul_2exp(term.value,term.value,64*i);
        ref_add(sum.value,sum.value,term.value);
    }
    ref_set(out,sum.value);
}
void certificate(NormalizedLocalKind kind,const std::vector<std::uint64_t>& a,
                 const std::vector<std::uint64_t>& d,const std::uint64_t* q){
    const auto n=d.size();Ref Q,low,high,target,lower,upper;
    ref_import(Q.value,n+1,-1,8,0,0,q);
    ref_sub_ui(low.value,Q.value,3);ref_add_ui(high.value,Q.value,3);
    bool power=d.back()==(UINT64_C(1)<<63);
    for(std::size_t i=0;i+1<n&&power;++i)power=d[i]==0;
    if(power){
        ref_mul_2exp(lower.value,low.value,64*n-1);
        ref_mul_2exp(upper.value,high.value,64*n-1);
    }else{product(lower.value,low.value,d);product(upper.value,high.value,d);}
    if(kind==NormalizedLocalKind::inverse){ref_set_ui(target.value,1);ref_mul_2exp(target.value,target.value,128*n);assert(q[n]==1);}
    else{ref_import(target.value,n+1,-1,8,0,0,a.data());ref_mul_2exp(target.value,target.value,64*n);}
    if(!(ref_cmp(lower.value,target.value)<0&&ref_cmp(upper.value,target.value)>0)){
        std::fprintf(stderr,"normalized local <3 ulp certificate failed: n=%zu kind=%u\n",n,unsigned(kind));assert(false);
    }
    ++certificates;
}
struct Program {
    NormalizedLocalInfo info{};
    sbn::workspace metadata;
    NormalizedLocalPrepared prepared{};
    Program(NormalizedLocalKind kind,std::size_t n){
        allocation_watch_start();const auto rc=normalized_local_query(kind,n,info);assert(!allocation_watch_stop());
        assert(rc==SBN3_SUPPORTED&&info.n==n&&info.kind==kind);
        metadata.reserve(info.metadata_bytes+64,info.alignment);
        auto bytes=metadata.on().scratch;
        std::fill(bytes.begin()+info.metadata_bytes,bytes.begin()+info.metadata_bytes+64,std::byte{0x4b});
        auto frame=Frame::external(bytes.data(),info.metadata_bytes);
        allocation_watch_start();prepared=normalized_local_prepare(kind,n,frame);assert(!allocation_watch_stop());
        assert(prepared.info.workspace_bytes==info.workspace_bytes&&prepared.info.record_count==info.record_count);
    }
    void guard_check(){auto bytes=metadata.on().scratch;for(std::size_t j=0;j<64;++j)assert(bytes[info.metadata_bytes+j]==std::byte{0x4b});}
};
struct Job {
    const NormalizedLocalPrepared* prepared;
    std::vector<std::uint64_t> a,d,q;
    sbn::workspace scratch;
    bool alias;
    Job(const Program& p,bool power,bool alias_output=false):prepared(&p.prepared),a(p.info.n+1),d(p.info.n),q(p.info.n+3,guard),alias(alias_output){
        for(auto& w:a)w=random_word();a.back()&=1;
        if(power){std::fill(d.begin(),d.end(),0);d.back()=UINT64_C(1)<<63;}
        else{for(auto& w:d)w=random_word();d.back()|=UINT64_C(1)<<63;}
        if(alias)std::copy(a.begin(),a.end(),q.begin());
        scratch.reserve(p.info.workspace_bytes+64,p.info.alignment);
        auto bytes=scratch.on().scratch;
        std::fill(bytes.begin()+p.info.workspace_bytes,bytes.begin()+p.info.workspace_bytes+64,std::byte{0x57});
    }
    void execute(){
        auto bytes=scratch.on().scratch;auto frame=Frame::external(bytes.data(),prepared->info.workspace_bytes);
        const bool division=prepared->info.kind==NormalizedLocalKind::divide;
        const sbn3_const_limbs numerator=division?sbn3_const_limbs{alias?q.data():a.data(),a.size()}:sbn3_const_limbs{};
        normalized_local_execute(*prepared,numerator,{d.data(),d.size()},{q.data(),d.size()+1},frame);
        assert(frame.used()==0);
    }
    void check(){
        assert(q[d.size()+1]==guard&&q[d.size()+2]==guard);
        auto bytes=scratch.on().scratch;for(std::size_t j=0;j<64;++j)assert(bytes[prepared->info.workspace_bytes+j]==std::byte{0x57});
        certificate(prepared->info.kind,a,d,q.data());
    }
};
void ordinary(){
    const std::size_t sizes[]{1,2,3,4,5,7,15,16,17,63,65,127,129,255,257,511,513,769,1025,2049};
    for(auto n:sizes)for(auto kind:{NormalizedLocalKind::inverse,NormalizedLocalKind::divide}){
        Program program(kind,n);
        for(unsigned pattern=0;pattern<3;++pattern){
            Job job(program,pattern==1,kind==NormalizedLocalKind::divide&&pattern==2);
            if(kind==NormalizedLocalKind::divide&&pattern==0)std::fill(job.a.begin(),job.a.end(),0);
            if(kind==NormalizedLocalKind::divide&&pattern==1){std::fill(job.a.begin(),job.a.end(),UINT64_MAX);job.a.back()=1;}
            if(kind==NormalizedLocalKind::inverse&&pattern==2)std::fill(job.d.begin(),job.d.end(),UINT64_MAX);
            const auto before_d=job.d,before_a=job.a;
            allocation_watch_start();job.execute();assert(!allocation_watch_stop());
            assert(job.d==before_d&&job.a==before_a);job.check();program.guard_check();
        }
    }
    // Exercise the compact extension without a quadratic large reference
    // multiplication: a power-of-two D makes both endpoint products shifts.
    for(auto kind:{NormalizedLocalKind::inverse,NormalizedLocalKind::divide}){
        const auto n=local_inverse_max_limbs+1;NormalizedLocalInfo info;
        allocation_watch_start();const auto rc=normalized_local_query(kind,n,info);assert(!allocation_watch_stop());
        if(rc!=SBN3_SUPPORTED)continue;
        Program program(kind,n);Job job(program,true,kind==NormalizedLocalKind::divide);
        allocation_watch_start();job.execute();assert(!allocation_watch_stop());job.check();program.guard_check();
    }
}
struct Parallel {Job* left;Job* right;};
void child(void* raw,sbn3_team_scope*){static_cast<Job*>(raw)->execute();}
void parent(void* raw,sbn3_team_scope* scope){
    auto& jobs=*static_cast<Parallel*>(raw);
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,child,jobs.left,child,jobs.right);
}
void active_scopes(){
    sbn3_arena* arena=nullptr;sbn3_team* team=nullptr;sbn3_lease control{};sbn3_error error{};
    sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
    assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);
    assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
    sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
    sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
    assert(sbn3_team_create(arena,&control,&tc,&team,&error)==SBN3_OK);
    for(auto kind:{NormalizedLocalKind::inverse,NormalizedLocalKind::divide}){
        Program program(kind,1025);Job left(program,false),right(program,false,kind==NormalizedLocalKind::divide);
        Parallel jobs{&left,&right};
        allocation_watch_start();sbn3_team_run(team,parent,&jobs);assert(!allocation_watch_stop());
        left.check();right.check();program.guard_check();
    }
    sbn3_team_destroy(team);sbn3_arena_release(arena,&control);
    sbn3_arena_stats stats{};sbn3_arena_get_stats(arena,&stats);assert(!stats.lease_references&&!stats.active_computations);
    sbn3_arena_destroy(arena);
}
}
int main(){
    NormalizedLocalInfo invalid;assert(normalized_local_query(NormalizedLocalKind::inverse,0,invalid)==SBN3_UNSUPPORTED);
    ordinary();active_scopes();
    std::printf("normalized local Newton: %zu independent exact certificates, complete replay, reuse, alias, active subscopes and allocation gates PASS\n",certificates);
}
