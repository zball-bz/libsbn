// Original libsbn tests; LGPL-3.0-or-later.
#include "float/short_division.hpp"
#include "sbn3/execution.hpp"
#include "runtime/scratch.hpp"
#include "runtime/team.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
namespace floating=sbn::v3::floating;
constexpr std::uint64_t guard=UINT64_C(0x97b431ec602f58da);
std::size_t exact_checks=0,local_checks=0,native_checks=0,virtual_checks=0;
struct Ref {ref_int value;Ref(){ref_init(value);}~Ref(){ref_clear(value);}Ref(const Ref&)=delete;};
std::uint64_t random_word(){static std::uint64_t state=UINT64_C(0x314159278de0a63b);
    state^=state<<13;state^=state>>7;state^=state<<17;return state;}
// Independently multiply Q by the raw divisor using only scalar products.
// A dense B^(k-1)*(top+1)-1 witness has an exact shift/subtract shortcut;
// sparse native-size witnesses cost one scalar pass per nonzero word.
void denominator_product(ref_number* out,const ref_number* q,const std::vector<std::uint64_t>& d){
    bool ones=true;for(std::size_t i=0;i+1<d.size();++i)ones&=d[i]==UINT64_MAX;
    Ref term;ref_set_ui(out,0);
    if(ones){
        if(d.back()==UINT64_MAX)ref_mul_2exp(term.value,q,64);else ref_mul_ui(term.value,q,d.back()+1);
        ref_mul_2exp(term.value,term.value,64*(d.size()-1));ref_sub(out,term.value,q);
    }else for(std::size_t i=0;i<d.size();++i)if(d[i]){
        ref_mul_ui(term.value,q,d[i]);ref_mul_2exp(term.value,term.value,64*i);ref_add(out,out,term.value);
    }
    ref_mul_2exp(out,out,std::countl_zero(d.back()));
}
void certificate(const std::vector<std::uint64_t>& A,std::int64_t shift,const std::vector<std::uint64_t>& D,
                 const std::vector<std::uint64_t>& Q,std::size_t n,sbn3_newton_kind kind){
    Ref numerator,denominator,q,product,upper;
    if(kind==SBN3_NEWTON_INVERSE){ref_set_ui(numerator.value,1);ref_mul_2exp(numerator.value,numerator.value,64*(n+D.size()));}
    else{
        ref_import(numerator.value,A.size(),-1,8,0,0,A.data());
        if(shift>=0)ref_mul_2exp(numerator.value,numerator.value,std::size_t(shift));
        else ref_fdiv_q_2exp(numerator.value,numerator.value,std::size_t(-shift));
        ref_mul_2exp(numerator.value,numerator.value,64*D.size());
    }
    ref_import(q.value,n+2,-1,8,0,0,Q.data());
    ref_import(denominator.value,D.size(),-1,8,0,0,D.data());
    ref_mul_2exp(denominator.value,denominator.value,std::countl_zero(D.back()));
    denominator_product(product.value,q.value,D);ref_add(upper.value,product.value,denominator.value);
    assert(ref_cmp(product.value,numerator.value)<=0&&ref_cmp(upper.value,numerator.value)>0);++exact_checks;
}
struct Invocation {
    const floating::ShortDivisionPrepared& program;sbn3_const_limbs A,D;sbn3_limbs Q;
    sbn::v3::Frame& work;std::int64_t shift=0;
};
void action(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Invocation*>(raw);
    floating::short_division_execute(c.program,c.A,c.D,c.Q,c.work,scope,c.shift);}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
void run(const floating::ShortDivisionPrepared& prepared,std::size_t n,std::size_t k,sbn3_newton_kind kind,
         unsigned variant,int source_shape,sbn3_team* team,bool child){
    std::vector<std::uint64_t> D(k),A,Q(n+4,guard);
    if(k>32768){D.front()=37;D[k/2]=5;D.back()=variant?1:UINT64_C(0x8000000000000001);}
    else if(k>=513){std::fill(D.begin(),D.end(),UINT64_MAX);D.back()=variant?3:UINT64_MAX;}
    else{for(auto& w:D)w=random_word();D.back()=variant==2?1:variant?3:(D.back()|UINT64_C(1)<<63);}
    std::int64_t shift=0;
    if(kind==SBN3_NEWTON_DIVIDE){
        A.resize(source_shape<0?n/2:source_shape>0?n+3:n);
        for(auto& w:A)w=random_word();
        A.back()=source_shape?3:(A.back()|UINT64_C(1)<<63);
        shift=std::int64_t(64*n)-std::int64_t(64*(A.size()-1)+std::bit_width(A.back()));
    }
    if(source_shape==2){A=D;shift=std::int64_t(64*n)-std::int64_t(64*(D.size()-1)+std::bit_width(D.back()));}
    const auto original_a=A,original_d=D;
    const auto& info=prepared.info;sbn::workspace space;
    space.reserve(info.workspace_bytes+64,info.alignment);auto bytes=space.on().scratch;
    std::fill(bytes.begin()+info.workspace_bytes,bytes.begin()+info.workspace_bytes+64,std::byte{0x76});
    auto work=sbn::v3::Frame::external(bytes.data(),info.workspace_bytes);
    const auto numerator=source_shape==2?sbn3_const_limbs{D.data(),D.size()}:sbn3_const_limbs{A.data(),A.size()};
    Invocation call{prepared,numerator,{D.data(),D.size()},{Q.data(),n+2},work,shift};
    allocation_watch_start();
    if(child){assert(team&&info.workers==2);
        sbn3_team_run(team,[](void* raw,sbn3_team_scope* scope){
            sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,[](void*,sbn3_team_scope*){},nullptr,
                [](void* p,sbn3_team_scope* sub){assert(sbn3_team_first_worker(sub)==2);action(p,sub);},raw);
        },&call);
    }else if(team)sbn3_team_run(team,action,&call);else sbn::v3::serial_team_run(action,&call);
    assert(!allocation_watch_stop());assert(work.used()==0&&work.peak()<=info.workspace_bytes);
    assert(A==original_a&&D==original_d&&Q[n+2]==guard&&Q[n+3]==guard);
    for(std::size_t i=0;i<64;++i)assert(bytes[info.workspace_bytes+i]==std::byte{0x76});
    assert(!Q[n+1]&&Q[n]<=2);certificate(A,shift,D,Q,n,kind);
    if(info.local)++local_checks;else ++native_checks;if(source_shape)++virtual_checks;
}
void recipe(std::size_t n,std::size_t k,sbn3_newton_kind kind,unsigned workers,sbn3_team* team,bool child){
    floating::ShortDivisionInfo info;allocation_watch_start();
    assert(floating::short_division_query(kind,n,k,workers,info)==SBN3_SUPPORTED);assert(!allocation_watch_stop());
    assert(info.local==(k<=32768)&&info.n==n&&info.k==k&&info.workers==workers);
    sbn::workspace storage;storage.reserve(info.metadata_bytes+64,info.alignment);auto bytes=storage.on().scratch;
    std::fill(bytes.begin()+info.metadata_bytes,bytes.begin()+info.metadata_bytes+64,std::byte{0x39});
    auto metadata=sbn::v3::Frame::external(bytes.data(),info.metadata_bytes);
    allocation_watch_start();const auto prepared=floating::short_division_prepare(kind,n,k,workers,metadata);
    assert(!allocation_watch_stop());
    for(unsigned variant=0;variant<(k<=129?3u:2u);++variant){
        run(prepared,n,k,kind,variant,0,team,child);
        if(kind==SBN3_NEWTON_DIVIDE&&k<=129){
            run(prepared,n,k,kind,variant,-1,team,child);run(prepared,n,k,kind,variant,1,team,child);
            run(prepared,n,k,kind,variant,2,team,child);
        }
        if(kind==SBN3_NEWTON_DIVIDE&&k>32768&&!variant)run(prepared,n,k,kind,variant,2,team,child);
    }
    for(std::size_t i=0;i<64;++i)assert(bytes[info.metadata_bytes+i]==std::byte{0x39});
}
}
int main(){
    floating::ShortDivisionInfo ignored;
    assert(floating::short_division_query(SBN3_NEWTON_DIVIDE,8,2,1,ignored)==SBN3_UNSUPPORTED);
    assert(floating::short_division_query(SBN3_NEWTON_DIVIDE,8,4,1,ignored)==SBN3_UNSUPPORTED);
    assert(floating::short_division_query(SBN3_NEWTON_DIVIDE,9,3,0,ignored)==SBN3_UNSUPPORTED);
    for(std::size_t k:{3u,7u,17u,129u,1025u,32768u})for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE})
        recipe(2*k+3,k,kind,1,nullptr,false);
    for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE})recipe(65541,32769,kind,1,nullptr,false);
    Team team;
    for(auto kind:{SBN3_NEWTON_DIVIDE,SBN3_NEWTON_INVERSE}){
        recipe(43,7,kind,2,team.value,true);
        recipe(65541,32769,kind,2,team.value,true);
    }
    std::printf("float short division: %zu independent exact quotient certificates (%zu local, %zu native, %zu virtual input shifts); prepared reuse/scopes/allocation PASS\n",
        exact_checks,local_checks,native_checks,virtual_checks);
}
