// Original libsbn tests; LGPL-3.0-or-later.
#include "float/constant_bank.hpp"
#include "float/constants.hpp"
#include "float/logarithm.hpp"
#include "float/exponential.hpp"
#include "float/elliptic.hpp"
#include "float/agm.hpp"
#include "runtime/team.hpp"
#include "sbn3/arena.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
namespace f=sbn::v3::floating;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
std::size_t checked=0;
void power2(ref_number* out,std::size_t bits){ref_set_ui(out,1);ref_mul_2exp(out,out,bits);}
std::uint64_t atan_fixed(ref_number* out,unsigned divisor,std::size_t bits,bool alternating){
    Ref unit,power,den,term;power2(unit.v,bits);ref_set_ui(power.v,divisor);ref_set_ui(out,0);std::size_t j=0;
    for(;;++j){ref_mul_ui(den.v,power.v,2*j+1);
        assert(((ref_sizeinbase(unit.v,2)+63)/64)*((ref_sizeinbase(den.v,2)+63)/64)<=1048576);
        ref_fdiv_q(term.v,unit.v,den.v);if(!ref_sgn(term.v))break;
        if(alternating&&(j&1))ref_sub(out,out,term.v);else ref_add(out,out,term.v);
        ref_mul_ui(power.v,power.v,std::uint64_t(divisor)*divisor);assert(j<bits+8);}
    return j+2;
}
void reference(Ball& out,f::ConstantKind kind,std::size_t bits){
    if(kind==f::ConstantKind::log2){out.radius=2*atan_fixed(out.center.v,3,bits,false);ref_mul_ui(out.center.v,out.center.v,2);}
    else{Ref a,b;const auto ea=atan_fixed(a.v,5,bits,true),eb=atan_fixed(b.v,239,bits,true);
        ref_mul_ui(a.v,a.v,16);ref_mul_ui(b.v,b.v,4);ref_sub(out.center.v,a.v,b.v);out.radius=16*ea+4*eb;}
    out.scale=-std::int64_t(bits);
}
void exp_one_reference(Ball& out,std::size_t bits){
    Ref term,den;power2(term.v,bits);ref_set(out.center.v,term.v);std::size_t j=1;
    for(;;++j){ref_set_ui(den.v,j);ref_fdiv_q(term.v,term.v,den.v);if(!ref_sgn(term.v))break;ref_add(out.center.v,out.center.v,term.v);}
    out.radius=8*j+16;out.scale=-std::int64_t(bits);
}
void certify(sbn::float_view point,sbn::float_report report,const Ball& truth,std::size_t p){
    assert(point.finite()&&!point.zero()&&!point.negative()&&report.error.kind!=sbn::error_kind::unbounded);
    Ref center,radius,lo,hi,expected,er,elo,ehi;
    ref_import(center.v,point.magnitude().size(),-1,8,0,0,point.magnitude().data());const auto width=ref_sizeinbase(center.v,2);
    assert(width<=p);assert(report.error.exact()||report.error.exponent2<=point.exponent2()+std::int64_t(width)-std::int64_t(p));
    auto scale=std::min(point.exponent2(),truth.scale);if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    ref_mul_2exp(center.v,center.v,std::size_t(point.exponent2()-scale));ref_set_ui(radius.v,report.error.exact()?0:1);
    if(!report.error.exact())ref_mul_2exp(radius.v,radius.v,std::size_t(report.error.exponent2-scale));
    ref_sub(lo.v,center.v,radius.v);ref_add(hi.v,center.v,radius.v);
    ref_mul_2exp(expected.v,truth.center.v,std::size_t(truth.scale-scale));ref_set_ui(er.v,truth.radius);ref_mul_2exp(er.v,er.v,std::size_t(truth.scale-scale));
    ref_sub(elo.v,expected.v,er.v);ref_add(ehi.v,expected.v,er.v);
    assert(ref_cmp(lo.v,elo.v)<=0&&ref_cmp(hi.v,ehi.v)>=0);++checked;
}
struct Team {sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned workers){sbn3_error e{};sbn3_arena_config a{std::size_t(1)<<28,32u<<20};assert(sbn3_arena_create(&a,&arena,&e)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&e)==SBN3_OK);sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config t{};t.workers=workers;t.stack_offset=65536;for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&e)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}};
f::ConstantBankView construct(f::ConstantBankDemand demand,sbn::workspace& values,sbn3_team* team){
    const auto workers=team?sbn3_team_workers(team):1;allocation_watch_start();const auto q=f::constant_bank_requirements(demand,workers);
    assert(!allocation_watch_stop()&&q.supported()&&q.value_bytes&&q.workspace_bytes);
    if(demand.log2_bits)assert(q.pi_bits>=f::cold_constant_working_precision(sbn::bits{demand.log2_bits}));
    values.reserve(q.value_bytes,q.value_alignment);f::ConstantBankView view;
    {
        sbn::workspace work;work.reserve(q.workspace_bytes,q.workspace_alignment);allocation_watch_start();
        const auto prepared=f::constant_bank_prepare(q,values.on().scratch.first(q.value_bytes),work.on().scratch.first(q.workspace_bytes));
        assert(!allocation_watch_stop());const auto pending=f::constant_bank_pending_binding(prepared);
        assert(pending.pending()&&!pending.query_only()&&f::constant_bank_supports(pending,demand));
        assert(!f::constant_bank_supports(f::ConstantBankBinding(prepared.pending),demand));
        const auto epoch=team?team->epoch:0;allocation_watch_start();view=f::constant_bank_execute(prepared,{{},team,nullptr});assert(!allocation_watch_stop());
        if(team)assert(team->epoch==epoch+1);
        assert(f::constant_bank_supports(view,demand));const auto published_epoch=team?team->epoch:0;
        allocation_watch_start();const auto again=f::constant_bank_execute(prepared,{{},team,nullptr});assert(!allocation_watch_stop());assert(again.state==view.state);
        if(team)assert(team->epoch==published_epoch);
        // A view must not retain a producer, Frame, or table pointer into
        // this construction span. Destroy it before any prefix/consumer.
        std::fill(work.on().scratch.begin(),work.on().scratch.end(),std::byte{0xa7});
    }
    assert(f::constant_bank_supports(view,demand));return view;
}
void prefixes(f::ConstantBankView view){
    const std::vector<std::byte> original(view.storage.begin(),view.storage.end());
    for(auto kind:{f::ConstantKind::pi,f::ConstantKind::log2}){
        const auto available=kind==f::ConstantKind::pi?view.pi_bits:view.log2_bits;if(!available)continue;
        const auto borrowed=f::constant_bank_borrow(view,kind);assert(borrowed.precision==available&&borrowed.point.finite());
        for(std::size_t p:{1u,2u,3u,63u,64u,65u,127u,257u,513u,1025u,2049u,4097u})if(p<=available){
            std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer output(words,sbn::bits{p});
            allocation_watch_start();const auto report=f::constant_bank_prefix(view,kind,output.ref());assert(!allocation_watch_stop());
            Ball truth;reference(truth,kind,p+256);certify(output.view(),report,truth,p);
        }
    }
    assert(std::equal(original.begin(),original.end(),view.storage.begin()));
}
void consumers(f::ConstantBankView view,sbn3_team* team){
    constexpr std::size_t p=65;const auto workers=team?sbn3_team_workers(team):1;
    const auto demand=f::constant_bank_merge(f::exponential_constant_demand(sbn::bits{p}),
        f::constant_bank_merge(f::logarithm_constant_demand(sbn::bits{p}),f::elliptic_constant_demand(sbn::bits{p})));
    assert(f::constant_bank_supports(view,demand));const auto facts=f::constant_bank_query_binding(demand);
    assert(facts.supplied()&&facts.query_only()&&!facts.values().state&&f::constant_bank_supports(facts,demand));
    const auto log=f::logarithm_program_requirements(sbn::bits{p},workers,view),
        exp=f::exponential_program_requirements(sbn::bits{p},workers,view),
        elliptic=f::elliptic_program_requirements(sbn::bits{p},workers,view);
    assert(log.supported()&&exp.supported()&&elliptic.supported());
    assert(log.scratch_bytes==f::logarithm_program_requirements(sbn::bits{p},workers,facts).scratch_bytes);
    assert(exp.scratch_bytes==f::exponential_program_requirements(sbn::bits{p},workers,facts).scratch_bytes);
    assert(elliptic.scratch_bytes==f::elliptic_program_requirements(sbn::bits{p},workers,facts).scratch_bytes);
    assert(log.scratch_bytes<f::logarithm_program_requirements(sbn::bits{p},workers).scratch_bytes);
    assert(exp.scratch_bytes<f::exponential_program_requirements(sbn::bits{p},workers).scratch_bytes);
    assert(elliptic.scratch_bytes<f::elliptic_program_requirements(sbn::bits{p},workers).scratch_bytes);
    const auto absent=f::constant_bank_query_binding({1,0});assert(!f::logarithm_program_requirements(sbn::bits{p},workers,absent).supported());
    const std::vector<std::byte> original(view.storage.begin(),view.storage.end());
    std::array<std::uint64_t,2> a{},b{};sbn::float_buffer output(a,sbn::bits{p}),second(b,sbn::bits{p});const std::uint64_t one=1,two=2;
    sbn::workspace scratch;scratch.reserve(log.scratch_bytes,log.scratch_alignment);allocation_watch_start();
    const auto lp=f::logarithm_program_prepare(sbn::bits{p},workers,scratch.on().scratch.first(log.scratch_bytes),view);
    assert(!allocation_watch_stop());Ball truth;reference(truth,f::ConstantKind::log2,p+256);
    for(unsigned repeat=0;repeat<2;++repeat){allocation_watch_start();const auto report=f::logarithm_program_execute(lp,output.ref(),
        sbn::float_view::from_magnitude({&two,1},0),f::LogarithmKind::log,{{},team,nullptr});assert(!allocation_watch_stop());certify(output.view(),report,truth,p);}
    scratch.reserve(exp.scratch_bytes,exp.scratch_alignment);allocation_watch_start();
    const auto ep=f::exponential_program_prepare(sbn::bits{p},workers,scratch.on().scratch.first(exp.scratch_bytes),view);assert(!allocation_watch_stop());
    exp_one_reference(truth,p+256);allocation_watch_start();const auto er=f::exponential_program_execute(ep,output.ref(),
        sbn::float_view::from_magnitude({&one,1},0),f::ExponentialKind::exp,{{},team,nullptr});assert(!allocation_watch_stop());certify(output.view(),er,truth,p);
    scratch.reserve(elliptic.scratch_bytes,elliptic.scratch_alignment);allocation_watch_start();
    const auto kp=f::elliptic_program_prepare(sbn::bits{p},workers,scratch.on().scratch.first(elliptic.scratch_bytes),view);assert(!allocation_watch_stop());
    reference(truth,f::ConstantKind::pi,p+256);--truth.scale;
    for(unsigned repeat=0;repeat<2;++repeat){allocation_watch_start();const auto kr=f::elliptic_program_execute(kp,output.ref(),second.ref(),{},f::EllipticParameter::parameter,{{},team,nullptr});
        assert(!allocation_watch_stop());certify(output.view(),kr.k,truth,p);certify(second.view(),kr.e,truth,p);}
    assert(std::equal(original.begin(),original.end(),view.storage.begin()));
}
void agm_epoch(sbn3_team* team){
    constexpr std::size_t p=1025;std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer output(words,sbn::bits{p});const std::uint64_t one=1,two=2;
    const auto a=sbn::float_view::from_magnitude({&one,1},0),b=sbn::float_view::from_magnitude({&two,1},0);
    const auto need=f::agm_program_requirements(sbn::bits{p},sbn3_team_workers(team));assert(need.supported());sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    const auto prepared=f::agm_program_prepare(sbn::bits{p},sbn3_team_workers(team),work.on().scratch.first(need.scratch_bytes));
    const auto before=team->epoch;allocation_watch_start();const auto report=f::agm_program_execute(prepared,output.ref(),a,b,{{},team,nullptr});
    assert(!allocation_watch_stop()&&team->epoch==before+1&&report.error.kind!=sbn::error_kind::unbounded);
    assert(sbn::compare(output.view(),a)==std::partial_ordering::greater&&sbn::compare(output.view(),b)==std::partial_ordering::less);
}
}
int main(){
    for(auto demand:{f::ConstantBankDemand{1,0},f::ConstantBankDemand{0,1},f::ConstantBankDemand{4097,4097}}){
        sbn::workspace values;const auto view=construct(demand,values,nullptr);prefixes(view);if(demand.log2_bits>=1025)consumers(view,nullptr);}
    for(unsigned workers:{1u,4u}){Team team(workers);sbn::workspace values;const auto view=construct({1537,1537},values,team.team);
        prefixes(view);consumers(view,team.team);agm_epoch(team.team);}
    std::printf("immutable constant bank/prefix certificates: %zu; lifecycle, shared consumers, and AGM epoch gates\n",checked);
}
