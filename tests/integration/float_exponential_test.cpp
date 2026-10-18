// Original libsbn tests; LGPL-3.0-or-later.
#include "float/exponential.hpp"
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
using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x58ba70d129de6c34);
std::size_t checked=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
struct Input {std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}};
void power2(ref_number* out,std::size_t n){ref_set_ui(out,1);ref_mul_2exp(out,out,n);}
void shift(ref_number* out,const ref_number* in,std::int64_t n){assert(n>=0&&n<1000000);ref_mul_2exp(out,in,std::size_t(n));}
void multiply(ref_number* out,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);std::size_t n=0;
    ref_export(words.data(),&n,-1,8,0,0,b);Ref sum,part;ref_set_ui(sum.v,0);
    for(std::size_t j=0;j<n;++j)if(words[j]){ref_mul_ui(part.v,a,words[j]);if(j)ref_mul_2exp(part.v,part.v,64*j);ref_add(sum.v,sum.v,part.v);}
    ref_set(out,sum.v);
}
// Direct independent Taylor summation; no production product, root, log, or
// exp is used by the oracle. For |x|<=8 each term's fixed-point error is
// <exp(8)<2^16 units. Summing errors and the final geometric tail gives the
// displayed radius, including alternating terms for negative inputs.
void reference(Ball& out,const Input& input,f::ExponentialKind kind,std::size_t bits){
    Ref a,term,prod,den;ref_import(a.v,input.words.size(),-1,8,0,0,input.words.data());
    assert(input.exponent+std::int64_t(ref_sizeinbase(a.v,2))<=4);power2(term.v,bits);
    if(kind==f::ExponentialKind::exp)ref_set(out.center.v,term.v);else ref_set_ui(out.center.v,0);
    std::size_t j=1;
    for(;;++j){
        multiply(prod.v,term.v,a.v);if(input.exponent>=0)ref_mul_2exp(prod.v,prod.v,std::size_t(input.exponent));
        ref_set_ui(den.v,j);ref_fdiv_q(term.v,prod.v,den.v); // one-word divisor: always independent
        if(input.exponent<0)ref_fdiv_q_2exp(term.v,term.v,std::size_t(-input.exponent));
        if(!ref_sgn(term.v))break;
        if(input.negative&&(j&1))ref_sub(out.center.v,out.center.v,term.v);else ref_add(out.center.v,out.center.v,term.v);
        assert(j<2*bits+64);
    }
    out.radius=(j+2)*(UINT64_C(1)<<18);out.scale=-std::int64_t(bits);
}
void certify(sbn::float_view out,sbn::float_report report,const Ball& truth,std::size_t p,rounding mode){
    assert(out.finite()&&!out.zero()&&report.error.kind!=sbn::error_kind::unbounded);
    Ref center,radius,lo,hi,exact,er,elo,ehi;
    ref_import(center.v,out.magnitude().size(),-1,8,0,0,out.magnitude().data());const auto width=ref_sizeinbase(center.v,2);
    assert(width<=p);if(out.negative())ref_neg(center.v,center.v);
    const auto ulp=out.exponent2()+std::int64_t(width)-std::int64_t(p);
    assert(report.error.exact()||report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
    auto scale=std::min(out.exponent2(),truth.scale);if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift(center.v,center.v,out.exponent2()-scale);ref_set_ui(radius.v,report.error.exact()?0:1);
    if(!report.error.exact())shift(radius.v,radius.v,report.error.exponent2-scale);
    ref_sub(lo.v,center.v,radius.v);ref_add(hi.v,center.v,radius.v);
    shift(exact.v,truth.center.v,truth.scale-scale);ref_set_ui(er.v,truth.radius);shift(er.v,er.v,truth.scale-scale);
    ref_sub(elo.v,exact.v,er.v);ref_add(ehi.v,exact.v,er.v);
    if(ref_cmp(lo.v,elo.v)>0||ref_cmp(hi.v,ehi.v)<0){std::fprintf(stderr,"exp enclosure failed p=%zu mode=%u exp=%lld error=%lld\n",
        p,unsigned(mode),(long long)out.exponent2(),(long long)report.error.exponent2);assert(false);}++checked;
}
struct Team {sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned workers){sbn3_error error{};sbn3_arena_config a{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&a,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config t{};t.workers=workers;t.stack_offset=65536;
        for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}};
struct ScopeCall {const f::FloatExponentialPrepared* prepared;sbn::float_ref out;sbn::float_view input;f::ExponentialKind kind;rounding mode;sbn::float_report report{};};
void scoped(void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<ScopeCall*>(opaque);
    c.report=f::exponential_program_execute(*c.prepared,c.out,c.input,c.kind,{{},nullptr,scope},c.mode);}
void run(const f::FloatExponentialPrepared& prepared,std::size_t p,const Input& input,f::ExponentialKind kind,const Ball& truth,
    rounding mode,sbn3_team* team,bool scope,bool alias){
    const auto limbs=(p+63)/64,capacity=alias?std::max(limbs,input.words.size()):limbs;
    std::vector<std::uint64_t> words(capacity+2,guard);if(alias)std::copy(input.words.begin(),input.words.end(),words.begin());
    sbn::float_buffer out({words.data(),capacity},sbn::bits{p});const auto original=input.words;
    const auto source=alias?sbn::float_view::from_magnitude({words.data(),input.words.size()},input.exponent,input.negative):input.view();
    allocation_watch_start();sbn::float_report report;
    if(scope){ScopeCall c{&prepared,out.ref(),source,kind,mode,{}};sbn3_team_run(team,scoped,&c);report=c.report;}
    else report=f::exponential_program_execute(prepared,out.ref(),source,kind,{{},team,nullptr},mode);
    assert(!allocation_watch_stop());assert(input.words==original&&words[capacity]==guard&&words[capacity+1]==guard);
    if(alias)for(std::size_t j=limbs;j<input.words.size();++j)assert(words[j]==input.words[j]);
    certify(out.view(),report,truth,p,mode);
}
void extremes(const f::FloatExponentialPrepared& prepared,std::size_t p,sbn3_team* team){
    std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});const std::uint64_t one=1,three=3;
    for(bool negative:{false,true})for(auto kind:{f::ExponentialKind::exp,f::ExponentialKind::expm1}){
        const std::int64_t exponent=-sbn::float_exponent_limit-1;
        const auto tiny=sbn::float_view::from_magnitude({&one,1},exponent,negative);
        allocation_watch_start();auto report=f::exponential_program_execute(prepared,out.ref(),tiny,kind,{{},team,nullptr});assert(!allocation_watch_stop());
        assert(out.view().magnitude().size()==1&&out.view().magnitude().data()[0]==1&&report.error.kind==sbn::error_kind::finite);
        if(kind==f::ExponentialKind::exp){assert(!out.view().negative()&&out.view().exponent2()==0&&report.error.exponent2>=exponent+1);}
        else assert(out.view().negative()==negative&&out.view().exponent2()==exponent&&report.error.exponent2>=2*exponent);
        ++checked;
        const auto huge=sbn::float_view::from_magnitude({&one,1},60,negative);
        const auto need=f::exponential_requirements(out.ref(),huge,kind,team?sbn3_team_workers(team):1);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();report=f::exponential_into(out.ref(),huge,kind,{{},team,nullptr});assert(!allocation_watch_stop());
        if(!negative)assert(!out.view().finite()&&!out.view().negative()&&sbn::has_flag(report.flags,sbn::float_flag::overflow));
        else if(kind==f::ExponentialKind::exp)assert(out.view().zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow)&&report.error.kind==sbn::error_kind::finite);
        else assert(sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},0,true))==std::partial_ordering::equivalent);
        ++checked;
        // This range failure is below the coarse early cutoff. It exercises
        // wide k, range policy after Newton, and virtual-gap subtraction.
        const auto boundary=sbn::float_view::from_magnitude({&three,1},58,negative);
        allocation_watch_start();report=f::exponential_program_execute(prepared,out.ref(),boundary,kind,{{},team,nullptr});assert(!allocation_watch_stop());
        if(!negative)assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::overflow));
        else if(kind==f::ExponentialKind::exp)assert(out.view().zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
        else assert(sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},0,true))==std::partial_ordering::equivalent);
        ++checked;
    }
}
void cases(std::size_t p,sbn3_team* team,bool many){
    const auto workers=team?sbn3_team_workers(team):1;allocation_watch_start();const auto need=f::exponential_program_requirements(sbn::bits{p},workers);
    assert(!allocation_watch_stop()&&need.supported());sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    allocation_watch_start();const auto prepared=f::exponential_program_prepare(sbn::bits{p},workers,work.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
    Ball truth;const Input one{{1},0,false};reference(truth,one,f::ExponentialKind::exp,p+512);
    for(auto mode:modes)run(prepared,p,one,f::ExponentialKind::exp,truth,mode,team,team!=nullptr,true);
    const Input negative{{1},0,true};reference(truth,negative,f::ExponentialKind::exp,p+512);
    run(prepared,p,negative,f::ExponentialKind::exp,truth,rounding::nearest_even,team,false,false);
    if(!many)return;
    for(auto kind:{f::ExponentialKind::exp,f::ExponentialKind::expm1})for(bool sign:{false,true}){
        const Input x{{13},-2,sign};reference(truth,x,kind,p+512);
        for(auto mode:modes)run(prepared,p,x,kind,truth,mode,team,false,false);
    }
    for(auto gap:{1u,7u,unsigned(p/2+8),unsigned(p+80)})for(bool sign:{false,true}){
        const Input x{{1},-std::int64_t(gap),sign};reference(truth,x,f::ExponentialKind::expm1,p+2*gap+512);
        for(auto mode:modes)run(prepared,p,x,f::ExponentialKind::expm1,truth,mode,team,false,true);
    }
    Input tall;tall.words.assign((p+300)/64+1,0);const auto low=64*(tall.words.size()-1);
    tall.words.back()=1;tall.words.front()=1;tall.exponent=-std::int64_t(low)-1;
    reference(truth,tall,f::ExponentialKind::exp,p+512);
    run(prepared,p,tall,f::ExponentialKind::exp,truth,rounding::nearest_even,team,false,true);
    extremes(prepared,p,team);
    const Input fresh{{3},-2,false};reference(truth,fresh,f::ExponentialKind::expm1,p+512);
    std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});sbn::workspace scratch;
    const auto request=f::exponential_requirements(out.ref(),fresh.view(),f::ExponentialKind::expm1,workers);scratch.reserve(request.scratch_bytes,request.scratch_alignment);
    allocation_watch_start();const auto report=f::exponential_into(out.ref(),fresh.view(),f::ExponentialKind::expm1,scratch.on(team));assert(!allocation_watch_stop());
    certify(out.view(),report,truth,p,rounding::nearest_even);
}
void specials(){
    std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    for(auto kind:{f::ExponentialKind::exp,f::ExponentialKind::expm1})for(bool negative:{false,true}){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        auto report=f::exponential_into(out.ref(),zero,kind);assert(report.error.exact());
        if(kind==f::ExponentialKind::exp)assert(!out.view().zero()&&!out.view().negative());else assert(out.view().zero()&&out.view().negative()==negative);
        report=f::exponential_into(out.ref(),sbn::float_view::infinity(negative),kind);
        if(!negative)assert(!out.view().finite()&&!out.view().negative());
        else if(kind==f::ExponentialKind::exp)assert(out.view().zero()&&report.error.exact());
        else assert(out.view().finite()&&out.view().negative()&&report.error.exact());
    }
    const auto report=f::exponential_into(out.ref(),sbn::float_view::nan(),f::ExponentialKind::exp);(void)report;assert(out.view().is_nan());
}
}
int main(){
    for(auto p:{1u,3u,7u,17u,63u,65u,127u,193u,257u})cases(p,nullptr,true);
    for(auto p:{509u,1025u,4097u})cases(p,nullptr,false);
    {Team team(1);cases(129,team.team,true);}{Team team(4);cases(257,team.team,true);}
    specials();std::printf("float exponential independent enclosures: %zu\n",checked);
}
