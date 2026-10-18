// Original libsbn tests; LGPL-3.0-or-later.
#include "float/real_power.hpp"
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
namespace f=sbn::v3::floating;using sbn::rounding;using Wide=__int128_t;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0xbe7390d145acf268);
std::size_t checked=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
struct Input {std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}};
void power2(ref_number* out,std::size_t bits){ref_set_ui(out,1);ref_mul_2exp(out,out,bits);}
void shift(ref_number* out,const ref_number* in,Wide n){assert(n>=0&&n<1000000);ref_mul_2exp(out,in,std::size_t(n));}
void multiply(ref_number* out,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);std::size_t count=0;
    ref_export(words.data(),&count,-1,8,0,0,b);Ref sum,part;ref_set_ui(sum.v,0);
    for(std::size_t j=0;j<count;++j)if(words[j]){ref_mul_ui(part.v,a,words[j]);if(j)ref_mul_2exp(part.v,part.v,64*j);ref_add(sum.v,sum.v,part.v);}
    ref_set(out,sum.v);
}
int compare_scaled(const ref_number* a,Wide ae,const ref_number* b,Wide be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=ae+ref_sizeinbase(a,2),bt=be+ref_sizeinbase(b,2);if(at!=bt)return at<bt?-1:1;
    Ref x,y;const auto base=std::min(ae,be);shift(x.v,a,ae-base);shift(y.v,b,be-base);return ref_cmp(x.v,y.v);
}
void magnitude_interval(sbn::float_view out,sbn::float_report report,Ref& lo,Ref& hi,std::int64_t& scale,std::size_t p,rounding mode){
    assert(out.finite()&&!out.zero()&&report.error.kind!=sbn::error_kind::unbounded);Ref center,radius;
    ref_import(center.v,out.magnitude().size(),-1,8,0,0,out.magnitude().data());const auto width=ref_sizeinbase(center.v,2);assert(width<=p);
    assert(report.error.exact()||report.error.exponent2<=out.exponent2()+std::int64_t(width)-std::int64_t(p)+(mode==rounding::nearest_even?0:1));
    scale=out.exponent2();if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift(center.v,center.v,out.exponent2()-scale);ref_set_ui(radius.v,report.error.exact()?0:1);
    if(!report.error.exact())shift(radius.v,radius.v,report.error.exponent2-scale);
    ref_sub(lo.v,center.v,radius.v);if(ref_sgn(lo.v)<0)ref_set_ui(lo.v,0);ref_add(hi.v,center.v,radius.v);
}
void certify_rational(const Input& base,std::int64_t numerator,unsigned denominator_bits,sbn::float_view out,
    sbn::float_report report,std::size_t p,rounding mode){
    assert(!base.negative||!denominator_bits);assert(out.negative()==(base.negative&&(numerator&1)));
    Ref lo,hi,temp,base_power,a,one;std::int64_t scale=0;magnitude_interval(out,report,lo,hi,scale,p,mode);
    for(unsigned j=0;j<denominator_bits;++j){multiply(temp.v,lo.v,lo.v);ref_set(lo.v,temp.v);multiply(temp.v,hi.v,hi.v);ref_set(hi.v,temp.v);}
    ref_import(a.v,base.words.size(),-1,8,0,0,base.words.data());ref_set_ui(base_power.v,1);const auto n=std::uint64_t(numerator<0?-numerator:numerator);
    for(std::uint64_t j=0;j<n;++j){multiply(temp.v,base_power.v,a.v);ref_set(base_power.v,temp.v);}
    const auto ys=Wide(scale)*(Wide(1)<<denominator_bits),xs=Wide(base.exponent)*n;
    if(numerator>=0)assert(compare_scaled(lo.v,ys,base_power.v,xs)<=0&&compare_scaled(hi.v,ys,base_power.v,xs)>=0);
    else{ref_set_ui(one.v,1);multiply(temp.v,lo.v,base_power.v);assert(compare_scaled(temp.v,ys+xs,one.v,0)<=0);
        multiply(temp.v,hi.v,base_power.v);assert(compare_scaled(temp.v,ys+xs,one.v,0)>=0);}
    ++checked;
}
// Exact small-gap logarithm series followed by an independent fixed-point
// Taylor exp. For x=1+/-2^-gap and y=2^gap, the exponent is
// +/-1 - 2^-gap/2 +/- 2^(-2gap)/3 ... . No production log/exp is used.
void near_one_reference(Ball& out,std::size_t gap,bool below,std::size_t bits){
    Ref z,term,den,magnitude,prod;ref_set_ui(z.v,0);std::size_t log_terms=0;
    for(std::size_t j=1;;++j){const auto drop=gap*(j-1);if(drop>bits)break;power2(term.v,bits-drop);ref_set_ui(den.v,j);ref_fdiv_q(term.v,term.v,den.v);
        if(below||(j%2==0))ref_sub(z.v,z.v,term.v);else ref_add(z.v,z.v,term.v);++log_terms;}
    const auto log_error=log_terms+2;const bool negative=ref_sgn(z.v)<0;ref_abs(magnitude.v,z.v);power2(term.v,bits);ref_set(out.center.v,term.v);
    std::size_t j=1;
    for(;;++j){multiply(prod.v,term.v,magnitude.v);ref_fdiv_q_2exp(prod.v,prod.v,bits);ref_set_ui(den.v,j);ref_fdiv_q(term.v,prod.v,den.v);
        if(!ref_sgn(term.v))break;if(negative&&(j&1))ref_sub(out.center.v,out.center.v,term.v);else ref_add(out.center.v,out.center.v,term.v);
        assert(j<bits+64);}
    // |z|<2, so exp sensitivity<8. Floor propagation and the tail are
    // bounded by64 units per term, with ample independent margin.
    out.radius=64*(j+2)+8*log_error;out.scale=-std::int64_t(bits);
}
void certify_ball(sbn::float_view out,sbn::float_report report,const Ball& truth,std::size_t p,rounding mode){
    assert(!out.negative());Ref lo,hi,center,radius,elo,ehi;std::int64_t scale=0;magnitude_interval(out,report,lo,hi,scale,p,mode);
    const auto common=std::min(scale,truth.scale);shift(lo.v,lo.v,scale-common);shift(hi.v,hi.v,scale-common);
    shift(center.v,truth.center.v,truth.scale-common);ref_set_ui(radius.v,truth.radius);shift(radius.v,radius.v,truth.scale-common);
    ref_sub(elo.v,center.v,radius.v);ref_add(ehi.v,center.v,radius.v);
    if(ref_cmp(lo.v,elo.v)>0||ref_cmp(hi.v,ehi.v)<0){std::fprintf(stderr,"near-one real power enclosure failed p=%zu mode=%u\n",p,unsigned(mode));assert(false);}++checked;
}
struct Team {sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned workers){sbn3_error e{};sbn3_arena_config a{std::size_t(1)<<28,32u<<20};assert(sbn3_arena_create(&a,&arena,&e)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&e)==SBN3_OK);sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config t{};t.workers=workers;t.stack_offset=65536;for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&e)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}};
struct Scoped {const f::FloatRealPowerPrepared* prepared;sbn::float_ref out;sbn::float_view x,y;rounding mode;sbn::float_report report{};};
void scoped(void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<Scoped*>(opaque);c.report=f::real_power_program_execute(*c.prepared,c.out,c.x,c.y,{{},nullptr,scope},c.mode);}
void run(const f::FloatRealPowerPrepared* prepared,std::size_t p,const Input& x,const Input& y,std::int64_t numerator,unsigned denominator_bits,
    const Ball* truth,rounding mode,sbn3_team* team,bool scope,unsigned alias){
    const auto limbs=(p+63)/64,capacity=std::max(limbs,alias==1?x.words.size():alias==2?y.words.size():0);
    std::vector<std::uint64_t> words(capacity+2,guard);const auto& source=alias==1?x:y;
    if(alias)std::copy(source.words.begin(),source.words.end(),words.begin());sbn::float_buffer out({words.data(),capacity},sbn::bits{p});
    const auto xv=alias==1?sbn::float_view::from_magnitude({words.data(),x.words.size()},x.exponent,x.negative):x.view();
    const auto yv=alias==2?sbn::float_view::from_magnitude({words.data(),y.words.size()},y.exponent,y.negative):y.view();
    sbn::workspace work;sbn::execution ex{{},team,nullptr};
    if(!prepared){allocation_watch_start();const auto need=f::real_power_requirements(out.ref(),xv,yv,team?sbn3_team_workers(team):1);assert(!allocation_watch_stop()&&need.supported());
        work.reserve(need.scratch_bytes,need.scratch_alignment);ex=work.on(team);}
    allocation_watch_start();sbn::float_report report;
    if(scope){assert(prepared);Scoped c{prepared,out.ref(),xv,yv,mode,{}};sbn3_team_run(team,scoped,&c);report=c.report;}
    else report=prepared?f::real_power_program_execute(*prepared,out.ref(),xv,yv,ex,mode):f::real_power_into(out.ref(),xv,yv,ex,mode);
    assert(!allocation_watch_stop());assert(words[capacity]==guard&&words[capacity+1]==guard);
    if(alias)for(std::size_t j=limbs;j<source.words.size();++j)assert(words[j]==source.words[j]);
    if(truth)certify_ball(out.view(),report,*truth,p,mode);else certify_rational(x,numerator,denominator_bits,out.view(),report,p,mode);
}
Input near_one(std::size_t gap,bool below){Input x;x.exponent=-std::int64_t(gap);x.words.assign(gap/64+1,0);
    if(below){for(std::size_t j=0;j<gap/64;++j)x.words[j]=UINT64_MAX;if(gap%64)x.words.back()=(UINT64_C(1)<<(gap%64))-1;}
    else{x.words.back()=UINT64_C(1)<<(gap%64);x.words[0]|=1;}while(!x.words.empty()&&!x.words.back())x.words.pop_back();return x;}
void cases(std::size_t p,sbn3_team* team,bool many){
    const auto workers=team?sbn3_team_workers(team):1;allocation_watch_start();const auto need=f::real_power_program_requirements(sbn::bits{p},workers);
    assert(!allocation_watch_stop()&&need.supported());sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);allocation_watch_start();
    const auto prepared=f::real_power_program_prepare(sbn::bits{p},workers,work.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
    const Input base{{3},-1,false};
    for(auto mode:modes)run(&prepared,p,base,Input{{3},-1,false},3,1,nullptr,mode,team,team!=nullptr,1);
    if(many||p==1025)for(bool below:{false,true}){
        const std::size_t gap=1000;const auto x=near_one(gap,below);const Input y{{1},std::int64_t(gap),false};Ball truth;
        near_one_reference(truth,gap,below,p+512);
        run(&prepared,p,x,y,0,0,&truth,rounding::nearest_even,team,false,1);
        if(many)run(nullptr,p,x,y,0,0,&truth,rounding::nearest_even,team,false,2);
    }
    if(!many)return;
    for(auto numerator:{-7LL,-3LL,3LL,7LL}){const auto mag=std::uint64_t(numerator<0?-numerator:numerator);
        run(&prepared,p,Input{{5},-2,false},Input{{mag},-2,numerator<0},std::int64_t(numerator),2,nullptr,rounding::nearest_even,team,false,2);}
    for(auto mode:modes)for(auto numerator:{-3LL,3LL}){const auto mag=std::uint64_t(numerator<0?-numerator:numerator);
        const Input x{{3},-1,true},y{{mag},0,numerator<0};
        run(&prepared,p,x,y,std::int64_t(numerator),0,nullptr,mode,team,false,1);
        run(nullptr,p,x,y,std::int64_t(numerator),0,nullptr,mode,team,false,2);}
}
void specials(){
    std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});const std::uint64_t one=1,two=2,three=3;
    const auto pos=sbn::float_view::from_magnitude({&one,1},0),neg=pos.negated(),half=sbn::float_view::from_magnitude({&one,1},-1);
    auto call=[&](sbn::float_view x,sbn::float_view y){const auto need=f::real_power_requirements(out.ref(),x,y);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=f::real_power_into(out.ref(),x,y);assert(!allocation_watch_stop());return report;};
    auto report=call(sbn::float_view::nan(),{});assert(sbn::compare(out.view(),pos)==std::partial_ordering::equivalent&&report.error.exact());
    report=call(pos,sbn::float_view::nan());assert(sbn::compare(out.view(),pos)==std::partial_ordering::equivalent&&report.error.exact());
    report=call(neg,sbn::float_view::infinity());assert(sbn::compare(out.view(),pos)==std::partial_ordering::equivalent);
    report=call(neg,half);assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
    for(bool sign:{false,true})for(bool reciprocal:{false,true}){
        const auto zero=sbn::float_view::from_magnitude({},0,sign),exponent=sbn::float_view::from_magnitude({&three,1},0,reciprocal);
        report=call(zero,exponent);assert(out.view().negative()==sign);
        if(reciprocal)assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));else assert(out.view().zero());
    }
    for(bool reciprocal:{false,true}){
        report=call(sbn::float_view::from_magnitude({},0,true),reciprocal?half.negated():half);
        assert(!out.view().negative());if(reciprocal)assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
        else assert(out.view().zero());
    }
    const std::uint64_t min_magnitude=UINT64_C(1)<<63;
    report=call(sbn::float_view::from_magnitude({&two,1},0,true),sbn::float_view::from_magnitude({&min_magnitude,1},0,true));
    assert(out.view().zero()&&!out.view().negative()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
    report=call(sbn::float_view::infinity(true),half);assert(!out.view().finite()&&!out.view().negative()&&!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    Input odd;odd.words.assign(16,0);odd.words[15]=UINT64_C(1)<<40;odd.words[0]=1;
    report=call(neg,odd.view());assert(sbn::compare(out.view(),neg)==std::partial_ordering::equivalent&&report.error.exact());
    odd.exponent=-1;report=call(sbn::float_view::from_magnitude({&two,1},0,true),odd.view());
    assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
    const auto need=f::real_power_program_requirements(sbn::bits{65});sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    const auto prepared=f::real_power_program_prepare(sbn::bits{65},1,work.on().scratch.first(need.scratch_bytes));
    odd.exponent=0;
    for(bool reciprocal:{false,true}){odd.negative=reciprocal;allocation_watch_start();report=f::real_power_program_execute(prepared,out.ref(),sbn::float_view::from_magnitude({&two,1},0,true),odd.view());
        assert(!allocation_watch_stop());assert(out.view().negative());if(reciprocal)assert(out.view().zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
        else assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::overflow));}
    for(bool tiny:{false,true}){
        const auto y=sbn::float_view::from_magnitude({&one,1},tiny?-sbn::float_exponent_limit-1:sbn::float_exponent_limit-1);
        allocation_watch_start();report=f::real_power_program_execute(prepared,out.ref(),sbn::float_view::from_magnitude({&two,1},0),y);assert(!allocation_watch_stop());
        if(tiny)assert(sbn::compare(out.view(),pos)==std::partial_ordering::equivalent&&report.error.kind==sbn::error_kind::finite);
        else assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::overflow));++checked;
    }
}
}
int main(){
    for(auto p:{1u,3u,17u,63u,65u,127u,257u})cases(p,nullptr,true);
    for(auto p:{509u,1025u,4097u})cases(p,nullptr,false);
    {Team team(1);cases(129,team.team,true);}{Team team(4);cases(193,team.team,true);}
    specials();std::printf("real power independent enclosures: %zu\n",checked);
}
