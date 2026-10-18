// Original libsbn tests; LGPL-3.0-or-later.
#include "float/derived_explog.hpp"
#include "sbn3/arena.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
namespace f=sbn::v3::floating;
using sbn::rounding;using Wide=__int128_t;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0xea43907bc8d125f6);
std::size_t checked=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
struct Input {std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}};
void shift(ref_number* out,const ref_number* in,Wide n){assert(n>=0&&n<1000000);ref_mul_2exp(out,in,std::size_t(n));}
void power2(ref_number* out,std::size_t bits){ref_set_ui(out,1);ref_mul_2exp(out,out,bits);}
void multiply(ref_number* out,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);std::size_t n=0;
    ref_export(words.data(),&n,-1,8,0,0,b);Ref sum,part;ref_set_ui(sum.v,0);
    for(std::size_t j=0;j<n;++j)if(words[j]){ref_mul_ui(part.v,a,words[j]);if(j)ref_mul_2exp(part.v,part.v,64*j);ref_add(sum.v,sum.v,part.v);}
    ref_set(out,sum.v);
}
void divide(ref_number* out,const ref_number* a,const ref_number* b){
    assert(((ref_sizeinbase(a,2)+63)/64)*((ref_sizeinbase(b,2)+63)/64)<=1048576);ref_fdiv_q(out,a,b);
}
int compare_scaled(const ref_number* a,Wide ae,const ref_number* b,Wide be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=ae+ref_sizeinbase(a,2),bt=be+ref_sizeinbase(b,2);if(at!=bt)return at<bt?-1:1;
    Ref x,y;const auto base=std::min(ae,be);shift(x.v,a,ae-base);shift(y.v,b,be-base);return ref_cmp(x.v,y.v);
}
void atanh(Ball& out,const ref_number* a,const ref_number* b,std::size_t bits){
    if(!ref_sgn(a)){ref_set_ui(out.center.v,0);out.radius=0;out.scale=-std::int64_t(bits);return;}
    Ref z,w,term,temp,odd,part;ref_mul_2exp(temp.v,a,bits);divide(z.v,temp.v,b);
    multiply(temp.v,z.v,z.v);ref_fdiv_q_2exp(w.v,temp.v,bits);ref_set(term.v,z.v);ref_set_ui(out.center.v,0);
    std::size_t j=0;while(ref_sgn(term.v)){
        ref_set_ui(odd.v,2*j+1);divide(part.v,term.v,odd.v);ref_add(out.center.v,out.center.v,part.v);
        multiply(temp.v,term.v,w.v);ref_fdiv_q_2exp(term.v,temp.v,bits);assert(++j<bits+8);}
    ref_mul_ui(out.center.v,out.center.v,2);out.radius=16*j+16;out.scale=-std::int64_t(bits);
}
void natural_log(Ball& out,const Input& input,std::size_t bits){
    Ref x,unit,a,b;ref_import(x.v,input.words.size(),-1,8,0,0,input.words.data());assert(!input.negative&&ref_sgn(x.v)>0);
    const auto width=ref_sizeinbase(x.v,2);const auto k=input.exponent+std::int64_t(width)-1;
    if(k==-1||k==0){power2(unit.v,std::size_t(-input.exponent));ref_sub(a.v,x.v,unit.v);const bool negative=ref_sgn(a.v)<0;
        ref_abs(a.v,a.v);ref_add(b.v,x.v,unit.v);atanh(out,a.v,b.v,bits);if(negative)ref_neg(out.center.v,out.center.v);return;}
    power2(unit.v,width-1);ref_sub(a.v,x.v,unit.v);ref_add(b.v,x.v,unit.v);atanh(out,a.v,b.v,bits);
    Ball log2;ref_set_ui(a.v,1);ref_set_ui(b.v,3);atanh(log2,a.v,b.v,bits);
    const auto magnitude=k<0?std::uint64_t(0)-std::uint64_t(k):std::uint64_t(k);assert(magnitude<1024);
    ref_mul_ui(log2.center.v,log2.center.v,magnitude);
    if(k<0)ref_sub(out.center.v,out.center.v,log2.center.v);else ref_add(out.center.v,out.center.v,log2.center.v);
    out.radius+=magnitude*log2.radius;
}
void logarithm_reference(Ball& out,const Input& input,f::DerivedExpLogKind kind,std::size_t bits){
    Ball numerator,denominator;natural_log(numerator,input,bits);
    natural_log(denominator,kind==f::DerivedExpLogKind::log2?Input{{2},0,false}:Input{{10},0,false},bits);
    Ref a,scaled;const bool negative=ref_sgn(numerator.center.v)<0;ref_abs(a.v,numerator.center.v);
    ref_mul_2exp(scaled.v,a.v,bits);divide(out.center.v,scaled.v,denominator.center.v);if(negative)ref_neg(out.center.v,out.center.v);
    // Tested |log(input)|<64, denominators>=1/2. This intentionally loose
    // independent radius includes both input intervals and integer flooring.
    out.radius=64*numerator.radius+1024*denominator.radius+32;out.scale=-std::int64_t(bits);
}
void interval(sbn::float_view out,sbn::float_report report,Ref& lo,Ref& hi,std::int64_t& scale,std::size_t p,rounding mode){
    assert(out.finite()&&report.error.kind!=sbn::error_kind::unbounded);Ref center,radius;
    ref_import(center.v,out.magnitude().size(),-1,8,0,0,out.magnitude().data());const auto width=ref_sizeinbase(center.v,2);assert(out.zero()||width<=p);
    if(out.negative())ref_neg(center.v,center.v);scale=out.exponent2();if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift(center.v,center.v,out.exponent2()-scale);ref_set_ui(radius.v,report.error.exact()?0:1);
    if(!report.error.exact())shift(radius.v,radius.v,report.error.exponent2-scale);
    ref_sub(lo.v,center.v,radius.v);ref_add(hi.v,center.v,radius.v);
    if(!out.zero())assert(report.error.exact()||report.error.exponent2<=out.exponent2()+std::int64_t(width)-std::int64_t(p)+(mode==rounding::nearest_even?0:1));
}
void certify_log(sbn::float_view out,sbn::float_report report,const Ball& truth,std::size_t p,rounding mode){
    Ref lo,hi,elo,ehi,center,radius;std::int64_t scale=0;interval(out,report,lo,hi,scale,p,mode);
    const auto common=std::min(scale,truth.scale);shift(lo.v,lo.v,scale-common);shift(hi.v,hi.v,scale-common);
    shift(center.v,truth.center.v,truth.scale-common);ref_set_ui(radius.v,truth.radius);shift(radius.v,radius.v,truth.scale-common);
    ref_sub(elo.v,center.v,radius.v);ref_add(ehi.v,center.v,radius.v);
    if(ref_cmp(lo.v,elo.v)>0||ref_cmp(hi.v,ehi.v)<0){std::fprintf(stderr,"derived log failed p=%zu mode=%u\n",p,unsigned(mode));assert(false);}++checked;
}
void certify_exp2(sbn::float_view out,sbn::float_report report,std::int64_t numerator,unsigned denominator_bits,std::size_t p,rounding mode){
    Ref lo,hi,temp,one;std::int64_t scale=0;interval(out,report,lo,hi,scale,p,mode);assert(!out.negative());
    if(ref_sgn(lo.v)<0)ref_set_ui(lo.v,0);
    for(unsigned j=0;j<denominator_bits;++j){multiply(temp.v,lo.v,lo.v);ref_set(lo.v,temp.v);multiply(temp.v,hi.v,hi.v);ref_set(hi.v,temp.v);}
    ref_set_ui(one.v,1);const auto exponent=Wide(scale)*(Wide(1)<<denominator_bits);
    assert(compare_scaled(lo.v,exponent,one.v,numerator)<=0&&compare_scaled(hi.v,exponent,one.v,numerator)>=0);++checked;
}
struct Team {sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned workers){sbn3_error e{};sbn3_arena_config a{std::size_t(1)<<28,32u<<20};assert(sbn3_arena_create(&a,&arena,&e)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&e)==SBN3_OK);sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config t{};t.workers=workers;t.stack_offset=65536;for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&e)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}};
struct Scoped {const f::FloatDerivedExpLogPrepared* prepared;sbn::float_ref out;sbn::float_view x;f::DerivedExpLogKind kind;rounding mode;sbn::float_report report{};};
void scoped(void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<Scoped*>(opaque);c.report=f::derived_explog_program_execute(*c.prepared,c.out,c.x,c.kind,{{},nullptr,scope},c.mode);}
void run(const f::FloatDerivedExpLogPrepared& prepared,std::size_t p,const Input& x,f::DerivedExpLogKind kind,
    const Ball* truth,std::int64_t numerator,unsigned denominator_bits,rounding mode,sbn3_team* team,bool scope,bool alias){
    const auto limbs=(p+63)/64,capacity=alias?std::max(limbs,x.words.size()):limbs;std::vector<std::uint64_t> words(capacity+2,guard);
    if(alias)std::copy(x.words.begin(),x.words.end(),words.begin());sbn::float_buffer out({words.data(),capacity},sbn::bits{p});
    const auto input=alias?sbn::float_view::from_magnitude({words.data(),x.words.size()},x.exponent,x.negative):x.view();
    allocation_watch_start();sbn::float_report report;
    if(scope){Scoped c{&prepared,out.ref(),input,kind,mode,{}};sbn3_team_run(team,scoped,&c);report=c.report;}
    else report=f::derived_explog_program_execute(prepared,out.ref(),input,kind,{{},team,nullptr},mode);
    assert(!allocation_watch_stop());assert(words[capacity]==guard&&words[capacity+1]==guard);
    if(alias)for(std::size_t j=limbs;j<x.words.size();++j)assert(words[j]==x.words[j]);
    if(kind==f::DerivedExpLogKind::exp2)certify_exp2(out.view(),report,numerator,denominator_bits,p,mode);else certify_log(out.view(),report,*truth,p,mode);
}
Input near_one(std::size_t gap,bool below){Input x;x.exponent=-std::int64_t(gap);x.words.assign(gap/64+1,0);
    if(below){for(std::size_t j=0;j<gap/64;++j)x.words[j]=UINT64_MAX;if(gap%64)x.words.back()=(UINT64_C(1)<<(gap%64))-1;}
    else{x.words.back()=UINT64_C(1)<<(gap%64);x.words[0]|=1;}while(!x.words.empty()&&!x.words.back())x.words.pop_back();return x;}
void exact_log(const f::FloatDerivedExpLogPrepared& prepared,std::size_t p,sbn3_team* team){
    Input x;x.words.assign(17,0);x.words.back()=UINT64_C(1)<<13;x.exponent=-1000;Ball truth;ref_set_ui(truth.center.v,37);
    for(auto mode:modes)run(prepared,p,x,f::DerivedExpLogKind::log2,&truth,0,0,mode,team,false,true);
}
void tall_fraction(const f::FloatDerivedExpLogPrepared& prepared,std::size_t p,sbn3_team* team){
    const auto gap=64*((p+575)/64),n=gap/64+1,limbs=(p+63)/64;const std::uint64_t one=1;
    for(bool minus:{false,true}){
        std::vector<std::uint64_t> words(n+2,minus?UINT64_MAX:0);words[0]=minus?UINT64_MAX:1;
        words[n-1]=minus?16:17;words[n]=words[n+1]=guard;
        sbn::float_buffer out({words.data(),n},sbn::bits{p});const auto x=sbn::float_view::from_magnitude({words.data(),n},-std::int64_t(gap));
        allocation_watch_start();const auto report=f::derived_explog_program_execute(prepared,out.ref(),x,f::DerivedExpLogKind::exp2,{{},team,nullptr});
        assert(!allocation_watch_stop());assert(sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},17))==std::partial_ordering::equivalent);
        // |2^(17+/-2^-gap)-2^17|<=2^(18-gap). A collapsed exact-integer
        // input would incorrectly return an exact report here.
        assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2>=18-std::int64_t(gap)&&
            report.error.exponent2<=18-std::int64_t(p));
        assert(words[n]==guard&&words[n+1]==guard);for(std::size_t j=limbs;j<n-1;++j)assert(words[j]==(minus?UINT64_MAX:0));++checked;
    }
}
void cases(std::size_t p,sbn3_team* team,bool many){
    const auto workers=team?sbn3_team_workers(team):1;
    for(auto kind:{f::DerivedExpLogKind::exp2,f::DerivedExpLogKind::log2,f::DerivedExpLogKind::log10}){
        allocation_watch_start();const auto need=f::derived_explog_program_requirements(sbn::bits{p},workers,kind);assert(!allocation_watch_stop()&&need.supported());
        sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);allocation_watch_start();
        const auto prepared=f::derived_explog_program_prepare(sbn::bits{p},workers,kind,storage.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
        if(kind==f::DerivedExpLogKind::exp2){
            for(auto mode:modes)run(prepared,p,Input{{1},-1,false},kind,nullptr,1,1,mode,team,team!=nullptr,true);
            if(many){
                for(auto numerator:{-7LL,-3LL,3LL,13LL}){const auto mag=numerator<0?std::uint64_t(-numerator):std::uint64_t(numerator);
                    run(prepared,p,Input{{mag},-2,numerator<0},kind,nullptr,std::int64_t(numerator),2,rounding::nearest_even,team,false,false);}
                const std::int64_t huge=std::int64_t(1)<<58;
                for(auto numerator:std::array<std::int64_t,2>{2*huge+1,-2*huge+1}){const auto mag=numerator<0?std::uint64_t(-numerator):std::uint64_t(numerator);
                    run(prepared,p,Input{{mag},-1,numerator<0},kind,nullptr,numerator,1,rounding::nearest_even,team,false,false);}
                tall_fraction(prepared,p,team);
            }
        }else{
            Ball truth;const Input input{{3},0,false};logarithm_reference(truth,input,kind,p+512);
            for(auto mode:modes)run(prepared,p,input,kind,&truth,0,0,mode,team,team!=nullptr,true);
            if(many){
                for(bool below:{false,true}){const auto gap=p+160;const auto x=near_one(gap,below);logarithm_reference(truth,x,kind,p+2*gap+512);
                    run(prepared,p,x,kind,&truth,0,0,rounding::nearest_even,team,false,true);}
                if(kind==f::DerivedExpLogKind::log2)exact_log(prepared,p,team);
                else{ref_set_ui(truth.center.v,3);truth.radius=0;truth.scale=0;
                    for(auto mode:modes)run(prepared,p,Input{{1000},0,false},kind,&truth,0,0,mode,team,false,false);}
            }
        }
        if(many){
            std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});const Input x=kind==f::DerivedExpLogKind::exp2?Input{{3},-1,false}:Input{{5},0,false};
            sbn::workspace scratch;const auto request=f::derived_explog_requirements(out.ref(),x.view(),kind,workers);scratch.reserve(request.scratch_bytes,request.scratch_alignment);
            allocation_watch_start();const auto report=f::derived_explog_into(out.ref(),x.view(),kind,scratch.on(team));assert(!allocation_watch_stop());
            if(kind==f::DerivedExpLogKind::exp2)certify_exp2(out.view(),report,3,1,p,rounding::nearest_even);
            else{Ball truth;logarithm_reference(truth,x,kind,p+512);certify_log(out.view(),report,truth,p,rounding::nearest_even);}
        }
    }
}
void specials(){
    std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});const std::uint64_t one=1;
    for(auto k:std::array<std::int64_t,4>{sbn::float_exponent_limit-1,sbn::float_exponent_limit,-sbn::float_exponent_limit-1,-sbn::float_exponent_limit-2}){
        const auto magnitude=k<0?std::uint64_t(-k):std::uint64_t(k);const auto input=sbn::float_view::from_magnitude({&magnitude,1},0,k<0);
        const auto need=f::derived_explog_requirements(out.ref(),input,f::DerivedExpLogKind::exp2);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=f::derived_explog_into(out.ref(),input,f::DerivedExpLogKind::exp2);assert(!allocation_watch_stop());
        if(k==sbn::float_exponent_limit)assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::overflow));
        else if(k<-sbn::float_exponent_limit-1)assert(out.view().zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
        else assert(out.view().finite()&&!out.view().zero()&&out.view().exponent2()==k&&report.error.exact());++checked;
    }
    for(auto kind:{f::DerivedExpLogKind::log2,f::DerivedExpLogKind::log10}){
        auto report=f::derived_explog_into(out.ref(),sbn::float_view::from_magnitude({&one,1},0),kind);assert(out.view().zero()&&report.error.exact());
        report=f::derived_explog_into(out.ref(),{},kind);assert(!out.view().finite()&&out.view().negative()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
        report=f::derived_explog_into(out.ref(),sbn::float_view::from_magnitude({&one,1},0,true),kind);assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
        report=f::derived_explog_into(out.ref(),sbn::float_view::infinity(),kind);assert(!out.view().finite()&&!out.view().negative());
        report=f::derived_explog_into(out.ref(),sbn::float_view::nan(),kind);assert(out.view().is_nan());
    }
    for(auto exponent:std::array<std::int64_t,2>{sbn::float_exponent_limit-1,-sbn::float_exponent_limit-1}){
        const auto input=sbn::float_view::from_magnitude({&one,1},exponent);
        const auto need=f::derived_explog_requirements(out.ref(),input,f::DerivedExpLogKind::log2);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=f::derived_explog_into(out.ref(),input,f::DerivedExpLogKind::log2);assert(!allocation_watch_stop());
        const auto magnitude=exponent<0?std::uint64_t(-exponent):std::uint64_t(exponent);
        assert(report.error.exact()&&sbn::compare(out.view(),sbn::float_view::from_magnitude({&magnitude,1},0,exponent<0))==std::partial_ordering::equivalent);++checked;
    }
}
}
int main(){
    for(auto p:{1u,3u,17u,63u,65u,127u,257u})cases(p,nullptr,true);
    for(auto p:{509u,1025u,4097u})cases(p,nullptr,false);
    {Team team(1);cases(129,team.team,true);}{Team team(4);cases(257,team.team,true);}
    specials();std::printf("derived exp/log independent enclosures: %zu\n",checked);
}
