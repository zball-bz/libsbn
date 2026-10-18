// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "float/rootn.hpp"
#include "runtime/team.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
using Wide=__int128_t;
using sbn::rounding;
namespace floating=sbn::v3::floating;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,
    rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x794e18bc035a6d2f);
std::size_t exact_checks=0,series_checks=0,prepared_checks=0,structural_checks=0;
struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
std::vector<std::uint64_t> words(const ref_number* a){
    std::vector<std::uint64_t> result((ref_sizeinbase(a,2)+63)/64);std::size_t count=0;
    ref_export(result.data(),&count,-1,8,0,0,a);result.resize(count);return result;
}
void product(ref_number* out,const ref_number* a,const ref_number* b){
    const auto limbs=words(b);Ref sum,term;ref_set_ui(sum.value,0);
    // ref_mul/ref_addmul can fall back to the production large-product
    // witness. Scalar accumulation stays independently exact at every size.
    for(std::size_t i=limbs.size();i-->0;){
        ref_mul_2exp(sum.value,sum.value,64);ref_mul_ui(term.value,a,limbs[i]);ref_add(sum.value,sum.value,term.value);
    }
    ref_set(out,sum.value);
}
void power(ref_number* out,const ref_number* input,std::uint64_t n){
    Ref base,sum;ref_set(base.value,input);ref_set_ui(sum.value,1);
    while(n){if(n&1)product(sum.value,sum.value,base.value);n>>=1;if(n)product(base.value,base.value,base.value);}
    ref_set(out,sum.value);
}
// Independent one-word long division; avoids the oracle's general divider.
void divide_word(ref_number* out,const ref_number* input,std::uint64_t d,bool upward){
    auto q=words(input);std::uint64_t r=0;
    for(std::size_t i=q.size();i-->0;){const __uint128_t current=(__uint128_t(r)<<64)|q[i];q[i]=std::uint64_t(current/d);r=std::uint64_t(current%d);}
    ref_import(out,q.size(),-1,8,0,0,q.data());if(upward&&r)ref_add_ui(out,out,1);
}
void shift(ref_number* a,Wide bits){assert(bits>=0&&bits<2000000);if(bits&&ref_sgn(a))ref_mul_2exp(a,a,std::size_t(bits));}
int compare_scaled(const ref_number* a,Wide ae,const ref_number* b,Wide be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const Wide at=ae+ref_sizeinbase(a,2),bt=be+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref left,right;ref_set(left.value,a);ref_set(right.value,b);const auto base=std::min(ae,be);
    shift(left.value,ae-base);shift(right.value,be-base);return ref_cmp(left.value,right.value);
}
struct Input {
    std::vector<std::uint64_t> magnitude;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(magnitude,exponent,negative);}
};
Input random_input(std::size_t width,std::int64_t top,bool negative){
    static std::uint64_t state=UINT64_C(0x27341e803f64b195);
    Input input;input.magnitude.resize((width+63)/64);
    for(auto& w:input.magnitude){state^=state<<13;state^=state>>7;state^=state<<17;w=state;}
    if(width%64)input.magnitude.back()&=(UINT64_C(1)<<(width%64))-1;
    input.magnitude.back()|=UINT64_C(1)<<((width-1)%64);input.magnitude.front()|=1;
    input.exponent=top-std::int64_t(width);input.negative=negative;return input;
}
std::size_t width(sbn::float_view x){
    const auto m=x.magnitude();return m.zero()?0:64*(m.size()-1)+std::bit_width(m.data()[m.size()-1]);
}
struct Endpoints {Ref lower,upper;Wide exponent=0;};
void endpoints(Endpoints& e,sbn::float_view result,sbn::float_report report,std::size_t precision,rounding mode,bool negative){
    assert(result.finite()&&!result.zero()&&result.negative()==negative&&width(result)<=precision);
    assert(report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid)&&!sbn::has_flag(report.flags,sbn::float_flag::underflow)&&
           !sbn::has_flag(report.flags,sbn::float_flag::overflow));
    const Wide ulp=Wide(result.exponent2())+width(result)-precision;
    assert(report.error.exact()||Wide(report.error.exponent2)<=ulp+(mode==rounding::nearest_even?0:1));
    Ref center,radius;ref_import(center.value,result.magnitude().size(),-1,8,0,0,result.magnitude().data());
    e.exponent=result.exponent2();
    if(report.error.exact())ref_set_ui(radius.value,0);
    else{e.exponent=std::min(e.exponent,Wide(report.error.exponent2));shift(center.value,Wide(result.exponent2())-e.exponent);
        ref_set_ui(radius.value,1);shift(radius.value,Wide(report.error.exponent2)-e.exponent);}
    ref_sub(e.lower.value,center.value,radius.value);if(ref_sgn(e.lower.value)<0)ref_set_ui(e.lower.value,0);
    ref_add(e.upper.value,center.value,radius.value);
}
void exact_certificate(const Input& input,std::uint64_t degree,const Endpoints& e){
    Ref target,lo,hi;ref_import(target.value,input.magnitude.size(),-1,8,0,0,input.magnitude.data());
    power(lo.value,e.lower.value,degree);power(hi.value,e.upper.value,degree);
    assert(compare_scaled(lo.value,e.exponent*degree,target.value,input.exponent)<=0);
    assert(compare_scaled(hi.value,e.exponent*degree,target.value,input.exponent)>=0);++exact_checks;
}
// A rigorous fixed-grid binomial enclosure for (1 +/- 1/16)^(1/n).
// Every coefficient operation uses independent scalar integer arithmetic.
// For j>=1, |binom(1/n,j)|<=1/(n*j); the remaining absolute tail is
// <=2*16^(-J-1)/n. No host floating-point value is an accuracy oracle.
struct Series {
    Input input;Ref lower,upper;Wide exponent=0;
    Series(std::uint64_t degree,bool below,std::size_t precision){
        input={{below?15u:17u},-4,below&&(degree&1)};
        const std::size_t Q=precision+256,J=(precision+128)/4+1;exponent=-Wide(Q);
        ref_set_ui(lower.value,1);shift(lower.value,Q);ref_set(upper.value,lower.value);
        Ref lo,hi,term,tmp;ref_set_ui(term.value,1);shift(term.value,Q-4);
        divide_word(lo.value,term.value,degree,false);divide_word(hi.value,term.value,degree,true);
        for(std::size_t j=1;j<=J;++j){
            const bool negative=below||!(j&1);
            if(negative){ref_sub(lower.value,lower.value,hi.value);ref_sub(upper.value,upper.value,lo.value);}
            else{ref_add(lower.value,lower.value,lo.value);ref_add(upper.value,upper.value,hi.value);}
            if(j<J){
                auto next=[&](ref_number* value,bool upward){
                    ref_mul_ui(tmp.value,value,j);ref_mul_ui(tmp.value,tmp.value,degree);ref_sub(tmp.value,tmp.value,value);
                    divide_word(tmp.value,tmp.value,degree,upward);divide_word(tmp.value,tmp.value,j+1,upward);
                    const bool remainder=ref_fdiv_ui(tmp.value,16)!=0;
                    ref_fdiv_q_2exp(value,tmp.value,4);if(upward&&remainder)ref_add_ui(value,value,1);
                };
                next(lo.value,false);next(hi.value,true);
            }
        }
        ref_set_ui(term.value,1);shift(term.value,Wide(Q)+1-4*Wide(J+1));divide_word(term.value,term.value,degree,true);
        ref_sub(lower.value,lower.value,term.value);ref_add(upper.value,upper.value,term.value);
    }
};
struct Call {
    sbn::float_ref out;sbn::float_view input;std::uint64_t degree;sbn::execution ex;rounding mode;
    const floating::FloatRootnPrepared* prepared=nullptr;sbn::float_report report{};
};
void execute(Call& call){call.report=call.prepared?floating::rootn_program_execute(*call.prepared,call.out,call.input,call.ex,call.mode):
    sbn::rootn_into(call.out,call.input,call.degree,call.ex,call.mode);}
void on_scope(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}
void run(const Input& input,std::uint64_t degree,std::size_t precision,rounding mode,sbn::workspace& scratch,
         bool alias=false,const Series* series=nullptr,const floating::FloatRootnPrepared* prepared=nullptr,
         sbn3_team* team=nullptr,bool scope=false,bool structural=false){
    const auto output_words=(precision+63)/64,capacity=alias?std::max(output_words,input.magnitude.size()):output_words;
    std::vector<std::uint64_t> output(capacity+2,guard);if(alias)std::copy(input.magnitude.begin(),input.magnitude.end(),output.begin());
    const auto original=input.magnitude;sbn::float_buffer result{{output.data(),capacity},sbn::bits{precision}};
    const auto source=alias?sbn::float_view::from_magnitude({output.data(),input.magnitude.size()},input.exponent,input.negative):input.view();
    sbn::execution ex{{},team,nullptr};
    if(!prepared){allocation_watch_start();const auto need=sbn::rootn_requirements(result.ref(),source,degree,team?sbn3_team_workers(team):1);
        assert(!allocation_watch_stop()&&need.supported()&&need.output_limbs==output_words);
        scratch.reserve(need.scratch_bytes+64,need.scratch_alignment);ex=scratch.on(team);ex.scratch=ex.scratch.first(need.scratch_bytes);
        std::fill(scratch.on().scratch.begin()+need.scratch_bytes,scratch.on().scratch.begin()+need.scratch_bytes+64,std::byte{0x59});}
    Call call{result.ref(),source,degree,ex,mode,prepared,{}};allocation_watch_start();
    if(scope){assert(team);sbn3_team_run(team,on_scope,&call);}else execute(call);
    assert(!allocation_watch_stop());assert(input.magnitude==original);
    assert(output[capacity]==guard&&output[capacity+1]==guard);
    if(alias)for(std::size_t i=output_words;i<input.magnitude.size();++i)assert(output[i]==input.magnitude[i]);
    if(!prepared)for(std::size_t j=0;j<64;++j)assert(scratch.on().scratch[ex.scratch.size()+j]==std::byte{0x59});
    Endpoints e;endpoints(e,result.view(),call.report,precision,mode,input.negative);
    if(series){assert(compare_scaled(e.lower.value,e.exponent,series->lower.value,series->exponent)<=0);
        assert(compare_scaled(e.upper.value,e.exponent,series->upper.value,series->exponent)>=0);++series_checks;}
    else if(structural){const std::uint64_t one=1;
        assert(sbn::compare(result.view(),sbn::float_view::from_magnitude({&one,1},-1))!=std::partial_ordering::less);
        assert(sbn::compare(result.view(),sbn::float_view::from_magnitude({&one,1},1))!=std::partial_ordering::greater);++structural_checks;}
    else exact_certificate(input,degree,e);
    if(prepared)++prepared_checks;
}
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
void exact_cases(){
    sbn::workspace scratch;
    for(std::size_t p:{1u,2u,3u,17u,53u,63u,64u,65u,127u,129u,257u,511u,513u,1025u,4097u}){
        for(std::uint64_t n:{1u,2u,3u,4u,7u}){
            const auto input=random_input(p<1000?131:257,int(p%5)-2,(n&1)&&p%2);
            for(auto mode:modes)run(input,n,p,mode,scratch,true);
        }
    }
    for(std::uint64_t n:{5u,17u,31u,63u})for(auto mode:modes)run(random_input(193,3,n&1),n,129,mode,scratch);
    const auto tall=random_input(8197,1,true);
    for(auto mode:modes)run(tall,3,129,mode,scratch,true);
    for(std::int64_t top:{-sbn::float_exponent_limit,sbn::float_exponent_limit})for(auto n:{3u,7u})
        for(auto mode:modes)run(random_input(131,top,false),n,129,mode,scratch);
    for(auto mode:modes){run({{27},0,true},3,65,mode,scratch,true);run({{81},0,false},4,65,mode,scratch);}
}
void huge_degrees(){
    sbn::workspace scratch;
    const std::uint64_t degrees[]{UINT64_C(1)<<32,(UINT64_C(1)<<63)-1,UINT64_C(1)<<63,UINT64_MAX-1,UINT64_MAX};
    for(std::size_t p:{1u,17u,65u,129u,257u,1025u})for(auto n:degrees)for(bool below:{false,true}){
        Series reference(n,below,p);for(auto mode:modes)run(reference.input,n,p,mode,scratch,true,&reference);
    }
    Series high(UINT64_MAX,false,4097);run(high.input,UINT64_MAX,4097,rounding::nearest_even,scratch,true,&high);
    // Distinguished structural checks for enormous exponent ratios; the
    // rigorous full-precision huge-degree certificates are the series above.
    for(std::int64_t top:{-sbn::float_exponent_limit,sbn::float_exponent_limit})
        for(auto mode:modes)run(random_input(73,top,false),UINT64_MAX,129,mode,scratch,false,nullptr,nullptr,nullptr,false,true);
}
void prepared_cases(){
    Team team;sbn::workspace unused;
    const std::uint64_t degrees[]{0,1,2,3,17,UINT64_MAX};
    for(auto n:degrees)for(std::size_t p:{65u,257u}){
        allocation_watch_start();const auto need=floating::rootn_program_requirements(sbn::bits{p},n,4);
        assert(!allocation_watch_stop()&&need.supported());sbn::workspace storage;storage.reserve(need.scratch_bytes+64,need.scratch_alignment);
        auto bytes=storage.on().scratch;std::fill(bytes.begin()+need.scratch_bytes,bytes.begin()+need.scratch_bytes+64,std::byte{0x6d});
        allocation_watch_start();const auto prepared=floating::rootn_program_prepare(sbn::bits{p},n,4,bytes.first(need.scratch_bytes));
        assert(!allocation_watch_stop());
        if(!n){std::array<std::uint64_t,5> output{};sbn::float_buffer out(output,sbn::bits{p});
            auto report=floating::rootn_program_execute(prepared,out.ref(),{},sbn::execution{{},team.value,nullptr});
            assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));}
        else if(n==UINT64_MAX){for(bool below:{false,true}){Series reference(n,below,p);
            for(auto mode:modes)run(reference.input,n,p,mode,unused,true,&reference,&prepared,team.value,true);}}
        else for(auto width:{53u,259u,4099u})for(auto mode:modes)
            run(random_input(width,0,n&1),n,p,mode,unused,true,nullptr,&prepared,team.value,true);
        for(std::size_t i=0;i<64;++i)assert(bytes[need.scratch_bytes+i]==std::byte{0x6d});
    }
    const auto input=random_input(193,1,true);
    const auto need=floating::rootn_program_requirements(sbn::bits{129},3,2);
    assert(need.supported());sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);
    const auto prepared=floating::rootn_program_prepare(sbn::bits{129},3,2,storage.on().scratch.first(need.scratch_bytes));
    std::array<std::uint64_t,3> output{};sbn::float_buffer out(output,sbn::bits{129});
    Call call{out.ref(),input.view(),3,{},rounding::nearest_even,&prepared,{}};
    allocation_watch_start();
    sbn3_team_run(team.value,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,
            [](void*,sbn3_team_scope*){},nullptr,
            [](void* payload,sbn3_team_scope* child){assert(sbn3_team_first_worker(child)==2);on_scope(payload,child);},raw);
    },&call);
    assert(!allocation_watch_stop());Endpoints e;endpoints(e,out.view(),call.report,129,rounding::nearest_even,true);
    exact_certificate(input,3,e);++prepared_checks;
}
void specials(){
    std::array<std::uint64_t,3> data{};sbn::float_buffer out(data,sbn::bits{129});
    const std::uint64_t one=1;
    for(auto mode:modes){
        for(auto x:{sbn::float_view{},sbn::float_view::nan(),sbn::float_view::infinity()}){
            auto r=sbn::rootn_into(out.ref(),x,0,{},mode);assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));}
        for(bool negative:{false,true})for(std::uint64_t n:{1u,2u,3u,4u}){
            auto r=sbn::rootn_into(out.ref(),sbn::float_view::from_magnitude({},0,negative),n,{},mode);
            assert(out.view().zero()&&out.view().negative()==negative&&r.error.exact());
            r=sbn::rootn_into(out.ref(),sbn::float_view::infinity(negative),n,{},mode);
            if(negative&&!(n&1))assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
            else assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==negative);
        }
        auto r=sbn::rootn_into(out.ref(),sbn::float_view::from_magnitude({&one,1},0,true),UINT64_MAX,{},mode);
        assert(r.error.exact()&&out.view().negative()&&sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},0,true))==std::partial_ordering::equivalent);
        r=sbn::rootn_into(out.ref(),sbn::float_view::from_magnitude({&one,1},0,true),UINT64_MAX-1,{},mode);
        assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
        r=sbn::rootn_into(out.ref(),sbn::float_view::from_magnitude({&one,1},-63),63,{},mode);
        assert(r.error.exact()&&sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},-1))==std::partial_ordering::equivalent);
    }
}
}
int main(){
    specials();exact_cases();huge_degrees();prepared_cases();
    std::printf("float rootn: %zu independent exact endpoint and %zu rigorous binomial certificates; %zu prepared, %zu separate structural cases; allocation/alias/scope gates PASS\n",
        exact_checks,series_checks,prepared_checks,structural_checks);
}
