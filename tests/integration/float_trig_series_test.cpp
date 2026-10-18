// Original libsbn tests; LGPL-3.0-or-later.
#include "float/trig_series.hpp"
#include "runtime/team.hpp"
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
namespace floating=sbn::v3::floating;using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x32107a5e8b649cfd);
std::size_t certified=0,prepared_checked=0,extreme_checked=0;
struct Ref {ref_int value;Ref(){ref_init(value);}~Ref(){ref_clear(value);}Ref(const Ref&)=delete;};
std::vector<std::uint64_t> words(const ref_number* a){
    std::vector<std::uint64_t> result((ref_sizeinbase(a,2)+63)/64);std::size_t count=0;
    ref_export(result.data(),&count,-1,8,0,0,a);result.resize(count);return result;
}
void product(ref_number* out,const ref_number* a,const ref_number* b){
    // Only independent scalar products; never the large ref_mul fallback.
    const auto digits=words(b);Ref sum,term;ref_set_ui(sum.value,0);
    for(std::size_t i=digits.size();i-->0;){ref_mul_2exp(sum.value,sum.value,64);
        ref_mul_ui(term.value,a,digits[i]);ref_add(sum.value,sum.value,term.value);}
    ref_set(out,sum.value);
}
void divide_word(ref_number* out,const ref_number* a,std::uint64_t d,bool upward){
    auto q=words(a);std::uint64_t remainder=0;
    for(std::size_t i=q.size();i-->0;){const __uint128_t x=(__uint128_t(remainder)<<64)|q[i];
        q[i]=std::uint64_t(x/d);remainder=std::uint64_t(x%d);}
    ref_import(out,q.size(),-1,8,0,0,q.data());if(upward&&remainder)ref_add_ui(out,out,1);
}
struct Input {std::vector<std::uint64_t> digits;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(digits,exponent,negative);}};
std::size_t width(sbn::float_view x){const auto m=x.magnitude();return m.zero()?0:64*(m.size()-1)+std::bit_width(m.data()[m.size()-1]);}
struct Reference {
    Ref sine_low,sine_high,cosine_low,cosine_high;std::int64_t exponent=0;
    Reference(const Input& input,std::size_t p){
        const auto top=input.exponent+std::int64_t(width(input.view()));
        const auto Q=p+256+std::size_t(std::max<std::int64_t>(0,-top));exponent=-std::int64_t(Q);
        Ref M,lo,hi,unit,tmp,remainder;ref_import(M.value,input.digits.size(),-1,8,0,0,input.digits.data());
        ref_set_ui(unit.value,1);ref_mul_2exp(unit.value,unit.value,Q);
        ref_set(lo.value,unit.value);ref_set(hi.value,unit.value);ref_set(cosine_low.value,unit.value);ref_set(cosine_high.value,unit.value);
        std::size_t J=1,bits=0;while(bits<Q+32){++J;bits+=std::bit_width(J)-1;}
        // Direct fixed-grid factorial series, independently directed at
        // every scalar operation. Either subseries tail is bounded by the
        // exp absolute tail 2/(J+1)! < one grid unit for |r|<=1.
        for(std::size_t j=1;j<=J;++j){
            auto next=[&](ref_number* term,bool upward){
                product(tmp.value,term,M.value);
                if(input.exponent<0){const auto count=std::size_t(-input.exponent);
                    ref_fdiv_r_2exp(remainder.value,tmp.value,count);const bool discarded=ref_sgn(remainder.value)!=0;
                    ref_fdiv_q_2exp(tmp.value,tmp.value,count);if(upward&&discarded)ref_add_ui(tmp.value,tmp.value,1);
                }else if(input.exponent)ref_mul_2exp(tmp.value,tmp.value,std::size_t(input.exponent));
                divide_word(term,tmp.value,j,upward);
            };
            next(lo.value,false);next(hi.value,true);
            auto* low=(j&1)?sine_low.value:cosine_low.value;
            auto* high=(j&1)?sine_high.value:cosine_high.value;
            const bool subtract=(j/2)&1;
            if(subtract){ref_sub(low,low,hi.value);ref_sub(high,high,lo.value);}
            else{ref_add(low,low,lo.value);ref_add(high,high,hi.value);}
        }
        ref_sub_ui(sine_low.value,sine_low.value,1);ref_add_ui(sine_high.value,sine_high.value,1);
        ref_sub_ui(cosine_low.value,cosine_low.value,1);ref_add_ui(cosine_high.value,cosine_high.value,1);
        if(input.negative){ref_set(tmp.value,sine_low.value);ref_neg(sine_low.value,sine_high.value);ref_neg(sine_high.value,tmp.value);}
    }
};
void shift(ref_number* a,std::int64_t n){assert(n>=0&&n<100000);if(n)ref_mul_2exp(a,a,std::size_t(n));}
void certificate(sbn::float_view value,sbn::float_report report,std::size_t p,rounding mode,bool negative,
                 const ref_number* reference_low,const ref_number* reference_high,std::int64_t reference_exponent){
    assert(value.finite()&&!value.zero()&&width(value)<=p&&value.negative()==negative&&report.error.kind==sbn::error_kind::finite);
    const auto grid=value.exponent2()+std::int64_t(width(value))-std::int64_t(p);
    assert(report.error.exponent2<=grid+(mode==rounding::nearest_even?0:1));
    const auto base=std::min({value.exponent2(),report.error.exponent2,reference_exponent});
    Ref center,radius,low,high,oracle_low,oracle_high;
    ref_import(center.value,value.magnitude().size(),-1,8,0,0,value.magnitude().data());
    if(value.negative())ref_neg(center.value,center.value);shift(center.value,value.exponent2()-base);
    ref_set_ui(radius.value,1);shift(radius.value,report.error.exponent2-base);
    ref_sub(low.value,center.value,radius.value);ref_add(high.value,center.value,radius.value);
    ref_set(oracle_low.value,reference_low);ref_set(oracle_high.value,reference_high);
    shift(oracle_low.value,reference_exponent-base);shift(oracle_high.value,reference_exponent-base);
    assert(ref_cmp(low.value,oracle_low.value)<=0&&ref_cmp(high.value,oracle_high.value)>=0);++certified;
}
struct Call {sbn::float_ref sine,cosine;sbn::float_view input;sbn::execution ex;rounding mode;
    const floating::FloatTrigSeriesPrepared* prepared=nullptr;floating::TrigSincosReport report{};};
void action(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;
    c.report=c.prepared?floating::sincos_series_execute(*c.prepared,c.sine,c.cosine,c.input,c.ex,c.mode):
        floating::trig_series_into(c.sine,c.cosine,c.input,c.ex,c.mode);}
void run(const Input& input,const Reference& reference,std::size_t sp,std::size_t cp,rounding mode,unsigned alias,
         const floating::FloatTrigSeriesPrepared* prepared=nullptr,sbn3_team* team=nullptr,bool child=false){
    const auto sn=(sp+63)/64,cn=(cp+63)/64;
    const auto sc=alias==1?std::max(sn,input.digits.size()):sn,cc=alias==2?std::max(cn,input.digits.size()):cn;
    std::vector<std::uint64_t> sv(sc+2,guard),cv(cc+2,guard);const auto original=input.digits;
    if(alias)std::copy(input.digits.begin(),input.digits.end(),alias==1?sv.begin():cv.begin());
    sbn::float_buffer sine{{sv.data(),sc},sbn::bits{sp}},cosine{{cv.data(),cc},sbn::bits{cp}};
    const auto source=alias?sbn::float_view::from_magnitude({alias==1?sv.data():cv.data(),input.digits.size()},input.exponent,input.negative):input.view();
    sbn::workspace scratch;sbn::execution ex{{},team,nullptr};std::size_t bytes=0;
    if(!prepared){allocation_watch_start();const auto need=floating::trig_series_requirements(sine.ref(),cosine.ref(),source,team?4:1);
        assert(!allocation_watch_stop()&&need.supported());bytes=need.scratch_bytes;scratch.reserve(bytes+64,need.scratch_alignment);
        ex=scratch.on(team);ex.scratch=ex.scratch.first(bytes);
        std::fill(scratch.on().scratch.begin()+bytes,scratch.on().scratch.begin()+bytes+64,std::byte{0x3e});}
    Call call{sine.ref(),cosine.ref(),source,ex,mode,prepared,{}};allocation_watch_start();
    if(child){assert(team);sbn3_team_run(team,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,[](void*,sbn3_team_scope*){},nullptr,
            [](void* p,sbn3_team_scope* sub){assert(sbn3_team_first_worker(sub)==2);action(p,sub);},raw);
    },&call);}else if(team)sbn3_team_run(team,action,&call);
    else call.report=prepared?floating::sincos_series_execute(*prepared,sine.ref(),cosine.ref(),source,ex,mode):
        floating::trig_series_into(sine.ref(),cosine.ref(),source,ex,mode);
    assert(!allocation_watch_stop());assert(input.digits==original&&sv[sc]==guard&&sv[sc+1]==guard&&cv[cc]==guard&&cv[cc+1]==guard);
    if(alias){const auto& v=alias==1?sv:cv;const auto n=alias==1?sn:cn;
        for(std::size_t i=n;i<input.digits.size();++i)assert(v[i]==input.digits[i]);}
    if(!prepared)for(std::size_t j=0;j<64;++j)assert(scratch.on().scratch[bytes+j]==std::byte{0x3e});
    certificate(sine.view(),call.report.sine,sp,mode,input.negative,reference.sine_low.value,reference.sine_high.value,reference.exponent);
    certificate(cosine.view(),call.report.cosine,cp,mode,false,reference.cosine_low.value,reference.cosine_high.value,reference.exponent);
    if(prepared)++prepared_checked;
}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
void ordinary(){
    for(std::size_t p:{1u,2u,17u,63u,64u,65u,127u,129u,257u,511u,513u,1025u,4096u}){
        const Input cases[]{{{1},0,false},{{3},-2,false},{{1},-10,false},{{3},-std::int64_t(p/2)-15,false}};
        for(auto input:cases)for(bool negative:{false,true}){input.negative=negative;Reference reference(input,p);
            for(auto mode:modes)run(input,reference,p,p,mode,negative?2:1);}
    }
    Input tall;tall.digits.resize((8197+63)/64,UINT64_C(0x92581ab407e63cdf));tall.digits.back()=19;tall.exponent=-8197;
    for(bool negative:{false,true}){tall.negative=negative;Reference reference(tall,257);
        for(auto mode:modes){run(tall,reference,257,65,mode,1);run(tall,reference,65,257,mode,2);}}
}
void prepared_cases(){
    Team team;
    for(std::size_t p:{65u,257u,1025u}){
        allocation_watch_start();const auto need=floating::trig_series_program_requirements(sbn::bits{p},2);
        assert(!allocation_watch_stop()&&need.supported());sbn::workspace storage;storage.reserve(need.scratch_bytes+64,need.scratch_alignment);
        auto bytes=storage.on().scratch;std::fill(bytes.begin()+need.scratch_bytes,bytes.begin()+need.scratch_bytes+64,std::byte{0x62});
        allocation_watch_start();const auto prepared=floating::trig_series_program_prepare(sbn::bits{p},2,bytes.first(need.scratch_bytes));
        assert(!allocation_watch_stop());
        for(bool negative:{false,true})for(int exponent:{-1,-11}){Input input{{1},exponent,negative};Reference reference(input,p);
            for(auto mode:modes){run(input,reference,p,17,mode,1,&prepared,team.value,true);run(input,reference,17,p,mode,2,&prepared,team.value,true);}}
        for(std::size_t j=0;j<64;++j)assert(bytes[need.scratch_bytes+j]==std::byte{0x62});
    }
}
void leaves_and_domain(){
    std::array<std::uint64_t,65> sv{},cv{};sbn::float_buffer sine(sv,sbn::bits{4096}),cosine(cv,sbn::bits{4096});
    const std::uint64_t one=1,three=3;
    for(bool negative:{false,true})for(auto mode:modes){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        const auto need=floating::trig_series_requirements(sine.ref(),cosine.ref(),zero);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=floating::trig_series_into(sine.ref(),cosine.ref(),zero,{},mode);assert(!allocation_watch_stop());
        assert(sine.view().zero()&&sine.view().negative()==negative&&report.sine.error.exact()&&report.cosine.error.exact());
        assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
    }
    for(auto x:{sbn::float_view::nan(),sbn::float_view::infinity(),sbn::float_view::from_magnitude({&three,1},-1)})
        assert(!floating::trig_series_requirements(sine.ref(),cosine.ref(),x).supported());
    assert(!floating::trig_series_program_requirements(sbn::bits{4097}).supported());
    std::array<std::uint64_t,2> a{},b{};sbn::float_buffer small_s(a,sbn::bits{65}),small_c(b,sbn::bits{65});
    for(bool negative:{false,true})for(auto mode:modes){
        const auto r=sbn::float_view::from_magnitude({&three,1},-sbn::float_exponent_limit-2,negative);
        const auto need=floating::trig_series_requirements(small_s.ref(),small_c.ref(),r);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=floating::trig_series_into(small_s.ref(),small_c.ref(),r,{},mode);assert(!allocation_watch_stop());
        assert(sbn::compare(small_s.view(),r)==std::partial_ordering::equivalent);
        assert(sbn::compare(small_c.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
        assert(report.sine.error.exponent2==-3*sbn::float_exponent_limit&&report.cosine.error.exponent2==-2*sbn::float_exponent_limit-1);++extreme_checked;
    }
}
}
int main(){
    leaves_and_domain();ordinary();prepared_cases();
    std::printf("reduced trig series: %zu independent component interval certificates, %zu prepared/scoped pairs, %zu separate extreme leaves; mixed precision/aliases/allocation PASS\n",
        certified,prepared_checked,extreme_checked);
}
