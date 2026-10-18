// Original libsbn tests; LGPL-3.0-or-later.
#include "float/exp_series.hpp"
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
namespace floating=sbn::v3::floating;
using Kind=floating::ExpSeriesKind;using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x6732ad40918febc5);
std::size_t certified=0,prepared_checked=0,extreme_checked=0;
struct Ref {ref_int value;Ref(){ref_init(value);}~Ref(){ref_clear(value);}Ref(const Ref&)=delete;};
std::vector<std::uint64_t> export_words(const ref_number* a){
    std::vector<std::uint64_t> result((ref_sizeinbase(a,2)+63)/64);std::size_t count=0;
    ref_export(result.data(),&count,-1,8,0,0,a);result.resize(count);return result;
}
// No ref_mul/ref_sqrt/general reference division: these witnesses use only
// independently exact scalar products, word divisions, shifts and sums.
void product(ref_number* out,const ref_number* a,const ref_number* b){
    const auto words=export_words(b);Ref sum,term;ref_set_ui(sum.value,0);
    for(std::size_t i=words.size();i-->0;){ref_mul_2exp(sum.value,sum.value,64);
        ref_mul_ui(term.value,a,words[i]);ref_add(sum.value,sum.value,term.value);}
    ref_set(out,sum.value);
}
void word_divide(ref_number* out,const ref_number* a,std::uint64_t divisor,bool upward){
    auto q=export_words(a);std::uint64_t remainder=0;
    for(std::size_t i=q.size();i-->0;){const __uint128_t x=(__uint128_t(remainder)<<64)|q[i];
        q[i]=std::uint64_t(x/divisor);remainder=std::uint64_t(x%divisor);}
    ref_import(out,q.size(),-1,8,0,0,q.data());if(upward&&remainder)ref_add_ui(out,out,1);
}
struct Input {std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}};
std::size_t width(sbn::float_view x){const auto m=x.magnitude();return m.zero()?0:64*(m.size()-1)+std::bit_width(m.data()[m.size()-1]);}
struct Reference {
    Ref exp_lower,exp_upper,m1_lower,m1_upper;std::int64_t exponent=0;
    Reference(const Input& input,std::size_t precision){
        const auto top=input.exponent+std::int64_t(width(input.view()));
        const auto Q=precision+256+std::size_t(std::max<std::int64_t>(0,-top));exponent=-std::int64_t(Q);
        Ref unit,M,lo,hi,tmp,remainder;ref_set_ui(unit.value,1);ref_mul_2exp(unit.value,unit.value,Q);
        ref_import(M.value,input.words.size(),-1,8,0,0,input.words.data());
        ref_set(lo.value,unit.value);ref_set(hi.value,unit.value);
        // Sum floor(log2(j)) is a lower bound for log2(J!). After J
        // terms, |r|<=1 gives tail <=2/(J+1)! < one Q-grid unit.
        std::size_t J=1,bits=0;while(bits<Q+32){++J;bits+=std::bit_width(J)-1;}
        for(std::size_t j=1;j<=J;++j){
            auto next=[&](ref_number* term,bool upward){
                product(tmp.value,term,M.value);
                if(input.exponent<0){const auto shift=std::size_t(-input.exponent);
                    ref_fdiv_r_2exp(remainder.value,tmp.value,shift);const bool omitted=ref_sgn(remainder.value)!=0;
                    ref_fdiv_q_2exp(tmp.value,tmp.value,shift);if(upward&&omitted)ref_add_ui(tmp.value,tmp.value,1);
                }else if(input.exponent)ref_mul_2exp(tmp.value,tmp.value,std::size_t(input.exponent));
                word_divide(term,tmp.value,j,upward);
            };
            next(lo.value,false);next(hi.value,true);
            if(input.negative&&(j&1)){ref_sub(m1_lower.value,m1_lower.value,hi.value);ref_sub(m1_upper.value,m1_upper.value,lo.value);}
            else{ref_add(m1_lower.value,m1_lower.value,lo.value);ref_add(m1_upper.value,m1_upper.value,hi.value);}
        }
        ref_sub_ui(m1_lower.value,m1_lower.value,1);ref_add_ui(m1_upper.value,m1_upper.value,1);
        ref_add(exp_lower.value,m1_lower.value,unit.value);ref_add(exp_upper.value,m1_upper.value,unit.value);
    }
};
void shift(ref_number* value,std::int64_t bits){assert(bits>=0&&bits<100000);if(bits)ref_mul_2exp(value,value,std::size_t(bits));}
void certify(sbn::float_view output,sbn::float_report report,std::size_t precision,rounding mode,Kind kind,
             const Input& input,const Reference& reference){
    assert(output.finite()&&!output.zero()&&width(output)<=precision&&output.negative()==(kind==Kind::expm1&&input.negative));
    assert(report.error.kind==sbn::error_kind::finite);
    const auto grid=output.exponent2()+std::int64_t(width(output))-std::int64_t(precision);
    assert(report.error.exponent2<=grid+(mode==rounding::nearest_even?0:1));
    const auto base=std::min({output.exponent2(),report.error.exponent2,reference.exponent});
    Ref center,radius,low,high,oracle_low,oracle_high;
    ref_import(center.value,output.magnitude().size(),-1,8,0,0,output.magnitude().data());
    if(output.negative())ref_neg(center.value,center.value);shift(center.value,output.exponent2()-base);
    ref_set_ui(radius.value,1);shift(radius.value,report.error.exponent2-base);
    ref_sub(low.value,center.value,radius.value);ref_add(high.value,center.value,radius.value);
    ref_set(oracle_low.value,kind==Kind::exp?reference.exp_lower.value:reference.m1_lower.value);
    ref_set(oracle_high.value,kind==Kind::exp?reference.exp_upper.value:reference.m1_upper.value);
    shift(oracle_low.value,reference.exponent-base);shift(oracle_high.value,reference.exponent-base);
    assert(ref_cmp(low.value,oracle_low.value)<=0&&ref_cmp(high.value,oracle_high.value)>=0);++certified;
}
struct Call {sbn::float_ref out;sbn::float_view input;Kind kind;sbn::execution ex;rounding mode;
    const floating::FloatExpSeriesPrepared* prepared=nullptr;sbn::float_report report{};};
void action(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;
    c.report=c.prepared?floating::exp_series_program_execute(*c.prepared,c.out,c.input,c.ex,c.mode):
        floating::exp_series_into(c.out,c.input,c.kind,c.ex,c.mode);}
void run(const Input& input,const Reference& reference,std::size_t precision,Kind kind,rounding mode,bool alias,
         const floating::FloatExpSeriesPrepared* prepared=nullptr,sbn3_team* team=nullptr,bool child=false){
    const auto count=(precision+63)/64,capacity=alias?std::max(count,input.words.size()):count;
    std::vector<std::uint64_t> output(capacity+2,guard);if(alias)std::copy(input.words.begin(),input.words.end(),output.begin());
    sbn::float_buffer out{{output.data(),capacity},sbn::bits{precision}};const auto original=input.words;
    const auto source=alias?sbn::float_view::from_magnitude({output.data(),input.words.size()},input.exponent,input.negative):input.view();
    sbn::workspace scratch;sbn::execution ex{{},team,nullptr};std::size_t bytes=0;
    if(!prepared){allocation_watch_start();const auto need=floating::exp_series_requirements(out.ref(),source,kind,team?4:1);
        assert(!allocation_watch_stop()&&need.supported());bytes=need.scratch_bytes;scratch.reserve(bytes+64,need.scratch_alignment);
        ex=scratch.on(team);ex.scratch=ex.scratch.first(bytes);
        std::fill(scratch.on().scratch.begin()+bytes,scratch.on().scratch.begin()+bytes+64,std::byte{0x5a});}
    Call call{out.ref(),source,kind,ex,mode,prepared,{}};allocation_watch_start();
    if(child){assert(team);sbn3_team_run(team,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,[](void*,sbn3_team_scope*){},nullptr,
            [](void* value,sbn3_team_scope* sub){assert(sbn3_team_first_worker(sub)==2);action(value,sub);},raw);
    },&call);}else if(team)sbn3_team_run(team,action,&call);
    else call.report=prepared?floating::exp_series_program_execute(*prepared,out.ref(),source,ex,mode):
        floating::exp_series_into(out.ref(),source,kind,ex,mode);
    assert(!allocation_watch_stop());assert(input.words==original&&output[capacity]==guard&&output[capacity+1]==guard);
    if(alias)for(std::size_t i=count;i<input.words.size();++i)assert(output[i]==input.words[i]);
    if(!prepared)for(std::size_t j=0;j<64;++j)assert(scratch.on().scratch[bytes+j]==std::byte{0x5a});
    certify(out.view(),call.report,precision,mode,kind,input,reference);if(prepared)++prepared_checked;
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
        const Input cases[]{{{1},0,false},{{3},-2,false},{{1},-10,false},{{3},-std::int64_t(p)-13,false}};
        for(auto input:cases)for(bool negative:{false,true}){input.negative=negative;Reference reference(input,p);
            for(auto kind:{Kind::exp,Kind::expm1})for(auto mode:modes)run(input,reference,p,kind,mode,true);}
    }
    Input tall;tall.words.resize((8197+63)/64,UINT64_C(0xe218fc74073a916b));tall.words.back()=17;tall.exponent=-8197;
    for(bool negative:{false,true}){tall.negative=negative;Reference reference(tall,129);
        for(auto kind:{Kind::exp,Kind::expm1})for(auto mode:modes)run(tall,reference,129,kind,mode,true);}
}
void prepared_cases(){
    Team team;
    for(std::size_t p:{65u,257u,1025u})for(bool negative:{false,true})for(auto kind:{Kind::exp,Kind::expm1}){
        allocation_watch_start();const auto need=floating::exp_series_program_requirements(sbn::bits{p},kind,negative,2);
        assert(!allocation_watch_stop()&&need.supported());sbn::workspace storage;storage.reserve(need.scratch_bytes+64,need.scratch_alignment);
        auto bytes=storage.on().scratch;std::fill(bytes.begin()+need.scratch_bytes,bytes.begin()+need.scratch_bytes+64,std::byte{0x31});
        allocation_watch_start();const auto prepared=floating::exp_series_program_prepare(sbn::bits{p},kind,negative,2,bytes.first(need.scratch_bytes));
        assert(!allocation_watch_stop());
        for(auto exponent:{-1,-11}){Input input{{1},exponent,negative};Reference reference(input,p);
            for(auto mode:modes)run(input,reference,p,kind,mode,true,&prepared,team.value,true);}
        for(std::size_t j=0;j<64;++j)assert(bytes[need.scratch_bytes+j]==std::byte{0x31});
    }
}
void leaves_and_domain(){
    std::array<std::uint64_t,65> words{};sbn::float_buffer out(words,sbn::bits{4096});const std::uint64_t one=1,three=3;
    for(bool negative:{false,true})for(auto kind:{Kind::exp,Kind::expm1}){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        const auto need=floating::exp_series_requirements(out.ref(),zero,kind);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=floating::exp_series_into(out.ref(),zero,kind);assert(!allocation_watch_stop()&&report.error.exact());
        if(kind==Kind::expm1)assert(out.view().zero()&&out.view().negative()==negative);
        else assert(sbn::compare(out.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
    }
    for(auto x:{sbn::float_view::nan(),sbn::float_view::infinity(),sbn::float_view::from_magnitude({&three,1},-1)})
        assert(!floating::exp_series_requirements(out.ref(),x,Kind::exp).supported());
    assert(!floating::exp_series_program_requirements(sbn::bits{4097},Kind::exp,false).supported());
    // Exact dyadic centers at the extreme lower exponent. These separate
    // leaf checks use the analytic |expm1(r)-r|<=r^2 certificate, not libm.
    std::array<std::uint64_t,2> data{};sbn::float_buffer small(data,sbn::bits{65});
    for(bool negative:{false,true})for(auto mode:modes){
        const auto r=sbn::float_view::from_magnitude({&three,1},-sbn::float_exponent_limit-2,negative);
        const auto need=floating::exp_series_requirements(small.ref(),r,Kind::expm1);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto report=floating::exp_series_into(small.ref(),r,Kind::expm1,{},mode);assert(!allocation_watch_stop());
        assert(sbn::compare(small.view(),r)==std::partial_ordering::equivalent&&report.error.kind==sbn::error_kind::finite);
        assert(report.error.exponent2== -2*sbn::float_exponent_limit);++extreme_checked;
    }
}
}
int main(){
    leaves_and_domain();ordinary();prepared_cases();
    std::printf("reduced exp series: %zu independent interval certificates, %zu prepared/scoped cases, %zu separate extreme leaves; guards/aliases/allocation PASS\n",
        certified,prepared_checked,extreme_checked);
}
