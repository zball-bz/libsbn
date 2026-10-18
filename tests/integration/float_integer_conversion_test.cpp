// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
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
constexpr sbn::rounding modes[]{sbn::rounding::nearest_even,sbn::rounding::toward_zero,
    sbn::rounding::upward,sbn::rounding::downward,sbn::rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x92e8704a63d15bcf);
std::size_t checked=0;
struct Ref {ref_int value;Ref(){ref_init(value);}~Ref(){ref_clear(value);}Ref(const Ref&)=delete;};
bool increment(sbn::rounding mode,bool negative,bool discarded,int half,bool odd){
    switch(mode){
        case sbn::rounding::nearest_even:return half>0||(half==0&&odd);
        case sbn::rounding::toward_zero:return false;
        case sbn::rounding::upward:return discarded&&!negative;
        case sbn::rounding::downward:return discarded&&negative;
        case sbn::rounding::away_zero:return discarded;
    }
    assert(false);return false;
}
void check_report(sbn::float_report report,bool discarded,sbn::rounding mode){
    assert(report.error.exact()==!discarded);
    assert(sbn::has_flag(report.flags,sbn::float_flag::rounded)==discarded);
    assert(report.flags==(discarded?sbn::float_flag::rounded:sbn::float_flag::none));
    if(discarded)assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2==(mode==sbn::rounding::nearest_even?-1:0));
}
void small_cases(){
    for(std::uint64_t m=0;m<260;++m)for(int exponent=-10;exponent<=9;++exponent)
        for(bool negative:{false,true})for(auto mode:modes){
            std::array<std::uint64_t,3> output{guard,guard,guard};sbn::integer_buffer result{{output.data(),2}};
            const auto input=sbn::float_view::from_magnitude({&m,1},exponent,negative);
            __uint128_t q=m;bool discarded=false;int half=-1;
            if(exponent>=0)q<<=exponent;
            else{const auto r=unsigned(-exponent);q=m>>r;const auto remainder=m&((UINT64_C(1)<<r)-1);
                discarded=remainder!=0;const auto threshold=UINT64_C(1)<<(r-1);half=remainder<threshold?-1:remainder>threshold?1:0;
                if(increment(mode,negative,discarded,half,q&1))++q;}
            allocation_watch_start();const auto need=sbn::to_integer_requirements(result.ref(),input,mode);
            assert(need.supported()&&!need.scratch_bytes);
            const auto report=sbn::to_integer_into(result.ref(),input,mode);assert(!allocation_watch_stop());
            const auto value=result.view();__uint128_t actual=0;
            for(std::size_t i=value.size();i-->0;)actual=(actual<<64)|value.data()[i];
            assert(actual==q&&value.negative()==bool(q&&negative));assert(output[2]==guard);
            check_report(report,discarded,mode);++checked;
        }
}
void oracle_round(ref_number* expected,const ref_number* input,std::int64_t exponent,bool negative,
                  sbn::rounding mode,bool& discarded){
    discarded=false;ref_set(expected,input);
    if(exponent>=0){ref_mul_2exp(expected,expected,std::size_t(exponent));}
    else{
        const auto r=std::size_t(-exponent);Ref remainder,half;
        ref_tdiv_q_2exp(expected,input,r);ref_fdiv_r_2exp(remainder.value,input,r);discarded=ref_sgn(remainder.value)!=0;
        int relation=-1;
        if(r<=ref_sizeinbase(input,2)){
            ref_set_ui(half.value,1);ref_mul_2exp(half.value,half.value,r-1);relation=ref_cmp(remainder.value,half.value);
        }
        if(increment(mode,negative,discarded,relation,ref_get_ui(expected)&1))ref_add_ui(expected,expected,1);
    }
    if(negative&&ref_sgn(expected))ref_neg(expected,expected);
}
void run(std::span<const std::uint64_t> input,std::int64_t exponent,bool negative,sbn::rounding mode,bool alias){
    Ref original,expected,actual;ref_import(original.value,input.size(),-1,8,0,0,input.data());bool discarded=false;
    oracle_round(expected.value,original.value,exponent,negative,mode,discarded);
    sbn::integer_buffer empty{{}};const auto source=sbn::float_view::from_magnitude(input,exponent,negative);
    allocation_watch_start();const auto quoted=sbn::to_integer_requirements(empty.ref(),source,mode);assert(!allocation_watch_stop());
    assert(quoted.status==sbn::support::capacity||quoted.supported());
    const auto capacity=alias?std::max(input.size(),quoted.output_limbs):quoted.output_limbs;
    std::vector<std::uint64_t> storage(capacity+2,guard);if(alias)std::copy(input.begin(),input.end(),storage.begin());
    sbn::integer_buffer out{{storage.data(),capacity}};
    const auto actual_input=alias?sbn::float_view::from_magnitude({storage.data(),input.size()},exponent,negative):source;
    allocation_watch_start();const auto need=sbn::to_integer_requirements(out.ref(),actual_input,mode);
    assert(need.supported()&&!need.scratch_bytes&&need.output_limbs==quoted.output_limbs);
    const auto report=sbn::to_integer_into(out.ref(),actual_input,mode);assert(!allocation_watch_stop());
    ref_import(actual.value,out.view().size(),-1,8,0,0,out.view().data());if(out.negative())ref_neg(actual.value,actual.value);
    assert(ref_cmp(actual.value,expected.value)==0);check_report(report,discarded,mode);
    assert(storage[capacity]==guard&&storage[capacity+1]==guard);
    if(alias)for(std::size_t i=need.output_limbs;i<input.size();++i)assert(storage[i]==input[i]);
    // Independent exact absolute-error check on a common dyadic grid.
    Ref source_scaled,result_scaled,difference,bound;ref_set(source_scaled.value,original.value);
    if(negative)ref_neg(source_scaled.value,source_scaled.value);ref_set(result_scaled.value,actual.value);
    const auto base=std::min<std::int64_t>(exponent,report.error.exact()?0:report.error.exponent2);
    ref_mul_2exp(source_scaled.value,source_scaled.value,std::size_t(exponent-base));
    ref_mul_2exp(result_scaled.value,result_scaled.value,std::size_t(-base));
    ref_sub(difference.value,result_scaled.value,source_scaled.value);ref_abs(difference.value,difference.value);
    if(report.error.exact())assert(!ref_sgn(difference.value));
    else{ref_set_ui(bound.value,1);ref_mul_2exp(bound.value,bound.value,std::size_t(report.error.exponent2-base));assert(ref_cmp(difference.value,bound.value)<=0);}
    ++checked;
}
void wider_cases(){
    std::uint64_t state=UINT64_C(0xb41dc086397e2a15);
    for(std::size_t bits:{1u,2u,63u,64u,65u,127u,128u,129u,257u,1025u,4097u}){
        std::vector<std::uint64_t> input((bits+63)/64);
        for(auto& w:input){state^=state<<13;state^=state>>7;state^=state<<17;w=state;}
        if(bits%64)input.back()&=(UINT64_C(1)<<(bits%64))-1;
        input.back()|=UINT64_C(1)<<((bits-1)%64);
        const std::int64_t shifts[]{-std::int64_t(bits)-1,-std::int64_t(bits),-129,-65,-64,-63,-1,0,1,63,64,65,129};
        for(auto exponent:shifts)for(bool negative:{false,true})for(auto mode:modes)
            for(bool alias:{false,true})run(input,exponent,negative,mode,alias);
    }
    // Carry from a complete all-ones retained limb; exact half-way parity;
    // fractional all-zero low limbs, and a high source word disappearing.
    const std::array<std::uint64_t,3> witnesses[]{
        {UINT64_MAX,UINT64_MAX,1},{0,UINT64_MAX,1},{1,1,0},{3,1,0},{0,1,0},{UINT64_MAX,1,0}};
    for(const auto& input:witnesses)for(std::int64_t exponent:{-1,-63,-64,-65,-127,-128})
        for(bool negative:{false,true})for(auto mode:modes)run(input,exponent,negative,mode,true);
}
void queries_and_virtual_shifts(){
    std::array<std::uint64_t,5> words{1,2,3,4,guard};sbn::integer_buffer out{{words.data(),4}};
    for(auto x:{sbn::float_view::nan(),sbn::float_view::infinity(),sbn::float_view::infinity(true)}){
        allocation_watch_start();const auto need=sbn::to_integer_requirements(out.ref(),x);assert(!allocation_watch_stop());
        assert(need.status==sbn::support::unsupported);
    }
    const auto overlapping=sbn::float_view::from_magnitude({words.data()+1,2},0);
    assert(sbn::to_integer_requirements(out.ref(),overlapping).status==sbn::support::unsupported);
    const std::uint64_t one=1;
    allocation_watch_start();const auto huge=sbn::to_integer_requirements(out.ref(),
        sbn::float_view::from_magnitude({&one,1},sbn::float_exponent_limit-1));assert(!allocation_watch_stop());
    assert(huge.status==sbn::support::capacity&&huge.output_limbs==std::size_t(sbn::float_exponent_limit)/64&&!huge.scratch_bytes);
    for(bool negative:{false,true})for(auto mode:modes){
        const auto tiny=sbn::float_view::from_magnitude({&one,1},-sbn::float_exponent_limit-1,negative);
        allocation_watch_start();const auto need=sbn::to_integer_requirements(out.ref(),tiny,mode);
        assert(need.supported()&&need.output_limbs<=1&&!need.scratch_bytes);
        const auto report=sbn::to_integer_into(out.ref(),tiny,mode);assert(!allocation_watch_stop());
        const bool up=mode==sbn::rounding::away_zero||(mode==sbn::rounding::upward&&!negative)||(mode==sbn::rounding::downward&&negative);
        assert(out.size()==std::size_t(up)&&out.negative()==(up&&negative));if(up)assert(out.view().data()[0]==1);
        check_report(report,true,mode);
    }
    for(bool negative:{false,true}){
        const auto zero=sbn::float_view::from_magnitude({},INT64_MIN,negative);
        auto report=sbn::to_integer_into(out.ref(),zero);assert(report.error.exact()&&!out.size()&&!out.negative());
    }
    assert(words[4]==guard);
}
}
int main(){
    queries_and_virtual_shifts();small_cases();wider_cases();
    std::printf("float integer conversion: %zu independent rounding/error checks; all modes, aliases, capacities and virtual shifts PASS\n",checked);
}
