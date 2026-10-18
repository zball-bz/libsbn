// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "float/power.hpp"
#include "runtime/team.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
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
constexpr std::uint64_t guard=UINT64_C(0xb1f97a0d452ac683);
std::uint64_t random_state=UINT64_C(0xe37af9417085bc62);
std::size_t exact_checks=0,analytic_checks=0,prepared_checks=0;
std::uint64_t magnitude(std::int64_t n){return n<0?std::uint64_t(0)-std::uint64_t(n):std::uint64_t(n);}
std::uint64_t random_word(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}
};
std::size_t word_bits(std::span<const std::uint64_t> words){
    auto n=words.size();while(n&&!words[n-1])--n;
    if(!n)return 0;std::size_t high=0;for(auto w=words[n-1];w;w>>=1)++high;
    return 64*(n-1)+high;
}
void shift(Ref& value,Wide bits){
    if(!ref_sgn(value.value))return;
    assert(bits>=0&&bits<2000000);if(bits)ref_mul_2exp(value.value,value.value,std::size_t(bits));
}
// Always independent. The generic ref_mul may use a production witness for
// large operands, so every product is accumulated from exact scalar products.
void product(ref_number* out,const ref_number* a,const ref_number* b){
    std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);std::size_t count=0;
    ref_export(words.data(),&count,-1,8,0,0,b);assert(count<=words.size());
    Ref sum,term;ref_set_ui(sum.value,0);
    for(std::size_t j=0;j<count;++j)if(words[j]){
        ref_mul_ui(term.value,a,words[j]);if(j)ref_mul_2exp(term.value,term.value,64*j);
        ref_add(sum.value,sum.value,term.value);
    }
    ref_set(out,sum.value);
}
int compare_scaled(const ref_number* a,Wide ae,const ref_number* b,Wide be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=ae+ref_sizeinbase(a,2),bt=be+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref left,right;ref_set(left.value,a);ref_set(right.value,b);
    const auto base=std::min(ae,be);shift(left,ae-base);shift(right,be-base);
    return ref_cmp(left.value,right.value);
}
Input exported(const ref_number* value,std::int64_t exponent,bool negative=false){
    Input input;input.words.resize((ref_sizeinbase(value,2)+63)/64);std::size_t count=0;
    ref_export(input.words.data(),&count,-1,8,0,0,value);assert(count==input.words.size());
    input.exponent=exponent;input.negative=negative;return input;
}
Input random_input(std::size_t bits,int top,bool negative=false){
    Input input;input.words.resize((bits+63)/64);
    for(auto& w:input.words)w=random_word();
    if(bits%64)input.words.back()&=(UINT64_C(1)<<(bits%64))-1;
    input.words.back()|=UINT64_C(1)<<((bits-1)%64);input.words.front()|=1;
    input.exponent=std::int64_t(top)-std::int64_t(bits);input.negative=negative;return input;
}
void exact_power(ref_number* out,const Input& input,std::uint64_t exponent){
    assert(exponent<=64);Ref base,acc;ref_import(base.value,input.words.size(),-1,8,0,0,input.words.data());
    ref_set_ui(acc.value,1);
    while(exponent){if(exponent&1)product(acc.value,acc.value,base.value);exponent>>=1;
        if(exponent)product(base.value,base.value,base.value);}
    ref_set(out,acc.value);
}
struct Endpoints {Ref low,high;Wide scale=0;};
void endpoints(Endpoints& bounds,sbn::float_view output,const sbn::float_report& report,
               std::size_t precision,rounding mode,bool negative){
    assert(output.finite()&&output.negative()==negative);
    assert(word_bits(output.magnitude().limbs())<=precision);
    assert(report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    if(!output.zero()&&!report.error.exact()){
        const auto ulp=Wide(output.exponent2())+word_bits(output.magnitude().limbs())-precision;
        assert(Wide(report.error.exponent2)<=ulp+(mode==rounding::nearest_even?0:1));
    }
    Ref center,radius;ref_import(center.value,output.magnitude().size(),-1,8,0,0,output.magnitude().data());
    bounds.scale=output.exponent2();
    if(report.error.exact())ref_set_ui(radius.value,0);
    else{
        bounds.scale=std::min(bounds.scale,Wide(report.error.exponent2));
        shift(center,Wide(output.exponent2())-bounds.scale);
        ref_set_ui(radius.value,1);shift(radius,Wide(report.error.exponent2)-bounds.scale);
    }
    ref_sub(bounds.low.value,center.value,radius.value);if(ref_sgn(bounds.low.value)<0)ref_set_ui(bounds.low.value,0);
    ref_add(bounds.high.value,center.value,radius.value);
}
void exact_certificate(const Input& input,std::int64_t exponent,sbn::float_view output,
                       const sbn::float_report& report,std::size_t precision,rounding mode){
    const auto n=magnitude(exponent);Endpoints bounds;
    endpoints(bounds,output,report,precision,mode,input.negative&&(n&1));
    Ref target,lower,upper,one;exact_power(target.value,input,n);ref_set_ui(one.value,1);
    const auto scale=Wide(input.exponent)*n;
    if(exponent<0){
        product(lower.value,bounds.low.value,target.value);product(upper.value,bounds.high.value,target.value);
        assert(compare_scaled(lower.value,bounds.scale+scale,one.value,0)<=0);
        assert(compare_scaled(upper.value,bounds.scale+scale,one.value,0)>=0);
    }else{
        assert(compare_scaled(bounds.low.value,bounds.scale,target.value,scale)<=0);
        assert(compare_scaled(bounds.high.value,bounds.scale,target.value,scale)>=0);
    }
    ++exact_checks;
}

// For x=1 +/- 2^-k, N=2^r and k=r+20, Nt=2^-20. Evaluate J binomial
// terms with exact integer coefficients and enclose the remaining tail by
// 2*(Nt)^(J+1). This gives an independent rigorous high-precision reference
// even for N=2^63, without creating the enormous exact integer power.
struct NearOne {
    Input input;
    Ref low,high;
    Wide scale;
    NearOne(unsigned r,bool below,std::size_t precision){
        const std::uint64_t n=UINT64_C(1)<<r;const std::size_t k=r+20,jmax=(precision+64)/20+1;
        Ref integer;ref_set_ui(integer.value,1);ref_mul_2exp(integer.value,integer.value,k);
        if(below)ref_sub_ui(integer.value,integer.value,1);else ref_add_ui(integer.value,integer.value,1);
        input=exported(integer.value,-std::int64_t(k));
        scale=-Wide(k*jmax);
        Ref coefficient,sum,term,tail;ref_set_ui(coefficient.value,1);ref_set_ui(sum.value,0);
        for(std::size_t j=0;j<=jmax;++j){
            ref_set(term.value,coefficient.value);shift(term,Wide(k*(jmax-j)));
            if(below&&(j&1))ref_sub(sum.value,sum.value,term.value);else ref_add(sum.value,sum.value,term.value);
            if(j<jmax){ref_mul_ui(coefficient.value,coefficient.value,n-j);ref_divexact_ui(coefficient.value,coefficient.value,j+1);}
        }
        ref_set_ui(tail.value,1);shift(tail,1-Wide(20*(jmax+1))-scale);
        ref_sub(low.value,sum.value,tail.value);ref_add(high.value,sum.value,tail.value);assert(ref_sgn(low.value)>0);
    }
};
void analytic_certificate(const NearOne& reference,std::int64_t exponent,sbn::float_view output,
                          const sbn::float_report& report,std::size_t precision,rounding mode){
    Endpoints bounds;endpoints(bounds,output,report,precision,mode,false);
    if(exponent>0){
        assert(compare_scaled(bounds.low.value,bounds.scale,reference.low.value,reference.scale)<=0);
        assert(compare_scaled(bounds.high.value,bounds.scale,reference.high.value,reference.scale)>=0);
    }else{
        Ref lower,upper,one;ref_set_ui(one.value,1);
        product(lower.value,bounds.low.value,reference.high.value);
        product(upper.value,bounds.high.value,reference.low.value);
        assert(compare_scaled(lower.value,bounds.scale+reference.scale,one.value,0)<=0);
        assert(compare_scaled(upper.value,bounds.scale+reference.scale,one.value,0)>=0);
    }
    ++analytic_checks;
}

struct ScopeCall {
    sbn::float_ref out;sbn::float_view input;std::int64_t exponent;sbn::execution ex;rounding mode;
    const floating::FloatPowerPrepared* prepared;sbn::float_report report{};
};
void scope_action(void* opaque,sbn3_team_scope* scope){
    auto& call=*static_cast<ScopeCall*>(opaque);call.ex.scope=scope;call.ex.team=nullptr;
    call.report=call.prepared?floating::power_program_execute(*call.prepared,call.out,call.input,call.ex,call.mode):
        sbn::powi_into(call.out,call.input,call.exponent,call.ex,call.mode);
}
void run(const Input& input,std::int64_t exponent,std::size_t precision,rounding mode,sbn::workspace& scratch,
         bool alias=false,sbn3_team* team=nullptr,bool scope=false,const floating::FloatPowerPrepared* prepared=nullptr,
         const NearOne* reference=nullptr){
    const auto output_words=(precision+63)/64,capacity=alias?std::max(output_words,input.words.size()):output_words;
    std::vector<std::uint64_t> words(capacity+2,guard);if(alias)std::copy(input.words.begin(),input.words.end(),words.begin());
    const auto original=input.words;
    sbn::float_buffer output{{words.data(),capacity},sbn::bits{precision}};
    const auto source=alias?sbn::float_view::from_magnitude({words.data(),input.words.size()},input.exponent,input.negative):input.view();
    sbn::execution ex{{},team,nullptr};
    if(!prepared){
        allocation_watch_start();const auto need=sbn::powi_requirements(output.ref(),source,exponent,team?sbn3_team_workers(team):1);
        assert(!allocation_watch_stop());assert(need.supported()&&need.output_limbs==output_words);
        scratch.reserve(need.scratch_bytes,need.scratch_alignment);ex=scratch.on(team);ex.scratch=ex.scratch.first(need.scratch_bytes);
    }
    sbn::float_report report;allocation_watch_start();
    if(scope){assert(team);ScopeCall call{output.ref(),source,exponent,ex,mode,prepared,{}};
        sbn3_team_run(team,scope_action,&call);report=call.report;
    }else report=prepared?floating::power_program_execute(*prepared,output.ref(),source,ex,mode):
        sbn::powi_into(output.ref(),source,exponent,ex,mode);
    assert(!allocation_watch_stop());assert(input.words==original);
    assert(words[capacity]==guard&&words[capacity+1]==guard);
    if(alias)for(std::size_t j=output_words;j<input.words.size();++j)assert(words[j]==input.words[j]);
    if(reference)analytic_certificate(*reference,exponent,output.view(),report,precision,mode);
    else exact_certificate(input,exponent,output.view(),report,precision,mode);
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
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_stats stats{};
        sbn3_arena_get_stats(arena,&stats);assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena);}
};
void exact_cases(){
    sbn::workspace scratch;
    const std::size_t precisions[]{1,2,3,17,31,53,63,64,65,127,129,257,511,513,1025,4097};
    const std::int64_t exponents[]{-37,-17,-3,-2,-1,0,1,2,3,7,16,37};
    for(auto p:precisions){
        const auto input=random_input(131+(p%5),int(p%3)-1,p%2);
        for(auto n:exponents)for(auto mode:modes)run(input,n,p,mode,scratch,n%2!=0);
    }
    const auto large=random_input(8197,1,true);
    for(auto n:{-3,-2,2,3})for(auto mode:modes)run(large,n,257,mode,scratch,true);
    // The positive square overflows, but its reciprocal is in the lowest
    // finite binade. Reciprocating the normalized base first preserves it.
    const Input extreme{{5},sbn::float_exponent_limit/2-2,false};
    for(auto mode:modes)run(extreme,-2,129,mode,scratch,true);
}
void huge_near_one(){
    sbn::workspace scratch;
    for(std::size_t p:{17u,65u,257u,1025u,4097u})for(unsigned r:{20u,40u,62u,63u})for(bool below:{false,true}){
        NearOne reference(r,below,p);
        const auto negative=r==63?std::numeric_limits<std::int64_t>::min():-std::int64_t(UINT64_C(1)<<r);
        for(auto mode:modes){
            run(reference.input,negative,p,mode,scratch,true,nullptr,false,nullptr,&reference);
            if(r<63)run(reference.input,-negative,p,mode,scratch,false,nullptr,false,nullptr,&reference);
        }
    }
}
void prepared_cases(){
    Team team;sbn::workspace unused;
    const std::int64_t degrees[]{-17,13,std::numeric_limits<std::int64_t>::min()};
    for(std::size_t p:{65u,257u,1025u})for(std::int64_t n:degrees){
        allocation_watch_start();const auto need=floating::power_program_requirements(sbn::bits{p},n,4);
        assert(!allocation_watch_stop());assert(need.supported());
        sbn::workspace storage;storage.reserve(need.scratch_bytes+64,need.scratch_alignment);auto bytes=storage.on().scratch;
        std::fill(bytes.begin()+need.scratch_bytes,bytes.begin()+need.scratch_bytes+64,std::byte{0x51});
        allocation_watch_start();const auto prepared=floating::power_program_prepare(sbn::bits{p},n,4,bytes.first(need.scratch_bytes));
        assert(!allocation_watch_stop());
        if(n==std::numeric_limits<std::int64_t>::min()){
            for(bool below:{false,true}){NearOne reference(63,below,p);
                for(auto mode:modes)run(reference.input,n,p,mode,unused,true,team.value,true,&prepared,&reference);}
        }else{
            for(auto width:{53u,259u,4099u}){const auto input=random_input(width,0,width%2);
                for(auto mode:modes)run(input,n,p,mode,unused,true,team.value,true,&prepared);}
        }
        for(std::size_t j=0;j<64;++j)assert(bytes[need.scratch_bytes+j]==std::byte{0x51});
    }
}
void specials(){
    alignas(64)std::array<std::uint64_t,3> data{};sbn::float_buffer out{data,sbn::bits{129}};
    const std::uint64_t one=1,two=2,three=3;
    for(auto mode:modes){
        for(auto x:{sbn::float_view::nan(),sbn::float_view::infinity(),sbn::float_view::infinity(true),sbn::float_view{}}){
            const auto report=sbn::powi_into(out.ref(),x,0,{},mode);
            assert(out.view().finite()&&!out.view().negative()&&out.view().magnitude().size()==1&&
                   out.view().magnitude().data()[0]==1&&out.view().exponent2()==0&&report.error.exact());
        }
        for(bool sign:{false,true})for(std::int64_t n:{-3,-2,2,3}){
            const auto zero=sbn::float_view::from_magnitude({},0,sign);
            auto report=sbn::powi_into(out.ref(),zero,n,{},mode);
            assert(out.view().negative()==(sign&&(magnitude(n)&1)));
            if(n<0)assert(!out.view().finite()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
            else assert(out.view().zero()&&report.error.exact());
            report=sbn::powi_into(out.ref(),sbn::float_view::infinity(sign),n,{},mode);
            assert(out.view().negative()==(sign&&(magnitude(n)&1)));
            assert(n<0?out.view().zero():!out.view().finite());
        }
        auto report=sbn::powi_into(out.ref(),sbn::float_view::from_magnitude({&one,1},0,true),
            std::numeric_limits<std::int64_t>::min(),{},mode);
        assert(!out.view().negative()&&out.view().magnitude().data()[0]==1&&report.error.exact());
        report=sbn::powi_into(out.ref(),sbn::float_view::from_magnitude({&two,1},0),
            std::numeric_limits<std::int64_t>::min(),{},mode);
        assert(out.view().zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
        assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2==std::numeric_limits<std::int64_t>::min());
        report=sbn::powi_into(out.ref(),sbn::float_view::nan(),-2,{},mode);assert(out.view().is_nan());
    }
    // Non-power-of-two huge exponents take the numerical wide-exponent path.
    sbn::workspace scratch;
    for(auto n:{std::numeric_limits<std::int64_t>::min(),std::numeric_limits<std::int64_t>::max()}){
        const auto input=sbn::float_view::from_magnitude({&three,1},0);
        const auto need=sbn::powi_requirements(out.ref(),input,n);assert(need.supported());
        scratch.reserve(need.scratch_bytes,need.scratch_alignment);
        allocation_watch_start();const auto report=sbn::powi_into(out.ref(),input,n,scratch.on());assert(!allocation_watch_stop());
        if(n<0){assert(out.view().zero()&&report.error.kind==sbn::error_kind::finite);
            assert(report.error.exponent2==std::numeric_limits<std::int64_t>::min());}
        else assert(out.view().classification()==sbn::float_class::infinity&&sbn::has_flag(report.flags,sbn::float_flag::overflow));
    }
}
void wide_unsigned_cases(){
    const std::uint64_t degrees[]{1,3,UINT64_C(1)<<63,UINT64_MAX-1,UINT64_MAX};
    const std::uint64_t one=1,three=3;
    for(std::size_t p:{17u,65u,257u})for(auto degree:degrees){
        allocation_watch_start();const auto need=floating::wide_power_program_requirements(sbn::bits{p},degree,1);
        assert(!allocation_watch_stop()&&need.supported());
        sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);
        allocation_watch_start();const auto prepared=floating::wide_power_program_prepare(sbn::bits{p},degree,1,
            storage.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
        for(std::int64_t e:{-2,-1,0,1,2}){
            allocation_watch_start();const auto result=floating::wide_power_program_execute(prepared,
                sbn::float_view::from_magnitude({&one,1},e));assert(!allocation_watch_stop());
            assert(sbn::compare(result.mantissa,sbn::float_view::from_magnitude({&one,1},-1))==std::partial_ordering::equivalent);
            assert(result.top_exponent==Wide(e)*degree+1);
            assert(result.relative_error.publish().exponent2<=-std::int64_t(p)-12);
        }
        allocation_watch_start();const auto result=floating::wide_power_program_execute(prepared,
            sbn::float_view::from_magnitude({&three,1},-1));assert(!allocation_watch_stop());
        assert(result.top_exponent>Wide(degree)/2&&result.top_exponent<=Wide(degree)+1);
        assert(!result.mantissa.zero()&&result.mantissa.finite()&&!result.mantissa.negative());
    }
    assert(!floating::wide_power_program_requirements(sbn::bits{65},0,1).supported());
}
}
int main(){
    specials();exact_cases();huge_near_one();prepared_cases();wide_unsigned_cases();
    std::printf("float powi: %zu independent exact and %zu rigorous binomial certificates (%zu prepared reuse); range, aliases, scopes and allocation gates PASS\n",
        exact_checks,analytic_checks,prepared_checks);
}
