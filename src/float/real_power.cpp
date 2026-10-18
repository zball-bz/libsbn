// Original libsbn code; LGPL-3.0-or-later.
#include "float/real_power.hpp"
#include "float/constants.hpp"
#include "float/logarithm.hpp"
#include "float/exponential.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3352505731);
struct IntegerInfo {bool integral=false,odd=false,small=false;std::int64_t value=0;};
IntegerInfo integer_info(float_view y) noexcept {
    IntegerInfo info;if(!y.finite())return info;
    if(y.zero()){info.integral=info.small=true;return info;}
    const auto width=magnitude_bits(y.magnitude());const auto exponent=y.exponent2();
    if(exponent>=0){info.integral=true;info.odd=!exponent&&(y.magnitude().data()[0]&1);}
    else{
        const auto cut=std::uint64_t(-__int128(exponent));if(cut>=width)return info;
        const auto whole=std::size_t(cut/64);for(std::size_t j=0;j<whole;++j)if(y.magnitude().data()[j])return info;
        const auto shift=unsigned(cut%64);
        if(shift&&(y.magnitude().data()[whole]&((UINT64_C(1)<<shift)-1)))return info;
        info.integral=true;info.odd=(y.magnitude().data()[whole]>>shift)&1;
    }
    if(real_top(y)>64)return info;
    std::uint64_t word=0;integer_buffer value({&word,1});const auto report=to_integer_into(value.ref(),y,rounding::toward_zero);
    require(report.error.exact()&&value.size()<=1,SBN3_FATAL_MATH,"real power integer classifier");
    if(y.negative()){
        if(word>(UINT64_C(1)<<63))return info;
        info.value=word==(UINT64_C(1)<<63)?INT64_MIN:-std::int64_t(word);
    }else{if(word>std::uint64_t(INT64_MAX))return info;info.value=std::int64_t(word);}
    info.small=true;return info;
}
float_view absolute(float_view x) noexcept{return x.negative()?x.negated():x;}
bool special_case(float_ref out,float_view x,float_view y,const IntegerInfo& yi,rounding mode,float_report& report) noexcept {
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    // C-like precedence: x^0 and +1^y are 1 even with a NaN argument.
    if(y.zero()||(x.finite()&&!x.negative()&&compare(x,one)==std::partial_ordering::equivalent)){
        report=round_into(out,one,mode);return true;}
    if(x.is_nan()||y.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(!y.finite()){
        const auto order=compare(absolute(x),one);
        if(order==std::partial_ordering::equivalent){report=round_into(out,one,mode);return true;}
        const bool infinity=(order==std::partial_ordering::greater)!=y.negative();
        report=special(out,infinity?float_class::infinity:float_class::finite,false);return true;
    }
    const bool negative=x.negative()&&yi.integral&&yi.odd;
    if(x.zero()){
        report=y.negative()?special(out,float_class::infinity,negative,float_flag::divide_by_zero):special(out,float_class::finite,negative);return true;}
    if(!x.finite()){
        report=special(out,y.negative()?float_class::finite:float_class::infinity,negative);return true;}
    if(x.negative()&&!yi.integral){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
    if(compare(absolute(x),one)==std::partial_ordering::equivalent){report=round_into(out,negative?one.negated():one,mode);return true;}
    return false;
}
bool simple(float_view x,float_view y,const IntegerInfo& yi) noexcept {
    if(!x.finite()||!y.finite()||x.zero()||y.zero()||(x.negative()&&!yi.integral))return true;
    const std::uint64_t one=1;return compare(absolute(x),float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent;
}
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;unsigned workers=1;
    std::size_t precision=0,working=0,bytes=0;
    RealArithmetic arithmetic{};ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};FloatLogarithmPrepared logarithm{};FloatExponentialPrepared exponential{};
    float_buffer* v[5]{};
};
struct Quote {requirements need{},logarithm{},exponential{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};std::size_t working=0,limbs=0;};
Quote quote(bits p,unsigned workers,ConstantBankBinding constants={}) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit||!workers||workers>32||
       !mul_size(p.count,2,q.working)||!add_size(q.working,256,q.working)||q.working>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="real power precision/workers";return q;}
    q.limbs=(q.working+63)/64;q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    const auto demand=real_power_constant_demand(p);
    if(constants.supplied()){
        if(!constant_bank_supports(constants,demand)){q.need.status=support::unsupported;q.need.reason="real power constants";return q;}
    }else{
        q.bank=constant_bank_requirements(demand,workers);
        if(!q.bank.supported()){q.need.status=q.bank.status;q.need.reason=q.bank.reason;return q;}
        constants=constant_bank_query_binding({q.bank.pi_bits,q.bank.log2_bits});
    }
    q.logarithm=logarithm_program_requirements(bits{q.working},workers,constants);
    q.exponential=exponential_program_requirements(bits{q.working},workers,constants);
    if(!q.arithmetic.need.supported()||!q.logarithm.supported()||!q.exponential.supported()){
        q.need.status=support::unsupported;q.need.reason="real power dependencies";return q;}
    q.need.scratch_alignment=std::max({q.arithmetic.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment,q.logarithm.scratch_alignment,q.exponential.scratch_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128);
    for(unsigned i=0;i<5;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment)&&
        real_append(n,q.logarithm.scratch_bytes,q.logarithm.scratch_alignment)&&real_append(n,q.exponential.scratch_bytes,q.exponential.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="real power storage bytes";}return q;
}
bool below(ErrorMagnitude radius,std::int64_t exponent) noexcept {const auto b=radius.publish();return b.exact()||(b.kind==error_kind::finite&&b.exponent2<=exponent);}
std::int64_t narrow(__int128 value) noexcept {require(value>=INT64_MIN&&value<=INT64_MAX,SBN3_FATAL_SIZE,"real power exponent");return std::int64_t(value);}
ErrorMagnitude prefix(float_ref out,float_view input) noexcept {
    const auto width=magnitude_bits(input.magnitude()),p=out.precision().count;
    const auto cut=width>p?width-p:0,kept=width-cut,n=(kept+63)/64,first=cut/64;const auto shift=unsigned(cut%64);
    auto* target=detail::float_access::begin(out);for(std::size_t j=0;j<n;++j){const auto at=first+j;
        target->data[j]=(input.magnitude().data()[at]>>shift)|(shift&&at+1<input.magnitude().size()?input.magnitude().data()[at+1]<<(64-shift):0);}
    target->size=n;target->negative=0;auto& m=detail::float_access::metadata(out);
    m.exponent2=narrow(__int128(input.exponent2())+cut);m.negative=input.negative();m.classification=float_class::finite;detail::float_access::finish(out);
    return cut?ErrorMagnitude::power_of_two(m.exponent2):ErrorMagnitude{};
}
struct Call {State* s;float_ref out;float_view x,y;bool negative;rounding mode;float_report report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;const execution ex{{},nullptr,scope};
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    if(s.owned_bank.state)s.constants=constant_bank_execute(s.owned_bank,ex);
    // Do not stage x before this call: the original near-one difference
    // may be far below W bits but amplified by a large y.
    const auto logarithm=logarithm_program_execute(s.logarithm,ref(0),absolute(call.x),LogarithmKind::log,ex);
    const auto log_radius=ErrorMagnitude::from(logarithm.error);
    require(val(0).finite()&&!val(0).zero()&&below(log_radius,real_top(val(0))-std::int64_t(s.working)+2),
        SBN3_FATAL_MATH,"real power logarithm radius");
    const auto sum=narrow(__int128(real_top(call.y))+real_top(val(0)));
    const bool negative_argument=call.y.negative()!=val(0).negative();
    // Exact |y|>=2^(Ty-1), true |log(x)|>=2^(TL-2). A sum>=64
    // therefore proves |Z|>=2^61 before a wide product is attempted.
    if(sum>=64){
        if(!negative_argument)call.report=special(call.out,float_class::infinity,call.negative,float_flag::overflow|float_flag::rounded);
        else{call.report=special(call.out,float_class::finite,call.negative,float_flag::underflow|float_flag::rounded);
            call.report.error=error_bound::power_of_two(-float_exponent_limit-1);}
        return;
    }
    if(sum+1<=-std::int64_t(s.precision)-20){
        const std::uint64_t one=1;const auto center=float_view::from_magnitude({&one,1},0,call.negative);
        // |Z|<=2^(Ty+TL+1); |exp(Z)-1|<=2|Z| in this tiny domain.
        call.report=real_finish(call.out,center,ErrorMagnitude::power_of_two(sum+2),call.mode);return;
    }
    const auto y_radius=prefix(ref(1),call.y);
    const auto product=real_mul(s.arithmetic,ref(2),val(1),val(0),ex);
    const auto z_radius=real_upper(val(1)).multiply(log_radius).add(real_upper(val(0)).multiply(y_radius))
        .add(log_radius.multiply(y_radius)).add(ErrorMagnitude::from(product.error));
    require(val(2).finite()&&!val(2).zero()&&below(z_radius,-std::int64_t(s.precision)-32),
        SBN3_FATAL_MATH,"real power moderate product radius");
    const auto exponential=exponential_program_execute(s.exponential,ref(3),val(2),ExponentialKind::exp,ex);
    if(!val(3).finite()){
        require(!val(3).is_nan(),SBN3_FATAL_MATH,"real power exponential domain");
        call.report=special(call.out,float_class::infinity,call.negative,exponential.flags);return;
    }
    const auto exp_radius=ErrorMagnitude::from(exponential.error);
    // For rZ<=1/2, exp changes by at most 2*exp(centerZ)*rZ.
    // exp(centerZ)<=|computed center|+rExp, including a zero underflow.
    const auto radius=exp_radius.add(real_upper(val(3)).add(exp_radius).multiply(z_radius).scale2(1));
    const auto point=call.negative?val(3).negated():val(3);
    call.report=real_finish(call.out,point,radius,call.mode);
    if(has_flag(exponential.flags,float_flag::underflow))call.report.flags=call.report.flags|float_flag::underflow|float_flag::rounded;
    const auto output=call.out.view();if(output.finite()&&!output.zero()){
        const auto limit=real_top(output)-std::int64_t(s.precision)+(call.mode==rounding::nearest_even?0:1);
        require(call.report.error.kind!=error_kind::unbounded&&(call.report.error.exact()||call.report.error.exponent2<=limit),
            SBN3_FATAL_MATH,"real power final radius");}
}
}
ConstantBankDemand real_power_constant_demand(bits p) noexcept{
    std::size_t working=0;
    if(!p.count||!mul_size(p.count,2,working)||!add_size(working,256,working)||working>float_precision_limit)return {};
    return constant_bank_merge(logarithm_constant_demand(bits{working}),exponential_constant_demand(bits{working}));
}
requirements real_power_program_requirements(bits p,unsigned workers,ConstantBankBinding constants) noexcept{return quote(p,workers,constants).need;}
FloatRealPowerPrepared real_power_program_prepare(bits precision,unsigned workers,std::span<std::byte> storage,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"real power bound constants");
    const auto q=quote(precision,workers,constants);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),SBN3_FATAL_WORKSPACE,"real power storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,constants.values().storage.data(),constants.values().storage.size()),
        SBN3_FATAL_ARGUMENT,"real power constant storage alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->precision=precision.count;s->working=q.working;s->bytes=q.need.scratch_bytes;s->workers=workers;
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);v=::new(raw)float_buffer({data,q.limbs},bits{q.working});}
    {auto sub=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);real_arithmetic_prepare(s->arithmetic,q.arithmetic,sub);}
    if(!constants.supplied()){
        auto* values=static_cast<std::byte*>(frame.allocate(q.bank.value_bytes,q.bank.value_alignment));
        auto* work=static_cast<std::byte*>(frame.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment));
        s->owned_bank=constant_bank_prepare(q.bank,{values,q.bank.value_bytes},{work,q.bank.workspace_bytes});
        constants=constant_bank_pending_binding(s->owned_bank);
    }
    s->constants=constants;
    auto* log=static_cast<std::byte*>(frame.allocate(q.logarithm.scratch_bytes,q.logarithm.scratch_alignment));
    s->logarithm=logarithm_program_prepare(bits{q.working},workers,{log,q.logarithm.scratch_bytes},constants);
    auto* exp=static_cast<std::byte*>(frame.allocate(q.exponential.scratch_bytes,q.exponential.scratch_alignment));
    s->exponential=exponential_program_prepare(bits{q.working},workers,{exp,q.exponential.scratch_bytes},constants);
    s->marker=magic;return {s,storage.first(q.need.scratch_bytes)};
}
float_report real_power_program_execute(const FloatRealPowerPrepared& prepared,float_ref out,float_view x,float_view y,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"real power identity");
    auto& s=*static_cast<State*>(prepared.state);require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"real power prepared shape");validate(x);validate(y);check_output(out,x,y);constant_bank_check_output(s.constants,out);
    const auto* dest=detail::float_access::begin(out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),dest->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"real power output storage alias");
    for(auto value:{x,y})require(!overlaps(prepared.storage.data(),prepared.storage.size(),value.magnitude().data(),value.magnitude().size()*8),SBN3_FATAL_ARGUMENT,"real power input storage alias");
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"real power concurrent use");
    const auto yi=integer_info(y);Call call{&s,out,x,y,x.negative()&&yi.odd,mode,{}};
    if(!special_case(out,x,y,yi,mode,call.report)){
        if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);}
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
requirements real_power_requirements(float_ref out,float_view x,float_view y,unsigned workers,ConstantBankBinding constants) noexcept {
    const auto p=out.precision().count;if(!p||p>float_precision_limit||!workers||workers>32)return {support::unsupported,0,128,0,"real power shape"};
    const auto limbs=(p+63)/64;if(out.capacity()<limbs)return {support::capacity,0,128,limbs,"real power capacity"};
    const auto yi=integer_info(y);if(simple(x,y,yi))return {support::yes,0,128,limbs,""};
    if(yi.small)return powi_requirements(out,x,yi.value,workers);
    return real_power_program_requirements(out.precision(),workers,constants);
}
float_report real_power_into(float_ref out,float_view x,float_view y,execution ex,rounding mode,ConstantBankBinding constants) noexcept {
    constant_bank_check_output(constants,out);validate(x);validate(y);check_output(out,x,y);const auto yi=integer_info(y);float_report report;
    if(special_case(out,x,y,yi,mode,report))return report;
    if(yi.small)return powi_into(out,x,yi.value,ex,mode);
    const auto need=real_power_requirements(out,x,y,ex.workers(),constants);require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    require(ex.scratch.size()>=need.scratch_bytes,SBN3_FATAL_WORKSPACE,"real power fresh workspace",need.scratch_bytes,ex.scratch.size());
    const auto prepared=real_power_program_prepare(out.precision(),ex.workers(),ex.scratch.first(need.scratch_bytes),constants);
    return real_power_program_execute(prepared,out,x,y,ex,mode);
}
}
namespace sbn {
requirements pow_requirements(float_ref out,float_view x,float_view y,unsigned workers) noexcept{return v3::floating::real_power_requirements(out,x,y,workers);}
float_report pow_into(float_ref out,float_view x,float_view y,execution ex,rounding mode) noexcept{return v3::floating::real_power_into(out,x,y,ex,mode);}
}
