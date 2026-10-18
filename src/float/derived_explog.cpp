// Original libsbn code; LGPL-3.0-or-later.
#include "float/derived_explog.hpp"
#include "float/constants.hpp"
#include "float/logarithm.hpp"
#include "float/exponential.hpp"
#include "float/exp_series.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3344455231);
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;unsigned workers=1;
    std::size_t precision=0,working=0,bytes=0;bool have_log2=false,have_log10=false,use_series=false;
    DerivedExpLogKind kind=DerivedExpLogKind::exp2;
    RealArithmetic arithmetic{};ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};
    FloatLogarithmPrepared logarithm{};FloatExponentialPrepared exponential{};
    FloatExpSeriesPrepared series{};
    ErrorMagnitude log2_radius{},log10_radius{};float_buffer* v[8]{};
};
struct Quote {requirements need{},logarithm{},exponential{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};std::size_t working=0,limbs=0;};
bool valid(DerivedExpLogKind kind) noexcept{return kind==DerivedExpLogKind::exp2||kind==DerivedExpLogKind::log2||kind==DerivedExpLogKind::log10;}
Quote quote(bits p,unsigned workers,DerivedExpLogKind kind,ConstantBankBinding constants={}) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit||!workers||workers>32||!valid(kind)||
       !add_size(p.count,128,q.working)||q.working>float_precision_limit){q.need.status=support::unsupported;q.need.reason="derived exp/log shape";return q;}
    q.limbs=(q.working+63)/64;q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    const auto demand=derived_explog_constant_demand(p,kind);
    if(constants.supplied()){
        if(!constant_bank_supports(constants,demand)){q.need.status=support::unsupported;q.need.reason="derived exp/log constants";return q;}
    }else{
        q.bank=constant_bank_requirements(demand,workers);
        if(!q.bank.supported()){q.need.status=q.bank.status;q.need.reason=q.bank.reason;return q;}
        constants=constant_bank_query_binding({q.bank.pi_bits,q.bank.log2_bits});
    }
    if(kind!=DerivedExpLogKind::exp2)q.logarithm=logarithm_program_requirements(bits{q.working},workers,constants);
    if(kind==DerivedExpLogKind::exp2)q.exponential=q.working<=4096?
        exp_series_program_requirements(bits{q.working},ExpSeriesKind::exp,false,workers):
        exponential_program_requirements(bits{q.working},workers,constants);
    if(!q.arithmetic.need.supported()||!q.bank.supported()||!q.logarithm.supported()||!q.exponential.supported()){
        q.need.status=support::unsupported;q.need.reason="derived exp/log dependencies";return q;}
    q.need.scratch_alignment=std::max({q.arithmetic.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment,
        q.logarithm.scratch_alignment,q.exponential.scratch_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128);
    for(unsigned i=0;i<8;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment)&&real_append(n,q.logarithm.scratch_bytes,q.logarithm.scratch_alignment)&&
        real_append(n,q.exponential.scratch_bytes,q.exponential.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="derived exp/log bytes";}return q;
}
bool power_two(float_view x,std::int64_t& exponent) noexcept {
    if(!x.finite()||x.zero()||x.negative())return false;
    bool found=false;
    for(std::size_t j=0;j<x.magnitude().size();++j){const auto word=x.magnitude().data()[j];if(!word)continue;
        if(found||!std::has_single_bit(word))return false;found=true;
        exponent=x.exponent2()+std::int64_t(64*j)+std::countr_zero(word);}
    return found;
}
bool integer64(float_view x,std::int64_t& value) noexcept {
    std::uint64_t word=0;integer_buffer out({&word,1});const auto report=to_integer_into(out.ref(),x,rounding::downward);
    require(out.size()<=1&&word<=std::uint64_t(INT64_MAX),SBN3_FATAL_SIZE,"exp2 reduction integer");
    value=out.negative()?-std::int64_t(word):std::int64_t(word);return report.error.exact();
}
float_report exponent_value(float_ref out,std::int64_t exponent,rounding mode) noexcept {
    const auto magnitude=exponent<0?std::uint64_t(0)-std::uint64_t(exponent):std::uint64_t(exponent);
    return round_magnitude(out,integer_view{{&magnitude,1}},0,exponent<0,mode);
}
bool trivial(float_ref out,float_view x,DerivedExpLogKind kind,rounding mode,float_report& report) noexcept {
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(kind==DerivedExpLogKind::exp2){
        if(x.zero()){report=round_into(out,one,mode);return true;}
        if(!x.finite()){
            report=x.negative()?special(out,float_class::finite,false):special(out,float_class::infinity,false);return true;}
        const auto top=real_top(x);
        if(top>=62){
            if(x.negative()){report=special(out,float_class::finite,false,float_flag::underflow|float_flag::rounded);
                report.error=error_bound::power_of_two(-float_exponent_limit-1);}
            else report=special(out,float_class::infinity,false,float_flag::overflow|float_flag::rounded);
            return true;
        }
        if(top<=-std::int64_t(out.precision().count)-18){report=real_finish(out,one,ErrorMagnitude::power_of_two(top+1),mode);return true;}
        std::int64_t k=0;if(integer64(x,k)){report=round_magnitude(out,integer_view{{&word,1}},k,false,mode);return true;}
    }else{
        if(x.zero()){report=special(out,float_class::infinity,true,float_flag::divide_by_zero);return true;}
        if(x.negative()){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(!x.finite()){report=special(out,float_class::infinity,false);return true;}
        std::int64_t exponent=0;
        if(kind==DerivedExpLogKind::log2&&power_two(x,exponent)){report=exponent_value(out,exponent,mode);return true;}
        if(compare(x,one)==std::partial_ordering::equivalent){report=special(out,float_class::finite,false);return true;}
    }
    return false;
}
bool trivial_shape(float_view x,DerivedExpLogKind kind,std::size_t precision) noexcept {
    if(!x.finite()||x.zero())return true;
    if(kind==DerivedExpLogKind::exp2){
        const auto top=real_top(x);if(top>=62||top<=-std::int64_t(precision)-18)return true;
        std::int64_t value=0;return integer64(x,value);
    }
    if(x.negative())return true;std::int64_t exponent=0;
    if(kind==DerivedExpLogKind::log2&&power_two(x,exponent))return true;
    const std::uint64_t one=1;return compare(x,float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent;
}
bool below(ErrorMagnitude radius,std::int64_t exponent) noexcept {
    const auto b=radius.publish();return b.exact()||(b.kind==error_kind::finite&&b.exponent2<=exponent);
}
ErrorMagnitude quotient_radius(float_view numerator,ErrorMagnitude rn,float_view denominator,ErrorMagnitude rd,error_bound local) noexcept {
    const auto t=real_top(denominator);require(below(rd,t-2),SBN3_FATAL_MATH,"derived logarithm denominator");
    return rn.scale2(2-t).add(real_upper(numerator).multiply(rd).scale2(3-2*t)).add(ErrorMagnitude::from(local));
}
struct Call {State* s;float_ref out;float_view input;DerivedExpLogKind kind;rounding mode;float_report report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;auto& ar=s.arithmetic;const execution ex{{},nullptr,scope};
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    if(s.owned_bank.state)s.constants=constant_bank_execute(s.owned_bank,ex);
    if(call.kind!=DerivedExpLogKind::log10&&!s.have_log2){
        s.log2_radius=ErrorMagnitude::from(constant_bank_prefix(s.constants,ConstantKind::log2,ref(6)).error);s.have_log2=true;}
    if(call.kind==DerivedExpLogKind::exp2){
        std::int64_t k=0;integer64(call.input,k);
        const std::uint64_t magnitude=k<0?std::uint64_t(0)-std::uint64_t(k):std::uint64_t(k);
        const auto integer=float_view::from_magnitude({&magnitude,1},0,k<0);
        // Exact cancellation of the original input's integer part precedes
        // every prefix/product operation. Tiny fractions survive tall inputs.
        auto report=real_add(ar,ref(0),call.input,integer.negated());
        const auto fraction_radius=ErrorMagnitude::from(report.error);
        report=real_mul(ar,ref(1),val(0),val(6),ex);
        const auto argument_radius=real_upper(val(0)).multiply(s.log2_radius)
            .add(real_upper(val(6)).multiply(fraction_radius)).add(fraction_radius.multiply(s.log2_radius))
            .add(ErrorMagnitude::from(report.error));
        report=s.use_series?exp_series_program_execute(s.series,ref(2),val(1),ex):
            exponential_program_execute(s.exponential,ref(2),val(1),ExponentialKind::exp,ex);
        // floor reduction gives fraction in [0,1). Both the exact and
        // rounded log2-scaled arguments are below1, so derivative<4.
        const auto radius=ErrorMagnitude::from(report.error).add(argument_radius.scale2(2)).scale2(k);
        const auto scaled=real_scaled(val(2),k);
        call.report=round_magnitude(call.out,scaled.magnitude(),scaled.exponent2(),false,call.mode);
        call.report.error=ErrorMagnitude::from(call.report.error).add(radius).publish();
    }else{
        if(call.kind==DerivedExpLogKind::log10&&!s.have_log10){
            const std::uint64_t ten=10;const auto value=float_view::from_magnitude({&ten,1},0);
            s.log10_radius=ErrorMagnitude::from(logarithm_program_execute(s.logarithm,ref(7),value,LogarithmKind::log,ex).error);
            s.have_log10=true;
        }
        // Always evaluate log on the ORIGINAL exact input. Staging x first
        // would destroy x=1+/-epsilon even though its logarithm is nonzero.
        auto report=logarithm_program_execute(s.logarithm,ref(0),call.input,LogarithmKind::log,ex);
        const auto numerator_radius=ErrorMagnitude::from(report.error);
        const auto denominator=call.kind==DerivedExpLogKind::log2?val(6):val(7);
        const auto denominator_radius=call.kind==DerivedExpLogKind::log2?s.log2_radius:s.log10_radius;
        report=real_divide(ar,ref(1),val(0),denominator,ex);
        const auto radius=quotient_radius(val(0),numerator_radius,denominator,denominator_radius,report.error);
        call.report=real_finish(call.out,val(1),radius,call.mode);
    }
    const auto result=call.out.view();if(result.finite()&&!result.zero()){
        const auto limit=real_top(result)-std::int64_t(s.precision)+(call.mode==rounding::nearest_even?0:1);
        require(call.report.error.kind!=error_kind::unbounded&&(call.report.error.exact()||call.report.error.exponent2<=limit),
            SBN3_FATAL_MATH,"derived exp/log final radius");}
}
}
ConstantBankDemand derived_explog_constant_demand(bits p,DerivedExpLogKind kind) noexcept{
    std::size_t working=0;
    if(!p.count||!valid(kind)||!add_size(p.count,128,working)||working>float_precision_limit)return {};
    ConstantBankDemand demand{0,kind==DerivedExpLogKind::log10?0:working};
    if(kind!=DerivedExpLogKind::exp2)demand=constant_bank_merge(demand,logarithm_constant_demand(bits{working}));
    else if(working>4096)demand=constant_bank_merge(demand,exponential_constant_demand(bits{working}));
    return demand;
}
requirements derived_explog_program_requirements(bits p,unsigned workers,DerivedExpLogKind kind,ConstantBankBinding constants) noexcept{return quote(p,workers,kind,constants).need;}
FloatDerivedExpLogPrepared derived_explog_program_prepare(bits precision,unsigned workers,DerivedExpLogKind kind,std::span<std::byte> storage,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"derived exp/log bound constants");
    const auto q=quote(precision,workers,kind,constants);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"derived exp/log storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,constants.values().storage.data(),constants.values().storage.size()),
        SBN3_FATAL_ARGUMENT,"derived exp/log constant storage alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->precision=precision.count;s->working=q.working;s->workers=workers;s->bytes=q.need.scratch_bytes;s->kind=kind;
    s->use_series=kind==DerivedExpLogKind::exp2&&q.working<=4096;
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);v=::new(raw)float_buffer({data,q.limbs},bits{q.working});}
    {auto sub=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);real_arithmetic_prepare(s->arithmetic,q.arithmetic,sub);}
    auto take=[&](requirements need){return std::span<std::byte>{static_cast<std::byte*>(frame.allocate(need.scratch_bytes,need.scratch_alignment)),need.scratch_bytes};};
    if(!constants.supplied()){
        auto* values=static_cast<std::byte*>(frame.allocate(q.bank.value_bytes,q.bank.value_alignment));
        auto* work=static_cast<std::byte*>(frame.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment));
        s->owned_bank=constant_bank_prepare(q.bank,{values,q.bank.value_bytes},{work,q.bank.workspace_bytes});
        constants=constant_bank_pending_binding(s->owned_bank);
    }
    s->constants=constants;
    if(kind!=DerivedExpLogKind::exp2)s->logarithm=logarithm_program_prepare(bits{q.working},workers,take(q.logarithm),constants);
    if(kind==DerivedExpLogKind::exp2){
        if(s->use_series)s->series=exp_series_program_prepare(bits{q.working},ExpSeriesKind::exp,false,workers,take(q.exponential));
        else s->exponential=exponential_program_prepare(bits{q.working},workers,take(q.exponential),constants);
    }
    s->marker=magic;return {s,storage.first(q.need.scratch_bytes)};
}
float_report derived_explog_program_execute(const FloatDerivedExpLogPrepared& prepared,float_ref out,float_view input,
    DerivedExpLogKind kind,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"derived exp/log identity");
    auto& s=*static_cast<State*>(prepared.state);require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&
        ex.workers()>=s.workers&&valid(kind)&&kind==s.kind,SBN3_FATAL_ARGUMENT,"derived exp/log program");
    validate(input);check_output(out,input);constant_bank_check_output(s.constants,out);const auto* dest=detail::float_access::begin(out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),dest->data,out.capacity()*8)&&
        !overlaps(prepared.storage.data(),prepared.storage.size(),input.magnitude().data(),input.magnitude().size()*8),SBN3_FATAL_ARGUMENT,"derived exp/log alias");
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"derived exp/log concurrent use");
    Call call{&s,out,input,kind,mode,{}};if(!trivial(out,input,kind,mode,call.report)){
        if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);}
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
requirements derived_explog_requirements(float_ref out,float_view input,DerivedExpLogKind kind,unsigned workers,ConstantBankBinding constants) noexcept {
    const auto p=out.precision().count;if(!p||p>float_precision_limit||!workers||workers>32||!valid(kind))return {support::unsupported,0,128,0,"derived exp/log shape"};
    const auto limbs=(p+63)/64;if(out.capacity()<limbs)return {support::capacity,0,128,limbs,"derived exp/log capacity"};
    if(trivial_shape(input,kind,p))return {support::yes,0,128,limbs,""};return derived_explog_program_requirements(out.precision(),workers,kind,constants);
}
float_report derived_explog_into(float_ref out,float_view input,DerivedExpLogKind kind,execution ex,rounding mode,ConstantBankBinding constants) noexcept {
    constant_bank_check_output(constants,out);require(valid(kind),SBN3_FATAL_ARGUMENT,"derived exp/log kind");validate(input);check_output(out,input);float_report report;
    if(trivial(out,input,kind,mode,report))return report;
    const auto need=derived_explog_requirements(out,input,kind,ex.workers(),constants);require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    require(ex.scratch.size()>=need.scratch_bytes,SBN3_FATAL_WORKSPACE,"derived exp/log fresh workspace",need.scratch_bytes,ex.scratch.size());
    const auto prepared=derived_explog_program_prepare(out.precision(),ex.workers(),kind,ex.scratch.first(need.scratch_bytes),constants);
    return derived_explog_program_execute(prepared,out,input,kind,ex,mode);
}
}
namespace sbn {
requirements exp2_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::derived_explog_requirements(out,x,v3::floating::DerivedExpLogKind::exp2,workers);}
requirements log2_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::derived_explog_requirements(out,x,v3::floating::DerivedExpLogKind::log2,workers);}
requirements log10_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::derived_explog_requirements(out,x,v3::floating::DerivedExpLogKind::log10,workers);}
float_report exp2_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::derived_explog_into(out,x,v3::floating::DerivedExpLogKind::exp2,ex,mode);}
float_report log2_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::derived_explog_into(out,x,v3::floating::DerivedExpLogKind::log2,ex,mode);}
float_report log10_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::derived_explog_into(out,x,v3::floating::DerivedExpLogKind::log10,ex,mode);}
}
