// Original libsbn code; LGPL-3.0-or-later.
#include "float/logarithm.hpp"
#include "float/diagnostics.hpp"
#include "float/constants.hpp"
#include "float/agm.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
ConstantBankDemand logarithm_constant_demand(bits p) noexcept {
    std::size_t q=0;if(!p.count||p.count>float_precision_limit||!mul_size(p.count,2,q)||!add_size(q,256,q)||q>float_precision_limit)return {};
    return {q,q};
}
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e334c4f4731);
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;
    std::size_t precision=0,working=0,bytes=0,shift=0,series_limit=0;
    unsigned workers=1;bool constants_ready=false;
    RealArithmetic arithmetic{};ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};FloatAgmPrepared agm{};
    ErrorMagnitude pi_radius{},log2_radius{};float_buffer* v[16]{};
};
struct Quote {requirements need{},agm{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};
    std::size_t working=0,limbs=0;};
Quote quote(bits p,unsigned workers,ConstantBankBinding constants={}) noexcept {
    Quote q;q.need.output_limbs=(p.count+63)/64;q.need.scratch_alignment=128;
    if(!p.count||p.count>float_precision_limit||!workers||workers>32||
       !mul_size(p.count,2,q.working)||!add_size(q.working,256,q.working)||q.working>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="logarithm precision/workers";return q;}
    q.limbs=(q.working+63)/64;q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    const ConstantBankDemand demand{q.working,q.working};
    if(constants.supplied()){
        if(!constant_bank_supports(constants,demand)){q.need.status=support::unsupported;q.need.reason="logarithm constant binding";return q;}
    }else q.bank=constant_bank_requirements(demand,workers);
    q.agm=agm_program_requirements(bits{q.working},workers);
    if(!q.arithmetic.need.supported()||!q.bank.supported()||!q.agm.supported()){
        q.need.status=support::unsupported;q.need.reason="logarithm dependencies";return q;}
    q.need.scratch_alignment=std::max({q.arithmetic.need.scratch_alignment,q.bank.value_alignment,
        q.bank.workspace_alignment,q.agm.scratch_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128);
    for(unsigned i=0;i<16;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment)&&
        real_append(n,q.agm.scratch_bytes,q.agm.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="logarithm storage bytes";}return q;
}
bool below(ErrorMagnitude x,std::int64_t exponent) noexcept {
    const auto b=x.publish();return b.exact()||(b.kind==error_kind::finite&&b.exponent2<=exponent);
}
std::int64_t narrow(__int128 value) noexcept {
    require(value>=INT64_MIN&&value<=INT64_MAX,SBN3_FATAL_SIZE,"logarithm scale");return std::int64_t(value);
}
// Bounded significant-prefix read. Never scans the omitted tail to discover
// whether it is zero. The returned absolute radius includes that whole tail.
ErrorMagnitude prefix(float_ref out,float_view input) noexcept {
    const auto width=magnitude_bits(input.magnitude()),p=out.precision().count;
    if(!width){special(out,float_class::finite,input.negative());return {};}
    const auto cut=width>p?width-p:0,kept=width-cut,words=(kept+63)/64,first=cut/64;
    const auto shift=unsigned(cut%64);auto* target=detail::float_access::begin(out);
    for(std::size_t j=0;j<words;++j){const auto at=first+j;
        target->data[j]=(input.magnitude().data()[at]>>shift)|
            (shift&&at+1<input.magnitude().size()?input.magnitude().data()[at+1]<<(64-shift):0);}
    target->size=words;target->negative=0;auto& metadata=detail::float_access::metadata(out);
    metadata.exponent2=narrow(__int128(input.exponent2())+cut);metadata.negative=input.negative();
    metadata.classification=float_class::finite;detail::float_access::finish(out);
    return cut?ErrorMagnitude::power_of_two(metadata.exponent2):ErrorMagnitude{};
}
ErrorMagnitude quotient_radius(float_view numerator,ErrorMagnitude rn,float_view denominator,
    ErrorMagnitude rd,error_bound local) noexcept {
    const auto t=real_top(denominator);require(below(rd,t-2),SBN3_FATAL_MATH,"logarithm denominator enclosure");
    return rn.scale2(2-t).add(real_upper(numerator).multiply(rd).scale2(3-2*t)).add(ErrorMagnitude::from(local));
}
bool special_case(float_ref out,float_view x,LogarithmKind kind,float_report& report) noexcept {
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(kind==LogarithmKind::log){
        if(x.zero()){report=special(out,float_class::infinity,true,float_flag::divide_by_zero);return true;}
        if(x.negative()){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(!x.finite()){report=special(out,float_class::infinity,false);return true;}
        if(compare(x,one)==std::partial_ordering::equivalent){report=special(out,float_class::finite,false);return true;}
    }else{
        if(x.zero()){report=special(out,float_class::finite,x.negative());return true;}
        const auto order=compare(x,one.negated());
        if(order==std::partial_ordering::less){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(order==std::partial_ordering::equivalent){report=special(out,float_class::infinity,true,float_flag::divide_by_zero);return true;}
        if(!x.finite()){report=special(out,float_class::infinity,false);return true;}
    }
    return false;
}
bool special_input(float_view x,LogarithmKind kind) noexcept {
    if(!x.finite()||x.zero())return true;
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    if(kind==LogarithmKind::log)return x.negative()||compare(x,one)==std::partial_ordering::equivalent;
    const auto order=compare(x,one.negated());
    return order==std::partial_ordering::less||order==std::partial_ordering::equivalent;
}
bool input_supported(float_view x) noexcept {
    if(x.magnitude().size()>float_precision_limit/64)return false;
    if(!x.finite()||x.zero())return true;
    const __int128 top=__int128(x.exponent2())+magnitude_bits(x.magnitude());
    return top>=-float_exponent_limit&&top<=float_exponent_limit;
}
struct Call {State* s;float_ref out;float_view input;LogarithmKind kind;rounding mode;float_report report{};};
void finish(Call& call,float_view center,ErrorMagnitude error) noexcept {
    call.report=real_finish(call.out,center,error,call.mode);
    const auto result=call.out.view();
    if(result.finite()&&!result.zero()){
        const auto limit=real_top(result)-std::int64_t(call.s->precision)+(call.mode==rounding::nearest_even?0:1);
        require(call.report.error.kind!=error_kind::unbounded&&
            (call.report.error.exact()||call.report.error.exponent2<=limit),SBN3_FATAL_MATH,"logarithm final radius");
    }
}
// v[0] is a center for an exact delta in [-1/2,1/2]. A tiny delta is
// returned with the scaled delta^2 remainder. Otherwise positive Horner
// handles at most ceil(log2(p+2)) atanh terms, never a p-term fallback.
bool local_log1p(Call& call,ErrorMagnitude delta_radius,execution ex) noexcept {
    auto& s=*call.s;auto& ar=s.arithmetic;const std::uint64_t word=1;
    const auto one=float_view::from_magnitude({&word,1},0);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    const auto delta_upper=real_upper(val(0)).add(delta_radius);
    if(below(delta_upper,-std::int64_t(s.precision)-16)){
        finish(call,val(0),delta_radius.add(delta_upper.multiply(delta_upper)));return true;
    }
    const auto den_report=real_add(ar,ref(1),real_scaled(one,1),val(0));
    const auto den_radius=delta_radius.add(ErrorMagnitude::from(den_report.error));
    const auto division=real_divide(ar,ref(2),val(0),val(1),ex);
    const auto z_radius=quotient_radius(val(0),delta_radius,val(1),den_radius,division.error);
    const auto z_upper=real_upper(val(2)).add(z_radius).publish();
    if(z_upper.kind!=error_kind::finite||z_upper.exponent2>=0)return false;
    const auto ze=z_upper.exponent2;const auto decay=__uint128_t(-__int128(ze))*2;
    const auto wanted=__uint128_t(s.precision)+20;
    const auto terms=std::size_t((wanted+decay-1)/decay);
    if(!terms||terms>s.series_limit)return false;
    real_square(ar,ref(3),val(2),ex);special(ref(4),float_class::finite,false);
    for(std::size_t j=terms;j-->0;){
        const std::uint64_t denominator=2*j+1;
        const auto divisor=float_view::from_magnitude({&denominator,1},0);
        real_divide(ar,ref(5),one,divisor,ex);
        real_mul(ar,ref(6),val(4),val(3),ex);real_add(ar,ref(4),val(5),val(6));
    }
    real_mul(ar,ref(13),real_scaled(val(2),1),val(4),ex);
    // On |z|<=1/3 the truncated atanh polynomial has derivative <3.
    // The positive Horner contraction gives <2^(top(z)+12-working)
    // absolute arithmetic error, including the rounded square and coefficients.
    const auto arithmetic=ErrorMagnitude::power_of_two(real_top(val(2))+12-std::int64_t(s.working));
    const auto tail=ErrorMagnitude::power_of_two(narrow(2+__int128(2*terms+1)*ze));
    finish(call,val(13),z_radius.scale2(2).add(arithmetic).add(tail));return true;
}
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;auto& ar=s.arithmetic;
    const execution ex{{},nullptr,scope};const std::uint64_t word=1,three=3;
    const auto one=float_view::from_magnitude({&word,1},0),half=real_scaled(one,-1);
    const auto one_half=float_view::from_magnitude({&three,1},-1);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    ErrorMagnitude input_radius{};std::int64_t k=0;
    if(call.kind==LogarithmKind::log&&compare(call.input,half)!=std::partial_ordering::less&&
       compare(call.input,one_half)!=std::partial_ordering::greater){
        // Exact finite-sum window reads the source before discarding any
        // cancellation-critical bits. Its output is a relative-accurate delta.
        const auto delta=real_add(ar,ref(0),call.input,one.negated());
        const auto radius=ErrorMagnitude::from(delta.error);
        if(local_log1p(call,radius,ex))return;
        const auto formed=real_add(ar,ref(14),one,val(0));input_radius=radius.add(ErrorMagnitude::from(formed.error));
        k=real_top(val(14))-1;round_into(ref(15),real_scaled(val(14),-k));input_radius=input_radius.scale2(-k);
    }else if(call.kind==LogarithmKind::log1p){
        const bool small=compare(call.input,half)!=std::partial_ordering::greater&&
            compare(call.input,half.negated())!=std::partial_ordering::less;
        if(small){
            const auto radius=prefix(ref(0),call.input);if(local_log1p(call,radius,ex))return;
            const auto formed=real_add(ar,ref(14),one,val(0));input_radius=radius.add(ErrorMagnitude::from(formed.error));
            k=real_top(val(14))-1;round_into(ref(15),real_scaled(val(14),-k));input_radius=input_radius.scale2(-k);
        }else if(call.input.negative()){
            // Near -1, stage the exact cancellation, not a rounded delta
            // which could first become exactly -1.
            const auto formed=real_add(ar,ref(14),one,call.input);input_radius=ErrorMagnitude::from(formed.error);
            k=real_top(val(14))-1;round_into(ref(15),real_scaled(val(14),-k));input_radius=input_radius.scale2(-k);
        }else{
            // Form normalized (1+delta) so the public exponent ceiling
            // cannot clip an otherwise perfectly representable logarithm.
            k=real_top(call.input)-1;input_radius=prefix(ref(14),real_scaled(call.input,-k));
            const auto formed=real_add(ar,ref(15),val(14),real_scaled(one,-k));
            input_radius=input_radius.add(ErrorMagnitude::from(formed.error));
            const auto adjust=real_top(val(15))-1;
            round_into(ref(15),real_scaled(val(15),-adjust));k+=adjust;input_radius=input_radius.scale2(-adjust);
        }
    }else{
        k=real_top(call.input)-1;input_radius=prefix(ref(15),real_scaled(call.input,-k));
    }
    require(val(15).finite()&&!val(15).zero()&&!val(15).negative(),SBN3_FATAL_MATH,"logarithm normalized input");
    if(!s.constants_ready){
        if(s.owned_bank.state)s.constants=constant_bank_execute(s.owned_bank,ex);
        s.pi_radius=ErrorMagnitude::from(constant_bank_prefix(s.constants,ConstantKind::pi,ref(11)).error);
        s.log2_radius=ErrorMagnitude::from(constant_bank_prefix(s.constants,ConstantKind::log2,ref(12)).error);s.constants_ready=true;
    }
    const auto inverse=real_divide(ar,ref(8),one,val(15),ex);
    auto v_radius=ErrorMagnitude::from(inverse.error).scale2(2-std::int64_t(s.shift));
    detail::float_access::metadata(ref(8)).exponent2+=2-std::int64_t(s.shift);detail::float_access::finish(ref(8));
    const auto relative_v=v_radius.scale2(1-real_top(val(8)));
    require(below(relative_v,4-std::int64_t(s.working)),SBN3_FATAL_MATH,"AGM log initializer error");
    const auto agm_report=agm_program_execute(s.agm,ref(7),one,val(8),ex);
    const auto m_radius=ErrorMagnitude::from(agm_report.error).add(real_upper(val(7)).multiply(relative_v).scale2(1));
    const auto divided=real_divide(ar,ref(9),val(11),real_scaled(val(7),1),ex);
    const auto quotient_error=quotient_radius(val(11),s.pi_radius,real_scaled(val(7),1),m_radius.scale2(1),divided.error);
    const auto coefficient=narrow(__int128(k)-s.shift);
    const std::uint64_t magnitude=coefficient<0?std::uint64_t(0)-std::uint64_t(coefficient):std::uint64_t(coefficient);
    const auto integer=float_view::from_magnitude({&magnitude,1},0,coefficient<0);
    const auto multiplied=real_mul(ar,ref(10),val(12),integer,ex);
    const auto scale_error=s.log2_radius.multiply(ErrorMagnitude::from_scaled_uint64(magnitude,0))
        .add(ErrorMagnitude::from(multiplied.error));
    const auto added=real_add(ar,ref(13),val(9),val(10));
    // Singular-limit bias <=(s+1)*2^(5-2s). Input normalization error
    // propagates through log with derivative <=2 on this certified domain.
    const auto bias=ErrorMagnitude::from_scaled_uint64(s.shift+1,5-2*std::int64_t(s.shift));
    finish(call,val(13),quotient_error.add(scale_error).add(ErrorMagnitude::from(added.error))
        .add(input_radius.scale2(1)).add(bias));
}
}
FloatDiagnostic logarithm_diagnostic(bits p,unsigned workers) noexcept {
    const auto q=quote(p,workers);FloatDiagnostic d;d.need=q.need;d.recipe="agm_log_with_local_series";
    d.working_bits=q.working;d.metadata_bytes=sizeof(State);d.constant_value_bytes=q.bank.value_bytes;
    d.constant_work_bytes=q.bank.workspace_bytes;return d;
}
requirements logarithm_program_requirements(bits p,unsigned workers,ConstantBankBinding constants) noexcept{return quote(p,workers,constants).need;}
FloatLogarithmPrepared logarithm_program_prepare(bits precision,unsigned workers,std::span<std::byte> storage,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"logarithm needs bound constants");
    const auto q=quote(precision,workers,constants);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"logarithm prepared storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,constants.values().storage.data(),constants.values().storage.size()),SBN3_FATAL_ARGUMENT,"logarithm constant/work alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->precision=precision.count;s->working=q.working;s->bytes=q.need.scratch_bytes;s->workers=workers;
    s->shift=q.working/2+32;s->series_limit=std::bit_width(precision.count+2);
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);
        v=::new(raw)float_buffer({data,q.limbs},bits{q.working});}
    {auto sub=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);real_arithmetic_prepare(s->arithmetic,q.arithmetic,sub);}
    if(!constants.supplied()){
        auto* values=static_cast<std::byte*>(frame.allocate(q.bank.value_bytes,q.bank.value_alignment));
        auto* work=static_cast<std::byte*>(frame.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment));
        s->owned_bank=constant_bank_prepare(q.bank,{values,q.bank.value_bytes},{work,q.bank.workspace_bytes});constants=constant_bank_pending_binding(s->owned_bank);
    }
    s->constants=constants;
    auto* agm=static_cast<std::byte*>(frame.allocate(q.agm.scratch_bytes,q.agm.scratch_alignment));
    s->agm=agm_program_prepare(bits{q.working},workers,{agm,q.agm.scratch_bytes});s->marker=magic;
    return {s,storage.first(q.need.scratch_bytes)};
}
float_report logarithm_program_execute(const FloatLogarithmPrepared& prepared,float_ref out,float_view input,
    LogarithmKind kind,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"logarithm prepared identity");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers&&
        (kind==LogarithmKind::log||kind==LogarithmKind::log1p),SBN3_FATAL_ARGUMENT,"logarithm prepared shape");
    validate(input);check_output(out,input);const auto* dest=detail::float_access::begin(out);
    constant_bank_check_output(s.constants,out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),dest->data,out.capacity()*8)&&
        !overlaps(prepared.storage.data(),prepared.storage.size(),input.magnitude().data(),input.magnitude().size()*8),
        SBN3_FATAL_ARGUMENT,"logarithm prepared value alias");
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"logarithm concurrent reuse");
    Call call{&s,out,input,kind,mode,{}};
    if(!special_case(out,input,kind,call.report)){
        if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);}
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
requirements logarithm_requirements(float_ref out,float_view input,LogarithmKind kind,unsigned workers,ConstantBankBinding constants) noexcept {
    const auto p=out.precision().count,limbs=p/64+unsigned(p%64!=0);
    if(!p||p>float_precision_limit||!workers||workers>32||
       (kind!=LogarithmKind::log&&kind!=LogarithmKind::log1p))return {support::unsupported,0,128,limbs,"logarithm scalar shape"};
    if(out.capacity()<limbs)return {support::capacity,0,128,limbs,"logarithm output capacity"};
    if(!input_supported(input))return {support::unsupported,0,128,limbs,"logarithm input domain"};
    if(special_input(input,kind))return {support::yes,0,128,limbs,""};
    return logarithm_program_requirements(out.precision(),workers,constants);
}
float_report logarithm_into(float_ref out,float_view input,LogarithmKind kind,execution ex,rounding mode,ConstantBankBinding constants) noexcept {
    require((kind==LogarithmKind::log||kind==LogarithmKind::log1p)&&ex.workers()&&ex.workers()<=32,SBN3_FATAL_ARGUMENT,"logarithm scalar shape");
    validate(input);check_output(out,input);constant_bank_check_output(constants,out);
    float_report report;if(special_case(out,input,kind,report))return report;
    const auto need=logarithm_requirements(out,input,kind,ex.workers(),constants);require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    require(ex.scratch.size()>=need.scratch_bytes,SBN3_FATAL_WORKSPACE,"logarithm fresh workspace",need.scratch_bytes,ex.scratch.size());
    const auto* dest=detail::float_access::begin(out);
    require(!overlaps(ex.scratch.data(),need.scratch_bytes,input.magnitude().data(),input.magnitude().size()*8)&&
        !overlaps(ex.scratch.data(),need.scratch_bytes,dest->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"logarithm fresh value/work alias");
    const auto prepared=logarithm_program_prepare(out.precision(),ex.workers(),ex.scratch.first(need.scratch_bytes),constants);
    return logarithm_program_execute(prepared,out,input,kind,ex,mode);
}
}
namespace sbn {
requirements log_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::logarithm_requirements(out,x,v3::floating::LogarithmKind::log,workers);}
requirements log1p_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::logarithm_requirements(out,x,v3::floating::LogarithmKind::log1p,workers);}
float_report log_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::logarithm_into(out,x,v3::floating::LogarithmKind::log,ex,mode);}
float_report log1p_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::logarithm_into(out,x,v3::floating::LogarithmKind::log1p,ex,mode);}
}
