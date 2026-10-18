// Original libsbn code; LGPL-3.0-or-later.
#include "float/elliptic.hpp"
#include "float/diagnostics.hpp"
#include "float/constants.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <new>
#include <utility>

namespace sbn::v3::floating {
ConstantBankDemand elliptic_constant_demand(bits p) noexcept {
    std::size_t q=0;if(!p.count||p.count>float_precision_limit||!add_size(p.count,256,q)||q>float_precision_limit)return {};return {q,0};
}
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e33454c4c31);
constexpr unsigned order=8;
// c_j/16^j are the negative nonconstant coefficients of reciprocal 2F1.
// Exact integer inversion, independent of any imported numerical table.
constexpr std::array<std::uint64_t,16> coefficients() {
    std::array<std::uint64_t,16> u{},c{};u[0]=1;std::uint64_t binomial=1;
    for(unsigned j=1;j<16;++j){binomial=binomial*2*(2*j-1)/j;u[j]=binomial*binomial;
        __uint128_t sum=0;for(unsigned i=1;i<j;++i)sum+=__uint128_t(c[i])*u[j-i];
        c[j]=u[j]-std::uint64_t(sum);}
    return c;
}
constexpr auto coefficient=coefficients();
static_assert(coefficient[1]==4&&coefficient[2]==20&&coefficient[3]==176);
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;
    std::size_t precision=0,working=0,iterations=0,bytes=0;
    unsigned workers=1;bool pi_ready=false;RealArithmetic arithmetic{};
    ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};ErrorMagnitude pi_radius{};
    float_buffer* v[20]{};
};
struct Quote {requirements need{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};
    std::size_t working=0,iterations=0,limbs=0;};
Quote quote(bits p,unsigned workers,ConstantBankBinding constants={}) noexcept {
    Quote q;q.need.output_limbs=(p.count+63)/64;q.need.scratch_alignment=128;
    if(!p.count||p.count>float_precision_limit||!workers||workers>32||
       !add_size(p.count,256,q.working)||q.working>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="elliptic precision/workers";return q;}
    q.iterations=66+std::bit_width(p.count+128);q.limbs=(q.working+63)/64;
    q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    if(constants.supplied()){
        if(!constant_bank_supports(constants,{q.working,0})){q.need.status=support::unsupported;q.need.reason="elliptic constant binding";return q;}
    }else q.bank=constant_bank_requirements({q.working,0},workers);
    if(!q.arithmetic.need.supported()||!q.bank.supported()){
        q.need.status=support::unsupported;q.need.reason="elliptic dependencies";return q;}
    q.need.scratch_alignment=std::max({q.arithmetic.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128);
    for(unsigned i=0;i<20;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="elliptic storage bytes";}
    return q;
}
bool below(ErrorMagnitude e,std::int64_t exponent) noexcept {
    const auto p=e.publish();return p.exact()||(p.kind==error_kind::finite&&p.exponent2<=exponent);
}
ErrorMagnitude quotient_radius(float_view numerator,ErrorMagnitude rn,float_view denominator,
                                ErrorMagnitude rd,error_bound local) noexcept {
    // denominator center >=2^(T-1), exact denominator >=2^(T-2).
    const auto t=real_top(denominator);
    require(below(rd,t-2),SBN3_FATAL_MATH,"elliptic quotient denominator radius");
    return rn.scale2(2-t).add(real_upper(numerator).multiply(rd).scale2(3-2*t)).add(ErrorMagnitude::from(local));
}
void disjoint(std::span<std::byte> work,float_ref k,float_ref e,float_view x) noexcept {
    const auto* a=detail::float_access::begin(k);const auto* b=detail::float_access::begin(e);
    require(!overlaps(a->data,k.capacity()*8,b->data,e.capacity()*8),SBN3_FATAL_ARGUMENT,"elliptic output overlap");
    require(!overlaps(work.data(),work.size(),a->data,k.capacity()*8)&&
            !overlaps(work.data(),work.size(),b->data,e.capacity()*8)&&
            !overlaps(work.data(),work.size(),x.magnitude().data(),x.magnitude().size()*8),
            SBN3_FATAL_ARGUMENT,"elliptic prepared storage alias");
}
bool special_case(float_ref k,float_ref e,float_view x,EllipticParameter kind,EllipticReport& reports) noexcept {
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    if(x.is_nan()){
        reports={special(k,float_class::quiet_nan,false),special(e,float_class::quiet_nan,false)};return true;}
    if(!x.finite()){
        if((kind==EllipticParameter::parameter&&x.negative())||(kind==EllipticParameter::complement&&!x.negative())){
            reports={special(k,float_class::finite,false),special(e,float_class::infinity,false)};return true;}
        reports={special(k,float_class::quiet_nan,false,float_flag::invalid),special(e,float_class::quiet_nan,false,float_flag::invalid)};return true;
    }
    const bool invalid=kind==EllipticParameter::parameter?compare(x,one)==std::partial_ordering::greater:
        x.negative()&&!x.zero();
    if(invalid){reports={special(k,float_class::quiet_nan,false,float_flag::invalid),
        special(e,float_class::quiet_nan,false,float_flag::invalid)};return true;}
    const bool endpoint=kind==EllipticParameter::parameter?compare(x,one)==std::partial_ordering::equivalent:x.zero();
    if(endpoint){reports={special(k,float_class::infinity,false),round_into(e,one)};return true;}
    return false;
}
struct Call {State* s;float_ref k,e;float_view input;EllipticParameter kind;rounding mode;EllipticReport report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;auto& ar=s.arithmetic;
    const execution ex{{},nullptr,scope};const std::uint64_t word=1;
    const auto one=float_view::from_magnitude({&word,1},0);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    ErrorMagnitude complement_radius{},factor_radius{};std::int64_t factor_scale=0;
    const bool transformed=call.kind==EllipticParameter::parameter?call.input.negative()&&!call.input.zero():
        compare(call.input,one)==std::partial_ordering::greater;
    if(transformed){
        // Normalize g=1-m before adding one. The public g might round out
        // of range, while sqrt(g), K and E are all still representable.
        auto positive=call.input.negative()?call.input.negated():call.input;
        const auto top=real_top(positive);factor_scale=top>2?top/2:0;
        const auto normalized=real_scaled(positive,-2*factor_scale);
        float_report g_report;
        if(call.kind==EllipticParameter::parameter)
            g_report=real_add(ar,ref(6),normalized,real_scaled(one,-2*factor_scale));
        else g_report=round_into(ref(6),normalized);
        const auto g_radius=ErrorMagnitude::from(g_report.error);
        auto report=real_root(ar,ref(5),val(6),ex);
        // g_normal >=1/2 and its tiny error imply sqrt sensitivity <2.
        factor_radius=ErrorMagnitude::from(report.error).add(g_radius.scale2(1));
        report=real_divide(ar,ref(19),one,val(6),ex);
        complement_radius=quotient_radius(one,{},val(6),g_radius,report.error).scale2(-2*factor_scale);
        detail::float_access::metadata(ref(19)).exponent2-=2*factor_scale;
        detail::float_access::finish(ref(19));
    }else{
        float_report report;
        if(call.kind==EllipticParameter::parameter)report=real_add(ar,ref(19),one,call.input.negated());
        else report=round_into(ref(19),call.input);
        complement_radius=ErrorMagnitude::from(report.error);round_into(ref(5),one);
    }
    require(val(19).finite()&&!val(19).zero()&&!val(19).negative(),SBN3_FATAL_MATH,"elliptic positive complement");
    const auto relative_complement=complement_radius.scale2(1-real_top(val(19)));
    require(below(relative_complement,8-std::int64_t(s.working)),SBN3_FATAL_MATH,"elliptic complement error");
    if(!s.pi_ready){
        if(s.owned_bank.state)s.constants=constant_bank_execute(s.owned_bank,ex);
        s.pi_radius=ErrorMagnitude::from(constant_bank_prefix(s.constants,ConstantKind::pi,ref(4)).error);s.pi_ready=true;
    }
    const auto pi_radius=s.pi_radius;
    if(!transformed&&compare(val(19),one)==std::partial_ordering::equivalent&&complement_radius.is_zero()){
        call.report={real_finish(call.k,real_scaled(val(4),-1),pi_radius.scale2(-1),call.mode),
                     real_finish(call.e,real_scaled(val(4),-1),pi_radius.scale2(-1),call.mode)};return;
    }
    round_into(ref(0),one);real_root(ar,ref(1),val(19),ex);special(ref(2),float_class::finite,false);
    const auto constant_report=real_add(ar,ref(3),real_scaled(one,-1),real_scaled(val(19),-1));
    const auto constant_radius=complement_radius.scale2(-1).add(ErrorMagnitude::from(constant_report.error));
    const auto initial_top=real_top(val(1));
    const auto ratio_bits=std::uint64_t(std::max<std::int64_t>(0,2-initial_top));
    const auto k_guard=unsigned(std::bit_width(2*(ratio_bits+2)-1));
    std::size_t r=0;std::int64_t rho_exponent=0,alpha_exponent=0;
    const auto local_exponent=32-std::int64_t(s.working);
    for(;;){
        real_add(ar,ref(6),real_scaled(val(0),-1),real_scaled(val(1),-1));
        real_add(ar,ref(7),real_scaled(val(0),-1),real_scaled(val(1),-1).negated());
        real_divide(ar,ref(8),val(7),val(6),ex);real_square(ar,ref(8),val(8),ex);
        const auto rho=real_upper(val(8)).add(ErrorMagnitude::power_of_two(local_exponent)).publish();
        rho_exponent=rho.exponent2;alpha_exponent=real_top(val(6))+1;
        const __int128 gate=-__int128(s.precision)-17-__int128(r)-k_guard-
            std::max<std::int64_t>(0,2*alpha_exponent);
        if(rho.kind==error_kind::finite&&rho_exponent<=-1&&__int128(order)*rho_exponent<=gate)break;
        require(r<s.iterations,SBN3_FATAL_MATH,"elliptic iteration bound");
        real_square(ar,ref(11),val(7),ex);
        real_add(ar,ref(2),val(2),real_scaled(val(11),std::int64_t(r)));
        real_mul(ar,ref(11),val(0),val(1),ex);real_root(ar,ref(1),val(11),ex);
        round_into(ref(0),val(6));++r;
    }
    // Normalize only A. x is already dimensionless. All local rounding is
    // covered by u=2^(32-working); this includes the complete forward input
    // envelope, not just polynomial roundoff.
    round_into(ref(6),real_scaled(val(6),-alpha_exponent));
    special(ref(9),float_class::finite,false);special(ref(10),float_class::finite,false);
    for(unsigned j=order;j-->0;){
        real_mul(ar,ref(11),val(10),val(8),ex);real_add(ar,ref(10),val(11),val(9));
        real_mul(ar,ref(11),val(9),val(8),ex);
        const auto cj=float_view::from_magnitude({&coefficient[j],1},-4*std::int64_t(j));
        real_add(ar,ref(9),val(11),cj);
    }
    real_add(ar,ref(12),one,val(9).negated());
    require(compare(val(12),real_scaled(one,-2))==std::partial_ordering::greater,SBN3_FATAL_MATH,"elliptic closure denominator");
    real_divide(ar,ref(13),val(10).negated(),val(12),ex);
    real_add(ar,ref(11),one,val(8).negated());real_mul(ar,ref(13),val(13),val(11),ex);
    real_add(ar,ref(13),one,real_scaled(val(13),1));
    real_mul(ar,ref(14),val(6),val(12),ex);
    detail::float_access::metadata(ref(14)).exponent2+=alpha_exponent;detail::float_access::finish(ref(14));
    real_square(ar,ref(15),val(6),ex);real_mul(ar,ref(15),val(15),val(8),ex);
    real_mul(ar,ref(15),val(15),val(13),ex);
    detail::float_access::metadata(ref(15)).exponent2+=1+2*alpha_exponent;detail::float_access::finish(ref(15));
    const auto tail_exponent=std::int64_t(order)*rho_exponent;
    const auto moment_radius=ErrorMagnitude::from_scaled_uint64(8*(order+1),tail_exponent)
        .add(ErrorMagnitude::from_scaled_uint64(272,local_exponent)).scale2(std::int64_t(r)+2*alpha_exponent);
    const auto value_radius=ErrorMagnitude::power_of_two(tail_exponent)
        .add(ErrorMagnitude::from_scaled_uint64(8,local_exponent)).scale2(alpha_exponent);
    const auto prefix_radius=ErrorMagnitude::from_scaled_uint64(32*(r+1),std::int64_t(r)+8-std::int64_t(s.working));
    auto report=real_add(ar,ref(16),val(2),real_scaled(val(15),std::int64_t(r)));
    auto correction_radius=moment_radius.add(prefix_radius).add(ErrorMagnitude::from(report.error));
    report=real_add(ar,ref(16),val(3),val(16).negated());
    correction_radius=correction_radius.add(constant_radius).add(ErrorMagnitude::from(report.error));
    report=real_divide(ar,ref(17),val(4),real_scaled(val(14),1),ex);
    auto k_radius=quotient_radius(val(4),pi_radius,real_scaled(val(14),1),value_radius.scale2(1),report.error);
    report=real_mul(ar,ref(18),val(17),val(16),ex);
    auto e_radius=real_upper(val(17)).multiply(correction_radius)
        .add(real_upper(val(16)).multiply(k_radius)).add(k_radius.multiply(correction_radius))
        .add(ErrorMagnitude::from(report.error));
    if(transformed){
        report=real_divide(ar,ref(19),val(17),val(5),ex);
        k_radius=quotient_radius(val(17),k_radius,val(5),factor_radius,report.error).scale2(-factor_scale);
        call.report.k=real_finish(call.k,real_scaled(val(19),-factor_scale),k_radius,call.mode);
        report=real_mul(ar,ref(19),val(18),val(5),ex);
        e_radius=real_upper(val(18)).multiply(factor_radius).add(real_upper(val(5)).multiply(e_radius))
            .add(e_radius.multiply(factor_radius)).add(ErrorMagnitude::from(report.error)).scale2(factor_scale);
        call.report.e=real_finish(call.e,real_scaled(val(19),factor_scale),e_radius,call.mode);
    }else call.report={real_finish(call.k,val(17),k_radius,call.mode),real_finish(call.e,val(18),e_radius,call.mode)};
    for(auto result:{std::pair{call.k.view(),call.report.k},std::pair{call.e.view(),call.report.e}}){
        require(result.first.finite()&&!result.first.zero()&&!result.first.negative(),SBN3_FATAL_MATH,"elliptic finite output");
        const auto limit=real_top(result.first)-std::int64_t(s.precision)+(call.mode==rounding::nearest_even?0:1);
        require(result.second.error.kind!=error_kind::unbounded&&
            (result.second.error.exact()||result.second.error.exponent2<=limit),SBN3_FATAL_MATH,"elliptic final radius");
    }
}
}
FloatDiagnostic elliptic_diagnostic(bits p,unsigned workers) noexcept {
    const auto q=quote(p,workers);FloatDiagnostic d;d.need=q.need;d.recipe="coupled_agm_ke";
    d.working_bits=q.working;d.metadata_bytes=sizeof(State);d.constant_value_bytes=q.bank.value_bytes;
    d.constant_work_bytes=q.bank.workspace_bytes;return d;
}
requirements elliptic_program_requirements(bits precision,unsigned workers,ConstantBankBinding constants) noexcept{return quote(precision,workers,constants).need;}
FloatEllipticPrepared elliptic_program_prepare(bits precision,unsigned workers,std::span<std::byte> storage,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"elliptic needs bound constants");
    const auto q=quote(precision,workers,constants);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"elliptic storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,constants.values().storage.data(),constants.values().storage.size()),SBN3_FATAL_ARGUMENT,"elliptic constant/work alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->precision=precision.count;s->working=q.working;s->iterations=q.iterations;s->bytes=q.need.scratch_bytes;s->workers=workers;
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);
        v=::new(raw)float_buffer({data,q.limbs},bits{q.working});}
    {auto work=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);
        real_arithmetic_prepare(s->arithmetic,q.arithmetic,work);}
    if(!constants.supplied()){
        auto* values=static_cast<std::byte*>(frame.allocate(q.bank.value_bytes,q.bank.value_alignment));
        auto* work=static_cast<std::byte*>(frame.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment));
        s->owned_bank=constant_bank_prepare(q.bank,{values,q.bank.value_bytes},{work,q.bank.workspace_bytes});constants=constant_bank_pending_binding(s->owned_bank);
    }
    s->constants=constants;
    s->marker=magic;return {s,storage.first(q.need.scratch_bytes)};
}
EllipticReport elliptic_program_execute(const FloatEllipticPrepared& prepared,float_ref k,float_ref e,
    float_view input,EllipticParameter kind,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),
        SBN3_FATAL_LIFETIME,"elliptic prepared identity");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&k.precision().count==s.precision&&
        e.precision().count==s.precision&&ex.workers()>=s.workers&&
        (kind==EllipticParameter::parameter||kind==EllipticParameter::complement),SBN3_FATAL_ARGUMENT,"elliptic prepared shape");
    validate(input);check_output(k,input);check_output(e,input);disjoint(prepared.storage,k,e,input);
    constant_bank_check_output(s.constants,k);constant_bank_check_output(s.constants,e);
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
        SBN3_FATAL_LIFETIME,"elliptic concurrent use");
    Call call{&s,k,e,input,kind,mode,{}};
    if(!special_case(k,e,input,kind,call.report)){
        if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);
    }
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
requirements elliptic_ke_requirements(float_ref k,float_ref e,float_view,EllipticParameter,unsigned workers,ConstantBankBinding constants) noexcept {
    if(k.precision().count!=e.precision().count)return {support::unsupported,0,128,0,"elliptic paired precision"};
    auto q=elliptic_program_requirements(k.precision(),workers,constants);
    if(k.capacity()<q.output_limbs||e.capacity()<q.output_limbs){q.status=support::capacity;q.reason="elliptic output capacity";}return q;
}
EllipticReport elliptic_ke_into(float_ref k,float_ref e,float_view input,EllipticParameter kind,execution ex,rounding mode,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"elliptic needs bound constants");validate(input);check_output(k,input);check_output(e,input);
    constant_bank_check_output(constants,k);constant_bank_check_output(constants,e);
    const auto q=elliptic_ke_requirements(k,e,input,kind,ex.workers(),constants);require(q.supported(),SBN3_FATAL_ARGUMENT,q.reason);
    require(ex.scratch.size()>=q.scratch_bytes,SBN3_FATAL_WORKSPACE,"elliptic fresh scratch",q.scratch_bytes,ex.scratch.size());
    disjoint(ex.scratch.first(q.scratch_bytes),k,e,input);
    auto prepared=elliptic_program_prepare(k.precision(),ex.workers(),ex.scratch.first(q.scratch_bytes),constants);
    return elliptic_program_execute(prepared,k,e,input,kind,ex,mode);
}
}
namespace sbn {
requirements elliptic_ke_requirements(float_ref k,float_ref e,float_view x,elliptic_argument kind,unsigned workers)noexcept{
    if(kind!=elliptic_argument::parameter&&kind!=elliptic_argument::complement)
        return {support::unsupported,0,128,0,"elliptic argument convention"};
    return v3::floating::elliptic_ke_requirements(k,e,x,
        kind==elliptic_argument::parameter?v3::floating::EllipticParameter::parameter:v3::floating::EllipticParameter::complement,workers);
}
elliptic_report elliptic_ke_into(float_ref k,float_ref e,float_view x,elliptic_argument kind,execution ex,rounding mode)noexcept{
    v3::require(kind==elliptic_argument::parameter||kind==elliptic_argument::complement,SBN3_FATAL_ARGUMENT,"elliptic argument convention");
    const auto r=v3::floating::elliptic_ke_into(k,e,x,
        kind==elliptic_argument::parameter?v3::floating::EllipticParameter::parameter:v3::floating::EllipticParameter::complement,ex,mode);
    return {r.k,r.e};
}
}
