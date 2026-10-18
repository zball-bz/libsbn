// Original libsbn code; LGPL-3.0-or-later.
#include "float/exponential.hpp"
#include "float/diagnostics.hpp"
#include "float/exp_series.hpp"
#include "float/logarithm.hpp"
#include "float/constants.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
ConstantBankDemand exponential_constant_demand(bits p) noexcept {
    std::size_t reduction=0,target=0;if(!p.count||p.count>float_precision_limit||!add_size(p.count,128,reduction)||
       reduction>float_precision_limit||!add_size(p.count,32,target))return {};
    ConstantBankDemand demand{0,reduction};
    for(std::size_t at=40;at<target;){at=std::min(target,2*at-2);const auto child=logarithm_constant_demand(bits{at+20});
        if(!child.pi_bits||!child.log2_bits)return {};demand=constant_bank_merge(demand,child);}
    return demand;
}
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3345585031);
constexpr std::size_t seed_precision=128,seed_accuracy=40;
struct Rung {std::size_t accuracy=0,precision=0;RealArithmetic arithmetic{};FloatLogarithmPrepared logarithm{};};
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;unsigned workers=1;
    std::size_t precision=0,reduction_precision=0,count=0,bytes=0,limbs=0;
    bool constant_ready=false;ErrorMagnitude log2_radius{};
    RealArithmetic reduction{},seed{};ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};Rung* rungs=nullptr;
    float_buffer* v[12]{};
};
struct Quote {requirements need{};ConstantBankQuote bank{};RealArithmeticPlan reduction{},seed{};
    std::size_t precision=0,count=0,limbs=0;};
bool below(ErrorMagnitude radius,std::int64_t exponent) noexcept {
    const auto e=radius.publish();return e.exact()||(e.kind==error_kind::finite&&e.exponent2<=exponent);
}
std::int64_t narrow(__int128 value) noexcept {
    require(value>=INT64_MIN&&value<=INT64_MAX,SBN3_FATAL_SIZE,"exponential exponent");return std::int64_t(value);
}
std::size_t next_accuracy(std::size_t current,std::size_t target) noexcept{return std::min(target,2*current-2);}
Quote quote(bits precision,unsigned workers,ConstantBankBinding constants={}) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(precision.count+63)/64;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32||
       !add_size(precision.count,128,q.precision)||q.precision>float_precision_limit){
        q.need.status=support::unsupported;q.need.reason="exponential precision/workers";return q;}
    const auto target=precision.count+32;
    for(auto at=seed_accuracy;at<target;at=next_accuracy(at,target))++q.count;
    q.limbs=(q.precision+63)/64;q.reduction=real_arithmetic_query(bits{q.precision},workers);
    q.seed=real_arithmetic_query(bits{seed_precision},workers);
    const auto demand=exponential_constant_demand(precision);
    if(!demand.log2_bits){q.need.status=support::unsupported;q.need.reason="exponential constant demand";return q;}
    if(constants.supplied()){
        if(!constant_bank_supports(constants,demand)){q.need.status=support::unsupported;q.need.reason="exponential constant binding";return q;}
    }else q.bank=constant_bank_requirements(demand,workers);
    const auto child_constants=constants.supplied()?constants:constant_bank_query_binding(demand);
    if(!q.reduction.need.supported()||!q.seed.need.supported()||!q.bank.supported()){
        q.need.status=support::unsupported;q.need.reason="exponential base dependencies";return q;}
    q.need.scratch_alignment=std::max({q.reduction.need.scratch_alignment,q.seed.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128)&&real_append(n,q.count*sizeof(Rung),128);
    for(unsigned i=0;i<12;++i)ok=ok&&real_append(n,sizeof(float_buffer),64)&&real_append(n,q.limbs*8,64);
    ok=ok&&real_append(n,q.reduction.need.scratch_bytes,q.reduction.need.scratch_alignment)&&
        real_append(n,q.seed.need.scratch_bytes,q.seed.need.scratch_alignment)&&real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&
        real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment);
    for(auto at=seed_accuracy;at<target;){
        at=next_accuracy(at,target);const auto stage=bits{at+20};
        const auto ar=real_arithmetic_query(stage,workers);const auto log=logarithm_program_requirements(stage,workers,child_constants);
        if(!ar.need.supported()||!log.supported()){q.need.status=support::unsupported;q.need.reason="exponential inverse rung";return q;}
        q.need.scratch_alignment=std::max({q.need.scratch_alignment,ar.need.scratch_alignment,log.scratch_alignment});
        ok=ok&&real_append(n,ar.need.scratch_bytes,ar.need.scratch_alignment)&&real_append(n,log.scratch_bytes,log.scratch_alignment);
    }
    if(!ok){q.need.status=support::unsupported;q.need.reason="exponential storage bytes";}return q;
}
ErrorMagnitude prefix(float_ref out,float_view input) noexcept {
    const auto width=magnitude_bits(input.magnitude()),p=out.precision().count;
    if(!width){special(out,float_class::finite,input.negative());return {};}
    const auto cut=width>p?width-p:0,kept=width-cut,n=(kept+63)/64,first=cut/64;const auto shift=unsigned(cut%64);
    auto* target=detail::float_access::begin(out);
    for(std::size_t j=0;j<n;++j){const auto at=first+j;target->data[j]=(input.magnitude().data()[at]>>shift)|
        (shift&&at+1<input.magnitude().size()?input.magnitude().data()[at+1]<<(64-shift):0);}
    target->size=n;target->negative=0;auto& metadata=detail::float_access::metadata(out);
    metadata.exponent2=narrow(__int128(input.exponent2())+cut);metadata.negative=input.negative();metadata.classification=float_class::finite;
    detail::float_access::finish(out);return cut?ErrorMagnitude::power_of_two(metadata.exponent2):ErrorMagnitude{};
}
bool simple(float_view x,ExponentialKind kind,std::size_t p) noexcept {
    return !x.finite()||x.zero()||real_top(x)>=61||
        (kind==ExponentialKind::exp&&real_top(x)<=-std::int64_t(p)-18);
}
bool special_case(float_ref out,float_view x,ExponentialKind kind,rounding mode,float_report& report) noexcept {
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0);
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(x.zero()){report=kind==ExponentialKind::exp?round_into(out,one,mode):special(out,float_class::finite,x.negative());return true;}
    if(!x.finite()){
        if(!x.negative())report=special(out,float_class::infinity,false);
        else report=kind==ExponentialKind::exp?special(out,float_class::finite,false):round_into(out,one.negated(),mode);
        return true;
    }
    const auto top=real_top(x);
    if(top>=61){
        if(!x.negative())report=special(out,float_class::infinity,false,float_flag::overflow|float_flag::rounded);
        else if(kind==ExponentialKind::exp){report=special(out,float_class::finite,false,float_flag::underflow|float_flag::rounded);
            report.error=error_bound::power_of_two(-float_exponent_limit-1);}
        else{report=round_into(out,one.negated(),mode);report.error=error_bound::power_of_two(-float_exponent_limit-1);}
        return true;
    }
    if(kind==ExponentialKind::exp&&top<=-std::int64_t(out.precision().count)-18){
        report=real_finish(out,one,ErrorMagnitude::power_of_two(top+1),mode);return true;}
    return false;
}
ErrorMagnitude quotient_radius(float_view numerator,ErrorMagnitude rn,float_view denominator,ErrorMagnitude rd,error_bound local) noexcept {
    const auto t=real_top(denominator);require(below(rd,t-2),SBN3_FATAL_MATH,"exponential reduction denominator");
    return rn.scale2(2-t).add(real_upper(numerator).multiply(rd).scale2(3-2*t)).add(ErrorMagnitude::from(local));
}
struct Call {State* s;float_ref out;float_view input;ExponentialKind kind;rounding mode;float_report report{};};
void precision(State& s,std::size_t bits) noexcept {
    // Values 9/10/11 retain the reduction source, log2, and exact reduced
    // target. Changing an output precision never rounds an existing point.
    for(unsigned i=0;i<9;++i)detail::float_access::metadata(s.v[i]->ref()).precision=bits;
}
void seed(State& s,bool correction,execution ex) noexcept {
    precision(s,seed_precision);auto& ar=s.seed;const std::uint64_t word=1;
    const auto one=float_view::from_magnitude({&word,1},0);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    round_into(ref(1),one);if(correction)special(ref(0),float_class::finite,false);else round_into(ref(0),one);
    ErrorMagnitude term_error{},sum_error{};const auto argument_upper=real_upper(val(11));
    for(std::uint64_t j=1;j<=16;++j){
        auto report=real_mul(ar,ref(2),val(1),val(11),ex);
        term_error=argument_upper.multiply(term_error).add(ErrorMagnitude::from(report.error));
        const auto integer=float_view::from_magnitude({&j,1},0);
        report=real_divide(ar,ref(1),val(2),integer,ex);
        // Division by j>=1 cannot increase the previous absolute radius.
        term_error=term_error.add(ErrorMagnitude::from(report.error));
        report=real_add(ar,ref(0),val(0),val(1));sum_error=sum_error.add(term_error).add(ErrorMagnitude::from(report.error));
    }
    // Sixteen Taylor terms on |r|<=1 give relative remainder <2^-44.
    // For expm1, |r|<=1/2 and |expm1(r)|>=|r|/2 give the same bound.
    // The actual arithmetic ledger must fit an additional <2^-44 relative.
    require(!val(0).zero()&&val(0).finite()&&below(sum_error,real_top(val(0))-46),SBN3_FATAL_MATH,"exponential Taylor seed error");
}
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Call*>(opaque);auto& s=*call.s;const execution ex{{},nullptr,scope};
    const std::uint64_t word=1;const auto one=float_view::from_magnitude({&word,1},0),half=real_scaled(one,-1);
    auto ref=[&](unsigned i){return s.v[i]->ref();};auto val=[&](unsigned i){return s.v[i]->view();};
    precision(s,s.reduction_precision);auto input_radius=prefix(ref(9),call.input);
    const auto input_upper=real_upper(val(9)).add(input_radius);
    if(call.kind==ExponentialKind::expm1&&below(input_upper,-std::int64_t(s.precision)-16)){
        call.report=real_finish(call.out,val(9),input_radius.add(input_upper.multiply(input_upper)),call.mode);return;
    }
    if(s.owned_bank.state)s.constants=constant_bank_execute(s.owned_bank,ex);
    const bool correction=call.kind==ExponentialKind::expm1&&
        compare(call.input,half)!=std::partial_ordering::greater&&compare(call.input,half.negated())!=std::partial_ordering::less;
    std::int64_t binary_scale=0;ErrorMagnitude target_radius{};
    if(correction){round_into(ref(11),val(9));target_radius=input_radius;}
    else{
        if(!s.constant_ready){s.log2_radius=ErrorMagnitude::from(constant_bank_prefix(s.constants,ConstantKind::log2,ref(10)).error);s.constant_ready=true;}
        auto report=real_divide(s.reduction,ref(1),val(9),val(10),ex);
        const auto quotient_error=quotient_radius(val(9),input_radius,val(10),s.log2_radius,report.error);
        require(below(quotient_error,-4),SBN3_FATAL_MATH,"exponential coarse quotient error");
        std::uint64_t storage=0;integer_buffer integer({&storage,1});to_integer_into(integer.ref(),val(1),rounding::nearest_even);
        require(integer.size()<=1&&storage<=std::uint64_t(INT64_MAX),SBN3_FATAL_SIZE,"exponential reduction integer");
        binary_scale=integer.negative()?-std::int64_t(storage):std::int64_t(storage);
        const auto coefficient=float_view::from_magnitude({&storage,integer.size()},0,integer.negative());
        report=real_mul(s.reduction,ref(2),val(10),coefficient,ex);
        const auto product_error=s.log2_radius.multiply(ErrorMagnitude::from_scaled_uint64(storage,0)).add(ErrorMagnitude::from(report.error));
        report=real_add(s.reduction,ref(11),val(9),val(2).negated());
        target_radius=input_radius.add(product_error).add(ErrorMagnitude::from(report.error));
        require(below(target_radius,-std::int64_t(s.precision)-60)&&
            compare(val(11),one)!=std::partial_ordering::greater&&compare(val(11),one.negated())!=std::partial_ordering::less,
            SBN3_FATAL_MATH,"exponential reduced target");
    }
    seed(s,correction,ex);std::size_t accuracy=seed_accuracy;
    for(std::size_t j=0;j<s.count;++j){
        auto& rung=s.rungs[j];precision(s,rung.precision);auto& ar=rung.arithmetic;
        const auto old_top=real_top(val(0));
        auto report=logarithm_program_execute(rung.logarithm,ref(1),val(0),
            correction?LogarithmKind::log1p:LogarithmKind::log,ex);
        auto residual_error=ErrorMagnitude::from(report.error);
        report=correction?real_add(ar,ref(2),val(1),val(11).negated()):real_add(ar,ref(2),val(11),val(1).negated());
        residual_error=residual_error.add(ErrorMagnitude::from(report.error));
        ErrorMagnitude arithmetic_error{};
        if(correction){
            report=real_add(ar,ref(3),one,val(0));const auto factor_error=ErrorMagnitude::from(report.error);
            report=real_mul(ar,ref(4),val(3),val(2),ex);
            arithmetic_error=real_upper(val(3)).multiply(residual_error)
                .add(real_upper(val(2)).add(residual_error).multiply(factor_error)).add(ErrorMagnitude::from(report.error));
            report=real_add(ar,ref(5),val(0),val(4).negated());arithmetic_error=arithmetic_error.add(ErrorMagnitude::from(report.error));
        }else{
            report=real_add(ar,ref(3),one,val(2));residual_error=residual_error.add(ErrorMagnitude::from(report.error));
            report=real_mul(ar,ref(5),val(0),val(3),ex);
            arithmetic_error=real_upper(val(0)).multiply(residual_error).add(ErrorMagnitude::from(report.error));
        }
        require(below(arithmetic_error,old_top-std::int64_t(rung.accuracy)-8),SBN3_FATAL_MATH,"exponential Newton arithmetic budget");
        round_into(ref(0),val(5));accuracy=rung.accuracy;
    }
    // |relative Newton error|<=2^-accuracy; converting to a computed-center
    // bound costs at most two. Reduction sensitivity costs at most four.
    ErrorMagnitude radius=real_upper(val(0)).multiply(ErrorMagnitude::power_of_two(1-std::int64_t(accuracy)));
    if(correction){radius=radius.add(target_radius.scale2(1));call.report=real_finish(call.out,val(0),radius,call.mode);}
    else{
        radius=radius.add(real_upper(val(0)).multiply(target_radius).scale2(2)).scale2(binary_scale);
        const auto scaled=real_scaled(val(0),binary_scale);
        if(call.kind==ExponentialKind::exp){
            call.report=round_magnitude(call.out,scaled.magnitude(),scaled.exponent2(),false,call.mode);
        }else{
            // sum_finite accepts a wide exact intermediate and applies the
            // public exponent policy only after the final subtraction.
            call.report=sum_finite(call.out,scaled,one.negated(),execution{s.reduction.sums,nullptr,nullptr},call.mode);
        }
        call.report.error=ErrorMagnitude::from(call.report.error).add(radius).publish();
    }
    const auto output=call.out.view();if(output.finite()&&!output.zero()){
        const auto limit=real_top(output)-std::int64_t(s.precision)+(call.mode==rounding::nearest_even?0:1);
        require(call.report.error.kind!=error_kind::unbounded&&
            (call.report.error.exact()||call.report.error.exponent2<=limit),SBN3_FATAL_MATH,"exponential final radius");}
}
}
requirements exponential_program_requirements(bits p,unsigned workers,ConstantBankBinding constants) noexcept{return quote(p,workers,constants).need;}
FloatExponentialPrepared exponential_program_prepare(bits precision_bits,unsigned workers,std::span<std::byte> storage,ConstantBankBinding constants) noexcept {
    require(!constants.query_only(),SBN3_FATAL_ARGUMENT,"exponential needs bound constants");
    const auto q=quote(precision_bits,workers,constants);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"exponential storage",q.need.scratch_bytes,storage.size());
    require(!overlaps(storage.data(),q.need.scratch_bytes,constants.values().storage.data(),constants.values().storage.size()),SBN3_FATAL_ARGUMENT,"exponential constant/work alias");
    auto frame=Frame::external(storage.data(),q.need.scratch_bytes);auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->precision=precision_bits.count;s->reduction_precision=q.precision;s->count=q.count;s->bytes=q.need.scratch_bytes;s->workers=workers;s->limbs=q.limbs;
    s->rungs=q.count?static_cast<Rung*>(frame.allocate(q.count*sizeof(Rung),128)):nullptr;
    for(std::size_t j=0;j<q.count;++j)::new(s->rungs+j)Rung{};
    for(auto& v:s->v){auto* raw=frame.allocate(sizeof(float_buffer),64);auto* data=frame.alloc<std::uint64_t>(q.limbs);v=::new(raw)float_buffer({data,q.limbs},bits{q.precision});}
    {auto sub=frame.subframe(q.reduction.need.scratch_bytes,q.reduction.need.scratch_alignment);real_arithmetic_prepare(s->reduction,q.reduction,sub);}
    {auto sub=frame.subframe(q.seed.need.scratch_bytes,q.seed.need.scratch_alignment);real_arithmetic_prepare(s->seed,q.seed,sub);}
    if(!constants.supplied()){
        auto* values=static_cast<std::byte*>(frame.allocate(q.bank.value_bytes,q.bank.value_alignment));
        auto* work=static_cast<std::byte*>(frame.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment));
        s->owned_bank=constant_bank_prepare(q.bank,{values,q.bank.value_bytes},{work,q.bank.workspace_bytes});constants=constant_bank_pending_binding(s->owned_bank);
    }
    s->constants=constants;
    auto accuracy=seed_accuracy;const auto target=precision_bits.count+32;
    for(std::size_t j=0;j<q.count;++j){
        accuracy=next_accuracy(accuracy,target);auto& rung=s->rungs[j];rung.accuracy=accuracy;rung.precision=accuracy+20;
        const auto ar=real_arithmetic_query(bits{rung.precision},workers);const auto log=logarithm_program_requirements(bits{rung.precision},workers,constants);
        {auto sub=frame.subframe(ar.need.scratch_bytes,ar.need.scratch_alignment);real_arithmetic_prepare(rung.arithmetic,ar,sub);}
        auto* span=static_cast<std::byte*>(frame.allocate(log.scratch_bytes,log.scratch_alignment));
        rung.logarithm=logarithm_program_prepare(bits{rung.precision},workers,{span,log.scratch_bytes},constants);
    }
    s->marker=magic;return {s,storage.first(q.need.scratch_bytes)};
}
float_report exponential_program_execute(const FloatExponentialPrepared& prepared,float_ref out,float_view input,
    ExponentialKind kind,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"exponential prepared identity");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers&&
        (kind==ExponentialKind::exp||kind==ExponentialKind::expm1),SBN3_FATAL_ARGUMENT,"exponential prepared shape");
    validate(input);check_output(out,input);const auto* dest=detail::float_access::begin(out);
    constant_bank_check_output(s.constants,out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),dest->data,out.capacity()*8)&&
        !overlaps(prepared.storage.data(),prepared.storage.size(),input.magnitude().data(),input.magnitude().size()*8),SBN3_FATAL_ARGUMENT,"exponential storage alias");
    std::uint32_t idle=0;require(__atomic_compare_exchange_n(&s.active,&idle,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"exponential concurrent reuse");
    Call call{&s,out,input,kind,mode,{}};
    if(!special_case(out,input,kind,mode,call.report)){
        if(ex.scope)calculate(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&call);else serial_team_run(calculate,&call);}
    __atomic_store_n(&s.active,0,__ATOMIC_RELEASE);return call.report;
}
bool series_input(float_view input,std::size_t precision)noexcept{
    if(precision>4096||!input.finite()||input.zero())return false;
    const auto top=input.exponent2()+std::int64_t(magnitude_bits(input.magnitude()));
    if(top<=0)return true;
    // A conservative dispatch avoids scanning a tall source solely to decide
    // whether it is exactly the boundary point one.
    return input.magnitude().size()==1&&std::has_single_bit(input.magnitude().data()[0])&&
        input.exponent2()+std::countr_zero(input.magnitude().data()[0])==0;
}
requirements exponential_requirements(float_ref out,float_view input,ExponentialKind kind,unsigned workers,ConstantBankBinding constants) noexcept {
    if(!out.precision().count||out.precision().count>float_precision_limit||!workers||workers>32||
       (kind!=ExponentialKind::exp&&kind!=ExponentialKind::expm1))
        return {support::unsupported,0,128,0,"exponential shape"};
    const auto limbs=(out.precision().count+63)/64;
    if(out.capacity()<limbs)return {support::capacity,0,128,limbs,"exponential output capacity"};
    if(simple(input,kind,out.precision().count))return {support::yes,0,128,limbs,""};
    if(series_input(input,out.precision().count))return exp_series_requirements(out,input,
        kind==ExponentialKind::exp?ExpSeriesKind::exp:ExpSeriesKind::expm1,workers);
    return exponential_program_requirements(out.precision(),workers,constants);
}
FloatDiagnostic exponential_diagnostic(float_ref out,float_view x,bool minus_one,unsigned workers) noexcept {
    const auto kind=minus_one?ExponentialKind::expm1:ExponentialKind::exp;
    FloatDiagnostic d;d.need=exponential_requirements(out,x,kind,workers);
    if(simple(x,kind,out.precision().count)){d.recipe="analytic";d.working_bits=out.precision().count;return d;}
    if(series_input(x,out.precision().count))return exp_series_diagnostic(out.precision(),minus_one,x.negative(),workers);
    const auto q=quote(out.precision(),workers);d.recipe="newton_agm_log";d.working_bits=q.precision;
    d.metadata_bytes=sizeof(State)+q.count*sizeof(Rung);d.constant_value_bytes=q.bank.value_bytes;
    d.constant_work_bytes=q.bank.workspace_bytes;return d;
}
float_report exponential_into(float_ref out,float_view input,ExponentialKind kind,execution ex,rounding mode,ConstantBankBinding constants) noexcept {
    require(kind==ExponentialKind::exp||kind==ExponentialKind::expm1,SBN3_FATAL_ARGUMENT,"exponential kind");
    validate(input);check_output(out,input);constant_bank_check_output(constants,out);float_report report;if(special_case(out,input,kind,mode,report))return report;
    if(series_input(input,out.precision().count))return exp_series_into(out,input,
        kind==ExponentialKind::exp?ExpSeriesKind::exp:ExpSeriesKind::expm1,ex,mode);
    const auto need=exponential_requirements(out,input,kind,ex.workers(),constants);require(need.supported(),SBN3_FATAL_ARGUMENT,need.reason);
    require(ex.scratch.size()>=need.scratch_bytes,SBN3_FATAL_WORKSPACE,"exponential fresh workspace",need.scratch_bytes,ex.scratch.size());
    const auto* dest=detail::float_access::begin(out);
    require(!overlaps(ex.scratch.data(),need.scratch_bytes,input.magnitude().data(),input.magnitude().size()*8)&&
        !overlaps(ex.scratch.data(),need.scratch_bytes,dest->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"exponential fresh value/work alias");
    const auto prepared=exponential_program_prepare(out.precision(),ex.workers(),ex.scratch.first(need.scratch_bytes),constants);
    return exponential_program_execute(prepared,out,input,kind,ex,mode);
}
}
namespace sbn {
requirements exp_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::exponential_requirements(out,x,v3::floating::ExponentialKind::exp,workers);}
requirements expm1_requirements(float_ref out,float_view x,unsigned workers) noexcept{return v3::floating::exponential_requirements(out,x,v3::floating::ExponentialKind::expm1,workers);}
float_report exp_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::exponential_into(out,x,v3::floating::ExponentialKind::exp,ex,mode);}
float_report expm1_into(float_ref out,float_view x,execution ex,rounding mode) noexcept{return v3::floating::exponential_into(out,x,v3::floating::ExponentialKind::expm1,ex,mode);}
}
