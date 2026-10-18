// Original libsbn code; LGPL-3.0-or-later.
#include "float/inverse_trig.hpp"
#include "float/constants.hpp"
#include "float/trig_agm.hpp"
#include <algorithm>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3349545231);
struct State {
    std::uint64_t marker=magic;std::uint32_t active=0;InverseTrigKind kind=InverseTrigKind::atan;
    std::size_t precision=0,working=0,pi_bits=0,bytes=0;unsigned workers=1;
    RealArithmetic arithmetic{};ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};FloatTrigAgmPrepared atan{};
    float_buffer* bank=nullptr;ErrorMagnitude pi_error{};bool pi_ready=false,owns_bank=false;
};
struct Quote {requirements need{},atan{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};
    std::size_t working=0,pi_bits=0;bool owns_bank=false;};
bool valid_kind(InverseTrigKind k) noexcept{return k==InverseTrigKind::atan||k==InverseTrigKind::atan2||k==InverseTrigKind::asin||k==InverseTrigKind::acos;}
Quote quote(bits p,unsigned workers,InverseTrigKind kind,ConstantBankBinding binding) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit/16||!workers||workers>32||!valid_kind(kind)){
        q.need.status=support::unsupported;q.need.reason="inverse trig precision/workers";return q;}
    q.working=2*p.count+256;q.pi_bits=2*q.working+128;
    const auto demand=inverse_trig_constant_demand(p,kind);auto children=binding;
    if(binding.supplied()){
        if(!constant_bank_supports(binding,demand)){q.need.status=support::unsupported;q.need.reason="inverse trig constant bank";return q;}
    }else{
        q.owns_bank=true;q.bank=constant_bank_requirements(demand,workers);
        if(!q.bank.supported()){q.need.status=support::unsupported;q.need.reason=q.bank.reason;return q;}
        children=constant_bank_query_binding({q.bank.pi_bits,q.bank.log2_bits});
    }
    q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    q.atan=trig_agm_requirements(bits{q.working},workers,TrigAgmKind::atan,true,children);
    if(!q.arithmetic.need.supported()||!q.atan.supported()){
        q.need.status=support::unsupported;q.need.reason="inverse trig child shape";return q;}
    q.need.scratch_alignment=std::max({std::size_t(128),q.arithmetic.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment,q.atan.scratch_alignment});
    std::size_t n=0;bool ok=real_append(n,sizeof(State),128)&&real_append(n,11*sizeof(float_buffer),64);
    for(unsigned j=0;j<11&&ok;++j)ok=real_append(n,(((j==10?q.pi_bits:q.working)+63)/64)*8,64);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);
    if(q.owns_bank)ok=ok&&real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment);
    ok=ok&&real_append(n,q.atan.scratch_bytes,q.atan.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="inverse trig storage extent";return q;}
    q.need.scratch_bytes=n;return q;
}
std::int64_t checked(__int128 e) noexcept {
    require(e>=INT64_MIN&&e<=INT64_MAX,SBN3_FATAL_SIZE,"inverse trig exponent");return std::int64_t(e);
}
float_view abs(float_view x) noexcept{return x.negative()?x.negated():x;}
float_view v(State& s,unsigned j) noexcept{return s.bank[j].view();}
void good(float_report r,bool allow_underflow=false) noexcept {
    require(r.error.kind!=error_kind::unbounded&&!has_flag(r.flags,float_flag::invalid)&&
        !has_flag(r.flags,float_flag::overflow)&&!has_flag(r.flags,float_flag::divide_by_zero)&&
        (allow_underflow||!has_flag(r.flags,float_flag::underflow)),SBN3_FATAL_MATH,"inverse trig arithmetic");
}
ErrorMagnitude prefix(float_ref out,float_view x) noexcept {
    if(x.zero()){special(out,float_class::finite,x.negative());return {};}
    const auto a=x.magnitude();const auto width=magnitude_bits(a),p=out.precision().count,cut=width>p?width-p:0;
    const auto first=cut/64,n=(width-cut+63)/64;const auto shift=unsigned(cut%64);auto* d=detail::float_access::begin(out);
    for(std::size_t j=0;j<n;++j){const auto i=first+j;d->data[j]=(a.data()[i]>>shift)|(shift&&i+1<a.size()?a.data()[i+1]<<(64-shift):0);}
    d->size=n;d->negative=0;auto& m=detail::float_access::metadata(out);m.classification=float_class::finite;
    m.exponent2=checked(__int128(x.exponent2())+cut);m.negative=x.negative();detail::float_access::finish(out);
    return cut?ErrorMagnitude::power_of_two(m.exponent2):ErrorMagnitude{};
}
void compute_pi(State& s,execution ex) noexcept {
    if(s.pi_ready)return;
    if(s.owns_bank)s.constants=ConstantBankBinding(constant_bank_execute(s.owned_bank,ex));
    const auto r=constant_bank_prefix(s.constants,ConstantKind::pi,s.bank[10].ref());good(r);
    require(r.error.exact()||(r.error.kind==error_kind::finite&&r.error.exponent2<=2-std::int64_t(s.pi_bits)),
        SBN3_FATAL_MATH,"inverse trig pi budget");
    s.pi_error=ErrorMagnitude::from(r.error);s.pi_ready=true;
}
struct Angle {float_view value{};ErrorMagnitude error{};bool nonzero=false;};
Angle pi_fraction(State& s,unsigned numerator,unsigned denominator_shift,bool negative,execution ex) noexcept {
    compute_pi(s,ex);auto value=real_scaled(v(s,10),-std::int64_t(denominator_shift));
    auto error=s.pi_error.scale2(-std::int64_t(denominator_shift));
    if(numerator!=1){const std::uint64_t n=numerator;
        const auto r=real_mul(s.arithmetic,s.bank[9].ref(),value,float_view::from_magnitude({&n,1},0),ex);good(r);
        value=v(s,9);error=error.multiply(ErrorMagnitude::from_scaled_uint64(numerator)).add(ErrorMagnitude::from(r.error));}
    return {negative?value.negated():value,error,true};
}
// Exact exceptional values that do not require pi or numerical preparation.
bool trivial(float_ref out,float_view a,float_view b,InverseTrigKind kind,float_report& r) noexcept {
    if(a.is_nan()||(kind==InverseTrigKind::atan2&&b.is_nan())){r=special(out,float_class::quiet_nan,false);return true;}
    if(kind==InverseTrigKind::asin||kind==InverseTrigKind::acos){
        const std::uint64_t one=1;const auto unit=float_view::from_magnitude({&one,1},0);
        if(!a.finite()||compare(abs(a),unit)==std::partial_ordering::greater){r=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(kind==InverseTrigKind::acos&&!a.negative()&&compare(a,unit)==std::partial_ordering::equivalent){r=special(out,float_class::finite,false);return true;}
        if(kind==InverseTrigKind::asin&&a.zero()){r=special(out,float_class::finite,a.negative());return true;}
    }else if(kind==InverseTrigKind::atan){if(a.zero()){r=special(out,float_class::finite,a.negative());return true;}}
    else if(kind==InverseTrigKind::atan2){
        if((a.zero()&&!b.negative())||(!b.finite()&&!b.negative()&&a.finite())){
            r=special(out,float_class::finite,a.negative());return true;}
    }
    return false;
}
Angle atan2_points(State& s,float_view y,float_view x,execution ex) noexcept {
    const bool negative=y.negative();
    if(!y.finite()&&!x.finite())return pi_fraction(s,x.negative()?3:1,2,negative,ex);
    if(y.zero())return pi_fraction(s,1,0,negative,ex); // positive-x zeros were exact shortcuts
    if(!y.finite()||x.zero())return pi_fraction(s,1,1,negative,ex);
    if(!x.finite())return pi_fraction(s,1,0,negative,ex); // remaining x is negative
    prefix(s.bank[0].ref(),abs(y));prefix(s.bank[1].ref(),abs(x));
    const bool swap=compare(v(s,0),v(s,1))==std::partial_ordering::greater;
    const auto numerator=v(s,swap?1:0),denominator=v(s,swap?0:1);
    const auto division=real_divide(s.arithmetic,s.bank[2].ref(),numerator,denominator,ex);good(division,true);
    // Prefixes preserve component signs and have relative error <2^(1-W).
    // Ratio-proportional transport preserves angles near the positive axis.
    auto error=ErrorMagnitude::from(division.error).add(ErrorMagnitude::power_of_two(
        checked(__int128(real_top(numerator))-real_top(denominator)+5-__int128(s.working))));
    float_view angle;
    if(v(s,2).zero())angle=v(s,2);
    else if(2*__int128(real_top(v(s,2)))<=-__int128(s.precision)-40){
        angle=v(s,2);error=error.add(ErrorMagnitude::power_of_two(checked(3*__int128(real_top(angle))-1)));
    }else{
        compute_pi(s,ex);
        const auto r=atan_agm_execute_with_pi(s.atan,s.bank[3].ref(),v(s,2),v(s,10),s.pi_error.publish(),ex,rounding::nearest_even);good(r);
        angle=v(s,3);error=error.add(ErrorMagnitude::from(r.error));
    }
    if(swap){compute_pi(s,ex);const auto r=real_add(s.arithmetic,s.bank[9].ref(),real_scaled(v(s,10),-1),angle.negated());good(r);
        angle=v(s,9);error=error.add(s.pi_error.scale2(-1)).add(ErrorMagnitude::from(r.error));}
    if(x.negative()){compute_pi(s,ex);const auto r=real_add(s.arithmetic,s.bank[9].ref(),v(s,10),angle.negated());good(r);
        angle=v(s,9);error=error.add(s.pi_error).add(ErrorMagnitude::from(r.error));}
    return {negative?angle.negated():angle,error,true};
}
Angle endpoint(State& s,float_view x,execution ex) noexcept {
    const std::uint64_t one=1;const auto unit=float_view::from_magnitude({&one,1},0);
    if(compare(abs(x),unit)==std::partial_ordering::equivalent)
        return pi_fraction(s,1,s.kind==InverseTrigKind::asin?1:0,s.kind==InverseTrigKind::asin&&x.negative(),ex);
    if(s.kind==InverseTrigKind::asin&&2*__int128(real_top(x))<=-__int128(s.precision)-40){
        auto e=prefix(s.bank[9].ref(),x);e=e.add(ErrorMagnitude::power_of_two(checked(3*__int128(real_top(x)))));
        return {v(s,9),e,true};
    }
    // These sums consume ORIGINAL x. Prefixing x first would erase a small
    // endpoint distance such as 1-(1-2^-N).
    const auto ur=real_add(s.arithmetic,s.bank[4].ref(),unit,x.negated());good(ur);
    const auto vr=real_add(s.arithmetic,s.bank[5].ref(),unit,x);good(vr);
    const auto eu=ErrorMagnitude::from(ur.error),ev=ErrorMagnitude::from(vr.error);
    auto ep=eu.multiply(real_upper(v(s,5))).add(ev.multiply(real_upper(v(s,4)))).add(eu.multiply(ev));
    const auto pr=real_mul(s.arithmetic,s.bank[6].ref(),v(s,4),v(s,5),ex);good(pr);ep=ep.add(ErrorMagnitude::from(pr.error));
    const auto sr=real_root(s.arithmetic,s.bank[7].ref(),v(s,6),ex);good(sr);
    const auto root_error=ErrorMagnitude::from(sr.error).add(ep.scale2(checked(2-__int128(real_top(v(s,7))))));
    const auto bound=root_error.publish();require(bound.exact()||(bound.kind==error_kind::finite&&bound.exponent2<=-4),
        SBN3_FATAL_MATH,"inverse trig root perturbation");
    // Along the root perturbation path the squared radius exceeds 1/2.
    // Both stable formulas have sensitivity <=2*|x| to that root.
    const auto transported=root_error.multiply(real_upper(x)).scale2(1);
    auto result=s.kind==InverseTrigKind::asin?atan2_points(s,x,v(s,7),ex):atan2_points(s,v(s,7),x,ex);
    result.error=result.error.add(transported);return result;
}
bool supported_input(float_view x) noexcept {
    if(x.magnitude().size()>float_precision_limit/64)return false;
    if(!x.finite()||x.zero())return true;
    const auto t=__int128(x.exponent2())+magnitude_bits(x.magnitude());return t>=-float_exponent_limit&&t<=float_exponent_limit;
}
void io(std::span<std::byte> storage,float_ref out,float_view a,float_view b) noexcept {
    validate(a);validate(b);check_output(out,a,b);const auto* d=detail::float_access::begin(out);
    for(auto x:{a,b})require(!overlaps(storage.data(),storage.size(),x.magnitude().data(),x.magnitude().size()*8),SBN3_FATAL_ARGUMENT,"inverse trig input/storage alias");
    require(!overlaps(storage.data(),storage.size(),d->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"inverse trig output/storage alias");
}
void constant_alias(ConstantBankBinding binding,float_ref out) noexcept {
    constant_bank_check_output(binding,out);
}
struct Use {State& s;explicit Use(State& x):s(x){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent inverse trig");}
    ~Use(){__atomic_store_n(&s.active,0,__ATOMIC_RELEASE);}};
}
ConstantBankDemand inverse_trig_constant_demand(bits p,InverseTrigKind kind) noexcept {
    if(!p.count||p.count>float_precision_limit/16||!valid_kind(kind))return {};
    const auto w=2*p.count+256;
    return constant_bank_merge({2*w+128,0},trig_agm_constant_demand(bits{w},TrigAgmKind::atan));
}
requirements inverse_trig_program_requirements(bits p,unsigned workers,InverseTrigKind kind,ConstantBankBinding binding) noexcept{return quote(p,workers,kind,binding).need;}
FloatInverseTrigPrepared inverse_trig_prepare(bits p,unsigned workers,InverseTrigKind kind,std::span<std::byte> storage,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"inverse trig query-only bank");
    const auto q=quote(p,workers,kind,binding);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),SBN3_FATAL_WORKSPACE,"inverse trig storage",q.need.scratch_bytes,storage.size());
    const auto values=binding.values().storage;
    require(!overlaps(storage.data(),storage.size(),values.data(),values.size()),SBN3_FATAL_ARGUMENT,"inverse trig bank/storage alias");
    auto frame=Frame::external(storage.data(),storage.size());auto* s=::new(frame.allocate(sizeof(State),128))State{};
    s->kind=kind;s->precision=p.count;s->working=q.working;s->pi_bits=q.pi_bits;s->workers=workers;s->bytes=q.need.scratch_bytes;
    s->bank=static_cast<float_buffer*>(frame.allocate(11*sizeof(float_buffer),64));
    for(unsigned j=0;j<11;++j){const auto precision=j==10?q.pi_bits:q.working,n=(precision+63)/64;
        ::new(s->bank+j)float_buffer({frame.alloc<std::uint64_t>(n),n},bits{precision});}
    {auto child=frame.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);real_arithmetic_prepare(s->arithmetic,q.arithmetic,child);}
    auto region=[&](std::size_t n,std::size_t a){return std::span<std::byte>{static_cast<std::byte*>(frame.allocate(n,a)),n};};
    s->constants=binding;s->owns_bank=q.owns_bank;
    if(q.owns_bank){const auto bank_values=region(q.bank.value_bytes,q.bank.value_alignment);
        const auto bank_work=region(q.bank.workspace_bytes,q.bank.workspace_alignment);
        s->owned_bank=constant_bank_prepare(q.bank,bank_values,bank_work);s->constants=constant_bank_pending_binding(s->owned_bank);}
    s->atan=trig_agm_prepare(bits{q.working},workers,TrigAgmKind::atan,region(q.atan.scratch_bytes,q.atan.scratch_alignment),true,s->constants);
    return {s,storage.first(q.need.scratch_bytes)};
}
float_report inverse_trig_execute(const FloatInverseTrigPrepared& prepared,float_ref out,float_view a,float_view b,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"inverse trig state");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"inverse trig prepared identity");
    if(s.kind!=InverseTrigKind::atan2)b={};io(prepared.storage,out,a,b);constant_alias(s.constants,out);Use use(s);float_report report;
    if(trivial(out,a,b,s.kind,report))return report;
    Angle result;
    if(s.kind==InverseTrigKind::atan){const std::uint64_t one=1;result=atan2_points(s,a,float_view::from_magnitude({&one,1},0),ex);}
    else if(s.kind==InverseTrigKind::atan2)result=atan2_points(s,a,b,ex);
    else result=endpoint(s,a,ex);
    report=real_finish(out,result.value,result.error,mode);
    if(out.view().zero()&&result.nonzero)report.flags=report.flags|float_flag::underflow|float_flag::rounded;
    if(out.view().finite()&&!out.view().zero()){
        const auto limit=checked(__int128(real_top(out.view()))-s.precision+(mode==rounding::nearest_even?0:1));
        require(report.error.exact()||(report.error.kind==error_kind::finite&&report.error.exponent2<=limit),
            SBN3_FATAL_MATH,"inverse trig final ulp budget");}
    return report;
}
requirements inverse_trig_requirements(float_ref out,float_view a,float_view b,unsigned workers,InverseTrigKind kind,ConstantBankBinding binding) noexcept {
    if(kind!=InverseTrigKind::atan2)b={};
    if(!supported_input(a)||!supported_input(b))return {support::unsupported,0,128,0,"inverse trig input"};
    auto q=quote(out.precision(),workers,kind,binding).need;
    if(out.capacity()<(out.precision().count+63)/64){q.status=support::capacity;q.reason="inverse trig output capacity";}return q;
}
float_report inverse_trig_into(float_ref out,float_view a,float_view b,execution ex,rounding mode,InverseTrigKind kind,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"inverse trig query-only bank");
    if(kind!=InverseTrigKind::atan2)b={};io(ex.scratch,out,a,b);constant_alias(binding,out);float_report report;
    require(valid_kind(kind),SBN3_FATAL_ARGUMENT,"inverse trig kind");if(trivial(out,a,b,kind,report))return report;
    const auto prepared=inverse_trig_prepare(out.precision(),ex.workers(),kind,ex.scratch,binding);
    return inverse_trig_execute(prepared,out,a,b,ex,mode);
}
}
namespace sbn {
#define SBN_INVERSE_TRIG_UNARY(name) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers)noexcept{ \
    return v3::floating::inverse_trig_requirements(out,x,{},workers,v3::floating::InverseTrigKind::name); } \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode)noexcept{ \
    return v3::floating::inverse_trig_into(out,x,{},ex,mode,v3::floating::InverseTrigKind::name); }
SBN_INVERSE_TRIG_UNARY(atan)
SBN_INVERSE_TRIG_UNARY(asin)
SBN_INVERSE_TRIG_UNARY(acos)
#undef SBN_INVERSE_TRIG_UNARY
requirements atan2_requirements(float_ref out,float_view y,float_view x,unsigned workers)noexcept{
    return v3::floating::inverse_trig_requirements(out,y,x,workers,v3::floating::InverseTrigKind::atan2);
}
float_report atan2_into(float_ref out,float_view y,float_view x,execution ex,rounding mode)noexcept{
    return v3::floating::inverse_trig_into(out,y,x,ex,mode,v3::floating::InverseTrigKind::atan2);
}
}
