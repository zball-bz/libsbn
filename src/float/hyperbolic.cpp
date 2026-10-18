// Original libsbn code; LGPL-3.0-or-later.
#include "float/hyperbolic.hpp"
#include "float/constants.hpp"
#include "float/exponential.hpp"
#include "float/logarithm.hpp"
#include <algorithm>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3348595031);
struct Ball {float_view value{};ErrorMagnitude error{};};
struct State {
    std::uint64_t marker=magic;std::uint32_t active=0;HyperbolicKind kind=HyperbolicKind::sinh;
    std::size_t precision=0,working=0,bytes=0;unsigned workers=1;bool constant_ready=false,owns_bank=false,bank_ready=false;
    RealArithmetic arithmetic{};FloatExponentialPrepared exponential{};FloatLogarithmPrepared logarithm{};
    ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};ErrorMagnitude log2_error{};float_buffer* bank=nullptr;
};
struct Quote {requirements need{},exp{},log{};ConstantBankQuote bank{};RealArithmeticPlan arithmetic{};std::size_t working=0;
    bool has_exp=false,has_log=false,owns_bank=false;};
bool valid_kind(HyperbolicKind k) noexcept{return k==HyperbolicKind::sinh||k==HyperbolicKind::cosh||k==HyperbolicKind::tanh||
    k==HyperbolicKind::asinh||k==HyperbolicKind::acosh||k==HyperbolicKind::atanh;}
Quote quote(bits p,unsigned workers,HyperbolicKind kind,ConstantBankBinding binding) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit/16||!workers||workers>32||!valid_kind(kind)){
        q.need.status=support::unsupported;q.need.reason="hyperbolic precision/workers";return q;}
    q.working=2*p.count+256;q.has_exp=kind==HyperbolicKind::sinh||kind==HyperbolicKind::cosh||kind==HyperbolicKind::tanh;
    q.has_log=!q.has_exp;
    const auto demand=hyperbolic_constant_demand(p,kind);auto children=binding;
    if(binding.supplied()){
        if(!constant_bank_supports(binding,demand)){q.need.status=support::unsupported;q.need.reason="hyperbolic constant bank";return q;}
    }else{
        q.owns_bank=true;q.bank=constant_bank_requirements(demand,workers);
        if(!q.bank.supported()){q.need.status=support::unsupported;q.need.reason=q.bank.reason;return q;}
        children=constant_bank_query_binding({q.bank.pi_bits,q.bank.log2_bits});
    }
    q.arithmetic=real_arithmetic_query(bits{q.working},workers);
    if(q.has_exp)q.exp=exponential_program_requirements(bits{q.working},workers,children);
    if(q.has_log)q.log=logarithm_program_requirements(bits{q.working},workers,children);
    if(!q.arithmetic.need.supported()||!q.exp.supported()||!q.log.supported()){
        q.need.status=support::unsupported;q.need.reason="hyperbolic child shape";return q;}
    q.need.scratch_alignment=std::max({std::size_t(128),q.arithmetic.need.scratch_alignment,q.exp.scratch_alignment,q.log.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment});
    auto& n=q.need.scratch_bytes;bool ok=real_append(n,sizeof(State),128)&&real_append(n,12*sizeof(float_buffer),64);
    for(unsigned j=0;j<12&&ok;++j)ok=real_append(n,((q.working+63)/64)*8,64);
    if(q.owns_bank)ok=ok&&real_append(n,q.bank.value_bytes,q.bank.value_alignment)&&real_append(n,q.bank.workspace_bytes,q.bank.workspace_alignment);
    ok=ok&&real_append(n,q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment)&&
        real_append(n,q.exp.scratch_bytes,q.exp.scratch_alignment)&&real_append(n,q.log.scratch_bytes,q.log.scratch_alignment);
    if(!ok){q.need.status=support::unsupported;q.need.reason="hyperbolic storage extent";}return q;
}
std::int64_t checked(__int128 x) noexcept {require(x>=INT64_MIN&&x<=INT64_MAX,SBN3_FATAL_SIZE,"hyperbolic exponent");return std::int64_t(x);}
float_view abs(float_view x) noexcept{return x.negative()?x.negated():x;}
float_view v(State& s,unsigned j) noexcept{return s.bank[j].view();}
Ball scaled(Ball a,std::int64_t n) noexcept{return {real_scaled(a.value,n),a.error.scale2(n)};}
Ball negated(Ball a) noexcept {a.value=a.value.negated();return a;}
void good(float_report r) noexcept {require(r.error.kind!=error_kind::unbounded&&!has_flag(r.flags,float_flag::invalid)&&
    !has_flag(r.flags,float_flag::overflow)&&!has_flag(r.flags,float_flag::underflow)&&!has_flag(r.flags,float_flag::divide_by_zero),
    SBN3_FATAL_MATH,"hyperbolic internal arithmetic");}
bool below(ErrorMagnitude e,std::int64_t exponent) noexcept {const auto b=e.publish();return b.exact()||(b.kind==error_kind::finite&&b.exponent2<=exponent);}
Ball prefix(State& s,unsigned out,float_view x) noexcept {
    if(x.zero()){special(s.bank[out].ref(),float_class::finite,x.negative());return {v(s,out),{}};}
    const auto a=x.magnitude();const auto length=magnitude_bits(a),cut=length>s.working?length-s.working:0;
    const auto n=(length-cut+63)/64,first=cut/64;const auto shift=unsigned(cut%64);auto* d=detail::float_access::begin(s.bank[out].ref());
    for(std::size_t j=0;j<n;++j){const auto i=first+j;d->data[j]=(a.data()[i]>>shift)|(shift&&i+1<a.size()?a.data()[i+1]<<(64-shift):0);}
    d->size=n;d->negative=0;auto& m=detail::float_access::metadata(s.bank[out].ref());m.classification=float_class::finite;
    m.negative=x.negative();m.exponent2=checked(__int128(x.exponent2())+cut);detail::float_access::finish(s.bank[out].ref());
    return {v(s,out),cut?ErrorMagnitude::power_of_two(m.exponent2):ErrorMagnitude{}};
}
Ball add(State& s,unsigned out,Ball a,Ball b) noexcept {
    const auto e=a.error.add(b.error);const auto r=real_add(s.arithmetic,s.bank[out].ref(),a.value,b.value);good(r);
    return {v(s,out),e.add(ErrorMagnitude::from(r.error))};
}
Ball multiply(State& s,unsigned out,Ball a,Ball b,execution ex) noexcept {
    const auto e=real_upper(a.value).multiply(b.error).add(real_upper(b.value).multiply(a.error)).add(a.error.multiply(b.error));
    const auto r=real_mul(s.arithmetic,s.bank[out].ref(),a.value,b.value,ex);good(r);return {v(s,out),e.add(ErrorMagnitude::from(r.error))};
}
Ball divide(State& s,unsigned out,Ball a,Ball b,execution ex) noexcept {
    const auto t=real_top(b.value);require(below(b.error,t-2),SBN3_FATAL_MATH,"hyperbolic denominator radius");
    const auto e=a.error.scale2(checked(2-__int128(t))).add(real_upper(a.value).multiply(b.error).scale2(checked(3-2*__int128(t))));
    const auto r=real_divide(s.arithmetic,s.bank[out].ref(),a.value,b.value,ex);good(r);return {v(s,out),e.add(ErrorMagnitude::from(r.error))};
}
Ball root(State& s,unsigned out,Ball a,execution ex) noexcept {
    const auto r=real_root(s.arithmetic,s.bank[out].ref(),a.value,ex);good(r);
    require(!v(s,out).zero(),SBN3_FATAL_MATH,"hyperbolic positive root");
    return {v(s,out),ErrorMagnitude::from(r.error).add(a.error.scale2(checked(2-__int128(real_top(v(s,out))))))};
}
Ball logarithm(State& s,unsigned out,Ball a,LogarithmKind kind,execution ex) noexcept {
    // Positive arguments only. Transport by the lower bound for a or 1+a;
    // this is essential for the enormous atanh argument near a pole.
    const auto lower=kind==LogarithmKind::log?real_top(a.value)-1:
        a.value.zero()?0:std::max(std::int64_t(0),real_top(a.value)-1);
    require(below(a.error,lower-2),SBN3_FATAL_MATH,"hyperbolic logarithm radius");
    const auto e=a.error.scale2(checked(1-__int128(lower)));
    const auto r=logarithm_program_execute(s.logarithm,s.bank[out].ref(),a.value,kind,ex);good(r);
    return {v(s,out),e.add(ErrorMagnitude::from(r.error))};
}
void ensure_bank(State& s,execution ex) noexcept {
    if(s.owns_bank&&!s.bank_ready){s.constants=ConstantBankBinding(constant_bank_execute(s.owned_bank,ex));s.bank_ready=true;}
}
Ball constant(State& s,execution ex) noexcept {
    ensure_bank(s,ex);
    if(!s.constant_ready){const auto r=constant_bank_prefix(s.constants,ConstantKind::log2,s.bank[11].ref());good(r);
        s.log2_error=ErrorMagnitude::from(r.error);s.constant_ready=true;}
    return {v(s,11),s.log2_error};
}
bool special_case(float_ref out,float_view x,HyperbolicKind kind,rounding mode,float_report& r) noexcept {
    const std::uint64_t one=1;const auto unit=float_view::from_magnitude({&one,1},0);
    if(x.is_nan()){r=special(out,float_class::quiet_nan,false);return true;}
    if(kind==HyperbolicKind::acosh){
        if(x.negative()||compare(x,unit)==std::partial_ordering::less){r=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(compare(x,unit)==std::partial_ordering::equivalent){r=special(out,float_class::finite,false);return true;}
    }
    if(kind==HyperbolicKind::atanh){const auto cmp=compare(abs(x),unit);
        if(cmp==std::partial_ordering::greater){r=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
        if(cmp==std::partial_ordering::equivalent){r=special(out,float_class::infinity,x.negative(),float_flag::divide_by_zero);return true;}}
    if(!x.finite()){
        if(kind==HyperbolicKind::tanh)r=round_into(out,x.negative()?unit.negated():unit,mode);
        else r=special(out,float_class::infinity,kind==HyperbolicKind::cosh?false:x.negative());return true;
    }
    if(x.zero()){r=kind==HyperbolicKind::cosh?round_into(out,unit,mode):special(out,float_class::finite,x.negative());return true;}
    if((kind==HyperbolicKind::sinh||kind==HyperbolicKind::cosh)&&real_top(x)>=61){
        r=special(out,float_class::infinity,kind==HyperbolicKind::sinh&&x.negative(),float_flag::overflow|float_flag::rounded);return true;}
    return false;
}
std::uint64_t floor_word(float_view x) noexcept {
    require(!x.negative()&&real_top(x)<=60,SBN3_FATAL_ARGUMENT,"hyperbolic bounded integer");const auto a=x.magnitude();
    if(x.exponent2()>=0)return a.data()[0]<<unsigned(x.exponent2());
    const auto bit=std::uint64_t(0)-std::uint64_t(x.exponent2());const auto i=std::size_t(bit/64);const auto shift=unsigned(bit%64);
    return i<a.size()?(a.data()[i]>>shift)|(shift&&i+1<a.size()?a.data()[i+1]<<(64-shift):0):0;
}
Ball calculate(State& s,float_view input,execution ex,bool& overflow) noexcept {
    const bool negative=input.negative();const auto a=abs(input);const auto E=real_top(a);
    const std::uint64_t one_word=1,two_word=2,threshold=s.precision+64;
    const Ball one{float_view::from_magnitude({&one_word,1},0),{}},two{float_view::from_magnitude({&two_word,1},0),{}};
    Ball result;
    if(s.kind!=HyperbolicKind::acosh&&2*__int128(E)<=-__int128(s.precision)-48){
        if(s.kind==HyperbolicKind::cosh){round_into(s.bank[9].ref(),one.value);return {v(s,9),ErrorMagnitude::power_of_two(checked(2*__int128(E)))};}
        result=prefix(s,9,input);result.error=result.error.add(ErrorMagnitude::power_of_two(checked(3*__int128(E))));return result;
    }
    const bool large_exp=compare(a,float_view::from_magnitude({&threshold,1},0))!=std::partial_ordering::less;
    if(s.kind==HyperbolicKind::tanh&&large_exp){round_into(s.bank[9].ref(),negative?one.value.negated():one.value);
        return {v(s,9),ErrorMagnitude::power_of_two(-2*std::int64_t(s.precision)-127)};}
    // Parent publishes once before any exp/log child can borrow its constants.
    ensure_bank(s,ex);
    if((s.kind==HyperbolicKind::sinh||s.kind==HyperbolicKind::cosh)&&large_exp){
        const auto argument=add(s,8,{a,{}},negated(constant(s,ex)));
        require(below(argument.error,-2),SBN3_FATAL_MATH,"hyperbolic exponential argument radius");
        const auto r=exponential_program_execute(s.exponential,s.bank[9].ref(),argument.value,ExponentialKind::exp,ex);
        if(!v(s,9).finite()){overflow=true;return {};}
        good(r);result={v(s,9),ErrorMagnitude::from(r.error).add(real_upper(v(s,9)).multiply(argument.error).scale2(2))
            .add(ErrorMagnitude::power_of_two(-std::int64_t(floor_word(a))-1))};
    }else if(s.kind==HyperbolicKind::sinh||s.kind==HyperbolicKind::cosh){
        const auto er=exponential_program_execute(s.exponential,s.bank[1].ref(),a,ExponentialKind::expm1,ex);good(er);
        // First evaluate the rational identities at the computed exact u.
        const Ball u{v(s,1),{}};const auto w=add(s,3,u,one),reciprocal=divide(s,2,one,w,ex);
        const auto t=add(s,4,u,two),numerator=multiply(s,5,u,t,ex);
        const auto sh=scaled(multiply(s,6,numerator,reciprocal,ex),-1);
        result=s.kind==HyperbolicKind::sinh?sh:add(s,7,sh,reciprocal);
        // d sinh/du <=1 and d cosh/du <=1/2 for u>=0.
        result.error=result.error.add(ErrorMagnitude::from(er.error));
    }else if(s.kind==HyperbolicKind::tanh){
        const auto er=exponential_program_execute(s.exponential,s.bank[1].ref(),real_scaled(a,1),ExponentialKind::expm1,ex);good(er);
        const Ball u{v(s,1),{}};const auto denominator=add(s,2,u,two);result=divide(s,9,u,denominator,ex);
        const auto eu=ErrorMagnitude::from(er.error);const auto t=real_top(denominator.value);
        require(below(eu.add(denominator.error),t-2),SBN3_FATAL_MATH,"tanh denominator perturbation");
        result.error=result.error.add(eu.scale2(checked(5-2*__int128(t))));
    }else if((s.kind==HyperbolicKind::asinh||s.kind==HyperbolicKind::acosh)&&2*__int128(E)>=__int128(s.precision)+80){
        const auto l=logarithm(s,9,{a,{}},LogarithmKind::log,ex);result=add(s,9,l,constant(s,ex));
        result.error=result.error.add(ErrorMagnitude::power_of_two(checked((s.kind==HyperbolicKind::acosh?2:0)-2*__int128(E))));
    }else if(s.kind==HyperbolicKind::asinh){
        const auto x=prefix(s,0,a),square=multiply(s,1,x,x,ex),radicand=add(s,2,square,one);
        const auto r=root(s,3,radicand,ex),denominator=add(s,4,r,one),term=divide(s,5,square,denominator,ex);
        result=logarithm(s,9,add(s,6,x,term),LogarithmKind::log1p,ex);
    }else if(s.kind==HyperbolicKind::acosh){
        // Original exact x is consumed by subtraction before any prefix.
        const auto delta=add(s,1,{input,{}},negated(one));const auto factor=add(s,2,delta,two);
        const auto r=root(s,4,multiply(s,3,delta,factor,ex),ex);
        result=logarithm(s,9,add(s,5,delta,r),LogarithmKind::log1p,ex);
    }else{
        // The endpoint distance is formed before truncating abs(x).
        const auto delta=add(s,1,one,negated({a,{}}));const auto x=prefix(s,0,a);
        const auto argument=divide(s,2,scaled(x,1),delta,ex);
        result=scaled(logarithm(s,9,argument,LogarithmKind::log1p,ex),-1);
    }
    if(negative&&s.kind!=HyperbolicKind::cosh&&s.kind!=HyperbolicKind::acosh)result.value=result.value.negated();return result;
}
void io(std::span<std::byte> storage,float_ref out,float_view x) noexcept {
    validate(x);check_output(out,x);const auto* o=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),x.magnitude().data(),x.magnitude().size()*8)&&
        !overlaps(storage.data(),storage.size(),o->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"hyperbolic storage alias");
}
void constant_alias(ConstantBankBinding binding,float_ref out) noexcept {
    constant_bank_check_output(binding,out);
}
struct Use {State& s;explicit Use(State& x):s(x){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent hyperbolic");}
    ~Use(){__atomic_store_n(&s.active,0,__ATOMIC_RELEASE);}};
}
ConstantBankDemand hyperbolic_constant_demand(bits p,HyperbolicKind kind) noexcept {
    if(!p.count||p.count>float_precision_limit/16||!valid_kind(kind))return {};
    const auto w=bits{2*p.count+256};
    const bool exp=kind==HyperbolicKind::sinh||kind==HyperbolicKind::cosh||kind==HyperbolicKind::tanh;
    auto demand=exp?exponential_constant_demand(w):logarithm_constant_demand(w);
    if(kind!=HyperbolicKind::tanh&&kind!=HyperbolicKind::atanh)demand=constant_bank_merge(demand,{0,w.count});return demand;
}
requirements hyperbolic_program_requirements(bits p,unsigned workers,HyperbolicKind kind,ConstantBankBinding binding) noexcept{return quote(p,workers,kind,binding).need;}
FloatHyperbolicPrepared hyperbolic_prepare(bits p,unsigned workers,HyperbolicKind kind,std::span<std::byte> storage,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"hyperbolic query-only bank");
    const auto q=quote(p,workers,kind,binding);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),SBN3_FATAL_WORKSPACE,"hyperbolic storage",q.need.scratch_bytes,storage.size());
    const auto values=binding.values().storage;
    require(!overlaps(storage.data(),storage.size(),values.data(),values.size()),SBN3_FATAL_ARGUMENT,"hyperbolic bank/storage alias");
    auto f=Frame::external(storage.data(),storage.size());auto* s=::new(f.allocate(sizeof(State),128))State{};
    s->kind=kind;s->precision=p.count;s->working=q.working;s->workers=workers;s->bytes=q.need.scratch_bytes;
    s->bank=static_cast<float_buffer*>(f.allocate(12*sizeof(float_buffer),64));const auto n=(q.working+63)/64;
    for(unsigned j=0;j<12;++j)::new(s->bank+j)float_buffer({f.alloc<std::uint64_t>(n),n},bits{q.working});
    auto region=[&](requirements r){return std::span<std::byte>{static_cast<std::byte*>(f.allocate(r.scratch_bytes,r.scratch_alignment)),r.scratch_bytes};};
    s->constants=binding;s->owns_bank=q.owns_bank;
    if(q.owns_bank){const auto bank_values=std::span<std::byte>{static_cast<std::byte*>(f.allocate(q.bank.value_bytes,q.bank.value_alignment)),q.bank.value_bytes};
        const auto bank_work=std::span<std::byte>{static_cast<std::byte*>(f.allocate(q.bank.workspace_bytes,q.bank.workspace_alignment)),q.bank.workspace_bytes};
        s->owned_bank=constant_bank_prepare(q.bank,bank_values,bank_work);s->constants=constant_bank_pending_binding(s->owned_bank);}
    {auto child=f.subframe(q.arithmetic.need.scratch_bytes,q.arithmetic.need.scratch_alignment);real_arithmetic_prepare(s->arithmetic,q.arithmetic,child);}
    if(q.has_exp)s->exponential=exponential_program_prepare(bits{q.working},workers,region(q.exp),s->constants);
    if(q.has_log)s->logarithm=logarithm_program_prepare(bits{q.working},workers,region(q.log),s->constants);
    return {s,storage.first(q.need.scratch_bytes)};
}
float_report hyperbolic_execute(const FloatHyperbolicPrepared& prepared,float_ref out,float_view x,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"hyperbolic state");
    auto& s=*static_cast<State*>(prepared.state);require(s.marker==magic&&s.bytes==prepared.storage.size()&&out.precision().count==s.precision&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"hyperbolic prepared identity");io(prepared.storage,out,x);constant_alias(s.constants,out);Use use(s);float_report report;
    if(special_case(out,x,s.kind,mode,report))return report;bool overflow=false;const auto result=calculate(s,x,ex,overflow);
    if(overflow)return special(out,float_class::infinity,s.kind==HyperbolicKind::sinh&&x.negative(),float_flag::overflow|float_flag::rounded);
    report=real_finish(out,result.value,result.error,mode);
    if(out.view().finite()&&!out.view().zero()){
        const auto limit=checked(__int128(real_top(out.view()))-s.precision+(mode==rounding::nearest_even?0:1));
        require(report.error.exact()||(report.error.kind==error_kind::finite&&report.error.exponent2<=limit),SBN3_FATAL_MATH,"hyperbolic final ulp budget");}
    return report;
}
requirements hyperbolic_requirements(float_ref out,float_view,unsigned workers,HyperbolicKind kind,ConstantBankBinding binding) noexcept {
    auto q=quote(out.precision(),workers,kind,binding).need;if(out.capacity()<(out.precision().count+63)/64){q.status=support::capacity;q.reason="hyperbolic output capacity";}return q;
}
float_report hyperbolic_into(float_ref out,float_view x,execution ex,rounding mode,HyperbolicKind kind,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"hyperbolic query-only bank");
    io(ex.scratch,out,x);constant_alias(binding,out);require(valid_kind(kind),SBN3_FATAL_ARGUMENT,"hyperbolic kind");float_report report;
    if(special_case(out,x,kind,mode,report))return report;
    return hyperbolic_execute(hyperbolic_prepare(out.precision(),ex.workers(),kind,ex.scratch,binding),out,x,ex,mode);
}
}
namespace sbn {
#define SBN_HYPERBOLIC_UNARY(name) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers)noexcept{ \
    return v3::floating::hyperbolic_requirements(out,x,workers,v3::floating::HyperbolicKind::name); } \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode)noexcept{ \
    return v3::floating::hyperbolic_into(out,x,ex,mode,v3::floating::HyperbolicKind::name); }
SBN_HYPERBOLIC_UNARY(sinh)
SBN_HYPERBOLIC_UNARY(cosh)
SBN_HYPERBOLIC_UNARY(tanh)
SBN_HYPERBOLIC_UNARY(asinh)
SBN_HYPERBOLIC_UNARY(acosh)
SBN_HYPERBOLIC_UNARY(atanh)
#undef SBN_HYPERBOLIC_UNARY
}
