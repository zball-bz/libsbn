// Original libsbn code; LGPL-3.0-or-later.
#include "float/agm.hpp"
#include "float/diagnostics.hpp"
#include "float/product.hpp"
#include "float/roots.hpp"
#include "float/error.hpp"
#include "runtime/team.hpp"
#include <algorithm>
#include <bit>
#include <limits>
#include <new>
#include <utility>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t agm_magic=UINT64_C(0x53424e3341474d31);
struct AgmState {
    std::uint64_t marker=0;
    std::uint32_t active=0;
    unsigned workers=1;
    std::size_t precision=0,working=0,iteration_limit=0,bytes=0;
    FloatProductPlan product_plan{};
    FloatProductPrepared product{};
    FloatRootPrepared root{};
    std::span<std::byte> sum_storage{};
    float_buffer a,b,next_a,geometric;

    AgmState(std::size_t p,std::size_t q,std::size_t limit,unsigned threads,
             std::uint64_t* av,std::uint64_t* bv,std::uint64_t* cv,
             std::uint64_t* dv,std::size_t limbs) noexcept
        :workers(threads),precision(p),working(q),iteration_limit(limit),
         a({av,limbs},bits{q}),b({bv,limbs},bits{q}),
         next_a({cv,limbs},bits{q}),geometric({dv,limbs},bits{q}) {}
};
struct Quote {
    requirements need{};
    FloatProductPlan product{};
    requirements root{};
    std::size_t working=0,limbs=0,iterations=0,sum_bytes=0;
};
struct Use {
    AgmState& state;
    explicit Use(AgmState& s) noexcept:state(s) {
        std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&state.active,&expected,1,false,
                    __ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
                SBN3_FATAL_LIFETIME,"prepared AGM already active");
    }
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}
};
bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment) noexcept {
    if(!bytes)return true;
    std::size_t at=0;
    return align_size(cursor,alignment,at)&&add_size(at,bytes,cursor);
}
Quote quote(bits precision,unsigned workers) noexcept {
    Quote p;
    p.need.scratch_alignment=128;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32){
        p.need.status=support::unsupported;p.need.reason="AGM precision/workers";return p;
    }
    p.need.output_limbs=(precision.count+63)/64;
    // 66 steps cover the entire public exponent-ratio domain. Remaining
    // steps cover quadratic convergence, with five further safety steps.
    p.iterations=72+std::bit_width(precision.count+8);
    // 2*(N+1)*eta < 2^(-p-7), eta=2^(8-q). Together with the
    // Q/16 gap budget this fits below one final nearest result ulp.
    const auto guards=std::bit_width(p.iterations+1)+16;
    if(!add_size(precision.count,guards,p.working)||p.working>float_precision_limit){
        p.need.status=support::unsupported;p.need.reason="AGM working precision";return p;
    }
    p.limbs=(p.working+63)/64;
    if(float_product_query(p.working,p.working,p.working,false,workers,p.product)!=SBN3_SUPPORTED){
        p.need.status=support::unsupported;p.need.reason="AGM product shape";return p;
    }
    p.root=root_program_requirements(bits{p.working},workers,RootKind::square_root);
    if(!p.root.supported()){
        p.need.status=support::unsupported;p.need.reason=p.root.reason;return p;
    }
    p.need.scratch_alignment=std::max<std::size_t>(
        128,std::max(p.product.alignment,p.root.scratch_alignment));
    // sum_finite's shared exact finite-sum window: two guard limbs plus
    // alignment. Its <=256-bit path uses an internal fixed stack window.
    if(p.working>256)p.sum_bytes=(p.limbs+2)*8+63;
    std::size_t bytes=0,value_bytes=0;
    if(!mul_size(p.limbs,8,value_bytes)||
       !append(bytes,sizeof(AgmState),128)||
       !append(bytes,value_bytes,64)||!append(bytes,value_bytes,64)||
       !append(bytes,value_bytes,64)||!append(bytes,value_bytes,64)||
       !append(bytes,p.product.scratch_bytes,p.product.alignment)||
       !append(bytes,p.root.scratch_bytes,p.root.scratch_alignment)||
       !append(bytes,p.sum_bytes,64)){
        p.need.status=support::unsupported;p.need.reason="AGM storage extent";return p;
    }
    p.need.scratch_bytes=bytes;
    return p;
}
bool input_supported(float_view x) noexcept {
    if(x.magnitude().size()>float_precision_limit/64)return false;
    if(!x.finite()||x.zero())return true;
    const __int128 top=__int128(x.exponent2())+magnitude_bits(x.magnitude());
    return top>=-float_exponent_limit&&top<=float_exponent_limit;
}
bool trivial_inputs(float_view a,float_view b) noexcept {
    return a.is_nan()||b.is_nan()||
           (a.negative()&&!a.zero())||(b.negative()&&!b.zero())||
           a.zero()||b.zero()||!a.finite()||!b.finite()||
           compare(a,b)==std::partial_ordering::equivalent;
}
bool trivial(float_ref out,float_view a,float_view b,rounding mode,
             float_report& report) noexcept {
    if(a.is_nan()||b.is_nan()){
        report=special(out,float_class::quiet_nan,false);return true;
    }
    if((a.negative()&&!a.zero())||(b.negative()&&!b.zero())){
        report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;
    }
    if(a.zero()||b.zero()){
        report=(!a.finite()||!b.finite())
            ?special(out,float_class::quiet_nan,false,float_flag::invalid)
            :special(out,float_class::finite,false);
        return true;
    }
    if(!a.finite()||!b.finite()){
        report=special(out,float_class::infinity,false);return true;
    }
    if(compare(a,b)==std::partial_ordering::equivalent){
        report=round_into(out,a,mode);return true;
    }
    return false;
}
std::int64_t checked_exponent(__int128 e) noexcept {
    require(e>=std::numeric_limits<std::int64_t>::min()&&
            e<=std::numeric_limits<std::int64_t>::max(),
            SBN3_FATAL_SIZE,"AGM internal exponent");
    return std::int64_t(e);
}
std::int64_t top(float_view x) noexcept {
    require(x.finite()&&!x.zero()&&!x.negative(),SBN3_FATAL_MATH,"AGM positive state");
    return checked_exponent(__int128(x.exponent2())+magnitude_bits(x.magnitude()));
}
// Downward significant-prefix staging is safe even at the upper exponent
// boundary. Omitted tails are bounded analytically by <2^(top-working);
// no scan of potentially enormous discarded source ranges is necessary.
void stage_prefix(float_ref out,float_view x) noexcept {
    const auto input=x.magnitude();
    const auto width=magnitude_bits(input),precision=out.precision().count;
    const auto cut=width>precision?width-precision:0;
    const auto kept=width-cut,words=(kept+63)/64,first=cut/64;
    const unsigned shift=unsigned(cut%64);
    const auto exponent=checked_exponent(__int128(x.exponent2())+cut);
    auto* r=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){
        const auto at=first+i;
        r->data[i]=(input.data()[at]>>shift)|
            (shift&&at+1<input.size()?input.data()[at+1]<<(64-shift):0);
    }
    r->size=words;r->negative=0;
    auto& metadata=detail::float_access::metadata(out);
    metadata.exponent2=exponent;metadata.negative=false;
    metadata.classification=float_class::finite;
    detail::float_access::finish(out);
}
ErrorMagnitude magnitude_upper(float_view x) noexcept {
    if(x.zero())return {};
    return ErrorMagnitude::power_of_two(top(x));
}
bool below(ErrorMagnitude error,std::int64_t exponent) noexcept {
    const auto bound=error.publish();
    return bound.exact()||(bound.kind==error_kind::finite&&bound.exponent2<=exponent);
}
void storage_disjoint(std::span<std::byte> storage,float_ref out,
                      float_view a,float_view b) noexcept {
    for(auto x:{a,b})require(!overlaps(storage.data(),storage.size(),
        x.magnitude().data(),x.magnitude().size()*8),SBN3_FATAL_ARGUMENT,
        "AGM input/program storage alias");
    const auto* output=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),output->data,out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"AGM output/program storage alias");
}
float_view scaled_view(float_view x,std::int64_t shift) noexcept {
    return float_view::from_magnitude(x.magnitude().limbs(),
        checked_exponent(__int128(x.exponent2())+shift),false);
}
float_view normalized_view(float_view x) noexcept {
    return float_view::from_magnitude(x.magnitude().limbs(),
        -std::int64_t(magnitude_bits(x.magnitude())),false);
}
// Project onto [lower,upper] without passing an out-of-public-range candidate
// to compare(). The exact geometric mean is in that interval, so projection
// cannot increase its absolute approximation error.
float_view project(float_view candidate,float_view lower,float_view upper) noexcept {
    const auto ct=top(candidate),lt=top(lower),ut=top(upper);
    if(ct<lt)return lower;
    if(ct>ut)return upper;
    if(compare(candidate,lower)==std::partial_ordering::less)return lower;
    if(compare(candidate,upper)==std::partial_ordering::greater)return upper;
    return candidate;
}
float_report finish(float_ref out,float_view upper,ErrorMagnitude gap,
                    std::size_t steps,const AgmState& state,rounding mode) noexcept {
    const auto upper_top=top(upper);
    // k=steps+1 includes initial prefix staging. Homogeneity and monotonicity
    // give |M_original-M_current| <= 2*k*eta*M_current, eta=2^(8-working).
    const auto propagated=ErrorMagnitude::from_scaled_uint64(2*(steps+1),
        checked_exponent(__int128(upper_top)+8-__int128(state.working)));
    auto report=round_into(out,upper,mode);
    report.error=ErrorMagnitude::from(report.error).add(gap).add(propagated).publish();
    const auto result=out.view();
    if(result.finite()&&!result.zero()){
        const auto limit=checked_exponent(__int128(top(result))-state.precision+
                         (mode==rounding::nearest_even?0:1));
        require(report.error.kind==error_kind::finite&&report.error.exponent2<=limit,
                SBN3_FATAL_MATH,"AGM final error budget");
    }
    return report;
}
} // namespace

FloatDiagnostic agm_diagnostic(bits p,unsigned workers) noexcept {
    const auto q=quote(p,workers);FloatDiagnostic d;d.need=q.need;d.recipe="real_agm";
    d.working_bits=q.working;d.metadata_bytes=sizeof(AgmState);d.constant_value_bytes=0;d.constant_work_bytes=0;
    d.table_bytes=q.product.tables_bytes;return d;
}
requirements agm_program_requirements(bits precision,unsigned workers) noexcept {
    return quote(precision,workers).need;
}
FloatAgmPrepared agm_program_prepare(bits precision,unsigned workers,
                                    std::span<std::byte> storage) noexcept {
    const auto p=quote(precision,workers);
    require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    require(storage.size()>=p.need.scratch_bytes&&
            !(std::uintptr_t(storage.data())&(p.need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"AGM program storage",p.need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());
    auto* raw=frame.allocate(sizeof(AgmState),128);
    auto* a=frame.alloc<std::uint64_t>(p.limbs);
    auto* b=frame.alloc<std::uint64_t>(p.limbs);
    auto* c=frame.alloc<std::uint64_t>(p.limbs);
    auto* d=frame.alloc<std::uint64_t>(p.limbs);
    auto* state=new(raw)AgmState(precision.count,p.working,p.iterations,workers,a,b,c,d,p.limbs);
    state->bytes=p.need.scratch_bytes;
    state->product_plan=p.product;
    std::span<std::byte> product_storage{};
    if(p.product.scratch_bytes)product_storage={
        static_cast<std::byte*>(frame.allocate(p.product.scratch_bytes,p.product.alignment)),
        p.product.scratch_bytes};
    state->product=float_product_prepare(state->product_plan,product_storage);
    const auto root_storage=std::span<std::byte>{
        static_cast<std::byte*>(frame.allocate(p.root.scratch_bytes,p.root.scratch_alignment)),
        p.root.scratch_bytes};
    state->root=root_program_prepare(bits{p.working},workers,RootKind::square_root,root_storage);
    if(p.sum_bytes)state->sum_storage={
        static_cast<std::byte*>(frame.allocate(p.sum_bytes,64)),p.sum_bytes};
    state->marker=agm_magic;
    return {state,storage.first(p.need.scratch_bytes)};
}
static float_report execute_scoped(AgmState& state,float_ref out,
    float_view a,float_view b,execution ex,rounding mode) noexcept {
    stage_prefix(state.a.ref(),a);
    stage_prefix(state.b.ref(),b);
    const auto eta_exponent=checked_exponent(8-__int128(state.working));
    const execution sum_execution{state.sum_storage,nullptr,nullptr};
    for(std::size_t steps=0;steps<=state.iteration_limit;++steps){
        const auto av=state.a.view(),bv=state.b.view();
        const bool ordered=compare(av,bv)!=std::partial_ordering::less;
        const auto upper=ordered?av:bv,lower=ordered?bv:av;
        const auto gap_report=sum_finite(state.next_a.ref(),upper,lower.negated(),
                                        sum_execution,rounding::nearest_even);
        const auto gap=magnitude_upper(state.next_a.view()).add(
            ErrorMagnitude::from(gap_report.error));
        const auto stop=checked_exponent(__int128(top(upper))-state.precision-4);
        if(below(gap,stop))return finish(out,upper,gap,steps,state,mode);
        require(steps<state.iteration_limit,SBN3_FATAL_MATH,"AGM convergence bound");

        // Half-scale BEFORE summing, so an overflowing a+b is never formed.
        const auto mean_report=sum_finite(state.next_a.ref(),scaled_view(av,-1),
            scaled_view(bv,-1),sum_execution,rounding::nearest_even);
        const auto mean_relative=ErrorMagnitude::from(mean_report.error).scale2(
            checked_exponent(2-__int128(top(state.next_a.view()))));
        require(below(mean_relative,eta_exponent),SBN3_FATAL_MATH,"AGM mean error budget");

        const auto at=top(av),bt=top(bv);
        const auto sum=__int128(at)+bt;
        const auto half=checked_exponent(sum/2-(sum%2<0));
        const auto parity=std::int64_t(sum-2*__int128(half));
        const auto product_report=float_product_execute(state.product,state.geometric.ref(),
            normalized_view(av),normalized_view(bv),ex,rounding::nearest_even);
        const auto root_report=root_program_execute(state.root,state.geometric.ref(),
            scaled_view(state.geometric.view(),parity),ex,rounding::nearest_even);
        // True normalized geometric mean >= 1/2. The product error passes
        // through sqrt with absolute amplification <=2^(parity+1).
        const auto geometric_relative=ErrorMagnitude::from(root_report.error).add(
            ErrorMagnitude::from(product_report.error).scale2(parity+1)).scale2(1);
        require(below(geometric_relative,eta_exponent),
                SBN3_FATAL_MATH,"AGM geometric error budget");
        const auto geometric=project(scaled_view(state.geometric.view(),half),lower,upper);
        if(geometric.magnitude().data()==state.geometric.view().magnitude().data()){
            // The usual path restores only the exponent. Do not copy a
            // complete q-bit root back onto itself merely to change scale.
            detail::float_access::metadata(state.geometric.ref()).exponent2=geometric.exponent2();
            detail::float_access::finish(state.geometric.ref());
        }else{
            const auto copy=round_into(state.geometric.ref(),geometric);
            require(copy.error.exact(),SBN3_FATAL_MATH,"AGM geometric scale copy");
        }
        std::swap(state.a,state.next_a);
        std::swap(state.b,state.geometric);
    }
    fatal(SBN3_FATAL_MATH,"AGM iteration fallthrough");
}
float_report agm_program_execute(const FloatAgmPrepared& prepared,float_ref out,
    float_view a,float_view b,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(AgmState)&&
            !(std::uintptr_t(prepared.state)&127),SBN3_FATAL_LIFETIME,"prepared AGM storage");
    auto& state=*static_cast<AgmState*>(prepared.state);
    require(state.marker==agm_magic&&state.bytes==prepared.storage.size(),SBN3_FATAL_LIFETIME,"prepared AGM identity");
    require(out.precision().count==state.precision&&ex.workers()>=state.workers,SBN3_FATAL_ARGUMENT,"prepared AGM precision/workers");
    validate(a);validate(b);check_output(out,a,b);storage_disjoint(prepared.storage,out,a,b);Use exclusive(state);
    float_report report;if(trivial(out,a,b,mode,report))return report;
    if(ex.scope)return execute_scoped(state,out,a,b,ex,mode);
    struct Call {AgmState* state;float_ref out;float_view a,b;rounding mode;float_report result{};};
    Call call{&state,out,a,b,mode,{}};
    auto action=[](void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(opaque);
        c.result=execute_scoped(*c.state,c.out,c.a,c.b,execution{{},nullptr,scope},c.mode);};
    if(ex.team)sbn3_team_run(ex.team,action,&call);else serial_team_run(action,&call);
    return call.result;
}
} // namespace sbn::v3::floating

namespace sbn {
requirements agm_requirements(float_ref out,float_view a,float_view b,unsigned workers) noexcept {
    using namespace v3::floating;
    const auto p=out.precision().count,n=p/64+unsigned(p%64!=0);
    if(!p||p>float_precision_limit||!workers||workers>32)
        return {support::unsupported,0,128,n,"AGM precision/workers"};
    if(out.capacity()<n)return {support::capacity,0,128,n,"AGM output capacity"};
    if(!input_supported(a)||!input_supported(b))
        return {support::unsupported,0,128,n,"AGM input domain"};
    if(trivial_inputs(a,b))return {support::yes,0,128,n,""};
    return agm_program_requirements(bits{p},workers);
}
float_report agm_into(float_ref out,float_view a,float_view b,execution ex,rounding mode) noexcept {
    using namespace v3::floating;
    validate(a);validate(b);check_output(out,a,b);
    float_report report;
    if(trivial(out,a,b,mode,report))return report;
    storage_disjoint(ex.scratch,out,a,b);
    const auto prepared=agm_program_prepare(out.precision(),ex.workers(),ex.scratch);
    return agm_program_execute(prepared,out,a,b,ex,mode);
}
} // namespace sbn
