// Original libsbn code; LGPL-3.0-or-later.
#include "float/rootn.hpp"
#include "float/diagnostics.hpp"
#include "float/power.hpp"
#include "float/division.hpp"
#include "float/product.hpp"
#include "float/roots.hpp"
#include "float/error.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <limits>
#include <new>

namespace sbn::v3::floating {
namespace {
using Wide=__int128_t;
constexpr std::uint64_t magic=UINT64_C(0x53424e3352544e31);
constexpr std::size_t maximum_stages=64,value_count=7;
enum Bank : unsigned {root_a,root_b,numerator,quotient,residual,correction,factor};
struct Step {std::size_t function=0,working=0,target=0;};
struct Stage {
    Step shape{};
    WidePositivePowerPrepared power{};
    FloatDivisionPrepared division{};
    FloatProductPlan product_plan{};
    FloatProductPrepared product{};
};
struct RootnState {
    std::uint64_t marker=0,degree=0;
    std::uint32_t active=0;
    unsigned workers=1,degree_bits=0;
    std::size_t precision=0,bytes=0,count=0,words=0;
    Stage* stages=nullptr;
    float_buffer* values=nullptr;
    std::span<std::byte> sum_storage{},word_storage{};
    FloatRootPrepared square_root{};
};
struct Shape {
    requirements need{};
    std::array<Step,maximum_stages> steps{};
    std::size_t count=0,working=0,function=0,sum_bytes=0,word_bytes=0;
};
struct Geometry {
    requirements power{},division{};
    FloatProductPlan product{};
};
std::int64_t narrow(Wide x) noexcept {
    require(x>=INT64_MIN&&x<=INT64_MAX,SBN3_FATAL_SIZE,"rootn internal exponent");return std::int64_t(x);
}
bool append(std::size_t& cursor,std::size_t bytes,std::size_t alignment) noexcept {
    if(!bytes)return true;
    std::size_t at=0;return align_size(cursor,alignment,at)&&add_size(at,bytes,cursor);
}
bool geometry(Step step,std::uint64_t degree,unsigned workers,Geometry& out) noexcept {
    out.power=wide_power_program_requirements(bits{step.function},degree,workers);
    out.division=division_program_requirements(bits{step.function},workers,SBN3_NEWTON_DIVIDE);
    return out.power.supported()&&out.division.supported()&&
        float_product_query(step.working,step.working,step.working,false,workers,out.product)==SBN3_SUPPORTED;
}
Shape quote(bits precision,std::uint64_t degree,unsigned workers) noexcept {
    Shape p;p.need.scratch_alignment=128;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32){
        p.need.status=support::unsupported;p.need.reason="rootn precision/workers";return p;
    }
    p.need.output_limbs=(precision.count+63)/64;
    std::size_t bytes=sizeof(RootnState);
    if(degree<=2){
        if(degree==2){
            const auto r=root_program_requirements(precision,workers,RootKind::square_root);
            if(!r.supported()){p.need=r;return p;}
            p.need.scratch_alignment=std::max(p.need.scratch_alignment,r.scratch_alignment);
            if(!append(bytes,r.scratch_bytes,r.scratch_alignment)){
                p.need.status=support::unsupported;p.need.reason="rootn square-root storage";return p;
            }
        }
        p.need.scratch_bytes=bytes;return p;
    }
    const auto b=std::size_t(std::bit_width(degree)),d=b-1;
    const auto target=std::size_t(std::max<Wide>(4,Wide(precision.count)+6-Wide(d)));
    const auto seed_target=std::min(target,d+8),seed_working=2*b+64;
    p.steps[p.count++]={b+32,seed_working,seed_target};
    std::size_t s=seed_target;
    while(s<target){
        s=std::min(target,2*s-1);
        if(p.count==maximum_stages){p.need.status=support::unsupported;p.need.reason="rootn stage count";return p;}
        p.steps[p.count++]={s+16,std::max(seed_working,s+d+40),s};
    }
    p.working=p.function=0;
    for(std::size_t i=0;i<p.count;++i){
        p.working=std::max(p.working,p.steps[i].working);p.function=std::max(p.function,p.steps[i].function);
    }
    if(p.working>float_precision_limit||p.function+16>float_precision_limit){
        p.need.status=support::unsupported;p.need.reason="rootn working precision";return p;
    }
    const auto words=(p.working+63)/64;
    if(p.working>256)p.sum_bytes=(words+2)*8+63;
    if(p.function>256)p.word_bytes=((p.function+63)/64+2)*8+63;
    std::size_t value_bytes=0;
    if(!mul_size(words,8,value_bytes)||!append(bytes,p.count*sizeof(Stage),alignof(Stage))||
       !append(bytes,value_count*sizeof(float_buffer),alignof(float_buffer))){
        p.need.status=support::unsupported;p.need.reason="rootn metadata size";return p;
    }
    for(std::size_t i=0;i<value_count;++i)if(!append(bytes,value_bytes,64)){
        p.need.status=support::unsupported;p.need.reason="rootn values size";return p;
    }
    for(std::size_t i=0;i<p.count;++i){
        Geometry g;
        if(!geometry(p.steps[i],degree,workers,g)){
            p.need.status=support::unsupported;p.need.reason="rootn arithmetic shape";return p;
        }
        p.need.scratch_alignment=std::max({p.need.scratch_alignment,g.power.scratch_alignment,
                                         g.division.scratch_alignment,g.product.alignment});
        if(!append(bytes,g.power.scratch_bytes,g.power.scratch_alignment)||
           !append(bytes,g.division.scratch_bytes,g.division.scratch_alignment)||
           !append(bytes,g.product.scratch_bytes,g.product.alignment)){
            p.need.status=support::unsupported;p.need.reason="rootn arithmetic storage";return p;
        }
    }
    if(!append(bytes,p.sum_bytes,64)||!append(bytes,p.word_bytes,64)){
        p.need.status=support::unsupported;p.need.reason="rootn linear storage";return p;
    }
    p.need.scratch_bytes=bytes;return p;
}
struct Reduction {Wide scale=0,remainder=0;};
Reduction reduce(float_view x,std::uint64_t degree) noexcept {
    const Wide top=Wide(x.exponent2())+magnitude_bits(x.magnitude()),n=Wide(degree);
    Wide h=top/n,k=top%n;
    if(k<0){--h;k+=n;}
    if(2*k>=n){++h;k-=n;}
    return {h,k};
}
bool power_two(float_view x,std::int64_t& exponent) noexcept {
    if(x.magnitude().size()!=1||!std::has_single_bit(x.magnitude().data()[0]))return false;
    exponent=x.exponent2()+std::countr_zero(x.magnitude().data()[0]);return true;
}
bool below(ErrorMagnitude bound,std::int64_t exponent) noexcept {
    const auto p=bound.publish();return p.exact()||(p.kind==error_kind::finite&&p.exponent2<=exponent);
}
ErrorMagnitude near_one_bound(Reduction reduction,std::uint64_t degree) noexcept {
    const auto k=reduction.remainder<0?-reduction.remainder:reduction.remainder;
    // |log(alpha)| < (|k|+1)/n. For the small accepted bound,
    // |alpha-1| <= 2*(|k|+1)/n <= 2*(|k|+1)*2^-floor(log2(n)).
    return ErrorMagnitude::from_scaled_uint64(std::uint64_t(k+1)).scale2(2-std::int64_t(std::bit_width(degree)));
}
bool simple_shape(float_view x,std::uint64_t degree,std::size_t precision) noexcept {
    if(degree<=1||!x.finite()||x.zero()||(x.negative()&&!(degree&1)))return true;
    if(degree==2)return false;
    std::int64_t e;
    if(power_two(x,e)&&Wide(e)%Wide(degree)==0)return true;
    return below(near_one_bound(reduce(x,degree),degree),-std::int64_t(precision)-3);
}
bool trivial(float_ref out,float_view x,std::uint64_t degree,rounding mode,float_report& report) noexcept {
    if(!degree){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
    if(degree==1){report=round_into(out,x,mode);return true;}
    if(degree==2)return false;
    if(x.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(x.zero()){report=special(out,float_class::finite,x.negative());return true;}
    if(x.negative()&&!(degree&1)){report=special(out,float_class::quiet_nan,false,float_flag::invalid);return true;}
    if(!x.finite()){report=special(out,float_class::infinity,x.negative());return true;}
    const std::uint64_t one=1;std::int64_t e;
    if(power_two(x,e)&&Wide(e)%Wide(degree)==0){
        report=round_magnitude(out,integer_view{{&one,1}},narrow(Wide(e)/Wide(degree)),x.negative(),mode);return true;
    }
    const auto reduction=reduce(x,degree);const auto error=near_one_bound(reduction,degree);
    if(below(error,-std::int64_t(out.precision().count)-3)){
        report=round_magnitude(out,integer_view{{&one,1}},narrow(reduction.scale),x.negative(),mode);
        report.error=ErrorMagnitude::from(report.error).add(error.scale2(narrow(reduction.scale))).publish();return true;
    }
    return false;
}
float_ref rebind(RootnState& s,Bank bank,std::size_t precision) noexcept {
    auto ref=s.values[bank].ref();detail::float_access::metadata(ref).precision=precision;return ref;
}
void stage_prefix(float_ref out,float_view x) noexcept {
    const auto m=x.magnitude();const auto width=magnitude_bits(m),p=out.precision().count;
    const auto cut=width>p?width-p:0,kept=width-cut,words=(kept+63)/64,first=cut/64;
    const unsigned shift=unsigned(cut%64);auto* destination=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){
        const auto j=first+i;
        destination->data[i]=(m.data()[j]>>shift)|(shift&&j+1<m.size()?m.data()[j+1]<<(64-shift):0);
    }
    destination->size=words;destination->negative=0;
    auto& metadata=detail::float_access::metadata(out);
    metadata.exponent2=narrow(Wide(x.exponent2())+cut);metadata.negative=false;metadata.classification=float_class::finite;
    detail::float_access::finish(out);
}
ErrorMagnitude magnitude_bound(float_view x) noexcept {
    if(x.zero())return {};
    const auto m=x.magnitude();const auto length=magnitude_bits(m);
    if(length<=64)return ErrorMagnitude::from_scaled_uint64(m.data()[0],x.exponent2());
    const auto cut=length-64,j=cut/64;const unsigned shift=unsigned(cut%64);
    const auto high=(m.data()[j]>>shift)|(shift&&j+1<m.size()?m.data()[j+1]<<(64-shift):0);
    const auto exponent=narrow(Wide(x.exponent2())+cut);
    return ErrorMagnitude::from_scaled_uint64(high,exponent).add(ErrorMagnitude::power_of_two(exponent));
}
struct Ratio {ErrorMagnitude residual_bound{};int direction=0;bool near=false;};
Ratio ratio(RootnState& s,Stage& stage,float_view y,float_view input,Wide k,sbn3_team_scope* scope) noexcept {
    const auto F=stage.shape.function;
    const auto normalized=float_view::from_magnitude(input.magnitude().limbs(),-std::int64_t(magnitude_bits(input.magnitude())));
    stage_prefix(rebind(s,numerator,F+16),normalized);
    const auto power=wide_power_program_execute(stage.power,y,execution{{},nullptr,scope});
    const auto e=k-power.top_exponent;
    if(e>1)return {{},1,false};
    if(e<-1)return {{},-1,false};
    const auto q=rebind(s,quotient,F);
    const auto division=division_program_execute(stage.division,q,s.values[numerator].view(),power.mantissa,
                                                execution{{},nullptr,scope},rounding::nearest_even);
    require(division.error.kind!=error_kind::unbounded&&!q.view().zero()&&q.view().finite(),
            SBN3_FATAL_MATH,"rootn normalized ratio");
    const auto D=float_view::from_magnitude(q.view().magnitude().limbs(),narrow(Wide(q.view().exponent2())+e));
    const std::uint64_t one=1;
    const auto subtract=subtract_into(rebind(s,residual,F+3),D,float_view::from_magnitude({&one,1},0),
                                      execution{s.sum_storage,nullptr,scope},rounding::nearest_even);
    require(subtract.error.exact(),SBN3_FATAL_MATH,"rootn exact residual subtraction");
    const auto r=s.values[residual].view();
    // Includes the actual division's guard-limb error, the numerator's
    // F+16 prefix, and the computed-relative positive-power certificate.
    // For e in [-1,1], D<=4 and their sum is <2^(5-F), including boundaries.
    return {magnitude_bound(r).add(ErrorMagnitude::power_of_two(5-std::int64_t(F))),
            r.zero()?0:(r.negative()?-1:1),true};
}
void update(RootnState& s,Stage& stage,Bank current,Bank next,sbn3_team_scope* scope) noexcept {
    const auto F=stage.shape.function,W=stage.shape.working;
    const auto r=s.values[residual].view();
    require(below(magnitude_bound(r),-1),SBN3_FATAL_MATH,"rootn Newton correction basin");
    const auto degree=float_view::from_magnitude({&s.degree,1},0);
    const auto correction_report=divide_into(rebind(s,correction,F),r,degree,
                                             execution{s.word_storage,nullptr,scope},rounding::nearest_even);
    require(correction_report.error.kind!=error_kind::unbounded,SBN3_FATAL_MATH,"rootn correction division");
    const std::uint64_t one=1;
    add_into(rebind(s,factor,W),float_view::from_magnitude({&one,1},0),s.values[correction].view(),
             execution{s.sum_storage,nullptr,scope},rounding::nearest_even);
    const auto report=float_product_execute(stage.product,rebind(s,next,W),s.values[current].view(),
                                            s.values[factor].view(),execution{{},nullptr,scope},rounding::nearest_even);
    require(report.error.kind!=error_kind::unbounded&&s.values[next].view().finite()&&!s.values[next].view().zero(),
            SBN3_FATAL_MATH,"rootn Newton product");
}
float_report finish(RootnState& s,float_ref out,float_view y,ErrorMagnitude rho,Wide scale,
                    bool negative,rounding mode) noexcept {
    const auto top=Wide(y.exponent2())+magnitude_bits(y.magnitude());
    const auto analytic=rho.scale2(narrow(top+2-Wide(s.degree_bits)+scale));
    auto report=round_magnitude(out,y.magnitude(),narrow(Wide(y.exponent2())+scale),negative,mode);
    report.error=ErrorMagnitude::from(report.error).add(analytic).publish();
    const auto v=out.view();
    require(v.finite()&&!v.zero()&&below(ErrorMagnitude::from(report.error),
        narrow(Wide(v.exponent2())+magnitude_bits(v.magnitude())-s.precision+(mode==rounding::nearest_even?0:1))),
        SBN3_FATAL_MATH,"rootn final error budget");
    return report;
}
struct Run {RootnState& state;float_ref out;float_view input;rounding mode;float_report report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Run*>(opaque);auto& s=call.state;
    const auto reduction=reduce(call.input,s.degree);auto& seed=s.stages[0];
    Bank current=root_a,next=root_b;
    __uint128_t lower=1,upper=4;std::int64_t exponent=-1;
    Ratio certificate;bool accepted=false;
    // At most b+8 bisections plus the final midpoint evaluation. All
    // midpoints have fewer than 80 significant bits and fit the seed bank.
    for(unsigned iteration=0;iteration<=s.degree_bits+8;++iteration){
        const __uint128_t middle=lower+upper;
        const std::uint64_t limbs[]{std::uint64_t(middle),std::uint64_t(middle>>64)};
        const auto report=round_magnitude(rebind(s,current,seed.shape.working),integer_view{{limbs,2}},exponent-1,false,rounding::nearest_even);
        require(report.error.exact(),SBN3_FATAL_MATH,"rootn exact seed midpoint");
        certificate=ratio(s,seed,s.values[current].view(),call.input,reduction.remainder,scope);
        if(certificate.near&&below(certificate.residual_bound,-5)){accepted=true;break;}
        require(certificate.direction,SBN3_FATAL_MATH,"rootn seed enclosure ordering");
        if(certificate.direction>0){lower=middle;upper*=2;}else{lower*=2;upper=middle;}
        --exponent;
    }
    require(accepted,SBN3_FATAL_MATH,"rootn bounded seed convergence");
    const auto stop=narrow(Wide(s.degree_bits)-1-Wide(s.precision)-5);
    auto done=[&]{return certificate.near&&below(certificate.residual_bound,stop);};
    auto emit=[&]{call.report=finish(s,call.out,s.values[current].view(),certificate.residual_bound,
                                   reduction.scale,call.input.negative(),call.mode);};
    if(done()){emit();return;}
    std::size_t accurate=4;
    while(accurate<seed.shape.target){
        update(s,seed,current,next,scope);std::swap(current,next);
        accurate=std::min(seed.shape.target,2*accurate-1);
        certificate=ratio(s,seed,s.values[current].view(),call.input,reduction.remainder,scope);
        require(certificate.near,SBN3_FATAL_MATH,"rootn seed refinement range");
        if(done()){emit();return;}
    }
    for(std::size_t i=1;i<s.count;++i){
        auto& stage=s.stages[i];
        certificate=ratio(s,stage,s.values[current].view(),call.input,reduction.remainder,scope);
        require(certificate.near,SBN3_FATAL_MATH,"rootn refinement ratio range");
        if(done()){emit();return;}
        update(s,stage,current,next,scope);std::swap(current,next);
    }
    certificate=ratio(s,s.stages[s.count-1],s.values[current].view(),call.input,reduction.remainder,scope);
    require(done(),SBN3_FATAL_MATH,"rootn final residual certificate");emit();
}
void disjoint(std::span<std::byte> storage,float_ref out,float_view input) noexcept {
    const auto* destination=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
            !overlaps(storage.data(),storage.size(),destination->data,out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"rootn value/program storage alias");
}
struct Use {
    RootnState& state;
    explicit Use(RootnState& s):state(s){std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
                SBN3_FATAL_LIFETIME,"concurrent prepared rootn");}
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}
};
}

FloatDiagnostic rootn_diagnostic(bits p,std::uint64_t degree,unsigned workers) noexcept {
    const auto q=quote(p,degree,workers);FloatDiagnostic d;d.need=q.need;d.recipe="certified_newton_rootn";
    d.working_bits=q.working;d.metadata_bytes=sizeof(RootnState)+q.count*sizeof(Stage);return d;
}
requirements rootn_program_requirements(bits precision,std::uint64_t degree,unsigned workers) noexcept {
    return quote(precision,degree,workers).need;
}
FloatRootnPrepared rootn_program_prepare(bits precision,std::uint64_t degree,unsigned workers,
                                        std::span<std::byte> storage) noexcept {
    const auto p=quote(precision,degree,workers);
    require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);
    require(storage.size()>=p.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(p.need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"rootn prepared storage",p.need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());
    auto* state=::new(frame.allocate(sizeof(RootnState),128))RootnState{};
    state->precision=precision.count;state->degree=degree;state->workers=workers;
    state->bytes=p.need.scratch_bytes;state->count=p.count;state->degree_bits=unsigned(std::bit_width(degree));
    auto region=[&](std::size_t bytes,std::size_t alignment){
        if(!bytes)return std::span<std::byte>{};
        return std::span<std::byte>{static_cast<std::byte*>(frame.allocate(bytes,alignment)),bytes};
    };
    if(degree==2){
        const auto r=root_program_requirements(precision,workers,RootKind::square_root);
        state->square_root=root_program_prepare(precision,workers,RootKind::square_root,region(r.scratch_bytes,r.scratch_alignment));
    }else if(degree>2){
        state->stages=static_cast<Stage*>(frame.allocate(p.count*sizeof(Stage),alignof(Stage)));
        for(std::size_t i=0;i<p.count;++i)::new(state->stages+i)Stage{};
        state->values=static_cast<float_buffer*>(frame.allocate(value_count*sizeof(float_buffer),alignof(float_buffer)));
        state->words=(p.working+63)/64;
        for(std::size_t i=0;i<value_count;++i){
            auto* data=static_cast<std::uint64_t*>(frame.allocate(state->words*8,64));
            ::new(state->values+i)float_buffer({data,state->words},bits{p.working});
        }
        for(std::size_t i=0;i<p.count;++i){
            auto& stage=state->stages[i];stage.shape=p.steps[i];Geometry g;
            require(geometry(stage.shape,degree,workers,g),SBN3_FATAL_ARGUMENT,"rootn arithmetic recipe");
            stage.product_plan=g.product;
            stage.power=wide_power_program_prepare(bits{stage.shape.function},degree,workers,region(g.power.scratch_bytes,g.power.scratch_alignment));
            stage.division=division_program_prepare(bits{stage.shape.function},workers,SBN3_NEWTON_DIVIDE,region(g.division.scratch_bytes,g.division.scratch_alignment));
            stage.product=float_product_prepare(stage.product_plan,region(g.product.scratch_bytes,g.product.alignment));
        }
        state->sum_storage=region(p.sum_bytes,64);state->word_storage=region(p.word_bytes,64);
    }
    state->marker=magic;return {state,storage.first(p.need.scratch_bytes)};
}
float_report rootn_program_execute(const FloatRootnPrepared& prepared,float_ref out,float_view input,
                                   execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(RootnState),
            SBN3_FATAL_LIFETIME,"prepared rootn state");
    auto& state=*static_cast<RootnState*>(prepared.state);
    require(state.marker==magic&&state.bytes==prepared.storage.size()&&out.precision().count==state.precision&&
            ex.workers()>=state.workers,SBN3_FATAL_ARGUMENT,"prepared rootn identity/shape");
    validate(input);check_output(out,input);disjoint(prepared.storage,out,input);Use exclusive(state);
    if(state.degree==2)return root_program_execute(state.square_root,out,input,ex,mode);
    float_report report;if(trivial(out,input,state.degree,mode,report))return report;
    Run run{state,out,input,mode};
    if(ex.scope)calculate(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&run);else serial_team_run(calculate,&run);
    return run.report;
}
}

namespace sbn {
requirements rootn_requirements(float_ref out,float_view input,std::uint64_t degree,unsigned workers) noexcept {
    using namespace v3::floating;
    const auto p=out.precision().count;
    if(!p||p>float_precision_limit||!workers||workers>32)return {support::unsupported,0,128,0,"rootn precision/workers"};
    validate(input);const auto words=(p+63)/64;
    if(out.capacity()<words)return {support::capacity,0,128,words,"rootn output capacity"};
    if(degree==2)return sqrt_requirements(out,input,workers);
    if(simple_shape(input,degree,p))return {support::yes,0,128,words,""};
    return rootn_program_requirements(out.precision(),degree,workers);
}
float_report rootn_into(float_ref out,float_view input,std::uint64_t degree,execution ex,rounding mode) noexcept {
    using namespace v3::floating;validate(input);check_output(out,input);
    if(degree==2)return sqrt_into(out,input,ex,mode);
    float_report report;if(trivial(out,input,degree,mode,report))return report;
    disjoint(ex.scratch,out,input);
    const auto prepared=rootn_program_prepare(out.precision(),degree,ex.workers(),ex.scratch);
    return rootn_program_execute(prepared,out,input,ex,mode);
}
}
