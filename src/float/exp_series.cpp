// Original libsbn code; LGPL-3.0-or-later.
#include "float/exp_series.hpp"
#include "float/diagnostics.hpp"
#include "float/product.hpp"
#include "float/division.hpp"
#include "float/error.hpp"
#include "core/x86_64/word.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3345585053);
constexpr std::size_t maximum_precision=4096,leaf_words=68;
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;
    std::size_t precision=0,working=0,bytes=0,terms=0,block=0,halvings=0,words=0;
    unsigned workers=1;bool negative=false;ExpSeriesKind kind=ExpSeriesKind::exp;
    FloatProductPlan plan{};FloatProductPrepared product{};FloatDivisionPrepared division{};
    float_buffer* values=nullptr;std::uint64_t* coefficient_product=nullptr;
    std::span<std::byte> sum{},word{};
};
struct Quote {
    requirements need{},division{},word{};FloatProductPlan product{};
    std::size_t working=0,terms=0,block=0,halvings=0,words=0,banks=0,sum=0;
};
bool append(std::size_t& size,std::size_t bytes,std::size_t alignment) noexcept {
    return !bytes||(align_size(size,alignment,size)&&add_size(size,bytes,size));
}
Quote quote(bits precision,ExpSeriesKind kind,bool negative,unsigned workers) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(precision.count+63)/64;
    if(!precision.count||precision.count>maximum_precision||!workers||workers>32||
       (kind!=ExpSeriesKind::exp&&kind!=ExpSeriesKind::expm1)){
        q.need.status=support::unsupported;q.need.reason="reduced exp series shape";return q;
    }
    q.halvings=2;while(q.halvings*q.halvings*q.halvings<precision.count)++q.halvings;
    q.terms=1+(precision.count+q.halvings+13+q.halvings-1)/q.halvings;
    q.block=1;while((q.block+1)*(q.block+1)<=q.terms)++q.block;
    q.block=std::min(q.block,std::size_t(120)/std::bit_width(q.terms));
    q.working=precision.count+q.halvings+std::bit_width(q.terms*(q.block+1)-1)+32;
    q.words=(q.working+63)/64;q.banks=q.block+7;
    if(float_product_query(q.working,q.working,q.working,false,workers,q.product)!=SBN3_SUPPORTED){
        q.need.status=support::unsupported;q.need.reason="reduced exp product shape";return q;
    }
    // Quote the actual two-word leaf once. Every Taylor denominator has at
    // most 120 bits, so this execution branch never queries a long inverse.
    std::array<std::uint64_t,leaf_words> temporary{};const std::uint64_t one=1,d[]{1,1};
    require(q.words<=temporary.size(),SBN3_FATAL_SIZE,"exp leaf quote capacity");
    float_buffer dummy(temporary,bits{q.working});
    q.word=divide_requirements(dummy.ref(),float_view::from_magnitude({&one,1},0),float_view::from_magnitude(d,0),workers);
    if(negative&&q.words>2)q.division=division_program_requirements(bits{q.working},workers,SBN3_NEWTON_DIVIDE);
    if(!q.word.supported()||!q.division.supported()){
        q.need.status=support::unsupported;q.need.reason="reduced exp division shape";return q;
    }
    q.need.scratch_alignment=std::max({q.need.scratch_alignment,q.product.alignment,q.word.scratch_alignment,q.division.scratch_alignment});
    q.sum=q.working>256?(q.words+2)*8+63:0;
    std::size_t bytes=0;
    bool ok=append(bytes,sizeof(State),128)&&append(bytes,q.banks*sizeof(float_buffer),64);
    for(std::size_t i=0;i<q.banks&&ok;++i)ok=append(bytes,q.words*8,64);
    ok=ok&&append(bytes,(q.words+2)*8,64)&&append(bytes,q.product.scratch_bytes,q.product.alignment)&&
       append(bytes,q.division.scratch_bytes,q.division.scratch_alignment)&&append(bytes,q.word.scratch_bytes,q.word.scratch_alignment)&&
       append(bytes,q.sum,64);
    if(!ok){q.need.status=support::unsupported;q.need.reason="reduced exp storage extent";return q;}
    q.need.scratch_bytes=bytes;return q;
}
std::int64_t top(float_view x) noexcept {return x.exponent2()+std::int64_t(magnitude_bits(x.magnitude()));}
bool domain(float_view x) noexcept {
    if(!x.finite())return false;
    validate(x);if(x.zero())return true;
    const auto t=top(x);if(t<=0)return true;if(t>1)return false;
    const std::uint64_t one=1;
    const auto positive=float_view::from_magnitude(x.magnitude().limbs(),x.exponent2());
    return compare(positive,float_view::from_magnitude({&one,1},0))!=std::partial_ordering::greater;
}
bool tiny(float_view x,std::size_t p) noexcept {return x.zero()||top(x)<=-std::int64_t(p)-8;}
bool stage(float_ref out,float_view x,std::int64_t scale=0) noexcept {
    const auto input=x.magnitude();const auto width=magnitude_bits(input),p=out.precision().count;
    const auto cut=width>p?width-p:0,kept=width-cut,words=(kept+63)/64,first=cut/64;
    const unsigned shift=unsigned(cut%64);auto* dst=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){const auto j=first+i;
        dst->data[i]=(input.data()[j]>>shift)|(shift&&j+1<input.size()?input.data()[j+1]<<(64-shift):0);}
    dst->size=words;dst->negative=0;auto& metadata=detail::float_access::metadata(out);
    metadata.exponent2=x.exponent2()+std::int64_t(cut)+scale;metadata.classification=float_class::finite;metadata.negative=false;
    detail::float_access::finish(out);return cut!=0;
}
float_report leaf(float_ref out,float_view x,ExpSeriesKind kind,rounding mode) noexcept {
    const std::uint64_t one=1;
    if(x.zero())return kind==ExpSeriesKind::expm1?special(out,float_class::finite,x.negative()):
        round_magnitude(out,integer_view{{&one,1}},0,false,mode);
    // Stage a bounded prefix before writing a possibly aliased output. For
    // |x|<=1/2, |expm1(x)-x|<=x^2. No halving or tiny product is performed.
    alignas(64)std::array<std::uint64_t,leaf_words> value{};
    alignas(64)std::array<std::uint64_t,76> sum{};
    const auto T=top(x);const auto p=out.precision().count,W=p+32;const bool negative=x.negative();
    float_buffer staged(value,bits{W});const bool truncated=stage(staged.ref(),x);
    auto point=negative?staged.view().negated():staged.view();
    auto report=kind==ExpSeriesKind::expm1?round_into(out,point,mode):
        add_into(out,float_view::from_magnitude({&one,1},0),point,
                 execution{{reinterpret_cast<std::byte*>(sum.data()),sizeof(sum)},nullptr,nullptr},mode);
    auto error=ErrorMagnitude::from(report.error).add(ErrorMagnitude::power_of_two(2*T));
    if(truncated)error=error.add(ErrorMagnitude::power_of_two(T-std::int64_t(W)));
    report.error=error.publish();return report;
}
void good(float_report report,float_view point,std::size_t W) noexcept {
    require(point.finite()&&report.error.kind!=error_kind::unbounded&&
            !has_flag(report.flags,float_flag::underflow)&&!has_flag(report.flags,float_flag::overflow),
            SBN3_FATAL_MATH,"exp series arithmetic range");
    require(report.error.exact()||(!point.zero()&&report.error.exponent2<=top(point)-std::int64_t(W)),
            SBN3_FATAL_MATH,"exp series arithmetic error");
}
float_report add(State& s,std::size_t out,float_view a,float_view b,sbn3_team_scope* scope) noexcept {
    auto report=add_into(s.values[out].ref(),a,b,execution{s.sum,nullptr,scope},rounding::nearest_even);
    good(report,s.values[out].view(),s.working);return report;
}
void multiply(State& s,std::size_t out,float_view a,float_view b,sbn3_team_scope* scope) noexcept {
    const auto report=float_product_execute(s.product,s.values[out].ref(),a,b,execution{{},nullptr,scope},rounding::nearest_even);
    good(report,s.values[out].view(),s.working);
}
void coefficient(State& s,std::size_t out,float_view a,__uint128_t c) noexcept {
    const std::uint64_t words[]{std::uint64_t(c),std::uint64_t(c>>64)};const auto count=words[1]?2u:1u;
    const auto input=a.magnitude();
    mul_basecase_assumed(s.coefficient_product,input.data(),input.size(),words,count);
    const auto report=round_magnitude(s.values[out].ref(),integer_view{{s.coefficient_product,input.size()+count}},
                                      a.exponent2(),false,rounding::nearest_even);
    good(report,s.values[out].view(),s.working);
}
// Put epsilon=2^-W. Every positive sum/product/short division contributes
// at most 8*epsilon logarithmic error; input prefix contributes 4*epsilon.
// A baby power contributes <=12*m*epsilon. Positivity then bounds one block
// by predecessor error +(20*m+16)*epsilon, hence the entire polynomial by
// 64*N*(m+1)*epsilon. Reconstruction y*(2+y) has logarithmic sensitivity<2;
// after s steps total computed-relative arithmetic error <=2^(-p-24).
// Taylor relative tail <=2*u^(N-1); after reconstruction and conversion to
// center-relative error it is <=2^(s+3-s*(N-1))<=2^(-p-10).
// Together a positive expm1 center has relative error <2^(-p-9).
struct Run {State& state;float_ref out;float_view input;rounding mode;float_report report{};};
void calculate(void* opaque,sbn3_team_scope* scope) {
    auto& call=*static_cast<Run*>(opaque);auto& s=call.state;
    const std::uint64_t one=1,two=2;
    const auto one_view=float_view::from_magnitude({&one,1},0),two_view=float_view::from_magnitude({&two,1},0);
    const auto F=s.block+1,B=F+1,term=F+2,tail=F+3,aux=F+4,result=F+5;
    round_into(s.values[0].ref(),one_view);stage(s.values[1].ref(),call.input,-std::int64_t(s.halvings));
    for(std::size_t i=2;i<=s.block;++i)multiply(s,i,s.values[i-1].view(),s.values[1].view(),scope);
    special(s.values[F].ref(),float_class::finite,false);
    std::size_t end=s.terms;
    while(end){
        const auto begin=((end-1)/s.block)*s.block,length=end-begin;
        special(s.values[B].ref(),float_class::finite,false);__uint128_t D=1;
        for(std::size_t l=length;l-->0;){
            D*=begin+l+1;
            require(!(D>>120),SBN3_FATAL_MATH,"exp exact denominator coefficient");
            if(!begin&&!l)continue; // Preserve expm1 directly: omit 1.
            coefficient(s,term,s.values[l].view(),D);
            add(s,B,s.values[B].view(),s.values[term].view(),scope);
        }
        multiply(s,tail,s.values[length].view(),s.values[F].view(),scope);
        add(s,aux,s.values[B].view(),s.values[tail].view(),scope);
        const std::uint64_t denominator[]{std::uint64_t(D),std::uint64_t(D>>64)};
        const auto report=divide_into(s.values[F].ref(),s.values[aux].view(),float_view::from_magnitude(denominator,0),
                                      execution{s.word,nullptr,scope},rounding::nearest_even);
        good(report,s.values[F].view(),s.working);end=begin;
    }
    std::size_t current=F,next=tail;
    for(std::size_t i=0;i<s.halvings;++i){
        add(s,B,two_view,s.values[current].view(),scope);
        multiply(s,next,s.values[current].view(),s.values[B].view(),scope);std::swap(current,next);
    }
    const auto y=s.values[current].view();
    require(!y.zero()&&!y.negative()&&compare(y,two_view)==std::partial_ordering::less,
            SBN3_FATAL_MATH,"exp series reconstructed range");
    auto error=ErrorMagnitude::power_of_two(top(y)-std::int64_t(s.precision)-9);
    auto point=y;
    if(s.negative||s.kind==ExpSeriesKind::exp){
        const auto denominator_report=add(s,B,one_view,y,scope);
        if(s.negative){
            const auto dividend=s.kind==ExpSeriesKind::expm1?y:one_view;
            const auto quotient_report=s.division.state?
                division_program_execute(s.division,s.values[result].ref(),dividend,s.values[B].view(),execution{{},nullptr,scope},rounding::nearest_even):
                divide_into(s.values[result].ref(),dividend,s.values[B].view(),execution{s.word,nullptr,scope},rounding::nearest_even);
            good(quotient_report,s.values[result].view(),s.working);
            auto denominator_error=ErrorMagnitude::from(denominator_report.error);
            if(s.kind==ExpSeriesKind::expm1)denominator_error=denominator_error.multiply(ErrorMagnitude::power_of_two(top(y)));
            error=error.add(denominator_error).add(ErrorMagnitude::from(quotient_report.error));
            point=s.kind==ExpSeriesKind::expm1?s.values[result].view().negated():s.values[result].view();
        }else{point=s.values[B].view();error=error.add(ErrorMagnitude::from(denominator_report.error));}
    }
    call.report=round_into(call.out,point,call.mode);
    call.report.error=ErrorMagnitude::from(call.report.error).add(error).publish();
    const auto output=call.out.view();const auto limit=top(output)-std::int64_t(s.precision)+(call.mode==rounding::nearest_even?0:1);
    require(output.finite()&&!output.zero()&&call.report.error.kind==error_kind::finite&&call.report.error.exponent2<=limit,
            SBN3_FATAL_MATH,"exp series final error budget");
}
void aliases(std::span<std::byte> storage,float_ref out,float_view input) noexcept {
    const auto* destination=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
            !overlaps(storage.data(),storage.size(),destination->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"exp series storage alias");
}
struct Use {State& state;explicit Use(State& s):state(s){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent exp series");}
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}};
}
FloatDiagnostic exp_series_diagnostic(bits p,bool minus_one,bool negative,unsigned workers) noexcept {
    const auto q=quote(p,minus_one?ExpSeriesKind::expm1:ExpSeriesKind::exp,negative,workers);
    FloatDiagnostic d;d.need=q.need;d.recipe="reduced_series";d.working_bits=q.working;
    d.metadata_bytes=sizeof(State)+q.banks*sizeof(float_buffer);d.table_bytes=q.product.tables_bytes;
    d.constant_value_bytes=0;d.constant_work_bytes=0;return d;
}
requirements exp_series_program_requirements(bits p,ExpSeriesKind kind,bool negative,unsigned workers) noexcept {
    return quote(p,kind,negative,workers).need;
}
FloatExpSeriesPrepared exp_series_program_prepare(bits p,ExpSeriesKind kind,bool negative,unsigned workers,
                                                  std::span<std::byte> storage) noexcept {
    const auto q=quote(p,kind,negative,workers);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"exp series storage",q.need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());auto* state=::new(frame.allocate(sizeof(State),128))State{};
    state->precision=p.count;state->working=q.working;state->bytes=q.need.scratch_bytes;state->terms=q.terms;
    state->block=q.block;state->halvings=q.halvings;state->words=q.words;state->workers=workers;
    state->kind=kind;state->negative=negative;state->plan=q.product;
    state->values=static_cast<float_buffer*>(frame.allocate(q.banks*sizeof(float_buffer),64));
    for(std::size_t i=0;i<q.banks;++i)::new(state->values+i)float_buffer({frame.alloc<std::uint64_t>(q.words),q.words},bits{q.working});
    state->coefficient_product=frame.alloc<std::uint64_t>(q.words+2);
    auto region=[&](std::size_t bytes,std::size_t alignment){return bytes?
        std::span<std::byte>{static_cast<std::byte*>(frame.allocate(bytes,alignment)),bytes}:std::span<std::byte>{};};
    state->product=float_product_prepare(state->plan,region(q.product.scratch_bytes,q.product.alignment));
    if(negative&&q.words>2)state->division=division_program_prepare(bits{q.working},workers,SBN3_NEWTON_DIVIDE,region(q.division.scratch_bytes,q.division.scratch_alignment));
    state->word=region(q.word.scratch_bytes,q.word.scratch_alignment);state->sum=region(q.sum,64);
    state->marker=magic;return {state,storage.first(q.need.scratch_bytes)};
}
float_report exp_series_program_execute(const FloatExpSeriesPrepared& prepared,float_ref out,float_view input,
                                         execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"exp series prepared state");
    auto& state=*static_cast<State*>(prepared.state);
    require(state.marker==magic&&state.bytes==prepared.storage.size()&&out.precision().count==state.precision&&
            ex.workers()>=state.workers&&(input.zero()||input.negative()==state.negative)&&domain(input),
            SBN3_FATAL_ARGUMENT,"exp series input/shape");
    check_output(out,input);aliases(prepared.storage,out,input);Use exclusive(state);
    if(tiny(input,state.precision))return leaf(out,input,state.kind,mode);
    Run run{state,out,input,mode};
    if(ex.scope)calculate(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&run);else serial_team_run(calculate,&run);
    return run.report;
}
requirements exp_series_requirements(float_ref out,float_view input,ExpSeriesKind kind,unsigned workers) noexcept {
    const auto p=out.precision().count;
    if(!p||p>maximum_precision||!workers||workers>32||!domain(input)||
       (kind!=ExpSeriesKind::exp&&kind!=ExpSeriesKind::expm1))return {support::unsupported,0,128,0,"exp series input/shape"};
    const auto words=(p+63)/64;if(out.capacity()<words)return {support::capacity,0,128,words,"exp series output capacity"};
    if(tiny(input,p))return {support::yes,0,128,words,""};
    return exp_series_program_requirements(out.precision(),kind,input.negative(),workers);
}
float_report exp_series_into(float_ref out,float_view input,ExpSeriesKind kind,execution ex,rounding mode) noexcept {
    const auto p=out.precision().count;
    require(p&&p<=maximum_precision&&domain(input)&&(kind==ExpSeriesKind::exp||kind==ExpSeriesKind::expm1),SBN3_FATAL_ARGUMENT,"exp series input/shape");
    check_output(out,input);if(tiny(input,p))return leaf(out,input,kind,mode);
    aliases(ex.scratch,out,input);const auto prepared=exp_series_program_prepare(out.precision(),kind,input.negative(),ex.workers(),ex.scratch);
    return exp_series_program_execute(prepared,out,input,ex,mode);
}
}
