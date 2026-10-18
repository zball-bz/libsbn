// Original libsbn code; LGPL-3.0-or-later.
#include "float/trig_series.hpp"
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
constexpr std::uint64_t magic=UINT64_C(0x53424e3354525352);
constexpr std::size_t maximum_precision=4096,leaf_words=68;
struct State {
    std::uint64_t marker=0;std::uint32_t active=0;
    std::size_t precision=0,working=0,bytes=0,terms=0,block=0,halvings=0,words=0;
    unsigned workers=1;
    FloatProductPlan mul_plan{},square_plan{};FloatProductPrepared mul{},square{};
    float_buffer* values=nullptr;std::uint64_t* coefficient_product=nullptr;
    std::span<std::byte> sum{},word{};
};
struct Quote {
    requirements need{},word{};FloatProductPlan mul{},square{};
    std::size_t working=0,terms=0,block=0,halvings=0,words=0,banks=0,sum=0;
};
bool append(std::size_t& at,std::size_t bytes,std::size_t alignment) noexcept {
    return !bytes||(align_size(at,alignment,at)&&add_size(at,bytes,at));
}
Quote quote(bits p,unsigned workers) noexcept {
    Quote q;q.need.scratch_alignment=128;
    if(!p.count||p.count>maximum_precision||!workers||workers>32){
        q.need.status=support::unsupported;q.need.reason="reduced trig series shape";return q;
    }
    q.need.output_limbs=(p.count+63)/64;
    q.halvings=2;while(q.halvings*q.halvings*q.halvings<p.count)++q.halvings;
    q.terms=(p.count+2*q.halvings+13+2*q.halvings-1)/(2*q.halvings);
    q.block=1;while((q.block+1)*(q.block+1)<=q.terms)++q.block;
    q.block=std::min(q.block,std::size_t(120)/(2*std::bit_width(2*q.terms+3)));
    q.working=p.count+2*q.halvings+std::bit_width(q.terms*(q.block+1)-1)+32;
    q.words=(q.working+63)/64;q.banks=q.block+11;
    if(float_product_query(q.working,q.working,q.working,false,workers,q.mul)!=SBN3_SUPPORTED||
       float_product_query(q.working,q.working,q.working,true,workers,q.square)!=SBN3_SUPPORTED){
        q.need.status=support::unsupported;q.need.reason="trig series product shape";return q;
    }
    std::array<std::uint64_t,leaf_words> storage{};const std::uint64_t one=1,d[]{1,1};
    require(q.words<=storage.size(),SBN3_FATAL_SIZE,"trig series leaf quote");
    float_buffer dummy(storage,bits{q.working});
    q.word=divide_requirements(dummy.ref(),float_view::from_magnitude({&one,1},0),float_view::from_magnitude(d,0),workers);
    if(!q.word.supported()){q.need=q.word;return q;}
    q.need.scratch_alignment=std::max({q.need.scratch_alignment,q.mul.alignment,q.square.alignment,q.word.scratch_alignment});
    q.sum=q.working>256?(q.words+2)*8+63:0;
    std::size_t bytes=0;bool ok=append(bytes,sizeof(State),128)&&append(bytes,q.banks*sizeof(float_buffer),64);
    for(std::size_t i=0;i<q.banks&&ok;++i)ok=append(bytes,q.words*8,64);
    ok=ok&&append(bytes,(q.words+2)*8,64)&&append(bytes,q.mul.scratch_bytes,q.mul.alignment)&&
        append(bytes,q.square.scratch_bytes,q.square.alignment)&&append(bytes,q.word.scratch_bytes,q.word.scratch_alignment)&&append(bytes,q.sum,64);
    if(!ok){q.need.status=support::unsupported;q.need.reason="trig series storage extent";return q;}
    q.need.scratch_bytes=bytes;return q;
}
std::int64_t top(float_view x) noexcept{return x.exponent2()+std::int64_t(magnitude_bits(x.magnitude()));}
bool domain(float_view x) noexcept {
    if(!x.finite())return false;validate(x);if(x.zero())return true;
    const auto t=top(x);if(t<=0)return true;if(t>1)return false;
    const std::uint64_t one=1;
    return compare(float_view::from_magnitude(x.magnitude().limbs(),x.exponent2()),
        float_view::from_magnitude({&one,1},0))!=std::partial_ordering::greater;
}
bool tiny(float_view x,std::size_t p) noexcept{return x.zero()||2*top(x)<=-std::int64_t(p)-10;}
bool prefix(float_ref out,float_view x,std::int64_t scale=0) noexcept {
    const auto m=x.magnitude();const auto width=magnitude_bits(m),p=out.precision().count;
    const auto cut=width>p?width-p:0,kept=width-cut,words=(kept+63)/64,first=cut/64;
    const unsigned shift=unsigned(cut%64);auto* dst=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){const auto j=first+i;
        dst->data[i]=(m.data()[j]>>shift)|(shift&&j+1<m.size()?m.data()[j+1]<<(64-shift):0);}
    dst->size=words;dst->negative=0;auto& metadata=detail::float_access::metadata(out);
    metadata.classification=float_class::finite;metadata.negative=false;
    metadata.exponent2=x.exponent2()+std::int64_t(cut)+scale;detail::float_access::finish(out);return cut!=0;
}
void output_checks(float_ref sine,float_ref cosine,float_view input,std::span<std::byte> storage={}) noexcept {
    check_output(sine,input);check_output(cosine,input);
    const auto* s=detail::float_access::begin(sine);const auto* c=detail::float_access::begin(cosine);
    require(!overlaps(s->data,sine.capacity()*8,c->data,cosine.capacity()*8),SBN3_FATAL_ARGUMENT,"trig series output alias");
    require(!overlaps(storage.data(),storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
            !overlaps(storage.data(),storage.size(),s->data,sine.capacity()*8)&&
            !overlaps(storage.data(),storage.size(),c->data,cosine.capacity()*8),SBN3_FATAL_ARGUMENT,"trig series storage alias");
}
TrigSincosReport leaf(float_ref sine,float_ref cosine,float_view x,rounding mode) noexcept {
    const std::uint64_t one=1;const bool negative=x.negative();
    if(x.zero())return {special(sine,float_class::finite,negative),round_magnitude(cosine,integer_view{{&one,1}},0,false,mode)};
    const auto P=std::max(sine.precision().count,cosine.precision().count),W=P+32;const auto T=top(x);
    alignas(64)std::array<std::uint64_t,leaf_words> words{};float_buffer staged(words,bits{W});
    const bool truncated=prefix(staged.ref(),x);const auto point=negative?staged.view().negated():staged.view();
    auto sr=round_into(sine,point,mode);
    auto error=ErrorMagnitude::from(sr.error).add(ErrorMagnitude::power_of_two(3*T));
    if(truncated)error=error.add(ErrorMagnitude::power_of_two(T-std::int64_t(W)));sr.error=error.publish();
    auto cr=round_magnitude(cosine,integer_view{{&one,1}},0,false,mode);
    cr.error=ErrorMagnitude::power_of_two(2*T-1).publish();return {sr,cr};
}
void good(float_report report,float_view point,std::size_t W) noexcept {
    require(point.finite()&&report.error.kind!=error_kind::unbounded&&
            !has_flag(report.flags,float_flag::underflow)&&!has_flag(report.flags,float_flag::overflow),
            SBN3_FATAL_MATH,"trig series arithmetic range");
    require(report.error.exact()||(!point.zero()&&report.error.exponent2<=top(point)-std::int64_t(W)),
            SBN3_FATAL_MATH,"trig series arithmetic error");
}
void multiply(State& s,std::size_t out,float_view a,float_view b,sbn3_team_scope* scope,bool square=false) noexcept {
    const auto report=float_product_execute(square?s.square:s.mul,s.values[out].ref(),a,b,execution{{},nullptr,scope},rounding::nearest_even);
    good(report,s.values[out].view(),s.working);
}
void add(State& s,std::size_t out,float_view a,float_view b,sbn3_team_scope* scope) noexcept {
    const auto report=add_into(s.values[out].ref(),a,b,execution{s.sum,nullptr,scope},rounding::nearest_even);
    good(report,s.values[out].view(),s.working);
}
void coefficient(State& s,std::size_t out,float_view a,__uint128_t c) noexcept {
    const std::uint64_t words[]{std::uint64_t(c),std::uint64_t(c>>64)};const auto count=words[1]?2u:1u;
    const auto m=a.magnitude();mul_basecase_assumed(s.coefficient_product,m.data(),m.size(),words,count);
    const auto report=round_magnitude(s.values[out].ref(),integer_view{{s.coefficient_product,m.size()+count}},a.exponent2(),a.negative(),rounding::nearest_even);
    good(report,s.values[out].view(),s.working);
}
void polynomial(State& s,bool sinc,sbn3_team_scope* scope) noexcept {
    const auto F=s.block+2,B=F+1,term=F+2,tail=F+3,aux=F+4;
    special(s.values[F].ref(),float_class::finite,false);std::size_t end=s.terms;
    while(end){
        const auto begin=((end-1)/s.block)*s.block,length=end-begin;
        special(s.values[B].ref(),float_class::finite,false);__uint128_t D=1;
        for(std::size_t l=length;l-->0;){
            const auto index=2*(begin+l)+(sinc?2u:1u);const auto ratio=index*(index+1);
            __uint128_t next=0;require(!__builtin_mul_overflow(D,__uint128_t(ratio),&next)&&!(next>>120),
                                      SBN3_FATAL_MATH,"trig exact denominator coefficient");D=next;
            coefficient(s,term,s.values[l].view(),D);add(s,B,s.values[B].view(),s.values[term].view(),scope);
        }
        multiply(s,tail,s.values[length].view(),s.values[F].view(),scope);
        add(s,aux,s.values[B].view(),s.values[tail].view(),scope);
        const std::uint64_t denominator[]{std::uint64_t(D),std::uint64_t(D>>64)};
        const auto report=divide_into(s.values[F].ref(),s.values[aux].view(),float_view::from_magnitude(denominator,0),
                                      execution{s.word,nullptr,scope},rounding::nearest_even);
        good(report,s.values[F].view(),s.working);end=begin;
    }
}
// |v|<=1/16, q_j>=2. Normalized absolute coefficient sums are <=32/31;
// every F_j is in [31/32,1] and its successor's sensitivity is <=(1/32)^len.
// With epsilon=2^-W, baby/short-product/sum errors therefore give initial
// max(relative sine error, absolute cosine error) <=2^14*N*(m+1)*epsilon,
// plus Taylor tail <=2*|u|^(2N). For a previous angle <=1/2 and errors<1/16,
// C>=7/8, |S|<=1/2, so double-angle gives E_next<=4E+16*epsilon.
// W=p+2s+ceil_log2(N*(m+1))+32 and 2sN>=p+2s+13 imply final E<2^(-p-11).
// The published radii below allow conversion to computed-relative sine
// error and all fixed-size outward error accumulation, before final rounding.
struct Run {State& state;float_ref sine,cosine;float_view input;rounding mode;TrigSincosReport report{};};
float_report publish(float_ref out,float_view center,ErrorMagnitude error,rounding mode) noexcept {
    auto report=round_into(out,center,mode);report.error=ErrorMagnitude::from(report.error).add(error).publish();
    const auto value=out.view();const auto limit=top(value)-std::int64_t(out.precision().count)+(mode==rounding::nearest_even?0:1);
    require(value.finite()&&!value.zero()&&report.error.kind==error_kind::finite&&report.error.exponent2<=limit,
            SBN3_FATAL_MATH,"trig series final error budget");return report;
}
void calculate(void* raw,sbn3_team_scope* scope) {
    auto& call=*static_cast<Run*>(raw);auto& s=call.state;const std::uint64_t one=1;
    const auto unit=float_view::from_magnitude({&one,1},0);
    const auto U=s.block+1,F=U+1,S=F+5,C=F+6,Next=F+7,Square=F+8;
    round_into(s.values[0].ref(),unit);prefix(s.values[U].ref(),call.input,-std::int64_t(s.halvings));
    multiply(s,1,s.values[U].view(),{},scope,true);detail::float_access::metadata(s.values[1].ref()).negative=true;
    for(std::size_t i=2;i<=s.block;++i)multiply(s,i,s.values[i-1].view(),s.values[1].view(),scope);
    polynomial(s,false,scope);round_into(s.values[C].ref(),s.values[F].view());
    polynomial(s,true,scope);multiply(s,S,s.values[U].view(),s.values[F].view(),scope);
    std::size_t current=S,next=Next;
    for(std::size_t i=0;i<s.halvings;++i){
        multiply(s,next,s.values[current].view(),s.values[C].view(),scope);
        const auto scaled=scale2_into(s.values[next].ref(),s.values[next].view(),1,rounding::nearest_even);
        require(scaled.error.exact(),SBN3_FATAL_MATH,"trig exact doubling");
        multiply(s,Square,s.values[current].view(),{},scope,true);
        const auto square=s.values[Square].view();
        const auto twice_negative=float_view::from_magnitude(square.magnitude().limbs(),square.exponent2()+1,true);
        add(s,C,unit,twice_negative,scope);std::swap(current,next);
    }
    const auto sine=s.values[current].view(),cosine=s.values[C].view();
    require(!sine.zero()&&!sine.negative()&&!cosine.zero()&&!cosine.negative(),SBN3_FATAL_MATH,"trig series component signs");
    const auto signed_sine=call.input.negative()?sine.negated():sine;
    call.report.sine=publish(call.sine,signed_sine,ErrorMagnitude::power_of_two(top(sine)-std::int64_t(s.precision)-9),call.mode);
    call.report.cosine=publish(call.cosine,cosine,ErrorMagnitude::power_of_two(-std::int64_t(s.precision)-9),call.mode);
}
struct Use {State& state;explicit Use(State& s):state(s){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),SBN3_FATAL_LIFETIME,"concurrent trig series");}
    ~Use(){__atomic_store_n(&state.active,0,__ATOMIC_RELEASE);}};
}
FloatDiagnostic trig_series_diagnostic(bits p,unsigned workers) noexcept {
    const auto q=quote(p,workers);FloatDiagnostic d;d.need=q.need;d.recipe="direct_series";d.working_bits=q.working;
    d.metadata_bytes=sizeof(State)+q.banks*sizeof(float_buffer);d.table_bytes=q.mul.tables_bytes+q.square.tables_bytes;
    d.constant_value_bytes=0;d.constant_work_bytes=0;return d;
}
requirements trig_series_program_requirements(bits p,unsigned workers) noexcept{return quote(p,workers).need;}
FloatTrigSeriesPrepared trig_series_program_prepare(bits p,unsigned workers,std::span<std::byte> storage) noexcept {
    const auto q=quote(p,workers);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
            SBN3_FATAL_WORKSPACE,"trig series storage",q.need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());auto* state=::new(frame.allocate(sizeof(State),128))State{};
    state->precision=p.count;state->working=q.working;state->bytes=q.need.scratch_bytes;state->terms=q.terms;state->block=q.block;
    state->halvings=q.halvings;state->words=q.words;state->workers=workers;state->mul_plan=q.mul;state->square_plan=q.square;
    state->values=static_cast<float_buffer*>(frame.allocate(q.banks*sizeof(float_buffer),64));
    for(std::size_t i=0;i<q.banks;++i)::new(state->values+i)float_buffer({frame.alloc<std::uint64_t>(q.words),q.words},bits{q.working});
    state->coefficient_product=frame.alloc<std::uint64_t>(q.words+2);
    auto region=[&](std::size_t bytes,std::size_t alignment){return bytes?
        std::span<std::byte>{static_cast<std::byte*>(frame.allocate(bytes,alignment)),bytes}:std::span<std::byte>{};};
    state->mul=float_product_prepare(state->mul_plan,region(q.mul.scratch_bytes,q.mul.alignment));
    state->square=float_product_prepare(state->square_plan,region(q.square.scratch_bytes,q.square.alignment));
    state->word=region(q.word.scratch_bytes,q.word.scratch_alignment);state->sum=region(q.sum,64);
    state->marker=magic;return {state,storage.first(q.need.scratch_bytes)};
}
TrigSincosReport sincos_series_execute(const FloatTrigSeriesPrepared& prepared,float_ref sine,float_ref cosine,
                                      float_view input,execution ex,rounding mode) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"trig series prepared state");
    auto& state=*static_cast<State*>(prepared.state);
    require(state.marker==magic&&state.bytes==prepared.storage.size()&&
            std::max(sine.precision().count,cosine.precision().count)==state.precision&&ex.workers()>=state.workers&&domain(input),
            SBN3_FATAL_ARGUMENT,"trig series input/shape");
    output_checks(sine,cosine,input,prepared.storage);Use exclusive(state);
    if(tiny(input,state.precision))return leaf(sine,cosine,input,mode);
    Run run{state,sine,cosine,input,mode};
    if(ex.scope)calculate(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,calculate,&run);else serial_team_run(calculate,&run);
    return run.report;
}
requirements trig_series_requirements(float_ref sine,float_ref cosine,float_view input,unsigned workers) noexcept {
    const auto p=std::max(sine.precision().count,cosine.precision().count);
    if(!p||p>maximum_precision||!workers||workers>32||!domain(input))return {support::unsupported,0,128,0,"trig series input/shape"};
    const auto words=(p+63)/64;
    if(sine.capacity()<(sine.precision().count+63)/64||cosine.capacity()<(cosine.precision().count+63)/64)
        return {support::capacity,0,128,words,"trig series output capacity"};
    if(tiny(input,p))return {support::yes,0,128,words,""};return trig_series_program_requirements(bits{p},workers);
}
TrigSincosReport trig_series_into(float_ref sine,float_ref cosine,float_view input,execution ex,rounding mode) noexcept {
    const auto p=std::max(sine.precision().count,cosine.precision().count);
    require(p&&p<=maximum_precision&&domain(input),SBN3_FATAL_ARGUMENT,"trig series input/shape");
    output_checks(sine,cosine,input,ex.scratch);if(tiny(input,p))return leaf(sine,cosine,input,mode);
    const auto prepared=trig_series_program_prepare(bits{p},ex.workers(),ex.scratch);
    return sincos_series_execute(prepared,sine,cosine,input,ex,mode);
}
}
