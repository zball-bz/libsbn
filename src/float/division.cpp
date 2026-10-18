// Original libsbn code; LGPL-3.0-or-later.
#include "float/division.hpp"
#include "float/diagnostics.hpp"
#include "float/error.hpp"
#include "float/short_division.hpp"
#include "runtime/team.hpp"
#include "value/divrem_words.hpp" // Retains the word-reciprocal donor notice.
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3346444956);
struct DivisionState {
    std::uint64_t marker=magic;std::size_t precision=0,n=0,bytes=0;
    sbn3_newton_kind kind=SBN3_NEWTON_DIVIDE;
    NormalizedNewtonPrepared program{};
    std::uint64_t *a=nullptr,*d=nullptr,*q=nullptr;
    void* workspace=nullptr;std::size_t workspace_bytes=0;
    std::uint32_t active=0;
};
struct Quote {requirements need{};NormalizedNewtonInfo kernel{};std::size_t n=0;};
Quote quote(bits precision,unsigned workers,sbn3_newton_kind kind)noexcept{
    Quote q;
    if(!precision.count||precision.count>float_precision_limit||!workers||workers>32||
       (kind!=SBN3_NEWTON_DIVIDE&&kind!=SBN3_NEWTON_INVERSE)){
        q.need={support::unsupported,0,128,0,"float division shape"};return q;
    }
    // The complete normalized error is <8 units. One full guard limb
    // leaves at least 60 bits of margin relative to a p-bit result ulp.
    q.need.output_limbs=(precision.count+63)/64;q.n=q.need.output_limbs+1;
    if(normalized_newton_query(kind,q.n,workers,q.kernel)!=SBN3_SUPPORTED){
        q.need.status=support::unsupported;q.need.reason="float normalized division support";return q;
    }
    q.need.scratch_alignment=std::max(std::size_t(128),q.kernel.alignment);
    std::size_t bytes=sizeof(DivisionState),values=0;
    const auto words=kind==SBN3_NEWTON_DIVIDE?3*q.n+2:2*q.n+1;
    if(!mul_size(words,8,values)||!add_size(bytes,values,bytes)||
       !add_size(bytes,q.kernel.metadata_bytes,bytes)||!add_size(bytes,q.kernel.workspace_bytes,bytes)||
       !add_size(bytes,7*q.need.scratch_alignment,bytes)){
        q.need.status=support::unsupported;q.need.reason="float division storage extent";return q;
    }
    q.need.scratch_bytes=bytes;return q;
}
bool power_two(float_view x,std::int64_t& exponent)noexcept{
    if(!x.finite()||x.magnitude().size()!=1||!std::has_single_bit(x.magnitude().data()[0]))return false;
    exponent=x.exponent2()+std::countr_zero(x.magnitude().data()[0]);return true;
}
// Effective short width is useful only when discovering it has bounded cost.
// A tall source stays on the prefix route; do not scan its complete low tail
// merely to discover an optional exact-scaling shortcut.
float_view effective_divisor(float_view x,std::size_t n)noexcept{
    if(!x.finite()||x.zero()||x.magnitude().size()>2*n)return x;
    const auto words=x.magnitude().limbs();std::size_t low=0;
    while(low+1<words.size()&&!words[low])++low;
    if(!low)return x;
    return float_view::from_magnitude(words.subspan(low),
        x.exponent2()+std::int64_t(64*low),x.negative());
}
bool trivial(float_ref out,float_view a,float_view b,rounding mode,bool inverse,float_report& report)noexcept{
    const std::uint64_t one=1;
    if(inverse)a=float_view::from_magnitude({&one,1},0);
    const bool sign=a.negative()!=b.negative();
    if(a.is_nan()||b.is_nan()){report=special(out,float_class::quiet_nan,false);return true;}
    if(!a.finite()){
        report=!b.finite()?special(out,float_class::quiet_nan,false,float_flag::invalid):special(out,float_class::infinity,sign);
        return true;
    }
    if(!b.finite()){report=special(out,float_class::finite,sign);return true;}
    if(b.zero()){
        report=a.zero()?special(out,float_class::quiet_nan,false,float_flag::invalid):
            special(out,float_class::infinity,sign,float_flag::divide_by_zero);
        return true;
    }
    if(a.zero()){report=special(out,float_class::finite,sign);return true;}
    std::int64_t exponent=0;
    if(power_two(b,exponent)){
        if(inverse)report=round_magnitude(out,integer_view{{&one,1}},-exponent,sign,mode);
        else report=scale2_into(out,b.negative()?a.negated():a,-exponent,mode);
        return true;
    }
    return false;
}
bool trivial_shape(float_view a,float_view b,bool inverse)noexcept{
    std::int64_t exponent;
    return (!inverse&&(!a.finite()||a.zero()))||!b.finite()||b.zero()||power_two(b,exponent);
}
// floor(magnitude * 2^(64*n-bit_length)): n words in [B^n/2,B^n).
// A negative source bit index denotes virtual low zero bits, never a buffer.
std::uint64_t prefix_word(std::size_t j,std::size_t n,integer_view x)noexcept{
    const auto shift=std::int64_t(magnitude_bits(x))-std::int64_t(64*n);
    const auto bit=std::int64_t(64*j)+shift;
    if(bit<0)return bit<=-64?0:x.data()[0]<<unsigned(-bit);
    const auto i=std::size_t(bit/64);const auto s=unsigned(bit%64);
    return i<x.size()?(x.data()[i]>>s)|(s&&i+1<x.size()?x.data()[i+1]<<(64-s):0):0;
}
void prefix(std::uint64_t* out,std::size_t n,integer_view x)noexcept{
    for(std::size_t j=0;j<n;++j)out[j]=prefix_word(j,n,x);
    require(out[n-1]>>63,SBN3_FATAL_MATH,"float division normalized prefix");
}
float_report finish(float_ref out,const std::uint64_t* q,std::size_t n,std::int64_t exponent,
                    bool negative,rounding mode)noexcept{
    const auto scale=exponent-std::int64_t(64*n);
    auto report=round_magnitude(out,integer_view{{q,n+1}},scale,negative,mode);
    if(has_flag(report.flags,float_flag::underflow))report.error=error_bound::power_of_two(exponent+1);
    else report.error=ErrorMagnitude::from(report.error).add(ErrorMagnitude::power_of_two(scale+3)).publish();
    return report;
}
// A fixed word divisor is an O(p) problem. Do not turn it into an n-word
// Newton inverse by padding its normalization with zeros. One reciprocal of
// the normalized word serves every quotient digit; input padding is virtual.
float_report word_divide(float_ref out,float_view a,float_view b,std::uint64_t* q,
                         std::size_t n,bool inverse,rounding mode)noexcept{
    const auto raw=b.magnitude().data()[0];const unsigned shift=std::countl_zero(raw);
    const auto d=raw<<shift,di=divrem_words::invert_limb(d);
    const auto be=b.exponent2()+64-shift;
    const auto ae=inverse?0:a.exponent2()+std::int64_t(magnitude_bits(a.magnitude()));
    const bool negative=(inverse?false:a.negative())!=b.negative();
    std::uint64_t r;
    if(inverse){
        r=1;
        for(std::size_t j=n+1;j-- >0;){udiv_qrnnd_preinv(q[j],r,r,0,d,di);}
    }else{
        r=prefix_word(n-1,n,a.magnitude());q[n]=r>=d;r-=q[n]?d:0;
        for(std::size_t j=n;j-- >0;){const auto w=j?prefix_word(j-1,n,a.magnitude()):0;
            udiv_qrnnd_preinv(q[j],r,r,w,d,di);}
    }
    return finish(out,q,n,ae-be,negative,mode);
}
// Normalizing a two-word divisor turns the scaled numerator into A*B^2.
// Stream those two virtual zero digits, retaining only the two-word remainder.
// One 3-by-2 reciprocal serves the whole quotient; no long inverse or copied
// shifted numerator is needed. The normalized inverse numerator is B^(n+2).
float_report two_word_divide(float_ref out,float_view a,float_view b,std::uint64_t* q,
                             std::size_t n,bool inverse,rounding mode)noexcept{
    const auto* d=b.magnitude().data();const unsigned shift=std::countl_zero(d[1]);
    const auto d0=d[0]<<shift,d1=(d[1]<<shift)|(shift?d[0]>>(64-shift):0);
    const auto inv=divrem_words::invert_pi1(d1,d0);
    const auto be=b.exponent2()+128-shift;
    const auto ae=inverse?0:a.exponent2()+std::int64_t(magnitude_bits(a.magnitude()));
    const bool negative=(inverse?false:a.negative())!=b.negative();
    // Consume the guaranteed zero leading quotient digit before the loop.
    std::uint64_t r1=inverse?1:0,r0=inverse?0:prefix_word(n-1,n,a.magnitude());
    for(std::size_t j=n+1;j-- >0;){
        const auto word=!inverse&&j>=2?prefix_word(j-2,n,a.magnitude()):0;
        udiv_qr_3by2(q[j],r1,r0,r1,r0,word,d1,d0,inv);
    }
    return finish(out,q,n,ae-be,negative,mode);
}
std::size_t word_bytes(bits p)noexcept{return p.count<=256?0:(((p.count+63)/64+2)*8+63);}
struct ShortQuote {requirements need{};ShortDivisionInfo program{};};
ShortQuote short_quote(bits p,std::size_t k,unsigned workers,bool inverse)noexcept{
    ShortQuote q;q.need.output_limbs=(p.count+63)/64;
    const auto n=q.need.output_limbs+1;
    if(short_division_query(inverse?SBN3_NEWTON_INVERSE:SBN3_NEWTON_DIVIDE,n,k,workers,q.program)!=SBN3_SUPPORTED){
        q.need={support::unsupported,0,128,q.need.output_limbs,"float short division shape"};return q;
    }
    q.need.scratch_alignment=q.program.alignment;
    std::size_t bytes=0;
    if(!mul_size(n+2,8,bytes)||!add_size(bytes,q.program.storage_bytes,bytes)||
       !add_size(bytes,3*q.program.alignment,bytes)){
        q.need.status=support::unsupported;q.need.reason="float short division extent";return q;
    }
    q.need.scratch_bytes=bytes;return q;
}
struct ShortCall {
    ShortDivisionPrepared program{};sbn3_const_limbs a{},d{};sbn3_limbs q{};
    void* work=nullptr;std::size_t work_bytes=0;std::int64_t shift=0;
};
void short_action(void* raw,sbn3_team_scope* scope){
    auto& call=*static_cast<ShortCall*>(raw);auto work=Frame::external(call.work,call.work_bytes);
    short_division_execute(call.program,call.a,call.d,call.q,work,scope,call.shift);
}
void aliases(const DivisionState& state,std::span<std::byte> storage,float_ref out,float_view a,float_view b)noexcept{
    for(auto x:{a,b})require(!overlaps(storage.data(),storage.size(),x.magnitude().data(),x.magnitude().size()*8),
        SBN3_FATAL_ARGUMENT,"float division value/storage alias");
    const auto* value=detail::float_access::begin(out);
    require(!overlaps(storage.data(),storage.size(),value->data,((state.precision+63)/64)*8),
        SBN3_FATAL_ARGUMENT,"float division output/storage alias");
}
struct Run {DivisionState* state;};
void action(void* opaque,sbn3_team_scope* scope){
    auto& s=*static_cast<Run*>(opaque)->state;
    auto work=Frame::external(s.workspace,s.workspace_bytes);
    normalized_newton_execute(s.program,s.kind==SBN3_NEWTON_DIVIDE?sbn3_const_limbs{s.a,s.n+1}:sbn3_const_limbs{},
        {s.d,s.n},{s.q,s.n+1},work,scope);
}
struct Use {
    DivisionState& s;
    explicit Use(DivisionState& state):s(state){std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
            SBN3_FATAL_LIFETIME,"concurrent prepared float division");}
    ~Use(){__atomic_store_n(&s.active,0,__ATOMIC_RELEASE);}
};
template<bool Describe> auto query_description(float_ref out,float_view a,float_view b,unsigned workers,bool inverse)noexcept{
    auto finish=[](requirements need,const char* recipe,std::size_t working=SIZE_MAX,std::size_t metadata=SIZE_MAX){
        if constexpr(Describe){FloatDiagnostic d;d.need=need;d.recipe=recipe;d.working_bits=working;d.metadata_bytes=metadata;
            d.constant_value_bytes=0;d.constant_work_bytes=0;return d;}
        else return need;
    };
    const auto p=out.precision();
    if(!p.count||p.count>float_precision_limit||!workers||workers>32)return finish({support::unsupported,0,128,0,"float division precision/workers"},"unsupported");
    validate(a);validate(b);
    const auto words=(p.count+63)/64;
    if(out.capacity()<words)return finish({support::capacity,0,128,words,"float division output capacity"},"capacity");
    b=effective_divisor(b,words+1);
    if(trivial_shape(a,b,inverse))return finish({support::yes,0,128,words,""},"analytic",p.count,0);
    if(b.magnitude().size()<=2)return finish({support::yes,word_bytes(p),64,words,""},"word_stream",(words+1)*64,0);
    if(b.magnitude().size()<(words+1)/2){const auto q=short_quote(p,b.magnitude().size(),workers,inverse);
        return finish(q.need,q.program.local?"short_local":"short_newton",q.program.n*64,q.program.metadata_bytes);}
    const auto q=quote(p,workers,inverse?SBN3_NEWTON_INVERSE:SBN3_NEWTON_DIVIDE);
    return finish(q.need,q.kernel.local?"normalized_local":"normalized_newton",q.n*64,sizeof(DivisionState)+q.kernel.metadata_bytes);
}
requirements query(float_ref out,float_view a,float_view b,unsigned workers,bool inverse)noexcept{
    return query_description<false>(out,a,b,workers,inverse);
}
float_report fresh(float_ref out,float_view a,float_view b,execution ex,rounding mode,bool inverse)noexcept{
    validate(a);validate(b);check_output(out,a,b);
    const auto original_b=b;b=effective_divisor(b,(out.precision().count+63)/64+1);
    float_report report;if(trivial(out,a,b,mode,inverse,report))return report;
    if(b.magnitude().size()<=2){
        const auto p=out.precision();const auto n=(p.count+63)/64+1;
        const auto run=b.magnitude().size()==1?word_divide:two_word_divide;
        if(p.count<=256){alignas(64)std::uint64_t q[7];return run(out,a,b,q,n,inverse,mode);}
        DivisionState description{};description.precision=p.count;aliases(description,ex.scratch,out,a,original_b);
        require(ex.scratch.size()>=word_bytes(p),SBN3_FATAL_WORKSPACE,"float word division scratch",word_bytes(p),ex.scratch.size());
        auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
        return run(out,a,b,frame.alloc<std::uint64_t>(n+1),n,inverse,mode);
    }
    DivisionState description{};description.precision=out.precision().count;aliases(description,ex.scratch,out,a,original_b);
    const auto n=(out.precision().count+63)/64+1;
    if(b.magnitude().size()<n/2){
        const auto q=short_quote(out.precision(),b.magnitude().size(),ex.workers(),inverse);
        require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
        require(ex.scratch.size()>=q.need.scratch_bytes&&
                !(std::uintptr_t(ex.scratch.data())&(q.need.scratch_alignment-1)),
                SBN3_FATAL_WORKSPACE,"float short division scratch",q.need.scratch_bytes,ex.scratch.size());
        const auto be=b.exponent2()+std::int64_t(magnitude_bits(b.magnitude()));
        const auto ae=inverse?0:a.exponent2()+std::int64_t(magnitude_bits(a.magnitude()));
        const bool negative=(inverse?false:a.negative())!=b.negative();
        auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
        auto* quotient=frame.alloc<std::uint64_t>(n+2);
        auto metadata=frame.subframe(q.program.metadata_bytes,q.program.alignment);
        const auto program=short_division_prepare(q.program.kind,n,q.program.k,ex.workers(),metadata);
        auto* work=frame.allocate(q.program.workspace_bytes,q.program.alignment);
        ShortCall call{program,inverse?sbn3_const_limbs{}:sbn3_const_limbs{a.magnitude().data(),a.magnitude().size()},
            {b.magnitude().data(),b.magnitude().size()},{quotient,n+2},work,q.program.workspace_bytes,
            inverse?0:std::int64_t(64*n)-std::int64_t(magnitude_bits(a.magnitude()))};
        if(ex.scope)short_action(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,short_action,&call);else serial_team_run(short_action,&call);
        return finish(out,quotient,n,ae-be,negative,mode);
    }
    const auto prepared=division_program_prepare(out.precision(),ex.workers(),inverse?SBN3_NEWTON_INVERSE:SBN3_NEWTON_DIVIDE,ex.scratch);
    return division_program_execute(prepared,out,a,original_b,ex,mode);
}
}
FloatDiagnostic division_diagnostic(float_ref out,float_view a,float_view b,bool inverse,unsigned workers) noexcept {
    return query_description<true>(out,a,b,workers,inverse);
}
requirements division_program_requirements(bits p,unsigned workers,sbn3_newton_kind kind)noexcept{return quote(p,workers,kind).need;}
FloatDivisionPrepared division_program_prepare(bits p,unsigned workers,sbn3_newton_kind kind,std::span<std::byte> storage)noexcept{
    const auto q=quote(p,workers,kind);
    require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"float division scratch",q.need.scratch_bytes,storage.size());
    auto frame=Frame::external(storage.data(),storage.size());
    auto* state=::new(frame.allocate(sizeof(DivisionState),128))DivisionState{};
    state->precision=p.count;state->kind=kind;state->n=q.n;state->bytes=q.need.scratch_bytes;
    {
        auto metadata=frame.subframe(q.kernel.metadata_bytes,q.kernel.alignment);
        state->program=normalized_newton_prepare(kind,q.n,workers,metadata);
    }
    if(kind==SBN3_NEWTON_DIVIDE)state->a=frame.alloc<std::uint64_t>(q.n+1);
    state->d=frame.alloc<std::uint64_t>(q.n);state->q=frame.alloc<std::uint64_t>(q.n+1);
    state->workspace=frame.allocate(q.kernel.workspace_bytes,q.kernel.alignment);state->workspace_bytes=q.kernel.workspace_bytes;
    return {state,storage.first(q.need.scratch_bytes)};
}
float_report division_program_execute(const FloatDivisionPrepared& prepared,float_ref out,float_view a,float_view b,
    execution ex,rounding mode)noexcept{
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(DivisionState),
        SBN3_FATAL_LIFETIME,"prepared float division state");
    auto& state=*static_cast<DivisionState*>(prepared.state);
    require(state.marker==magic&&state.bytes==prepared.storage.size()&&out.precision().count==state.precision&&
        ex.workers()>=state.program.info.workers,SBN3_FATAL_ARGUMENT,"prepared float division identity/shape");
    const bool inverse=state.kind==SBN3_NEWTON_INVERSE;
    if(inverse)a={};
    validate(a);validate(b);check_output(out,a,b);aliases(state,prepared.storage,out,a,b);Use exclusive(state);
    b=effective_divisor(b,state.n);
    float_report report;if(trivial(out,a,b,mode,inverse,report))return report;
    if(b.magnitude().size()==1)return word_divide(out,a,b,state.q,state.n,inverse,mode);
    if(b.magnitude().size()==2)return two_word_divide(out,a,b,state.q,state.n,inverse,mode);
    const auto be=b.exponent2()+std::int64_t(magnitude_bits(b.magnitude()));
    const auto ae=inverse?0:a.exponent2()+std::int64_t(magnitude_bits(a.magnitude()));
    const bool negative=(inverse?false:a.negative())!=b.negative();
    prefix(state.d,state.n,b.magnitude());
    if(!inverse){prefix(state.a,state.n,a.magnitude());state.a[state.n]=0;}
    Run run{&state};
    if(ex.scope)action(&run,ex.scope);else if(ex.team)sbn3_team_run(ex.team,action,&run);else serial_team_run(action,&run);
    return finish(out,state.q,state.n,ae-be,negative,mode);
}
}
namespace sbn {
requirements divide_requirements(float_ref out,float_view a,float_view b,unsigned workers)noexcept{return v3::floating::query(out,a,b,workers,false);}
requirements reciprocal_requirements(float_ref out,float_view x,unsigned workers)noexcept{return v3::floating::query(out,{},x,workers,true);}
float_report divide_into(float_ref out,float_view a,float_view b,execution ex,rounding mode)noexcept{return v3::floating::fresh(out,a,b,ex,mode,false);}
float_report reciprocal_into(float_ref out,float_view x,execution ex,rounding mode)noexcept{return v3::floating::fresh(out,{},x,ex,mode,true);}
}
