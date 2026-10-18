// Original libsbn code; LGPL-3.0-or-later.
#include "float/trig.hpp"
#include "float/diagnostics.hpp"
#include "float/trig_series.hpp"
#include "float/constants.hpp"
#include "float/product.hpp"
#include "float/division.hpp"
#include "float/error.hpp"
#include "product/program.hpp"
#include <algorithm>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::size_t guard_bits=64;
std::int64_t checked(__int128 e) noexcept {
    require(e>=INT64_MIN&&e<=INT64_MAX,SBN3_FATAL_SIZE,"trig exponent");return std::int64_t(e);
}
std::int64_t top(float_view x) noexcept{return checked(__int128(x.exponent2())+magnitude_bits(x.magnitude()));}
float_view scaled(float_view x,std::int64_t shift) noexcept {
    return float_view::from_magnitude(x.magnitude().limbs(),checked(__int128(x.exponent2())+shift),x.negative());
}
std::uint64_t word_at(sbn3_const_limbs x,std::int64_t bit) noexcept {
    if(bit<0)return bit<=-64?0:(x.count?x.data[0]<<unsigned(-bit):0);
    const auto i=std::size_t(bit/64);const auto shift=unsigned(bit%64);
    return i<x.count?(x.data[i]>>shift)|(shift&&i+1<x.count?x.data[i+1]<<(64-shift):0):0;
}
// Descriptor of (integer*2^-J mod 4)-nearest_integer, with q mod 4.
// Complemented residual words are generated lazily, including the carry.
struct Phase {
    sbn3_const_limbs source{};std::int64_t J=0;std::size_t words=0,first=0,length=0;
    unsigned quadrant=0;bool negative=false;
    std::uint64_t low(std::size_t i)const noexcept {
        auto w=i<source.count?source.data[i]:0;
        if(J%64&&i==std::size_t(J/64))w&=(UINT64_C(1)<<unsigned(J%64))-1;
        return w;
    }
    std::uint64_t at(std::size_t i)const noexcept {
        if(i>=words)return 0;auto w=low(i);
        if(negative)w=i<first?0:i==first?std::uint64_t(0)-w:~w;
        if(J%64&&i==std::size_t(J/64))w&=(UINT64_C(1)<<unsigned(J%64))-1;
        return w;
    }
};
Phase phase(sbn3_const_limbs x,std::int64_t J) noexcept {
    Phase p;p.source=x;p.J=J;
    if(J<=0){const auto shift=-__int128(J);p.quadrant=shift>=2?0:unsigned((x.count?x.data[0]:0)<<unsigned(shift))&3;return p;}
    p.quadrant=unsigned(word_at(x,J)&3);p.negative=(word_at(x,J-1)&1)!=0;
    if(p.negative)p.quadrant=(p.quadrant+1)&3;
    p.words=std::min(x.count,std::size_t((J+63)/64));
    if(p.negative){while(p.first<p.words&&!p.low(p.first))++p.first;require(p.first<p.words,SBN3_FATAL_MATH,"trig residual complement");}
    auto n=p.words;while(n&&!p.at(n-1))--n;
    if(n)p.length=64*(n-1)+64-unsigned(__builtin_clzll(p.at(n-1)));
    return p;
}
ErrorMagnitude phase_prefix(float_ref out,const Phase& p) noexcept {
    if(!p.length){special(out,float_class::finite,false);return {};}
    const auto width=out.precision().count,cut=p.length>width?p.length-width:0;
    const auto kept=p.length-cut,n=(kept+63)/64,first=cut/64;const auto shift=unsigned(cut%64);
    auto* d=detail::float_access::begin(out);
    for(std::size_t i=0;i<n;++i)d->data[i]=(p.at(first+i)>>shift)|(shift?p.at(first+i+1)<<(64-shift):0);
    d->size=n;d->negative=0;auto& m=detail::float_access::metadata(out);
    m.classification=float_class::finite;m.negative=p.negative;m.exponent2=checked(__int128(cut)-p.J);detail::float_access::finish(out);
    return cut?ErrorMagnitude::power_of_two(m.exponent2):ErrorMagnitude{};
}
ErrorMagnitude prefix(float_ref out,float_view x) noexcept {
    if(x.zero()){special(out,float_class::finite,x.negative());return {};}
    const auto input=x.magnitude();const auto bits=magnitude_bits(input),p=out.precision().count,cut=bits>p?bits-p:0;
    const auto n=(bits-cut+63)/64;auto* d=detail::float_access::begin(out);
    for(std::size_t j=0;j<n;++j)d->data[j]=word_at({input.data(),input.size()},std::int64_t(cut+64*j));
    d->size=n;d->negative=0;auto& m=detail::float_access::metadata(out);
    m.classification=float_class::finite;m.negative=x.negative();m.exponent2=checked(__int128(x.exponent2())+cut);
    detail::float_access::finish(out);return cut?ErrorMagnitude::power_of_two(m.exponent2):ErrorMagnitude{};
}
bool append(std::size_t& at,std::size_t n,std::size_t a) noexcept{return !n||(align_size(at,a,at)&&add_size(at,n,at));}
struct Plan {
    requirements need{},inverse{},kernel{};ConstantBankQuote bank{};ConstantBankDemand demand{};
    FloatProductPlan angle_product{};ProductProgramPlan exact_product{};
    std::size_t p=0,Q=0,W=0,constant=0,L=0,H=0,keep=0,an=0,bn=0,capacity=0,tables=0;
    std::int64_t E=0,J=0,source_exponent=0;TrigKind kind=TrigKind::radians;
    bool trivial=false,tiny=false,lattice=false,small_product=false,direct_series=false,series=false,owns_bank=false;unsigned quadrant=0,workers=1;
};
Plan quote(bits sine,bits cosine,float_view x,unsigned workers,TrigKind kind,ConstantBankBinding binding) noexcept {
    Plan p;p.kind=kind;p.workers=workers;p.p=std::max(sine.count,cosine.count);
    p.need.scratch_alignment=128;p.need.output_limbs=(p.p+63)/64;
    if(!sine.count||!cosine.count||p.p>float_precision_limit/8||!workers||workers>32||
       (kind!=TrigKind::radians&&kind!=TrigKind::pi_scaled)||x.magnitude().size()>float_precision_limit/64){
        p.need.status=support::unsupported;p.need.reason="trig input/precision shape";return p;}
    if(binding.supplied()&&!constant_bank_supports(binding,{})){
        p.need.status=support::unsupported;p.need.reason="trig constant bank identity";return p;}
    p.trivial=!x.finite()||x.zero();if(p.trivial)return p;
    const auto E=__int128(x.exponent2())+magnitude_bits(x.magnitude());
    if(E<-float_exponent_limit||E>float_exponent_limit){p.need.status=support::unsupported;p.need.reason="trig input exponent";return p;}
    p.E=std::int64_t(E);p.H=magnitude_bits(x.magnitude());p.Q=p.p+16;p.W=p.Q+32;
    p.tiny=kind==TrigKind::radians&&p.E<=-std::int64_t((p.p+8)/2);
    if(kind==TrigKind::pi_scaled){
        const auto reduced=phase({x.magnitude().data(),x.magnitude().size()},checked(-__int128(x.exponent2())-1));
        if(!reduced.length){p.lattice=true;p.quadrant=reduced.quadrant;return p;}
    }
    if(p.tiny)return p;
    if(kind==TrigKind::radians&&p.p<=4096&&
       (p.E<=0||(x.magnitude().size()==1&&std::has_single_bit(x.magnitude().data()[0])&&
                  x.exponent2()+std::countr_zero(x.magnitude().data()[0])==0))){
        p.direct_series=true;p.kernel=trig_series_program_requirements(bits{p.p},workers);
        p.need=p.kernel;return p;
    }
    p.constant=p.W+64;
    if(kind==TrigKind::radians){
        const auto L=E+p.Q+guard_bits;
        if(L<=0||L>__int128(float_precision_limit/4)){p.need.status=support::unsupported;p.need.reason="trig constant precision";return p;}
        p.L=std::size_t(L);p.constant=std::max(p.constant,p.L+128);
        p.keep=std::min(p.H,p.L);p.source_exponent=checked(__int128(x.exponent2())+p.H-p.keep);
        p.J=checked(__int128(p.L)-p.source_exponent);
        p.an=(p.keep+63)/64;p.bn=(std::min(p.L,std::size_t(p.J)+2)+63)/64;
        p.small_product=p.an<=6&&p.bn<=6;p.capacity=p.an+p.bn;
        if(!p.small_product){sbn3_mul_options o{};o.workers=workers;o.borrow_output=1;
            if(product_program_query(p.an,p.bn,o,p.exact_product)!=SBN3_SUPPORTED){p.need.status=support::unsupported;p.need.reason="trig exact window product";return p;}
            p.capacity=sbn3_mul_output_capacity(&p.exact_product.info);p.tables=p.exact_product.prepared_bytes+128;
            p.need.scratch_alignment=std::max({p.need.scratch_alignment,p.exact_product.info.output_alignment,p.exact_product.info.workspace_alignment});}
        p.inverse=division_program_requirements(bits{p.constant},workers,SBN3_NEWTON_DIVIDE);
        if(!p.inverse.supported()){p.need.status=support::unsupported;p.need.reason=p.inverse.reason;return p;}
    }
    p.series=p.Q<=4096;
    p.demand={p.constant,0};if(!p.series)p.demand=constant_bank_merge(p.demand,trig_agm_constant_demand(bits{p.Q},TrigAgmKind::reduced_sincos));
    auto children=binding;
    if(binding.supplied()){
        if(!constant_bank_supports(binding,p.demand)){p.need.status=support::unsupported;p.need.reason="trig constant bank precision";return p;}
    }else{
        p.owns_bank=true;p.bank=constant_bank_requirements(p.demand,workers);
        if(!p.bank.supported()){p.need.status=support::unsupported;p.need.reason=p.bank.reason;return p;}
        children=constant_bank_query_binding({p.bank.pi_bits,p.bank.log2_bits});
    }
    p.kernel=p.series?trig_series_program_requirements(bits{p.Q},workers):
        trig_agm_requirements(bits{p.Q},workers,TrigAgmKind::reduced_sincos,false,children);
    if(!p.kernel.supported()){
        p.need.status=support::unsupported;p.need.reason=p.kernel.reason;return p;}
    if(float_product_query(p.W,p.W,p.W,false,workers,p.angle_product)!=SBN3_SUPPORTED){
        p.need.status=support::unsupported;p.need.reason="trig angle product";return p;}
    p.need.scratch_alignment=std::max({p.need.scratch_alignment,p.bank.value_alignment,p.bank.workspace_alignment,p.kernel.scratch_alignment,
        p.inverse.scratch_alignment,p.angle_product.alignment});
    std::size_t bytes=0;bool ok=append(bytes,8*sizeof(float_buffer),64);
    for(unsigned j=0;j<8&&ok;++j){const auto bits=j==3||j==4?p.Q:j==5||j==6?p.constant:p.W;ok=append(bytes,((bits+63)/64)*8,64);}
    if(p.owns_bank)ok=ok&&append(bytes,p.bank.value_bytes,p.bank.value_alignment)&&append(bytes,p.bank.workspace_bytes,p.bank.workspace_alignment);
    ok=ok&&append(bytes,p.kernel.scratch_bytes,p.kernel.scratch_alignment)&&
        append(bytes,p.angle_product.scratch_bytes,p.angle_product.alignment);
    if(kind==TrigKind::radians){
        ok=ok&&append(bytes,p.inverse.scratch_bytes,p.inverse.scratch_alignment)&&append(bytes,p.an*8,64)&&append(bytes,p.bn*8,64)&&
            append(bytes,p.capacity*8,p.small_product?64:p.exact_product.info.output_alignment);
        if(!p.small_product)ok=ok&&append(bytes,p.tables,128)&&append(bytes,p.exact_product.info.workspace_bytes,p.exact_product.info.workspace_alignment);
    }
    ok=ok&&add_size(bytes,p.need.scratch_alignment-1,bytes);
    if(!ok){p.need.status=support::unsupported;p.need.reason="trig storage extent";return p;}
    p.need.scratch_bytes=bytes;return p;
}
void good(float_report r) noexcept {
    require(r.error.kind!=error_kind::unbounded&&!has_flag(r.flags,float_flag::invalid)&&
        !has_flag(r.flags,float_flag::overflow)&&!has_flag(r.flags,float_flag::underflow)&&
        !has_flag(r.flags,float_flag::divide_by_zero),SBN3_FATAL_MATH,"trig reduction arithmetic");
}
float_report finish(float_ref out,float_view x,ErrorMagnitude e,rounding mode) noexcept {
    auto r=round_into(out,x,mode);r.error=ErrorMagnitude::from(r.error).add(e).publish();return r;
}
TrigSincosReport publish(float_ref sine,float_ref cosine,float_view s,float_view c,
    ErrorMagnitude es,ErrorMagnitude ec,unsigned q,bool negative,rounding mode) noexcept {
    if(q&1){std::swap(s,c);std::swap(es,ec);}if(q&2)s=s.negated();if((q+1)&2)c=c.negated();if(negative)s=s.negated();
    return {finish(sine,s,es,mode),finish(cosine,c,ec,mode)};
}
void output_checks(float_ref s,float_ref c,float_view x,execution ex) noexcept {
    validate(x);check_output(s,x);check_output(c,x);const auto* a=detail::float_access::begin(s);const auto* b=detail::float_access::begin(c);
    require(!overlaps(a->data,s.capacity()*8,b->data,c.capacity()*8),SBN3_FATAL_ARGUMENT,"trig output alias");
    require(!overlaps(ex.scratch.data(),ex.scratch.size(),x.magnitude().data(),x.magnitude().size()*8)&&
        !overlaps(ex.scratch.data(),ex.scratch.size(),a->data,s.capacity()*8)&&
        !overlaps(ex.scratch.data(),ex.scratch.size(),b->data,c.capacity()*8),SBN3_FATAL_ARGUMENT,"trig scratch alias");
}
void bank_aliases(ConstantBankBinding binding,float_ref out,std::span<std::byte> scratch) noexcept {
    constant_bank_check_output(binding,out);const auto values=binding.values().storage;
    require(!overlaps(values.data(),values.size(),scratch.data(),scratch.size()),SBN3_FATAL_ARGUMENT,"trig constant bank alias");
}
struct ProductCall {const ProductProgram* p;void* work;std::size_t bytes;sbn3_const_limbs a,b;sbn3_limbs out;};
void multiply_action(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<ProductCall*>(raw);auto f=Frame::external(c.work,c.bytes);
    product_program_execute(*c.p,f,scope,c.a,c.b,c.out);}
}
ConstantBankDemand trig_constant_demand(bits sine,bits cosine,float_view x,TrigKind kind) noexcept {
    const auto p=std::max(sine.count,cosine.count);
    if(!sine.count||!cosine.count||p>float_precision_limit/8||!x.finite()||x.zero()||
       x.magnitude().size()>float_precision_limit/64||(kind!=TrigKind::radians&&kind!=TrigKind::pi_scaled))return {};
    const auto E=__int128(x.exponent2())+magnitude_bits(x.magnitude());if(E<-float_exponent_limit||E>float_exponent_limit)return {};
    if(kind==TrigKind::radians){
        if(E<=-__int128((p+8)/2))return {};
        if(p<=4096&&(E<=0||(x.magnitude().size()==1&&std::has_single_bit(x.magnitude().data()[0])&&
             x.exponent2()+std::countr_zero(x.magnitude().data()[0])==0)))return {};
    }else if(!phase({x.magnitude().data(),x.magnitude().size()},checked(-__int128(x.exponent2())-1)).length)return {};
    const auto Q=p+16;ConstantBankDemand d{p+112,0};
    if(kind==TrigKind::radians){const auto L=E+Q+guard_bits;if(L<=0||L>__int128(float_precision_limit/4))return {};
        d.pi_bits=std::max(d.pi_bits,std::size_t(L)+128);}
    if(Q>4096)d=constant_bank_merge(d,trig_agm_constant_demand(bits{Q},TrigAgmKind::reduced_sincos));return d;
}
FloatDiagnostic trig_diagnostic(bits s,bits c,float_view x,bool pi_scaled,unsigned workers) noexcept {
    const auto q=quote(s,c,x,workers,pi_scaled?TrigKind::pi_scaled:TrigKind::radians,{});
    if(q.direct_series)return trig_series_diagnostic(s.count>=c.count?s:c,workers);
    FloatDiagnostic d;d.need=q.need;d.recipe=q.trivial||q.tiny||q.lattice?"analytic":q.direct_series?"direct_series":q.series?"reduced_series":"complex_agm_newton";
    d.working_bits=q.W;d.constant_value_bytes=q.bank.value_bytes;d.constant_work_bytes=q.bank.workspace_bytes;
    d.table_bytes=q.tables;return d;
}
requirements trig_requirements(float_ref s,float_ref c,float_view x,unsigned workers,TrigKind kind,ConstantBankBinding binding) noexcept {
    auto need=quote(s.precision(),c.precision(),x,workers,kind,binding).need;
    if(need.supported()&&(s.capacity()<(s.precision().count+63)/64||c.capacity()<(c.precision().count+63)/64)){
        need.status=support::capacity;need.reason="trig output capacity";}
    return need;
}
// Round a bounded high-word view, carrying a separate error for the omitted
// low source. Only the original exact-base alias is admitted by the caller;
// round_magnitude reads this higher-address window before forward output writes.
static float_report round_prefix(float_ref out,float_view x,rounding mode,ErrorMagnitude extra)noexcept{
    const auto words=x.magnitude().limbs();const auto keep=std::min(words.size(),(out.precision().count+191)/64);
    const auto cut=words.size()-keep;const auto scale=checked(__int128(x.exponent2())+64*__int128(cut));
    auto report=round_magnitude(out,integer_view{words.subspan(cut)},scale,x.negative(),mode);
    auto error=ErrorMagnitude::from(report.error).add(extra);
    if(cut)error=error.add(ErrorMagnitude::power_of_two(scale));
    report.error=error.publish();
    return report;
}
static float_report tiny_sine(float_ref out,float_view x,std::int64_t E,rounding mode)noexcept{
    return round_prefix(out,x,mode,ErrorMagnitude::power_of_two(checked(3*__int128(E))));
}
// Exact pi-scaled lattice values have their own signed-zero convention:
// sinpi(integer) has the input sign; cospi(half-integer) is positive zero.
static float_view lattice_point(bool cosine,unsigned quadrant,bool negative)noexcept{
    static constexpr std::uint64_t one=1;
    static constexpr int sin_values[]{0,1,0,-1},cos_values[]{1,0,-1,0};
    const int value=cosine?cos_values[quadrant]:sin_values[quadrant];
    if(!value)return float_view::from_magnitude({},0,!cosine&&negative);
    return float_view::from_magnitude({&one,1},0,(value<0)!=(!cosine&&negative));
}
TrigSincosReport trig_into(float_ref sine,float_ref cosine,float_view x,execution ex,rounding mode,TrigKind kind,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"trig query-only bank");
    output_checks(sine,cosine,x,ex);bank_aliases(binding,sine,ex.scratch);bank_aliases(binding,cosine,ex.scratch);
    const auto p=quote(sine.precision(),cosine.precision(),x,ex.workers(),kind,binding);
    require(p.need.supported(),SBN3_FATAL_ARGUMENT,p.need.reason);const bool negative=x.negative();
    const std::uint64_t one=1,two=2;const auto unit=float_view::from_magnitude({&one,1},0);
    if(!x.finite()){const auto flag=x.is_nan()?float_flag::none:float_flag::invalid;
        return {special(sine,float_class::quiet_nan,false,flag),special(cosine,float_class::quiet_nan,false,flag)};}
    if(x.zero()||p.lattice){const auto q=p.lattice?p.quadrant:0;
        return {round_into(sine,lattice_point(false,q,negative),mode),round_into(cosine,lattice_point(true,q,negative),mode)};}
    if(p.tiny)return {tiny_sine(sine,x,p.E,mode),finish(cosine,unit,ErrorMagnitude::power_of_two(checked(2*__int128(p.E)-1)),mode)};
    require(ex.scratch.size()>=p.need.scratch_bytes,SBN3_FATAL_WORKSPACE,"trig scratch",p.need.scratch_bytes,ex.scratch.size());
    if(p.direct_series){
        const auto program=trig_series_program_prepare(bits{p.p},ex.workers(),ex.scratch);
        return sincos_series_execute(program,sine,cosine,x,ex,mode);
    }
    auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
    auto region=[&](std::size_t n,std::size_t a){return std::span<std::byte>{static_cast<std::byte*>(frame.allocate(n,a)),n};};
    auto* banks=static_cast<float_buffer*>(frame.allocate(8*sizeof(float_buffer),64));
    for(unsigned j=0;j<8;++j){const auto precision=j==3||j==4?p.Q:j==5||j==6?p.constant:p.W,n=(precision+63)/64;
        ::new(banks+j)float_buffer({frame.alloc<std::uint64_t>(n),n},bits{precision});}
    ConstantBankPrepared owned_bank{};
    if(p.owns_bank){const auto values=region(p.bank.value_bytes,p.bank.value_alignment),work=region(p.bank.workspace_bytes,p.bank.workspace_alignment);
        owned_bank=constant_bank_prepare(p.bank,values,work);binding=constant_bank_pending_binding(owned_bank);}
    const auto kernel_storage=region(p.kernel.scratch_bytes,p.kernel.scratch_alignment);
    FloatTrigAgmPrepared kernel{};FloatTrigSeriesPrepared series{};
    if(p.series)series=trig_series_program_prepare(bits{p.Q},ex.workers(),kernel_storage);
    else kernel=trig_agm_prepare(bits{p.Q},ex.workers(),TrigAgmKind::reduced_sincos,kernel_storage,false,binding);
    const auto angle=float_product_prepare(p.angle_product,region(p.angle_product.scratch_bytes,p.angle_product.alignment));
    FloatDivisionPrepared inverse{};ProductProgram exact{};std::uint64_t *a=nullptr,*b=nullptr,*result=nullptr;void* work=nullptr;
    if(kind==TrigKind::radians){
        inverse=division_program_prepare(bits{p.constant},ex.workers(),SBN3_NEWTON_DIVIDE,region(p.inverse.scratch_bytes,p.inverse.scratch_alignment));
        a=frame.alloc<std::uint64_t>(p.an);b=frame.alloc<std::uint64_t>(p.bn);
        result=static_cast<std::uint64_t*>(frame.allocate(p.capacity*8,p.small_product?64:p.exact_product.info.output_alignment));
        if(!p.small_product){auto tables=frame.subframe(p.tables,128);exact=product_program_prepare(p.exact_product,tables);
            work=frame.allocate(p.exact_product.info.workspace_bytes,p.exact_product.info.workspace_alignment);}
    }
    // All queries, table construction and bindings precede this first seed.
    if(p.owns_bank)binding=ConstantBankBinding(constant_bank_execute(owned_bank,ex));
    const auto pi_report=constant_bank_prefix(binding,ConstantKind::pi,banks[5].ref());good(pi_report);
    const auto small_pi_error=ErrorMagnitude::from(pi_report.error).add(prefix(banks[7].ref(),banks[5].view()));
    Phase reduced;ErrorMagnitude argument_error;
    if(kind==TrigKind::radians){
        const auto reciprocal_report=division_program_execute(inverse,banks[6].ref(),float_view::from_magnitude({&two,1},0),banks[5].view(),ex);good(reciprocal_report);
        // 2/pi is uniformly conditioned on this positive pi enclosure. The
        // chosen 128 guard bits make its numerical error <2^-L, so flooring
        // to L fractional bits leaves total coefficient error <2^(1-L).
        require(pi_report.error.exact()||pi_report.error.exponent2<=-std::int64_t(p.L)-8,SBN3_FATAL_MATH,"trig pi reduction budget");
        require(reciprocal_report.error.exact()||reciprocal_report.error.exponent2<=-std::int64_t(p.L)-8,SBN3_FATAL_MATH,"trig inverse reduction budget");
        const auto cut=p.H-p.keep;const auto source=x.magnitude();
        for(std::size_t j=0;j<p.an;++j)a[j]=word_at({source.data(),source.size()},std::int64_t(cut+64*j));
        const auto cv=banks[6].view();const auto ci=cv.magnitude();const auto shift=checked(-__int128(cv.exponent2())-p.L);
        for(std::size_t j=0;j<p.bn;++j)b[j]=word_at({ci.data(),ci.size()},checked(__int128(shift)+64*j));
        const auto window=std::min(p.L,std::size_t(p.J)+2);if(window%64)b[p.bn-1]&=(UINT64_C(1)<<unsigned(window%64))-1;
        if(p.small_product)sbn3_mul_basecase(result,p.capacity,a,p.an,b,p.bn);
        else{ProductCall call{&exact,work,p.exact_product.info.workspace_bytes,{a,p.an},{b,p.bn},{result,p.capacity}};
            if(ex.scope)multiply_action(&call,ex.scope);else if(ex.team)sbn3_team_run(ex.team,multiply_action,&call);else serial_team_run(multiply_action,&call);}
        reduced=phase({result,p.an+p.bn},p.J);
        argument_error=ErrorMagnitude::power_of_two(checked(__int128(p.E)-p.L+2));
        if(cut)argument_error=argument_error.add(ErrorMagnitude::power_of_two(p.source_exponent));
    }else reduced=phase({x.magnitude().data(),x.magnitude().size()},checked(-__int128(x.exponent2())-1));
    argument_error=argument_error.add(phase_prefix(banks[1].ref(),reduced).scale2(1));
    const auto phase_value=banks[1].view();
    const auto phase_upper=phase_value.zero()?ErrorMagnitude{}:ErrorMagnitude::power_of_two(top(phase_value));
    argument_error=argument_error.add(phase_upper.multiply(small_pi_error.scale2(-1)));
    const auto angle_report=float_product_execute(angle,banks[2].ref(),phase_value,scaled(banks[7].view(),-1),ex);good(angle_report);
    argument_error=argument_error.add(ErrorMagnitude::from(angle_report.error));
    const auto reports=p.series?sincos_series_execute(series,banks[3].ref(),banks[4].ref(),banks[2].view(),ex):
        sincos_agm_execute(kernel,banks[3].ref(),banks[4].ref(),banks[2].view(),ex);
    return publish(sine,cosine,banks[3].view(),banks[4].view(),
        ErrorMagnitude::from(reports.sine.error).add(argument_error),ErrorMagnitude::from(reports.cosine.error).add(argument_error),
        reduced.quadrant,negative,mode);
}
requirements unary_trig_requirements(float_ref out,float_view x,unsigned workers,TrigKind kind,ConstantBankBinding binding)noexcept{
    auto p=quote(out.precision(),out.precision(),x,workers,kind,binding);
    if(!p.need.supported())return p.need;
    if(out.capacity()<p.need.output_limbs){p.need.status=support::capacity;p.need.reason="trig output capacity";return p.need;}
    if(!p.need.scratch_bytes)return p.need;
    std::size_t bytes=0;
    if(!mul_size(p.need.output_limbs,8,bytes)||!add_size(bytes,p.need.scratch_bytes,bytes)||
       !add_size(bytes,2*p.need.scratch_alignment,bytes)){
        p.need.status=support::unsupported;p.need.reason="unary trig storage";return p.need;}
    p.need.scratch_bytes=bytes;return p.need;
}
float_report unary_trig_into(float_ref out,float_view x,bool cosine,TrigKind kind,execution ex,rounding mode,ConstantBankBinding binding)noexcept{
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"unary trig query-only bank");
    validate(x);check_output(out,x);bank_aliases(binding,out,ex.scratch);
    const auto plan=quote(out.precision(),out.precision(),x,ex.workers(),kind,binding);
    require(plan.need.supported(),SBN3_FATAL_ARGUMENT,plan.need.reason);
    if(!x.finite())return special(out,float_class::quiet_nan,false,x.is_nan()?float_flag::none:float_flag::invalid);
    if(plan.tiny){
        if(!cosine)return tiny_sine(out,x,plan.E,mode);
        const std::uint64_t one=1;return finish(out,float_view::from_magnitude({&one,1},0),
            ErrorMagnitude::power_of_two(checked(2*__int128(plan.E)-1)),mode);
    }
    if(x.zero()||plan.lattice){
        const auto q=plan.lattice?plan.quadrant:0;
        return round_into(out,lattice_point(cosine,q,x.negative()),mode);
    }
    const auto* destination=detail::float_access::begin(out);
    require(!overlaps(ex.scratch.data(),ex.scratch.size(),x.magnitude().data(),x.magnitude().size()*8)&&
            !overlaps(ex.scratch.data(),ex.scratch.size(),destination->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"unary trig scratch alias");
    auto frame=Frame::external(ex.scratch.data(),ex.scratch.size());
    const auto words=(out.precision().count+63)/64;
    float_buffer discarded({frame.alloc<std::uint64_t>(words),words},out.precision());
    auto* work=static_cast<std::byte*>(frame.allocate(plan.need.scratch_bytes,plan.need.scratch_alignment));
    const auto report=trig_into(cosine?discarded.ref():out,cosine?out:discarded.ref(),x,
        execution{{work,plan.need.scratch_bytes},ex.team,ex.scope},mode,kind,binding);
    return cosine?report.cosine:report.sine;
}
}
namespace sbn {
requirements sincos_requirements(float_ref s,float_ref c,float_view x,unsigned workers)noexcept{
    return v3::floating::trig_requirements(s,c,x,workers,v3::floating::TrigKind::radians);
}
requirements sincospi_requirements(float_ref s,float_ref c,float_view x,unsigned workers)noexcept{
    return v3::floating::trig_requirements(s,c,x,workers,v3::floating::TrigKind::pi_scaled);
}
sincos_report sincos_into(float_ref s,float_ref c,float_view x,execution ex,rounding mode)noexcept{
    const auto r=v3::floating::trig_into(s,c,x,ex,mode,v3::floating::TrigKind::radians);
    return {r.sine,r.cosine};
}
sincos_report sincospi_into(float_ref s,float_ref c,float_view x,execution ex,rounding mode)noexcept{
    const auto r=v3::floating::trig_into(s,c,x,ex,mode,v3::floating::TrigKind::pi_scaled);
    return {r.sine,r.cosine};
}
#define SBN_TRIG_UNARY(name,kind,cosine) \
requirements name##_requirements(float_ref out,float_view x,unsigned workers)noexcept{ \
    return v3::floating::unary_trig_requirements(out,x,workers,v3::floating::TrigKind::kind); } \
float_report name##_into(float_ref out,float_view x,execution ex,rounding mode)noexcept{ \
    return v3::floating::unary_trig_into(out,x,cosine,v3::floating::TrigKind::kind,ex,mode); }
SBN_TRIG_UNARY(sin,radians,false)
SBN_TRIG_UNARY(cos,radians,true)
SBN_TRIG_UNARY(sinpi,pi_scaled,false)
SBN_TRIG_UNARY(cospi,pi_scaled,true)
#undef SBN_TRIG_UNARY
}
