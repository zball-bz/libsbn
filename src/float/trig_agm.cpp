// Original libsbn code; LGPL-3.0-or-later.
#include "float/trig_agm.hpp"
#include "float/complex.hpp"
#include "float/constants.hpp"
#include "float/error.hpp"
#include <algorithm>
#include <bit>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e3354524731);
struct Shape {std::size_t target=0,m=0,iterations=0,working=0,scalar=0;requirements complex{};};
struct Stage {Shape shape{};FloatComplexPrepared complex{};float_buffer* bank=nullptr;};
struct State {
    std::uint64_t marker=magic;std::uint32_t active=0;TrigAgmKind kind=TrigAgmKind::atan;
    std::size_t precision=0,count=0,bytes=0,pi_bits=0;unsigned workers=1;
    bool external_pi=false,pi_ready=false,owns_bank=false;
    Stage* stages=nullptr;ConstantBankPrepared owned_bank{};ConstantBankBinding constants{};ConstantValueView pi{};
};
struct Quote {requirements need{};ConstantBankQuote bank{};std::size_t count=0,pi_bits=0;bool owns_bank=false;};
std::size_t lg(std::size_t n) noexcept{return n<=1?0:std::bit_width(n-1);}
bool append(std::size_t& at,std::size_t bytes,std::size_t align) noexcept {
    return !bytes||(align_size(at,align,at)&&add_size(at,bytes,at));
}
Shape shape_precision(std::size_t p) noexcept {
    Shape s;s.target=p;s.m=(p+8)/2+lg(p+16)+4;
    while(2*s.m<p+lg(s.m+4)+8)++s.m;
    const auto L=lg(s.m+4),j0=lg(s.m)+1,j1=lg(p+L+12);
    s.iterations=j0+j1;s.working=p+2*s.iterations+L+16;
    s.scalar=complex_scalar_precision(bits{s.working});return s;
}
Shape shape(std::size_t p,unsigned workers) noexcept {
    auto s=shape_precision(p);s.complex=complex_program_requirements(bits{s.working},workers);return s;
}
std::size_t stages(bits p,TrigAgmKind kind) noexcept {
    if(kind==TrigAgmKind::atan)return 1;
    std::size_t n=0,q=5;while(q<p.count+6){++n;q=2*q-3;}return n;
}
std::size_t target_at(bits p,TrigAgmKind kind,std::size_t j) noexcept {
    if(kind==TrigAgmKind::atan)return p.count+8;
    std::size_t q=5;while(j--)q=2*q-3;return 2*q+6;
}
Quote quote(bits p,unsigned workers,TrigAgmKind kind,bool external_pi,ConstantBankBinding binding) noexcept {
    Quote q;q.need.scratch_alignment=128;q.need.output_limbs=(p.count+63)/64;
    if(!p.count||p.count>float_precision_limit/4||!workers||workers>32||
       (kind!=TrigAgmKind::atan&&kind!=TrigAgmKind::reduced_sincos)||
       (external_pi&&kind!=TrigAgmKind::atan)){
        q.need.status=support::unsupported;q.need.reason="AGM trig shape";return q;}
    q.count=stages(p,kind);std::size_t n=0;
    bool ok=append(n,sizeof(State),128)&&append(n,q.count*sizeof(Stage),64);
    for(std::size_t j=0;j<q.count&&ok;++j){const auto s=shape(target_at(p,kind,j),workers);
        if(!s.complex.supported()){q.need.status=support::unsupported;q.need.reason=s.complex.reason;return q;}
        q.pi_bits=std::max(q.pi_bits,s.scalar+4);
        q.need.scratch_alignment=std::max(q.need.scratch_alignment,s.complex.scratch_alignment);
        ok=append(n,16*sizeof(float_buffer),64);
        for(unsigned k=0;k<16&&ok;++k)ok=append(n,((s.scalar+63)/64)*8,64);
        ok=ok&&append(n,s.complex.scratch_bytes,s.complex.scratch_alignment);
    }
    if(binding.supplied()){
        if(!constant_bank_supports(binding,{q.pi_bits,0})){
            q.need.status=support::unsupported;q.need.reason="AGM trig constant bank";return q;}
    }else if(!external_pi){
        q.owns_bank=true;q.bank=constant_bank_requirements({q.pi_bits,0},workers);
        if(!q.bank.supported()){q.need.status=support::unsupported;q.need.reason=q.bank.reason;return q;}
        q.need.scratch_alignment=std::max({q.need.scratch_alignment,q.bank.value_alignment,q.bank.workspace_alignment});
        ok=ok&&append(n,q.bank.value_bytes,q.bank.value_alignment)&&append(n,q.bank.workspace_bytes,q.bank.workspace_alignment);
    }
    if(!ok){q.need.status=support::unsupported;q.need.reason="AGM trig storage extent";return q;}
    q.need.scratch_bytes=n;return q;
}
std::int64_t checked(__int128 e) noexcept {
    require(e>=INT64_MIN&&e<=INT64_MAX,SBN3_FATAL_SIZE,"AGM trig exponent");return std::int64_t(e);
}
float_view scaled(float_view x,std::int64_t shift) noexcept {
    return float_view::from_magnitude(x.magnitude().limbs(),checked(__int128(x.exponent2())+shift),x.negative());
}
void prefix(float_ref out,float_view x) noexcept {
    if(x.zero()){special(out,float_class::finite,x.negative());return;}
    const auto a=x.magnitude();const auto length=magnitude_bits(a),p=out.precision().count,cut=length>p?length-p:0;
    const auto kept=length-cut,words=(kept+63)/64,first=cut/64;const unsigned shift=unsigned(cut%64);
    auto* d=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){const auto j=first+i;
        d->data[i]=(a.data()[j]>>shift)|(shift&&j+1<a.size()?a.data()[j+1]<<(64-shift):0);}
    d->size=words;d->negative=0;auto& m=detail::float_access::metadata(out);
    m.classification=float_class::finite;m.negative=x.negative();m.exponent2=checked(__int128(x.exponent2())+cut);
    detail::float_access::finish(out);
}
void good(float_report r) noexcept {
    require(r.error.kind!=error_kind::unbounded&&!has_flag(r.flags,float_flag::invalid)&&
        !has_flag(r.flags,float_flag::overflow)&&!has_flag(r.flags,float_flag::underflow)&&
        !has_flag(r.flags,float_flag::divide_by_zero),SBN3_FATAL_MATH,"AGM trig scalar range");
}
float_view v(Stage& s,unsigned k) noexcept{return s.bank[k].view();}
void real(Stage& s,ComplexRealKind kind,unsigned out,float_view a,float_view b,execution ex) noexcept {
    good(complex_real_execute(s.complex,kind,s.bank[out].ref(),a,b,ex));
}
float_view one_view(const std::uint64_t& one) noexcept{return float_view::from_magnitude({&one,1},0);}
void atan_stage(Stage& s,float_ref out,float_view t,float_view pi,execution ex) noexcept {
    const bool negative=t.negative();if(negative)t=t.negated();
    const std::uint64_t one=1;const auto unit=one_view(one);
    good(round_into(s.bank[0].ref(),unit));special(s.bank[1].ref(),float_class::finite,false);
    std::int64_t ha=0,hb=0,hp=0;
    auto pair=[&](unsigned j,std::int64_t h){return ScaledComplexView{v(s,j),v(s,j+1),h};};
    auto output=[&](unsigned j,std::int64_t& h){return ScaledComplexRef{s.bank[j].ref(),s.bank[j+1].ref(),&h};};
    good(complex_program_execute(s.complex,ComplexKind::reciprocal,output(2,hb),{unit,t,0},{},ex));
    hb=checked(__int128(hb)+2-__int128(s.shape.m));
    for(std::size_t j=0;j<s.shape.iterations;++j){
        good(complex_program_execute(s.complex,ComplexKind::multiply,output(4,hp),pair(0,ha),pair(2,hb),ex));
        good(complex_program_execute(s.complex,ComplexKind::mean,output(0,ha),pair(0,ha),pair(2,hb),ex));
        good(complex_program_execute(s.complex,ComplexKind::square_root,output(2,hb),pair(4,hp),{},ex));
    }
    good(complex_program_execute(s.complex,ComplexKind::mean,output(0,ha),pair(0,ha),pair(2,hb),ex));
    good(complex_program_execute(s.complex,ComplexKind::reciprocal,output(4,hp),pair(0,ha),{},ex));
    prefix(s.bank[7].ref(),pi);
    real(s,ComplexRealKind::multiply,6,v(s,5),scaled(v(s,7),-1),ex);
    good(scale2_into(out,negative?v(s,6).negated():v(s,6),hp));
}
// Fixed-cost independent atan seed: two half-angle transformations, then 12
// alternating terms at |v|<1/3. Total error is well below 2^-12.
void seed_atan(Stage& s,float_view x,execution ex) noexcept {
    const std::uint64_t one=1;const auto unit=one_view(one);prefix(s.bank[11].ref(),x);
    for(unsigned j=0;j<2;++j){
        real(s,ComplexRealKind::multiply,12,v(s,11),v(s,11),ex);
        real(s,ComplexRealKind::add,12,v(s,12),unit,ex);
        real(s,ComplexRealKind::square_root,13,v(s,12),{},ex);
        real(s,ComplexRealKind::add,13,v(s,13),unit,ex);
        real(s,ComplexRealKind::divide,11,v(s,11),v(s,13),ex);
    }
    real(s,ComplexRealKind::multiply,13,v(s,11),v(s,11),ex);
    prefix(s.bank[14].ref(),v(s,11));prefix(s.bank[10].ref(),v(s,11));
    for(unsigned k=1;k<12;++k){
        real(s,ComplexRealKind::multiply,14,v(s,14),v(s,13),ex);
        const std::uint64_t d=2*k+1;
        real(s,ComplexRealKind::divide,15,v(s,14),one_view(d),ex);
        real(s,ComplexRealKind::add,10,v(s,10),k%2?v(s,15).negated():v(s,15),ex);
    }
    good(scale2_into(s.bank[10].ref(),v(s,10),2));
}
void seed(Stage& s,float_view r,execution ex) noexcept {
    prefix(s.bank[9].ref(),r);int lo=-512,hi=512,mid=0;
    const std::uint64_t one=1;const auto threshold=float_view::from_magnitude({&one,1},-10);
    for(unsigned j=0;j<8;++j){
        mid=(lo+hi)/2;const auto word=std::uint64_t(mid<0?-mid:mid);
        good(round_into(s.bank[8].ref(),float_view::from_magnitude({&word,1},-8,mid<0)));
        seed_atan(s,v(s,8),ex);
        real(s,ComplexRealKind::subtract,12,v(s,10),v(s,9),ex);
        auto residual=v(s,12);const bool neg=residual.negative();if(neg)residual=residual.negated();
        if(compare(residual,threshold)!=std::partial_ordering::greater)return;
        if(neg)lo=mid;else hi=mid;
    }
    mid=(lo+hi)/2;const auto word=std::uint64_t(mid<0?-mid:mid);
    good(round_into(s.bank[8].ref(),float_view::from_magnitude({&word,1},-8,mid<0)));
}
struct Use {State& s;explicit Use(State& x):s(x){std::uint32_t expected=0;
    require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
        SBN3_FATAL_LIFETIME,"concurrent AGM trig");}~Use(){__atomic_store_n(&s.active,0,__ATOMIC_RELEASE);}};
State& state(const FloatTrigAgmPrepared& p,TrigAgmKind kind,execution ex) noexcept {
    require(p.state&&p.state==p.storage.data()&&p.storage.size()>=sizeof(State),SBN3_FATAL_LIFETIME,"AGM trig state");
    auto& s=*static_cast<State*>(p.state);
    require(s.marker==magic&&s.kind==kind&&s.bytes==p.storage.size()&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"AGM trig prepared identity");return s;
}
void validate_io(const FloatTrigAgmPrepared& p,State& s,float_ref out,float_view input) noexcept {
    require(out.precision().count==s.precision&&input.finite(),SBN3_FATAL_ARGUMENT,"AGM trig precision/finite input");
    validate(input);check_output(out,input);const auto* a=detail::float_access::begin(out);
    require(!overlaps(p.storage.data(),p.storage.size(),input.magnitude().data(),input.magnitude().size()*8)&&
        !overlaps(p.storage.data(),p.storage.size(),a->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"AGM trig storage alias");
    constant_bank_check_output(s.constants,out);
    const std::uint64_t limit=s.kind==TrigAgmKind::atan?2:1;
    require(compare(input.negative()?input.negated():input,one_view(limit))!=std::partial_ordering::greater,
        SBN3_FATAL_ARGUMENT,"AGM trig reduced domain");
}
float_report finish(float_ref out,float_view x,std::int64_t radius,rounding mode) noexcept {
    auto r=round_into(out,x,mode);r.error=ErrorMagnitude::from(r.error).add(ErrorMagnitude::power_of_two(radius)).publish();return r;
}
void compute_pi(State& s,execution ex) noexcept {
    require(s.constants.supplied(),SBN3_FATAL_ARGUMENT,"AGM trig requires supplied pi");
    if(s.pi_ready)return;
    if(s.owns_bank)s.constants=ConstantBankBinding(constant_bank_execute(s.owned_bank,ex));
    s.pi=constant_bank_borrow(s.constants.values(),ConstantKind::pi);
    require(s.pi.error.exact()||(s.pi.error.kind==error_kind::finite&&
        s.pi.error.exponent2<=2-std::int64_t(s.pi_bits)),SBN3_FATAL_MATH,"AGM trig pi error budget");
    s.pi_ready=true;
}
}
ConstantBankDemand trig_agm_constant_demand(bits p,TrigAgmKind kind) noexcept {
    if(!p.count||p.count>float_precision_limit/4||(kind!=TrigAgmKind::atan&&kind!=TrigAgmKind::reduced_sincos))return {};
    ConstantBankDemand d;for(std::size_t j=0;j<stages(p,kind);++j)d.pi_bits=std::max(d.pi_bits,shape_precision(target_at(p,kind,j)).scalar+4);
    return d;
}
requirements trig_agm_requirements(bits p,unsigned workers,TrigAgmKind kind,bool external_pi,ConstantBankBinding binding) noexcept{return quote(p,workers,kind,external_pi,binding).need;}
FloatTrigAgmPrepared trig_agm_prepare(bits p,unsigned workers,TrigAgmKind kind,std::span<std::byte> storage,bool external_pi,ConstantBankBinding binding) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"AGM trig query-only bank");
    const auto q=quote(p,workers,kind,external_pi,binding);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"AGM trig storage",q.need.scratch_bytes,storage.size());
    const auto values=binding.values().storage;
    require(!overlaps(storage.data(),storage.size(),values.data(),values.size()),SBN3_FATAL_ARGUMENT,"AGM trig bank/storage alias");
    auto f=Frame::external(storage.data(),storage.size());auto* s=::new(f.allocate(sizeof(State),128))State{};
    s->kind=kind;s->precision=p.count;s->workers=workers;s->count=q.count;s->bytes=q.need.scratch_bytes;
    s->pi_bits=q.pi_bits;s->external_pi=external_pi;
    s->stages=static_cast<Stage*>(f.allocate(q.count*sizeof(Stage),64));
    auto region=[&](std::size_t n,std::size_t a){return std::span<std::byte>{static_cast<std::byte*>(f.allocate(n,a)),n};};
    for(std::size_t j=0;j<q.count;++j){auto* st=::new(s->stages+j)Stage{};st->shape=shape(target_at(p,kind,j),workers);
        st->bank=static_cast<float_buffer*>(f.allocate(16*sizeof(float_buffer),64));const auto n=(st->shape.scalar+63)/64;
        for(unsigned k=0;k<16;++k)::new(st->bank+k)float_buffer({f.alloc<std::uint64_t>(n),n},bits{st->shape.scalar});
        st->complex=complex_program_prepare(bits{st->shape.working},workers,
            region(st->shape.complex.scratch_bytes,st->shape.complex.scratch_alignment));
    }
    s->constants=binding;s->owns_bank=q.owns_bank;
    if(q.owns_bank){
        const auto bank_values=region(q.bank.value_bytes,q.bank.value_alignment);
        const auto bank_work=region(q.bank.workspace_bytes,q.bank.workspace_alignment);
        s->owned_bank=constant_bank_prepare(q.bank,bank_values,bank_work);
        s->constants=constant_bank_pending_binding(s->owned_bank);
    }
    return {s,storage.first(q.need.scratch_bytes)};
}
float_report atan_agm_execute(const FloatTrigAgmPrepared& prepared,float_ref out,float_view x,execution ex,rounding mode) noexcept {
    auto& s=state(prepared,TrigAgmKind::atan,ex);validate_io(prepared,s,out,x);Use use(s);
    if(x.zero())return special(out,float_class::finite,x.negative());
    const auto top=checked(__int128(x.exponent2())+magnitude_bits(x.magnitude()));
    if(top<=-std::int64_t((s.precision+8)/2))
        return finish(out,x,checked(3*__int128(top)),mode);
    compute_pi(s,ex);auto& st=s.stages[0];
    atan_stage(st,st.bank[10].ref(),x,s.pi.point,ex);
    return finish(out,v(st,10),-std::int64_t(st.shape.target),mode);
}
float_report atan_agm_execute_with_pi(const FloatTrigAgmPrepared& prepared,float_ref out,float_view x,
    float_view pi,error_bound pi_error,execution ex,rounding mode) noexcept {
    auto& s=state(prepared,TrigAgmKind::atan,ex);validate_io(prepared,s,out,x);Use use(s);
    validate(pi);const std::uint64_t three=3,four=4;
    require(pi.finite()&&!pi.negative()&&compare(pi,one_view(three))!=std::partial_ordering::less&&
            compare(pi,one_view(four))!=std::partial_ordering::greater&&
            (pi_error.exact()||(pi_error.kind==error_kind::finite&&pi_error.exponent2<=2-std::int64_t(s.pi_bits))),
            SBN3_FATAL_MATH,"AGM trig supplied pi contract");
    const auto* destination=detail::float_access::begin(out);
    require(!overlaps(pi.magnitude().data(),pi.magnitude().size()*8,prepared.storage.data(),prepared.storage.size())&&
            !overlaps(pi.magnitude().data(),pi.magnitude().size()*8,destination->data,out.capacity()*8),
            SBN3_FATAL_ARGUMENT,"AGM trig supplied pi alias");
    if(x.zero())return special(out,float_class::finite,x.negative());
    const auto top=checked(__int128(x.exponent2())+magnitude_bits(x.magnitude()));
    if(top<=-std::int64_t((s.precision+8)/2))return finish(out,x,checked(3*__int128(top)),mode);
    auto& st=s.stages[0];atan_stage(st,st.bank[10].ref(),x,pi,ex);
    return finish(out,v(st,10),-std::int64_t(st.shape.target),mode);
}
TrigSincosReport sincos_agm_execute(const FloatTrigAgmPrepared& prepared,float_ref sine,float_ref cosine,
    float_view r,execution ex,rounding mode) noexcept {
    auto& s=state(prepared,TrigAgmKind::reduced_sincos,ex);validate_io(prepared,s,sine,r);validate_io(prepared,s,cosine,r);
    const auto* a=detail::float_access::begin(sine);const auto* b=detail::float_access::begin(cosine);
    require(!overlaps(a->data,sine.capacity()*8,b->data,cosine.capacity()*8),SBN3_FATAL_ARGUMENT,"sincos output alias");
    Use use(s);const std::uint64_t one=1;const auto unit=one_view(one);
    if(r.zero()){const auto negative=r.negative();return {special(sine,float_class::finite,negative),round_into(cosine,unit,mode)};}
    const auto top=checked(__int128(r.exponent2())+magnitude_bits(r.magnitude()));
    if(top<=-std::int64_t((s.precision+8)/2))
        return {finish(sine,r,checked(3*__int128(top)),mode),
                finish(cosine,unit,checked(2*__int128(top)-1),mode)};
    compute_pi(s,ex);seed(s.stages[0],r,ex);
    for(std::size_t j=0;j<s.count;++j){auto& st=s.stages[j];
        if(j)prefix(st.bank[8].ref(),v(s.stages[j-1],8));
        prefix(st.bank[9].ref(),r);
        atan_stage(st,st.bank[10].ref(),v(st,8),s.pi.point,ex);
        real(st,ComplexRealKind::subtract,11,v(st,9),v(st,10),ex);
        real(st,ComplexRealKind::multiply,12,v(st,8),v(st,8),ex);
        real(st,ComplexRealKind::add,12,v(st,12),unit,ex);
        real(st,ComplexRealKind::multiply,13,v(st,12),v(st,11),ex);
        real(st,ComplexRealKind::add,8,v(st,8),v(st,13),ex);
    }
    auto& st=s.stages[s.count-1];real(st,ComplexRealKind::multiply,11,v(st,8),v(st,8),ex);
    real(st,ComplexRealKind::add,11,v(st,11),unit,ex);
    real(st,ComplexRealKind::square_root,12,v(st,11),{},ex);
    real(st,ComplexRealKind::divide,13,unit,v(st,12),ex);
    real(st,ComplexRealKind::multiply,14,v(st,8),v(st,13),ex);
    const auto radius=-std::int64_t(s.precision)-2;
    return {finish(sine,v(st,14),radius,mode),finish(cosine,v(st,13),radius,mode)};
}
}
