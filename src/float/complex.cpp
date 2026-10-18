// Original libsbn code; LGPL-3.0-or-later.
#include "float/complex.hpp"
#include "float/product.hpp"
#include "float/roots.hpp"
#include "float/division.hpp"
#include <algorithm>
#include <limits>
#include <new>

namespace sbn::v3::floating {
namespace {
constexpr std::uint64_t magic=UINT64_C(0x53424e33434f4d31);
struct State {
    std::uint64_t marker=magic;std::uint32_t active=0;
    std::size_t bits=0,scalar=0,bytes=0;unsigned workers=1;
    FloatProductPlan plan{};FloatProductPrepared product{};
    FloatRootPrepared root{};FloatDivisionPrepared divide{};
    float_buffer* bank=nullptr;std::span<std::byte> add{};
};
struct Quote {
    requirements need{},root{},divide{};FloatProductPlan product{};
    std::size_t scalar=0,limbs=0,add=0;
};
bool append(std::size_t& at,std::size_t bytes,std::size_t alignment) noexcept {
    return !bytes||(align_size(at,alignment,at)&&add_size(at,bytes,at));
}
Quote quote(bits p,unsigned workers) noexcept {
    Quote q;q.need.scratch_alignment=128;
    q.scalar=complex_scalar_precision(p);
    if(!q.scalar||!workers||workers>32){q.need.status=support::unsupported;q.need.reason="complex precision/workers";return q;}
    q.limbs=(q.scalar+63)/64;q.need.output_limbs=q.limbs;
    if(float_product_query(q.scalar,q.scalar,q.scalar,false,workers,q.product)!=SBN3_SUPPORTED){
        q.need.status=support::unsupported;q.need.reason="complex product shape";return q;}
    q.root=root_program_requirements(bits{q.scalar},workers,RootKind::square_root);
    q.divide=division_program_requirements(bits{q.scalar},workers,SBN3_NEWTON_DIVIDE);
    if(!q.root.supported()||!q.divide.supported()){
        q.need.status=support::unsupported;q.need.reason=!q.root.supported()?q.root.reason:q.divide.reason;return q;}
    q.need.scratch_alignment=std::max({std::size_t(128),q.product.alignment,q.root.scratch_alignment,q.divide.scratch_alignment});
    q.add=q.scalar>256?(q.limbs+2)*8+63:0;
    std::size_t n=0;
    bool ok=append(n,sizeof(State),128)&&append(n,8*sizeof(float_buffer),64);
    for(unsigned j=0;j<8&&ok;++j)ok=append(n,q.limbs*8,64);
    ok=ok&&append(n,q.product.scratch_bytes,q.product.alignment)&&
        append(n,q.root.scratch_bytes,q.root.scratch_alignment)&&
        append(n,q.divide.scratch_bytes,q.divide.scratch_alignment)&&append(n,q.add,64);
    if(!ok){q.need.status=support::unsupported;q.need.reason="complex storage extent";return q;}
    q.need.scratch_bytes=n;return q;
}
std::int64_t checked(__int128 x) noexcept {
    require(x>=INT64_MIN&&x<=INT64_MAX,SBN3_FATAL_SIZE,"complex exponent");return std::int64_t(x);
}
bool zero(ScaledComplexView z) noexcept{return z.real.zero()&&z.imag.zero();}
std::int64_t top(float_view x,std::int64_t scale=0) noexcept {
    return checked(__int128(x.exponent2())+magnitude_bits(x.magnitude())+scale);
}
std::int64_t height(ScaledComplexView x) noexcept {
    if(x.real.zero())return x.imag.zero()?0:top(x.imag,x.scale);
    if(x.imag.zero())return top(x.real,x.scale);
    return std::max(top(x.real,x.scale),top(x.imag,x.scale));
}
float_view scaled(float_view x,std::int64_t e) noexcept {
    return float_view::from_magnitude(x.magnitude().limbs(),checked(__int128(x.exponent2())+e),x.negative());
}
void stage(float_ref out,float_view x,std::int64_t source_scale,std::int64_t h,std::size_t s) noexcept {
    if(x.zero()||__int128(top(x,source_scale))-h<=-__int128(s+8)){
        special(out,float_class::finite,x.negative());return;
    }
    const auto input=x.magnitude();const auto width=magnitude_bits(input);
    const auto cut=width>s?width-s:0,kept=width-cut,words=(kept+63)/64;
    const auto first=cut/64;const unsigned shift=unsigned(cut%64);
    auto* dst=detail::float_access::begin(out);
    for(std::size_t i=0;i<words;++i){const auto j=first+i;
        dst->data[i]=(input.data()[j]>>shift)|(shift&&j+1<input.size()?input.data()[j+1]<<(64-shift):0);}
    dst->size=words;dst->negative=0;
    auto& m=detail::float_access::metadata(out);m.classification=float_class::finite;
    m.negative=x.negative();m.exponent2=checked(__int128(x.exponent2())+cut+source_scale-h);
    detail::float_access::finish(out);
}
void input(State& s,unsigned at,ScaledComplexView x,std::int64_t h) noexcept {
    stage(s.bank[at].ref(),x.real,x.scale,h,s.scalar);
    stage(s.bank[at+1].ref(),x.imag,x.scale,h,s.scalar);
}
void good(float_report r) noexcept {
    require(r.error.kind!=error_kind::unbounded&&!has_flag(r.flags,float_flag::invalid)&&
        !has_flag(r.flags,float_flag::overflow)&&!has_flag(r.flags,float_flag::underflow)&&
        !has_flag(r.flags,float_flag::divide_by_zero),SBN3_FATAL_MATH,"complex scalar range");
}
void product(State& s,unsigned out,float_view a,float_view b,execution ex) noexcept {
    good(float_product_execute(s.product,s.bank[out].ref(),a,b,ex));
}
void add(State& s,unsigned out,float_view a,float_view b,execution ex) noexcept {
    ex.scratch=s.add;good(add_into(s.bank[out].ref(),a,b,ex));
}
void root(State& s,unsigned out,float_view a,execution ex) noexcept {
    good(root_program_execute(s.root,s.bank[out].ref(),a,ex));
}
void divide(State& s,unsigned out,float_view a,float_view b,execution ex) noexcept {
    good(division_program_execute(s.divide,s.bank[out].ref(),a,b,ex));
}
void aliases(State& s,std::span<std::byte> span,ScaledComplexRef out,ScaledComplexView a,ScaledComplexView b) noexcept {
    require(out.scale&&out.real.precision().count==s.scalar&&out.imag.precision().count==s.scalar,
        SBN3_FATAL_ARGUMENT,"complex output precision");
    for(auto x:{a.real,a.imag,b.real,b.imag}){
        require(x.finite(),SBN3_FATAL_ARGUMENT,"complex finite point");validate(x);
        require(!overlaps(span.data(),span.size(),x.magnitude().data(),x.magnitude().size()*8),
            SBN3_FATAL_ARGUMENT,"complex input/storage alias");}
    check_output(out.real,a.real,a.imag);check_output(out.real,b.real,b.imag);
    check_output(out.imag,a.real,a.imag);check_output(out.imag,b.real,b.imag);
    const auto* re=detail::float_access::begin(out.real);const auto* im=detail::float_access::begin(out.imag);
    require(!overlaps(re->data,out.real.capacity()*8,im->data,out.imag.capacity()*8),
        SBN3_FATAL_ARGUMENT,"complex component alias");
    for(auto o:{out.real,out.imag}){const auto* p=detail::float_access::begin(o);
        require(!overlaps(span.data(),span.size(),p->data,o.capacity()*8),SBN3_FATAL_ARGUMENT,"complex output/storage alias");}
    require(!overlaps(span.data(),span.size(),out.scale,sizeof(*out.scale)),SBN3_FATAL_ARGUMENT,"complex scale/storage alias");
}
struct Use {
    State& s;explicit Use(State& x):s(x){std::uint32_t expected=0;
        require(__atomic_compare_exchange_n(&s.active,&expected,1,false,__ATOMIC_ACQUIRE,__ATOMIC_RELAXED),
            SBN3_FATAL_LIFETIME,"concurrent complex program");}
    ~Use(){__atomic_store_n(&s.active,0,__ATOMIC_RELEASE);}
};
}
std::size_t complex_scalar_precision(bits p) noexcept {
    if(!p.count||p.count>float_precision_limit-16)return 0;
    return std::max(std::size_t(32),p.count+16);
}
requirements complex_program_requirements(bits p,unsigned workers) noexcept{return quote(p,workers).need;}
FloatComplexPrepared complex_program_prepare(bits p,unsigned workers,std::span<std::byte> storage) noexcept {
    const auto q=quote(p,workers);require(q.need.supported(),SBN3_FATAL_ARGUMENT,q.need.reason);
    require(storage.size()>=q.need.scratch_bytes&&!(std::uintptr_t(storage.data())&(q.need.scratch_alignment-1)),
        SBN3_FATAL_WORKSPACE,"complex storage",q.need.scratch_bytes,storage.size());
    auto f=Frame::external(storage.data(),storage.size());auto* s=::new(f.allocate(sizeof(State),128))State{};
    s->bits=p.count;s->scalar=q.scalar;s->bytes=q.need.scratch_bytes;s->workers=workers;s->plan=q.product;
    s->bank=static_cast<float_buffer*>(f.allocate(8*sizeof(float_buffer),64));
    for(unsigned j=0;j<8;++j)::new(s->bank+j)float_buffer({f.alloc<std::uint64_t>(q.limbs),q.limbs},bits{q.scalar});
    auto region=[&](std::size_t n,std::size_t align){return std::span<std::byte>{static_cast<std::byte*>(f.allocate(n,align)),n};};
    s->product=float_product_prepare(s->plan,region(q.product.scratch_bytes,q.product.alignment));
    s->root=root_program_prepare(bits{q.scalar},workers,RootKind::square_root,region(q.root.scratch_bytes,q.root.scratch_alignment));
    s->divide=division_program_prepare(bits{q.scalar},workers,SBN3_NEWTON_DIVIDE,region(q.divide.scratch_bytes,q.divide.scratch_alignment));
    if(q.add)s->add=region(q.add,64);
    return {s,storage.first(q.need.scratch_bytes)};
}
float_report complex_program_execute(const FloatComplexPrepared& prepared,ComplexKind kind,
    ScaledComplexRef out,ScaledComplexView a,ScaledComplexView b,execution ex) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),
        SBN3_FATAL_LIFETIME,"complex prepared state");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&ex.workers()>=s.workers,
        SBN3_FATAL_ARGUMENT,"complex prepared identity");
    require(kind==ComplexKind::mean||kind==ComplexKind::multiply||kind==ComplexKind::square_root||kind==ComplexKind::reciprocal,
        SBN3_FATAL_ARGUMENT,"complex operation");
    if(kind==ComplexKind::square_root||kind==ComplexKind::reciprocal)b={};
    aliases(s,prepared.storage,out,a,b);Use use(s);
    if((kind==ComplexKind::multiply&&(zero(a)||zero(b)))||
       (kind==ComplexKind::mean&&zero(a)&&zero(b))||(kind==ComplexKind::square_root&&zero(a))){
        special(out.real,float_class::finite,false);special(out.imag,float_class::finite,false);*out.scale=0;return {};}
    require(kind!=ComplexKind::reciprocal||!zero(a),SBN3_FATAL_ARGUMENT,"complex reciprocal zero");
    auto v=[&](unsigned j){return s.bank[j].view();};std::int64_t scale=0;
    const auto ha=height(a),hb=height(b);
    if(kind==ComplexKind::mean){
        const auto h=zero(a)?hb:zero(b)?ha:std::max(ha,hb);input(s,0,a,h);input(s,2,b,h);
        add(s,4,v(0),v(2),ex);add(s,5,v(1),v(3),ex);scale=checked(__int128(h)-1);
    }else if(kind==ComplexKind::multiply){
        input(s,0,a,ha);input(s,2,b,hb);
        product(s,6,v(0),v(2),ex);product(s,7,v(1),v(3),ex);add(s,4,v(6),v(7).negated(),ex);
        product(s,6,v(0),v(3),ex);product(s,7,v(1),v(2),ex);add(s,5,v(6),v(7),ex);
        scale=checked(__int128(ha)+hb);
    }else if(kind==ComplexKind::reciprocal){
        input(s,0,a,ha);product(s,6,v(0),v(0),ex);product(s,7,v(1),v(1),ex);add(s,6,v(6),v(7),ex);
        const std::uint64_t one=1;divide(s,7,float_view::from_magnitude({&one,1},0),v(6),ex);
        product(s,4,v(0),v(7),ex);product(s,5,v(1).negated(),v(7),ex);scale=checked(-__int128(ha));
    }else{
        const auto q=ha/2-std::int64_t(ha%2<0),parity=ha-2*q;
        input(s,0,a,ha);
        if(parity){good(scale2_into(s.bank[0].ref(),v(0),1));good(scale2_into(s.bank[1].ref(),v(1),1));}
        product(s,6,v(0),v(0),ex);product(s,7,v(1),v(1),ex);add(s,6,v(6),v(7),ex);
        root(s,7,v(6),ex);add(s,6,v(7),v(0),ex);
        require(!v(6).zero()&&!v(6).negative(),SBN3_FATAL_ARGUMENT,"complex square root sector");
        root(s,4,scaled(v(6),-1),ex);divide(s,5,v(1),scaled(v(4),1),ex);scale=q;
    }
    ScaledComplexView result{v(4),v(5),scale};
    require(!zero(result),SBN3_FATAL_MATH,"complex nonzero result");
    const auto h=height(result);stage(out.real,result.real,scale,h,s.scalar);stage(out.imag,result.imag,scale,h,s.scalar);*out.scale=h;
    // Complete wrapper error <=128*rho = 2^(-W)/32. Normalized center norm
    // is <sqrt(2); conversion to this power-of-two absolute bound is safe.
    return {error_bound::power_of_two(checked(__int128(h)-s.bits-4)),float_flag::rounded};
}
float_report complex_real_execute(const FloatComplexPrepared& prepared,ComplexRealKind kind,
    float_ref out,float_view a,float_view b,execution ex) noexcept {
    require(prepared.state&&prepared.state==prepared.storage.data()&&prepared.storage.size()>=sizeof(State),
        SBN3_FATAL_LIFETIME,"complex scalar prepared state");
    auto& s=*static_cast<State*>(prepared.state);
    require(s.marker==magic&&s.bytes==prepared.storage.size()&&ex.workers()>=s.workers&&out.precision().count==s.scalar,
        SBN3_FATAL_ARGUMENT,"complex scalar prepared identity");
    validate(a);validate(b);check_output(out,a,b);
    for(auto x:{a,b})require(!overlaps(prepared.storage.data(),prepared.storage.size(),x.magnitude().data(),x.magnitude().size()*8),
        SBN3_FATAL_ARGUMENT,"complex scalar input/storage alias");
    const auto* o=detail::float_access::begin(out);
    require(!overlaps(prepared.storage.data(),prepared.storage.size(),o->data,out.capacity()*8),
        SBN3_FATAL_ARGUMENT,"complex scalar output/storage alias");
    Use use(s);
    switch(kind){
    case ComplexRealKind::add:ex.scratch=s.add;return add_into(out,a,b,ex);
    case ComplexRealKind::subtract:ex.scratch=s.add;return subtract_into(out,a,b,ex);
    case ComplexRealKind::multiply:return float_product_execute(s.product,out,a,b,ex);
    case ComplexRealKind::square_root:return root_program_execute(s.root,out,a,ex);
    case ComplexRealKind::divide:return division_program_execute(s.divide,out,a,b,ex);
    }
    require(false,SBN3_FATAL_ARGUMENT,"complex scalar kind");return {};
}
}
