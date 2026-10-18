// Original libsbn tests; LGPL-3.0-or-later.
#include "float/trig.hpp"
#include "sbn3/arena.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
namespace f=sbn::v3::floating;
using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,
    rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x31fae750428c691b);
std::size_t certificates=0;
struct Int {
    ref_int n;
    Int(){ref_init(n);ref_set_ui(n,0);}explicit Int(std::uint64_t x){ref_init(n);ref_set_ui(n,x);}
    Int(const Int& x){ref_init(n);ref_set(n,x.n);}~Int(){ref_clear(n);}
    Int& operator=(const Int& x){if(this!=&x)ref_set(n,x.n);return *this;}
};
Int add(const Int& a,const Int& b){Int x;ref_add(x.n,a.n,b.n);return x;}
Int sub(const Int& a,const Int& b){Int x;ref_sub(x.n,a.n,b.n);return x;}
Int neg(const Int& a){Int x;ref_neg(x.n,a.n);return x;}
Int shift(Int a,std::size_t n){assert(n<1000000);if(n)ref_mul_2exp(a.n,a.n,n);return a;}
Int floor2(const Int& a,std::size_t n){Int x;ref_fdiv_q_2exp(x.n,a.n,n);return x;}
Int ceil2(const Int& a,std::size_t n){return neg(floor2(neg(a),n));}
Int word(const Int& a,std::uint64_t n){Int x;ref_mul_ui(x.n,a.n,n);return x;}
// Force independent exact scalar products at every oracle size. ref_mul and
// ref_sqrt have production-backed large-input paths and are never called.
Int multiply(const Int& a,const Int& b){
    Int x,aa=a,bb=b,term;const bool negative=(ref_sgn(a.n)<0)!=(ref_sgn(b.n)<0);
    ref_abs(aa.n,aa.n);ref_abs(bb.n,bb.n);if(!ref_sgn(aa.n)||!ref_sgn(bb.n))return x;
    std::vector<std::uint64_t> limbs((ref_sizeinbase(bb.n,2)+63)/64);std::size_t count=0;
    ref_export(limbs.data(),&count,-1,8,0,0,bb.n);assert(count==limbs.size());
    for(std::size_t j=0;j<limbs.size();++j)if(limbs[j]){
        ref_mul_ui(term.n,aa.n,limbs[j]);if(j)ref_mul_2exp(term.n,term.n,64*j);ref_add(x.n,x.n,term.n);}
    if(negative)ref_neg(x.n,x.n);return x;
}
Int floor_word(const Int& a,std::uint64_t n){
    assert(n);Int x,aa=a,den(n);const bool negative=ref_sgn(a.n)<0;ref_abs(aa.n,aa.n);
    // A one-word divisor always uses the independent scalar oracle path.
    ref_fdiv_q(x.n,aa.n,den.n);
    if(negative){if(ref_fdiv_ui(aa.n,n))ref_add_ui(x.n,x.n,1);ref_neg(x.n,x.n);}return x;
}
Int ceil_word(const Int& a,std::uint64_t n){return neg(floor_word(neg(a),n));}
// Independent binary long division for the two argument-transformation
// endpoints. Avoid even the oracle's large-denominator certified divider.
Int positive_divide(const Int& numerator,const Int& denominator,bool ceil){
    assert(ref_sgn(numerator.n)>=0&&ref_sgn(denominator.n)>0);Int q,r=numerator;
    if(ref_cmp(r.n,denominator.n)>=0){
        const auto distance=ref_sizeinbase(r.n,2)-ref_sizeinbase(denominator.n,2);
        for(std::size_t j=distance+1;j-- >0;){const auto d=shift(denominator,j);
            if(ref_cmp(r.n,d.n)>=0){ref_sub(r.n,r.n,d.n);const auto bit=shift(Int(1),j);ref_add(q.n,q.n,bit.n);}}
    }
    if(ceil&&ref_sgn(r.n))ref_add_ui(q.n,q.n,1);return q;
}
struct Interval {Int lo,hi;};
Interval exact_interval(const Int& a){return {a,a};}
Interval add(const Interval& a,const Interval& b){return {add(a.lo,b.lo),add(a.hi,b.hi)};}
Interval neg(const Interval& a){return {neg(a.hi),neg(a.lo)};}
Interval sub(const Interval& a,const Interval& b){return add(a,neg(b));}
Interval word(const Interval& a,std::uint64_t n){return {word(a.lo,n),word(a.hi,n)};}
Interval divide_word(const Interval& a,std::uint64_t n){return {floor_word(a.lo,n),ceil_word(a.hi,n)};}
Interval multiply(const Interval& a,const Interval& b,std::size_t F){
    Int products[]{multiply(a.lo,b.lo),multiply(a.lo,b.hi),multiply(a.hi,b.lo),multiply(a.hi,b.hi)};
    unsigned lo=0,hi=0;for(unsigned j=1;j<4;++j){if(ref_cmp(products[j].n,products[lo].n)<0)lo=j;
        if(ref_cmp(products[j].n,products[hi].n)>0)hi=j;}
    return {floor2(products[lo],F),ceil2(products[hi],F)};
}
Interval atan_small(const Interval& argument,std::size_t F){
    if(ref_sgn(argument.hi.n)<=0){if(!ref_sgn(argument.lo.n))return argument;return neg(atan_small(neg(argument),F));}
    // If the enclosure crosses zero, atan(x) stays between its endpoints.
    if(ref_sgn(argument.lo.n)<0)return argument;
    const auto square=multiply(argument,argument,F);Interval power=argument,sum=argument;
    for(std::size_t k=1;k<F+16;++k){
        power=multiply(power,square,F);const auto term=divide_word(power,2*k+1);
        if(ref_cmp_ui(term.hi.n,1)<=0){
            // Alternating-series next-term bound; use a symmetric enclosure.
            return {sub(sum.lo,term.hi),add(sum.hi,term.hi)};
        }
        sum=k%2?sub(sum,term):add(sum,term);
    }
    assert(false);return {};
}
Interval machin_pi(std::size_t F){
    const auto unit=exact_interval(shift(Int(1),F));
    return sub(word(atan_small(divide_word(unit,5),F),16),word(atan_small(divide_word(unit,239),F),4));
}
struct Input {
    std::vector<std::uint64_t> words;std::int64_t e=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,e,negative);}
};
Int magnitude(sbn::float_view x){Int a;ref_import(a.n,x.magnitude().size(),-1,8,0,0,x.magnitude().data());
    if(x.negative())ref_neg(a.n,a.n);return a;}
Interval input_interval(const Input& x,std::size_t F){
    auto a=magnitude(x.view());const auto shift_bits=__int128(x.e)+F;
    if(shift_bits>=0){assert(shift_bits<1000000);return exact_interval(shift(a,std::size_t(shift_bits)));}
    assert(-shift_bits<1000000);return {floor2(a,std::size_t(-shift_bits)),ceil2(a,std::size_t(-shift_bits))};
}
Int abs_upper(const Interval& x){Int a=x.lo,b=x.hi;ref_abs(a.n,a.n);ref_abs(b.n,b.n);return ref_cmp(a.n,b.n)>0?a:b;}
struct Pair {Interval sine,cosine;};
Pair taylor(const Interval& x,std::size_t F){
    const auto unit=exact_interval(shift(Int(1),F));const auto square=multiply(x,x,F);
    Interval st=x,ct=unit,s=x,c=unit;
    for(std::size_t k=1;k<F+16;++k){
        st=divide_word(multiply(st,square,F),(2*k)*(2*k+1));
        ct=divide_word(multiply(ct,square,F),(2*k-1)*(2*k));
        const auto es=abs_upper(st),ec=abs_upper(ct);
        if(k>=3&&ref_cmp_ui(es.n,1)<=0&&ref_cmp_ui(ec.n,1)<=0)
            return {{sub(s.lo,es),add(s.hi,es)},{sub(c.lo,ec),add(c.hi,ec)}};
        s=k%2?sub(s,st):add(s,st);c=k%2?sub(c,ct):add(c,ct);
    }
    assert(false);return {};
}
Pair rotate(Pair x,unsigned q,bool negative){
    if(q&1)std::swap(x.sine,x.cosine);if(q&2)x.sine=neg(x.sine);
    if((q+1)&2)x.cosine=neg(x.cosine);if(negative)x.sine=neg(x.sine);return x;
}
Interval down(const Interval& x,std::size_t bits){return {floor2(x.lo,bits),ceil2(x.hi,bits)};}
Pair reference(const Input& input,std::size_t F,f::TrigKind kind){
    auto x=input;x.negative=false;unsigned q=0;Interval residual;
    if(kind==f::TrigKind::radians){
        const auto E=std::int64_t(ref_sizeinbase(magnitude(x.view()).n,2))+x.e;
        const auto extra=std::size_t(std::max(std::int64_t(0),E))+16,Fc=F+extra;
        const auto pi=machin_pi(Fc),X=input_interval(x,Fc);
        const auto quotient=positive_divide(add(word(X.lo,4),pi.hi),word(pi.hi,2),false);
        q=unsigned(ref_fdiv_ui(quotient.n,4));
        const Interval multiple{multiply(quotient,pi.lo),multiply(quotient,pi.hi)};
        residual=down(sub(X,divide_word(multiple,2)),extra);
    }else{
        Int m=magnitude(x.view());const auto J=-x.e-1;Int fraction;
        if(J<=0){q=J<=-2?0:unsigned(ref_fdiv_ui(m.n,4)<<unsigned(-J))&3;}
        else{
            assert(J<1000000);Int whole;ref_fdiv_q_2exp(whole.n,m.n,std::size_t(J));q=unsigned(ref_fdiv_ui(whole.n,4));
            ref_fdiv_r_2exp(fraction.n,m.n,std::size_t(J));const auto half=shift(Int(1),std::size_t(J-1));
            if(ref_cmp(fraction.n,half.n)>=0){q=(q+1)&3;fraction=sub(fraction,shift(Int(1),std::size_t(J)));}
        }
        Interval phase;
        if(J>0&&std::size_t(J)>F)phase={floor2(fraction,std::size_t(J)-F),ceil2(fraction,std::size_t(J)-F)};
        else phase=exact_interval(shift(fraction,J>0?F-std::size_t(J):F));
        residual=multiply(phase,divide_word(machin_pi(F),2),F);
    }
    const auto unit=shift(Int(1),F);assert(ref_cmp(abs_upper(residual).n,unit.n)<=0);
    return rotate(taylor(residual,F),q,input.negative);
}
Input exported(const Int& value,std::int64_t exponent){Input x;x.e=exponent;x.negative=ref_sgn(value.n)<0;Int a=value;ref_abs(a.n,a.n);
    if(!ref_sgn(a.n))return x;x.words.resize((ref_sizeinbase(a.n,2)+63)/64);std::size_t n=0;
    ref_export(x.words.data(),&n,-1,8,0,0,a.n);assert(n==x.words.size());return x;}
void certificate(sbn::float_view x,sbn::float_report report,const Interval& ref,std::size_t F,std::size_t p){
    assert(x.finite()&&report.error.kind!=sbn::error_kind::unbounded);
    const auto e=report.error.exact()?x.exponent2():report.error.exponent2;
    const auto base=std::min({x.exponent2(),e,-std::int64_t(F)});
    const auto center=shift(magnitude(x),std::size_t(x.exponent2()-base));
    const auto radius=report.error.exact()?Int(0):shift(Int(1),std::size_t(e-base));
    const auto lo=shift(ref.lo,std::size_t(-std::int64_t(F)-base)),hi=shift(ref.hi,std::size_t(-std::int64_t(F)-base));
    if(ref_cmp(sub(center,radius).n,lo.n)>0||ref_cmp(hi.n,add(center,radius).n)>0){
        std::fprintf(stderr,"general trig certificate p=%zu error=%lld\n",p,(long long)e);assert(false);}
    assert(report.error.exact()||e<=4-std::int64_t(p));++certificates;
}
class Scratch {
    std::vector<std::byte> bytes;std::size_t offset=0,n=0;
public:
    explicit Scratch(const sbn::requirements& q):bytes(q.scratch_bytes+q.scratch_alignment+128,std::byte{0x4a}),n(q.scratch_bytes){
        const auto b=reinterpret_cast<std::uintptr_t>(bytes.data());offset=((b+64+q.scratch_alignment-1)&~(q.scratch_alignment-1))-b;}
    std::span<std::byte> span(){return {bytes.data()+offset,n};}
    void check()const{for(std::size_t j=0;j<offset;++j)assert(bytes[j]==std::byte{0x4a});
        for(std::size_t j=offset+n;j<bytes.size();++j)assert(bytes[j]==std::byte{0x4a});}
};
struct Call {sbn::float_ref sine,cosine;sbn::float_view x;sbn::execution ex;rounding mode;f::TrigKind kind;f::TrigSincosReport report;f::ConstantBankBinding constants{};};
void execute(Call& c){c.report=f::trig_into(c.sine,c.cosine,c.x,c.ex,c.mode,c.kind,c.constants);}
void scope_call(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}
void noop(void*,sbn3_team_scope*){}
void child_call(void* raw,sbn3_team_scope* scope){sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,noop,nullptr,scope_call,raw);}
void run(const Input& x,std::size_t p,f::TrigKind kind,rounding mode=rounding::nearest_even,unsigned alias=0,
         sbn3_team* team=nullptr,unsigned scope=0,const Pair* known=nullptr,std::size_t F=0,f::ConstantBankBinding constants={}){
    if(!F)F=p+112;const auto expected=known?*known:reference(x,F,kind);
    const auto capacity=std::max((p+63)/64,x.words.size());std::vector<std::uint64_t> sv(capacity+2,guard),cv(capacity+2,guard);
    if(alias)std::copy(x.words.begin(),x.words.end(),alias==1?sv.begin():cv.begin());
    sbn::float_buffer sine({sv.data(),capacity},sbn::bits{p}),cosine({cv.data(),capacity},sbn::bits{p});
    auto xv=x.view();if(alias)xv=sbn::float_view::from_magnitude({alias==1?sv.data():cv.data(),x.words.size()},x.e,x.negative);
    const auto workers=scope==2?2:team?sbn3_team_workers(team):1;
    allocation_watch_start();const auto need=f::trig_requirements(sine.ref(),cosine.ref(),xv,workers,kind,constants);assert(!allocation_watch_stop());
    assert(need.supported()&&need.scratch_bytes<(std::size_t(1)<<29));Scratch scratch(need);
    Call c{sine.ref(),cosine.ref(),xv,{scratch.span(),team,nullptr},mode,kind,{},constants};
    allocation_watch_start();if(scope)sbn3_team_run(team,scope==1?scope_call:child_call,&c);else execute(c);assert(!allocation_watch_stop());
    certificate(sine.view(),c.report.sine,expected.sine,F,p);certificate(cosine.view(),c.report.cosine,expected.cosine,F,p);
    assert(sv[capacity]==guard&&sv[capacity+1]==guard&&cv[capacity]==guard&&cv[capacity+1]==guard);scratch.check();
}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease lease{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config cfg{std::size_t(1)<<28,32u<<20};assert(sbn3_arena_create(&cfg,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&lease);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&lease,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&lease);sbn3_arena_destroy(arena);}
};
void lattice_and_specials(){
    std::vector<std::uint64_t> sv(2),cv(2);sbn::float_buffer sine(sv,sbn::bits{65}),cosine(cv,sbn::bits{65});
    for(std::uint64_t n=0;n<16;++n)for(bool negative:{false,true}){
        const auto x=sbn::float_view::from_magnitude({&n,1},-1,negative);
        const auto q=f::trig_requirements(sine.ref(),cosine.ref(),x,1,f::TrigKind::pi_scaled);assert(q.supported()&&!q.scratch_bytes);
        allocation_watch_start();const auto report=f::trig_into(sine.ref(),cosine.ref(),x,{},rounding::nearest_even,f::TrigKind::pi_scaled);assert(!allocation_watch_stop());
        assert(report.sine.error.exact()&&report.cosine.error.exact());const std::uint64_t one=1;
        const auto s=n%2?sbn::float_view::from_magnitude({&one,1},0,((n&2)!=0)!=negative):sbn::float_view{};
        const auto c=n%2?sbn::float_view{}:sbn::float_view::from_magnitude({&one,1},0,(n&2)!=0);
        assert(sbn::compare(sine.view(),s)==std::partial_ordering::equivalent&&sbn::compare(cosine.view(),c)==std::partial_ordering::equivalent);
        if(n%2)assert(!cosine.view().negative());else assert(sine.view().negative()==negative);
        assert(sbn::sinpi_requirements(sine.ref(),x).scratch_bytes==0);
        assert(sbn::cospi_requirements(cosine.ref(),x).scratch_bytes==0);
        allocation_watch_start();
        const auto sr=sbn::sinpi_into(sine.ref(),x),cr=sbn::cospi_into(cosine.ref(),x);
        assert(!allocation_watch_stop()&&sr.error.exact()&&cr.error.exact());
        assert(sbn::compare(sine.view(),s)==std::partial_ordering::equivalent&&sbn::compare(cosine.view(),c)==std::partial_ordering::equivalent);
        if(n%2)assert(!cosine.view().negative());else assert(sine.view().negative()==negative);
    }
    const std::uint64_t one=1;const auto huge=sbn::float_view::from_magnitude({&one,1},sbn::float_exponent_limit-1);
    assert(!f::trig_requirements(sine.ref(),cosine.ref(),huge,1,f::TrigKind::radians).supported());
    const auto cheap=f::trig_requirements(sine.ref(),cosine.ref(),huge,1,f::TrigKind::pi_scaled);assert(cheap.supported()&&!cheap.scratch_bytes);
    allocation_watch_start();const auto exact=f::trig_into(sine.ref(),cosine.ref(),huge,{},rounding::nearest_even,f::TrigKind::pi_scaled);assert(!allocation_watch_stop());
    assert(exact.sine.error.exact()&&exact.cosine.error.exact()&&sine.view().zero());
    assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
    for(auto x:{sbn::float_view::infinity(),sbn::float_view::nan()})for(auto kind:{f::TrigKind::radians,f::TrigKind::pi_scaled}){
        allocation_watch_start();const auto r=f::trig_into(sine.ref(),cosine.ref(),x,{},rounding::nearest_even,kind);assert(!allocation_watch_stop());
        assert(sine.view().is_nan()&&cosine.view().is_nan()&&r.sine.error.kind==sbn::error_kind::unbounded);
    }
}
void tiny_pi(){
    constexpr std::size_t p=65,F=p+112;const auto pi=machin_pi(F);const Interval expected{sub(pi.lo,Int(1)),add(pi.hi,Int(1))};
    const std::uint64_t one=1;std::vector<std::uint64_t> a(2),b(2);sbn::float_buffer sine(a,sbn::bits{p}),cosine(b,sbn::bits{p});
    for(const auto e:{std::int64_t(-1000),std::int64_t(-1000000000000LL),-sbn::float_exponent_limit}){
        const auto x=sbn::float_view::from_magnitude({&one,1},e);const auto need=f::trig_requirements(sine.ref(),cosine.ref(),x,1,f::TrigKind::pi_scaled);assert(need.supported());Scratch scratch(need);
        allocation_watch_start();auto report=f::trig_into(sine.ref(),cosine.ref(),x,{scratch.span()},rounding::nearest_even,f::TrigKind::pi_scaled);assert(!allocation_watch_stop());
        auto y=sine.view();assert(!y.zero());y=sbn::float_view::from_magnitude(y.magnitude().limbs(),y.exponent2()-e);
        assert(report.sine.error.kind==sbn::error_kind::finite);report.sine.error.exponent2-=e;certificate(y,report.sine,expected,F,p);
        assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
        assert(report.cosine.error.kind==sbn::error_kind::finite&&report.cosine.error.exponent2>=2*e+3);scratch.check();
        const auto rn=f::trig_requirements(sine.ref(),cosine.ref(),x,1,f::TrigKind::radians);assert(rn.supported());Scratch rs(rn);
        allocation_watch_start();const auto direct=f::trig_into(sine.ref(),cosine.ref(),x,{rs.span()},rounding::nearest_even,f::TrigKind::radians);assert(!allocation_watch_stop());
        assert(sbn::compare(sine.view(),x)==std::partial_ordering::equivalent);
        assert(direct.sine.error.kind==sbn::error_kind::finite&&direct.sine.error.exponent2>=3*(e+1)-2);
        assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);rs.check();
    }
}
void known_large_shift(){
    constexpr std::size_t p=257,F=p+112,B=1025,Fc=F+B+16;const auto pi=machin_pi(Fc);
    // x = lower(pi)*2^(B-1) + 3/8, an exact dyadic. Its known quotient
    // 2^B is divisible by four. Bound the small phase without rediscovering q.
    const auto delta=shift(Int(3),Fc-B-2);const auto mantissa=add(pi.lo,delta);
    const auto x=exported(mantissa,std::int64_t(B)-1-std::int64_t(Fc));
    const Interval residual_big{add(shift(Int(3),Fc-3),shift(sub(pi.lo,pi.hi),B-1)),shift(Int(3),Fc-3)};
    const auto expected=taylor(down(residual_big,Fc-F),F);run(x,p,f::TrigKind::radians,rounding::nearest_even,1,nullptr,0,&expected,F);
}
void external_constants(){
    for(bool large_phase:{false,true}){
        const std::size_t p=large_phase?65:4097;
        const Input x=large_phase?Input{{1},4097,false}:Input{{3},-2,false};
        const auto demand=f::trig_constant_demand(sbn::bits{p},sbn::bits{p},x.view(),f::TrigKind::radians);
        assert(demand.pi_bits>p);const auto quote=f::constant_bank_requirements(demand,1);assert(quote.supported());
        sbn::workspace values;values.reserve(quote.value_bytes,quote.value_alignment);f::ConstantBankView bank;
        {
            sbn::workspace construction;construction.reserve(quote.workspace_bytes,quote.workspace_alignment);
            const auto prepared=f::constant_bank_prepare(quote,values.on().scratch,construction.on().scratch);
            bank=f::constant_bank_execute(prepared);
        }
        // Independent Machin/Taylor certificate after producer work is dead:
        // both the large phase and the high-p AGM reduced kernel borrow pi.
        run(x,p,f::TrigKind::radians,rounding::nearest_even,1,nullptr,0,nullptr,0,bank);
    }
}
}
int main(){
    for(std::size_t p:{1u,53u,65u,257u,1025u,4097u}){
        run(Input{{3},-2,false},p,f::TrigKind::radians,rounding::nearest_even,1);
        run(Input{{3},0,true},p,f::TrigKind::radians,rounding::nearest_even,2);
        run(Input{{7},-4,false},p,f::TrigKind::pi_scaled,rounding::nearest_even,2);
    }
    for(auto mode:modes){run(Input{{13},-2,true},53,f::TrigKind::radians,mode,1);run(Input{{19},-3,true},53,f::TrigKind::pi_scaled,mode,2);}
    run(Input{{3},257,false},257,f::TrigKind::radians);run(Input{{1},4097,false},65,f::TrigKind::radians);
    Input near_integer=exported(add(shift(Int(1),1200),Int(1)),-200);run(near_integer,65,f::TrigKind::pi_scaled,rounding::nearest_even,1,nullptr,0,nullptr,400);
    near_integer=exported(sub(shift(Int(1),1200),Int(1)),-200);run(near_integer,65,f::TrigKind::pi_scaled,rounding::nearest_even,2,nullptr,0,nullptr,400);
    near_integer=exported(add(add(shift(Int(1),1200),shift(Int(1),199)),Int(1)),-200);run(near_integer,65,f::TrigKind::pi_scaled,rounding::nearest_even,1,nullptr,0,nullptr,400);
    Input wide;wide.words.resize(257,UINT64_C(0x89abcdef01234567));wide.words.back()=3;wide.e=-16386;run(wide,65,f::TrigKind::radians,rounding::nearest_even,1);
    known_large_shift();lattice_and_specials();tiny_pi();external_constants();Team team;
    run(Input{{5},127,false},65,f::TrigKind::radians,rounding::nearest_even,1,team.value,1);
    run(Input{{27},-3,true},65,f::TrigKind::pi_scaled,rounding::nearest_even,2,team.value,2);
    std::printf("general sincos/sinpi/cospi: %zu independent Taylor/Machin phase certificates; large exponents, exact lattice, tiny residuals, aliases, scopes and allocation gates PASS\n",certificates);
}
