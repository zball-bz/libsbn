// Original libsbn tests; LGPL-3.0-or-later.
#include "float/trig_agm.hpp"
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
Int ratio(const Int& numerator,const Int& denominator,std::size_t F,bool upper){
    Int a=numerator;const bool negative=ref_sgn(a.n)<0;ref_abs(a.n,a.n);a=shift(a,F);
    auto q=positive_divide(a,denominator,negative?!upper:upper);return negative?neg(q):q;
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
Interval atan_oracle(const Input& input,std::size_t F,const Interval& pi){
    auto x=input;const bool negative=x.negative;x.negative=false;auto a=input_interval(x,F);
    const auto unit=shift(Int(1),F),half=floor2(unit,1);Interval value;
    if(ref_cmp(a.hi.n,half.n)<=0)value=atan_small(a,F);
    else{
        // atan(x)=pi/4+atan((x-1)/(x+1)); transformed |t|<=1/3.
        const Interval t{ratio(sub(a.lo,unit),add(a.lo,unit),F,false),
                         ratio(sub(a.hi,unit),add(a.hi,unit),F,true)};
        value=add(divide_word(pi,4),atan_small(t,F));
    }
    return negative?neg(value):value;
}
struct Pair {Interval sine,cosine;};
Pair sincos_oracle(const Input& input,std::size_t F){
    auto x=input;const bool negative=x.negative;x.negative=false;const auto a=input_interval(x,F);
    const auto square=multiply(a,a,F),unit=exact_interval(shift(Int(1),F));
    Interval st=a,ct=unit,sine=a,cosine=unit;
    for(std::size_t k=1;k<F+16;++k){
        st=divide_word(multiply(st,square,F),(2*k)*(2*k+1));
        ct=divide_word(multiply(ct,square,F),(2*k-1)*(2*k));
        if(ref_cmp_ui(st.hi.n,1)<=0&&ref_cmp_ui(ct.hi.n,1)<=0){
            sine={sub(sine.lo,st.hi),add(sine.hi,st.hi)};
            cosine={sub(cosine.lo,ct.hi),add(cosine.hi,ct.hi)};
            return {negative?neg(sine):sine,cosine};
        }
        sine=k%2?sub(sine,st):add(sine,st);cosine=k%2?sub(cosine,ct):add(cosine,ct);
    }
    assert(false);return {};
}
void certify(sbn::float_view y,sbn::float_report r,const Interval& reference,std::size_t F,std::size_t p){
    assert(y.finite()&&r.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(r.flags,sbn::float_flag::invalid)&&!sbn::has_flag(r.flags,sbn::float_flag::overflow)&&
        !sbn::has_flag(r.flags,sbn::float_flag::underflow));
    const auto center=magnitude(y);const auto scale=std::min({y.exponent2(),-std::int64_t(F),r.error.exact()?y.exponent2():r.error.exponent2});
    const auto c=shift(center,std::size_t(y.exponent2()-scale));
    const auto error=r.error.exact()?Int(0):shift(Int(1),std::size_t(r.error.exponent2-scale));
    const auto lower=shift(reference.lo,std::size_t(-std::int64_t(F)-scale)),upper=shift(reference.hi,std::size_t(-std::int64_t(F)-scale));
    if(ref_cmp(sub(c,error).n,lower.n)>0||ref_cmp(upper.n,add(c,error).n)>0){
        std::fprintf(stderr,"AGM trig endpoint failed p=%zu error=%lld\n",p,(long long)r.error.exponent2);assert(false);}
    // Absolute bounds must remain useful, not merely enclose everything.
    assert(r.error.exact()||r.error.exponent2<=3-std::int64_t(p));++certificates;
}
void norm_certificate(sbn::float_view sine,sbn::float_view cosine,const f::TrigSincosReport& report){
    const auto se=report.sine.error,ce=report.cosine.error;
    const auto base=std::min({sine.exponent2(),cosine.exponent2(),
        se.exact()?sine.exponent2():se.exponent2,ce.exact()?cosine.exponent2():ce.exponent2});
    assert(base<=0&&base>-1000000);
    auto s=shift(magnitude(sine),std::size_t(sine.exponent2()-base));
    auto c=shift(magnitude(cosine),std::size_t(cosine.exponent2()-base));
    const auto es=se.exact()?Int(0):shift(Int(1),std::size_t(se.exponent2-base));
    const auto ec=ce.exact()?Int(0):shift(Int(1),std::size_t(ce.exponent2-base));
    auto difference=sub(add(multiply(s,s),multiply(c,c)),shift(Int(1),std::size_t(-2*base)));
    ref_abs(difference.n,difference.n);ref_abs(s.n,s.n);ref_abs(c.n,c.n);
    const auto bound=add(add(word(multiply(s,es),2),multiply(es,es)),
                         add(word(multiply(c,ec),2),multiply(ec,ec)));
    assert(ref_cmp(difference.n,bound.n)<=0);++certificates;
}
class Scratch {
    std::vector<std::byte> bytes;std::size_t offset=0,n=0;
public:
    explicit Scratch(const sbn::requirements& q):bytes(q.scratch_bytes+q.scratch_alignment+128,std::byte{0x7c}),n(q.scratch_bytes){
        const auto base=reinterpret_cast<std::uintptr_t>(bytes.data());offset=((base+64+q.scratch_alignment-1)&~(q.scratch_alignment-1))-base;}
    std::span<std::byte> span(){return {bytes.data()+offset,n};}
    void check()const{for(std::size_t j=0;j<offset;++j)assert(bytes[j]==std::byte{0x7c});
        for(std::size_t j=offset+n;j<bytes.size();++j)assert(bytes[j]==std::byte{0x7c});}
};
sbn::requirements query(std::size_t p,unsigned workers,f::TrigAgmKind kind){
    allocation_watch_start();const auto q=f::trig_agm_requirements(sbn::bits{p},workers,kind);assert(!allocation_watch_stop());
    assert(q.supported()&&q.scratch_bytes<(std::size_t(1)<<29));return q;
}
struct Prepared {
    std::size_t p;f::TrigAgmKind kind;sbn::requirements need;Scratch storage;f::FloatTrigAgmPrepared handle;
    Prepared(std::size_t precision,f::TrigAgmKind k,unsigned workers=1):p(precision),kind(k),need(query(p,workers,k)),storage(need){
        allocation_watch_start();handle=f::trig_agm_prepare(sbn::bits{p},workers,kind,storage.span());assert(!allocation_watch_stop());}
};
struct Call {Prepared* p;sbn::float_ref sine,cosine;sbn::float_view x;sbn::execution ex;rounding mode;f::TrigSincosReport report;};
void execute(Call& c){if(c.p->kind==f::TrigAgmKind::atan)c.report.sine=f::atan_agm_execute(c.p->handle,c.sine,c.x,c.ex,c.mode);
    else c.report=f::sincos_agm_execute(c.p->handle,c.sine,c.cosine,c.x,c.ex,c.mode);}
void in_scope(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex={};c.ex.scope=scope;execute(c);}
void noop(void*,sbn3_team_scope*){}
void in_child(void* raw,sbn3_team_scope* scope){sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,noop,nullptr,in_scope,raw);}
void run(Prepared& p,const Input& x,const Interval& a,const Interval* b,std::size_t F,rounding mode,
         unsigned alias=0,sbn3_team* team=nullptr,unsigned scope=0){
    const auto n=(p.p+63)/64,capacity=std::max(n,x.words.size());
    std::vector<std::uint64_t> sv(capacity+2,guard),cv(capacity+2,guard);
    if(alias)std::copy(x.words.begin(),x.words.end(),alias==1?sv.begin():cv.begin());
    sbn::float_buffer sine({sv.data(),capacity},sbn::bits{p.p}),cosine({cv.data(),capacity},sbn::bits{p.p});
    auto input=x.view();if(alias)input=sbn::float_view::from_magnitude({alias==1?sv.data():cv.data(),x.words.size()},x.e,x.negative);
    Call c{&p,sine.ref(),cosine.ref(),input,{{},team,nullptr},mode,{}};
    allocation_watch_start();if(scope)sbn3_team_run(team,scope==1?in_scope:in_child,&c);else execute(c);assert(!allocation_watch_stop());
    certify(sine.view(),c.report.sine,a,F,p.p);
    if(b){certify(cosine.view(),c.report.cosine,*b,F,p.p);norm_certificate(sine.view(),cosine.view(),c.report);}
    assert(sv[capacity]==guard&&sv[capacity+1]==guard&&cv[capacity]==guard&&cv[capacity+1]==guard);p.storage.check();
}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease lease{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error e{};sbn3_arena_config cfg{std::size_t(1)<<28,32u<<20};assert(sbn3_arena_create(&cfg,&arena,&e)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&e)==SBN3_OK);sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&lease);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&lease,&tc,&value,&e)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&lease);sbn3_arena_destroy(arena);}
};
void sweep(std::size_t p){
    const std::size_t F=p+96;const auto pi=machin_pi(F);Prepared atan(p,f::TrigAgmKind::atan),sc(p,f::TrigAgmKind::reduced_sincos);
    const Input atan_inputs[]{Input{{2},0,false},Input{{2},0,true},Input{{1},0,false},Input{{1},0,true},Input{{3},-3,false}};
    for(unsigned j=0;j<5;++j){if(p>1025&&j!=0&&j!=3)continue;const auto& x=atan_inputs[j];const auto expected=atan_oracle(x,F,pi);
        run(atan,x,expected,nullptr,F,rounding::nearest_even,j%2);
        if(p==53||p==257)for(auto mode:modes)run(atan,x,expected,nullptr,F,mode,1);
    }
    const Input sincos_inputs[]{Input{{1},0,false},Input{{1},0,true},Input{{3},-2,false},Input{{13},-4,true}};
    for(unsigned j=0;j<4;++j){if(p>1025&&j>1)continue;const auto& x=sincos_inputs[j];const auto expected=sincos_oracle(x,F);
        run(sc,x,expected.sine,&expected.cosine,F,rounding::nearest_even,1+j%2);
        if(p==53||p==257)for(auto mode:modes)run(sc,x,expected.sine,&expected.cosine,F,mode,2);
    }
}
void zeros_tiny_and_oversized(){
    constexpr std::size_t p=65;Prepared atan(p,f::TrigAgmKind::atan),sc(p,f::TrigAgmKind::reduced_sincos);
    std::vector<std::uint64_t> a(2),b(2);sbn::float_buffer sine(a,sbn::bits{p}),cosine(b,sbn::bits{p});const std::uint64_t one=1;
    for(bool negative:{false,true})for(auto mode:modes){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        allocation_watch_start();const auto ar=f::atan_agm_execute(atan.handle,sine.ref(),zero,{},mode);
        const auto sr=f::sincos_agm_execute(sc.handle,sine.ref(),cosine.ref(),zero,{},mode);assert(!allocation_watch_stop());
        assert(ar.error.exact()&&sr.sine.error.exact()&&sr.cosine.error.exact());assert(sine.view().zero()&&sine.view().negative()==negative);
        assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
        for(auto e:{std::int64_t(-200),std::int64_t(-1000000000000LL),-sbn::float_exponent_limit}){
            const auto x=sbn::float_view::from_magnitude({&one,1},e,negative);
            allocation_watch_start();const auto at=f::atan_agm_execute(atan.handle,sine.ref(),x,{},mode);assert(!allocation_watch_stop());
            assert(sbn::compare(sine.view(),x)==std::partial_ordering::equivalent);
            assert(at.error.kind==sbn::error_kind::finite&&at.error.exponent2>=3*(e+1)-1&&at.error.exponent2<-std::int64_t(p));
            allocation_watch_start();const auto st=f::sincos_agm_execute(sc.handle,sine.ref(),cosine.ref(),x,{},mode);assert(!allocation_watch_stop());
            assert(sbn::compare(sine.view(),x)==std::partial_ordering::equivalent);
            assert(sbn::compare(cosine.view(),sbn::float_view::from_magnitude({&one,1},0))==std::partial_ordering::equivalent);
            assert(st.sine.error.kind==sbn::error_kind::finite&&st.sine.error.exponent2>=3*(e+1)-2);
            assert(st.cosine.error.kind==sbn::error_kind::finite&&st.cosine.error.exponent2>=2*(e+1)-1);certificates+=3;
        }
    }
    Input large;large.words.resize(257,UINT64_C(0x789abcde03216475));large.words.back()=3;large.e=-16386;
    const auto F=p+96;const auto pi=machin_pi(F);const auto expected=atan_oracle(large,F,pi);run(atan,large,expected,nullptr,F,rounding::nearest_even,1);
    const auto pair=sincos_oracle(large,F);run(sc,large,pair.sine,&pair.cosine,F,rounding::nearest_even,2);
    assert(!f::trig_agm_requirements(sbn::bits{0},1,f::TrigAgmKind::atan).supported());
    assert(!f::trig_agm_requirements(sbn::bits{65},0,f::TrigAgmKind::atan).supported());
    assert(!f::trig_agm_requirements(sbn::bits{sbn::float_precision_limit},1,f::TrigAgmKind::reduced_sincos).supported());
}
void scopes(){
    Team team;constexpr std::size_t p=257,F=p+96;const auto pi=machin_pi(F);const Input x{{7},-3,false};
    const auto at=atan_oracle(x,F,pi);const auto sc=sincos_oracle(x,F);
    for(unsigned workers:{4u,2u}){
        Prepared ap(p,f::TrigAgmKind::atan,workers),sp(p,f::TrigAgmKind::reduced_sincos,workers);
        run(ap,x,at,nullptr,F,rounding::nearest_even,1,team.value,workers==4?1:2);
        run(sp,x,sc.sine,&sc.cosine,F,rounding::nearest_even,2,team.value,workers==4?1:2);
    }
}
}
int main(){
    for(std::size_t p:{1u,3u,17u,53u,65u,127u,257u,513u,1025u,2049u,4097u})sweep(p);
    zeros_tiny_and_oversized();scopes();
    std::printf("AGM atan/sincos: %zu independent fixed-point interval/Taylor/Machin certificates; modes, reuse, aliases, tiny exponents, scopes and allocation gates PASS\n",certificates);
}
