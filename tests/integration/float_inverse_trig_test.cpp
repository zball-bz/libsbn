// Original libsbn tests; LGPL-3.0-or-later.
#include "float/inverse_trig.hpp"
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
// Restoring binary square root: exact add/sub/shift only, no ref_sqrt and no
// production reciprocal/root witness. remainder certifies floor vs ceil.
struct Root {Int value,remainder;};
Root integer_root(const Int& n){
    assert(ref_sgn(n.n)>=0);Root r;const auto bits=ref_sgn(n.n)?ref_sizeinbase(n.n,2):0;
    std::vector<std::uint64_t> limbs((bits+63)/64);std::size_t count=0;
    if(bits)ref_export(limbs.data(),&count,-1,8,0,0,n.n);
    for(std::size_t j=(bits+1)/2;j-- >0;){
        const auto digit=(limbs[(2*j)/64]>>((2*j)%64))&3;
        r.remainder=add(word(r.remainder,4),Int(digit));const auto trial=add(word(r.value,4),Int(1));
        r.value=word(r.value,2);
        if(ref_cmp(r.remainder.n,trial.n)>=0){r.remainder=sub(r.remainder,trial);r.value=add(r.value,Int(1));}
    }
    return r;
}
Interval root(const Interval& x,std::size_t F){
    assert(ref_sgn(x.lo.n)>=0);const auto lo=integer_root(shift(x.lo,F)),hi=integer_root(shift(x.hi,F));
    return {lo.value,ref_sgn(hi.remainder.n)?add(hi.value,Int(1)):hi.value};
}
Interval divide_positive(const Interval& a,const Interval& b,std::size_t F){
    assert(ref_sgn(a.lo.n)>=0&&ref_sgn(b.lo.n)>0);
    return {ratio(a.lo,b.hi,F,false),ratio(a.hi,b.lo,F,true)};
}
Interval atan_positive(const Interval& x,std::size_t F,const Interval& pi){
    const auto one=shift(Int(1),F),half=floor2(one,1);
    if(ref_cmp(x.hi.n,half.n)<=0)return atan_small(x,F);
    const Interval t{ratio(sub(x.lo,one),add(x.lo,one),F,false),ratio(sub(x.hi,one),add(x.hi,one),F,true)};
    return add(divide_word(pi,4),atan_small(t,F));
}
Interval atan2_interval(const Interval& y,const Interval& x,bool yn,bool xn,std::size_t F,const Interval& pi){
    Interval angle;
    if(!ref_sgn(y.hi.n))angle=xn?pi:exact_interval(Int(0));
    else if(!ref_sgn(x.hi.n))angle=divide_word(pi,2);
    else{
        const bool swap=ref_cmp(y.lo.n,x.lo.n)>0;
        angle=atan_positive(divide_positive(swap?x:y,swap?y:x,F),F,pi);
        if(swap)angle=sub(divide_word(pi,2),angle);if(xn)angle=sub(pi,angle);
    }
    return yn?neg(angle):angle;
}
Interval reference(const Input& a,const Input& b,f::InverseTrigKind kind,std::size_t F,const Interval& pi){
    if(kind==f::InverseTrigKind::asin||kind==f::InverseTrigKind::acos){
        auto input=a;input.negative=false;const auto x=input_interval(input,F),one=exact_interval(shift(Int(1),F));
        // Deliberately use 1-x^2, independently of the implementation's
        // (1-x)(1+x), at enough precision to enclose endpoint cancellation.
        const auto y=root(sub(one,multiply(x,x,F)),F);
        return kind==f::InverseTrigKind::asin?atan2_interval(x,y,a.negative,false,F,pi):
            atan2_interval(y,x,false,a.negative,F,pi);
    }
    Input y=a,x=kind==f::InverseTrigKind::atan?Input{{1},0,false}:b;
    const bool yn=y.negative,xn=x.negative;y.negative=x.negative=false;
    const auto yt=std::int64_t(ref_sizeinbase(magnitude(y.view()).n,2))+y.e;
    const auto xt=std::int64_t(ref_sizeinbase(magnitude(x.view()).n,2))+x.e;
    const auto common=std::max(yt,xt);y.e-=common;x.e-=common;
    return atan2_interval(input_interval(y,F),input_interval(x,F),yn,xn,F,pi);
}
Input exported(const Int& n,std::int64_t e){Input a;a.e=e;a.negative=ref_sgn(n.n)<0;Int x=n;ref_abs(x.n,x.n);
    if(!ref_sgn(x.n))return a;a.words.resize((ref_sizeinbase(x.n,2)+63)/64);std::size_t count=0;
    ref_export(a.words.data(),&count,-1,8,0,0,x.n);assert(count==a.words.size());return a;}
void certify(sbn::float_view x,sbn::float_report report,const Interval& expected,std::size_t F,std::size_t p,rounding mode){
    assert(x.finite()&&report.error.kind!=sbn::error_kind::unbounded);
    const auto e=report.error.exact()?x.exponent2():report.error.exponent2;
    const auto base=std::min({x.exponent2(),e,-std::int64_t(F)});
    const auto center=shift(magnitude(x),std::size_t(x.exponent2()-base));
    const auto radius=report.error.exact()?Int(0):shift(Int(1),std::size_t(e-base));
    const auto lo=shift(expected.lo,std::size_t(-std::int64_t(F)-base)),hi=shift(expected.hi,std::size_t(-std::int64_t(F)-base));
    if(ref_cmp(sub(center,radius).n,lo.n)>0||ref_cmp(hi.n,add(center,radius).n)>0){
        std::fprintf(stderr,"inverse trig bound failure p=%zu mode=%u error=%lld\n",p,unsigned(mode),(long long)e);assert(false);}
    if(!x.zero()){const auto top=x.exponent2()+std::int64_t(ref_sizeinbase(magnitude(x).n,2));
        assert(report.error.exact()||e<=top-std::int64_t(p)+(mode==rounding::nearest_even?0:1));}
    ++certificates;
}
class Scratch {
    std::vector<std::byte> bytes;std::size_t offset=0,n=0;
public:
    explicit Scratch(const sbn::requirements& q):bytes(q.scratch_bytes+q.scratch_alignment+128,std::byte{0x3e}),n(q.scratch_bytes){
        const auto b=reinterpret_cast<std::uintptr_t>(bytes.data());offset=((b+64+q.scratch_alignment-1)&~(q.scratch_alignment-1))-b;}
    std::span<std::byte> span(){return {bytes.data()+offset,n};}
    void check()const{for(std::size_t j=0;j<offset;++j)assert(bytes[j]==std::byte{0x3e});
        for(std::size_t j=offset+n;j<bytes.size();++j)assert(bytes[j]==std::byte{0x3e});}
};
sbn::requirements query(std::size_t p,unsigned workers,f::InverseTrigKind kind){
    allocation_watch_start();const auto q=f::inverse_trig_program_requirements(sbn::bits{p},workers,kind);assert(!allocation_watch_stop());
    assert(q.supported()&&q.scratch_bytes<(std::size_t(1)<<29));return q;
}
struct Prepared {
    std::size_t p;f::InverseTrigKind kind;sbn::requirements need;Scratch storage;f::FloatInverseTrigPrepared handle;
    Prepared(std::size_t precision,f::InverseTrigKind k,unsigned workers=1):p(precision),kind(k),need(query(p,workers,k)),storage(need){
        allocation_watch_start();handle=f::inverse_trig_prepare(sbn::bits{p},workers,kind,storage.span());assert(!allocation_watch_stop());}
};
struct Call {Prepared* p;sbn::float_ref out;sbn::float_view a,b;sbn::execution ex;rounding mode;bool fresh;sbn::float_report report;};
void execute(Call& c){c.report=c.fresh?f::inverse_trig_into(c.out,c.a,c.b,c.ex,c.mode,c.p->kind):
    f::inverse_trig_execute(c.p->handle,c.out,c.a,c.b,c.ex,c.mode);}
void scope_call(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}
void noop(void*,sbn3_team_scope*){}
void child_call(void* raw,sbn3_team_scope* scope){sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,noop,nullptr,scope_call,raw);}
void run(Prepared& p,const Input& a,const Input& b,const Interval& expected,std::size_t F,rounding mode=rounding::nearest_even,
         unsigned alias=0,bool fresh=false,sbn3_team* team=nullptr,unsigned scope=0){
    const auto capacity=std::max({(p.p+63)/64,a.words.size(),b.words.size()});std::vector<std::uint64_t> words(capacity+2,guard);
    if(alias){const auto& input=alias==1?a:b;std::copy(input.words.begin(),input.words.end(),words.begin());}
    sbn::float_buffer out({words.data(),capacity},sbn::bits{p.p});auto av=a.view(),bv=b.view();
    if(alias==1)av=sbn::float_view::from_magnitude({words.data(),a.words.size()},a.e,a.negative);
    if(alias==2)bv=sbn::float_view::from_magnitude({words.data(),b.words.size()},b.e,b.negative);
    const auto saved_a=a.words,saved_b=b.words;
    Call c{&p,out.ref(),av,bv,{p.storage.span(),team,nullptr},mode,fresh,{}};
    allocation_watch_start();if(scope)sbn3_team_run(team,scope==1?scope_call:child_call,&c);else execute(c);assert(!allocation_watch_stop());
    certify(out.view(),c.report,expected,F,p.p,mode);assert(words[capacity]==guard&&words[capacity+1]==guard);
    assert(a.words==saved_a&&b.words==saved_b);p.storage.check();
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
    const std::size_t F=p+112;const auto pi=machin_pi(F);
    for(auto kind:{f::InverseTrigKind::atan,f::InverseTrigKind::atan2,f::InverseTrigKind::asin,f::InverseTrigKind::acos}){
        Prepared prepared(p,kind);Input a{{1},kind==f::InverseTrigKind::atan||kind==f::InverseTrigKind::atan2?0:-1,false},b{{1},0,false};
        const auto special=divide_word(pi,kind==f::InverseTrigKind::asin?6:kind==f::InverseTrigKind::acos?3:4);
        run(prepared,a,b,special,F,rounding::nearest_even,1,true);
        a.negative=true;const auto negative=kind==f::InverseTrigKind::acos?divide_word(word(pi,2),3):neg(special);
        run(prepared,a,b,negative,F,rounding::nearest_even,1);
        if(p<=513){a={{3},-3,false};b={{7},-3,kind==f::InverseTrigKind::atan2};const auto expected=reference(a,b,kind,F,pi);
            run(prepared,a,b,expected,F,rounding::nearest_even,kind==f::InverseTrigKind::atan2?2:1);
            if(p==53)for(auto mode:modes)run(prepared,a,b,expected,F,mode,1);
        }
    }
}
void endpoints(){
    constexpr std::size_t p=65,N=1200,F=p+N+112;const auto pi=machin_pi(F);
    auto x=exported(sub(shift(Int(1),N),Int(1)),-std::int64_t(N));
    for(auto kind:{f::InverseTrigKind::asin,f::InverseTrigKind::acos}){Prepared prepared(p,kind);
        for(bool negative:{false,true}){x.negative=negative;const auto expected=reference(x,{},kind,F,pi);
            run(prepared,x,{},expected,F,rounding::nearest_even,1);}}
    // Distinct non-limb precision and endpoint distance; relative acos bound.
    constexpr std::size_t p2=4097,N2=257,F2=p2+N2+112;const auto pi2=machin_pi(F2);Prepared acos(p2,f::InverseTrigKind::acos);
    x=exported(sub(shift(Int(1),N2),Int(1)),-std::int64_t(N2));run(acos,x,{},reference(x,{},acos.kind,F2,pi2),F2);
}
void tiny_and_specials(){
    constexpr std::size_t p=65,F=p+112;const auto pi=machin_pi(F);std::vector<std::uint64_t> words(2);sbn::float_buffer out(words,sbn::bits{p});
    const std::uint64_t one=1;const auto unit=sbn::float_view::from_magnitude({&one,1},0);
    for(auto kind:{f::InverseTrigKind::atan,f::InverseTrigKind::asin}){Prepared prepared(p,kind);
        for(bool negative:{false,true}){
            allocation_watch_start();const auto z=f::inverse_trig_execute(prepared.handle,out.ref(),sbn::float_view::from_magnitude({},0,negative));assert(!allocation_watch_stop());
            assert(z.error.exact()&&out.view().zero()&&out.view().negative()==negative);
        }
        for(bool negative:{false,true})for(auto e:{std::int64_t(-1000),std::int64_t(-1000000000000LL),-sbn::float_exponent_limit}){
            const auto x=sbn::float_view::from_magnitude({&one,1},e,negative);
            allocation_watch_start();const auto r=f::inverse_trig_execute(prepared.handle,out.ref(),x);assert(!allocation_watch_stop());
            assert(sbn::compare(out.view(),x)==std::partial_ordering::equivalent);
            assert(r.error.kind==sbn::error_kind::finite&&r.error.exponent2>=3*e-3&&r.error.exponent2<=e+1-std::int64_t(p));++certificates;
        }
        allocation_watch_start();const auto inf=f::inverse_trig_execute(prepared.handle,out.ref(),sbn::float_view::infinity(true));assert(!allocation_watch_stop());
        if(kind==f::InverseTrigKind::atan)certify(out.view(),inf,neg(divide_word(pi,2)),F,p,rounding::nearest_even);
        else assert(out.view().is_nan()&&sbn::has_flag(inf.flags,sbn::float_flag::invalid));
    }
    Prepared prepared(p,f::InverseTrigKind::atan2);
    for(bool yn:{false,true})for(bool xn:{false,true}){
        const auto yzero=sbn::float_view::from_magnitude({},0,yn),xzero=sbn::float_view::from_magnitude({},0,xn);
        allocation_watch_start();auto r=f::inverse_trig_execute(prepared.handle,out.ref(),yzero,xzero);assert(!allocation_watch_stop());
        if(xn)certify(out.view(),r,yn?neg(pi):pi,F,p,rounding::nearest_even);
        else assert(r.error.exact()&&out.view().zero()&&out.view().negative()==yn);
        allocation_watch_start();r=f::inverse_trig_execute(prepared.handle,out.ref(),sbn::float_view::infinity(yn),sbn::float_view::infinity(xn));assert(!allocation_watch_stop());
        auto expected=divide_word(word(pi,xn?3:1),4);if(yn)expected=neg(expected);certify(out.view(),r,expected,F,p,rounding::nearest_even);
        allocation_watch_start();r=f::inverse_trig_execute(prepared.handle,out.ref(),yn?unit.negated():unit,sbn::float_view::infinity(xn));assert(!allocation_watch_stop());
        if(xn)certify(out.view(),r,yn?neg(pi):pi,F,p,rounding::nearest_even);
        else assert(r.error.exact()&&out.view().zero()&&out.view().negative()==yn);
    }
    const auto tiny=sbn::float_view::from_magnitude({&one,1},-sbn::float_exponent_limit);
    const auto huge=sbn::float_view::from_magnitude({&one,1},sbn::float_exponent_limit-1);
    allocation_watch_start();auto r=f::inverse_trig_execute(prepared.handle,out.ref(),tiny,huge);assert(!allocation_watch_stop());
    assert(out.view().zero()&&sbn::has_flag(r.flags,sbn::float_flag::underflow)&&r.error.kind==sbn::error_kind::finite&&r.error.exponent2<-sbn::float_exponent_limit);
    allocation_watch_start();r=f::inverse_trig_execute(prepared.handle,out.ref(),tiny,huge.negated());assert(!allocation_watch_stop());
    assert(!sbn::has_flag(r.flags,sbn::float_flag::underflow));certify(out.view(),r,{sub(pi.lo,Int(1)),add(pi.hi,Int(1))},F,p,rounding::nearest_even);
    for(auto kind:{f::InverseTrigKind::asin,f::InverseTrigKind::acos}){Prepared bad(p,kind);
        allocation_watch_start();r=f::inverse_trig_execute(bad.handle,out.ref(),unit);assert(!allocation_watch_stop());
        if(kind==f::InverseTrigKind::acos)assert(out.view().zero()&&r.error.exact());
        else certify(out.view(),r,divide_word(pi,2),F,p,rounding::nearest_even);
        const auto input=exported(add(shift(Int(1),512),Int(1)),-512);
        allocation_watch_start();r=f::inverse_trig_execute(bad.handle,out.ref(),input.view());assert(!allocation_watch_stop());
        assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
        allocation_watch_start();r=f::inverse_trig_execute(bad.handle,out.ref(),sbn::float_view::nan());assert(!allocation_watch_stop());assert(out.view().is_nan());
    }
    assert(!f::inverse_trig_program_requirements(sbn::bits{0},1,f::InverseTrigKind::atan).supported());
    assert(!f::inverse_trig_program_requirements(sbn::bits{65},0,f::InverseTrigKind::acos).supported());
}
void scopes_and_scale(){
    Team team;constexpr std::size_t p=129,F=p+112;const auto pi=machin_pi(F);
    for(unsigned workers:{4u,2u}){Prepared prepared(p,f::InverseTrigKind::atan2,workers);
        const Input y{{3},1000000000000LL,true},x{{4},1000000000000LL,true};const auto expected=reference(y,x,prepared.kind,F,pi);
        run(prepared,y,x,expected,F,rounding::nearest_even,2,false,team.value,workers==4?1:2);}
}
}
int main(){
    for(std::size_t p:{1u,3u,53u,65u,127u,257u,513u,1025u,4097u})sweep(p);
    endpoints();tiny_and_specials();scopes_and_scale();
    std::printf("inverse trig: %zu independent Taylor/rational/restoring-root endpoint certificates; relative tiny bounds, C99 quadrants, aliases, prepared reuse, scopes and allocation gates PASS\n",certificates);
}
