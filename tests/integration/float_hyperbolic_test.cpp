// Original libsbn tests; LGPL-3.0-or-later.
#include "float/hyperbolic.hpp"
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
Interval atanh_series(const Interval& z,std::size_t F){
    assert(ref_sgn(z.lo.n)>=0);const auto square=multiply(z,z,F);Interval power=z,sum=z;
    for(std::size_t k=1;k<F+16;++k){
        power=multiply(power,square,F);const auto term=divide_word(power,2*k+1);
        // z<=1/2: the remaining positive tail is below 2*next_term.
        if(ref_cmp_ui(term.hi.n,1)<=0)return {sum.lo,add(sum.hi,word(term.hi,2))};
        sum=add(sum,term);
    }
    assert(false);return {};
}
Interval ln2(std::size_t F){return word(atanh_series(divide_word(exact_interval(shift(Int(1),F)),3),F),2);}
Interval scaled(const Interval& x,std::int64_t k){
    return k>=0?Interval{shift(x.lo,std::size_t(k)),shift(x.hi,std::size_t(k))}:
        Interval{floor2(x.lo,std::size_t(-k)),ceil2(x.hi,std::size_t(-k))};
}
Interval logarithm(const Interval& x,std::size_t F,const Interval& log2){
    assert(ref_sgn(x.lo.n)>0);const auto k=std::int64_t(ref_sizeinbase(x.lo.n,2))-std::int64_t(F)-1;
    const auto m=scaled(x,-k);const auto one=shift(Int(1),F);
    const Interval z{ratio(sub(m.lo,one),add(m.lo,one),F,false),ratio(sub(m.hi,one),add(m.hi,one),F,true)};
    auto value=word(atanh_series(z,F),2);const auto exponent=k<0?std::uint64_t(0)-std::uint64_t(k):std::uint64_t(k);
    const auto correction=word(log2,exponent);return k<0?sub(value,correction):add(value,correction);
}
Interval exponential(const Interval& x,std::size_t F){
    assert(ref_sgn(x.lo.n)>=0);const auto upper=ceil2(x.hi,F);assert(ref_fits_ulong_p(upper.n));const auto bound=ref_get_ui(upper.n);
    assert(bound<=512);const auto unit=exact_interval(shift(Int(1),F));Interval term=unit,sum=unit;
    for(std::size_t k=1;k<4*(F+bound+16);++k){
        term=divide_word(multiply(term,x,F),k);
        if(k>=2*bound&&ref_cmp_ui(term.hi.n,1)<=0)return {sum.lo,add(sum.hi,word(term.hi,2))};
        sum=add(sum,term);
    }
    assert(false);return {};
}
Interval reference(const Input& input,f::HyperbolicKind kind,std::size_t F,const Interval& log2){
    auto x=input;x.negative=false;const auto a=input_interval(x,F),one=exact_interval(shift(Int(1),F));Interval result;
    if(kind==f::HyperbolicKind::sinh||kind==f::HyperbolicKind::cosh||kind==f::HyperbolicKind::tanh){
        const auto ep=exponential(a,F),em=divide_positive(one,ep,F);
        const auto s=divide_word(sub(ep,em),2),c=divide_word(add(ep,em),2);
        result=kind==f::HyperbolicKind::sinh?s:kind==f::HyperbolicKind::cosh?c:divide_positive(s,c,F);
    }else if(kind==f::HyperbolicKind::asinh){
        result=logarithm(add(a,root(add(one,multiply(a,a,F)),F)),F,log2);
    }else if(kind==f::HyperbolicKind::acosh){
        result=logarithm(add(a,root(sub(multiply(a,a,F),one),F)),F,log2);
    }else result=divide_word(logarithm(divide_positive(add(one,a),sub(one,a),F),F,log2),2);
    return input.negative&&kind!=f::HyperbolicKind::cosh&&kind!=f::HyperbolicKind::acosh?neg(result):result;
}
Input exported(const Int& n,std::int64_t e){Input x;x.e=e;x.negative=ref_sgn(n.n)<0;Int a=n;ref_abs(a.n,a.n);
    if(!ref_sgn(a.n))return x;x.words.resize((ref_sizeinbase(a.n,2)+63)/64);std::size_t count=0;
    ref_export(x.words.data(),&count,-1,8,0,0,a.n);assert(count==x.words.size());return x;}
void certify(sbn::float_view x,sbn::float_report r,const Interval& expected,std::size_t F,std::size_t p,rounding mode,
             f::HyperbolicKind kind,const Input& input,std::int64_t result_scale){
    assert(x.finite()&&r.error.kind!=sbn::error_kind::unbounded);const auto e=r.error.exact()?x.exponent2():r.error.exponent2;
    const auto base=std::min({x.exponent2(),e,-std::int64_t(F)});const auto center=shift(magnitude(x),std::size_t(x.exponent2()-base));
    const auto radius=r.error.exact()?Int(0):shift(Int(1),std::size_t(e-base));
    const auto lo=shift(expected.lo,std::size_t(-std::int64_t(F)-base)),hi=shift(expected.hi,std::size_t(-std::int64_t(F)-base));
    const bool below=ref_cmp(sub(center,radius).n,lo.n)>0,above=ref_cmp(hi.n,add(center,radius).n)>0;
    if(below||above){
        constexpr const char* names[]{"sinh","cosh","tanh","asinh","acosh","atanh"};
        const auto width=sub(expected.hi,expected.lo);
        std::fprintf(stderr,"hyperbolic endpoint failure function=%s p=%zu mode=%u error=%lld oracle_F=%zu oracle_width_bits=%zu lower=%d upper=%d input_sign=%u input_exp=%lld input_limbs=%zu input_low=%llx input_high=%llx result_scale=%lld result_exp=%lld\n",
            names[unsigned(kind)],p,unsigned(mode),(long long)e,F,ref_sgn(width.n)?ref_sizeinbase(width.n,2):0,
            int(below),int(above),unsigned(input.negative),(long long)input.e,input.words.size(),
            (unsigned long long)(input.words.empty()?0:input.words.front()),(unsigned long long)(input.words.empty()?0:input.words.back()),
            (long long)result_scale,(long long)x.exponent2());assert(false);}
    if(!x.zero()){const auto top=x.exponent2()+std::int64_t(ref_sizeinbase(magnitude(x).n,2));
        assert(r.error.exact()||e<=top-std::int64_t(p)+(mode==rounding::nearest_even?0:1));}++certificates;
}
class Scratch {
    std::vector<std::byte> bytes;std::size_t offset=0,n=0;
public:
    explicit Scratch(const sbn::requirements& q):bytes(q.scratch_bytes+q.scratch_alignment+128,std::byte{0x2b}),n(q.scratch_bytes){
        const auto base=reinterpret_cast<std::uintptr_t>(bytes.data());offset=((base+64+q.scratch_alignment-1)&~(q.scratch_alignment-1))-base;}
    std::span<std::byte> span(){return {bytes.data()+offset,n};}
    void check()const{for(std::size_t j=0;j<offset;++j)assert(bytes[j]==std::byte{0x2b});
        for(std::size_t j=offset+n;j<bytes.size();++j)assert(bytes[j]==std::byte{0x2b});}
};
sbn::requirements query(std::size_t p,unsigned workers,f::HyperbolicKind kind){
    allocation_watch_start();const auto q=f::hyperbolic_program_requirements(sbn::bits{p},workers,kind);assert(!allocation_watch_stop());
    assert(q.supported()&&q.scratch_bytes<(std::size_t(1)<<29));return q;
}
struct Prepared {
    std::size_t p;f::HyperbolicKind kind;sbn::requirements need;Scratch storage;f::FloatHyperbolicPrepared handle;
    Prepared(std::size_t precision,f::HyperbolicKind k,unsigned workers=1):p(precision),kind(k),need(query(p,workers,k)),storage(need){
        allocation_watch_start();handle=f::hyperbolic_prepare(sbn::bits{p},workers,kind,storage.span());assert(!allocation_watch_stop());}
};
struct Call {Prepared* p;sbn::float_ref out;sbn::float_view x;sbn::execution ex;rounding mode;bool fresh;sbn::float_report report;};
void execute(Call& c){c.report=c.fresh?f::hyperbolic_into(c.out,c.x,c.ex,c.mode,c.p->kind):f::hyperbolic_execute(c.p->handle,c.out,c.x,c.ex,c.mode);}
void scope_call(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}
void noop(void*,sbn3_team_scope*){}
void child_call(void* raw,sbn3_team_scope* scope){sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,noop,nullptr,scope_call,raw);}
void run(Prepared& p,const Input& x,const Interval& expected,std::size_t F,rounding mode=rounding::nearest_even,
         bool alias=false,bool fresh=false,sbn3_team* team=nullptr,unsigned scope=0,std::int64_t scale=0){
    const auto capacity=std::max((p.p+63)/64,x.words.size());std::vector<std::uint64_t> words(capacity+2,guard);
    if(alias)std::copy(x.words.begin(),x.words.end(),words.begin());sbn::float_buffer out({words.data(),capacity},sbn::bits{p.p});
    auto source=x.view();if(alias)source=sbn::float_view::from_magnitude({words.data(),x.words.size()},x.e,x.negative);
    const auto saved=x.words;Call c{&p,out.ref(),source,{p.storage.span(),team,nullptr},mode,fresh,{}};
    allocation_watch_start();if(scope)sbn3_team_run(team,scope==1?scope_call:child_call,&c);else execute(c);assert(!allocation_watch_stop());
    auto y=out.view();if(scale){assert(y.finite());y=sbn::float_view::from_magnitude(y.magnitude().limbs(),y.exponent2()-scale,y.negative());
        if(!c.report.error.exact())c.report.error.exponent2-=scale;}
    certify(y,c.report,expected,F,p.p,mode,p.kind,x,scale);assert(words[capacity]==guard&&words[capacity+1]==guard&&x.words==saved);p.storage.check();
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
void sweep(std::size_t p){
    const std::size_t F=p+128;const auto log2=ln2(F);
    for(auto kind:{f::HyperbolicKind::sinh,f::HyperbolicKind::cosh,f::HyperbolicKind::tanh,f::HyperbolicKind::asinh,f::HyperbolicKind::acosh,f::HyperbolicKind::atanh}){
        Prepared prepared(p,kind);Input x{{3},-3,false};
        if(kind==f::HyperbolicKind::asinh)x={{3},-2,false};if(kind==f::HyperbolicKind::acosh)x={{5},-2,false};
        // Algebraic certificates asinh(3/4)=acosh(5/4)=log(2).
        const auto expected=kind==f::HyperbolicKind::asinh||kind==f::HyperbolicKind::acosh?log2:reference(x,kind,F,log2);
        run(prepared,x,expected,F,rounding::nearest_even,true,true);
        if(p<=257){if(kind!=f::HyperbolicKind::cosh&&kind!=f::HyperbolicKind::acosh)x.negative=true;
            const auto opposite=x.negative?neg(expected):expected;run(prepared,x,opposite,F,rounding::nearest_even,true);
            if(p==53)for(auto mode:modes)run(prepared,x,opposite,F,mode,true);}
    }
}
void endpoints_and_large(){
    constexpr std::size_t p=65,N=1200,F=p+N+128;const auto log2=ln2(F);
    Prepared acosh(p,f::HyperbolicKind::acosh),atanh(p,f::HyperbolicKind::atanh);
    const auto near_above=exported(add(shift(Int(1),N),Int(1)),-std::int64_t(N));
    run(acosh,near_above,reference(near_above,acosh.kind,F,log2),F,rounding::nearest_even,true);
    const auto near_below=exported(sub(shift(Int(1),N),Int(1)),-std::int64_t(N));
    const auto exact_ratio=exact_interval(shift(sub(shift(Int(1),N+1),Int(1)),F));
    const auto expected=divide_word(logarithm(exact_ratio,F,log2),2);run(atanh,near_below,expected,F,rounding::nearest_even,true);
    const std::size_t smallF=p+128;const auto l2=ln2(smallF);
    for(auto kind:{f::HyperbolicKind::asinh,f::HyperbolicKind::acosh}){Prepared prepared(p,kind);
        for(auto e:{std::int64_t(2000),std::int64_t(1000000000000LL),sbn::float_exponent_limit-1}){
            const Input x{{1},e,false};auto value=word(l2,std::uint64_t(e+1));
            if(kind==f::HyperbolicKind::asinh)value.hi=add(value.hi,Int(1));else value.lo=sub(value.lo,Int(1));
            run(prepared,x,value,smallF,rounding::nearest_even,true);
        }}
    for(auto kind:{f::HyperbolicKind::sinh,f::HyperbolicKind::cosh}){
        Prepared prepared(53,kind);const Input x{{128},0,false};const std::size_t F1=181;const auto l=ln2(F1);
        run(prepared,x,reference(x,kind,F1,l),F1,rounding::nearest_even,true);}
    // A saturated result may carry a much tighter radius than one output
    // ulp. The former F=181 quotient enclosure was too wide to certify the
    // valid 2^-233 report for tanh(128). Use an independent analytic enclosure:
    // 0 < 1-tanh(128) = 2/(exp(256)+1) < 2*2^-256 = 2^-255,
    // since e>2. No production bound or certificate assertion is weakened.
    constexpr std::size_t saturation_F=320;
    const auto unit=shift(Int(1),saturation_F),tail=shift(Int(1),saturation_F-255);
    const Interval saturated{sub(unit,tail),unit};Prepared tanh(53,f::HyperbolicKind::tanh);
    for(bool negative:{false,true})for(auto mode:modes)
        run(tanh,Input{{128},0,negative},negative?neg(saturated):saturated,saturation_F,mode,true);
}
void last_finite_binade(){
    constexpr std::size_t p=65,F=p+128,Fc=p+320;const auto l=ln2(Fc);
    const auto coefficient=2*std::uint64_t(sbn::float_exponent_limit)+1;
    const auto x=exported(word(l.lo,coefficient),-std::int64_t(Fc)-1);
    const auto delta=ceil_word(word(sub(l.hi,l.lo),coefficient),2);
    const auto error=add(ceil2(word(delta,4),Fc-F),Int(1));
    const auto sqrt2=root(exact_interval(shift(Int(2),F)),F);const Interval expected{sub(sqrt2.lo,error),add(sqrt2.hi,error)};
    for(auto kind:{f::HyperbolicKind::sinh,f::HyperbolicKind::cosh}){Prepared prepared(p,kind);
        run(prepared,x,expected,F,rounding::nearest_even,true,false,nullptr,0,sbn::float_exponent_limit-1);}
}
void tiny_specials_and_scopes(){
    constexpr std::size_t p=65;std::vector<std::uint64_t> words(2);sbn::float_buffer out(words,sbn::bits{p});const std::uint64_t one=1;
    const auto unit=sbn::float_view::from_magnitude({&one,1},0);
    for(auto kind:{f::HyperbolicKind::sinh,f::HyperbolicKind::cosh,f::HyperbolicKind::tanh,f::HyperbolicKind::asinh,f::HyperbolicKind::acosh,f::HyperbolicKind::atanh}){
        Prepared prepared(p,kind);
        for(bool negative:{false,true}){
            const auto zero=sbn::float_view::from_magnitude({},0,negative);
            allocation_watch_start();auto r=f::hyperbolic_execute(prepared.handle,out.ref(),zero);assert(!allocation_watch_stop());
            if(kind==f::HyperbolicKind::acosh)assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
            else if(kind==f::HyperbolicKind::cosh)assert(r.error.exact()&&sbn::compare(out.view(),unit)==std::partial_ordering::equivalent);
            else assert(r.error.exact()&&out.view().zero()&&out.view().negative()==negative);
            allocation_watch_start();r=f::hyperbolic_execute(prepared.handle,out.ref(),sbn::float_view::infinity(negative));assert(!allocation_watch_stop());
            if(kind==f::HyperbolicKind::atanh||(kind==f::HyperbolicKind::acosh&&negative))assert(out.view().is_nan());
            else if(kind==f::HyperbolicKind::tanh)assert(r.error.exact()&&sbn::compare(out.view(),negative?unit.negated():unit)==std::partial_ordering::equivalent);
            else assert(out.view().classification()==sbn::float_class::infinity);
            if(kind==f::HyperbolicKind::acosh)continue;
            for(auto e:{std::int64_t(-1000),std::int64_t(-1000000000000LL),-sbn::float_exponent_limit}){
                const auto x=sbn::float_view::from_magnitude({&one,1},e,negative);
                allocation_watch_start();r=f::hyperbolic_execute(prepared.handle,out.ref(),x);assert(!allocation_watch_stop());
                assert(sbn::compare(out.view(),kind==f::HyperbolicKind::cosh?unit:x)==std::partial_ordering::equivalent);
                assert(r.error.kind==sbn::error_kind::finite);
                assert(r.error.exponent2>=(kind==f::HyperbolicKind::cosh?2*e-2:3*e-3));++certificates;
            }
        }
        allocation_watch_start();const auto nan=f::hyperbolic_execute(prepared.handle,out.ref(),sbn::float_view::nan());assert(!allocation_watch_stop());
        assert(out.view().is_nan()&&nan.error.kind==sbn::error_kind::unbounded);
        if(kind==f::HyperbolicKind::atanh){allocation_watch_start();auto r=f::hyperbolic_execute(prepared.handle,out.ref(),unit.negated());assert(!allocation_watch_stop());
            assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()&&sbn::has_flag(r.flags,sbn::float_flag::divide_by_zero));}
        if(kind==f::HyperbolicKind::acosh){allocation_watch_start();const auto r=f::hyperbolic_execute(prepared.handle,out.ref(),unit);assert(!allocation_watch_stop());
            assert(out.view().zero()&&r.error.exact());}
        if(kind==f::HyperbolicKind::acosh||kind==f::HyperbolicKind::atanh){
            const auto endpoint=shift(Int(1),512);
            const auto invalid=exported(kind==f::HyperbolicKind::acosh?sub(endpoint,Int(1)):add(endpoint,Int(1)),-512);
            allocation_watch_start();const auto r=f::hyperbolic_execute(prepared.handle,out.ref(),invalid.view());assert(!allocation_watch_stop());
            assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));}
        if(kind==f::HyperbolicKind::sinh||kind==f::HyperbolicKind::cosh){allocation_watch_start();auto r=f::hyperbolic_execute(prepared.handle,out.ref(),sbn::float_view::from_magnitude({&one,1},60));assert(!allocation_watch_stop());
            assert(out.view().classification()==sbn::float_class::infinity&&sbn::has_flag(r.flags,sbn::float_flag::overflow));}
    }
    Team team;const std::size_t F=p+128;const auto l=ln2(F);const Input x{{3},-3,false};
    for(unsigned workers:{4u,2u}){Prepared prepared(p,f::HyperbolicKind::asinh,workers);
        run(prepared,x,reference(x,prepared.kind,F,l),F,rounding::nearest_even,true,false,team.value,workers==4?1:2);}
    assert(!f::hyperbolic_program_requirements(sbn::bits{0},1,f::HyperbolicKind::sinh).supported());
    assert(!f::hyperbolic_program_requirements(sbn::bits{65},0,f::HyperbolicKind::atanh).supported());
}
}
int main(){
    for(std::size_t p:{1u,53u,65u,257u,1025u,4097u})sweep(p);
    endpoints_and_large();last_finite_binade();tiny_specials_and_scopes();
    std::printf("hyperbolic: %zu independent interval/series/restoring-root certificates; last finite binade, endpoint distances, tiny/huge exponents, modes, aliases, scopes and allocation gates PASS\n",certificates);
}
