// Original libsbn tests; LGPL-3.0-or-later.
#include "float/complex.hpp"
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
constexpr std::uint64_t guard=UINT64_C(0xab617e56273bc409);
std::uint64_t random_state=UINT64_C(0x184ec639276a804b);
std::size_t certificates=0;
std::uint64_t random_word(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
struct Exact {
    ref_int n;std::int64_t e=0;
    Exact(){ref_init(n);ref_set_ui(n,0);}
    Exact(const Exact& x):e(x.e){ref_init(n);ref_set(n,x.n);}
    Exact& operator=(const Exact& x){if(this!=&x){ref_set(n,x.n);e=x.e;}return *this;}
    ~Exact(){ref_clear(n);}
};
Exact integer(std::uint64_t n,std::int64_t e=0){Exact x;ref_set_ui(x.n,n);x.e=e;return x;}
void shift(Exact& x,std::int64_t n){assert(n>=0&&n<1000000);if(n&&ref_sgn(x.n))ref_mul_2exp(x.n,x.n,std::size_t(n));}
Exact neg(Exact x){ref_neg(x.n,x.n);return x;}
Exact add(const Exact& a,const Exact& b){Exact x=a,y=b;x.e=std::min(a.e,b.e);shift(x,a.e-x.e);shift(y,b.e-x.e);ref_add(x.n,x.n,y.n);return x;}
Exact sub(const Exact& a,const Exact& b){return add(a,neg(b));}
// Deliberately no ref_mul: its large-input path can call a production witness.
Exact mul(const Exact& a,const Exact& b){
    Exact x,aa=a,bb=b,term;const bool negative=(ref_sgn(a.n)<0)!=(ref_sgn(b.n)<0);
    ref_abs(aa.n,aa.n);ref_abs(bb.n,bb.n);x.e=a.e+b.e;
    if(!ref_sgn(aa.n)||!ref_sgn(bb.n))return x;
    std::vector<std::uint64_t> words((ref_sizeinbase(bb.n,2)+63)/64);std::size_t count=0;
    ref_export(words.data(),&count,-1,8,0,0,bb.n);assert(count==words.size());
    for(std::size_t j=0;j<words.size();++j)if(words[j]){
        ref_mul_ui(term.n,aa.n,words[j]);if(j)ref_mul_2exp(term.n,term.n,64*j);ref_add(x.n,x.n,term.n);}
    if(negative)ref_neg(x.n,x.n);return x;
}
bool le(const Exact& a,const Exact& b){Exact d=sub(a,b);return ref_sgn(d.n)<=0;}
struct Input {
    std::vector<std::uint64_t> words;std::int64_t e=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,e,negative);}
};
Input exported(const Exact& x){Input a;a.e=x.e;a.negative=ref_sgn(x.n)<0;
    if(!ref_sgn(x.n))return a;Exact t=x;ref_abs(t.n,t.n);a.words.resize((ref_sizeinbase(t.n,2)+63)/64);
    std::size_t count=0;ref_export(a.words.data(),&count,-1,8,0,0,t.n);assert(count==a.words.size());return a;}
Exact exact(sbn::float_view x,std::int64_t scale=0){Exact a;a.e=x.exponent2()+scale;
    ref_import(a.n,x.magnitude().size(),-1,8,0,0,x.magnitude().data());if(x.negative())ref_neg(a.n,a.n);return a;}
Input random_input(std::size_t bits,int top,bool negative){Input x;x.words.resize((bits+63)/64);
    for(auto& w:x.words)w=random_word();if(bits%64)x.words.back()&=(UINT64_C(1)<<(bits%64))-1;
    x.words.back()|=UINT64_C(1)<<((bits-1)%64);x.words.front()|=1;x.e=top-std::int64_t(bits);x.negative=negative;return x;}
struct Point {Input re,im;std::int64_t scale=0;f::ScaledComplexView view()const{return {re.view(),im.view(),scale};}};
struct Pair {Exact re,im;};
Pair exact(f::ScaledComplexView z){return {exact(z.real,z.scale),exact(z.imag,z.scale)};}
Pair product(const Pair& a,const Pair& b){return {sub(mul(a.re,b.re),mul(a.im,b.im)),add(mul(a.re,b.im),mul(a.im,b.re))};}
Exact norm2(const Pair& x){return add(mul(x.re,x.re),mul(x.im,x.im));}
Point point(const Pair& x){return {exported(x.re),exported(x.im),0};}
void certify(f::ComplexKind kind,const Point& a,const Point& b,const Pair* root,
             f::ScaledComplexView y,sbn::float_report report){
    assert(y.real.finite()&&y.imag.finite());assert(report.error.kind!=sbn::error_kind::unbounded);
    const auto av=exact(a.view()),bv=exact(b.view()),yv=exact(y);
    Exact radius=report.error.exact()?integer(0):integer(1,report.error.exponent2);Pair delta;Exact right;
    if(kind==f::ComplexKind::reciprocal){
        const auto d=norm2(av);delta={sub(mul(yv.re,d),av.re),add(mul(yv.im,d),av.im)};
        right=mul(mul(radius,radius),mul(d,d));
    }else{
        Pair wanted;
        if(kind==f::ComplexKind::multiply)wanted=product(av,bv);
        else if(kind==f::ComplexKind::mean){wanted={add(av.re,bv.re),add(av.im,bv.im)};--wanted.re.e;--wanted.im.e;}
        else {assert(root);wanted=*root;assert(!y.real.negative());}
        delta={sub(yv.re,wanted.re),sub(yv.im,wanted.im)};right=mul(radius,radius);
    }
    if(!le(norm2(delta),right)){
        std::fprintf(stderr,"complex certificate failed kind=%u radius=%lld\n",unsigned(kind),(long long)report.error.exponent2);assert(false);}
    ++certificates;
}
class Scratch {
    std::vector<std::byte> bytes;std::size_t offset=0,n=0;
public:
    explicit Scratch(const sbn::requirements& q):bytes(q.scratch_bytes+q.scratch_alignment+128,std::byte{0x69}),n(q.scratch_bytes){
        const auto base=reinterpret_cast<std::uintptr_t>(bytes.data());offset=((base+64+q.scratch_alignment-1)&~(q.scratch_alignment-1))-base;}
    std::span<std::byte> span(){return {bytes.data()+offset,n};}
    void check()const{for(std::size_t j=0;j<offset;++j)assert(bytes[j]==std::byte{0x69});
        for(std::size_t j=offset+n;j<bytes.size();++j)assert(bytes[j]==std::byte{0x69});}
};
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config cfg{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&cfg,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
struct Call {const f::FloatComplexPrepared* prepared;f::ComplexKind kind;f::ScaledComplexRef out;
    f::ScaledComplexView a,b;sbn::execution ex;sbn::float_report report;};
void execute(Call& c){c.report=f::complex_program_execute(*c.prepared,c.kind,c.out,c.a,c.b,c.ex);}
void scope_call(void* p,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(p);c.ex={};c.ex.scope=scope;execute(c);}
void run(const f::FloatComplexPrepared& prepared,std::size_t p,f::ComplexKind kind,const Point& a,const Point& b,
         const Pair* root=nullptr,bool alias=false,sbn3_team* team=nullptr,bool scope=false){
    const auto s=f::complex_scalar_precision(sbn::bits{p}),n=(s+63)/64;
    const auto nr=alias?std::max(n,a.re.words.size()):n,ni=alias?std::max(n,a.im.words.size()):n;
    std::vector<std::uint64_t> real(nr+2,guard),imag(ni+2,guard);
    if(alias){std::copy(a.re.words.begin(),a.re.words.end(),real.begin());std::copy(a.im.words.begin(),a.im.words.end(),imag.begin());}
    sbn::float_buffer re({real.data(),nr},sbn::bits{s}),im({imag.data(),ni},sbn::bits{s});std::int64_t scale=a.scale;
    auto av=a.view();if(alias)av={sbn::float_view::from_magnitude({real.data(),a.re.words.size()},a.re.e,a.re.negative),
        sbn::float_view::from_magnitude({imag.data(),a.im.words.size()},a.im.e,a.im.negative),a.scale};
    Call c{&prepared,kind,{re.ref(),im.ref(),&scale},av,b.view(),{{},team,nullptr},{}};
    allocation_watch_start();if(scope)sbn3_team_run(team,scope_call,&c);else execute(c);assert(!allocation_watch_stop());
    assert(real[nr]==guard&&real[nr+1]==guard&&imag[ni]==guard&&imag[ni+1]==guard);
    certify(kind,a,b,root,c.out.view(),c.report);
}
void sweep(std::size_t p,sbn3_team* team=nullptr){
    const unsigned workers=team?4:1;
    allocation_watch_start();const auto q=f::complex_program_requirements(sbn::bits{p},workers);assert(!allocation_watch_stop());
    assert(q.supported()&&q.scratch_bytes<(std::size_t(1)<<29));Scratch scratch(q);
    allocation_watch_start();const auto prepared=f::complex_program_prepare(sbn::bits{p},workers,scratch.span());assert(!allocation_watch_stop());
    for(unsigned repeat=0;repeat<2;++repeat){
        Point a{random_input(p+239,0, false),random_input(p+181,-1,repeat!=0),-7};
        Point b{random_input(p+97,0,false),random_input(p+311,-1,repeat!=0),11};
        run(prepared,p,f::ComplexKind::mean,a,b,nullptr,repeat,team,team&&repeat);
        run(prepared,p,f::ComplexKind::multiply,a,b,nullptr,repeat,team,team&&repeat);
        run(prepared,p,f::ComplexKind::reciprocal,a,{},nullptr,repeat,team,team&&repeat);
        const auto root=exact(a.view());const auto squared=point(product(root,root));
        run(prepared,p,f::ComplexKind::square_root,squared,{},&root,repeat,team,team&&repeat);
    }
    scratch.check();
}
void zero_and_extreme(){
    constexpr std::size_t p=65;const auto s=f::complex_scalar_precision(sbn::bits{p});
    const auto q=f::complex_program_requirements(sbn::bits{p});Scratch storage(q);
    const auto prepared=f::complex_program_prepare(sbn::bits{p},1,storage.span());
    Point zero{},one{Input{{1},0,false},{},0};const Pair root=exact(zero.view());
    run(prepared,p,f::ComplexKind::mean,zero,zero);run(prepared,p,f::ComplexKind::multiply,zero,one);
    run(prepared,p,f::ComplexKind::square_root,zero,{},&root);run(prepared,p,f::ComplexKind::mean,zero,one);
    // Huge equal/opposite external scales never enter the scalar kernels.
    auto large=one,small=one;large.scale=1000000000000LL;small.scale=-large.scale;
    run(prepared,p,f::ComplexKind::multiply,large,small);
    const auto external=large.scale;large.scale=0; // reciprocal certificate at ordinary scale, then compare metadata scaling.
    std::vector<std::uint64_t> re((s+63)/64),im((s+63)/64);sbn::float_buffer rr(re,sbn::bits{s}),ii(im,sbn::bits{s});
    std::int64_t scale=0;auto a=one;a.im={{1},-sbn::float_exponent_limit,false};
    allocation_watch_start();const auto report=f::complex_program_execute(prepared,f::ComplexKind::multiply,
        {rr.ref(),ii.ref(),&scale},a.view(),one.view());assert(!allocation_watch_stop());
    assert(ii.view().zero());assert(sbn::compare(rr.view(),sbn::float_view::from_magnitude(one.re.words,-scale))==std::partial_ordering::equivalent);
    assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2>-sbn::float_exponent_limit+8);
    a=one;a.scale=external;
    allocation_watch_start();f::complex_program_execute(prepared,f::ComplexKind::reciprocal,{rr.ref(),ii.ref(),&scale},a.view());
    assert(!allocation_watch_stop());assert(ii.view().zero()&&scale==-external+1);
    assert(sbn::compare(rr.view(),sbn::float_view::from_magnitude(one.re.words,-1))==std::partial_ordering::equivalent);
    storage.check();
    assert(!f::complex_program_requirements(sbn::bits{0}).supported());
    assert(!f::complex_program_requirements(sbn::bits{65},0).supported());
    assert(!f::complex_program_requirements(sbn::bits{sbn::float_precision_limit}).supported());
}
}
int main(){
    for(std::size_t p:{1u,2u,3u,31u,53u,63u,64u,65u,127u,129u,257u,513u,1025u,2049u,4097u,8193u})sweep(p);
    zero_and_extreme();Team team;sweep(1025,team.value);
    std::printf("complex prepared primitives: %zu independent exact-dyadic norm certificates, aliases, ranges, W4 scopes and allocation gates PASS\n",certificates);
}
