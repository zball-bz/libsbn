// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "sbn3/arena.h"
#include "float/fma.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

namespace {
using sbn::rounding;
namespace floating=sbn::v3::floating;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,
    rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0xa392671ef08dc45b);
constexpr std::byte scratch_guard{0x73};
std::uint64_t random_state=UINT64_C(0xd1387942ca0fb6e5);
std::size_t checked=0;

struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    bool negative=false;
    sbn::float_view view()const noexcept {
        return sbn::float_view::from_magnitude(words,exponent,negative);
    }
};
std::uint64_t random_word(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
std::size_t word_bits(std::span<const std::uint64_t> a){
    auto n=a.size();while(n&&!a[n-1])--n;if(!n)return 0;
    std::size_t hi=0;for(auto x=a[n-1];x;x>>=1)++hi;return 64*(n-1)+hi;
}
Input random_input(std::size_t bits,int top=1,bool negative=false){
    assert(bits);Input r;r.words.resize((bits+63)/64);
    for(auto& w:r.words)w=random_word();
    if(bits%64)r.words.back()&=(UINT64_C(1)<<(bits%64))-1;
    r.words.back()|=UINT64_C(1)<<((bits-1)%64);r.words.front()|=1;
    r.exponent=std::int64_t(top)-std::int64_t(bits);r.negative=negative;return r;
}
Input exported(const ref_number* n,std::int64_t exponent,bool negative=false){
    assert(ref_sgn(n)>0);Input r;r.words.resize((ref_sizeinbase(n,2)+63)/64);
    std::size_t count=0;ref_export(r.words.data(),&count,-1,8,0,0,n);
    assert(count==r.words.size());r.exponent=exponent;r.negative=negative;return r;
}
Input near_power(std::size_t bit,int delta,std::int64_t exponent=0,bool negative=false){
    Ref n;ref_set_ui(n.value,1);ref_mul_2exp(n.value,n.value,bit);
    if(delta<0)ref_sub_ui(n.value,n.value,1);else if(delta>0)ref_add_ui(n.value,n.value,1);
    return exported(n.value,exponent,negative);
}
void import(Ref& n,std::span<const std::uint64_t> words){ref_import(n.value,words.size(),-1,8,0,0,words.data());}
void shift_nonzero(Ref& n,__int128 bits){
    if(!ref_sgn(n.value))return;assert(bits>=0&&bits<1000000);
    if(bits)ref_mul_2exp(n.value,n.value,std::size_t(bits));
}
// tests/oracle's general ref_mul has a production witness path for large
// inputs. The scalar reference path below is independent at every tested size.
void exact_product(Ref& product,const Input& a,const Input& b){
    Ref av,term;import(av,a.words);ref_set_ui(product.value,0);
    for(std::size_t i=0;i<b.words.size();++i)if(b.words[i]){
        ref_mul_ui(term.value,av.value,b.words[i]);
        if(i)ref_mul_2exp(term.value,term.value,64*i);
        ref_add(product.value,product.value,term.value);
    }
    if(a.negative!=b.negative)ref_neg(product.value,product.value);
}
std::int64_t exact_sum(Ref& sum,const Input& a,const Input& b,const Input& c){
    Ref product,addend;exact_product(product,a,b);import(addend,c.words);
    if(c.negative)ref_neg(addend.value,addend.value);
    const auto pe=a.exponent+b.exponent;
    const auto scale=!ref_sgn(product.value)?c.exponent:
        !ref_sgn(addend.value)?pe:std::min(pe,c.exponent);
    shift_nonzero(product,__int128(pe)-scale);shift_nonzero(addend,__int128(c.exponent)-scale);
    ref_add(sum.value,product.value,addend.value);return scale;
}
int compare_scaled(const ref_number* a,std::int64_t ae,const ref_number* b,std::int64_t be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=__int128(ae)+ref_sizeinbase(a,2),bt=__int128(be)+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref l,r;ref_set(l.value,a);ref_set(r.value,b);const auto scale=std::min(ae,be);
    shift_nonzero(l,__int128(ae)-scale);shift_nonzero(r,__int128(be)-scale);return ref_cmp(l.value,r.value);
}
bool zero_sign(const Input& a,const Input& b,const Input& c,rounding mode){
    const bool az=!word_bits(a.words),bz=!word_bits(b.words),cz=!word_bits(c.words);
    const bool ps=a.negative!=b.negative;
    return (az||bz)&&cz&&ps==c.negative?ps:mode==rounding::downward;
}

// Exactly one independent rounding of the complete signed a*b+c. No
// production multiply, add, round, or comparison participates in this oracle.
void certify(const Input& a,const Input& b,const Input& c,sbn::float_view y,
             const sbn::float_report& report,std::size_t precision,rounding mode){
    Ref sum,magnitude,q,remainder,half,center,radius,difference;
    const auto scale=exact_sum(sum,a,b,c);const bool negative=ref_sgn(sum.value)<0;
    if(!ref_sgn(sum.value)){
        assert(y.zero()&&y.negative()==zero_sign(a,b,c,mode));
        assert(report.error.exact()&&report.flags==sbn::float_flag::none);return;
    }
    ref_abs(magnitude.value,sum.value);
    const auto length=ref_sizeinbase(magnitude.value,2),cut=length>precision?length-precision:0;
    ref_fdiv_q_2exp(q.value,magnitude.value,cut);ref_fdiv_r_2exp(remainder.value,magnitude.value,cut);
    bool up=false;
    if(ref_sgn(remainder.value))switch(mode){
        case rounding::nearest_even:
            ref_set_ui(half.value,1);ref_mul_2exp(half.value,half.value,cut-1);
            up=ref_cmp(remainder.value,half.value)>0||
                (ref_cmp(remainder.value,half.value)==0&&ref_fdiv_ui(q.value,2));break;
        case rounding::toward_zero:break;
        case rounding::upward:up=!negative;break;
        case rounding::downward:up=negative;break;
        case rounding::away_zero:up=true;break;
    }
    if(up)ref_add_ui(q.value,q.value,1);
    const auto qe=scale+std::int64_t(cut);
    const auto top=__int128(qe)+ref_sizeinbase(q.value,2);
    assert(y.negative()==negative);
    if(top>sbn::float_exponent_limit){
        assert(y.classification()==sbn::float_class::infinity);
        assert(sbn::has_flag(report.flags,sbn::float_flag::overflow));
        assert(report.error.kind==sbn::error_kind::unbounded);return;
    }
    assert(y.finite()&&report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    if(top<-sbn::float_exponent_limit){
        assert(y.zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
        assert(report.error.kind==sbn::error_kind::finite);
        ref_set_ui(radius.value,1);
        assert(compare_scaled(magnitude.value,scale,radius.value,report.error.exponent2)<=0);return;
    }
    assert(!y.zero()&&word_bits(y.magnitude().limbs())<=precision);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::overflow));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::underflow));
    import(center,y.magnitude().limbs());
    if(compare_scaled(q.value,qe,center.value,y.exponent2())){
        std::fprintf(stderr,"FMA single-round mismatch p=%zu mode=%u a_bits=%zu b_bits=%zu c_bits=%zu\n",
            precision,unsigned(mode),word_bits(a.words),word_bits(b.words),word_bits(c.words));assert(false);
    }
    if(!ref_sgn(remainder.value)){
        assert(report.error.exact()&&!sbn::has_flag(report.flags,sbn::float_flag::rounded));return;
    }
    assert(report.error.kind==sbn::error_kind::finite&&sbn::has_flag(report.flags,sbn::float_flag::rounded));
    auto base=std::min({scale,y.exponent2(),report.error.exponent2});
    shift_nonzero(magnitude,__int128(scale)-base);shift_nonzero(center,__int128(y.exponent2())-base);
    ref_sub(difference.value,center.value,magnitude.value);ref_abs(difference.value,difference.value);
    ref_set_ui(radius.value,1);shift_nonzero(radius,__int128(report.error.exponent2)-base);
    assert(ref_cmp(difference.value,radius.value)<=0);
    const auto ulp=y.exponent2()+std::int64_t(word_bits(y.magnitude().limbs()))-std::int64_t(precision);
    assert(report.error.exponent2<=ulp-(mode==rounding::nearest_even?1:0));
}

class Scratch {
    std::vector<std::byte> bytes_;std::size_t n_;
public:
    explicit Scratch(std::size_t n):bytes_(n+36,scratch_guard),n_(n){}
    std::span<std::byte> span(){return {bytes_.data()+17,n_};}
    void guards()const {
        assert(std::all_of(bytes_.begin(),bytes_.begin()+17,[](auto x){return x==scratch_guard;}));
        assert(std::all_of(bytes_.begin()+17+n_,bytes_.end(),[](auto x){return x==scratch_guard;}));
    }
};
struct Call {
    sbn::float_ref out;sbn::float_view a,b,c;sbn::execution ex;rounding mode;
    const floating::FloatFmaPlan* plan=nullptr;sbn::float_report report;
};
void execute(Call& c){
    c.report=c.plan?floating::float_fma_execute(*c.plan,c.out,c.a,c.b,c.c,c.ex,c.mode):
        sbn::fma_into(c.out,c.a,c.b,c.c,c.ex,c.mode);
}
void on_scope(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}

void run_case(const Input& a,const Input& b,const Input& c,std::size_t precision,
              rounding mode,unsigned alias=0,sbn3_team* team=nullptr,bool scope=false,
              const floating::FloatFmaPlan* plan=nullptr){
    const Input* inputs[]{&a,&b,&c};const auto* aliased=alias?inputs[alias-1]:nullptr;
    const auto out_words=(precision+63)/64;
    const auto capacity=aliased?std::max(out_words,aliased->words.size()):out_words;
    std::vector<std::uint64_t> storage(capacity+2,guard);
    if(aliased)std::copy(aliased->words.begin(),aliased->words.end(),storage.begin());
    const auto saved_a=a.words,saved_b=b.words,saved_c=c.words;
    sbn::float_buffer output({storage.data(),capacity},sbn::bits{precision});
    sbn::float_view v[]{a.view(),b.view(),c.view()};
    if(alias)v[alias-1]=sbn::float_view::from_magnitude(
        {storage.data(),aliased->words.size()},aliased->exponent,aliased->negative);
    const unsigned workers=team?sbn3_team_workers(team):1;
    allocation_watch_start();const auto need=sbn::fma_requirements(output.ref(),v[0],v[1],v[2],workers);
    assert(!allocation_watch_stop());assert(need.supported()&&need.output_limbs==out_words);
    if(precision<=256&&a.words.size()<=6&&b.words.size()<=6)assert(!need.scratch_bytes);
    Scratch scratch(plan?plan->scratch_bytes:need.scratch_bytes);
    Call call{output.ref(),v[0],v[1],v[2],{scratch.span(),team,nullptr},mode,plan,{}};
    allocation_watch_start();
    if(scope){assert(team);sbn3_team_run(team,on_scope,&call);}else execute(call);
    assert(!allocation_watch_stop());scratch.guards();
    assert(storage[capacity]==guard&&storage[capacity+1]==guard);
    assert(a.words==saved_a&&b.words==saved_b&&c.words==saved_c);
    if(aliased)for(std::size_t i=out_words;i<aliased->words.size();++i)assert(storage[i]==aliased->words[i]);
    certify(a,b,c,output.view(),call.report,precision,mode);++checked;
}

class Team {
    sbn3_arena* arena_=nullptr;sbn3_lease control_{};
public:
    sbn3_team* value=nullptr;
    explicit Team(unsigned workers){
        sbn3_arena_config config{std::size_t(1)<<28,32u<<20};sbn3_error error{};
        assert(sbn3_arena_create(&config,&arena_,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena_,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena_,0,sbn3_team_storage_bytes(),&control_);
        sbn3_team_config tc{};tc.workers=workers;tc.stack_offset=65536;for(auto& x:tc.cpu_ids)x=-1;
        assert(sbn3_team_create(arena_,&control_,&tc,&value,&error)==SBN3_OK);
    }
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena_,&control_);sbn3_arena_stats s{};
        sbn3_arena_get_stats(arena_,&s);assert(!s.lease_references&&!s.active_computations);sbn3_arena_destroy(arena_);}
    Team(const Team&)=delete;Team&operator=(const Team&)=delete;
};

void random_cases(){
    for(std::size_t p=1;p<=257;++p){
        const auto a=random_input(p+53,-1,p&1),b=random_input(p+117,3,p&2),c=random_input(p+397,1,p&4);
        for(auto mode:modes)run_case(a,b,c,p,mode,p%4);
    }
    for(std::size_t p:{383u,513u,769u,1025u,2049u,4097u,8193u}){
        const auto a=random_input(p+197,-2,true),b=random_input(p+461,4,false),c=random_input(2*p+71,2,true);
        for(auto mode:modes)for(unsigned alias:{0u,1u,2u,3u})run_case(a,b,c,p,mode,alias);
        const Input small_a{{3},-2,false},small_b{{5},-3,true};
        for(auto mode:modes)run_case(small_a,small_b,c,p,mode,3);
    }
}
void cancellation(){
    for(std::size_t n:{5u,63u,64u,65u,191u,383u,385u,1025u,4097u,8193u}){
        const auto a=near_power(n,1),b=near_power(n,-1),c=near_power(2*n,0,0,true);
        for(std::size_t p:{1u,3u,65u,257u,513u})for(auto mode:modes)
            for(unsigned alias:{0u,1u,2u,3u})run_case(a,b,c,p,mode,alias);
        // Exact cancellation to signed zero, with an independent full product.
        Ref product;exact_product(product,a,b);const auto cancel=exported(product.value,0,true);
        for(auto mode:modes)run_case(a,b,cancel,65,mode,3);
    }
    // Bounded prefixes would lose these distant correction bits entirely.
    const auto a=near_power(32771,1),b=near_power(32771,-1),c=near_power(65542,0,0,true);
    for(auto mode:modes)run_case(a,b,c,3,mode,3);
}
void range_cases(){
    const Input one{{1},0,false},two{{1},1,false},half{{1},-1,false};
    for(auto mode:modes)for(bool negative:{false,true}){
        // Product alone overflows; cancellation leaves a normal finite value.
        const Input a{{3},sbn::float_exponent_limit-2,negative};
        auto c=a;c.negative=!negative;
        run_case(a,two,c,65,mode,3);
        // Product alone underflows; addition recovers a normal finite value.
        const Input tiny{{1},-sbn::float_exponent_limit-1,negative};
        run_case(tiny,half,tiny,65,mode,1);
        const Input zero{{},0,false};
        run_case(a,two,zero,65,mode);run_case(tiny,half,zero,65,mode);
        auto opposite=tiny;opposite.negative=!negative;
        run_case(tiny,half,opposite,65,mode,3);
        const Input x{{3},1000000000000LL,negative},y{{5},-1000000000000LL,false},z{{7},-2,!negative};
        run_case(x,y,z,65,mode,2);
        run_case(tiny,one,zero,65,mode);
    }
}

// A remote bit above/below a midpoint is decisive even though its exponent
// gap cannot be materialized. A finite proxy of the same sign, much smaller
// than a target ulp, gives the same unique rounded result for all five modes.
void virtual_gaps(){
    const Input one{{1},0,false};constexpr std::int64_t gap=1000000000000LL;
    for(std::size_t p:{1u,3u,53u,64u,65u,129u,257u})for(bool negative:{false,true})
        for(bool subtract:{false,true})for(unsigned parity:{0u,1u})for(auto mode:modes){
            Ref n;ref_set_ui(n.value,1);ref_mul_2exp(n.value,n.value,p);
            ref_add_ui(n.value,n.value,parity&&p>1?3:1);
            const auto a=exported(n.value,-std::int64_t(p),negative);
            const Input remote{{1},-gap,negative!=subtract};
            const Input proxy{{1},-std::int64_t(p)-20,negative!=subtract};
            std::vector<std::uint64_t> storage((p+63)/64+2,guard);
            sbn::float_buffer out({storage.data(),(p+63)/64},sbn::bits{p});
            const auto need=sbn::fma_requirements(out.ref(),a.view(),one.view(),remote.view());
            assert(need.supported());Scratch scratch(need.scratch_bytes);
            allocation_watch_start();const auto report=sbn::fma_into(out.ref(),a.view(),one.view(),remote.view(),{scratch.span()},mode);
            assert(!allocation_watch_stop());scratch.guards();
            // Use the proxy ONLY to check the rounded value: its exact error
            // can differ from the remote-bit error by a tiny amount.
            Ref sum,expected,q,remainder,halfulp,center;const auto scale=exact_sum(sum,a,one,proxy);
            ref_abs(expected.value,sum.value);const auto cut=ref_sizeinbase(expected.value,2)-p;
            ref_fdiv_q_2exp(q.value,expected.value,cut);ref_fdiv_r_2exp(remainder.value,expected.value,cut);
            bool up=false;
            if(mode==rounding::nearest_even){ref_set_ui(halfulp.value,1);ref_mul_2exp(halfulp.value,halfulp.value,cut-1);
                up=ref_cmp(remainder.value,halfulp.value)>0||
                    (ref_cmp(remainder.value,halfulp.value)==0&&ref_fdiv_ui(q.value,2));}
            else up=ref_sgn(remainder.value)&&(mode==rounding::away_zero||
                (mode==rounding::upward&&!negative)||(mode==rounding::downward&&negative));
            if(up)ref_add_ui(q.value,q.value,1);import(center,out.view().magnitude().limbs());
            assert(compare_scaled(q.value,scale+std::int64_t(cut),center.value,out.view().exponent2())==0);
            assert(out.view().negative()==negative&&report.error.kind==sbn::error_kind::finite);
            assert(sbn::has_flag(report.flags,sbn::float_flag::rounded));
            const auto ulp=out.view().exponent2()+std::int64_t(word_bits(out.view().magnitude().limbs()))-std::int64_t(p);
            assert(report.error.exponent2<=ulp-(mode==rounding::nearest_even?1:0));
            assert(storage[(p+63)/64]==guard&&storage[(p+63)/64+1]==guard);++checked;
        }
}

void special_cases(){
    alignas(64)std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    const Input one{{1},0,false};
    for(auto mode:modes){
        for(bool an:{false,true})for(bool bn:{false,true})for(bool cn:{false,true}){
            const Input a{{},0,an},b{{3},0,bn},c{{},0,cn};run_case(a,b,c,65,mode);
            const auto inf=sbn::float_view::infinity(an);
            allocation_watch_start();auto r=sbn::fma_into(out.ref(),inf,b.view(),sbn::float_view::infinity(cn),{},mode);
            assert(!allocation_watch_stop());
            if((an!=bn)!=cn)assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
            else assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==cn);
            r=sbn::fma_into(out.ref(),a.view(),inf,one.view(),{},mode);
            assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
            r=sbn::fma_into(out.ref(),a.view(),b.view(),sbn::float_view::infinity(cn),{},mode);
            assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==cn);
        }
        for(unsigned position:{0u,1u,2u}){
            sbn::float_view v[]{one.view(),one.view(),one.view()};v[position]=sbn::float_view::nan();
            const auto r=sbn::fma_into(out.ref(),v[0],v[1],v[2],{},mode);(void)r;assert(out.view().is_nan());
        }
    }
    std::array<std::uint64_t,1> small{guard};sbn::float_buffer short_out(small,sbn::bits{257});
    const auto a=random_input(811),b=random_input(719),c=random_input(1027);
    allocation_watch_start();const auto need=sbn::fma_requirements(short_out.ref(),a.view(),b.view(),c.view());
    assert(!allocation_watch_stop());assert(need.status==sbn::support::capacity&&need.output_limbs==5&&small[0]==guard);
    for(unsigned workers:{0u,33u})assert(sbn::fma_requirements(out.ref(),one.view(),one.view(),one.view(),workers).status==sbn::support::unsupported);
}

void compiled_and_teams(){
    for(unsigned workers:{1u,4u}){
        Team team(workers);
        for(std::size_t p:{65u,257u,1025u,4097u,8193u}){
            const auto a=random_input(p+337,-2,true),b=random_input(p+521,3,false),c=random_input(p+619,1,true);
            floating::FloatFmaPlan plan{};
            allocation_watch_start();const auto rc=floating::float_fma_query(p,a.words.size(),b.words.size(),true,workers,plan);
            assert(!allocation_watch_stop());assert(rc==SBN3_SUPPORTED);
            for(auto mode:modes){
                run_case(a,b,c,p,mode,1,team.value,false,&plan);
                auto aa=a,bb=b,cc=c;aa.words[0]^=0x71;bb.words[0]^=0x91;cc.negative=!cc.negative;
                run_case(aa,bb,cc,p,mode,3,team.value,true,&plan);
                run_case(a,b,c,p,mode,2,team.value);
            }
            const Input zero{{},0,false};
            run_case(a,b,zero,p,rounding::nearest_even,0,team.value,false,&plan);
        }
    }
}
}

int main(){
    special_cases();random_cases();cancellation();range_cases();virtual_gaps();compiled_and_teams();
    std::printf("float FMA: %zu independent exact-product/sum single-round certificates; cancellation, aliases, exponent gaps/ranges, compiled W1/W4 and allocation gates PASS\n",checked);
}
