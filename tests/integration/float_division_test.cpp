// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "sbn3/arena.h"
#include "float/division.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
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
constexpr std::size_t precisions[]{1,2,3,7,17,31,32,53,63,64,65,95,127,128,129,
    173,191,255,256,257,383,511,513,769,1023,1025,1537,2049,3073,4093,4097,6145,8201,16385,32771};
constexpr std::uint64_t guard=UINT64_C(0xd6851a93f7240cbe);
constexpr std::byte scratch_guard{0x59};
std::uint64_t state=UINT64_C(0xc93f182d46a057be);
std::size_t checked=0,prepared_checked=0;

struct Ref {
    ref_int value;
    Ref(){ref_init(value);}~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;
};
struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    bool negative=false;
    sbn::float_view view()const noexcept{return sbn::float_view::from_magnitude(words,exponent,negative);}
};
std::uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
std::size_t word_bits(std::span<const std::uint64_t> a){
    auto n=a.size();while(n&&!a[n-1])--n;if(!n)return 0;
    std::size_t hi=0;for(auto w=a[n-1];w;w>>=1)++hi;return 64*(n-1)+hi;
}
Input random_input(std::size_t bits,int top=1,bool negative=false){
    assert(bits);Input r;r.words.resize((bits+63)/64);for(auto& w:r.words)w=random_word();
    if(bits%64)r.words.back()&=(UINT64_C(1)<<(bits%64))-1;
    r.words.back()|=UINT64_C(1)<<((bits-1)%64);r.words[0]|=1;
    r.exponent=std::int64_t(top)-std::int64_t(bits);r.negative=negative;return r;
}
Input near_power(std::size_t bit,int delta,int top=1,bool negative=false){
    Ref n;ref_set_ui(n.value,1);ref_mul_2exp(n.value,n.value,bit);
    if(delta<0)ref_sub_ui(n.value,n.value,1);else if(delta>0)ref_add_ui(n.value,n.value,1);
    Input r;r.words.resize((ref_sizeinbase(n.value,2)+63)/64);std::size_t count=0;
    ref_export(r.words.data(),&count,-1,8,0,0,n.value);assert(count==r.words.size());
    r.exponent=std::int64_t(top)-std::int64_t(ref_sizeinbase(n.value,2));r.negative=negative;return r;
}
void import(Ref& n,std::span<const std::uint64_t> words){ref_import(n.value,words.size(),-1,8,0,0,words.data());}
void shift_nonzero(Ref& n,__int128 bits){
    if(!ref_sgn(n.value))return;assert(bits>=0&&bits<4000000);
    if(bits)ref_mul_2exp(n.value,n.value,std::size_t(bits));
}
int compare_scaled(const ref_number* a,__int128 ae,const ref_number* b,__int128 be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=ae+ref_sizeinbase(a,2),bt=be+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref l,r;ref_set(l.value,a);ref_set(r.value,b);const auto base=std::min(ae,be);
    shift_nonzero(l,ae-base);shift_nonzero(r,be-base);return ref_cmp(l.value,r.value);
}
// Independent exact arithmetic for every size. Sparse denominators cost only
// one scalar multiply/shift per nonzero word, including the million-bit test.
void multiply_denominator(Ref& result,const ref_number* endpoint,const Input& denominator){
    Ref accumulator,term;ref_set_ui(accumulator.value,0);
    for(std::size_t j=0;j<denominator.words.size();++j)if(denominator.words[j]){
        ref_mul_ui(term.value,endpoint,denominator.words[j]);
        if(j)ref_mul_2exp(term.value,term.value,64*j);
        ref_add(accumulator.value,accumulator.value,term.value);
    }
    ref_set(result.value,accumulator.value);
}

enum class Range {normal,underflow,overflow};
void certify(const Input& numerator,const Input& denominator,sbn::float_view y,
    const sbn::float_report& report,std::size_t precision,rounding mode,bool inverse,Range range=Range::normal){
    const bool negative=(inverse?false:numerator.negative)!=denominator.negative;
    assert(y.negative()==negative);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    if(range==Range::overflow){
        assert(y.classification()==sbn::float_class::infinity&&sbn::has_flag(report.flags,sbn::float_flag::overflow));
        assert(report.error.kind==sbn::error_kind::unbounded);return;
    }
    assert(y.finite()&&word_bits(y.magnitude().limbs())<=precision);
    assert(report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::overflow));
    if(range==Range::underflow)assert(y.zero()&&sbn::has_flag(report.flags,sbn::float_flag::underflow));
    else assert(!y.zero()&&!sbn::has_flag(report.flags,sbn::float_flag::underflow));
    Ref a,center,radius,lo,hi,product;
    if(inverse)ref_set_ui(a.value,1);else import(a,numerator.words);
    const __int128 ae=inverse?0:numerator.exponent;
    import(center,y.magnitude().limbs());
    auto scale=y.zero()&&!report.error.exact()?report.error.exponent2:y.exponent2();
    if(report.error.exact())ref_set_ui(radius.value,0);
    else{
        if(!y.zero()){
            const auto ulp=__int128(y.exponent2())+word_bits(y.magnitude().limbs())-precision;
            assert(report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
        }
        scale=std::min(scale,report.error.exponent2);
        shift_nonzero(center,__int128(y.exponent2())-scale);
        ref_set_ui(radius.value,1);shift_nonzero(radius,__int128(report.error.exponent2)-scale);
    }
    ref_sub(lo.value,center.value,radius.value);if(ref_sgn(lo.value)<0)ref_set_ui(lo.value,0);
    ref_add(hi.value,center.value,radius.value);
    multiply_denominator(product,lo.value,denominator);
    const bool lower=compare_scaled(product.value,__int128(scale)+denominator.exponent,a.value,ae)<=0;
    multiply_denominator(product,hi.value,denominator);
    const bool upper=compare_scaled(product.value,__int128(scale)+denominator.exponent,a.value,ae)>=0;
    if(!lower||!upper){
        std::fprintf(stderr,"float division endpoint failed p=%zu inverse=%d mode=%u A_bits=%zu D_bits=%zu Aexp=%lld Dexp=%lld bound=%lld\n",
            precision,int(inverse),unsigned(mode),word_bits(numerator.words),word_bits(denominator.words),
            (long long)numerator.exponent,(long long)denominator.exponent,(long long)report.error.exponent2);
        assert(false);
    }
}

class Scratch {
    std::vector<std::byte> bytes_;std::byte* base_=nullptr;std::size_t size_=0;
public:
    Scratch(std::size_t size,std::size_t alignment):bytes_(size+alignment+128,scratch_guard),size_(size){
        assert(alignment&&!(alignment&(alignment-1)));
        const auto p=(reinterpret_cast<std::uintptr_t>(bytes_.data())+64+alignment-1)&~std::uintptr_t(alignment-1);
        base_=reinterpret_cast<std::byte*>(p);
    }
    std::span<std::byte> span(){return {base_,size_};}
    void guards()const {
        for(std::size_t j=1;j<=64;++j)assert(base_[-std::ptrdiff_t(j)]==scratch_guard);
        for(std::size_t j=0;j<64;++j)assert(base_[size_+j]==scratch_guard);
    }
};
struct Call {
    sbn::float_ref out;sbn::float_view a,b;sbn::execution ex;rounding mode;bool inverse;
    const floating::FloatDivisionPrepared* prepared=nullptr;
    sbn::float_report report;unsigned seen_first=0,seen_width=0;
    std::atomic<unsigned>* arrivals=nullptr;
};
void execute(Call& c){
    c.report=c.prepared?floating::division_program_execute(*c.prepared,c.out,c.a,c.b,c.ex,c.mode):
        c.inverse?sbn::reciprocal_into(c.out,c.b,c.ex,c.mode):sbn::divide_into(c.out,c.a,c.b,c.ex,c.mode);
}
void on_scope(void* raw,sbn3_team_scope* scope){
    auto& c=*static_cast<Call*>(raw);c.seen_first=sbn3_team_first_worker(scope);c.seen_width=sbn3_team_width(scope);
    if(c.arrivals){c.arrivals->fetch_add(1,std::memory_order_release);while(c.arrivals->load(std::memory_order_acquire)<2){}}
    c.ex.team=nullptr;c.ex.scope=scope;execute(c);
}
void noop(void*,sbn3_team_scope*){}
void right_scope(void* raw,sbn3_team_scope* scope){
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,noop,nullptr,on_scope,raw);
}

// scope_mode: 0 public root dispatch, 1 active W4 root, 2 active first=2 child.
void run_case(const Input& a,const Input& d,std::size_t p,rounding mode,bool inverse,
    unsigned alias=0,sbn3_team* team=nullptr,unsigned scope_mode=0,
    const floating::FloatDivisionPrepared* prepared=nullptr,Range range=Range::normal){
    const auto out_words=(p+63)/64;const Input* aliased=alias==1?&a:alias==2?&d:nullptr;
    const auto capacity=aliased?std::max(out_words,aliased->words.size()):out_words;
    std::vector<std::uint64_t> words(capacity+2,guard);
    if(aliased)std::copy(aliased->words.begin(),aliased->words.end(),words.begin());
    const auto saved_a=a.words,saved_d=d.words;
    sbn::float_buffer out({words.data(),capacity},sbn::bits{p});
    auto av=a.view(),dv=d.view();
    if(alias==1)av=sbn::float_view::from_magnitude({words.data(),a.words.size()},a.exponent,a.negative);
    if(alias==2)dv=sbn::float_view::from_magnitude({words.data(),d.words.size()},d.exponent,d.negative);
    const unsigned workers=scope_mode==2?2:team?sbn3_team_workers(team):1;
    allocation_watch_start();
    const auto need=inverse?sbn::reciprocal_requirements(out.ref(),dv,workers):sbn::divide_requirements(out.ref(),av,dv,workers);
    assert(!allocation_watch_stop());assert(need.supported()&&need.output_limbs==out_words);
    assert(need.scratch_bytes<(std::size_t(512)<<20));
    Scratch scratch(prepared?0:need.scratch_bytes,need.scratch_alignment);
    Call c{out.ref(),av,dv,{scratch.span(),team,nullptr},mode,inverse,prepared,{}};
    allocation_watch_start();
    if(scope_mode){assert(team);sbn3_team_run(team,scope_mode==2?right_scope:on_scope,&c);}else execute(c);
    assert(!allocation_watch_stop());scratch.guards();
    if(scope_mode==2)assert(c.seen_first==2&&c.seen_width==2);
    if(scope_mode==1)assert(c.seen_first==0&&c.seen_width==sbn3_team_workers(team));
    assert(words[capacity]==guard&&words[capacity+1]==guard);
    if(aliased)for(std::size_t j=out_words;j<aliased->words.size();++j)assert(words[j]==aliased->words[j]);
    assert(a.words==saved_a&&d.words==saved_d);
    certify(a,d,out.view(),c.report,p,mode,inverse,range);++checked;if(prepared)++prepared_checked;
}

class Team {
    sbn3_arena* arena_=nullptr;sbn3_lease control_{};
public:
    sbn3_team* value=nullptr;
    Team(){
        sbn3_arena_config config{std::size_t(1)<<28,32u<<20};sbn3_error error{};
        assert(sbn3_arena_create(&config,&arena_,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena_,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena_,0,sbn3_team_storage_bytes(),&control_);
        sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;for(auto& x:tc.cpu_ids)x=-1;
        assert(sbn3_team_create(arena_,&control_,&tc,&value,&error)==SBN3_OK);
    }
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena_,&control_);sbn3_arena_stats stats{};
        sbn3_arena_get_stats(arena_,&stats);assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena_);}
    Team(const Team&)=delete;Team&operator=(const Team&)=delete;
};

void precision_cases(){
    for(auto p:precisions){
        const auto a=random_input(p+197,int(p%7)-3,p&1),d=random_input(p+331,2,p&2);
        const auto near_low=near_power(p+149,-1,-2,true),near_high=near_power(p+151,1,3,false);
        for(auto mode:modes){
            run_case(a,d,p,mode,false,1);run_case(a,d,p,mode,false,2);
            run_case(a,d,p,mode,true,2);
            run_case(near_low,near_high,p,mode,false);run_case(a,near_low,p,mode,true);
        }
    }
    // Source height varies independently of output precision.
    const auto long_a=random_input(65549,1,true),long_d=random_input(65573,-1,false);
    const Input three{{3},0,false};
    for(std::size_t p:{1u,3u,53u,65u,129u,257u,513u})for(auto mode:modes){
        run_case(long_a,long_d,p,mode,false,1);run_case(three,long_d,p,mode,false,2);
        run_case(three,long_d,p,mode,true,2);
    }
    for(std::size_t p:{65u,257u,1025u}){
        std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});
        const auto short_a=random_input(p+233),short_d=random_input(p+271);
        const auto x=sbn::divide_requirements(out.ref(),short_a.view(),short_d.view());
        const auto y=sbn::divide_requirements(out.ref(),long_a.view(),long_d.view());
        const auto u=sbn::reciprocal_requirements(out.ref(),short_d.view()),v=sbn::reciprocal_requirements(out.ref(),long_d.view());
        assert(x.supported()&&y.supported()&&u.supported()&&v.supported());
        assert(x.scratch_bytes==y.scratch_bytes&&x.scratch_alignment==y.scratch_alignment);
        assert(u.scratch_bytes==v.scratch_bytes&&u.scratch_alignment==v.scratch_alignment);
    }
}

void prepared_reuse(std::size_t p,bool inverse,unsigned workers=1,sbn3_team* team=nullptr,unsigned scope_mode=0){
    const auto kind=inverse?SBN3_NEWTON_INVERSE:SBN3_NEWTON_DIVIDE;
    allocation_watch_start();const auto need=floating::division_program_requirements(sbn::bits{p},workers,kind);
    assert(!allocation_watch_stop());assert(need.supported());
    Scratch scratch(need.scratch_bytes,need.scratch_alignment);
    allocation_watch_start();const auto prepared=floating::division_program_prepare(sbn::bits{p},workers,kind,scratch.span());
    assert(!allocation_watch_stop());assert(prepared.state==prepared.storage.data());
    for(unsigned repeat=0;repeat<3;++repeat){
        const auto a=random_input(repeat==1?7:p+193+repeat*67,-2,repeat&1);
        const auto d=random_input(repeat==1?9:p+277+repeat*83,2,repeat&2);
        for(auto mode:modes){
            run_case(a,d,p,mode,inverse,inverse?2:repeat%3,team,scope_mode,&prepared);
            scratch.guards();
        }
    }
    // A prepared inverse ignores its numerator even when it is NaN.
    if(inverse){
        const Input d{{3},-1,true};std::vector<std::uint64_t> words((p+63)/64);
        sbn::float_buffer out(words,sbn::bits{p});
        allocation_watch_start();const auto r=floating::division_program_execute(prepared,out.ref(),sbn::float_view::nan(),d.view(),{{},team,nullptr});
        assert(!allocation_watch_stop());certify({},d,out.view(),r,p,rounding::nearest_even,true);++checked;++prepared_checked;
    }
}

void power_two_and_ranges(){
    const Input three{{3},0,false};
    for(std::size_t p:{1u,3u,65u,129u,257u,1025u})for(unsigned bit:{0u,1u,17u,63u})
        for(bool negative:{false,true})for(auto mode:modes){
            const Input d{{UINT64_C(1)<<bit},-77,negative};const auto a=random_input(p+23,2,!negative);
            std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});
            const auto dn=sbn::divide_requirements(out.ref(),a.view(),d.view());
            const auto rn=sbn::reciprocal_requirements(out.ref(),d.view());
            assert(dn.supported()&&rn.supported()&&!dn.scratch_bytes&&!rn.scratch_bytes);
            allocation_watch_start();const auto exact=sbn::reciprocal_into(out.ref(),d.view(),{},mode);
            assert(!allocation_watch_stop());assert(exact.error.exact()&&exact.flags==sbn::float_flag::none);
            run_case(a,d,p,mode,false,1);run_case(three,d,p,mode,true,2);
        }
    for(auto mode:modes)for(bool negative:{false,true}){
        const Input a{{3},sbn::float_exponent_limit-2,negative},d{{3},sbn::float_exponent_limit-2,false};
        run_case(a,d,129,mode,false,1);
        const Input low{{3},-sbn::float_exponent_limit-2,negative};
        run_case(low,d,129,mode,false,2,nullptr,0,nullptr,Range::underflow);
        run_case(a,low,129,mode,false,1,nullptr,0,nullptr,Range::overflow);
        run_case(three,low,129,mode,true,2,nullptr,0,nullptr,Range::overflow);
        run_case(three,d,129,mode,true);
        const Input x{{7},1000000000000LL,negative},y{{5},-1000000000000LL,false};
        run_case(x,y,129,mode,false);run_case(three,x,129,mode,true);
    }
    // Same mathematical power of two, multiword representation: generic path.
    const Input multiword{{0,UINT64_C(1)<<17},-81,true};
    for(auto mode:modes){run_case(three,multiword,513,mode,false,2);run_case(three,multiword,513,mode,true,2);}
}

void specials_and_queries(){
    alignas(128)std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    const Input one{{1},0,false},three{{3},0,false};
    for(auto mode:modes)for(bool an:{false,true})for(bool dn:{false,true}){
        auto a=three,d=three;a.negative=an;d.negative=dn;
        const auto az=sbn::float_view::from_magnitude({},0,an),dz=sbn::float_view::from_magnitude({},0,dn);
        allocation_watch_start();auto r=sbn::divide_into(out.ref(),az,d.view(),{},mode);
        assert(!allocation_watch_stop());assert(out.view().zero()&&out.view().negative()==(an!=dn)&&r.error.exact());
        r=sbn::divide_into(out.ref(),a.view(),dz,{},mode);
        assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==(an!=dn));
        assert(sbn::has_flag(r.flags,sbn::float_flag::divide_by_zero));
        r=sbn::divide_into(out.ref(),az,dz,{},mode);assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
        r=sbn::divide_into(out.ref(),sbn::float_view::infinity(an),sbn::float_view::infinity(dn),{},mode);
        assert(out.view().is_nan()&&sbn::has_flag(r.flags,sbn::float_flag::invalid));
        r=sbn::divide_into(out.ref(),a.view(),sbn::float_view::infinity(dn),{},mode);
        assert(out.view().zero()&&out.view().negative()==(an!=dn)&&r.error.exact());
        r=sbn::divide_into(out.ref(),sbn::float_view::infinity(an),d.view(),{},mode);
        assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==(an!=dn));
        r=sbn::reciprocal_into(out.ref(),dz,{},mode);
        assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==dn&&sbn::has_flag(r.flags,sbn::float_flag::divide_by_zero));
        r=sbn::reciprocal_into(out.ref(),sbn::float_view::infinity(dn),{},mode);
        assert(out.view().zero()&&out.view().negative()==dn&&r.error.exact());
    }
    for(auto mode:modes){
        auto r=sbn::divide_into(out.ref(),sbn::float_view::nan(),three.view(),{},mode);assert(out.view().is_nan());
        r=sbn::divide_into(out.ref(),one.view(),sbn::float_view::nan(),{},mode);assert(out.view().is_nan());
        r=sbn::reciprocal_into(out.ref(),sbn::float_view::nan(),{},mode);assert(out.view().is_nan());
    }
    std::array<std::uint64_t,1> tiny{guard};sbn::float_buffer short_out(tiny,sbn::bits{257});
    assert(sbn::divide_requirements(short_out.ref(),one.view(),three.view()).status==sbn::support::capacity);
    assert(sbn::reciprocal_requirements(short_out.ref(),three.view()).status==sbn::support::capacity);
    assert(tiny[0]==guard);
    for(unsigned workers:{0u,33u}){
        assert(!sbn::divide_requirements(out.ref(),one.view(),three.view(),workers).supported());
        assert(!sbn::reciprocal_requirements(out.ref(),three.view(),workers).supported());
    }
    assert(!floating::division_program_requirements(sbn::bits{65},1,SBN3_NEWTON_RSQRT).supported());
}

void same_base_alias(){
    for(std::size_t p:{65u,257u,4097u})for(const auto height:{p+573,std::size_t(173)})for(auto mode:modes){
        const auto input=random_input(height,1,true);
        std::vector<std::uint64_t> words=input.words;
        words.resize(std::max(words.size(),(p+63)/64)+2,guard);
        const auto capacity=words.size()-2;
        sbn::float_buffer out({words.data(),capacity},sbn::bits{p});
        const auto value=sbn::float_view::from_magnitude({words.data(),input.words.size()},input.exponent,input.negative);
        const auto need=sbn::divide_requirements(out.ref(),value,value);assert(need.supported());
        Scratch scratch(need.scratch_bytes,need.scratch_alignment);
        allocation_watch_start();const auto report=sbn::divide_into(out.ref(),value,value,{scratch.span()},mode);
        assert(!allocation_watch_stop());scratch.guards();
        assert(words[capacity]==guard&&words[capacity+1]==guard);
        certify(input,input,out.view(),report,p,mode,false);++checked;
    }
}

void short_divisors(){
    const Input a{{UINT64_C(0xa791ca02b64917d3),UINT64_C(0x2174)},-47,true};
    const Input divisors[]{
        {{1,1},-64,false},{{UINT64_MAX,1},-65,true},
        {{UINT64_MAX,UINT64_MAX},-127,false},{{0,0,5,3},-193,true},
        {{0,0,1},-128,false}};
    for(std::size_t p:{1u,65u,127u,129u,191u,255u,257u,513u,4097u,65537u}){
        std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});
        for(const auto& d:divisors){
            const auto need=sbn::divide_requirements(out.ref(),a.view(),d.view());
            assert(need.supported());
            if(p<=256)assert(!need.scratch_bytes);
            else assert(need.scratch_bytes<=(((p+63)/64+2)*8+63));
            for(auto mode:modes){
                run_case(a,d,p,mode,false,2);
                run_case(a,d,p,mode,true,2);
            }
        }
    }
    // A source much taller than the demanded prefix must not be scanned just
    // to prove an optional power-of-two fast path.
    Input tall;tall.words.resize(33);tall.words.back()=1;tall.exponent=-2048;
    std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    assert(sbn::divide_requirements(out.ref(),a.view(),tall.view()).scratch_bytes);
    run_case(a,tall,65,rounding::nearest_even,false,2);
    for(std::size_t p:{513u,2049u,8193u,32771u}){
        const auto numerator=random_input(p+193,-2,true);
        for(std::size_t k:{3u,7u,17u,65u,129u})if(k<((p+63)/64+1)/2){
            const auto denominator=random_input(64*k-7,3,k&1);
            for(auto mode:modes){
                run_case(numerator,denominator,p,mode,false,2);
                run_case(numerator,denominator,p,mode,true,2);
            }
        }
    }
}

struct Pair {Call* left;Call* right;};
void pair_scope(void* raw,sbn3_team_scope* scope){auto& p=*static_cast<Pair*>(raw);
    sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,on_scope,p.left,on_scope,p.right);}
void concurrent_prepared(Team& team){
    constexpr std::size_t p0=4097,p1=8201;
    const auto a0=random_input(5113,-2,true),d0=random_input(5297,3),a1=random_input(37),d1=random_input(9187,-1,true);
    std::vector<std::uint64_t> w0((p0+63)/64+2,guard),w1((p1+63)/64+2,guard);
    sbn::float_buffer out0({w0.data(),(p0+63)/64},sbn::bits{p0}),out1({w1.data(),(p1+63)/64},sbn::bits{p1});
    const auto q0=floating::division_program_requirements(sbn::bits{p0},2,SBN3_NEWTON_DIVIDE);
    const auto q1=floating::division_program_requirements(sbn::bits{p1},2,SBN3_NEWTON_INVERSE);
    assert(q0.supported()&&q1.supported());Scratch s0(q0.scratch_bytes,q0.scratch_alignment),s1(q1.scratch_bytes,q1.scratch_alignment);
    allocation_watch_start();
    const auto b0=floating::division_program_prepare(sbn::bits{p0},2,SBN3_NEWTON_DIVIDE,s0.span());
    const auto b1=floating::division_program_prepare(sbn::bits{p1},2,SBN3_NEWTON_INVERSE,s1.span());
    assert(!allocation_watch_stop());
    for(auto mode:{rounding::nearest_even,rounding::downward}){
        std::atomic<unsigned> arrivals{0};
        Call c0{out0.ref(),a0.view(),d0.view(),{},mode,false,&b0,{},0,0,&arrivals};
        Call c1{out1.ref(),a1.view(),d1.view(),{},mode,true,&b1,{},0,0,&arrivals};Pair pair{&c0,&c1};
        allocation_watch_start();sbn3_team_run(team.value,pair_scope,&pair);assert(!allocation_watch_stop());
        assert(arrivals.load()==2&&c0.seen_first==0&&c1.seen_first==2&&c0.seen_width==2&&c1.seen_width==2);
        s0.guards();s1.guards();assert(w0[(p0+63)/64]==guard&&w1[(p1+63)/64]==guard);
        certify(a0,d0,out0.view(),c0.report,p0,mode,false);certify(a1,d1,out1.view(),c1.report,p1,mode,true);
        checked+=2;prepared_checked+=2;
    }
}
void teams_and_native(){
    Team team;
    for(std::size_t p:{257u,1025u,4097u,8201u}){
        const auto a=random_input(p+417,-2,true),d=random_input(p+619,2,false);
        for(auto mode:modes)for(bool inverse:{false,true}){
            run_case(a,d,p,mode,inverse,inverse?2:1,team.value,1);
            run_case(a,d,p,mode,inverse,2,team.value,2);
        }
    }
    prepared_reuse(4097,false,4,team.value,1);prepared_reuse(8201,true,2,team.value,2);
    concurrent_prepared(team);
    // A genuinely wide denominator keeps native normalized coverage after
    // effective-width and two-word shortcuts. Two nonzero words keep the
    // independent endpoint certificate linear in the full output width.
    constexpr std::size_t p=(std::size_t(1)<<20)+1;
    const Input a{{3},0,false};Input d;
    d.words.resize((p+63)/64);d.words.front()=d.words.back()=1;
    d.exponent=-std::int64_t(64*(d.words.size()-1));
    std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});
    const auto need=sbn::divide_requirements(out.ref(),a.view(),d.view(),4);
    assert(need.supported()&&need.scratch_bytes&&need.scratch_bytes<(std::size_t(512)<<20));
    run_case(a,d,p,rounding::nearest_even,false,0,team.value,1);
}
}

int main(){
    specials_and_queries();precision_cases();power_two_and_ranges();same_base_alias();short_divisors();
    for(std::size_t p:{1u,65u,257u,513u,1025u,4097u,8201u})for(bool inverse:{false,true})prepared_reuse(p,inverse);
    teams_and_native();
    std::printf("float divide/reciprocal: %zu independent endpoint certificates (%zu prepared); modes, aliases, huge exact inputs, ranges, W4 child/concurrent scopes and native million-bit case PASS\n",checked,prepared_checked);
}
