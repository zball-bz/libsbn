// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "sbn3/arena.h"
#include "float/agm.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();

namespace {
namespace floating=sbn::v3::floating;
using sbn::rounding;
constexpr std::uint64_t guard=UINT64_C(0xa64d3b19c572ef08);
constexpr std::array modes{rounding::nearest_even,rounding::toward_zero,
    rounding::upward,rounding::downward,rounding::away_zero};
std::size_t certified=0,prepared_certified=0,coarse_checked=0;

std::size_t magnitude_width(sbn::float_view value){
    const auto words=value.magnitude().limbs();
    std::size_t count=words.size();
    while(count&&!words[count-1])--count;
    if(!count)return 0;
    unsigned high=0;
    for(auto word=words[count-1];word;word>>=1)++high;
    return 64*(count-1)+high;
}
struct Ref {
    ref_int n;
    Ref(){ref_init(n);}
    explicit Ref(std::uint64_t value):Ref(){ref_set_ui(n,value);}
    Ref(const Ref& other):Ref(){ref_set(n,other.n);}
    Ref& operator=(const Ref& other){if(this!=&other)ref_set(n,other.n);return *this;}
    ~Ref(){ref_clear(n);}
};
std::vector<std::uint64_t> export_words(const Ref& x){
    std::vector<std::uint64_t> words(std::max<std::size_t>(1,(ref_sizeinbase(x.n,2)+63)/64));
    std::size_t count=0;ref_export(words.data(),&count,-1,8,0,0,x.n);
    words.resize(count);return words;
}
// Deliberately avoid ref_mul/ref_addmul: their large-product oracle can call
// a production multiplication plus residue witnesses. ref_mul_ui always uses
// the oracle's independent exact scalar schoolbook routine.
void scalar_product(Ref& out,const Ref& a,const Ref& b){
    const auto words=export_words(b);
    Ref accumulator,term;
    for(std::size_t i=words.size();i-->0;){
        ref_mul_2exp(accumulator.n,accumulator.n,64);
        ref_mul_ui(term.n,a.n,words[i]);
        ref_add(accumulator.n,accumulator.n,term.n);
    }
    ref_set(out.n,accumulator.n);
}
// Base-4 restoring square root; no ref_sqrt and no production multiplication.
// After consuming a digit pair, trial=(2*q+1) for q already doubled.
void restoring_root(Ref& out,const Ref& input,bool upward){
    assert(ref_sgn(input.n)>=0);
    const auto words=export_words(input);
    const auto width=ref_sgn(input.n)?ref_sizeinbase(input.n,2):0;
    Ref root,remainder,trial;
    for(std::size_t pair=(width+1)/2;pair-->0;){
        const auto bit=2*pair;
        const auto digit=(words[bit/64]>>(bit%64))&3;
        ref_mul_2exp(remainder.n,remainder.n,2);
        ref_add_ui(remainder.n,remainder.n,digit);
        ref_mul_2exp(root.n,root.n,1);
        ref_mul_2exp(trial.n,root.n,1);
        ref_add_ui(trial.n,trial.n,1);
        if(ref_cmp(remainder.n,trial.n)>=0){
            ref_sub(remainder.n,remainder.n,trial.n);
            ref_add_ui(root.n,root.n,1);
        }
    }
    if(upward&&ref_sgn(remainder.n))ref_add_ui(root.n,root.n,1);
    ref_set(out.n,root.n);
}
void directed_mean(Ref& out,const Ref& a,const Ref& b,bool upward){
    Ref sum;ref_add(sum.n,a.n,b.n);
    const bool odd=ref_get_ui(sum.n)&1;
    ref_fdiv_q_2exp(out.n,sum.n,1);
    if(upward&&odd)ref_add_ui(out.n,out.n,1);
}
void oracle_self_checks(){
    for(std::uint64_t n=0;n<=4096;n+=7){
        Ref input(n),floor,ceil;
        restoring_root(floor,input,false);restoring_root(ceil,input,true);
        std::uint64_t root=0;
        while((root+1)*(root+1)<=n)++root;
        assert(ref_cmp_ui(floor.n,root)==0);
        assert(ref_cmp_ui(ceil.n,root+unsigned(root*root!=n))==0);
    }
    // A closed-form 4097-bit witness validates both independent routines.
    constexpr std::size_t k=4097;
    Ref a(1),product,expected(1),power(1),root;
    ref_mul_2exp(a.n,a.n,k);ref_sub_ui(a.n,a.n,1);
    scalar_product(product,a,a);
    ref_mul_2exp(expected.n,expected.n,2*k);
    ref_mul_2exp(power.n,power.n,k+1);
    ref_sub(expected.n,expected.n,power.n);ref_add_ui(expected.n,expected.n,1);
    assert(ref_cmp(product.n,expected.n)==0);
    restoring_root(root,product,false);assert(ref_cmp(root.n,a.n)==0);
    ref_sub_ui(product.n,product.n,1);
    restoring_root(root,product,false);
    ref_sub_ui(expected.n,a.n,1);assert(ref_cmp(root.n,expected.n)==0);
}

struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    sbn::float_view view()const noexcept{return sbn::float_view::from_magnitude(words,exponent);}
};
Input small(std::uint64_t m,std::int64_t e=0){return {{m},e};}
Input scaled(const Input& x,std::int64_t scale){
    Input result=x;
    const __int128 e=__int128(x.exponent)+scale;
    assert(e>=INT64_MIN&&e<=INT64_MAX);result.exponent=std::int64_t(e);return result;
}
std::uint64_t random_word(){
    static std::uint64_t state=UINT64_C(0x41474d5245463236);
    state^=state<<13;state^=state>>7;state^=state<<17;return state;
}
Input normalized(std::size_t width){
    Input x;x.words.resize((width+63)/64);
    for(auto& word:x.words)word=random_word();
    if(width%64)x.words.back()&=(std::uint64_t(1)<<(width%64))-1;
    x.words.back()|=std::uint64_t(1)<<((width-1)%64);
    x.exponent=1-std::int64_t(width);return x; // [1,2)
}
Input almost_one(std::size_t fractional_bits){
    Input x;x.words.assign(fractional_bits/64+1,0);
    x.words[0]=1;x.words[fractional_bits/64]|=std::uint64_t(1)<<(fractional_bits%64);
    x.exponent=-std::int64_t(fractional_bits);return x;
}
void import(Ref& out,const Input& x){
    ref_import(out.n,x.words.size(),-1,8,0,0,x.words.data());
}
void fixed_input(Ref& low,Ref& high,const Input& x,std::size_t q){
    Ref value,restored;import(value,x);
    const __int128 shift=__int128(x.exponent)+q;
    assert(shift>=-100000&&shift<=100000);
    if(shift>=0){
        ref_mul_2exp(low.n,value.n,std::size_t(shift));ref_set(high.n,low.n);
    }else{
        const auto cut=std::size_t(-shift);
        ref_fdiv_q_2exp(low.n,value.n,cut);
        ref_mul_2exp(restored.n,low.n,cut);
        ref_set(high.n,low.n);
        if(ref_cmp(restored.n,value.n)!=0)ref_add_ui(high.n,high.n,1);
    }
}
struct Oracle {
    Ref lower,upper;
    std::size_t q=0;
};
Oracle agm_oracle(const Input& a,const Input& b,std::size_t precision){
    Oracle result;result.q=precision+256;
    Ref al,ah,bl,bh,nal,nah,nbl,nbh,product,width,limit(1);
    fixed_input(al,ah,a,result.q);fixed_input(bl,bh,b,result.q);
    assert(ref_sgn(al.n)>0&&ref_sgn(bl.n)>0);
    ref_mul_2exp(limit.n,limit.n,64);
    for(unsigned iteration=0;iteration<128;++iteration){
        directed_mean(nal,al,bl,false);directed_mean(nah,ah,bh,true);
        scalar_product(product,al,bl);restoring_root(nbl,product,false);
        scalar_product(product,ah,bh);restoring_root(nbh,product,true);
        al=nal;ah=nah;bl=nbl;bh=nbh;
        // After the first true iteration arithmetic >= geometric, regardless
        // of initial ordering. Outward coordinate bounds enclose those exact
        // iterates, so [bl,ah] encloses their common AGM limit.
        ref_sub(width.n,ah.n,bl.n);
        assert(ref_sgn(width.n)>=0);
        if(ref_cmp(width.n,limit.n)<=0){
            result.lower=bl;result.upper=ah;return result;
        }
    }
    assert(false&&"independent fixed-grid AGM interval failed to contract");
    return result;
}
void certify(sbn::float_view value,const sbn::float_report& report,
             const Oracle& oracle,std::int64_t scale,std::size_t precision,rounding mode){
    assert(value.finite()&&!value.zero()&&!value.negative());
    const auto width=magnitude_width(value);
    assert(width<=precision);
    Ref center,radius(1),lower,upper;
    ref_import(center.n,value.magnitude().size(),-1,8,0,0,value.magnitude().data());
    const __int128 center_shift=__int128(value.exponent2())-scale+oracle.q;
    assert(center_shift>=0&&center_shift<=100000);
    ref_mul_2exp(center.n,center.n,std::size_t(center_shift));
    if(report.error.exact()){
        ref_set_ui(radius.n,0);
    }else{
        assert(report.error.kind==sbn::error_kind::finite);
        const __int128 error_shift=__int128(report.error.exponent2)-scale+oracle.q;
        assert(error_shift>=0&&error_shift<=100000);
        ref_mul_2exp(radius.n,radius.n,std::size_t(error_shift));
        const __int128 allowed=__int128(value.exponent2())+width-precision+
            (mode==rounding::nearest_even?0:1);
        assert(report.error.exponent2<=allowed);
    }
    ref_sub(lower.n,center.n,radius.n);ref_add(upper.n,center.n,radius.n);
    assert(ref_cmp(lower.n,oracle.lower.n)<=0);
    assert(ref_cmp(upper.n,oracle.upper.n)>=0);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::overflow));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::underflow));
}
class Scratch {
    sbn::workspace work_;
    std::size_t bytes_;
public:
    explicit Scratch(const sbn::requirements& need):bytes_(need.scratch_bytes){
        assert(need.supported());
        work_.reserve(bytes_+64,need.scratch_alignment);
        auto storage=work_.on().scratch;
        std::fill(storage.begin()+bytes_,storage.begin()+bytes_+64,std::byte{0x6b});
    }
    std::span<std::byte> span(){return work_.on().scratch.first(bytes_);}
    void guards(){
        const auto storage=work_.on().scratch;
        for(std::size_t i=bytes_;i<bytes_+64;++i)assert(storage[i]==std::byte{0x6b});
    }
};
struct Call {
    sbn::float_ref out;
    sbn::float_view a,b;
    sbn::execution ex;
    rounding mode;
    const floating::FloatAgmPrepared* prepared;
    sbn::float_report report{};
};
void execute(Call& c){
    c.report=c.prepared?floating::agm_program_execute(*c.prepared,c.out,c.a,c.b,c.ex,c.mode):
        sbn::agm_into(c.out,c.a,c.b,c.ex,c.mode);
}
void on_scope(void* opaque,sbn3_team_scope* scope){
    auto& c=*static_cast<Call*>(opaque);c.ex.team=nullptr;c.ex.scope=scope;execute(c);
}
void run_case(const Input& a,const Input& b,std::size_t p,rounding mode,
              const Oracle& oracle,std::int64_t scale=0,unsigned alias=0,
              sbn3_team* team=nullptr,bool scope=false,
              const floating::FloatAgmPrepared* prepared=nullptr){
    const auto output_words=(p+63)/64;
    const Input* aliased=alias==1?&a:alias==2?&b:nullptr;
    const auto capacity=aliased?std::max(output_words,aliased->words.size()):output_words;
    std::vector<std::uint64_t> output(capacity+2,guard);
    if(aliased)std::copy(aliased->words.begin(),aliased->words.end(),output.begin());
    const auto before_a=a.words,before_b=b.words;
    sbn::float_buffer out({output.data(),capacity},sbn::bits{p});
    const auto av=alias==1?sbn::float_view::from_magnitude(
        {output.data(),a.words.size()},a.exponent):a.view();
    const auto bv=alias==2?sbn::float_view::from_magnitude(
        {output.data(),b.words.size()},b.exponent):b.view();
    const auto workers=team?sbn3_team_workers(team):1;
    const auto need=sbn::agm_requirements(out.ref(),av,bv,workers);
    assert(need.supported()&&need.output_limbs==output_words);
    Scratch scratch(prepared?sbn::requirements{}:need);
    Call call{out.ref(),av,bv,{scratch.span(),team,nullptr},mode,prepared,{}};
    allocation_watch_start();
    if(scope){assert(team);sbn3_team_run(team,on_scope,&call);}else execute(call);
    assert(!allocation_watch_stop());
    scratch.guards();assert(output[capacity]==guard&&output[capacity+1]==guard);
    assert(a.words==before_a&&b.words==before_b);
    if(aliased)for(std::size_t i=output_words;i<aliased->words.size();++i)
        assert(output[i]==aliased->words[i]);
    certify(out.view(),call.report,oracle,scale,p,mode);
    ++certified;if(prepared)++prepared_certified;
}
class Team {
    sbn3_arena* arena_=nullptr;sbn3_lease control_{};
public:
    sbn3_team* value=nullptr;
    explicit Team(unsigned workers){
        sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena_,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena_,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena_,0,sbn3_team_storage_bytes(),&control_);
        sbn3_team_config tc{};tc.workers=workers;tc.stack_offset=65536;
        for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena_,&control_,&tc,&value,&error)==SBN3_OK);
    }
    ~Team(){
        sbn3_team_destroy(value);sbn3_arena_release(arena_,&control_);
        sbn3_arena_stats stats{};sbn3_arena_get_stats(arena_,&stats);
        assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena_);
    }
    Team(const Team&)=delete;Team&operator=(const Team&)=delete;
};
void ordinary_cases(){
    const auto a=small(1),b=small(1,-2);
    for(std::size_t p:{1u,2u,3u,7u,31u,53u,63u,64u,65u,127u,129u,
                       191u,257u,383u,513u,769u,1025u,1537u,2053u,3073u,4097u}){
        const auto oracle=agm_oracle(a,b,p);
        for(auto mode:modes)run_case(a,b,p,mode,oracle);
        // Symmetry is certified against the same independent mathematical
        // interval; bit-identical rounded centers are not an extra contract.
        run_case(b,a,p,rounding::nearest_even,oracle);
        if(p==65||p==257||p==4097){
            run_case(a,b,p,rounding::nearest_even,oracle,0,1);
            run_case(a,b,p,rounding::downward,oracle,0,2);
        }
    }
    for(std::size_t p:{65u,257u,1025u,4097u}){
        const auto a_long=normalized(4*p+257),b_long=normalized(3*p+193);
        const auto oracle=agm_oracle(a_long,b_long,p);
        for(auto mode:modes)run_case(a_long,b_long,p,mode,oracle);
        run_case(a_long,b_long,p,rounding::nearest_even,oracle,0,1);
        run_case(a_long,b_long,p,rounding::upward,oracle,0,2);
        const auto near=almost_one(4*p+257);
        const auto near_oracle=agm_oracle(a,near,p);
        run_case(a,near,p,rounding::nearest_even,near_oracle);
        // Quote depends on demanded output precision, not full source height.
        std::vector<std::uint64_t> words((p+63)/64);
        sbn::float_buffer out(words,sbn::bits{p});
        assert(sbn::agm_requirements(out.ref(),a.view(),b.view()).scratch_bytes==
               sbn::agm_requirements(out.ref(),a_long.view(),b_long.view()).scratch_bytes);
    }
}
void scale_and_prepared_cases(){
    const auto a=small(3,-1),b=small(5,-2);
    constexpr std::array scales{
        -sbn::float_exponent_limit-1,-INT64_C(1000000000000),
        INT64_C(1000000000000),sbn::float_exponent_limit-1};
    for(std::size_t p:{65u,257u,1025u}){
        const auto oracle=agm_oracle(a,b,p);
        const auto need=floating::agm_program_requirements(sbn::bits{p},1);
        Scratch storage(need);
        allocation_watch_start();
        const auto prepared=floating::agm_program_prepare(sbn::bits{p},1,storage.span());
        assert(!allocation_watch_stop());
        for(auto scale:scales){
            const auto av=scaled(a,scale),bv=scaled(b,scale);
            for(auto mode:modes)run_case(av,bv,p,mode,oracle,scale,0,nullptr,false,&prepared);
        }
        const auto c=small(1),d=small(1,-2);
        const auto other=agm_oracle(c,d,p);
        run_case(c,d,p,rounding::nearest_even,other,0,0,nullptr,false,&prepared);
        storage.guards();
    }
}
void team_and_scope_cases(){
    Team team(4);
    for(std::size_t p:{65u,257u,1025u}){
        const auto a=small(1),b=small(3,-3);
        const auto oracle=agm_oracle(a,b,p);
        run_case(a,b,p,rounding::nearest_even,oracle,0,0,team.value);
        run_case(a,b,p,rounding::downward,oracle,0,1,team.value,true);
        const auto need=floating::agm_program_requirements(sbn::bits{p},4);
        Scratch storage(need);
        allocation_watch_start();
        const auto prepared=floating::agm_program_prepare(sbn::bits{p},4,storage.span());
        assert(!allocation_watch_stop());
        run_case(b,a,p,rounding::nearest_even,oracle,0,2,team.value,true,&prepared);
        storage.guards();
    }
    // A prepared two-worker program executing on workers 2..3 of a W4 Team.
    constexpr std::size_t p=257;
    const auto a=small(1),b=small(1,-1);
    const auto oracle=agm_oracle(a,b,p);
    const auto need=floating::agm_program_requirements(sbn::bits{p},2);
    Scratch storage(need);
    const auto prepared=floating::agm_program_prepare(sbn::bits{p},2,storage.span());
    std::array<std::uint64_t,5> words{};
    sbn::float_buffer out(words,sbn::bits{p});
    Call call{out.ref(),a.view(),b.view(),{},rounding::nearest_even,&prepared,{}};
    allocation_watch_start();
    sbn3_team_run(team.value,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,
            [](void*,sbn3_team_scope*){},nullptr,
            [](void* payload,sbn3_team_scope* child){
                assert(sbn3_team_first_worker(child)==2);on_scope(payload,child);
            },raw);
    },&call);
    assert(!allocation_watch_stop());
    certify(out.view(),call.report,oracle,0,p,rounding::nearest_even);
    storage.guards();++certified;++prepared_certified;
}
int compare_power(const Ref& positive,std::int64_t exponent,std::int64_t power){
    const auto width=ref_sizeinbase(positive.n,2);
    const __int128 top=__int128(exponent)+width;
    if(top!=__int128(power)+1)return top>__int128(power)+1?1:-1;
    Ref target(1);
    const __int128 shift=__int128(power)-exponent;
    assert(shift>=0&&shift<=100000);
    ref_mul_2exp(target.n,target.n,std::size_t(shift));
    return ref_cmp(positive.n,target.n);
}
void extreme_ratio_coarse_checks(){
    // These are independently justified coarse bounds, NOT p-bit certificates.
    // For H>=32, t=2^-H and L=(H+2)log(2), the complementary K expansion gives
    // L <= K < 2L. With 3<pi<4 and 1/2<log(2)<1,
    //       1/2 < (H+2)*AGM(1,2^-H) < 4.
    // No double-precision value is used as an accuracy oracle.
    for(std::uint64_t h:{UINT64_C(32),UINT64_C(1000),UINT64_C(1000000000000),
                         std::uint64_t(2*sbn::float_exponent_limit)}){
        const auto scale=h>std::uint64_t(sbn::float_exponent_limit)
            ?sbn::float_exponent_limit-1:std::int64_t(0);
        const auto a=small(1,scale),b=small(1,scale-std::int64_t(h));
        for(std::size_t p:{65u,257u}){
            std::vector<std::uint64_t> words((p+63)/64);
            sbn::float_buffer out(words,sbn::bits{p});
            const auto need=sbn::agm_requirements(out.ref(),a.view(),b.view());
            Scratch storage(need);
            allocation_watch_start();
            const auto report=sbn::agm_into(out.ref(),a.view(),b.view(),
                                            {storage.span(),nullptr,nullptr});
            assert(!allocation_watch_stop());
            const auto value=out.view();
            assert(value.finite()&&!value.zero()&&!value.negative());
            assert(report.error.kind==sbn::error_kind::finite);
            const auto width=magnitude_width(value);
            assert(width<=p);
            assert(report.error.exponent2<=__int128(value.exponent2())+width-p);
            Ref normalized;
            ref_import(normalized.n,value.magnitude().size(),-1,8,0,0,value.magnitude().data());
            ref_mul_ui(normalized.n,normalized.n,h+2);
            const __int128 exponent=__int128(value.exponent2())-scale;
            assert(exponent>=INT64_MIN&&exponent<=INT64_MAX);
            assert(compare_power(normalized,std::int64_t(exponent),-1)>0);
            assert(compare_power(normalized,std::int64_t(exponent),2)<0);
            storage.guards();++coarse_checked;
        }
    }
}
void special_cases(){
    std::array<std::uint64_t,3> words{};
    sbn::float_buffer out(words,sbn::bits{129});
    const auto zero=small(0),one=small(1);
    const auto infinity=sbn::float_view::infinity(),nan=sbn::float_view::nan();
    for(auto mode:modes){
        for(auto input:{zero.view(),zero.view().negated(),one.view()}){
            const auto need=sbn::agm_requirements(out.ref(),zero.view(),input);
            assert(need.supported()&&need.scratch_bytes==0);
            allocation_watch_start();
            const auto report=sbn::agm_into(out.ref(),zero.view(),input,{},mode);
            assert(!allocation_watch_stop());
            assert(out.view().zero()&&!out.view().negative()&&report.error.exact());
        }
        for(auto pair:std::array{
            std::array{sbn::float_view::infinity(true),one.view()},
            std::array{one.view().negated(),zero.view()},
            std::array{infinity,zero.view()}}){
            allocation_watch_start();
            const auto report=sbn::agm_into(out.ref(),pair[0],pair[1],{},mode);
            assert(!allocation_watch_stop());
            assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
        }
        allocation_watch_start();
        sbn::agm_into(out.ref(),infinity,one.view(),{},mode);
        assert(!allocation_watch_stop());
        assert(out.view().classification()==sbn::float_class::infinity&&!out.view().negative());
        sbn::agm_into(out.ref(),nan,one.view(),{},mode);assert(out.view().is_nan());
    }
    for(std::size_t p:{1u,3u,65u,129u,257u}){
        const auto input=normalized(2*p+17);
        std::vector<std::uint64_t> expected_words((p+63)/64),actual_words((p+63)/64);
        sbn::float_buffer expected(expected_words,sbn::bits{p}),actual(actual_words,sbn::bits{p});
        for(auto mode:modes){
            const auto er=sbn::round_into(expected.ref(),input.view(),mode);
            const auto need=sbn::agm_requirements(actual.ref(),input.view(),input.view());
            assert(need.supported()&&need.scratch_bytes==0);
            allocation_watch_start();
            const auto ar=sbn::agm_into(actual.ref(),input.view(),input.view(),{},mode);
            assert(!allocation_watch_stop());
            assert(sbn::compare(expected.view(),actual.view())==std::partial_ordering::equivalent);
            assert(er.error.kind==ar.error.kind&&er.error.exponent2==ar.error.exponent2);
        }
    }
}
} // namespace

int main(){
    oracle_self_checks();
    special_cases();
    ordinary_cases();
    scale_and_prepared_cases();
    team_and_scope_cases();
    extreme_ratio_coarse_checks();
    std::printf("float AGM: %zu independent fixed-grid interval certificates (%zu prepared), "
                "%zu separately labelled coarse extreme-ratio checks; specials, aliases, "
                "scales, scopes and allocation gates PASS\n",
                certified,prepared_certified,coarse_checked);
}
