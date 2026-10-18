// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "sbn3/arena.h"
#include "float/product.hpp"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
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
constexpr std::uint64_t guard=UINT64_C(0xe4f639a1702bc85d);
constexpr std::byte scratch_guard{0x6d};
std::uint64_t state=UINT64_C(0x83e917a4265cf0bd);
std::size_t checked=0,prepared_checked=0;

struct Ref {
    ref_int value;
    Ref(){ref_init(value);}
    ~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;
    Ref&operator=(const Ref&)=delete;
};
struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    bool negative=false;
    sbn::float_view view()const noexcept {
        return sbn::float_view::from_magnitude(words,exponent,negative);
    }
};
std::uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
std::size_t word_bits(std::span<const std::uint64_t> words){
    auto n=words.size();while(n&&!words[n-1])--n;
    if(!n)return 0;
    std::size_t high=0;for(auto w=words[n-1];w;w>>=1)++high;
    return 64*(n-1)+high;
}
Input random_input(std::size_t bits,int top=1,bool negative=false){
    assert(bits);Input r;r.words.resize((bits+63)/64);
    for(auto& w:r.words)w=random_word();
    if(bits%64)r.words.back()&=(UINT64_C(1)<<(bits%64))-1;
    r.words.back()|=UINT64_C(1)<<((bits-1)%64);
    r.words.front()|=1;
    r.exponent=std::int64_t(top)-std::int64_t(bits);r.negative=negative;return r;
}
Input exported(const ref_number* n,std::int64_t exponent=0,bool negative=false){
    assert(ref_sgn(n)>0);Input r;
    r.words.resize((ref_sizeinbase(n,2)+63)/64);std::size_t count=0;
    ref_export(r.words.data(),&count,-1,8,0,0,n);assert(count==r.words.size());
    r.exponent=exponent;r.negative=negative;return r;
}
Input near_power(std::size_t bit,int delta,int top=1,bool negative=false){
    Ref n;ref_set_ui(n.value,1);ref_mul_2exp(n.value,n.value,bit);
    if(delta<0)ref_sub_ui(n.value,n.value,1);else if(delta>0)ref_add_ui(n.value,n.value,1);
    return exported(n.value,std::int64_t(top)-std::int64_t(ref_sizeinbase(n.value,2)),negative);
}
void import(Ref& n,std::span<const std::uint64_t> words){
    ref_import(n.value,words.size(),-1,8,0,0,words.data());
}
// ref_mul deliberately uses a production witness for large operands. Force
// independent exact scalar products here, including the longest source test.
void exact_product(Ref& result,const Ref& a,std::span<const std::uint64_t> b){
    ref_set_ui(result.value,0);Ref term;
    for(std::size_t i=0;i<b.size();++i)if(b[i]){
        ref_mul_ui(term.value,a.value,b[i]);
        if(i)ref_mul_2exp(term.value,term.value,64*i);
        ref_add(result.value,result.value,term.value);
    }
}
void shift_nonzero(Ref& n,__int128 bits){
    if(!ref_sgn(n.value))return;
    assert(bits>=0&&bits<1000000);
    if(bits)ref_mul_2exp(n.value,n.value,std::size_t(bits));
}
int compare_scaled(const ref_number* a,std::int64_t ae,const ref_number* b,std::int64_t be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const auto at=__int128(ae)+ref_sizeinbase(a,2),bt=__int128(be)+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref left,right;ref_set(left.value,a);ref_set(right.value,b);
    const auto base=std::min(ae,be);
    shift_nonzero(left,__int128(ae)-base);shift_nonzero(right,__int128(be)-base);
    return ref_cmp(left.value,right.value);
}

// Independently round the FULL exact product, only in the untruncated-input
// cases where the public adapter promises that value. This reference never
// reconstructs the production prefixes or calls the production rounder.
void check_exact_round(const ref_number* product,std::int64_t exponent,bool negative,
                       sbn::float_view result,std::size_t precision,rounding mode){
    Ref q,remainder,half,center;
    const auto length=ref_sizeinbase(product,2),cut=length>precision?length-precision:0;
    ref_fdiv_q_2exp(q.value,product,cut);ref_fdiv_r_2exp(remainder.value,product,cut);
    bool increment=false;
    if(ref_sgn(remainder.value)){
        switch(mode){
            case rounding::nearest_even:
                assert(cut);ref_set_ui(half.value,1);ref_mul_2exp(half.value,half.value,cut-1);
                increment=ref_cmp(remainder.value,half.value)>0||
                    (ref_cmp(remainder.value,half.value)==0&&ref_fdiv_ui(q.value,2));break;
            case rounding::toward_zero:break;
            case rounding::upward:increment=!negative;break;
            case rounding::downward:increment=negative;break;
            case rounding::away_zero:increment=true;break;
        }
    }
    if(increment)ref_add_ui(q.value,q.value,1);
    import(center,result.magnitude().limbs());
    assert(compare_scaled(q.value,exponent+std::int64_t(cut),center.value,result.exponent2())==0);
}

// The certificate uses an exact schoolbook product from tests/oracle, all
// original input limbs and exact dyadic subtraction. In particular, the
// truncated-input check cannot pass by merely matching the implementation's
// approximate prefix product.
void certify(const Input& a,const Input& b,sbn::float_view y,const sbn::float_report& report,
             std::size_t precision,rounding mode,bool allow_underflow=false){
    assert(y.finite());assert(word_bits(y.magnitude().limbs())<=precision);
    assert(y.negative()==(a.negative!=b.negative));
    assert(report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::overflow));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    if(!allow_underflow)assert(!y.zero()&&!sbn::has_flag(report.flags,sbn::float_flag::underflow));
    Ref av,exact,center,radius,difference;
    import(av,a.words);exact_product(exact,av,b.words);
    import(center,y.magnitude().limbs());
    const auto exact_exponent=a.exponent+b.exponent;
    if(!y.zero()&&word_bits(a.words)<=precision+128&&word_bits(b.words)<=precision+128)
        check_exact_round(exact.value,exact_exponent,a.negative!=b.negative,y,precision,mode);
    auto scale=exact_exponent;
    if(!y.zero())scale=std::min(scale,y.exponent2());
    if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift_nonzero(exact,__int128(exact_exponent)-scale);
    shift_nonzero(center,__int128(y.exponent2())-scale);
    ref_sub(difference.value,center.value,exact.value);ref_abs(difference.value,difference.value);
    if(report.error.exact())assert(!ref_sgn(difference.value));
    else{
        ref_set_ui(radius.value,1);shift_nonzero(radius,__int128(report.error.exponent2)-scale);
        if(ref_cmp(difference.value,radius.value)>0){
            std::fprintf(stderr,"float product bound failed p=%zu mode=%u a_bits=%zu b_bits=%zu error_exp=%lld\n",
                precision,unsigned(mode),word_bits(a.words),word_bits(b.words),(long long)report.error.exponent2);
            assert(false);
        }
        if(!y.zero()){
            const auto ulp=y.exponent2()+std::int64_t(word_bits(y.magnitude().limbs()))-std::int64_t(precision);
            assert(report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
        }
    }
    if(!y.zero()&&(word_bits(a.words)>precision+128||word_bits(b.words)>precision+128))
        assert(report.error.kind==sbn::error_kind::finite);
}

class Scratch {
    std::vector<std::byte> bytes_;
    std::size_t size_=0;
public:
    explicit Scratch(std::size_t size):bytes_(size+36,scratch_guard),size_(size){}
    std::span<std::byte> span(){return {bytes_.data()+17,size_};}
    void guards()const {
        assert(std::all_of(bytes_.begin(),bytes_.begin()+17,[](auto b){return b==scratch_guard;}));
        assert(std::all_of(bytes_.begin()+17+size_,bytes_.end(),[](auto b){return b==scratch_guard;}));
    }
};
std::uint64_t digest(const void* data,std::size_t bytes){
    const auto* p=static_cast<const unsigned char*>(data);std::uint64_t h=1469598103934665603ULL;
    for(std::size_t i=0;i<bytes;++i)h=(h^p[i])*1099511628211ULL;
    return h;
}
struct Call {
    sbn::float_ref out;sbn::float_view a,b;sbn::execution ex;
    const floating::FloatProductPrepared* prepared;
    rounding mode;bool square;sbn::float_report report;
};
void execute(Call& c){
    if(c.prepared)c.report=floating::float_product_execute(*c.prepared,c.out,c.a,c.b,c.ex,c.mode);
    else c.report=c.square?sbn::square_into(c.out,c.a,c.ex,c.mode):sbn::mul_into(c.out,c.a,c.b,c.ex,c.mode);
}
void on_scope(void* opaque,sbn3_team_scope* scope){
    auto& c=*static_cast<Call*>(opaque);c.ex.team=nullptr;c.ex.scope=scope;execute(c);
}

// alias: 0 separate, 1 output==left, 2 output==right. The source Input
// objects stay immutable and are retained as the exact oracle operands.
void run_case(const Input& a,const Input& supplied_b,std::size_t precision,rounding mode,
              bool square,unsigned alias=0,sbn3_team* team=nullptr,bool scope=false,
              const floating::FloatProductPrepared* prepared=nullptr,bool allow_underflow=false){
    const auto& b=square?a:supplied_b;
    const auto output_words=(precision+63)/64;
    const auto* aliased=alias==1?&a:alias==2?&b:nullptr;
    const auto capacity=aliased?std::max(output_words,aliased->words.size()):output_words;
    std::vector<std::uint64_t> output(capacity+2,guard);
    if(aliased)std::copy(aliased->words.begin(),aliased->words.end(),output.begin());
    const auto saved_a=a.words,saved_b=b.words;
    sbn::float_buffer target({output.data(),capacity},sbn::bits{precision});
    auto av=a.view(),bv=b.view();
    if(alias==1)av=sbn::float_view::from_magnitude({output.data(),a.words.size()},a.exponent,a.negative);
    if(alias==2)bv=sbn::float_view::from_magnitude({output.data(),b.words.size()},b.exponent,b.negative);
    if(square&&alias==2)av=bv;
    const unsigned workers=team?sbn3_team_workers(team):1;
    allocation_watch_start();
    const auto need=square?sbn::square_requirements(target.ref(),av,workers):
        sbn::multiply_requirements(target.ref(),av,bv,workers);
    assert(!allocation_watch_stop());assert(need.supported()&&need.output_limbs==output_words);
    if(precision<=256)assert(!need.scratch_bytes);
    Scratch scratch(prepared?0:need.scratch_bytes);
    Call call{target.ref(),av,bv,{scratch.span(),team,nullptr},prepared,mode,square,{}};
    allocation_watch_start();
    if(scope){assert(team);sbn3_team_run(team,on_scope,&call);}else execute(call);
    assert(!allocation_watch_stop());
    scratch.guards();assert(output[capacity]==guard&&output[capacity+1]==guard);
    assert(a.words==saved_a&&b.words==saved_b);
    if(aliased)for(std::size_t i=output_words;i<aliased->words.size();++i)assert(output[i]==aliased->words[i]);
    certify(a,b,target.view(),call.report,precision,mode,allow_underflow);
    ++checked;if(prepared)++prepared_checked;
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
        sbn3_team_config tc{};tc.workers=workers;tc.stack_offset=65536;for(auto& cpu:tc.cpu_ids)cpu=-1;
        assert(sbn3_team_create(arena_,&control_,&tc,&value,&error)==SBN3_OK);
    }
    ~Team(){
        sbn3_team_destroy(value);sbn3_arena_release(arena_,&control_);
        sbn3_arena_stats stats{};sbn3_arena_get_stats(arena_,&stats);
        assert(!stats.lease_references&&!stats.active_computations);sbn3_arena_destroy(arena_);
    }
    Team(const Team&)=delete;Team&operator=(const Team&)=delete;
};

void precision_sweep(){
    // Every bit precision through 257, not merely limb boundaries. Two
    // distinct height regimes test exact product rounding and input truncation.
    for(std::size_t p=1;p<=257;++p){
        const auto a=random_input(p+17,int(p%7)-3,(p&1)!=0);
        const auto b=random_input(p+31,2,(p&2)!=0);
        const auto high=random_input(p+331,-1,(p&4)!=0);
        for(auto mode:modes){
            run_case(a,b,p,mode,false,p%3);
            run_case(high,b,p,mode,false,(p+1)%3);
            run_case(high,high,p,mode,true,p%2);
        }
    }
    for(std::size_t p:{383u,511u,513u,769u,1023u,1025u,1537u,2049u,4093u,4097u,6145u,8193u,8201u,16385u}){
        const auto a=random_input(p+213,-3,true),b=random_input(p+397,4,false);
        const auto exact_a=random_input(p/2+3,1),exact_b=random_input(p/2+9,-1,true);
        for(auto mode:modes){
            run_case(a,b,p,mode,false,1);run_case(a,b,p,mode,false,2);
            run_case(a,a,p,mode,true,1);run_case(exact_a,exact_b,p,mode,false);
        }
    }
}

void exact_midpoints(){
    for(std::size_t p:{1u,2u,3u,31u,32u,53u,63u,64u,65u,127u,128u,129u,255u,256u,257u,513u,1025u})
        for(bool negative:{false,true})for(unsigned parity:{0u,1u})for(unsigned tail:{0u,3u,4u,5u}){
            Ref n;ref_set_ui(n.value,1);ref_mul_2exp(n.value,n.value,p+2);
            if(parity&&p>1)ref_add_ui(n.value,n.value,8);
            ref_add_ui(n.value,n.value,tail);
            const auto a=exported(n.value,-37,negative);const Input one{{1},17,false};
            for(auto mode:modes)run_case(a,one,p,mode,false,parity?1:2);
        }
    // All-one significands and near powers stress carry renormalization.
    for(std::size_t p:{1u,63u,64u,65u,129u,257u,1025u,4097u})for(int delta:{-1,0,1}){
        const auto a=near_power(p+149,delta,-1,true),b=near_power(p+151,-delta,2,false);
        for(auto mode:modes){run_case(a,b,p,mode,false,1);run_case(a,a,p,mode,true);}
    }
}

void huge_sources_and_memory(){
    const auto huge=random_input(65549,2,true);const Input short_input{{UINT64_C(0x107925)},-19,false};
    for(std::size_t p:{1u,3u,53u,65u,129u,257u,513u})for(auto mode:modes)
        run_case(huge,short_input,p,mode,false,1);
    // A few full exact large squares; the independent oracle still sees all
    // source limbs. Avoid multiplying this expensive witness by the whole grid.
    for(auto mode:modes)run_case(huge,huge,65,mode,true,1);
    for(std::size_t p:{257u,513u,1025u,4097u}){
        auto modest=random_input(p+145,1);auto enormous=random_input(65549,1);
        std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});
        allocation_watch_start();
        const auto m=sbn::multiply_requirements(out.ref(),modest.view(),modest.view());
        const auto h=sbn::multiply_requirements(out.ref(),enormous.view(),enormous.view());
        const auto ms=sbn::square_requirements(out.ref(),modest.view());
        const auto hs=sbn::square_requirements(out.ref(),enormous.view());
        assert(!allocation_watch_stop());
        assert(m.supported()&&h.supported()&&ms.supported()&&hs.supported());
        assert(m.scratch_bytes==h.scratch_bytes&&m.scratch_alignment==h.scratch_alignment);
        assert(ms.scratch_bytes==hs.scratch_bytes&&ms.scratch_alignment==hs.scratch_alignment);
    }
    // The omitted range is all zero and the observed prefix product is exact:
    // the wrapper must nevertheless retain its analytic, non-exact report.
    const auto power=near_power(65548,0,1,false);const Input one{{1},0,false};
    alignas(64)std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    allocation_watch_start();const auto r=sbn::mul_into(out.ref(),power.view(),one.view());
    assert(!allocation_watch_stop());assert(r.error.kind==sbn::error_kind::finite);
    assert(!sbn::has_flag(r.flags,sbn::float_flag::rounded));
    certify(power,one,out.view(),r,65,rounding::nearest_even);++checked;
}

void prepared_reuse(std::size_t precision,bool square,sbn3_team* team=nullptr){
    const unsigned workers=team?sbn3_team_workers(team):1;
    floating::FloatProductPlan plan{};
    allocation_watch_start();
    const auto status=floating::float_product_query(precision,precision+128,precision+128,square,workers,plan);
    assert(!allocation_watch_stop());assert(status==SBN3_SUPPORTED);
    assert(plan.a_prefix_bits<=precision+128&&plan.b_prefix_bits<=precision+128);
    assert(plan.product_capacity>=plan.product_limbs);
    Scratch scratch(plan.scratch_bytes);
    allocation_watch_start();const auto prepared=floating::float_product_prepare(plan,scratch.span());
    assert(!allocation_watch_stop());
    const auto plan_identity=digest(&plan,sizeof(plan));
    const auto table_identity=digest(prepared.product.prepared,prepared.product.prepared_bytes);
    for(unsigned repeat=0;repeat<4;++repeat){
        // The same geometry handles shorter inputs, exact inputs, and inputs
        // much wider than the nominal bound because retained prefixes fit.
        const auto abits=repeat==0?precision+451:repeat==1?37:repeat==2?precision+128:precision+1139;
        const auto bbits=repeat==0?precision+719:repeat==1?3:repeat==2?precision+17:precision+947;
        const auto a=random_input(abits,-2,repeat&1),b=random_input(bbits,3,repeat&2);
        for(auto mode:modes){
            run_case(a,b,precision,mode,square,repeat%3,team,team&&repeat==3,&prepared);
            assert(digest(&plan,sizeof(plan))==plan_identity);
            assert(digest(prepared.product.prepared,prepared.product.prepared_bytes)==table_identity);
            scratch.guards();
        }
    }
}

void teams(){
    for(unsigned workers:{1u,4u}){
        Team team(workers);
        for(std::size_t p:{257u,1025u,4097u,8201u}){
            const auto a=random_input(p+333,-1,true),b=random_input(p+529,2,false);
            for(auto mode:modes){
                run_case(a,b,p,mode,false,1,team.value);
                run_case(a,a,p,mode,true,1,team.value,true);
            }
        }
        prepared_reuse(4097,false,team.value);prepared_reuse(8201,true,team.value);
    }
    Team team(16);const auto a=random_input(33301,-1,true),b=random_input(33719,1,false);
    run_case(a,b,32769,rounding::nearest_even,false,2,team.value,true);
}

void specials_ranges_queries(){
    alignas(64)std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});
    const Input one{{1},0,false},three{{3},-1,false};
    for(auto mode:modes){
        for(bool an:{false,true})for(bool bn:{false,true}){
            const auto zero=sbn::float_view::from_magnitude({},0,an);
            auto b=three;b.negative=bn;
            allocation_watch_start();auto report=sbn::mul_into(out.ref(),zero,b.view(),{},mode);
            assert(!allocation_watch_stop());assert(out.view().zero()&&out.view().negative()==(an!=bn));
            assert(report.error.exact()&&report.flags==sbn::float_flag::none);
            report=sbn::square_into(out.ref(),zero,{},mode);
            assert(out.view().zero()&&!out.view().negative()&&report.error.exact());
            report=sbn::mul_into(out.ref(),sbn::float_view::infinity(an),b.view(),{},mode);
            assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==(an!=bn));
            report=sbn::mul_into(out.ref(),zero,sbn::float_view::infinity(bn),{},mode);
            assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
        }
        for(bool negative:{false,true}){
            auto report=sbn::square_into(out.ref(),sbn::float_view::infinity(negative),{},mode);
            assert(out.view().classification()==sbn::float_class::infinity&&!out.view().negative());
            auto maximum=one;maximum.exponent=sbn::float_exponent_limit-1;maximum.negative=negative;
            const Input two{{1},1,false};
            report=sbn::mul_into(out.ref(),maximum.view(),two.view(),{},mode);
            assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==negative);
            assert(sbn::has_flag(report.flags,sbn::float_flag::overflow)&&report.error.kind==sbn::error_kind::unbounded);
            auto minimum=one;minimum.exponent=-sbn::float_exponent_limit-1;minimum.negative=negative;
            const Input half{{1},-1,false};
            report=sbn::mul_into(out.ref(),minimum.view(),half.view(),{},mode);
            assert(out.view().zero()&&out.view().negative()==negative);
            assert(sbn::has_flag(report.flags,sbn::float_flag::underflow));
            certify(minimum,half,out.view(),report,65,mode,true);++checked;
            // Exercise the separated early-underflow path and a normal result
            // at the exponent boundary without constructing virtual gaps.
            run_case(minimum,minimum,65,mode,true,0,nullptr,false,nullptr,true);
            run_case(maximum,one,65,mode,false);run_case(minimum,one,65,mode,false);
        }
        auto report=sbn::mul_into(out.ref(),sbn::float_view::nan(),three.view(),{},mode);
        assert(out.view().is_nan());
        report=sbn::square_into(out.ref(),sbn::float_view::nan(),{},mode);assert(out.view().is_nan());
        const Input a{{3},1000000000000LL,true},b{{5},-1000000000000LL,false};
        run_case(a,b,65,mode,false,1);
    }
    std::array<std::uint64_t,1> tiny{guard};sbn::float_buffer too_small(tiny,sbn::bits{257});
    const auto large=random_input(809,1);
    allocation_watch_start();
    const auto m=sbn::multiply_requirements(too_small.ref(),large.view(),large.view());
    const auto s=sbn::square_requirements(too_small.ref(),large.view());
    assert(!allocation_watch_stop());assert(m.status==sbn::support::capacity&&s.status==sbn::support::capacity);
    assert(m.output_limbs==5&&s.output_limbs==5&&tiny[0]==guard);
    for(unsigned workers:{0u,33u}){
        assert(sbn::multiply_requirements(out.ref(),three.view(),one.view(),workers).status==sbn::support::unsupported);
        assert(sbn::square_requirements(out.ref(),three.view(),workers).status==sbn::support::unsupported);
    }
    floating::FloatProductPlan rejected{};
    assert(floating::float_product_query(65,0,1,false,1,rejected)==SBN3_UNSUPPORTED);
    assert(floating::float_product_query(65,1,1,false,0,rejected)==SBN3_UNSUPPORTED);
}
}

int main(){
    specials_ranges_queries();precision_sweep();exact_midpoints();huge_sources_and_memory();
    for(std::size_t p:{1u,65u,257u,513u,1025u,4097u,8193u})for(bool square:{false,true})
        prepared_reuse(p,square);
    teams();
    std::printf("float mul/square: %zu full exact-product certificates (%zu prepared reuse); all modes, prefixes, aliases, ranges, W1/W4/W16, scope and allocation gates PASS\n",
        checked,prepared_checked);
}
