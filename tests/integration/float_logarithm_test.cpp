// Original libsbn tests; LGPL-3.0-or-later.
#include "float/logarithm.hpp"
#include "common/query_audit.hpp"
#include "sbn3/mul.h"
#include "sbn3/arena.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
#if SBN3_CHECK_SMALL
namespace {unsigned watching_queries=0;std::uint64_t observed_queries=0;}
extern "C" void sbn3_query_audit_observer(const char*) noexcept {
    if(__atomic_load_n(&watching_queries,__ATOMIC_ACQUIRE))__atomic_add_fetch(&observed_queries,1,__ATOMIC_RELAXED);
}
#endif
namespace {
namespace f=sbn::v3::floating;
using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x356c0bade28f4197);
std::size_t checked=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;Ref&operator=(const Ref&)=delete;};
struct Ball {Ref center;std::uint64_t radius=0;std::int64_t scale=0;};
struct Input {std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}};
void power2(ref_number* out,std::size_t n){ref_set_ui(out,1);ref_mul_2exp(out,out,n);}
void shift(ref_number* out,const ref_number* in,std::int64_t n){assert(n>=0&&n<1000000);ref_mul_2exp(out,in,std::size_t(n));}
void divide(ref_number* out,const ref_number* a,const ref_number* b){
    // Beyond this threshold the general oracle may use a production witness.
    assert(((ref_sizeinbase(a,2)+63)/64)*((ref_sizeinbase(b,2)+63)/64)<=1048576);
    ref_fdiv_q(out,a,b);
}
void multiply(ref_number* out,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);
    std::size_t n=0;ref_export(words.data(),&n,-1,8,0,0,b);Ref sum,part;ref_set_ui(sum.v,0);
    for(std::size_t j=0;j<n;++j)if(words[j]){ref_mul_ui(part.v,a,words[j]);if(j)ref_mul_2exp(part.v,part.v,64*j);ref_add(sum.v,sum.v,part.v);}
    ref_set(out,sum.v);
}
void log2_reference(Ball& out,std::size_t bits){
    Ref unit,power,den,term;power2(unit.v,bits);ref_set_ui(power.v,3);ref_set_ui(out.center.v,0);std::size_t j=0;
    for(;;++j){ref_mul_ui(den.v,power.v,2*j+1);divide(term.v,unit.v,den.v);if(!ref_sgn(term.v))break;
        ref_add(out.center.v,out.center.v,term.v);ref_mul_ui(power.v,power.v,9);assert(j<bits+8);}
    ref_mul_ui(out.center.v,out.center.v,2);out.radius=2*j+4;out.scale=-std::int64_t(bits);
}
// Independent fixed-point positive atanh series for exact 0<=a/b<=1/3.
// z's floor loses<1 unit, w=z^2 loses<2, term recurrence loses<3,
// division by odd integers loses<4, and the final omitted tail loses<4.
void atanh_reference(Ball& out,const ref_number* a,const ref_number* b,std::size_t bits){
    if(!ref_sgn(a)){ref_set_ui(out.center.v,0);out.radius=0;out.scale=-std::int64_t(bits);return;}
    Ref z,w,term,scaled,product,odd,part;ref_mul_2exp(scaled.v,a,bits);divide(z.v,scaled.v,b);
    multiply(product.v,z.v,z.v);ref_fdiv_q_2exp(w.v,product.v,bits);ref_set(term.v,z.v);ref_set_ui(out.center.v,0);
    std::size_t j=0;while(ref_sgn(term.v)){
        ref_set_ui(odd.v,2*j+1);divide(part.v,term.v,odd.v);ref_add(out.center.v,out.center.v,part.v);
        multiply(product.v,term.v,w.v);ref_fdiv_q_2exp(term.v,product.v,bits);assert(++j<bits+8);
    }
    ref_mul_ui(out.center.v,out.center.v,2);out.radius=16*j+16;out.scale=-std::int64_t(bits);
}
void reference(Ball& out,const Input& input,f::LogarithmKind kind,std::size_t bits){
    Ref x,unit,num,den;ref_import(x.v,input.words.size(),-1,8,0,0,input.words.data());auto exponent=input.exponent;
    if(kind==f::LogarithmKind::log1p){
        assert(exponent>-100000&&exponent<100000);
        if(exponent<0){power2(unit.v,std::size_t(-exponent));if(input.negative)ref_sub(x.v,unit.v,x.v);else ref_add(x.v,x.v,unit.v);}
        else{ref_mul_2exp(x.v,x.v,std::size_t(exponent));assert(!input.negative);ref_add_ui(x.v,x.v,1);exponent=0;}
    }
    assert(ref_sgn(x.v)>0);const auto width=ref_sizeinbase(x.v,2);const auto k=exponent+std::int64_t(width)-1;
    if(k==-1||k==0){
        // Preserve near-one cancellation in the independent reference too.
        assert(exponent<=0);power2(unit.v,std::size_t(-exponent));ref_sub(num.v,x.v,unit.v);
        const bool negative=ref_sgn(num.v)<0;ref_abs(num.v,num.v);ref_add(den.v,x.v,unit.v);
        atanh_reference(out,num.v,den.v,bits);if(negative)ref_neg(out.center.v,out.center.v);return;
    }
    power2(unit.v,width-1);ref_sub(num.v,x.v,unit.v);ref_add(den.v,x.v,unit.v);atanh_reference(out,num.v,den.v,bits);
    Ball log2;log2_reference(log2,bits);const auto magnitude=k<0?std::uint64_t(0)-std::uint64_t(k):std::uint64_t(k);
    assert(magnitude<1000000);ref_mul_ui(log2.center.v,log2.center.v,magnitude);
    if(k<0)ref_sub(out.center.v,out.center.v,log2.center.v);else ref_add(out.center.v,out.center.v,log2.center.v);
    out.radius+=magnitude*log2.radius;
}
void certify(sbn::float_view output,sbn::float_report report,const Ball& truth,std::size_t p,rounding mode){
    assert(output.finite()&&report.error.kind!=sbn::error_kind::unbounded);
    if(!ref_sgn(truth.center.v)&&!truth.radius){assert(output.zero()&&report.error.exact());++checked;return;}
    assert(!output.zero());Ref center,radius,lo,hi,expected,expected_radius,elo,ehi;
    ref_import(center.v,output.magnitude().size(),-1,8,0,0,output.magnitude().data());
    const auto width=ref_sizeinbase(center.v,2);assert(width<=p);if(output.negative())ref_neg(center.v,center.v);
    const auto ulp=output.exponent2()+std::int64_t(width)-std::int64_t(p);
    assert(report.error.exact()||report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
    auto scale=std::min(output.exponent2(),truth.scale);if(!report.error.exact())scale=std::min(scale,report.error.exponent2);
    shift(center.v,center.v,output.exponent2()-scale);ref_set_ui(radius.v,report.error.exact()?0:1);
    if(!report.error.exact())shift(radius.v,radius.v,report.error.exponent2-scale);
    ref_sub(lo.v,center.v,radius.v);ref_add(hi.v,center.v,radius.v);
    shift(expected.v,truth.center.v,truth.scale-scale);ref_set_ui(expected_radius.v,truth.radius);shift(expected_radius.v,expected_radius.v,truth.scale-scale);
    ref_sub(elo.v,expected.v,expected_radius.v);ref_add(ehi.v,expected.v,expected_radius.v);
    if(ref_cmp(lo.v,elo.v)>0||ref_cmp(hi.v,ehi.v)<0){std::fprintf(stderr,"log enclosure failed p=%zu mode=%u exp=%lld radius=%lld\n",
        p,unsigned(mode),(long long)output.exponent2(),(long long)report.error.exponent2);assert(false);}++checked;
}
struct Team {sbn3_arena* arena=nullptr;sbn3_lease control{};sbn3_team* team=nullptr;
    explicit Team(unsigned workers){sbn3_error error{};sbn3_arena_config a{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&a,&arena,&error)==SBN3_OK);assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config t{};t.workers=workers;t.stack_offset=65536;
        for(auto& cpu:t.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&t,&team,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(team);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}};
struct ScopeCall {const f::FloatLogarithmPrepared* program;sbn::float_ref out;sbn::float_view input;f::LogarithmKind kind;rounding mode;sbn::float_report report{};};
void scoped(void* opaque,sbn3_team_scope* scope){auto& c=*static_cast<ScopeCall*>(opaque);
    c.report=f::logarithm_program_execute(*c.program,c.out,c.input,c.kind,{{},nullptr,scope},c.mode);}
void run(const f::FloatLogarithmPrepared& prepared,std::size_t p,const Input& input,f::LogarithmKind kind,const Ball& truth,
         rounding mode,sbn3_team* team,bool scope,bool alias){
    const auto limbs=(p+63)/64,capacity=alias?std::max(limbs,input.words.size()):limbs;
    std::vector<std::uint64_t> words(capacity+2,guard);if(alias)std::copy(input.words.begin(),input.words.end(),words.begin());
    sbn::float_buffer out({words.data(),capacity},sbn::bits{p});const auto original=input.words;
    const auto source=alias?sbn::float_view::from_magnitude({words.data(),input.words.size()},input.exponent,input.negative):input.view();
    allocation_watch_start();sbn::float_report report;
    if(scope){ScopeCall call{&prepared,out.ref(),source,kind,mode,{}};sbn3_team_run(team,scoped,&call);report=call.report;}
    else report=f::logarithm_program_execute(prepared,out.ref(),source,kind,{{},team,nullptr},mode);
    assert(!allocation_watch_stop());assert(words[capacity]==guard&&words[capacity+1]==guard&&input.words==original);
    if(alias)for(std::size_t j=limbs;j<input.words.size();++j)assert(words[j]==input.words[j]);
    certify(out.view(),report,truth,p,mode);
}
Input near_one(std::size_t gap,bool below){
    Input input;input.exponent=-std::int64_t(gap);input.words.assign(gap/64+1,0);
    if(below){for(std::size_t j=0;j<gap/64;++j)input.words[j]=UINT64_MAX;
        if(gap%64)input.words[gap/64]=(UINT64_C(1)<<(gap%64))-1;}
    else{input.words[gap/64]=UINT64_C(1)<<(gap%64);input.words[0]|=1;}
    while(!input.words.empty()&&!input.words.back())input.words.pop_back();return input;
}
void tiny_extreme(const f::FloatLogarithmPrepared& prepared,std::size_t p,sbn3_team* team){
    std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});const std::uint64_t one=1;
    for(auto exponent:std::array<std::int64_t,2>{-sbn::float_exponent_limit-1,-1000000})for(bool negative:{false,true})for(auto mode:modes){
        const auto source=sbn::float_view::from_magnitude({&one,1},std::int64_t(exponent),negative);
        allocation_watch_start();const auto report=f::logarithm_program_execute(prepared,out.ref(),source,f::LogarithmKind::log1p,{{},team,nullptr},mode);
        assert(!allocation_watch_stop());assert(out.view().negative()==negative&&out.view().exponent2()==exponent&&out.view().magnitude().size()==1&&out.view().magnitude().data()[0]==1);
        assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2>=2*exponent&&
            report.error.exponent2<=exponent+1-std::int64_t(p)+(mode==rounding::nearest_even?0:1));++checked;
    }
}
void cases(std::size_t p,sbn3_team* team,bool many){
    const unsigned workers=team?sbn3_team_workers(team):1;allocation_watch_start();const auto need=f::logarithm_program_requirements(sbn::bits{p},workers);
    assert(!allocation_watch_stop()&&need.supported());sbn::workspace storage;storage.reserve(need.scratch_bytes,need.scratch_alignment);
    allocation_watch_start();const auto prepared=f::logarithm_program_prepare(sbn::bits{p},workers,storage.on().scratch.first(need.scratch_bytes));assert(!allocation_watch_stop());
    Ball truth;const Input three{{3},0,false};reference(truth,three,f::LogarithmKind::log,p+512);
    for(auto mode:modes)run(prepared,p,three,f::LogarithmKind::log,truth,mode,team,team!=nullptr,true);
    const Input fraction{{3},-3,false};reference(truth,fraction,f::LogarithmKind::log,p+512);
    run(prepared,p,fraction,f::LogarithmKind::log,truth,rounding::nearest_even,team,false,false);
    if(!many)return;
    for(auto kind:{f::LogarithmKind::log,f::LogarithmKind::log1p}){
        const Input ordinary{{5},kind==f::LogarithmKind::log?-1:-3,false};reference(truth,ordinary,kind,p+512);
        for(auto mode:modes)run(prepared,p,ordinary,kind,truth,mode,team,false,false);
    }
    for(auto gap:{7u,unsigned(p/3+8),unsigned(p+80)})for(bool negative:{false,true}){
        const Input delta{{1},-std::int64_t(gap),negative};reference(truth,delta,f::LogarithmKind::log1p,p+2*gap+512);
        for(auto mode:modes)run(prepared,p,delta,f::LogarithmKind::log1p,truth,mode,team,false,true);
        const auto point=near_one(gap,negative);reference(truth,point,f::LogarithmKind::log,p+2*gap+512);
        run(prepared,p,point,f::LogarithmKind::log,truth,rounding::nearest_even,team,false,true);
    }
    auto endpoint=near_one(p+300,true);endpoint.negative=true;reference(truth,endpoint,f::LogarithmKind::log1p,p+512);
    run(prepared,p,endpoint,f::LogarithmKind::log1p,truth,rounding::nearest_even,team,false,true);
    // Exponent magnitude affects metadata, not allocated mantissa length.
    const std::int64_t huge=std::int64_t(1)<<58;log2_reference(truth,p+512);truth.scale+=58;
    run(prepared,p,Input{{1},huge,false},f::LogarithmKind::log,truth,rounding::nearest_even,team,false,false);
    ++truth.radius;run(prepared,p,Input{{1},huge,false},f::LogarithmKind::log1p,truth,rounding::nearest_even,team,false,false);
    --truth.radius;ref_neg(truth.center.v,truth.center.v);
    run(prepared,p,Input{{1},-huge,false},f::LogarithmKind::log,truth,rounding::nearest_even,team,false,false);
    tiny_extreme(prepared,p,team);
    const Input fresh_input{{7},-2,false};reference(truth,fresh_input,f::LogarithmKind::log,p+512);
    std::vector<std::uint64_t> words((p+63)/64);sbn::float_buffer out(words,sbn::bits{p});sbn::workspace scratch;
    const auto fresh=f::logarithm_requirements(out.ref(),fresh_input.view(),f::LogarithmKind::log,workers);scratch.reserve(fresh.scratch_bytes,fresh.scratch_alignment);
    allocation_watch_start();const auto report=f::logarithm_into(out.ref(),fresh_input.view(),f::LogarithmKind::log,scratch.on(team));assert(!allocation_watch_stop());
    certify(out.view(),report,truth,p,rounding::nearest_even);
}
void specials(){
    const auto need=f::logarithm_program_requirements(sbn::bits{65});sbn::workspace work;work.reserve(need.scratch_bytes,need.scratch_alignment);
    const auto prepared=f::logarithm_program_prepare(sbn::bits{65},1,work.on().scratch.first(need.scratch_bytes));
    std::array<std::uint64_t,2> words{};sbn::float_buffer out(words,sbn::bits{65});const std::uint64_t one=1,two=2;
    const auto one_view=sbn::float_view::from_magnitude({&one,1},0);
    auto report=f::logarithm_program_execute(prepared,out.ref(),one_view);assert(out.view().zero()&&report.error.exact());
    for(bool negative:{false,true}){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        report=f::logarithm_program_execute(prepared,out.ref(),zero);assert(!out.view().finite()&&out.view().negative()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
        report=f::logarithm_program_execute(prepared,out.ref(),zero,f::LogarithmKind::log1p);assert(out.view().zero()&&out.view().negative()==negative&&report.error.exact());
    }
    report=f::logarithm_program_execute(prepared,out.ref(),one_view.negated(),f::LogarithmKind::log1p);
    assert(!out.view().finite()&&out.view().negative()&&sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    for(auto input:{one_view.negated(),sbn::float_view::infinity(true)}){
        report=f::logarithm_program_execute(prepared,out.ref(),input);assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));}
    report=f::logarithm_program_execute(prepared,out.ref(),sbn::float_view::from_magnitude({&two,1},0,true),f::LogarithmKind::log1p);
    assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
    report=f::logarithm_program_execute(prepared,out.ref(),sbn::float_view::infinity());assert(!out.view().finite()&&!out.view().negative());
    report=f::logarithm_program_execute(prepared,out.ref(),sbn::float_view::nan());assert(out.view().is_nan());
}
void fresh_special_dispatch(){
    // The checked hook is calibrated independently; uninstrumented native
    // builds still require zero scratch and zero allocation, but make no
    // claim about observing every possible internal planning entry point.
#if SBN3_CHECK_SMALL
    __atomic_store_n(&observed_queries,0,__ATOMIC_RELAXED);__atomic_store_n(&watching_queries,1,__ATOMIC_RELEASE);
    sbn3_mul_plan plan{};sbn3_mul_info info{};sbn3_mul_options options{};options.workers=1;
    const sbn3_product_spec spec{17,19};(void)sbn3_mul_query(&spec,&options,&plan,&info);
    __atomic_store_n(&watching_queries,0,__ATOMIC_RELEASE);assert(__atomic_load_n(&observed_queries,__ATOMIC_RELAXED)>0);
#endif
    std::array<std::uint64_t,3> words{};sbn::float_buffer out(words,sbn::bits{129});
    const std::uint64_t one=1,two=2;const auto one_view=sbn::float_view::from_magnitude({&one,1},0);
    std::array<std::uint64_t,17> tall{};tall.back()=1;
    struct Case {sbn::float_view x;f::LogarithmKind kind;sbn::float_class type;bool negative;sbn::float_flag flag;};
    const Case cases[]{
        {{},f::LogarithmKind::log,sbn::float_class::infinity,true,sbn::float_flag::divide_by_zero},
        {sbn::float_view::from_magnitude({},0,true),f::LogarithmKind::log,sbn::float_class::infinity,true,sbn::float_flag::divide_by_zero},
        {one_view,f::LogarithmKind::log,sbn::float_class::finite,false,sbn::float_flag::none},
        {sbn::float_view::from_magnitude(tall,-1024),f::LogarithmKind::log,sbn::float_class::finite,false,sbn::float_flag::none},
        {one_view.negated(),f::LogarithmKind::log,sbn::float_class::quiet_nan,false,sbn::float_flag::invalid},
        {sbn::float_view::infinity(true),f::LogarithmKind::log,sbn::float_class::quiet_nan,false,sbn::float_flag::invalid},
        {sbn::float_view::infinity(),f::LogarithmKind::log,sbn::float_class::infinity,false,sbn::float_flag::none},
        {sbn::float_view::nan(),f::LogarithmKind::log,sbn::float_class::quiet_nan,false,sbn::float_flag::none},
        {{},f::LogarithmKind::log1p,sbn::float_class::finite,false,sbn::float_flag::none},
        {sbn::float_view::from_magnitude({},0,true),f::LogarithmKind::log1p,sbn::float_class::finite,true,sbn::float_flag::none},
        {one_view.negated(),f::LogarithmKind::log1p,sbn::float_class::infinity,true,sbn::float_flag::divide_by_zero},
        {sbn::float_view::from_magnitude({&two,1},0,true),f::LogarithmKind::log1p,sbn::float_class::quiet_nan,false,sbn::float_flag::invalid},
        {sbn::float_view::infinity(true),f::LogarithmKind::log1p,sbn::float_class::quiet_nan,false,sbn::float_flag::invalid},
        {sbn::float_view::infinity(),f::LogarithmKind::log1p,sbn::float_class::infinity,false,sbn::float_flag::none},
        {sbn::float_view::nan(),f::LogarithmKind::log1p,sbn::float_class::quiet_nan,false,sbn::float_flag::none}};
#if SBN3_CHECK_SMALL
    __atomic_store_n(&observed_queries,0,__ATOMIC_RELAXED);__atomic_store_n(&watching_queries,1,__ATOMIC_RELEASE);
#endif
    allocation_watch_start();
    for(const auto& c:cases)for(auto mode:modes){
        const auto need=c.kind==f::LogarithmKind::log?sbn::log_requirements(out.ref(),c.x,1):sbn::log1p_requirements(out.ref(),c.x,1);
        assert(need.supported()&&!need.scratch_bytes&&need.output_limbs==3);
        const auto report=c.kind==f::LogarithmKind::log?sbn::log_into(out.ref(),c.x,{},mode):sbn::log1p_into(out.ref(),c.x,{},mode);
        assert(out.view().classification()==c.type&&out.view().negative()==c.negative&&report.flags==c.flag);
        if(c.type==sbn::float_class::finite)assert(out.view().zero()&&report.error.exact());
        else assert(report.error.kind==sbn::error_kind::unbounded);++checked;
    }
    assert(!allocation_watch_stop());
#if SBN3_CHECK_SMALL
    __atomic_store_n(&watching_queries,0,__ATOMIC_RELEASE);assert(!__atomic_load_n(&observed_queries,__ATOMIC_RELAXED));
#endif
    assert(!f::logarithm_requirements(out.ref(),one_view,f::LogarithmKind::log,0).supported());
    assert(!f::logarithm_requirements(out.ref(),one_view,static_cast<f::LogarithmKind>(99),1).supported());
    sbn::float_buffer short_output({},sbn::bits{129});
    assert(f::logarithm_requirements(short_output.ref(),one_view,f::LogarithmKind::log,1).status==sbn::support::capacity);
}
}
int main(){
    fresh_special_dispatch();
    for(auto p:{1u,3u,17u,63u,65u,127u,193u,257u})cases(p,nullptr,true);
    for(auto p:{509u,1025u,4097u})cases(p,nullptr,false);
    {Team team(1);cases(129,team.team,true);}{Team team(4);cases(257,team.team,true);}
    specials();std::printf("float logarithm independent enclosures: %zu\n",checked);
}
