// Original libsbn tests; LGPL-3.0-or-later.
#include "float/bit_burst.hpp"
#include "common/small_checks.h"
#include "runtime/team.hpp"
#include "sbn3/mul.h"
#include "../oracle/oracle.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
std::atomic<bool> watch_queries{false};
std::atomic<std::uint64_t> query_count{0};
}
extern "C" void sbn3_query_audit_observer(const char*) noexcept {
    if(watch_queries.load(std::memory_order_relaxed))query_count.fetch_add(1,std::memory_order_relaxed);
}
namespace {
namespace floating=sbn::v3::floating;
using Kind=floating::BitBurstKind;using sbn::rounding;
constexpr rounding modes[]{rounding::nearest_even,rounding::toward_zero,rounding::upward,rounding::downward,rounding::away_zero};
constexpr std::uint64_t guard=UINT64_C(0x125ed06348b97acf);
std::size_t certified=0,wide_runs=0,prepared_runs=0,tiny_runs=0;
struct Ref {ref_int v;Ref(){ref_init(v);}~Ref(){ref_clear(v);}Ref(const Ref&)=delete;};
struct Interval {Ref lo,hi;};
struct Input {
    std::vector<std::uint64_t> words;std::int64_t exponent=0;bool negative=false;
    sbn::float_view view()const{return sbn::float_view::from_magnitude(words,exponent,negative);}
};
std::size_t width(sbn::float_view x){const auto m=x.magnitude();return m.zero()?0:64*(m.size()-1)+std::bit_width(m.data()[m.size()-1]);}
std::vector<std::uint64_t> export_words(const ref_number* a){
    std::vector<std::uint64_t> result((ref_sizeinbase(a,2)+63)/64);std::size_t count=0;
    ref_export(result.data(),&count,-1,8,0,0,a);result.resize(count);return result;
}
void product(ref_number* out,const ref_number* a,const ref_number* b){
    // ref_mul has a production fallback for large operands. This deliberately
    // uses only independently implemented scalar multiplications and addition.
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);const auto digits=export_words(b);Ref sum,term;
    ref_set_ui(sum.v,0);
    for(std::size_t i=digits.size();i-->0;){ref_mul_2exp(sum.v,sum.v,64);ref_mul_ui(term.v,a,digits[i]);ref_add(sum.v,sum.v,term.v);}
    ref_set(out,sum.v);
}
void divide_word(ref_number* out,const ref_number* in,std::uint64_t divisor,bool upward){
    assert(divisor&&ref_sgn(in)>=0);auto q=export_words(in);std::uint64_t remainder=0;
    for(std::size_t i=q.size();i-->0;){const __uint128_t value=(__uint128_t(remainder)<<64)|q[i];
        q[i]=std::uint64_t(value/divisor);remainder=std::uint64_t(value%divisor);}
    ref_import(out,q.size(),-1,8,0,0,q.data());if(upward&&remainder)ref_add_ui(out,out,1);
}
void divide_pow2(ref_number* out,const ref_number* in,std::size_t shift,bool upward){
    Ref remainder;ref_fdiv_r_2exp(remainder.v,in,shift);const bool discarded=ref_sgn(remainder.v)!=0;
    ref_fdiv_q_2exp(out,in,shift);if(upward&&discarded)ref_add_ui(out,out,1);
}
void negate(Interval& x){Ref t;ref_set(t.v,x.lo.v);ref_neg(x.lo.v,x.hi.v);ref_neg(x.hi.v,t.v);}
void widen(Interval& x,const ref_number* amount){ref_sub(x.lo.v,x.lo.v,amount);ref_add(x.hi.v,x.hi.v,amount);}
void multiply_grid(Interval& out,const Interval& a,const ref_number* positive,std::size_t Q){
    Ref lo,hi;product(lo.v,a.lo.v,positive);product(hi.v,a.hi.v,positive);
    divide_pow2(out.lo.v,lo.v,Q,false);divide_pow2(out.hi.v,hi.v,Q,true);
}
struct Reference {
    Interval exponential,expm1,sine,cosine;std::int64_t exponent=0;
    // Dense high-precision cases are 1/2+h, where h starts beyond p/2.
    // A rigorous first-order perturbation plus an independently evaluated
    // short-argument Taylor series avoids an expensive full-height reference
    // tree while certifying every requested output bit of a wide-leaf call.
    Reference(const Input& input,std::size_t p,bool perturb=false){
        const auto top=input.exponent+std::int64_t(width(input.view()));
        const std::size_t Q=p+256+std::size_t(std::max<std::int64_t>(0,-top));exponent=-std::int64_t(Q);
        const Input base=perturb?Input{{1},-1,input.negative}:input;
        Ref M,unit,lo,hi,tmp,tail;
        ref_import(M.v,base.words.size(),-1,8,0,0,base.words.data());
        ref_set_ui(unit.v,1);ref_mul_2exp(unit.v,unit.v,Q);
        ref_set(lo.v,unit.v);ref_set(hi.v,unit.v);
        ref_set(exponential.lo.v,unit.v);ref_set(exponential.hi.v,unit.v);
        ref_set(cosine.lo.v,unit.v);ref_set(cosine.hi.v,unit.v);
        std::size_t J=1,factorial_bits=0;
        while(factorial_bits<Q+40){++J;factorial_bits+=std::bit_width(J)-1;}
        // J! >= 2^factorial_bits. Both subseries tails and the exp tail
        // are below 2/(J+1)! < one Q-grid unit on |x|<=1.
        for(std::size_t j=1;j<=J;++j){
            auto next=[&](ref_number* term,bool upward){
                product(tmp.v,term,M.v);
                if(base.exponent<0)divide_pow2(tmp.v,tmp.v,std::size_t(-base.exponent),upward);
                else if(base.exponent)ref_mul_2exp(tmp.v,tmp.v,std::size_t(base.exponent));
                divide_word(term,tmp.v,j,upward);
            };
            next(lo.v,false);next(hi.v,true);
            auto accumulate=[&](Interval& sum,bool minus){
                if(minus){ref_sub(sum.lo.v,sum.lo.v,hi.v);ref_sub(sum.hi.v,sum.hi.v,lo.v);}
                else{ref_add(sum.lo.v,sum.lo.v,lo.v);ref_add(sum.hi.v,sum.hi.v,hi.v);}
            };
            accumulate(exponential,input.negative&&(j&1));
            accumulate((j&1)?sine:cosine,(j/2)&1);
        }
        ref_set_ui(tail.v,1);widen(exponential,tail.v);widen(sine,tail.v);widen(cosine,tail.v);
        if(perturb){
            const auto k=(p+1)/2+70;assert(input.exponent==-std::int64_t(p+97)&&Q>=p+97);
            Ref magnitude,h,half,factor,remainder;
            ref_import(magnitude.v,input.words.size(),-1,8,0,0,input.words.data());
            ref_mul_2exp(magnitude.v,magnitude.v,Q-(p+97));
            ref_fdiv_q_2exp(half.v,unit.v,1);ref_sub(h.v,magnitude.v,half.v);assert(ref_sgn(h.v)>0);
            // 0<h<2^(1-k). Since exp(1/2)<2 and |h|<1/2, each
            // Taylor remainder used here is <=2^(4-2k), including exp(-x).
            ref_set_ui(remainder.v,1);assert(Q+4>=2*k);ref_mul_2exp(remainder.v,remainder.v,Q+4-2*k);
            if(input.negative)ref_sub(factor.v,unit.v,h.v);else ref_add(factor.v,unit.v,h.v);
            multiply_grid(exponential,exponential,factor.v,Q);widen(exponential,remainder.v);
            Interval hs,hc;multiply_grid(hs,sine,h.v,Q);multiply_grid(hc,cosine,h.v,Q);
            ref_add(sine.lo.v,sine.lo.v,hc.lo.v);ref_add(sine.hi.v,sine.hi.v,hc.hi.v);
            ref_sub(cosine.lo.v,cosine.lo.v,hs.hi.v);ref_sub(cosine.hi.v,cosine.hi.v,hs.lo.v);
            widen(sine,remainder.v);widen(cosine,remainder.v);
        }
        ref_sub(expm1.lo.v,exponential.lo.v,unit.v);ref_sub(expm1.hi.v,exponential.hi.v,unit.v);
        if(input.negative)negate(sine);
    }
};
void shift(ref_number* x,std::int64_t bits){assert(bits>=0&&bits<1000000);if(bits)ref_mul_2exp(x,x,std::size_t(bits));}
void certificate(sbn::float_view value,sbn::float_report report,std::size_t p,rounding mode,
                 const Interval& expected,std::int64_t expected_exponent){
    assert(value.finite()&&!value.zero()&&width(value)<=p&&report.error.kind==sbn::error_kind::finite);
    const auto ulp=value.exponent2()+std::int64_t(width(value))-std::int64_t(p);
    assert(report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
    const auto common=std::min({value.exponent2(),report.error.exponent2,expected_exponent});
    Ref center,radius,low,high,oracle_low,oracle_high;
    ref_import(center.v,value.magnitude().size(),-1,8,0,0,value.magnitude().data());
    if(value.negative())ref_neg(center.v,center.v);shift(center.v,value.exponent2()-common);
    ref_set_ui(radius.v,1);shift(radius.v,report.error.exponent2-common);
    ref_sub(low.v,center.v,radius.v);ref_add(high.v,center.v,radius.v);
    ref_set(oracle_low.v,expected.lo.v);ref_set(oracle_high.v,expected.hi.v);
    shift(oracle_low.v,expected_exponent-common);shift(oracle_high.v,expected_exponent-common);
    assert(ref_cmp(low.v,oracle_low.v)<=0&&ref_cmp(high.v,oracle_high.v)>=0);++certified;
}
class Team {
    sbn3_arena* arena=nullptr;sbn3_lease control{};
public:
    sbn3_team* value=nullptr;
    Team(){sbn3_error error{};sbn3_arena_config config{std::size_t(1)<<28,32u<<20};
        assert(sbn3_arena_create(&config,&arena,&error)==SBN3_OK);
        assert(sbn3_arena_prepare(arena,0,sbn3_team_storage_bytes(),&error)==SBN3_OK);
        sbn3_arena_acquire(arena,0,sbn3_team_storage_bytes(),&control);sbn3_team_config tc{};tc.workers=4;tc.stack_offset=65536;
        for(auto& cpu:tc.cpu_ids)cpu=-1;assert(sbn3_team_create(arena,&control,&tc,&value,&error)==SBN3_OK);}
    ~Team(){sbn3_team_destroy(value);sbn3_arena_release(arena,&control);sbn3_arena_destroy(arena);}
};
struct Call {
    Kind kind;sbn::float_ref out,other;sbn::float_view input;sbn::execution ex;rounding mode;
    const floating::FloatBitBurstPrepared* prepared=nullptr;floating::TrigSincosReport reports{};
};
void execute(Call& c){
    if(c.kind==Kind::sincos)c.reports=c.prepared?floating::sincos_bit_burst_execute(*c.prepared,c.out,c.other,c.input,c.ex,c.mode):
        floating::bit_burst_sincos_into(c.out,c.other,c.input,c.ex,c.mode);
    else c.reports.sine=c.prepared?floating::exp_bit_burst_execute(*c.prepared,c.out,c.input,c.ex,c.mode):
        floating::bit_burst_into(c.out,c.input,c.kind,c.ex,c.mode);
}
void scoped(void* raw,sbn3_team_scope* scope){auto& c=*static_cast<Call*>(raw);c.ex.team=nullptr;c.ex.scope=scope;execute(c);}
void run(const Input& input,const Reference& reference,std::size_t p,Kind kind,rounding mode,unsigned alias,
         const floating::FloatBitBurstPrepared* prepared=nullptr,sbn3_team* team=nullptr,bool child=false,bool expect_wide=false,
         std::size_t cp=0){
    if(!cp)cp=p;const auto n=(p+63)/64,m=(cp+63)/64;
    const auto ac=alias==1?std::max(n,input.words.size()):n,bc=alias==2?std::max(m,input.words.size()):m;
    std::vector<std::uint64_t> a(ac+2,guard),b(bc+2,guard);const auto original=input.words;
    if(alias)std::copy(input.words.begin(),input.words.end(),alias==1?a.begin():b.begin());
    sbn::float_buffer out{{a.data(),ac},sbn::bits{p}},other{{b.data(),bc},sbn::bits{cp}};
    const auto source=alias?sbn::float_view::from_magnitude({alias==1?a.data():b.data(),input.words.size()},input.exponent,input.negative):input.view();
    sbn::workspace scratch;sbn::execution ex{{},team,nullptr};std::size_t extent=0;
    if(!prepared){allocation_watch_start();const auto need=kind==Kind::sincos?
            floating::bit_burst_sincos_requirements(out.ref(),other.ref(),source,team?4:1):
            floating::bit_burst_requirements(out.ref(),source,kind,team?4:1);
        assert(!allocation_watch_stop()&&need.supported());extent=need.scratch_bytes;scratch.reserve(extent+64,need.scratch_alignment);
        ex=scratch.on(team);ex.scratch=ex.scratch.first(extent);
        std::fill(scratch.on().scratch.begin()+extent,scratch.on().scratch.begin()+extent+64,std::byte{0x51});}
    Call call{kind,out.ref(),other.ref(),source,ex,mode,prepared,{}};
    query_count.store(0,std::memory_order_relaxed);watch_queries.store(prepared!=nullptr,std::memory_order_relaxed);allocation_watch_start();
    if(child){assert(team);sbn3_team_run(team,[](void* raw,sbn3_team_scope* scope){
        sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,[](void*,sbn3_team_scope*){},nullptr,
            [](void* p,sbn3_team_scope* sub){assert(sbn3_team_first_worker(sub)==2);scoped(p,sub);},raw);
    },&call);}else if(team)sbn3_team_run(team,scoped,&call);else execute(call);
    const auto allocations=allocation_watch_stop();watch_queries.store(false,std::memory_order_relaxed);
    assert(!allocations&&!query_count.load(std::memory_order_relaxed));
    assert(input.words==original&&a[ac]==guard&&a[ac+1]==guard&&b[bc]==guard&&b[bc+1]==guard);
    if(alias){const auto& v=alias==1?a:b;for(std::size_t i=alias==1?n:m;i<input.words.size();++i)assert(v[i]==input.words[i]);}
    if(!prepared)for(std::size_t i=0;i<64;++i)assert(scratch.on().scratch[extent+i]==std::byte{0x51});
    const auto& primary=kind==Kind::exp?reference.exponential:kind==Kind::expm1?reference.expm1:reference.sine;
    certificate(out.view(),call.reports.sine,p,mode,primary,reference.exponent);
    if(kind==Kind::sincos)certificate(other.view(),call.reports.cosine,cp,mode,reference.cosine,reference.exponent);
    if(prepared){++prepared_runs;const auto stats=floating::bit_burst_statistics(*prepared);
        assert(stats.nonzero_slices);if(expect_wide){assert(stats.max_runtime_leaf_limbs>64);++wide_runs;}}
}
Input dense_perturbation(std::size_t p,bool negative){
    const auto H=p+97,k=(p+1)/2+70,high=H-k;Input x;x.words.resize((H+63)/64);x.exponent=-std::int64_t(H);x.negative=negative;
    for(std::size_t i=0;i<=high/64;++i)x.words[i]=UINT64_C(0xa91276dc4308f5eb)^(UINT64_C(0x9e3779b97f4a7c15)*i);
    x.words[high/64]&=(high%64==63?UINT64_MAX:(UINT64_C(1)<<(high%64+1))-1);
    x.words[high/64]|=UINT64_C(1)<<(high%64);x.words[(H-1)/64]|=UINT64_C(1)<<((H-1)%64);return x;
}
void ordinary(){
    for(std::size_t p:{1u,17u,65u,129u,257u,1025u}){
        const Input cases[]{{{1},0,false},{{3},-2,false},{{1},-9,false}};
        for(auto input:cases)for(bool negative:{false,true}){input.negative=negative;Reference reference(input,p);
            for(auto kind:{Kind::exp,Kind::expm1,Kind::sincos})for(auto mode:modes)
                run(input,reference,p,kind,mode,kind==Kind::sincos&&negative?2:1);}
    }
    Input dense;dense.words.resize(19,UINT64_C(0xb417629edca5038f));dense.words.back()=UINT64_C(0x157);dense.exponent=-1162;
    for(bool negative:{false,true}){dense.negative=negative;Reference reference(dense,513);
        for(auto kind:{Kind::exp,Kind::expm1,Kind::sincos})run(dense,reference,513,kind,rounding::nearest_even,1);}
    Input tiny_tall;tiny_tall.words.resize(129,UINT64_C(0xf1287bd63e04ca95));tiny_tall.words.back()=31;tiny_tall.exponent=-8297;
    for(bool negative:{false,true}){tiny_tall.negative=negative;Reference reference(tiny_tall,65);
        for(auto kind:{Kind::exp,Kind::expm1,Kind::sincos})for(auto mode:modes)
            run(tiny_tall,reference,65,kind,mode,kind==Kind::sincos?2:1);}
    // 8197 input bits, top=-1200 and p=1025 force the tiny expm1
    // prefix into the quoted caller scratch (17 words), beyond the stack
    // leaf. The oracle grid is Q=2481, safely below its ~2^-2225 ulp.
    // This is an expm1 certificate; no claim is made here about a tighter
    // cosine radius at an independently chosen much smaller argument.
    tiny_tall.exponent=-9397;
    std::array<std::uint64_t,17> tiny_output{};sbn::float_buffer tiny_buffer(tiny_output,sbn::bits{1025});
    const auto tiny_need=floating::bit_burst_requirements(tiny_buffer.ref(),tiny_tall.view(),Kind::expm1);
    assert(tiny_need.supported()&&tiny_need.scratch_bytes>=17*8&&tiny_need.scratch_bytes<4096);
    for(bool negative:{false,true}){tiny_tall.negative=negative;Reference reference(tiny_tall,1025);
        for(auto mode:modes)run(tiny_tall,reference,1025,Kind::expm1,mode,1);}
    Input tiny_mixed{{UINT64_C(0x137ca06d95e248bf),UINT64_C(0x8901b35c8e72f46d),1},-729,false};
    for(bool negative:{false,true}){tiny_mixed.negative=negative;Reference reference(tiny_mixed,1025);
        for(auto mode:modes){run(tiny_mixed,reference,65,Kind::sincos,mode,1,nullptr,nullptr,false,false,1025);
            run(tiny_mixed,reference,1025,Kind::sincos,mode,2,nullptr,nullptr,false,false,65);}}
}
void prepared_wide(){
    Team team;
    for(std::size_t p:{8193u,12289u,16387u})for(auto kind:{Kind::exp,Kind::expm1,Kind::sincos}){
        allocation_watch_start();const auto layout=floating::bit_burst_program_layout(sbn::bits{p+16},2,kind);
        assert(!allocation_watch_stop()&&layout.need.supported()&&layout.working_bits>p);
        assert(layout.max_exact_limbs*64<8*layout.working_bits&&layout.phase_bytes&&layout.metadata_bytes&&layout.live_value_bytes);
        assert(layout.need.scratch_bytes>=layout.metadata_bytes+layout.shared_table_budget+layout.live_value_bytes+layout.phase_bytes);
        sbn::workspace storage;storage.reserve(layout.need.scratch_bytes+64,layout.need.scratch_alignment);auto bytes=storage.on().scratch;
        std::fill(bytes.begin()+layout.need.scratch_bytes,bytes.begin()+layout.need.scratch_bytes+64,std::byte{0x37});
        allocation_watch_start();const auto prepared=floating::bit_burst_program_prepare(sbn::bits{p+16},2,kind,bytes.first(layout.need.scratch_bytes));
        assert(!allocation_watch_stop());assert(floating::bit_burst_statistics(prepared).shared_table_bytes<=layout.shared_table_budget);
        for(bool negative:{false,true}){const auto input=dense_perturbation(p,negative);Reference reference(input,p,true);
            for(auto mode:{rounding::nearest_even,rounding::downward})
                run(input,reference,p,kind,mode,kind==Kind::sincos&&negative?2:1,&prepared,team.value,true,p>=12289);
        }
        for(std::size_t i=0;i<64;++i)assert(bytes[layout.need.scratch_bytes+i]==std::byte{0x37});
    }
}
void leaves_and_domain(){
    std::array<std::uint64_t,2>a{},b{};sbn::float_buffer out(a,sbn::bits{65}),other(b,sbn::bits{65});
    const std::uint64_t one=1,three=3;const auto unit=sbn::float_view::from_magnitude({&one,1},0);
    for(bool negative:{false,true})for(auto mode:modes){
        const auto zero=sbn::float_view::from_magnitude({},0,negative);
        for(auto kind:{Kind::exp,Kind::expm1}){const auto need=floating::bit_burst_requirements(out.ref(),zero,kind);assert(need.supported()&&!need.scratch_bytes);
            allocation_watch_start();const auto report=floating::bit_burst_into(out.ref(),zero,kind,{},mode);assert(!allocation_watch_stop()&&report.error.exact());
            if(kind==Kind::exp)assert(sbn::compare(out.view(),unit)==std::partial_ordering::equivalent);
            else assert(out.view().zero()&&out.view().negative()==negative);}
        const auto x=sbn::float_view::from_magnitude({&three,1},-sbn::float_exponent_limit-2,negative);
        for(auto kind:{Kind::exp,Kind::expm1}){const auto need=floating::bit_burst_requirements(out.ref(),x,kind);assert(need.supported()&&!need.scratch_bytes);
            allocation_watch_start();const auto report=floating::bit_burst_into(out.ref(),x,kind,{},mode);assert(!allocation_watch_stop());
            assert(sbn::compare(out.view(),kind==Kind::exp?unit:x)==std::partial_ordering::equivalent);
            assert(report.error.kind==sbn::error_kind::finite&&report.error.exponent2==(kind==Kind::exp?-sbn::float_exponent_limit+1:-2*sbn::float_exponent_limit));++tiny_runs;}
        const auto need=floating::bit_burst_sincos_requirements(out.ref(),other.ref(),x);assert(need.supported()&&!need.scratch_bytes);
        allocation_watch_start();const auto pair=floating::bit_burst_sincos_into(out.ref(),other.ref(),x,{},mode);assert(!allocation_watch_stop());
        assert(sbn::compare(out.view(),x)==std::partial_ordering::equivalent&&sbn::compare(other.view(),unit)==std::partial_ordering::equivalent);
        assert(pair.sine.error.exponent2==-3*sbn::float_exponent_limit&&pair.cosine.error.exponent2==-2*sbn::float_exponent_limit-1);++tiny_runs;
        // tiny exp must capture the input magnitude before publishing an
        // aliased one; otherwise the returned radius can lose 63 bits.
        a[0]=UINT64_MAX;
        const auto aliased=sbn::float_view::from_magnitude({a.data(),1},-sbn::float_exponent_limit-64,negative);
        allocation_watch_start();const auto exp_report=floating::bit_burst_into(out.ref(),aliased,Kind::exp,{},mode);assert(!allocation_watch_stop());
        assert(sbn::compare(out.view(),unit)==std::partial_ordering::equivalent&&exp_report.error.exponent2==-sbn::float_exponent_limit+1);++tiny_runs;
    }
    for(auto x:{sbn::float_view::nan(),sbn::float_view::infinity(),sbn::float_view::from_magnitude({&three,1},-1)}){
        assert(!floating::bit_burst_requirements(out.ref(),x,Kind::exp).supported());
        assert(!floating::bit_burst_sincos_requirements(out.ref(),other.ref(),x).supported());}
    assert(!floating::bit_burst_program_requirements(sbn::bits{0},1,Kind::exp).supported());
    assert(!floating::bit_burst_program_requirements(sbn::bits{65},0,Kind::sincos).supported());
}
void calibrate_observer(){
#if SBN3_CHECK_SMALL
    sbn3_mul_plan plan{};sbn3_mul_info info{};sbn3_mul_options options{};options.workers=1;
    const sbn3_product_spec spec{8,8};
    query_count.store(0);watch_queries.store(true);assert(sbn3_mul_query(&spec,&options,&plan,&info)==SBN3_SUPPORTED);
    watch_queries.store(false);assert(query_count.load()>0);
#endif
}
}
int main(){
    calibrate_observer();ordinary();prepared_wide();leaves_and_domain();
    std::printf("bit-burst: %zu independent interval certificates, %zu prepared replays, %zu executed >64-limb leaves, %zu extreme analytic leaves\n",
                certified,prepared_runs,wide_runs,tiny_runs);
}
