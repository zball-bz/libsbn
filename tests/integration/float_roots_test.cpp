// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/float.hpp"
#include "sbn3/arena.h"
#include "float/roots.hpp"
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
constexpr std::size_t precisions[]{1,2,3,7,31,32,53,63,64,65,95,127,128,
    129,191,255,257,383,511,513,769,1023,1025,1537,2049,4093,4097,6145,8201};
constexpr std::uint64_t guard=UINT64_C(0xa96283fe70145dbc);
std::uint64_t state=UINT64_C(0xc207516bd936428f);
std::size_t checked=0,prepared_checked=0;

struct Ref {
    ref_int value;
    Ref(){ref_init(value);}
    ~Ref(){ref_clear(value);}
    Ref(const Ref&)=delete;
    Ref&operator=(const Ref&)=delete;
};
// The general ref_mul switches to a production-backed witness for large
// operands. Build every product here from the oracle's independent exact
// multiply-by-one-word, addition and shifts instead. Export before writing
// result, and accumulate separately so result may alias either input.
void independent_product(ref_number* result,const ref_number* a,const ref_number* b){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    std::vector<std::uint64_t> words((ref_sizeinbase(b,2)+63)/64);
    std::size_t count=0;ref_export(words.data(),&count,-1,8,0,0,b);
    assert(count<=words.size());
    Ref accumulator,term;ref_set_ui(accumulator.value,0);
    for(std::size_t i=0;i<count;++i)if(words[i]){
        ref_mul_ui(term.value,a,words[i]);
        if(i)ref_mul_2exp(term.value,term.value,64*i);
        ref_add(accumulator.value,accumulator.value,term.value);
    }
    ref_set(result,accumulator.value);
}
struct Input {
    std::vector<std::uint64_t> words;
    std::int64_t exponent=0;
    sbn::float_view view()const noexcept {
        return sbn::float_view::from_magnitude(std::span<const std::uint64_t>(words),exponent);
    }
};
std::uint64_t random_word(){state^=state<<13;state^=state>>7;state^=state<<17;return state;}
std::size_t word_bits(std::span<const std::uint64_t> words){
    auto n=words.size();while(n&&!words[n-1])--n;
    if(!n)return 0;
    std::size_t high=0;for(auto v=words[n-1];v;v>>=1)++high;
    return 64*(n-1)+high;
}
Input exported(const ref_number* value,std::int64_t exponent){
    assert(ref_sgn(value)>0);
    Input result;result.words.resize((ref_sizeinbase(value,2)+63)/64);
    std::size_t count=0;ref_export(result.words.data(),&count,-1,8,0,0,value);
    assert(count==result.words.size());result.exponent=exponent;return result;
}
Input random_input(std::size_t precision,int offset){
    Input x;x.words.resize((precision+211+63)/64);
    for(auto& w:x.words)w=random_word();
    // Include a partly occupied top word and a low nonzero tail.
    x.words.back()=(x.words.back()&UINT64_C(0x1ffff))|UINT64_C(0x10000);
    x.words.front()|=1;
    x.exponent=-std::int64_t(word_bits(x.words))+offset;
    return x;
}
Input near_power(std::size_t bit,int delta,int offset){
    Ref a;ref_set_ui(a.value,1);ref_mul_2exp(a.value,a.value,bit);
    if(delta<0)ref_sub_ui(a.value,a.value,1);else if(delta>0)ref_add_ui(a.value,a.value,1);
    return exported(a.value,-std::int64_t(bit)+offset);
}
Input near_square(std::size_t precision,int delta,int offset){
    const auto root_bits=precision/2+37;
    Ref r,a;ref_set_ui(r.value,1);ref_mul_2exp(r.value,r.value,root_bits);
    ref_add_ui(r.value,r.value,UINT64_C(0x1bd13));
    independent_product(a.value,r.value,r.value);
    if(delta<0)ref_sub_ui(a.value,a.value,1);else if(delta>0)ref_add_ui(a.value,a.value,1);
    return exported(a.value,-std::int64_t(2*root_bits)+offset);
}

// Compare nonnegative dyadics without materializing an enormous exponent
// gap. Equal top exponents imply that the remaining shift is bounded by the
// sizes of these test integers, even at the public exponent limits.
int compare_scaled(const ref_number* a,std::int64_t ae,const ref_number* b,std::int64_t be){
    assert(ref_sgn(a)>=0&&ref_sgn(b)>=0);
    if(!ref_sgn(a)||!ref_sgn(b))return (ref_sgn(a)>0)-(ref_sgn(b)>0);
    const __int128 at=__int128(ae)+ref_sizeinbase(a,2),bt=__int128(be)+ref_sizeinbase(b,2);
    if(at!=bt)return at<bt?-1:1;
    Ref left,right;ref_set(left.value,a);ref_set(right.value,b);
    const __int128 shift=__int128(ae)-be;
    assert(shift>-1000000&&shift<1000000);
    if(shift>0)ref_mul_2exp(left.value,left.value,std::size_t(shift));
    else if(shift<0)ref_mul_2exp(right.value,right.value,std::size_t(-shift));
    return ref_cmp(left.value,right.value);
}

// No floating reference, production root, or production multiplication is
// involved in this certificate. The returned radius produces exact rational
// endpoints L,H. Verify L^2 <= x <= H^2, or x L^2 <= 1 <= x H^2.
void certify(const Input& input,sbn::float_view y,const sbn::float_report& report,
             std::size_t precision,rounding mode,bool inverse){
    assert(y.finite()&&!y.zero()&&!y.negative());
    assert(word_bits(y.magnitude().limbs())<=precision);
    assert(report.error.kind!=sbn::error_kind::unbounded);
    assert(!sbn::has_flag(report.flags,sbn::float_flag::invalid));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::overflow));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::underflow));
    assert(!sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
    Ref x,center,radius,lo,hi,squared,other,one;
    ref_import(x.value,input.words.size(),-1,8,0,0,input.words.data());
    ref_import(center.value,y.magnitude().size(),-1,8,0,0,y.magnitude().data());
    auto scale=y.exponent2();
    if(report.error.exact())ref_set_ui(radius.value,0);
    else{
        const auto ulp=y.exponent2()+std::int64_t(word_bits(y.magnitude().limbs()))-std::int64_t(precision);
        assert(report.error.exponent2<=ulp+(mode==rounding::nearest_even?0:1));
        scale=std::min(scale,report.error.exponent2);
        const auto shift=y.exponent2()-scale;
        assert(shift>=0&&shift<1000000);
        ref_mul_2exp(center.value,center.value,std::size_t(shift));
        ref_set_ui(radius.value,1);
        const auto radius_shift=report.error.exponent2-scale;
        assert(radius_shift>=0&&radius_shift<1000000);
        ref_mul_2exp(radius.value,radius.value,std::size_t(radius_shift));
    }
    ref_sub(lo.value,center.value,radius.value);if(ref_sgn(lo.value)<0)ref_set_ui(lo.value,0);
    ref_add(hi.value,center.value,radius.value);ref_set_ui(one.value,1);
    independent_product(squared.value,lo.value,lo.value);
    bool lower=false,upper=false;
    if(inverse){
        independent_product(other.value,squared.value,x.value);
        lower=compare_scaled(other.value,input.exponent+2*scale,one.value,0)<=0;
        independent_product(squared.value,hi.value,hi.value);independent_product(other.value,squared.value,x.value);
        upper=compare_scaled(other.value,input.exponent+2*scale,one.value,0)>=0;
    }else{
        lower=compare_scaled(squared.value,2*scale,x.value,input.exponent)<=0;
        independent_product(squared.value,hi.value,hi.value);
        upper=compare_scaled(squared.value,2*scale,x.value,input.exponent)>=0;
    }
    if(!lower||!upper){
        std::fprintf(stderr,"float root enclosure failed: p=%zu inverse=%d mode=%u input_words=%zu exponent=%lld bound=%lld\n",
            precision,int(inverse),unsigned(mode),input.words.size(),(long long)input.exponent,(long long)report.error.exponent2);
        assert(false);
    }
}

struct ScopeCall {
    sbn::float_ref out;
    sbn::float_view input;
    sbn::execution ex;
    rounding mode;
    bool inverse;
    const floating::FloatRootPrepared* prepared;
    sbn::float_report report;
};
void on_scope(void* opaque,sbn3_team_scope* scope){
    auto& c=*static_cast<ScopeCall*>(opaque);c.ex.team=nullptr;c.ex.scope=scope;
    c.report=c.prepared?floating::root_program_execute(*c.prepared,c.out,c.input,c.ex,c.mode):
        c.inverse?sbn::rsqrt_into(c.out,c.input,c.ex,c.mode):sbn::sqrt_into(c.out,c.input,c.ex,c.mode);
}
void run_case(const Input& input,std::size_t precision,rounding mode,bool inverse,
              sbn::workspace& scratch,sbn3_team* team=nullptr,bool alias=false,bool scope=false,
              const floating::FloatRootPrepared* prepared=nullptr){
    const auto output_words=(precision+63)/64;
    const auto capacity=alias?std::max(output_words,input.words.size()):output_words;
    std::vector<std::uint64_t> storage(capacity+2,guard);
    if(alias)std::copy(input.words.begin(),input.words.end(),storage.begin());
    const auto original=input.words;
    sbn::float_buffer output{std::span<std::uint64_t>(storage.data(),capacity),sbn::bits{precision}};
    const auto source=alias?sbn::float_view::from_magnitude(std::span<const std::uint64_t>(storage.data(),input.words.size()),input.exponent):input.view();
    sbn::execution ex{{},team,nullptr};
    if(!prepared){
        allocation_watch_start();
        const auto need=inverse?sbn::rsqrt_requirements(output.ref(),source,team?sbn3_team_workers(team):1):
            sbn::sqrt_requirements(output.ref(),source,team?sbn3_team_workers(team):1);
        const auto query_allocations=allocation_watch_stop();
        assert(!query_allocations&&need.supported());assert(need.output_limbs==output_words);
        scratch.reserve(need.scratch_bytes,need.scratch_alignment);
        ex=scratch.on(team);ex.scratch=ex.scratch.first(need.scratch_bytes);
    }
    sbn::float_report report;
    allocation_watch_start();
    if(scope){
        assert(team);ScopeCall call{output.ref(),source,ex,mode,inverse,prepared,{}};
        sbn3_team_run(team,on_scope,&call);report=call.report;
    }else report=prepared?floating::root_program_execute(*prepared,output.ref(),source,ex,mode):
        inverse?sbn::rsqrt_into(output.ref(),source,ex,mode):sbn::sqrt_into(output.ref(),source,ex,mode);
    const auto execute_allocations=allocation_watch_stop();
    assert(!execute_allocations);
    assert(storage[capacity]==guard&&storage[capacity+1]==guard);
    assert(input.words==original);
    if(alias)for(std::size_t i=output_words;i<input.words.size();++i)assert(storage[i]==input.words[i]);
    certify(input,output.view(),report,precision,mode,inverse);++checked;if(prepared)++prepared_checked;
}

class Team {
    sbn3_arena* arena_=nullptr;
    sbn3_lease control_{};
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
    Team(const Team&)=delete;
    Team&operator=(const Team&)=delete;
};

void special_cases(){
    alignas(64)std::array<std::uint64_t,2> words{};
    sbn::float_buffer out{words,sbn::bits{65}};
    const std::uint64_t one=1,three=3;
    for(auto mode:modes){
        for(bool negative:{false,true}){
            const auto zero=sbn::float_view::from_magnitude({},0,negative);
            allocation_watch_start();
            auto report=sbn::sqrt_into(out.ref(),zero,{},mode);
            assert(!allocation_watch_stop());assert(out.view().zero()&&out.view().negative()==negative);
            assert(report.error.exact()&&report.flags==sbn::float_flag::none);
            report=sbn::rsqrt_into(out.ref(),zero,{},mode);
            assert(out.view().classification()==sbn::float_class::infinity&&out.view().negative()==negative);
            assert(sbn::has_flag(report.flags,sbn::float_flag::divide_by_zero));
        }
        auto report=sbn::sqrt_into(out.ref(),sbn::float_view::infinity(),{},mode);
        assert(out.view().classification()==sbn::float_class::infinity&&!out.view().negative());
        report=sbn::rsqrt_into(out.ref(),sbn::float_view::infinity(),{},mode);
        assert(out.view().zero()&&!out.view().negative()&&report.error.exact());
        for(bool inverse:{false,true})for(auto x:{sbn::float_view::infinity(true),
            sbn::float_view::from_magnitude({&three,1},0,true)}){
            report=inverse?sbn::rsqrt_into(out.ref(),x,{},mode):sbn::sqrt_into(out.ref(),x,{},mode);
            assert(out.view().is_nan()&&sbn::has_flag(report.flags,sbn::float_flag::invalid));
        }
        for(bool inverse:{false,true}){
            report=inverse?sbn::rsqrt_into(out.ref(),sbn::float_view::nan(),{},mode):sbn::sqrt_into(out.ref(),sbn::float_view::nan(),{},mode);
            assert(out.view().is_nan());
            for(std::int64_t exponent:{-1000000000LL,-130LL,-2LL,0LL,2LL,126LL,1000000000LL}){
                const auto x=sbn::float_view::from_magnitude({&one,1},exponent);
                const auto need=inverse?sbn::rsqrt_requirements(out.ref(),x):sbn::sqrt_requirements(out.ref(),x);
                assert(need.supported()&&!need.scratch_bytes);
                allocation_watch_start();
                report=inverse?sbn::rsqrt_into(out.ref(),x,{},mode):sbn::sqrt_into(out.ref(),x,{},mode);
                assert(!allocation_watch_stop());assert(report.error.exact()&&report.flags==sbn::float_flag::none);
                assert(out.view().magnitude().size()==1&&out.view().magnitude().data()[0]==1);
                assert(out.view().exponent2()==(inverse?-exponent/2:exponent/2));
            }
        }
    }
    std::array<std::uint64_t,1> short_words{};
    sbn::float_buffer short_out{short_words,sbn::bits{129}};
    assert(sbn::sqrt_requirements(short_out.ref(),sbn::float_view::from_magnitude({&three,1},0)).status==sbn::support::capacity);
    assert(!sbn::rsqrt_requirements(out.ref(),sbn::float_view::from_magnitude({&three,1},0),0).supported());
}
void precision_sweep(){
    sbn::workspace scratch;
    for(auto p:precisions){
        std::array<Input,7> inputs{random_input(p,int(p%7)-3),near_power(p+17,-1,-1),
            near_power(p+17,1,0),near_square(p,-1,1),near_square(p,0,-1),near_square(p,1,0),Input{{2},0}};
        for(std::size_t i=0;i<inputs.size();++i)for(auto mode:modes)for(bool inverse:{false,true})
            run_case(inputs[i],p,mode,inverse,scratch,nullptr,i==0||i==4);
    }
    const std::int64_t exponents[]{-sbn::float_exponent_limit-2,-1000000001LL,-65,-64,-3,-2,-1,
        0,1,2,63,64,1000000001LL,sbn::float_exponent_limit-2};
    for(auto exponent:exponents){
        const Input input{{3},exponent};
        for(auto mode:modes)for(bool inverse:{false,true})run_case(input,129,mode,inverse,scratch);
    }
    // Much larger exact source than target: discarded input limbs must be
    // covered by the prefix certificate, including an aliased output.
    const auto long_input=random_input(32771,1);
    for(auto p:{3u,65u,257u})for(auto mode:modes)for(bool inverse:{false,true})
        run_case(long_input,p,mode,inverse,scratch,nullptr,true);
}
void team_cases(){
    sbn::workspace scratch;
    for(unsigned workers:{1u,4u}){
        Team team(workers);
        for(auto p:{257u,1025u,4097u,8201u}){
            const auto input=random_input(p,-1);
            for(auto mode:modes)for(bool inverse:{false,true})
                run_case(input,p,mode,inverse,scratch,team.value,true);
        }
        const auto input=near_square(4097,-1,1);
        for(auto mode:modes)for(bool inverse:{false,true})
            run_case(input,4097,mode,inverse,scratch,team.value,false,true);
    }
}
void prepared_cases(){
    sbn::workspace unused;
    for(unsigned workers:{1u,4u}){
        Team team(workers);
        for(std::size_t p:{65u,257u,1025u,4097u,8201u})for(bool inverse:{false,true}){
            const auto kind=inverse?floating::RootKind::reciprocal_square_root:floating::RootKind::square_root;
            allocation_watch_start();
            const auto need=floating::root_program_requirements(sbn::bits{p},workers,kind);
            assert(!allocation_watch_stop());assert(need.supported());
            assert(need.output_limbs==(p+63)/64&&need.scratch_bytes);
            sbn::workspace storage;storage.reserve(need.scratch_bytes+64,need.scratch_alignment);
            auto bytes=storage.on().scratch;
            std::fill(bytes.begin()+need.scratch_bytes,bytes.begin()+need.scratch_bytes+64,std::byte{0x39});
            allocation_watch_start();
            const auto prepared=floating::root_program_prepare(sbn::bits{p},workers,kind,bytes.first(need.scratch_bytes));
            assert(!allocation_watch_stop());assert(prepared.state&&prepared.storage.size()==need.scratch_bytes);
            // One prepared program handles different source widths, exponent
            // parity, aliases and modes. No requirements call occurs in this
            // loop, and root_program_execute has no planning entry.
            std::array<Input,5> inputs{Input{{2},0},random_input(p,1),near_square(p,-1,-1),
                random_input(2*p+1027,0),Input{{1},-130}};
            for(std::size_t i=0;i<inputs.size();++i)for(auto mode:modes)
                run_case(inputs[i],p,mode,inverse,unused,team.value,i==1||i==3,i%2==0,&prepared);
            for(std::size_t i=0;i<64;++i)assert(bytes[need.scratch_bytes+i]==std::byte{0x39});
            // Reuse after the exact shortcut must reinitialize all mutable
            // iterate banks while retaining the immutable prepared tables.
            run_case(near_power(p+31,1,-1),p,rounding::nearest_even,inverse,unused,team.value,true,false,&prepared);
        }
    }
    assert(!floating::root_program_requirements(sbn::bits{0},1,floating::RootKind::square_root).supported());
    assert(!floating::root_program_requirements(sbn::bits{129},0,floating::RootKind::square_root).supported());
}
void nonzero_first_worker_cases(){
    Team team(4);sbn::workspace scratch;
    constexpr std::size_t precision=8193;
    const auto input=random_input(precision,-1);
    std::vector<std::uint64_t> words((precision+63)/64);
    sbn::float_buffer out{words,sbn::bits{precision}};
    for(bool inverse:{false,true}){
        const auto kind=inverse?floating::RootKind::reciprocal_square_root:floating::RootKind::square_root;
        const auto need=floating::root_program_requirements(sbn::bits{precision},2,kind);
        assert(need.supported());scratch.reserve(need.scratch_bytes,need.scratch_alignment);
        const auto prepared=floating::root_program_prepare(sbn::bits{precision},2,kind,scratch.on().scratch.first(need.scratch_bytes));
        ScopeCall call{out.ref(),input.view(),{},rounding::nearest_even,inverse,&prepared,{}};
        allocation_watch_start();
        sbn3_team_run(team.value,[](void* opaque,sbn3_team_scope* scope){
            sbn3_team_invoke2(scope,SBN3_PARALLEL_CHILDREN,2,
                [](void*,sbn3_team_scope*){},nullptr,
                [](void* value,sbn3_team_scope* child){assert(sbn3_team_first_worker(child)==2);on_scope(value,child);},opaque);
        },&call);
        assert(!allocation_watch_stop());
        certify(input,out.view(),call.report,precision,rounding::nearest_even,inverse);
        ++checked;++prepared_checked;
    }
}
}

int main(){
    special_cases();precision_sweep();team_cases();prepared_cases();nonzero_first_worker_cases();
    std::printf("float sqrt/rsqrt: %zu independent exact endpoint certificates (%zu prepared reuse); precision, exponent, alias, W1/W4, scope and allocation gates PASS\n",checked,prepared_checked);
}
