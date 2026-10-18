// Original libsbn tests; LGPL-3.0-or-later.
#include "sbn3/sbn.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>

extern "C" void allocation_watch_start();
extern "C" std::uint64_t allocation_watch_stop();
namespace {
using i128=__int128_t;
using u128=__uint128_t;
constexpr unsigned F=112;
constexpr u128 unit=u128(1)<<F;
struct Interval {i128 lo,hi;};
u128 ceil_div(u128 a,u128 b){return a/b+(a%b!=0);}
// Independent fixed-point intervals for atan(1/q) and atanh(1/q).
// Power recurrence rounds both endpoints outwards. After stopping, the
// absolute remaining sum is <2*next_power_hi on this same grid.
Interval small_series(unsigned q,bool alternating){
    u128 lo=unit/q,hi=ceil_div(unit,q);Interval sum{};
    for(unsigned k=0;k<128;++k){
        const auto a=i128(lo/(2*k+1)),b=i128(ceil_div(hi,2*k+1));
        if(alternating&&(k&1)){sum.lo-=b;sum.hi-=a;}
        else{sum.lo+=a;sum.hi+=b;}
        lo/=q*q;hi=ceil_div(hi,q*q);
        if(hi<=1)return {sum.lo-i128(2*hi),sum.hi+i128(2*hi)};
    }
    assert(false);return {};
}
Interval reference(sbn::constant_kind kind){
    if(kind==sbn::constant_kind::ln2){const auto a=small_series(3,false);return {2*a.lo,2*a.hi};}
    const auto a=small_series(5,true),b=small_series(239,true);
    return {16*a.lo-4*b.hi,16*a.hi-4*b.lo};
}
void certificate(sbn::float_view value,sbn::float_report report,Interval expected){
    assert(value.finite()&&!value.zero()&&!value.negative());
    assert(value.magnitude().size()<=2&&report.error.kind==sbn::error_kind::finite);
    u128 significand=0;
    for(std::size_t i=value.magnitude().size();i--;)significand=(significand<<64)|value.magnitude().data()[i];
    const auto shift=value.exponent2()+F,rshift=report.error.exponent2+F;
    assert(shift>=0&&shift<128&&rshift>=0&&rshift<120);
    const auto point=i128(significand<<unsigned(shift)),radius=i128(1)<<unsigned(rshift);
    assert(point-radius<=expected.lo&&point+radius>=expected.hi);
}
void check_prefixes(sbn::constant_view bank){
    for(auto kind:{sbn::constant_kind::pi,sbn::constant_kind::ln2}){
        const auto expected=reference(kind);
        for(unsigned p:{17u,53u,65u}){
            std::array<std::uint64_t,2> limbs{};sbn::float_buffer out(limbs,sbn::bits{p});
            allocation_watch_start();const auto report=sbn::copy_constant_into(out.ref(),bank,kind);
            assert(!allocation_watch_stop());certificate(out.view(),report,expected);
        }
    }
}
struct RecordingResource final:sbn::memory_resource {
    sbn::buffer_resource storage;unsigned allocations=0,releases=0;
    explicit RecordingResource(std::span<std::byte> bytes):storage(bytes){}
    sbn::resource_error try_allocate(std::size_t n,std::size_t a,sbn::allocation& out)noexcept override{
        ++allocations;return storage.try_allocate(n,a,out);
    }
    void deallocate(sbn::allocation value)noexcept override{++releases;storage.deallocate(value);}
    sbn::resource_statistics statistics()const noexcept override{return storage.statistics();}
};
template<class T>concept RvalueConstantView=requires(T&& object){std::move(object).view();};
static_assert(!RvalueConstantView<sbn::constant_bank>);
void raw_storage_lifetimes(){
    const sbn::constant_demand demand{513,257};
    const auto need=sbn::constants_requirements(demand);assert(need.supported());
    sbn::workspace values,work;values.reserve(need.value_bytes,need.value_alignment);
    work.reserve(need.work.scratch_bytes,need.work.scratch_alignment);
    allocation_watch_start();
    const auto bank=sbn::constants_into(values.on().scratch,demand,work.on());
    assert(!allocation_watch_stop()&&bank.supports(demand));
    assert(bank.precision().pi_bits>=513&&bank.precision().ln2_bits>=257);
    assert(!bank.supports({bank.precision().pi_bits+1,0}));
    // Immutable values, their descriptors and error reports cannot refer to
    // any producer state or temporaries in the now-dead construction span.
    std::memset(work.on().scratch.data(),0xa7,work.on().scratch.size());
    check_prefixes(bank);
    const auto pi=bank.get(sbn::constant_kind::pi);
    assert(pi.precision.count==bank.precision().pi_bits&&pi.point.finite());
}
void owned_and_context(){
    alignas(4096)std::array<std::byte,65536> bytes{};RecordingResource values(bytes);
    const sbn::constant_demand demand{257,129};const auto need=sbn::constants_requirements(demand);
    sbn::workspace work;work.reserve(need.work.scratch_bytes,need.work.scratch_alignment);
    {
        sbn::constant_bank bank(values);
        assert(bank.view().empty()&&values.allocations==0);
        assert(bank.view().supports({})&&!bank.try_prepare({},{}));
        assert(sbn::constants_requirements({}).supported()&&values.allocations==0);
        const auto insufficient=bank.try_prepare(demand,{});
        assert(insufficient&&bank.view().empty()&&values.allocations==0);
        assert(!bank.try_prepare(demand,work.on()));
        const auto borrowed=bank.view();check_prefixes(borrowed);
        const auto count=values.allocations;
        assert(bank.try_prepare({65537,65537},{}));
        assert(bank.view().supports(demand)&&values.allocations==count);
        check_prefixes(borrowed);
        allocation_watch_start();assert(!bank.try_prepare({65,65},{}));assert(!allocation_watch_stop());
        assert(values.allocations==count);
        sbn::constant_bank moved(std::move(bank));
        assert(bank.view().empty()&&moved.view().supports(demand));check_prefixes(borrowed);
        const sbn::constant_demand larger{1025,513};
        const auto growth=sbn::constants_requirements(larger);assert(growth.supported());
        work.reserve(growth.work.scratch_bytes,growth.work.scratch_alignment);
        moved.prepare(larger,work.on()); // old borrowed view is now invalid
        const auto replacement_view=moved.view();check_prefixes(replacement_view);
        sbn::constant_bank assigned(values);assigned.prepare(demand,work.on());
        assigned=std::move(moved); // releases assigned's previous allocation
        assert(moved.view().empty()&&assigned.view().supports(larger));
        check_prefixes(replacement_view);
    }
    assert(values.statistics().live_allocations==0&&values.allocations==values.releases);
    {
        // A context's value resource and its scratch/persistent-constant
        // resource remain distinct. Shared construction work can be reused.
        sbn::context context(values,sbn::default_resource());
        assert(context.constants().empty());context.prepare_constants(demand);
        auto bank=context.constants();check_prefixes(bank);
        const auto ex=context.execution();std::memset(ex.scratch.data(),0x71,ex.scratch.size());
        check_prefixes(bank);
        allocation_watch_start();assert(!context.try_prepare_constants({65,65}));assert(!allocation_watch_stop());
    }
    assert(values.statistics().live_allocations==0);
}
void public_function_bindings(){
    using Query=sbn::requirements(*)(sbn::float_ref,sbn::float_view,unsigned,sbn::constant_view)noexcept;
    using Run=sbn::float_report(*)(sbn::float_ref,sbn::float_view,sbn::execution,sbn::rounding,sbn::constant_view)noexcept;
    using Demand=sbn::constant_demand(*)(sbn::bits)noexcept;
    struct Case {Query query;Run run;Demand demand;double input,lower,upper;};
    const Case cases[]{
        {sbn::log_requirements,sbn::log_into,sbn::log_constants,2,0.5,1},
        {sbn::log1p_requirements,sbn::log1p_into,sbn::log_constants,0.5,0.25,0.5},
        {sbn::exp_requirements,sbn::exp_into,sbn::exp_constants,0.5,1.5,2},
        {sbn::expm1_requirements,sbn::expm1_into,sbn::exp_constants,0.5,0.5,1},
        {sbn::sin_requirements,sbn::sin_into,nullptr,0.5,0.25,0.75},
        {sbn::cos_requirements,sbn::cos_into,nullptr,0.5,0.75,1},
        {sbn::sinpi_requirements,sbn::sinpi_into,nullptr,0.125,0.25,0.5},
        {sbn::cospi_requirements,sbn::cospi_into,nullptr,0.125,0.75,1},
        {sbn::atan_requirements,sbn::atan_into,sbn::atan_constants,0.5,0.25,0.5},
        {sbn::asin_requirements,sbn::asin_into,sbn::asin_constants,0.5,0.5,1},
        {sbn::acos_requirements,sbn::acos_into,sbn::acos_constants,0.5,1,1.5},
        {sbn::sinh_requirements,sbn::sinh_into,sbn::sinh_constants,0.5,0.5,1},
        {sbn::cosh_requirements,sbn::cosh_into,sbn::cosh_constants,0.5,1,1.5},
        {sbn::tanh_requirements,sbn::tanh_into,sbn::tanh_constants,0.5,0.25,0.5},
        {sbn::asinh_requirements,sbn::asinh_into,sbn::asinh_constants,0.5,0.25,0.5},
        {sbn::acosh_requirements,sbn::acosh_into,sbn::acosh_constants,1.5,0.5,1},
        {sbn::atanh_requirements,sbn::atanh_into,sbn::atanh_constants,0.5,0.5,1},
        {sbn::exp2_requirements,sbn::exp2_into,sbn::exp2_constants,0.5,1,1.5},
        {sbn::log2_requirements,sbn::log2_into,sbn::log2_constants,1.5,0.5,1},
        {sbn::log10_requirements,sbn::log10_into,sbn::log10_constants,1.5,0.125,0.25}};
    const sbn::bits precision{65};sbn::constant_demand demand{};
    auto merge=[&](sbn::constant_demand d){demand=sbn::merge_constants(demand,d);};
    for(const auto& c:cases)if(c.demand)merge(c.demand(precision));
    merge(sbn::pow_constants(precision));merge(sbn::elliptic_constants(precision));
    const auto half=sbn::bigfloat::from(0.5,precision),one=sbn::bigfloat::from(1,precision),two=sbn::bigfloat::from(2,precision);
    merge(sbn::sincos_constants(precision,precision,half.view()));
    merge(sbn::sincospi_constants(precision,precision,half.view()));
    sbn::context context;context.prepare_constants(demand);const auto bank=context.constants();
    std::array<std::uint64_t,2> a{},b{};sbn::float_buffer output(a,precision),second(b,precision);
    for(const auto& c:cases){
        const auto input=sbn::bigfloat::from(c.input,precision);
        const auto need=c.query(output.ref(),input.view(),1,bank);assert(need.supported());context.reserve(need);
        allocation_watch_start();const auto report=c.run(output.ref(),input.view(),context.execution(),sbn::rounding::nearest_even,bank);
        assert(!allocation_watch_stop()&&report.error.kind!=sbn::error_kind::unbounded);
        const auto lower=sbn::bigfloat::from(c.lower),upper=sbn::bigfloat::from(c.upper);
        assert(sbn::compare(output.view(),lower.view())>0&&sbn::compare(output.view(),upper.view())<0);
    }
    const auto cold=sbn::log_requirements(output.ref(),two.view());
    const auto warm=sbn::log_requirements(output.ref(),two.view(),1,bank);
    assert(cold.supported()&&warm.supported()&&warm.scratch_bytes<cold.scratch_bytes);
    context.reserve(warm);
    const auto log=sbn::log_into(output.ref(),two.view(),context.execution(),sbn::rounding::nearest_even,bank);
    certificate(output.view(),log,reference(sbn::constant_kind::ln2));
    const auto pair_need=sbn::sincospi_requirements(output.ref(),second.ref(),half.view(),1,bank);
    context.reserve(pair_need);
    const auto pair=sbn::sincospi_into(output.ref(),second.ref(),half.view(),context.execution(),sbn::rounding::nearest_even,bank);
    assert(pair.sine.error.exact()&&pair.cosine.error.exact());
    assert(sbn::compare(output.view(),one.view())==0&&second.view().zero()&&!second.view().negative());
    context.reserve(sbn::pow_requirements(output.ref(),two.view(),half.view(),1,bank));
    const auto power=sbn::pow_into(output.ref(),two.view(),half.view(),context.execution(),sbn::rounding::nearest_even,bank);
    assert(power.error.kind!=sbn::error_kind::unbounded&&sbn::compare(output.view(),one.view())>0);
    context.reserve(sbn::atan2_requirements(output.ref(),one.view(),one.view(),1,bank));
    const auto angle=sbn::atan2_into(output.ref(),one.view(),one.view(),context.execution(),sbn::rounding::nearest_even,bank);
    assert(angle.error.kind!=sbn::error_kind::unbounded&&sbn::compare(output.view(),half.view())>0&&sbn::compare(output.view(),one.view())<0);
    context.reserve(sbn::elliptic_ke_requirements(output.ref(),second.ref(),half.view(),sbn::elliptic_argument::parameter,1,bank));
    const auto elliptic=sbn::elliptic_ke_into(output.ref(),second.ref(),half.view(),sbn::elliptic_argument::parameter,context.execution(),sbn::rounding::nearest_even,bank);
    assert(elliptic.k.error.kind!=sbn::error_kind::unbounded&&elliptic.e.error.kind!=sbn::error_kind::unbounded);
    check_prefixes(bank);
}
}
int main(){raw_storage_lifetimes();owned_and_context();public_function_bindings();
    std::puts("float constant bank: independent Machin/atanh prefix enclosures, allocation-free construction/copy, immutable value/work separation, ownership and context reuse PASS");}
