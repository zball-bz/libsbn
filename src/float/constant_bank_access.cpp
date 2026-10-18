// Original libsbn code; LGPL-3.0-or-later.
#include "float/constant_bank_internal.hpp"
#include "float/error.hpp"
#include <algorithm>

namespace sbn::v3::floating {
namespace {
std::int64_t top(float_view point) noexcept {
    require(point.finite()&&!point.zero(),SBN3_FATAL_MATH,"constant bank finite point");
    const __int128 value=__int128(point.exponent2())+magnitude_bits(point.magnitude());
    require(value>=INT64_MIN&&value<=INT64_MAX,SBN3_FATAL_SIZE,"constant bank point exponent");return std::int64_t(value);
}
}
namespace constant_bank_detail {
const Values* checked(ConstantBankView view) noexcept {
    if(!view.state||view.state!=view.storage.data()||view.storage.size()<sizeof(Values)||(std::uintptr_t(view.state)&127))return nullptr;
    const auto* s=static_cast<const Values*>(view.state);
    if(s->marker!=value_magic||s->bytes!=view.storage.size()||s->pi_bits!=view.pi_bits||s->log2_bits!=view.log2_bits)return nullptr;
    return s;
}
bool radius_good(float_view point,error_bound error,std::size_t bits,ConstantKind kind) noexcept {
    if(!point.finite()||point.zero()||point.negative()||error.kind==error_kind::unbounded)return false;
    const auto exponent=top(point);const bool pi=kind==ConstantKind::pi;
    // One-bit nearest pi is4. Retain the canonical constant's absolute
    // radius requirement rather than enlarging it to the carried binade.
    const bool binade=pi?(exponent==2||(bits==1&&exponent==3)):exponent==0;
    const auto constant_top=pi?2:0;
    return binade&&magnitude_bits(point.magnitude())<=bits&&(error.exact()||error.exponent2<=constant_top-std::int64_t(bits));
}
}
namespace {
ConstantValueView entry(ConstantBankView view,ConstantKind kind,bool pending) noexcept {
    const auto* s=constant_bank_detail::checked(view);
    require(s&&(pending||__atomic_load_n(&s->ready,__ATOMIC_ACQUIRE)),SBN3_FATAL_LIFETIME,"constant bank publication");
    const bool pi=kind==ConstantKind::pi;require(pi||kind==ConstantKind::log2,SBN3_FATAL_ARGUMENT,"constant bank entry");
    const auto* value=pi?s->pi:s->log2;const auto precision=pi?s->pi_bits:s->log2_bits;
    require(value&&__atomic_load_n(pi?&s->pi_ready:&s->log2_ready,__ATOMIC_ACQUIRE),SBN3_FATAL_LIFETIME,"constant bank slot publication");
    const auto report=pi?s->pi_report:s->log2_report;
    require(constant_bank_detail::radius_good(value->view(),report.error,precision,kind),SBN3_FATAL_MATH,"constant bank stored radius");
    return {value->view(),report.error,precision};
}
}
ConstantBankDemand constant_bank_merge(ConstantBankDemand a,ConstantBankDemand b) noexcept {
    return {std::max(a.pi_bits,b.pi_bits),std::max(a.log2_bits,b.log2_bits)};
}
bool constant_bank_supports(ConstantBankBinding binding,ConstantBankDemand demand) noexcept {
    if(binding.query_only())return binding.values().pi_bits>=demand.pi_bits&&binding.values().log2_bits>=demand.log2_bits;
    const auto* s=constant_bank_detail::checked(binding.values());if(!s||s->pi_bits<demand.pi_bits||s->log2_bits<demand.log2_bits)return false;
    if(binding.pending())return true;
    if(!__atomic_load_n(&s->ready,__ATOMIC_ACQUIRE))return false;
    if(demand.pi_bits&&(!s->pi||!s->pi_ready||!constant_bank_detail::radius_good(s->pi->view(),s->pi_report.error,s->pi_bits,ConstantKind::pi)))return false;
    if(demand.log2_bits&&(!s->log2||!s->log2_ready||!constant_bank_detail::radius_good(s->log2->view(),s->log2_report.error,s->log2_bits,ConstantKind::log2)))return false;
    return true;
}
ConstantBankBinding constant_bank_query_binding(ConstantBankDemand available) noexcept {
    ConstantBankBinding result;result.values_.pi_bits=available.pi_bits;result.values_.log2_bits=available.log2_bits;result.query_=true;return result;
}
void constant_bank_check_output(ConstantBankBinding binding,float_ref out) noexcept {
    require(!binding.query_only(),SBN3_FATAL_ARGUMENT,"constant query facts are not executable");
    if(!binding.supplied())return;
    require(constant_bank_supports(binding,{}),SBN3_FATAL_LIFETIME,"constant bank output binding");
    const auto* value=detail::float_access::begin(out);
    require(!overlaps(binding.values().storage.data(),binding.values().storage.size(),value->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"immutable constant output alias");
}
ConstantValueView constant_bank_borrow(ConstantBankView view,ConstantKind kind) noexcept{return entry(view,kind,false);}
float_report constant_bank_prefix(ConstantBankBinding binding,ConstantKind kind,float_ref out) noexcept {
    const auto value=entry(binding.values(),kind,binding.pending());const auto p=out.precision().count;
    require(p<=value.precision,SBN3_FATAL_ARGUMENT,"constant bank prefix precision");check_output(out,{});
    const auto* target=detail::float_access::begin(out);
    require(!overlaps(binding.values().storage.data(),binding.values().storage.size(),target->data,out.capacity()*8),SBN3_FATAL_ARGUMENT,"constant bank immutable output alias");
    const auto source=value.point.magnitude();const auto keep=(p+63)/64+2;
    const auto drop=source.size()>keep?source.size()-keep:0;
    const auto exponent=value.point.exponent2()+std::int64_t(64*drop);const integer_view high{source.limbs().subspan(drop)};
    auto report=round_magnitude(out,high,exponent,false,rounding::nearest_even);
    auto radius=ErrorMagnitude::from(report.error).add(ErrorMagnitude::from(value.error));
    if(drop)radius=radius.add(ErrorMagnitude::power_of_two(exponent));report.error=radius.publish();
    require(report.error.kind!=error_kind::unbounded&&(report.error.exact()||report.error.exponent2<=top(out.view())-std::int64_t(p)),
        SBN3_FATAL_MATH,"constant bank prefix radius");return report;
}
}
