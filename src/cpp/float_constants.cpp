// Original libsbn code; LGPL-3.0-or-later.
#include "sbn3/float_constants.hpp"
#include "float/constant_bank.hpp"
#include "float/constants.hpp"
#include "cpp/float_constant_access.hpp"

namespace sbn {
namespace f=v3::floating;

namespace {
f::ConstantKind kind_of(constant_kind kind)noexcept{
    detail::require(kind==constant_kind::pi||kind==constant_kind::ln2,
                    SBN3_FATAL_ARGUMENT,"constant kind");
    return kind==constant_kind::pi?f::ConstantKind::pi:f::ConstantKind::log2;
}
f::ConstantBankDemand demand_of(constant_demand demand)noexcept{return {demand.pi_bits,demand.ln2_bits};}

}
bool constant_view::supports(constant_demand demand)const noexcept{
    if(!demand.pi_bits&&!demand.ln2_bits)return true;
    return f::constant_bank_supports(detail::constant_access::internal(*this),demand_of(demand));
}
constant_approximation constant_view::get(constant_kind kind)const noexcept{
    const auto value=f::constant_bank_borrow(detail::constant_access::internal(*this),kind_of(kind));
    return {value.point,value.error,bits{value.precision}};
}
constant_storage_requirements constants_requirements(constant_demand demand,unsigned workers)noexcept{
    if(!demand.pi_bits&&!demand.ln2_bits&&workers&&workers<=32)return {};
    const auto quote=f::constant_bank_requirements(demand_of(demand),workers);
    return {{quote.status,quote.workspace_bytes,quote.workspace_alignment,0,quote.reason},
            quote.value_bytes,quote.value_alignment,{quote.pi_bits,quote.log2_bits}};
}
constant_view constants_into(std::span<std::byte> values,constant_demand demand,execution ex)noexcept{
    if(!demand.pi_bits&&!demand.ln2_bits){
        detail::require(ex.workers()&&ex.workers()<=32,SBN3_FATAL_ARGUMENT,"constant bank workers");return {};
    }
    const auto quote=f::constant_bank_requirements(demand_of(demand),ex.workers());
    detail::require(quote.supported(),SBN3_FATAL_ARGUMENT,quote.reason);
    const auto program=f::constant_bank_prepare(quote,values,ex.scratch);
    return detail::constant_access::external(f::constant_bank_execute(program,ex));
}
float_report copy_constant_into(float_ref out,constant_view bank,constant_kind kind)noexcept{
    return f::constant_bank_prefix(detail::constant_access::internal(bank),kind_of(kind),out);
}
resource_error constant_bank::try_prepare(constant_demand demand,execution ex)noexcept{
    if(!demand.pi_bits&&!demand.ln2_bits)return {};
    if(!ready_.empty()&&ready_.supports(demand))return {};
    const auto quote=constants_requirements(demand,ex.workers());
    if(!quote.supported())return {resource_errc::capacity,quote.value_bytes,values_.bytes,0,quote.work.reason};
    if(ex.scratch.size()<quote.work.scratch_bytes||
       (quote.work.scratch_bytes&&(std::uintptr_t(ex.scratch.data())&(quote.work.scratch_alignment-1))))
        return {resource_errc::capacity,quote.work.scratch_bytes,ex.scratch.size(),0,"constant construction scratch"};
    if(values_.bytes<quote.value_bytes||
       (std::uintptr_t(values_.data)&(quote.value_alignment-1))){
        allocation replacement;const auto error=resource_->try_allocate(quote.value_bytes,quote.value_alignment,replacement);
        if(error)return error;
        if(values_.data)resource_->deallocate(values_);
        values_=replacement;
    }
    ready_={};
    ready_=constants_into({static_cast<std::byte*>(values_.data),values_.bytes},demand,ex);
    return {};
}
}
