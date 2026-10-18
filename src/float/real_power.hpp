// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
struct FloatRealPowerPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand real_power_constant_demand(bits) noexcept;
requirements real_power_program_requirements(bits,unsigned=1,ConstantBankBinding={}) noexcept;
FloatRealPowerPrepared real_power_program_prepare(bits,unsigned,std::span<std::byte>,ConstantBankBinding={}) noexcept;
float_report real_power_program_execute(const FloatRealPowerPrepared&,float_ref,float_view base,float_view exponent,
    execution={},rounding=rounding::nearest_even) noexcept;
// Fresh exact int64 exponents select the existing integer-power recipe
// before admission. Explicit prepared execution always uses its sealed
// generic log/product/exp program, apart from exact/special leaves.
requirements real_power_requirements(float_ref,float_view,float_view,unsigned=1,ConstantBankBinding={}) noexcept;
float_report real_power_into(float_ref,float_view,float_view,execution={},rounding=rounding::nearest_even,ConstantBankBinding={}) noexcept;
}
