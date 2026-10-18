// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class ExponentialKind { exp, expm1 };
struct FloatExponentialPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand exponential_constant_demand(bits) noexcept;
requirements exponential_program_requirements(bits,unsigned=1,ConstantBankBinding={}) noexcept;
FloatExponentialPrepared exponential_program_prepare(bits,unsigned,std::span<std::byte>,ConstantBankBinding={}) noexcept;
float_report exponential_program_execute(const FloatExponentialPrepared&,float_ref,float_view,
    ExponentialKind=ExponentialKind::exp,execution={},rounding=rounding::nearest_even) noexcept;
requirements exponential_requirements(float_ref,float_view,ExponentialKind,unsigned=1,ConstantBankBinding={}) noexcept;
float_report exponential_into(float_ref,float_view,ExponentialKind,execution={},rounding=rounding::nearest_even,ConstantBankBinding={}) noexcept;
}
