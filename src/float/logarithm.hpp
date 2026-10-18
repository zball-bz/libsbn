// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class LogarithmKind { log, log1p };
struct FloatLogarithmPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand logarithm_constant_demand(bits) noexcept;
requirements logarithm_program_requirements(bits,unsigned=1,ConstantBankBinding={}) noexcept;
FloatLogarithmPrepared logarithm_program_prepare(bits,unsigned,std::span<std::byte>,ConstantBankBinding={}) noexcept;
// Exclusive reuse. Immutable pi/log2 values are retained after the first
// AGM execution; local log1p calls need not initialize either constant.
// Every numerical shape is compiled before the first execution.
float_report logarithm_program_execute(const FloatLogarithmPrepared&,float_ref,float_view,
    LogarithmKind=LogarithmKind::log,execution={},rounding=rounding::nearest_even) noexcept;
requirements logarithm_requirements(float_ref,float_view,LogarithmKind,unsigned=1,ConstantBankBinding={}) noexcept;
float_report logarithm_into(float_ref,float_view,LogarithmKind,execution={},
    rounding=rounding::nearest_even,ConstantBankBinding={}) noexcept;
}
