// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class DerivedExpLogKind { exp2, log2, log10 };
struct FloatDerivedExpLogPrepared {void* state=nullptr;std::span<std::byte> storage{};};
// Kind is sealed by prepare; only its necessary children are admitted.
ConstantBankDemand derived_explog_constant_demand(bits,DerivedExpLogKind) noexcept;
requirements derived_explog_program_requirements(bits,unsigned,DerivedExpLogKind,ConstantBankBinding={}) noexcept;
FloatDerivedExpLogPrepared derived_explog_program_prepare(bits,unsigned,DerivedExpLogKind,std::span<std::byte>,ConstantBankBinding={}) noexcept;
float_report derived_explog_program_execute(const FloatDerivedExpLogPrepared&,float_ref,float_view,
    DerivedExpLogKind,execution={},rounding=rounding::nearest_even) noexcept;
requirements derived_explog_requirements(float_ref,float_view,DerivedExpLogKind,unsigned=1,ConstantBankBinding={}) noexcept;
float_report derived_explog_into(float_ref,float_view,DerivedExpLogKind,execution={},rounding=rounding::nearest_even,ConstantBankBinding={}) noexcept;
}
