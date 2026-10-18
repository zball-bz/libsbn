// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class HyperbolicKind {sinh,cosh,tanh,asinh,acosh,atanh};
struct FloatHyperbolicPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand hyperbolic_constant_demand(bits,HyperbolicKind) noexcept;
// The parent merges its own and exp/log child demands. A supplied bank must
// outlive the program; without one the full bank value/build storage is quoted.
requirements hyperbolic_program_requirements(bits,unsigned,HyperbolicKind,ConstantBankBinding={}) noexcept;
FloatHyperbolicPrepared hyperbolic_prepare(bits,unsigned,HyperbolicKind,std::span<std::byte>,ConstantBankBinding={}) noexcept;
float_report hyperbolic_execute(const FloatHyperbolicPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
requirements hyperbolic_requirements(float_ref,float_view,unsigned,HyperbolicKind,ConstantBankBinding={}) noexcept;
float_report hyperbolic_into(float_ref,float_view,execution,rounding,HyperbolicKind,ConstantBankBinding={}) noexcept;
// Exact point inputs; same-base aliases are supported. Prepared reuse retains
// immutable constants and owns the SUM of its child spans. Every shape is
// sealed before execution, using an existing Team/scope. Finite normal bounds:
// <=one result precision ulp nearest; <=two for other mantissa rounding modes.
}
