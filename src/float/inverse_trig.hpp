// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class InverseTrigKind {atan,atan2,asin,acos};
struct FloatInverseTrigPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand inverse_trig_constant_demand(bits,InverseTrigKind) noexcept;
// A supplied immutable bank must outlive the prepared program. Otherwise one
// merged bank is included in scratch and published lazily on its first use.
requirements inverse_trig_program_requirements(bits,unsigned,InverseTrigKind,ConstantBankBinding={}) noexcept;
FloatInverseTrigPrepared inverse_trig_prepare(bits,unsigned,InverseTrigKind,std::span<std::byte>,ConstantBankBinding={}) noexcept;
// atan2 uses a=y,b=x; unary kinds ignore b. Nearest finite normal results have
// <=one result precision ulp error, other modes <=two; modes round computed
// approximations. Tiny outputs retain scaled bounds, including final underflow.
// Point inputs are exact. Prepared storage is exclusive, serially reusable;
// existing Team/scopes are borrowed. Exact-base output/input aliases work.
float_report inverse_trig_execute(const FloatInverseTrigPrepared&,float_ref,
    float_view a,float_view b={},execution={},rounding=rounding::nearest_even) noexcept;
requirements inverse_trig_requirements(float_ref,float_view a,float_view b,
    unsigned,InverseTrigKind,ConstantBankBinding={}) noexcept;
float_report inverse_trig_into(float_ref,float_view a,float_view b,
    execution,rounding,InverseTrigKind,ConstantBankBinding={}) noexcept;
}
