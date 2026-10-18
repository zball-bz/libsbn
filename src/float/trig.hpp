// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/trig_agm.hpp"
namespace sbn::v3::floating {
enum class TrigKind {radians,pi_scaled};
// Fresh general real frontends. The complete cold constant, exact phase-window
// product and reduced kernel storage is quoted; execute makes all plans before
// evaluating constants. Input points may alias either distinct output bank.
// Reports bound absolute error, including range reduction. pi_scaled evaluates
// sin(pi*x),cos(pi*x) with exact dyadic phase reduction; integer/half-integer
// inputs are exact. sinpi(integer) zero has the input sign; cospi at a
// half-integer is positive zero, independently of quadrant sign changes.
// A supplied immutable bank remains alive for the call; otherwise the fresh
// parent builds one merged bank and lends it to its reduced evaluator.
ConstantBankDemand trig_constant_demand(bits sine,bits cosine,float_view,TrigKind) noexcept;
requirements trig_requirements(float_ref sine,float_ref cosine,float_view,
    unsigned workers,TrigKind,ConstantBankBinding={}) noexcept;
TrigSincosReport trig_into(float_ref sine,float_ref cosine,float_view,
    execution,rounding,TrigKind,ConstantBankBinding={}) noexcept;
requirements unary_trig_requirements(float_ref,float_view,unsigned,TrigKind,ConstantBankBinding={}) noexcept;
float_report unary_trig_into(float_ref,float_view,bool cosine,TrigKind,execution,rounding,ConstantBankBinding={}) noexcept;
}
