// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class TrigAgmKind {atan,reduced_sincos};
struct FloatTrigAgmPrepared {void* state=nullptr;std::span<std::byte> storage{};};
struct TrigSincosReport {float_report sine{},cosine{};};
ConstantBankDemand trig_agm_constant_demand(bits,TrigAgmKind) noexcept;
// A supplied bank is borrowed and must outlive the prepared program. Without
// one, prepare owns a bank; legacy external_pi=true omits that bank for the
// explicit atan_agm_execute_with_pi entry point. Query facts cannot execute.
requirements trig_agm_requirements(bits,unsigned,TrigAgmKind,bool external_pi=false,ConstantBankBinding={}) noexcept;
FloatTrigAgmPrepared trig_agm_prepare(bits,unsigned,TrigAgmKind,std::span<std::byte>,bool external_pi=false,ConstantBankBinding={}) noexcept;
// Finite exact points only: atan |x|<=2; reduced sincos |r|<=1.
// Output precision matches prepare. Reports are absolute bounds; relative
// accuracy near a zero is the frontend's responsibility. Both outputs must
// be distinct; exact-base aliases with the input are supported.
float_report atan_agm_execute(const FloatTrigAgmPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
// The caller supplies an immutable approximation of mathematical pi with the
// stated absolute error. external_pi preparation omits its owned constant
// program/value. Supplied pi must be disjoint from output and prepared work.
float_report atan_agm_execute_with_pi(const FloatTrigAgmPrepared&,float_ref,float_view,
    float_view pi,error_bound pi_error,execution={},rounding=rounding::nearest_even) noexcept;
TrigSincosReport sincos_agm_execute(const FloatTrigAgmPrepared&,float_ref sine,
    float_ref cosine,float_view,execution={},rounding=rounding::nearest_even) noexcept;
}
