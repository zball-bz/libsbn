// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/constant_bank.hpp"
namespace sbn::v3::floating {
enum class EllipticParameter { parameter, complement };
struct EllipticReport { float_report k{},e{}; };
struct FloatEllipticPrepared {void* state=nullptr;std::span<std::byte> storage{};};
ConstantBankDemand elliptic_constant_demand(bits) noexcept;
requirements elliptic_program_requirements(bits,unsigned=1,ConstantBankBinding={}) noexcept;
FloatEllipticPrepared elliptic_program_prepare(bits,unsigned,std::span<std::byte>,ConstantBankBinding={}) noexcept;
// Complete real K(m), E(m), or their direct-complement input c=1-m.
// Both outputs have the prepared precision and must be disjoint. Either may
// alias the exact input's base. The exclusive prepared span is separate.
EllipticReport elliptic_program_execute(const FloatEllipticPrepared&,float_ref k,float_ref e,
    float_view,EllipticParameter=EllipticParameter::parameter,execution={},
    rounding=rounding::nearest_even) noexcept;
requirements elliptic_ke_requirements(float_ref,float_ref,float_view,
    EllipticParameter=EllipticParameter::parameter,unsigned=1,ConstantBankBinding={}) noexcept;
EllipticReport elliptic_ke_into(float_ref,float_ref,float_view,
    EllipticParameter=EllipticParameter::parameter,execution={},
    rounding=rounding::nearest_even,ConstantBankBinding={}) noexcept;
}
