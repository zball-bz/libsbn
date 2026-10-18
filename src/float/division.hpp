// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "algorithms/normalized_newton.hpp"

namespace sbn::v3::floating {
struct FloatDivisionPrepared {void* state=nullptr;std::span<std::byte> storage{};};
requirements division_program_requirements(bits,unsigned,sbn3_newton_kind)noexcept;
FloatDivisionPrepared division_program_prepare(bits,unsigned,sbn3_newton_kind,std::span<std::byte>)noexcept;
// INVERSE ignores a and reciprocates b. The exclusive prepared span contains
// both immutable recipes and mutable prefix/output banks; reuse is serial.
float_report division_program_execute(const FloatDivisionPrepared&,float_ref,float_view a,float_view b,
    execution={},rounding=rounding::nearest_even)noexcept;
}
