// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/product.hpp"
#include "float/roots.hpp"
#include "float/division.hpp"
#include "float/error.hpp"
#include "float/constant_bank.hpp"

namespace sbn::v3::floating {
struct ColdConstantPrepared { void* state=nullptr; std::span<std::byte> storage{}; };
std::size_t cold_constant_working_precision(bits) noexcept;
requirements cold_constant_requirements(bits,unsigned,ConstantKind,ConstantBankBinding supplied_pi={}) noexcept;
ColdConstantPrepared cold_constant_prepare(bits,unsigned,ConstantKind,std::span<std::byte>,ConstantBankBinding supplied_pi={}) noexcept;
float_report cold_constant_execute(const ColdConstantPrepared&,float_ref,execution={},
                                  rounding=rounding::nearest_even) noexcept;

// Private fixed-precision scalar bundle shared by the cold-constant and
// elliptic compositions. Every prepared child owns a disjoint quoted span;
// this initial implementation deliberately quotes their SUM, not their max.
struct RealArithmeticPlan {
    requirements need{},root_need{},division_need{},sum_need{};
    FloatProductPlan multiply{},square{};
    std::size_t precision=0; unsigned workers=1;
};
struct RealArithmetic {
    RealArithmeticPlan plan{};
    FloatProductPrepared multiply{},square{};
    FloatRootPrepared root{};
    FloatDivisionPrepared division{};
    std::span<std::byte> sums{};
};
RealArithmeticPlan real_arithmetic_query(bits,unsigned) noexcept;
void real_arithmetic_prepare(RealArithmetic&,const RealArithmeticPlan&,Frame&) noexcept;
float_report real_add(RealArithmetic&,float_ref,float_view,float_view) noexcept;
float_report real_mul(RealArithmetic&,float_ref,float_view,float_view,execution) noexcept;
float_report real_square(RealArithmetic&,float_ref,float_view,execution) noexcept;
float_report real_root(RealArithmetic&,float_ref,float_view,execution) noexcept;
float_report real_divide(RealArithmetic&,float_ref,float_view,float_view,execution) noexcept;
bool real_append(std::size_t&,std::size_t,std::size_t) noexcept;
std::int64_t real_top(float_view) noexcept;
float_view real_scaled(float_view,std::int64_t) noexcept;
ErrorMagnitude real_upper(float_view) noexcept;
float_report real_finish(float_ref,float_view,ErrorMagnitude,rounding) noexcept;
}
