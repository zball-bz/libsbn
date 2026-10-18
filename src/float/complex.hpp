// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"

namespace sbn::v3::floating {
// Value = (real + i*imag)*2^scale. The external exponent is not passed to
// scalar arithmetic. Components and outputs must be finite exact points.
struct ScaledComplexView {float_view real{},imag{};std::int64_t scale=0;};
struct ScaledComplexRef {
    float_ref real,imag;std::int64_t* scale;
    ScaledComplexView view()const noexcept{return {real.view(),imag.view(),*scale};}
};
enum class ComplexKind {mean,multiply,square_root,reciprocal};
struct FloatComplexPrepared {void* state=nullptr;std::span<std::byte> storage{};};
std::size_t complex_scalar_precision(bits) noexcept;
requirements complex_program_requirements(bits,unsigned workers=1) noexcept;
FloatComplexPrepared complex_program_prepare(bits,unsigned,std::span<std::byte>) noexcept;
// Outputs retain complex_scalar_precision(W) bits. Relative norm error <=2^-W;
// report.error is an absolute complex-norm bound, including prefix omission.
// Mean requires |a+b| >= (|a|+|b|)/2. Root requires Re(a)>=-2|a|/3.
// Reciprocal requires a!=0. Root(0), zero products and mean(0,0) are exact.
// b is ignored by unary operations. All values must be disjoint from storage;
// whole input-pair aliases are supported, overlapping output components are not.
// Preparation owns all queries; execution is serially reusable on a Team/scope.
float_report complex_program_execute(const FloatComplexPrepared&,ComplexKind,
    ScaledComplexRef,ScaledComplexView a,ScaledComplexView b={},execution={}) noexcept;
enum class ComplexRealKind {add,subtract,multiply,square_root,divide};
// Reuses the same prepared scalar providers between complex calls. No nested
// or concurrent use of the prepared handle is allowed. Output has S bits.
float_report complex_real_execute(const FloatComplexPrepared&,ComplexRealKind,
    float_ref,float_view a,float_view b={},execution={}) noexcept;
}
