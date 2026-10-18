// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
namespace sbn::v3::floating {
enum class ExpSeriesKind {exp,expm1};
struct FloatExpSeriesPrepared {void* state=nullptr;std::span<std::byte> storage{};};
// Reduced real |r|<=1, 1<=p<=4096 only. Binding the input sign avoids a
// general division plan for positive recipes; negative recipes quote one
// such plan for their final reciprocal/rational transform.
requirements exp_series_program_requirements(bits,ExpSeriesKind,bool negative,unsigned=1) noexcept;
FloatExpSeriesPrepared exp_series_program_prepare(bits,ExpSeriesKind,bool negative,unsigned,std::span<std::byte>) noexcept;
float_report exp_series_program_execute(const FloatExpSeriesPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
requirements exp_series_requirements(float_ref,float_view,ExpSeriesKind,unsigned=1) noexcept;
float_report exp_series_into(float_ref,float_view,ExpSeriesKind,execution={},rounding=rounding::nearest_even) noexcept;
}
