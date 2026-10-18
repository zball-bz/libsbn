// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/trig_agm.hpp"
namespace sbn::v3::floating {
struct FloatTrigSeriesPrepared {void* state=nullptr;std::span<std::byte> storage{};};
// Reduced exact real input |r|<=1; max(output precisions)=p, 1<=p<=4096.
// Shared rectangular Taylor powers, separate componentwise error bounds.
requirements trig_series_program_requirements(bits,unsigned=1) noexcept;
FloatTrigSeriesPrepared trig_series_program_prepare(bits,unsigned,std::span<std::byte>) noexcept;
TrigSincosReport sincos_series_execute(const FloatTrigSeriesPrepared&,float_ref sine,float_ref cosine,
    float_view,execution={},rounding=rounding::nearest_even) noexcept;
requirements trig_series_requirements(float_ref sine,float_ref cosine,float_view,unsigned=1) noexcept;
TrigSincosReport trig_series_into(float_ref sine,float_ref cosine,float_view,execution={},
    rounding=rounding::nearest_even) noexcept;
}
