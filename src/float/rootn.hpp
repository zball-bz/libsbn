// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"

namespace sbn::v3::floating {
struct FloatRootnPrepared {void* state=nullptr;std::span<std::byte> storage{};};
requirements rootn_program_requirements(bits,std::uint64_t,unsigned) noexcept;
FloatRootnPrepared rootn_program_prepare(bits,std::uint64_t,unsigned,std::span<std::byte>) noexcept;
// A fixed degree/precision/worker program, with every arithmetic recipe
// prepared before the certified seed. Values cannot overlap its exclusive
// storage. Existing Team/scope only; no query or allocation during execution.
float_report rootn_program_execute(const FloatRootnPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
}
