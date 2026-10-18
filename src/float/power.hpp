// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"
#include "float/error.hpp"

namespace sbn::v3::floating {
struct FloatPowerPrepared {void* state=nullptr;std::span<std::byte> storage{};};
requirements power_program_requirements(bits,std::int64_t,unsigned) noexcept;
FloatPowerPrepared power_program_prepare(bits,std::int64_t,unsigned,std::span<std::byte>) noexcept;
// Fixed precision/exponent/worker recipe. The exclusive prepared span holds
// all tables and rolling values. ex supplies only an existing Team/scope;
// ex.scratch is not used. No query or ordinary allocation occurs here.
float_report power_program_execute(const FloatPowerPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;

// Positive unsigned power without public exponent clipping. The center is a
// lower bound for the exact power; |P-center| <= relative_error*center.
// Its normalized mantissa borrows prepared storage until the next execution.
struct WidePositivePowerPrepared {void* state=nullptr;std::span<std::byte> storage{};};
struct WidePositivePowerValue {
    float_view mantissa{};
    __int128_t top_exponent=0;
    ErrorMagnitude relative_error{};
};
requirements wide_power_program_requirements(bits function_precision,std::uint64_t degree,unsigned workers) noexcept;
WidePositivePowerPrepared wide_power_program_prepare(bits,std::uint64_t,unsigned,std::span<std::byte>) noexcept;
// degree>=1, positive finite nonzero input; no query/allocation during replay.
WidePositivePowerValue wide_power_program_execute(const WidePositivePowerPrepared&,float_view,execution={}) noexcept;
}
