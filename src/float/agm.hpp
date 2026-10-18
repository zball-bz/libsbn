// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"

namespace sbn::v3::floating {
// Reusable positive-real, value-only AGM program. Preparation owns all root
// and product planning/table construction; execute performs no plan queries.
// Caller keeps this exclusive span alive. Reuse is serial and accepts new
// exact input points, including wider inputs than the working precision.
struct FloatAgmPrepared {
    void* state=nullptr;
    std::span<std::byte> storage{};
};
requirements agm_program_requirements(bits precision,unsigned workers) noexcept;
FloatAgmPrepared agm_program_prepare(bits precision,unsigned workers,
                                    std::span<std::byte> storage) noexcept;
// Output precision must match prepare; ex supplies an existing Team/scope.
// Values must not overlap program storage. Exact-base output/input aliases
// are safe because both inputs are staged before any caller output write.
float_report agm_program_execute(const FloatAgmPrepared&,float_ref,
    float_view a,float_view b,execution={},rounding=rounding::nearest_even) noexcept;
}

