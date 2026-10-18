// Original libsbn code; LGPL-3.0-or-later.
#pragma once
#include "float/core.hpp"

namespace sbn::v3::floating {
enum class RootKind { square_root, reciprocal_square_root };
// The caller owns this exclusive span for the handle's entire lifetime.
// Preparation compiles every rung and prepares its immutable tables. Reuse
// is serial; execution accepts any exact input width and makes no queries.
struct FloatRootPrepared {
    void* state=nullptr;
    std::span<std::byte> storage{};
};
requirements root_program_requirements(bits,unsigned,RootKind) noexcept;
FloatRootPrepared root_program_prepare(bits,unsigned,RootKind,std::span<std::byte>) noexcept;
// ex supplies only an existing Team/scope; its scratch member is not used.
// Output precision must match prepare, and values must not overlap storage.
float_report root_program_execute(const FloatRootPrepared&,float_ref,float_view,
    execution={},rounding=rounding::nearest_even) noexcept;
requirements root_requirements(float_ref, float_view, unsigned, RootKind) noexcept;
float_report root_into(float_ref, float_view, execution, rounding, RootKind) noexcept;
}
