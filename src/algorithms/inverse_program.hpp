#pragma once
#include "algorithms/inverse_rung.hpp"
namespace sbn::v3 {
// Prepared, one-shot reciprocal program. Rungs are ordered seed -> target;
// tables, handles and product bindings already exist. values[0/1] each have
// next_precision(target)+1 limbs when the final output is external (target+1
// also works), and are disjoint from input and all rung workspaces. Output
// may be the inactive value buffer on the last rung (index rung_count%2).
struct InverseProgram {
    size_t target, seed_limbs, rung_count;
    const InverseRung *rungs;
    uint64_t *values[2];
};
void inverse_program(const InverseProgram &, const uint64_t *D, uint64_t *U) noexcept;
} // namespace sbn::v3
