#pragma once
#include "algorithms/inverse_rung.hpp"
namespace sbn::v3 {
// Prepared, one-shot reciprocal program. Rungs are ordered seed -> target;
// tables, handles and product bindings already exist. The caller's target+1
// word output holds every intermediate approximation; no value ping-pong
// storage is required.
struct InverseProgram {
    size_t target, seed_limbs, rung_count;
    const InverseRung *rungs;
    void *local_seed_workspace = nullptr;
    size_t local_seed_bytes = 0;
};
void inverse_program(const InverseProgram &, const uint64_t *D, uint64_t *U) noexcept;
} // namespace sbn::v3
