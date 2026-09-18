#pragma once
#include <stddef.h>
#include "tuning/native_policy.hpp"
namespace sbn::v3 {
// Zen5 AI Max 395, Clang 21.1.8. Whole-operation 4/8/16-worker probes:
// bench/probes/yc_newton_2026-09-09/v3_{1788942485385291129,
// 1788942706260148412,1788942775558850146}. Initial conservative cutoff;
// Include 16 words for the ladder's existing guard/padding near 2^18.
// It caps a rung's team, never changes arithmetic feasibility or precision.
constexpr unsigned newton_rung_workers(size_t n, unsigned maximum) noexcept {
    const unsigned cap = n < native_policy::newton_single_worker_below ? 1
                         : n <= native_policy::newton_middle_band_max  ? native_policy::newton_middle_workers
                                                                       : maximum;
    return maximum < cap ? maximum : cap;
}
} // namespace sbn::v3
