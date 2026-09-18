#pragma once
#include <stddef.h>
namespace sbn::v3::divrem_tuning {
// Initial shape thresholds for the exact division service. Zen5 AI Max 395,
// Clang 21.1.8. Calibrated by experiments/results/divrem_2026-09-18 (crossover
// sweep of the schoolbook entry against the block-Barrett service); the
// values only choose between exact algorithms and never change results.
inline constexpr size_t schoolbook_max_divisor = 64;   // dn <= this: 3/2 schoolbook
inline constexpr size_t schoolbook_max_quotient = 16;  // quotient limbs <= this: schoolbook (O(dn) per limb)
// Planning seed for the block inverse: a Newton ladder to precision n costs
// about this many linear n x n products (rungs sum to ~2 products of the
// final size plus the smaller rungs). Only orders block-size candidates.
inline constexpr double inverse_cost_ratio = 3.0;
// Arithmetic guarantees (docs/divrem-design): |qhat-q|<=7, |E|<7D.
inline constexpr unsigned correction_limit = 8;
inline constexpr size_t ring_guard_words = 2; // ring >= dn + guard for the signed residual lift
} // namespace sbn::v3::divrem_tuning
